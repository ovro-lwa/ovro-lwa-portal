"""Progressive (chunk-aligned) async dynamic-spectrum fill plus linked sky tile.

Standalone experiment code for the ``async-test`` branch. It does not import
``ovro_lwa_portal.viz``. Readers work on a raw ``zarr.Array`` (Option A/E) or a
lazy xarray ``DataArray``; painters are headless (matplotlib Agg, Bokeh
document serialization) so the benchmark runs without a browser.

Main-thread model
-----------------
Workers only read and return numpy blocks. The calling ("main") thread drains a
queue, writes blocks into a fixed ``(n_time, n_freq)`` grid, and repaints at
most once per ``paint_interval_s``. That mirrors a notebook io_loop: paints are
the only main-thread stalls.
"""

from __future__ import annotations

import queue
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FillUnit:
    """Half-open ``[t0, t1) × [f0, f1)`` block of the dynspec grid."""

    t0: int
    t1: int
    f0: int
    f1: int


def plan_fill_units(n_time: int, n_freq: int, time_chunk: int, freq_chunk: int) -> list[FillUnit]:
    """One unit per on-disk chunk footprint along time × frequency."""
    return [
        FillUnit(t0, min(t0 + time_chunk, n_time), f0, min(f0 + freq_chunk, n_freq))
        for t0 in range(0, n_time, time_chunk)
        for f0 in range(0, n_freq, freq_chunk)
    ]


class ZarrPixelReader:
    """Read a pixel column block straight from a 5-D ``zarr.Array``."""

    def __init__(self, array, *, pol_idx: int, l_idx: int, m_idx: int) -> None:
        self.array = array
        self.pol_idx = int(pol_idx)
        self.l_idx = int(l_idx)
        self.m_idx = int(m_idx)

    def __call__(self, unit: FillUnit) -> np.ndarray:
        return np.asarray(
            self.array[unit.t0 : unit.t1, unit.f0 : unit.f1, self.pol_idx, self.l_idx, self.m_idx]
        )


class XarrayPixelReader:
    """Read a pixel column block from a lazy (non-dask) xarray ``DataArray``."""

    def __init__(self, column) -> None:
        self.column = column

    def __call__(self, unit: FillUnit) -> np.ndarray:
        return np.asarray(
            self.column.isel(time=slice(unit.t0, unit.t1), frequency=slice(unit.f0, unit.f1)).values
        )


class ZarrTrackedPixelReader:
    """Read a dynspec block at per-time ``(l, m)`` pixels (RA/Dec track).

    ``l_indices`` / ``m_indices`` / ``visible`` have length ``n_time`` (full cube).
    Invisible times are filled with NaN. When every visible time in a unit shares
    the same pixel, the read collapses to one zarr slice (same as
    :class:`ZarrPixelReader`).
    """

    def __init__(
        self,
        array,
        *,
        pol_idx: int,
        l_indices: np.ndarray,
        m_indices: np.ndarray,
        visible: np.ndarray,
    ) -> None:
        self.array = array
        self.pol_idx = int(pol_idx)
        self.l_indices = np.asarray(l_indices, dtype=int)
        self.m_indices = np.asarray(m_indices, dtype=int)
        self.visible = np.asarray(visible, dtype=bool)

    def __call__(self, unit: FillUnit) -> np.ndarray:
        n_t = unit.t1 - unit.t0
        n_f = unit.f1 - unit.f0
        out = np.full((n_t, n_f), np.nan, dtype=np.float32)
        vis = self.visible[unit.t0 : unit.t1]
        if not np.any(vis):
            return out
        ls = self.l_indices[unit.t0 : unit.t1]
        ms = self.m_indices[unit.t0 : unit.t1]
        # Fast path: one pixel for all times in this unit (fixed or track-stalled).
        if np.all(ls[vis] == ls[vis][0]) and np.all(ms[vis] == ms[vis][0]) and np.all(vis):
            out[:] = self.array[
                unit.t0 : unit.t1, unit.f0 : unit.f1, self.pol_idx, int(ls[0]), int(ms[0])
            ]
            return out
        for i, t in enumerate(range(unit.t0, unit.t1)):
            if not self.visible[t]:
                continue
            out[i] = self.array[
                t, unit.f0 : unit.f1, self.pol_idx, int(self.l_indices[t]), int(self.m_indices[t])
            ]
        return out


def read_sky_tile_zarr(array, t: int, f: int, p: int, l_slice: slice, m_slice: slice) -> np.ndarray:
    return np.asarray(array[t, f, p, l_slice, m_slice], dtype=np.float32)


def _chunk_aligned_ranges(start: int, stop: int, chunk: int) -> list[tuple[int, int]]:
    edges = [start, *range((start // chunk + 1) * chunk, stop, chunk), stop]
    return list(zip(edges[:-1], edges[1:]))


def read_sky_tile_parallel(
    array, t: int, f: int, p: int, l_slice: slice, m_slice: slice, executor: Executor
) -> np.ndarray:
    """Read a sky tile as one task per spatial chunk (zarr v2 reads chunks serially)."""
    l_ranges = _chunk_aligned_ranges(l_slice.start, l_slice.stop, array.chunks[3])
    m_ranges = _chunk_aligned_ranges(m_slice.start, m_slice.stop, array.chunks[4])
    if len(l_ranges) * len(m_ranges) == 1:
        return executor.submit(read_sky_tile_zarr, array, t, f, p, l_slice, m_slice).result()
    out = np.empty((l_slice.stop - l_slice.start, m_slice.stop - m_slice.start), dtype=np.float32)

    def _read(lr: tuple[int, int], mr: tuple[int, int]) -> None:
        out[lr[0] - l_slice.start : lr[1] - l_slice.start, mr[0] - m_slice.start : mr[1] - m_slice.start] = (
            array[t, f, p, lr[0] : lr[1], mr[0] : mr[1]]
        )

    futures = [executor.submit(_read, lr, mr) for lr in l_ranges for mr in m_ranges]
    for fut in futures:
        fut.result()
    return out


@dataclass
class FillStats:
    """Timings are seconds since the fill started (``perf_counter`` deltas)."""

    n_units: int
    t_first_data: float = float("nan")
    t_first_paint: float = float("nan")
    t_complete_data: float = float("nan")
    t_complete_paint: float = float("nan")
    paint_durations: list[float] = field(default_factory=list)

    @property
    def n_paints(self) -> int:
        return len(self.paint_durations)

    @property
    def max_paint(self) -> float:
        return max(self.paint_durations) if self.paint_durations else float("nan")

    @property
    def total_paint(self) -> float:
        return float(sum(self.paint_durations))


def progressive_fill(
    read_unit: Callable[[FillUnit], np.ndarray],
    units: Sequence[FillUnit],
    grid: np.ndarray,
    *,
    paint: Callable[[np.ndarray], None],
    executor: Executor | None = None,
    max_workers: int = 8,
    paint_interval_s: float = 0.1,
    submit: Callable | None = None,
) -> FillStats:
    """Fill ``grid`` block by block as reads finish; paint at a bounded rate.

    ``submit(fn, unit)`` can override task submission (e.g. a Dask client);
    it must return an object with ``add_done_callback(cb)`` whose callback
    receives a future-like with ``.result()``.
    """
    stats = FillStats(n_units=len(units))
    done_q: queue.SimpleQueue = queue.SimpleQueue()
    own_pool = None
    if submit is None:
        if executor is None:
            own_pool = ThreadPoolExecutor(max_workers=max_workers)
            executor = own_pool
        submit = executor.submit

    start = time.perf_counter()
    for unit in units:
        fut = submit(read_unit, unit)
        fut.add_done_callback(lambda f, u=unit: done_q.put((u, f)))

    remaining = len(units)
    last_paint = start
    dirty = False
    try:
        while remaining:
            timeout = max(0.0, paint_interval_s - (time.perf_counter() - last_paint)) if dirty else None
            try:
                unit, fut = done_q.get(timeout=timeout)
            except queue.Empty:
                unit = None
            if unit is not None:
                grid[unit.t0 : unit.t1, unit.f0 : unit.f1] = fut.result()
                remaining -= 1
                dirty = True
                if remaining == len(units) - 1:
                    stats.t_first_data = time.perf_counter() - start
                if remaining == 0:
                    stats.t_complete_data = time.perf_counter() - start
            now = time.perf_counter()
            if dirty and (remaining == 0 or now - last_paint >= paint_interval_s or stats.n_paints == 0):
                p0 = time.perf_counter()
                paint(grid)
                p1 = time.perf_counter()
                stats.paint_durations.append(p1 - p0)
                if stats.n_paints == 1:
                    stats.t_first_paint = p1 - start
                last_paint = p1
                dirty = False
        stats.t_complete_paint = time.perf_counter() - start
    finally:
        if own_pool is not None:
            own_pool.shutdown(wait=True)
    return stats


# --------------------------------------------------------------------------
# Painters (headless). Each exposes ``paint_sky(tile)`` and ``paint_dynspec(grid)``.
# --------------------------------------------------------------------------


def _finite_limits(data: np.ndarray) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [2, 98])
    return float(lo), float(hi) if hi > lo else float(lo) + 1.0


class MatplotlibLinkedPainter:
    """Two-pane matplotlib figure rendered with Agg (a real raster draw)."""

    def __init__(self, n_time: int, n_freq: int, tile_shape: tuple[int, int], *, blit: bool = False) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        self.blit = blit
        self.fig = Figure(figsize=(11, 4.5), dpi=100)
        self.canvas = FigureCanvasAgg(self.fig)
        ax_sky, ax_dyn = self.fig.subplots(1, 2)
        self.sky_im = ax_sky.imshow(np.zeros(tile_shape).T, origin="lower", cmap="inferno")
        ax_sky.set_title("Sky tile")
        self.dyn_im = ax_dyn.imshow(
            np.full((n_freq, n_time), np.nan), origin="lower", aspect="auto", cmap="viridis"
        )
        ax_dyn.set_xlabel("time index")
        ax_dyn.set_ylabel("frequency index")
        ax_dyn.set_title("Dynamic spectrum")
        self.ax_sky, self.ax_dyn = ax_sky, ax_dyn
        self.canvas.draw()
        self._backgrounds = {
            ax: self.canvas.copy_from_bbox(ax.bbox) for ax in (ax_sky, ax_dyn)
        }

    def _redraw(self, ax, artist) -> None:
        if not self.blit:
            self.canvas.draw()
            return
        # Only the image artist is re-rasterized; axes chrome comes from the cache.
        self.canvas.restore_region(self._backgrounds[ax])
        ax.draw_artist(artist)
        self.canvas.blit(ax.bbox)

    def paint_sky(self, tile: np.ndarray) -> None:
        self.sky_im.set_data(tile.T)
        self.sky_im.set_clim(*_finite_limits(tile))
        self._redraw(self.ax_sky, self.sky_im)

    def paint_dynspec(self, grid: np.ndarray) -> None:
        self.dyn_im.set_data(grid.T)
        self.dyn_im.set_clim(*_finite_limits(grid))
        self._redraw(self.ax_dyn, self.dyn_im)


class BokehLinkedPainter:
    """Two Bokeh image figures in one Document; paint = mutate + serialize patch.

    Serializing the ``PATCH-DOC`` message is what Panel/Bokeh does before sending
    a notebook comm update. Browser rendering is not measured.
    """

    def __init__(self, n_time: int, n_freq: int, tile_shape: tuple[int, int]) -> None:
        from bokeh.document import Document
        from bokeh.models import ColumnDataSource, LinearColorMapper
        from bokeh.plotting import figure
        from bokeh.protocol import Protocol

        self._protocol = Protocol()
        self.doc = Document()
        self._events: list = []
        self.sky_src = ColumnDataSource({"image": [np.zeros(tile_shape, dtype=np.float32).T]})
        self.dyn_src = ColumnDataSource({"image": [np.zeros((n_freq, n_time))]})
        self.sky_cmap = LinearColorMapper(palette="Inferno256")
        self.dyn_cmap = LinearColorMapper(palette="Viridis256")
        sky = figure(width=450, height=450, title="Sky tile")
        sky.image("image", source=self.sky_src, x=0, y=0, dw=tile_shape[0], dh=tile_shape[1], color_mapper=self.sky_cmap)
        dyn = figure(width=600, height=450, title="Dynamic spectrum")
        dyn.image("image", source=self.dyn_src, x=0, y=0, dw=n_time, dh=n_freq, color_mapper=self.dyn_cmap)
        self.doc.add_root(sky)
        self.doc.add_root(dyn)
        self.doc.to_json()  # first mount
        self.doc.callbacks.on_change(lambda event: self._events.append(event))

    def _flush(self) -> None:
        events, self._events = self._events, []
        if events:
            msg = self._protocol.create("PATCH-DOC", events)
            # Force serialization of header/content like a comm send would.
            _ = msg.header_json, msg.content_json, msg.buffers

    def paint_sky(self, tile: np.ndarray) -> None:
        lo, hi = _finite_limits(tile)
        self.sky_src.data = {"image": [np.ascontiguousarray(tile.T)]}
        self.sky_cmap.update(low=lo, high=hi)
        self._flush()

    def paint_dynspec(self, grid: np.ndarray) -> None:
        lo, hi = _finite_limits(grid)
        self.dyn_src.data = {"image": [np.ascontiguousarray(grid.T)]}
        self.dyn_cmap.update(low=lo, high=hi)
        self._flush()
