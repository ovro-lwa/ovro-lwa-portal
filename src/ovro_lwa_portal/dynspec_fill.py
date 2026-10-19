"""Progressive dynamic-spectrum fill for Source Review.

Workers read one time x frequency chunk at the per-time tracked pixel. The
caller paints the assembled grid on the UI thread. This does not load a sky
slice and does not move the selected heatmap cell.
"""

from __future__ import annotations

import queue
import time
from collections.abc import Callable
from concurrent.futures import Executor, Future
from dataclasses import dataclass

import numpy as np
import xarray as xr

from ovro_lwa_portal.zarr_direct import DirectZarrArray


@dataclass(frozen=True)
class FillUnit:
    """Half-open ``[t0, t1) x [f0, f1)`` block of the spectrum grid."""

    t0: int
    t1: int
    f0: int
    f1: int


def plan_fill_units(
    n_time: int,
    n_freq: int,
    time_chunk: int,
    freq_chunk: int,
) -> list[FillUnit]:
    """One unit per chunk footprint along time and frequency."""
    t_step = max(1, int(time_chunk))
    f_step = max(1, int(freq_chunk))
    return [
        FillUnit(t0, min(t0 + t_step, n_time), f0, min(f0 + f_step, n_freq))
        for t0 in range(0, n_time, t_step)
        for f0 in range(0, n_freq, f_step)
    ]


def spectrum_chunk_lengths(
    sky: xr.DataArray, direct: DirectZarrArray | None = None
) -> tuple[int, int]:
    """Time and frequency chunk lengths: on-disk when ``direct``, else Dask."""
    if direct is not None:
        return int(direct.chunks[0]), int(direct.chunks[1])
    return _axis_chunk(sky, "time"), _axis_chunk(sky, "frequency")


def _axis_chunk(sky: xr.DataArray, dim: str) -> int:
    size = int(sky.sizes[dim])
    chunks = getattr(sky.data, "chunks", None)
    if not chunks:
        return size
    axis = int(sky.get_axis_num(dim))
    if axis >= len(chunks):
        return size
    block = chunks[axis]
    if isinstance(block, tuple):
        return max(1, int(block[0])) if block else size
    return max(1, int(block))


class TrackedPixelReader:
    """Read a spectrum block at per-time ``(l, m)`` pixels.

    ``l_indices``, ``m_indices``, and ``visible`` each have length ``n_time``.
    Invisible times stay NaN. A block whose visible times share one pixel is
    one slice. With ``direct``, a moving pixel is one Zarr coordinate
    selection; without it, each visible time is read through xarray.
    """

    def __init__(
        self,
        sky: xr.DataArray,
        *,
        pol_idx: int,
        l_indices: np.ndarray,
        m_indices: np.ndarray,
        visible: np.ndarray,
        direct: DirectZarrArray | None = None,
    ) -> None:
        self._sky = sky
        self._has_pol = "polarization" in sky.dims
        self.pol_idx = int(pol_idx)
        self.l_indices = np.asarray(l_indices, dtype=int)
        self.m_indices = np.asarray(m_indices, dtype=int)
        self.visible = np.asarray(visible, dtype=bool)
        self.direct = direct

    def __call__(self, unit: FillUnit) -> np.ndarray:
        n_t = unit.t1 - unit.t0
        n_f = unit.f1 - unit.f0
        out = np.full((n_t, n_f), np.nan, dtype=np.float32)
        vis = self.visible[unit.t0 : unit.t1]
        if not np.any(vis):
            return out
        ls = self.l_indices[unit.t0 : unit.t1]
        ms = self.m_indices[unit.t0 : unit.t1]
        same_pixel = bool(np.all(ls[vis] == ls[vis][0]) and np.all(ms[vis] == ms[vis][0]))
        direct = self.direct
        if direct is not None:
            freqs = slice(unit.f0, unit.f1)
            if same_pixel and bool(np.all(vis)):
                out[:] = direct.column(
                    slice(unit.t0, unit.t1), freqs, self.pol_idx, int(ls[0]), int(ms[0])
                )
            else:
                rows = np.flatnonzero(vis)
                out[rows] = direct.points(unit.t0 + rows, freqs, self.pol_idx, ls[rows], ms[rows])
            return out
        if same_pixel and bool(np.all(vis)):
            out[:] = self._column(slice(unit.t0, unit.t1), slice(unit.f0, unit.f1), int(ls[0]), int(ms[0]))
            return out
        for i, t in enumerate(range(unit.t0, unit.t1)):
            if not self.visible[t]:
                continue
            out[i] = self._column(
                t,
                slice(unit.f0, unit.f1),
                int(self.l_indices[t]),
                int(self.m_indices[t]),
            )
        return out

    def _column(
        self,
        time_sel: int | slice,
        freq_sel: slice,
        l_idx: int,
        m_idx: int,
    ) -> np.ndarray:
        sel: dict[str, int | slice] = {
            "time": time_sel,
            "frequency": freq_sel,
            "l": l_idx,
            "m": m_idx,
        }
        if self._has_pol:
            sel["polarization"] = self.pol_idx
        block = self._sky.isel(sel)
        if isinstance(time_sel, slice):
            block = block.transpose("time", "frequency")
        return np.asarray(block.values, dtype=np.float32)


class ProgressiveSpectrumFill:
    """Submit chunk reads and paint the grid as they finish.

    ``paint(grid, done)`` and ``on_error(exc)`` run on ``schedule_main``. A
    newer :meth:`start` drops results from the previous fill.
    """

    def __init__(
        self,
        executor: Executor,
        schedule_main: Callable[[Callable[[], None]], None],
        *,
        paint_interval_s: float = 0.1,
    ) -> None:
        self.executor = executor
        self.schedule_main = schedule_main
        self.paint_interval_s = float(paint_interval_s)
        self._gen = 0
        self._queue: queue.SimpleQueue[tuple[int, FillUnit, Future[np.ndarray]]] = queue.SimpleQueue()

    def start(
        self,
        reader: Callable[[FillUnit], np.ndarray],
        *,
        n_time: int,
        n_freq: int,
        time_chunk: int,
        freq_chunk: int,
        paint: Callable[[np.ndarray, bool], None],
        on_error: Callable[[BaseException], None],
        on_chunk: Callable[[int, int], None] | None = None,
    ) -> None:
        self._gen += 1
        gen = self._gen
        grid = np.full((int(n_time), int(n_freq)), np.nan, dtype=np.float64)
        units = plan_fill_units(int(n_time), int(n_freq), int(time_chunk), int(freq_chunk))
        state = {
            "remaining": len(units),
            "total": len(units),
            "last": 0.0,
            "painted": False,
            "failed": False,
        }

        def _drain() -> None:
            self._drain(gen, grid, state, paint, on_error, on_chunk)

        if not units:
            self.schedule_main(lambda: paint(grid.copy(), True))
            return
        for unit in units:
            fut = self.executor.submit(reader, unit)

            def _done(
                done: Future[np.ndarray],
                u: FillUnit = unit,
                drain: Callable[[], None] = _drain,
            ) -> None:
                self._enqueue(gen, u, done, drain)

            fut.add_done_callback(_done)

    def _enqueue(
        self,
        gen: int,
        unit: FillUnit,
        fut: Future[np.ndarray],
        drain: Callable[[], None],
    ) -> None:
        self._queue.put((gen, unit, fut))
        self.schedule_main(drain)

    def _drain(
        self,
        gen: int,
        grid: np.ndarray,
        state: dict[str, float | int | bool],
        paint: Callable[[np.ndarray, bool], None],
        on_error: Callable[[BaseException], None],
        on_chunk: Callable[[int, int], None] | None = None,
    ) -> None:
        if gen != self._gen or state["failed"]:
            return
        wrote = False
        while True:
            try:
                item_gen, unit, fut = self._queue.get_nowait()
            except queue.Empty:
                break
            if item_gen != gen:
                continue
            try:
                block = np.asarray(fut.result(), dtype=np.float64)
            except Exception as exc:  # noqa: BLE001 — worker errors reach on_error
                state["failed"] = True
                self._gen += 1
                on_error(exc)
                return
            grid[unit.t0 : unit.t1, unit.f0 : unit.f1] = block
            state["remaining"] = int(state["remaining"]) - 1
            wrote = True
            if on_chunk is not None:
                done = int(state["total"]) - int(state["remaining"])
                on_chunk(done, int(state["total"]))
        if not wrote:
            return
        now = time.perf_counter()
        remaining = int(state["remaining"])
        interval_elapsed = now - float(state["last"]) >= self.paint_interval_s
        due = (not state["painted"]) or remaining == 0 or interval_elapsed
        if not due:
            return
        state["painted"] = True
        state["last"] = now
        paint(grid.copy(), remaining == 0)
