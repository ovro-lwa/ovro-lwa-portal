"""Live linked sky-tile + progressive dynamic-spectrum UI (Panel + Bokeh).

Experiment code for branch ``async-test``; independent of ``ovro_lwa_portal.viz``.

Threading model
---------------
* Worker threads (``ThreadPoolExecutor``) only read zarr blocks.
* A tornado ``PeriodicCallback`` on the kernel io_loop drains finished blocks,
  mutates the Bokeh ``ColumnDataSource`` in place, and pushes the notebook comm
  once per tick (coalesced paints, like ``progressive_fill`` in the benchmark).
* Tapping a dynspec cell submits a sky-tile read; its paint is scheduled back
  onto the io_loop. Stale tile reads (superseded by a newer tap) are dropped.
"""

from __future__ import annotations

import queue
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zarr

from dynspec_async import (
    FillUnit,
    ZarrPixelReader,
    ZarrTrackedPixelReader,
    plan_fill_units,
    read_sky_tile_zarr,
)


def _finite_limits(data: np.ndarray) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [2, 98])
    return float(lo), float(hi) if hi > lo else float(lo) + 1.0


def jupyter_schedule_main(callback: Callable[[], None]) -> None:
    """Run ``callback`` on the IPython kernel io_loop (never inline on a worker)."""
    from IPython import get_ipython

    get_ipython().kernel.io_loop.add_callback(callback)


def jupyter_push(*views) -> None:
    """Push pending Bokeh model changes for registered Panel notebook views."""
    from panel.io.notebook import push
    from panel.io.state import state

    pushed: set[int] = set()
    for view in views:
        for ref in list(getattr(view, "_models", {})):
            if ref not in state._views:
                continue
            _viewable, _root, doc, comm = state._views[ref]
            if comm is not None and id(doc) not in pushed:
                push(doc, comm)
                pushed.add(id(doc))


class LinkedAsyncView:
    """Two linked panes: sky tile at ``(t, f)`` and dynspec at fixed ``(l, m)``."""

    def __init__(
        self,
        sky_store: str,
        dyn_store: str | None = None,
        *,
        l_slice: slice,
        m_slice: slice,
        l_idx: int,
        m_idx: int,
        pol_idx: int = 0,
        max_workers: int = 4,
        paint_interval_ms: int = 100,
        schedule_main: Callable[[Callable[[], None]], None] = jupyter_schedule_main,
        push: Callable[..., None] = jupyter_push,
        start_periodic: Callable | None = None,
        on_slice: Callable[[int, int], None] | None = None,
    ) -> None:
        import panel as pn
        from bokeh.models import ColumnDataSource, LinearColorMapper, Rect
        from bokeh.plotting import figure

        self.sky_arr = zarr.open_consolidated(sky_store, mode="r")["SKY"]
        self.dyn_arr = zarr.open_consolidated(dyn_store or sky_store, mode="r")["SKY"]
        self.l_slice, self.m_slice = l_slice, m_slice
        self.l_idx, self.m_idx, self.pol_idx = l_idx, m_idx, pol_idx
        self.n_time, self.n_freq = self.dyn_arr.shape[:2]
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.paint_interval_ms = paint_interval_ms
        self._schedule_main = schedule_main
        self._push = push
        self._start_periodic = start_periodic or self._tornado_periodic
        self.on_slice = on_slice  # optional SkyWidget/other hook; called on io_loop after tile paint
        # When set, a dynspec tap updates only this hook (no Bokeh sky-tile read).
        self.on_tf_only: Callable[[int, int], None] | None = None
        self._done: queue.SimpleQueue = queue.SimpleQueue()
        self._remaining = 0
        self._periodic = None
        self._tile_generation = 0
        self.grid = np.full((self.n_time, self.n_freq), np.nan, dtype=np.float32)
        self.log: list[str] = []
        # Optional RA/Dec track (same arrays as bench_radec_track / accessor).
        self._track_l: np.ndarray | None = None
        self._track_m: np.ndarray | None = None
        self._track_visible: np.ndarray | None = None
        self._mode = "fixed"
        self._title_prefix = f"l={l_idx}, m={m_idx}"
        self._track_gen = 0

        n_l = l_slice.stop - l_slice.start
        n_m = m_slice.stop - m_slice.start
        self._tile_n_l = n_l
        self._tile_n_m = n_m
        self.sky_cmap = LinearColorMapper(palette="Inferno256")
        self.dyn_cmap = LinearColorMapper(palette="Viridis256", nan_color="#dddddd")
        self.sky_src = ColumnDataSource({"image": [np.zeros((n_m, n_l), dtype=np.float32)]})
        self.dyn_src = ColumnDataSource({"image": [self.grid.T.copy()]})
        self.sel_src = ColumnDataSource({"x": [0.5], "y": [0.5]})
        self.cross_src = ColumnDataSource(
            {"x": [l_idx - l_slice.start], "y": [m_idx - m_slice.start]}
        )

        self.sky_fig = figure(width=430, height=430, title="Sky tile", match_aspect=True)
        self.sky_fig.image("image", source=self.sky_src, x=0, y=0, dw=n_l, dh=n_m, color_mapper=self.sky_cmap)
        self.sky_fig.scatter("x", "y", source=self.cross_src, marker="cross", size=14, color="cyan")
        from bokeh.events import Tap
        from bokeh.models import TapTool

        self.dyn_fig = figure(
            width=640,
            height=430,
            title="Dynamic spectrum (loading…)",
            x_axis_label="time index",
            y_axis_label="frequency index",
            tools="pan,wheel_zoom,reset",  # TapTool added explicitly below
            active_drag=None,
            active_inspect=None,
        )
        self.dyn_fig.image(
            "image",
            source=self.dyn_src,
            x=0,
            y=0,
            dw=self.n_time,
            dh=self.n_freq,
            color_mapper=self.dyn_cmap,
        )
        self.dyn_fig.add_glyph(
            self.sel_src,
            Rect(x="x", y="y", width=1, height=1, fill_alpha=0, line_color="red", line_width=2),
        )
        tap_tool = TapTool()
        self.dyn_fig.add_tools(tap_tool)
        self.dyn_fig.toolbar.active_tap = tap_tool
        # Bind once on the figure; display ``dyn_pane`` only once (see notebook F).
        self.dyn_fig.on_event(Tap, self._on_tap)

        self.sky_pane = pn.pane.Bokeh(self.sky_fig)
        self.dyn_pane = pn.pane.Bokeh(self.dyn_fig)
        self.status = pn.pane.Markdown("Ready")
        # Dynspec is NOT in ``layout``: mounting it here then again in section F
        # breaks Jupyter tap callbacks on the shared Bokeh figure. Show
        # ``dyn_pane`` once (next to SkyWidget in section F).
        self.layout = pn.Column(self.status, self.sky_pane)

    def refresh_dynspec_display(self) -> None:
        """Push the current dynspec grid to a already-displayed ``dyn_pane``."""
        lo, hi = _finite_limits(self.grid)
        self.dyn_src.data = {"image": [self.grid.T.copy()]}
        self.dyn_cmap.update(low=lo, high=hi)
        self._push(self.dyn_pane)

    # ------------------------------------------------------------------ public

    def start(self, t: int = 0, f: int = 0) -> None:
        """Load the initial sky tile and begin the progressive fixed-pixel fill."""
        self._mode = "fixed"
        self._title_prefix = f"l={self.l_idx}, m={self.m_idx}"
        self._track_l = self._track_m = self._track_visible = None
        self._begin_fill(
            ZarrPixelReader(self.dyn_arr, pol_idx=self.pol_idx, l_idx=self.l_idx, m_idx=self.m_idx),
            t=t,
            f=f,
        )

    def start_tracked(
        self,
        l_indices: np.ndarray,
        m_indices: np.ndarray,
        visible: np.ndarray,
        *,
        t: int | None = None,
        f: int = 0,
        title: str = "tracked",
        sync_slice: bool = True,
    ) -> None:
        """Progressive fill at per-time pixels (same reader as ``bench_radec_track``).

        ``l_indices`` / ``m_indices`` / ``visible`` come from
        ``ds.radport._compute_pixel_track(ra, dec)``. When ``sync_slice`` is true,
        the first sky tile uses the first visible time (or ``t`` if given and
        visible). When it is false, only the heatmap values change.
        """
        self._mode = "tracked"
        self._title_prefix = title
        self._track_l = np.asarray(l_indices, dtype=int)
        self._track_m = np.asarray(m_indices, dtype=int)
        self._track_visible = np.asarray(visible, dtype=bool)
        if sync_slice and t is None:
            vis = np.flatnonzero(self._track_visible)
            t = int(vis[0]) if vis.size else 0
        elif t is None:
            t = 0
        self._begin_fill(
            ZarrTrackedPixelReader(
                self.dyn_arr,
                pol_idx=self.pol_idx,
                l_indices=self._track_l,
                m_indices=self._track_m,
                visible=self._track_visible,
            ),
            t=t,
            f=f,
            sync_slice=sync_slice,
        )

    def schedule_tracked_fill(
        self,
        ra: float,
        dec: float,
        track: Callable[[float, float], tuple[np.ndarray, np.ndarray, np.ndarray]],
        *,
        title: str | None = None,
        on_started: Callable[[], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        """Pixel-track off the io_loop, then :meth:`start_tracked` on the main thread.

        ``track(ra, dec)`` is the accessor pixel track (slow on a full store). A newer
        call drops an older result so the heatmap matches the latest request.
        """
        self._track_gen += 1
        gen = self._track_gen
        ra_f, dec_f = float(ra), float(dec)
        label = title or f"view {ra_f:.3f}, {dec_f:.3f}"

        def _work() -> None:
            try:
                l_idx, m_idx, visible = track(ra_f, dec_f)
            except Exception as exc:  # noqa: BLE001 — track() is caller code
                err = exc

                def _fail() -> None:
                    if gen != self._track_gen:
                        return
                    if on_error is not None:
                        on_error(err)
                    else:
                        self.log.append(f"heatmap track failed: {err}")

                self._schedule_main(_fail)
                return

            def _start() -> None:
                if gen != self._track_gen:
                    self.log.append("heatmap track dropped (newer request)")
                    return
                self.start_tracked(l_idx, m_idx, visible, title=label, sync_slice=False)
                if on_started is not None:
                    on_started()

            self._schedule_main(_start)

        self.pool.submit(_work)

    def _begin_fill(self, reader, *, t: int, f: int, sync_slice: bool = True) -> None:
        self.grid[:] = np.nan
        self._t_start = time.perf_counter()
        if sync_slice:
            self.request_tile(t, f)
        units = plan_fill_units(self.n_time, self.n_freq, *self.dyn_arr.chunks[:2])
        self._remaining = len(units)
        # Drop any leftover results from a previous start.
        while True:
            try:
                self._done.get_nowait()
            except queue.Empty:
                break
        for unit in units:
            fut = self.pool.submit(reader, unit)
            fut.add_done_callback(lambda fu, u=unit: self._done.put((u, fu)))
        if self._periodic is not None:
            self._periodic.stop()
        self._periodic = self._start_periodic(self._drain, self.paint_interval_ms)

    def _tile_slices_for_time(self, t: int) -> tuple[slice, slice, int, int]:
        """Return ``(l_slice, m_slice, cross_x, cross_y)`` for the sky pane at ``t``."""
        if self._mode == "tracked" and self._track_visible is not None and self._track_visible[t]:
            li = int(self._track_l[t])
            mi = int(self._track_m[t])
            half_l = self._tile_n_l // 2
            half_m = self._tile_n_m // 2
            n_l_full = self.sky_arr.shape[3]
            n_m_full = self.sky_arr.shape[4]
            l0 = max(0, min(li - half_l, n_l_full - self._tile_n_l))
            m0 = max(0, min(mi - half_m, n_m_full - self._tile_n_m))
            return slice(l0, l0 + self._tile_n_l), slice(m0, m0 + self._tile_n_m), li - l0, mi - m0
        return (
            self.l_slice,
            self.m_slice,
            self.l_idx - self.l_slice.start,
            self.m_idx - self.m_slice.start,
        )

    def request_tile(self, t: int, f: int) -> None:
        self._tile_generation += 1
        gen = self._tile_generation
        submitted = time.perf_counter()
        l_sl, m_sl, cx, cy = self._tile_slices_for_time(t)
        fut = self.pool.submit(read_sky_tile_zarr, self.sky_arr, t, f, self.pol_idx, l_sl, m_sl)
        fut.add_done_callback(
            lambda fu: self._schedule_main(
                lambda: self._paint_tile(fu, gen, t, f, submitted, cross=(cx, cy))
            )
        )

    def close(self) -> None:
        if self._periodic is not None:
            self._periodic.stop()
        self.pool.shutdown(wait=False, cancel_futures=True)

    # --------------------------------------------------------------- io_loop

    @staticmethod
    def _tornado_periodic(callback, interval_ms):
        from IPython import get_ipython
        from tornado.ioloop import PeriodicCallback

        loop = get_ipython().kernel.io_loop
        pc = PeriodicCallback(callback, interval_ms)
        loop.add_callback(pc.start)
        return pc

    def _drain(self) -> None:
        changed = False
        while True:
            try:
                unit, fut = self._done.get_nowait()
            except queue.Empty:
                break
            self._write(unit, fut.result())
            changed = True
        if not changed:
            return
        lo, hi = _finite_limits(self.grid)
        self.dyn_src.data = {"image": [self.grid.T.copy()]}
        self.dyn_cmap.update(low=lo, high=hi)
        elapsed = time.perf_counter() - self._t_start
        if self._remaining == 0:
            self.dyn_fig.title.text = (
                f"Dynamic spectrum @ {self._title_prefix} — complete {elapsed:.2f} s"
            )
            if self._periodic is not None:
                self._periodic.stop()
            self.log.append(f"dynspec complete {elapsed:.3f} s")
        else:
            self.dyn_fig.title.text = (
                f"Dynamic spectrum @ {self._title_prefix} — "
                f"{self._remaining} blocks pending ({elapsed:.2f} s)"
            )
        self._push(self.layout, self.dyn_pane)

    def _write(self, unit: FillUnit, block: np.ndarray) -> None:
        self.grid[unit.t0 : unit.t1, unit.f0 : unit.f1] = block
        self._remaining -= 1

    def _paint_tile(
        self, fut, gen: int, t: int, f: int, submitted: float, *, cross: tuple[int, int]
    ) -> None:
        if gen != self._tile_generation:
            return
        tile = fut.result()
        lo, hi = _finite_limits(tile)
        self.sky_src.data = {"image": [np.ascontiguousarray(tile.T)]}
        self.sky_cmap.update(low=lo, high=hi)
        self.cross_src.data = {"x": [cross[0]], "y": [cross[1]]}
        extra = ""
        if self._mode == "tracked" and self._track_l is not None:
            extra = f" pixel=({int(self._track_l[t])},{int(self._track_m[t])})"
        self.sky_fig.title.text = f"Sky tile t={t} f={f}{extra}"
        self.sel_src.data = {"x": [t + 0.5], "y": [f + 0.5]}
        msg = f"tile t={t} f={f} in {time.perf_counter() - submitted:.3f} s{extra}"
        self.status.object = msg
        self.log.append(msg)
        self._push(self.layout, self.sky_pane, self.dyn_pane)
        if self.on_slice is not None:
            try:
                self.on_slice(int(t), int(f))
            except Exception as exc:  # pragma: no cover - UI hook must not kill paint
                self.log.append(f"on_slice error: {exc}")

    def _on_tap(self, event) -> None:
        t = int(np.clip(np.floor(event.x), 0, self.n_time - 1))
        f = int(np.clip(np.floor(event.y), 0, self.n_freq - 1))
        # Move the red box immediately (do not wait for the async tile read).
        self.sel_src.data = {"x": [t + 0.5], "y": [f + 0.5]}
        self.log.append(f"dynspec tap → t={t} f={f}")
        self.status.object = f"dynspec tap → t={t} f={f}"
        self._push(self.dyn_pane, self.layout, self.status)
        if self.on_tf_only is not None:
            try:
                self.on_tf_only(t, f)
            except Exception as exc:  # noqa: BLE001 — UI hook must not kill the tap
                self.log.append(f"on_tf_only error: {exc}")
            return
        self.request_tile(t, f)
