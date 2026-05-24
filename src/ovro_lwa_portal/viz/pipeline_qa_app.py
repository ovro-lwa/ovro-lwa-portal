"""JupyterLab Panel application for per-day pipeline QA review."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import ipywidgets as widgets
import numpy as np
import pandas as pd
import panel as pn
import param
import xarray as xr
from astropy.coordinates import SkyCoord
import astropy.units as u
from astrowidget import SkyWidget
from bokeh.events import Tap
from bokeh.models import ColumnDataSource, FixedTicker, HoverTool
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

from ovro_lwa_portal.viz._imports import check_viz_deps
from ovro_lwa_portal.viz.pipeline_qa import (
    LogFn,
    convert_button_disabled,
    convert_button_label,
    convert_missing_zarr,
    day_summary_table,
    default_select_day,
    load_qa_datasets,
    qa_days,
    scan_coverage,
    zarr_status,
)

check_viz_deps()

logger = logging.getLogger(__name__)

ZENITH_L = 0.0
ZENITH_M = 0.0
ZENITH_PATCH_RADIUS = 10
DEFAULT_FREQ_IDX = 8
DEFAULT_FOV_DEG = 25.0
THERMAL_NOISE_GRID_COLS = 4


def _schedule_ipython_main(callback: Callable[[], None]) -> None:
    """Run callback on the IPython kernel event loop."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        kernel = getattr(ip, "kernel", None) if ip is not None else None
        io_loop = getattr(kernel, "io_loop", None) if kernel is not None else None
        if io_loop is not None:
            io_loop.add_callback(callback)
            return
    except Exception:
        pass
    callback()


def _run_on_main_thread(callback: Callable[[], None]) -> None:
    """Run a callback on the active notebook/UI thread when possible."""
    try:
        pn.state.execute(callback)
    except Exception:
        _schedule_ipython_main(callback)


def _push_panel_layout(*views: pn.viewable.Viewable) -> None:
    """Push Panel layout changes to the notebook frontend."""
    try:
        from panel.io.notebook import push, push_on_root
        from panel.io.state import state
    except ImportError:
        return

    pushed_docs: set[int] = set()
    for view in views:
        if not view._models:
            continue
        if not any(ref in state._views for ref in view._models):
            continue
        for ref in view._models:
            if ref not in state._views:
                continue
            _viewable, root, doc, comm = state._views[ref]
            if comm and "embedded" not in root.tags:
                doc_id = id(doc)
                if doc_id in pushed_docs:
                    continue
                try:
                    push(doc, comm)
                    pushed_docs.add(doc_id)
                except Exception as exc:
                    logger.debug("Panel notebook push failed: %s", exc, exc_info=True)
            else:
                try:
                    push_on_root(ref)
                except Exception as exc:
                    logger.debug("Panel push_on_root failed: %s", exc, exc_info=True)


ACTIVITY_LOG_HEIGHT_PX = 150


def _format_activity_log_display(text: str) -> str:
    """Format log lines for display (newest first)."""
    lines = [line for line in text.splitlines() if line]
    return "\n".join(reversed(lines)) if lines else ""


def _format_activity_log_html(text: str) -> str:
    """Render the activity log in a fixed-height scrollable block (newest first)."""
    import html

    display = _format_activity_log_display(text) or " "
    body = html.escape(display)
    return (
        f'<div style="height:{ACTIVITY_LOG_HEIGHT_PX}px;overflow-y:auto;overflow-x:hidden;'
        'border:1px solid #ccc;border-radius:4px;padding:8px;background:#fafafa;'
        'font-family:monospace;font-size:12px;white-space:pre-wrap;margin:0;'
        'box-sizing:border-box;">'
        f"{body}</div>"
    )


def _qa_tile_label(lst_hour: str, obs_date: str, n_subbands: int) -> str:
    subband_word = "subband" if n_subbands == 1 else "subbands"
    return f"**{lst_hour}** · {obs_date} · {n_subbands} {subband_word}"


def build_thermal_noise_grid(
    summary_df: pd.DataFrame,
    obs_date: str,
    *,
    n_cols: int = THERMAL_NOISE_GRID_COLS,
    open_full_size: Callable[[str, str], None] | None = None,
) -> pn.Column:
    """Grid of thermal-noise PNGs labeled by LST hour, date, and subband count."""
    if summary_df.empty:
        return pn.Column(
            pn.pane.Markdown("*No Wideband QA hours for this day.*"),
            sizing_mode="stretch_width",
        )

    tiles: list[pn.Column] = []
    for row in summary_df.itertuples(index=False):
        png_path = str(row.thermal_noise_png)
        lst_hour = str(row.lst_hour)
        n_subbands = int(row.n_subbands)
        label = _qa_tile_label(lst_hour, obs_date, n_subbands)

        if Path(png_path).is_file():
            img: Any = pn.pane.PNG(
                png_path,
                height=180,
                sizing_mode="scale_width",
            )
        else:
            img = pn.pane.Markdown("*thermal_noise_vs_subband.png missing*")

        footer: list[Any] = [pn.pane.Markdown(label)]
        if open_full_size and Path(png_path).is_file():
            title = f"thermal_noise_vs_subband — {obs_date} {lst_hour}"

            def _open(
                _event: Any,
                path: str = png_path,
                tile_title: str = title,
            ) -> None:
                open_full_size(path, tile_title)

            btn = pn.widgets.Button(name="Full size", width=90, height=28)
            btn.on_click(_open)
            footer.append(btn)

        tiles.append(
            pn.Column(
                img,
                *footer,
                width=280,
                margin=(6, 6),
            )
        )

    grid_rows: list[pn.Row] = []
    for start in range(0, len(tiles), n_cols):
        grid_rows.append(
            pn.Row(*tiles[start : start + n_cols], sizing_mode="stretch_width")
        )
    return pn.Column(*grid_rows, sizing_mode="stretch_width")


class ScrollLog:
    """Timestamped log buffer for the preparation pane."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def append(self, message: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self._lines.append(f"[{stamp}] {message}")

    def clear(self) -> None:
        self._lines.clear()

    @property
    def text(self) -> str:
        return "\n".join(self._lines)


def first_valid_time_idx(dataset: xr.Dataset, freq_idx: int) -> int:
    """First time index where SKY has finite data at the given frequency."""
    freq_idx = int(np.clip(freq_idx, 0, int(dataset.sizes["frequency"]) - 1))
    sky = dataset.SKY.isel(frequency=freq_idx, polarization=0)
    for t in range(sky.sizes["time"]):
        if np.isfinite(sky.isel(time=t).values).any():
            return t
    return 0


def default_stat_slice(
    stat_map: np.ndarray,
    dataset: xr.Dataset,
    *,
    preferred_freq_idx: int = DEFAULT_FREQ_IDX,
) -> tuple[int, int]:
    """Pick default (time, freq) from the zenith stat map with SKY fallback."""
    n_times, n_freqs = stat_map.shape
    if n_times == 0 or n_freqs == 0:
        return 0, 0

    pref = int(np.clip(preferred_freq_idx, 0, n_freqs - 1))
    finite_at_pref = np.flatnonzero(np.isfinite(stat_map[:, pref]))
    if finite_at_pref.size:
        return int(finite_at_pref[0]), pref

    finite = np.argwhere(np.isfinite(stat_map))
    if finite.size:
        time_idx, freq_idx = finite[0]
        return int(time_idx), int(freq_idx)

    return first_valid_time_idx(dataset, pref), pref


def zenith_lm_coord(dataset: xr.Dataset, time_idx: int) -> SkyCoord:
    """RA/Dec of the fixed (l=0, m=0) grid point for one time slice."""
    wcs = dataset.radport._get_wcs(time_idx=time_idx)
    l_idx, m_idx = dataset.radport.nearest_lm_idx(ZENITH_L, ZENITH_M)
    coord = wcs.pixel_to_world(l_idx, m_idx)
    return SkyCoord(ra=coord.ra, dec=coord.dec)


def _reduce_patch_rms(patch: np.ndarray) -> np.ndarray:
    """RMS = sqrt(mean(x^2)) for each frequency plane (Stokes V about zero)."""
    patch_arr = np.asarray(patch)
    values = np.full(patch_arr.shape[0], np.nan, dtype=np.float64)
    for fi in range(patch_arr.shape[0]):
        finite = patch_arr[fi][np.isfinite(patch_arr[fi])]
        if finite.size:
            values[fi] = np.sqrt(np.mean(finite**2))
    return values


def compute_zenith_std_map(dataset: xr.Dataset, radius: int = ZENITH_PATCH_RADIUS) -> np.ndarray:
    """Spatial STD in a fixed (l=0, m=0) patch for each (time, frequency) cell."""
    result = dataset.radport.patch_statistic(
        l=ZENITH_L,
        m=ZENITH_M,
        statistic="std",
        radius=radius,
    )
    return np.asarray(result.stat_map.values, dtype=np.float64)


def compute_zenith_rms_map(dataset: xr.Dataset, radius: int = ZENITH_PATCH_RADIUS) -> np.ndarray:
    """Spatial RMS in a fixed (l=0, m=0) patch for each (time, frequency) cell."""
    l_idx, m_idx = dataset.radport.nearest_lm_idx(ZENITH_L, ZENITH_M)
    n_times = int(dataset.sizes["time"])
    n_freqs = int(dataset.sizes["frequency"])
    stat_map = np.full((n_times, n_freqs), np.nan, dtype=np.float64)
    vis_times, patches = dataset.radport._extract_tracked_patch_cubes(
        l_indices=np.full(n_times, l_idx, dtype=int),
        m_indices=np.full(n_times, m_idx, dtype=int),
        visible=np.ones(n_times, dtype=bool),
        var="SKY",
        pol=0,
        radius=radius,
    )
    for ti, patch in zip(vis_times, patches, strict=True):
        stat_map[int(ti)] = _reduce_patch_rms(np.asarray(patch))
    return stat_map


def _heatmap_cell_center(idx: int) -> float:
    """Data-space coordinate at the center of a heatmap cell."""
    return idx + 0.5


def _heatmap_index_from_coord(coord: float, n: int) -> int:
    """Map a tap/data coordinate to a zero-based cell index."""
    if n <= 0:
        return 0
    return int(np.clip(int(np.floor(float(coord))), 0, n - 1))


def _heatmap_axis_ticks(n: int, *, max_ticks: int = 24) -> tuple[list[float], dict[float, str]]:
    """Tick positions at cell centers with integer index labels."""
    if n <= 0:
        return [], {}
    step = 1 if n <= max_ticks else int(np.ceil(n / max_ticks))
    indices = range(0, n, step)
    ticks = [_heatmap_cell_center(i) for i in indices]
    labels = {tick: str(i) for tick, i in zip(ticks, indices, strict=True)}
    return ticks, labels


def _build_heatmap_hover_source(stat_map: np.ndarray) -> ColumnDataSource:
    """ColumnDataSource for per-cell hover tooltips on the zenith heatmap."""
    n_times, n_freqs = stat_map.shape
    time_idx, freq_idx = np.meshgrid(
        np.arange(n_times, dtype=int),
        np.arange(n_freqs, dtype=int),
        indexing="ij",
    )
    return ColumnDataSource(
        data={
            "x": time_idx.ravel() + 0.5,
            "y": freq_idx.ravel() + 0.5,
            "time_idx": time_idx.ravel(),
            "freq_idx": freq_idx.ravel(),
            "value": stat_map.astype(float, copy=False).ravel(),
        }
    )


class _ZenithHeatmapSelector:
    """Bokeh zenith heatmap embedded in Panel (same JS stack as the rest of the app)."""

    def __init__(
        self,
        stat_map: np.ndarray,
        *,
        metric_label: str,
        time_idx: int,
        freq_idx: int,
        on_select: Callable[[int, int], None],
    ) -> None:
        self._stat_map = stat_map
        self._metric_label = metric_label
        self._time_idx = time_idx
        self._freq_idx = freq_idx
        self._on_select = on_select
        self._plot = self._build_plot()
        self.pane = pn.pane.Bokeh(self._plot, width=520, height=420, sizing_mode="fixed")

    def _clim(self) -> tuple[float, float] | None:
        finite = self._stat_map[np.isfinite(self._stat_map)]
        if finite.size == 0:
            return None
        lo, hi = np.percentile(finite, [2, 98])
        return float(lo), float(hi)

    def _image_uint8(self) -> np.ndarray:
        data = self._stat_map.T.astype(np.float64, copy=True)
        clim = self._clim()
        if clim is not None:
            data = np.clip(data, clim[0], clim[1])
            span = clim[1] - clim[0]
            if span > 0:
                data = (data - clim[0]) / span
            else:
                data = np.zeros_like(data)
        data = np.nan_to_num(data, nan=0.0)
        return (data * 255).astype(np.uint8)

    def _build_plot(self):
        n_times, n_freqs = self._stat_map.shape
        plot = figure(
            width=520,
            height=400,
            title=f"Zenith patch {self._metric_label} (hover for values; click to set slice)",
            x_range=(0, n_times),
            y_range=(0, n_freqs),
            tools="pan,wheel_zoom,reset,tap",
            active_drag="pan",
            active_tap="tap",
        )
        plot.image(
            image=[self._image_uint8()],
            x=0,
            y=0,
            dw=n_times,
            dh=n_freqs,
            palette=Viridis256,
            level="image",
        )
        hover_renderer = plot.rect(
            x="x",
            y="y",
            width=1,
            height=1,
            source=_build_heatmap_hover_source(self._stat_map),
            fill_alpha=0,
            line_alpha=0,
            hover_fill_alpha=0,
            hover_line_alpha=0,
        )
        plot.add_tools(
            HoverTool(
                renderers=[hover_renderer],
                tooltips=[
                    ("Time idx", "@time_idx"),
                    ("Freq idx", "@freq_idx"),
                    (self._metric_label, "@value{0.3g}"),
                ],
            )
        )
        self._marker = plot.scatter(
            x=[_heatmap_cell_center(self._time_idx)],
            y=[_heatmap_cell_center(self._freq_idx)],
            size=18,
            marker="cross",
            line_color="cyan",
            fill_color=None,
            line_width=2,
        )
        x_ticks, x_labels = _heatmap_axis_ticks(n_times)
        y_ticks, y_labels = _heatmap_axis_ticks(n_freqs)
        plot.xaxis.ticker = FixedTicker(ticks=x_ticks)
        plot.yaxis.ticker = FixedTicker(ticks=y_ticks)
        plot.xaxis.major_label_overrides = x_labels
        plot.yaxis.major_label_overrides = y_labels
        plot.xaxis.axis_label = "Time index"
        plot.yaxis.axis_label = "Frequency index"
        plot.on_event(Tap, self._on_tap)
        return plot

    def _on_tap(self, event: Tap) -> None:
        if event.x is None or event.y is None:
            return
        n_times, n_freqs = self._stat_map.shape
        time_idx = _heatmap_index_from_coord(event.x, n_times)
        freq_idx = _heatmap_index_from_coord(event.y, n_freqs)
        self._on_select(time_idx, freq_idx)

    def set_slice(self, time_idx: int, freq_idx: int, *, push: bool = True) -> None:
        self._time_idx = time_idx
        self._freq_idx = freq_idx
        cx = _heatmap_cell_center(time_idx)
        cy = _heatmap_cell_center(freq_idx)
        self._marker.data_source.patch({"x": [(0, cx)], "y": [(0, cy)]})
        if push:
            _run_on_main_thread(lambda: _push_panel_layout(self.pane))

    def set_data(self, stat_map: np.ndarray, *, time_idx: int, freq_idx: int) -> None:
        self._stat_map = stat_map
        self._time_idx = time_idx
        self._freq_idx = freq_idx
        self._plot = self._build_plot()
        self.pane.object = self._plot

    def dispose(self) -> None:
        return


class ZenithReviewPanel(param.Parameterized):
    """Heatmap + SkyWidget zenith review with Param-driven slice selection."""

    time_idx = param.Integer(default=0, doc="Selected time index from the heatmap.")
    freq_idx = param.Integer(default=0, doc="Selected frequency index from the heatmap.")
    stokes_label = param.String(default="", doc="Stokes parameter label (I or V).")
    metric_label = param.String(default="", doc="Zenith patch metric label (STD or RMS).")

    def __init__(
        self,
        dataset: xr.Dataset,
        stat_map: np.ndarray,
        *,
        stokes_label: str,
        metric_label: str,
    ) -> None:
        self._dataset = dataset
        self._stat_map = stat_map
        self._n_times = int(dataset.sizes["time"])
        self._n_freqs = int(dataset.sizes["frequency"])
        self._sky_widget: SkyWidget | None = None
        self._push_root: Callable[[], None] | None = None

        default_time, default_freq = default_stat_slice(stat_map, dataset)
        super().__init__(
            time_idx=default_time,
            freq_idx=default_freq,
            stokes_label=stokes_label,
            metric_label=metric_label,
        )
        self.param.time_idx.bounds = (0, max(0, self._n_times - 1))
        self.param.freq_idx.bounds = (0, max(0, self._n_freqs - 1))

        self._heatmap = _ZenithHeatmapSelector(
            stat_map,
            metric_label=metric_label,
            time_idx=self.time_idx,
            freq_idx=self.freq_idx,
            on_select=self._select_slice,
        )

        self._header = pn.pane.Markdown(
            f"**Stokes {stokes_label}** — click the heatmap or use the sliders to choose "
            "time and frequency. The matching sky view appears directly below."
        )
        self._status_pane = pn.pane.Markdown(self._slice_status, sizing_mode="stretch_width")
        self._time_slider = pn.widgets.IntSlider.from_param(
            self.param.time_idx,
            name="Time index",
        )
        self._freq_slider = pn.widgets.IntSlider.from_param(
            self.param.freq_idx,
            name="Frequency index",
        )
        self._layout = pn.Column(
            self._header,
            self._status_pane,
            self._time_slider,
            self._freq_slider,
            self._heatmap.pane,
            width=ZENITH_REVIEW_COLUMN_WIDTH,
            sizing_mode="fixed",
            margin=(0, ZENITH_REVIEW_COLUMN_GAP, 0, 0),
        )
        self.param.watch(self._on_slice_changed, ["time_idx", "freq_idx"])
        self._on_slice_changed()

    def set_push_root(self, callback: Callable[[], None] | None) -> None:
        """Register the displayed dashboard root push callback (required in JupyterLab)."""
        self._push_root = callback

    @param.depends("time_idx", "freq_idx")
    def _slice_status(self) -> str:
        """Slice summary shown above the heatmap (reactive via Param + Panel)."""
        coord = zenith_lm_coord(self._dataset, self.time_idx)
        metric_val = float(self._stat_map[self.time_idx, self.freq_idx])
        freq_mhz = float(self._dataset.frequency.values[self.freq_idx]) / 1e6
        return (
            f"**Stokes {self.stokes_label} zenith (l=0, m=0)** | time={self.time_idx}, "
            f"freq={self.freq_idx} ({freq_mhz:.1f} MHz)"
            f" | center RA={coord.ra.to_string(unit=u.hour, precision=1)}, "
            f"Dec={coord.dec.to_string(unit=u.deg, precision=1)}"
            f" | patch {self.metric_label}={metric_val:.3g}"
        )

    @staticmethod
    def _bind_sky_dataset(widget: SkyWidget, dataset: xr.Dataset) -> None:
        widget.set_dataset(dataset, max_size=1024)

    def mount_sky(self, dataset: xr.Dataset | None = None) -> SkyWidget:
        """Create SkyWidget for native Jupyter display (not embedded in the Bokeh layout)."""
        if self._sky_widget is not None:
            return self._sky_widget

        self._sky_widget = SkyWidget()
        self._sky_widget.colormap = "inferno"
        self._sky_widget.background_survey = ""
        self._sky_widget.invert_horizontal_pan = True
        self._sky_widget.layout = widgets.Layout(
            width=f"{ZENITH_REVIEW_COLUMN_WIDTH}px",
            min_width=f"{ZENITH_REVIEW_COLUMN_WIDTH}px",
            max_width=f"{ZENITH_REVIEW_COLUMN_WIDTH}px",
            height=f"{ZENITH_REVIEW_COLUMN_WIDTH}px",
            min_height=f"{ZENITH_REVIEW_COLUMN_WIDTH}px",
        )
        bind_ds = dataset if dataset is not None else self._dataset
        self._bind_sky_dataset(self._sky_widget, bind_ds)
        self._on_slice_changed()
        return self._sky_widget

    def _select_slice(self, time_idx: int, freq_idx: int) -> None:
        """Heatmap tap handler: update Param slice indices on the notebook UI thread."""
        time_idx = int(np.clip(time_idx, 0, self._n_times - 1))
        freq_idx = int(np.clip(freq_idx, 0, self._n_freqs - 1))

        def _apply() -> None:
            with param.parameterized.batch_call_watchers(self):
                self.time_idx = time_idx
                self.freq_idx = freq_idx

        _run_on_main_thread(_apply)

    def _on_slice_changed(self, *_events: param.parameterized.Event) -> None:
        """Sync heatmap crosshair and SkyWidget when slice Param values change."""
        self._heatmap.set_slice(self.time_idx, self.freq_idx, push=False)
        if self._sky_widget is not None:
            coord = zenith_lm_coord(self._dataset, self.time_idx)
            self._sky_widget.update_slice(
                self.time_idx,
                self.freq_idx,
                center=coord,
                fov=DEFAULT_FOV_DEG * u.deg,
                percentile_low=2,
                percentile_high=98,
            )
            send_state = getattr(self._sky_widget, "send_state", None)
            if callable(send_state):
                send_state()
        self._push_slice_ui()

    def _push_slice_ui(self) -> None:
        """Push the displayed dashboard so nested zenith controls update in JupyterLab."""
        if self._push_root is not None:
            self._push_root()
        else:
            _push_panel_layout(self._layout, self._heatmap.pane)

    @property
    def layout(self) -> pn.Column:
        return self._layout

    @property
    def heatmap_column(self) -> pn.Column:
        """Header, status, sliders, and heatmap for side-by-side zenith review layout."""
        return self._layout

    def dispose(self) -> None:
        self._heatmap.dispose()
        self._sky_widget = None
        self._push_root = None


@dataclass(frozen=True)
class _StokesSectionSpec:
    """Configuration for one Stokes I/V zenith review section."""

    stokes: str
    heading: str
    metric_label: str
    waiting_message: str
    missing_zarr_message: str
    compute_stat_map: Callable[[xr.Dataset], np.ndarray]


_STOKES_SECTIONS: tuple[_StokesSectionSpec, ...] = (
    _StokesSectionSpec(
        stokes="I",
        heading="## Stokes I — zenith patch STD",
        metric_label="STD",
        waiting_message="*Load a day with QA Zarr to enable Stokes I review.*",
        missing_zarr_message="*Convert FITS → Zarr to enable Stokes I review.*",
        compute_stat_map=compute_zenith_std_map,
    ),
    _StokesSectionSpec(
        stokes="V",
        heading="## Stokes V — zenith patch RMS",
        metric_label="RMS",
        waiting_message="*Load a day with QA Zarr to enable Stokes I review.*",
        missing_zarr_message="*Run **Convert** to build the Stokes V Zarr and enable review.*",
        compute_stat_map=compute_zenith_rms_map,
    ),
)

_NO_QA_ZARR_MESSAGE = (
    "*No QA Zarr stores for this day. Click **Convert FITS → Zarr** to build them.*"
)

_ZENITH_PLACEHOLDER = (
    "*Select an observation day to build Stokes I/V heatmaps and sky views.*"
)

_ZENITH_SKY_PLACEHOLDER = (
    "*Sky views appear directly below the heatmaps after a day is selected (when QA Zarr exists).*"
)

ZENITH_REVIEW_COLUMN_WIDTH = 520
ZENITH_REVIEW_COLUMN_GAP = 8
ZENITH_REVIEW_ROW_WIDTH = 2 * ZENITH_REVIEW_COLUMN_WIDTH + ZENITH_REVIEW_COLUMN_GAP
# SkyWidget height + label + ipywidgets padding (must fit inside the Panel IPyWidget pane).
ZENITH_SKY_PANE_HEIGHT = ZENITH_REVIEW_COLUMN_WIDTH + 64
ZENITH_SKY_ROW_MARGIN = (0, 0, 32, 0)


class _SkyWidgetHost:
    """Embedded ipywidgets sky columns inside the Panel layout (JupyterLab only)."""

    def __init__(self) -> None:
        self._containers: dict[str, widgets.VBox] = {
            spec.stokes: widgets.VBox(
                children=[widgets.HTML(_ZENITH_SKY_PLACEHOLDER)],
                layout=self._container_layout(),
            )
            for spec in _STOKES_SECTIONS
        }
        self._panes: dict[str, pn.pane.IPyWidget] = {}
        for index, spec in enumerate(_STOKES_SECTIONS):
            column_margin = (
                (0, ZENITH_REVIEW_COLUMN_GAP, 0, 0) if index == 0 else (0, 0, 0, 0)
            )
            self._panes[spec.stokes] = pn.pane.IPyWidget(
                self._containers[spec.stokes],
                width=ZENITH_REVIEW_COLUMN_WIDTH,
                height=ZENITH_SKY_PANE_HEIGHT,
                sizing_mode="fixed",
                margin=column_margin,
            )
        self._panel_row = pn.Row(
            *[self._panes[spec.stokes] for spec in _STOKES_SECTIONS],
            sizing_mode="fixed",
            width=ZENITH_REVIEW_ROW_WIDTH,
            margin=ZENITH_SKY_ROW_MARGIN,
        )

    @staticmethod
    def _container_layout() -> widgets.Layout:
        col_width = f"{ZENITH_REVIEW_COLUMN_WIDTH}px"
        return widgets.Layout(
            width=col_width,
            min_width=col_width,
            overflow="hidden",
        )

    @property
    def panel_row(self) -> pn.Row:
        """Panel row aligned with the zenith heatmap columns above."""
        return self._panel_row

    @property
    def widget(self) -> widgets.HBox:
        """Native ipywidgets row for the sky columns."""
        return widgets.HBox(
            list(self._containers.values()),
            layout=widgets.Layout(
                width=f"{ZENITH_REVIEW_ROW_WIDTH}px",
                min_width=f"{ZENITH_REVIEW_ROW_WIDTH}px",
            ),
        )

    def show_placeholder(self) -> None:
        """Render the waiting message in each sky column."""
        for container in self._containers.values():
            container.children = [widgets.HTML(_ZENITH_SKY_PLACEHOLDER)]

    def reset(self) -> None:
        self.show_placeholder()

    def mount(self, panels: dict[str, "ZenithReviewPanel | None"]) -> None:
        """Create SkyWidgets and show them in the matching columns."""
        for spec in _STOKES_SECTIONS:
            container = self._containers[spec.stokes]
            panel = panels.get(spec.stokes)
            if panel is None:
                container.children = [widgets.HTML(_ZENITH_SKY_PLACEHOLDER)]
                continue
            sky = panel.mount_sky()
            container.children = [
                widgets.HTML(f"<strong>Stokes {spec.stokes} sky view</strong>"),
                sky,
            ]


class _StokesReviewHolder:
    """Builds Stokes I/V review columns when zenith panels are loaded."""

    def __init__(self) -> None:
        self._panels: dict[str, ZenithReviewPanel | None] = {
            spec.stokes: None for spec in _STOKES_SECTIONS
        }

    @staticmethod
    def _build_section(spec: _StokesSectionSpec, content: pn.viewable.Viewable) -> pn.Column:
        return pn.Column(
            pn.pane.Markdown(spec.heading),
            content,
            margin=(0, 0, 16, 0),
            sizing_mode="stretch_width",
        )

    def _dispose_panels(self) -> None:
        for stokes, panel in self._panels.items():
            if panel is not None:
                panel.dispose()
                self._panels[stokes] = None

    def build_section_contents(
        self,
        datasets: dict[str, xr.Dataset],
        log: LogFn,
        flush: Callable[[], None] | None = None,
    ) -> dict[str, pn.viewable.Viewable]:
        """Build Stokes I/V review panel content for stable zenith section slots."""
        self._dispose_panels()
        return {
            spec.stokes: self._build_section_content_for_spec(
                spec,
                datasets,
                log,
                flush=flush,
            )
            for spec in _STOKES_SECTIONS
        }

    def build_column(
        self,
        datasets: dict[str, xr.Dataset],
        log: LogFn,
        flush: Callable[[], None] | None = None,
    ) -> pn.Column:
        """Compute stat maps and assemble heatmaps in a side-by-side row."""
        contents = self.build_section_contents(datasets, log, flush=flush)
        heatmap_columns = [
            self._build_section(spec, contents[spec.stokes]) for spec in _STOKES_SECTIONS
        ]
        return pn.Column(
            pn.Row(*heatmap_columns, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    def _build_section_content_for_spec(
        self,
        spec: _StokesSectionSpec,
        datasets: dict[str, xr.Dataset],
        log: LogFn,
        *,
        flush: Callable[[], None] | None = None,
    ) -> pn.viewable.Viewable:
        if spec.stokes not in datasets:
            return pn.pane.Markdown(spec.missing_zarr_message)

        dataset = datasets[spec.stokes]
        log(
            f"Computing Stokes {spec.stokes} zenith patch {spec.metric_label} "
            f"(radius={ZENITH_PATCH_RADIUS}, ~{2 * ZENITH_PATCH_RADIUS + 1} px)…"
        )
        if flush is not None:
            flush()
        started = time.perf_counter()
        stat_map = spec.compute_stat_map(dataset)
        log(
            f"Stokes {spec.stokes} zenith {spec.metric_label} map ready "
            f"({time.perf_counter() - started:.1f}s)."
        )
        if flush is not None:
            flush()

        log(f"Building Stokes {spec.stokes} review panel…")
        if flush is not None:
            flush()
        widget_started = time.perf_counter()
        panel = ZenithReviewPanel(
            dataset,
            stat_map,
            stokes_label=spec.stokes,
            metric_label=spec.metric_label,
        )
        self._panels[spec.stokes] = panel
        log(
            f"Stokes {spec.stokes} review panel ready "
            f"({time.perf_counter() - widget_started:.1f}s)."
        )
        if flush is not None:
            flush()
        return panel.heatmap_column

    def build_no_zarr_contents(self) -> dict[str, pn.viewable.Viewable]:
        """Placeholder content when the selected day has no QA Zarr stores."""
        self._dispose_panels()
        return {
            spec.stokes: pn.pane.Markdown(spec.missing_zarr_message)
            for spec in _STOKES_SECTIONS
        }

    def build_no_zarr_column(self) -> pn.Column:
        """Column shown when the selected day has no QA Zarr stores."""
        contents = self.build_no_zarr_contents()
        columns = [
            self._build_section(spec, contents[spec.stokes]) for spec in _STOKES_SECTIONS
        ]
        return pn.Column(
            pn.Row(*columns, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    def dispose(self) -> None:
        self._dispose_panels()


def build_zenith_review_panel(
    dataset: xr.Dataset,
    stat_map: np.ndarray,
    *,
    stokes_label: str,
    metric_label: str,
) -> pn.Column:
    """Build a one-shot zenith review panel (prefer :class:`ZenithReviewPanel` for reuse)."""
    panel = ZenithReviewPanel(
        dataset,
        stat_map,
        stokes_label=stokes_label,
        metric_label=metric_label,
    )
    panel.mount_sky(dataset)
    return panel.layout


def build_stokes_review_column(
    datasets: dict[str, xr.Dataset],
    log: LogFn,
) -> pn.Column:
    """Stack Stokes I and V zenith review panels vertically."""
    holder = _StokesReviewHolder()
    return holder.build_column(datasets, log)


@dataclass
class _DayLoadPayload:
    """Summary data fetched when the user selects an observation day."""

    select_day: str
    summary_df: pd.DataFrame


class _LoadSuperseded(Exception):
    """Raised when a newer day load replaces this one."""


class PipelineQAApp(param.Parameterized):
    """Interactive Panel dashboard for one-day pipeline QA."""

    select_day = param.Selector(default=None, objects=[])
    log_text = param.String(default="")
    error_message = param.String(default="")
    scanning = param.Boolean(default=True)
    converting = param.Boolean(default=False)
    loading_day = param.Boolean(default=False)
    loading_zenith = param.Boolean(default=False)

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._coverage: pd.DataFrame = pd.DataFrame()
        self._scroll_log = ScrollLog()
        self._summary_df: pd.DataFrame = pd.DataFrame()
        self._loaded_day: str | None = None
        self._stokes_review = _StokesReviewHolder()
        self._sky_host = _SkyWidgetHost()
        self._zenith_banner = pn.pane.Markdown(_ZENITH_PLACEHOLDER)
        self._zenith_section_content: dict[str, pn.Column] = {}
        self._zenith_sections: dict[str, pn.Column] = {}
        for index, spec in enumerate(_STOKES_SECTIONS):
            column_margin = (
                (0, ZENITH_REVIEW_COLUMN_GAP, 0, 0) if index == 0 else (0, 0, 0, 0)
            )
            content = pn.Column(
                pn.pane.Markdown(spec.waiting_message),
                sizing_mode="fixed",
                width=ZENITH_REVIEW_COLUMN_WIDTH,
            )
            self._zenith_section_content[spec.stokes] = content
            self._zenith_sections[spec.stokes] = pn.Column(
                pn.pane.Markdown(spec.heading),
                content,
                width=ZENITH_REVIEW_COLUMN_WIDTH,
                sizing_mode="fixed",
                margin=column_margin,
            )
        self._zenith_review_row = pn.Row(
            *[self._zenith_sections[spec.stokes] for spec in _STOKES_SECTIONS],
            sizing_mode="stretch_width",
        )
        self._zenith_loading_spinner = pn.indicators.LoadingSpinner(
            value=False,
            size=40,
            name="Loading zenith panels",
        )
        self._zenith_loading_row = pn.Row(
            self._zenith_loading_spinner,
            pn.pane.Markdown("Loading zenith review panels…"),
            visible=False,
            sizing_mode="stretch_width",
        )
        self._zenith_slot = pn.Column(
            self._zenith_banner,
            self._zenith_loading_row,
            self._zenith_review_row,
            sizing_mode="stretch_width",
        )
        self._zenith_load_button = pn.widgets.Button(
            name="Reload zenith panels",
            button_type="default",
            disabled=True,
        )
        self._zenith_load_button.on_click(self._on_zenith_load_click)
        self._qa_grid = pn.Column(
            pn.pane.Markdown("*Select an observation day to build the thermal-noise QA grid.*"),
            sizing_mode="stretch_width",
        )
        self._day_selector = pn.widgets.Select.from_param(
            self.param.select_day,
            name="Observation day",
            width=220,
        )
        self.param.watch(self._on_select_day_changed, "select_day")
        self._convert_button = pn.widgets.Button(
            name="Convert FITS → Zarr",
            button_type="primary",
            disabled=True,
        )
        self._convert_button.on_click(self._on_convert_click)
        self._close_modal_button = pn.widgets.Button(name="Close", button_type="default")
        self._close_modal_button.on_click(lambda _event: self._close_modal())
        self._modal_container = pn.Column(sizing_mode="stretch_width")
        self._log_pane = pn.pane.HTML(
            _format_activity_log_html(""),
            sizing_mode="stretch_width",
            height=ACTIVITY_LOG_HEIGHT_PX,
        )
        self._error_alert = pn.panel(
            self._error_alert_view,
            sizing_mode="stretch_width",
        )
        self._layout: pn.Column | None = None
        self._scan_started = False
        self._load_seq = 0
        self._active_datasets: dict[str, xr.Dataset] = {}

    @property
    def sky_widgets(self) -> widgets.HBox:
        """Embedded ipywidgets row for the sky columns."""
        return self._sky_host.widget

    @property
    def busy(self) -> bool:
        """True while scan, day load, zenith load, or conversion is in progress."""
        return bool(
            self.scanning or self.loading_day or self.loading_zenith or self.converting
        )

    @param.depends("log_text", watch=True)
    def _sync_log_pane(self) -> None:
        self._log_pane.object = _format_activity_log_html(self.log_text)

    @param.depends("error_message")
    def _error_alert_view(self) -> pn.viewable.Viewable:
        if not self.error_message:
            return pn.Spacer(height=0, width=0)
        return pn.pane.Alert(
            self.error_message,
            alert_type="danger",
            sizing_mode="stretch_width",
        )

    @param.depends("loading_zenith", watch=True)
    def _sync_zenith_loading_indicator(self) -> None:
        self._zenith_loading_row.visible = self.loading_zenith
        self._zenith_loading_spinner.value = self.loading_zenith

    @param.depends(
        "select_day",
        "scanning",
        "loading_day",
        "loading_zenith",
        "converting",
        watch=True,
    )
    def _sync_action_controls(self) -> None:
        """Keep day selector and action buttons aligned with Param state."""
        self._day_selector.disabled = self.busy
        self._sync_convert_button()
        self._sync_zenith_button()

    def _execute(self, callback: Any) -> None:
        try:
            pn.state.execute(callback)
        except Exception:
            callback()

    def _push_panel_roots(self) -> None:
        """Push the dashboard layout to the notebook frontend."""
        if self._layout is not None:
            _push_panel_layout(self._layout)

    def _reset_zenith_sections(self, *, banner: str | None = None) -> None:
        """Restore zenith placeholders without replacing the displayed layout tree."""
        self._stokes_review.dispose()
        self._sky_host.reset()
        self._zenith_banner.object = banner if banner is not None else _ZENITH_PLACEHOLDER
        for spec in _STOKES_SECTIONS:
            self._zenith_section_content[spec.stokes].objects = [
                pn.pane.Markdown(spec.waiting_message),
            ]

    def _mount_zenith_sections(
        self,
        section_contents: dict[str, pn.viewable.Viewable],
        *,
        banner: str,
    ) -> None:
        """Populate stable zenith section slots and mount sky widgets below."""
        self._zenith_banner.object = banner
        push_root = lambda: self._execute(self._push_panel_roots)
        for spec in _STOKES_SECTIONS:
            self._zenith_section_content[spec.stokes].objects = [
                section_contents[spec.stokes],
            ]
            panel = self._stokes_review._panels.get(spec.stokes)
            if panel is not None:
                panel.set_push_root(push_root)
        self._sky_host.mount(self._stokes_review._panels)
        self._execute(self._push_zenith_root)

    def _sync_day_selector(self, days: list[str], value: str | None) -> None:
        """Update day options and selection from scan results."""
        with param.parameterized.batch_call_watchers(self):
            self.param.select_day.objects = days
            self.select_day = value

    def _on_select_day_changed(self, event: param.parameterized.Event) -> None:
        new_day = event.new
        if new_day is None:
            return
        old_day = event.old
        if old_day not in (None, param.Undefined) and old_day == new_day:
            return
        if self._loaded_day != new_day:
            self._release_active_datasets()
            self._reset_zenith_sections()

            def _push() -> None:
                self._push_panel_roots()

            self._execute(_push)
        self._begin_load_day()

    def _on_zenith_load_click(self, _event: Any) -> None:
        self._begin_zenith_load()

    def _sync_zenith_button(self) -> None:
        """Enable manual zenith reload once the selected day has been loaded."""
        if self.select_day is None:
            self._zenith_load_button.disabled = True
            return
        status = zarr_status(self.select_day)
        has_zarr = status["I"] or status["V"]
        day_ready = self._loaded_day == self.select_day
        self._zenith_load_button.disabled = (
            self.busy or not day_ready or not has_zarr
        )

    def _sync_log(self, *, defer: bool = False) -> None:
        text = self._scroll_log.text
        if defer:

            def _push() -> None:
                self.log_text = text

            self._execute(_push)
        else:
            self.log_text = text

    def _log(self, message: str, *, sync: bool = True, defer: bool = False) -> None:
        self._scroll_log.append(message)
        if sync:
            self._sync_log(defer=defer)

    def _flush_log(self) -> None:
        """Push pending log lines to the UI before a blocking call."""
        self._sync_log()

    def _release_active_datasets(self) -> None:
        """Close cached datasets before opening new stores."""
        for ds in self._active_datasets.values():
            close = getattr(ds, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        self._active_datasets.clear()

    def _clear_error(self) -> None:
        self.error_message = ""

    def _log_error(self, message: str) -> None:
        """Record an error in the log pane and show an alert."""
        self.error_message = message
        self._log(f"ERROR: {message}")

    def _sync_convert_button(self) -> None:
        """Sync convert button label, color, and disabled state with the selected day."""
        if self.select_day is None:
            self._convert_button.disabled = True
            self._convert_button.button_type = "default"
            return
        status = zarr_status(self.select_day)
        zarr_complete = status["I"] and status["V"]
        self._convert_button.name = convert_button_label(status)
        self._convert_button.disabled = convert_button_disabled(
            status,
            converting=self.converting,
        ) or self.busy
        self._convert_button.button_type = "default" if zarr_complete or self.busy else "primary"

    def _start_initial_scan(self) -> None:
        self.scanning = True
        self._scroll_log.clear()
        self._sync_log()
        self._log("Scanning pipeline tree…", sync=False)
        self._sync_log()

        def _run() -> None:
            try:
                coverage = scan_coverage()
                days = qa_days(coverage)
                default_day = default_select_day(coverage)

                def _apply() -> None:
                    self._coverage = coverage
                    self.scanning = False
                    if not days:
                        self._log_error("No Wideband QA days found under the pipeline root.")
                        self._sync_day_selector([], None)
                    else:
                        self._clear_error()
                        self._sync_day_selector(days, default_day)
                        self._log(
                            f"Found {len(days)} Wideband QA day(s). "
                            f"Loading {default_day}…"
                        )
                    self._sync_log()

                self._execute(_apply)
            except Exception as exc:
                def _fail() -> None:
                    self.scanning = False
                    self._log_error(f"Scan failed: {exc}")

                self._execute(_fail)

        threading.Thread(target=_run, daemon=True).start()

    def _begin_load_day(self) -> None:
        """Load thermal-noise summary and QA grid for the selected day."""
        if self.converting or self.loading_zenith or self.select_day is None or self._coverage.empty:
            return

        select_day = self.select_day
        self._load_seq += 1
        load_seq = self._load_seq
        self.loading_day = True
        self._log(f"Loading QA data for {select_day}…")
        self._flush_log()
        self._release_active_datasets()
        self._stokes_review.dispose()
        self._run_day_load(select_day, load_seq)

    def _auto_load_zenith_if_ready(self, load_seq: int) -> None:
        """Start zenith review automatically after a successful day load."""
        if not self._is_current_load(load_seq):
            return
        if self.select_day is None or self._loaded_day != self.select_day:
            return
        status = zarr_status(self.select_day)
        if not (status["I"] or status["V"]):
            return
        self._begin_zenith_load(load_seq=load_seq)

    def _begin_zenith_load(self, *, load_seq: int | None = None) -> None:
        """Build Stokes I/V heatmaps and SkyWidgets for the loaded day."""
        if (
            self.converting
            or self.loading_day
            or self.loading_zenith
            or self.select_day is None
            or self._loaded_day != self.select_day
        ):
            return

        if load_seq is None:
            load_seq = self._load_seq
        elif not self._is_current_load(load_seq):
            return

        select_day = self.select_day
        self.loading_zenith = True
        self._log(f"Loading zenith review panels for {select_day}…")
        self._flush_log()
        self._release_active_datasets()
        self._reset_zenith_sections(banner="*Loading zenith panels…*")
        self._execute(self._push_zenith_root)

        def _run() -> None:
            try:
                if not self._is_current_load(load_seq):
                    raise _LoadSuperseded
                status = zarr_status(select_day)
                if not (status["I"] or status["V"]):
                    section_contents = self._stokes_review.build_no_zarr_contents()
                    banner = _NO_QA_ZARR_MESSAGE
                else:
                    datasets = load_qa_datasets(
                        select_day,
                        self._log,
                        flush=self._flush_log,
                    )
                    if not self._is_current_load(load_seq):
                        raise _LoadSuperseded
                    self._active_datasets = dict(datasets)
                    section_contents = self._stokes_review.build_section_contents(
                        datasets,
                        self._log,
                        flush=self._flush_log,
                    )
                    banner = (
                        "*Zenith heatmaps loaded. Click a cell or move the sliders "
                        "to inspect time/frequency slices.*"
                    )
                    if not self._is_current_load(load_seq):
                        raise _LoadSuperseded

                def _mount() -> None:
                    if not self._is_current_load(load_seq):
                        self._finish_zenith_load(load_seq=load_seq)
                        return
                    self._mount_zenith_sections(section_contents, banner=banner)
                    self._clear_error()
                    self._log(f"Zenith review panels ready for {select_day}.")
                    self._finish_zenith_load(load_seq=load_seq)

                _schedule_ipython_main(_mount)
            except _LoadSuperseded:

                def _abort() -> None:
                    self._finish_zenith_load(load_seq=load_seq)

                _schedule_ipython_main(_abort)
            except Exception as exc:
                import traceback

                def _fail() -> None:
                    if not self._is_current_load(load_seq):
                        self._finish_zenith_load(load_seq=load_seq)
                        return
                    self._log_error(f"Failed to load zenith panels for {select_day}: {exc}")
                    self._log(traceback.format_exc(), sync=False)
                    self._reset_zenith_sections()
                    self._finish_zenith_load(load_seq=load_seq)

                _schedule_ipython_main(_fail)

        threading.Thread(target=_run, daemon=True).start()

    def _finish_zenith_load(self, *, load_seq: int | None = None) -> None:
        """Clear zenith loading state and refresh controls."""
        if load_seq is not None and not self._is_current_load(load_seq):
            return
        self.loading_zenith = False
        self._flush_log()
        self._execute(self._push_zenith_root)

    def _push_zenith_root(self) -> None:
        """Push zenith heatmaps after nested panel updates."""
        self._push_panel_roots()

    def _run_day_load(self, select_day: str, load_seq: int) -> None:
        """Fetch Zarr data and rebuild widgets on the main thread."""
        applied = False
        try:
            if not self._is_current_load(load_seq):
                raise _LoadSuperseded
            payload = self._fetch_day_data(select_day, load_seq=load_seq)
            if not self._is_current_load(load_seq):
                raise _LoadSuperseded
            applied = self._apply_day_payload(payload, load_seq=load_seq)
        except _LoadSuperseded:
            pass
        except Exception as exc:
            self._fail_day_content(select_day, exc, load_seq=load_seq)
        finally:
            self._finish_load_day(load_seq=load_seq, auto_zenith=applied)

    def _fetch_day_data(
        self,
        select_day: str,
        *,
        load_seq: int,
    ) -> _DayLoadPayload:
        """Load the thermal-noise summary table for one day."""
        if not self._is_current_load(load_seq):
            raise _LoadSuperseded

        self._log(f"Building thermal-noise summary for {select_day}…")
        summary_df = day_summary_table(select_day, self._coverage)
        self._log(f"Thermal-noise summary ready ({len(summary_df)} LST hour(s)).")
        self._flush_log()

        if not self._is_current_load(load_seq):
            raise _LoadSuperseded

        status = zarr_status(select_day)
        if status["I"] or status["V"]:
            self._log(
                f"QA Zarr available for {select_day}. "
                "Loading Stokes I/V zenith review after the thermal-noise grid…"
            )
        else:
            self._log(
                f"No QA Zarr stores for {select_day}. "
                "Use **Convert FITS → Zarr** before loading zenith panels."
            )
        self._flush_log()

        return _DayLoadPayload(
            select_day=select_day,
            summary_df=summary_df,
        )

    def _apply_day_payload(
        self,
        payload: _DayLoadPayload,
        *,
        load_seq: int | None = None,
    ) -> bool:
        if load_seq is not None and not self._is_current_load(load_seq):
            return False

        select_day = payload.select_day
        self._loaded_day = select_day
        self._active_datasets.clear()
        self._reset_zenith_sections()
        self._summary_df = payload.summary_df
        thermal_grid = build_thermal_noise_grid(
            payload.summary_df,
            select_day,
            open_full_size=self._open_modal,
        )
        self._qa_grid.objects = [thermal_grid]
        self._clear_error()
        self._log(
            f"Loaded QA data for {select_day} "
            f"({len(payload.summary_df)} thermal-noise plot(s))."
        )
        self._flush_log()

        def _push_loaded_day() -> None:
            self._push_panel_roots()

        self._execute(_push_loaded_day)
        return True

    def _is_current_load(self, load_seq: int) -> bool:
        return load_seq == self._load_seq

    def _fail_day_content(
        self,
        select_day: str,
        exc: Exception,
        *,
        load_seq: int | None = None,
    ) -> None:
        if load_seq is not None and not self._is_current_load(load_seq):
            return
        self._log_error(f"Failed to load QA data for {select_day}: {exc}")
        self._flush_log()

    def _finish_load_day(
        self,
        *,
        load_seq: int | None = None,
        auto_zenith: bool = False,
    ) -> None:
        if load_seq is not None and not self._is_current_load(load_seq):
            return
        self.loading_day = False
        if auto_zenith and load_seq is not None:
            self._auto_load_zenith_if_ready(load_seq)

    def _load_day(self, *, silent: bool) -> None:
        """Synchronous load used after FITS→Zarr conversion."""
        select_day = self.select_day
        if select_day is None:
            return

        self._load_seq += 1
        load_seq = self._load_seq
        self.loading_day = True
        if not silent:
            self._log(f"Refreshing QA data for {select_day}…")
            self._flush_log()
        self._release_active_datasets()
        self._run_day_load(select_day, load_seq)

    def _open_modal(self, png_path: str, title: str) -> None:
        self._modal_container.objects = [
            pn.pane.Markdown(f"### {title}"),
            pn.pane.PNG(png_path, sizing_mode="scale_width"),
            self._close_modal_button,
        ]
        self._push_panel_roots()

    def _close_modal(self) -> None:
        self._modal_container.objects = []
        self._push_panel_roots()

    def _on_convert_click(self, _event: Any) -> None:
        if self.converting or self.select_day is None:
            return
        select_day = self.select_day
        self.converting = True
        self._scroll_log.clear()
        self._sync_log()
        self._log(f"Converting FITS → Zarr for {select_day}…")

        def _run() -> None:
            try:
                convert_missing_zarr(
                    select_day,
                    self._coverage,
                    lambda msg: self._log(msg, sync=True, defer=True),
                )
                self._execute(lambda: self._load_day(silent=False))
            except Exception as exc:
                def _fail() -> None:
                    self._log_error(f"Conversion failed: {exc}")

                self._execute(_fail)
            finally:

                def _finish() -> None:
                    self.converting = False
                    self._sync_log()

                self._execute(_finish)

        threading.Thread(target=_run, daemon=True).start()

    def _build_layouts(self) -> None:
        """Build the single Panel layout with sky widgets below the heatmaps."""
        if self._layout is not None:
            return

        header = pn.pane.Markdown(
            "# Pipeline QA check\n\n"
            "Scan finds available days automatically. Select a day to load the "
            "thermal-noise QA grid and Stokes I/V zenith review. "
            "Matching sky views appear directly below the heatmaps."
        )
        log_section = pn.Column(
            pn.pane.Markdown("**Activity log**"),
            self._log_pane,
            sizing_mode="stretch_width",
        )
        self._layout = pn.Column(
            header,
            pn.Row(
                self._day_selector,
                self._convert_button,
                sizing_mode="stretch_width",
            ),
            log_section,
            self._error_alert,
            pn.pane.Markdown("### Zenith review (Stokes I / V)"),
            pn.Row(self._zenith_load_button, sizing_mode="stretch_width"),
            self._zenith_slot,
            self._sky_host.panel_row,
            pn.pane.Markdown("### Thermal-noise QA by LST hour"),
            self._qa_grid,
            self._modal_container,
            sizing_mode="stretch_width",
        )

    def panel(self) -> pn.Column:
        """Return the JupyterLab dashboard layout."""
        self._build_layouts()

        if not self._scan_started:
            self._scan_started = True
            pn.state.onload(self._start_initial_scan)

        assert self._layout is not None
        return self._layout


def display_pipeline_qa_app(app: PipelineQAApp | None = None) -> PipelineQAApp:
    """Display the QA dashboard in JupyterLab (single Panel document + embedded sky row)."""
    from IPython.display import display

    try:
        pn.extension("ipywidgets")
    except Exception as exc:
        logger.debug("Panel ipywidgets extension unavailable: %s", exc)

    app = app or PipelineQAApp()
    app._build_layouts()

    if not app._scan_started:
        app._scan_started = True
        pn.state.onload(app._start_initial_scan)

    assert app._layout is not None
    display(app._layout)
    return app
