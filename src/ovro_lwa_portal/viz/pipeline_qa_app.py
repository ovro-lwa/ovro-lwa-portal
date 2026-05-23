"""Panel application for per-day pipeline QA review."""

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
    """Run callback on the IPython kernel event loop (required for display_id updates)."""
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


def _push_panel_layout(*views: pn.viewable.Viewable) -> None:
    """Push Panel layout changes to the notebook frontend."""
    try:
        from panel.io.notebook import push, push_on_root
        from panel.io.state import state
    except ImportError:
        return

    pushed_docs: set[int] = set()
    for view in views:
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


def _activity_log_html(text: str) -> str:
    """Render the activity log in a scrollable monospace block (newest first)."""
    import html

    lines = [line for line in text.splitlines() if line]
    display = "\n".join(reversed(lines)) if lines else " "
    body = html.escape(display)
    return (
        '<div id="pipeline-qa-activity-log" '
        'style="height:150px;overflow-y:auto;border:1px solid #ccc;border-radius:4px;'
        'padding:8px;background:#fafafa;font-family:monospace;font-size:12px;'
        'white-space:pre-wrap;margin:0;">'
        f"{body}</div>"
        "<script>"
        "requestAnimationFrame(function(){"
        'var el=document.getElementById("pipeline-qa-activity-log");'
        "if(el){el.scrollTop=0;}"
        "});"
        "</script>"
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
                styles={
                    "border": "1px solid #e0e0e0",
                    "border-radius": "4px",
                    "padding": "6px",
                },
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
            title=f"Zenith patch {self._metric_label} (click to set slice)",
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
        self._marker = plot.scatter(
            x=[self._time_idx],
            y=[self._freq_idx],
            size=18,
            marker="cross",
            line_color="cyan",
            fill_color=None,
            line_width=2,
        )
        plot.xaxis.axis_label = "Time index"
        plot.yaxis.axis_label = "Frequency index"
        plot.on_event(Tap, self._on_tap)
        return plot

    def _on_tap(self, event: Tap) -> None:
        if event.x is None or event.y is None:
            return
        n_times, n_freqs = self._stat_map.shape
        time_idx = int(np.clip(int(round(float(event.x))), 0, n_times - 1))
        freq_idx = int(np.clip(int(round(float(event.y))), 0, n_freqs - 1))
        self._on_select(time_idx, freq_idx)

    def set_slice(self, time_idx: int, freq_idx: int) -> None:
        self._time_idx = time_idx
        self._freq_idx = freq_idx
        self._marker.data_source.data = {"x": [time_idx], "y": [freq_idx]}

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
            f"**Stokes {stokes_label}** — click the heatmap to choose time and frequency. "
            "The matching sky view appears in the row below."
        )
        self._status_pane = pn.pane.Markdown("")
        self._layout = pn.Column(
            self._header,
            self._status_pane,
            self._heatmap.pane,
            margin=(0, 0, 24, 0),
            sizing_mode="stretch_width",
        )

        self.param.watch(self._on_slice_changed, ["time_idx", "freq_idx"])
        self._on_slice_changed()

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
        self._sky_widget.layout = widgets.Layout(width="520px", height="520px")
        bind_ds = dataset if dataset is not None else self._dataset
        self._bind_sky_dataset(self._sky_widget, bind_ds)
        self._on_slice_changed()
        return self._sky_widget

    def _select_slice(self, time_idx: int, freq_idx: int) -> None:
        """Heatmap tap handler: update Param slice indices (drives sky + status)."""
        self.time_idx = int(np.clip(time_idx, 0, self._n_times - 1))
        self.freq_idx = int(np.clip(freq_idx, 0, self._n_freqs - 1))

    def _on_slice_changed(self, *_events: param.parameterized.Event) -> None:
        """Sync heatmap marker, status text, and SkyWidget for the current slice."""
        self._heatmap.set_slice(self.time_idx, self.freq_idx)
        self._sync_status()
        if self._sky_widget is None:
            return
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

    def _sync_status(self) -> None:
        coord = zenith_lm_coord(self._dataset, self.time_idx)
        metric_val = float(self._stat_map[self.time_idx, self.freq_idx])
        freq_mhz = float(self._dataset.frequency.values[self.freq_idx]) / 1e6
        self._status_pane.object = (
            f"**Stokes {self.stokes_label} zenith (l=0, m=0)** | time={self.time_idx}, "
            f"freq={self.freq_idx} ({freq_mhz:.1f} MHz)"
            f" | center RA={coord.ra.to_string(unit=u.hour, precision=1)}, "
            f"Dec={coord.dec.to_string(unit=u.deg, precision=1)}"
            f" | patch {self.metric_label}={metric_val:.3g}"
        )

    @property
    def layout(self) -> pn.Column:
        return self._layout

    @property
    def heatmap_column(self) -> pn.Column:
        """Header, status, and heatmap for side-by-side zenith review layout."""
        return pn.Column(
            self._header,
            self._status_pane,
            self._heatmap.pane,
            width=ZENITH_REVIEW_COLUMN_WIDTH,
            sizing_mode="fixed",
            margin=(0, 8, 0, 0),
        )

    def dispose(self) -> None:
        self._heatmap.dispose()
        self._sky_widget = None


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
        waiting_message="*Use **Load zenith panels** after **Load day**.*",
        missing_zarr_message="*Convert FITS → Zarr to enable Stokes I review.*",
        compute_stat_map=compute_zenith_std_map,
    ),
    _StokesSectionSpec(
        stokes="V",
        heading="## Stokes V — zenith patch RMS",
        metric_label="RMS",
        waiting_message="*Use **Load zenith panels** after **Load day**.*",
        missing_zarr_message="*Run **Convert** to build the Stokes V Zarr and enable review.*",
        compute_stat_map=compute_zenith_rms_map,
    ),
)

_NO_QA_ZARR_MESSAGE = (
    "*No QA Zarr stores for this day. Click **Convert FITS → Zarr** to build them.*"
)

_ZENITH_PLACEHOLDER = (
    "*Select a day, click **Load day**, then **Load zenith panels** "
    "for Stokes I/V heatmaps and sky views.*"
)

_ZENITH_SKY_PLACEHOLDER = (
    "*Sky views appear in the row below the heatmaps after **Load zenith panels**.*"
)

ZENITH_REVIEW_COLUMN_WIDTH = 520

_SKY_DISPLAY_ID = "ovro-lwa-portal-pipeline-qa-sky-host"


class _SkyWidgetHost:
    """Native ipywidgets sky area (cell output via IPython display_id, not Output capture)."""

    def __init__(self) -> None:
        self._display_id = _SKY_DISPLAY_ID
        self._displayed = False

    @property
    def widget(self) -> widgets.HTML:
        """Compatibility handle; sky content is published via :meth:`mark_displayed`."""
        return widgets.HTML("")

    def _sky_shell(self, body: widgets.Widget) -> widgets.VBox:
        return widgets.VBox(
            [body],
            layout=widgets.Layout(
                width="100%",
                min_height="560px",
                border="1px solid #e0e0e0",
                padding="8px",
            ),
        )

    def _publish(self, body: widgets.Widget, *, first: bool = False) -> None:
        from IPython.display import display

        shell = self._sky_shell(body)
        if first:
            display(shell, display_id=self._display_id)
        else:
            display(shell, display_id=self._display_id, update=True)

    def show_placeholder(self) -> None:
        """Render the waiting message into the sky output area."""
        if not self._displayed:
            return
        self._publish(widgets.HTML(_ZENITH_SKY_PLACEHOLDER))

    def mark_displayed(self) -> None:
        """Register the sky display_id and show the placeholder."""
        self._displayed = True
        self._publish(widgets.HTML(_ZENITH_SKY_PLACEHOLDER), first=True)

    def reset(self) -> None:
        if self._displayed:
            self.show_placeholder()

    def mount(self, panels: dict[str, "ZenithReviewPanel | None"]) -> None:
        """Create SkyWidgets and publish them to the sky display_id."""
        if not self._displayed:
            return

        rows: list[Any] = []
        for spec in _STOKES_SECTIONS:
            panel = panels.get(spec.stokes)
            if panel is None:
                continue
            sky = panel.mount_sky()
            rows.append(widgets.HTML(f"<strong>Stokes {spec.stokes} sky view</strong>"))
            rows.append(sky)

        if rows:
            sky_columns = [
                widgets.VBox(
                    [rows[i], rows[i + 1]],
                    layout=widgets.Layout(
                        width=f"{ZENITH_REVIEW_COLUMN_WIDTH}px",
                        min_width=f"{ZENITH_REVIEW_COLUMN_WIDTH}px",
                    ),
                )
                for i in range(0, len(rows), 2)
            ]
            self._publish(
                widgets.HBox(
                    sky_columns,
                    layout=widgets.Layout(
                        width="100%",
                        justify_content="flex-start",
                    ),
                ),
            )
        else:
            self.show_placeholder()


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

    def build_column(
        self,
        datasets: dict[str, xr.Dataset],
        log: LogFn,
        flush: Callable[[], None] | None = None,
    ) -> pn.Column:
        """Compute stat maps and assemble heatmaps in a side-by-side row."""
        self._dispose_panels()
        heatmap_columns: list[pn.Column] = []
        for spec in _STOKES_SECTIONS:
            heatmap_columns.append(self._build_section_for_spec(spec, datasets, log, flush=flush))
        return pn.Column(
            pn.Row(*heatmap_columns, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        )

    def _build_section_for_spec(
        self,
        spec: _StokesSectionSpec,
        datasets: dict[str, xr.Dataset],
        log: LogFn,
        *,
        flush: Callable[[], None] | None = None,
    ) -> pn.Column:
        if spec.stokes not in datasets:
            return pn.Column(
                pn.pane.Markdown(spec.heading),
                pn.pane.Markdown(spec.missing_zarr_message),
                width=ZENITH_REVIEW_COLUMN_WIDTH,
                sizing_mode="fixed",
                margin=(0, 8, 0, 0),
            )

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

    def build_no_zarr_column(self) -> pn.Column:
        """Column shown when the selected day has no QA Zarr stores."""
        self._dispose_panels()
        columns = [
            pn.Column(
                pn.pane.Markdown(spec.heading),
                pn.pane.Markdown(spec.missing_zarr_message),
                width=ZENITH_REVIEW_COLUMN_WIDTH,
                sizing_mode="fixed",
                margin=(0, 8, 0, 0),
            )
            for spec in _STOKES_SECTIONS
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
    """Summary data fetched when the user clicks Load day."""

    select_day: str
    summary_df: pd.DataFrame


class _LoadSuperseded(Exception):
    """Raised when a newer day load replaces this one."""


class PipelineQAApp(param.Parameterized):
    """Interactive Panel dashboard for one-day pipeline QA."""

    select_day = param.Selector(default=None, objects=[])
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
        self._zenith_slot = pn.Column(
            pn.pane.Markdown(_ZENITH_PLACEHOLDER),
            sizing_mode="stretch_width",
        )
        self._zenith_load_button = pn.widgets.Button(
            name="Load zenith panels",
            button_type="default",
            disabled=True,
        )
        self._zenith_load_button.on_click(self._on_zenith_load_click)
        self._qa_grid = pn.Column(
            pn.pane.Markdown(
                "*Select a day and click **Load day** to build the thermal-noise QA grid.*"
            ),
            sizing_mode="stretch_width",
        )
        self._syncing_day = False
        self._day_selector = pn.widgets.Select(
            name="Observation day",
            options=[],
            width=220,
        )
        self._day_selector.param.watch(self._on_day_selector_changed, "value")
        self._load_button = pn.widgets.Button(
            name="Load day",
            button_type="primary",
            disabled=True,
        )
        self._load_button.on_click(self._on_load_click)
        self._convert_button = pn.widgets.Button(
            name="Convert FITS → Zarr",
            button_type="default",
            disabled=True,
        )
        self._convert_button.on_click(self._on_convert_click)
        self._close_modal_button = pn.widgets.Button(name="Close", button_type="default")
        self._close_modal_button.on_click(lambda _event: self._close_modal())
        self._modal_container = pn.Column(sizing_mode="stretch_width")
        self._log_pane = pn.pane.HTML(
            _activity_log_html(""),
            sizing_mode="stretch_width",
        )
        self._layout: pn.Column | None = None
        self._scan_started = False
        self._load_seq = 0
        self._active_datasets: dict[str, xr.Dataset] = {}

    @property
    def sky_widgets(self) -> widgets.HTML:
        """Sky area is published via IPython display_id (see :func:`display_pipeline_qa_app`)."""
        return self._sky_host.widget

    def _execute(self, callback: Any) -> None:
        try:
            pn.state.execute(callback)
        except Exception:
            callback()

    def _sync_day_selector(self, days: list[str], value: str | None) -> None:
        """Push day options into the Select widget (needed in JupyterLab)."""
        self._syncing_day = True
        try:
            self.param.select_day.objects = days
            self._day_selector.options = days
            self._day_selector.value = value
        finally:
            self._syncing_day = False

    def _on_day_selector_changed(self, event: param.parameterized.Event) -> None:
        if self._syncing_day or event.new is None or event.new == self.select_day:
            return
        self.select_day = event.new
        if self._loaded_day != event.new:
            self._release_active_datasets()
            self._stokes_review.dispose()
            self._sky_host.reset()
            self._zenith_slot.objects = [pn.pane.Markdown(_ZENITH_PLACEHOLDER)]

            def _push() -> None:
                views: list[pn.viewable.Viewable] = [self._zenith_slot]
                if self._layout is not None:
                    views.append(self._layout)
                _push_panel_layout(*views)

            self._execute(_push)
        self._refresh_action_buttons()

    def _on_load_click(self, _event: Any) -> None:
        self._begin_load_day()

    def _on_zenith_load_click(self, _event: Any) -> None:
        self._begin_zenith_load()

    def _refresh_action_buttons(self) -> None:
        """Sync Load / Convert / zenith button state with the selected day."""
        busy = self.loading_day or self.loading_zenith or self.converting or self.scanning
        no_day = self.select_day is None or self._coverage.empty
        self._load_button.disabled = busy or no_day
        self._refresh_convert_button()
        self._refresh_zenith_button()

    def _refresh_zenith_button(self) -> None:
        """Enable zenith load once Load day has run and QA Zarr exists."""
        if self.select_day is None:
            self._zenith_load_button.disabled = True
            return
        status = zarr_status(self.select_day)
        has_zarr = status["I"] or status["V"]
        day_ready = self._loaded_day == self.select_day
        busy = self.loading_zenith or self.loading_day or self.converting or self.scanning
        self._zenith_load_button.disabled = busy or not day_ready or not has_zarr

    def _sync_log(self, *, defer: bool = False) -> None:
        html = _activity_log_html(self._scroll_log.text)
        if defer:

            def _push() -> None:
                self._log_pane.object = html

            self._execute(_push)
        else:
            self._log_pane.object = html

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

    def _log_error(self, message: str) -> None:
        """Record an error in the log pane."""
        self._log(f"ERROR: {message}")

    def _refresh_convert_button(self) -> None:
        """Sync convert button label/disabled state with the selected day."""
        if self.select_day is None:
            self._convert_button.disabled = True
            return
        status = zarr_status(self.select_day)
        self._convert_button.name = convert_button_label(status)
        self._convert_button.disabled = convert_button_disabled(
            status,
            converting=self.converting,
        )

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
                        self.select_day = None
                    else:
                        self._sync_day_selector(days, default_day)
                        self.select_day = default_day
                        self._log(
                            f"Found {len(days)} Wideband QA day(s). "
                            "Select a day, click **Load day**, then **Load zenith panels**."
                        )
                        self._refresh_action_buttons()
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
        self._day_selector.disabled = True
        self._load_button.disabled = True
        self._convert_button.disabled = True
        self._zenith_load_button.disabled = True
        self._log(f"Loading QA data for {select_day}…")
        self._flush_log()
        self._release_active_datasets()
        self._stokes_review.dispose()
        self._run_day_load(select_day, load_seq)

    def _begin_zenith_load(self) -> None:
        """Build Stokes I/V heatmaps (Panel) and SkyWidgets (native ipywidgets) for the loaded day."""
        if (
            self.converting
            or self.loading_day
            or self.loading_zenith
            or self.select_day is None
            or self._loaded_day != self.select_day
        ):
            return

        select_day = self.select_day
        self.loading_zenith = True
        self._zenith_load_button.disabled = True
        self._load_button.disabled = True
        self._convert_button.disabled = True
        self._day_selector.disabled = True
        self._log(f"Loading zenith review panels for {select_day}…")
        self._flush_log()
        self._release_active_datasets()
        self._stokes_review.dispose()
        self._sky_host.reset()
        self._zenith_slot.objects = [
            pn.pane.Markdown("*Loading zenith panels…*"),
        ]
        self._execute(self._push_zenith_root)

        def _run() -> None:
            try:
                status = zarr_status(select_day)
                if not (status["I"] or status["V"]):
                    review_column = self._stokes_review.build_no_zarr_column()
                else:
                    datasets = load_qa_datasets(
                        select_day,
                        self._log,
                        flush=self._flush_log,
                    )
                    self._active_datasets = dict(datasets)
                    review_column = self._stokes_review.build_column(
                        datasets,
                        self._log,
                        flush=self._flush_log,
                    )

                def _mount() -> None:
                    self._mount_zenith_column(review_column)
                    self._log(f"Zenith review panels ready for {select_day}.")
                    self._finish_zenith_load()

                _schedule_ipython_main(_mount)
            except Exception as exc:
                import traceback

                def _fail() -> None:
                    self._log_error(f"Failed to load zenith panels for {select_day}: {exc}")
                    self._log(traceback.format_exc(), sync=False)
                    self._zenith_slot.objects = [pn.pane.Markdown(_ZENITH_PLACEHOLDER)]
                    self._sky_host.reset()
                    self._finish_zenith_load()

                _schedule_ipython_main(_fail)

        threading.Thread(target=_run, daemon=True).start()

    def _finish_zenith_load(self) -> None:
        """Clear zenith loading state and refresh controls."""
        self.loading_zenith = False
        self._day_selector.disabled = self.converting
        self._refresh_action_buttons()
        self._flush_log()
        self._execute(self._push_zenith_root)

    def _push_zenith_root(self) -> None:
        """Push zenith slot and app root after nested panel updates."""
        views: list[pn.viewable.Viewable] = [self._zenith_slot]
        if self._layout is not None:
            views.append(self._layout)
        _push_panel_layout(*views)

    def _mount_zenith_column(self, review_column: pn.Column) -> None:
        """Swap zenith heatmaps into the dashboard and mount sky widgets below."""
        self._zenith_slot.objects = [review_column]
        if not self._sky_host._displayed:
            self._sky_host.mark_displayed()
        self._sky_host.mount(self._stokes_review._panels)
        self._execute(self._push_zenith_root)

    def _run_day_load(self, select_day: str, load_seq: int) -> None:
        """Fetch Zarr data and rebuild widgets on the main thread."""
        try:
            if not self._is_current_load(load_seq):
                raise _LoadSuperseded
            payload = self._fetch_day_data(select_day, load_seq=load_seq)
            if not self._is_current_load(load_seq):
                raise _LoadSuperseded
            self._apply_day_payload(payload, load_seq=load_seq)
        except _LoadSuperseded:
            pass
        except Exception as exc:
            self._fail_day_content(select_day, exc, load_seq=load_seq)
        finally:
            self._finish_load_day(load_seq=load_seq)

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
                "Click **Load zenith panels** for Stokes I/V review."
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
    ) -> None:
        if load_seq is not None and not self._is_current_load(load_seq):
            return

        select_day = payload.select_day
        self._loaded_day = select_day
        self._active_datasets.clear()
        self._stokes_review.dispose()
        self._sky_host.reset()
        self._summary_df = payload.summary_df
        thermal_grid = build_thermal_noise_grid(
            payload.summary_df,
            select_day,
            open_full_size=self._open_modal,
        )
        self._qa_grid.objects = [thermal_grid]
        self._zenith_slot.objects = [pn.pane.Markdown(_ZENITH_PLACEHOLDER)]
        self._refresh_convert_button()
        self._log(
            f"Loaded QA data for {select_day} "
            f"({len(payload.summary_df)} thermal-noise plot(s))."
        )
        self._flush_log()

        def _push_loaded_day() -> None:
            views: list[pn.viewable.Viewable] = [
                self._qa_grid,
                self._zenith_slot,
            ]
            if self._layout is not None:
                views.append(self._layout)
            _push_panel_layout(*views)

        self._execute(_push_loaded_day)

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
        self._refresh_convert_button()
        self._flush_log()

    def _finish_load_day(self, *, load_seq: int | None = None) -> None:
        if load_seq is not None and not self._is_current_load(load_seq):
            return
        self.loading_day = False
        self._day_selector.disabled = self.converting
        self._refresh_action_buttons()

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

    def _close_modal(self) -> None:
        self._modal_container.objects = []

    def _on_convert_click(self, _event: Any) -> None:
        if self.converting or self.select_day is None:
            return
        select_day = self.select_day
        self.converting = True
        self._scroll_log.clear()
        self._sync_log()
        self._log(f"Converting FITS → Zarr for {select_day}…")
        self._convert_button.disabled = True
        self._load_button.disabled = True
        self._zenith_load_button.disabled = True
        self._day_selector.disabled = True

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
                    self._refresh_action_buttons()
                    self._sync_log()

                self._execute(_finish)

        threading.Thread(target=_run, daemon=True).start()

    def panel(self) -> pn.Column:
        """Return the full Panel layout."""
        if self._layout is None:
            header = pn.pane.Markdown(
                "# Pipeline QA check\n\n"
                "Scan finds available days automatically. Select a day and click **Load day** "
                "for the thermal-noise QA grid, then **Load zenith panels** for Stokes I/V "
                "heatmaps side by side. Matching sky views appear in the row below."
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
                    self._load_button,
                    self._convert_button,
                    sizing_mode="stretch_width",
                ),
                log_section,
                pn.pane.Markdown("### Zenith review (Stokes I / V)"),
                pn.Row(self._zenith_load_button, sizing_mode="stretch_width"),
                self._zenith_slot,
                pn.pane.Markdown("### Thermal-noise QA by LST hour"),
                self._qa_grid,
                self._modal_container,
                sizing_mode="stretch_width",
            )

        if not self._scan_started:
            self._scan_started = True
            pn.state.onload(self._start_initial_scan)

        return self._layout


def display_pipeline_qa_app(app: PipelineQAApp | None = None) -> PipelineQAApp:
    """Display the Panel dashboard and native sky-widget area in Jupyter."""
    from IPython.display import display

    app = app or PipelineQAApp()
    display(app.panel())
    app._sky_host.mark_displayed()
    return app
