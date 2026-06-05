"""JupyterLab Panel application for per-day pipeline QA review."""

from __future__ import annotations

import logging
import math
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
    PipelineQAConfig,
    convert_button_disabled,
    convert_button_label,
    convert_missing_zarr,
    day_summary_table,
    load_dewarp_summary_dataframe,
    load_flux_check_hybrid_dataframe,
    load_qa_datasets,
    qa_days,
    resolve_pipeline_qa_config,
    scan_coverage,
    zarr_status,
)
from ovro_lwa_portal.accessor import _has_per_time_wcs_header_str, _read_wcs_header_str
from ovro_lwa_portal.viz.dewarp_summary_plots import build_dewarp_shift_panel
from ovro_lwa_portal.viz.flux_check_plots import (
    build_flux_ratio_figures,
    build_flux_ratio_panel_grid,
)

check_viz_deps()


def _patch_astrowidget_get_wcs() -> None:
    """Use strict per-time WCS lookup so late time indices do not fall back to time 0."""
    import astrowidget.wcs as awcs

    if getattr(awcs.get_wcs, "_ovro_portal_patched", False):
        return

    original_get_wcs = awcs.get_wcs

    def get_wcs(ds: xr.Dataset, var: str = "SKY", time_idx: int = 0):
        from astropy.io.fits import Header
        from astropy.wcs import WCS

        if _has_per_time_wcs_header_str(ds):
            hdr_str = _read_wcs_header_str(ds, var=var, time_idx=int(time_idx))
            if hdr_str is None:
                n_time = int(ds.sizes.get("time", 0))
                msg = (
                    f"Missing WCS metadata for time index {time_idx} "
                    f"(dataset has {n_time} time steps with per-time wcs_header_str). "
                    "Re-run FITS→Zarr conversion for this day so each time step stores "
                    "a WCS header."
                )
                raise ValueError(msg)
            wcs = WCS(Header.fromstring(hdr_str, sep="\n"))
            if not wcs.has_celestial:
                msg = "WCS header has no celestial axes (RA/Dec)"
                raise ValueError(msg)
            return wcs.celestial
        return original_get_wcs(ds, var=var, time_idx=time_idx)

    get_wcs._ovro_portal_patched = True  # type: ignore[attr-defined]
    awcs.get_wcs = get_wcs


_patch_astrowidget_get_wcs()


def bind_sky_widget_dataset(
    widget: SkyWidget,
    dataset: xr.Dataset,
    *,
    max_size: int = 1024,
) -> None:
    """Load the SKY cube without displaying; call :meth:`~astrowidget.SkyWidget.update_slice` next."""
    widget.set_dataset(dataset, max_size=max_size, defer_display=True)


logger = logging.getLogger(__name__)

ZENITH_L = 0.0
ZENITH_M = 0.0
ZENITH_PATCH_RADIUS = 10
DEFAULT_FREQ_IDX = 8
DEFAULT_FOV_DEG = 25.0
THERMAL_NOISE_GRID_COLS = 4


_IPYTHON_IO_LOOP: Any = None


def _capture_ipython_io_loop() -> None:
    """Cache the IPython kernel ``io_loop`` from the main thread.

    Background worker threads cannot call ``get_ipython()`` reliably in
    Jupyter. Call this once from a notebook setup cell (after imports).
    """
    global _IPYTHON_IO_LOOP
    _IPYTHON_IO_LOOP = None
    try:
        from IPython import get_ipython

        ip = get_ipython()
        kernel = getattr(ip, "kernel", None) if ip is not None else None
        if kernel is not None:
            _IPYTHON_IO_LOOP = getattr(kernel, "io_loop", None)
    except Exception:
        pass


def _schedule_ipython_main(callback: Callable[[], None]) -> None:
    """Run callback on the IPython kernel event loop."""
    io_loop = _IPYTHON_IO_LOOP
    if io_loop is not None:
        try:
            io_loop.add_callback(callback)
            return
        except Exception:
            pass
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
    """Run a callback on the IPython/Panel notebook UI thread.

    Do not run synchronously when ``state.curdoc`` is set: Bokeh heatmap tap handlers
    execute with the heatmap pane's document active, and an in-place callback there
    updates Param objects without refreshing the embedded Panel sliders.
    """
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
    pushed_roots: set[str] = set()
    for view in views:
        if not view._models:
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
            elif ref not in pushed_roots:
                try:
                    push_on_root(ref)
                    pushed_roots.add(ref)
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
    thermal_noise_plot_name: str = "thermal_noise_vs_subband",
    open_full_size: Callable[[str, str], None] | None = None,
) -> pn.Column:
    """Thermal-noise QA PNGs labeled by LST hour, date, and subband count."""
    if summary_df.empty:
        return pn.Column(
            pn.pane.Markdown("*No QA hours with thermal-noise plots for this day.*"),
            sizing_mode="stretch_width",
        )

    png_height = 360 if n_cols <= 1 else 180
    tiles: list[pn.Column] = []
    for row in summary_df.itertuples(index=False):
        png_path = str(row.thermal_noise_png)
        lst_hour = str(row.lst_hour)
        n_subbands = int(row.n_subbands)
        label = _qa_tile_label(lst_hour, obs_date, n_subbands)

        if Path(png_path).is_file():
            img: Any = pn.pane.PNG(
                png_path,
                height=png_height,
                sizing_mode="scale_width",
            )
        else:
            img = pn.pane.Markdown(f"*{thermal_noise_plot_name}.png missing*")

        footer: list[Any] = [pn.pane.Markdown(label)]
        if open_full_size and Path(png_path).is_file():
            title = f"{thermal_noise_plot_name} — {obs_date} {lst_hour}"

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
    return pn.Column(
        *grid_rows,
        sizing_mode="stretch_width",
        max_width=ZENITH_REVIEW_ROW_WIDTH,
    )


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
    """RA/Dec of the fixed (l=0, m=0) grid point for one time slice.

    Uses :meth:`~ovro_lwa_portal.accessor.RadportAccessor.pixel_to_coords` so zenith
    matches the time-dependent SIN geometry (same for Stokes I and V).
    """
    l_idx, m_idx = dataset.radport.nearest_lm_idx(ZENITH_L, ZENITH_M)
    ra_deg, dec_deg = dataset.radport.pixel_to_coords(l_idx, m_idx, time_idx=int(time_idx))
    return SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="fk5")


def sky_view_center(dataset: xr.Dataset, time_idx: int) -> SkyCoord:
    """View center for :class:`~astrowidget.SkyWidget` at one time index.

    When the Zarr store has per-time ``wcs_header_str`` (incremental ingest), use
    that slice's FITS WCS phase center so the sphere matches the displayed image.
    Otherwise use :func:`zenith_lm_coord` (analytical SIN at zenith).
    """
    from astrowidget.wcs import get_wcs

    if _has_per_time_wcs_header_str(dataset):
        wcs = get_wcs(dataset, time_idx=int(time_idx))
        return SkyCoord(
            ra=float(wcs.wcs.crval[0]) * u.deg,
            dec=float(wcs.wcs.crval[1]) * u.deg,
            frame="fk5",
        )
    return zenith_lm_coord(dataset, time_idx)


def compute_zenith_std_map(
    dataset: xr.Dataset,
    radius: int = ZENITH_PATCH_RADIUS,
) -> np.ndarray:
    """Spatial STD in a fixed (l, m) patch for each (time, frequency) cell.

    Uses a fixed pixel half-width so zenith QA works on stores without synthesized
    beam metadata (common for pipeline QA Zarr until BEAM is populated).
    """
    l_idx, m_idx = dataset.radport.nearest_lm_idx(ZENITH_L, ZENITH_M)
    sky = dataset["SKY"].isel(polarization=0)
    n_times = int(sky.sizes["time"])
    n_freqs = int(sky.sizes["frequency"])
    n_l = int(sky.sizes["l"])
    n_m = int(sky.sizes["m"])
    r = max(0, int(radius))
    stat = np.full((n_times, n_freqs), np.nan, dtype=np.float64)
    li0 = int(l_idx)
    mi0 = int(m_idx)
    l_sl = slice(max(0, li0 - r), min(n_l, li0 + r + 1))
    m_sl = slice(max(0, mi0 - r), min(n_m, mi0 + r + 1))
    for t in range(n_times):
        for f in range(n_freqs):
            patch = sky.isel(time=t, frequency=f, l=l_sl, m=m_sl).values
            finite = patch[np.isfinite(patch)]
            if finite.size:
                stat[t, f] = float(np.std(finite))
    return stat


def _time_days_since_start(time_values: np.ndarray) -> np.ndarray:
    """Convert dataset ``time`` coordinates to elapsed days from the first sample."""
    from astropy.time import Time

    tv = np.asarray(time_values)
    if tv.dtype == object:
        mjd = np.asarray(Time(tv.tolist()).mjd, dtype=np.float64)
    elif np.issubdtype(tv.dtype, np.datetime64):
        mjd = np.asarray(Time(tv, format="datetime64").mjd, dtype=np.float64)
    else:
        values = tv.astype(np.float64, copy=False)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.zeros_like(values, dtype=np.float64)
        origin = float(values[0])
        if np.nanmax(np.abs(finite)) > 1e12:
            return (values - origin) / 86_400_000_000_000.0
        if np.nanmax(np.abs(finite)) > 1e7:
            return (values - origin) / 86_400.0
        return values - origin
    return mjd - float(mjd[0])


def _zenith_heatmap_time_days(dataset: xr.Dataset) -> np.ndarray:
    """Elapsed days since the first time sample (for heatmap axis labels)."""
    return _time_days_since_start(dataset.coords["time"].values)


def _zenith_heatmap_freq_mhz(dataset: xr.Dataset) -> np.ndarray:
    """Frequency coordinate in MHz (for heatmap axis labels)."""
    return np.asarray(dataset.coords["frequency"].values, dtype=np.float64) / 1e6


def _zenith_heatmap_lst_hours(dataset: xr.Dataset) -> np.ndarray:
    """Mean local sidereal time in hours (0–24) for each dataset time sample."""
    from astropy import units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    from astropy.utils.iers import conf as iers_conf

    observatory = EarthLocation(
        lat=37.2339 * u.deg, lon=-118.2817 * u.deg, height=1222 * u.m
    )
    mjd = np.asarray(dataset.coords["time"].values, dtype=np.float64)
    orig = iers_conf.auto_download
    try:
        iers_conf.auto_download = False
        times = Time(mjd, format="mjd", scale="utc")
        lst_deg = np.asarray(
            times.sidereal_time("mean", longitude=observatory.lon).deg,
            dtype=np.float64,
        )
    finally:
        iers_conf.auto_download = orig
    return np.mod(lst_deg / 15.0, 24.0)


def _format_lst_hour_label(lst_hour: float) -> str:
    """Format LST as a directory-style hour label such as ``08h``."""
    hour = int(round(float(lst_hour))) % 24
    return f"{hour:02d}h"


def _format_freq_mhz_label(mhz: float) -> str:
    return f"{mhz:.1f}"


def _heatmap_cell_center(idx: int) -> float:
    """Data-space coordinate at the center of a heatmap cell."""
    return idx + 0.5


def _heatmap_index_from_coord(coord: float, n: int) -> int:
    """Map a tap/data coordinate to a zero-based cell index."""
    if n <= 0:
        return 0
    return int(np.clip(int(np.floor(float(coord))), 0, n - 1))


def _heatmap_axis_ticks(
    n: int,
    values: np.ndarray | None = None,
    *,
    format_value: Callable[[Any], str] | None = None,
    max_ticks: int = 24,
) -> tuple[list[float], dict[float, str]]:
    """Tick positions at cell centers with index or physical-value labels."""
    if n <= 0:
        return [], {}
    step = 1 if n <= max_ticks else int(np.ceil(n / max_ticks))
    indices = range(0, n, step)
    ticks = [_heatmap_cell_center(i) for i in indices]
    if values is not None and format_value is not None and len(values) >= n:
        labels = {
            tick: format_value(values[i]) for tick, i in zip(ticks, indices, strict=True)
        }
    else:
        labels = {tick: str(i) for tick, i in zip(ticks, indices, strict=True)}
    return ticks, labels


def _build_heatmap_hover_source(
    stat_map: np.ndarray,
    *,
    lst_hours: np.ndarray,
    freq_mhz: np.ndarray,
) -> ColumnDataSource:
    """ColumnDataSource for per-cell hover tooltips on the zenith heatmap."""
    n_times, n_freqs = stat_map.shape
    time_idx, freq_idx = np.meshgrid(
        np.arange(n_times, dtype=int),
        np.arange(n_freqs, dtype=int),
        indexing="ij",
    )
    flat_time = time_idx.ravel()
    flat_freq = freq_idx.ravel()
    lst_labels = [_format_lst_hour_label(float(h)) for h in lst_hours[flat_time]]
    return ColumnDataSource(
        data={
            "x": flat_time + 0.5,
            "y": flat_freq + 0.5,
            "time_idx": flat_time,
            "freq_idx": flat_freq,
            "lst_hour": lst_labels,
            "freq_mhz": freq_mhz[flat_freq],
            "value": stat_map.astype(float, copy=False).ravel(),
        }
    )


class _ZenithHeatmapSelector:
    """Bokeh zenith heatmap embedded in Panel (read-only map; tap updates shared sliders)."""

    def __init__(
        self,
        stat_map: np.ndarray,
        *,
        metric_label: str,
        lst_hours: np.ndarray,
        freq_mhz: np.ndarray,
        on_select: Callable[[int, int], None],
    ) -> None:
        self._stat_map = stat_map
        self._metric_label = metric_label
        self._lst_hours = np.asarray(lst_hours, dtype=np.float64)
        self._freq_mhz = np.asarray(freq_mhz, dtype=np.float64)
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
            title=(
                f"Zenith patch {self._metric_label} "
                "(hover for values; click to set time/frequency sliders)"
            ),
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
            source=_build_heatmap_hover_source(
                self._stat_map,
                lst_hours=self._lst_hours,
                freq_mhz=self._freq_mhz,
            ),
            fill_alpha=0,
            line_alpha=0,
            hover_fill_alpha=0,
            hover_line_alpha=0,
        )
        plot.add_tools(
            HoverTool(
                renderers=[hover_renderer],
                tooltips=[
                    ("LST hour", "@lst_hour"),
                    ("Freq (MHz)", "@freq_mhz{0.1}"),
                    ("Time idx", "@time_idx"),
                    ("Freq idx", "@freq_idx"),
                    (self._metric_label, "@value{0.3g}"),
                ],
            )
        )
        x_ticks, x_labels = _heatmap_axis_ticks(
            n_times,
            self._lst_hours,
            format_value=_format_lst_hour_label,
        )
        y_ticks, y_labels = _heatmap_axis_ticks(
            n_freqs,
            self._freq_mhz,
            format_value=_format_freq_mhz_label,
        )
        plot.xaxis.ticker = FixedTicker(ticks=x_ticks)
        plot.yaxis.ticker = FixedTicker(ticks=y_ticks)
        plot.xaxis.major_label_overrides = x_labels
        plot.yaxis.major_label_overrides = y_labels
        plot.xaxis.axis_label = "LST hour"
        plot.yaxis.axis_label = "Frequency (MHz)"
        plot.xaxis.major_label_orientation = math.pi / 4
        plot.min_border_bottom = 80
        plot.on_event(Tap, self._on_tap)
        return plot

    def _on_tap(self, event: Tap) -> None:
        if event.x is None or event.y is None:
            return
        n_times, n_freqs = self._stat_map.shape
        time_idx = _heatmap_index_from_coord(event.x, n_times)
        freq_idx = _heatmap_index_from_coord(event.y, n_freqs)

        def _dispatch() -> None:
            self._on_select(time_idx, freq_idx)

        # Bokeh tap callbacks run on the heatmap document; schedule on the kernel loop.
        _schedule_ipython_main(_dispatch)

    def set_data(
        self,
        stat_map: np.ndarray,
        *,
        lst_hours: np.ndarray,
        freq_mhz: np.ndarray,
    ) -> None:
        self._stat_map = stat_map
        self._lst_hours = np.asarray(lst_hours, dtype=np.float64)
        self._freq_mhz = np.asarray(freq_mhz, dtype=np.float64)
        self._plot = self._build_plot()
        self.pane.object = self._plot

    def dispose(self) -> None:
        return


class ZenithSliceSelection(param.Parameterized):
    """Shared time/frequency slice indices for all Stokes zenith review panels."""

    time_idx = param.Integer(default=0, bounds=(0, 0), doc="Selected time index.")
    freq_idx = param.Integer(default=0, bounds=(0, 0), doc="Selected frequency index.")

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._push_root: Callable[[], None] | None = None

    def set_push_root(self, callback: Callable[[], None] | None) -> None:
        """Register the displayed dashboard root push callback (required in JupyterLab)."""
        self._push_root = callback

    def configure(
        self,
        *,
        n_times: int,
        n_freqs: int,
        default_time: int,
        default_freq: int,
    ) -> None:
        """Set slider bounds and the initial slice for a loaded observation day."""
        max_time = max(0, int(n_times) - 1)
        max_freq = max(0, int(n_freqs) - 1)
        self.param.time_idx.bounds = (0, max_time)
        self.param.freq_idx.bounds = (0, max_freq)
        with param.parameterized.batch_call_watchers(self):
            self.time_idx = int(np.clip(default_time, 0, max_time))
            self.freq_idx = int(np.clip(default_freq, 0, max_freq))

    def apply_slice(self, time_idx: int, freq_idx: int) -> None:
        """Update slice indices on the notebook UI thread."""
        t_lo, t_hi = self.param.time_idx.bounds
        f_lo, f_hi = self.param.freq_idx.bounds

        def _apply() -> None:
            with param.parameterized.batch_call_watchers(self):
                self.time_idx = int(np.clip(time_idx, t_lo, t_hi))
                self.freq_idx = int(np.clip(freq_idx, f_lo, f_hi))

        _run_on_main_thread(_apply)

    def _push_ui(self) -> None:
        if self._push_root is not None:
            self._push_root()


class ZenithReviewPanel(param.Parameterized):
    """Heatmap + SkyWidget zenith review linked to shared slice selection."""

    stokes_label = param.String(default="", doc="Stokes parameter label (I or V).")
    metric_label = param.String(default="", doc="Zenith patch metric label (STD).")

    def __init__(
        self,
        dataset: xr.Dataset,
        stat_map: np.ndarray,
        *,
        slice_selection: ZenithSliceSelection,
        stokes_label: str,
        metric_label: str,
        on_heatmap_select: Callable[[str, int, int], None] | None = None,
    ) -> None:
        self._dataset = dataset
        self._stat_map = stat_map
        self._slice_selection = slice_selection
        self._on_heatmap_select = on_heatmap_select
        self._n_times = int(dataset.sizes["time"])
        self._n_freqs = int(dataset.sizes["frequency"])

        super().__init__(
            stokes_label=stokes_label,
            metric_label=metric_label,
        )

        self._heatmap = _ZenithHeatmapSelector(
            stat_map,
            metric_label=metric_label,
            lst_hours=_zenith_heatmap_lst_hours(dataset),
            freq_mhz=_zenith_heatmap_freq_mhz(dataset),
            on_select=self._select_slice,
        )

        self._header = pn.pane.Markdown(
            f"**Click on heatmap or use sliders below to update time/frequency/Stokes view**"
        )
        self._layout = pn.Column(
            self._header,
            self._heatmap.pane,
            width=ZENITH_REVIEW_COLUMN_WIDTH,
            sizing_mode="fixed",
            margin=(0, ZENITH_REVIEW_COLUMN_GAP, 0, 0),
        )

    @property
    def time_idx(self) -> int:
        return int(self._slice_selection.time_idx)

    @property
    def freq_idx(self) -> int:
        return int(self._slice_selection.freq_idx)

    def _format_slice_status(self, time_idx: int, freq_idx: int) -> str:
        """Slice summary shown above the heatmap (bound to shared slice params)."""
        time_idx = int(time_idx)
        freq_idx = int(freq_idx)
        try:
            coord = sky_view_center(self._dataset, time_idx)
            metric_val = float(self._stat_map[time_idx, freq_idx])
            time_day = float(_zenith_heatmap_time_days(self._dataset)[time_idx])
            freq_mhz = float(_zenith_heatmap_freq_mhz(self._dataset)[freq_idx])
            return (
                f"**Stokes {self.stokes_label}** | time={time_day:.3f} d ({time_idx}), "
                f"freq={freq_mhz:.1f} MHz ({freq_idx})"
                f" | center RA={coord.ra.to_string(unit=u.hour, precision=1)}, "
                f"Dec={coord.dec.to_string(unit=u.deg, precision=1)}"
                f" | patch {self.metric_label}={metric_val:.3g}"
            )
        except Exception as exc:
            logger.warning(
                "Zenith status update failed for Stokes %s slice (%s, %s): %s",
                self.stokes_label,
                time_idx,
                freq_idx,
                exc,
                exc_info=True,
            )
            return (
                f"**Stokes {self.stokes_label}** | time_idx={time_idx}, "
                f"freq_idx={freq_idx} | status unavailable ({exc})"
            )

    def _select_slice(self, time_idx: int, freq_idx: int) -> None:
        """Heatmap tap handler: update shared slice (and sky Stokes when wired)."""
        time_idx = int(np.clip(time_idx, 0, self._n_times - 1))
        freq_idx = int(np.clip(freq_idx, 0, self._n_freqs - 1))
        if self._on_heatmap_select is not None:
            self._on_heatmap_select(self.stokes_label, time_idx, freq_idx)
        else:
            self._slice_selection.apply_slice(time_idx, freq_idx)

    @property
    def layout(self) -> pn.Column:
        return self._layout

    @property
    def heatmap_column(self) -> pn.Column:
        """Header and heatmap for side-by-side zenith review layout."""
        return self._layout

    def dispose(self) -> None:
        self._heatmap.dispose()


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
        heading="## Stokes V — zenith patch STD",
        metric_label="STD",
        waiting_message="*Load a day with QA Zarr to enable Stokes V review.*",
        missing_zarr_message="*Run **Convert** to build the Stokes V Zarr and enable review.*",
        compute_stat_map=compute_zenith_std_map,
    ),
)

_NO_QA_ZARR_MESSAGE = (
    "*No QA Zarr stores for this day. Click **Convert FITS → Zarr** to build them.*"
)

_ZENITH_PLACEHOLDER = (
    "*Select an observation day to build Stokes I/V heatmaps and sky views.*"
)

_ZENITH_SKY_PLACEHOLDER = (
    "*Sky view appears below the heatmaps after a day is selected (when QA Zarr exists).*"
)

ZENITH_REVIEW_COLUMN_WIDTH = 520
ZENITH_REVIEW_COLUMN_GAP = 8
ZENITH_REVIEW_ROW_WIDTH = 2 * ZENITH_REVIEW_COLUMN_WIDTH + ZENITH_REVIEW_COLUMN_GAP
ZENITH_SKY_WIDGET_WIDTH = ZENITH_REVIEW_ROW_WIDTH
# astrowidget's WebGL canvas defaults to 600px tall (see widget.js); width is 100%.
ZENITH_SKY_WIDGET_HEIGHT = 600
ZENITH_SKY_LABEL_HEIGHT = 22
ZENITH_SKY_PANE_HEIGHT = ZENITH_SKY_WIDGET_HEIGHT + ZENITH_SKY_LABEL_HEIGHT
ZENITH_SKY_ROW_MARGIN = (0, 0, 0, 0)

_ZENITH_SKY_STATUS_IDLE = (
    "*Sky view: use the sliders or click a heatmap cell; the status line shows load progress.*"
)


class _StokesReviewHolder(param.Parameterized):
    """Builds Stokes I/V heatmaps and a shared sky view for zenith review."""

    sky_stokes = param.Selector(default="I", objects=["I", "V"], doc="Stokes parameter for sky view.")
    loading_sky = param.Boolean(default=False, doc="True while the sky widget is updating.")

    def __init__(self) -> None:
        super().__init__()
        self._datasets: dict[str, xr.Dataset] = {}
        self._sky_widget: SkyWidget | None = None
        self._sky_update_seq = 0
        self._pending_reset_center = False
        self._slice_selection = ZenithSliceSelection()
        self._extra_push_views: Callable[[], list[pn.viewable.Viewable]] | None = None
        self._slice_push_watcher = self._slice_selection.param.watch(
            self._on_slice_indices_changed,
            ["time_idx", "freq_idx"],
        )
        self._slice_watcher = self._slice_selection.param.watch(
            self._on_slice_selection_changed,
            ["time_idx", "freq_idx"],
        )
        self._sky_watcher = self.param.watch(self._on_sky_stokes_changed, "sky_stokes")
        self._time_slider = pn.widgets.IntSlider.from_param(
            self._slice_selection.param.time_idx,
            name="Time index",
            width=400,
        )
        self._freq_slider = pn.widgets.IntSlider.from_param(
            self._slice_selection.param.freq_idx,
            name="Frequency index",
            width=400,
        )
        self._stokes_toggle = pn.widgets.RadioButtonGroup.from_param(
            self.param.sky_stokes,
            name="Sky view",
            options={"Stokes I": "I", "Stokes V": "V"},
            button_type="default",
            width=200,
        )
        self._controls_row = pn.Row(
            self._time_slider,
            self._freq_slider,
            self._stokes_toggle,
            sizing_mode="stretch_width",
            max_width=ZENITH_REVIEW_ROW_WIDTH,
            visible=False,
        )
        sky_container_height = f"{ZENITH_SKY_PANE_HEIGHT}px"
        self._sky_container = widgets.VBox(
            children=[widgets.HTML(_ZENITH_SKY_PLACEHOLDER)],
            layout=widgets.Layout(
                width=f"{ZENITH_REVIEW_ROW_WIDTH}px",
                min_width=f"{ZENITH_REVIEW_ROW_WIDTH}px",
                height=sky_container_height,
                max_height=sky_container_height,
                overflow="hidden",
            ),
        )
        self._sky_pane = pn.pane.IPyWidget(
            self._sky_container,
            width=ZENITH_REVIEW_ROW_WIDTH,
            height=ZENITH_SKY_PANE_HEIGHT,
            sizing_mode="fixed",
            margin=ZENITH_SKY_ROW_MARGIN,
        )
        self._sky_status_pane = pn.pane.Markdown(
            _ZENITH_SKY_STATUS_IDLE,
            sizing_mode="stretch_width",
        )
        self._sky_status_spinner = pn.indicators.LoadingSpinner(
            value=False,
            size=20,
            name="",
        )
        self._sky_status_row = pn.Row(
            self._sky_status_spinner,
            self._sky_status_pane,
            sizing_mode="stretch_width",
            max_width=ZENITH_REVIEW_ROW_WIDTH,
            margin=(0, 0, 4, 0),
        )
        self._sky_error_alert = pn.pane.Alert(
            alert_type="danger",
            visible=False,
            sizing_mode="stretch_width",
            max_width=ZENITH_REVIEW_ROW_WIDTH,
        )
        self._zenith_footer = pn.Column(
            self._controls_row,
            self._sky_status_row,
            self._sky_error_alert,
            self._sky_pane,
            sizing_mode="stretch_width",
            max_width=ZENITH_REVIEW_ROW_WIDTH,
        )
        self._panels: dict[str, ZenithReviewPanel | None] = {
            spec.stokes: None for spec in _STOKES_SECTIONS
        }
        self._heatmap_status_panes: dict[str, pn.pane.Markdown] = {}
        for index, spec in enumerate(_STOKES_SECTIONS):
            column_margin = (
                (0, ZENITH_REVIEW_COLUMN_GAP, 0, 0) if index == 0 else (0, 0, 0, 0)
            )
            self._heatmap_status_panes[spec.stokes] = pn.pane.Markdown(
                "",
                sizing_mode="fixed",
                width=ZENITH_REVIEW_COLUMN_WIDTH,
                margin=column_margin,
            )
        self._heatmap_status_row = pn.Row(
            *[self._heatmap_status_panes[spec.stokes] for spec in _STOKES_SECTIONS],
            sizing_mode="stretch_width",
            max_width=ZENITH_REVIEW_ROW_WIDTH,
            visible=False,
            margin=(0, 0, 4, 0),
        )
        self._sky_bound_stokes: str | None = None
        self._skip_sky_watch = False
        self._ignore_slice_watcher_for_sky = False
        self._suppress_post_heatmap_sky_watchers = False
        self._sky_last_time_idx: int | None = None
        self._sky_last_stokes: str | None = None
        self.param.watch(self._sync_sky_loading_indicator, "loading_sky")

    @property
    def zenith_footer(self) -> pn.Column:
        """Shared slice controls and sky view below the heatmap row."""
        return self._zenith_footer

    @property
    def heatmap_status_row(self) -> pn.Row:
        """Per-Stokes slice summary above heatmaps (hoisted for JupyterLab push)."""
        return self._heatmap_status_row

    @property
    def slice_selection(self) -> ZenithSliceSelection:
        return self._slice_selection

    @property
    def sky_widget(self) -> SkyWidget | None:
        return self._sky_widget

    def set_push_root(self, callback: Callable[[], None] | None) -> None:
        self._slice_selection.set_push_root(callback)

    def set_extra_push_views(
        self, callback: Callable[[], list[pn.viewable.Viewable]] | None
    ) -> None:
        """Register zenith row/slot views for notebook push (nested panes may lack comms)."""
        self._extra_push_views = callback

    def _refresh_heatmap_status_row(self) -> None:
        """Update hoisted per-Stokes status markdown from shared slice indices."""
        time_idx = int(self._slice_selection.time_idx)
        freq_idx = int(self._slice_selection.freq_idx)
        for spec in _STOKES_SECTIONS:
            panel = self._panels.get(spec.stokes)
            pane = self._heatmap_status_panes[spec.stokes]
            if panel is not None:
                pane.object = panel._format_slice_status(time_idx, freq_idx)
            else:
                pane.object = ""

    def _sync_slice_controls_to_ui(self) -> None:
        """Mirror shared slice params on sliders and push embedded Panel views."""
        time_idx = int(self._slice_selection.time_idx)
        freq_idx = int(self._slice_selection.freq_idx)
        if int(self._time_slider.value) != time_idx:
            self._time_slider.value = time_idx
        if int(self._freq_slider.value) != freq_idx:
            self._freq_slider.value = freq_idx
        if self._heatmap_status_row.visible:
            self._refresh_heatmap_status_row()
        _push_panel_layout(
            self._controls_row,
            self._time_slider,
            self._freq_slider,
            self._heatmap_status_row,
            self._zenith_footer,
        )
        push = self._slice_selection._push_root
        if push is not None:
            push()

    def _push_zenith_ui(self) -> None:
        """Push hoisted heatmap status row and dashboard root (JupyterLab embedded layout)."""
        push_views: list[pn.viewable.Viewable] = [self._heatmap_status_row]
        if self._extra_push_views is not None:
            push_views.extend(self._extra_push_views())
        _push_panel_layout(*push_views)
        push = self._slice_selection._push_root
        if push is not None:
            push()

    def _sky_slice_label(self) -> str:
        return (
            f"time {int(self._slice_selection.time_idx)}, "
            f"freq {int(self._slice_selection.freq_idx)}"
        )

    def _update_sky_status(self, *, phase: str) -> None:
        """Refresh the sky status markdown (selection vs load state)."""
        stokes = self.sky_stokes
        slice_label = self._sky_slice_label()
        messages = {
            "idle": _ZENITH_SKY_STATUS_IDLE,
            "loading_dataset": f"**Sky** · opening Stokes {stokes} cube…",
            "loading_slice": f"**Sky (Stokes {stokes})** · {slice_label} · *loading slice…*",
            "ready": f"**Sky (Stokes {stokes})** · {slice_label} · ready",
            "error": f"**Sky (Stokes {stokes})** · {slice_label} · update failed",
        }
        self._sky_status_pane.object = messages.get(phase, messages["idle"])

    def _sync_sky_loading_indicator(self, *_events: param.parameterized.Event) -> None:
        self._sky_status_spinner.visible = self.loading_sky
        self._sky_status_spinner.value = self.loading_sky

    def _push_sky_ui(self) -> None:
        _push_panel_layout(
            self._sky_status_row,
            self._sky_error_alert,
            self._zenith_footer,
        )

    def _clear_sky_error(self) -> None:
        self._sky_error_alert.object = ""
        self._sky_error_alert.visible = False

    def _show_sky_error(self, message: str) -> None:
        self._sky_error_alert.object = message
        self._sky_error_alert.visible = True

    def _request_sky_update(self) -> None:
        """Refresh the sky widget for the current slice selection."""
        if self._sky_widget is None:
            return
        self._start_sky_update()

    def _start_sky_update(self) -> None:
        """Load or refresh the sky widget for the current slice selection."""
        if self._sky_widget is None:
            return

        self._sky_update_seq += 1
        request_id = self._sky_update_seq
        time_idx = int(self._slice_selection.time_idx)
        freq_idx = int(self._slice_selection.freq_idx)
        stokes = self.sky_stokes
        rebind = self._sky_bound_stokes != self.sky_stokes
        reset_center = self._pending_reset_center
        self._pending_reset_center = False
        widget = self._sky_widget
        if reset_center and widget is not None:
            # Compare to the widget's displayed slice, not _sky_last_time_idx (can be
            # ahead of the frontend after a partial sync — high-LST clicks then skip
            # recenter and the new image lands off-screen).
            reset_center = (
                int(getattr(widget, "time_idx", -1)) != time_idx
                or stokes != self._sky_last_stokes
                or rebind
            )

        self.loading_sky = True
        self._clear_sky_error()
        status_phase = "loading_dataset" if rebind else "loading_slice"
        self._update_sky_status(phase=status_phase)
        self._sync_sky_loading_indicator()
        self._push_sky_ui()

        if request_id != self._sky_update_seq:
            return
        center: SkyCoord | None = None
        fov: Any = None
        try:
            if reset_center:
                dataset = self._datasets.get(stokes)
                if dataset is None:
                    msg = f"No dataset loaded for Stokes {stokes}"
                    raise ValueError(msg)
                center = sky_view_center(dataset, time_idx)
                fov = DEFAULT_FOV_DEG * u.deg
            updated = self._execute_sky_update(
                request_id=request_id,
                rebind=rebind,
                time_idx=time_idx,
                freq_idx=freq_idx,
                stokes=stokes,
                reset_center=reset_center,
                center=center,
                fov=fov,
            )
        except Exception as exc:
            self._finish_sky_update(request_id, exc)
            return
        if not updated:
            if request_id == self._sky_update_seq:
                msg = "Sky update was skipped (stale request)"
                self._finish_sky_update(request_id, RuntimeError(msg))
            return
        self._finish_sky_update(request_id, None)

    def _execute_sky_update(
        self,
        *,
        request_id: int,
        rebind: bool,
        time_idx: int,
        freq_idx: int,
        stokes: str,
        reset_center: bool,
        center: SkyCoord | None,
        fov: Any,
    ) -> bool:
        if request_id != self._sky_update_seq:
            return False
        widget = self._sky_widget
        if widget is None:
            return False
        dataset = self._datasets.get(stokes)
        if dataset is None:
            msg = f"No dataset loaded for Stokes {stokes}"
            raise ValueError(msg)
        if rebind or self._sky_bound_stokes != stokes:
            self._bind_sky_dataset(widget, dataset)
            self._sky_bound_stokes = stokes
            self._sky_last_time_idx = None
        kwargs: dict[str, Any] = {
            "percentile_low": 2,
            "percentile_high": 98,
        }
        widget_time = getattr(widget, "time_idx", None)
        time_changed = (
            widget_time is not None and int(widget_time) != int(time_idx)
        )
        need_recenter = reset_center or time_changed
        if need_recenter:
            if center is None:
                center = sky_view_center(dataset, time_idx)
            if fov is None:
                fov = DEFAULT_FOV_DEG * u.deg
            kwargs["center"] = center
            kwargs["fov"] = fov
        revision_before = int(getattr(widget, "image_revision", 0))
        widget.update_slice(time_idx, freq_idx, **kwargs)
        if int(widget.time_idx) != int(time_idx) or int(widget.freq_idx) != int(freq_idx):
            msg = (
                f"SkyWidget slice indices ({widget.time_idx}, {widget.freq_idx}) "
                f"do not match requested ({time_idx}, {freq_idx})"
            )
            raise RuntimeError(msg)
        revision_after = int(getattr(widget, "image_revision", 0))
        if revision_after <= revision_before:
            msg = (
                f"SkyWidget did not refresh image data for slice "
                f"({time_idx}, {freq_idx})"
            )
            raise RuntimeError(msg)
        self._sky_last_time_idx = time_idx
        self._sky_last_stokes = stokes
        self._notify_sky_widget()
        return True

    def _notify_sky_widget(self) -> None:
        """Push SkyWidget state to the Jupyter frontend after a slice change."""
        widget = self._sky_widget
        if widget is None:
            return
        send_state = getattr(widget, "send_state", None)
        if callable(send_state):
            try:
                send_state()
            except Exception as exc:
                logger.debug("SkyWidget send_state failed: %s", exc, exc_info=True)
        _push_panel_layout(self._sky_pane)

    def _finish_sky_update(self, request_id: int, error: BaseException | None) -> None:
        if request_id != self._sky_update_seq:
            return
        self.loading_sky = False
        if error is not None:
            self._show_sky_error(str(error))
            self._update_sky_status(phase="error")
        else:
            self._clear_sky_error()
            self._update_sky_status(phase="ready")
        self._sync_sky_loading_indicator()
        self._push_sky_ui()

    def _sky_widget_shows_slice(self, time_idx: int, freq_idx: int, *, stokes: str) -> bool:
        """True when the sky widget already displays the requested slice."""
        widget = self._sky_widget
        if widget is None or self._sky_bound_stokes != stokes:
            return False
        if int(getattr(widget, "image_revision", 0)) <= 0:
            return False
        return int(widget.time_idx) == int(time_idx) and int(widget.freq_idx) == int(freq_idx)

    def _on_slice_selection_changed(self, *events: param.parameterized.Event) -> None:
        """Sync status and sky view when sliders or heatmap taps change the slice."""
        if self._suppress_post_heatmap_sky_watchers or self._ignore_slice_watcher_for_sky:
            return
        if self._skip_sky_watch:
            return
        time_idx = int(self._slice_selection.time_idx)
        freq_idx = int(self._slice_selection.freq_idx)
        event = events[0] if events else None
        if event is None or getattr(event, "name", None) == "time_idx":
            self._pending_reset_center = True
        if self._sky_widget_shows_slice(time_idx, freq_idx, stokes=self.sky_stokes):
            return
        self._request_sky_update()

    def _on_slice_indices_changed(self, *_events: param.parameterized.Event) -> None:
        """Refresh heatmap status text and push sliders to the JupyterLab frontend."""
        if not self._controls_row.visible:
            return
        self._sync_slice_controls_to_ui()

    def _sync_zenith_status_lines(self, *, push: bool = True) -> None:
        """Refresh hoisted heatmap status markdown and push to the JupyterLab frontend."""
        if not self._heatmap_status_row.visible:
            return
        self._refresh_heatmap_status_row()
        if push:
            self._push_zenith_ui()

    def _on_sky_stokes_changed(self, *_events: param.parameterized.Event) -> None:
        if self._skip_sky_watch or self._suppress_post_heatmap_sky_watchers:
            return
        self._pending_reset_center = True
        self._request_sky_update()

    def select_slice_from_heatmap(self, stokes: str, time_idx: int, freq_idx: int) -> None:
        """Set shared time/frequency and switch sky view to the clicked heatmap's Stokes."""
        t_lo, t_hi = self._slice_selection.param.time_idx.bounds
        f_lo, f_hi = self._slice_selection.param.freq_idx.bounds
        time_idx = int(np.clip(time_idx, t_lo, t_hi))
        freq_idx = int(np.clip(freq_idx, f_lo, f_hi))

        def _apply() -> None:
            self._skip_sky_watch = True
            self._ignore_slice_watcher_for_sky = True
            self._suppress_post_heatmap_sky_watchers = True
            try:
                with param.parameterized.batch_call_watchers(self):
                    if stokes in self.param.sky_stokes.objects:
                        self.sky_stokes = stokes
                with param.parameterized.batch_call_watchers(self._slice_selection):
                    self._slice_selection.time_idx = time_idx
                    self._slice_selection.freq_idx = freq_idx
                self._pending_reset_center = True
                self._request_sky_update()
                self._sync_slice_controls_to_ui()
            finally:
                self._skip_sky_watch = False
                self._suppress_post_heatmap_sky_watchers = False
                self._ignore_slice_watcher_for_sky = False

        _run_on_main_thread(_apply)

    @staticmethod
    def _bind_sky_dataset(widget: SkyWidget, dataset: xr.Dataset) -> None:
        """Load cube only; first frame comes from :meth:`update_slice`."""
        bind_sky_widget_dataset(widget, dataset)

    def bind_datasets(self, datasets: dict[str, xr.Dataset]) -> None:
        """Remember loaded Stokes datasets and constrain the sky-view toggle."""
        self._datasets = dict(datasets)
        self._sky_bound_stokes = None
        available = [spec.stokes for spec in _STOKES_SECTIONS if spec.stokes in self._datasets]
        with param.parameterized.batch_call_watchers(self):
            self.param.sky_stokes.objects = available
            if available:
                if self.sky_stokes not in available:
                    self.sky_stokes = available[0]
            toggle_options = {f"Stokes {stokes}": stokes for stokes in available}
            self._stokes_toggle.options = toggle_options
            self._stokes_toggle.disabled = len(available) < 2

    def mount_sky(self) -> None:
        """Create the shared SkyWidget below the heatmaps when datasets are ready."""
        self.reset_sky()
        if not self._datasets:
            return

        self._sky_widget = SkyWidget()
        self._sky_widget.colormap = "inferno"
        self._sky_widget.background_survey = ""
        self._sky_widget.invert_horizontal_pan = True
        width = f"{ZENITH_SKY_WIDGET_WIDTH}px"
        height = f"{ZENITH_SKY_WIDGET_HEIGHT}px"
        self._sky_widget.layout = widgets.Layout(
            width=width,
            min_width=width,
            max_width=width,
            height=height,
            min_height=height,
            max_height=height,
        )
        self._sky_container.children = [
            widgets.HTML("<strong>Sky view</strong>"),
            self._sky_widget,
        ]
        self._sky_bound_stokes = None
        self._pending_reset_center = True
        self._update_sky_status(phase="idle")
        self._request_sky_update()

    def reset_sky(self) -> None:
        """Restore the sky-view placeholder."""
        self._sky_update_seq += 1
        self.loading_sky = False
        self._sky_widget = None
        self._sky_bound_stokes = None
        self._sky_last_time_idx = None
        self._sky_last_stokes = None
        self._ignore_slice_watcher_for_sky = False
        self._suppress_post_heatmap_sky_watchers = False
        self._clear_sky_error()
        self._update_sky_status(phase="idle")
        self._sync_sky_loading_indicator()
        self._sky_container.children = [widgets.HTML(_ZENITH_SKY_PLACEHOLDER)]

    def _configure_slice_selection(self) -> None:
        """Set shared slider bounds and default slice from the first loaded panel."""
        primary: ZenithReviewPanel | None = None
        for spec in _STOKES_SECTIONS:
            panel = self._panels.get(spec.stokes)
            if panel is not None:
                primary = panel
                break

        if primary is None:
            self._controls_row.visible = False
            self._heatmap_status_row.visible = False
            return

        default_time, default_freq = default_stat_slice(primary._stat_map, primary._dataset)
        self._slice_selection.configure(
            n_times=primary._n_times,
            n_freqs=primary._n_freqs,
            default_time=default_time,
            default_freq=default_freq,
        )
        self._controls_row.visible = True
        self._heatmap_status_row.visible = True

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
        self._controls_row.visible = False
        self._heatmap_status_row.visible = False
        for pane in self._heatmap_status_panes.values():
            pane.object = ""
        self.set_extra_push_views(None)
        self.reset_sky()
        self._datasets.clear()

    def build_section_contents(
        self,
        datasets: dict[str, xr.Dataset],
        log: LogFn,
        flush: Callable[[], None] | None = None,
    ) -> dict[str, pn.viewable.Viewable]:
        """Build Stokes I/V review panel content for stable zenith section slots."""
        self._dispose_panels()
        self.bind_datasets(datasets)
        contents = {
            spec.stokes: self._build_section_content_for_spec(
                spec,
                datasets,
                log,
                flush=flush,
            )
            for spec in _STOKES_SECTIONS
        }
        self._finalize_section_build()
        return contents

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
            slice_selection=self._slice_selection,
            stokes_label=spec.stokes,
            metric_label=spec.metric_label,
            on_heatmap_select=self.select_slice_from_heatmap,
        )
        self._panels[spec.stokes] = panel
        log(
            f"Stokes {spec.stokes} review panel ready "
            f"({time.perf_counter() - widget_started:.1f}s)."
        )
        if flush is not None:
            flush()
        return panel.heatmap_column

    def _finalize_section_build(self) -> None:
        """Configure shared slice controls after all Stokes panels are built."""
        self._configure_slice_selection()
        self._sync_zenith_status_lines()

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
    slice_selection = ZenithSliceSelection()
    n_times = int(dataset.sizes["time"])
    n_freqs = int(dataset.sizes["frequency"])
    default_time, default_freq = default_stat_slice(stat_map, dataset)
    slice_selection.configure(
        n_times=n_times,
        n_freqs=n_freqs,
        default_time=default_time,
        default_freq=default_freq,
    )
    holder = _StokesReviewHolder()
    holder.bind_datasets({stokes_label: dataset})
    holder.sky_stokes = stokes_label
    panel = ZenithReviewPanel(
        dataset,
        stat_map,
        slice_selection=holder.slice_selection,
        stokes_label=stokes_label,
        metric_label=metric_label,
    )
    holder._panels[stokes_label] = panel
    holder._configure_slice_selection()
    holder.mount_sky()
    return pn.Column(panel.layout, holder.zenith_footer)


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

    def __init__(
        self,
        *,
        qa_config: PipelineQAConfig | None = None,
        **params: Any,
    ) -> None:
        super().__init__(**params)
        self._qa_config = qa_config or PipelineQAConfig.default()
        self._coverage: pd.DataFrame = pd.DataFrame()
        self._scroll_log = ScrollLog()
        self._summary_df: pd.DataFrame = pd.DataFrame()
        self._loaded_day: str | None = None
        self._stokes_review = _StokesReviewHolder()
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
            self._stokes_review.heatmap_status_row,
            self._zenith_review_row,
            self._stokes_review.zenith_footer,
            sizing_mode="stretch_width",
        )
        self._zenith_load_button = pn.widgets.Button(
            name="Reload zenith panels",
            button_type="default",
            disabled=True,
        )
        self._zenith_load_button.on_click(self._on_zenith_load_click)
        self._qa_grid = pn.Column(
            pn.pane.Markdown(
                "*Select an observation day to build the thermal-noise PNG grid "
                "and dewarp summary heatmap.*"
            ),
            sizing_mode="stretch_width",
        )
        self._flux_ratio_grid = pn.Column(
            pn.pane.Markdown(
                "*Hybrid flux ratio plots (imfit / model) appear after a day is loaded.*"
            ),
            sizing_mode="stretch_width",
        )
        self._day_selector = pn.widgets.Select.from_param(
            self.param.select_day,
            name="Observation day",
            width=220,
        )
        self._day_selector.param.watch(self._on_day_selector_value, "value")
        self._programmatic_day_sync = False
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
        # Keep the day dropdown interactive while QA/zenith loads run in the background.
        self._day_selector.disabled = self.scanning or self.converting
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

    def _reset_thermal_qa_section(self) -> None:
        """Restore thermal-noise PNG grid and dewarp summary placeholders."""
        self._qa_grid.objects = [
            pn.pane.Markdown(
                "*Select an observation day to build the thermal-noise PNG grid "
                "and dewarp summary heatmap.*"
            ),
        ]

    def _reset_flux_ratio_grid(self) -> None:
        """Restore flux ratio placeholder content."""
        self._flux_ratio_grid.objects = [
            pn.pane.Markdown(
                "*Hybrid flux ratio plots (imfit / model) appear after a day is loaded.*"
            ),
        ]

    def _build_thermal_qa_section(self, select_day: str) -> pn.Column:
        """Thermal-noise PNG tile grid with dewarp median-shift heatmap below."""
        thermal_grid = build_thermal_noise_grid(
            self._summary_df,
            select_day,
            n_cols=self._qa_config.thermal_noise_grid_cols,
            thermal_noise_plot_name=self._qa_config.thermal_noise_plot_name,
            open_full_size=self._open_modal,
        )
        dewarp_df = load_dewarp_summary_dataframe(
            select_day,
            self._coverage,
            config=self._qa_config,
        )
        dewarp_panel = build_dewarp_shift_panel(dewarp_df)
        if not dewarp_df.empty:
            self._log(
                f"Built dewarp median-shift heatmap from {len(dewarp_df)} row(s) "
                f"({self._qa_config.dewarp_summary_csv_glob})."
            )
        return pn.Column(
            thermal_grid,
            pn.pane.Markdown("#### Dewarp median shift (LST × frequency)"),
            dewarp_panel,
            sizing_mode="stretch_width",
        )

    def _build_flux_ratio_grid(self, select_day: str) -> pn.Column:
        """Load flux-check CSVs and build the Bokeh heatmap grid for one day."""
        flux_df = load_flux_check_hybrid_dataframe(
            select_day,
            self._coverage,
            config=self._qa_config,
        )
        if flux_df.empty:
            return pn.Column(
                pn.pane.Markdown(
                    f"*No flux_check_hybrid.csv files found "
                    f"({self._qa_config.flux_check_csv_glob}) for this day.*"
                ),
                sizing_mode="stretch_width",
            )
        figures = build_flux_ratio_figures(flux_df)
        sources = ", ".join(sorted(figures))
        self._log(
            f"Built hybrid flux ratio plots for {len(figures)} source(s): {sources} "
            f"({len(flux_df)} CSV row(s))."
        )
        return build_flux_ratio_panel_grid(figures)

    def _reset_zenith_sections(self, *, banner: str | None = None) -> None:
        """Restore zenith placeholders without replacing the displayed layout tree."""
        self._stokes_review.dispose()
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
        # Push zenith row + full layout (nested status panes may not have their own comms).
        self._stokes_review.set_push_root(self._push_zenith_root)
        self._stokes_review.set_extra_push_views(
            lambda: [
                self._stokes_review.heatmap_status_row,
                self._zenith_review_row,
                self._zenith_slot,
            ]
        )
        for spec in _STOKES_SECTIONS:
            self._zenith_section_content[spec.stokes].objects = [
                section_contents[spec.stokes],
            ]
        self._stokes_review.mount_sky()
        self._execute(self._push_zenith_root)

    def _sync_day_selector(self, days: list[str], value: str | None) -> None:
        """Update day options and selection from scan results."""
        normalized_days = [str(day) for day in days]
        normalized_value = str(value) if value is not None else None
        self._programmatic_day_sync = True
        try:
            with param.parameterized.batch_call_watchers(self):
                self.param.select_day.objects = normalized_days
                self.select_day = normalized_value
        finally:
            self._programmatic_day_sync = False

    def _supersede_inflight_work(self) -> None:
        """Cancel in-flight zenith/day work so a new observation day can load."""
        self._load_seq += 1
        self.loading_zenith = False
        self.loading_day = False

    def _on_day_selector_value(self, event: param.parameterized.Event) -> None:
        """Handle user-driven changes from the Panel day dropdown."""
        if self._programmatic_day_sync:
            return
        new_day = event.new
        if new_day is None:
            return
        self._handle_day_selection(str(new_day), previous=event.old)

    def _handle_day_selection(self, new_day: str, *, previous: Any = None) -> None:
        """Load QA content for one observation day selected in the dropdown."""
        if self.scanning or self._coverage.empty:
            return
        old_day = previous
        if (
            old_day not in (None, param.Undefined)
            and str(old_day) == new_day
            and self._loaded_day == new_day
        ):
            return
        if self._loaded_day != new_day:
            self._supersede_inflight_work()
            self._release_active_datasets()
            self._reset_zenith_sections()
            self._reset_flux_ratio_grid()
            self._reset_thermal_qa_section()
            self._qa_grid.objects = [
                pn.pane.Markdown("*Loading thermal-noise QA grid and dewarp summary…*"),
            ]

            def _push() -> None:
                self._push_panel_roots()

            self._execute(_push)
        if self.select_day != new_day:
            self._programmatic_day_sync = True
            try:
                with param.parameterized.batch_call_watchers(self):
                    self.select_day = new_day
            finally:
                self._programmatic_day_sync = False
        self._begin_load_day()

    def _on_select_day_changed(self, event: param.parameterized.Event) -> None:
        """Backward-compatible alias; prefer :meth:`_handle_day_selection`."""
        if event.new is None:
            return
        self._handle_day_selection(str(event.new), previous=event.old)

    def _on_zenith_load_click(self, _event: Any) -> None:
        self._begin_zenith_load()

    def _sync_zenith_button(self) -> None:
        """Enable manual zenith reload once the selected day has been loaded."""
        if self.select_day is None:
            self._zenith_load_button.disabled = True
            return
        status = zarr_status(self.select_day, config=self._qa_config)
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
        status = zarr_status(self.select_day, config=self._qa_config)
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
                coverage = scan_coverage(config=self._qa_config)
                days = qa_days(coverage)

                def _apply() -> None:
                    pending_day = self.select_day
                    self._coverage = coverage
                    self.scanning = False
                    if not days:
                        self._log_error(
                            f"No {self._qa_config.qa_run_label} QA days found under the pipeline root."
                        )
                        self._sync_day_selector([], None)
                    else:
                        self._clear_error()
                        preferred = (
                            pending_day if pending_day in days else None
                        )
                        self._sync_day_selector(days, preferred)
                        if preferred is not None:
                            self._begin_load_day()
                            self._log(
                                f"Found {len(days)} {self._qa_config.qa_run_label} QA day(s). "
                                f"Loading QA data for {preferred}…"
                            )
                        else:
                            self._log(
                                f"Found {len(days)} {self._qa_config.qa_run_label} QA day(s). "
                                "Select a day from the dropdown to load QA data."
                            )
                    self._sync_log()

                self._execute(_apply)
            except Exception as exc:
                err = exc

                def _fail(err: BaseException = err) -> None:
                    self.scanning = False
                    self._log_error(f"Scan failed: {err}")

                self._execute(_fail)

        threading.Thread(target=_run, daemon=True).start()

    def _begin_load_day(self) -> None:
        """Load thermal-noise summary and QA grid for the selected day."""
        if self.converting:
            return
        if self.select_day is None:
            return
        if self._coverage.empty:
            return

        select_day = self.select_day
        self._load_seq += 1
        load_seq = self._load_seq
        self.loading_day = True
        self._log(f"Loading QA data for {select_day}…")
        self._flush_log()
        self._release_active_datasets()
        self._stokes_review.dispose()
        self._start_day_load_thread(select_day, load_seq)

    def _start_day_load_thread(self, select_day: str, load_seq: int) -> None:
        """Run day QA loading off the notebook UI thread."""

        def _run() -> None:
            self._run_day_load(select_day, load_seq)

        threading.Thread(target=_run, daemon=True).start()

    def _auto_load_zenith_if_ready(self, load_seq: int) -> None:
        """Start zenith review automatically after a successful day load."""
        if not self._is_current_load(load_seq):
            return
        if self.select_day is None or self._loaded_day != self.select_day:
            return
        status = zarr_status(self.select_day, config=self._qa_config)
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
                status = zarr_status(select_day, config=self._qa_config)
                if not (status["I"] or status["V"]):
                    section_contents = self._stokes_review.build_no_zarr_contents()
                    banner = _NO_QA_ZARR_MESSAGE
                else:
                    datasets = load_qa_datasets(
                        select_day,
                        self._log,
                        flush=self._flush_log,
                        config=self._qa_config,
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
                        "*Zenith heatmaps loaded. Use the sliders and Stokes toggle below "
                        "the heatmaps, or click a cell, to inspect slices.*"
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

                err = exc
                tb = traceback.format_exc()

                def _fail(err: BaseException = err, tb: str = tb) -> None:
                    if not self._is_current_load(load_seq):
                        self._finish_zenith_load(load_seq=load_seq)
                        return
                    self._log_error(f"Failed to load zenith panels for {select_day}: {err}")
                    self._log(tb, sync=False)
                    self._reset_zenith_sections()
                    self._finish_zenith_load(load_seq=load_seq)

                _schedule_ipython_main(_fail)

        threading.Thread(target=_run, daemon=True).start()

    def _finish_zenith_load(self, *, load_seq: int | None = None) -> None:
        """Clear zenith loading state and refresh controls."""
        self.loading_zenith = False
        if load_seq is not None and not self._is_current_load(load_seq):
            return
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

        status = zarr_status(select_day, config=self._qa_config)
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
        self._qa_grid.objects = [self._build_thermal_qa_section(select_day)]
        self._flux_ratio_grid.objects = [self._build_flux_ratio_grid(select_day)]
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
        self.loading_day = False
        if load_seq is not None and not self._is_current_load(load_seq):
            return
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
        self._start_day_load_thread(select_day, load_seq)

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
                    config=self._qa_config,
                )
            except Exception as exc:
                err = exc

                def _fail(err: BaseException = err) -> None:
                    self.converting = False
                    self._log_error(f"Conversion failed: {err}")
                    self._sync_log()

                _schedule_ipython_main(_fail)
                return

            def _after_convert() -> None:
                # Clear converting before refresh so _auto_load_zenith_if_ready is not blocked.
                self.converting = False
                status = zarr_status(select_day, config=self._qa_config)
                if not (status["I"] or status["V"]):
                    self._log_error(
                        "Conversion finished but no QA Zarr store was created. "
                        "Check the notebook/kernel log for errors after the staging line."
                    )
                else:
                    self._load_day(silent=False)
                self._sync_log()

            _schedule_ipython_main(_after_convert)

        threading.Thread(target=_run, daemon=True).start()

    def _build_layouts(self) -> None:
        """Build the single Panel layout with sky widgets below the heatmaps."""
        if self._layout is not None:
            return

        header = pn.pane.Markdown(
            "# Pipeline QA check\n\n"
            "Scan finds available days automatically. Select a day to load Stokes I/V "
            "zenith review, hybrid flux ratio plots, thermal-noise PNG grid, and dewarp summary. "
            "A shared sky view appears below the heatmaps."
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
            pn.pane.Markdown("### Hybrid flux ratio (imfit / model)"),
            self._flux_ratio_grid,
            pn.pane.Markdown("### Thermal-noise QA (PNG grid & dewarp summary)"),
            self._qa_grid,
            self._modal_container,
            sizing_mode="stretch_width",
        )

    def panel(self) -> pn.Column:
        """Return the JupyterLab dashboard layout."""
        self._build_layouts()

        _schedule_initial_scan(self)

        assert self._layout is not None
        return self._layout


def _schedule_initial_scan(app: PipelineQAApp) -> None:
    """Start the pipeline scan once, even if ``pn.state.onload`` never fires."""
    if app._scan_started:
        return
    app._scan_started = True
    if pn.state.loaded:
        app._start_initial_scan()
    else:
        pn.state.onload(app._start_initial_scan)


def display_pipeline_qa_app(
    app: PipelineQAApp | None = None,
    *,
    qa_config: PipelineQAConfig | None = None,
    pipeline_root: Path | str | None = None,
    symlink_root: Path | str | None = None,
    zarr_root: Path | str | None = None,
    i_fits_glob: str | None = None,
    v_fits_glob: str | None = None,
) -> PipelineQAApp:
    """Display the QA dashboard in JupyterLab (single Panel document + embedded sky view).

    Parameters
    ----------
    app
        Existing app instance. When omitted, a new :class:`PipelineQAApp` is created.
    qa_config
        Full configuration object. Individual path/glob arguments override fields on
        ``qa_config`` when both are provided.
    pipeline_root
        Root of the exopipe tree to scan for QA runs (phase1 Wideband or phase2 Science).
    symlink_root
        Directory for FITS symlink staging and fixed-header side products during conversion.
    zarr_root
        Directory where per-day Stokes I/V QA Zarr stores are written and loaded from.
    i_fits_glob, v_fits_glob
        Glob patterns (under ``{run}/*/{I|V}/deep/``) used for FITS→Zarr conversion.
    """
    from IPython.display import display

    try:
        pn.extension("ipywidgets")
    except Exception as exc:
        logger.debug("Panel ipywidgets extension unavailable: %s", exc)

    resolved_config = resolve_pipeline_qa_config(
        config=qa_config,
        pipeline_root=pipeline_root,
        symlink_root=symlink_root,
        zarr_root=zarr_root,
        i_fits_glob=i_fits_glob,
        v_fits_glob=v_fits_glob,
    )
    app = app or PipelineQAApp(qa_config=resolved_config)
    app._build_layouts()

    _schedule_initial_scan(app)

    assert app._layout is not None
    display(app._layout)
    return app
