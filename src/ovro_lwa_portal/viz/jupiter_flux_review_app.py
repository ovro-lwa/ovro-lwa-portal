"""Panel Jupiter flux review app (phase2 QA Zarr dynspec + SkyWidget).

Extracted from ``notebooks/jupiter_flux_review.ipynb`` for testability. The
notebook keeps path/config constants and calls :func:`configure_jupiter_flux_review_notebook`
before constructing :class:`JupiterFluxReview`.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import ipywidgets as widgets
import numpy as np
import ovro_lwa_portal as ovro
import panel as pn
import param
import xarray as xr
import astropy.units as u
from astropy.coordinates import SkyCoord
from astrowidget import SkyWidget
from bokeh.events import Tap
from bokeh.models import ColumnDataSource, FixedTicker, HoverTool
from bokeh.plotting import figure

from ovro_lwa_portal.viz.jupiter_flux_review_data import (
    JupiterLoad,
    format_flux_hover,
    jupiter_at_observation_start,
    jupiter_color_mapper,
    jupiter_flux_map,
    list_phase2_i_qa_zarrs,
    zarr_path_to_day,
)
from ovro_lwa_portal.viz.panel_ui_session import (
    CallbackPanelUISession,
    JupyterPanelUISession,
    PanelUISession,
)
from ovro_lwa_portal.viz.pipeline_qa import PipelineQAConfig
from ovro_lwa_portal.viz.pipeline_qa_app import (
    ACTIVITY_LOG_HEIGHT_PX,
    ScrollLog,
    _capture_ipython_io_loop,
    _format_activity_log_html,
    _patch_astrowidget_get_wcs,
    bind_sky_widget_dataset,
    schedule_when_panel_loaded,
)
from ovro_lwa_portal.viz.source_review import run_dataset_load
from ovro_lwa_portal.viz.source_review_data import (
    _PROGRESS_STAGE_LABELS,
    _format_lst_hour_label,
    _format_patch_fit_diagnostics,
    _heatmap_index_from_coord,
    _patch_fit_hover_columns,
    lst_hours_for_dataset,
)

__all__ = [
    "JupiterFluxReview",
    "JupiterFluxReviewConfig",
    "configure_jupiter_flux_review_notebook",
]

JUPITER_SKY_FOV_DEG = 10.0


@dataclass(frozen=True)
class JupiterFluxReviewConfig:
    """Runtime tuning knobs normally set in the notebook config cell."""

    zarr_lm_chunk: int = 512


def _placeholder_dynspec_figure() -> figure:
    plot = figure(
        width=1000,
        height=400,
        x_range=(0, 1),
        y_range=(0, 1),
        tools="",
        active_drag=None,
        active_tap=None,
        title="Dynamic spectrum loads after selecting a QA Zarr store…",
    )
    plot.xaxis.visible = False
    plot.yaxis.visible = False
    plot.outline_line_alpha = 0
    return plot


def configure_jupiter_flux_review_notebook() -> None:
    """One-time notebook setup: astrowidget patch, Panel extension, io_loop capture."""
    _patch_astrowidget_get_wcs()
    pn.extension("bokeh", sizing_mode="stretch_width")
    _capture_ipython_io_loop()


class JupiterFluxReview(param.Parameterized):
    """Dynamic spectrum toward Jupiter with linked SkyWidget."""

    select_zarr = param.Selector(default=None, objects=[], doc="Phase2 QA Zarr store.")
    flux_method = param.Selector(
        default="dynamic_spectrum",
        objects=["dynamic_spectrum", "patch_fit", "patch_max"],
        doc="Flux: tracked pixel, Gaussian patch fit, or patch maximum.",
    )
    loading = param.Boolean(default=False)
    status = param.String(default="Select a QA Zarr store.")
    log_text = param.String(default="")

    def __init__(
        self,
        config: PipelineQAConfig,
        *,
        patch_fit_scale: float = 3.0,
        patch_fit_max_reduced_chi_squared: float = 200.0,
        review_config: JupiterFluxReviewConfig | None = None,
        dispatch_override: Callable[[Callable[[], None]], None] | None = None,
        ui_session: PanelUISession | None = None,
        **params: Any,
    ) -> None:
        self._config = config
        self._review_config = review_config or JupiterFluxReviewConfig()
        self._patch_fit_scale = float(patch_fit_scale)
        self._patch_fit_max_reduced_chi_squared = float(patch_fit_max_reduced_chi_squared)
        self._dispatch_override = dispatch_override
        self._ui_session_override = ui_session
        self._ui_session: PanelUISession | None = None
        self._scroll_log = ScrollLog()
        self._zarr_paths: dict[str, Path] = {}
        self._dataset: xr.Dataset | None = None
        self._dynspec: xr.DataArray | None = None
        self._patch_fit_result: object | None = None
        self._jupiter: SkyCoord | None = None
        self._lst_hours: np.ndarray | None = None
        self._freq_mhz: np.ndarray | None = None
        self._sky_widget: SkyWidget | None = None
        self._time_idx = 0
        self._freq_idx = 0
        self._loaded_label: str | None = None
        self._initial_load_scheduled = False
        self._load_job_id = 0

        zarr_paths = list_phase2_i_qa_zarrs(config)
        labels: list[str] = []
        for path in zarr_paths:
            day = zarr_path_to_day(path, stem=config.i_qa_zarr_stem)
            label = f"{day} ({path.name})"
            labels.append(label)
            self._zarr_paths[label] = path

        default = labels[-1] if labels else None
        super().__init__(select_zarr=default, **params)
        self.param.select_zarr.objects = labels

        self._heatmap_pane = pn.pane.Bokeh(
            _placeholder_dynspec_figure(),
            height=420,
            sizing_mode="stretch_width",
        )
        self._sky_container = widgets.VBox(
            children=[widgets.HTML("<i>Sky view loads after selecting a Zarr store.</i>")],
            layout=widgets.Layout(width="100%", min_height="620px"),
        )
        self._sky_pane = pn.pane.IPyWidget(
            self._sky_container,
            height=620,
            sizing_mode="stretch_width",
        )
        self._status_pane = pn.pane.Markdown("")
        self._log_widget = widgets.HTML(
            value=_format_activity_log_html(""),
            layout=widgets.Layout(
                width="100%",
                height=f"{ACTIVITY_LOG_HEIGHT_PX}px",
                border="1px solid #ccc",
            ),
        )
        self._log_pane = pn.pane.IPyWidget(
            self._log_widget,
            sizing_mode="stretch_width",
            height=ACTIVITY_LOG_HEIGHT_PX,
        )
        self._selector = pn.widgets.Select.from_param(
            self.param.select_zarr,
            name="Phase2 QA Zarr (Stokes I)",
            width=520,
        )
        self._flux_method_selector = pn.widgets.Select.from_param(
            self.param.flux_method,
            name="Flux method",
            width=200,
        )
        self._spinner = pn.indicators.LoadingSpinner(value=False, size=24, name="")
        self._layout = pn.Column(
            pn.Row(
                self._selector,
                self._flux_method_selector,
                self._spinner,
                margin=(0, 0, 8, 0),
            ),
            pn.Column(
                pn.pane.Markdown("**Activity log**"),
                self._log_pane,
                sizing_mode="stretch_width",
            ),
            self._status_pane,
            self._heatmap_pane,
            self._sky_pane,
            sizing_mode="stretch_width",
            max_width=1048,
        )
        self.param.watch(self._on_select_zarr, "select_zarr")
        self.param.watch(self._on_flux_method_change, "flux_method")
        if labels:
            self._log(f"Found {len(labels)} phase2 QA Zarr store(s).")
        else:
            self._log("No phase2 QA Zarr stores found under the configured root.")
            self._set_status("No phase2 QA Zarr stores found under the configured root.")
        self._log_dask_dashboard()
        self._refresh_log_widget()

    @property
    def panel(self) -> pn.Column:
        self._ensure_initial_load()
        return self._layout

    def _notebook_ui_views(self) -> tuple[pn.viewable.Viewable, ...]:
        return (
            self._layout,
            self._status_pane,
            self._spinner,
            self._heatmap_pane,
            self._selector,
            self._flux_method_selector,
        )

    @property
    def _ui(self) -> PanelUISession:
        if self._ui_session is None:
            if self._ui_session_override is not None:
                self._ui_session = self._ui_session_override
            elif self._dispatch_override is not None:
                self._ui_session = CallbackPanelUISession(
                    self._dispatch_override,
                    root_views=self._notebook_ui_views,
                )
            else:
                self._ui_session = JupyterPanelUISession(self._notebook_ui_views)
        return self._ui_session

    def _dispatch(self, callback: Callable[[], None]) -> None:
        self._ui.dispatch(callback)

    def _publish_dynspec_figure(self, figure: object) -> None:
        """Push the dynspec Bokeh model via the validated publish path."""
        self._ui.publish_bokeh_figure(self._heatmap_pane, figure)

    def _ensure_initial_load(self) -> None:
        if self._initial_load_scheduled or self.select_zarr is None:
            return
        if self._loaded_label == self.select_zarr and self._dataset is not None:
            return
        self._initial_load_scheduled = True
        _capture_ipython_io_loop()
        schedule_when_panel_loaded(self._load_selected_zarr)

    def _on_select_zarr(self, *_events: param.parameterized.Event) -> None:
        if self.select_zarr is None:
            return
        self._load_selected_zarr()

    def _on_flux_method_change(self, *_events: param.parameterized.Event) -> None:
        if self.select_zarr is not None and not self.loading:
            self._load_selected_zarr()

    def _flux_method_label(self) -> str:
        if self.flux_method == "patch_fit":
            return "patch_fit (shifted Gaussian)"
        if self.flux_method == "patch_max":
            return "patch_max (patch maximum)"
        return "dynamic_spectrum (tracked pixel)"

    def _set_status(self, text: str) -> None:
        self.status = text

    @param.depends("status", watch=True)
    def _sync_status_pane(self) -> None:
        self._ui.sync_status_pane(self._status_pane, self.status)

    def _refresh_log_widget(self) -> None:
        self._log_widget.value = _format_activity_log_html(self.log_text)

    def _sync_log(self) -> None:
        self.log_text = self._scroll_log.text
        self._refresh_log_widget()

    def _log(self, message: str) -> None:
        self._scroll_log.append(message)
        self._sync_log()

    def _log_dask_dashboard(self) -> None:
        try:
            from dask.distributed import get_client

            client = get_client()
            self._log(f"Dask dashboard: {client.dashboard_link}")
        except Exception:
            pass

    def _sync_spinner(self, value: bool) -> None:
        self._ui.sync_spinner(self._spinner, value=value, visible=value)

    def _flux_progress_callback(self, job_id: int) -> Callable[[str, int, int, str], None]:
        last_key: dict[str, tuple[int, str]] = {}

        def _callback(stage: str, current: int, total: int, message: str) -> None:
            if job_id != self._load_job_id:
                return
            if stage == "track" and total > 1 and current not in (0, total):
                return
            if stage in ("extract", "track") and total > 1:
                key = (current, message)
                if current not in (0, total) and last_key.get(stage) == key:
                    return
                last_key[stage] = key
            label = _PROGRESS_STAGE_LABELS.get(stage, stage)
            if "in progress" in message or (stage == "extract" and current < total):
                text = f"{label}: {message}"
            else:
                pct = int(round(100.0 * int(current) / int(total))) if total else 0
                text = f"{label}: {message} ({current}/{total}, {pct}%)"

            def _push() -> None:
                self._log(text)

            self._dispatch(_push)

        return _callback

    def _load_selected_zarr(self) -> None:
        label = self.select_zarr
        if label is None:
            self._set_status("No phase2 QA Zarr stores found under the configured root.")
            return
        path = self._zarr_paths[label]
        self._load_job_id += 1
        job_id = self._load_job_id

        def _start() -> None:
            self.loading = True
            self._sync_spinner(True)
            self._log(f"Loading {path.name}…")
            self._set_status("Loading…")

        self._dispatch(_start)

        def _open(report: Callable[[str], None]) -> JupiterLoad:
            t0 = time.perf_counter()
            report(f"Opening Stokes I Zarr at {path}…")
            ds = ovro.open_dataset(path, chunks="auto").chunk(
                {
                    "l": self._review_config.zarr_lm_chunk,
                    "m": self._review_config.zarr_lm_chunk,
                }
            )
            report(
                f"Opened Zarr ({int(ds.sizes['time'])} times, "
                f"{int(ds.sizes['frequency'])} frequencies, "
                f"{int(ds.sizes['l'])}×{int(ds.sizes['m'])} pixels)."
            )
            report("Computing Jupiter ephemeris at observation start…")
            jupiter = jupiter_at_observation_start(ds)
            method_label = self._flux_method_label()
            report(
                f"Extracting flux map via {method_label} toward Jupiter "
                f"(RA={float(jupiter.ra.deg):.3f}°, Dec={float(jupiter.dec.deg):.3f}°)…"
            )
            dynspec, patch_fit = jupiter_flux_map(
                ds,
                jupiter,
                method=self.flux_method,
                patch_fit_scale=self._patch_fit_scale,
                patch_fit_max_reduced_chi_squared=self._patch_fit_max_reduced_chi_squared,
                progress_callback=self._flux_progress_callback(job_id),
            )
            report(f"Flux extraction finished in {time.perf_counter() - t0:.1f} s ({method_label}).")
            lst_hours = lst_hours_for_dataset(ds)
            freq_mhz = np.asarray(ds.coords["frequency"].values, dtype=np.float64) / 1e6
            return JupiterLoad(
                dataset=ds,
                dynspec=dynspec,
                jupiter=jupiter,
                lst_hours=lst_hours,
                freq_mhz=freq_mhz,
                patch_fit_result=patch_fit,
            )

        def _work() -> None:
            run_dataset_load(
                open_dataset=_open,
                dispatch=self._dispatch,
                on_loaded=lambda load: self._finish_load(load, job_id=job_id),
                on_error=lambda exc: self._finish_load(None, job_id=job_id, error=exc),
                log=self._log,
            )

        threading.Thread(target=_work, daemon=True).start()

    def _finish_load(
        self,
        load: JupiterLoad | None,
        *,
        job_id: int,
        error: BaseException | None = None,
    ) -> None:
        if job_id != self._load_job_id:
            return

        def _apply() -> None:
            self.loading = False
            self._sync_spinner(False)
            if error is not None:
                self._log(f"ERROR: {error}")
                self._set_status(f"**Load failed:** {error}")
                return

            assert load is not None
            label = self.select_zarr
            assert label is not None

            self._dataset = load.dataset
            self._dynspec = load.dynspec
            self._patch_fit_result = load.patch_fit_result
            self._jupiter = load.jupiter
            self._lst_hours = load.lst_hours
            self._freq_mhz = load.freq_mhz
            self._loaded_label = label

            if load.patch_fit_result is not None:
                n_accepted = int(load.patch_fit_result.fit_accepted_map.sum().values)
                n_cells = int(load.patch_fit_result.fit_accepted_map.size)
                self._log(
                    f"patch_fit quality: {n_accepted}/{n_cells} cells accepted "
                    f"(χ²_red ≤ {self._patch_fit_max_reduced_chi_squared:g})"
                )

            jupiter = load.jupiter
            jupiter_ra = jupiter.ra.to_string(unit=u.hour, precision=1)
            jupiter_dec = jupiter.dec.to_string(unit=u.deg, precision=1)
            self._set_status(
                f"**Jupiter at observation start:** RA={jupiter_ra}, Dec={jupiter_dec} · "
                f"{int(load.dataset.sizes['time'])} times × "
                f"{int(load.dataset.sizes['frequency'])} frequencies · "
                f"Flux: {self._flux_method_label()} · "
                "Click the dynamic spectrum to center the sky view on Jupiter."
            )

            self._mount_sky_widget(load.dataset)
            self._time_idx, self._freq_idx = self._default_slice(load.dynspec.values)
            self._publish_dynspec_figure(self._build_dynspec_figure(load.dynspec.values))
            self._update_sky(self._time_idx, self._freq_idx)
            self._log(
                f"Ready — Jupiter dynamic spectrum and sky view loaded for "
                f"{self._zarr_paths[label].name}."
            )

        self._dispatch(_apply)

    def _default_slice(self, values: np.ndarray) -> tuple[int, int]:
        finite = np.argwhere(np.isfinite(values))
        if finite.size:
            t_idx, f_idx = finite[len(finite) // 2]
            return int(t_idx), int(f_idx)
        return 0, 0

    def _mount_sky_widget(self, ds: xr.Dataset) -> None:
        widget = SkyWidget()
        widget.colormap = "inferno"
        widget.background_survey = ""
        widget.invert_horizontal_pan = True
        max_size = max(256, int(ds.sizes["l"]) // 2)
        bind_sky_widget_dataset(widget, ds, max_size=max_size)
        self._sky_widget = widget
        self._sky_container.children = [widget]

    def _update_sky(self, time_idx: int, freq_idx: int) -> None:
        widget = self._sky_widget
        jupiter = self._jupiter
        if widget is None or jupiter is None:
            return
        widget.update_slice(
            time_idx=int(time_idx),
            freq_idx=int(freq_idx),
            center=jupiter,
            fov=JUPITER_SKY_FOV_DEG * u.deg,
            percentile_low=2,
            percentile_high=98,
        )
        send_state = getattr(widget, "send_state", None)
        if callable(send_state):
            send_state()

    def _on_heatmap_tap(self, time_idx: int, freq_idx: int) -> None:
        self._time_idx = time_idx
        self._freq_idx = freq_idx
        if self._jupiter is None or self._lst_hours is None or self._freq_mhz is None:
            return
        ra = self._jupiter.ra.to_string(unit=u.hour, precision=1)
        dec = self._jupiter.dec.to_string(unit=u.deg, precision=1)
        lst = _format_lst_hour_label(float(self._lst_hours[time_idx]))
        freq = float(self._freq_mhz[freq_idx])
        status = (
            f"**Selected slice:** LST {lst}, {freq:.1f} MHz · "
            f"Jupiter (t₀) RA={ra}, Dec={dec}"
        )
        if self.flux_method == "patch_fit" and self._patch_fit_result is not None:
            diag_md = _format_patch_fit_diagnostics(
                self._patch_fit_result, time_idx, freq_idx
            )
            status = f"{status}\n\n{diag_md}"
            self._log(diag_md.replace("**patch_fit** ", "patch_fit "))
        self._set_status(status)
        self._log(
            f"Sky view updated — time {time_idx}, freq {freq_idx} "
            f"({freq:.1f} MHz), FOV {JUPITER_SKY_FOV_DEG:.0f}°."
        )
        self._update_sky(time_idx, freq_idx)

    def _build_dynspec_figure(self, values: np.ndarray) -> figure:
        assert self._lst_hours is not None and self._freq_mhz is not None
        n_times, n_freqs = values.shape
        mapper = jupiter_color_mapper(values.astype(np.float64, copy=False))

        if self.flux_method == "patch_fit":
            plot_title = (
                "Gaussian-fit peak flux toward Jupiter — hover for χ², peak RA/Dec, offsets"
            )
            flux_tooltip = ("Peak flux (Jy/beam)", "@peak_flux_display")
        elif self.flux_method == "patch_max":
            plot_title = "Patch-max flux toward Jupiter (RA/Dec at observation start)"
            flux_tooltip = ("Patch max (Jy/beam)", "@flux_display")
        else:
            plot_title = "Dynamic spectrum toward Jupiter (RA/Dec at observation start)"
            flux_tooltip = ("Flux (Jy/beam)", "@flux_display")
        plot = figure(
            width=1000,
            height=400,
            title=plot_title,
            x_range=(0, n_times),
            y_range=(0, n_freqs),
            tools="pan,wheel_zoom,reset,tap",
            active_drag="pan",
            active_tap="tap",
        )
        plot.image(
            image=[values.T.astype(np.float64, copy=False)],
            x=0,
            y=0,
            dw=n_times,
            dh=n_freqs,
            color_mapper=mapper,
        )

        time_idx, freq_idx = np.meshgrid(
            np.arange(n_times, dtype=int),
            np.arange(n_freqs, dtype=int),
            indexing="ij",
        )
        flat_time = time_idx.ravel()
        flat_freq = freq_idx.ravel()
        hover_data: dict[str, object] = {
            "x": flat_time + 0.5,
            "y": flat_freq + 0.5,
            "time_idx": flat_time,
            "freq_idx": flat_freq,
            "lst_hour": [
                _format_lst_hour_label(float(h)) for h in self._lst_hours[flat_time]
            ],
            "freq_mhz": self._freq_mhz[flat_freq],
            "flux_display": format_flux_hover(values),
        }
        tooltips: list[tuple[str, str]] = [
            ("LST hour", "@lst_hour"),
            ("Freq (MHz)", "@freq_mhz{0.1}"),
            ("Time idx", "@time_idx"),
            ("Freq idx", "@freq_idx"),
            flux_tooltip,
        ]
        if self.flux_method == "patch_fit" and self._patch_fit_result is not None:
            hover_data.update(_patch_fit_hover_columns(self._patch_fit_result))
            tooltips.extend(
                [
                    ("Patch max (Jy)", "@patch_max_display"),
                    ("χ²_red", "@chi2_display"),
                    ("Fit accepted", "@fit_accepted_display"),
                    ("Peak RA", "@peak_ra_display"),
                    ("Peak Dec", "@peak_dec_display"),
                    ("Offset (l,m px)", "@offset_display"),
                ]
            )
        hover_src = ColumnDataSource(data=hover_data)
        hover_renderer = plot.rect(
            x="x",
            y="y",
            width=1,
            height=1,
            source=hover_src,
            fill_alpha=0,
            line_alpha=0,
        )
        plot.add_tools(
            HoverTool(
                renderers=[hover_renderer],
                tooltips=tooltips,
            )
        )

        def _axis_ticks(
            n: int, axis_values: np.ndarray, fmt: Callable[[float], str]
        ) -> tuple[list[float], dict[float, str]]:
            step = 1 if n <= 24 else int(np.ceil(n / 24))
            indices = range(0, n, step)
            ticks = [i + 0.5 for i in indices]
            labels = {tick: fmt(float(axis_values[i])) for tick, i in zip(ticks, indices, strict=True)}
            return ticks, labels

        x_ticks, x_labels = _axis_ticks(
            n_times, self._lst_hours, _format_lst_hour_label
        )
        y_ticks, y_labels = _axis_ticks(
            n_freqs, self._freq_mhz, lambda v: f"{float(v):.1f}"
        )
        plot.xaxis.ticker = FixedTicker(ticks=x_ticks)
        plot.yaxis.ticker = FixedTicker(ticks=y_ticks)
        plot.xaxis.major_label_overrides = x_labels
        plot.yaxis.major_label_overrides = y_labels
        plot.xaxis.axis_label = "LST hour"
        plot.yaxis.axis_label = "Frequency (MHz)"
        plot.xaxis.major_label_orientation = math.pi / 4

        def _on_tap(event: Tap) -> None:
            if event.x is None or event.y is None:
                return
            t_idx = _heatmap_index_from_coord(event.x, n_times)
            f_idx = _heatmap_index_from_coord(event.y, n_freqs)
            self._dispatch(lambda: self._on_heatmap_tap(t_idx, f_idx))

        plot.on_event(Tap, _on_tap)
        return plot
