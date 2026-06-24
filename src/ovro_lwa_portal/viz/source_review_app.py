"""Panel source review app (SkyWidget + time–frequency heatmap).

Extracted from ``notebooks/source_review.ipynb`` for testability. The notebook
keeps path/config constants and calls :func:`configure_source_review_notebook`
before constructing :class:`SourceReview`.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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

from ovro_lwa_portal import resolve_coordinate_string
from ovro_lwa_portal.io import DataSourceError
from ovro_lwa_portal.name_resolution import format_icrs_degree_pair
from ovro_lwa_portal.viz.hips import compute_hips_percentile_cuts, hips_background_survey_url
from ovro_lwa_portal.viz.pipeline_qa_app import (
    ACTIVITY_LOG_HEIGHT_PX,
    ScrollLog,
    _capture_ipython_io_loop,
    _format_activity_log_html,
    _patch_astrowidget_get_wcs,
    _schedule_ipython_main,
    bind_sky_widget_dataset,
    defer_after_notebook_hold,
    publish_bokeh_pane_to_notebook,
    push_bokeh_pane_mutation_to_notebook,
    schedule_when_panel_loaded,
    set_notebook_pane_object,
)
from ovro_lwa_portal.viz.panel_ui_session import (
    CallbackPanelUISession,
    JupyterPanelUISession,
    PanelUISession,
)
from ovro_lwa_portal.viz.source_review import (
    DatasetLoad,
    finalize_dataset_load,
    plan_center_action,
    run_dataset_load,
)
from ovro_lwa_portal.accessor import PatchFitCellResult
from ovro_lwa_portal.viz.source_review_data import (
    HEATMAP_METHOD_OPTIONS,
    HeatmapLoad,
    _PROGRESS_STAGE_LABELS,
    _color_mapper,
    _format_patch_fit_diagnostics,
    _heatmap_index_from_coord,
    _row_hover,
    build_source_from_coordinate,
    calendar_mmdd_labels_for_time_coord,
    compute_overlay_patch_fit,
    compute_source_heatmap,
    diagnose_heatmap_coverage,
    filter_known_source_names,
    first_valid_sky_slice,
    format_heatmap_time_axis_label,
    load_known_sources,
    lst_hours_for_dataset,
    resolve_known_sources_path,
)

__all__ = [
    "SourceReview",
    "SourceReviewConfig",
    "configure_source_review_notebook",
]


@dataclass
class _HeatmapBokehHandles:
    """Live Bokeh models for the heatmap pane (mutate in place after first publish)."""

    plot: figure
    image_renderer: Any
    hover_source: ColumnDataSource
    hover_tool: HoverTool


@dataclass(frozen=True)
class SourceReviewConfig:
    """Runtime paths and tuning knobs normally set in the notebook config cell."""

    zarr_lm_chunk: int = 512
    skip_first_valid_sky_scan: bool = True
    use_ned_fallback: bool = True
    ned_timeout_s: float = 10.0
    hips_root: Path = Path("/lustre/pipeline/calibration/hips")
    hips_background: Path = Path(
        "/lustre/pipeline/calibration/hips/Blue_I_deep_Taper_Robust-0.75_Jan25.hips"
    )
    hips_http_prefix: str = "/calibration/hips"
    hips_background_percentile_low: float = 2.0
    hips_background_percentile_high: float = 98.0
    log_overlay_timing: bool = False


LOADING_SPINNER_HTML = (
    '<div style="display:inline-block;width:22px;height:22px;vertical-align:middle;'
    'border:3px solid #ddd;border-top-color:#0d6efd;border-radius:50%;'
    'animation:ovro-lwa-spin 0.8s linear infinite"></div>'
    "<style>@keyframes ovro-lwa-spin{to{transform:rotate(360deg)}}</style>"
)


def _placeholder_heatmap_figure() -> figure:
    """Minimal Bokeh plot shown before the Zarr store opens.

    Ensures the heatmap ``pn.pane.Bokeh`` is a real figure in the layout comm
    from the first render (not an empty Spacer placeholder that is hard to
    replace from io-loop callbacks).
    """
    plot = figure(
        width=1000,
        height=400,
        x_range=(0, 1),
        y_range=(0, 1),
        tools="",
        active_drag=None,
        active_tap=None,
        title="Heatmap loads after the Zarr store opens…",
    )
    plot.xaxis.visible = False
    plot.yaxis.visible = False
    plot.outline_line_alpha = 0
    return plot


def configure_source_review_notebook() -> None:
    """One-time notebook setup: astrowidget patch, Panel extension, io_loop capture."""
    _patch_astrowidget_get_wcs()
    pn.extension("bokeh", sizing_mode="stretch_width")
    _capture_ipython_io_loop()


class SourceReview(param.Parameterized):
    """Sky-position heatmap + SkyWidget review (Jupiter-style Panel UI)."""

    coordinate_string = param.String(
        default="",
        doc="ICRS RA/Dec in degrees or a source name (Center or Generate heatmap).",
    )
    heatmap_method = param.Selector(
        default="mad",
        objects=HEATMAP_METHOD_OPTIONS,
        doc="Quantity plotted in the time–frequency heatmap.",
    )
    loading = param.Boolean(default=False)
    status = param.String(default="Opening Zarr store…")
    log_text = param.String(default="")

    def __init__(
        self,
        zarr_path: Path,
        *,
        coordinate_string: str = "",
        known_sources_path: Path | None = None,
        patch_scale: float,
        sky_fov_deg: float,
        patch_fit_max_reduced_chi_squared: float,
        config: SourceReviewConfig | None = None,
        validate_zarr: bool = True,
        dispatch_override: Callable[[Callable[[], None]], None] | None = None,
        ui_session: PanelUISession | None = None,
        **params,
    ) -> None:
        self._zarr_path = Path(zarr_path)
        self._known_sources_path = Path(known_sources_path) if known_sources_path else None
        if self._known_sources_path is not None:
            self._known_source_names = load_known_sources(self._known_sources_path)
        else:
            self._known_source_names = []
        self._patch_scale = float(patch_scale)
        self._sky_fov_deg = float(sky_fov_deg)
        self._patch_fit_max_chi2 = float(patch_fit_max_reduced_chi_squared)
        self._config = config or SourceReviewConfig()
        self._dispatch_override = dispatch_override
        self._ui_session_override = ui_session
        self._ui_session: PanelUISession | None = None
        self._scroll_log = ScrollLog()
        self._dataset: xr.Dataset | None = None
        self._cache: dict[tuple[str, str], HeatmapLoad] = {}
        self._heatmap_values: np.ndarray | None = None
        self._overlay_patch_fit_result: PatchFitCellResult | None = None
        self._patch_stat_result: object | None = None
        self._current_source: dict | None = None
        self._coord: SkyCoord | None = None
        # Sky target the most recently *generated* dynamic-spectrum heatmap
        # belongs to (distinct from self._coord, the current overlay center).
        self._heatmap_coord: SkyCoord | None = None
        # Whether the radio overlay is shown; toggled by the Overlay button.
        self._overlay_enabled = True
        self._suppress_overlay_toggle = False
        self._lst_hours: np.ndarray | None = None
        self._time_day_labels: np.ndarray | None = None
        self._freq_mhz: np.ndarray | None = None
        self._sky_widget: SkyWidget | None = None
        self._hips_background_url = hips_background_survey_url(
            self._config.hips_background,
            hips_root=self._config.hips_root,
            http_prefix=self._config.hips_http_prefix,
        )
        self._time_idx = 0
        self._freq_idx = 0
        self._default_time_idx = 0
        self._default_freq_idx = 0
        self._heatmap_job_id = 0
        self._heatmap_grid_ready = False
        self._heatmap_bokeh_handles: _HeatmapBokehHandles | None = None
        self._last_heatmap_figure: object | None = None
        self._overlay_load_generation = 0
        self._overlay_fit_job_id = 0
        self._fit_overlay_sync_token = 0
        self._last_sent_image_revision: int | None = None

        super().__init__(coordinate_string=coordinate_string.strip(), **params)

        self._heatmap_pane = pn.pane.Bokeh(
            _placeholder_heatmap_figure(),
            height=420,
            sizing_mode="stretch_width",
        )
        self._sky_container = widgets.VBox(
            children=[widgets.HTML("<i>Sky view loads after the Zarr store opens.</i>")],
            layout=widgets.Layout(width="100%", min_height="620px"),
        )
        self._sky_pane = pn.pane.IPyWidget(self._sky_container, height=620, sizing_mode="stretch_width")
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
        self._method_selector = pn.widgets.Select.from_param(
            self.param.heatmap_method,
            name="Heatmap method",
            width=220,
        )
        # ipywidgets spinner: same comm path as the activity log (Panel LoadingSpinner
        # often never spins or clears at the wrong time in live Jupyter).
        self._loading_widget = widgets.HTML(value="", layout=widgets.Layout(width="28px"))
        self._loading_pane = pn.pane.IPyWidget(
            self._loading_widget,
            width=32,
            height=32,
            sizing_mode="fixed",
        )
        initial_coord = coordinate_string.strip()
        self._coord_input = pn.widgets.AutocompleteInput(
            name="Coordinate",
            value=initial_coord,
            value_input=initial_coord,
            placeholder="RA°, Dec° or source name — Center or Generate heatmap",
            options=filter_known_source_names(initial_coord, self._known_source_names),
            search_strategy="includes",
            case_sensitive=False,
            restrict=False,
            min_characters=1,
            sizing_mode="stretch_width",
        )
        self._coord_input.param.watch(self._on_coord_value_input, "value_input")
        self._coord_input.param.watch(self._on_coord_value, "value")
        self._suppress_coord_value_handler = False
        self._coord_slew = pn.widgets.Button(
            name="Center",
            button_type="primary",
            width=80,
        )
        self._coord_slew.on_click(self._on_slew)
        self._coord_generate = pn.widgets.Button(
            name="Generate heatmap",
            button_type="primary",
            width=150,
        )
        self._coord_generate.on_click(self._on_generate_heatmap)
        self._overlay_toggle = pn.widgets.Toggle(
            name="Overlay: on",
            value=True,
            button_type="success",
            width=110,
        )
        self._overlay_toggle.param.watch(self._on_overlay_toggle, "value")
        self._fit_overlay_button = pn.widgets.Button(
            name="Fit overlay",
            width=110,
            disabled=True,
        )
        self._fit_overlay_button.on_click(self._on_fit_overlay)
        self._layout = pn.Column(
            pn.Row(
                self._coord_input,
                self._coord_slew,
                self._coord_generate,
                self._overlay_toggle,
                self._fit_overlay_button,
                self._method_selector,
                self._loading_pane,
                sizing_mode="stretch_width",
                margin=(0, 0, 8, 0),
            ),
            self._status_pane,
            self._sky_pane,
            self._heatmap_pane,
            pn.Column(
                pn.pane.Markdown("**Activity log**"),
                self._log_pane,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
            max_width=1048,
        )
        self.param.watch(self._on_heatmap_method_change, "heatmap_method")
        if self._known_sources_path is not None:
            resolved_sources = resolve_known_sources_path(self._known_sources_path)
            if resolved_sources is None:
                self._log(
                    f"WARNING: Known sources file not found: {self._known_sources_path} "
                    f"(cwd={Path.cwd()})"
                )
            elif self._known_source_names:
                self._log(
                    f"Known sources: {len(self._known_source_names)} names from {resolved_sources}"
                )
        self._log(f"Zarr: {self._zarr_path}")
        props = self._config.hips_background / "properties"
        if props.is_file():
            self._log(f"HiPS background: {self._hips_background_url}")
        else:
            self._log(
                f"WARNING: HiPS not found on disk ({self._config.hips_background}); "
                f"background URL is {self._hips_background_url!r}"
            )
        self._set_status(
            "**Enter a coordinate** (`RA°, Dec°` or source name), then click "
            "**Center** or **Generate heatmap**."
        )
        if validate_zarr:
            try:
                resolved = ovro.validate_local_zarr_store(self._zarr_path)
                if resolved != self._zarr_path.resolve():
                    self._log(f"Resolved Zarr path: {resolved}")
                    self._zarr_path = resolved
            except (FileNotFoundError, DataSourceError) as exc:
                self.loading = False
                self._refresh_loading_widget(False)
                self._log(f"ERROR: {exc}")
                self._set_status(
                    "**Invalid Zarr path** — correct `ZARR_PATH` in the config cell "
                    "and re-run the launch cell."
                )
                return
        self._log_dask_dashboard()
        self._refresh_log_widget()
        self._open_scheduled = False

    @property
    def panel(self) -> pn.Column:
        self._ensure_dataset_open()
        return self._layout

    def _notebook_ui_views(self) -> tuple[pn.viewable.Viewable, ...]:
        # Include panes we mutate often so _push_panel_layout matches jupiter_flux_review.
        return (
            self._layout,
            self._status_pane,
            self._heatmap_pane,
            self._coord_input,
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

    def _publish_heatmap_figure(
        self,
        figure: object,
        *,
        after_publish: Callable[[], None] | None = None,
    ) -> None:
        """Push a Bokeh heatmap figure on the next io-loop turn (Jupiter push path)."""
        self._last_heatmap_figure = figure
        self._ui.publish_bokeh_figure(
            self._heatmap_pane,
            figure,
            after_publish=after_publish,
        )

    def _republish_heatmap_figure_for_notebook(self) -> None:
        """Confirm push on a fresh io_loop turn (layout-root-only notebook comms)."""
        published = self._heatmap_pane.object
        if published is None and self._heatmap_values is not None:
            self._heatmap_bokeh_handles = None
            published = self._refresh_heatmap_figure(self._heatmap_values)
        if published is None:
            return
        # Force a nested-pane swap when confirm re-assigns the same figure object.
        self._heatmap_pane.object = None
        publish_bokeh_pane_to_notebook(
            self._heatmap_pane,
            published,
            *self._notebook_ui_views(),
            force_push=True,
        )

    def _publish_heatmap_mutation_to_notebook(
        self,
        *,
        after_publish: Callable[[], None] | None = None,
    ) -> None:
        """Push in-place heatmap edits after the active hold cycle (or on io_loop)."""

        def _push() -> None:
            push_bokeh_pane_mutation_to_notebook(
                self._heatmap_pane,
                *self._notebook_ui_views(),
                force_push=True,
            )
            if after_publish is not None:
                after_publish()

        defer_after_notebook_hold(_push)

    def _push_heatmap_mutation_to_notebook(self) -> None:
        """Push in-place Bokeh edits (title, image data) on a fresh io_loop turn."""
        self._publish_heatmap_mutation_to_notebook()

    def _ensure_dataset_open(self) -> None:
        if self._open_scheduled or self._dataset is not None:
            return
        self._open_scheduled = True
        _capture_ipython_io_loop()
        schedule_when_panel_loaded(self._open_dataset)

    def _dispatch(self, callback: Callable[[], None]) -> None:
        self._ui.dispatch(callback)

    def _schedule_ui_action(self, callback: Callable[[], None]) -> None:
        """Run a Panel button/param handler on the kernel io_loop (``JupyterPanelUISession``)."""
        self._dispatch(callback)

    def _active_coordinate_text(self) -> str:
        typing = (self._coord_input.value_input or "").strip()
        committed = (self._coord_input.value or "").strip()
        if not typing and not committed:
            return (self.coordinate_string or "").strip()
        if committed and typing and committed.casefold() != typing.casefold():
            # AutocompleteInput can leave value_input as a typed prefix after a
            # dropdown pick; value holds the committed completion.
            if committed.casefold().startswith(typing.casefold()) and (
                committed in self._known_source_names or len(committed) > len(typing)
            ):
                return committed
            return typing
        return typing or committed or (self.coordinate_string or "").strip()

    def _coord_match_options(self, text: str) -> list[str]:
        return filter_known_source_names(text, self._known_source_names)

    def _on_coord_value_input(self, event) -> None:
        text = event.new or ""
        self.coordinate_string = text
        options = self._coord_match_options(text)
        if list(self._coord_input.options) != options:
            self._coord_input.options = options

    def _on_coord_value(self, event) -> None:
        if self._suppress_coord_value_handler:
            return
        text = str(event.new or "").strip()
        if not text:
            return
        self.coordinate_string = text
        with param.parameterized.discard_events(self._coord_input):
            self._coord_input.value_input = text
        self._log_coordinate_resolution(text)

    def _log_coordinate_resolution(self, coord_text: str) -> None:
        """Resolve a coordinate string and log RA/Dec (tab completion / dropdown pick)."""
        coord_text = coord_text.strip()
        if not coord_text:
            return
        resolution, messages = resolve_coordinate_string(
            coord_text,
            use_ned_fallback=self._config.use_ned_fallback,
            ned_timeout=self._config.ned_timeout_s,
            known_source_names=frozenset(
                name.casefold() for name in self._known_source_names
            ),
        )
        for message in messages:
            self._log(message)
        if resolution is None:
            self._log(
                f"WARNING: Could not resolve coordinate {coord_text!r} "
                "(expected RA/Dec in degrees or a resolvable name)."
            )
            return
        coord = resolution.coord
        resolver_note = f" [{resolution.resolver}]"
        self._log(
            f"Coordinate {coord_text!r} → RA={coord.ra.deg:.4f}°, "
            f"Dec={coord.dec.deg:.4f}°{resolver_note}."
        )

    def _set_coordinate_field_from_text(
        self, text: str, *, log_prefix: str | None = None
    ) -> None:
        text = text.strip()
        if not text:
            return
        self.coordinate_string = text
        self._suppress_coord_value_handler = True
        try:
            self._ui.sync_coordinate_field(
                self._coord_input,
                value=text,
                value_input=text,
            )
        finally:
            self._suppress_coord_value_handler = False
        if log_prefix:
            self._log(f"{log_prefix} {text!r}")

    def _refresh_loading_widget(self, active: bool) -> None:
        self._loading_widget.value = LOADING_SPINNER_HTML if active else ""
        send_state = getattr(self._loading_widget, "send_state", None)
        if callable(send_state):
            send_state()

    def _sync_spinner(self, value: bool) -> None:
        self._refresh_loading_widget(value)

    def _overlay_fit_source(self) -> dict | None:
        """RA/Dec dict for overlay patch fit (field position, not heatmap target)."""
        if self._coord is not None:
            label = self.coordinate_string.strip() or "overlay"
            return build_source_from_coordinate(label, self._coord)
        return self._current_source

    def _sync_fit_overlay_button(self) -> None:
        """Update Fit overlay disabled state without blocking UI actions on Zarr I/O."""
        if self._dataset is None or self._coord is None:
            self._fit_overlay_button.disabled = True
            return
        self._fit_overlay_sync_token += 1
        token = self._fit_overlay_sync_token
        dataset = self._dataset
        time_idx = int(self._time_idx)
        freq_idx = int(self._freq_idx)

        def _work() -> None:
            try:
                populated = dataset.radport._var_cell_has_finite_data(
                    time_idx=time_idx,
                    frequency_idx=freq_idx,
                )
            except (ValueError, KeyError, AttributeError):
                populated = False

            def _apply() -> None:
                if token != self._fit_overlay_sync_token:
                    return
                self._fit_overlay_button.disabled = not populated

            _schedule_ipython_main(_apply)

        threading.Thread(target=_work, daemon=True).start()

    def _invalidate_overlay_patch_fit(self) -> None:
        self._overlay_fit_job_id += 1
        self._overlay_patch_fit_result = None
        self._sync_fit_overlay_button()

    def _on_fit_overlay(self, _event: object | None = None) -> None:
        self._schedule_ui_action(self._on_fit_overlay_impl)

    def _on_fit_overlay_impl(self) -> None:
        if self._fit_overlay_button.disabled:
            return
        self._load_overlay_fit()

    def _load_overlay_fit(self) -> None:
        if self._dataset is None or self._coord is None:
            return
        src = self._overlay_fit_source()
        if src is None:
            return
        self._overlay_fit_job_id += 1
        job_id = self._overlay_fit_job_id
        time_idx = int(self._time_idx)
        freq_idx = int(self._freq_idx)
        dataset = self._dataset

        def _begin() -> None:
            self.loading = True
            self._sync_spinner(True)
            self._log(
                f"Fitting overlay patch for {src['name']} "
                f"(t={time_idx}, f={freq_idx})…"
            )

        _schedule_ipython_main(_begin)

        def _work() -> None:
            try:
                result = compute_overlay_patch_fit(
                    dataset,
                    src,
                    time_idx=time_idx,
                    freq_idx=freq_idx,
                    scale=self._patch_scale,
                    patch_fit_max_reduced_chi_squared=self._patch_fit_max_chi2,
                )
            except Exception as exc:
                captured_error = exc
                _schedule_ipython_main(
                    lambda err=captured_error: self._finish_overlay_fit(
                        None, err, job_id
                    )
                )
                return
            captured_result = result
            _schedule_ipython_main(
                lambda fit=captured_result: self._finish_overlay_fit(
                    fit, None, job_id
                )
            )

        threading.Thread(target=_work, daemon=True).start()

    def _finish_overlay_fit(
        self,
        result: PatchFitCellResult | None,
        error: BaseException | None,
        job_id: int,
    ) -> None:
        if job_id != self._overlay_fit_job_id:
            return
        self.loading = False
        self._sync_spinner(False)
        if error is not None:
            self._log(f"ERROR (fit overlay): {error}")
            self._dispatch(
                lambda: self._set_status(f"**Fit overlay failed:** {error}")
            )
            return
        assert result is not None
        self._overlay_patch_fit_result = result
        diag = result.cell_diagnostics(result.time_idx, result.frequency_idx)
        peak_s = (
            f"{diag['peak']:.3g}"
            if np.isfinite(float(diag["peak"]))
            else "n/a (masked)"
        )
        self._log(
            f"Fit overlay finished t={result.time_idx} f={result.frequency_idx}: "
            f"χ²_red={diag['reduced_chi_squared']:.3g}, peak={peak_s} Jy"
        )

        def _update_status() -> None:
            if self._lst_hours is None or self._freq_mhz is None:
                status = _format_patch_fit_diagnostics(
                    result, result.time_idx, result.frequency_idx
                )
            else:
                lst = format_heatmap_time_axis_label(
                    np.asarray(self._dataset.coords["time"].values),
                    result.time_idx,
                    self._lst_hours,
                    day_labels=self._time_day_labels,
                )
                freq = float(self._freq_mhz[result.frequency_idx])
                name = (
                    self._current_source["name"]
                    if self._current_source is not None
                    else "field"
                )
                status = (
                    f"**{name}** · LST {lst}, {freq:.1f} MHz "
                    f"(t={result.time_idx}, f={result.frequency_idx})\n\n"
                    f"{_format_patch_fit_diagnostics(result, result.time_idx, result.frequency_idx)}"
                )
            self._set_status(status)

        self._dispatch(_update_status)
        self._sync_fit_overlay_button()

    def _schedule_overlay_slice_load(
        self,
        time_idx: int,
        freq_idx: int,
        *,
        center_on_target: bool = False,
        center: SkyCoord | None = None,
        preserve_view: bool = False,
        manage_spinner: bool = True,
    ) -> None:
        """Load an overlay slice on the kernel io_loop without a Panel dispatch batch.

        Wrapping ``update_slice`` in ``defer_dispatch`` runs two full layout pushes
        (before and after the Zarr read) on large notebooks; the spinner then tracks
        Panel comm latency instead of the actual slice load.
        """
        if self._sky_widget is None:
            if manage_spinner:
                self._clear_loading_indicator()
            return

        if manage_spinner:
            self.loading = True
            self._sync_spinner(True)

        self._overlay_load_generation += 1
        generation = self._overlay_load_generation

        def _run() -> None:
            if generation != self._overlay_load_generation:
                return
            try:
                self._update_sky(
                    int(time_idx),
                    int(freq_idx),
                    center_on_target=center_on_target,
                    center=center,
                    preserve_view=preserve_view,
                    log_loading=True,
                    overlay_generation=generation,
                )
            finally:
                if manage_spinner and generation == self._overlay_load_generation:
                    self._clear_loading_indicator()

        self._ui.schedule(_run)

    def _sync_coordinate_field(self) -> str:
        resolved = self._active_coordinate_text()
        if resolved:
            self._set_coordinate_field_from_text(resolved)
        return resolved

    def _on_sky_widget_click(self, _change) -> None:
        widget = self._sky_widget
        if widget is None:
            return
        ra_deg, dec_deg = (float(x) for x in widget.clicked_coord)

        def _apply() -> None:
            if not (math.isfinite(ra_deg) and math.isfinite(dec_deg)):
                self._log(f"WARNING: Sky click at non-finite RA/Dec ({ra_deg}, {dec_deg}).")
                return
            try:
                text = format_icrs_degree_pair(ra_deg, dec_deg)
            except ValueError as exc:
                self._log(f"WARNING: {exc}")
                return
            self._log_click_projection_diagnostics(ra_deg, dec_deg)
            self._set_coordinate_field_from_text(text, log_prefix="Sky click →")
            self._set_status(
                f"Sky click — RA={ra_deg:.4f}°, Dec={dec_deg:.4f}° · "
                "**Center** reprojects the overlay here; **Generate heatmap** tracks this position."
            )

        self._schedule_ui_action(_apply)

    def _resolve_active_coordinate(self) -> tuple[SkyCoord, str] | None:
        coord_text = self._active_coordinate_text().strip()
        if not coord_text:
            self._log("WARNING: Coordinate is empty — enter RA/Dec or a source name.")
            return None

        resolution, messages = resolve_coordinate_string(
            coord_text,
            use_ned_fallback=self._config.use_ned_fallback,
            ned_timeout=self._config.ned_timeout_s,
            known_source_names=frozenset(
                name.casefold() for name in self._known_source_names
            ),
        )
        for message in messages:
            self._log(message)
        if resolution is None:
            self._log(
                f"WARNING: Could not resolve coordinate {coord_text!r} "
                "(expected RA/Dec in degrees or a resolvable name)."
            )
            return None

        coord = resolution.coord
        label = resolution.canonical_name or coord_text
        resolver_note = f" [{resolution.resolver}]"
        self._log(
            f"Coordinate {coord_text!r} → RA={coord.ra.deg:.4f}°, "
            f"Dec={coord.dec.deg:.4f}°{resolver_note}."
        )
        return coord, label

    def _apply_coordinate_from_field(self) -> bool:
        resolved = self._resolve_active_coordinate()
        if resolved is None:
            return False
        coord, label = resolved
        self._current_source = build_source_from_coordinate(label, coord)
        # Current overlay center — clicking any heatmap cell loads a slice here.
        self._coord = coord
        self._cache.clear()
        return True

    def _target_matches_coord(self, coord: SkyCoord) -> bool:
        """True when the active heatmap/overlay belongs to ``coord``."""
        if self._coord is None or self._heatmap_values is None:
            return False
        widget = self._sky_widget
        if widget is None or widget.image_shape == (0, 0):
            return False
        return bool(self._coord.separation(coord) < 1 * u.arcsec)

    def _set_sky_crosshair(self, widget: object, coord: SkyCoord | None) -> None:
        """Pin the sky crosshair to a catalog position (requires astrowidget >= crosshair traits)."""
        setter = getattr(widget, "set_crosshair", None)
        if callable(setter):
            setter(coord)
            return
        if not hasattr(widget, "crosshair_ra"):
            return
        if coord is None:
            widget.crosshair_ra = -999.0
            widget.crosshair_dec = -999.0
        else:
            icrs = coord.icrs
            widget.crosshair_ra = float(icrs.ra.deg)
            widget.crosshair_dec = float(icrs.dec.deg)

    def _on_slew(self, _event: object | None = None) -> None:
        self._schedule_ui_action(self._on_slew_impl)

    def _on_slew_impl(self) -> None:
        self._sync_coordinate_field()
        resolved = self._resolve_active_coordinate()
        if resolved is None:
            return
        field_coord, label = resolved
        widget = self._sky_widget
        if widget is None:
            self._log("Sky widget not ready — wait for the Zarr store to open.")
            return

        plan = plan_center_action(
            field_coord,
            heatmap_coord=self._heatmap_coord,
            has_overlay=widget.image_shape != (0, 0),
        )
        # Center establishes the current overlay center for later heatmap clicks.
        self._coord = field_coord
        self._sync_fit_overlay_button()

        def _maybe_reset_heatmap() -> None:
            if not plan.field_matches_heatmap:
                self._reset_heatmap_to_zeros()
            elif self._heatmap_pane.object is None:
                self._ensure_heatmap_grid()

        defer_after_notebook_hold(_maybe_reset_heatmap)

        widget.goto(plan.goto_center, fov=self._sky_fov_deg * u.deg)
        self._set_sky_crosshair(widget, plan.goto_center)

        pos = (
            f"RA={plan.goto_center.ra.deg:.4f}°, "
            f"Dec={plan.goto_center.dec.deg:.4f}°"
        )

        if plan.overlay_center is not None:
            self._log(f"Centering on {label} ({pos}) — loading overlay…")
            if plan.field_matches_heatmap:
                self._set_status(
                    f"**{label}** centering ({pos}) — radio overlay reprojecting…"
                )
            else:
                self._set_status(
                    f"Centering ({pos}) — radio overlay reprojecting… "
                    "Heatmap will reset to zeros."
                )
            self._schedule_overlay_slice_load(
                self._time_idx,
                self._freq_idx,
                center_on_target=True,
                center=plan.overlay_center,
                manage_spinner=True,
            )
        else:
            self._log(
                f"Centered HiPS on {label} ({pos}). "
                "Click a heatmap cell to load a slice, or Generate heatmap for the dynamic spectrum."
            )
            self._set_status(
                f"**{label}** centered on the HiPS sky ({pos}). "
                "**Click a heatmap cell** to load that time/frequency as an overlay, "
                "or **Generate heatmap** for the dynamic spectrum."
            )

        self._log_overlay_diagnostics(plan.goto_center, context=f"Center[{plan.reason}]")
        self._force_send_sky_widget_state(widget)

    def _maybe_send_sky_widget_state(self, widget: object, *, force: bool = False) -> float:
        """Push widget traits to the browser; skip when ``image_revision`` is unchanged."""
        send_state = getattr(widget, "send_state", None)
        if not callable(send_state):
            return 0.0
        rev = int(getattr(widget, "image_revision", 0))
        if not force and self._last_sent_image_revision == rev:
            return 0.0
        t0 = time.perf_counter()
        send_state()
        comm_ms = (time.perf_counter() - t0) * 1000.0
        self._last_sent_image_revision = rev
        return comm_ms

    def _force_send_sky_widget_state(self, widget: object) -> None:
        """Always push widget state (Center/goto paths also change view traits)."""
        self._maybe_send_sky_widget_state(widget, force=True)

    def _log_overlay_push_timing(self, widget: object, *, comm_ms: float) -> None:
        profile = getattr(widget, "_profile_last_push", None) or {}
        zarr_ms = float(profile.get("zarr_ms", 0.0))
        reproject_ms = float(profile.get("reproject_ms", 0.0))
        nbytes = int(profile.get("bytes", len(getattr(widget, "image_data", b"") or b"")))
        self._log(
            f"Overlay push: Zarr {zarr_ms:.0f} ms, reproject {reproject_ms:.0f} ms, "
            f"comm {comm_ms:.0f} ms, {nbytes / 1024:.0f} KB"
        )

    def _log_overlay_diagnostics(self, intended_center: SkyCoord, *, context: str) -> None:
        """Log realized widget view/CRVAL vs the intended center.

        Compares the coordinate we asked for against what the widget actually
        applied (``view_ra``/``view_dec`` from ``goto``) and the reprojected
        overlay phase center (``crval`` from ``update_slice``). Lets a live
        session localize a wrong overlay to the controller, the view, or the
        WebGL reprojection layer instead of guessing.
        """
        widget = self._sky_widget
        if widget is None:
            return
        try:
            view_ra = float(widget.view_ra)
            view_dec = float(widget.view_dec)
            crval = tuple(float(x) for x in widget.crval)
        except (TypeError, ValueError, AttributeError) as exc:
            self._log(f"[diag] {context}: could not read widget view/CRVAL ({exc}).")
            return
        d_view = SkyCoord(
            ra=view_ra * u.deg, dec=view_dec * u.deg, frame="icrs"
        ).separation(intended_center).to(u.arcsec).value
        d_crval = SkyCoord(
            ra=crval[0] * u.deg, dec=crval[1] * u.deg, frame="icrs"
        ).separation(intended_center).to(u.arcsec).value
        self._log(
            f"[diag] {context}: intended RA={intended_center.ra.deg:.4f}°, "
            f"Dec={intended_center.dec.deg:.4f}° | view=({view_ra:.4f}, {view_dec:.4f}) "
            f"Δ={d_view:.1f}″ | crval=({crval[0]:.4f}, {crval[1]:.4f}) Δ={d_crval:.1f}″"
        )

    def _log_click_projection_diagnostics(self, ra_deg: float, dec_deg: float) -> None:
        """Log the same click pixel read through HiPS (Aladin) vs WebGL (SIN).

        ``clicked_coord_debug`` carries ``[hips_ra, hips_dec, webgl_ra, webgl_dec]``
        for the click. A large HiPS↔WebGL separation means the radio overlay and
        HiPS background are projected inconsistently, so the reported click and
        the visible overlay source disagree. Also reports the overlay ``crval``
        so a misregistered overlay (source drawn at the wrong sky spot) is
        distinguishable from a wrong click readout.
        """
        widget = self._sky_widget
        if widget is None:
            return
        debug = list(getattr(widget, "clicked_coord_debug", []) or [])
        if len(debug) < 4:
            return
        hips_ra, hips_dec, webgl_ra, webgl_dec = (float(x) for x in debug[:4])
        parts: list[str] = []
        if math.isfinite(hips_ra) and math.isfinite(hips_dec):
            parts.append(f"hips=({hips_ra:.4f}, {hips_dec:.4f})")
        if math.isfinite(webgl_ra) and math.isfinite(webgl_dec):
            parts.append(f"webgl=({webgl_ra:.4f}, {webgl_dec:.4f})")
        if (
            math.isfinite(hips_ra)
            and math.isfinite(hips_dec)
            and math.isfinite(webgl_ra)
            and math.isfinite(webgl_dec)
        ):
            d_proj = (
                SkyCoord(ra=hips_ra * u.deg, dec=hips_dec * u.deg, frame="icrs")
                .separation(SkyCoord(ra=webgl_ra * u.deg, dec=webgl_dec * u.deg, frame="icrs"))
                .to(u.arcsec)
                .value
            )
            parts.append(f"hips↔webgl Δ={d_proj:.1f}″")
        try:
            crval = tuple(float(x) for x in widget.crval)
            parts.append(f"crval=({crval[0]:.4f}, {crval[1]:.4f})")
        except (TypeError, ValueError, AttributeError):
            pass
        self._log(f"[diag] Click reported=({ra_deg:.4f}, {dec_deg:.4f}) | " + " | ".join(parts))

    def _on_generate_heatmap(self, _event: object | None = None) -> None:
        self._schedule_ui_action(self._on_generate_heatmap_impl)

    def _on_generate_heatmap_impl(self) -> None:
        self._sync_coordinate_field()
        if not self._apply_coordinate_from_field():
            return
        if self._dataset is None:
            self._log(
                "WARNING: Zarr store not open yet — wait for "
                "'Opened —' in the activity log before generating a heatmap."
            )
            self._set_status(
                "**Zarr still opening** — wait for **Opened —** in the activity log, "
                "then **Generate heatmap**."
            )
            return
        self._load_heatmap()

    def _heatmap_figure_title(self, *, subject: str | None = None) -> str:
        """Bokeh title: target name + heatmap method (matches the method selector)."""
        if subject is None:
            src = self._current_source
            subject = src["name"] if src is not None else "Time × frequency"
        return (
            f"{subject} — {self.heatmap_method} "
            f"({self._heatmap_method_label()}; click a cell for sky view)"
        )

    def _heatmap_method_label(self) -> str:
        labels = {
            "dynamic_spectrum": "tracked centre pixel",
            "patch_max": "patch maximum",
            "mad": "patch MAD",
            "std": "patch std",
            "mean": "patch mean",
            "min": "patch min",
        }
        return labels.get(self.heatmap_method, self.heatmap_method)

    def _set_status(self, text: str) -> None:
        self.status = text

    @param.depends("status", watch=True)
    def _sync_status_pane(self) -> None:
        self._ui.sync_status_pane(self._status_pane, self.status)

    def _refresh_log_widget(self) -> None:
        """Update the ipywidgets activity log (separate comm from Panel layout)."""
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

    def _heatmap_progress_callback(self) -> Callable[[str, int, int, str], None]:
        last_key: dict[str, tuple[int, str]] = {}

        def _callback(stage: str, current: int, total: int, message: str) -> None:
            # Pixel track is fast; log start + finish only (skip per-batch lines).
            if stage == "track" and total > 1 and current not in (0, total):
                return
            if stage in ("extract", "reduce", "fit") and total > 1:
                key = (current, message)
                if current not in (0, total) and last_key.get(stage) == key:
                    return
                last_key[stage] = key
            label = _PROGRESS_STAGE_LABELS.get(stage, stage)
            if total > 0:
                pct = int(round(100.0 * int(current) / int(total)))
                text = f"{label}: {message} ({current}/{total}, {pct}%)"
            else:
                text = f"{label}: {message}"

            # Activity log is ipywidgets — schedule on the io_loop without a Panel
            # dispatch batch. Progress ``dispatch`` calls each end with
            # ``_push_panel_layout`` and can republish the zeros heatmap after
            # ``publish_bokeh_figure`` on long compute jobs.
            _schedule_ipython_main(lambda msg=text: self._log(msg))

        return _callback


    def _open_dataset(self) -> None:
        def _start() -> None:
            self.loading = True
            self._sync_spinner(True)
            self._log(
                f"Opening {self._zarr_path} (chunks='auto', l/m={self._config.zarr_lm_chunk})…"
            )

        self._dispatch(_start)

        def _open(report: Callable[[str], None]) -> DatasetLoad:
            t_open = time.perf_counter()

            t_meta = time.perf_counter()
            ds = ovro.open_dataset(self._zarr_path, chunks="auto").chunk(
                {"l": self._config.zarr_lm_chunk, "m": self._config.zarr_lm_chunk}
            )
            report(f"Zarr opened ({time.perf_counter() - t_meta:.1f} s)")

            if self._config.skip_first_valid_sky_scan:
                t0, f0 = 0, int(ds.sizes["frequency"]) // 2
                report(
                    f"Using default slice time={t0}, freq={f0} "
                    "(self._config.skip_first_valid_sky_scan=True)"
                )
            else:
                report("Scanning centre pixel for first valid time index…")
                t_scan = time.perf_counter()
                t0, f0 = first_valid_sky_slice(ds)
                report(f"Centre-pixel scan complete ({time.perf_counter() - t_scan:.1f} s)")

            report("Computing LST labels for time axis…")
            t_lst = time.perf_counter()
            lst_hours = lst_hours_for_dataset(ds)
            freq_mhz = np.asarray(ds.coords["frequency"].values, dtype=np.float64) / 1e6
            report(
                f"Coordinates ready — {int(ds.sizes['time'])}×{int(ds.sizes['frequency'])} "
                f"heatmap grid; total open ({time.perf_counter() - t_lst:.1f} s)"
            )
            report(f"Open pipeline finished in {time.perf_counter() - t_open:.1f} s")
            return DatasetLoad(
                dataset=ds,
                default_time_idx=t0,
                default_freq_idx=f0,
                lst_hours=lst_hours,
                freq_mhz=freq_mhz,
            )

        def _work() -> None:
            run_dataset_load(
                open_dataset=_open,
                dispatch=self._dispatch,
                log_dispatch=_schedule_ipython_main,
                on_loaded=lambda load: self._finish_open(
                    load.dataset,
                    load.default_time_idx,
                    load.default_freq_idx,
                    load.lst_hours,
                    load.freq_mhz,
                    None,
                ),
                on_error=lambda exc: self._finish_open(None, None, None, None, None, exc),
                log=self._log,
            )

        import threading

        threading.Thread(target=_work, daemon=True).start()


    def _finish_open(
        self,
        ds: xr.Dataset | None,
        default_time_idx: int | None,
        default_freq_idx: int | None,
        lst_hours: np.ndarray | None,
        freq_mhz: np.ndarray | None,
        error: BaseException | None,
    ) -> None:
        if error is not None:
            self.loading = False
            self._sync_spinner(False)
            err_text = str(error).strip()
            self._log(f"ERROR: {err_text}")
            first_line = err_text.splitlines()[0] if err_text else repr(error)
            self._set_status(f"**Load failed:** {first_line}")
            return

        assert ds is not None and lst_hours is not None and freq_mhz is not None
        assert default_time_idx is not None and default_freq_idx is not None

        self._dataset = ds
        self._lst_hours = lst_hours
        self._time_day_labels = calendar_mmdd_labels_for_time_coord(ds.coords["time"].values)
        self._freq_mhz = freq_mhz
        self._default_time_idx = int(default_time_idx)
        self._default_freq_idx = int(default_freq_idx)
        self._time_idx = self._default_time_idx
        self._freq_idx = self._default_freq_idx
        self._heatmap_bokeh_handles = None
        self._heatmap_grid_ready = False
        self._heatmap_values = None

        self._log(
            f"Opened — {int(ds.sizes['time'])} times × {int(ds.sizes['frequency'])} freqs, "
            f"{int(ds.sizes['l'])}×{int(ds.sizes['m'])} px, WCS={ds.radport.has_wcs}."
        )

        def _clear_loading() -> None:
            self.loading = False
            self._sync_spinner(False)

        def _report_step_error(step: str, exc: BaseException) -> None:
            self._log(f"ERROR: post-open step {step!r} failed: {exc}")

        finalize_dataset_load(
            mount_sky=lambda: self._mount_sky_widget(ds),
            build_heatmap_grid=lambda: self._ensure_heatmap_grid(),
            clear_loading=_clear_loading,
            on_step_error=_report_step_error,
        )
        self._set_status(
            f"**Zarr ready** — {int(ds.sizes['time'])} times × {int(ds.sizes['frequency'])} freqs. "
            "Enter a coordinate, then **click a heatmap cell** to load a slice or "
            "**Generate heatmap** for the dynamic spectrum."
        )
        self._sync_fit_overlay_button()

    def _on_heatmap_method_change(self, *_events) -> None:
        self._schedule_ui_action(self._on_heatmap_method_change_impl)

    def _on_heatmap_method_change_impl(self) -> None:
        """Update the heatmap title for the new method; do not recompute until Generate."""
        self._sync_heatmap_method_display()

    def _sync_heatmap_method_display(self) -> None:
        """Refresh heatmap chrome for ``heatmap_method`` without loading Zarr data."""
        if self._dataset is None or self.loading:
            return
        if self._heatmap_values is not None:
            self._refresh_heatmap_figure(self._heatmap_values)
            self._push_heatmap_mutation_to_notebook()
            return
        if self._heatmap_grid_ready:
            self._ensure_heatmap_grid(force=True, announce=False)

    def _load_heatmap(self) -> None:
        if self._dataset is None or self._current_source is None:
            return
        src = self._current_source
        method = str(self.heatmap_method)
        cache_key = (self.coordinate_string.strip(), method)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            self._dispatch(lambda: self._apply_heatmap(src, cached))
            return

        self._heatmap_job_id += 1
        job_id = self._heatmap_job_id
        self._invalidate_overlay_patch_fit()
        self._begin_heatmap_load(src)

        def _work() -> None:
            t0 = time.perf_counter()
            try:
                payload = compute_source_heatmap(
                    self._dataset,
                    src,
                    method=method,
                    scale=self._patch_scale,
                    patch_fit_max_reduced_chi_squared=self._patch_fit_max_chi2,
                    progress_callback=self._heatmap_progress_callback(),
                )
            except Exception as exc:
                captured_error = exc
                _schedule_ipython_main(
                    lambda err=captured_error: self._finish_heatmap(
                        src, None, err, job_id, t0
                    )
                )
                return
            captured_payload = payload
            _schedule_ipython_main(
                lambda result=captured_payload: self._finish_heatmap(
                    src, result, None, job_id, t0
                )
            )

        import threading

        threading.Thread(target=_work, daemon=True).start()

    def _begin_heatmap_load(self, src: dict) -> None:
        """Show spinner/status for a new heatmap job."""

        def _begin() -> None:
            self.loading = True
            self._sync_spinner(True)
            n_times = int(self._dataset.sizes["time"])
            n_freqs = int(self._dataset.sizes["frequency"])
            label = src["name"]
            self._log(
                f"Computing {self._heatmap_method_label()} for {label} "
                f"({n_times} times × {n_freqs} freqs, "
                f"RA={src['ra']:.4f}°, Dec={src['dec']:.4f}°)…"
            )
            status_text = (
                f"Computing **{self._heatmap_method_label()}** for **{label}**…"
            )
            with param.parameterized.discard_events(self):
                self.status = status_text
            self._ui.sync_status_pane(self._status_pane, status_text)

        # Avoid a dispatch batch ``_push_panel_layout`` on the zeros heatmap while
        # compute runs — that full layout push can race the post-finish publish.
        _schedule_ipython_main(_begin)

    def _finish_heatmap(
        self,
        src: dict,
        payload: HeatmapLoad | None,
        error: BaseException | None,
        job_id: int,
        started_at: float,
    ) -> None:
        if job_id != self._heatmap_job_id:
            return
        elapsed_s = time.perf_counter() - started_at
        if error is not None:
            self.loading = False
            self._sync_spinner(False)
            self._log(f"ERROR ({src['name']}): {error}")
            self._set_status(f"**Heatmap failed for {src['name']}:** {error}")
            return

        assert payload is not None
        cache_key = (self.coordinate_string.strip(), str(self.heatmap_method))
        self._cache[cache_key] = payload
        arr = payload.values
        finite = arr[np.isfinite(arr)]
        if finite.size:
            self._log(
                f"{src['name']} ({self._heatmap_method_label()}): "
                f"range [{float(finite.min()):.3g}, {float(finite.max()):.3g}] "
                f"({finite.size}/{arr.size} finite cells)"
            )
        else:
            self._log(f"{src['name']} ({self._heatmap_method_label()}): no finite values")
        self._log(
            f"Finished {src['name']} ({self._heatmap_method_label()}) in {elapsed_s:.1f} s"
        )
        self._apply_heatmap(src, payload)

        def _log_coverage_hint() -> None:
            if self._dataset is None:
                return
            hint = diagnose_heatmap_coverage(
                self._dataset,
                src,
                arr,
                method=str(self.heatmap_method),
                patch_fit_max_reduced_chi_squared=self._patch_fit_max_chi2,
            )
            if hint:
                self._log(f"Hint: {hint}")

        _schedule_ipython_main(_log_coverage_hint)

    def _clear_loading_indicator(self) -> None:
        self.loading = False
        self._sync_spinner(False)

    def _apply_heatmap(self, src: dict, payload: HeatmapLoad) -> None:
        self._current_source = src
        self._heatmap_values = payload.values
        self._patch_stat_result = payload.patch_stat_result
        self._invalidate_overlay_patch_fit()
        self._coord = SkyCoord(ra=src["ra"] * u.deg, dec=src["dec"] * u.deg, frame="icrs")
        self._heatmap_coord = self._coord
        self._time_idx, self._freq_idx = self._default_slice(payload.values)
        # Grid placeholder and deferred open-time grid must not clobber this figure.
        self._heatmap_grid_ready = True

        # Mutate the live zeros-grid figure in place when it is already mounted.
        # Object-swap + confirm republish is unreliable on layout-root-only comms;
        # mutation push matches the method-dropdown path (validated in live Jupyter).
        had_live_figure = (
            self._heatmap_bokeh_handles is not None
            and self._heatmap_pane.object is self._heatmap_bokeh_handles.plot
        )
        if not had_live_figure:
            self._heatmap_bokeh_handles = None
        figure = self._refresh_heatmap_figure(payload.values)

        ra_h = self._coord.ra.to_string(unit=u.hour, precision=1)
        dec_s = self._coord.dec.to_string(unit=u.deg, precision=1)
        overlay_hint = (
            "**Click the heatmap** to inspect a time/frequency slice."
            if self._overlay_enabled
            else "Overlay is **off** — toggle **Overlay** to show slices."
        )
        status_text = (
            f"**{src['name']}** — l={src['l']:.2f}°, b={src['b']:.2f}°, "
            f"RA={ra_h}, Dec={dec_s} · "
            f"Heatmap: **{self._heatmap_method_label()}** (scale={self._patch_scale:g}) · "
            f"{overlay_hint}"
        )

        # Invalidate in-flight heatmap-tap overlay loads before scheduling the
        # post-generate slice so a slow follow-up cannot overwrite a newer tap.
        self._overlay_load_generation += 1
        overlay_generation = self._overlay_load_generation

        def _load_overlay_after_heatmap() -> None:
            if overlay_generation != self._overlay_load_generation:
                return
            if (
                self._overlay_enabled
                and self._sky_widget is not None
                and self._coord is not None
            ):
                self._schedule_overlay_slice_load(
                    self._time_idx,
                    self._freq_idx,
                    center_on_target=True,
                    center=self._coord,
                    manage_spinner=True,
                )

        def _after_heatmap_publish() -> None:
            self._set_status(status_text)
            self._clear_loading_indicator()
            self._ui.schedule(_load_overlay_after_heatmap)

        if had_live_figure:
            self._publish_heatmap_mutation_to_notebook(after_publish=_after_heatmap_publish)
        else:
            self._publish_heatmap_figure(figure, after_publish=_after_heatmap_publish)
        self._sync_fit_overlay_button()

    def _default_slice(self, values: np.ndarray) -> tuple[int, int]:
        finite = np.argwhere(np.isfinite(values))
        if finite.size:
            t_idx, f_idx = finite[len(finite) // 2]
            return int(t_idx), int(f_idx)
        return self._default_time_idx, self._default_freq_idx

    def _mount_sky_widget(self, ds: xr.Dataset) -> None:
        """Match ``jupiter_flux_review.ipynb`` (proven with per-time CRVAL)."""
        widget = SkyWidget()
        widget.colormap = "inferno"
        widget.background_survey = self._hips_background_url
        try:
            cut_lo, cut_hi = compute_hips_percentile_cuts(
                self._config.hips_background,
                percentile_low=self._config.hips_background_percentile_low,
                percentile_high=self._config.hips_background_percentile_high,
            )
            widget.background_cut_min = cut_lo
            widget.background_cut_max = cut_hi
            self._log(
                f"HiPS cuts ({self._config.hips_background_percentile_low:g}/"
                f"{self._config.hips_background_percentile_high:g} pct): "
                f"{cut_lo:.4g} .. {cut_hi:.4g}"
            )
        except (FileNotFoundError, ValueError) as exc:
            self._log(f"WARNING: HiPS display cuts not set — {exc}")
        widget.invert_horizontal_pan = True
        max_size = max(256, int(ds.sizes["l"]) // 2)
        bind_sky_widget_dataset(widget, ds, max_size=max_size)
        if self._hips_background_url:
            widget.overlay_view_lock = True
            widget.observe(self._on_view_gesture_revision, names="view_gesture_revision")
        widget.observe(self._on_sky_widget_click, names="click_tick")
        self._sky_widget = widget
        self._sky_container.children = [widget]

    def _on_view_gesture_revision(self, change) -> None:
        """Activity-log hook when pan/zoom triggers debounced overlay reproject."""
        if change.get("type") != "change" or change.get("name") != "view_gesture_revision":
            return
        rev = change.get("new")
        self._schedule_ui_action(lambda: self._log(f"Overlay view lock: gesture revision {rev}"))

    def _update_sky(
        self,
        time_idx: int,
        freq_idx: int,
        *,
        center_on_target: bool = False,
        center: SkyCoord | None = None,
        preserve_view: bool = False,
        log_loading: bool = False,
        overlay_generation: int | None = None,
    ) -> None:
        widget = self._sky_widget
        if widget is None:
            return
        if overlay_generation is not None and overlay_generation != self._overlay_load_generation:
            return
        reproject_center = center if center is not None else self._coord
        if reproject_center is None:
            return
        if log_loading and self._freq_mhz is not None:
            freq = float(self._freq_mhz[int(freq_idx)])
            self._log(
                f"Loading overlay slice t={int(time_idx)}, f={int(freq_idx)} "
                f"({freq:.1f} MHz)…"
            )
        if preserve_view:
            widget.update_slice(
                time_idx=int(time_idx),
                freq_idx=int(freq_idx),
                view_lock=True,
                percentile_low=2,
                percentile_high=98,
            )
            diag_center = widget.view_center_skycoord()
            diag_context = "update_sky[heatmap]"
        elif center_on_target or not widget.overlay_view_lock:
            widget.update_slice(
                time_idx=int(time_idx),
                freq_idx=int(freq_idx),
                center=reproject_center,
                fov=self._sky_fov_deg * u.deg,
                percentile_low=2,
                percentile_high=98,
            )
            diag_center = reproject_center
            diag_context = "update_sky[center]"
        else:
            widget.update_slice(
                time_idx=int(time_idx),
                freq_idx=int(freq_idx),
                view_lock=True,
                percentile_low=2,
                percentile_high=98,
            )
            diag_center = widget.view_center_skycoord()
            diag_context = "update_sky[view_lock]"
        if overlay_generation is not None and overlay_generation != self._overlay_load_generation:
            return
        self._log_overlay_diagnostics(diag_center, context=diag_context)
        if log_loading:
            self._log(
                f"Overlay slice loaded (t={int(time_idx)}, f={int(freq_idx)})."
            )
        # User-visible overlay loads always push widget state; revision-gated
        # skip is only for silent/internal paths.
        comm_ms = self._maybe_send_sky_widget_state(widget, force=log_loading)
        if self._config.log_overlay_timing and log_loading:
            self._log_overlay_push_timing(widget, comm_ms=comm_ms)

    def _on_heatmap_tap(self, time_idx: int, freq_idx: int) -> None:
        self._time_idx = time_idx
        self._freq_idx = freq_idx
        self._invalidate_overlay_patch_fit()
        if self._lst_hours is None or self._freq_mhz is None:
            return

        lst = format_heatmap_time_axis_label(
            np.asarray(self._dataset.coords["time"].values),
            time_idx,
            self._lst_hours,
            day_labels=self._time_day_labels,
        )
        freq = float(self._freq_mhz[freq_idx])

        # Clicking any cell loads that Zarr slice centered on the current
        # coordinate. Resolve the field lazily so the user can type a coordinate
        # and click a cell without first pressing Center/Generate.
        coord = self._coord
        if coord is None:
            self._sync_coordinate_field()
            if self._active_coordinate_text().strip() and self._apply_coordinate_from_field():
                coord = self._coord
        if coord is None:
            self._set_status(
                f"Selected t={time_idx}, f={freq_idx} ({freq:.1f} MHz) · "
                "**Enter a coordinate** (RA/Dec or name), then click a cell to load "
                "that slice as an overlay."
            )
            self._log(
                f"Heatmap cell t={time_idx}, f={freq_idx} ({freq:.1f} MHz) selected — "
                "no coordinate set yet."
            )
            return

        name = self._current_source["name"] if self._current_source is not None else "field"
        val = (
            float(self._heatmap_values[time_idx, freq_idx])
            if self._heatmap_values is not None
            else float("nan")
        )
        val_s = f"{val:.3g}" if np.isfinite(val) else "n/a"

        ra_h = coord.ra.to_string(unit=u.hour, precision=1)
        dec_s = coord.dec.to_string(unit=u.deg, precision=1)
        track_note = ""
        if self._dataset is not None:
            try:
                li, mi = self._dataset.radport.coords_to_pixel(
                    float(coord.ra.deg), float(coord.dec.deg), time_idx=time_idx
                )
                tr_ra, tr_dec = self._dataset.radport.pixel_to_coords(li, mi, time_idx=time_idx)
                track_note = (
                    f" · tracked@slice RA={tr_ra:.4f}°, Dec={tr_dec:.4f}° "
                    f"(pix {li},{mi})"
                )
            except ValueError as exc:
                track_note = f" · tracked@slice: {exc}"
        self._set_overlay_toggle_display(True)
        status = (
            f"**{name}** · LST {lst}, {freq:.1f} MHz (t={time_idx}, f={freq_idx}) · "
            f"{self._heatmap_method_label()}={val_s} · target RA={ra_h}, Dec={dec_s}"
            f"{track_note}"
        )
        if (
            self._overlay_patch_fit_result is not None
            and self._overlay_patch_fit_result.time_idx == time_idx
            and self._overlay_patch_fit_result.frequency_idx == freq_idx
        ):
            status = (
                f"{status}\n\n"
                f"{_format_patch_fit_diagnostics(self._overlay_patch_fit_result, time_idx, freq_idx)}"
            )
        self._set_status(status)
        self._log(
            f"Heatmap cell — {name}, t={time_idx}, f={freq_idx} ({freq:.1f} MHz); "
            "loading overlay at current view."
        )
        self._schedule_overlay_slice_load(
            time_idx,
            freq_idx,
            preserve_view=True,
        )

    def _reset_heatmap_to_zeros(self) -> None:
        """Replace the heatmap with a zeros grid when centering on a new position."""
        self._invalidate_overlay_patch_fit()
        self._patch_stat_result = None
        self._heatmap_coord = None
        self._ensure_heatmap_grid(force=True, announce=False)

    def _set_overlay_toggle_display(self, enabled: bool) -> None:
        """Sync overlay toggle state without running the toggle handler."""
        self._overlay_enabled = bool(enabled)
        self._overlay_toggle.name = (
            "Overlay: on" if self._overlay_enabled else "Overlay: off"
        )
        self._overlay_toggle.button_type = (
            "success" if self._overlay_enabled else "default"
        )
        if self._overlay_toggle.value != self._overlay_enabled:
            self._suppress_overlay_toggle = True
            try:
                self._overlay_toggle.value = self._overlay_enabled
            finally:
                self._suppress_overlay_toggle = False

    def _ensure_heatmap_grid(
        self,
        *,
        force: bool = False,
        announce: bool = True,
    ) -> None:
        """Display a clickable zeros heatmap so any time/frequency slice can be
        loaded as an overlay before a dynamic spectrum is computed.

        The grid spans the full Zarr ``time × frequency`` shape; clicking a cell
        loads that slice from the Zarr centered on the current coordinate. A
        subsequently generated dynamic spectrum replaces these zeros in place.

        With ``force=True`` the grid is rebuilt even if one is already shown
        (used to discard a stale computed spectrum while keeping a grid to click).
        """
        if self._dataset is None:
            return
        if not force and self._heatmap_grid_ready:
            return
        if (
            not force
            and self._current_source is not None
            and self._heatmap_values is not None
            and np.any(np.isfinite(self._heatmap_values) & (self._heatmap_values != 0))
        ):
            self._heatmap_grid_ready = True
            return
        if self._lst_hours is None or self._freq_mhz is None:
            return
        n_times = int(self._dataset.sizes["time"])
        n_freqs = int(self._dataset.sizes["frequency"])
        self._heatmap_values = np.zeros((n_times, n_freqs), dtype=np.float64)
        figure = self._refresh_heatmap_figure(self._heatmap_values)
        self._heatmap_grid_ready = True
        had_live_figure = (
            self._heatmap_bokeh_handles is not None
            and self._heatmap_pane.object is self._heatmap_bokeh_handles.plot
        )
        if had_live_figure:
            self._push_heatmap_mutation_to_notebook()
        else:
            self._publish_heatmap_figure(figure)
        if announce:
            self._log(
                f"Heatmap grid ready ({n_times} times × {n_freqs} freqs) — enter a "
                "coordinate and click a cell to load that slice as an overlay."
            )

    def _on_overlay_toggle(self, event: object) -> None:
        """Show/hide the radio overlay without touching the heatmap."""
        if self._suppress_overlay_toggle:
            return
        enabled = bool(getattr(event, "new", self._overlay_toggle.value))

        def _run() -> None:
            self._on_overlay_toggle_impl(enabled)

        self._schedule_ui_action(_run)

    def _on_overlay_toggle_impl(self, enabled: bool) -> None:
        self._overlay_enabled = enabled
        self._set_overlay_toggle_display(self._overlay_enabled)
        widget = self._sky_widget
        if widget is None:
            self._set_status("Overlay toggle ready once the Zarr store finishes opening.")
            return
        if self._overlay_enabled:
            if self._coord is not None:
                self._schedule_overlay_slice_load(
                    self._time_idx,
                    self._freq_idx,
                    center_on_target=True,
                    center=self._coord,
                )
                msg = f"Overlay **on** — slice t={self._time_idx}, f={self._freq_idx}."
            else:
                msg = (
                    "Overlay **on** — enter a coordinate and click a heatmap cell "
                    "to load a slice."
                )
            self._log(msg.replace("**", ""))
            self._set_status(msg)
        else:
            clear_image = getattr(widget, "clear_image", None)
            if callable(clear_image):
                clear_image()
            self._force_send_sky_widget_state(widget)
            self._log("Overlay off — hidden.")
            self._set_status("Overlay **off** — HiPS background only.")

    def _refresh_heatmap_figure(self, values: np.ndarray) -> figure:
        """Return the live heatmap figure, creating it once then mutating in place."""
        if self._heatmap_bokeh_handles is None:
            handles = self._create_heatmap_figure_shell(values)
            self._heatmap_bokeh_handles = handles
            return handles.plot
        self._update_heatmap_figure_shell(self._heatmap_bokeh_handles, values)
        return self._heatmap_bokeh_handles.plot

    def _heatmap_hover_payload(
        self,
        values: np.ndarray,
        *,
        n_times: int,
        n_freqs: int,
    ) -> tuple[dict[str, object], list[tuple[str, str]]]:
        time_idx, freq_idx = np.meshgrid(
            np.arange(n_times, dtype=int),
            np.arange(n_freqs, dtype=int),
            indexing="ij",
        )
        flat_time = time_idx.ravel()
        flat_freq = freq_idx.ravel()
        time_values = np.asarray(self._dataset.coords["time"].values)
        hover_data: dict[str, object] = {
            "x": flat_time + 0.5,
            "y": flat_freq + 0.5,
            "time_idx": flat_time,
            "freq_idx": flat_freq,
            "lst_hour": [
                format_heatmap_time_axis_label(
                    time_values,
                    int(t),
                    self._lst_hours,  # type: ignore[arg-type]
                    day_labels=self._time_day_labels,
                )
                for t in flat_time
            ],
            "freq_mhz": self._freq_mhz[flat_freq],  # type: ignore[index]
            "value_display": _row_hover(values),
        }
        tooltips: list[tuple[str, str]] = [
            ("Day · LST", "@lst_hour"),
            ("Freq (MHz)", "@freq_mhz{0.1}"),
            ("Time idx", "@time_idx"),
            ("Freq idx", "@freq_idx"),
            ("Value", "@value_display"),
        ]
        return hover_data, tooltips

    def _heatmap_axis_ticks(
        self,
        n: int,
        axis_values: np.ndarray,
        fmt: Callable[[float], str],
    ) -> tuple[list[float], dict[float, str]]:
        step = 1 if n <= 24 else int(np.ceil(n / 24))
        indices = range(0, n, step)
        ticks = [i + 0.5 for i in indices]
        labels = {tick: fmt(float(axis_values[i])) for tick, i in zip(ticks, indices, strict=True)}
        return ticks, labels

    def _create_heatmap_figure_shell(self, values: np.ndarray) -> _HeatmapBokehHandles:
        n_times, n_freqs = values.shape
        mapper = _color_mapper(values.astype(np.float64, copy=False))
        src = self._current_source
        title_name = src["name"] if src is not None else None
        plot = figure(
            width=1000,
            height=400,
            title=self._heatmap_figure_title(subject=title_name),
            x_range=(0, n_times),
            y_range=(0, n_freqs),
            tools="pan,wheel_zoom,reset,tap",
            active_drag="pan",
            active_tap="tap",
        )
        image_renderer = plot.image(
            image=[values.T.astype(np.float64, copy=False)],
            x=0,
            y=0,
            dw=n_times,
            dh=n_freqs,
            color_mapper=mapper,
        )
        hover_data, tooltips = self._heatmap_hover_payload(
            values, n_times=n_times, n_freqs=n_freqs
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
        hover_tool = HoverTool(renderers=[hover_renderer], tooltips=tooltips)
        plot.add_tools(hover_tool)

        time_values = np.asarray(self._dataset.coords["time"].values)
        x_ticks, x_labels = self._heatmap_axis_ticks(
            n_times,
            np.arange(n_times, dtype=float),
            lambda i: format_heatmap_time_axis_label(
                time_values,
                int(i),
                self._lst_hours,  # type: ignore[arg-type]
                day_labels=self._time_day_labels,
            ),
        )
        y_ticks, y_labels = self._heatmap_axis_ticks(
            n_freqs, self._freq_mhz, lambda v: f"{float(v):.1f}"  # type: ignore[arg-type]
        )
        plot.xaxis.ticker = FixedTicker(ticks=x_ticks)
        plot.yaxis.ticker = FixedTicker(ticks=y_ticks)
        plot.xaxis.major_label_overrides = x_labels
        plot.yaxis.major_label_overrides = y_labels
        plot.xaxis.axis_label = "Day · LST hour"
        plot.yaxis.axis_label = "Frequency (MHz)"
        plot.xaxis.major_label_orientation = math.pi / 4

        def _on_tap(event: Tap) -> None:
            if event.x is None or event.y is None or self._heatmap_values is None:
                return
            n_t, n_f = self._heatmap_values.shape
            t_idx = _heatmap_index_from_coord(event.x, n_t)
            f_idx = _heatmap_index_from_coord(event.y, n_f)
            self._schedule_ui_action(lambda: self._on_heatmap_tap(t_idx, f_idx))

        plot.on_event(Tap, _on_tap)
        return _HeatmapBokehHandles(
            plot=plot,
            image_renderer=image_renderer,
            hover_source=hover_src,
            hover_tool=hover_tool,
        )

    def _update_heatmap_figure_shell(
        self,
        handles: _HeatmapBokehHandles,
        values: np.ndarray,
    ) -> None:
        n_times, n_freqs = values.shape
        plot = handles.plot
        src = self._current_source
        title_name = src["name"] if src is not None else None
        plot.title.text = self._heatmap_figure_title(subject=title_name)
        plot.x_range.start = 0
        plot.x_range.end = n_times
        plot.y_range.start = 0
        plot.y_range.end = n_freqs

        glyph = handles.image_renderer.glyph
        handles.image_renderer.data_source.data = {
            "image": [values.T.astype(np.float64, copy=False)],
        }
        glyph.dw = n_times
        glyph.dh = n_freqs
        glyph.color_mapper = _color_mapper(values.astype(np.float64, copy=False))

        hover_data, tooltips = self._heatmap_hover_payload(
            values, n_times=n_times, n_freqs=n_freqs
        )
        handles.hover_source.data = hover_data
        handles.hover_tool.tooltips = tooltips

        time_values = np.asarray(self._dataset.coords["time"].values)
        x_ticks, x_labels = self._heatmap_axis_ticks(
            n_times,
            np.arange(n_times, dtype=float),
            lambda i: format_heatmap_time_axis_label(
                time_values,
                int(i),
                self._lst_hours,  # type: ignore[arg-type]
                day_labels=self._time_day_labels,
            ),
        )
        y_ticks, y_labels = self._heatmap_axis_ticks(
            n_freqs, self._freq_mhz, lambda v: f"{float(v):.1f}"  # type: ignore[arg-type]
        )
        plot.xaxis.ticker = FixedTicker(ticks=x_ticks)
        plot.yaxis.ticker = FixedTicker(ticks=y_ticks)
        plot.xaxis.major_label_overrides = x_labels
        plot.yaxis.major_label_overrides = y_labels
