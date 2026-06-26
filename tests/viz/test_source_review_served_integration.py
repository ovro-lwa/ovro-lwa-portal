"""Integration tests: threaded Zarr open under ``panel serve`` (ServedPanelUISession)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore:Message serialization failed with:UserWarning",
    "ignore:There is no current event loop:DeprecationWarning",
)

pn = pytest.importorskip("panel")
pytest.importorskip("astrowidget")

from ovro_lwa_portal.viz import source_review_app as sra
from ovro_lwa_portal.viz.panel_ui_session import ServedPanelUISession
from ovro_lwa_portal.viz.source_review import DatasetLoad, run_dataset_load
from ovro_lwa_portal.viz.source_review_app import SourceReview, SourceReviewConfig
from ovro_lwa_portal.viz.source_review_data import HeatmapLoad, build_source_from_coordinate
from tests.viz.panel_ui_testkit import BokehTickHarness, PanelUITestHarness
from tests.viz.test_source_review_ui_integration import (
    CAS_A,
    _assert_heatmap_bokeh_model_live,
    _heatmap_title,
    _heatmap_values_max,
    _make_dataset,
)


def _mount_review_served(
    tmp_path: Path,
    *,
    layout_only: bool = False,
) -> tuple[PanelUITestHarness, SourceReview, ServedPanelUISession, BokehTickHarness]:
    """SourceReview with production ``ServedPanelUISession`` + Bokeh tick flusher."""
    zarr = tmp_path / "store.zarr"
    zarr.mkdir()
    holder: dict[str, SourceReview] = {}

    def _root_views() -> tuple:
        review = holder["review"]
        return (
            review._layout,
            review._status_pane,
            review._heatmap_pane,
            review._coord_input,
        )

    session = ServedPanelUISession(_root_views)
    review = SourceReview(
        zarr,
        patch_scale=5.0,
        sky_fov_deg=8.0,
        patch_fit_max_reduced_chi_squared=10.0,
        config=SourceReviewConfig(
            hips_root=tmp_path,
            hips_background=tmp_path / "missing.hips",
        ),
        validate_zarr=False,
        ui_session=session,
    )
    holder["review"] = review

    harness = PanelUITestHarness()
    if layout_only:
        harness.mount_layout_only(review._layout)
    else:
        harness.mount(review._layout)
    ticks = harness.served_ticks()
    session.bind_document(harness.doc)
    return harness, review, session, ticks


def _drain_served_ticks(ticks: BokehTickHarness, *, timeout_s: float = 2.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        pending = list(ticks.doc.callbacks._session_callbacks)
        if not pending:
            return
        ticks.flush_ticks()
        time.sleep(0.001)


def _flush_served_ticks(ticks: BokehTickHarness) -> None:
    """Drain scheduled next-tick callbacks (including nested schedules)."""
    while list(ticks.doc.callbacks._session_callbacks):
        ticks.flush_ticks()


def test_served_loading_spinner_syncs_spin_css(tmp_path: Path) -> None:
    """``LoadingSpinner`` needs widget model sync under ``panel serve``."""
    harness, review, _session, _ticks = _mount_review_served(tmp_path)
    review._sync_spinner(True)
    model = harness.bokeh_model(review._loading_pane, review._layout)
    assert review._loading_pane.value is True
    assert "spin" in model.css_classes
    review._sync_spinner(False)
    model = harness.bokeh_model(review._loading_pane, review._layout)
    assert review._loading_pane.value is False
    assert "spin" not in model.css_classes


def test_serve_center_hi_ps_only_goto_then_single_remount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Center on serve: goto in Python, one bundle remount (not at goto time)."""
    from unittest.mock import MagicMock

    import astropy.units as u

    _harness, review, _session, ticks = _mount_review_served(tmp_path, layout_only=True)
    review._dataset = _make_dataset()
    review.coordinate_string = "Cas A"

    widget = MagicMock()
    widget.image_shape = (0, 0)
    widget.image_revision = 0
    review._sky_widget = widget

    remounts = 0

    def _count_remount(_w: object) -> None:
        nonlocal remounts
        remounts += 1

    monkeypatch.setattr(
        review,
        "_resolve_active_coordinate",
        lambda: (CAS_A, "Cas A"),
    )
    monkeypatch.setattr(review, "_remount_sky_ipywidget_model", _count_remount)
    monkeypatch.setattr(review, "_sync_fit_overlay_button", lambda: None)
    monkeypatch.setattr(review, "_log_overlay_diagnostics", lambda *a, **k: None)
    monkeypatch.setattr(review, "_reset_heatmap_to_zeros", lambda: None)

    review._on_slew_impl()
    _flush_served_ticks(ticks)

    widget.goto.assert_called_once()
    goto_target = widget.goto.call_args[0][0]
    assert goto_target.separation(CAS_A) < 1 * u.arcsec
    assert widget.goto.call_args[1]["fov"] == 8.0 * u.deg
    widget.set_crosshair.assert_called_once()
    assert remounts == 1


def test_served_sky_bundle_remount_replaces_bokeh_model(tmp_path: Path) -> None:
    """Serve mode must replace the Bokeh IPyWidget model, not rely on send_state."""
    from panel.io.state import set_curdoc

    from astrowidget import SkyWidget
    from ovro_lwa_portal.viz.source_review_app import ServeSkyWidget, configure_source_review_serve

    configure_source_review_serve()
    harness, review, session, ticks = _mount_review_served(tmp_path)
    session.bind_document(harness.doc)
    with set_curdoc(harness.doc):
        widget = ServeSkyWidget()
    review._sky_widget = widget
    review._install_sky_ipywidget_pane(widget)
    ref = next(iter(review._sky_pane._models))
    old_model = review._sky_pane._models[ref][0]
    widget.view_ra = 12.5
    review._remount_sky_ipywidget_model(widget)
    new_model = review._sky_pane._models[ref][0]
    assert new_model is not old_model


def test_served_worker_open_updates_panel_html_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: worker-thread Zarr open must update Panel HTML log, not ipywidgets."""
    harness, review, _session, ticks = _mount_review_served(tmp_path, layout_only=True)
    monkeypatch.setattr(review, "_mount_sky_widget", lambda _ds: None)
    assert review._log_widget is None
    assert isinstance(review._log_pane, pn.pane.HTML)

    log_threads: list[threading.Thread] = []
    real_refresh = review._refresh_log_widget

    def _track_refresh() -> None:
        log_threads.append(threading.current_thread())
        real_refresh()

    monkeypatch.setattr(review, "_refresh_log_widget", _track_refresh)

    open_done = threading.Event()

    def _slow_open(report) -> DatasetLoad:
        report("Opening zarr metadata (slow)…")
        time.sleep(0.05)
        ds = _make_dataset()
        report("Zarr opened (0.1 s)")
        lst_hours = np.linspace(4.0, 5.0, int(ds.sizes["time"]))
        freq_mhz = np.asarray(ds.coords["frequency"].values, dtype=np.float64) / 1e6
        report("Coordinates ready")
        open_done.set()
        return DatasetLoad(
            dataset=ds,
            default_time_idx=0,
            default_freq_idx=0,
            lst_hours=lst_hours,
            freq_mhz=freq_mhz,
        )

    def _work() -> None:
        run_dataset_load(
            open_dataset=_slow_open,
            dispatch=review._dispatch,
            log_dispatch=review._schedule_main,
            on_loaded=lambda load: review._finish_open(
                load.dataset,
                load.default_time_idx,
                load.default_freq_idx,
                load.lst_hours,
                load.freq_mhz,
                None,
            ),
            on_error=lambda exc: review._finish_open(None, None, None, None, None, exc),
            log=review._log,
        )

    review._dispatch(
        lambda: (
            setattr(review, "loading", True),
            review._sync_spinner(True),
            review._log("Opening store…"),
        )
    )
    _flush_served_ticks(ticks)
    threading.Thread(target=_work, daemon=True).start()

    deadline = time.perf_counter() + 2.0
    while not open_done.is_set() and time.perf_counter() < deadline:
        _drain_served_ticks(ticks, timeout_s=0.05)
        time.sleep(0.01)
    _flush_served_ticks(ticks)

    assert review._dataset is not None
    assert "Opened —" in review.log_text
    assert "Opened —" in (review._log_pane.object or "")
    assert log_threads
    assert all(t is threading.main_thread() for t in log_threads)
    assert "click a cell" in _heatmap_title(harness, review).lower()


def test_served_session_slow_zarr_open_then_generate_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interleave slow Zarr open with Generate under ``ServedPanelUISession``."""
    harness, review, _session, ticks = _mount_review_served(tmp_path, layout_only=True)
    monkeypatch.setattr(review, "_mount_sky_widget", lambda _ds: None)
    open_done = threading.Event()

    def _slow_open(report) -> DatasetLoad:
        report("Opening zarr metadata (slow)…")
        time.sleep(0.05)
        ds = _make_dataset()
        report("Zarr opened (0.1 s)")
        report("Computing LST labels for time axis…")
        lst_hours = np.linspace(4.0, 5.0, int(ds.sizes["time"]))
        freq_mhz = np.asarray(ds.coords["frequency"].values, dtype=np.float64) / 1e6
        report("Coordinates ready")
        open_done.set()
        return DatasetLoad(
            dataset=ds,
            default_time_idx=0,
            default_freq_idx=0,
            lst_hours=lst_hours,
            freq_mhz=freq_mhz,
        )

    def _work() -> None:
        run_dataset_load(
            open_dataset=_slow_open,
            dispatch=review._dispatch,
            log_dispatch=review._schedule_main,
            on_loaded=lambda load: review._finish_open(
                load.dataset,
                load.default_time_idx,
                load.default_freq_idx,
                load.lst_hours,
                load.freq_mhz,
                None,
            ),
            on_error=lambda exc: review._finish_open(None, None, None, None, None, exc),
            log=review._log,
        )

    review._dispatch(
        lambda: (
            setattr(review, "loading", True),
            review._sync_spinner(True),
            review._log("Opening store…"),
        )
    )
    _flush_served_ticks(ticks)
    threading.Thread(target=_work, daemon=True).start()

    deadline = time.perf_counter() + 2.0
    while not open_done.is_set() and time.perf_counter() < deadline:
        _drain_served_ticks(ticks, timeout_s=0.05)
        time.sleep(0.01)
    _flush_served_ticks(ticks)

    assert review._dataset is not None
    assert "click a cell" in _heatmap_title(harness, review).lower()

    review.coordinate_string = "Cas A"
    review._coord_input.value = "Cas A"
    review._coord_input.value_input = "Cas A"
    review._current_source = build_source_from_coordinate("Cas A", CAS_A)

    def _fast_compute(*_args, **_kwargs) -> HeatmapLoad:
        return HeatmapLoad(
            values=np.full((6, 4), 88.0),
            patch_fit_result=None,
            patch_stat_result=None,
        )

    monkeypatch.setattr(sra, "compute_source_heatmap", _fast_compute)
    review._on_generate_heatmap()
    _flush_served_ticks(ticks)

    assert _heatmap_values_max(review) == pytest.approx(88.0)
    _assert_heatmap_bokeh_model_live(harness, review)
