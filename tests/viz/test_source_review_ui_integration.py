"""Integration tests: heatmap publish + spinner + coordinate field together."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from astropy.coordinates import SkyCoord

pn = pytest.importorskip("panel")
pytest.importorskip("astrowidget")

from ovro_lwa_portal.viz import pipeline_qa_app as pqa
from ovro_lwa_portal.viz import source_review_app as sra
from ovro_lwa_portal.viz.source_review import DatasetLoad, run_dataset_load
from ovro_lwa_portal.viz.source_review_app import SourceReview, SourceReviewConfig
from ovro_lwa_portal.viz.source_review_data import HeatmapLoad, build_source_from_coordinate
from tests.viz.panel_ui_testkit import PanelUITestHarness, QueuedIOLoop

CAS_A = SkyCoord(ra=350.85 * u.deg, dec=58.815 * u.deg, frame="icrs")


def _mount_review(tmp_path: Path) -> tuple[PanelUITestHarness, SourceReview]:
    zarr = tmp_path / "store.zarr"
    zarr.mkdir()
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
    )
    harness = PanelUITestHarness()
    session = harness.mount(review._layout)
    review._ui_session_override = session
    review._ui_session = session
    return harness, review


def _seed_dataset(review: SourceReview, *, n_times: int = 6, n_freqs: int = 4) -> None:
    times = np.arange(
        np.datetime64("2025-01-01T00:00:00"),
        np.datetime64("2025-01-01T00:00:00") + np.timedelta64(n_times, "s"),
        np.timedelta64(1, "s"),
    )
    freqs = np.linspace(50e6, 55e6, n_freqs)
    ds = xr.Dataset(
        {
            "SKY": (("time", "frequency", "l", "m"), np.zeros((n_times, n_freqs, 8, 8))),
        },
        coords={
            "time": times,
            "frequency": freqs,
            "l": np.arange(8),
            "m": np.arange(8),
        },
    )
    review._dataset = ds
    review._lst_hours = np.linspace(4.0, 5.0, n_times)
    review._freq_mhz = freqs / 1e6
    review._default_time_idx = 0
    review._default_freq_idx = 0
    review._time_idx = 0
    review._freq_idx = 0
    review._current_source = build_source_from_coordinate("Cas A", CAS_A)


def _heatmap_title(harness: PanelUITestHarness, review: SourceReview) -> str:
    return harness.bokeh_model(review._heatmap_pane, review._layout).title.text


def _heatmap_values_max(review: SourceReview) -> float:
    if review._heatmap_values is None:
        return 0.0
    return float(np.nanmax(review._heatmap_values))


def _assert_heatmap_bokeh_model_live(harness: PanelUITestHarness, review: SourceReview) -> None:
    fig = review._heatmap_pane.object
    assert fig is not None
    model = harness.bokeh_model(review._heatmap_pane, review._layout)
    assert model.title.text == fig.title.text
    assert "Heatmap loads" not in model.title.text


def _spinner_spinning(review: SourceReview) -> bool:
    return "ovro-lwa-spin" in (review._loading_widget.value or "")


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_open_grid_replaces_placeholder_heatmap(tmp_path: Path) -> None:
    harness, review = _mount_review(tmp_path)
    _seed_dataset(review)

    placeholder_title = _heatmap_title(harness, review)
    assert "Heatmap loads" in placeholder_title

    harness.run_ui(harness.session(review._layout), review._ensure_heatmap_grid)

    assert _heatmap_title(harness, review) != placeholder_title
    assert "click a cell" in _heatmap_title(harness, review).lower()
    assert _heatmap_values_max(review) == 0.0
    _assert_heatmap_bokeh_model_live(harness, review)


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_finish_heatmap_inside_dispatch_updates_bokeh_model(tmp_path: Path) -> None:
    """Regression: publish must run after hold, not during _finish_heatmap dispatch."""
    harness, review = _mount_review(tmp_path)
    _seed_dataset(review)
    harness.run_ui(harness.session(review._layout), review._ensure_heatmap_grid)

    review._heatmap_job_id = 1
    assert _heatmap_values_max(review) == 0.0
    values = np.linspace(1.0, 24.0, 24).reshape(6, 4)
    payload = HeatmapLoad(values=values, patch_fit_result=None, patch_stat_result=None)
    src = review._current_source
    assert src is not None

    harness.run_ui(
        harness.session(review._layout),
        lambda: review._finish_heatmap(src, payload, None, job_id=1, started_at=time.perf_counter()),
    )

    assert "Cas A" in _heatmap_title(harness, review)
    assert _heatmap_values_max(review) == pytest.approx(24.0)
    _assert_heatmap_bokeh_model_live(harness, review)


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_generate_flow_spinner_and_heatmap_and_coord_on_sky_click(tmp_path: Path) -> None:
    harness, review = _mount_review(tmp_path)
    _seed_dataset(review)
    harness.run_ui(harness.session(review._layout), review._ensure_heatmap_grid)

    review._heatmap_job_id = 1
    src = review._current_source
    assert src is not None

    harness.run_ui(
        harness.session(review._layout),
        lambda: (
            review._sync_spinner(True),
            setattr(review, "loading", True),
        ),
    )
    assert _spinner_spinning(review)

    values = np.full((6, 4), 42.0)
    payload = HeatmapLoad(values=values, patch_fit_result=None, patch_stat_result=None)
    harness.run_ui(
        harness.session(review._layout),
        lambda: review._finish_heatmap(src, payload, None, job_id=1, started_at=time.perf_counter()),
    )

    assert not _spinner_spinning(review)
    assert _heatmap_values_max(review) == pytest.approx(42.0)
    _assert_heatmap_bokeh_model_live(harness, review)

    review._sky_widget = MagicMock()
    review._sky_widget.clicked_coord = (123.4567, 45.6789)
    review._on_sky_widget_click(None)

    coord_model = harness.bokeh_model(review._coord_input, review._layout)
    assert "123.4567" in coord_model.value
    assert "45.6789" in coord_model.value


def _mount_review_jupyter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    layout_only: bool = False,
) -> tuple[PanelUITestHarness, SourceReview, QueuedIOLoop]:
    """SourceReview with production ``JupyterPanelUISession`` + queued io_loop."""
    from ovro_lwa_portal.viz.panel_ui_session import JupyterPanelUISession

    zarr = tmp_path / "store.zarr"
    zarr.mkdir()
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
    )
    harness = PanelUITestHarness()
    if layout_only:
        harness.mount_layout_only(review._layout)
    else:
        harness.mount(review._layout)
    loop = QueuedIOLoop()
    monkeypatch.setattr(pqa, "_IPYTHON_IO_LOOP", loop)
    monkeypatch.setattr(pqa, "_resolve_ipython_event_loop", lambda: loop)
    monkeypatch.setattr(pqa, "_is_jupyter_kernel_context", lambda: True)
    monkeypatch.setattr(pqa, "_schedule_ipython_main", loop.add_callback)
    review._ui_session = JupyterPanelUISession(review._notebook_ui_views)
    return harness, review, loop


def _make_dataset(*, n_times: int = 6, n_freqs: int = 4) -> xr.Dataset:
    from tests.test_fits_to_zarr import _make_sin_wcs_header_str

    times = np.arange(
        np.datetime64("2025-01-01T00:00:00"),
        np.datetime64("2025-01-01T00:00:00") + np.timedelta64(n_times, "s"),
        np.timedelta64(1, "s"),
    )
    freqs = np.linspace(50e6, 55e6, n_freqs)
    hdr = _make_sin_wcs_header_str(nx=8, ny=8, crval1=350.85, crval2=58.815)
    ds = xr.Dataset(
        {
            "SKY": (
                ("time", "frequency", "polarization", "l", "m"),
                np.zeros((n_times, n_freqs, 1, 8, 8)),
            ),
            "wcs_header_str": (["time"], [hdr] * n_times),
        },
        coords={
            "time": times,
            "frequency": freqs,
            "polarization": [0],
            "l": np.arange(8),
            "m": np.arange(8),
        },
    )
    return ds


def _drain_io_loop(loop: QueuedIOLoop, *, timeout_s: float = 2.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while loop.callbacks and time.perf_counter() < deadline:
        loop.flush()
        time.sleep(0.001)


def _flush_jupyter_io(loop: QueuedIOLoop) -> None:
    """Drain scheduled kernel callbacks (dispatch, then after-hold publish)."""
    while loop.callbacks:
        loop.flush()


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_jupyter_session_generate_before_deferred_grid_does_not_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: deferred open-time grid must not overwrite a finished heatmap."""
    harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch)
    _seed_dataset(review)

    review._heatmap_job_id = 1
    src = review._current_source
    assert src is not None
    payload = HeatmapLoad(
        values=np.full((6, 4), 77.0),
        patch_fit_result=None,
        patch_stat_result=None,
    )

    review._ui.dispatch(
        lambda: review._finish_heatmap(
            src, payload, None, job_id=1, started_at=time.perf_counter()
        )
    )
    _flush_jupyter_io(loop)
    assert _heatmap_values_max(review) == pytest.approx(77.0)

    review._ui.defer_dispatch(review._ensure_heatmap_grid)
    _flush_jupyter_io(loop)

    assert _heatmap_values_max(review) == pytest.approx(77.0)
    _assert_heatmap_bokeh_model_live(harness, review)


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_jupyter_session_open_generate_and_sky_click(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production session path: deferred Bokeh/widget updates flush on io_loop."""
    harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch)
    _seed_dataset(review)

    review._ui.defer_dispatch(review._ensure_heatmap_grid)
    _flush_jupyter_io(loop)
    _assert_heatmap_bokeh_model_live(harness, review)

    review._heatmap_job_id = 1
    src = review._current_source
    assert src is not None

    review._ui.dispatch(
        lambda: (
            review._sync_spinner(True),
            setattr(review, "loading", True),
        )
    )
    _flush_jupyter_io(loop)
    assert _spinner_spinning(review)

    payload = HeatmapLoad(
        values=np.full((6, 4), 99.0),
        patch_fit_result=None,
        patch_stat_result=None,
    )
    review._ui.dispatch(
        lambda: review._finish_heatmap(
            src, payload, None, job_id=1, started_at=time.perf_counter()
        )
    )
    _flush_jupyter_io(loop)

    assert not _spinner_spinning(review)
    assert _heatmap_values_max(review) == pytest.approx(99.0)
    _assert_heatmap_bokeh_model_live(harness, review)

    review._sky_widget = MagicMock()
    review._sky_widget.clicked_coord = (6.5916, 64.0770)
    review._on_sky_widget_click(None)
    _flush_jupyter_io(loop)

    coord_model = harness.bokeh_model(review._coord_input, review._layout)
    assert "6.5916" in coord_model.value
    assert "64.0770" in coord_model.value


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_jupyter_session_generate_button_updates_heatmap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate heatmap button must publish through the production dispatch path."""
    from ovro_lwa_portal.viz import source_review_app as sra

    harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch)
    _seed_dataset(review)
    review.coordinate_string = "Cas A"
    review._coord_input.value = "Cas A"
    review._coord_input.value_input = "Cas A"

    review._ui.defer_dispatch(review._ensure_heatmap_grid)
    _flush_jupyter_io(loop)

    values = np.full((6, 4), 55.0)

    def _fast_compute(*_args, **_kwargs) -> HeatmapLoad:
        return HeatmapLoad(values=values, patch_fit_result=None, patch_stat_result=None)

    monkeypatch.setattr(sra, "compute_source_heatmap", _fast_compute)

    review._on_generate_heatmap()
    _flush_jupyter_io(loop)
    while loop.callbacks:
        loop.flush()

    assert _heatmap_values_max(review) == pytest.approx(55.0)
    _assert_heatmap_bokeh_model_live(harness, review)


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_jupyter_layout_only_generate_syncs_nested_heatmap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Notebook-like comm: layout root registered, nested heatmap pane is not."""
    sync_targets: list[pn.viewable.Viewable] = []
    real_sync = pqa.sync_pane_to_notebook

    def _track_sync(pane: pn.viewable.Viewable, *root_views: pn.viewable.Viewable) -> None:
        sync_targets.append(pane)
        real_sync(pane, *root_views)

    monkeypatch.setattr(pqa, "sync_pane_to_notebook", _track_sync)

    harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review.coordinate_string = "Cas A"
    review._coord_input.value = "Cas A"
    review._coord_input.value_input = "Cas A"

    review._ui.defer_dispatch(review._ensure_heatmap_grid)
    _flush_jupyter_io(loop)

    values = np.full((6, 4), 61.0)

    def _fast_compute(*_args, **_kwargs) -> HeatmapLoad:
        return HeatmapLoad(values=values, patch_fit_result=None, patch_stat_result=None)

    monkeypatch.setattr(sra, "compute_source_heatmap", _fast_compute)

    review._on_generate_heatmap()
    _drain_io_loop(loop)

    assert review._heatmap_pane in sync_targets
    assert _heatmap_values_max(review) == pytest.approx(61.0)
    _assert_heatmap_bokeh_model_live(harness, review)


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_jupyter_session_slow_zarr_open_then_generate_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interleave slow Zarr open progress with Generate; heatmap must still publish."""
    harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
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
    _flush_jupyter_io(loop)
    threading.Thread(target=_work, daemon=True).start()

    deadline = time.perf_counter() + 2.0
    while not open_done.is_set() and time.perf_counter() < deadline:
        _drain_io_loop(loop, timeout_s=0.05)
        time.sleep(0.01)
    _drain_io_loop(loop)

    assert review._dataset is not None
    assert "click a cell" in _heatmap_title(harness, review).lower()

    review.coordinate_string = "Cas A"
    review._coord_input.value = "Cas A"
    review._coord_input.value_input = "Cas A"

    def _fast_compute(*_args, **_kwargs) -> HeatmapLoad:
        return HeatmapLoad(
            values=np.full((6, 4), 88.0),
            patch_fit_result=None,
            patch_stat_result=None,
        )

    monkeypatch.setattr(sra, "compute_source_heatmap", _fast_compute)
    review._on_generate_heatmap()
    _drain_io_loop(loop)

    assert _heatmap_values_max(review) == pytest.approx(88.0)
    _assert_heatmap_bokeh_model_live(harness, review)


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_jupyter_spinner_stays_until_heatmap_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spinner stays on through compute, heatmap publish, and overlay load."""
    compute_started = threading.Event()
    release_compute = threading.Event()

    def _slow_compute(*_args, **_kwargs) -> HeatmapLoad:
        compute_started.set()
        assert release_compute.wait(timeout=2.0)
        return HeatmapLoad(
            values=np.full((6, 4), 71.0),
            patch_fit_result=None,
            patch_stat_result=None,
        )

    harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review.coordinate_string = "Cas A"
    review._coord_input.value = "Cas A"
    review._coord_input.value_input = "Cas A"
    monkeypatch.setattr(sra, "compute_source_heatmap", _slow_compute)

    review._on_generate_heatmap()
    _drain_io_loop(loop, timeout_s=2.0)

    assert compute_started.is_set()
    assert review.loading is True
    assert _spinner_spinning(review)

    release_compute.set()
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        _drain_io_loop(loop, timeout_s=0.1)
        if not review.loading:
            break
        time.sleep(0.01)

    assert review.loading is False
    assert not _spinner_spinning(review)
    assert _heatmap_values_max(review) == pytest.approx(71.0)
    _assert_heatmap_bokeh_model_live(harness, review)


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_overlay_toggle_and_heatmap_tap_use_schedule_not_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlay loads must not wrap Zarr reads in Panel dispatch (double layout push)."""
    harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review._coord = CAS_A
    review._heatmap_values = np.zeros((6, 4))
    widget = MagicMock()
    widget.overlay_view_lock = True
    widget.crval = (350.85, 58.815)
    widget.view_ra = 350.85
    widget.view_dec = 58.815
    widget.view_center_skycoord.return_value = CAS_A
    review._sky_widget = widget

    scheduled: list[str] = []
    real_schedule = review._ui.schedule

    def _track_schedule(callback) -> None:
        scheduled.append("schedule")
        real_schedule(callback)

    monkeypatch.setattr(review._ui, "schedule", _track_schedule)
    monkeypatch.setattr(review._ui, "defer_dispatch", lambda _cb: scheduled.append("defer_dispatch"))

    review._on_overlay_toggle_impl(True)
    assert review.loading is True
    assert _spinner_spinning(review)
    _flush_jupyter_io(loop)

    assert review.loading is False
    assert not _spinner_spinning(review)
    assert "schedule" in scheduled
    assert "defer_dispatch" not in scheduled
    widget.update_slice.assert_called_once()

    scheduled.clear()
    review._on_heatmap_tap(2, 1)
    assert review.loading is True
    _flush_jupyter_io(loop)

    assert review.loading is False
    assert scheduled == ["schedule"]
    assert widget.update_slice.call_count == 2
