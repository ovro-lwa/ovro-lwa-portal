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
CYG_A = SkyCoord(ra=299.8682 * u.deg, dec=40.7339 * u.deg, frame="icrs")


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
    assert "mad" in _heatmap_title(harness, review)
    assert _heatmap_values_max(review) == 0.0
    _assert_heatmap_bokeh_model_live(harness, review)


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_heatmap_title_updates_in_place_on_method_change(tmp_path: Path) -> None:
    """Regression: title/method changes must mutate the registered Bokeh figure."""
    harness, review = _mount_review(tmp_path)
    _seed_dataset(review)
    harness.run_ui(harness.session(review._layout), review._ensure_heatmap_grid)

    plot = review._heatmap_pane.object
    assert plot is not None
    assert "mad" in harness.bokeh_model(review._heatmap_pane, review._layout).title.text

    review.heatmap_method = "std"
    harness.run_ui(
        harness.session(review._layout),
        review._on_heatmap_method_change_impl,
    )

    assert review._heatmap_pane.object is plot
    title = harness.bokeh_model(review._heatmap_pane, review._layout).title.text
    assert "std" in title
    assert "mad" not in title


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
    assert "mad" in _heatmap_title(harness, review)
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
def test_generate_heatmap_mutation_push_reaches_browser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: Generate must mutate the mounted zeros-grid figure for live Jupyter."""
    mutation_calls: list[bool] = []
    publish_calls: list[bool] = []
    real_mutation = pqa.push_bokeh_pane_mutation_to_notebook
    real_publish = pqa.publish_bokeh_pane_to_notebook

    def _track_mutation(
        pane: object,
        *root_views: object,
        force_push: bool = False,
    ) -> None:
        mutation_calls.append(bool(force_push))
        real_mutation(pane, *root_views, force_push=force_push)

    def _track_publish(
        pane: object,
        value: object,
        *root_views: object,
        force_push: bool = False,
    ) -> None:
        publish_calls.append(bool(force_push))
        real_publish(pane, value, *root_views, force_push=force_push)

    monkeypatch.setattr(sra, "push_bokeh_pane_mutation_to_notebook", _track_mutation)
    monkeypatch.setattr(pqa, "push_bokeh_pane_mutation_to_notebook", _track_mutation)
    monkeypatch.setattr(sra, "publish_bokeh_pane_to_notebook", _track_publish)
    monkeypatch.setattr(pqa, "publish_bokeh_pane_to_notebook", _track_publish)

    harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review.coordinate_string = "Cas A"
    review._coord_input.value = "Cas A"
    review._coord_input.value_input = "Cas A"
    review._ui.defer_dispatch(review._ensure_heatmap_grid)
    _flush_jupyter_io(loop)
    mutation_calls.clear()
    publish_calls.clear()

    values = np.full((6, 4), 71.0)

    def _fast_compute(*_args, **_kwargs) -> HeatmapLoad:
        return HeatmapLoad(values=values, patch_fit_result=None, patch_stat_result=None)

    monkeypatch.setattr(sra, "compute_source_heatmap", _fast_compute)
    review._on_generate_heatmap()
    _flush_jupyter_io(loop)
    while loop.callbacks:
        loop.flush()

    assert _heatmap_values_max(review) == pytest.approx(71.0)
    assert mutation_calls and mutation_calls[-1] is True
    assert publish_calls == []
    _assert_heatmap_bokeh_model_live(harness, review)


def test_method_change_does_not_auto_compute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    harness, review = _mount_review(tmp_path)
    _seed_dataset(review)
    review._coord_input.value = "Cas A"
    review._apply_coordinate_from_field()
    harness.run_ui(harness.session(review._layout), review._ensure_heatmap_grid)

    called: list[str] = []

    def _fail_compute(*_args, **_kwargs) -> HeatmapLoad:
        called.append("compute")
        raise AssertionError("compute_source_heatmap must not run on method change")

    monkeypatch.setattr(sra, "compute_source_heatmap", _fail_compute)
    review.heatmap_method = "std"
    harness.run_ui(harness.session(review._layout), review._on_heatmap_method_change_impl)

    assert called == []
    assert "std" in harness.bokeh_model(review._heatmap_pane, review._layout).title.text


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_jupyter_method_change_mutates_heatmap_title_after_generate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Method dropdown must push in-place title edits to layout-root-only comms."""
    from ovro_lwa_portal.viz import pipeline_qa_app as pqa

    mutation_calls: list[bool] = []
    real_mutation = pqa.push_bokeh_pane_mutation_to_notebook

    def _track_mutation(pane: object, *root_views: object, force_push: bool = False) -> None:
        mutation_calls.append(bool(force_push))
        real_mutation(pane, *root_views, force_push=force_push)

    monkeypatch.setattr(sra, "push_bokeh_pane_mutation_to_notebook", _track_mutation)
    monkeypatch.setattr(pqa, "push_bokeh_pane_mutation_to_notebook", _track_mutation)

    harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review.coordinate_string = "Cas A"
    review._coord_input.value = "Cas A"
    review._coord_input.value_input = "Cas A"

    src = review._current_source
    assert src is not None
    review._apply_heatmap(
        src,
        HeatmapLoad(
            values=np.full((6, 4), 12.0),
            patch_fit_result=None,
            patch_stat_result=None,
        ),
    )
    _flush_jupyter_io(loop)
    assert "mad" in harness.bokeh_model(review._heatmap_pane, review._layout).title.text

    mutation_calls.clear()
    review.heatmap_method = "patch_max"
    review._dispatch(review._on_heatmap_method_change_impl)
    _flush_jupyter_io(loop)

    title = harness.bokeh_model(review._heatmap_pane, review._layout).title.text
    assert "patch_max" in title
    assert mutation_calls and mutation_calls[-1] is True


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
    """Spinner stays on through compute until the heatmap is confirmed in the browser."""
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
        if abs(_heatmap_values_max(review) - 71.0) < 1e-6 and not review.loading:
            break
        time.sleep(0.01)

    assert not review.loading
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


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_overlay_single_flight_drops_stale_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the latest scheduled overlay load runs when several are queued quickly."""
    harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review._coord = CAS_A
    widget = MagicMock()
    widget.overlay_view_lock = True
    widget.crval = (350.85, 58.815)
    widget.view_ra = 350.85
    widget.view_dec = 58.815
    widget.view_center_skycoord.return_value = CAS_A
    widget.image_revision = 0
    review._sky_widget = widget

    update_calls: list[tuple[int, int]] = []

    def _record_update_slice(time_idx, freq_idx, **kwargs) -> None:
        update_calls.append((int(time_idx), int(freq_idx)))
        widget.image_revision = int(widget.image_revision) + 1

    widget.update_slice.side_effect = _record_update_slice

    review._schedule_overlay_slice_load(0, 0, preserve_view=True)
    review._schedule_overlay_slice_load(5, 0, preserve_view=True)
    _flush_jupyter_io(loop)

    assert update_calls == [(5, 0)]
    assert review.loading is False
    assert not _spinner_spinning(review)


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_center_schedules_overlay_instead_of_blocking_update_sky(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Center must not call ``_update_sky`` synchronously inside the dispatch batch."""
    _harness, review, _loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review._coord = CAS_A
    review._heatmap_coord = CAS_A

    widget = MagicMock()
    widget.image_shape = (512, 512)
    widget.overlay_view_lock = True
    widget.image_revision = 0
    review._sky_widget = widget

    scheduled_overlay: list[tuple] = []
    update_sky_calls: list[tuple] = []

    monkeypatch.setattr(
        review,
        "_resolve_active_coordinate",
        lambda: (CAS_A, "Cas A"),
    )
    monkeypatch.setattr(
        review,
        "_schedule_overlay_slice_load",
        lambda *args, **kwargs: scheduled_overlay.append((args, kwargs)),
    )
    monkeypatch.setattr(
        review,
        "_update_sky",
        lambda *args, **kwargs: update_sky_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(review, "_reset_heatmap_to_zeros", lambda: None)
    monkeypatch.setattr(review, "_sync_fit_overlay_button", lambda: None)
    monkeypatch.setattr(review, "_log_overlay_diagnostics", lambda *a, **k: None)
    monkeypatch.setattr(review, "_force_send_sky_widget_state", lambda *a, **k: None)

    review._on_slew_impl()

    assert not update_sky_calls
    assert len(scheduled_overlay) == 1
    _args, kwargs = scheduled_overlay[0]
    assert kwargs.get("center_on_target") is True
    assert kwargs.get("center") == CAS_A
    widget.goto.assert_called_once()
    widget.set_crosshair.assert_called_once_with(CAS_A)


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_center_before_first_generate_skips_heatmap_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sky-click → Center before any Generate must not republish the zeros grid."""
    _harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review._heatmap_coord = None
    review._heatmap_grid_ready = True

    clicked = SkyCoord(ra=180.0 * u.deg, dec=37.0 * u.deg, frame="icrs")

    widget = MagicMock()
    widget.image_shape = (0, 0)
    widget.overlay_view_lock = True
    widget.image_revision = 0
    review._sky_widget = widget

    reset_calls: list[bool] = []

    def _track_reset() -> None:
        reset_calls.append(True)

    monkeypatch.setattr(review, "_reset_heatmap_to_zeros", _track_reset)
    monkeypatch.setattr(
        review,
        "_resolve_active_coordinate",
        lambda: (clicked, "180.0000, 37.0000"),
    )
    monkeypatch.setattr(review, "_sync_fit_overlay_button", lambda: None)
    monkeypatch.setattr(review, "_log_overlay_diagnostics", lambda *a, **k: None)
    monkeypatch.setattr(review, "_force_send_sky_widget_state", lambda *a, **k: None)

    review._dispatch(review._on_slew_impl)
    _flush_jupyter_io(loop)

    assert reset_calls == []


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_center_with_overlay_skips_heatmap_reset_for_mismatched_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlay present: Center on a new field must not wipe the computed heatmap."""
    _harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review._heatmap_coord = CAS_A

    widget = MagicMock()
    widget.image_shape = (512, 512)
    widget.overlay_view_lock = True
    widget.image_revision = 0
    review._sky_widget = widget

    reset_calls: list[bool] = []

    monkeypatch.setattr(
        review,
        "_reset_heatmap_to_zeros",
        lambda: reset_calls.append(True),
    )
    monkeypatch.setattr(
        review,
        "_resolve_active_coordinate",
        lambda: (CYG_A, "Cyg A"),
    )
    monkeypatch.setattr(review, "_schedule_overlay_slice_load", lambda *a, **k: None)
    monkeypatch.setattr(review, "_sync_fit_overlay_button", lambda: None)
    monkeypatch.setattr(review, "_log_overlay_diagnostics", lambda *a, **k: None)
    monkeypatch.setattr(review, "_force_send_sky_widget_state", lambda *a, **k: None)

    review._dispatch(review._on_slew_impl)
    _flush_jupyter_io(loop)

    assert reset_calls == []


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_center_without_overlay_resets_mismatched_heatmap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No overlay: Center on a new field discards a stale computed heatmap."""
    _harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review._heatmap_coord = CYG_A

    widget = MagicMock()
    widget.image_shape = (0, 0)
    review._sky_widget = widget

    reset_calls: list[bool] = []

    monkeypatch.setattr(
        review,
        "_reset_heatmap_to_zeros",
        lambda: reset_calls.append(True),
    )
    monkeypatch.setattr(
        review,
        "_resolve_active_coordinate",
        lambda: (CAS_A, "Cas A"),
    )
    monkeypatch.setattr(review, "_sync_fit_overlay_button", lambda: None)
    monkeypatch.setattr(review, "_log_overlay_diagnostics", lambda *a, **k: None)
    monkeypatch.setattr(review, "_force_send_sky_widget_state", lambda *a, **k: None)

    review._dispatch(review._on_slew_impl)
    _flush_jupyter_io(loop)

    assert reset_calls == [True]


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_post_generate_overlay_skipped_when_heatmap_tap_supersedes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Post-generate overlay must not run after the user taps a newer heatmap cell."""
    harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review._coord = CAS_A
    widget = MagicMock()
    widget.overlay_view_lock = True
    widget.crval = (350.85, 58.815)
    widget.view_ra = 350.85
    widget.view_dec = 58.815
    widget.view_center_skycoord.return_value = CAS_A
    widget.image_revision = 0
    review._sky_widget = widget

    update_calls: list[tuple[int, int]] = []

    def _record_update_slice(time_idx, freq_idx, **kwargs) -> None:
        update_calls.append((int(time_idx), int(freq_idx)))
        widget.image_revision = int(widget.image_revision) + 1

    widget.update_slice.side_effect = _record_update_slice

    src = review._current_source
    assert src is not None
    payload = HeatmapLoad(
        values=np.full((6, 4), 42.0),
        patch_fit_result=None,
        patch_stat_result=None,
    )
    review._apply_heatmap(src, payload)
    _flush_jupyter_io(loop)

    review._schedule_overlay_slice_load(5, 0, preserve_view=True)
    _flush_jupyter_io(loop)

    assert (5, 0) in update_calls
    assert (0, 0) not in update_calls or update_calls[-1] == (5, 0)


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_late_progress_dispatch_does_not_clobber_published_heatmap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a progress dispatch queued before publish must not reset the figure."""
    harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review._heatmap_job_id = 1
    src = review._current_source
    assert src is not None
    payload = HeatmapLoad(
        values=np.full((6, 4), 55.0),
        patch_fit_result=None,
        patch_stat_result=None,
    )

    review._ui.dispatch(
        lambda: review._finish_heatmap(
            src, payload, None, job_id=1, started_at=time.perf_counter()
        )
    )
    # Simulate a progress-log dispatch still queued from compute.
    review._ui.dispatch(lambda: review._log("late progress line"))
    _flush_jupyter_io(loop)

    assert _heatmap_values_max(review) == pytest.approx(55.0)
    _assert_heatmap_bokeh_model_live(harness, review)
    assert harness.bokeh_model(review._heatmap_pane, review._layout).title.text.startswith(
        "Cas A"
    )


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_heatmap_progress_logs_do_not_dispatch_panel_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: progress logging must not run full dispatch (stale heatmap push)."""
    harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)

    dispatch_calls: list[str] = []
    real_dispatch = review._ui.dispatch

    def _track_dispatch(callback, **_kwargs) -> None:
        dispatch_calls.append("dispatch")
        real_dispatch(callback)

    monkeypatch.setattr(review._ui, "dispatch", _track_dispatch)
    scheduled: list[str] = []
    monkeypatch.setattr(
        sra,
        "_schedule_ipython_main",
        lambda fn: scheduled.append("schedule") or loop.add_callback(fn),
    )

    progress = review._heatmap_progress_callback()
    progress("extract", 1, 10, "tracking pixels")

    assert dispatch_calls == []
    assert scheduled == ["schedule"]
    _flush_jupyter_io(loop)
    assert "tracking pixels" in review.log_text
    assert "(1/10, 10%)" in review.log_text


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_heatmap_progress_throttles_pixel_track_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pixel track logs start and finish only — intermediate batches are omitted."""
    harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    monkeypatch.setattr(
        sra,
        "_schedule_ipython_main",
        lambda fn: loop.add_callback(fn),
    )

    progress = review._heatmap_progress_callback()
    progress("track", 0, 120, "Mapping RA/Dec to image pixels (per-time WCS)")
    progress("track", 60, 120, "Mapping RA/Dec to image pixels (per-time WCS)")
    progress("track", 120, 120, "Mapping RA/Dec to image pixels (per-time WCS)")
    _flush_jupyter_io(loop)

    lines = [line for line in review.log_text.splitlines() if "Pixel track" in line]
    assert len(lines) == 2
    assert "(0/120" in lines[0]
    assert "(120/120" in lines[1]


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_overlay_fit_begin_logs_schedule_not_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fit overlay start logs via ipywidgets schedule, not Panel dispatch batches."""
    import threading
    import time

    from ovro_lwa_portal.accessor import PatchFitCellResult, format_radec_sexagesimal

    harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review._coord = CAS_A

    dispatch_calls: list[str] = []
    real_dispatch = review._ui.dispatch

    def _track_dispatch(callback, **_kwargs) -> None:
        dispatch_calls.append("dispatch")
        real_dispatch(callback)

    monkeypatch.setattr(review._ui, "dispatch", _track_dispatch)
    scheduled: list[str] = []
    monkeypatch.setattr(
        sra,
        "_schedule_ipython_main",
        lambda fn: scheduled.append("schedule") or loop.add_callback(fn),
    )

    ra_s, dec_s = format_radec_sexagesimal(350.85, 58.815)
    worker_done = threading.Event()

    def _fake_fit(*_args, **_kwargs) -> PatchFitCellResult:
        worker_done.wait(timeout=2.0)
        return PatchFitCellResult(
            time_idx=0,
            frequency_idx=0,
            fit_accepted=True,
            reduced_chi_squared=1.0,
            peak=10.0,
            peak_ra_deg=350.85,
            peak_dec_deg=58.815,
            peak_ra=ra_s,
            peak_dec=dec_s,
            x_offset_pixels=0.0,
            y_offset_pixels=0.0,
            peak_offset_pixels=0.0,
            center_flux=1.0,
            patch_max=11.0,
            background=0.0,
            widthx=3.0,
            widthy=3.0,
            scale=5.0,
            max_reduced_chi_squared=10.0,
            allow_position_offset=True,
            patch_radius_pixels=4,
        )

    monkeypatch.setattr(sra, "compute_overlay_patch_fit", _fake_fit)
    review._fit_overlay_button.disabled = False
    review._load_overlay_fit()
    _flush_jupyter_io(loop)

    assert scheduled
    assert dispatch_calls == []
    assert "Fitting overlay patch" in review.log_text

    worker_done.set()
    deadline = time.monotonic() + 2.0
    while len(scheduled) < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    _flush_jupyter_io(loop)
    assert dispatch_calls
    assert "Fit overlay finished" in review.log_text
