"""Integration tests: heatmap publish + spinner + coordinate field together."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import astropy.units as u
import numpy as np
import pytest
import xarray as xr
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

pn = pytest.importorskip("panel")
pytest.importorskip("astrowidget")

from ovro_lwa_portal import fov_overlay, open_dataset
from ovro_lwa_portal.viz import pipeline_qa_app as pqa
from ovro_lwa_portal.viz import source_review_app as sra
from ovro_lwa_portal.viz.source_review import DatasetLoad, run_dataset_load
from ovro_lwa_portal.viz.source_review_app import SourceReview, SourceReviewConfig
from ovro_lwa_portal.viz.source_review_data import HeatmapLoad, build_source_from_coordinate
from ovro_lwa_portal.zarr_direct import DirectZarrArray
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


@pytest.mark.parametrize("method", ["dynamic_spectrum", "patch_max"])
def test_cached_heatmap_request_invalidates_older_job(
    tmp_path: Path,
    method: str,
) -> None:
    """A queued cache hit must supersede in-flight work and ignore stale paints."""
    _, review = _mount_review(tmp_path)
    _seed_dataset(review)
    review.heatmap_method = method
    review.coordinate_string = "Cas A"
    review._cache[("Cas A", method, str(review.stokes))] = HeatmapLoad(
        values=np.full((6, 4), 42.0),
        patch_fit_result=None,
        patch_stat_result=None,
    )
    review._heatmap_job_id = 1
    scheduled: list[Callable[[], None]] = []
    review._schedule_main = lambda callback: scheduled.append(callback)  # type: ignore[method-assign]
    review._dispatch = lambda callback: scheduled.append(callback)  # type: ignore[method-assign]

    review._load_heatmap()

    assert review._heatmap_job_id == 2
    assert len(scheduled) == 1
    review._heatmap_job_id += 1
    scheduled[0]()

    assert review._heatmap_values is None


def test_finish_heatmap_ignores_stale_deferred_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completion queued for publish cannot overwrite a newer heatmap request."""
    _, review = _mount_review(tmp_path)
    _seed_dataset(review)
    review._heatmap_job_id = 1
    src = review._current_source
    assert src is not None
    payload = HeatmapLoad(
        values=np.full((6, 4), 43.0),
        patch_fit_result=None,
        patch_stat_result=None,
    )
    deferred: list[Callable[[], None]] = []
    monkeypatch.setattr(sra, "_run_after_hold", lambda callback: deferred.append(callback))

    review._finish_heatmap(src, payload, None, job_id=1, started_at=time.perf_counter())

    assert len(deferred) == 1
    review._heatmap_job_id += 1
    deferred[0]()

    assert review._heatmap_values is None


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


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_notebook_sky_embed_includes_widget_mounted_after_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first Panel snapshot is the placeholder; mounting must re-embed."""
    from ipywidgets import HTML

    harness, review, _loop = _mount_review_jupyter(tmp_path, monkeypatch)
    assert review._sky_pane._models
    review._sky_container.children = [HTML("SKY-MOUNTED")]
    review._refresh_notebook_sky_embed()

    _ref, (model, _parent) = next(iter(review._sky_pane._models.items()))
    blob = str(model.bundle)
    assert "SKY-MOUNTED" in blob
    assert "Sky view loads after the Zarr store opens." not in blob
    sky_model = harness.bokeh_model(review._sky_pane, review._layout)
    assert sky_model is model
    parent = sky_model.document
    assert parent is not None
    n_models = len(parent.roots)
    review._sky_container.children = [HTML("SKY-MOVED")]
    review._refresh_notebook_sky_embed()
    _ref, (again, _parent) = next(iter(review._sky_pane._models.items()))
    assert again is model
    assert "SKY-MOVED" in str(model.bundle)
    assert len(parent.roots) == n_models


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
    cutout_calls: list[int] = []
    real_schedule = review._ui.schedule

    def _track_schedule(callback) -> None:
        scheduled.append("schedule")
        real_schedule(callback)

    def _record_cutout(*_args, **_kwargs) -> bool:
        cutout_calls.append(1)
        return False

    monkeypatch.setattr(review._ui, "schedule", _track_schedule)
    monkeypatch.setattr(review._ui, "defer_dispatch", lambda _cb: scheduled.append("defer_dispatch"))
    monkeypatch.setattr(review, "_load_overlay_cutout", _record_cutout)

    review._on_overlay_toggle_impl(True)
    assert review.loading is True
    assert _spinner_spinning(review)
    _flush_jupyter_io(loop)

    assert review.loading is False
    assert not _spinner_spinning(review)
    assert "schedule" in scheduled
    assert "defer_dispatch" not in scheduled
    assert cutout_calls == [1]

    scheduled.clear()
    review._on_heatmap_tap(2, 1)
    assert review.loading is True
    _flush_jupyter_io(loop)

    assert review.loading is False
    assert scheduled == ["schedule"]
    assert cutout_calls == [1, 1]


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

    def _record_cutout(time_idx, freq_idx, **kwargs) -> bool:
        update_calls.append((int(time_idx), int(freq_idx)))
        return False

    monkeypatch.setattr(review, "_load_overlay_cutout", _record_cutout)

    review._schedule_overlay_slice_load(0, 0, preserve_view=True)
    review._schedule_overlay_slice_load(5, 0, preserve_view=True)
    _flush_jupyter_io(loop)

    assert update_calls == [(5, 0)]
    assert review.loading is False
    assert not _spinner_spinning(review)


class _CutoutSky:
    """SkyWidget stand-in that records shader pushes and does not invent view writes."""

    def __init__(self) -> None:
        self.view_ra = 180.0
        self.view_dec = 40.0
        self.view_fov = 8.0
        self.opacity = 1.0
        self.crval = (180.0, 40.0)
        self.image_shape = (0, 0)
        self.image_data = b""
        self.image_revision = 0
        self.overlay_view_lock = False
        self.view_gesture_revision = 0
        self.pushes: list[np.ndarray] = []
        self.update_slice_calls: list[dict[str, object]] = []

    def view_center_skycoord(self) -> SkyCoord:
        return SkyCoord(self.view_ra, self.view_dec, unit="deg", frame="icrs")

    def update_slice(self, *_args: object, **kwargs: object) -> None:
        self.update_slice_calls.append(dict(kwargs))

    def push_shader_frame(self, data: np.ndarray, wcs: object, **_kwargs: object) -> None:
        array = np.asarray(data)
        self.pushes.append(array)
        self.image_data = array.tobytes()
        self.image_shape = tuple(int(x) for x in array.shape)
        self.image_revision += 1

    def send_state(self) -> None:
        return None

    def unobserve(self, *_args: object, **_kwargs: object) -> None:
        return None

    def observe(self, *_args: object, **_kwargs: object) -> None:
        return None


def _wait_until(loop: QueuedIOLoop, predicate, *, timeout_s: float = 5.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        _drain_io_loop(loop, timeout_s=0.05)
        if predicate():
            return
        time.sleep(0.01)
    msg = "condition was not met"
    raise AssertionError(msg)


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_heatmap_tap_fades_cutout_without_moving_the_view(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A heatmap tap reloads (time, freq) at the current view and fades first."""
    from astropy.wcs import WCS

    from ovro_lwa_portal import fov_overlay

    _harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    sky = np.ones((6, 4, 8, 8), dtype=np.float32)
    sky[1] = np.nan
    review._dataset["SKY"] = (("time", "frequency", "l", "m"), sky)
    review._coord = CAS_A
    widget = _CutoutSky()
    review._sky_widget = widget

    def _wcs(*_args: object, **_kwargs: object) -> WCS:
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        wcs.wcs.crval = [180.0, 40.0]
        wcs.wcs.cdelt = [-1.0, 1.0]
        wcs.wcs.crpix = [4.5, 4.5]
        wcs.wcs.cunit = ["deg", "deg"]
        return wcs

    seen: list[float] = []
    real_extract = fov_overlay.extract_fov_cutout

    def _extract(*args: object, **kwargs: object) -> np.ndarray:
        seen.append(widget.opacity)
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(fov_overlay, "celestial_wcs_for_time", _wcs)
    monkeypatch.setattr(
        fov_overlay,
        "_project_native_sin",
        lambda data, cut_wcs: (np.asarray(data, dtype=np.float32), cut_wcs, 0.2),
    )
    monkeypatch.setattr(fov_overlay, "extract_fov_cutout", _extract)

    review._on_heatmap_tap(0, 0)
    _wait_until(loop, lambda: widget.opacity == 1.0 and len(widget.pushes) == 1)

    assert seen[0] == pytest.approx(0.35)
    assert widget.update_slice_calls == []
    assert (widget.view_ra, widget.view_dec, widget.view_fov) == (180.0, 40.0, 8.0)
    assert review.loading is False

    seen.clear()
    review._on_heatmap_tap(1, 0)
    _wait_until(loop, lambda: widget.opacity == 1.0 and review.loading is False)

    assert seen[0] == pytest.approx(0.35)
    assert len(widget.pushes) == 1
    assert widget.update_slice_calls == []
    assert (widget.view_ra, widget.view_dec, widget.view_fov) == (180.0, 40.0, 8.0)


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_large_cutout_push_does_not_call_update_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 782^2 Source Review overlay must not call ``update_slice(center=)``."""
    from astropy.wcs import WCS

    from ovro_lwa_portal import fov_overlay

    _harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review._coord = CAS_A
    widget = _CutoutSky()
    review._sky_widget = widget

    def _wcs(*_args: object, **_kwargs: object) -> WCS:
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        wcs.wcs.crval = [180.0, 40.0]
        wcs.wcs.cdelt = [-0.037, 0.037]
        wcs.wcs.crpix = [391.5, 391.5]
        wcs.wcs.cunit = ["deg", "deg"]
        return wcs

    tile = np.ones((782, 782), dtype=np.float32)
    monkeypatch.setattr(fov_overlay, "celestial_wcs_for_time", _wcs)
    monkeypatch.setattr(
        fov_overlay,
        "extract_fov_cutout",
        lambda *_a, **_k: tile,
    )
    monkeypatch.setattr(
        fov_overlay,
        "_project_native_sin",
        lambda data, cut_wcs: (np.asarray(data, dtype=np.float32), cut_wcs, 0.2),
    )

    review._on_heatmap_tap(0, 0)
    _wait_until(loop, lambda: len(widget.pushes) == 1 and widget.opacity == 1.0)

    assert widget.pushes[0].shape == (782, 782)
    assert widget.update_slice_calls == []
    assert (widget.view_ra, widget.view_dec, widget.view_fov) == (180.0, 40.0, 8.0)


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
def test_review_catalog_position_skips_heatmap_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A catalog click must not republish the zeros grid before the new spectrum."""
    _harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review._heatmap_coord = CYG_A
    review._heatmap_values = np.ones((6, 4))

    widget = MagicMock()
    widget.image_shape = (0, 0)
    review._sky_widget = widget

    reset_calls: list[bool] = []
    load_calls: list[bool] = []
    overlay_calls: list[dict[str, object]] = []

    monkeypatch.setattr(review, "_reset_heatmap_to_zeros", lambda: reset_calls.append(True))
    monkeypatch.setattr(review, "_load_heatmap", lambda: load_calls.append(True))
    monkeypatch.setattr(
        review,
        "_schedule_overlay_slice_load",
        lambda *args, **kwargs: overlay_calls.append(kwargs),
    )
    monkeypatch.setattr(review, "_sync_fit_overlay_button", lambda: None)
    monkeypatch.setattr(review, "_log_overlay_diagnostics", lambda *a, **k: None)

    review.review_catalog_position("280.46630 -1.96395", name="J1841-0157", stokes="I")
    _flush_jupyter_io(loop)

    assert reset_calls == []
    assert load_calls == [True]
    assert len(overlay_calls) == 1
    assert overlay_calls[0].get("center_on_target") is True
    widget.goto.assert_called_once()
    widget.set_crosshair.assert_called_once()
    assert review._current_source is not None
    assert review._current_source["name"] == "J1841-0157"
    assert review._heatmap_coord == CYG_A


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_review_catalog_position_switches_stokes_without_zeros_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stokes on a catalog click must not clear the heatmap before Generate."""
    _harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review.stokes = "I"
    review._heatmap_coord = CYG_A
    review._heatmap_values = np.ones((6, 4))

    widget = MagicMock()
    widget.image_shape = (0, 0)
    review._sky_widget = widget

    grid_calls: list[bool] = []

    def _grid(*_args: object, **_kwargs: object) -> None:
        grid_calls.append(True)

    monkeypatch.setattr(review, "_ensure_heatmap_grid", _grid)
    monkeypatch.setattr(review, "_load_heatmap", lambda: None)
    monkeypatch.setattr(review, "_schedule_overlay_slice_load", lambda *a, **k: None)
    monkeypatch.setattr(review, "_sync_fit_overlay_button", lambda: None)

    review.review_catalog_position("148.28526 +7.92746", name="J0953+0755", stokes="V")
    _flush_jupyter_io(loop)

    assert review.stokes == "V"
    assert grid_calls == []
    assert review._heatmap_coord == CYG_A
    assert review._current_source is not None
    assert review._current_source["name"] == "J0953+0755"


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_catalog_load_fades_the_previous_spectrum_until_new_values_paint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load dims the current spectrum; the next paint restores it."""
    _harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review._heatmap_values = np.ones((6, 4))
    review._refresh_heatmap_figure(review._heatmap_values)
    glyph = review._heatmap_bokeh_handles.image_renderer.glyph
    assert glyph.global_alpha == 1.0

    monkeypatch.setattr(review, "_load_heatmap", lambda: None)
    monkeypatch.setattr(review, "_schedule_overlay_slice_load", lambda *a, **k: None)
    monkeypatch.setattr(review, "_sync_fit_overlay_button", lambda: None)
    widget = MagicMock()
    widget.image_shape = (8, 8)
    review._sky_widget = widget

    review.review_catalog_position("280.46630 -1.96395", name="J1841-0157", stokes="I")
    _flush_jupyter_io(loop)
    assert glyph.global_alpha == 0.35

    review._update_heatmap_figure_shell(
        review._heatmap_bokeh_handles,
        np.full((6, 4), 2.0),
    )
    assert glyph.global_alpha == 1.0


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_review_catalog_position_keeps_detection_slice_for_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A catalog time/frequency survives the post-heatmap overlay choice."""
    _harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review._lst_hours = np.arange(6, dtype=float)
    review._freq_mhz = np.array([18.0, 32.0, 46.0, 82.0])
    widget = MagicMock()
    widget.image_shape = (8, 8)
    review._sky_widget = widget
    monkeypatch.setattr(review, "_load_heatmap", lambda: None)
    monkeypatch.setattr(review, "_schedule_overlay_slice_load", lambda *a, **k: None)
    monkeypatch.setattr(review, "_sync_fit_overlay_button", lambda: None)

    review.review_catalog_position(
        "280.46630 -1.96395",
        name="J1841-0157",
        stokes="I",
        time_idx=4,
        freq_idx=1,
    )
    _flush_jupyter_io(loop)

    assert "32.0 MHz" in review.status
    assert "01-01 04h" in review.status
    assert review._pending_overlay_slice == (4, 1)
    chosen = review._consume_pending_overlay_slice(np.zeros((6, 4)))
    assert chosen == (4, 1)
    assert review._pending_overlay_slice is None
    assert review._consume_pending_overlay_slice(np.zeros((6, 4))) == review._default_slice(
        np.zeros((6, 4))
    )


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_dynamic_spectrum_request_does_not_leak_detection_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _harness, review = _mount_review(tmp_path)
    _seed_dataset(review)
    review._pending_overlay_slice = (4, 1)
    monkeypatch.setattr(review, "_start_dynamic_spectrum", lambda *_args: None)

    review._load_heatmap()

    assert review._pending_overlay_slice is None
    assert review._consume_pending_overlay_slice(np.zeros((6, 4))) == review._default_slice(
        np.zeros((6, 4))
    )


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_heatmap_request_owns_its_detection_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _harness, review = _mount_review(tmp_path)
    _seed_dataset(review)
    review.heatmap_method = "patch_max"
    review._pending_overlay_slice = (4, 1)
    review._cache[("", "patch_max", "I")] = HeatmapLoad(values=np.ones((6, 4)))
    monkeypatch.setattr(review, "_dispatch", lambda callback: callback())

    review._load_heatmap()
    assert review._pending_overlay_slice is None

    assert (review._time_idx, review._freq_idx) == (4, 1)


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_generate_after_center_rebinds_desynced_heatmap_pane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Center layout pushes can desync pane.object from handles; Generate must recover."""
    from bokeh.plotting import figure as bokeh_figure

    from ovro_lwa_portal.viz import source_review_app as sra

    harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review._ui.defer_dispatch(review._ensure_heatmap_grid)
    _flush_jupyter_io(loop)

    assert review._heatmap_bokeh_handles is not None
    review._heatmap_pane.object = bokeh_figure(width=100, height=100)

    clicked = SkyCoord(ra=187.2779 * u.deg, dec=2.0524 * u.deg, frame="icrs")
    widget = MagicMock()
    widget.image_shape = (0, 0)
    widget.overlay_view_lock = True
    widget.image_revision = 0
    review._sky_widget = widget

    monkeypatch.setattr(
        review,
        "_resolve_active_coordinate",
        lambda: (clicked, "3C 1"),
    )
    monkeypatch.setattr(review, "_schedule_overlay_slice_load", lambda *a, **k: None)
    monkeypatch.setattr(review, "_sync_fit_overlay_button", lambda: None)
    monkeypatch.setattr(review, "_log_overlay_diagnostics", lambda *a, **k: None)
    monkeypatch.setattr(review, "_force_send_sky_widget_state", lambda *a, **k: None)

    review._dispatch(review._on_slew_impl)
    _flush_jupyter_io(loop)

    can_mutate, rebound = review._sync_heatmap_pane_to_handles()
    assert can_mutate is True
    assert rebound is True
    assert review._heatmap_pane.object is review._heatmap_bokeh_handles.plot

    values = np.full((6, 4), 91.0)

    def _fast_compute(*_args, **_kwargs) -> HeatmapLoad:
        return HeatmapLoad(values=values, patch_fit_result=None, patch_stat_result=None)

    monkeypatch.setattr(sra, "compute_source_heatmap", _fast_compute)
    review._on_generate_heatmap()
    _flush_jupyter_io(loop)
    while loop.callbacks:
        loop.flush()

    assert _heatmap_values_max(review) == pytest.approx(91.0)
    _assert_heatmap_bokeh_model_live(harness, review)


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_sky_click_center_generate_publishes_heatmap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: sky click → Center → Generate must reach the browser (worker finish path)."""
    from ovro_lwa_portal.viz import source_review_app as sra

    harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review._ui.defer_dispatch(review._ensure_heatmap_grid)
    _flush_jupyter_io(loop)

    clicked = SkyCoord(ra=187.2779 * u.deg, dec=2.0524 * u.deg, frame="icrs")
    widget = MagicMock()
    widget.image_shape = (0, 0)
    widget.overlay_view_lock = True
    widget.image_revision = 0
    widget.clicked_coord = (float(clicked.ra.deg), float(clicked.dec.deg))
    review._sky_widget = widget

    review._on_sky_widget_click(None)
    _flush_jupyter_io(loop)

    monkeypatch.setattr(
        review,
        "_resolve_active_coordinate",
        lambda: (clicked, "3C 1"),
    )
    monkeypatch.setattr(review, "_schedule_overlay_slice_load", lambda *a, **k: None)
    monkeypatch.setattr(review, "_sync_fit_overlay_button", lambda: None)
    monkeypatch.setattr(review, "_log_overlay_diagnostics", lambda *a, **k: None)
    monkeypatch.setattr(review, "_force_send_sky_widget_state", lambda *a, **k: None)

    review._dispatch(review._on_slew_impl)
    _flush_jupyter_io(loop)

    values = np.full((6, 4), 88.0)

    def _fast_compute(*_args, **_kwargs) -> HeatmapLoad:
        return HeatmapLoad(values=values, patch_fit_result=None, patch_stat_result=None)

    monkeypatch.setattr(sra, "compute_source_heatmap", _fast_compute)
    review._on_generate_heatmap()
    _flush_jupyter_io(loop)
    while loop.callbacks:
        loop.flush()

    assert _heatmap_values_max(review) == pytest.approx(88.0)
    _assert_heatmap_bokeh_model_live(harness, review)
    assert "3C 1" in harness.bokeh_model(review._heatmap_pane, review._layout).title.text


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

    def _record_cutout(time_idx, freq_idx, **kwargs) -> bool:
        update_calls.append((int(time_idx), int(freq_idx)))
        return False

    monkeypatch.setattr(review, "_load_overlay_cutout", _record_cutout)

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


def _control_names(review: SourceReview) -> list[str]:
    row = review._layout.objects[0]
    return [str(getattr(child, "name", "")) for child in row.objects]


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_control_row_names_and_order_are_unchanged(tmp_path: Path) -> None:
    """Phase 3 must not add a heatmap button or reorder the existing row."""
    _harness, review = _mount_review(tmp_path)
    assert _control_names(review)[:7] == [
        "Coordinate",
        "Center",
        "Generate heatmap",
        "Stokes",
        "Overlay: on",
        "Fit overlay",
        "Heatmap method",
    ]


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_dynamic_spectrum_generate_paints_click_without_moving_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate heatmap fills the click coordinate and leaves the overlay cell put."""
    mutation_calls: list[bool] = []
    publish_calls: list[bool] = []
    real_mutation = pqa.push_bokeh_pane_mutation_to_notebook
    real_publish = pqa.publish_bokeh_pane_to_notebook

    def _track_mutation(pane: object, *root_views: object, force_push: bool = False) -> None:
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
    sky = np.zeros((6, 4, 8, 8), dtype=np.float32)
    sky[:, :, 2, 3] = 12.5
    review._dataset["SKY"] = (("time", "frequency", "l", "m"), sky)
    review._time_idx = 4
    review._freq_idx = 2
    review._ui.defer_dispatch(review._ensure_heatmap_grid)
    _flush_jupyter_io(loop)

    widget = MagicMock()
    widget.clicked_coord = (10.25, -20.5)
    widget.view_ra = 180.0
    widget.view_dec = 40.0
    widget.view_fov = 8.0
    review._sky_widget = widget
    review.heatmap_method = "dynamic_spectrum"
    _flush_jupyter_io(loop)
    mutation_calls.clear()
    publish_calls.clear()

    tracked: list[tuple[float, float]] = []

    def _track(ra: float, dec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tracked.append((float(ra), float(dec)))
        return np.full(6, 2), np.full(6, 3), np.ones(6, dtype=bool)

    monkeypatch.setattr(review, "_pixel_track", _track)
    overlay_calls: list[str] = []
    monkeypatch.setattr(
        review,
        "_schedule_overlay_slice_load",
        lambda *_a, **_k: overlay_calls.append("schedule"),
    )
    monkeypatch.setattr(
        review,
        "_update_sky",
        lambda *_a, **_k: overlay_calls.append("update") or False,
    )
    compute_calls: list[str] = []

    def _fail_compute(*_args: object, **_kwargs: object) -> HeatmapLoad:
        compute_calls.append("compute")
        raise AssertionError("dynamic_spectrum must not use compute_source_heatmap")

    monkeypatch.setattr(sra, "compute_source_heatmap", _fail_compute)

    review._on_sky_widget_click(None)
    _flush_jupyter_io(loop)
    review._on_generate_heatmap()
    _wait_until(
        loop,
        lambda: review._heatmap_values is not None
        and float(np.nanmax(review._heatmap_values)) == pytest.approx(12.5)
        and review.loading is False,
    )

    assert tracked
    assert tracked[0][0] == pytest.approx(10.25)
    assert tracked[0][1] == pytest.approx(-20.5)
    assert overlay_calls == []
    assert compute_calls == []
    assert (review._time_idx, review._freq_idx) == (4, 2)
    assert (widget.view_ra, widget.view_dec) == (180.0, 40.0)
    assert mutation_calls
    assert publish_calls == []
    _assert_heatmap_bokeh_model_live(harness, review)


def _zarr_backed_dataset(review: SourceReview, tmp_path: Path) -> xr.Dataset:
    """Reopen the seeded dataset from a 5-D Zarr store so reads can go direct."""
    seeded = review._dataset
    assert seeded is not None
    sky = np.zeros((6, 4, 1, 8, 8), dtype=np.float32)
    sky[:, :, 0, 2, 3] = 12.5
    ds = xr.Dataset(
        {"SKY": (("time", "frequency", "polarization", "l", "m"), sky)},
        coords={**{k: seeded.coords[k] for k in ("time", "frequency", "l", "m")}, "polarization": [1]},
    )
    path = tmp_path / "direct.zarr"
    ds.to_zarr(path, mode="w", consolidated=True, encoding={"SKY": {"chunks": (2, 4, 1, 4, 4)}})
    opened = open_dataset(path, chunks={}, validate=False)
    review._dataset = opened
    return opened


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
@pytest.mark.filterwarnings("ignore:Zarr store .* has small on-disk l/m chunks:UserWarning")
def test_zarr_backed_store_reads_spectrum_and_cutout_direct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate and heatmap taps read on-disk chunks directly, not through Dask."""
    _harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    _zarr_backed_dataset(review, tmp_path)
    review._ui.defer_dispatch(review._ensure_heatmap_grid)
    _flush_jupyter_io(loop)

    direct_reads: list[str] = []
    real_column = DirectZarrArray.column
    real_getitem = DirectZarrArray.__getitem__

    def _column(self: DirectZarrArray, *args: object, **kwargs: object) -> np.ndarray:
        direct_reads.append("column")
        return real_column(self, *args, **kwargs)

    def _getitem(self: DirectZarrArray, key: tuple) -> np.ndarray:
        direct_reads.append("plane")
        return real_getitem(self, key)

    monkeypatch.setattr(DirectZarrArray, "column", _column)
    monkeypatch.setattr(DirectZarrArray, "__getitem__", _getitem)

    widget = _CutoutSky()
    widget.clicked_coord = (10.25, -20.5)
    review._sky_widget = widget
    review.heatmap_method = "dynamic_spectrum"
    monkeypatch.setattr(
        review, "_pixel_track", lambda *_a: (np.full(6, 2), np.full(6, 3), np.ones(6, dtype=bool))
    )
    review._on_sky_widget_click(None)
    _flush_jupyter_io(loop)
    review._on_generate_heatmap()
    _wait_until(
        loop,
        lambda: review._heatmap_values is not None
        and float(np.nanmax(review._heatmap_values)) == pytest.approx(12.5)
        and review.loading is False,
    )
    assert "column" in direct_reads
    assert review._direct_sky_array() is not None

    def _wcs(*_args: object, **_kwargs: object) -> WCS:
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---SIN", "DEC--SIN"]
        wcs.wcs.crval = [180.0, 40.0]
        wcs.wcs.cdelt = [-1.0, 1.0]
        wcs.wcs.crpix = [4.5, 4.5]
        wcs.wcs.cunit = ["deg", "deg"]
        return wcs

    monkeypatch.setattr(fov_overlay, "celestial_wcs_for_time", _wcs)
    monkeypatch.setattr(
        fov_overlay,
        "_project_native_sin",
        lambda data, cut_wcs: (np.asarray(data, dtype=np.float32), cut_wcs, 0.2),
    )
    review._coord = CAS_A
    review._on_heatmap_tap(0, 0)
    _wait_until(loop, lambda: len(widget.pushes) == 1 and widget.opacity == 1.0)

    session = review._fov_session
    assert session is not None
    assert isinstance(session.sky_arr, DirectZarrArray)
    assert session.tile_executor is review._read_pool()
    assert session.tile_executor is not session.executor
    assert "plane" in direct_reads
    assert float(np.nanmax(widget.pushes[0])) == pytest.approx(12.5)


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_generate_with_empty_coordinate_does_not_fill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _harness, review, loop = _mount_review_jupyter(tmp_path, monkeypatch, layout_only=True)
    _seed_dataset(review)
    review.coordinate_string = ""
    review._coord_input.value = ""
    review._coord_input.value_input = ""
    review.heatmap_method = "dynamic_spectrum"
    tracked: list[str] = []
    monkeypatch.setattr(review, "_pixel_track", lambda *_a, **_k: tracked.append("track"))
    review._on_generate_heatmap()
    _flush_jupyter_io(loop)
    assert tracked == []
    assert "empty" in review.log_text.lower()
    assert review._heatmap_values is None
