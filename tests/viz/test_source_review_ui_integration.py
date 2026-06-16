"""Integration tests: heatmap publish + spinner + coordinate field together."""

from __future__ import annotations

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


def _spinner_spinning(harness: PanelUITestHarness, review: SourceReview) -> bool:
    model = harness.bokeh_model(review._spinner, review._layout)
    return "spin" in model.css_classes


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
    assert _spinner_spinning(harness, review)

    values = np.full((6, 4), 42.0)
    payload = HeatmapLoad(values=values, patch_fit_result=None, patch_stat_result=None)
    harness.run_ui(
        harness.session(review._layout),
        lambda: review._finish_heatmap(src, payload, None, job_id=1, started_at=time.perf_counter()),
    )

    assert not _spinner_spinning(harness, review)
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
) -> tuple[PanelUITestHarness, SourceReview, QueuedIOLoop]:
    """SourceReview with production ``JupyterPanelUISession`` + queued io_loop."""
    from ovro_lwa_portal.viz import pipeline_qa_app as pqa
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
    harness.mount(review._layout)
    loop = QueuedIOLoop()
    monkeypatch.setattr(pqa, "_IPYTHON_IO_LOOP", loop)
    monkeypatch.setattr(pqa, "_resolve_ipython_event_loop", lambda: loop)
    monkeypatch.setattr(pqa, "_is_jupyter_kernel_context", lambda: True)
    monkeypatch.setattr(pqa, "_schedule_ipython_main", loop.add_callback)
    review._ui_session = JupyterPanelUISession(review._notebook_ui_views)
    return harness, review, loop


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
    assert _spinner_spinning(harness, review)

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

    assert not _spinner_spinning(harness, review)
    assert _heatmap_values_max(review) == pytest.approx(99.0)
    _assert_heatmap_bokeh_model_live(harness, review)

    review._sky_widget = MagicMock()
    review._sky_widget.clicked_coord = (6.5916, 64.0770)
    review._on_sky_widget_click(None)
    _flush_jupyter_io(loop)

    coord_model = harness.bokeh_model(review._coord_input, review._layout)
    assert "6.5916" in coord_model.value
    assert "64.0770" in coord_model.value
