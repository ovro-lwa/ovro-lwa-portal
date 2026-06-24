"""Tests for the extracted SourceReview Panel app."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pn = pytest.importorskip("panel")
pytest.importorskip("astrowidget")

import astropy.units as u
from astropy.coordinates import SkyCoord

from ovro_lwa_portal.viz.source_review_app import SourceReview, SourceReviewConfig


def test_source_review_builds_layout_without_zarr_validation(tmp_path: Path) -> None:
    zarr = tmp_path / "missing.zarr"
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

    assert review.panel is review._layout
    assert review._log_pane is not None
    assert "Zarr:" in review.log_text


def test_source_review_log_updates_via_inline_dispatch(tmp_path: Path) -> None:
    zarr = tmp_path / "store.zarr"
    zarr.mkdir()

    def _inline_dispatch(callback) -> None:
        callback()

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
        dispatch_override=_inline_dispatch,
    )

    review._dispatch(lambda: review._log("dispatch test line"))
    assert "dispatch test line" in review.log_text


def test_ui_action_handlers_schedule_through_dispatch(tmp_path: Path) -> None:
    """Panel buttons and param actions must enter the notebook dispatch batch."""
    zarr = tmp_path / "store.zarr"
    zarr.mkdir()
    scheduled: list[str] = []

    def _recording_dispatch(callback) -> None:
        scheduled.append(callback.__name__ if hasattr(callback, "__name__") else "lambda")
        callback()

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
        dispatch_override=_recording_dispatch,
    )

    review._on_slew()
    assert scheduled and scheduled[-1] == "_on_slew_impl"

    scheduled.clear()
    review._on_generate_heatmap()
    assert scheduled and scheduled[-1] == "_on_generate_heatmap_impl"

    scheduled.clear()
    review._on_heatmap_method_change()
    assert scheduled and scheduled[-1] == "_on_heatmap_method_change_impl"

    scheduled.clear()
    review._on_overlay_toggle(MagicMock(new=True))
    assert scheduled and scheduled[-1] == "_run"


def test_fit_overlay_button_exists(tmp_path: Path) -> None:
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
    assert review._fit_overlay_button.name == "Fit overlay"
    assert review._fit_overlay_button.disabled is True


def test_fit_overlay_click_schedules_impl(tmp_path: Path) -> None:
    zarr = tmp_path / "store.zarr"
    zarr.mkdir()
    scheduled: list[str] = []

    def _recording_dispatch(callback) -> None:
        scheduled.append(callback.__name__ if hasattr(callback, "__name__") else "lambda")
        callback()

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
        dispatch_override=_recording_dispatch,
    )

    started: list[bool] = []
    review._load_overlay_fit = lambda: started.append(True)  # type: ignore[method-assign]
    review._fit_overlay_button.disabled = False

    review._on_fit_overlay(None)
    assert scheduled and scheduled[-1] == "_on_fit_overlay_impl"
    assert started == [True]


def test_fit_overlay_button_sync_is_nonblocking(
    tmp_path: Path,
    valid_ovro_dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fit overlay gating must not scan the full cube on the UI thread."""
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
    review._dataset = valid_ovro_dataset
    review._coord = SkyCoord(ra=350.85 * u.deg, dec=58.815 * u.deg, frame="icrs")
    review._time_idx = 0
    review._freq_idx = 0

    def _fail_full_cache(self, **_kwargs) -> None:
        msg = "ensure_patch_metadata_cache must not run for Fit overlay button gating"
        raise AssertionError(msg)

    monkeypatch.setattr(
        "ovro_lwa_portal.accessor.RadportAccessor.ensure_patch_metadata_cache",
        _fail_full_cache,
    )
    review._sync_fit_overlay_button()
    assert review._fit_overlay_button.disabled is True
