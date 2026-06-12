"""Tests for the extracted SourceReview Panel app."""

from __future__ import annotations

from pathlib import Path

import pytest

pn = pytest.importorskip("panel")
pytest.importorskip("astrowidget")

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
