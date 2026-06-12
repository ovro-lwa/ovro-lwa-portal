"""Tests for the extracted JupiterFluxReview Panel app."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

pn = pytest.importorskip("panel")
pytest.importorskip("astrowidget")

from ovro_lwa_portal.viz.jupiter_flux_review_app import (
    JupiterFluxReview,
    JupiterFluxReviewConfig,
)
from ovro_lwa_portal.viz.jupiter_flux_review_data import zarr_path_to_day
from ovro_lwa_portal.viz.pipeline_qa import PipelineQAConfig


def _qa_config(tmp_path: Path) -> PipelineQAConfig:
    zarr_root = tmp_path / "zarr"
    zarr_root.mkdir()
    stem = "pipelineQA-phase2-I-NoTaper-Robust-0"
    store = zarr_root / f"{stem}-20250111.zarr"
    store.mkdir()
    (store / ".zgroup").write_text("{}")
    return replace(
        PipelineQAConfig.phase2_default(),
        zarr_root=zarr_root,
        i_qa_zarr_stem=stem,
    )


def test_zarr_path_to_day_parses_phase2_stem() -> None:
    stem = "pipelineQA-phase2-I-NoTaper-Robust-0"
    path = Path(f"/fast/claw/{stem}-20250111.zarr")
    assert zarr_path_to_day(path, stem=stem) == "2025-01-11"


def test_jupiter_flux_review_builds_layout(tmp_path: Path) -> None:
    cfg = _qa_config(tmp_path)

    def _inline_dispatch(callback) -> None:
        callback()

    review = JupiterFluxReview(
        cfg,
        review_config=JupiterFluxReviewConfig(zarr_lm_chunk=512),
        dispatch_override=_inline_dispatch,
    )

    assert review.panel is review._layout
    assert review._log_pane is not None
    assert "phase2 QA Zarr store(s)" in review.log_text
    assert review.select_zarr is not None


def test_jupiter_flux_review_log_updates_via_dispatch(tmp_path: Path) -> None:
    cfg = _qa_config(tmp_path)

    def _inline_dispatch(callback) -> None:
        callback()

    review = JupiterFluxReview(
        cfg,
        dispatch_override=_inline_dispatch,
    )
    review._dispatch(lambda: review._log("dispatch test line"))
    assert "dispatch test line" in review.log_text
