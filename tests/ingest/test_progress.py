"""Tests for ingest progress mapping."""

from __future__ import annotations

import pytest

from ovro_lwa_portal.ingest.progress import ingest_progress_percent


def test_ingest_progress_percent_setup_phase() -> None:
    assert ingest_progress_percent("setup", 0, 4) == 0.0
    assert ingest_progress_percent("setup", 4, 4) == 10.0


def test_ingest_progress_percent_converting_phase() -> None:
    assert ingest_progress_percent("converting", 0, 2) == 10.0
    assert ingest_progress_percent("converting", 1, 2) == pytest.approx(55.0)
    assert ingest_progress_percent("converting", 2, 2) == 100.0


def test_ingest_progress_percent_legacy_linear() -> None:
    assert ingest_progress_percent("other", 1, 4) == 25.0
