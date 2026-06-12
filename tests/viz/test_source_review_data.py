"""Tests for source review data helpers (no Panel/notebook comm)."""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from ovro_lwa_portal.viz.source_review_data import (
    build_source_from_coordinate,
    filter_known_source_names,
    load_known_sources,
    resolve_known_sources_path,
)


def test_filter_known_source_names_prefix_match() -> None:
    names = ["Cas A", "Cyg A", "3C 273"]
    assert filter_known_source_names("cas", names) == ["Cas A"]
    assert filter_known_source_names("12.3", names) == []


def test_build_source_from_coordinate_fields() -> None:
    coord = SkyCoord(ra=350.85 * u.deg, dec=58.815 * u.deg, frame="icrs")
    src = build_source_from_coordinate("Cas A", coord)
    assert src["name"] == "Cas A"
    assert src["ra"] == pytest.approx(350.85)
    assert src["dec"] == pytest.approx(58.815)
    assert "l" in src and "b" in src


def test_load_known_sources_from_repo_fixture(tmp_path: Path) -> None:
    yaml_path = tmp_path / "known_sources.yaml"
    yaml_path.write_text(
        "sources:\n  - name: Cas A\n  - Cyg A\n",
        encoding="utf-8",
    )
    names = load_known_sources(yaml_path)
    assert names == ["Cas A", "Cyg A"]
    assert resolve_known_sources_path(yaml_path) == yaml_path.resolve()
