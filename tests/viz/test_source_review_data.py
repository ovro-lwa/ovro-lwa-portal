"""Tests for source review data helpers (no Panel/notebook comm)."""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from ovro_lwa_portal.viz.source_review_data import (
    build_source_from_coordinate,
    calendar_mmdd_labels_for_time_coord,
    filter_known_source_names,
    format_heatmap_time_axis_label,
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


def test_format_heatmap_time_axis_label_includes_day_and_lst() -> None:
    from astropy.time import Time

    times = np.array([60000.0, 60001.5])
    day_labels = calendar_mmdd_labels_for_time_coord(times)
    lst_hours = np.array([4.0, 5.5])
    assert format_heatmap_time_axis_label(
        times, 0, lst_hours, day_labels=day_labels
    ) == f"{day_labels[0]} 04h"
    assert format_heatmap_time_axis_label(
        times, 1, lst_hours, day_labels=day_labels
    ) == f"{day_labels[1]} 06h"
    assert day_labels[0] != "01-01" or Time(60000.0, format="mjd", scale="utc").isot[5:10] == "01-01"


def test_calendar_mmdd_labels_from_mjd_not_datetime64_epoch() -> None:
    from astropy.time import Time

    mjd = np.array([60000.0, 60001.0])
    labels = calendar_mmdd_labels_for_time_coord(mjd)
    expected = np.array([Time(v, format="mjd", scale="utc").isot[5:10] for v in mjd])
    assert labels.tolist() == expected.tolist()
    assert labels[0] != "01-01" or expected[0] == "01-01"
