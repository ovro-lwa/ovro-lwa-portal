"""Tests for name resolution (degrees, from_name, NED fallback)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord

from ovro_lwa_portal.name_resolution import (
    CoordinateResolution,
    resolve_coordinate_string,
    resolve_via_ned,
)


def test_resolve_degree_pair() -> None:
    result, messages = resolve_coordinate_string("61.14, 37.16", use_ned_fallback=False)
    assert messages == []
    assert result is not None
    assert result.resolver == "degrees"
    assert result.coord.ra.deg == pytest.approx(61.14)
    assert result.coord.dec.deg == pytest.approx(37.16)


@patch("ovro_lwa_portal.name_resolution.SkyCoord.from_name")
def test_resolve_from_name_before_ned(mock_from_name: MagicMock) -> None:
    mock_from_name.return_value = SkyCoord(ra=83.6 * u.deg, dec=22.0 * u.deg, frame="icrs")
    with patch("ovro_lwa_portal.name_resolution.resolve_via_ned") as mock_ned:
        result, messages = resolve_coordinate_string("Cas A")
    assert result is not None
    assert result.resolver == "from_name"
    assert messages == []
    mock_ned.assert_not_called()


@patch("ovro_lwa_portal.name_resolution.SkyCoord.from_name", side_effect=Exception("not found"))
@patch("ovro_lwa_portal.name_resolution.resolve_via_ned")
def test_ned_fallback_when_from_name_fails(mock_ned: MagicMock, _mock_from_name: MagicMock) -> None:
    mock_ned.return_value = (
        SkyCoord(ra=10.68 * u.deg, dec=41.27 * u.deg, frame="icrs"),
        ["Resolved via NED (MESSIER 031)."],
        "MESSIER 031",
    )
    result, messages = resolve_coordinate_string("m31", use_ned_fallback=True)
    assert result is not None
    assert result.resolver == "ned"
    assert result.canonical_name == "MESSIER 031"
    assert messages == ["Resolved via NED (MESSIER 031)."]
    mock_ned.assert_called_once()


@patch("ovro_lwa_portal.name_resolution.SkyCoord.from_name", side_effect=Exception("not found"))
def test_ned_fallback_disabled(_mock_from_name: MagicMock) -> None:
    result, messages = resolve_coordinate_string("m31", use_ned_fallback=False)
    assert result is None
    assert len(messages) == 1
    assert "WARNING" in messages[0]


@patch("ovro_lwa_portal.name_resolution.SkyCoord.from_name")
def test_resolve_j2000_embedded_sexagesimal_before_from_name(mock_from_name: MagicMock) -> None:
    result, messages = resolve_coordinate_string(
        "VLSSr J020032.8-351910",
        use_ned_fallback=False,
    )
    assert messages == []
    assert result is not None
    assert result.resolver == "j2000_name"
    assert result.coord.ra.deg == pytest.approx(30.136666666666667)
    assert result.coord.dec.deg == pytest.approx(-35.31944444444445)
    mock_from_name.assert_not_called()


@patch("ovro_lwa_portal.name_resolution.SkyCoord.from_name")
def test_resolve_j2000_name_without_catalog_prefix(mock_from_name: MagicMock) -> None:
    result, messages = resolve_coordinate_string("J123045.6+270512.3", use_ned_fallback=False)
    assert result is not None
    assert result.resolver == "j2000_name"
    assert result.coord.ra.deg == pytest.approx(187.69016666666668)
    assert result.coord.dec.deg == pytest.approx(27.08675)
    mock_from_name.assert_not_called()


@patch("ovro_lwa_portal.name_resolution._try_j2000_embedded_sexagesimal")
@patch("ovro_lwa_portal.name_resolution.SkyCoord.from_name")
def test_known_source_skips_j2000_parse(
    mock_from_name: MagicMock,
    mock_j2000: MagicMock,
) -> None:
    mock_from_name.return_value = SkyCoord(ra=83.6 * u.deg, dec=22.0 * u.deg, frame="icrs")
    result, messages = resolve_coordinate_string(
        "Cas A",
        use_ned_fallback=False,
        known_source_names=frozenset({"cas a"}),
    )
    assert result is not None
    assert result.resolver == "from_name"
    mock_j2000.assert_not_called()
    mock_from_name.assert_called_once_with("Cas A")


@patch("ovro_lwa_portal.name_resolution.requests.post")
def test_resolve_via_ned_known_object(mock_post: MagicMock) -> None:
    mock_post.return_value.json.return_value = {
        "ResultCode": 3,
        "Preferred": {
            "Name": "MESSIER 031",
            "Position": {"RA": 10.68479292, "Dec": 41.269065},
        },
        "Interpreted": {"Name": "MESSIER 031"},
    }
    mock_post.return_value.raise_for_status = MagicMock()
    coord, messages, canonical = resolve_via_ned("m31")
    assert canonical == "MESSIER 031"
    assert coord is not None
    assert coord.ra.deg == pytest.approx(10.68479292)
    assert coord.dec.deg == pytest.approx(41.269065)
    assert messages[0].startswith("Resolved via NED")


@patch("ovro_lwa_portal.name_resolution.requests.post")
def test_resolve_via_ned_ambiguous_strict(mock_post: MagicMock) -> None:
    mock_post.return_value.json.return_value = {
        "ResultCode": 1,
        "Interpreted": {"Aliases": ["ANDROMEDA GALAXY", "ANDROMEDA I"]},
    }
    mock_post.return_value.raise_for_status = MagicMock()
    coord, messages, canonical = resolve_via_ned("andromeda")
    assert canonical is None
    assert coord is None
    assert len(messages) == 1
    assert "ambiguous" in messages[0].lower()
    assert "ANDROMEDA GALAXY" in messages[0]
