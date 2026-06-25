"""Tests for local HiPS URL helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from ovro_lwa_portal.viz.hips import (
    compute_hips_percentile_cuts,
    configure_hips_background,
    hips_background_survey_url,
    hips_http_server_survey_url,
    normalize_hips_survey_name,
    resolve_hips_survey_path,
)


def _write_tile(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(values.astype(np.float32))
    hdu.writeto(path, overwrite=True)


def test_hips_background_survey_url_relative_prefix(tmp_path: Path, monkeypatch) -> None:
    hips_root = tmp_path / "hips"
    survey = hips_root / "Blue_I_deep.hips"
    survey.mkdir(parents=True)
    monkeypatch.delenv("OVRO_HIPS_HTTP_BASE", raising=False)
    monkeypatch.delenv("OVRO_HIPS_HTTP_PREFIX", raising=False)

    url = hips_background_survey_url(survey, hips_root=hips_root, http_prefix="/calibration/hips")
    assert url == "/calibration/hips/Blue_I_deep.hips/"


def test_normalize_hips_survey_name() -> None:
    assert normalize_hips_survey_name("Blue_I_deep") == "Blue_I_deep.hips"
    assert normalize_hips_survey_name("Blue_I_deep.hips/") == "Blue_I_deep.hips"


def test_resolve_hips_survey_path_relative(tmp_path: Path) -> None:
    hips_root = tmp_path / "hips"
    survey = hips_root / "Blue_I_deep.hips"
    survey.mkdir(parents=True)

    assert resolve_hips_survey_path("Blue_I_deep", hips_root=hips_root) == survey.resolve()


def test_hips_http_server_survey_url() -> None:
    url = hips_http_server_survey_url(
        "Blue_I_deep_Taper_Robust-0.75_Jan25",
        port=3005,
        host="calim10.example.edu",
    )
    assert url == "http://calim10.example.edu:3005/Blue_I_deep_Taper_Robust-0.75_Jan25.hips/"


def test_configure_hips_background_external_server(tmp_path: Path) -> None:
    hips_root = tmp_path / "hips"
    survey = hips_root / "Blue_I_deep.hips"
    survey.mkdir(parents=True)

    cfg = configure_hips_background(
        "Blue_I_deep",
        hips_root=hips_root,
        http_port=3005,
        http_host="localhost",
    )

    assert cfg.disk_path == survey.resolve()
    assert cfg.url == "http://localhost:3005/Blue_I_deep.hips/"


def test_configure_hips_background_jupyter_extension(tmp_path: Path) -> None:
    hips_root = tmp_path / "hips"
    survey = hips_root / "Blue_I_deep.hips"
    survey.mkdir(parents=True)

    cfg = configure_hips_background(
        survey,
        hips_root=hips_root,
        http_port=None,
        jupyter_http_prefix="/calibration/hips",
    )

    assert cfg.url == "/calibration/hips/Blue_I_deep.hips/"


def test_hips_background_survey_url_absolute_http_base(tmp_path: Path) -> None:
    hips_root = tmp_path / "hips"
    survey = hips_root / "Blue_I_deep.hips"
    survey.mkdir(parents=True)

    url = hips_background_survey_url(
        survey,
        hips_root=hips_root,
        http_prefix="https://example.org/calibration/hips",
    )
    assert url == "https://example.org/calibration/hips/Blue_I_deep.hips/"


def test_compute_hips_percentile_cuts(tmp_path: Path) -> None:
    hips = tmp_path / "survey.hips"
    _write_tile(hips / "Norder0/Dir0/Npix0.fits", np.linspace(0.0, 100.0, 100, dtype=np.float32))
    _write_tile(hips / "Norder0/Dir0/Npix1.fits", np.linspace(100.0, 200.0, 100, dtype=np.float32))

    lo, hi = compute_hips_percentile_cuts(hips, percentile_low=2, percentile_high=98)

    assert lo == pytest.approx(2.0, abs=5.0)
    assert hi == pytest.approx(198.0, abs=5.0)


def test_compute_hips_percentile_cuts_missing_tiles(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No FITS tiles"):
        compute_hips_percentile_cuts(tmp_path / "empty.hips")


def test_register_hips_panel_serve_adds_toplevel_pattern(tmp_path: Path) -> None:
    from bokeh.server.urls import toplevel_patterns

    from ovro_lwa_portal.viz import hips_server as hs

    hips_root = tmp_path / "hips"
    hips_root.mkdir()
    hs._HIPS_PANEL_PATTERN = None  # noqa: SLF001
    prefix = f"/test-hips-{tmp_path.name}"

    before = len(toplevel_patterns)
    hs.register_hips_panel_serve(hips_root, prefix)

    assert len(toplevel_patterns) == before + 1
    entry = toplevel_patterns[-1]
    pattern = entry[0]
    handler = entry[1]
    kwargs = entry[2] if len(entry) > 2 else {}
    assert "test-hips" in pattern.replace("\\", "")
    assert kwargs["path"] == str(hips_root.resolve())
    from tornado.web import StaticFileHandler

    assert handler is StaticFileHandler

    # Idempotent second call.
    hs.register_hips_panel_serve(hips_root, prefix)
    assert len(toplevel_patterns) == before + 1
