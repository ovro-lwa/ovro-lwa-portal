"""Tests for local HiPS URL helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from ovro_lwa_portal.viz.hips import compute_hips_percentile_cuts, hips_background_survey_url


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
