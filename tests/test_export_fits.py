"""Tests for ``ovro_lwa_portal.export_fits`` (Phase 1 export module)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from astropy.io import fits

from ovro_lwa_portal import export_fits
from ovro_lwa_portal.fits_to_zarr_xradio import _fits_header_bytes_for_slice

from tests.test_fits_to_zarr import _make_sin_wcs_header_str


def _synthetic_export_dataset(
    *,
    nl: int = 8,
    nm: int = 8,
    freq_hz: float = 55e6,
    stokes: float = 1.0,
) -> xr.Dataset:
    post_wcs = _make_sin_wcs_header_str(nx=nl, ny=nm, crval1=180.0, crval2=45.0)
    primary = fits.Header.fromstring(post_wcs, sep="\n")
    primary["RESTFREQ"] = freq_hz
    primary["DATE-OBS"] = "2025-01-20T04:03:33"
    primary["TELESCOP"] = "OVRO-LWA"
    payload = _fits_header_bytes_for_slice(
        primary,
        post_regrid_wcs_hdr=post_wcs,
        nl=nl,
        nm=nm,
        freq_hz=freq_hz,
        stokes=stokes,
    )
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "SKY": (
                ("time", "frequency", "polarization", "m", "l"),
                rng.standard_normal((1, 1, 1, nm, nl), dtype=np.float32),
            ),
            "fits_header_str": (
                ("time", "frequency", "polarization"),
                np.array([[[np.bytes_(payload)]]], dtype=object),
            ),
        },
        coords={
            "time": ("time", np.array([60695.0])),
            "frequency": ("frequency", np.array([freq_hz])),
            "polarization": ("polarization", np.array([stokes])),
            "l": ("l", np.arange(nl, dtype=float)),
            "m": ("m", np.arange(nm, dtype=float)),
        },
    )


def test_build_fits_hdu_naxis4_singleton() -> None:
    """Exported HDU uses OVRO 4D singleton layout aligned with Zarr coords."""
    freq_hz = 55e6
    stokes = 4.0
    ds = _synthetic_export_dataset(freq_hz=freq_hz, stokes=stokes)
    hdu = export_fits.build_fits_hdu(ds, time_idx=0, freq_idx=0, pol_idx=0)

    assert hdu.data is not None
    assert hdu.data.shape == (1, 1, ds.sizes["m"], ds.sizes["l"])
    assert int(hdu.header["NAXIS"]) == 4
    assert int(hdu.header["NAXIS3"]) == 1
    assert int(hdu.header["NAXIS4"]) == 1
    assert str(hdu.header["CTYPE3"]).strip() == "FREQ"
    assert str(hdu.header["CTYPE4"]).strip() == "STOKES"
    assert float(hdu.header["CRVAL3"]) == pytest.approx(freq_hz)
    assert float(hdu.header["CRVAL4"]) == pytest.approx(stokes)
    assert int(hdu.header["BITPIX"]) == -32
    assert "BSCALE" not in hdu.header


def test_build_fits_data_array_matches_sky_slice() -> None:
    ds = _synthetic_export_dataset()
    expected = ds["SKY"].isel(time=0, frequency=0, polarization=0).values.astype(
        np.float32
    )
    data = export_fits.build_fits_data_array(ds, time_idx=0, freq_idx=0, pol_idx=0)
    assert data.shape == (1, 1, ds.sizes["m"], ds.sizes["l"])
    np.testing.assert_array_equal(data[0, 0], expected)


def test_write_fits_slice_writes_readable_file(tmp_path: Path) -> None:
    ds = _synthetic_export_dataset()
    out = tmp_path / "slice.fits"
    export_fits.write_fits_slice(ds, out, time_idx=0, freq_idx=0, pol_idx=0)

    with fits.open(out) as hdul:
        assert len(hdul) == 1
        assert hdul[0].data.shape == (1, 1, ds.sizes["m"], ds.sizes["l"])
        assert int(hdul[0].header["NAXIS"]) == 4


def test_build_fits_hdulist_single_primary() -> None:
    ds = _synthetic_export_dataset()
    hdul = export_fits.build_fits_hdulist(ds)
    assert len(hdul) == 1
    assert isinstance(hdul[0], fits.PrimaryHDU)


def test_export_missing_fits_header_str_raises() -> None:
    ds = xr.Dataset(
        {
            "SKY": (
                ("time", "frequency", "polarization", "m", "l"),
                np.zeros((1, 1, 1, 4, 4), dtype=np.float32),
            ),
            "wcs_header_str": (("time",), np.array([np.bytes_(b"NAXIS   = 2")], dtype=object)),
        },
        coords={
            "time": [60695.0],
            "frequency": [55e6],
            "polarization": [1.0],
            "l": np.arange(4),
            "m": np.arange(4),
        },
    )
    with pytest.raises(ValueError, match="fits_header_str"):
        export_fits.build_fits_hdu(ds)


def test_export_rejects_beam_variable() -> None:
    ds = _synthetic_export_dataset()
    with pytest.raises(ValueError, match="BEAM"):
        export_fits.build_fits_hdu(ds, var="BEAM")


def test_export_module_has_no_viz_import_chain() -> None:
    import ast
    from pathlib import Path

    source = Path(export_fits.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("ovro_lwa_portal.viz")
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("ovro_lwa_portal.viz")
