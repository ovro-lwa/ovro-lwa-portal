"""Tests for FITS to Zarr conversion utilities."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pytest
from astropy.io import fits


def _import_module():
    """Import module under test if optional dependencies are available."""
    try:
        from ovro_lwa_portal import fits_to_zarr_xradio
    except ImportError as e:
        pytest.skip(f"xradio dependencies not available: {e}")
    return fits_to_zarr_xradio


def test_time_does_not_fall_back_to_filename(tmp_path: Path):
    """Without ``-image-YYYYMMDD_HHMMSS`` or DATE-OBS, time key stays unknown."""
    mod = _import_module()
    fpath = tmp_path / "20240524_050009_41MHz_averaged_20000_iterations-I-image.fits"
    fits.PrimaryHDU(data=[[1.0]], header=fits.Header({"SIMPLE": True})).writeto(fpath)
    time_key, _, notes = mod._extract_group_metadata(fpath)
    assert time_key is None
    assert "time-from-filename" not in notes


def test_mhz_from_name():
    """Test extracting MHz from filename."""
    # Extract MHZ_RE pattern from module
    module_path = Path(__file__).parent.parent / "src" / "ovro_lwa_portal" / "fits_to_zarr_xradio.py"
    content = module_path.read_text()
    mhz_match = re.search(r'MHZ_RE = re\.compile\(r"([^"]+)"\)', content)
    assert mhz_match is not None
    MHZ_RE = re.compile(mhz_match.group(1))

    # Replicate _mhz_from_name logic
    path = Path("20240524_050009_41MHz_averaged_20000_iterations-I-image.fits")
    m = MHZ_RE.search(path.name)
    mhz = int(m.group(1)) if m else 10**9

    assert mhz == 41


def test_mhz_from_name_multiple_digits():
    """Test extracting MHz from filename with larger frequency."""
    # Extract MHZ_RE pattern from module
    module_path = Path(__file__).parent.parent / "src" / "ovro_lwa_portal" / "fits_to_zarr_xradio.py"
    content = module_path.read_text()
    mhz_match = re.search(r'MHZ_RE = re\.compile\(r"([^"]+)"\)', content)
    assert mhz_match is not None
    MHZ_RE = re.compile(mhz_match.group(1))

    path = Path("20240524_050009_82MHz_averaged_20000_iterations-I-image.fits")
    m = MHZ_RE.search(path.name)
    mhz = int(m.group(1)) if m else 10**9

    assert mhz == 82


def test_mhz_from_name_no_match():
    """Test extracting MHz from filename without frequency."""
    # Extract MHZ_RE pattern from module
    module_path = Path(__file__).parent.parent / "src" / "ovro_lwa_portal" / "fits_to_zarr_xradio.py"
    content = module_path.read_text()
    mhz_match = re.search(r'MHZ_RE = re\.compile\(r"([^"]+)"\)', content)
    assert mhz_match is not None
    MHZ_RE = re.compile(mhz_match.group(1))

    path = Path("invalid.fits")
    m = MHZ_RE.search(path.name)
    mhz = int(m.group(1)) if m else 10**9

    # Should return sentinel value
    assert mhz == 10**9


def test_mhz_from_name_hyphen_after_mhz_like_dewarp_staging(tmp_path: Path) -> None:
    """``_NNMHz-`` in staged dewarp basenames must resolve (hyphenated OVRO product tag)."""
    mod = _import_module()
    name = "20250101_040422__18MHz-I-Deep-Taper-Robust-0-image-20250101_040422.pbcorr_dewarp.fits"
    fpath = tmp_path / name
    fits.PrimaryHDU(data=[[1.0]], header=fits.Header({"SIMPLE": True})).writeto(fpath)
    _, frequency_hz, notes = mod._extract_group_metadata(fpath)
    assert frequency_hz == pytest.approx(18e6)
    assert "frequency-from-filename" in notes


def test_mhz_from_name_leading_subband_token() -> None:
    """Phase2 dewarped basenames start with ``NNMHz-`` (no leading underscore)."""
    mod = _import_module()
    assert mod._mhz_from_name(Path("82MHz-I-NoTaper-3581s-Robust-0-image.fits")) == 82


def test_time_key_from_phase2_dewarped_basename(tmp_path: Path) -> None:
    """Phase2 products encode time as ``YYYYMMDD_HHMMSS-image`` before the suffix."""
    mod = _import_module()
    name = "82MHz-I-NoTaper-3581s-Robust-0-20241218_033402-image.pbcorr_dewarped.fits"
    fpath = tmp_path / name
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-12-18T00:00:00", "RESTFREQ": 82e6}),
    ).writeto(fpath)

    time_key, frequency_hz, notes = mod._extract_group_metadata(fpath)

    assert time_key == "20241218_033402"
    assert frequency_hz == pytest.approx(82e6)
    assert "time-from-filename" in notes


def test_discover_groups_phase2_dewarped_basename_merges_subbands(tmp_path: Path) -> None:
    """Subbands sharing the same phase2 image-time token belong to one time step."""
    mod = _import_module()
    a = tmp_path / "18MHz-I-NoTaper-3581s-Robust-0-20241218_033402-image.pbcorr_dewarped.fits"
    b = tmp_path / "82MHz-I-NoTaper-3581s-Robust-0-20241218_033402-image.pbcorr_dewarped.fits"
    for path, mhz in ((a, 18e6), (b, 82e6)):
        fits.PrimaryHDU(
            data=[[1.0]],
            header=fits.Header({"DATE-OBS": "2024-12-18T01:00:00", "RESTFREQ": mhz}),
        ).writeto(path)

    groups = mod._discover_groups(tmp_path)

    assert list(groups.keys()) == ["20241218_033402"]
    assert {p.name for p in groups["20241218_033402"]} == {a.name, b.name}


def test_module_can_be_imported():
    """Test that the fits_to_zarr_xradio module can be imported."""
    fits_to_zarr_xradio = _import_module()
    assert hasattr(fits_to_zarr_xradio, "convert_fits_dir_to_zarr")
    assert hasattr(fits_to_zarr_xradio, "MHZ_RE")


def test_select_reference_shape_index_deterministic():
    """Largest LM shape should be selected deterministically."""
    fits_to_zarr_xradio = _import_module()

    shapes = [(3122, 3122), (4096, 4096), (4096, 3000), (4096, 4096)]
    idx = fits_to_zarr_xradio._select_reference_shape_index(shapes)

    # Tie on (4096, 4096) should pick first occurrence.
    assert idx == 1


def test_peek_lm_shape_reads_dimensions_from_header(tmp_path: Path):
    """Peeked LM shape should match NAXIS2/NAXIS1 from FITS header."""
    import numpy as np

    mod = _import_module()
    fpath = tmp_path / "lm_shape_from_header.fits"
    fits.PrimaryHDU(data=np.zeros((5, 7), dtype=np.float32)).writeto(fpath)

    assert mod._peek_lm_shape(fpath) == (7, 5)


def test_peek_lm_shape_errors_when_naxis_less_than_two(tmp_path: Path):
    """Peeking LM shape should fail clearly for non-image FITS data."""
    import numpy as np

    mod = _import_module()
    fpath = tmp_path / "not_2d.fits"
    fits.PrimaryHDU(data=np.zeros(8, dtype=np.float32)).writeto(fpath)

    with pytest.raises(RuntimeError, match="NAXIS=1"):
        mod._peek_lm_shape(fpath)


def test_assign_canonical_frequency_for_stack_uses_mhz_token_over_identical_header_freq():
    """Basename ``_NNNMHz_`` must set the stack ``frequency`` coord when xradio agrees on Hz."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    xds = xr.Dataset(
        {"SKY": (["frequency", "l", "m"], np.zeros((1, 2, 2), dtype=np.float32))},
        coords={
            "frequency": np.array([73.8e6], dtype=np.float64),
            "l": np.array([0.0, 1.0]),
            "m": np.array([0.0, 1.0]),
        },
    )
    fp = Path("20240524_041000_55MHz_averaged-I-image_fixed.fits")
    out = mod._assign_canonical_frequency_for_stack(
        xds, fp, group_metadata_source="fits"
    )
    assert float(out["frequency"].values[0]) == pytest.approx(55e6)


def test_combine_by_coords_two_slices_no_duplicate_frequency_after_canonical_assign():
    """Two OVRO-style names at different MHz must combine without duplicate ``frequency``."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    coords_common = {
        "l": np.array([0.0, 1.0]),
        "m": np.array([0.0, 1.0]),
    }
    dup_header_freq = np.array([73.8e6], dtype=np.float64)
    x41 = xr.Dataset(
        {"SKY": (["frequency", "l", "m"], np.ones((1, 2, 2), dtype=np.float32) * 41.0)},
        coords={"frequency": dup_header_freq, **coords_common},
    )
    x55 = xr.Dataset(
        {"SKY": (["frequency", "l", "m"], np.ones((1, 2, 2), dtype=np.float32) * 55.0)},
        coords={"frequency": dup_header_freq, **coords_common},
    )
    p41 = Path("t__20240524_041000_41MHz_averaged-I-image.fits")
    p55 = Path("t__20240524_041000_55MHz_averaged-I-image.fits")
    x41 = mod._assign_canonical_frequency_for_stack(x41, p41, group_metadata_source="filename")
    x55 = mod._assign_canonical_frequency_for_stack(x55, p55, group_metadata_source="filename")
    merged = xr.combine_by_coords(
        [x41, x55],
        combine_attrs="drop",
        data_vars="minimal",
        coords="minimal",
        compat="no_conflicts",
    )
    merged = merged.sortby("frequency")
    assert merged.sizes["frequency"] == 2
    np.testing.assert_allclose(
        merged["frequency"].values, np.array([41e6, 55e6], dtype=np.float64)
    )


def _make_sin_wcs_header_str(
    *, nx: int, ny: int, crval1: float, crval2: float, cdelt: float = 0.1
) -> str:
    """Build a minimal 2D ``RA---SIN`` / ``DEC--SIN`` FITS header string.

    Used by ``_regrid_to_reference_lm`` tests because that function now recomputes
    ``right_ascension``/``declination`` from the persisted WCS, so fixtures need
    real celestial WCS info (CTYPE/CRVAL/CRPIX/CDELT), not placeholder strings.
    """
    from astropy.io import fits as _fits

    h = _fits.Header()
    h["NAXIS"] = 2
    h["NAXIS1"] = nx
    h["NAXIS2"] = ny
    h["CTYPE1"] = "RA---SIN"
    h["CTYPE2"] = "DEC--SIN"
    h["CRVAL1"] = float(crval1)
    h["CRVAL2"] = float(crval2)
    h["CRPIX1"] = (nx + 1) / 2.0
    h["CRPIX2"] = (ny + 1) / 2.0
    h["CDELT1"] = -float(cdelt)
    h["CDELT2"] = float(cdelt)
    h["RADESYS"] = "FK5"
    h["EQUINOX"] = 2000.0
    return h.tostring(sep="\n")


def _radec_from_header_str(
    hdr_str: str, *, nl: int, nm: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``(ra, dec)`` arrays of shape ``(nl, nm)`` using a 2D celestial WCS."""
    from astropy.io import fits as _fits
    from astropy.wcs import WCS as _WCS

    w = _WCS(_fits.Header.fromstring(hdr_str, sep="\n"))
    yy, xx = np.indices((nm, nl), dtype=float)
    ra2d, dec2d = w.all_pix2world(xx, yy, 0)
    return np.transpose(ra2d), np.transpose(dec2d)


# ``np`` is imported above only inside test bodies in this file; pull it to module scope so
# the helpers above can use it without re-importing.
import numpy as np  # noqa: E402


def _encoded_fits_header_bytes(hdr_str: str, *, nl: int, nm: int) -> np.bytes_:
    """Encode a pixel-faithful ``fits_header_str`` payload for synthetic datasets."""
    from astropy.io import fits as afits

    mod = _import_module()
    return np.bytes_(
        mod._fits_header_bytes_for_slice(
            afits.Header.fromstring(hdr_str, sep="\n"),
            post_regrid_wcs_hdr=hdr_str,
            nl=nl,
            nm=nm,
        )
    )


def _attach_ingest_header_fixtures(
    xds,
    hdr_str: str,
    *,
    include_fits_header_str: bool = True,
):
    """Attach in-memory ingest header fields used by regrid/combine helpers."""
    out = xds.copy(deep=False)
    out.attrs["fits_wcs_header"] = hdr_str
    out.attrs["_fits_primary_header_str"] = hdr_str
    out = out.assign(wcs_header_str=((), np.bytes_(hdr_str.encode("utf-8"))))
    if include_fits_header_str:
        mod = _import_module()
        out = mod._assign_pixel_faithful_fits_header_str(
            out, post_regrid_wcs_hdr=hdr_str
        )
    return out


def test_regrid_to_reference_lm_mixed_shapes():
    """Smaller (m,l) grids interpolate onto the reference LM grid."""
    import xarray as xr

    fits_to_zarr_xradio = _import_module()

    l_ref = np.linspace(-1.0, 1.0, 6)
    m_ref = np.linspace(-1.0, 1.0, 5)
    rng = np.random.default_rng(0)
    sky_ref = rng.standard_normal((5, 6))
    # Both ref and source share the same per-time CRVAL → output RA/Dec must equal
    # what a fresh WCS evaluation on ref's pixel grid would produce.
    hdr_ref = _make_sin_wcs_header_str(nx=6, ny=5, crval1=180.0, crval2=45.0)
    expected_ra, expected_dec = _radec_from_header_str(hdr_ref, nl=6, nm=5)
    xds_ref = xr.Dataset(
        data_vars={"SKY": (("m", "l"), sky_ref)},
        coords={
            "l": ("l", l_ref),
            "m": ("m", m_ref),
            "right_ascension": (("l", "m"), expected_ra),
            "declination": (("l", "m"), expected_dec),
        },
    )
    xds_ref = _attach_ingest_header_fixtures(xds_ref, hdr_ref)

    l_sm = np.linspace(-0.5, 0.5, 4)
    m_sm = np.linspace(-0.5, 0.5, 3)
    sky_sm = rng.standard_normal((3, 4))
    hdr_sm = _make_sin_wcs_header_str(nx=4, ny=3, crval1=180.0, crval2=45.0)
    xds_sm = xr.Dataset(
        data_vars={"SKY": (("m", "l"), sky_sm)},
        coords={
            "l": ("l", l_sm),
            "m": ("m", m_sm),
            "right_ascension": (("m", "l"), np.zeros((3, 4))),
            "declination": (("m", "l"), np.zeros((3, 4))),
        },
    )
    xds_sm = _attach_ingest_header_fixtures(xds_sm, hdr_sm)

    out = fits_to_zarr_xradio._regrid_to_reference_lm(xds_sm, xds_ref)

    assert out.sizes["m"] == 5
    assert out.sizes["l"] == 6
    np.testing.assert_allclose(out["l"].values, l_ref)
    np.testing.assert_allclose(out["m"].values, m_ref)
    # The persisted WCS now reflects ref's pixel grid + source's CRVAL (here identical).
    out_hdr_str = out.attrs["fits_wcs_header"]
    assert out["SKY"].attrs["fits_wcs_header"] == out_hdr_str
    np.testing.assert_allclose(out["right_ascension"].values, expected_ra)
    np.testing.assert_allclose(out["declination"].values, expected_dec)
    # Pixel-faithful header keeps post-regrid celestial reference values.
    from astropy.io import fits as afits

    out_hdr = afits.Header.fromstring(
        bytes(out["fits_header_str"].values.item()).decode("utf-8"), sep="\n"
    )
    assert out_hdr["CRVAL1"] == pytest.approx(180.0)
    assert out_hdr["CRVAL2"] == pytest.approx(45.0)


def test_regrid_to_reference_lm_same_shape_different_index_coords():
    """4096²-shaped slices with different ``l``/``m`` vectors must still align to ``ref``.

    Otherwise :func:`xarray.combine_by_coords` outer-joins ``l``/``m`` and spatial
    dimensions blow up (e.g. ``3 × 4096`` for three subbands).
    """
    import xarray as xr

    mod = _import_module()
    n = 8
    l_ref = np.linspace(-1.0, 1.0, n)
    m_ref = np.linspace(-1.0, 1.0, n)
    sky = np.arange(n * n, dtype=np.float64).reshape(n, n)
    hdr = _make_sin_wcs_header_str(nx=n, ny=n, crval1=180.0, crval2=45.0)

    def mk_ds(l_arr: np.ndarray, m_arr: np.ndarray) -> xr.Dataset:
        ds = xr.Dataset(
            {"SKY": (("m", "l"), sky.copy())},
            coords={
                "l": ("l", l_arr),
                "m": ("m", m_arr),
                "right_ascension": (("m", "l"), np.full((n, n), 180.0)),
                "declination": (("m", "l"), np.full((n, n), 45.0)),
            },
        )
        return _attach_ingest_header_fixtures(ds, hdr)

    xds_ref = mk_ds(l_ref, m_ref)
    l_other = np.linspace(-0.5, 0.5, n)
    xds_other = mk_ds(l_other, m_ref.copy())

    out = mod._regrid_to_reference_lm(xds_other, xds_ref)

    assert out.sizes["l"] == n
    assert out.sizes["m"] == n
    np.testing.assert_allclose(out["l"].values, l_ref)
    np.testing.assert_allclose(out["m"].values, m_ref)


def test_regrid_to_reference_lm_requires_l_m_coords():
    """Regridding must fail clearly when ``l``/``m`` coordinates are absent."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    l_ref = np.linspace(-1.0, 1.0, 4)
    m_ref = np.linspace(-1.0, 1.0, 4)
    sky_ref = np.zeros((4, 4))
    xds_ref = xr.Dataset(
        {"SKY": (("m", "l"), sky_ref)},
        coords={
            "l": ("l", l_ref),
            "m": ("m", m_ref),
            "right_ascension": (("m", "l"), np.zeros((4, 4))),
            "declination": (("m", "l"), np.zeros((4, 4))),
        },
        attrs={"fits_wcs_header": "SIMPLE  =                   T"},
    )
    xds_no_coords = xr.Dataset({"SKY": (("m", "l"), np.zeros((2, 3)))})

    with pytest.raises(RuntimeError, match="missing"):
        mod._regrid_to_reference_lm(xds_no_coords, xds_ref)


def test_regrid_to_reference_lm_error_includes_source_label():
    """Interpolation failures should name the source file when provided."""
    from unittest.mock import patch

    import xarray as xr

    mod = _import_module()
    rng = np.random.default_rng(2)
    l_ref = np.linspace(-1.0, 1.0, 5)
    m_ref = np.linspace(-1.0, 1.0, 5)
    sky_ref = rng.standard_normal((5, 5))
    hdr_ref = _make_sin_wcs_header_str(nx=5, ny=5, crval1=180.0, crval2=45.0)
    xds_ref = xr.Dataset(
        data_vars={"SKY": (("m", "l"), sky_ref)},
        coords={
            "l": ("l", l_ref),
            "m": ("m", m_ref),
            "right_ascension": (("m", "l"), np.full((5, 5), 180.0)),
            "declination": (("m", "l"), np.full((5, 5), 45.0)),
        },
        attrs={"fits_wcs_header": hdr_ref},
    )
    l_sm = np.linspace(-0.5, 0.5, 4)
    m_sm = np.linspace(-0.5, 0.5, 3)
    sky_sm = rng.standard_normal((3, 4))
    xds_sm = xr.Dataset(
        data_vars={"SKY": (("m", "l"), sky_sm)},
        coords={
            "l": ("l", l_sm),
            "m": ("m", m_sm),
            "right_ascension": (("m", "l"), np.zeros((3, 4))),
            "declination": (("m", "l"), np.zeros((3, 4))),
        },
        attrs={"fits_wcs_header": hdr_ref},
    )

    with (
        patch.object(xr.Dataset, "interp", side_effect=ValueError("simulated interp failure")),
        pytest.raises(RuntimeError, match="bad_file.fits"),
    ):
        mod._regrid_to_reference_lm(xds_sm, xds_ref, source_label="bad_file.fits")


def test_regrid_to_reference_lm_uses_source_crval_for_radec():
    """Output RA/Dec must come from source's per-time CRVAL on ref's pixel grid.

    Reproduces the
    ``Celestial coordinate grids differ by up to <large> arcsec across N frequency
    slice(s)`` warning scenario:

    * The global LM reference is built once from the earliest time step → its
      ``CRVAL1``/``CRVAL2`` reflect that reference FITS header.
    * At later time steps, each source FITS carries its own native ``CRVAL``.
    * Sky positions for a single time step must therefore use the source's CRVAL
      (otherwise subbands that fell through the short-circuit and those that were
      actually regridded disagree by the LST advance between the reference time and
      the current time → frequency-dependent RA/Dec in the combined dataset).
    """
    import xarray as xr

    mod = _import_module()
    n_ref = 6
    l_ref = np.linspace(-1.0, 1.0, n_ref)
    m_ref = np.linspace(-1.0, 1.0, n_ref)
    rng = np.random.default_rng(7)
    sky_ref = rng.standard_normal((n_ref, n_ref))

    ref_crval = (180.0, 45.0)
    src_crval = (200.0, 40.0)  # later time step → different native phase center

    hdr_ref = _make_sin_wcs_header_str(
        nx=n_ref, ny=n_ref, crval1=ref_crval[0], crval2=ref_crval[1]
    )
    xds_ref = _attach_ingest_header_fixtures(
        xr.Dataset(
            data_vars={"SKY": (("m", "l"), sky_ref)},
            coords={
                "l": ("l", l_ref),
                "m": ("m", m_ref),
                "right_ascension": (("l", "m"), np.full((n_ref, n_ref), ref_crval[0])),
                "declination": (("l", "m"), np.full((n_ref, n_ref), ref_crval[1])),
            },
        ),
        hdr_ref,
    )

    n_sm = 4
    l_sm = np.linspace(-0.5, 0.5, n_sm)
    m_sm = np.linspace(-0.5, 0.5, n_sm)
    sky_sm = rng.standard_normal((n_sm, n_sm))
    hdr_sm = _make_sin_wcs_header_str(
        nx=n_sm, ny=n_sm, crval1=src_crval[0], crval2=src_crval[1]
    )
    xds_sm = _attach_ingest_header_fixtures(
        xr.Dataset(
            data_vars={"SKY": (("m", "l"), sky_sm)},
            coords={
                "l": ("l", l_sm),
                "m": ("m", m_sm),
                "right_ascension": (("m", "l"), np.zeros((n_sm, n_sm))),
                "declination": (("m", "l"), np.zeros((n_sm, n_sm))),
            },
        ),
        hdr_sm,
    )

    out = mod._regrid_to_reference_lm(xds_sm, xds_ref)

    assert out.sizes["l"] == n_ref
    assert out.sizes["m"] == n_ref

    # The persisted header must keep ref's pixel grid (CRPIX/CDELT/CTYPE) and adopt
    # source's celestial reference value.
    from astropy.io import fits
    from astropy.wcs import WCS

    out_hdr = fits.Header.fromstring(out.attrs["fits_wcs_header"], sep="\n")
    assert out_hdr["CRVAL1"] == pytest.approx(src_crval[0])
    assert out_hdr["CRVAL2"] == pytest.approx(src_crval[1])
    assert out_hdr["CRPIX1"] == pytest.approx((n_ref + 1) / 2.0)
    assert out_hdr["CRPIX2"] == pytest.approx((n_ref + 1) / 2.0)
    assert out_hdr["CTYPE1"].startswith("RA")
    assert out_hdr["CTYPE2"].startswith("DEC")

    # Output RA/Dec must equal what a fresh WCS evaluation produces on the hybrid header.
    yy, xx = np.indices((n_ref, n_ref), dtype=float)
    ra_ref, dec_ref = WCS(out_hdr).all_pix2world(xx, yy, 0)
    np.testing.assert_allclose(out["right_ascension"].values, np.transpose(ra_ref))
    np.testing.assert_allclose(out["declination"].values, np.transpose(dec_ref))

    # And they must *not* equal ref's RA/Dec (which would silently mix obs times).
    assert not np.allclose(out["right_ascension"].values, ref_crval[0])
    assert out["right_ascension"].dims == ("l", "m")
    assert out["declination"].dims == ("l", "m")

    # Persisted strings agree across attrs and the 0-D wcs_header_str variable.
    out_hdr_str = out.attrs["fits_wcs_header"]
    assert out["SKY"].attrs["fits_wcs_header"] == out_hdr_str
    # Pixel-faithful header keeps post-regrid celestial reference values.
    from astropy.io import fits as afits

    out_hdr = afits.Header.fromstring(
        bytes(out["fits_header_str"].values.item()).decode("utf-8"), sep="\n"
    )
    assert out_hdr["CRVAL1"] == pytest.approx(src_crval[0])
    assert out_hdr["CRVAL2"] == pytest.approx(src_crval[1])


def test_load_global_lm_reference_selects_largest_shape(monkeypatch, tmp_path: Path):
    """Global reference must load the FITS whose LM shape wins the max-shape rule."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    f_small = tmp_path / "small.fits"
    f_large = tmp_path / "large.fits"
    f_small.touch()
    f_large.touch()
    by_time = {"20240101_120000": [f_small, f_large]}

    monkeypatch.setattr(mod, "fix_fits_headers", lambda files, fd, skip_existing=True: list(files))

    def fake_getheader(fp: Path) -> fits.Header:
        hdr = fits.Header()
        hdr["NAXIS"] = 2
        n = 32 if fp.name.startswith("small") else 64
        hdr["NAXIS1"] = n
        hdr["NAXIS2"] = n
        hdr["BMAJ"] = 0.1
        hdr["BMIN"] = 0.1
        return hdr

    monkeypatch.setattr(mod, "_getheader_for_ingest", fake_getheader)

    loaded: list[Path] = []

    def fake_load(fp: Path, chunk_lm: int = 1024) -> xr.Dataset:
        loaded.append(fp)
        n = 64
        l_ = np.linspace(-1.0, 1.0, n)
        m_ = np.linspace(-1.0, 1.0, n)
        sky = np.zeros((n, n))
        hdr = "SIMPLE  =                   T\nNAXIS   =                    2"
        return (
            xr.Dataset(
                {"SKY": (("m", "l"), sky)},
                coords={
                    "l": ("l", l_),
                    "m": ("m", m_),
                    "frequency": ("frequency", np.array([1.4e8])),
                    "right_ascension": (("m", "l"), np.full((n, n), 180.0)),
                    "declination": (("m", "l"), np.full((n, n), 45.0)),
                },
                attrs={"fits_wcs_header": hdr},
            )
            .assign(wcs_header_str=((), np.bytes_(hdr.encode("utf-8"))))
        )

    monkeypatch.setattr(mod, "_load_for_combine", fake_load)

    out = mod._load_global_lm_reference_dataset(
        by_time,
        tmp_path / "fixed",
        chunk_lm=0,
        fix_headers_on_demand=True,
    )

    assert loaded == [f_large]
    assert int(out.sizes["m"]) == 64
    assert int(out.sizes["l"]) == 64


def test_load_global_lm_reference_passes_target_size_to_resample(monkeypatch, tmp_path: Path) -> None:
    """When ``target_size`` is set, the reference dataset is passed through resampling."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    f_small = tmp_path / "small.fits"
    f_large = tmp_path / "large.fits"
    f_small.touch()
    f_large.touch()
    by_time = {"20240101_120000": [f_small, f_large]}

    monkeypatch.setattr(mod, "fix_fits_headers", lambda files, fd, skip_existing=True: list(files))

    def fake_getheader(fp: Path) -> fits.Header:
        hdr = fits.Header()
        hdr["NAXIS"] = 2
        n = 32 if fp.name.startswith("small") else 64
        hdr["NAXIS1"] = n
        hdr["NAXIS2"] = n
        hdr["BMAJ"] = 0.1
        hdr["BMIN"] = 0.1
        return hdr

    monkeypatch.setattr(mod, "_getheader_for_ingest", fake_getheader)

    def fake_load(fp: Path, chunk_lm: int = 1024) -> xr.Dataset:
        n = 64
        l_ = np.linspace(-1.0, 1.0, n)
        m_ = np.linspace(-1.0, 1.0, n)
        sky = np.zeros((n, n))
        hdr = "SIMPLE  =                   T\nNAXIS   =                    2"
        return (
            xr.Dataset(
                {"SKY": (("m", "l"), sky)},
                coords={
                    "l": ("l", l_),
                    "m": ("m", m_),
                    "frequency": ("frequency", np.array([1.4e8])),
                    "right_ascension": (("m", "l"), np.full((n, n), 180.0)),
                    "declination": (("m", "l"), np.full((n, n), 45.0)),
                },
                attrs={"fits_wcs_header": hdr},
            )
            .assign(wcs_header_str=((), np.bytes_(hdr.encode("utf-8"))))
        )

    calls: list[tuple[Path, int]] = []

    def fake_resample(xds: xr.Dataset, ref_fp: Path, *, target_size: int, chunk_lm: int) -> xr.Dataset:
        calls.append((ref_fp, int(target_size)))
        return xds

    monkeypatch.setattr(mod, "_load_for_combine", fake_load)
    monkeypatch.setattr(mod, "_resample_lm_reference_to_target_size", fake_resample)

    mod._load_global_lm_reference_dataset(
        by_time,
        tmp_path / "fixed",
        chunk_lm=0,
        fix_headers_on_demand=True,
        target_size=4096,
    )

    assert len(calls) == 1
    assert calls[0][1] == 4096
    assert calls[0][0].name == f_large.name


def test_load_global_lm_reference_respects_max_time_groups(monkeypatch, tmp_path: Path) -> None:
    """Only the first *max_time_groups* observation times are header-scanned."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    by_time = {
        "20240101_120000": [tmp_path / "t1.fits"],
        "20240101_120100": [tmp_path / "t2.fits"],
        "20240101_120200": [tmp_path / "t3.fits"],
    }
    for files in by_time.values():
        files[0].touch()

    peeked: list[str] = []

    def fake_peek(fp: Path) -> tuple[int, int]:
        peeked.append(fp.name)
        return (64, 64)

    monkeypatch.setattr(mod, "fix_fits_headers", lambda files, fd, skip_existing=True: list(files))
    monkeypatch.setattr(mod, "_peek_lm_shape", fake_peek)

    def fake_load(fp: Path, chunk_lm: int = 1024) -> xr.Dataset:
        n = 64
        l_ = np.linspace(-1.0, 1.0, n)
        m_ = np.linspace(-1.0, 1.0, n)
        hdr = "SIMPLE  =                   T\nNAXIS   =                    2"
        return (
            xr.Dataset(
                {"SKY": (("m", "l"), np.zeros((n, n)))},
                coords={"l": ("l", l_), "m": ("m", m_), "frequency": ("frequency", np.array([1.4e8]))},
                attrs={"fits_wcs_header": hdr},
            )
            .assign(wcs_header_str=((), np.bytes_(hdr.encode("utf-8"))))
        )

    monkeypatch.setattr(mod, "_load_for_combine", fake_load)

    mod._load_global_lm_reference_dataset(
        by_time,
        tmp_path / "fixed",
        chunk_lm=0,
        fix_headers_on_demand=False,
        max_time_groups=2,
    )

    assert len(peeked) == 2
    assert all("t3" not in name for name in peeked)


def test_lm_shape_from_header_reads_naxis(tmp_path: Path) -> None:
    import numpy as np

    mod = _import_module()
    data = np.zeros((5, 7), dtype=np.float32)
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = 7
    hdr["NAXIS2"] = 5
    fpath = tmp_path / "shape.fits"
    fits.writeto(fpath, data, hdr, overwrite=True)
    assert mod._lm_shape_from_header(fits.getheader(fpath)) == (7, 5)


def test_format_data_size_uses_binary_units() -> None:
    mod = _import_module()
    assert mod.format_data_size(0) == "0 B"
    assert mod.format_data_size(1023) == "1023 B"
    assert mod.format_data_size(1024) == "1.0 KiB"
    assert mod.format_data_size(42_000_000) == "40.1 MiB"


def test_estimate_zarr_store_bytes_uses_stokes_i_sample(tmp_path: Path) -> None:
    """Zarr size estimate scales the reference LM grid by the total file count."""
    import numpy as np

    mod = _import_module()

    def write_shape(path: Path, n_l: int, n_m: int) -> None:
        data = np.zeros((n_m, n_l), dtype=np.float32)
        hdr = fits.Header()
        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = n_l
        hdr["NAXIS2"] = n_m
        fits.writeto(path, data, hdr, overwrite=True)

    f1 = tmp_path / "82MHz-I-Taper-602s-Robust-0-20260419_071829-image.pbcorr_dewarped.fits"
    f2 = tmp_path / "82MHz-I-Taper-602s-Robust-0-20260419_072826-image.pbcorr_dewarped.fits"
    write_shape(f1, 10, 20)
    write_shape(f2, 30, 40)
    by_time = {"t1": [f1], "t2": [f2]}

    nbytes = mod.estimate_zarr_store_bytes(by_time)
    assert nbytes == 2 * 10 * 20 * mod._ZARR_ESTIMATE_BYTES_PER_PIXEL


def test_estimate_zarr_store_bytes_uses_largest_subband_in_time_group(tmp_path: Path) -> None:
    """Multi-subband time groups use the largest Stokes-I grid (global LM reference rule)."""
    import numpy as np

    mod = _import_module()

    def write_subband(path: Path, mhz: int, n_l: int, n_m: int) -> None:
        data = np.zeros((n_m, n_l), dtype=np.float32)
        hdr = fits.Header()
        hdr["NAXIS"] = 2
        hdr["NAXIS1"] = n_l
        hdr["NAXIS2"] = n_m
        hdr["RESTFREQ"] = float(mhz * 1e6)
        fits.writeto(path, data, hdr, overwrite=True)

    tkey = "20260419_071829"
    low = tmp_path / f"41MHz-I-Taper-602s-Robust-0-{tkey}-image.pbcorr_dewarped.fits"
    high = tmp_path / f"82MHz-I-Taper-602s-Robust-0-{tkey}-image.pbcorr_dewarped.fits"
    write_subband(low, 41, 100, 100)
    write_subband(high, 82, 200, 200)
    by_time = {tkey: [low, high]}

    nbytes = mod.estimate_zarr_store_bytes(by_time)
    assert nbytes == 2 * 200 * 200 * mod._ZARR_ESTIMATE_BYTES_PER_PIXEL


def test_stokes_label_from_dewarped_basename() -> None:
    mod = _import_module()
    assert (
        mod._stokes_label_from_basename(
            Path("82MHz-I-Taper-602s-Robust-0-20260419_071829-image.pbcorr_dewarped.fits")
        )
        == 1
    )
    assert (
        mod._stokes_label_from_basename(
            Path("82MHz-V-Taper-602s-Robust-0-20260419_071829-image.pbcorr_dewarped.fits")
        )
        == 4
    )


def test_discover_groups_keeps_dewarped_i_and_v_same_time_freq(tmp_path: Path) -> None:
    """Dewarped ``NNMHz-I-Taper`` and ``NNMHz-V-Taper`` products stay separate Stokes bins."""
    mod = _import_module()
    time_key = "20260419_071829"
    f_i = tmp_path / f"82MHz-I-Taper-602s-Robust-0-{time_key}-image.pbcorr_dewarped.fits"
    f_v = tmp_path / f"82MHz-V-Taper-602s-Robust-0-{time_key}-image.pbcorr_dewarped.fits"
    _write_ovro_stokes_fits(f_i, stokes=1)
    _write_ovro_stokes_fits(f_v, stokes=1)

    groups = mod._discover_groups(tmp_path)

    assert time_key in groups
    assert {p.name for p in groups[time_key]} == {f_i.name, f_v.name}


def test_assert_same_lm_clear_error_on_length_mismatch():
    """Length mismatch must raise RuntimeError, not a NumPy broadcast error."""
    import numpy as np

    fits_to_zarr_xradio = _import_module()

    ref = (np.linspace(-1, 1, 4096), np.linspace(-1, 1, 4096))
    cur = (np.linspace(-1, 1, 3122), np.linspace(-1, 1, 3122))
    with pytest.raises(RuntimeError, match="length mismatch"):
        fits_to_zarr_xradio._assert_same_lm(ref, cur)


def test_extract_group_metadata_from_header(tmp_path: Path):
    """Header metadata should provide both time and frequency."""
    mod = _import_module()
    fpath = tmp_path / "arbitrary_name.fits"
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-05-24T05:00:09.0", "RESTFREQ": 4.1e7}),
    ).writeto(fpath)

    time_key, frequency_hz, notes = mod._extract_group_metadata(fpath)

    assert time_key == "20240524_050009"
    assert frequency_hz == pytest.approx(4.1e7)
    assert notes == []


def test_extract_group_metadata_filename_time_overrides_header(tmp_path: Path) -> None:
    """Basename ``-image-`` stamp is the default time key and overrides DATE-OBS."""
    mod = _import_module()
    fpath = tmp_path / "18MHz-I-Deep-Taper-Robust-0-image-20241221_102109_x.fits"
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-12-21T00:00:00", "RESTFREQ": 18e6}),
    ).writeto(fpath)

    time_key, frequency_hz, notes = mod._extract_group_metadata(fpath)

    assert time_key == "20241221_102109"
    assert frequency_hz == pytest.approx(18e6)
    assert notes == []

    time_header, _, _ = mod._extract_group_metadata(fpath, time_key_source="header")
    assert time_header == "20241221_000000"


def test_discover_groups_filename_time_merges_same_image_id(tmp_path: Path) -> None:
    """Same ``-image-YYYYMMDD_HHMMSS`` basename groups together even if DATE-OBS differs."""
    mod = _import_module()
    a = tmp_path / "18MHz-I-Deep-Taper-Robust-0-image-20241221_102109_a.fits"
    b = tmp_path / "73MHz-I-Deep-Taper-Robust-0-image-20241221_102109_b.fits"
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-12-21T01:00:00", "RESTFREQ": 18e6}),
    ).writeto(a)
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-12-22T23:00:00", "RESTFREQ": 73e6}),
    ).writeto(b)

    by_header = mod._discover_groups(tmp_path, time_key_source="header")
    by_name = mod._discover_groups(tmp_path)

    assert len(by_header) == 2
    assert len(by_name) == 1
    assert list(by_name.keys()) == ["20241221_102109"]
    assert {p.name for p in by_name["20241221_102109"]} == {a.name, b.name}


def test_harmonize_subband_time_coords_collapses_mjd_spread() -> None:
    """Mismatched per-subband ``time`` coords must collapse before frequency stacking."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    mjd_a = 60662.13476852
    mjd_b = mjd_a + 30.0 / 86400.0

    def _slice(mjd: float) -> xr.Dataset:
        return xr.Dataset(
            {
                "SKY": (
                    ("time", "frequency", "polarization", "l", "m"),
                    np.zeros((1, 1, 1, 2, 2), dtype=np.float32),
                )
            },
            coords={
                "time": ("time", [mjd]),
                "frequency": ("frequency", [41e6]),
                "polarization": ("polarization", [1]),
                "l": ("l", [0, 1]),
                "m": ("m", [0, 1]),
            },
        )

    harmonized = mod._harmonize_subband_time_coords_for_stack([_slice(mjd_a), _slice(mjd_b)])
    assert float(harmonized[0]["time"].values[0]) == pytest.approx(mjd_a)
    assert float(harmonized[1]["time"].values[0]) == pytest.approx(mjd_a)


def test_discover_groups_filename_only_skips_getheader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filename-only discovery uses basename time and ``_NNNMHz_`` without ``getheader``."""
    mod = _import_module()

    def boom(*_a: object, **_k: object) -> None:
        pytest.fail("fits.getheader should not be called in filename-only discovery")

    monkeypatch.setattr(mod.fits, "getheader", boom)

    a = tmp_path / "18MHz-I-Deep-Taper-Robust-0-image-20241221_102109_a.fits"
    b = tmp_path / "73MHz-I-Deep-Taper-Robust-0-image-20241221_102109_b.fits"
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-12-21T01:00:00", "RESTFREQ": 99e6}),
    ).writeto(a)
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-12-22T23:00:00", "RESTFREQ": 1e6}),
    ).writeto(b)

    groups = mod._discover_groups(tmp_path, group_metadata_source="filename")

    assert list(groups.keys()) == ["20241221_102109"]
    assert [p.name for p in groups["20241221_102109"]] == [a.name, b.name]


def test_extract_group_metadata_requires_date_obs(tmp_path: Path):
    """OVRO-style name without ``-image-`` stamp needs DATE-OBS for the time key."""
    mod = _import_module()
    fpath = tmp_path / "20240524_050009_41MHz_averaged_20000_iterations-I-image.fits"
    fits.PrimaryHDU(data=[[1.0]], header=fits.Header({"SIMPLE": True})).writeto(fpath)

    time_key, frequency_hz, notes = mod._extract_group_metadata(fpath)

    assert time_key is None
    assert frequency_hz == pytest.approx(4.1e7)
    assert "frequency-from-filename" in notes


def test_extract_group_metadata_filename_time_overrides_header(tmp_path: Path) -> None:
    """``time_key_source='filename'`` prefers ``-image-`` basename over DATE-OBS."""
    mod = _import_module()
    fpath = tmp_path / "18MHz-I-Deep-Taper-Robust-0-image-20241221_102109_x.fits"
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-12-21T00:00:00", "RESTFREQ": 18e6}),
    ).writeto(fpath)

    time_key, frequency_hz, notes = mod._extract_group_metadata(fpath, time_key_source="filename")

    assert time_key == "20241221_102109"
    assert frequency_hz == pytest.approx(18e6)
    assert "time-from-filename" in notes


def test_discover_groups_filename_time_merges_same_image_id(tmp_path: Path) -> None:
    """Same ``-image-YYYYMMDD_HHMMSS`` basename groups together even if DATE-OBS differs."""
    mod = _import_module()
    a = tmp_path / "18MHz-I-Deep-Taper-Robust-0-image-20241221_102109_a.fits"
    b = tmp_path / "73MHz-I-Deep-Taper-Robust-0-image-20241221_102109_b.fits"
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-12-21T01:00:00", "RESTFREQ": 18e6}),
    ).writeto(a)
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-12-22T23:00:00", "RESTFREQ": 73e6}),
    ).writeto(b)

    by_header = mod._discover_groups(tmp_path, time_key_source="header")
    by_name = mod._discover_groups(tmp_path, time_key_source="filename")

    assert len(by_header) == 2
    assert len(by_name) == 1
    assert list(by_name.keys()) == ["20241221_102109"]
    assert {p.name for p in by_name["20241221_102109"]} == {a.name, b.name}


def test_discover_groups_duplicate_without_resolver_keeps_first(tmp_path: Path):
    """Same time + same discovery frequency bin: keep the first file and warn; do not stack duplicates."""
    mod = _import_module()
    hdr = fits.Header({"DATE-OBS": "2024-05-24T05:00:09.0", "RESTFREQ": 4.1e7})
    f1 = tmp_path / "first_name.fits"
    f2 = tmp_path / "second_name.fits"
    fits.PrimaryHDU(data=[[1.0]], header=hdr).writeto(f1)
    fits.PrimaryHDU(data=[[1.0]], header=hdr).writeto(f2)

    groups = mod._discover_groups(tmp_path)
    assert groups["20240524_050009"] == [f1]


def test_discover_groups_header_frequency_jitter_single_plane(tmp_path: Path):
    """RESTFREQ differing by <<23 kHz should map to one binned subband (one FITS kept)."""
    mod = _import_module()
    t = "2024-05-24T05:00:09.0"
    f1 = tmp_path / "a.fits"
    f2 = tmp_path / "b.fits"
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": t, "RESTFREQ": 4.1e7}),
    ).writeto(f1)
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": t, "RESTFREQ": 4.1e7 + 100.0}),
    ).writeto(f2)
    groups = mod._discover_groups(tmp_path)
    assert groups["20240524_050009"] == [f1]


def test_discover_groups_freq_bin_hz_controls_merge_window(tmp_path: Path):
    """Narrow bin (10 kHz) splits ~15 kHz RESTFREQ offset; 23 kHz bin merges as one subband."""
    mod = _import_module()
    t = "2024-05-24T05:00:09.0"
    f1 = tmp_path / "a.fits"
    f2 = tmp_path / "b.fits"
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": t, "RESTFREQ": 4.1e7}),
    ).writeto(f1)
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": t, "RESTFREQ": 4.1e7 + 15_000.0}),
    ).writeto(f2)

    merged = mod._discover_groups(tmp_path, freq_bin_hz=23_000.0)
    assert merged["20240524_050009"] == [f1]

    split = mod._discover_groups(tmp_path, freq_bin_hz=10_000.0)
    assert len(split["20240524_050009"]) == 2


def test_discover_groups_duplicate_with_resolver_selects_one(tmp_path: Path):
    """Resolver should choose one candidate for duplicate time/frequency groups."""
    mod = _import_module()
    hdr = fits.Header({"DATE-OBS": "2024-05-24T05:00:09.0", "RESTFREQ": 4.1e7})
    f1 = tmp_path / "candidate_a.fits"
    f2 = tmp_path / "candidate_b.fits"
    fits.PrimaryHDU(data=[[1.0]], header=hdr).writeto(f1)
    fits.PrimaryHDU(data=[[1.0]], header=hdr).writeto(f2)

    def choose_second(_time_key: str, _freq_hz: float, candidates: list[Path]) -> Path:
        return candidates[-1]

    groups = mod._discover_groups(tmp_path, duplicate_resolver=choose_second)

    assert "20240524_050009" in groups
    assert groups["20240524_050009"] == [f2]


def test_discover_groups_triple_duplicate_resolver_sees_fresh_candidates(tmp_path: Path):
    """After resolving two-way duplicate, third file must not reuse stale candidate list."""
    mod = _import_module()
    hdr = fits.Header({"DATE-OBS": "2024-05-24T05:00:09.0", "RESTFREQ": 4.1e7})
    paths = [tmp_path / f"candidate_{i}.fits" for i in range(3)]
    for p in paths:
        fits.PrimaryHDU(data=[[1.0]], header=hdr).writeto(p)

    resolver_calls: list[list[Path]] = []

    def record_first(_time_key: str, _freq_hz: float, candidates: list[Path]) -> Path:
        resolver_calls.append(list(candidates))
        return candidates[0]

    groups = mod._discover_groups(tmp_path, duplicate_resolver=record_first)

    assert groups["20240524_050009"] == [paths[0]]
    assert len(resolver_calls) == 2
    assert resolver_calls[0] == paths[:2]
    assert resolver_calls[1] == [paths[0], paths[2]]


def test_discover_groups_header_based_frequency_sorting(tmp_path: Path):
    """Groups should be deterministically sorted by header frequency."""
    mod = _import_module()
    # Write intentionally out-of-order names with opposite order frequencies.
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-05-24T05:00:09.0", "RESTFREQ": 8.2e7}),
    ).writeto(tmp_path / "aaa_name.fits")
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"DATE-OBS": "2024-05-24T05:00:09.0", "RESTFREQ": 4.1e7}),
    ).writeto(tmp_path / "zzz_name.fits")

    groups = mod._discover_groups(tmp_path)
    group_files = groups["20240524_050009"]

    freqs = [mod._extract_group_metadata(p)[1] for p in group_files]
    assert freqs == [pytest.approx(4.1e7), pytest.approx(8.2e7)]


def test_discover_groups_skips_file_without_time_or_frequency_metadata(tmp_path: Path):
    """Files with no usable header or filename metadata should be skipped."""
    mod = _import_module()
    fits.PrimaryHDU(data=[[1.0]], header=fits.Header({"SIMPLE": True})).writeto(
        tmp_path / "unparseable_name.fits"
    )

    groups = mod._discover_groups(tmp_path)

    assert groups == {}


def test_time_key_from_lst_color_filename(tmp_path: Path) -> None:
    """LST color-band basenames encode date, LST hour, and time bin."""
    mod = _import_module()
    name = "Blue_I_10min_Taper_Robust-0_pbcorr_20250508_LST22h_t0001.fits"
    assert mod._time_key_from_lst_color_filename(tmp_path / name) == "20250508_LST22h_t0001"
    assert mod._time_key_from_lst_color_filename(tmp_path / "no_match.fits") is None


def test_extract_group_metadata_lst_color(tmp_path: Path) -> None:
    """LST color-band metadata uses basename time and header frequency."""
    mod = _import_module()
    fpath = tmp_path / "Green_I_10min_Taper_Robust-0_pbcorr_20250508_LST22h_t0002.fits"
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"RESTFREQ": 55.0e6}),
    ).writeto(fpath)

    time_key, frequency_hz, notes = mod._extract_group_metadata_lst_color(fpath)

    assert time_key == "20250508_LST22h_t0002"
    assert frequency_hz == pytest.approx(55.0e6)
    assert "time-from-lst-color-filename" in notes
    assert "frequency-from-header" in notes


def test_discover_groups_lst_color_groups_subbands_by_time_and_header_mhz(
    tmp_path: Path,
) -> None:
    """Blue/Green/Red at the same LST bin group together; distinct bins stay separate."""
    mod = _import_module()
    blue = tmp_path / "Blue_I_10min_Taper_Robust-0_pbcorr_20250508_LST22h_t0001.fits"
    green = tmp_path / "Green_I_10min_Taper_Robust-0_pbcorr_20250508_LST22h_t0001.fits"
    red_other_bin = tmp_path / "Red_I_10min_Taper_Robust-0_pbcorr_20250508_LST22h_t0002.fits"
    for path, hz in ((blue, 41e6), (green, 55e6), (red_other_bin, 73e6)):
        fits.PrimaryHDU(
            data=[[1.0]],
            header=fits.Header({"RESTFREQ": hz}),
        ).writeto(path)

    groups = mod._discover_groups(tmp_path, filename_convention="lst-color")

    assert set(groups.keys()) == {"20250508_LST22h_t0001", "20250508_LST22h_t0002"}
    assert {p.name for p in groups["20250508_LST22h_t0001"]} == {blue.name, green.name}
    assert groups["20250508_LST22h_t0002"] == [red_other_bin]


def test_discover_groups_lst_color_header_frequency_jitter_single_plane(tmp_path: Path) -> None:
    """Header MHz jitter within the discovery bin merges duplicate subbands."""
    mod = _import_module()
    f1 = tmp_path / "Blue_I_10min_Taper_Robust-0_pbcorr_20250508_LST22h_t0001.fits"
    f2 = tmp_path / "Blue_I_10min_Taper_Robust-0_pbcorr_20250508_LST22h_t0001_dup.fits"
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"RESTFREQ": 41.0e6}),
    ).writeto(f1)
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"RESTFREQ": 41.0e6 + 100.0}),
    ).writeto(f2)

    groups = mod._discover_groups(tmp_path, filename_convention="lst-color")

    assert groups["20250508_LST22h_t0001"] == [f1]


def test_discover_groups_lst_color_rejects_filename_metadata_source(tmp_path: Path) -> None:
    """LST color grouping requires FITS headers for subband frequency."""
    mod = _import_module()

    with pytest.raises(ValueError, match='use group_metadata_source="fits"'):
        mod._discover_groups(
            tmp_path,
            filename_convention="lst-color",
            group_metadata_source="filename",
        )


def test_rechunk_lm_for_zarr_uniform_spatial_chunks():
    """Irregular dask chunks along l/m must become uniform for Zarr compatibility."""
    import dask.array as da
    import numpy as np
    import xarray as xr

    mod = _import_module()
    arr = np.random.default_rng(0).random((512, 512))
    data = da.from_array(arr, chunks=((256, 256), (256, 128, 128)))
    l = np.linspace(-1.0, 1.0, 512)
    m = np.linspace(-1.0, 1.0, 512)
    ds = xr.Dataset(
        {"SKY": (("m", "l"), data)},
        coords={"l": ("l", l), "m": ("m", m)},
    )
    out = mod._rechunk_lm_for_zarr(ds, chunk_lm=256)
    chunks = out["SKY"].data.chunks
    assert chunks[0] == (256, 256)
    assert chunks[1] == (256, 256)


def test_rechunk_lm_for_zarr_chunk_lm_zero_single_spatial_chunk():
    """chunk_lm=0 should use one chunk per spatial axis (still uniform)."""
    import dask.array as da
    import numpy as np
    import xarray as xr

    mod = _import_module()
    arr = np.random.default_rng(1).random((100, 100))
    data = da.from_array(arr, chunks=((50, 50), (50, 50)))
    l = np.linspace(-1.0, 1.0, 100)
    m = np.linspace(-1.0, 1.0, 100)
    ds = xr.Dataset({"SKY": (("m", "l"), data)}, coords={"l": ("l", l), "m": ("m", m)})
    out = mod._rechunk_lm_for_zarr(ds, chunk_lm=0)
    assert out["SKY"].data.chunks == ((100,), (100,))


def test_rechunk_lm_for_zarr_fixes_irregular_wcs_header_str_chunks(tmp_path):
    """Non-uniform dask chunks on aux vars (e.g. wcs along frequency) must be Zarr-safe."""
    import dask.array as da
    import numpy as np
    import xarray as xr

    mod = _import_module()
    n = 4
    l = np.linspace(-1.0, 1.0, 32)
    m = np.linspace(-1.0, 1.0, 32)
    sky = da.random.random((32, 32), chunks=(16, 16))
    hdr = np.array([np.bytes_(b"x" * 20) for _ in range(n)], dtype=np.bytes_)
    w = da.from_array(hdr, chunks=((2, 1, 1),))
    ds = xr.Dataset(
        {"SKY": (("m", "l"), sky), "wcs_header_str": (("frequency",), w)},
        coords={
            "l": ("l", l),
            "m": ("m", m),
            "frequency": np.arange(n, dtype=np.float64),
        },
    )
    out = mod._rechunk_lm_for_zarr(ds, chunk_lm=8)
    assert out["wcs_header_str"].data.chunks == ((4,),)
    out.to_zarr(tmp_path / "t.zarr", mode="w", consolidated=False)


def test_rechunk_lm_for_zarr_strips_coord_encoding_conflicts(tmp_path):
    """Stale ``encoding['chunks']`` on coords must not break ``to_zarr`` (Dask vs Zarr grid)."""
    import dask.array as da
    import numpy as np
    import xarray as xr

    mod = _import_module()
    nf, ny, nx = 2, 32, 32
    sky = da.random.random((nf, ny, nx), chunks=(1, 16, 16))
    ra = da.random.random((nf, ny, nx), chunks=(1, 16, 16))
    dec = da.random.random((nf, ny, nx), chunks=(1, 16, 16))
    ds = xr.Dataset(
        {"SKY": (("frequency", "m", "l"), sky)},
        coords={
            "frequency": np.arange(nf),
            "l": np.linspace(-1, 1, nx),
            "m": np.linspace(-1, 1, ny),
            "right_ascension": (("frequency", "m", "l"), ra),
            "declination": (("frequency", "m", "l"), dec),
        },
    )
    ds["right_ascension"].encoding = {"chunks": (2, 128, 128)}
    out = mod._rechunk_lm_for_zarr(ds, chunk_lm=8)
    assert out["right_ascension"].encoding == {}
    assert out["declination"].encoding == {}
    out.to_zarr(tmp_path / "coord.zarr", mode="w", consolidated=False)


def test_rechunk_lm_for_zarr_fixes_nonuniform_coord_time_chunks(tmp_path):
    """Coords with time chunks like (2,1,1) should be rechunked to Zarr-safe layout."""
    import dask.array as da
    import numpy as np
    import xarray as xr

    mod = _import_module()
    nt, ny, nx = 4, 32, 32
    sky = da.random.random((nt, ny, nx), chunks=(1, 16, 16))
    # Deliberately non-uniform time chunks to match runtime failure.
    ra_np = np.random.default_rng(2).random((nt, ny, nx))
    dec_np = np.random.default_rng(3).random((nt, ny, nx))
    ra = da.from_array(ra_np, chunks=((2, 1, 1), (16, 16), (16, 16)))
    dec = da.from_array(dec_np, chunks=((2, 1, 1), (16, 16), (16, 16)))

    ds = xr.Dataset(
        {"SKY": (("time", "m", "l"), sky)},
        coords={
            "time": np.arange(nt),
            "l": np.linspace(-1, 1, nx),
            "m": np.linspace(-1, 1, ny),
            "right_ascension": (("time", "m", "l"), ra),
            "declination": (("time", "m", "l"), dec),
        },
    )
    out = mod._rechunk_lm_for_zarr(ds, chunk_lm=8)
    assert hasattr(out["right_ascension"].data, "chunks")
    assert out["right_ascension"].data.chunks[0] == (4,)
    out.to_zarr(tmp_path / "coord_nonuniform_time.zarr", mode="w", consolidated=False)


def test_discover_groups_filename_fallback_compatibility(tmp_path: Path):
    """Files missing DATE-OBS and the ``-image-`` time stamp should be skipped."""
    mod = _import_module()
    fits.PrimaryHDU(data=[[1.0]], header=fits.Header({"SIMPLE": True})).writeto(
        tmp_path / "20240524_050009_41MHz_averaged_20000_iterations-I-image.fits"
    )
    fits.PrimaryHDU(data=[[1.0]], header=fits.Header({"SIMPLE": True})).writeto(
        tmp_path / "20240524_050009_82MHz_averaged_20000_iterations-I-image.fits"
    )

    groups = mod._discover_groups(tmp_path)
    assert groups == {}


def test_invalid_beam_reason_missing_keywords():
    """Headers without BMAJ or BMIN should be reported as missing."""
    mod = _import_module()
    assert mod._invalid_beam_reason(fits.Header({})) == "missing BMAJ/BMIN"
    assert mod._invalid_beam_reason(fits.Header({"BMAJ": 0.1})) == "missing BMIN"
    assert mod._invalid_beam_reason(fits.Header({"BMIN": 0.1})) == "missing BMAJ"


def test_invalid_beam_reason_zero_or_negative():
    """Zero or negative beam axes should each be flagged."""
    mod = _import_module()
    assert mod._invalid_beam_reason(fits.Header({"BMAJ": 0.0, "BMIN": 0.1})) == "BMAJ=0.0"
    assert mod._invalid_beam_reason(fits.Header({"BMAJ": 0.1, "BMIN": 0.0})) == "BMIN=0.0"
    reason_both_zero = mod._invalid_beam_reason(fits.Header({"BMAJ": 0.0, "BMIN": 0.0}))
    assert "BMAJ=0.0" in reason_both_zero and "BMIN=0.0" in reason_both_zero
    assert mod._invalid_beam_reason(fits.Header({"BMAJ": -1.0, "BMIN": 0.1})) == "BMAJ=-1.0"


def test_invalid_beam_reason_non_finite_value():
    """Non-finite stand-ins (handled defensively even though astropy rejects NaN in headers)."""
    import numpy as _np

    mod = _import_module()

    class _NonFiniteHeader:
        def __init__(self, mapping):
            self._mapping = mapping

        def __contains__(self, key):
            return key in self._mapping

        def __getitem__(self, key):
            return self._mapping[key]

    hdr = _NonFiniteHeader({"BMAJ": _np.inf, "BMIN": 0.1})
    assert mod._invalid_beam_reason(hdr) == "BMAJ=inf"

    hdr2 = _NonFiniteHeader({"BMAJ": "not_a_number", "BMIN": 0.1})
    assert mod._invalid_beam_reason(hdr2).startswith("BMAJ=<non-numeric:")


def test_invalid_beam_reason_valid_beam_returns_none():
    """Finite, strictly positive BMAJ/BMIN means the file should be kept."""
    mod = _import_module()
    assert mod._invalid_beam_reason(fits.Header({"BMAJ": 0.1, "BMIN": 0.05})) is None
    assert mod._invalid_beam_reason(fits.Header({"BMAJ": 1e-6, "BMIN": 1e-6})) is None


def test_fix_headers_reads_image_hdu_when_primary_empty(tmp_path: Path) -> None:
    """Fpacked-style empty primary + image extension must expose BMAJ/BMIN to header fix."""
    mod = _import_module()

    img_hdr = fits.Header(
        {
            "NAXIS": 2,
            "NAXIS1": 8,
            "NAXIS2": 8,
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CRVAL1": 180.0,
            "CRVAL2": 45.0,
            "CRPIX1": 4.0,
            "CRPIX2": 4.0,
            "CDELT1": -0.1,
            "CDELT2": 0.1,
            "BMAJ": 0.2,
            "BMIN": 0.1,
            "BPA": 15.0,
        }
    )
    primary = fits.PrimaryHDU(header=fits.Header({"SIMPLE": True, "BITPIX": -32, "NAXIS": 0}))
    image = fits.ImageHDU(data=np.ones((8, 8), dtype=np.float32), header=img_hdr)
    src = tmp_path / "64MHz-Clean-Snapshot-20250120_040013-image.fits.fs"
    fits.HDUList([primary, image]).writeto(src)

    fixed_dir = tmp_path / "fixed"
    fixed = mod.fix_fits_headers([src], fixed_dir)
    assert len(fixed) == 1
    with fits.open(fixed[0]) as hdul:
        hdr = hdul[0].header
    assert float(hdr["BMAJ"]) == 0.2
    assert float(hdr["BMIN"]) == 0.1


def test_repair_zero_beam_from_nearby_time_same_frequency(tmp_path: Path) -> None:
    """Placeholder beam should be filled from the nearest other time at the same MHz."""
    mod = _import_module()

    donor_time = "20250120_040000"
    zero_time = "20250120_040343"
    donor = tmp_path / f"55MHz-Clean-Snapshot-{donor_time}-image.fits"
    zero_src = tmp_path / f"55MHz-Clean-Snapshot-{zero_time}-image.fits"
    other_band = tmp_path / f"18MHz-Clean-Snapshot-{donor_time}-image.fits"
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"BMAJ": 0.2, "BMIN": 0.1, "BPA": 30.0}),
    ).writeto(donor)
    fits.PrimaryHDU(
        data=[[2.0]],
        header=fits.Header({"BMAJ": 0.0, "BMIN": 0.0, "BPA": 0.0}),
    ).writeto(zero_src)
    fits.PrimaryHDU(
        data=[[3.0]],
        header=fits.Header({"BMAJ": 0.5, "BMIN": 0.4, "BPA": 10.0}),
    ).writeto(other_band)

    zero_out = tmp_path / "out_zero.fits"
    fits.PrimaryHDU(
        data=[[2.0]],
        header=fits.Header({"BMAJ": 0.0, "BMIN": 0.0, "BPA": 0.0}),
    ).writeto(zero_out, overwrite=True)

    by_time = {
        donor_time: [donor, other_band],
        zero_time: [zero_src],
    }
    n = mod.repair_zero_beam_from_nearby_time(
        [zero_src],
        [zero_out],
        zero_time,
        by_time,
    )
    assert n == 1
    with fits.open(zero_out) as hdul:
        hdr = hdul[0].header
    assert float(hdr["BMAJ"]) == 0.2
    assert float(hdr["BMIN"]) == 0.1
    assert float(hdr["BPA"]) == 30.0


def test_filter_invalid_beam_files_drops_zero_and_missing(tmp_path: Path, caplog):
    """Files with missing or zero BMAJ/BMIN must be dropped with a warning."""
    import logging

    mod = _import_module()

    good = tmp_path / "good.fits"
    bad_missing = tmp_path / "bad_missing.fits"
    bad_zero = tmp_path / "bad_zero.fits"
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"BMAJ": 0.1, "BMIN": 0.1}),
    ).writeto(good)
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({}),
    ).writeto(bad_missing)
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"BMAJ": 0.0, "BMIN": 0.0}),
    ).writeto(bad_zero)

    caplog.set_level(logging.WARNING, logger="ovro_lwa_portal.fits_to_zarr_xradio")
    by_time = {"20240524_050009": [good, bad_missing, bad_zero]}
    filtered = mod._filter_invalid_beam_files(by_time)

    assert filtered == {"20240524_050009": [good]}
    assert "bad_missing.fits" in caplog.text
    assert "bad_zero.fits" in caplog.text
    assert "missing BMAJ/BMIN" in caplog.text
    assert "BMAJ=0.0" in caplog.text


def test_filter_invalid_beam_files_drops_empty_time_keys(tmp_path: Path, caplog):
    """Time keys whose files all fail the beam check must be removed entirely."""
    import logging

    mod = _import_module()

    bad_a = tmp_path / "bad_a.fits"
    bad_b = tmp_path / "bad_b.fits"
    good_c = tmp_path / "good_c.fits"
    fits.PrimaryHDU(data=[[1.0]], header=fits.Header({"BMAJ": 0.0, "BMIN": 0.0})).writeto(bad_a)
    fits.PrimaryHDU(data=[[1.0]], header=fits.Header({})).writeto(bad_b)
    fits.PrimaryHDU(data=[[1.0]], header=fits.Header({"BMAJ": 0.1, "BMIN": 0.1})).writeto(good_c)

    caplog.set_level(logging.WARNING, logger="ovro_lwa_portal.fits_to_zarr_xradio")
    by_time = {
        "t_all_bad": [bad_a, bad_b],
        "t_one_good": [bad_a, good_c],
    }
    filtered = mod._filter_invalid_beam_files(by_time)

    assert "t_all_bad" not in filtered
    assert filtered.get("t_one_good") == [good_c]
    assert "t_all_bad" in caplog.text


def test_filter_invalid_beam_files_logs_unreadable_files(tmp_path: Path, caplog):
    """Missing or unreadable files should drop the file rather than abort the run."""
    import logging

    mod = _import_module()

    real = tmp_path / "real.fits"
    fits.PrimaryHDU(data=[[1.0]], header=fits.Header({"BMAJ": 0.1, "BMIN": 0.1})).writeto(real)
    missing = tmp_path / "does_not_exist.fits"

    caplog.set_level(logging.WARNING, logger="ovro_lwa_portal.fits_to_zarr_xradio")
    filtered = mod._filter_invalid_beam_files({"t": [real, missing]})

    assert filtered == {"t": [real]}
    assert "does_not_exist.fits" in caplog.text
    assert "cannot stat file" in caplog.text


def test_truncated_fits_reason_detects_short_file(tmp_path: Path) -> None:
    """Partial files must be flagged before image data is read."""
    import numpy as np

    mod = _import_module()
    full = tmp_path / "full.fits"
    data = np.zeros((8, 8), dtype=np.float32)
    fits.PrimaryHDU(
        data=data,
        header=fits.Header({"BMAJ": 0.1, "BMIN": 0.1, "BITPIX": -32, "NAXIS": 2, "NAXIS1": 8, "NAXIS2": 8}),
    ).writeto(full)
    assert mod._truncated_fits_reason(full) is None

    with fits.open(full, memmap=True) as hdul:
        required = int(hdul[0]._data_offset) + int(hdul[0].size)
    short = tmp_path / "short.fits"
    short.write_bytes(full.read_bytes()[: required - 100])
    reason = mod._truncated_fits_reason(short)
    assert reason is not None
    assert "truncated" in reason.lower()


def test_truncated_fits_reason_detects_empty_file(tmp_path: Path) -> None:
    """Zero-byte paths are treated as corrupt."""
    mod = _import_module()
    empty = tmp_path / "empty.fits"
    empty.write_bytes(b"")
    assert mod._truncated_fits_reason(empty) == "empty file (0 bytes)"


def test_filter_invalid_beam_files_drops_truncated(tmp_path: Path, caplog) -> None:
    """Truncated FITS must be dropped during discovery with a clear warning."""
    import logging
    import numpy as np

    mod = _import_module()
    good = tmp_path / "good.fits"
    fits.PrimaryHDU(
        data=np.zeros((4, 4), dtype=np.float32),
        header=fits.Header({"BMAJ": 0.1, "BMIN": 0.1, "BITPIX": -32, "NAXIS": 2, "NAXIS1": 4, "NAXIS2": 4}),
    ).writeto(good)
    with fits.open(good, memmap=True) as hdul:
        required = int(hdul[0]._data_offset) + int(hdul[0].size)
    bad = tmp_path / "bad.fits"
    bad.write_bytes(good.read_bytes()[: required - 100])

    caplog.set_level(logging.WARNING, logger="ovro_lwa_portal.fits_to_zarr_xradio")
    filtered = mod._filter_invalid_beam_files({"20240524_050009": [good, bad]})

    assert filtered == {"20240524_050009": [good]}
    assert "bad.fits" in caplog.text
    assert "truncated" in caplog.text.lower()


def test_filter_lst_color_groups_drops_mismatched_header_times(tmp_path: Path, caplog) -> None:
    """Lst-color groups with conflicting DATE-OBS across subbands must be excluded."""
    import logging

    mod = _import_module()
    stem = "20241218_LST02h_t0002"
    good_paths = []
    for color, date_obs in (
        ("Blue", "2024-12-18T04:09:01.0"),
        ("Green", "2024-12-18T04:09:01.0"),
        ("Red", "2024-12-18T04:09:01.0"),
    ):
        fp = tmp_path / f"{color}_I_10min_Taper_Robust-0_pbcorr_dewarped_{stem}.fits"
        fits.PrimaryHDU(
            data=[[1.0]],
            header=fits.Header({"DATE-OBS": date_obs, "RESTFRQ": 4.1e7}),
        ).writeto(fp)
        good_paths.append(fp)

    bad_paths = []
    for color, date_obs in (
        ("Blue", "2024-12-18T04:09:01.0"),
        ("Green", "2024-12-18T04:18:58.0"),
        ("Red", "2024-12-18T04:09:01.0"),
    ):
        fp = tmp_path / f"{color}_bad_{stem}.fits"
        fits.PrimaryHDU(
            data=[[1.0]],
            header=fits.Header({"DATE-OBS": date_obs, "RESTFRQ": 4.1e7}),
        ).writeto(fp)
        bad_paths.append(fp)

    by_time = {
        "20241218_LST02h_t0001": good_paths,
        stem: bad_paths,
    }
    caplog.set_level(logging.WARNING, logger="ovro_lwa_portal.fits_to_zarr_xradio")
    filtered = mod._filter_lst_color_groups_with_mismatched_header_times(by_time)

    assert list(filtered.keys()) == ["20241218_LST02h_t0001"]
    assert filtered["20241218_LST02h_t0001"] == good_paths
    assert stem not in filtered
    assert "Dropping lst-color time group" in caplog.text
    assert "20241218_040901" in caplog.text
    assert "20241218_041858" in caplog.text


def test_fix_headers_raises_invalid_beam_error_on_missing_beam(tmp_path: Path):
    """``_fix_headers`` must refuse to invent a placeholder beam for unfit images."""
    import numpy as _np

    mod = _import_module()
    in_path = tmp_path / "no_beam.fits"
    out_path = tmp_path / "no_beam_fixed.fits"

    data = _np.zeros((1, 1, 4, 4), dtype=_np.float32)
    header = fits.Header(
        {
            "NAXIS": 4,
            "NAXIS1": 4,
            "NAXIS2": 4,
            "NAXIS3": 1,
            "NAXIS4": 1,
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CTYPE3": "FREQ",
            "CTYPE4": "STOKES",
            "CRVAL3": 4.1e7,
            "CRPIX3": 1.0,
            "CDELT3": 1.0,
            "CRVAL4": 1.0,
            "CRPIX4": 1.0,
            "CDELT4": 1.0,
        }
    )
    fits.PrimaryHDU(data=data, header=header).writeto(in_path)

    with pytest.raises(mod.InvalidBeamError, match="missing BMAJ/BMIN"):
        mod._fix_headers(in_path, out_path)

    assert not out_path.exists(), "Output FITS must not be written for invalid-beam inputs."


def test_fix_headers_raises_invalid_beam_error_on_zero_beam(tmp_path: Path):
    """Zero ``BMAJ``/``BMIN`` must also be rejected — no placeholder beam ever lands on disk."""
    import numpy as _np

    mod = _import_module()
    in_path = tmp_path / "zero_beam.fits"
    out_path = tmp_path / "zero_beam_fixed.fits"

    data = _np.zeros((1, 1, 4, 4), dtype=_np.float32)
    header = fits.Header(
        {
            "NAXIS": 4,
            "NAXIS1": 4,
            "NAXIS2": 4,
            "NAXIS3": 1,
            "NAXIS4": 1,
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CTYPE3": "FREQ",
            "CTYPE4": "STOKES",
            "CRVAL3": 4.1e7,
            "CRPIX3": 1.0,
            "CDELT3": 1.0,
            "CRVAL4": 1.0,
            "CRPIX4": 1.0,
            "CDELT4": 1.0,
            "BMAJ": 0.0,
            "BMIN": 0.0,
        }
    )
    fits.PrimaryHDU(data=data, header=header).writeto(in_path)

    with pytest.raises(mod.InvalidBeamError, match="BMAJ=0.0"):
        mod._fix_headers(in_path, out_path)


def test_fix_fits_headers_skips_invalid_beam_and_returns_only_valid(tmp_path: Path, caplog):
    """``fix_fits_headers`` must drop invalid-beam files, log a warning, and clean partials."""
    import logging

    import numpy as _np

    mod = _import_module()

    def _write(path: Path, *, with_beam: bool) -> None:
        h = fits.Header(
            {
                "NAXIS": 2,
                "NAXIS1": 4,
                "NAXIS2": 4,
                "CTYPE1": "RA---SIN",
                "CTYPE2": "DEC--SIN",
                "CRVAL1": 180.0,
                "CRVAL2": 45.0,
                "CRPIX1": 2.5,
                "CRPIX2": 2.5,
                "CDELT1": -0.03,
                "CDELT2": 0.03,
                "CUNIT1": "deg",
                "CUNIT2": "deg",
            }
        )
        if with_beam:
            h["BMAJ"] = 0.1
            h["BMIN"] = 0.1
        fits.PrimaryHDU(data=_np.zeros((4, 4), dtype=_np.float32), header=h).writeto(path)

    good = tmp_path / "good_70MHz.fits"
    bad = tmp_path / "bad_74MHz.fits"
    _write(good, with_beam=True)
    _write(bad, with_beam=False)

    fixed_dir = tmp_path / "fixed"
    caplog.set_level(logging.WARNING, logger="ovro_lwa_portal.fits_to_zarr_xradio")
    fixed_paths = mod.fix_fits_headers([good, bad], fixed_dir, skip_existing=False)

    assert fixed_paths == [fixed_dir / "good_70MHz_fixed.fits"]
    assert (fixed_dir / "good_70MHz_fixed.fits").exists()
    assert not (fixed_dir / "bad_74MHz_fixed.fits").exists()
    assert "bad_74MHz.fits" in caplog.text
    assert "missing BMAJ/BMIN" in caplog.text


def test_fix_headers_preserves_real_beam_when_present(tmp_path: Path):
    """``_fix_headers`` must keep the input's real synthesized beam, not clobber it."""
    import numpy as _np

    mod = _import_module()
    in_path = tmp_path / "input.fits"
    out_path = tmp_path / "output_fixed.fits"

    data = _np.zeros((1, 1, 4, 4), dtype=_np.float32)
    header = fits.Header(
        {
            "NAXIS": 4,
            "NAXIS1": 4,
            "NAXIS2": 4,
            "NAXIS3": 1,
            "NAXIS4": 1,
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CTYPE3": "FREQ",
            "CTYPE4": "STOKES",
            "CRVAL3": 4.1e7,
            "CRPIX3": 1.0,
            "CDELT3": 1.0,
            "CRVAL4": 1.0,
            "CRPIX4": 1.0,
            "CDELT4": 1.0,
            "BMAJ": 0.0421,
            "BMIN": 0.0317,
            "BPA": 12.5,
        }
    )
    fits.PrimaryHDU(data=data, header=header).writeto(in_path)
    mod._fix_headers(in_path, out_path)

    out_hdr = fits.getheader(out_path, ext=0)
    assert out_hdr["BMAJ"] == pytest.approx(0.0421)
    assert out_hdr["BMIN"] == pytest.approx(0.0317)
    assert out_hdr["BPA"] == pytest.approx(12.5)


def test_fix_headers_adds_stokes_axis_when_missing(tmp_path: Path):
    """Header fixing should add a singleton STOKES axis for 3D FREQ-only cubes."""
    import numpy as np

    mod = _import_module()
    in_path = tmp_path / "input.fits"
    out_path = tmp_path / "output_fixed.fits"

    data = np.zeros((1, 4, 4), dtype=np.float32)
    header = fits.Header(
        {
            "NAXIS": 3,
            "NAXIS1": 4,
            "NAXIS2": 4,
            "NAXIS3": 1,
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CTYPE3": "FREQ",
            "CRVAL3": 4.1e7,
            "CRPIX3": 1.0,
            "CDELT3": 1.0,
            "CUNIT3": "Hz",
            "BMAJ": 0.1,
            "BMIN": 0.1,
        }
    )
    fits.PrimaryHDU(data=data, header=header).writeto(in_path)

    mod._fix_headers(in_path, out_path)

    with fits.open(out_path) as hdul:
        hdr = hdul[0].header
        out_data = hdul[0].data

    assert hdr["NAXIS"] == 4
    assert hdr["CTYPE4"] == "STOKES"
    assert hdr["CRVAL4"] == pytest.approx(1.0)
    assert hdr["CRPIX4"] == pytest.approx(1.0)
    assert out_data.shape == (1, 1, 4, 4)


def test_fix_headers_normalizes_string_stokes_crval_on_2d_promotion(tmp_path: Path) -> None:
    """2D images with leftover ``CRVAL4='I'`` must promote to numeric Stokes codes."""
    mod = _import_module()
    in_path = tmp_path / "82MHz-I-Taper-test.fits"
    out_path = tmp_path / "82MHz-I-Taper-test_fixed.fits"

    data = np.ones((4, 4), dtype=np.float32)
    header = fits.Header(
        {
            "NAXIS": 2,
            "NAXIS1": 4,
            "NAXIS2": 4,
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CRVAL1": 180.0,
            "CRVAL2": 45.0,
            "CRPIX1": 2.5,
            "CRPIX2": 2.5,
            "CDELT1": -0.01,
            "CDELT2": 0.01,
            "CRVAL4": "I",
            "RESTFREQ": 82e6,
            "BMAJ": 0.25,
            "BMIN": 0.25,
        }
    )
    fits.PrimaryHDU(data=data, header=header).writeto(in_path)

    mod._fix_headers(in_path, out_path)

    with fits.open(out_path) as hdul:
        hdr = hdul[0].header
    assert hdr["NAXIS"] == 4
    assert hdr["CTYPE4"] == "STOKES"
    assert float(hdr["CRVAL4"]) == pytest.approx(1.0)


def test_stokes_numeric_from_value_accepts_string_labels() -> None:
    """Stokes helpers map OVRO string tokens to FITS numeric codes."""
    mod = _import_module()
    assert mod._stokes_numeric_from_value("I") == pytest.approx(1.0)
    assert mod._stokes_numeric_from_value(np.str_("V")) == pytest.approx(4.0)
    assert mod._stokes_value_from_header(fits.Header({"CTYPE4": "STOKES", "CRVAL4": "I", "NAXIS": 4})) == pytest.approx(1.0)


def test_fix_headers_2d_promotion_uses_restfrq_for_synthetic_freq_axis(tmp_path: Path) -> None:
    """2D dewarped planes often carry ``RESTFRQ`` only; promotion must not default to 60 MHz."""
    import numpy as np

    mod = _import_module()
    in_path = tmp_path / "Blue_I_10min_Taper_Robust-0_pbcorr_dewarped_20241218_LST01h_t0001.fits"
    out_path = tmp_path / "output_fixed.fits"

    data = np.zeros((8, 8), dtype=np.float32)
    header = fits.Header(
        {
            "NAXIS": 2,
            "NAXIS1": 8,
            "NAXIS2": 8,
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "RESTFRQ": 73.77609456783534e6,
            "BMAJ": 0.1,
            "BMIN": 0.1,
        }
    )
    fits.PrimaryHDU(data=data, header=header).writeto(in_path)

    mod._fix_headers(in_path, out_path)

    with fits.open(out_path) as hdul:
        hdr = hdul[0].header

    assert hdr["NAXIS"] == 4
    assert hdr["CTYPE3"] == "FREQ"
    assert hdr["CRVAL3"] == pytest.approx(73.77609456783534e6)
    assert hdr["RESTFREQ"] == pytest.approx(73.77609456783534e6)

    hz = mod._canonical_stack_frequency_hz(
        out_path,
        group_metadata_source="fits",
        filename_convention="lst-color",
    )
    assert hz == pytest.approx(73.77609456783534e6)


def test_normalize_time_key_from_datetime64():
    """Datetime64 values should normalize to discovery-style time keys."""
    mod = _import_module()
    value = mod.np.datetime64("2024-12-18T06:33:36.987654321")

    out = mod._normalize_time_key(value)

    assert out == "20241218_063336"


def test_normalize_time_key_from_mjd_float():
    """Numeric MJD time coordinates should normalize to discovery-style keys."""
    mod = _import_module()
    # 2024-12-20T03:00:00 UTC in MJD.
    out = mod._normalize_time_key(60664.125)

    assert out == "20241220_030000"


def test_existing_time_keys_from_zarr(tmp_path: Path):
    """Existing Zarr time coordinates should map to a set of normalized keys."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    out_zarr = tmp_path / "existing.zarr"
    ds = xr.Dataset(
        {"SKY": (("time", "m", "l"), np.zeros((2, 2, 2), dtype=np.float32))},
        coords={
            "time": np.array(["2024-12-18T06:33:36", "2024-12-18T06:33:37"], dtype="datetime64[ns]"),
            "m": np.array([0.0, 1.0]),
            "l": np.array([0.0, 1.0]),
        },
    )
    ds.to_zarr(out_zarr, mode="w", consolidated=False)

    keys = mod._existing_time_keys_from_zarr(out_zarr)

    assert keys == {"20241218_063336", "20241218_063337"}


def test_existing_time_keys_from_zarr_missing_time_raises(tmp_path: Path):
    """Resume helper should fail clearly when existing Zarr has no time coordinate."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    out_zarr = tmp_path / "no_time.zarr"
    ds = xr.Dataset(
        {"SKY": (("m", "l"), np.zeros((2, 2), dtype=np.float32))},
        coords={"m": np.array([0.0, 1.0]), "l": np.array([0.0, 1.0])},
    )
    ds.to_zarr(out_zarr, mode="w", consolidated=False)

    with pytest.raises(RuntimeError, match="has no 'time' coordinate"):
        mod._existing_time_keys_from_zarr(out_zarr)


def test_reindex_time_step_to_expected_frequencies_fills_missing_with_nan():
    """Per-time datasets should be expanded to the expected subband axis."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    xds_t = xr.Dataset(
        {
            "SKY": (
                ("time", "frequency", "m", "l"),
                np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2),
            )
        },
        coords={
            "time": np.array(["2024-12-18T06:33:36"], dtype="datetime64[s]"),
            "frequency": np.array([41_000_000.0, 55_000_000.0]),
            "m": np.array([0.0, 1.0]),
            "l": np.array([0.0, 1.0]),
        },
    )

    out = mod._reindex_time_step_to_expected_frequencies(
        xds_t,
        [41_000_000.0, 48_000_000.0, 55_000_000.0],
    )

    assert out.sizes["frequency"] == 3
    assert np.allclose(out["frequency"].values, [41_000_000.0, 48_000_000.0, 55_000_000.0])
    # Added frequency plane is all NaN in data variables.
    assert np.isnan(out["SKY"].isel(frequency=1).values).all()


def test_reindex_time_step_preserves_empty_fits_header_str_fill() -> None:
    """Missing frequency subbands must not store ``np.nan`` in ``fits_header_str``."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    primary = fits.Header()
    primary["SIMPLE"] = True
    primary["BITPIX"] = -32
    payload = mod._fits_header_bytes_for_slice(
        primary,
        post_regrid_wcs_hdr=_make_sin_wcs_header_str(nx=2, ny=2, crval1=180.0, crval2=45.0),
        nl=2,
        nm=2,
    )
    xds_t = xr.Dataset(
        {
            "SKY": (
                ("time", "frequency", "m", "l"),
                np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2),
            ),
            "fits_header_str": (("time", "frequency"), np.array([[payload, payload]])),
        },
        coords={
            "time": np.array(["2024-12-18T06:33:36"], dtype="datetime64[s]"),
            "frequency": np.array([41_000_000.0, 55_000_000.0]),
            "m": np.array([0.0, 1.0]),
            "l": np.array([0.0, 1.0]),
        },
    )

    out = mod._reindex_time_step_to_expected_frequencies(
        xds_t,
        [41_000_000.0, 48_000_000.0, 55_000_000.0],
    )

    filled = out["fits_header_str"].isel(frequency=1).values
    assert filled == np.bytes_(b"") or filled == b""
    from ovro_lwa_portal.accessor import _decode_wcs_header_bytes

    assert _decode_wcs_header_bytes(filled) == ""


def test_sky_coord_cache_is_bounded_and_clearable() -> None:
    """Sky-coord LRU must not grow without bound across many WCS variants."""
    from astropy.wcs import WCS

    mod = _import_module()
    mod._clear_sky_coord_cache()

    base = fits.Header()
    base.update(
        {
            "SIMPLE": True,
            "BITPIX": -32,
            "NAXIS": 2,
            "NAXIS1": 8,
            "NAXIS2": 8,
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CRVAL1": 180.0,
            "CRVAL2": 37.0,
            "CRPIX1": 4.5,
            "CRPIX2": 4.5,
            "CDELT1": -0.1,
            "CDELT2": 0.1,
            "CUNIT1": "deg",
            "CUNIT2": "deg",
        }
    )
    for i in range(60):
        hdr = base.copy()
        hdr["CRVAL1"] = 180.0 + i
        hdr_str = WCS(hdr).celestial.to_header().tostring(sep="\n")
        ra, dec = mod._compute_sky_coord_arrays(8, 8, hdr_str)
        assert ra.shape == (8, 8)
        assert dec.shape == (8, 8)
    assert mod._sky_coord_cache_size() <= mod._SKY_COORD_CACHE_MAXSIZE
    mod._clear_sky_coord_cache()
    assert mod._sky_coord_cache_size() == 0


def test_discovery_completed_matches_mjd_when_filename_differs(tmp_path: Path) -> None:
    """Resume must treat a group as done when DATE-OBS MJD is already in the Zarr."""
    import numpy as np
    import xarray as xr
    from astropy.time import Time

    mod = _import_module()
    obs = Time("2025-01-20T04:00:18", scale="utc")
    mjd = float(obs.mjd)

    out_zarr = tmp_path / "store.zarr"
    xr.Dataset(
        {"SKY": (("time",), np.array([0.0]))},
        coords={"time": ("time", np.array([mjd], dtype=np.float64))},
    ).to_zarr(out_zarr, mode="w")

    completed_keys, completed_mjds = mod._completed_times_in_zarr(out_zarr, rebuild=False)
    assert obs.to_datetime().strftime("%Y%m%d_%H%M%S") in completed_keys

    data = np.zeros((4, 4), dtype=np.float32)
    img_hdr = fits.Header(
        {
            "NAXIS": 2,
            "NAXIS1": 4,
            "NAXIS2": 4,
            "DATE-OBS": "2025-01-20T04:00:18",
            "BMAJ": 0.1,
            "BMIN": 0.1,
        }
    )
    primary = fits.PrimaryHDU(header=fits.Header({"SIMPLE": True, "BITPIX": -32, "NAXIS": 0}))
    image = fits.ImageHDU(data=data, header=img_hdr)
    fpath = tmp_path / "50MHz-Clean-Snapshot-20250120_040333-image.fits.fs"
    fits.HDUList([primary, image]).writeto(fpath, overwrite=True)

    discovery_key = "20250120_040333"
    assert discovery_key not in completed_keys
    assert mod._discovery_time_key_completed_in_zarr(
        discovery_key, [fpath], completed_keys, completed_mjds
    )


def test_write_or_append_omits_fits_wcs_header_when_fits_header_str_present(
    tmp_path: Path,
) -> None:
    """Zarr must not persist time-0 ``fits_wcs_header`` on SKY when headers vary per time."""
    import xarray as xr

    mod = _import_module()
    out_zarr = tmp_path / "wcs_attrs.zarr"
    l_ = np.linspace(-1.0, 1.0, 8)
    m_ = np.linspace(-1.0, 1.0, 8)
    hdr0_str = _make_sin_wcs_header_str(nx=8, ny=8, crval1=10.0, crval2=20.0)
    hdr1_str = _make_sin_wcs_header_str(nx=8, ny=8, crval1=20.0, crval2=30.0)
    hdr0 = mod._fits_header_bytes_for_slice(
        fits.Header.fromstring(hdr0_str, sep="\n"),
        post_regrid_wcs_hdr=hdr0_str,
        nl=8,
        nm=8,
    )
    hdr1 = mod._fits_header_bytes_for_slice(
        fits.Header.fromstring(hdr1_str, sep="\n"),
        post_regrid_wcs_hdr=hdr1_str,
        nl=8,
        nm=8,
    )
    mjd0, mjd1 = 60695.17, 60695.18

    def _step(mjd: float, hdr: bytes, value: float) -> xr.Dataset:
        ds = xr.Dataset(
            {
                "SKY": (
                    ("time", "frequency", "polarization", "m", "l"),
                    np.full((1, 1, 1, 8, 8), value, dtype=np.float32),
                ),
                "fits_header_str": (("time",), np.array([np.bytes_(hdr)])),
            },
            coords={
                "time": ("time", np.array([mjd], dtype=np.float64)),
                "frequency": ("frequency", np.array([5.0e7])),
                "polarization": ("polarization", np.array([1.0])),
                "l": ("l", l_),
                "m": ("m", m_),
            },
            attrs={"fits_wcs_header": "stale-global"},
        )
        ds["SKY"].attrs["fits_wcs_header"] = "stale-sky"
        ds["right_ascension"] = (("m", "l"), np.zeros((8, 8)))
        ds["declination"] = (("m", "l"), np.zeros((8, 8)))
        ds["right_ascension"].attrs["fits_wcs_header"] = "stale-coord"
        return ds

    mod._write_or_append_zarr(_step(mjd0, hdr0, 1.0), out_zarr, first_write=True, chunk_lm=4)
    mod._write_or_append_zarr(_step(mjd1, hdr1, 2.0), out_zarr, first_write=False, chunk_lm=4)

    with xr.open_zarr(out_zarr, consolidated=False) as ds:
        assert int(ds.sizes["time"]) == 2
        assert "fits_wcs_header" not in ds.attrs
        assert "fits_wcs_header" not in ds["SKY"].attrs
        assert "fits_wcs_header" not in ds["right_ascension"].attrs
        assert "wcs_header_str" not in ds
        assert bytes(ds["fits_header_str"].isel(time=1).values.item()) == hdr1


def test_write_or_append_skips_duplicate_mjd_by_default(tmp_path: Path, caplog) -> None:
    """Duplicate observation times must not overwrite existing Zarr rows by default."""
    import logging
    import numpy as np
    import xarray as xr

    mod = _import_module()
    caplog.set_level(logging.INFO, logger="ovro_lwa_portal.fits_to_zarr_xradio")

    out_zarr = tmp_path / "store.zarr"
    mjd = 60695.17
    l_ = np.linspace(-1.0, 1.0, 8)
    m_ = np.linspace(-1.0, 1.0, 8)
    wcs_hdr = mod._fits_header_bytes_for_slice(
        fits.Header.fromstring(
            _make_sin_wcs_header_str(nx=8, ny=8, crval1=180.0, crval2=45.0),
            sep="\n",
        ),
        post_regrid_wcs_hdr=_make_sin_wcs_header_str(nx=8, ny=8, crval1=180.0, crval2=45.0),
        nl=8,
        nm=8,
    )
    base = xr.Dataset(
        {
            "SKY": (
                ("time", "frequency", "polarization", "m", "l"),
                np.zeros((1, 1, 1, 8, 8), dtype=np.float32),
            ),
            "fits_header_str": (("time",), np.array([np.bytes_(wcs_hdr)])),
        },
        coords={
            "time": ("time", np.array([mjd], dtype=np.float64)),
            "frequency": ("frequency", np.array([5.0e7])),
            "polarization": ("polarization", np.array([1.0])),
            "l": ("l", l_),
            "m": ("m", m_),
        },
    )
    mod._write_or_append_zarr(base, out_zarr, first_write=True, chunk_lm=4)

    dup = base.copy(deep=True)
    dup["SKY"].values[...] = 99.0
    mod._write_or_append_zarr(dup, out_zarr, first_write=False, chunk_lm=4)

    with xr.open_zarr(out_zarr, consolidated=False) as ds:
        assert int(ds.sizes["time"]) == 1
        assert float(ds["SKY"].values[0, 0, 0, 0, 0]) == 0.0
    assert "Skipping write: time row" in caplog.text
    assert "Overwriting existing time row" not in caplog.text


def test_convert_resume_skips_already_ingested_times(monkeypatch, tmp_path: Path):
    """Resume mode should only process discovered timesteps missing from output Zarr."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_zarr = out_dir / "ovro_lwa_full_lm_only.zarr"
    out_zarr.mkdir()

    f1 = tmp_path / "a.fits"
    f2 = tmp_path / "b.fits"
    f1.touch()
    f2.touch()

    by_time = {"20241218_063336": [f1], "20241218_063337": [f2]}
    monkeypatch.setattr(mod, "_discover_groups", lambda *_args, **_kwargs: by_time)
    monkeypatch.setattr(mod, "_filter_invalid_beam_files", lambda groups, **kwargs: groups)
    monkeypatch.setattr(
        mod,
        "_filter_completed_time_keys",
        lambda by_time, _out, **_: {k: v for k, v in by_time.items() if k != "20241218_063336"},
    )
    monkeypatch.setattr(
        mod,
        "_global_frequency_coord_hz",
        lambda *_args, **_kwargs: np.asarray([4.1e7], dtype=np.float64),
    )

    ref = xr.Dataset(coords={"l": ("l", np.array([0.0, 1.0])), "m": ("m", np.array([0.0, 1.0]))})
    monkeypatch.setattr(mod, "_load_global_lm_reference_dataset", lambda *_args, **_kwargs: ref)

    xds_t = xr.Dataset(
        {"SKY": (("time", "m", "l"), np.zeros((1, 2, 2), dtype=np.float32))},
        coords={
            "time": np.array(["2024-12-18T06:33:37"], dtype="datetime64[s]"),
            "m": np.array([0.0, 1.0]),
            "l": np.array([0.0, 1.0]),
            "frequency": np.array([4.1e7]),
        },
    )
    monkeypatch.setattr(mod, "_combine_time_step", lambda *_args, **_kwargs: (xds_t, [4.1e7], []))

    write_calls: list[bool] = []
    monkeypatch.setattr(
        mod,
        "_write_or_append_zarr",
        lambda _xds, _out, *, first_write, chunk_lm: write_calls.append(first_write),
    )

    consolidate_calls: list[Path] = []
    monkeypatch.setattr(
        mod,
        "_consolidate_zarr_metadata",
        lambda path: consolidate_calls.append(path),
    )

    result = mod.convert_fits_dir_to_zarr(
        input_dir=tmp_path,
        out_dir=out_dir,
        resume=True,
        rebuild=False,
    )

    assert result == out_zarr
    assert len(write_calls) == 1
    assert write_calls == [False]
    assert consolidate_calls == [out_zarr]


def test_convert_skips_consolidate_when_disabled(monkeypatch, tmp_path: Path):
    """Per-time ingest callers can defer consolidation until the full batch finishes."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_zarr = out_dir / "ovro_lwa_full_lm_only.zarr"

    f1 = tmp_path / "a.fits"
    f1.touch()
    by_time = {"20241218_063336": [f1]}
    monkeypatch.setattr(mod, "_discover_groups", lambda *_args, **_kwargs: by_time)
    monkeypatch.setattr(mod, "_filter_invalid_beam_files", lambda groups, **kwargs: groups)
    monkeypatch.setattr(
        mod,
        "_global_frequency_coord_hz",
        lambda *_args, **_kwargs: np.asarray([4.1e7], dtype=np.float64),
    )

    ref = xr.Dataset(coords={"l": ("l", np.array([0.0, 1.0])), "m": ("m", np.array([0.0, 1.0]))})
    monkeypatch.setattr(mod, "_load_global_lm_reference_dataset", lambda *_args, **_kwargs: ref)

    xds_t = xr.Dataset(
        {"SKY": (("time", "m", "l"), np.zeros((1, 2, 2), dtype=np.float32))},
        coords={
            "time": np.array(["2024-12-18T06:33:37"], dtype="datetime64[s]"),
            "m": np.array([0.0, 1.0]),
            "l": np.array([0.0, 1.0]),
            "frequency": np.array([4.1e7]),
        },
    )
    monkeypatch.setattr(mod, "_combine_time_step", lambda *_args, **_kwargs: (xds_t, [4.1e7], []))
    monkeypatch.setattr(mod, "_write_or_append_zarr", lambda *_args, **_kwargs: None)

    consolidate_calls: list[Path] = []
    monkeypatch.setattr(
        mod,
        "_consolidate_zarr_metadata",
        lambda path: consolidate_calls.append(path),
    )

    mod.convert_fits_dir_to_zarr(
        input_dir=tmp_path,
        out_dir=out_dir,
        consolidate_metadata_at_end=False,
    )
    assert consolidate_calls == []


def test_convert_resume_returns_early_when_no_pending(monkeypatch, tmp_path: Path):
    """Resume mode should exit without combine/write when all times already exist."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    out_zarr = out_dir / "ovro_lwa_full_lm_only.zarr"
    out_zarr.mkdir()

    f1 = tmp_path / "a.fits"
    f1.touch()

    by_time = {"20241218_063336": [f1]}
    monkeypatch.setattr(mod, "_discover_groups", lambda *_args, **_kwargs: by_time)
    monkeypatch.setattr(mod, "_filter_invalid_beam_files", lambda groups, **kwargs: groups)
    monkeypatch.setattr(
        mod,
        "_filter_completed_time_keys",
        lambda _by_time, _out, **_: {},
    )

    ref = xr.Dataset(coords={"l": ("l", np.array([0.0, 1.0])), "m": ("m", np.array([0.0, 1.0]))})
    monkeypatch.setattr(mod, "_load_global_lm_reference_dataset", lambda *_args, **_kwargs: ref)

    combine_calls: list[bool] = []
    monkeypatch.setattr(
        mod,
        "_combine_time_step",
        lambda *_args, **_kwargs: combine_calls.append(True),  # pragma: no cover
    )
    write_calls: list[bool] = []
    monkeypatch.setattr(
        mod,
        "_write_or_append_zarr",
        lambda *_args, **_kwargs: write_calls.append(True),  # pragma: no cover
    )
    consolidate_calls: list[Path] = []
    monkeypatch.setattr(
        mod,
        "_consolidate_zarr_metadata",
        lambda path: consolidate_calls.append(path),
    )

    result = mod.convert_fits_dir_to_zarr(
        input_dir=tmp_path,
        out_dir=out_dir,
        resume=True,
        rebuild=False,
    )

    assert result == out_zarr
    assert combine_calls == []
    assert write_calls == []
    assert consolidate_calls == [out_zarr]


def test_lm_reference_from_existing_zarr_reads_fits_header_str(tmp_path: Path) -> None:
    """Stokes V convert must load LM reference WCS from ``fits_header_str`` Zarr stores."""
    import xarray as xr

    mod = _import_module()
    hdr_str = _make_sin_wcs_header_str(nx=8, ny=8, crval1=180.0, crval2=45.0)
    hdr = mod._fits_header_bytes_for_slice(
        fits.Header.fromstring(hdr_str, sep="\n"),
        post_regrid_wcs_hdr=hdr_str,
        nl=8,
        nm=8,
    )
    out_zarr = tmp_path / "fits_hdr_ref.zarr"
    ds = xr.Dataset(
        {
            "SKY": (
                ("time", "frequency", "polarization", "m", "l"),
                np.zeros((1, 2, 1, 8, 8), dtype=np.float32),
            ),
            "fits_header_str": (
                ("time", "frequency"),
                np.array([[np.bytes_(hdr), np.bytes_(hdr)]], dtype=object),
            ),
        },
        coords={
            "time": ("time", np.array([59000.0])),
            "frequency": ("frequency", np.array([55e6, 65e6])),
            "polarization": ("polarization", np.array([1.0])),
            "l": ("l", np.linspace(-0.1, 0.1, 8)),
            "m": ("m", np.linspace(-0.1, 0.1, 8)),
        },
    )
    ds.to_zarr(out_zarr, mode="w", consolidated=False)

    ref = mod._lm_reference_from_existing_zarr(out_zarr)
    assert "fits_wcs_header" in ref.attrs
    from astropy.io import fits as afits
    from astropy.wcs import WCS

    ref_hdr = afits.Header.fromstring(ref.attrs["fits_wcs_header"], sep="\n")
    assert WCS(ref_hdr).wcs.crval[0] == pytest.approx(180.0)
    assert ref.sizes == {"l": 8, "m": 8}


def test_read_wcs_header_str_from_time_promoted_zarr(tmp_path: Path) -> None:
    """Resume must read WCS from stores where ``wcs_header_str`` spans ``time``."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    hdr = _make_sin_wcs_header_str(nx=8, ny=8, crval1=180.0, crval2=45.0)
    hdr_encoded = hdr.encode("utf-8")
    hdr_bytes = np.bytes_(hdr_encoded)
    n_time = 4
    nl = nm = 8
    out_zarr = tmp_path / "resume_probe.zarr"
    wcs_per_time = np.array([hdr_encoded] * n_time, dtype=f"S{len(hdr_encoded)}")
    ds = xr.Dataset(
        {
            "SKY": (
                ("time", "frequency", "m", "l"),
                np.zeros((n_time, 2, nm, nl), dtype=np.float32),
            ),
            "wcs_header_str": (("time",), wcs_per_time),
        },
        coords={
            "time": ("time", np.linspace(59000.0, 59000.0 + n_time - 1, n_time)),
            "frequency": ("frequency", np.array([55e6, 65e6])),
            "l": ("l", np.linspace(-0.1, 0.1, nl)),
            "m": ("m", np.linspace(-0.1, 0.1, nm)),
        },
    )
    ds.to_zarr(out_zarr, mode="w", consolidated=False)

    ref = mod._lm_reference_from_existing_zarr(out_zarr)
    assert ref.attrs.get("fits_wcs_header") == hdr
    assert ref.sizes == {"l": nl, "m": nm}


def test_fix_headers_relabels_singleton_axis_to_stokes_for_4d(tmp_path: Path) -> None:
    """4D cubes with a mis-tagged length-1 axis must expose a literal ``STOKES`` CTYPE for xradio."""
    import numpy as np

    mod = _import_module()
    in_path = tmp_path / "in4d.fits"
    out_path = tmp_path / "out4d_fixed.fits"
    data = np.zeros((1, 1, 4, 4), dtype=np.float32)
    header = fits.Header(
        {
            "NAXIS": 4,
            "NAXIS1": 4,
            "NAXIS2": 4,
            "NAXIS3": 1,
            "NAXIS4": 1,
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CTYPE3": "FREQ",
            "CTYPE4": "TABULAR",
            "CRVAL1": 180.0,
            "CRVAL2": 45.0,
            "CRVAL3": 4.1e7,
            "CRVAL4": 0.0,
            "CRPIX1": 2.0,
            "CRPIX2": 2.0,
            "CRPIX3": 1.0,
            "CRPIX4": 1.0,
            "CDELT1": -0.03,
            "CDELT2": 0.03,
            "CDELT3": 1.0,
            "CDELT4": 1.0,
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CUNIT3": "Hz",
            "CUNIT4": "",
            "DATE-OBS": "2024-01-01T00:00:00",
            "RADESYS": "FK5",
            "EQUINOX": 2000.0,
            "LONPOLE": 180.0,
            "TELESCOP": "TEST",
            "BMAJ": 0.1,
            "BMIN": 0.1,
        }
    )
    fits.PrimaryHDU(data=data, header=header).writeto(in_path)

    mod._fix_headers(in_path, out_path)

    with fits.open(out_path) as hdul:
        hdr = hdul[0].header
        assert hdr["CTYPE4"] == "STOKES"
        assert hdr["CRVAL4"] == pytest.approx(1.0)

    xds = mod._read_fits_via_xradio(out_path, do_sky_coords=False, compute_mask=False)
    assert "SKY" in xds.data_vars


def test_fix_headers_strips_padded_stokes_ctype_for_xradio(tmp_path: Path) -> None:
    """Trailing spaces on ``STOKES`` must not break xradio ``ctype.index('STOKES')``."""
    import numpy as np

    mod = _import_module()
    in_path = tmp_path / "pad_stokes.fits"
    out_path = tmp_path / "pad_stokes_fixed.fits"
    data = np.zeros((1, 1, 4, 4), dtype=np.float32)
    header = fits.Header(
        {
            "NAXIS": 4,
            "NAXIS1": 4,
            "NAXIS2": 4,
            "NAXIS3": 1,
            "NAXIS4": 1,
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CTYPE3": "FREQ",
            "CTYPE4": "STOKES   ",
            "CRVAL1": 180.0,
            "CRVAL2": 45.0,
            "CRVAL3": 4.1e7,
            "CRVAL4": 1.0,
            "CRPIX1": 2.0,
            "CRPIX2": 2.0,
            "CRPIX3": 1.0,
            "CRPIX4": 1.0,
            "CDELT1": -0.03,
            "CDELT2": 0.03,
            "CDELT3": 1.0,
            "CDELT4": 1.0,
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CUNIT3": "Hz",
            "CUNIT4": "",
            "DATE-OBS": "2024-01-01T00:00:00",
            "RADESYS": "FK5",
            "EQUINOX": 2000.0,
            "LONPOLE": 180.0,
            "TELESCOP": "TEST",
            "BMAJ": 0.1,
            "BMIN": 0.1,
        }
    )
    fits.PrimaryHDU(data=data, header=header).writeto(in_path)

    mod._fix_headers(in_path, out_path)

    ctypes = [fits.getheader(out_path)[f"CTYPE{i}"] for i in range(1, 5)]
    assert ctypes[-1] == "STOKES"

    xds = mod._read_fits_via_xradio(out_path, do_sky_coords=False, compute_mask=False)
    assert "SKY" in xds.data_vars


def test_fix_headers_preserves_crval_from_input_not_filename(tmp_path: Path):
    """_fix_headers keeps native CRVAL1/2 even when the basename has an image-time stamp."""
    import numpy as np

    mod = _import_module()
    in_path = tmp_path / "18MHz-I-Deep-Taper-Robust-0-image-20241218_030201-test.fits"
    out_path = tmp_path / "18MHz-I-Deep-Taper-Robust-0-image-20241218_030201-test_fixed.fits"

    data = np.zeros((8, 8), dtype=np.float32)
    header = fits.Header(
        {
            "NAXIS": 2,
            "NAXIS1": 8,
            "NAXIS2": 8,
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CRVAL1": 12.5,
            "CRVAL2": 37.2,
            "CRPIX1": 4.5,
            "CRPIX2": 4.5,
            "CDELT1": -0.03,
            "CDELT2": 0.03,
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "DATE-OBS": "2024-12-18T03:00:01.4",
            "TIMESYS": "UTC",
            "BMAJ": 0.1,
            "BMIN": 0.1,
        }
    )
    fits.PrimaryHDU(data=data, header=header).writeto(in_path)

    mod._fix_headers(in_path, out_path)

    with fits.open(out_path) as hdul:
        hdr = hdul[0].header

    assert hdr["CRVAL1"] == pytest.approx(12.5)
    assert hdr["CRVAL2"] == pytest.approx(37.2)
    assert hdr["LATPOLE"] == pytest.approx(37.2)
    assert hdr["RADESYS"] == "FK5"


def test_fix_headers_leaves_crval_without_image_timestamp_in_name(tmp_path: Path):
    """If the basename has no ``-image-YYYYMMDD_HHMMSS``, CRVAL1/2 are not overwritten."""
    import numpy as np

    mod = _import_module()
    in_path = tmp_path / "no_stamp.fits"
    out_path = tmp_path / "no_stamp_fixed.fits"

    data = np.zeros((8, 8), dtype=np.float32)
    header = fits.Header(
        {
            "NAXIS": 2,
            "NAXIS1": 8,
            "NAXIS2": 8,
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CRVAL1": 1.25,
            "CRVAL2": 2.5,
            "CRPIX1": 4.5,
            "CRPIX2": 4.5,
            "CDELT1": -0.03,
            "CDELT2": 0.03,
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "BMAJ": 0.1,
            "BMIN": 0.1,
        }
    )
    fits.PrimaryHDU(data=data, header=header).writeto(in_path)

    mod._fix_headers(in_path, out_path)

    with fits.open(out_path) as hdul:
        hdr = hdul[0].header

    assert hdr["CRVAL1"] == pytest.approx(1.25)
    assert hdr["CRVAL2"] == pytest.approx(2.5)


def test_patch_celestial_crval_in_header_str_keeps_pixel_grid():
    """CRVAL repair must preserve CRPIX/CDELT while adopting native FITS phase center."""
    mod = _import_module()
    ref_hdr = _make_sin_wcs_header_str(nx=8, ny=8, crval1=99.0, crval2=88.0)
    src = fits.Header(
        {
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CRVAL1": 12.5,
            "CRVAL2": 37.2,
            "RADESYS": "FK5",
        }
    )
    out = mod._patch_celestial_crval_in_header_str(ref_hdr, src)
    out_hdr = fits.Header.fromstring(out, sep="\n")
    ref_parsed = fits.Header.fromstring(ref_hdr, sep="\n")
    assert out_hdr["CRVAL1"] == pytest.approx(12.5)
    assert out_hdr["CRVAL2"] == pytest.approx(37.2)
    assert out_hdr["LATPOLE"] == pytest.approx(37.2)
    assert out_hdr["CRPIX1"] == pytest.approx(ref_parsed["CRPIX1"])
    assert out_hdr["CDELT1"] == pytest.approx(ref_parsed["CDELT1"])


def test_resolve_discovery_keys_for_zarr_times_handles_date_obs_offset():
    """Zarr DATE-OBS keys can differ from filename -image- stamps by a few seconds."""
    import numpy as np
    from astropy.time import Time

    mod = _import_module()
    z_time = np.array(
        [
            Time("2025-01-20T04:00:08", scale="utc").mjd,
            Time("2025-01-20T04:00:18", scale="utc").mjd,
        ],
        dtype=np.float64,
    )
    by_time = {
        "20250120_040013": [Path("a.fits")],
        "20250120_040023": [Path("b.fits")],
    }
    resolved, stats = mod._resolve_discovery_keys_for_zarr_times(z_time, by_time)
    assert resolved == ["20250120_040013", "20250120_040023"]
    assert stats["index"] == 2
    assert stats["unresolved"] == 0


def test_repair_zarr_crval_from_fits_patches_wcs_header_str(tmp_path: Path):
    """In-place repair replaces stored CRVAL from native FITS without re-ingest."""
    import numpy as np
    import zarr
    from astropy.time import Time

    mod = _import_module()
    store = tmp_path / "repair_me.zarr"
    wrong_hdr = _make_sin_wcs_header_str(nx=4, ny=4, crval1=200.0, crval2=50.0)
    discovery_key = "20250120_040013"
    zarr_mjd = Time("2025-01-20T04:00:08", scale="utc").mjd

    root = zarr.group(store)
    root.create_dataset("time", data=np.array([zarr_mjd], dtype=np.float64))
    root.create_dataset(
        "wcs_header_str",
        data=np.array([np.bytes_(wrong_hdr.encode("utf-8"))], dtype="S"),
        chunks=(1,),
    )
    root["wcs_header_str"].attrs["_ARRAY_DIMENSIONS"] = ["time"]

    fits_path = tmp_path / f"18MHz-Clean-Snapshot-{discovery_key}-image.fits"
    data = np.zeros((4, 4), dtype=np.float32)
    fits.PrimaryHDU(
        data=data,
        header=fits.Header(
            {
                "NAXIS": 2,
                "NAXIS1": 4,
                "NAXIS2": 4,
                "CTYPE1": "RA---SIN",
                "CTYPE2": "DEC--SIN",
                "CRVAL1": 61.14,
                "CRVAL2": 37.16,
                "CRPIX1": 2.5,
                "CRPIX2": 2.5,
                "CDELT1": -0.03,
                "CDELT2": 0.03,
                "CUNIT1": "deg",
                "CUNIT2": "deg",
                "BMAJ": 0.1,
                "BMIN": 0.1,
            }
        ),
    ).writeto(fits_path)

    result = mod.repair_zarr_crval_from_fits(
        store,
        {discovery_key: [fits_path]},
        group_metadata_source="filename",
        backup_suffix=".bak",
    )
    assert result["patched_rows"] == 1
    assert result["max_crval_delta_deg"]["ra"] > 100.0

    zg = zarr.open_group(str(store), mode="r")
    fixed_hdr = fits.Header.fromstring(
        zg["wcs_header_str"][0].decode("utf-8"),
        sep="\n",
    )
    assert fixed_hdr["CRVAL1"] == pytest.approx(61.14)
    assert fixed_hdr["CRVAL2"] == pytest.approx(37.16)
    assert fixed_hdr["CRPIX1"] == pytest.approx((4 + 1) / 2.0)


def test_collapse_wcs_header_str_when_ra_dec_have_no_frequency_dim():
    """Single-subband combine can leave wcs_header_str on (frequency,) while RA/Dec are (l, m)."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    hdr_a, hdr_b = b"hdr-a", b"hdr-b"
    ds = xr.Dataset(
        {
            "SKY": (("l", "m"), np.ones((4, 5))),
            "wcs_header_str": (("frequency",), np.array([np.bytes_(hdr_a), np.bytes_(hdr_b)])),
        },
        coords={
            "frequency": np.array([45e6, 55e6]),
            "l": np.linspace(-0.1, 0.1, 4),
            "m": np.linspace(-0.1, 0.1, 5),
            "right_ascension": (("l", "m"), np.full((4, 5), 100.0)),
            "declination": (("l", "m"), np.full((4, 5), 40.0)),
        },
    )
    out = mod._harmonize_celestial_coords_independent_of_frequency(ds)
    assert out["wcs_header_str"].dims == ()
    assert bytes(out["wcs_header_str"].values.item()) == hdr_a


def test_align_time_dimension_broadcasts_scalar_wcs_to_time_frequency_schema():
    """Scalar per-step WCS must align when an existing store uses (time, frequency)."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    hdr = np.bytes_(b"NAXIS = 2\nCRVAL1 = 180.0\nCRVAL2 = 45.0\n")
    schema = xr.Dataset(
        {
            "SKY": (("time", "frequency", "m", "l"), np.zeros((2, 3, 4, 4))),
            "wcs_header_str": (
                ("time", "frequency"),
                np.array([[hdr] * 3, [hdr] * 3], dtype=object),
            ),
        },
        coords={
            "time": [60000.0, 60001.0],
            "frequency": [45e6, 55e6, 65e6],
            "l": np.arange(4),
            "m": np.arange(4),
        },
    )
    incoming = xr.Dataset(
        {
            "SKY": (("frequency", "m", "l"), np.zeros((3, 4, 4))),
            "wcs_header_str": ((), hdr),
        },
        coords={
            "time": np.array([60002.0]),
            "frequency": schema.coords["frequency"],
            "l": np.arange(4),
            "m": np.arange(4),
        },
    )
    aligned = mod._align_time_dimension_for_zarr_write(incoming, schema=schema)
    assert aligned["wcs_header_str"].dims == ("time", "frequency")
    assert aligned["wcs_header_str"].sizes["time"] == 1
    assert bytes(aligned["wcs_header_str"].isel(time=0, frequency=0).values.item()) == bytes(hdr)


def test_assert_nonempty_wcs_header_str_before_zarr_write_raises():
    import numpy as np
    import xarray as xr

    mod = _import_module()
    ds = xr.Dataset(
        {
            "SKY": (("time", "frequency", "m", "l"), np.zeros((1, 2, 4, 4))),
            "wcs_header_str": (("time",), np.array([np.bytes_(b"")])),
        },
        coords={"time": [60000.0], "frequency": [45e6, 55e6], "l": np.arange(4), "m": np.arange(4)},
    )
    with pytest.raises(RuntimeError, match="wcs_header_str is empty"):
        mod._assert_nonempty_wcs_header_str_before_zarr_write(ds)


def test_harmonize_celestial_coords_collapses_frequency_dim():
    """After combine, RA/Dec should be ``(l, m)`` only when slices share one WCS."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    nm, nl, nf = 5, 6, 2
    ra0_ml = np.broadcast_to(np.linspace(100.0, 110.0, nl), (nm, nl)).copy()
    ra0 = ra0_ml.T
    ra = np.stack([ra0, ra0], axis=0)
    dec = np.stack(
        [np.full((nl, nm), 40.0, dtype=np.float64), np.full((nl, nm), 40.0, dtype=np.float64)],
        axis=0,
    )
    hdr = "NAXIS = 2\nCRVAL1 = 105"
    ds = xr.Dataset(
        {"SKY": (("frequency", "l", "m"), np.ones((nf, nl, nm)))},
        coords={
            "frequency": np.array([45e6, 55e6], dtype=float),
            "l": np.linspace(-0.1, 0.1, nl),
            "m": np.linspace(-0.1, 0.1, nm),
            "right_ascension": (("frequency", "l", "m"), ra),
            "declination": (("frequency", "l", "m"), dec),
        },
    )
    ds["right_ascension"].attrs["fits_wcs_header"] = hdr
    ds["declination"].attrs["fits_wcs_header"] = hdr

    out = mod._harmonize_celestial_coords_independent_of_frequency(ds)
    assert "frequency" not in out.right_ascension.dims
    assert out.right_ascension.shape == (nl, nm)
    np.testing.assert_allclose(out.right_ascension.values, ra0)
    assert out["SKY"].attrs.get("fits_wcs_header") == hdr


def test_harmonize_celestial_coords_warns_on_large_wcs_drift(caplog):
    """Large per-channel RA/Dec drift vs reference should emit one warning."""
    import logging

    import numpy as np
    import xarray as xr

    mod = _import_module()
    nm, nl, nf = 5, 6, 2
    ra0_ml = np.broadcast_to(np.linspace(100.0, 110.0, nl), (nm, nl)).copy()
    ra0 = ra0_ml.T
    ra1 = ra0 + 2.0
    ra = np.stack([ra0, ra1], axis=0)
    dec = np.stack(
        [np.full((nl, nm), 40.0, dtype=np.float64), np.full((nl, nm), 40.0, dtype=np.float64)],
        axis=0,
    )
    ds = xr.Dataset(
        {"SKY": (("frequency", "l", "m"), np.ones((nf, nl, nm)))},
        coords={
            "frequency": np.array([45e6, 55e6], dtype=float),
            "l": np.linspace(-0.1, 0.1, nl),
            "m": np.linspace(-0.1, 0.1, nm),
            "right_ascension": (("frequency", "l", "m"), ra),
            "declination": (("frequency", "l", "m"), dec),
        },
    )
    caplog.set_level(logging.WARNING, logger="ovro_lwa_portal.fits_to_zarr_xradio")
    out = mod._harmonize_celestial_coords_independent_of_frequency(ds, warn_max_sep_arcsec=60.0)
    assert "Celestial coordinate grids differ" in caplog.text
    assert "frequency" not in out.right_ascension.dims


def test_harmonize_celestial_coords_samples_dask_backed_coords(monkeypatch):
    """Dask-backed celestial coords should be sampled before drift computation."""
    import numpy as np
    import xarray as xr

    da = pytest.importorskip("dask.array")
    mod = _import_module()
    nm, nl, nf = 300, 300, 2
    ra0 = np.broadcast_to(np.linspace(100.0, 110.0, nl), (nm, nl)).copy()
    ra = np.stack([ra0, ra0 + 0.01], axis=0)
    dec = np.stack(
        [np.full((nm, nl), 40.0, dtype=np.float64), np.full((nm, nl), 40.0, dtype=np.float64)],
        axis=0,
    )
    ds = xr.Dataset(
        {"SKY": (("frequency", "m", "l"), np.ones((nf, nm, nl)))},
        coords={
            "frequency": np.array([45e6, 55e6], dtype=float),
            "l": np.linspace(-0.1, 0.1, nl),
            "m": np.linspace(-0.1, 0.1, nm),
            "right_ascension": (
                ("frequency", "m", "l"),
                da.from_array(ra, chunks=(1, 75, 75)),
            ),
            "declination": (
                ("frequency", "m", "l"),
                da.from_array(dec, chunks=(1, 75, 75)),
            ),
        },
    )

    captured = {}
    original = mod._sky_sep_max_vs_ref_arcsec

    def _capture(
        ra_arr,
        dec_arr,
        *,
        ref_idx,
        max_points=mod._CELESTIAL_DRIFT_SAMPLE_MAX_POINTS,
    ):
        captured["shape"] = tuple(ra_arr.shape)
        return original(ra_arr, dec_arr, ref_idx=ref_idx, max_points=max_points)

    monkeypatch.setattr(mod, "_sky_sep_max_vs_ref_arcsec", _capture)
    out = mod._harmonize_celestial_coords_independent_of_frequency(ds)
    assert captured["shape"] == (nf, 1, mod._CELESTIAL_DRIFT_SAMPLE_MAX_POINTS)
    assert "frequency" not in out.right_ascension.dims


def test_align_zarr_velocity_coord_expands_when_schema_has_time_dimension():
    """Append path: velocity only on ``frequency`` in coords vs store ``(time, frequency)``."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    nf = 4
    freq = np.linspace(30e6, 78e6, nf)
    time_schema = np.array([58400.0])
    velocity_on_disk = np.arange(nf * 1, dtype=np.float64).reshape(1, nf)
    schema = xr.Dataset(
        coords={
            "time": ("time", time_schema),
            "frequency": ("frequency", freq),
            "velocity": (("time", "frequency"), velocity_on_disk),
        },
    )
    incoming = xr.Dataset(
        coords={
            "time": ("time", [58401.5]),
            "frequency": ("frequency", freq),
            "velocity": ("frequency", np.arange(nf, dtype=np.float64) + 1.0),
        },
    )

    aligned = mod._align_time_dimension_for_zarr_write(incoming, schema=schema)
    assert aligned.coords["velocity"].dims == ("time", "frequency")
    np.testing.assert_array_equal(
        aligned.coords["velocity"].values,
        np.broadcast_to(np.arange(nf, dtype=np.float64) + 1.0, (1, nf)),
    )
    assert aligned.coords["frequency"].dims == ("frequency",)


def test_align_zarr_first_write_keeps_dimension_coords_1d():
    """First write: frequency / l stays 1D; frequency-only auxiliary coords gain ``time``."""
    import numpy as np
    import xarray as xr

    mod = _import_module()
    nf = 3
    freq = np.linspace(55e6, 65e6, nf)
    nl = 2
    l = np.linspace(-0.1, 0.1, nl)
    ds = xr.Dataset(
        coords={
            "time": ("time", [59000.0]),
            "frequency": ("frequency", freq),
            "l": ("l", l),
            "velocity": ("frequency", np.zeros(nf, dtype=np.float32)),
        },
    )
    aligned = mod._align_time_dimension_for_zarr_write(ds)

    assert aligned.coords["frequency"].dims == ("frequency",)
    assert aligned.coords["l"].dims == ("l",)
    assert aligned.coords["velocity"].dims == ("time", "frequency")


def test_fits_header_bytes_for_slice_patches_celestial_and_shape() -> None:
    """Pixel-faithful headers keep provenance while adopting post-regrid celestial cards."""
    from astropy.io import fits as afits
    from astropy.wcs import WCS

    mod = _import_module()
    primary = afits.Header()
    primary["SIMPLE"] = True
    primary["BITPIX"] = -32
    primary["NAXIS"] = 4
    primary["NAXIS1"] = 4
    primary["NAXIS2"] = 4
    primary["NAXIS3"] = 1
    primary["NAXIS4"] = 1
    primary["CTYPE3"] = "FREQ"
    primary["CRVAL3"] = 55e6
    primary["CTYPE4"] = "STOKES"
    primary["CRVAL4"] = 1.0
    primary["DATE-OBS"] = "2025-01-20T04:03:33"
    primary["TELESCOP"] = "OVRO-LWA"
    primary["BMAJ"] = 0.25
    primary["BMIN"] = 0.20
    primary["BPA"] = 15.0

    post_wcs = _make_sin_wcs_header_str(nx=6, ny=5, crval1=181.0, crval2=46.0)
    payload = mod._fits_header_bytes_for_slice(
        primary, post_regrid_wcs_hdr=post_wcs, nl=6, nm=5
    )
    out = afits.Header.fromstring(payload.decode("utf-8"), sep="\n")
    assert int(out["NAXIS"]) == 4
    assert int(out["NAXIS1"]) == 6
    assert int(out["NAXIS2"]) == 5
    assert int(out["NAXIS3"]) == 1
    assert int(out["NAXIS4"]) == 1
    assert int(out["BITPIX"]) == -32
    assert str(out["CTYPE3"]).strip() == "FREQ"
    assert str(out["CTYPE4"]).strip() == "STOKES"
    assert out["DATE-OBS"] == "2025-01-20T04:03:33"
    assert out["TELESCOP"] == "OVRO-LWA"
    assert out["RESTFREQ"] == pytest.approx(55e6)
    assert out["CRVAL3"] == pytest.approx(55e6)
    assert out["CRVAL1"] == pytest.approx(181.0)
    assert out["CRVAL2"] == pytest.approx(46.0)
    assert out["CRVAL4"] == pytest.approx(1.0)
    assert "BSCALE" not in out
    from astropy.wcs import FITSFixedWarning
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        assert WCS(out).celestial.wcs.crval[0] == pytest.approx(181.0)


def test_fits_header_str_preserves_singleton_freq_stokes_cards() -> None:
    """Ingest stores ``NAXIS=4`` singleton FREQ/Stokes cards matching Zarr coords."""
    import xarray as xr
    from astropy.io import fits as afits

    from ovro_lwa_portal.accessor import _read_fits_header_str

    mod = _import_module()
    freq_hz = 55e6
    post_wcs = _make_sin_wcs_header_str(nx=4, ny=4, crval1=180.0, crval2=45.0)
    primary = afits.Header.fromstring(post_wcs, sep="\n")
    primary["RESTFREQ"] = freq_hz
    primary["DATE-OBS"] = "2025-01-20T04:03:33"
    payload = mod._fits_header_bytes_for_slice(
        primary,
        post_regrid_wcs_hdr=post_wcs,
        nl=4,
        nm=4,
        freq_hz=freq_hz,
        stokes=4.0,
    )
    ds = xr.Dataset(
        {
            "SKY": (
                ("time", "frequency", "polarization", "m", "l"),
                np.zeros((1, 1, 1, 4, 4), dtype=np.float32),
            ),
            "fits_header_str": (
                ("time", "frequency", "polarization"),
                np.array([[[np.bytes_(payload)]]], dtype=object),
            ),
        },
        coords={
            "time": ("time", np.array([60695.0])),
            "frequency": ("frequency", np.array([freq_hz])),
            "polarization": ("polarization", np.array([4.0])),
            "l": ("l", np.arange(4)),
            "m": ("m", np.arange(4)),
        },
    )
    hdr_str = _read_fits_header_str(ds, time_idx=0, freq_idx=0, pol_idx=0)
    out = afits.Header.fromstring(hdr_str, sep="\n")
    assert int(out["NAXIS"]) == 4
    assert int(out["NAXIS3"]) == 1
    assert int(out["NAXIS4"]) == 1
    assert str(out["CTYPE3"]).strip() == "FREQ"
    assert str(out["CTYPE4"]).strip() == "STOKES"
    assert float(out["CRVAL3"]) == pytest.approx(freq_hz)
    assert float(out["CRVAL4"]) == pytest.approx(4.0)
    assert float(ds.coords["frequency"].values[0]) == pytest.approx(float(out["CRVAL3"]))
    assert float(ds.coords["polarization"].values[0]) == pytest.approx(float(out["CRVAL4"]))


def test_assert_nonempty_fits_header_str_before_zarr_write_raises() -> None:
    import xarray as xr

    mod = _import_module()
    ds = xr.Dataset(
        {
            "SKY": (("time", "frequency", "m", "l"), np.zeros((1, 2, 4, 4))),
            "fits_header_str": (("time",), np.array([np.bytes_(b"")])),
        },
        coords={"time": [60000.0], "frequency": [45e6, 55e6], "l": np.arange(4), "m": np.arange(4)},
    )
    with pytest.raises(RuntimeError, match="fits_header_str is empty"):
        mod._assert_nonempty_fits_header_str_before_zarr_write(ds)


def test_new_ingest_omits_wcs_header_str_from_zarr(tmp_path: Path) -> None:
    """New ingest writes ``fits_header_str`` only — no legacy ``wcs_header_str`` on disk."""
    import xarray as xr

    mod = _import_module()
    out_zarr = tmp_path / "fits_headers.zarr"
    hdr_str = _make_sin_wcs_header_str(nx=8, ny=8, crval1=180.0, crval2=45.0)
    hdr = mod._fits_header_bytes_for_slice(
        fits.Header.fromstring(hdr_str, sep="\n"),
        post_regrid_wcs_hdr=hdr_str,
        nl=8,
        nm=8,
    )
    step = xr.Dataset(
        {
            "SKY": (
                ("time", "frequency", "polarization", "m", "l"),
                np.zeros((1, 2, 1, 8, 8), dtype=np.float32),
            ),
            "fits_header_str": (
                ("time", "frequency"),
                np.array([[np.bytes_(hdr), np.bytes_(hdr)]], dtype=object),
            ),
        },
        coords={
            "time": ("time", np.array([60695.17])),
            "frequency": ("frequency", np.array([45e6, 55e6])),
            "polarization": ("polarization", np.array([1.0])),
            "l": ("l", np.linspace(-1, 1, 8)),
            "m": ("m", np.linspace(-1, 1, 8)),
        },
    )
    mod._write_or_append_zarr(step, out_zarr, first_write=True, chunk_lm=4)
    with xr.open_zarr(out_zarr, consolidated=False) as ds:
        assert "fits_header_str" in ds
        assert "wcs_header_str" not in ds
        assert ds["fits_header_str"].dims == ("time", "frequency")


def test_read_wcs_header_str_derived_from_fits_header_str() -> None:
    """Portal WCS reads the celestial subset from ``fits_header_str``."""
    import xarray as xr

    from ovro_lwa_portal.accessor import _read_wcs_header_str

    hdr0 = _make_sin_wcs_header_str(nx=8, ny=8, crval1=180.0, crval2=45.0)
    hdr1 = _make_sin_wcs_header_str(nx=8, ny=8, crval1=190.0, crval2=50.0)
    mod = _import_module()
    enc0 = mod._fits_header_bytes_for_slice(
        fits.Header.fromstring(hdr0, sep="\n"),
        post_regrid_wcs_hdr=hdr0,
        nl=8,
        nm=8,
    )
    enc1 = mod._fits_header_bytes_for_slice(
        fits.Header.fromstring(hdr1, sep="\n"),
        post_regrid_wcs_hdr=hdr1,
        nl=8,
        nm=8,
    )
    ds = xr.Dataset(
        {
            "SKY": (
                ("time", "frequency", "polarization", "m", "l"),
                np.zeros((2, 1, 1, 8, 8), dtype=np.float32),
            ),
            "fits_header_str": (
                ("time",),
                np.array([np.bytes_(enc0), np.bytes_(enc1)], dtype=object),
            ),
        },
        coords={
            "time": ("time", np.arange(2, dtype=float)),
            "frequency": ("frequency", np.array([55e6])),
            "polarization": ("polarization", np.array([1.0])),
            "l": ("l", np.linspace(-0.1, 0.1, 8)),
            "m": ("m", np.linspace(-0.1, 0.1, 8)),
        },
    )
    wcs1 = _read_wcs_header_str(ds, time_idx=1)
    assert wcs1 is not None
    from astropy.io import fits as afits
    from astropy.wcs import WCS

    assert WCS(afits.Header.fromstring(wcs1, sep="\n")).wcs.crval[0] == pytest.approx(190.0)
    assert ds.radport._get_wcs(time_idx=1).wcs.crval[0] == pytest.approx(190.0)


def test_polarization_coord_stokes_values() -> None:
    """``polarization`` coord carries FITS Stokes codes from stored headers."""
    import xarray as xr

    mod = _import_module()
    hdr_i = _make_sin_wcs_header_str(nx=4, ny=4, crval1=180.0, crval2=45.0)
    primary_i = fits.Header.fromstring(hdr_i, sep="\n")
    primary_i["CTYPE4"] = "STOKES"
    primary_i["CRVAL4"] = 1.0
    bytes_i = mod._fits_header_bytes_for_slice(
        primary_i, post_regrid_wcs_hdr=hdr_i, nl=4, nm=4
    )
    ds = xr.Dataset(
        {
            "SKY": (("polarization", "m", "l"), np.zeros((1, 4, 4))),
            "fits_header_str": ((), np.bytes_(bytes_i)),
        },
        coords={
            "polarization": ("polarization", [99]),
            "l": ("l", np.arange(4)),
            "m": ("m", np.arange(4)),
        },
    )
    out = mod._set_polarization_coord_from_fits_headers(ds)
    assert float(out.coords["polarization"].values[0]) == pytest.approx(1.0)


def test_fits_header_str_not_collapsed_on_combine() -> None:
    """``fits_header_str`` must remain per-frequency after celestial harmonization."""
    import xarray as xr

    mod = _import_module()
    hdr_a = _make_sin_wcs_header_str(nx=4, ny=4, crval1=180.0, crval2=45.0)
    hdr_b = _make_sin_wcs_header_str(nx=4, ny=4, crval1=180.1, crval2=45.1)
    bytes_a = mod._fits_header_bytes_for_slice(
        fits.Header.fromstring(hdr_a, sep="\n"),
        post_regrid_wcs_hdr=hdr_a,
        nl=4,
        nm=4,
    )
    bytes_b = mod._fits_header_bytes_for_slice(
        fits.Header.fromstring(hdr_b, sep="\n"),
        post_regrid_wcs_hdr=hdr_b,
        nl=4,
        nm=4,
    )
    ra = np.full((2, 4, 4), 180.0)
    dec = np.full((2, 4, 4), 45.0)
    ds = xr.Dataset(
        {
            "SKY": (("frequency", "l", "m"), np.ones((2, 4, 4))),
            "fits_header_str": (
                ("frequency",),
                np.array([np.bytes_(bytes_a), np.bytes_(bytes_b)], dtype=object),
            ),
        },
        coords={
            "frequency": np.array([45e6, 55e6], dtype=float),
            "l": np.linspace(-0.1, 0.1, 4),
            "m": np.linspace(-0.1, 0.1, 4),
            "right_ascension": (("frequency", "l", "m"), ra),
            "declination": (("frequency", "l", "m"), dec),
        },
    )
    out = mod._harmonize_celestial_coords_independent_of_frequency(ds)
    assert out["fits_header_str"].dims == ("frequency",)
    assert bytes(out["fits_header_str"].isel(frequency=1).values.item()) == bytes_b


def _write_ovro_stokes_fits(
    path: Path,
    *,
    stokes: int,
    mhz: int = 18,
    time_key: str = "20240817_120000",
    pixel_value: float | None = None,
    n: int = 8,
    restfreq_hz: float | None = None,
) -> None:
    """Write a minimal OVRO-style 4D FITS with one frequency and one Stokes plane."""
    pix = float(stokes) if pixel_value is None else float(pixel_value)
    data = np.full((1, 1, n, n), pix, dtype=np.float32)
    freq_hz = float(restfreq_hz) if restfreq_hz is not None else float(mhz) * 1e6
    date_obs = datetime.strptime(time_key, "%Y%m%d_%H%M%S").strftime("%Y-%m-%dT%H:%M:%S.0")
    from astropy.time import Time

    mjd_obs = Time(date_obs, format="fits", scale="utc").mjd
    header = fits.Header(
        {
            "NAXIS": 4,
            "NAXIS1": n,
            "NAXIS2": n,
            "NAXIS3": 1,
            "NAXIS4": 1,
            "CTYPE1": "RA---SIN",
            "CTYPE2": "DEC--SIN",
            "CTYPE3": "FREQ",
            "CTYPE4": "STOKES",
            "CRVAL1": 180.0,
            "CRVAL2": 45.0,
            "CRVAL3": freq_hz,
            "CRVAL4": float(stokes),
            "CRPIX1": (n + 1) / 2.0,
            "CRPIX2": (n + 1) / 2.0,
            "CRPIX3": 1.0,
            "CRPIX4": 1.0,
            "CDELT1": -0.03,
            "CDELT2": 0.03,
            "CDELT3": 1.0,
            "CDELT4": 1.0,
            "CUNIT1": "deg",
            "CUNIT2": "deg",
            "CUNIT3": "Hz",
            "CUNIT4": "",
            "DATE-OBS": date_obs,
            "MJD-OBS": mjd_obs,
            "TELESCOP": "OVRO-LWA",
            "RADESYS": "FK5",
            "EQUINOX": 2000.0,
            "LONPOLE": 180.0,
            "BMAJ": 0.1,
            "BMIN": 0.1,
            "BUNIT": "Jy/beam",
        }
    )
    fits.PrimaryHDU(data=data, header=header).writeto(path)


def test_discover_groups_keeps_i_and_v_same_time_freq(tmp_path: Path) -> None:
    """Stokes I and V at the same time/subband must not be treated as duplicates."""
    mod = _import_module()
    time_key = "20240817_120000"
    f_i = tmp_path / f"18MHz-Clean-Snapshot-{time_key}-image-I.fits"
    f_v = tmp_path / f"18MHz-Clean-Snapshot-{time_key}-image-V.fits"
    _write_ovro_stokes_fits(f_i, stokes=1)
    _write_ovro_stokes_fits(f_v, stokes=4)

    groups = mod._discover_groups(tmp_path)

    assert time_key in groups
    assert {p.name for p in groups[time_key]} == {f_i.name, f_v.name}


def test_combine_time_step_stacks_i_and_v_polarization(tmp_path: Path) -> None:
    """One time step with I+V FITS stacks along ``polarization`` with sorted Stokes coords."""
    mod = _import_module()
    time_key = "20240817_120000"
    f_i = tmp_path / f"18MHz-Clean-Snapshot-{time_key}-image-I.fits"
    f_v = tmp_path / f"18MHz-Clean-Snapshot-{time_key}-image-V.fits"
    _write_ovro_stokes_fits(f_i, stokes=1, pixel_value=1.0)
    _write_ovro_stokes_fits(f_v, stokes=4, pixel_value=4.0)
    fixed_dir = tmp_path / "fixed"
    fixed_dir.mkdir()

    xds_t, freqs, _ = mod._combine_time_step(
        [f_i, f_v],
        fixed_dir,
        chunk_lm=0,
        fix_headers_on_demand=True,
    )

    assert int(xds_t.sizes["polarization"]) == 2
    assert list(np.sort(xds_t.coords["polarization"].values)) == [1.0, 4.0]
    assert len(freqs) == 1
    pol_vals = list(xds_t.coords["polarization"].values)
    i_idx = pol_vals.index(1.0)
    v_idx = pol_vals.index(4.0)
    assert float(np.nanmean(xds_t["SKY"].isel(polarization=i_idx).values)) == pytest.approx(1.0)
    assert float(np.nanmean(xds_t["SKY"].isel(polarization=v_idx).values)) == pytest.approx(4.0)
    assert "fits_header_str" in xds_t.data_vars
    assert "polarization" in xds_t["fits_header_str"].dims


def test_stokes_key_prefers_basename_over_misleading_header(tmp_path: Path) -> None:
    """Stacking uses ``NNMHz-I-`` / ``NNMHz-V-`` basename tokens, not stray ``CRVAL4``."""
    mod = _import_module()
    time_key = "20240817_120000"
    f_i = tmp_path / f"82MHz-I-Taper-602s-Robust-0-{time_key}-image.pbcorr_dewarped.fits"
    f_v = tmp_path / f"82MHz-V-Taper-602s-Robust-0-{time_key}-image.pbcorr_dewarped.fits"
    # Headers claim Stokes Q (2) and U (3) while basenames encode I and V.
    _write_ovro_stokes_fits(f_i, stokes=2, pixel_value=1.0)
    _write_ovro_stokes_fits(f_v, stokes=3, pixel_value=4.0)
    fixed_dir = tmp_path / "fixed"
    fixed_dir.mkdir()

    xds_t, _, _ = mod._combine_time_step(
        [f_i, f_v],
        fixed_dir,
        chunk_lm=0,
        fix_headers_on_demand=True,
    )

    assert int(xds_t.sizes["polarization"]) == 2
    assert list(np.sort(xds_t.coords["polarization"].values)) == [1.0, 4.0]


def test_combine_time_step_dedupes_two_i_in_different_discovery_freq_bins(tmp_path: Path) -> None:
    """Two I products with the same ``NNMHz`` basename but header Hz in different 23 kHz bins.

    Discovery keeps both (different frequency bins, same Stokes I). Stack must not
  produce ``polarization=[1, 4, 1]`` when combined with V at that subband.
    """
    mod = _import_module()
    time_key = "20260419_071829"
    f_i_a = tmp_path / f"82MHz-I-Taper-602s-Robust-0-{time_key}-image.pbcorr_dewarped.fits"
    f_i_b = tmp_path / f"82MHz-I-Taper-602s-Robust-0-{time_key}-image-rerun.pbcorr_dewarped.fits"
    f_v = tmp_path / f"82MHz-V-Taper-602s-Robust-0-{time_key}-image.pbcorr_dewarped.fits"
    _write_ovro_stokes_fits(f_i_a, stokes=1, mhz=82, time_key=time_key, restfreq_hz=82.0e6)
    _write_ovro_stokes_fits(
        f_i_b, stokes=1, mhz=82, time_key=time_key, restfreq_hz=82.0e6 + 50_000.0
    )
    _write_ovro_stokes_fits(f_v, stokes=4, mhz=82, time_key=time_key, restfreq_hz=82.0e6)

    groups = mod._discover_groups(tmp_path)
    assert len(groups[time_key]) == 3

    fixed_dir = tmp_path / "fixed"
    fixed_dir.mkdir()
    xds_t, _, _ = mod._combine_time_step(
        groups[time_key],
        fixed_dir,
        chunk_lm=0,
        fix_headers_on_demand=True,
    )

    assert int(xds_t.sizes["polarization"]) == 2
    assert list(np.sort(xds_t.coords["polarization"].values)) == [1.0, 4.0]


def test_align_time_step_to_polarization_grid_fills_missing_stokes() -> None:
    """Append alignment adds NaN planes for Stokes present in the store but not the step."""
    import xarray as xr

    mod = _import_module()
    ds = xr.Dataset(
        {"SKY": (("polarization", "m", "l"), np.ones((2, 2, 2)))},
        coords={
            "polarization": ("polarization", [1.0, 4.0]),
            "l": ("l", [0, 1]),
            "m": ("m", [0, 1]),
        },
    )
    aligned = mod._align_time_step_to_polarization_grid(ds, np.array([1.0, 4.0, 2.0]))
    assert int(aligned.sizes["polarization"]) == 3
    assert list(aligned.coords["polarization"].values) == [1.0, 4.0, 2.0]
    q_plane = aligned["SKY"].isel(polarization=2).values
    assert np.isnan(q_plane).all()


def test_convert_fits_dir_to_zarr_i_and_v_single_store(tmp_path: Path) -> None:
    """End-to-end: separate I and V FITS ingest into one Zarr with ``polarization=[1, 4]``."""
    import xarray as xr

    mod = _import_module()
    time_key = "20240817_120000"
    f_i = tmp_path / f"18MHz-Clean-Snapshot-{time_key}-image-I.fits"
    f_v = tmp_path / f"18MHz-Clean-Snapshot-{time_key}-image-V.fits"
    _write_ovro_stokes_fits(f_i, stokes=1, pixel_value=1.0)
    _write_ovro_stokes_fits(f_v, stokes=4, pixel_value=4.0)
    out_dir = tmp_path / "zarr_out"
    fixed_dir = tmp_path / "fixed"

    out_zarr = mod.convert_fits_dir_to_zarr(
        input_dir=tmp_path,
        out_dir=out_dir,
        zarr_name="combined_iv.zarr",
        fixed_dir=fixed_dir,
        chunk_lm=4,
        rebuild=True,
        consolidate_metadata_at_end=False,
    )

    with xr.open_zarr(out_zarr, consolidated=False) as ds:
        assert "fits_header_str" in ds
        assert "wcs_header_str" not in ds
        assert int(ds.sizes["polarization"]) == 2
        assert list(np.sort(ds.coords["polarization"].values)) == [1.0, 4.0]
        assert int(ds.sizes["time"]) == 1
        pol_vals = list(ds.coords["polarization"].values)
        i_idx = pol_vals.index(1.0)
        v_idx = pol_vals.index(4.0)
        assert float(np.nanmean(ds["SKY"].isel(time=0, polarization=i_idx).values)) == pytest.approx(
            1.0
        )
        assert float(np.nanmean(ds["SKY"].isel(time=0, polarization=v_idx).values)) == pytest.approx(
            4.0
        )
        from astropy.io import fits as afits

        from ovro_lwa_portal.accessor import _read_fits_header_str

        for pol_idx, stokes in ((i_idx, 1.0), (v_idx, 4.0)):
            hdr_str = _read_fits_header_str(ds, time_idx=0, freq_idx=0, pol_idx=pol_idx)
            hdr = afits.Header.fromstring(hdr_str, sep="\n")
            assert int(hdr["NAXIS"]) == 4
            assert int(hdr["NAXIS3"]) == 1
            assert int(hdr["NAXIS4"]) == 1
            assert str(hdr["CTYPE3"]).strip() == "FREQ"
            assert str(hdr["CTYPE4"]).strip() == "STOKES"
            assert float(hdr["CRVAL4"]) == pytest.approx(stokes)


def test_align_time_step_does_not_duplicate_i_for_legacy_store_template() -> None:
    """Label-based align must not copy I into a second Stokes-1 slot (reindex would)."""
    import xarray as xr

    mod = _import_module()
    ds = xr.Dataset(
        {"SKY": (("polarization",), np.array([10.0, 40.0], dtype=np.float64))},
        coords={"polarization": ("polarization", [1.0, 4.0])},
    )
    aligned = mod._align_time_step_to_polarization_grid(ds, np.array([1.0, 4.0, 1.0]))
    assert list(aligned.coords["polarization"].values) == [1.0, 4.0, 1.0]
    assert float(aligned["SKY"].isel(polarization=0).values) == pytest.approx(10.0)
    assert float(aligned["SKY"].isel(polarization=1).values) == pytest.approx(40.0)
    assert np.isnan(float(aligned["SKY"].isel(polarization=2).values))


def test_canonicalize_polarization_coord_keeps_best_i_plane() -> None:
    """Duplicate I planes collapse to the one with more finite data."""
    import xarray as xr

    mod = _import_module()
    ds = xr.Dataset(
        {
            "SKY": (
                ("polarization", "m", "l"),
                np.array(
                    [
                        [[1.0, 1.0], [1.0, 1.0]],
                        [[4.0, 4.0], [4.0, 4.0]],
                        [[np.nan, np.nan], [np.nan, np.nan]],
                    ]
                ),
            )
        },
        coords={
            "polarization": ("polarization", [1.0, 4.0, 1.0]),
            "l": ("l", [0, 1]),
            "m": ("m", [0, 1]),
        },
    )
    out = mod._canonicalize_polarization_coord(ds)
    assert list(out.coords["polarization"].values) == [1.0, 4.0]
    assert float(out["SKY"].isel(polarization=0).mean()) == pytest.approx(1.0)
    assert float(out["SKY"].isel(polarization=1).mean()) == pytest.approx(4.0)


def test_repair_zarr_polarization_drops_duplicate_plane(tmp_path: Path) -> None:
    """In-place repair rewrites arrays to unique Stokes polarization."""
    import xarray as xr

    mod = _import_module()
    store = tmp_path / "dup_pol.zarr"
    ds = xr.Dataset(
        {
            "SKY": (
                ("time", "frequency", "polarization", "l", "m"),
                np.array(
                    [[[[[1.0]], [[4.0]], [[1.0]]]]],
                    dtype=np.float64,
                ),
            ),
            "fits_header_str": (
                ("time", "frequency", "polarization"),
                np.array([[["hdr-i", "hdr-v", "hdr-dup"]]], dtype="S8"),
            ),
        },
        coords={
            "time": [0.0],
            "frequency": [70e6],
            "polarization": [1.0, 4.0, 1.0],
            "l": [0],
            "m": [0],
        },
    )
    ds.to_zarr(store, mode="w", consolidated=False)

    result = mod.repair_zarr_polarization(store, skip_backup=True)
    assert result["changed"] is True
    assert result["polarization_after"] == [1.0, 4.0]

    with xr.open_zarr(store, consolidated=False) as repaired:
        assert list(repaired.coords["polarization"].values) == [1.0, 4.0]
        assert int(repaired.sizes["polarization"]) == 2
        assert float(repaired["SKY"].isel(polarization=0).values.item()) == pytest.approx(1.0)
        assert float(repaired["SKY"].isel(polarization=1).values.item()) == pytest.approx(4.0)
