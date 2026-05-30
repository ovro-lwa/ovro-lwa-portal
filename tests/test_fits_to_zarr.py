"""Tests for FITS to Zarr conversion utilities."""

from __future__ import annotations

import re
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
        attrs={"fits_wcs_header": hdr_ref},
    )
    xds_ref = xds_ref.assign(wcs_header_str=((), np.bytes_(hdr_ref.encode("utf-8"))))

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
        attrs={"fits_wcs_header": hdr_sm},
    )
    xds_sm = xds_sm.assign(wcs_header_str=((), np.bytes_(hdr_sm.encode("utf-8"))))

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
    assert bytes(out["wcs_header_str"].values.item()) == out_hdr_str.encode("utf-8")


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
        return xr.Dataset(
            {"SKY": (("m", "l"), sky.copy())},
            coords={
                "l": ("l", l_arr),
                "m": ("m", m_arr),
                "right_ascension": (("m", "l"), np.full((n, n), 180.0)),
                "declination": (("m", "l"), np.full((n, n), 45.0)),
            },
            attrs={"fits_wcs_header": hdr},
        ).assign(wcs_header_str=((), np.bytes_(hdr.encode("utf-8"))))

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
      ``CRVAL1``/``CRVAL2`` reflect the FK5 zenith at *that* instant.
    * At later time steps, each source FITS gets re-stamped by ``_fix_headers`` with
      *its own* obs-time zenith ``CRVAL``.
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
    src_crval = (200.0, 40.0)  # later obs time → zenith shifted in RA, slight Dec drift

    hdr_ref = _make_sin_wcs_header_str(
        nx=n_ref, ny=n_ref, crval1=ref_crval[0], crval2=ref_crval[1]
    )
    xds_ref = xr.Dataset(
        data_vars={"SKY": (("m", "l"), sky_ref)},
        coords={
            "l": ("l", l_ref),
            "m": ("m", m_ref),
            "right_ascension": (("l", "m"), np.full((n_ref, n_ref), ref_crval[0])),
            "declination": (("l", "m"), np.full((n_ref, n_ref), ref_crval[1])),
        },
        attrs={"fits_wcs_header": hdr_ref},
    ).assign(wcs_header_str=((), np.bytes_(hdr_ref.encode("utf-8"))))

    n_sm = 4
    l_sm = np.linspace(-0.5, 0.5, n_sm)
    m_sm = np.linspace(-0.5, 0.5, n_sm)
    sky_sm = rng.standard_normal((n_sm, n_sm))
    hdr_sm = _make_sin_wcs_header_str(
        nx=n_sm, ny=n_sm, crval1=src_crval[0], crval2=src_crval[1]
    )
    xds_sm = xr.Dataset(
        data_vars={"SKY": (("m", "l"), sky_sm)},
        coords={
            "l": ("l", l_sm),
            "m": ("m", m_sm),
            "right_ascension": (("m", "l"), np.zeros((n_sm, n_sm))),
            "declination": (("m", "l"), np.zeros((n_sm, n_sm))),
        },
        attrs={"fits_wcs_header": hdr_sm},
    ).assign(wcs_header_str=((), np.bytes_(hdr_sm.encode("utf-8"))))

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
    assert bytes(out["wcs_header_str"].values.item()) == out_hdr_str.encode("utf-8")


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
    monkeypatch.setattr(
        mod,
        "_peek_lm_shape",
        lambda fp: (32, 32) if fp.name.startswith("small") else (64, 64),
    )

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
    monkeypatch.setattr(
        mod,
        "_peek_lm_shape",
        lambda fp: (32, 32) if fp.name.startswith("small") else (64, 64),
    )

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
    """Header read failures should drop the file rather than abort the run."""
    import logging

    mod = _import_module()

    real = tmp_path / "real.fits"
    fits.PrimaryHDU(data=[[1.0]], header=fits.Header({"BMAJ": 0.1, "BMIN": 0.1})).writeto(real)
    missing = tmp_path / "does_not_exist.fits"

    caplog.set_level(logging.WARNING, logger="ovro_lwa_portal.fits_to_zarr_xradio")
    filtered = mod._filter_invalid_beam_files({"t": [real, missing]})

    assert filtered == {"t": [real]}
    assert "does_not_exist.fits" in caplog.text
    assert "could not read primary header" in caplog.text


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
    assert hdr["CDELT4"] == pytest.approx(1.0)
    assert out_data is not None
    assert out_data.shape == (1, 1, 4, 4)


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
    monkeypatch.setattr(mod, "_filter_invalid_beam_files", lambda groups: groups)
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
    monkeypatch.setattr(mod, "_filter_invalid_beam_files", lambda groups: groups)
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
    monkeypatch.setattr(mod, "_filter_invalid_beam_files", lambda groups: groups)
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


def test_fix_headers_sets_crval_to_fk5_zenith_from_filename_image_stamp(tmp_path: Path):
    """_fix_headers sets CRVAL1/2 from FK5 zenith at ``-image-YYYYMMDD_HHMMSS`` in the basename."""
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
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
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

    # 20241218_030201 → 2024-12-18 03:02:01 UTC (not DATE-OBS).
    assert hdr["CRVAL1"] == pytest.approx(14.0996845, rel=0, abs=1e-4)
    assert hdr["CRVAL2"] == pytest.approx(37.0948037, rel=0, abs=1e-4)
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
