"""Tests for ``ovro_lwa_portal.export_fits`` and FITS→Zarr→FITS→Zarr round-trip.

Round-trip tests compare exported FITS headers to the **pixel-faithful reference**
stored in ``fits_header_str`` after ingest (not pre-regrid input when regrid
ran). Header parity uses ``HEADER_ALLOWLIST``; pixels use ``rtol=1e-5``,
``atol=1e-4`` Jy/beam.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr
from astropy.io import fits

from ovro_lwa_portal import export_fits, open_dataset, write_fits_slice
from ovro_lwa_portal.accessor import _read_fits_header_str
from ovro_lwa_portal.fits_to_zarr_xradio import _fits_header_bytes_for_slice

from tests.test_fits_to_zarr import _import_module, _make_sin_wcs_header_str, _write_ovro_stokes_fits

if TYPE_CHECKING:
    from ovro_lwa_portal import fits_to_zarr_xradio as fits_mod_type

HEADER_ALLOWLIST: tuple[str, ...] = (
    "CRVAL1",
    "CRVAL2",
    "CRVAL3",
    "CRVAL4",
    "CDELT1",
    "CDELT2",
    "CDELT3",
    "CDELT4",
    "CRPIX1",
    "CRPIX2",
    "CRPIX3",
    "CRPIX4",
    "CTYPE1",
    "CTYPE2",
    "CTYPE3",
    "CTYPE4",
    "CUNIT3",
    "RESTFREQ",
    "RESTFRQ",
    "BMAJ",
    "BMIN",
    "BPA",
    "BUNIT",
    "DATE-OBS",
    "TELESCOP",
    "BITPIX",
    "NAXIS",
    "NAXIS1",
    "NAXIS2",
    "NAXIS3",
    "NAXIS4",
)

_STRING_HEADER_KEYS = frozenset(
    {
        "CTYPE1",
        "CTYPE2",
        "CTYPE3",
        "CTYPE4",
        "CUNIT3",
        "BUNIT",
        "DATE-OBS",
        "TELESCOP",
    }
)


def _fits_mod() -> fits_mod_type:
    return _import_module()


def _assert_header_allowlist_match(
    reference: fits.Header,
    exported: fits.Header,
    *,
    context: str = "",
) -> None:
    prefix = f"{context}: " if context else ""
    for key in HEADER_ALLOWLIST:
        if key not in reference:
            continue
        assert key in exported, f"{prefix}missing exported keyword {key!r}"
        ref_val = reference[key]
        got_val = exported[key]
        if key in _STRING_HEADER_KEYS or isinstance(ref_val, str):
            assert str(got_val).strip() == str(ref_val).strip(), (
                f"{prefix}{key}: {got_val!r} != {ref_val!r}"
            )
        else:
            assert float(got_val) == pytest.approx(float(ref_val)), (
                f"{prefix}{key}: {got_val} != {ref_val}"
            )


def _pixel_faithful_reference_header(
    ds: xr.Dataset,
    *,
    time_idx: int,
    freq_idx: int,
    pol_idx: int,
) -> fits.Header:
    """Expected export header from persisted ``fits_header_str`` + live coords."""
    return export_fits.build_fits_header(
        ds,
        time_idx=time_idx,
        freq_idx=freq_idx,
        pol_idx=pol_idx,
    )


def _write_provenance_fits(
    path: Path,
    *,
    stokes: int,
    mhz: int = 18,
    time_key: str = "20240817_120000",
    pixel_value: float | None = None,
    n: int = 8,
    crval1: float = 180.0,
    crval2: float = 45.0,
    history: str | None = "ovro-lwa-portal round-trip test",
) -> None:
    """Write a minimal OVRO-style FITS with provenance keywords for round-trip tests."""
    _write_ovro_stokes_fits(
        path,
        stokes=stokes,
        mhz=mhz,
        time_key=time_key,
        pixel_value=pixel_value,
        n=n,
    )
    with fits.open(path, mode="update") as hdul:
        hdul[0].header["CRVAL1"] = crval1
        hdul[0].header["CRVAL2"] = crval2
        if history is not None:
            hdul[0].header["HISTORY"] = history


@dataclass(frozen=True)
class RoundtripResult:
    zarr_a: Path
    exported_fits: Path
    zarr_b: Path
    ds_a: xr.Dataset
    ds_b: xr.Dataset


def _roundtrip_fits_zarr_fits_zarr(
    input_fits: list[Path],
    tmp_path: Path,
    *,
    time_idx: int = 0,
    freq_idx: int = 0,
    pol_idx: int = 0,
    chunk_lm: int = 4,
    zarr_name: str = "zarr_a.zarr",
    exported_name: str = "exported.fits",
    pixel_rtol: float = 0.0,
    pixel_atol: float = 0.0,
) -> RoundtripResult:
    """Run FITS→Zarr→FITS→Zarr for one exported slice."""
    mod = _fits_mod()
    in_dir = tmp_path / "input_fits"
    in_dir.mkdir(parents=True, exist_ok=True)
    for src in input_fits:
        shutil.copy2(src, in_dir / src.name)

    zarr_a = mod.convert_fits_dir_to_zarr(
        input_dir=in_dir,
        out_dir=tmp_path / "zarr_a",
        zarr_name=zarr_name,
        fixed_dir=tmp_path / "fixed_a",
        chunk_lm=chunk_lm,
        rebuild=True,
        consolidate_metadata_at_end=False,
    )
    ds_a = open_dataset(zarr_a, chunks={})

    exported_fits = tmp_path / exported_name
    write_fits_slice(
        ds_a,
        exported_fits,
        time_idx=time_idx,
        freq_idx=freq_idx,
        pol_idx=pol_idx,
        overwrite=True,
    )

    ref_hdr = _pixel_faithful_reference_header(
        ds_a,
        time_idx=time_idx,
        freq_idx=freq_idx,
        pol_idx=pol_idx,
    )
    with fits.open(exported_fits) as hdul:
        assert hdul[0].data is not None
        assert hdul[0].data.shape == (1, 1, ds_a.sizes["m"], ds_a.sizes["l"])
        assert int(hdul[0].header["NAXIS3"]) == 1
        assert int(hdul[0].header["NAXIS4"]) == 1
        _assert_header_allowlist_match(ref_hdr, hdul[0].header)

    reingest_dir = tmp_path / "reingest_fits"
    reingest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported_fits, reingest_dir / exported_name)
    zarr_b = mod.convert_fits_dir_to_zarr(
        input_dir=reingest_dir,
        out_dir=tmp_path / "zarr_b",
        zarr_name="zarr_b.zarr",
        fixed_dir=tmp_path / "fixed_b",
        chunk_lm=chunk_lm,
        rebuild=True,
        consolidate_metadata_at_end=False,
    )
    ds_b = open_dataset(zarr_b, chunks={})

    sky_a = (
        ds_a["SKY"]
        .isel(time=time_idx, frequency=freq_idx, polarization=pol_idx)
        .values.astype(np.float32)
    )
    sky_b = (
        ds_b["SKY"]
        .isel(time=0, frequency=0, polarization=0)
        .values.astype(np.float32)
    )
    np.testing.assert_allclose(sky_a, sky_b, rtol=pixel_rtol, atol=pixel_atol)

    hdr_a = fits.Header.fromstring(
        _read_fits_header_str(ds_a, time_idx=time_idx, freq_idx=freq_idx, pol_idx=pol_idx),
        sep="\n",
    )
    hdr_b = fits.Header.fromstring(
        _read_fits_header_str(ds_b, time_idx=0, freq_idx=0, pol_idx=0),
        sep="\n",
    )
    _assert_header_allowlist_match(hdr_a, hdr_b, context="zarr_b vs zarr_a")

    freq_a = float(np.asarray(ds_a.coords["frequency"].values).ravel()[freq_idx])
    freq_b = float(np.asarray(ds_b.coords["frequency"].values).ravel()[0])
    assert freq_b == pytest.approx(freq_a)

    pol_a = float(np.asarray(ds_a.coords["polarization"].values).ravel()[pol_idx])
    pol_b = float(np.asarray(ds_b.coords["polarization"].values).ravel()[0])
    assert pol_b == pytest.approx(pol_a)

    return RoundtripResult(
        zarr_a=zarr_a,
        exported_fits=exported_fits,
        zarr_b=zarr_b,
        ds_a=ds_a,
        ds_b=ds_b,
    )


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


def _synthetic_export_dataset_iv(
    *,
    nl: int = 8,
    nm: int = 8,
    freq_hz: float = 55e6,
) -> xr.Dataset:
    """Synthetic Zarr-like dataset with Stokes I and V planes."""
    import ovro_lwa_portal  # noqa: F401 — register radport accessor

    ds_i = _synthetic_export_dataset(nl=nl, nm=nm, freq_hz=freq_hz, stokes=1.0)
    ds_v = _synthetic_export_dataset(nl=nl, nm=nm, freq_hz=freq_hz, stokes=4.0)
    header_i = ds_i["fits_header_str"].isel(time=0, frequency=0, polarization=0).item()
    header_v = ds_v["fits_header_str"].isel(time=0, frequency=0, polarization=0).item()
    return xr.Dataset(
        {
            "SKY": (
                ("time", "frequency", "polarization", "m", "l"),
                np.stack(
                    [
                        ds_i["SKY"].isel(polarization=0).values,
                        ds_v["SKY"].isel(polarization=0).values,
                    ],
                    axis=2,
                ),
            ),
            "fits_header_str": (
                ("time", "frequency", "polarization"),
                np.array([[[header_i, header_v]]], dtype=object),
            ),
        },
        coords={
            "time": ds_i.coords["time"],
            "frequency": ds_i.coords["frequency"],
            "polarization": ("polarization", np.array([1.0, 4.0])),
            "l": ds_i.coords["l"],
            "m": ds_i.coords["m"],
        },
    )


class TestRadportExportFits:
    """Tests for RadportAccessor FITS export wrappers (Phase 2)."""

    def test_build_fits_hdu_delegates_to_export_module(self) -> None:
        import ovro_lwa_portal  # noqa: F401

        ds = _synthetic_export_dataset()
        hdu = ds.radport.build_fits_hdu(time_idx=0, freq_idx=0, pol_idx=0)
        assert int(hdu.header["NAXIS"]) == 4
        assert hdu.data.shape == (1, 1, ds.sizes["m"], ds.sizes["l"])

    def test_export_fits_writes_single_file(self, tmp_path: Path) -> None:
        import ovro_lwa_portal  # noqa: F401

        ds = _synthetic_export_dataset()
        files = ds.radport.export_fits(tmp_path, time_indices=[0], freq_indices=[0])
        assert len(files) == 1
        assert Path(files[0]).exists()
        assert files[0].endswith("_s1.fits")

    def test_export_fits_writes_all_time_freq_combinations(self, tmp_path: Path) -> None:
        import ovro_lwa_portal  # noqa: F401

        post_wcs = _make_sin_wcs_header_str(nx=4, ny=4, crval1=180.0, crval2=45.0)
        payloads = []
        for freq, stokes in ((45e6, 1.0), (55e6, 1.0)):
            primary = fits.Header.fromstring(post_wcs, sep="\n")
            primary["RESTFREQ"] = freq
            payloads.append(
                np.bytes_(
                    _fits_header_bytes_for_slice(
                        primary,
                        post_regrid_wcs_hdr=post_wcs,
                        nl=4,
                        nm=4,
                        freq_hz=freq,
                        stokes=stokes,
                    )
                )
            )
        ds = xr.Dataset(
            {
                "SKY": (
                    ("time", "frequency", "polarization", "m", "l"),
                    np.zeros((2, 2, 1, 4, 4), dtype=np.float32),
                ),
                "fits_header_str": (
                    ("time", "frequency", "polarization"),
                    np.array(
                        [
                            [[payloads[0]], [payloads[1]]],
                            [[payloads[0]], [payloads[1]]],
                        ],
                        dtype=object,
                    ),
                ),
            },
            coords={
                "time": ("time", np.array([60695.0, 60695.1])),
                "frequency": ("frequency", np.array([45e6, 55e6])),
                "polarization": ("polarization", np.array([1.0])),
                "l": ("l", np.arange(4)),
                "m": ("m", np.arange(4)),
            },
        )
        files = ds.radport.export_fits(tmp_path)
        assert len(files) == 4

    def test_export_fits_pol_indices_writes_iv(self, tmp_path: Path) -> None:
        import ovro_lwa_portal  # noqa: F401

        ds = _synthetic_export_dataset_iv()
        files = ds.radport.export_fits(
            tmp_path,
            time_indices=[0],
            freq_indices=[0],
            pol_indices=[0, 1],
        )
        assert len(files) == 2
        stokes_in_names = sorted(Path(p).name for p in files)
        assert any("_s1.fits" in name for name in stokes_in_names)
        assert any("_s4.fits" in name for name in stokes_in_names)

    def test_export_fits_custom_filename_template(self, tmp_path: Path) -> None:
        import ovro_lwa_portal  # noqa: F401

        ds = _synthetic_export_dataset()
        files = ds.radport.export_fits(
            tmp_path,
            time_indices=[0],
            freq_indices=[0],
            filename_template="export_{time_idx}_{freq_idx}_p{pol_idx}.fits",
        )
        assert files[0].endswith("export_0_0_p0.fits")


class TestExportFitsRoundtrip:
    """FITS→Zarr→FITS→Zarr round-trip and header parity tests (Phase 3)."""

    pytestmark = pytest.mark.filterwarnings(
        "ignore:Zarr store .* has small on-disk l/m chunks:UserWarning",
        "ignore:Failed to open Zarr store with consolidated metadata:RuntimeWarning",
        "ignore:In a future version of xarray the default value for data_vars:FutureWarning",
        "ignore:In a future version of xarray the default value for coords:FutureWarning",
        "ignore:Using fallback metadata for:UserWarning",
    )

    def test_ingest_fits_header_str_matches_pixel_faithful_reference(
        self, tmp_path: Path
    ) -> None:
        """Ingested ``fits_header_str`` matches export's pixel-faithful reference."""
        fits_path = tmp_path / "18MHz-Clean-Snapshot-20240817_120000-image-I.fits"
        _write_provenance_fits(fits_path, stokes=1, pixel_value=2.0)
        result = _roundtrip_fits_zarr_fits_zarr([fits_path], tmp_path / "rt_ref")
        ref_hdr = _pixel_faithful_reference_header(result.ds_a, time_idx=0, freq_idx=0, pol_idx=0)
        stored_hdr = fits.Header.fromstring(
            _read_fits_header_str(result.ds_a, time_idx=0, freq_idx=0, pol_idx=0),
            sep="\n",
        )
        _assert_header_allowlist_match(ref_hdr, stored_hdr, context="stored vs export ref")

    def test_roundtrip_single_fits_file(self, tmp_path: Path) -> None:
        fits_path = tmp_path / "18MHz-Clean-Snapshot-20240817_120000-image-I.fits"
        _write_provenance_fits(fits_path, stokes=1, pixel_value=3.5)
        _roundtrip_fits_zarr_fits_zarr([fits_path], tmp_path / "rt_single")

    def test_roundtrip_multi_time_incremental(self, tmp_path: Path) -> None:
        f0 = tmp_path / "18MHz-Clean-Snapshot-20240817_120000-image-I.fits"
        f1 = tmp_path / "18MHz-Clean-Snapshot-20240817_130000-image-I.fits"
        _write_provenance_fits(f0, stokes=1, time_key="20240817_120000", pixel_value=1.0)
        _write_provenance_fits(
            f1,
            stokes=1,
            time_key="20240817_130000",
            pixel_value=2.0,
            crval1=190.0,
            crval2=50.0,
        )
        result = _roundtrip_fits_zarr_fits_zarr([f0, f1], tmp_path / "rt_time")
        assert int(result.ds_a.sizes["time"]) == 2
        hdr0 = fits.Header.fromstring(
            _read_fits_header_str(result.ds_a, time_idx=0, freq_idx=0, pol_idx=0),
            sep="\n",
        )
        hdr1 = fits.Header.fromstring(
            _read_fits_header_str(result.ds_a, time_idx=1, freq_idx=0, pol_idx=0),
            sep="\n",
        )
        assert float(hdr0["CRVAL1"]) == pytest.approx(180.0)
        assert float(hdr1["CRVAL1"]) == pytest.approx(190.0)

    def test_roundtrip_multi_frequency_subband(self, tmp_path: Path) -> None:
        time_key = "20240817_120000"
        f18 = tmp_path / f"18MHz-Clean-Snapshot-{time_key}-image-I.fits"
        f23 = tmp_path / f"23MHz-Clean-Snapshot-{time_key}-image-I.fits"
        _write_provenance_fits(f18, stokes=1, mhz=18, pixel_value=1.0, crval1=180.0)
        _write_provenance_fits(
            f23,
            stokes=1,
            mhz=23,
            pixel_value=2.0,
            crval1=180.5,
            crval2=45.5,
        )
        result = _roundtrip_fits_zarr_fits_zarr([f18, f23], tmp_path / "rt_freq", freq_idx=1)
        assert int(result.ds_a.sizes["frequency"]) == 2
        hdr = fits.Header.fromstring(
            _read_fits_header_str(result.ds_a, time_idx=0, freq_idx=1, pol_idx=0),
            sep="\n",
        )
        assert float(hdr["RESTFREQ"]) == pytest.approx(23e6)
        assert float(hdr["CRVAL1"]) == pytest.approx(180.5)
        assert float(hdr["CRVAL2"]) == pytest.approx(45.5)

    def test_roundtrip_polarization_slice(self, tmp_path: Path) -> None:
        import ovro_lwa_portal  # noqa: F401

        time_key = "20240817_120000"
        f_i = tmp_path / f"18MHz-Clean-Snapshot-{time_key}-image-I.fits"
        f_v = tmp_path / f"18MHz-Clean-Snapshot-{time_key}-image-V.fits"
        _write_provenance_fits(f_i, stokes=1, pixel_value=1.0)
        _write_provenance_fits(f_v, stokes=4, pixel_value=4.0)
        mod = _fits_mod()
        in_dir = tmp_path / "input_iv"
        in_dir.mkdir()
        shutil.copy2(f_i, in_dir / f_i.name)
        shutil.copy2(f_v, in_dir / f_v.name)
        zarr_a = mod.convert_fits_dir_to_zarr(
            input_dir=in_dir,
            out_dir=tmp_path / "zarr_iv",
            zarr_name="iv.zarr",
            fixed_dir=tmp_path / "fixed_iv",
            chunk_lm=4,
            rebuild=True,
            consolidate_metadata_at_end=False,
        )
        ds = open_dataset(zarr_a, chunks={})
        export_dir = tmp_path / "exported_iv"
        files = ds.radport.export_fits(
            export_dir,
            time_indices=[0],
            freq_indices=[0],
            pol_indices=[0, 1],
        )
        assert len(files) == 2
        for pol_idx, stokes in ((0, 1.0), (1, 4.0)):
            with fits.open(files[pol_idx]) as hdul:
                assert int(hdul[0].header["NAXIS4"]) == 1
                assert float(hdul[0].header["CRVAL4"]) == pytest.approx(stokes)

    def test_roundtrip_stokes_i_and_v_wcs_equivalent(self, tmp_path: Path) -> None:
        time_key = "20240817_120000"
        f_i = tmp_path / f"18MHz-Clean-Snapshot-{time_key}-image-I.fits"
        f_v = tmp_path / f"18MHz-Clean-Snapshot-{time_key}-image-V.fits"
        _write_provenance_fits(f_i, stokes=1, pixel_value=1.0)
        _write_provenance_fits(f_v, stokes=4, pixel_value=9.0)
        mod = _fits_mod()
        in_dir = tmp_path / "input_iv"
        in_dir.mkdir()
        shutil.copy2(f_i, in_dir / f_i.name)
        shutil.copy2(f_v, in_dir / f_v.name)
        zarr_a = mod.convert_fits_dir_to_zarr(
            input_dir=in_dir,
            out_dir=tmp_path / "zarr_iv",
            zarr_name="iv.zarr",
            fixed_dir=tmp_path / "fixed_iv",
            chunk_lm=4,
            rebuild=True,
            consolidate_metadata_at_end=False,
        )
        ds = open_dataset(zarr_a, chunks={})
        out_i = tmp_path / "export_i.fits"
        out_v = tmp_path / "export_v.fits"
        write_fits_slice(ds, out_i, time_idx=0, freq_idx=0, pol_idx=0, overwrite=True)
        write_fits_slice(ds, out_v, time_idx=0, freq_idx=0, pol_idx=1, overwrite=True)
        with fits.open(out_i) as hi, fits.open(out_v) as hv:
            assert float(hi[0].header["CRVAL1"]) == pytest.approx(float(hv[0].header["CRVAL1"]))
            assert float(hi[0].header["CRVAL2"]) == pytest.approx(float(hv[0].header["CRVAL2"]))
            assert not np.allclose(hi[0].data, hv[0].data)

    def test_roundtrip_header_provenance_keywords(self, tmp_path: Path) -> None:
        fits_path = tmp_path / "18MHz-Clean-Snapshot-20240817_120000-image-I.fits"
        _write_provenance_fits(
            fits_path,
            stokes=1,
            history="ovro-lwa-portal provenance round-trip",
        )
        result = _roundtrip_fits_zarr_fits_zarr([fits_path], tmp_path / "rt_prov")
        with fits.open(result.exported_fits) as hdul:
            hdr = hdul[0].header
            assert str(hdr["TELESCOP"]).strip() == "OVRO-LWA"
            assert "2024-08-17" in str(hdr["DATE-OBS"])
            assert "ovro-lwa-portal provenance round-trip" in str(hdr["HISTORY"])

    def test_roundtrip_after_regrid(self, tmp_path: Path) -> None:
        time_key = "20240817_120000"
        f_small = tmp_path / f"18MHz-Clean-Snapshot-{time_key}-image-I.fits"
        f_large = tmp_path / f"23MHz-Clean-Snapshot-{time_key}-image-I.fits"
        _write_provenance_fits(f_small, stokes=1, mhz=18, n=6, pixel_value=1.0)
        _write_provenance_fits(f_large, stokes=1, mhz=23, n=10, pixel_value=2.0)
        with fits.open(f_small) as hdul:
            input_crpix = float(hdul[0].header["CRPIX1"])
        result = _roundtrip_fits_zarr_fits_zarr(
            [f_small, f_large],
            tmp_path / "rt_regrid",
            freq_idx=0,
        )
        with fits.open(result.exported_fits) as hdul:
            export_crpix = float(hdul[0].header["CRPIX1"])
        stored_hdr = fits.Header.fromstring(
            _read_fits_header_str(result.ds_a, time_idx=0, freq_idx=0, pol_idx=0),
            sep="\n",
        )
        assert export_crpix == pytest.approx(float(stored_hdr["CRPIX1"]))
        assert export_crpix != pytest.approx(input_crpix)

    def test_export_then_xradio_read(self, tmp_path: Path) -> None:
        mod = _fits_mod()
        fits_path = tmp_path / "18MHz-Clean-Snapshot-20240817_120000-image-I.fits"
        _write_provenance_fits(fits_path, stokes=1)
        result = _roundtrip_fits_zarr_fits_zarr([fits_path], tmp_path / "rt_xradio")
        xds = mod._read_fits_via_xradio(
            result.exported_fits,
            do_sky_coords=False,
            compute_mask=False,
        )
        assert "SKY" in xds.data_vars or "DATA" in xds.data_vars

    def test_accessor_export_fits_writes_multiple(self, tmp_path: Path) -> None:
        import ovro_lwa_portal  # noqa: F401

        time_key = "20240817_120000"
        f18 = tmp_path / f"18MHz-Clean-Snapshot-{time_key}-image-I.fits"
        f23 = tmp_path / f"23MHz-Clean-Snapshot-{time_key}-image-I.fits"
        _write_provenance_fits(f18, stokes=1, mhz=18, pixel_value=1.0)
        _write_provenance_fits(f23, stokes=1, mhz=23, pixel_value=2.0)
        mod = _fits_mod()
        in_dir = tmp_path / "input_mf"
        in_dir.mkdir()
        shutil.copy2(f18, in_dir / f18.name)
        shutil.copy2(f23, in_dir / f23.name)
        zarr_a = mod.convert_fits_dir_to_zarr(
            input_dir=in_dir,
            out_dir=tmp_path / "zarr_mf",
            zarr_name="mf.zarr",
            fixed_dir=tmp_path / "fixed_mf",
            chunk_lm=4,
            rebuild=True,
            consolidate_metadata_at_end=False,
        )
        ds = open_dataset(zarr_a, chunks={})
        export_dir = tmp_path / "batch_export"
        files = ds.radport.export_fits(export_dir)
        assert len(files) == 2
        _roundtrip_fits_zarr_fits_zarr([Path(files[0])], tmp_path / "rt_from_export")
