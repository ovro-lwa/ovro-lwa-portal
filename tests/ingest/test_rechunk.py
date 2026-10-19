"""Tests for ``ovro-ingest rechunk``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr
from typer.testing import CliRunner

from ovro_lwa_portal.ingest.cli import app
from ovro_lwa_portal.ingest.rechunk import (
    RechunkConfig,
    RechunkUnit,
    image_cube_variables,
    plan_rechunk_units,
    rechunk_store,
    target_chunks,
)

runner = CliRunner()


def _write_store(path: Path) -> xr.Dataset:
    rng = np.random.default_rng(1)
    n_t, n_f, n_p, n_l, n_m = 7, 3, 2, 10, 9
    sky = rng.normal(size=(n_t, n_f, n_p, n_l, n_m)).astype(np.float32)
    sky[2, 1] = np.nan
    headers = np.full((n_t, n_f, n_p), b"SIMPLE  =                    T", dtype="S80")
    ds = xr.Dataset(
        {
            "SKY": (("time", "frequency", "polarization", "l", "m"), sky),
            "fits_header_str": (("time", "frequency", "polarization"), headers),
        },
        coords={
            "time": np.arange(n_t, dtype=np.float64),
            "frequency": np.array([40e6, 50e6, 60e6]),
            "polarization": np.array([1, 4]),
            "l": np.linspace(-0.5, 0.5, n_l),
            "m": np.linspace(-0.5, 0.5, n_m),
        },
    )
    ds.attrs["note"] = "fixture"
    ds.SKY.attrs["units"] = "Jy/beam"
    encoding = {"SKY": {"chunks": (1, 2, 1, 4, 4)}}
    ds.to_zarr(path, mode="w", consolidated=True, encoding=encoding)
    return ds


def test_target_chunks_default_layout() -> None:
    assert target_chunks((415, 15, 2, 3122, 3122), RechunkConfig()) == (32, 15, 1, 128, 128)
    assert target_chunks((5, 4, 1, 64, 64), RechunkConfig()) == (5, 4, 1, 64, 64)
    assert target_chunks((40, 12, 2, 300, 300), RechunkConfig(freq_chunk=6)) == (32, 6, 1, 128, 128)


def test_plan_units_are_disjoint_and_cover_whole_chunks() -> None:
    shape = (7, 3, 2, 10, 9)
    chunks = (3, 3, 1, 4, 4)
    row_bytes = 3 * 3 * 1 * 4 * 9 * 4
    units = plan_rechunk_units(shape, chunks, itemsize=4, max_unit_bytes=row_bytes)
    assert RechunkUnit(0, 3, 0, 1, 0, 4) in units
    covered = np.zeros((7, 2, 10), dtype=int)
    for u in units:
        assert u.t0 % 3 == 0
        assert u.l0 % 4 == 0
        covered[u.t0 : u.t1, u.p0 : u.p1, u.l0 : u.l1] += 1
    assert np.all(covered == 1)


def test_rechunk_store_preserves_values_and_copies_metadata(tmp_path: Path) -> None:
    src = tmp_path / "src.zarr"
    dst = tmp_path / "dst.zarr"
    original = _write_store(src)
    cfg = RechunkConfig(time_chunk=3, lm_chunk=4, workers=3, max_unit_bytes=1, verify_samples=8)

    report = rechunk_store(src, dst, cfg)

    assert report["variables"]["SKY"]["chunks"] == [3, 3, 1, 4, 4]
    assert report["verified_samples"] == 8
    assert not (tmp_path / "dst.zarr.partial").exists()
    group = zarr.open_consolidated(str(dst), mode="r")
    assert group["SKY"].chunks == (3, 3, 1, 4, 4)
    assert group["SKY"].compressor == zarr.open_group(str(src))["SKY"].compressor
    assert image_cube_variables(group) == ["SKY"]

    out = xr.open_zarr(dst, consolidated=True)
    np.testing.assert_array_equal(out.SKY.values, original.SKY.values)
    np.testing.assert_array_equal(out.fits_header_str.values, original.fits_header_str.values)
    np.testing.assert_array_equal(out.polarization.values, original.polarization.values)
    assert out.attrs["note"] == "fixture"
    assert out.SKY.attrs["units"] == "Jy/beam"


def test_rechunk_store_refuses_existing_dest(tmp_path: Path) -> None:
    src = tmp_path / "src.zarr"
    dst = tmp_path / "dst.zarr"
    _write_store(src)
    dst.mkdir()
    with pytest.raises(FileExistsError):
        rechunk_store(src, dst, RechunkConfig(time_chunk=3, lm_chunk=4))
    with pytest.raises(ValueError, match="outside source"):
        rechunk_store(src, src / "nested.zarr", RechunkConfig())


def test_cli_rechunk(tmp_path: Path) -> None:
    src = tmp_path / "src.zarr"
    dst = tmp_path / "dst.zarr"
    _write_store(src)
    result = runner.invoke(
        app,
        ["rechunk", str(src), str(dst), "--chunk-time", "4", "--chunk-lm", "5", "--workers", "2"],
        env={"NO_COLOR": "1", "FORCE_COLOR": "0"},
    )
    assert result.exit_code == 0, result.stdout
    assert "Rechunk completed" in result.stdout
    assert zarr.open_group(str(dst), mode="r")["SKY"].chunks == (4, 3, 1, 5, 5)
