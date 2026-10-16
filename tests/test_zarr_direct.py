"""Direct Zarr chunk reads behind ``SKY``."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ovro_lwa_portal import open_dataset
from ovro_lwa_portal.dynspec_fill import TrackedPixelReader, plan_fill_units, spectrum_chunk_lengths
from ovro_lwa_portal.io import cap_lm_chunks, warn_if_suboptimal_lm_chunks
from ovro_lwa_portal.zarr_direct import direct_zarr_array, read_workers

pytestmark = pytest.mark.filterwarnings(
    "ignore:Zarr store .* has small on-disk l/m chunks:UserWarning"
)


def _cube(n_t: int = 6, n_f: int = 4, n_p: int = 2, n_l: int = 12, n_m: int = 10) -> xr.Dataset:
    rng = np.random.default_rng(3)
    sky = rng.normal(size=(n_t, n_f, n_p, n_l, n_m)).astype(np.float32)
    sky[1, 2] = np.nan
    return xr.Dataset(
        {"SKY": (("time", "frequency", "polarization", "l", "m"), sky)},
        coords={
            "time": np.arange(n_t, dtype=np.float64) + 60000.0,
            "frequency": np.linspace(40e6, 70e6, n_f),
            "polarization": np.array([1, 4]),
            "l": np.linspace(-0.4, 0.4, n_l),
            "m": np.linspace(-0.4, 0.4, n_m),
        },
    )


@pytest.fixture
def cube_store(tmp_path: Path) -> Path:
    path = tmp_path / "cube.zarr"
    _cube().to_zarr(
        path, mode="w", consolidated=True, encoding={"SKY": {"chunks": (2, 3, 1, 5, 4)}}
    )
    return path


def _open(path: Path) -> xr.Dataset:
    return open_dataset(path, chunks={}, validate=False)


def test_direct_array_resolves_from_open_dataset(cube_store: Path) -> None:
    ds = _open(cube_store)
    direct = direct_zarr_array(ds)
    assert direct is not None
    assert direct.chunks == (2, 3, 1, 5, 4)
    assert direct.shape == (6, 4, 2, 12, 10)
    plane = direct[3, 1, 1, slice(2, 9), slice(0, 7)]
    np.testing.assert_array_equal(
        plane, ds.SKY.isel(time=3, frequency=1, polarization=1).values[2:9, :7]
    )


def test_direct_array_maps_polarization_subset(cube_store: Path) -> None:
    ds = _open(cube_store).isel(polarization=[1])
    direct = direct_zarr_array(ds)
    assert direct is not None
    assert direct.shape[2] == 1
    np.testing.assert_array_equal(
        direct[0, 0, 0, slice(0, 12), slice(0, 10)],
        ds.SKY.isel(time=0, frequency=0, polarization=0).values,
    )


def test_direct_array_rejects_modified_or_unbacked_data(cube_store: Path) -> None:
    ds = _open(cube_store)
    assert direct_zarr_array(_cube()) is None
    assert direct_zarr_array(ds.isel(time=slice(1, None))) is None
    assert direct_zarr_array(ds.sortby("time", ascending=False)) is None
    replaced = ds.assign(SKY=ds.SKY * 2)
    assert direct_zarr_array(replaced) is None


def test_tracked_reader_direct_matches_xarray(cube_store: Path) -> None:
    ds = _open(cube_store)
    direct = direct_zarr_array(ds)
    assert direct is not None
    l_idx = np.array([0, 3, 3, 7, 11, 5])
    m_idx = np.array([9, 2, 2, 4, 0, 5])
    visible = np.array([True, True, False, True, True, True])
    kwargs = {"pol_idx": 1, "l_indices": l_idx, "m_indices": m_idx, "visible": visible}
    via_xr = TrackedPixelReader(ds.SKY, **kwargs)
    via_zarr = TrackedPixelReader(ds.SKY, direct=direct, **kwargs)
    t_chunk, f_chunk = spectrum_chunk_lengths(ds.SKY, direct)
    assert (t_chunk, f_chunk) == (2, 3)
    for unit in plan_fill_units(6, 4, t_chunk, f_chunk):
        np.testing.assert_array_equal(via_zarr(unit), via_xr(unit))


def test_dynamic_spectrum_fixed_pixel_direct_matches_dask(cube_store: Path) -> None:
    ds = _open(cube_store)
    in_memory = ds.copy(deep=True).load()
    in_memory.encoding.clear()
    fast = ds.radport.dynamic_spectrum(l=0.1, m=-0.2, pol=1)
    slow = in_memory.radport.dynamic_spectrum(l=0.1, m=-0.2, pol=1)
    xr.testing.assert_identical(fast, slow)


def test_dynamic_spectrum_tracked_direct_matches_dask(
    cube_store: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ds = _open(cube_store)
    in_memory = ds.copy(deep=True).load()
    in_memory.encoding.clear()
    track = (
        np.array([0, 3, 3, 7, 11, 5]),
        np.array([9, 2, 2, 4, 0, 5]),
        np.array([True, True, False, True, True, True]),
    )
    accessor_cls = type(ds.radport)
    monkeypatch.setattr(accessor_cls, "_resolve_coordinates", lambda _self, **_kw: track)
    fast = ds.radport.dynamic_spectrum(ra=10.0, dec=20.0, pol=0)
    slow = in_memory.radport.dynamic_spectrum(ra=10.0, dec=20.0, pol=0)
    xr.testing.assert_identical(fast, slow)


def test_cap_lm_chunks_only_splits_coarser_chunks(cube_store: Path) -> None:
    ds = _open(cube_store)
    assert cap_lm_chunks(ds, 8).SKY.data.chunksize[3:] == (5, 4)
    assert cap_lm_chunks(ds, 3).SKY.data.chunksize[3:] == (3, 3)
    assert cap_lm_chunks(ds, None) is ds


def test_time_blocked_small_lm_chunks_do_not_warn(tmp_path: Path) -> None:
    path = tmp_path / "blocked.zarr"
    sky = np.zeros((32, 4, 1, 1024, 1024), dtype=np.float32)
    ds = xr.Dataset({"SKY": (("time", "frequency", "polarization", "l", "m"), sky)})
    ds.to_zarr(path, mode="w", encoding={"SKY": {"chunks": (32, 4, 1, 128, 128)}}, compute=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_suboptimal_lm_chunks(path)


def test_read_workers_scale_with_chunk_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("os.cpu_count", lambda: 48)
    assert read_workers(1_200_000) == 4
    assert read_workers(10_000_000) == 8
    assert read_workers(31_500_000) == 16
    monkeypatch.setattr("os.cpu_count", lambda: 2)
    assert read_workers(31_000_000) == 2
