"""Standalone async (threaded) extract of SKY planes from an OVRO-LWA Zarr cube.

This module is intentionally separate from ``ovro_lwa_portal.viz`` (SourceReview,
SkyWidget, HoloViews explorers). It only depends on numpy/xarray and the
portal ``open_dataset`` entry point for consistent Zarr open behavior.

Typical flow
------------
1. ``ds = open_cube(path)``
2. ``arr = extract_plane(ds, time_idx, freq_idx, pol_idx)``  # sync
3. ``fut = extract_plane_async(ds, ...)`` then ``arr = fut.result()``
4. ``extract_planes_parallel(ds, keys)`` for many ``(t, f, p)`` planes
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class PlaneKey:
    """Indices into a SKY cube."""

    time_idx: int
    freq_idx: int
    pol_idx: int = 0


@dataclass(frozen=True)
class PlaneResult:
    """Materialized 2D plane plus the key that produced it."""

    key: PlaneKey
    data: np.ndarray  # shape (n_l, n_m), float32
    elapsed_s: float


def open_cube(
    source: str | Path,
    *,
    chunks: dict[str, int] | str | None = "auto",
    lm_chunk: int | None = 512,
    storage_options: dict | None = None,
) -> xr.Dataset:
    """Open a Zarr store via ``ovro_lwa_portal.open_dataset``.

    Parameters
    ----------
    source
        Local path, ``s3://`` / ``https://`` URL, or DOI string.
    chunks
        Passed to ``open_dataset`` (default ``\"auto\"``).
    lm_chunk
        If set, rechunk ``l`` and ``m`` after open (matches SourceReview read
        pattern). Pass ``None`` to leave chunks unchanged.
    storage_options
        Forwarded to ``open_dataset`` for remote stores.
    """
    import ovro_lwa_portal as ovro

    kwargs: dict = {"chunks": chunks}
    if storage_options is not None:
        kwargs["storage_options"] = storage_options
    ds = ovro.open_dataset(source, **kwargs)
    if lm_chunk is not None and "l" in ds.sizes and "m" in ds.sizes:
        ds = ds.chunk({"l": int(lm_chunk), "m": int(lm_chunk)})
    return ds


def _stride_for_max_size(n: int, max_size: int | None) -> int:
    if max_size is None or max_size <= 0 or n <= max_size:
        return 1
    return max(1, -(-n // max_size))


def extract_plane(
    ds: xr.Dataset,
    time_idx: int,
    freq_idx: int,
    pol_idx: int = 0,
    *,
    var: str = "SKY",
    max_size: int | None = 512,
) -> np.ndarray:
    """Select one ``(time, frequency, polarization)`` plane and return numpy.

    Returns
    -------
    np.ndarray
        Float32 array shaped ``(n_l, n_m)`` after optional striding. This is
        array-axis order (numpy row = ``l``), not a matplotlib transpose.
    """
    da = ds[var].isel(
        time=int(time_idx),
        frequency=int(freq_idx),
        polarization=int(pol_idx),
    )
    # Materialize (Dask/Zarr → numpy). Stride after read keeps display small.
    data = np.asarray(da.values, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"expected 2D plane, got shape {data.shape}")
    s_l = _stride_for_max_size(data.shape[0], max_size)
    s_m = _stride_for_max_size(data.shape[1], max_size)
    if s_l > 1 or s_m > 1:
        data = data[::s_l, ::s_m]
    return data


def extract_plane_timed(
    ds: xr.Dataset,
    key: PlaneKey,
    *,
    var: str = "SKY",
    max_size: int | None = 512,
) -> PlaneResult:
    """Like :func:`extract_plane` but returns timing metadata."""
    import time

    t0 = time.perf_counter()
    data = extract_plane(
        ds,
        key.time_idx,
        key.freq_idx,
        key.pol_idx,
        var=var,
        max_size=max_size,
    )
    return PlaneResult(key=key, data=data, elapsed_s=time.perf_counter() - t0)


def extract_plane_async(
    ds: xr.Dataset,
    time_idx: int,
    freq_idx: int,
    pol_idx: int = 0,
    *,
    executor: ThreadPoolExecutor,
    var: str = "SKY",
    max_size: int | None = 512,
) -> Future[PlaneResult]:
    """Submit a single-plane extract on ``executor``; return a Future."""
    key = PlaneKey(time_idx, freq_idx, pol_idx)
    return executor.submit(
        extract_plane_timed, ds, key, var=var, max_size=max_size
    )


def extract_planes_parallel(
    ds: xr.Dataset,
    keys: Sequence[PlaneKey] | Iterable[PlaneKey],
    *,
    max_workers: int = 4,
    var: str = "SKY",
    max_size: int | None = 512,
) -> list[PlaneResult]:
    """Extract many planes concurrently; results follow ``keys`` order."""
    key_list = list(keys)
    if not key_list:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(extract_plane_timed, ds, key, var=var, max_size=max_size)
            for key in key_list
        ]
        return [fut.result() for fut in futures]


def make_demo_dataset(
    *,
    n_time: int = 8,
    n_freq: int = 4,
    n_pol: int = 2,
    n_l: int = 256,
    n_m: int = 256,
) -> xr.Dataset:
    """In-memory SKY cube for offline demos (no disk Zarr required)."""
    rng = np.random.default_rng(0)
    sky = rng.standard_normal((n_time, n_freq, n_pol, n_l, n_m), dtype=np.float32)
    # Soft circular source drifting slowly in time for visual checks
    yy, xx = np.mgrid[0:n_l, 0:n_m]
    for ti in range(n_time):
        cy = n_l / 2 + 20 * np.sin(ti / 2.0)
        cx = n_m / 2 + 20 * np.cos(ti / 2.0)
        rr = (yy - cy) ** 2 + (xx - cx) ** 2
        sky[ti, :, :, :, :] += (2.5 * np.exp(-rr / (2 * 18.0**2))).astype(np.float32)

    return xr.Dataset(
        {"SKY": (["time", "frequency", "polarization", "l", "m"], sky)},
        coords={
            "time": np.arange(n_time),
            "frequency": np.linspace(50e6, 80e6, n_freq),
            "polarization": np.arange(n_pol),
            "l": np.linspace(-1.0, 1.0, n_l),
            "m": np.linspace(-1.0, 1.0, n_m),
        },
    )


def robust_limits(data: np.ndarray, low: float = 2.0, high: float = 98.0) -> tuple[float, float]:
    """Percentile display limits over finite pixels."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, low)), float(np.percentile(finite, high))
