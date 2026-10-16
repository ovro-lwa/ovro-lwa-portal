"""Chunk-aligned reads from the Zarr array behind an image cube variable.

A pixel spectrum or a sky cutout touches a few kilobytes to megabytes, but
through Dask it builds one task per chunk over a graph of the whole cube.
:func:`direct_zarr_array` returns the on-disk ``zarr.Array`` for ``SKY`` (or
another ``(time, frequency, polarization, l, m)`` variable) so readers can index
it directly, one thread per chunk. It returns ``None`` whenever the variable no
longer matches the store, and callers then fall back to xarray indexing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
import zarr

__all__ = [
    "IMAGE_CUBE_DIMS",
    "ZARR_STORE_ENCODING_KEY",
    "DirectZarrArray",
    "attach_zarr_store",
    "direct_zarr_array",
    "read_workers",
]

ZARR_STORE_ENCODING_KEY = "ovro_lwa_portal_zarr_store"
_GROUP_ENCODING_KEY = "ovro_lwa_portal_zarr_group"
_ARRAY_GRAPH_LAYERS_ENCODING_KEY = "ovro_lwa_portal_zarr_array_graph_layers"
IMAGE_CUBE_DIMS = ("time", "frequency", "polarization", "l", "m")


def attach_zarr_store(ds: xr.Dataset, store: Any) -> xr.Dataset:
    """Record the store ``ds`` was opened from. Survives ``isel`` / ``chunk``."""
    ds.encoding[ZARR_STORE_ENCODING_KEY] = store
    ds.encoding[_ARRAY_GRAPH_LAYERS_ENCODING_KEY] = {
        name: frozenset(data.data.dask.layers)
        for name, data in ds.data_vars.items()
        if hasattr(getattr(data.data, "dask", None), "layers")
    }
    return ds


def read_workers(chunk_nbytes: int) -> int:
    """Thread count for chunk reads.

    Small chunks spend most of their time in zarr's Python path, so more than
    four threads contend for the GIL. Larger chunks spend it in Blosc, which
    releases the GIL.
    """
    nbytes = int(chunk_nbytes)
    if nbytes <= 4 * 1024**2:
        workers = 4
    elif nbytes <= 16 * 1024**2:
        workers = 8
    else:
        workers = 16
    return max(1, min(workers, os.cpu_count() or workers))


@dataclass(frozen=True)
class DirectZarrArray:
    """Dataset-indexed view of a 5-D on-disk array.

    ``shape`` and the polarization index are in the dataset's index space;
    ``pol_map`` translates polarization to the store's index. ``chunks`` is
    the on-disk chunk shape.
    """

    array: Any
    pol_map: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        n_t, n_f, _n_p, n_l, n_m = self.array.shape
        return (int(n_t), int(n_f), int(self.pol_map.size), int(n_l), int(n_m))

    @property
    def chunks(self) -> tuple[int, ...]:
        return tuple(int(c) for c in self.array.chunks)

    @property
    def chunk_nbytes(self) -> int:
        return int(np.prod(self.chunks)) * int(self.array.dtype.itemsize)

    def _pol(self, p: int) -> int:
        return int(self.pol_map[int(p)])

    def __getitem__(self, key: tuple[Any, ...]) -> np.ndarray:
        t, f, p, l_sel, m_sel = key
        return np.asarray(self.array[t, f, self._pol(p), l_sel, m_sel], dtype=np.float32)

    def column(self, t: slice, f: slice, p: int, l_idx: int, m_idx: int) -> np.ndarray:
        """``(n_t, n_f)`` spectrum block at one fixed pixel."""
        return np.asarray(
            self.array[t, f, self._pol(p), int(l_idx), int(m_idx)], dtype=np.float32
        ).reshape(t.stop - t.start, f.stop - f.start)

    def points(
        self,
        t_idx: np.ndarray,
        f: slice,
        p: int,
        l_idx: np.ndarray,
        m_idx: np.ndarray,
    ) -> np.ndarray:
        """``(len(t_idx), n_f)`` values at per-time pixels ``(l_idx[i], m_idx[i])``.

        One coordinate selection, so zarr decodes each touched chunk once even
        when several times share it.
        """
        t_col = np.asarray(t_idx, dtype=np.intp)[:, None]
        freqs = np.arange(f.start, f.stop, dtype=np.intp)[None, :]
        pol = np.full_like(t_col, self._pol(p))
        l_col = np.asarray(l_idx, dtype=np.intp)[:, None]
        m_col = np.asarray(m_idx, dtype=np.intp)[:, None]
        values = self.array.get_coordinate_selection((t_col, freqs, pol, l_col, m_col))
        return np.asarray(values, dtype=np.float32)


def _open_group(ds: xr.Dataset) -> Any | None:
    group = ds.encoding.get(_GROUP_ENCODING_KEY)
    if group is not None:
        return group
    store = ds.encoding.get(ZARR_STORE_ENCODING_KEY)
    if store is None:
        return None
    try:
        group = zarr.open_consolidated(store, mode="r")
    except (KeyError, ValueError, OSError):
        try:
            group = zarr.open_group(store, mode="r")
        except (KeyError, ValueError, OSError):
            return None
    ds.encoding[_GROUP_ENCODING_KEY] = group
    return group


def _coords_match(ds_values: np.ndarray, raw_values: np.ndarray) -> bool:
    if ds_values.shape != raw_values.shape:
        return False
    if np.issubdtype(ds_values.dtype, np.number) and np.issubdtype(raw_values.dtype, np.number):
        return bool(np.array_equal(ds_values, raw_values))
    return True


def _pol_map(ds: xr.Dataset, group: Any, n_raw: int) -> np.ndarray | None:
    n_ds = int(ds.sizes["polarization"])
    if "polarization" not in ds.coords or "polarization" not in group:
        return np.arange(n_ds) if n_ds == n_raw else None
    ds_pol = np.asarray(ds.coords["polarization"].values)
    raw_pol = np.asarray(group["polarization"][:])
    if raw_pol.dtype.kind in "SU" or ds_pol.dtype.kind in "SU":
        raw_pol = raw_pol.astype(str)
        ds_pol = ds_pol.astype(str)
    out = np.empty(n_ds, dtype=np.intp)
    for i, value in enumerate(ds_pol):
        hits = np.flatnonzero(raw_pol == value)
        if hits.size != 1:
            return None
        out[i] = int(hits[0])
    return out


def _retains_zarr_backing(ds: xr.Dataset, var: str) -> bool:
    """Check that ``ds[var]`` is still backed by its original Zarr Dask graph."""
    original_layers = ds.encoding.get(_ARRAY_GRAPH_LAYERS_ENCODING_KEY, {}).get(var)
    graph = getattr(ds[var].data, "dask", None)
    layers = getattr(graph, "layers", None)
    if not original_layers or layers is None or not original_layers.issubset(layers):
        return False
    return all(
        name in original_layers or name.startswith(("getitem-", "rechunk-"))
        for name in layers
    )


def direct_zarr_array(ds: xr.Dataset, var: str = "SKY") -> DirectZarrArray | None:
    """On-disk array for ``ds[var]``, or ``None`` when direct reads are unsafe.

    Requires the store recorded by :func:`attach_zarr_store`, a variable that
    still carries its on-disk chunk encoding (not recomputed), dims
    ``(time, frequency, polarization, l, m)``, and ``time`` / ``frequency`` /
    ``l`` / ``m`` that match the store. ``polarization`` may be a subset.
    """
    if var not in ds.data_vars:
        return None
    da = ds[var]
    if tuple(da.dims) != IMAGE_CUBE_DIMS or "preferred_chunks" not in da.encoding:
        return None
    if not _retains_zarr_backing(ds, var):
        return None
    group = _open_group(ds)
    if group is None or var not in group:
        return None
    raw = group[var]
    if len(raw.shape) != 5 or raw.dtype.kind != "f" or raw.dtype.itemsize != 4:
        return None
    for axis, dim in enumerate(IMAGE_CUBE_DIMS):
        if dim == "polarization":
            continue
        if int(raw.shape[axis]) != int(da.sizes[dim]):
            return None
        if (
            dim in ds.coords
            and dim in group
            and not _coords_match(np.asarray(ds.coords[dim].values), np.asarray(group[dim][:]))
        ):
            return None
    pol_map = _pol_map(ds, group, int(raw.shape[2]))
    if pol_map is None:
        return None
    return DirectZarrArray(array=raw, pol_map=pol_map)
