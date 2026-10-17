"""Rewrite a finished Zarr store into the interactive review layout.

Incremental ingest appends one time step at a time, so image cubes land on
disk as ``(1, f, 1, lm, lm)`` chunks. A pixel spectrum on that layout opens
thousands of chunks. :func:`rechunk_store` rewrites every
``(time, frequency, polarization, l, m)`` array to
``(32, all frequencies, 1, 128, 128)`` by default: a spectrum then reads about a
dozen chunks and a sky cutout a few dozen, from one copy of the data.

Every other array is copied file-for-file. The output is built under
``<dest>.partial`` and renamed to ``dest`` only after a sample check against the
source passes.
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zarr

__all__ = [
    "DEFAULT_LM_CHUNK",
    "DEFAULT_TIME_CHUNK",
    "IMAGE_CUBE_DIMS",
    "RechunkConfig",
    "RechunkUnit",
    "image_cube_variables",
    "plan_rechunk_units",
    "rechunk_store",
    "target_chunks",
    "verify_rechunk",
]

DEFAULT_TIME_CHUNK = 32
DEFAULT_LM_CHUNK = 128
IMAGE_CUBE_DIMS = ("time", "frequency", "polarization", "l", "m")


@dataclass(frozen=True)
class RechunkConfig:
    """Target chunk shape and execution limits.

    ``freq_chunk=0`` keeps the whole frequency axis in one chunk.
    ``max_unit_bytes`` bounds the decoded block each worker holds in memory.
    """

    time_chunk: int = DEFAULT_TIME_CHUNK
    freq_chunk: int = 0
    lm_chunk: int = DEFAULT_LM_CHUNK
    workers: int = 8
    max_unit_bytes: int = 4 * 1024**3
    verify_samples: int = 64

    def __post_init__(self) -> None:
        for name in ("time_chunk", "lm_chunk", "workers", "max_unit_bytes"):
            if int(getattr(self, name)) <= 0:
                msg = f"{name} must be positive, got {getattr(self, name)}"
                raise ValueError(msg)
        if int(self.freq_chunk) < 0:
            msg = f"freq_chunk must be >= 0, got {self.freq_chunk}"
            raise ValueError(msg)


@dataclass(frozen=True)
class RechunkUnit:
    """Half-open block ``[t0, t1) x all freq x [p0, p1) x [l0, l1) x all m``."""

    t0: int
    t1: int
    p0: int
    p1: int
    l0: int
    l1: int


@dataclass
class RechunkReport:
    source: str
    dest: str
    variables: dict[str, dict[str, Any]] = field(default_factory=dict)
    units: int = 0
    seconds: float = 0.0
    bytes_on_disk: int = 0
    verified_samples: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "dest": self.dest,
            "variables": self.variables,
            "units": self.units,
            "seconds": round(self.seconds, 1),
            "bytes_on_disk": self.bytes_on_disk,
            "verified_samples": self.verified_samples,
        }


def image_cube_variables(group: zarr.Group) -> list[str]:
    """Names of arrays laid out as ``(time, frequency, polarization, l, m)``."""
    names = []
    for name, arr in group.arrays():
        dims = arr.attrs.get("_ARRAY_DIMENSIONS")
        if isinstance(dims, list) and tuple(dims) == IMAGE_CUBE_DIMS:
            names.append(name)
    return sorted(names)


def target_chunks(shape: tuple[int, ...], config: RechunkConfig) -> tuple[int, int, int, int, int]:
    """On-disk chunk shape for a 5-D image cube of ``shape``."""
    n_t, n_f, _n_p, n_l, n_m = (int(x) for x in shape)
    freq = int(config.freq_chunk) or n_f
    return (
        min(int(config.time_chunk), n_t),
        min(freq, n_f),
        1,
        min(int(config.lm_chunk), n_l),
        min(int(config.lm_chunk), n_m),
    )


def plan_rechunk_units(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    *,
    itemsize: int,
    max_unit_bytes: int,
) -> list[RechunkUnit]:
    """Split the cube into disjoint blocks that each cover whole target chunks.

    A unit spans one time chunk, one polarization, all frequencies, all ``m``,
    and as many ``l`` chunk rows as fit in ``max_unit_bytes`` (at least one).
    No two units write the same target chunk.
    """
    n_t, n_f, n_p, n_l, n_m = (int(x) for x in shape)
    ct, _cf, cp, cl, _cm = (int(x) for x in chunks)
    row_bytes = ct * n_f * cp * cl * n_m * int(itemsize)
    rows_per_unit = max(1, int(max_unit_bytes) // max(1, row_bytes))
    band = rows_per_unit * cl
    return [
        RechunkUnit(t0, min(t0 + ct, n_t), p0, min(p0 + cp, n_p), l0, min(l0 + band, n_l))
        for t0 in range(0, n_t, ct)
        for p0 in range(0, n_p, cp)
        for l0 in range(0, n_l, band)
    ]


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _copy_untouched(source: Path, dest: Path, skip: set[str]) -> None:
    for src in sorted(source.iterdir()):
        entry = src.name
        if entry in skip or entry == ".zmetadata":
            continue
        if src.is_dir():
            shutil.copytree(src, dest / entry)
        else:
            shutil.copy2(src, dest / entry)


def _create_like(
    group: zarr.Group, name: str, src: zarr.Array, chunks: tuple[int, ...]
) -> zarr.Array:
    kwargs: dict[str, Any] = {
        "shape": src.shape,
        "chunks": chunks,
        "dtype": src.dtype,
        "compressor": src.compressor,
        "fill_value": src.fill_value,
        "order": src.order,
        "filters": src.filters,
        "overwrite": True,
    }
    separator = getattr(src, "_dimension_separator", None)
    if separator:
        kwargs["dimension_separator"] = separator
    dst = group.create(name, **kwargs)
    dst.attrs.put(dict(src.attrs))
    return dst


def verify_rechunk(
    source: str | Path,
    dest: str | Path,
    variables: list[str],
    *,
    samples: int = 64,
    seed: int = 0,
) -> int:
    """Compare random pixel spectra between ``source`` and ``dest``.

    Each sample reads one ``(time block, all frequencies, pol, l, m)`` column
    from both stores. Raises ``ValueError`` on any mismatch; returns the
    number of columns compared.
    """
    src_group = zarr.open_group(str(source), mode="r")
    dst_group = zarr.open_group(str(dest), mode="r")
    rng = np.random.default_rng(seed)
    checked = 0
    for name in variables:
        src = src_group[name]
        dst = dst_group[name]
        if src.shape != dst.shape or src.dtype != dst.dtype:
            msg = f"{name}: shape/dtype {dst.shape}/{dst.dtype} != source {src.shape}/{src.dtype}"
            raise ValueError(msg)
        n_t, _n_f, n_p, n_l, n_m = src.shape
        for _ in range(int(samples)):
            t0 = int(rng.integers(0, n_t))
            t1 = min(n_t, t0 + int(dst.chunks[0]))
            p = int(rng.integers(0, n_p))
            li = int(rng.integers(0, n_l))
            mi = int(rng.integers(0, n_m))
            a = np.asarray(src[t0:t1, :, p, li, mi])
            b = np.asarray(dst[t0:t1, :, p, li, mi])
            if not np.array_equal(a, b, equal_nan=True):
                msg = f"{name}: mismatch at t[{t0}:{t1}] p={p} l={li} m={mi}"
                raise ValueError(msg)
            checked += 1
    return checked


def rechunk_store(
    source: str | Path,
    dest: str | Path,
    config: RechunkConfig | None = None,
    *,
    overwrite: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Write a copy of ``source`` with image cubes rechunked per ``config``.

    ``progress(done_units, total_units)`` is called from worker threads.
    An existing ``dest`` is replaced only when ``overwrite`` is true, and only
    after the new copy is complete and verified.
    """
    cfg = config or RechunkConfig()
    src_path = Path(source).expanduser().resolve()
    dst_path = Path(dest).expanduser().resolve()
    output_paths = (dst_path, dst_path.with_name(dst_path.name + ".partial").resolve())
    for output_path in output_paths:
        if (
            src_path == output_path
            or src_path in output_path.parents
            or output_path in src_path.parents
        ):
            msg = f"dest and staging must be outside source and not contain it: {output_path}"
            raise ValueError(msg)
    if dst_path.exists() and not overwrite:
        msg = f"dest exists (pass overwrite=True to replace): {dst_path}"
        raise FileExistsError(msg)

    src_group = zarr.open_group(str(src_path), mode="r")
    variables = image_cube_variables(src_group)
    if not variables:
        msg = f"No {IMAGE_CUBE_DIMS} arrays in {src_path}"
        raise ValueError(msg)

    partial = dst_path.with_name(dst_path.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    report = RechunkReport(source=str(src_path), dest=str(dst_path))
    t_start = time.perf_counter()
    _copy_untouched(src_path, partial, set(variables))
    dst_group = zarr.open_group(str(partial), mode="r+")

    jobs: list[tuple[zarr.Array, zarr.Array, RechunkUnit]] = []
    for name in variables:
        src = src_group[name]
        chunks = target_chunks(src.shape, cfg)
        dst = _create_like(dst_group, name, src, chunks)
        units = plan_rechunk_units(
            src.shape,
            chunks,
            itemsize=src.dtype.itemsize,
            max_unit_bytes=cfg.max_unit_bytes,
        )
        report.variables[name] = {
            "shape": list(src.shape),
            "source_chunks": list(src.chunks),
            "chunks": list(chunks),
            "units": len(units),
        }
        jobs.extend((src, dst, unit) for unit in units)
    report.units = len(jobs)

    def _copy(src: zarr.Array, dst: zarr.Array, u: RechunkUnit) -> None:
        block = src[u.t0 : u.t1, :, u.p0 : u.p1, u.l0 : u.l1, :]
        dst[u.t0 : u.t1, :, u.p0 : u.p1, u.l0 : u.l1, :] = block

    done = 0
    with ThreadPoolExecutor(max_workers=int(cfg.workers), thread_name_prefix="rechunk") as pool:
        futures = [pool.submit(_copy, src, dst, unit) for src, dst, unit in jobs]
        for fut in as_completed(futures):
            fut.result()
            done += 1
            if progress is not None:
                progress(done, len(jobs))

    zarr.consolidate_metadata(str(partial))
    report.verified_samples = verify_rechunk(
        src_path, partial, variables, samples=cfg.verify_samples
    )
    if dst_path.exists():
        shutil.rmtree(dst_path)
    partial.rename(dst_path)
    report.seconds = time.perf_counter() - t_start
    report.bytes_on_disk = _dir_size(dst_path)
    return report.as_dict()
