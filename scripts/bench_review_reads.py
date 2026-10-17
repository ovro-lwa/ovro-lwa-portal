"""Time Source Review reads on one or more stores (warm cache, median of repeats).

Measures, per store:

* open: ``open_dataset(chunks={})`` + ``cap_lm_chunks(512)`` (current), and
  optionally the old ``chunks="auto"`` + ``.chunk(512)`` recipe;
* overlay: one 8° FOV cutout through the direct Zarr reader at each worker count;
* spectrum: a fixed-pixel and an RA/Dec-tracked dynamic spectrum through the
  direct reader at each worker count (track time reported separately).

Usage::

    pixi run python scripts/bench_review_reads.py \\
        /fast/claw/IV_1hMHz_Taper_shflux_coadd.zarr \\
        /fast/claw/IV_1hMHz_Taper_shflux_coadd_R2.zarr --old-open
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ovro_lwa_portal import open_dataset
from ovro_lwa_portal.accessor import _read_spectrum_direct
from ovro_lwa_portal.fov_overlay import celestial_wcs_for_time, extract_fov_cutout, plan_fov_cutout
from ovro_lwa_portal.io import cap_lm_chunks
from ovro_lwa_portal.zarr_direct import direct_zarr_array


def _median_ms(fn: Any, repeats: int) -> tuple[float, Any]:
    times = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), out


def bench_store(
    path: str, *, workers: list[int], repeats: int, old_open: bool, pad_frac: float
) -> dict[str, Any]:
    row: dict[str, Any] = {"store": path}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        ds = cap_lm_chunks(open_dataset(path, chunks={}), 512)
        row["open_s"] = round(time.perf_counter() - t0, 2)
        if old_open:
            t0 = time.perf_counter()
            open_dataset(path, chunks="auto").chunk({"l": 512, "m": 512})
            row["old_open_s"] = round(time.perf_counter() - t0, 2)

    direct = direct_zarr_array(ds)
    if direct is None:
        msg = f"{path}: direct Zarr reads unavailable"
        raise RuntimeError(msg)
    row["chunks"] = list(direct.chunks)
    row["chunk_mb"] = round(direct.chunk_nbytes / 1e6, 1)

    pol = 0
    n_time, n_freq = int(direct.shape[0]), int(direct.shape[1])
    t_idx, f_idx, wcs = next(
        (t, f, w)
        for t in range(n_time // 2, n_time)
        for f in range(n_freq // 2, n_freq)
        if (w := celestial_wcs_for_time(ds, t, freq_idx=f)) is not None
    )
    row["slice"] = [t_idx, f_idx]
    ra0, dec0 = (float(v) for v in wcs.wcs.crval)
    n_l, n_m = int(direct.shape[3]), int(direct.shape[4])
    spec = plan_fov_cutout(
        wcs, ra0 + 3.0, dec0 + 2.0, 8.0,
        n_l=n_l, n_m=n_m, chunk_l=direct.chunks[3], chunk_m=direct.chunks[4],
        pad_frac=pad_frac,
    )
    row["cutout_shape"] = list(spec.shape)
    row["cutout_chunks"] = spec.n_chunks

    track = ds.radport._compute_pixel_track(ra0, dec0, pol=pol)
    t0 = time.perf_counter()
    track = ds.radport._compute_pixel_track(ra0, dec0, pol=pol)
    row["track_ms"] = round((time.perf_counter() - t0) * 1000.0)
    n_t = int(direct.shape[0])
    fixed = (np.full(n_t, n_l // 2), np.full(n_t, n_m // 2), np.ones(n_t, dtype=bool))

    row["by_workers"] = {}
    for w in workers:
        with ThreadPoolExecutor(max_workers=w) as pool:
            def _cutout(pool: ThreadPoolExecutor = pool) -> np.ndarray:
                return extract_fov_cutout(
                    direct, t_idx, f_idx, pol, spec.l_slice, spec.m_slice, executor=pool
                )

            cut_ms, cut = _median_ms(_cutout, repeats)

        def _spectrum(l_idx: np.ndarray, m_idx: np.ndarray, vis: np.ndarray) -> np.ndarray:
            return _read_spectrum_direct(
                direct, ds.SKY, pol=pol, l_indices=l_idx, m_indices=m_idx, visible=vis,
                progress_callback=None, message="",
            )

        import ovro_lwa_portal.accessor as acc

        real = acc.read_workers
        acc.read_workers = lambda _nbytes, w=w: w
        try:
            fixed_ms, fixed_vals = _median_ms(lambda: _spectrum(*fixed), repeats)
            tracked_ms, tracked_vals = _median_ms(lambda: _spectrum(*track), repeats)
        finally:
            acc.read_workers = real
        row["by_workers"][w] = {
            "cutout_ms": round(cut_ms, 1),
            "fixed_spectrum_ms": round(fixed_ms, 1),
            "tracked_extract_ms": round(tracked_ms, 1),
        }
        row["checksums"] = {
            "cutout": float(np.nansum(cut)),
            "fixed": float(np.nansum(fixed_vals)),
            "tracked": float(np.nansum(tracked_vals)),
        }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stores", nargs="+")
    parser.add_argument("--workers", type=int, nargs="*", default=[4, 8, 16])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--old-open", action="store_true", help="also time chunks='auto' + 512")
    parser.add_argument(
        "--pad-frac", type=float, default=0.5, help="cutout padding (Source Review uses 0.5)"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    rows = [
        bench_store(
            s,
            workers=args.workers,
            repeats=args.repeats,
            old_open=args.old_open,
            pad_frac=args.pad_frac,
        )
        for s in args.stores
    ]
    text = json.dumps(rows, indent=2)
    print(text)
    if args.out is not None:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
