"""Benchmark RA/Dec-tracked dynamic spectrum: accessor vs progressive async.

Stages timed separately
-----------------------
1. ``track`` — ``radport._compute_pixel_track(ra, dec)`` (per-time WCS → pixels)
2. ``extract`` — load ``(time, frequency)`` at those pixels
3. ``total`` — full ``radport.dynamic_spectrum(ra=, dec=)`` (track + extract)

Approaches for extract
----------------------
C-native   Existing accessor extract after a precomputed track (vectorized Dask).
A-tracked  Chunk-aligned zarr reads on a ``ThreadPoolExecutor`` via
           :class:`dynspec_async.ZarrTrackedPixelReader` + ``progressive_fill``.

Default catalog position is the phase center at ``time_idx=0`` (fixed celestial
source; pixels drift as CRVAL moves). Override with ``--ra`` / ``--dec``.

Usage::

    pixi run python notebooks/async_cube_slice/bench_radec_track.py \
        --store /fast/claw/IV_1hMHz_Taper_shflux_coadd.zarr \
        --workers 4 --out notebooks/async_cube_slice/results/radec_track.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dynspec_async import (  # noqa: E402
    ZarrTrackedPixelReader,
    plan_fill_units,
    progressive_fill,
)

warnings.filterwarnings("ignore")


def _crval_at(ds, time_idx: int = 0) -> tuple[float, float]:
    from ovro_lwa_portal.accessor import _parse_sin_celestial_keywords, _read_wcs_header_str

    for fi in range(int(ds.sizes["frequency"])):
        text = _read_wcs_header_str(ds, time_idx=time_idx, freq_idx=fi)
        if not text:
            continue
        kw = _parse_sin_celestial_keywords(text)
        if kw is not None and "CRVAL1" in kw and "CRVAL2" in kw:
            return float(kw["CRVAL1"]), float(kw["CRVAL2"])
    raise RuntimeError(f"No CRVAL at time_idx={time_idx}")


def _summarize(rows: list[dict], keys: list[str]) -> dict:
    out = {}
    for key in keys:
        vals = [r[key] for r in rows]
        out[key] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
    return out


def _fmt(stat: dict) -> str:
    return f"{stat['median'] * 1e3:.0f} [{stat['min'] * 1e3:.0f}–{stat['max'] * 1e3:.0f}]"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", default="/fast/claw/IV_1hMHz_Taper_shflux_coadd.zarr")
    p.add_argument("--ra", type=float, default=None)
    p.add_argument("--dec", type=float, default=None)
    p.add_argument("--pol", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--paint-interval", type=float, default=0.1)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import ovro_lwa_portal as ovro

    t_open = time.perf_counter()
    ds = ovro.open_dataset(args.store, chunks={}, validate=False)
    open_s = time.perf_counter() - t_open

    ra = args.ra
    dec = args.dec
    if ra is None or dec is None:
        ra, dec = _crval_at(ds, time_idx=0)

    # Warm track once for metadata
    l_idx, m_idx, visible = ds.radport._compute_pixel_track(ra, dec)
    n_vis = int(np.count_nonzero(visible))
    unique_lm = len({(int(l_idx[t]), int(m_idx[t])) for t in range(len(visible)) if visible[t]})
    changes = 0
    vis_t = np.flatnonzero(visible)
    if vis_t.size > 1:
        changes = int(
            np.count_nonzero(np.diff(l_idx[vis_t])) + np.count_nonzero(np.diff(m_idx[vis_t]))
        )

    print(
        f"catalog RA={ra:.4f} Dec={dec:.4f} | visible={n_vis}/{len(visible)} "
        f"unique_lm={unique_lm} index_changes={changes} | open={open_s:.2f}s",
        flush=True,
    )

    arr = zarr.open_consolidated(args.store, mode="r")["SKY"]
    n_time, n_freq = arr.shape[:2]
    tc, fc = arr.chunks[:2]
    units = plan_fill_units(n_time, n_freq, tc, fc)

    # --- full accessor path ---
    def one_c_total() -> dict:
        t0 = time.perf_counter()
        da = ds.radport.dynamic_spectrum(ra=ra, dec=dec, pol=args.pol)
        grid = np.asarray(da.values, dtype=np.float32)
        t1 = time.perf_counter()
        return {
            "T_total": t1 - t0,
            "checksum": float(np.nansum(grid)),
            "finite": float(np.isfinite(grid).mean()),
        }

    # --- track only ---
    def one_track() -> dict:
        t0 = time.perf_counter()
        li, mi, vis = ds.radport._compute_pixel_track(ra, dec)
        t1 = time.perf_counter()
        return {"T_track": t1 - t0, "n_vis": int(np.count_nonzero(vis))}

    # --- C extract only (reuse warm track arrays from outer scope each call) ---
    data_var = ds["SKY"].isel(polarization=args.pol)

    def one_c_extract() -> dict:
        t0 = time.perf_counter()
        vis_times, rows = ds.radport._extract_tracked_pixel_vectors(
            data_var, l_indices=l_idx, m_indices=m_idx, visible=visible
        )
        out = np.full((n_time, n_freq), np.nan, dtype=np.float32)
        for ti, row in zip(vis_times, rows, strict=True):
            out[int(ti)] = np.asarray(row, dtype=np.float32)
        t1 = time.perf_counter()
        return {
            "T_extract": t1 - t0,
            "checksum": float(np.nansum(out)),
            "finite": float(np.isfinite(out).mean()),
        }

    # --- A-tracked progressive extract ---
    def one_a_extract() -> dict:
        reader = ZarrTrackedPixelReader(
            arr,
            pol_idx=args.pol,
            l_indices=l_idx,
            m_indices=m_idx,
            visible=visible,
        )
        grid = np.full((n_time, n_freq), np.nan, dtype=np.float32)
        paints: list[int] = []

        def paint(_g: np.ndarray) -> None:
            paints.append(1)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            stats = progressive_fill(
                reader,
                units,
                grid,
                paint=paint,
                executor=pool,
                paint_interval_s=args.paint_interval,
            )
        return {
            "T_extract": stats.t_complete_data,
            "T_first": stats.t_first_data,
            "T_complete_paint": stats.t_complete_paint,
            "n_paints": len(paints),
            "checksum": float(np.nansum(grid)),
            "finite": float(np.isfinite(grid).mean()),
        }

    # Warm + timed
    for warm_fn in (one_track, one_c_extract, one_a_extract, one_c_total):
        warm_fn()

    track_rows = [one_track() for _ in range(args.repeats)]
    c_ext_rows = [one_c_extract() for _ in range(args.repeats)]
    a_ext_rows = [one_a_extract() for _ in range(args.repeats)]
    c_tot_rows = [one_c_total() for _ in range(args.repeats)]

    track_s = _summarize(track_rows, ["T_track"])
    c_ext_s = _summarize(c_ext_rows, ["T_extract"])
    a_ext_s = _summarize(a_ext_rows, ["T_extract", "T_first", "T_complete_paint"])
    c_tot_s = _summarize(c_tot_rows, ["T_total"])

    # Parity: A vs C extract
    match = np.isclose(c_ext_rows[-1]["checksum"], a_ext_rows[-1]["checksum"], rtol=1e-4, atol=1e-2)

    print(
        f"track              {_fmt(track_s['T_track'])} ms",
        flush=True,
    )
    print(
        f"C-native extract   {_fmt(c_ext_s['T_extract'])} ms  "
        f"finite={c_ext_rows[-1]['finite']:.3f} sum={c_ext_rows[-1]['checksum']:.4g}",
        flush=True,
    )
    print(
        f"A-tracked extract  {_fmt(a_ext_s['T_extract'])} ms  "
        f"first={_fmt(a_ext_s['T_first'])} paint={_fmt(a_ext_s['T_complete_paint'])}  "
        f"finite={a_ext_rows[-1]['finite']:.3f} sum={a_ext_rows[-1]['checksum']:.4g} "
        f"paints={a_ext_rows[-1]['n_paints']}",
        flush=True,
    )
    print(
        f"C-native total     {_fmt(c_tot_s['T_total'])} ms  "
        f"(track+extract inside dynamic_spectrum)",
        flush=True,
    )
    print(
        f"A-tracked total*   ~{_fmt({'median': track_s['T_track']['median'] + a_ext_s['T_extract']['median'], 'min': 0, 'max': 0})} ms  "
        f"(track median + A extract median; *composed)",
        flush=True,
    )
    print(f"checksum A vs C extract match: {bool(match)}", flush=True)

    record = {
        "store": args.store,
        "ra": ra,
        "dec": dec,
        "pol": args.pol,
        "workers": args.workers,
        "open_s": open_s,
        "n_visible": n_vis,
        "n_times": int(len(visible)),
        "unique_lm": unique_lm,
        "index_changes": changes,
        "n_units": len(units),
        "chunks": list(arr.chunks),
        "summary": {
            "track": track_s,
            "C_extract": c_ext_s,
            "A_tracked_extract": a_ext_s,
            "C_total": c_tot_s,
        },
        "checksum_match": bool(match),
        "rows": {
            "track": track_rows,
            "C_extract": c_ext_rows,
            "A_tracked_extract": a_ext_rows,
            "C_total": c_tot_rows,
        },
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as fh:
            fh.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
