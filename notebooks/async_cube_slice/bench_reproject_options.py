"""Async sky-chunk load plus one-time reprojection options.

Same threading model as the Bokeh dynamic spectrum: worker threads read Zarr
chunks, the main thread assembles the tile, then one projection runs as the
single "paint". The three stable projections (naive SIN at the slice phase
center, CAR, HEALPix) are compared with the current per-slew view-centered SIN
warp.

CAR and HEALPix both sample with ``scipy.ndimage.map_coordinates`` (``order=1``),
the same resampler as astrowidget. HEALPix pixels come from ``healpy.query_disc``
and ``healpy.pix2ang`` (RING, nside matched to the plate scale).

Example::

    pixi run python notebooks/async_cube_slice/bench_reproject_options.py --quick
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

warnings.filterwarnings("ignore")
os.environ.setdefault("ASTROWIDGET_PROFILE_PUSH", "1")


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def _nside_for_cdelt_deg(cdelt_deg: float) -> int:
    """Smallest power-of-two nside whose pixel scale is <= ``cdelt_deg``."""
    # sqrt(pi / (3 nside^2)) radians is the HEALPix pixel scale.
    nside = math.sqrt(math.pi / 3.0) * (180.0 / math.pi) / max(float(cdelt_deg), 1e-6)
    order = max(0, math.ceil(math.log2(max(nside, 1.0))))
    return int(2**order)


def _check_healpix() -> str:
    """Confirm the installed healpy matches its published pixel example."""
    import healpy as hp

    pix = int(hp.ang2pix(16, np.pi / 2, 0.0))
    if pix != 1440:
        msg = f"healpy ang2pix self-check failed: got {pix}, expected 1440"
        raise RuntimeError(msg)
    return str(hp.__version__)


def _sample_map(data: np.ndarray, src_l: np.ndarray, src_m: np.ndarray) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    out = map_coordinates(
        data.astype(np.float64, copy=False),
        [src_l, src_m],
        order=1,
        mode="constant",
        cval=np.nan,
    )
    return out.astype(np.float32, copy=False)


def project_native_sin(data: np.ndarray, wcs, *, ra_deg: float, dec_deg: float):
    """Naive SIN at a chosen center. Slice phase center, or the current view."""
    from astrowidget.wcs import apply_reproject_maps, build_reproject_maps

    t0 = time.perf_counter()
    maps = build_reproject_maps(wcs, data.shape, crval_ra=ra_deg, crval_dec=dec_deg)
    build_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    out = apply_reproject_maps(data, maps)
    apply_ms = (time.perf_counter() - t1) * 1000.0
    return out, build_ms, apply_ms


def project_car(data: np.ndarray, wcs):
    """One CAR grid at the native pixel scale, sampled from the SIN cutout."""
    from astropy.wcs import WCS

    n_l, n_m = data.shape
    ra0 = float(wcs.wcs.crval[0])
    dec0 = float(wcs.wcs.crval[1])
    car = WCS(naxis=2)
    car.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    car.wcs.crval = [ra0, dec0]
    car.wcs.cdelt = [float(wcs.wcs.cdelt[0]), float(wcs.wcs.cdelt[1])]
    car.wcs.crpix = [float(wcs.wcs.crpix[0]), float(wcs.wcs.crpix[1])]
    car.wcs.cunit = ["deg", "deg"]

    t0 = time.perf_counter()
    ll, mm = np.meshgrid(np.arange(n_l, dtype=np.float64), np.arange(n_m, dtype=np.float64), indexing="ij")
    ra, dec = car.all_pix2world(ll, mm, 0)
    src_l, src_m = wcs.all_world2pix(ra, dec, 0)
    build_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    out = _sample_map(data, src_l, src_m)
    apply_ms = (time.perf_counter() - t1) * 1000.0
    return out, build_ms, apply_ms


def project_healpix(data: np.ndarray, wcs):
    """Partial RING HEALPix covering the cutout, at the native pixel scale.

    ``healpy.query_disc`` lists pixels inside the disc from the cutout center
    out to the farthest corner. Centers that fall outside the array are dropped
    before sampling.
    """
    import healpy as hp

    n_l, n_m = data.shape
    ra_mid, dec_mid = wcs.all_pix2world([(n_l - 1) / 2.0], [(n_m - 1) / 2.0], 0)
    theta0 = np.deg2rad(90.0 - float(np.asarray(dec_mid).reshape(-1)[0]))
    phi0 = np.deg2rad(float(np.asarray(ra_mid).reshape(-1)[0]))
    corners_l = np.array([0, 0, n_l - 1, n_l - 1], dtype=np.float64)
    corners_m = np.array([0, n_m - 1, 0, n_m - 1], dtype=np.float64)
    ra_k, dec_k = wcs.all_pix2world(corners_l, corners_m, 0)
    th = np.deg2rad(90.0 - np.asarray(dec_k, dtype=np.float64))
    ph = np.deg2rad(np.asarray(ra_k, dtype=np.float64))
    cosang = np.sin(theta0) * np.sin(th) * np.cos(phi0 - ph) + np.cos(theta0) * np.cos(th)
    radius = float(np.max(np.arccos(np.clip(cosang, -1.0, 1.0))))
    cdelt = min(abs(float(wcs.wcs.cdelt[0])), abs(float(wcs.wcs.cdelt[1])))
    nside = _nside_for_cdelt_deg(cdelt)
    vec = hp.ang2vec(theta0, phi0)

    t0 = time.perf_counter()
    ipix = hp.query_disc(nside, vec, radius)
    ra, dec = hp.pix2ang(nside, ipix, lonlat=True)
    src_l, src_m = wcs.all_world2pix(ra, dec, 0)
    src_l = np.asarray(src_l, dtype=np.float64)
    src_m = np.asarray(src_m, dtype=np.float64)
    inside = (
        np.isfinite(src_l)
        & np.isfinite(src_m)
        & (src_l >= -0.5)
        & (src_l <= n_l - 0.5)
        & (src_m >= -0.5)
        & (src_m <= n_m - 0.5)
    )
    src_l = src_l[inside]
    src_m = src_m[inside]
    build_ms = (time.perf_counter() - t0) * 1000.0
    t1 = time.perf_counter()
    out = _sample_map(data, src_l, src_m) if src_l.size else np.array([], dtype=np.float32)
    apply_ms = (time.perf_counter() - t1) * 1000.0
    return out, build_ms, apply_ms, int(nside), int(out.size)


def _finite_frac(data: np.ndarray) -> float:
    if data.size == 0:
        return 0.0
    return float(np.isfinite(data).mean())


def _time_method(fn, repeats: int) -> dict[str, float]:
    # One discarded call, then the timed repeats.
    warm = fn()
    builds, applies, totals, finite = [], [], [], []
    n_out = int(warm[0].size)
    nside = int(warm[3]) if len(warm) > 3 else None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        total = (time.perf_counter() - t0) * 1000.0
        out, build_ms, apply_ms = result[0], result[1], result[2]
        builds.append(build_ms)
        applies.append(apply_ms)
        totals.append(total)
        finite.append(_finite_frac(out))
        n_out = int(out.size)
    timed: dict[str, float] = {
        "build_ms": _median(builds),
        "apply_ms": _median(applies),
        "project_ms": _median(totals),
        "finite": _median(finite),
        "n_out": n_out,
    }
    if nside is not None:
        timed["nside"] = nside
    return timed


def _phase_center_chunk(wcs, ra_deg: float, dec_deg: float, n_l: int, n_m: int, chunk_l: int, chunk_m: int):
    """The single on-disk chunk that contains the slice phase center."""
    from fov_cutout import FovCutoutSpec

    pix_l, pix_m = wcs.world_to_pixel_values(ra_deg, dec_deg)
    li = int(np.clip(np.floor(float(np.asarray(pix_l).reshape(-1)[0])), 0, n_l - 1))
    mi = int(np.clip(np.floor(float(np.asarray(pix_m).reshape(-1)[0])), 0, n_m - 1))
    l0 = (li // chunk_l) * chunk_l
    m0 = (mi // chunk_m) * chunk_m
    l1 = min(n_l, l0 + chunk_l)
    m1 = min(n_m, m0 + chunk_m)
    return FovCutoutSpec(
        l_slice=slice(l0, l1),
        m_slice=slice(m0, m1),
        n_l=l1 - l0,
        n_m=m1 - m0,
        n_chunks_l=1,
        n_chunks_m=1,
    )


def _read_tile_timed(array, t: int, f: int, p: int, l_slice: slice, m_slice: slice, executor):
    """Async per-chunk read. Returns data, full read ms, and first-chunk ms."""
    from dynspec_async import _chunk_aligned_ranges

    l_ranges = _chunk_aligned_ranges(l_slice.start, l_slice.stop, array.chunks[3])
    m_ranges = _chunk_aligned_ranges(m_slice.start, m_slice.stop, array.chunks[4])
    t0 = time.perf_counter()

    def _one(lr: tuple[int, int], mr: tuple[int, int]) -> np.ndarray:
        return np.asarray(array[t, f, p, lr[0] : lr[1], mr[0] : mr[1]], dtype=np.float32)

    futures = {
        executor.submit(_one, lr, mr): (lr, mr)
        for lr in l_ranges
        for mr in m_ranges
    }
    out = np.empty((l_slice.stop - l_slice.start, m_slice.stop - m_slice.start), dtype=np.float32)
    first_ms = None
    for fut in as_completed(futures):
        lr, mr = futures[fut]
        block = fut.result()
        if first_ms is None:
            first_ms = (time.perf_counter() - t0) * 1000.0
        out[
            lr[0] - l_slice.start : lr[1] - l_slice.start,
            mr[0] - m_slice.start : mr[1] - m_slice.start,
        ] = block
    read_ms = (time.perf_counter() - t0) * 1000.0
    return out, read_ms, float(first_ms if first_ms is not None else read_ms)


def main() -> int:
    healpy_version = _check_healpix()

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--store", default="/fast/claw/IV_1hMHz_Taper_shflux_coadd.zarr")
    p.add_argument("--fov", type=float, default=8.0)
    p.add_argument("--time", type=int, default=0)
    p.add_argument("--freq", type=int, default=0)
    p.add_argument("--pol", type=int, default=0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--slew-deg",
        type=float,
        default=2.0,
        help="View-center RA offset for the per-slew SIN warp",
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default=str(_HERE / "results" / "reproject_options.jsonl"))
    args = p.parse_args()

    import zarr

    import ovro_lwa_portal as ovro
    from fov_cutout import adjust_wcs_for_cutout, celestial_wcs_for_time, plan_fov_cutout

    store = args.store
    if not Path(store).exists():
        print(f"store not found: {store}", file=sys.stderr)
        return 2
    repeats = 2 if args.quick else max(1, int(args.repeats))

    ds = ovro.open_dataset(store, chunks={}, validate=False)
    sky_arr = zarr.open_consolidated(store, mode="r")["SKY"]
    wcs_full = celestial_wcs_for_time(ds, int(args.time), freq_idx=int(args.freq))
    if wcs_full is None:
        print("no WCS", file=sys.stderr)
        return 2
    ra0 = float(wcs_full.wcs.crval[0])
    dec0 = float(wcs_full.wcs.crval[1])
    n_l = int(sky_arr.shape[3])
    n_m = int(sky_arr.shape[4])
    chunk_l = int(sky_arr.chunks[3])
    chunk_m = int(sky_arr.chunks[4])

    one = _phase_center_chunk(wcs_full, ra0, dec0, n_l, n_m, chunk_l, chunk_m)
    wide = plan_fov_cutout(
        wcs_full,
        ra0,
        dec0,
        float(args.fov),
        n_l=n_l,
        n_m=n_m,
        chunk_l=chunk_l,
        chunk_m=chunk_m,
    )
    cdelt = min(abs(float(wcs_full.wcs.cdelt[0])), abs(float(wcs_full.wcs.cdelt[1])))
    cases = [
        ("1-chunk", one, cdelt * max(one.n_l, one.n_m)),
        (f"{args.fov:g}deg", wide, float(args.fov)),
    ]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=int(args.workers))
    rows: list[dict] = []

    print(
        f"store={store} t={args.time} f={args.freq} phase=({ra0:.3f},{dec0:.3f}) "
        f"chunks=({chunk_l},{chunk_m}) workers={args.workers}"
    )
    print(
        f"{'case':<10} {'proj':<12} {'chunks':>6} {'shape':>12} {'first':>8} {'read':>8} "
        f"{'build':>8} {'apply':>8} {'project':>8} {'e2e':>8} {'finite':>7}"
    )

    for label, spec, fov_deg in cases:
        cut_wcs = adjust_wcs_for_cutout(wcs_full, spec.l_slice.start, spec.m_slice.start)
        # Warm the page cache, then time. Workers read chunks; the main thread projects once.
        _read_tile_timed(
            sky_arr, int(args.time), int(args.freq), int(args.pol), spec.l_slice, spec.m_slice, pool
        )
        reads: list[float] = []
        firsts: list[float] = []
        data = None
        for _ in range(repeats):
            data, read_ms, first_ms = _read_tile_timed(
                sky_arr,
                int(args.time),
                int(args.freq),
                int(args.pol),
                spec.l_slice,
                spec.m_slice,
                pool,
            )
            reads.append(read_ms)
            firsts.append(first_ms)
        assert data is not None
        read_med = _median(reads)
        first_med = _median(firsts)
        slew = float(args.slew_deg)
        methods = {
            "view-sin": lambda data=data, cut_wcs=cut_wcs, slew=slew: project_native_sin(
                data, cut_wcs, ra_deg=ra0 + slew, dec_deg=dec0
            ),
            "native-sin": lambda data=data, cut_wcs=cut_wcs: project_native_sin(
                data, cut_wcs, ra_deg=ra0, dec_deg=dec0
            ),
            "car": lambda data=data, cut_wcs=cut_wcs: project_car(data, cut_wcs),
            "healpix": lambda data=data, cut_wcs=cut_wcs: project_healpix(data, cut_wcs),
        }
        for name, fn in methods.items():
            timed = _time_method(fn, repeats)
            extra = f" nside={int(timed['nside'])}" if "nside" in timed else ""
            row = {
                "case": label,
                "fov_deg": float(fov_deg),
                "proj": name,
                "n_chunks": int(spec.n_chunks),
                "shape": list(spec.shape),
                "first_ms": first_med,
                "read_ms": read_med,
                "workers": int(args.workers),
                "slew_deg": slew,
                **{k: timed[k] for k in ("build_ms", "apply_ms", "project_ms", "finite", "n_out")},
                "e2e_ms": read_med + timed["project_ms"],
            }
            if "nside" in timed:
                row["nside"] = int(timed["nside"])
                row["healpy"] = healpy_version
                row["healpix_query"] = "query_disc"
            rows.append(row)
            print(
                f"{label:<10} {name:<12} {spec.n_chunks:6d} {tuple(spec.shape)!s:>12} "
                f"{first_med:8.1f} {read_med:8.1f} {timed['build_ms']:8.1f} {timed['apply_ms']:8.1f} "
                f"{timed['project_ms']:8.1f} {row['e2e_ms']:8.1f} {timed['finite']:7.3f}{extra}"
            )

    pool.shutdown(wait=False, cancel_futures=True)
    with out_path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
