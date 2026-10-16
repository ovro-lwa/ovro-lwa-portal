"""Compare full-plane max_size overlay vs FOV-native cutout (Approach 1).

Warm-only timings on a local Zarr. Does not require a browser.

Example::

    pixi run python notebooks/async_cube_slice/bench_fov_cutout.py --quick
    pixi run python notebooks/async_cube_slice/bench_fov_cutout.py \\
        --store /fast/claw/IV_1hMHz_Taper_shflux_coadd.zarr --fov 8 --repeats 5
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

warnings.filterwarnings("ignore")
os.environ.setdefault("ASTROWIDGET_PROFILE_PUSH", "1")


def _median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def _fmt(ms: float) -> str:
    return f"{ms:8.1f}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--store",
        default="/fast/claw/IV_1hMHz_Taper_shflux_coadd.zarr",
        help="Zarr path",
    )
    p.add_argument("--fov", type=float, default=8.0, help="FOV degrees")
    p.add_argument("--time", type=int, default=0)
    p.add_argument("--freq", type=int, default=0)
    p.add_argument("--pol", type=int, default=0)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--max-sizes",
        default="256,512,1024",
        help="Comma-separated max_size values for the full-plane baseline",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--quick",
        action="store_true",
        help="One max_size (512) and 2 timed repeats after one discard",
    )
    args = p.parse_args()

    import ovro_lwa_portal as ovro
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astrowidget import SkyWidget
    from ovro_lwa_portal.accessor import _parse_sin_celestial_keywords, _read_wcs_header_str
    from ovro_lwa_portal.viz.pipeline_qa_app import bind_sky_widget_dataset
    import zarr

    from fov_cutout import plan_fov_cutout, push_fov_overlay, celestial_wcs_for_time

    store = args.store
    if not Path(store).exists():
        print(f"store not found: {store}", file=sys.stderr)
        return 2

    repeats = 2 if args.quick else max(1, int(args.repeats))
    max_sizes = [512] if args.quick else [int(x) for x in args.max_sizes.split(",") if x.strip()]

    ds = ovro.open_dataset(store, chunks={}, validate=False)
    sky_arr = zarr.open_consolidated(store, mode="r")["SKY"]
    hdr = _read_wcs_header_str(ds, time_idx=int(args.time), freq_idx=int(args.freq))
    if not hdr:
        print("no WCS at requested time", file=sys.stderr)
        return 2
    kw = _parse_sin_celestial_keywords(hdr)
    center = SkyCoord(float(kw["CRVAL1"]), float(kw["CRVAL2"]), unit="deg", frame="fk5")

    wcs = celestial_wcs_for_time(ds, int(args.time), freq_idx=int(args.freq))
    spec = plan_fov_cutout(
        wcs,
        float(center.icrs.ra.deg),
        float(center.icrs.dec.deg),
        float(args.fov),
        n_l=int(sky_arr.shape[3]),
        n_m=int(sky_arr.shape[4]),
        chunk_l=int(sky_arr.chunks[3]),
        chunk_m=int(sky_arr.chunks[4]),
    )
    print(
        f"store={store}\n"
        f"FOV={args.fov}° center=({center.ra.deg:.4f},{center.dec.deg:.4f}) "
        f"t={args.time} f={args.freq}\n"
        f"FOV cutout shape={spec.shape} chunks={spec.n_chunks} "
        f"l={spec.l_slice} m={spec.m_slice}"
    )
    print(
        f"{'path':<22} {'shape':>14} {'MB':>7} {'zarr_ms':>8} {'reproj_ms':>9} {'total_ms':>8}"
    )

    # --- baseline: full-plane update_slice at each max_size ---
    for ms in max_sizes:
        sky = SkyWidget()
        bind_sky_widget_dataset(sky, ds, max_size=ms)
        # discard
        sky.update_slice(int(args.time), int(args.freq), center=center, fov=args.fov * u.deg)
        totals, zarrs, reprojs = [], [], []
        shape = None
        nbytes = 0
        for _ in range(repeats):
            # clear cube cache so each repeat re-reads
            if sky._cube is not None:
                sky._cube._load_slice.cache_clear()
            t0 = time.perf_counter()
            sky.update_slice(
                int(args.time), int(args.freq), center=center, fov=args.fov * u.deg
            )
            totals.append((time.perf_counter() - t0) * 1000.0)
            prof = getattr(sky, "_profile_last_push", {}) or {}
            zarrs.append(float(prof.get("zarr_ms", float("nan"))))
            reprojs.append(float(prof.get("reproject_ms", float("nan"))))
            shape = tuple(sky.image_shape)
            nbytes = len(sky.image_data) if sky.image_data else 0
        print(
            f"{'full-plane max='+str(ms):<22} {str(shape):>14} {nbytes/1e6:7.2f} "
            f"{_fmt(_median(zarrs))} {_fmt(_median(reprojs))} {_fmt(_median(totals))}"
        )

    # --- FOV native cutout ---
    sky = SkyWidget()
    bind_sky_widget_dataset(sky, ds, max_size=256)  # unused by FOV path
    pool = ThreadPoolExecutor(max_workers=int(args.workers))
    push_fov_overlay(
        sky, ds, sky_arr, int(args.time), int(args.freq), center, float(args.fov),
        pol_idx=int(args.pol), executor=pool,
    )  # discard
    totals, zarrs, reprojs = [], [], []
    shape = None
    nbytes = 0
    n_chunks = 0
    for _ in range(repeats):
        prof = push_fov_overlay(
            sky,
            ds,
            sky_arr,
            int(args.time),
            int(args.freq),
            center,
            float(args.fov),
            pol_idx=int(args.pol),
            executor=pool,
        )
        totals.append(float(prof["total_ms"]))
        zarrs.append(float(prof["zarr_ms"]))
        reprojs.append(float(prof.get("reproject_ms", float("nan"))))
        shape = prof["shape"]
        nbytes = int(prof["nbytes"])
        n_chunks = int(prof["n_chunks"])
    print(
        f"{'FOV-native':<22} {str(shape):>14} {nbytes/1e6:7.2f} "
        f"{_fmt(_median(zarrs))} {_fmt(_median(reprojs))} {_fmt(_median(totals))}  "
        f"(chunks={n_chunks})"
    )
    pool.shutdown(wait=False, cancel_futures=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
