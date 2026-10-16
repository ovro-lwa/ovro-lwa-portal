"""Benchmark the linked sky-tile + dynamic-spectrum UI (experiment v1).

Approaches
----------
A        Progressive chunk-aligned fill, raw zarr reads on a ThreadPoolExecutor,
         matplotlib (Agg) painter.
A-blit   Same as A but matplotlib repaints only the changed axes (blit).
A-bokeh  Same reads as A, Bokeh painter (isolates painter library from I/O design).
A-xr     Same as A but reads through a lazy (non-dask) xarray DataArray.
AD       Same as A but tasks go to a ``dask.distributed`` LocalCluster (processes).
C        Production recipe: ``open_dataset(chunks="auto").chunk({"l": 512, "m": 512})``,
         ``radport.dynamic_spectrum(l=, m=)`` on one worker thread, one Bokeh
         mutation paint when complete (SourceReview Generate path, no SkyWidget).
C-native Same as C but ``open_dataset(chunks={})`` (dask chunks = zarr chunks).

Usage::

    pixi run python notebooks/async_cube_slice/bench_linked.py \
        --store /fast/claw/IV_1hMHz_Taper_shflux_coadd.zarr \
        --approach A --workers 8 --label full --out results.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dynspec_async import (  # noqa: E402
    BokehLinkedPainter,
    MatplotlibLinkedPainter,
    XarrayPixelReader,
    ZarrPixelReader,
    plan_fill_units,
    progressive_fill,
    read_sky_tile_parallel,
)

FULL_TILE = (1173, 1564)
FULL_PIXEL = 1368


def chunk_cost(store: str, sel: tuple) -> tuple[int, int]:
    """Count chunks and compressed bytes touched by a basic-slice selection."""
    arr = zarr.open_consolidated(store, mode="r")["SKY"]
    ranges = []
    for s, c, n in zip(sel, arr.chunks, arr.shape, strict=True):
        if isinstance(s, slice):
            start, stop = s.start or 0, s.stop if s.stop is not None else n
        else:
            start, stop = int(s), int(s) + 1
        ranges.append(range(start // c, (stop - 1) // c + 1))
    sep = getattr(arr, "_dimension_separator", ".") or "."
    n_chunks = 0
    n_bytes = 0
    import itertools

    for idx in itertools.product(*ranges):
        n_chunks += 1
        path = os.path.join(store, "SKY", sep.join(map(str, idx)))
        if os.path.exists(path):
            n_bytes += os.path.getsize(path)
    return n_chunks, n_bytes


def _thread_future(fn, *args) -> Future:
    """Run ``fn`` on a daemon thread like SourceReview's worker pattern."""
    fut: Future = Future()

    def _run() -> None:
        try:
            fut.set_result(fn(*args))
        except BaseException as exc:  # pragma: no cover - surfaced via result()
            fut.set_exception(exc)

    threading.Thread(target=_run, daemon=True).start()
    return fut


class Setup:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.sky_store = args.sky_store or args.store
        self.dyn_store = args.dyn_store or args.store
        self.sky_arr = zarr.open_consolidated(self.sky_store, mode="r")["SKY"]
        self.dyn_arr = zarr.open_consolidated(self.dyn_store, mode="r")["SKY"]
        n_l = self.sky_arr.shape[3]
        if n_l == FULL_TILE[1] - FULL_TILE[0]:
            self.tile = slice(0, n_l)
            self.pix = FULL_PIXEL - FULL_TILE[0]
        else:
            self.tile = slice(*FULL_TILE)
            self.pix = FULL_PIXEL
        self.n_time, self.n_freq = self.dyn_arr.shape[:2]
        self.pol = 0
        self.t0, self.f0 = 0, 0
        self.t_link, self.f_link = self.n_time // 2, self.n_freq // 2
        tile_len = self.tile.stop - self.tile.start
        self.tile_shape = (tile_len, tile_len)

    def cost(self) -> dict:
        sky_n, sky_b = chunk_cost(self.sky_store, (self.t0, self.f0, self.pol, self.tile, self.tile))
        dyn_n, dyn_b = chunk_cost(self.dyn_store, (slice(None), slice(None), self.pol, self.pix, self.pix))
        return {"sky_chunks": sky_n, "sky_bytes": sky_b, "dyn_chunks": dyn_n, "dyn_bytes": dyn_b}


# --------------------------------------------------------------------------
# Approach A family
# --------------------------------------------------------------------------


def run_progressive(setup: Setup, *, painter_kind: str, reader_kind: str, workers: int, client=None) -> dict:
    a = setup.args
    if painter_kind == "bokeh":
        painter = BokehLinkedPainter(setup.n_time, setup.n_freq, setup.tile_shape)
    else:
        painter = MatplotlibLinkedPainter(
            setup.n_time, setup.n_freq, setup.tile_shape, blit=painter_kind == "mpl-blit"
        )
    pool = ThreadPoolExecutor(max_workers=workers)

    def sky_read(t: int, f: int) -> np.ndarray:
        # Spatial chunks fan out on the pool; the benchmark's main thread just waits.
        return read_sky_tile_parallel(setup.sky_arr, t, f, setup.pol, setup.tile, setup.tile, pool)

    def timed_sky(t: int, f: int) -> tuple[float, float, float]:
        s0 = time.perf_counter()
        tile = sky_read(t, f)
        s1 = time.perf_counter()
        painter.paint_sky(tile)
        s2 = time.perf_counter()
        return s1 - s0, s2 - s0, s2 - s1

    sky_data, sky_paint, sky_stall = timed_sky(setup.t0, setup.f0)

    if reader_kind == "zarr":
        reader = ZarrPixelReader(setup.dyn_arr, pol_idx=setup.pol, l_idx=setup.pix, m_idx=setup.pix)
    else:
        import xarray as xr

        ds = xr.open_zarr(setup.dyn_store, chunks=None, consolidated=True)
        reader = XarrayPixelReader(ds["SKY"].isel(polarization=setup.pol, l=setup.pix, m=setup.pix))

    tc, fc = setup.dyn_arr.chunks[:2]
    units = plan_fill_units(setup.n_time, setup.n_freq, tc, fc)
    grid = np.full((setup.n_time, setup.n_freq), np.nan, dtype=np.float32)
    submit = None
    if client is not None:
        submit = lambda fn, unit: client.submit(fn, unit, pure=False)  # noqa: E731
    stats = progressive_fill(
        reader,
        units,
        grid,
        paint=painter.paint_dynspec,
        executor=pool,
        paint_interval_s=a.paint_interval,
        submit=submit,
    )

    link_data, link_paint, link_stall = timed_sky(setup.t_link, setup.f_link)
    pool.shutdown(wait=True)
    return {
        "T_sky_data": sky_data,
        "T_sky_paint": sky_paint,
        "T_dynspec_first_data": stats.t_first_data,
        "T_dynspec_first_paint": stats.t_first_paint,
        "T_dynspec_complete_data": stats.t_complete_data,
        "T_dynspec_complete_paint": stats.t_complete_paint,
        "T_link_data": link_data,
        "T_link_paint": link_paint,
        "n_units": stats.n_units,
        "n_paints": stats.n_paints,
        "max_stall": max(stats.max_paint, sky_stall, link_stall),
        "total_paint": stats.total_paint,
        "finite_frac": float(np.isfinite(grid).mean()),
        "checksum": float(np.nansum(grid)),
    }


# --------------------------------------------------------------------------
# Approach C family
# --------------------------------------------------------------------------


class CState:
    def __init__(self, setup: Setup, native: bool) -> None:
        import ovro_lwa_portal as ovro

        self.setup = setup
        chunks = {} if native else "auto"
        kw = {"validate": False} if setup.args.no_validate else {}
        self.sky_ds = ovro.open_dataset(setup.sky_store, chunks=chunks, **kw)
        self.dyn_ds = (
            self.sky_ds if setup.dyn_store == setup.sky_store else ovro.open_dataset(setup.dyn_store, chunks=chunks, **kw)
        )
        if not native:
            self.sky_ds = self.sky_ds.chunk({"l": 512, "m": 512})
            self.dyn_ds = self.dyn_ds.chunk({"l": 512, "m": 512})
        self.l_val = float(self.dyn_ds["l"].values[setup.pix])
        self.m_val = float(self.dyn_ds["m"].values[setup.pix])


def run_c(setup: Setup, state: CState) -> dict:
    painter = BokehLinkedPainter(setup.n_time, setup.n_freq, setup.tile_shape)

    def sky_read(t: int, f: int) -> np.ndarray:
        sel = state.sky_ds["SKY"].isel(time=t, frequency=f, polarization=setup.pol, l=setup.tile, m=setup.tile)
        return np.asarray(sel.values, dtype=np.float32)

    def timed_sky(t: int, f: int) -> tuple[float, float, float]:
        s0 = time.perf_counter()
        tile = _thread_future(sky_read, t, f).result()
        s1 = time.perf_counter()
        painter.paint_sky(tile)
        s2 = time.perf_counter()
        return s1 - s0, s2 - s0, s2 - s1

    sky_data, sky_paint, sky_stall = timed_sky(setup.t0, setup.f0)

    def dyn_read() -> np.ndarray:
        da = state.dyn_ds.radport.dynamic_spectrum(l=state.l_val, m=state.m_val, pol=setup.pol)
        assert int(da.attrs["l_idx"]) == setup.pix and int(da.attrs["m_idx"]) == setup.pix, da.attrs
        return np.asarray(da.values, dtype=np.float32)

    d0 = time.perf_counter()
    grid = _thread_future(dyn_read).result()
    d1 = time.perf_counter()
    painter.paint_dynspec(grid)
    d2 = time.perf_counter()

    link_data, link_paint, link_stall = timed_sky(setup.t_link, setup.f_link)
    return {
        "T_sky_data": sky_data,
        "T_sky_paint": sky_paint,
        "T_dynspec_first_data": d1 - d0,
        "T_dynspec_first_paint": d2 - d0,
        "T_dynspec_complete_data": d1 - d0,
        "T_dynspec_complete_paint": d2 - d0,
        "T_link_data": link_data,
        "T_link_paint": link_paint,
        "n_units": 1,
        "n_paints": 1,
        "max_stall": max(d2 - d1, sky_stall, link_stall),
        "total_paint": d2 - d1,
        "finite_frac": float(np.isfinite(grid).mean()),
        "checksum": float(np.nansum(grid)),
    }


# --------------------------------------------------------------------------


METRICS = [
    "T_sky_data",
    "T_sky_paint",
    "T_dynspec_first_paint",
    "T_dynspec_complete_data",
    "T_dynspec_complete_paint",
    "T_link_paint",
    "max_stall",
]


def summarize(rows: list[dict]) -> dict:
    out = {}
    for key in METRICS + ["T_dynspec_first_data", "T_link_data", "total_paint", "n_paints"]:
        vals = [r[key] for r in rows]
        out[key] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--store", default="/fast/claw/IV_1hMHz_Taper_shflux_coadd.zarr")
    p.add_argument("--sky-store", default=None)
    p.add_argument("--dyn-store", default=None)
    p.add_argument("--approach", required=True, choices=["A", "A-blit", "A-bokeh", "A-xr", "AD", "C", "C-native"])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--paint-interval", type=float, default=0.1)
    p.add_argument("--label", default="")
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    setup = Setup(args)
    client = None
    c_state = None
    if args.approach == "AD":
        from dask.distributed import Client, LocalCluster

        cluster = LocalCluster(n_workers=args.workers, threads_per_worker=1, processes=True, dashboard_address=None)
        client = Client(cluster)
    if args.approach in ("C", "C-native"):
        o0 = time.perf_counter()
        c_state = CState(setup, native=args.approach == "C-native")
        open_s = time.perf_counter() - o0
    else:
        open_s = 0.0

    def one() -> dict:
        if args.approach == "A":
            return run_progressive(setup, painter_kind="mpl", reader_kind="zarr", workers=args.workers)
        if args.approach == "A-blit":
            return run_progressive(setup, painter_kind="mpl-blit", reader_kind="zarr", workers=args.workers)
        if args.approach == "A-bokeh":
            return run_progressive(setup, painter_kind="bokeh", reader_kind="zarr", workers=args.workers)
        if args.approach == "A-xr":
            return run_progressive(setup, painter_kind="mpl", reader_kind="xr", workers=args.workers)
        if args.approach == "AD":
            return run_progressive(setup, painter_kind="mpl", reader_kind="zarr", workers=args.workers, client=client)
        return run_c(setup, c_state)

    warm = one()  # warm-up, discarded
    rows = [one() for _ in range(args.repeats)]
    if client is not None:
        client.close()

    record = {
        "label": args.label,
        "approach": args.approach,
        "workers": args.workers if args.approach.startswith("A") else "dask-default",
        "sky_store": setup.sky_store,
        "dyn_store": setup.dyn_store,
        "dyn_chunks": list(setup.dyn_arr.chunks),
        "sky_chunks_shape": list(setup.sky_arr.chunks),
        "paint_interval": args.paint_interval,
        "open_s": open_s,
        **setup.cost(),
        "warmup": warm,
        "summary": summarize(rows),
        "rows": rows,
    }
    s = record["summary"]
    line = (
        f"{args.label:>10} {args.approach:>8} w={record['workers']!s:>12} | "
        + " ".join(f"{k}={s[k]['median']:.3f}" for k in METRICS)
        + f" | paints={s['n_paints']['median']:.0f} finite={rows[-1]['finite_frac']:.3f} sum={rows[-1]['checksum']:.4g}"
    )
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
