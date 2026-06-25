#!/usr/bin/env python
"""Profile source-review heatmap methods on a small Zarr subset.

Full-cube ``patch_fit`` is no longer a heatmap method; use ``--fit-overlay-cell``
to time single-cell overlay fitting (``patch_fit_cell`` / Fit overlay button).
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import xarray as xr

from ovro_lwa_portal.io import open_dataset
from ovro_lwa_portal.viz.source_review_data import (
    HEATMAP_METHOD_OPTIONS,
    compute_overlay_patch_fit,
    compute_source_heatmap,
)

DEFAULT_ZARR = "/fast/claw/pipelineQA-phase2-I-NoTaper-Robust-0-20241218.zarr"
# Typical review target in the OVRO-LWA field.
DEFAULT_SRC = {"name": "test", "ra": 180.0, "dec": 37.0, "l": 0.0, "b": 0.0}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=Path(DEFAULT_ZARR),
        help="Zarr store to profile (default: QA phase-2 store on calim)",
    )
    parser.add_argument(
        "--n-time",
        type=int,
        default=8,
        help="Number of time steps in the profiling subset",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=3.0,
        help="Patch scale passed to heatmap / overlay fit",
    )
    parser.add_argument(
        "--patch-fit-max-chi2",
        type=float,
        default=3.0,
        help="Maximum reduced chi-squared for overlay patch fit",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Cold + warm repeats per heatmap method",
    )
    parser.add_argument(
        "--fit-overlay-cell",
        action="store_true",
        help="Time overlay patch fit for one (time, frequency) cell",
    )
    parser.add_argument(
        "--time-idx",
        type=int,
        default=0,
        help="Time index for --fit-overlay-cell",
    )
    parser.add_argument(
        "--freq-idx",
        type=int,
        default=0,
        help="Frequency index for --fit-overlay-cell",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run cProfile on representative heatmap methods",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Heatmap methods to time (default: all HEATMAP_METHOD_OPTIONS)",
    )
    return parser.parse_args(argv)


def subset_dataset(ds: xr.Dataset, n_time: int) -> xr.Dataset:
    """Keep the first ``n_time`` steps and modest LM chunks for laptop profiling."""
    ds = ds.isel(time=slice(0, n_time))
    return ds.chunk({"time": 1, "frequency": 1, "l": 512, "m": 512})


@contextmanager
def _sync_dask():
    import dask

    with dask.config.set(scheduler="synchronous"):
        yield


def _finite_frac(values: np.ndarray) -> float:
    return float(np.isfinite(values).sum()) / values.size if values.size else 0.0


def time_method(
    ds: xr.Dataset,
    method: str,
    src: dict,
    *,
    scale: float,
    patch_fit_max_chi2: float,
    repeats: int,
) -> tuple[float, np.ndarray]:
    times: list[float] = []
    values: np.ndarray | None = None
    for i in range(repeats):
        t0 = time.perf_counter()
        payload = compute_source_heatmap(
            ds,
            src,
            method=method,
            scale=scale,
            patch_fit_max_reduced_chi_squared=patch_fit_max_chi2,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        values = payload.values
        label = "cold" if i == 0 else "warm"
        print(f"  {method:18s} {label}: {elapsed:6.2f}s  finite={_finite_frac(values):.1%}")
    assert values is not None
    return min(times[1:]) if len(times) > 1 else times[0], values


def time_overlay_fit_cell(
    ds: xr.Dataset,
    src: dict,
    *,
    time_idx: int,
    freq_idx: int,
    scale: float,
    patch_fit_max_chi2: float,
    repeats: int = 2,
) -> float | None:
    """Time ``compute_overlay_patch_fit`` (Fit overlay button backend)."""
    times: list[float] = []
    for i in range(repeats):
        t0 = time.perf_counter()
        try:
            result = compute_overlay_patch_fit(
                ds,
                src,
                time_idx=time_idx,
                freq_idx=freq_idx,
                scale=scale,
                patch_fit_max_reduced_chi_squared=patch_fit_max_chi2,
            )
        except Exception as exc:
            print(
                f"  fit_overlay_cell   FAILED t={time_idx} f={freq_idx}: {exc}",
            )
            return None
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        label = "cold" if i == 0 else "warm"
        diag = result.cell_diagnostics(time_idx, freq_idx)
        peak = diag.get("peak", float("nan"))
        chi2 = diag.get("reduced_chi_squared", float("nan"))
        print(
            f"  fit_overlay_cell   {label}: {elapsed:6.2f}s  "
            f"t={time_idx} f={freq_idx} peak={peak:.3g} chi2_red={chi2:.3g}"
        )
    return min(times[1:]) if len(times) > 1 else times[0]


def profile_method(
    ds: xr.Dataset,
    method: str,
    src: dict,
    *,
    scale: float,
    patch_fit_max_chi2: float,
) -> None:
    print(f"\n--- cProfile: {method} ---")
    pr = cProfile.Profile()
    pr.enable()
    compute_source_heatmap(
        ds,
        src,
        method=method,
        scale=scale,
        patch_fit_max_reduced_chi_squared=patch_fit_max_chi2,
    )
    pr.disable()
    stream = io.StringIO()
    stats = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
    stats.print_stats(15)
    print(stream.getvalue())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    zarr_path = args.zarr_path
    if not zarr_path.exists():
        print(
            f"Zarr store not found: {zarr_path}\n"
            "Pass --zarr-path to a local QA store or run on a host with /fast/claw/…",
            file=sys.stderr,
        )
        return 1

    methods = args.methods if args.methods is not None else list(HEATMAP_METHOD_OPTIONS)
    unknown = [m for m in methods if m not in HEATMAP_METHOD_OPTIONS]
    if unknown:
        print(
            f"Unknown heatmap method(s): {unknown}. "
            f"Valid: {', '.join(HEATMAP_METHOD_OPTIONS)}",
            file=sys.stderr,
        )
        return 1
    if "patch_fit" in methods:
        print(
            "patch_fit is not a heatmap method; use --fit-overlay-cell for single-cell timing.",
            file=sys.stderr,
        )
        return 1

    print(f"Store: {zarr_path}")
    ds_full = open_dataset(zarr_path, chunks="auto")
    ds = subset_dataset(ds_full, args.n_time)
    print(
        f"Subset: time={ds.sizes['time']}, frequency={ds.sizes['frequency']}, "
        f"SKY shape=({ds.sizes['l']}, {ds.sizes['m']})"
    )

    with _sync_dask():
        if args.fit_overlay_cell:
            print("\n== Overlay patch fit (single cell) ==")
            warm_fit = time_overlay_fit_cell(
                ds,
                DEFAULT_SRC,
                time_idx=args.time_idx,
                freq_idx=args.freq_idx,
                scale=args.scale,
                patch_fit_max_chi2=args.patch_fit_max_chi2,
                repeats=args.repeats,
            )
            if warm_fit is not None:
                print(f"  fit_overlay_cell warm: {warm_fit:.2f}s")

        if methods:
            print("\n== Wall times (warm run) ==")
            results: list[tuple[str, float]] = []
            for method in methods:
                try:
                    elapsed, _ = time_method(
                        ds,
                        method,
                        DEFAULT_SRC,
                        scale=args.scale,
                        patch_fit_max_chi2=args.patch_fit_max_chi2,
                        repeats=args.repeats,
                    )
                    results.append((method, elapsed))
                except Exception as exc:
                    print(f"  {method:18s} FAILED: {exc}")

            if results:
                baseline = next(
                    (t for m, t in results if m == "dynamic_spectrum"),
                    results[0][1],
                )
                print("\n== Relative to dynamic_spectrum ==")
                for method, elapsed in sorted(results, key=lambda x: x[1]):
                    rel = elapsed / baseline if baseline > 0 else float("inf")
                    print(f"  {method:18s} {elapsed:6.2f}s  ({rel:5.1f}x)")

        if args.profile and methods:
            for method in ("dynamic_spectrum", "patch_max", "std"):
                if method in methods:
                    profile_method(
                        ds,
                        method,
                        DEFAULT_SRC,
                        scale=args.scale,
                        patch_fit_max_chi2=args.patch_fit_max_chi2,
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
