#!/usr/bin/env python3
"""Profile SkyWidget overlay slice load (Zarr → display buffer).

Usage::

    pixi run python scripts/profile_overlay_slice.py
    pixi run python scripts/profile_overlay_slice.py /path/to/store.zarr

Measures the same stages as ``SourceReview._update_sky`` → ``SkyWidget.update_slice``.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ASTROWIDGET_SRC = _REPO_ROOT.parent / "astrowidget" / "src"
if _ASTROWIDGET_SRC.is_dir():
    sys.path.insert(0, str(_ASTROWIDGET_SRC))

import numpy as np
import xarray as xr
from astropy.coordinates import SkyCoord
from astropy.io.fits import Header
from astropy.wcs import WCS
from astropy import units as u

# Repo + editable astrowidget on path via pixi
from astrowidget import SkyWidget
from astrowidget.cube import PreloadedCube
from astrowidget.wcs import get_wcs, reproject_for_shader_display


def _make_wcs_header(ra: float = 350.85, dec: float = 58.815) -> str:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [-0.003, 0.003]
    w.wcs.crpix = [256.5, 256.5]
    w.wcs.cunit = ["deg", "deg"]
    return w.to_header().tostring(sep="\n")


def _build_dataset(
    *,
    n_time: int = 120,
    n_freq: int = 16,
    n_l: int = 512,
    n_m: int = 512,
    per_time_wcs: bool = True,
) -> xr.Dataset:
    rng = np.random.default_rng(0)
    times = np.arange(n_time, dtype=np.int64)
    freqs = np.linspace(50e6, 55e6, n_freq)
    sky = rng.standard_normal((n_time, n_freq, 1, n_l, n_m), dtype=np.float32)

    if per_time_wcs:
        headers = []
        for t in range(n_time):
            ra = 350.85 + 0.01 * np.sin(t / 10.0)
            dec = 58.815 + 0.005 * np.cos(t / 10.0)
            headers.append(_make_wcs_header(ra, dec))
        wcs_var = xr.DataArray(
            np.array(headers, dtype=object),
            dims=["time"],
            coords={"time": times},
        )
    else:
        wcs_var = xr.DataArray(np.array(_make_wcs_header()), dims=[])

    return xr.Dataset(
        {
            "SKY": (["time", "frequency", "polarization", "l", "m"], sky),
            "wcs_header_str": wcs_var,
        },
        coords={
            "time": times,
            "frequency": freqs,
            "polarization": [0],
            "l": np.arange(n_l),
            "m": np.arange(n_m),
        },
    )


def _write_zarr(
    ds: xr.Dataset,
    path: Path,
    *,
    lm_chunk: int,
    time_chunk: int = 1,
    freq_chunk: int = 1,
) -> None:
    chunked = ds.chunk(
        {
            "time": time_chunk,
            "frequency": freq_chunk,
            "polarization": 1,
            "l": lm_chunk,
            "m": lm_chunk,
        }
    )
    chunked.to_zarr(path, mode="w")


@contextmanager
def _timer(label: str, timings: dict[str, list[float]]) -> Iterator[None]:
    t0 = time.perf_counter()
    yield
    timings[label].append(time.perf_counter() - t0)


def _summarize(timings: dict[str, list[float]]) -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    for key, vals in timings.items():
        if not vals:
            continue
        rows.append((key, statistics.mean(vals), max(vals)))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def _profile_cube_stages(
    cube: PreloadedCube,
    ds: xr.Dataset,
    *,
    time_idx: int,
    freq_idx: int,
    center: SkyCoord | None,
    label: str,
    repeats: int,
) -> dict[str, list[float]]:
    timings: dict[str, list[float]] = {k: [] for k in (
        "get_wcs",
        "zarr_slice_load",
        "reproject_shader",
        "percentile_scale",
        "trait_bytes",
        "total_push_frame",
    )}

    from astrowidget.wcs import adjust_wcs_for_array_stride, wcs_projection_matches_naive_shader

    for _ in range(repeats):
        cube._load_slice.cache_clear()

        with _timer("get_wcs", timings):
            wcs = get_wcs(ds, time_idx=time_idx)
            display_wcs = adjust_wcs_for_array_stride(
                wcs, cube.stride_l, cube.stride_m
            )

        with _timer("zarr_slice_load", timings):
            data = cube.image(time_idx, freq_idx)

        reproject_center = center
        if reproject_center is not None:
            view_ra = float(reproject_center.icrs.ra.deg)
            view_dec = float(reproject_center.icrs.dec.deg)
        else:
            view_ra = float(display_wcs.wcs.crval[0])
            view_dec = float(display_wcs.wcs.crval[1])

        with _timer("reproject_shader", timings):
            if center is not None or not wcs_projection_matches_naive_shader(display_wcs):
                data, display_wcs = reproject_for_shader_display(
                    data,
                    display_wcs,
                    crval_ra=view_ra,
                    crval_dec=view_dec,
                )

        with _timer("percentile_scale", timings):
            finite = data[np.isfinite(data)]
            if finite.size:
                np.percentile(finite, [2, 98])

        with _timer("trait_bytes", timings):
            _ = data.astype(np.float32, copy=False).tobytes()

        with _timer("total_push_frame", timings):
            pass  # placeholder; sum printed separately

    print(f"\n=== {label} (n={repeats}, t={time_idx}, f={freq_idx}) ===")
    for name, mean_s, max_s in _summarize(timings):
        if name == "total_push_frame":
            continue
        print(f"  {name:22s}  mean={mean_s * 1000:8.1f} ms  max={max_s * 1000:8.1f} ms")
    total_mean = sum(statistics.mean(timings[k]) for k in timings if k != "total_push_frame")
    print(f"  {'sum(stages)':22s}  mean={total_mean * 1000:8.1f} ms")
    return timings


def _profile_widget_update(
    ds: xr.Dataset,
    *,
    max_size: int,
    time_idx: int,
    freq_idx: int,
    center: SkyCoord | None,
    preserve_view: bool,
    label: str,
    repeats: int,
) -> None:
    widget = SkyWidget()
    widget.set_dataset(ds, max_size=max_size, defer_display=True)
    widget.overlay_view_lock = True
    coord = center or SkyCoord(ra=350.85 * u.deg, dec=58.815 * u.deg)

    vals: list[float] = []
    for i in range(repeats):
        if i == 0:
            cube = widget._cube
            assert cube is not None
            cube._load_slice.cache_clear()
        t0 = time.perf_counter()
        if preserve_view:
            widget.update_slice(time_idx, freq_idx, view_lock=True)
        elif center is not None:
            widget.update_slice(
                time_idx,
                freq_idx,
                center=coord,
                fov=8.0 * u.deg,
            )
        else:
            widget.update_slice(time_idx, freq_idx)
        vals.append(time.perf_counter() - t0)

    print(f"\n=== {label} update_slice end-to-end ===")
    print(f"  cold mean={statistics.mean(vals) * 1000:.1f} ms  max={max(vals) * 1000:.1f} ms")
    # warm
    warm: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        if preserve_view:
            widget.update_slice(time_idx, freq_idx, view_lock=True)
        elif center is not None:
            widget.update_slice(time_idx, freq_idx, center=coord, fov=8.0 * u.deg)
        else:
            widget.update_slice(time_idx, freq_idx)
        warm.append(time.perf_counter() - t0)
    print(f"  warm mean={statistics.mean(warm) * 1000:.1f} ms  (LRU + no WCS rebuild)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "zarr_path",
        nargs="?",
        help="Optional on-disk Zarr (uses open_dataset-style load). Otherwise synthetic tmp Zarr.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--n-time", type=int, default=120)
    parser.add_argument("--n-freq", type=int, default=16)
    parser.add_argument("--n-lm", type=int, default=512)
    args = parser.parse_args(argv)

    n_l = args.n_lm
    max_size = max(256, n_l // 2)  # source_review_app._mount_sky_widget
    center = SkyCoord(ra=350.85 * u.deg, dec=58.815 * u.deg, frame="icrs")

    if args.zarr_path:
        from ovro_lwa_portal import open_dataset
        from ovro_lwa_portal.io import summarize_lm_chunks

        zarr_path = Path(args.zarr_path)
        try:
            chunk_summary = summarize_lm_chunks(zarr_path)
            print(
                "  on-disk SKY l/m chunks: "
                f"l_min={chunk_summary['l_min']} m_min={chunk_summary['m_min']} "
                f"(l={chunk_summary['l_chunks']}, m={chunk_summary['m_chunks']})"
            )
        except Exception as exc:
            print(f"  on-disk chunk summary unavailable: {exc}")
        print(f"Loading {zarr_path} (chunks auto + l/m=512)…")
        t0 = time.perf_counter()
        ds = open_dataset(zarr_path, chunks="auto").chunk({"l": 512, "m": 512})
        print(f"  open_dataset ready in {(time.perf_counter() - t0):.2f} s")
        print(f"  shape: {dict(ds.sizes)}  SKY chunks: {ds['SKY'].chunks}")
    else:
        import tempfile

        tmp = Path(tempfile.gettempdir()) / f"overlay_profile_{uuid.uuid4().hex}.zarr"
        print(f"Building synthetic Zarr at {tmp} …")
        ds_mem = _build_dataset(n_time=args.n_time, n_freq=args.n_freq, n_l=n_l, n_m=n_l)
        _write_zarr(ds_mem, tmp, lm_chunk=512)
        from ovro_lwa_portal import open_dataset

        ds = open_dataset(tmp, chunks="auto").chunk({"l": 512, "m": 512})
        print(f"  shape: {dict(ds.sizes)}  SKY chunks: {ds['SKY'].chunks}")

    cube = PreloadedCube(ds, max_size=max_size)
    print(
        f"\nDisplay grid: stride=({cube.stride_l}, {cube.stride_m}) "
        f"→ {len(cube.l_vals)}×{len(cube.m_vals)} pixels (max_size={max_size})"
    )

    t_mid = args.n_time // 2
    f_mid = args.n_freq // 2

    _profile_cube_stages(
        cube,
        ds,
        time_idx=t_mid,
        freq_idx=f_mid,
        center=center,
        label="cold slice + catalog center (generate overlay)",
        repeats=args.repeats,
    )
    _profile_cube_stages(
        cube,
        ds,
        time_idx=t_mid,
        freq_idx=f_mid,
        center=None,
        label="cold slice + view_lock center (heatmap tap)",
        repeats=args.repeats,
    )
    # warm cache
    cube.image(t_mid, f_mid)
    _profile_cube_stages(
        cube,
        ds,
        time_idx=t_mid,
        freq_idx=f_mid,
        center=center,
        label="warm LRU + catalog center",
        repeats=args.repeats,
    )

    _profile_widget_update(
        ds,
        max_size=max_size,
        time_idx=t_mid,
        freq_idx=f_mid,
        center=center,
        preserve_view=False,
        label="SkyWidget catalog center (after generate)",
        repeats=args.repeats,
    )
    _profile_widget_update(
        ds,
        max_size=max_size,
        time_idx=t_mid,
        freq_idx=f_mid,
        center=None,
        preserve_view=True,
        label="SkyWidget preserve_view (heatmap cell)",
        repeats=args.repeats,
    )

    # Chunk sensitivity on synthetic store only
    if not args.zarr_path:
        print("\n=== Zarr chunk layout sensitivity (cold slice load only) ===")
        for lm_chunk in (n_l, 512, 128):
            p = Path(tempfile.gettempdir()) / f"overlay_profile_c{lm_chunk}_{uuid.uuid4().hex}.zarr"
            _write_zarr(ds_mem, p, lm_chunk=lm_chunk)
            ds_c = open_dataset(p, chunks="auto").chunk({"l": 512, "m": 512})
            c = PreloadedCube(ds_c, max_size=max_size)
            c._load_slice.cache_clear()
            t0 = time.perf_counter()
            _ = c.image(t_mid, f_mid)
            dt = time.perf_counter() - t0
            print(f"  lm_chunk={lm_chunk:4d}  cold load={dt * 1000:7.1f} ms  SKY chunks={ds_c['SKY'].chunks}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
