#!/usr/bin/env python3
"""Audit per-time celestial WCS stored in an OVRO-LWA ingest Zarr store.

Reports CRVAL1/CRVAL2 from each time step's ``wcs_header_str`` and how much the
phase center drifts in RA/Dec.  Per-time drift reflects the native FITS headers
written into each time step during ingest, not filename-derived zenith estimates.

Usage:
    pixi run python scripts/audit_zarr_wcs_timeline.py \\
        /fast/claw/I-Clean-Snapshot-20250120-LST4-/I-Clean-Snapshot-20250120-LST4-5.zarr

    # Compare against one original FITS (before/after fix_headers):
    pixi run python scripts/audit_zarr_wcs_timeline.py /path/to/store.zarr \\
        --sample-fits /lustre/claw/.../some-image.fits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy import units as u
from astropy.wcs import WCS

from ovro_lwa_portal.fits_to_zarr_xradio import (
    _decode_wcs_header_payload,
    _fix_headers,
    _obstime_from_fits_filename,
    _zenith_fk5_crvals_deg,
)


def _crval_deg_from_header_str(hdr_str: str) -> tuple[float, float] | None:
    if not hdr_str.strip():
        return None
    hdr = fits.Header.fromstring(hdr_str, sep="\n")
    wcs = WCS(hdr).celestial
    if wcs.naxis != 2:
        return None
    return float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1])


def _read_zarr_wcs_timeline(store: Path) -> list[tuple[float, float, str]]:
    """Return (mjd, ra_deg, dec_deg, time_label) per time index."""
    zg = zarr.open_group(str(store), mode="r")
    if "wcs_header_str" not in zg:
        msg = f"{store} has no wcs_header_str array"
        raise KeyError(msg)
    if "time" not in zg:
        msg = f"{store} has no time coordinate"
        raise KeyError(msg)

    wcs_arr = zg["wcs_header_str"]
    times = np.atleast_1d(np.asarray(zg["time"][:], dtype=np.float64))
    n_time = int(times.size)
    if wcs_arr.shape[0] != n_time:
        msg = (
            f"wcs_header_str length {wcs_arr.shape[0]} != time length {n_time}; "
            f"shape={wcs_arr.shape}"
        )
        raise ValueError(msg)

    rows: list[tuple[float, float, str]] = []
    for ti in range(n_time):
        raw = wcs_arr[ti]
        if wcs_arr.ndim >= 2:
            raw = raw[0]
        hdr_str = _decode_wcs_header_payload(raw)
        crval = _crval_deg_from_header_str(hdr_str)
        if crval is None:
            rows.append((float(times[ti]), float("nan"), float("nan"), f"t={ti} (empty)"))
            continue
        t_iso = Time(float(times[ti]), format="mjd", scale="utc").isot
        rows.append((float(times[ti]), crval[0], crval[1], t_iso))
    return rows


def _report_timeline(rows: list[tuple[float, float, str]]) -> int:
    print(f"Time steps: {len(rows)}")
    print(f"{'i':>4}  {'MJD':>14}  {'CRVAL1 (deg)':>14}  {'CRVAL2 (deg)':>14}  UTC")
    ra_vals: list[float] = []
    dec_vals: list[float] = []
    for i, (mjd, ra, dec, label) in enumerate(rows):
        print(f"{i:4d}  {mjd:14.8f}  {ra:14.6f}  {dec:14.6f}  {label}")
        if np.isfinite(ra) and np.isfinite(dec):
            ra_vals.append(ra)
            dec_vals.append(dec)

    if len(ra_vals) < 2:
        print("\nNot enough finite CRVAL rows to summarize drift.")
        return 0

    ra_arr = np.asarray(ra_vals, dtype=np.float64)
    dec_arr = np.asarray(dec_vals, dtype=np.float64)
    ra_span = float(np.max(ra_arr) - np.min(ra_arr))
    dec_span = float(np.max(dec_arr) - np.min(dec_arr))

    c0 = SkyCoord(ra=ra_arr[0] * u.deg, dec=dec_arr[0] * u.deg, frame="icrs")
    c1 = SkyCoord(ra=ra_arr[-1] * u.deg, dec=dec_arr[-1] * u.deg, frame="icrs")
    sep = c0.separation(c1).arcsec

    print("\nSummary (finite CRVAL rows only):")
    print(f"  RA span (max - min):  {ra_span:.4f} deg")
    print(f"  Dec span:             {dec_span:.4f} deg")
    print(f"  First–last separation: {sep:.1f} arcsec")

    if ra_span > 0.01 or dec_span > 0.01:
        print(
            "\nInterpretation: per-time CRVAL drift is expected when native FITS "
            "phase centers change across the observation. SkyWidget and radport use "
            "wcs_header_str[time], so RA/Dec axis labels change when stepping through "
            "time — verify against the source FITS headers, not filename timestamps."
        )
        return 1
    print("\nCRVAL is stable across time (within 0.01 deg).")
    return 0


def _compare_sample_fits(sample: Path) -> None:
    print(f"\nSample FITS: {sample}")
    with fits.open(sample, memmap=True) as hdul:
        raw_hdr = hdul[0].header.copy()
    raw_crval = _crval_deg_from_header_str(
        WCS(raw_hdr).celestial.to_header().tostring(sep="\n")
    )
    obs = _obstime_from_fits_filename(sample)
    zen = _zenith_fk5_crvals_deg(raw_hdr, obs) if obs is not None else None

    fixed = sample.parent / f"{sample.stem}_audit_fixed.fits"
    if fixed.exists():
        fixed.unlink()
    _fix_headers(sample, fixed)
    with fits.open(fixed, memmap=True) as hdul:
        fix_hdr = hdul[0].header
    fix_crval = (float(fix_hdr["CRVAL1"]), float(fix_hdr["CRVAL2"]))

    print(f"  Native FITS CRVAL:     {raw_crval}")
    print(f"  Zenith at filename:    {zen}  (diagnostic only; ingest does not use this)")
    print(f"  After _fix_headers:    {fix_crval}")
    if raw_crval:
        dra = abs(raw_crval[0] - fix_crval[0])
        ddec = abs(raw_crval[1] - fix_crval[1])
        print(f"  |native - fixed|:      RA {dra:.6f} deg, Dec {ddec:.6f} deg")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_store", type=Path, help="Path to .zarr store")
    parser.add_argument(
        "--sample-fits",
        type=Path,
        default=None,
        help="Optional pipeline FITS to compare native vs _fix_headers CRVAL",
    )
    args = parser.parse_args(argv)

    store = args.zarr_store.expanduser().resolve()
    if not store.exists():
        print(f"ERROR: store not found: {store}", file=sys.stderr)
        return 2

    print(f"Zarr store: {store}\n")
    rows = _read_zarr_wcs_timeline(store)
    code = _report_timeline(rows)

    if args.sample_fits is not None:
        _compare_sample_fits(args.sample_fits.expanduser().resolve())

    return code


if __name__ == "__main__":
    raise SystemExit(main())
