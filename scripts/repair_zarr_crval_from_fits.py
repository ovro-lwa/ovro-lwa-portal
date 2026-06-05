#!/usr/bin/env python3
"""Repair per-time Zarr WCS CRVAL from native pipeline FITS headers.

Stores ingested before ``_fix_headers`` stopped overwriting ``CRVAL1``/``CRVAL2``
from filename timestamps may have incorrect phase centers in ``wcs_header_str``.
This script re-reads the native celestial reference from the same ``*.fits.fs``
sources used by :file:`ingest_per_time_convert.py`, patches each time row in
place (preserving the LM pixel grid), and leaves filename tokens for grouping only.

Usage (I-Clean-Snapshot defaults)::

    pixi run bash scripts/repair-I-Clean-Snapshot-20250120-LST4-5-crval.sh

Or directly::

    pixi run python scripts/repair_zarr_crval_from_fits.py \\
        --glob-pattern '/lustre/.../*image.fits.fs' \\
        --zarr-path /fast/claw/.../store.zarr \\
        --work-root /lustre/claw/.../crval_repair_work \\
        --skip-backup

Dry-run first::

    pixi run python scripts/repair_zarr_crval_from_fits.py ... --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from ovro_lwa_portal.fits_to_zarr_xradio import (
    _discover_groups,
    repair_zarr_crval_from_fits,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import ingest_per_time_convert as _ingest_convert  # noqa: E402

logger = logging.getLogger(__name__)


def _collect_sources(glob_pattern: str) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(glob_pattern))]


def _discover_from_glob(glob_pattern: str) -> dict[str, list[Path]]:
    sources = _collect_sources(glob_pattern)
    if not sources:
        return {}
    return _ingest_convert.discover_sources_by_time(sources)


class _OnDemandFunpack:
    """Funpack one discovery group at a time; delete prior work dir after each use."""

    def __init__(self, work_root: Path) -> None:
        self._work_root = work_root
        self._active_dir: Path | None = None

    def __call__(self, discovery_key: str) -> list[Path]:
        self._cleanup_active()
        sources = self._sources_by_key.get(discovery_key, ())
        if not sources:
            return []
        work_dir = self._work_root / discovery_key
        if work_dir.exists():
            shutil.rmtree(work_dir)
        try:
            unpacked = _ingest_convert.funpack_time_group(list(sources), work_dir)
        except subprocess.CalledProcessError as exc:
            msg = f"funpack failed for time {discovery_key}: {exc}"
            raise RuntimeError(msg) from exc
        self._active_dir = work_dir
        return unpacked

    def bind_sources(self, sources_by_key: dict[str, list[Path]]) -> None:
        self._sources_by_key = sources_by_key

    def _cleanup_active(self) -> None:
        if self._active_dir is not None and self._active_dir.exists():
            shutil.rmtree(self._active_dir, ignore_errors=True)
        self._active_dir = None

    def close(self) -> None:
        self._cleanup_active()


def _read_zarr_time_array(zarr_path: Path) -> "np.ndarray":
    import numpy as np
    import zarr

    zg = zarr.open_group(str(zarr_path), mode="r")
    if "time" not in zg:
        msg = f"{zarr_path} has no time coordinate"
        raise KeyError(msg)
    return np.atleast_1d(np.asarray(zg["time"][:], dtype=np.float64))


def run_crval_repair(
    zarr_path: Path,
    *,
    glob_pattern: str | None,
    fits_dir: Path | None,
    work_root: Path,
    backup_suffix: str,
    skip_backup: bool,
    dry_run: bool,
    max_time_delta_sec: float,
) -> dict[str, object]:
    """Resolve FITS paths and patch ``wcs_header_str`` CRVAL in *zarr_path*."""
    zarr_path = zarr_path.expanduser().resolve()
    z_time = _read_zarr_time_array(zarr_path)
    logger.info("Zarr has %d time step(s)", int(z_time.size))

    funpack: _OnDemandFunpack | None = None
    try:
        if fits_dir is not None:
            fits_dir = fits_dir.expanduser().resolve()
            if not fits_dir.is_dir():
                msg = f"FITS directory not found: {fits_dir}"
                raise FileNotFoundError(msg)
            by_time = _discover_groups(fits_dir, group_metadata_source="filename")
            logger.info("Discovered %d time group(s) under %s", len(by_time), fits_dir)
            resolve_fits_paths = None
        elif glob_pattern is not None:
            work_root.mkdir(parents=True, exist_ok=True)
            by_time = _discover_from_glob(glob_pattern)
            if not by_time:
                msg = f"Glob matched no groupable sources: {glob_pattern}"
                raise FileNotFoundError(msg)
            funpack = _OnDemandFunpack(work_root)
            funpack.bind_sources(by_time)
            resolve_fits_paths = funpack
            logger.info(
                "Discovered %d filename time group(s); funpack-on-demand per Zarr step",
                len(by_time),
            )
        else:
            msg = "Provide --glob-pattern or --fits-dir"
            raise ValueError(msg)

        return repair_zarr_crval_from_fits(
            zarr_path,
            by_time,
            group_metadata_source="filename",
            backup_suffix=backup_suffix,
            skip_backup=skip_backup,
            dry_run=dry_run,
            max_time_delta_sec=max_time_delta_sec,
            resolve_fits_paths=resolve_fits_paths,
        )
    finally:
        if funpack is not None:
            funpack.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Patch Zarr wcs_header_str CRVAL from native pipeline FITS headers.",
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        required=True,
        help="Path to existing .zarr store",
    )
    parser.add_argument(
        "--glob-pattern",
        default=None,
        help="Python glob for pipeline *.fits.fs sources (same as ingest script)",
    )
    parser.add_argument(
        "--fits-dir",
        type=Path,
        default=None,
        help="Optional directory of uncompressed .fits (e.g. staged {time_key}__*.fits)",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("/tmp/ovro-crval-repair-work"),
        help="Per-time funpack scratch directory (removed after each time)",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".backup-before-crval-repair",
        help="Suffix for full-store backup before in-place writes",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        help="Patch wcs_header_str without copying the full Zarr store",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report CRVAL deltas without modifying the store",
    )
    parser.add_argument(
        "--max-time-delta-sec",
        type=float,
        default=6.0,
        help=(
            "Max UTC gap when pairing Zarr DATE-OBS times with filename -image- stamps "
            "(OVRO products often differ by ~5 s)"
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.glob_pattern is None and args.fits_dir is None:
        parser.error("one of --glob-pattern or --fits-dir is required")

    try:
        result = run_crval_repair(
            args.zarr_path,
            glob_pattern=args.glob_pattern,
            fits_dir=args.fits_dir,
            work_root=args.work_root,
            backup_suffix=args.backup_suffix,
            skip_backup=args.skip_backup,
            dry_run=args.dry_run,
            max_time_delta_sec=args.max_time_delta_sec,
        )
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, FileExistsError) as exc:
        logger.error("%s", exc)
        return 1

    print(json.dumps(result, indent=2, default=str))
    if result["patched_rows"] == 0:
        logger.warning("No wcs_header_str rows were changed")
    elif result["dry_run"]:
        logger.info("Dry run complete — re-run without --dry-run to apply")
    else:
        if result["backup"]:
            logger.info(
                "Patched %d time row(s); backup at %s",
                result["patched_rows"],
                result["backup"],
            )
        else:
            logger.info("Patched %d time row(s) (no full-store backup)", result["patched_rows"])
        logger.info(
            "Verify: pixi run python scripts/audit_zarr_wcs_timeline.py %s",
            result["store"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
