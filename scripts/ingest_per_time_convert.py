#!/usr/bin/env python3
"""Ingest compressed snapshot FITS one observation time at a time (low peak disk).

For each time group (from filename ``-image-YYYYMMDD_HHMMSS``):

1. Symlink each ``*.fits.fs`` as ``*.fits.fz`` (``.fs`` → ``.fz``), run ``funpack`` on the symlink
2. Stage ``{time_key}__*.fits``, run :class:`~ovro_lwa_portal.ingest.core.FITSToZarrConverter`
3. Remove work directory and staged files before the next time

Global LM reference and frequency axis are derived before the per-time loop:
header reads from the first few time groups only for the LM grid; frequency labels
from filenames across all sources (no up-front funpack of the full dataset).
"""

from __future__ import annotations

import argparse
import glob
import logging
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from ovro_lwa_portal.fits_to_zarr_xradio import (
    _DISCOVERY_FREQ_BIN_HZ,
    _clear_sky_coord_cache,
    _discovery_frequency_sort_tuple,
    _extract_group_metadata_filename_only,
    _global_frequency_coord_hz,
    _load_global_lm_reference_dataset,
    _consolidate_zarr_metadata,
    repair_zero_beam_from_nearby_time,
)
from ovro_lwa_portal.ingest.core import ConversionConfig, FITSToZarrConverter
from ovro_lwa_portal.ingest.dewarp_convert import remove_staged_files_for_time_key
from ovro_lwa_portal.ingest.discovery import IngestDiscoveryConfig, prepare_ingest_time_groups

logger = logging.getLogger(__name__)


def discover_sources_by_time(
    source_paths: Sequence[Path],
    *,
    freq_bin_hz: float = _DISCOVERY_FREQ_BIN_HZ,
) -> dict[str, list[Path]]:
    """Group pipeline ``*.fits.fs`` paths by time and binned frequency (filename only)."""
    if freq_bin_hz <= 0.0:
        msg = f"freq_bin_hz must be positive, got {freq_bin_hz}"
        raise ValueError(msg)

    by_time: dict[str, list[Path]] = {}
    by_time_freq: dict[str, dict[int, list[Path]]] = {}

    for f in sorted(source_paths):
        time_key, frequency_hz, notes = _extract_group_metadata_filename_only(f)
        if time_key is None:
            logger.warning(
                "Skipping %s: missing -image-YYYYMMDD_HHMMSS in basename.",
                f.name,
            )
            continue
        if frequency_hz is None:
            logger.warning(
                "Could not determine frequency for %s; duplicate detection disabled.",
                f.name,
            )
            by_time.setdefault(time_key, []).append(f)
            continue

        freq_key = int(round(frequency_hz / freq_bin_hz))
        time_freq_map = by_time_freq.setdefault(time_key, {})
        candidates = time_freq_map.setdefault(freq_key, [])
        candidates.append(f)

        if len(candidates) > 1:
            kept = candidates[0]
            logger.warning(
                "Multiple sources share time=%s and frequency bin %s: using %s only.",
                time_key,
                freq_key,
                kept.name,
            )
            time_freq_map[freq_key] = [kept]
            by_time.setdefault(time_key, [])
            by_time[time_key] = [p for p in by_time[time_key] if p not in candidates]
            by_time[time_key].append(kept)
            continue

        if notes:
            logger.warning("Fallback metadata for %s: %s", f.name, ", ".join(notes))
        by_time.setdefault(time_key, []).append(f)

    sort_key = lambda p: _discovery_frequency_sort_tuple(p, group_metadata_source="filename")
    for time_key, files in by_time.items():
        by_time[time_key] = sorted(files, key=sort_key)
    return by_time


def _freq_label_from_source(src: Path) -> str:
    """Return the ``*MHz`` directory name above ``snapshots_clean``."""
    return src.parent.parent.name


def _fz_symlink_name(src: Path, seen: set[str]) -> str:
    """Basename with suffix ``.fs`` replaced by ``.fz`` (e.g. ``*.fits.fs`` → ``*.fits.fz``)."""
    if not src.name.endswith(".fs"):
        msg = f"Expected a .fs suffix on {src}"
        raise ValueError(msg)
    link_name = f"{src.name.removesuffix('.fs')}.fz"
    if link_name in seen:
        run_tag = src.parents[2].name if len(src.parents) > 2 else src.parent.name
        link_name = f"{run_tag}__{link_name}"
    seen.add(link_name)
    return link_name


def funpack_time_group(sources: list[Path], work_dir: Path) -> list[Path]:
    """Symlink ``.fs`` → ``.fz``, funpack the symlink, return uncompressed ``.fits`` paths."""
    fz_dir = work_dir / "fz"
    fits_root = work_dir / "fits"
    fz_dir.mkdir(parents=True, exist_ok=True)

    seen_names: set[str] = set()
    unpacked: list[Path] = []

    for src in sources:
        link_name = _fz_symlink_name(src, seen_names)
        fz_path = fz_dir / link_name
        if fz_path.is_symlink() or fz_path.exists():
            fz_path.unlink()
        fz_path.symlink_to(src.resolve())

        # funpack reads the .fz symlink; writes uncompressed .fits (drop .fz only).
        out_fits = fits_root / _freq_label_from_source(src) / link_name.removesuffix(".fz")
        out_fits.parent.mkdir(parents=True, exist_ok=True)
        if not out_fits.exists():
            subprocess.run(
                ["funpack", "-O", str(out_fits), str(fz_path)],
                check=True,
            )
        unpacked.append(out_fits)

    return unpacked


def _stage_time_group(staging_dir: Path, time_key: str, files: list[Path]) -> int:
    staging_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in files:
        dest = staging_dir / f"{time_key}__{src.name}"
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(src.resolve())
        n += 1
    return n


def _cleanup_time_work(work_dir: Path, staging_dir: Path, time_key: str) -> None:
    shutil.rmtree(work_dir, ignore_errors=True)
    removed = remove_staged_files_for_time_key(staging_dir, time_key)
    logger.info("Removed %d staged FITS and work dir for time %s", removed, time_key)


def run_ingest_pipeline(
    source_paths: Sequence[Path],
    work_root: Path,
    staging_dir: Path,
    output_dir: Path,
    fixed_dir: Path,
    *,
    zarr_name: str,
    chunk_lm: int = 1024,
    rebuild: bool = False,
    resume: bool = True,
    discovery: IngestDiscoveryConfig | None = None,
) -> Path:
    """Funpack and convert one time group at a time; delete intermediates after each."""
    discovery = discovery or IngestDiscoveryConfig(
        group_metadata_source="filename",
        time_key_source="filename",
    )

    by_time_all = discover_sources_by_time(
        source_paths,
        freq_bin_hz=discovery.freq_bin_hz,
    )
    if not by_time_all:
        msg = "No groupable source files after discovery."
        raise FileNotFoundError(msg)

    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    out_zarr = output_dir / zarr_name

    logger.info(
        "Discovered %d source file(s) in %d time group(s).",
        sum(len(v) for v in by_time_all.values()),
        len(by_time_all),
    )

    lm_ref_ds = _load_global_lm_reference_dataset(
        by_time_all,
        fixed_dir,
        chunk_lm=chunk_lm,
        fix_headers_on_demand=True,
        group_metadata_source="filename",
    ).copy(deep=True)

    global_freq_hz = _global_frequency_coord_hz(
        by_time_all,
        group_metadata_source="filename",
    )

    by_time = prepare_ingest_time_groups(
        by_time_all,
        out_zarr=out_zarr,
        rebuild=rebuild,
        resume=resume,
        require_73mhz=False,
        context="convert",
        filter_invalid_beam=False,
    )
    if not by_time:
        logger.info("Every time key is already in %s.", out_zarr)
        _consolidate_zarr_metadata(out_zarr)
        return out_zarr

    first_zarr_write = not (out_zarr.exists() and not rebuild)
    time_keys_sorted = sorted(by_time.keys())
    total = len(time_keys_sorted)

    for idx, tkey in enumerate(time_keys_sorted):
        sources = list(by_time[tkey])
        work_dir = work_root / tkey
        _cleanup_time_work(work_dir, staging_dir, tkey)

        logger.info(
            "Time %s (%d/%d): funpack %d source(s) -> %s",
            tkey,
            idx + 1,
            total,
            len(sources),
            work_dir,
        )
        unpacked = funpack_time_group(sources, work_dir)
        repaired = repair_zero_beam_from_nearby_time(
            sources,
            unpacked,
            tkey,
            by_time_all,
            freq_bin_hz=discovery.freq_bin_hz,
        )
        if repaired:
            logger.info(
                "Repaired synthesized beam from nearby time on %d file(s) for time %s",
                repaired,
                tkey,
            )

        n_staged = _stage_time_group(staging_dir, tkey, unpacked)
        logger.info("Staged %d FITS for convert", n_staged)

        config = ConversionConfig(
            input_dir=staging_dir,
            output_dir=output_dir,
            zarr_name=zarr_name,
            fixed_dir=fixed_dir,
            chunk_lm=chunk_lm,
            rebuild=first_zarr_write,
            resume=False,
            fix_headers_on_demand=True,
            cleanup_fixed_fits=True,
            discovery_freq_bin_hz=discovery.freq_bin_hz,
            time_keys_only=(tkey,),
            lm_reference_ds=lm_ref_ds,
            group_metadata_source="filename",
            discovery_time_key_source="filename",
            consolidate_metadata_at_end=False,
            global_frequency_coord_hz=global_freq_hz if first_zarr_write else None,
        )
        FITSToZarrConverter(config).convert()
        first_zarr_write = False
        _clear_sky_coord_cache()

        _cleanup_time_work(work_dir, staging_dir, tkey)

    _consolidate_zarr_metadata(out_zarr)
    return out_zarr


def _collect_sources(glob_pattern: str) -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(glob_pattern))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest snapshot FITS: funpack and convert one time group at a time.",
    )
    parser.add_argument(
        "--glob-pattern",
        required=True,
        help="Python glob for pipeline *.fits.fs sources",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        required=True,
        help="Per-time funpack work directories (removed after each time)",
    )
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixed-dir", type=Path, required=True)
    parser.add_argument("--zarr-name", required=True)
    parser.add_argument("--chunk-lm", type=int, default=1024)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    sources = _collect_sources(args.glob_pattern)
    if not sources:
        logger.error("Glob matched no files: %s", args.glob_pattern)
        return 1

    try:
        out = run_ingest_pipeline(
            sources,
            args.work_root,
            args.staging_dir,
            args.output_dir,
            args.fixed_dir,
            zarr_name=args.zarr_name,
            chunk_lm=args.chunk_lm,
            rebuild=args.rebuild,
            resume=not args.no_resume,
        )
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as exc:
        logger.error("%s", exc)
        return 1

    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
