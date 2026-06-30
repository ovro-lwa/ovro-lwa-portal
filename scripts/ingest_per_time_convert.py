#!/usr/bin/env python3
"""Ingest compressed snapshot FITS one observation time at a time (low peak disk).

Thin wrapper around :func:`ovro_lwa_portal.ingest.per_time_convert.run_per_time_glob_convert`.
Prefer ``ovro-ingest convert STAGING_DIR OUTPUT_DIR --glob-pattern '...'`` for the same
workflow with CLI progress reporting.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from ovro_lwa_portal.ingest.discovery import IngestDiscoveryConfig, collect_glob_sources
from ovro_lwa_portal.ingest.per_time_convert import PerTimeGlobConvertConfig, run_per_time_glob_convert

logger = logging.getLogger(__name__)


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

    if not collect_glob_sources(args.glob_pattern):
        logger.error("Glob matched no files: %s", args.glob_pattern)
        return 1

    config = PerTimeGlobConvertConfig(
        glob_pattern=args.glob_pattern,
        staging_dir=args.staging_dir,
        work_root=args.work_root,
        output_dir=args.output_dir,
        fixed_dir=args.fixed_dir,
        zarr_name=args.zarr_name,
        chunk_lm=args.chunk_lm,
        rebuild=args.rebuild,
        resume=not args.no_resume,
        discovery=IngestDiscoveryConfig(
            group_metadata_source="filename",
            time_key_source="filename",
        ),
    )

    try:
        out = run_per_time_glob_convert(config)
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        logger.error("%s", exc)
        return 1

    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
