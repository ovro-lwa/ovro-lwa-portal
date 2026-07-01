"""Per-time FITS→Zarr conversion from a glob pattern with prefixed staging."""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ovro_lwa_portal.fits_to_zarr_xradio import (
    _clear_sky_coord_cache,
    _consolidate_zarr_metadata,
    _global_frequency_coord_hz,
    _load_global_lm_reference_dataset,
    repair_zero_beam_from_nearby_time,
)
from ovro_lwa_portal.ingest.core import ConversionConfig, FITSToZarrConverter
from ovro_lwa_portal.ingest.dewarp_convert import remove_staged_files_for_time_key
from ovro_lwa_portal.ingest.discovery import (
    GlobConvertDiscoveryPlan,
    IngestDiscoveryConfig,
    collect_glob_sources,
    discover_time_grouped_paths,
    prepare_ingest_time_groups,
)
from ovro_lwa_portal.ingest.progress import report_ingest_progress

__all__ = [
    "PerTimeGlobConvertConfig",
    "funpack_time_group",
    "run_per_time_glob_convert",
    "sources_need_funpack",
    "stage_time_group_symlinks",
]

logger = logging.getLogger(__name__)


def sources_need_funpack(sources: Sequence[Path]) -> bool:
    """Return True when any source path uses the pipeline ``.fs`` compressed suffix."""
    return any(p.name.endswith(".fs") for p in sources)


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

        out_fits = fits_root / _freq_label_from_source(src) / link_name.removesuffix(".fz")
        out_fits.parent.mkdir(parents=True, exist_ok=True)
        if not out_fits.exists():
            subprocess.run(
                ["funpack", "-O", str(out_fits), str(fz_path)],
                check=True,
            )
        unpacked.append(out_fits)

    return unpacked


def _staged_symlink_name(time_key: str, src: Path, seen: set[str]) -> str:
    """Build a flat staging basename; disambiguate duplicate basenames with a parent tag."""
    link_name = f"{time_key}__{src.name}"
    if link_name in seen:
        run_tag = src.parents[2].name if len(src.parents) > 2 else src.parent.name
        link_name = f"{time_key}__{run_tag}__{src.name}"
    seen.add(link_name)
    return link_name


def stage_time_group_symlinks(staging_dir: Path, time_key: str, files: Sequence[Path]) -> int:
    """Symlink FITS into *staging_dir* as ``{time_key}__{basename}`` to avoid collisions."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    seen_names: set[str] = set()
    n = 0
    for src in files:
        dest = staging_dir / _staged_symlink_name(time_key, src, seen_names)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        dest.symlink_to(src.resolve())
        n += 1
    return n


def _cleanup_time_work(work_dir: Path, staging_dir: Path, time_key: str) -> None:
    shutil.rmtree(work_dir, ignore_errors=True)
    removed = remove_staged_files_for_time_key(staging_dir, time_key)
    logger.info("Removed %d staged FITS and work dir for time %s", removed, time_key)


@dataclass(frozen=True)
class PerTimeGlobConvertConfig:
    """Configuration for glob-driven per-time FITS→Zarr conversion."""

    glob_pattern: str
    staging_dir: Path
    output_dir: Path
    fixed_dir: Path
    zarr_name: str = "ovro_lwa_full_lm_only.zarr"
    work_root: Path | None = None
    chunk_lm: int = 1024
    rebuild: bool = False
    resume: bool = True
    fix_headers_on_demand: bool = True
    cleanup_fixed_fits: bool = True
    funpack: bool | None = None
    repair_zero_beam: bool | None = None
    discovery: IngestDiscoveryConfig | None = None
    lm_reference_target_size: int | None = None
    duplicate_resolver: Callable[[str, float, list[Path]], Path] | None = None
    verbose: bool = False
    prepared_discovery: GlobConvertDiscoveryPlan | None = None


def run_per_time_glob_convert(
    config: PerTimeGlobConvertConfig,
    *,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> Path:
    """Discover sources via glob, convert one observation time at a time, then clean staging."""
    discovery = config.discovery or IngestDiscoveryConfig(
        group_metadata_source="filename",
        time_key_source="filename",
    )

    plan = config.prepared_discovery
    if plan is not None:
        source_paths = list(plan.source_paths)
        by_time_all = plan.by_time_filtered
        use_funpack = plan.use_funpack
        repair_zero_beam = (
            config.repair_zero_beam
            if config.repair_zero_beam is not None
            else use_funpack
        )
    else:
        source_paths = collect_glob_sources(config.glob_pattern)
        if not source_paths:
            msg = f"Glob matched no files: {config.glob_pattern}"
            raise FileNotFoundError(msg)

        use_funpack = (
            config.funpack
            if config.funpack is not None
            else sources_need_funpack(source_paths)
        )
        repair_zero_beam = (
            config.repair_zero_beam
            if config.repair_zero_beam is not None
            else use_funpack
        )

        by_time_all = discover_time_grouped_paths(
            source_paths,
            duplicate_resolver=config.duplicate_resolver,
            discovery=discovery,
        )
        if not by_time_all:
            msg = "No groupable source files after discovery."
            raise FileNotFoundError(msg)

    work_root = config.work_root
    if use_funpack and work_root is None:
        msg = "work_root is required when funpack is enabled (compressed .fs sources)"
        raise ValueError(msg)
    if work_root is None:
        work_root = config.staging_dir / ".work"

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.fixed_dir.mkdir(parents=True, exist_ok=True)
    config.staging_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    out_zarr = config.output_dir / config.zarr_name

    logger.info(
        "Discovered %d source file(s) in %d time group(s).",
        sum(len(v) for v in by_time_all.values()),
        len(by_time_all),
    )

    report_ingest_progress(
        progress_callback,
        "setup",
        0,
        2,
        "Loading global LM reference grid…",
    )
    lm_ref_ds = _load_global_lm_reference_dataset(
        by_time_all,
        config.fixed_dir,
        chunk_lm=config.chunk_lm,
        fix_headers_on_demand=config.fix_headers_on_demand,
        target_size=config.lm_reference_target_size,
        group_metadata_source=discovery.group_metadata_source,
        filename_convention=discovery.filename_convention,
        discovery_metadata=plan.discovery_metadata if plan is not None else None,
    ).copy(deep=True)

    report_ingest_progress(
        progress_callback,
        "setup",
        1,
        2,
        "Building global frequency axis…",
    )
    global_freq_hz = _global_frequency_coord_hz(
        by_time_all,
        group_metadata_source=discovery.group_metadata_source,
        filename_convention=discovery.filename_convention,
        discovery_metadata=plan.discovery_metadata if plan is not None else None,
    )

    by_time = (
        plan.to_process
        if plan is not None
        else prepare_ingest_time_groups(
            by_time_all,
            out_zarr=out_zarr,
            rebuild=config.rebuild,
            resume=config.resume,
            require_73mhz=False,
            context="convert",
            filter_invalid_beam=not repair_zero_beam,
        )
    )
    if not by_time:
        logger.info("Every time key is already in %s.", out_zarr)
        _consolidate_zarr_metadata(out_zarr)
        return out_zarr

    first_zarr_write = not (out_zarr.exists() and not config.rebuild)
    time_keys_sorted = sorted(by_time.keys())
    total = len(time_keys_sorted)

    for idx, tkey in enumerate(time_keys_sorted):
        sources = list(by_time[tkey])
        work_dir = work_root / tkey
        _cleanup_time_work(work_dir, config.staging_dir, tkey)

        report_ingest_progress(
            progress_callback,
            "converting",
            idx,
            total,
            f"Time step {idx + 1}/{total}: preparing {tkey}",
        )

        logger.info(
            "Time %s (%d/%d): preparing %d source(s)",
            tkey,
            idx + 1,
            total,
            len(sources),
        )

        if use_funpack:
            logger.info("Funpacking into %s", work_dir)
            prepared = funpack_time_group(sources, work_dir)
            if repair_zero_beam:
                repaired = repair_zero_beam_from_nearby_time(
                    sources,
                    prepared,
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
        else:
            prepared = sources

        n_staged = stage_time_group_symlinks(config.staging_dir, tkey, prepared)
        logger.info("Staged %d FITS for convert", n_staged)

        convert_config = ConversionConfig(
            input_dir=config.staging_dir,
            output_dir=config.output_dir,
            zarr_name=config.zarr_name,
            fixed_dir=config.fixed_dir,
            chunk_lm=config.chunk_lm,
            rebuild=first_zarr_write,
            resume=False,
            fix_headers_on_demand=config.fix_headers_on_demand,
            cleanup_fixed_fits=config.cleanup_fixed_fits,
            duplicate_resolver=config.duplicate_resolver,
            discovery_freq_bin_hz=discovery.freq_bin_hz,
            time_keys_only=(tkey,),
            lm_reference_ds=lm_ref_ds,
            group_metadata_source=discovery.group_metadata_source,
            discovery_time_key_source=discovery.time_key_source,
            discovery_filename_convention=discovery.filename_convention,
            lm_reference_target_size=config.lm_reference_target_size,
            consolidate_metadata_at_end=False,
            global_frequency_coord_hz=global_freq_hz if first_zarr_write else None,
            verbose=config.verbose,
        )
        # Inner convert handles one staged time key; outer loop owns progress reporting.
        FITSToZarrConverter(convert_config, progress_callback=None).convert()
        report_ingest_progress(
            progress_callback,
            "converting",
            idx + 1,
            total,
            f"Time step {idx + 1}/{total}: wrote {tkey}",
        )
        first_zarr_write = False
        _clear_sky_coord_cache()
        _cleanup_time_work(work_dir, config.staging_dir, tkey)

    _consolidate_zarr_metadata(out_zarr)
    return out_zarr
