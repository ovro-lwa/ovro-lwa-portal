"""Shared FITS discovery and pre-ingest filtering for convert and dewarp-convert."""

from __future__ import annotations

import glob
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal

from ovro_lwa_portal.fits_to_zarr_xradio import (
    _DISCOVERY_FREQ_BIN_HZ,
    DiscoveryFilenameConvention,
    _discover_groups,
    _discover_groups_from_files,
    _filter_completed_time_keys,
    _filter_invalid_beam_files,
)

__all__ = [
    "DEFAULT_INGEST_DISCOVERY",
    "IngestDiscoveryConfig",
    "collect_glob_sources",
    "discover_time_grouped_fits",
    "discover_time_grouped_paths",
    "prepare_ingest_time_groups",
]


@dataclass(frozen=True)
class IngestDiscoveryConfig:
    """Parameters shared by ``convert``, ``dewarp-convert``, and ``audit-metadata``."""

    freq_bin_hz: float = _DISCOVERY_FREQ_BIN_HZ
    group_metadata_source: Literal["fits", "filename"] = "fits"
    time_key_source: Literal["header", "filename"] = "filename"
    filename_convention: DiscoveryFilenameConvention = "image"


DEFAULT_INGEST_DISCOVERY = IngestDiscoveryConfig()


def collect_glob_sources(glob_pattern: str) -> list[Path]:
    """Return sorted paths matched by a Python :func:`glob.glob` pattern."""
    return [Path(p) for p in sorted(glob.glob(glob_pattern))]


def discover_time_grouped_fits(
    in_dir: Path,
    *,
    duplicate_resolver: Callable[[str, float, List[Path]], Path] | None = None,
    discovery: IngestDiscoveryConfig | None = None,
) -> Dict[str, List[Path]]:
    """Group FITS under *in_dir* by observation time and frequency bin."""
    cfg = discovery or DEFAULT_INGEST_DISCOVERY
    return _discover_groups(
        in_dir,
        duplicate_resolver=duplicate_resolver,
        freq_bin_hz=cfg.freq_bin_hz,
        time_key_source=cfg.time_key_source,
        group_metadata_source=cfg.group_metadata_source,
        filename_convention=cfg.filename_convention,
    )


def discover_time_grouped_paths(
    paths: Sequence[Path],
    *,
    duplicate_resolver: Callable[[str, float, List[Path]], Path] | None = None,
    discovery: IngestDiscoveryConfig | None = None,
) -> Dict[str, List[Path]]:
    """Group explicit FITS paths by observation time and frequency bin."""
    cfg = discovery or DEFAULT_INGEST_DISCOVERY
    return _discover_groups_from_files(
        paths,
        duplicate_resolver=duplicate_resolver,
        freq_bin_hz=cfg.freq_bin_hz,
        time_key_source=cfg.time_key_source,
        group_metadata_source=cfg.group_metadata_source,
        filename_convention=cfg.filename_convention,
    )


def prepare_ingest_time_groups(
    by_time: Dict[str, List[Path]],
    *,
    out_zarr: Path | None = None,
    rebuild: bool = False,
    resume: bool = True,
    require_73mhz: bool = False,
    context: str = "convert",
    filter_invalid_beam: bool = True,
) -> Dict[str, List[Path]]:
    """Apply truncation/beam validity, optional 73 MHz, and optional resume filters.

    Set ``filter_invalid_beam=False`` when a downstream step repairs placeholder
    ``BMAJ``/``BMIN`` (e.g. per-time funpack + nearby-time beam copy) before convert.
    """
    if filter_invalid_beam:
        filtered = _filter_invalid_beam_files(by_time)
    else:
        filtered = dict(by_time)
    if require_73mhz:
        from ovro_lwa_portal.ingest.dewarp_convert import (
            _filter_time_groups_without_cascade_reference,
        )

        filtered = _filter_time_groups_without_cascade_reference(filtered)
    if resume and out_zarr is not None and not rebuild:
        filtered = _filter_completed_time_keys(
            filtered, out_zarr, rebuild=False, context=context
        )
    return filtered
