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
    _extract_group_metadata_for_discovery,
    _filter_completed_time_keys,
    _filter_invalid_beam_files,
    _filter_lst_color_groups_with_mismatched_header_times,
    _stokes_key_for_discovery,
    estimate_zarr_store_bytes,
    format_data_size,
)

__all__ = [
    "DEFAULT_INGEST_DISCOVERY",
    "IngestDiscoveryConfig",
    "IngestDiscoverySummary",
    "collect_glob_sources",
    "discover_time_grouped_fits",
    "discover_time_grouped_paths",
    "format_data_size",
    "plan_convert_discovery",
    "prepare_ingest_time_groups",
    "summarize_time_grouped_fits",
]

_STOKES_LABELS: dict[int, str] = {
    1: "I",
    2: "Q",
    3: "U",
    4: "V",
    -1: "RR",
    -2: "LL",
    -3: "RL",
    -4: "LR",
}


@dataclass(frozen=True)
class IngestDiscoveryConfig:
    """Parameters shared by ``convert``, ``dewarp-convert``, and ``audit-metadata``."""

    freq_bin_hz: float = _DISCOVERY_FREQ_BIN_HZ
    group_metadata_source: Literal["fits", "filename"] = "fits"
    time_key_source: Literal["header", "filename"] = "filename"
    filename_convention: DiscoveryFilenameConvention = "image"


DEFAULT_INGEST_DISCOVERY = IngestDiscoveryConfig()


@dataclass(frozen=True)
class IngestDiscoverySummary:
    """Counts of FITS inputs and (time, frequency, polarization) groups."""

    input_files: int
    time_groups: int
    frequency_groups: int
    polarization_groups: int
    polarization_labels: tuple[str, ...]
    time_frequency_polarization_cells: int
    estimated_zarr_bytes: int | None = None

    @property
    def zarr_shape_hint(self) -> str:
        """Human-readable ``(time, frequency, polarization)`` extent."""
        return (
            f"({self.time_groups}, {self.frequency_groups}, {self.polarization_groups})"
        )

    @property
    def estimated_zarr_size(self) -> str | None:
        """Human-readable Zarr size estimate (4 B/pixel), if available."""
        if self.estimated_zarr_bytes is None:
            return None
        return format_data_size(self.estimated_zarr_bytes)


def summarize_time_grouped_fits(
    by_time: Dict[str, List[Path]],
    *,
    discovery: IngestDiscoveryConfig | None = None,
) -> IngestDiscoverySummary:
    """Summarize grouped FITS paths by time, frequency subband, and polarization."""
    cfg = discovery or DEFAULT_INGEST_DISCOVERY
    freq_bins: set[int] = set()
    stokes_keys: set[int] = set()
    input_files = 0

    for files in by_time.values():
        input_files += len(files)
        for fp in files:
            _, frequency_hz, _ = _extract_group_metadata_for_discovery(
                fp,
                filename_convention=cfg.filename_convention,
                group_metadata_source=cfg.group_metadata_source,
                time_key_source=cfg.time_key_source,
            )
            if frequency_hz is not None:
                freq_bins.add(int(round(float(frequency_hz) / cfg.freq_bin_hz)))
            stokes_keys.add(
                _stokes_key_for_discovery(
                    fp,
                    group_metadata_source=cfg.group_metadata_source,
                )
            )

    labels = tuple(
        _STOKES_LABELS.get(code, str(code)) for code in sorted(stokes_keys)
    )
    return IngestDiscoverySummary(
        input_files=input_files,
        time_groups=len(by_time),
        frequency_groups=len(freq_bins),
        polarization_groups=len(stokes_keys),
        polarization_labels=labels,
        time_frequency_polarization_cells=input_files,
        estimated_zarr_bytes=estimate_zarr_store_bytes(by_time),
    )


def _apply_convert_discovery_filters(
    by_time: Dict[str, List[Path]],
    *,
    discovery: IngestDiscoveryConfig,
    filter_invalid_beam: bool,
) -> Dict[str, List[Path]]:
    filtered = _filter_invalid_beam_files(by_time) if filter_invalid_beam else dict(by_time)
    if discovery.filename_convention == "lst-color":
        filtered = _filter_lst_color_groups_with_mismatched_header_times(filtered)
    return filtered


def plan_convert_discovery(
    by_time: Dict[str, List[Path]],
    *,
    discovery: IngestDiscoveryConfig,
    out_zarr: Path,
    rebuild: bool,
    resume: bool,
    filter_invalid_beam: bool = True,
) -> tuple[IngestDiscoverySummary, IngestDiscoverySummary]:
    """Return discovery and to-process summaries for convert / per-time glob."""
    filtered = _apply_convert_discovery_filters(
        by_time,
        discovery=discovery,
        filter_invalid_beam=filter_invalid_beam,
    )
    discovered = summarize_time_grouped_fits(filtered, discovery=discovery)
    to_process_groups = prepare_ingest_time_groups(
        filtered,
        out_zarr=out_zarr,
        rebuild=rebuild,
        resume=resume,
        require_73mhz=False,
        context="convert",
        filter_invalid_beam=False,
    )
    to_process = summarize_time_grouped_fits(to_process_groups, discovery=discovery)
    return discovered, to_process


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
