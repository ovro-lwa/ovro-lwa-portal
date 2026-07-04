"""Persist glob discovery metadata beside the output Zarr for fast convert reruns."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from astropy.io import fits

from ovro_lwa_portal.fits_to_zarr_xradio import (
    _DiscoveryFileMetadata,
    _lm_shape_from_discovery_metadata,
    _peek_lm_shape,
)
from ovro_lwa_portal.ingest.discovery import (
    GlobConvertDiscoveryPlan,
    IngestDiscoveryConfig,
    IngestDiscoverySummary,
    collect_glob_sources,
    prepare_ingest_time_groups,
    summarize_time_grouped_fits,
)

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1

__all__ = [
    "discovery_sidecar_path_for_zarr",
    "load_glob_discovery_sidecar",
    "save_glob_discovery_sidecar",
]


def discovery_sidecar_path_for_zarr(out_zarr: Path) -> Path:
    """Return ``{zarr_stem}_metadata.json`` next to the Zarr store directory."""
    name = out_zarr.name
    if name.endswith(".zarr"):
        stem = name[: -len(".zarr")]
    else:
        stem = name
    return out_zarr.parent / f"{stem}_metadata.json"


def _discovery_config_payload(discovery: IngestDiscoveryConfig) -> dict[str, Any]:
    return {
        "freq_bin_hz": float(discovery.freq_bin_hz),
        "group_metadata_source": discovery.group_metadata_source,
        "time_key_source": discovery.time_key_source,
        "filename_convention": discovery.filename_convention,
    }


def _discovery_config_from_payload(payload: dict[str, Any]) -> IngestDiscoveryConfig:
    return IngestDiscoveryConfig(
        freq_bin_hz=float(payload["freq_bin_hz"]),
        group_metadata_source=payload["group_metadata_source"],
        time_key_source=payload["time_key_source"],
        filename_convention=payload["filename_convention"],
    )


def _lm_shape_for_sidecar(fp: Path, meta: _DiscoveryFileMetadata) -> tuple[int, int] | None:
    shape = _lm_shape_from_discovery_metadata(fp, meta)
    if shape is not None:
        return shape
    try:
        return _peek_lm_shape(fp)
    except Exception:
        return None


def _header_text_for_sidecar(meta: _DiscoveryFileMetadata) -> str | None:
    if meta.ingest_header is None:
        return None
    return meta.ingest_header.tostring(sep="\n")


def _metadata_from_sidecar_entry(
    entry: dict[str, Any],
) -> _DiscoveryFileMetadata:
    notes = tuple(str(n) for n in entry.get("notes", ()))
    time_key = entry.get("time_key")
    frequency_hz = entry.get("frequency_hz")
    stokes_key = int(entry["stokes_key"])
    hdr: fits.Header | None = None
    header_text = entry.get("ingest_header_text")
    if header_text:
        hdr = fits.Header.fromstring(str(header_text), sep="\n")
    elif entry.get("lm_naxis1") is not None and entry.get("lm_naxis2") is not None:
        hdr = fits.Header(
            {
                "NAXIS": 2,
                "NAXIS1": int(entry["lm_naxis1"]),
                "NAXIS2": int(entry["lm_naxis2"]),
            }
        )
    freq = float(frequency_hz) if frequency_hz is not None else None
    return _DiscoveryFileMetadata(
        time_key if time_key is None else str(time_key),
        freq,
        notes,
        stokes_key,
        hdr,
    )


def _paths_by_time_from_payload(
    payload: dict[str, list[str]],
) -> dict[str, list[Path]]:
    return {tkey: [Path(p) for p in paths] for tkey, paths in payload.items()}


def save_glob_discovery_sidecar(
    sidecar_path: Path,
    *,
    glob_pattern: str,
    discovery: IngestDiscoveryConfig,
    plan: GlobConvertDiscoveryPlan,
) -> None:
    """Write discovery results for *plan* to *sidecar_path*."""
    files: dict[str, Any] = {}
    for fp, meta in plan.discovery_metadata.items():
        resolved = str(fp.resolve())
        try:
            st = fp.stat()
        except OSError:
            continue
        shape = _lm_shape_for_sidecar(fp, meta)
        entry: dict[str, Any] = {
            "mtime_ns": int(st.st_mtime_ns),
            "size": int(st.st_size),
            "time_key": meta.time_key,
            "frequency_hz": meta.frequency_hz,
            "stokes_key": int(meta.stokes_key),
            "notes": list(meta.notes),
        }
        header_text = _header_text_for_sidecar(meta)
        if header_text is not None:
            entry["ingest_header_text"] = header_text
        if shape is not None:
            entry["lm_naxis1"] = int(shape[0])
            entry["lm_naxis2"] = int(shape[1])
        files[resolved] = entry

    payload = {
        "schema_version": _SCHEMA_VERSION,
        "glob_pattern": glob_pattern,
        "discovery": _discovery_config_payload(discovery),
        "filter_invalid_beam": bool(plan.filter_invalid_beam),
        "use_funpack": bool(plan.use_funpack),
        "files": files,
        "by_time_filtered": {
            tkey: [str(p.resolve()) for p in paths]
            for tkey, paths in plan.by_time_filtered.items()
        },
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = sidecar_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(sidecar_path)
    logger.info("Wrote discovery sidecar (%d files): %s", len(files), sidecar_path)


def _sidecar_matches_glob(
    payload: dict[str, Any],
    source_paths: list[Path],
) -> bool:
    files = payload.get("files", {})
    current = {str(p.resolve()) for p in source_paths}
    if set(files.keys()) != current:
        logger.info(
            "Discovery sidecar stale: glob file set changed (%d cached, %d current).",
            len(files),
            len(current),
        )
        return False
    for fp in source_paths:
        key = str(fp.resolve())
        entry = files[key]
        try:
            st = fp.stat()
        except OSError:
            logger.info("Discovery sidecar stale: missing file %s", fp)
            return False
        if int(st.st_mtime_ns) != int(entry["mtime_ns"]) or int(st.st_size) != int(
            entry["size"]
        ):
            logger.info("Discovery sidecar stale: file identity changed for %s", fp.name)
            return False
    return True


def load_glob_discovery_sidecar(
    sidecar_path: Path,
    *,
    glob_pattern: str,
    discovery: IngestDiscoveryConfig,
    out_zarr: Path,
    rebuild: bool,
    resume: bool,
    funpack: bool | None,
) -> GlobConvertDiscoveryPlan | None:
    """Load a cached plan when the sidecar matches the current glob and file mtimes."""
    if not sidecar_path.is_file():
        return None
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read discovery sidecar %s: %s", sidecar_path, exc)
        return None

    if int(payload.get("schema_version", 0)) != _SCHEMA_VERSION:
        logger.info("Discovery sidecar stale: unsupported schema version.")
        return None
    if payload.get("glob_pattern") != glob_pattern:
        logger.info("Discovery sidecar stale: glob pattern changed.")
        return None
    cached_discovery = _discovery_config_from_payload(payload["discovery"])
    if cached_discovery != discovery:
        logger.info("Discovery sidecar stale: discovery settings changed.")
        return None

    from ovro_lwa_portal.ingest.per_time_convert import sources_need_funpack

    source_paths = collect_glob_sources(glob_pattern)
    if not source_paths:
        return None
    use_funpack = funpack if funpack is not None else sources_need_funpack(source_paths)
    filter_invalid_beam = not use_funpack
    if bool(payload.get("use_funpack")) != bool(use_funpack):
        logger.info("Discovery sidecar stale: funpack setting changed.")
        return None
    if bool(payload.get("filter_invalid_beam")) != bool(filter_invalid_beam):
        logger.info("Discovery sidecar stale: beam-filter setting changed.")
        return None
    if not _sidecar_matches_glob(payload, source_paths):
        return None

    discovery_metadata: dict[Path, _DiscoveryFileMetadata] = {}
    for path_str, entry in payload["files"].items():
        discovery_metadata[Path(path_str)] = _metadata_from_sidecar_entry(entry)

    by_time_filtered = _paths_by_time_from_payload(payload["by_time_filtered"])
    discovered = summarize_time_grouped_fits(
        by_time_filtered,
        discovery=discovery,
        discovery_metadata=discovery_metadata,
    )
    to_process = prepare_ingest_time_groups(
        by_time_filtered,
        out_zarr=out_zarr,
        rebuild=rebuild,
        resume=resume,
        require_73mhz=False,
        context="convert",
        filter_invalid_beam=False,
    )
    to_process_summary = summarize_time_grouped_fits(
        to_process,
        discovery=discovery,
        discovery_metadata=discovery_metadata,
    )
    logger.info(
        "Loaded discovery sidecar (%d files, %d time groups): %s",
        len(source_paths),
        len(by_time_filtered),
        sidecar_path,
    )
    return GlobConvertDiscoveryPlan(
        source_paths=tuple(source_paths),
        by_time_all=by_time_filtered,
        by_time_filtered=by_time_filtered,
        discovery_metadata=discovery_metadata,
        to_process=to_process,
        discovered=discovered,
        to_process_summary=to_process_summary,
        filter_invalid_beam=filter_invalid_beam,
        use_funpack=use_funpack,
    )
