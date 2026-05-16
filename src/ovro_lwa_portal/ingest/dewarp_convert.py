"""Dewarp (``image_plane_correction.flow.flow_cascade73MHz``) staging before FITS→Zarr."""

from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from ovro_lwa_portal.fits_to_zarr_xradio import (
    _consolidate_zarr_metadata,
    _load_global_lm_reference_dataset,
)
from ovro_lwa_portal.ingest.discovery import (
    IngestDiscoveryConfig,
    discover_time_grouped_fits,
    prepare_ingest_time_groups,
)

__all__ = [
    "collect_cascade_fits",
    "import_flow_cascade73mhz",
    "remove_staged_files_for_time_key",
    "run_cascade_for_time_key",
    "run_cascade_per_time_group",
    "dewarp_and_convert_append_each_time",
]

logger = logging.getLogger(__name__)


# Subband-token regex used to test whether a basename carries a given
# ``<N>MHz`` tag without false-positive matches on substrings like
# ``1473MHz``. Matches start-of-string, ``_``, or ``-`` before the digits
# and the same separators (or ``.``/end-of-string) after the ``MHz`` token.
_CASCADE_SUBBAND_TOKEN_RE = re.compile(r"(?:^|[_-])(\d+)MHz(?:[_-]|\.|$)")


def _file_subband_mhz(fp: Path) -> int | None:
    """Return the subband (in MHz) parsed from *fp*'s basename, or ``None`` if absent.

    Mirrors the OVRO-LWA naming convention (e.g. ``73MHz-I-Deep-…`` or
    ``_73MHz_averaged_…``). Stricter than ``MHZ_RE`` in
    ``ovro_lwa_portal.fits_to_zarr_xradio``: it accepts start-of-string and the
    hyphen separator that OVRO product names use, while still requiring a
    boundary so ``1473MHz`` does not match the ``73MHz`` cascade token.
    """
    m = _CASCADE_SUBBAND_TOKEN_RE.search(fp.name)
    return int(m.group(1)) if m else None


def _has_cascade_reference_subband(files: Sequence[Path], *, target_mhz: int = 73) -> bool:
    """Whether any path in *files* carries the ``{target_mhz}MHz`` cascade reference tag.

    ``image_plane_correction.flow.flow_cascade73MHz`` is anchored at 73 MHz and
    locates its peer subbands by substring-replacing the literal ``73MHz`` token
    in each input path. A group with no 73 MHz file cannot be cascaded, so we
    use this check (rather than the FITS header frequency) to decide whether the
    cascade can run for the group.
    """
    return any(_file_subband_mhz(fp) == target_mhz for fp in files)


def _filter_time_groups_without_cascade_reference(
    by_time: dict[str, list[Path]],
    *,
    target_mhz: int = 73,
) -> dict[str, list[Path]]:
    """Drop time groups whose files have no ``{target_mhz}MHz`` reference subband.

    The dewarp cascade ``flow_cascade73MHz`` is anchored at 73 MHz and locates
    peer subbands by substring replacement of ``73MHz`` in each input path.
    Time groups missing that reference cannot be cascaded; this filter removes
    them with a warning so the (expensive) cascade is never invoked on an
    incomplete group and the downstream Zarr is not contaminated with partially
    dewarped products for those observation times.

    Parameters
    ----------
    by_time
        Discovery output mapping observation-time key → list of FITS file paths,
        typically already passed through
        :func:`~ovro_lwa_portal.fits_to_zarr_xradio._filter_invalid_beam_files`.
    target_mhz
        Subband MHz the cascade is anchored on. Defaults to ``73`` to match
        ``flow_cascade73MHz``.

    Returns
    -------
    dict[str, list[Path]]
        Same shape as *by_time* but with groups missing the reference subband
        dropped entirely.
    """
    filtered: dict[str, list[Path]] = {}
    for tkey, files in by_time.items():
        if _has_cascade_reference_subband(files, target_mhz=target_mhz):
            filtered[tkey] = files
            continue
        subbands_present = sorted({mhz for mhz in (_file_subband_mhz(fp) for fp in files) if mhz is not None})
        logger.warning(
            "Skipping time=%s for dewarp cascade: no %dMHz reference image found "
            "among %d file(s) (subbands present: %s); group cannot be cascaded.",
            tkey,
            target_mhz,
            len(files),
            subbands_present if subbands_present else "<none parseable from basenames>",
        )
    return filtered


def import_flow_cascade73mhz() -> Callable[..., Any]:
    """Import and return ``image_plane_correction.flow.flow_cascade73MHz``.

    Raises
    ------
    ImportError
        If ``image_plane_correction`` is not installed, or ``flow`` has no
        ``flow_cascade73MHz``.
    """
    try:
        from image_plane_correction import flow as flow_mod
    except ImportError as e:
        msg = (
            "Could not import `image_plane_correction.flow` (install the "
            "`image-plane-correction` / `image_plane_correction` package in the "
            "same Python environment you use to run `ovro-ingest`)."
        )
        raise ImportError(msg) from e
    fn = getattr(flow_mod, "flow_cascade73MHz", None)
    if fn is None:
        msg = (
            "`image_plane_correction.flow` has no attribute `flow_cascade73MHz`. "
            "Check your image_plane_correction version."
        )
        raise ImportError(msg)
    return fn


def collect_cascade_fits(outroot: Path) -> list[Path]:
    """Return sorted ``*.fits`` paths under *outroot* (non-recursive, then recursive)."""
    direct = sorted(outroot.glob("*.fits"))
    if direct:
        return direct
    return sorted(outroot.rglob("*.fits"))


def _link_or_copy(src: Path, dest: Path) -> None:
    """Symlink *src* to *dest* when possible; otherwise copy."""
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dest)


def remove_staged_files_for_time_key(staging_dir: Path, time_key: str) -> int:
    """Remove ``{time_key}__*.fits`` under *staging_dir*; return count removed."""
    n = 0
    for p in staging_dir.glob(f"{time_key}__*.fits"):
        try:
            p.unlink(missing_ok=True)
            n += 1
        except OSError as exc:
            logger.warning("Could not remove staged FITS %s: %s", p, exc)
    return n


def run_cascade_for_time_key(
    time_key: str,
    files: Sequence[Path],
    cascade_parent: Path,
    staging_dir: Path,
    *,
    cascade_fn: Callable[..., Any],
    cleaned: bool = True,
    qa: bool = True,
    use_best_pb_model: bool = True,
    bright_source_flux_qa: bool = True,
    write: bool = True,
    target_size: int | None = None,
) -> int:
    """Run ``flow_cascade73MHz`` for one time key and link outputs into *staging_dir*.

    Staged names are ``{time_key}__{basename}`` (same as :func:`run_cascade_per_time_group`).

    Returns
    -------
    int
        Number of FITS linked or copied into *staging_dir* for this time key.
    """
    outroot = cascade_parent / time_key
    if outroot.exists():
        shutil.rmtree(outroot)
    outroot.mkdir(parents=True, exist_ok=True)
    fns: Sequence[str] = [str(p.absolute()) for p in files]
    logger.info(
        "Running flow_cascade73MHz for time_key=%s (%d files) → %s",
        time_key,
        len(fns),
        outroot,
    )
    cascade_fn(
        image_filenames=fns,
        cleaned=cleaned,
        qa=qa,
        use_best_pb_model=use_best_pb_model,
        bright_source_flux_qa=bright_source_flux_qa,
        write=write,
        outroot=str(outroot),
        target_size=target_size,
    )
    produced = collect_cascade_fits(outroot)
    if not produced:
        msg = (
            f"flow_cascade73MHz produced no *.fits under {outroot}. "
            "Check the flow package output layout or enable write=True."
        )
        raise RuntimeError(msg)
    n_staged = 0
    for p in produced:
        dest = staging_dir / f"{time_key}__{p.name}"
        if dest.exists():
            dest.unlink()
        _link_or_copy(p, dest)
        n_staged += 1
    return n_staged


def run_cascade_per_time_group(
    input_dir: Path,
    cascade_parent: Path,
    staging_dir: Path,
    *,
    discovery_freq_bin_hz: float,
    duplicate_resolver: Callable[[str, float, list[Path]], Path] | None = None,
    cascade_fn: Callable[..., Any] | None = None,
    cleaned: bool = True,
    qa: bool = True,
    use_best_pb_model: bool = True,
    bright_source_flux_qa: bool = True,
    write: bool = True,
    target_size: int | None = None,
    clear_staging: bool = True,
    group_metadata_source: Literal["fits", "filename"] = "fits",
    time_key_source: Literal["header", "filename"] = "filename",
    out_zarr: Path | None = None,
    rebuild: bool = False,
    resume: bool = True,
    fixed_dir: Path | None = None,
    chunk_lm: int = 1024,
    fix_headers_on_demand: bool = True,
) -> tuple[int, list[str], Any | None]:
    """Run ``flow_cascade73MHz`` once per observation-time group and stage outputs.

    FITS under *input_dir* are grouped like :func:`convert_fits_dir_to_zarr`. With the
    default ``group_metadata_source="fits"``, the time key prefers the basename
    ``-image-YYYYMMDD_HHMMSS`` stamp when present (so multi-band images that share a pipeline
    image id stay in one group even if ``DATE-OBS`` differs between symlink targets). If that
    pattern is missing, grouping falls back to ``DATE-OBS`` like the Zarr ingest path.
    Pass ``group_metadata_source="filename"`` to infer time and subband **only** from the
    basename (no FITS header reads during discovery); that requires the ``-image-`` stamp and
    ``_NNNMHz_`` tokens when frequency bins matter.

    For each time key, all subband files in that group are passed as ``image_filenames`` to
    ``image_plane_correction.flow.flow_cascade73MHz`` with ``outroot=cascade_parent / time_key``.
    Paths are passed as :meth:`pathlib.Path.absolute` strings (not :meth:`~pathlib.Path.resolve`)
    so symlinks in a flat staging directory are preserved: ``flow_cascade73MHz`` locates peers
    by replacing the ``73MHz`` token in each path string, which only matches the real
    on-disk layout when every subband shares the same directory prefix (or when unresolved
    symlinks all differ only by that token).

    Time groups whose files have no ``73MHz`` subband tag in their basenames are dropped
    via :func:`_filter_time_groups_without_cascade_reference` with a warning **before** the
    cascade is invoked. ``flow_cascade73MHz`` is anchored at 73 MHz, so a group lacking that
    reference cannot be cascaded. Those observation times do not appear in the staging
    directory and therefore do not reach the downstream Zarr (consistent with how
    invalid-beam files and resume-skipped time keys are handled).

    Produced ``*.fits`` (immediate children of *outroot*, else recursive) are linked
    or copied into *staging_dir* as ``{time_key}__{basename}`` so the flat directory
    can be consumed by :meth:`ovro_lwa_portal.ingest.core.FITSToZarrConverter.convert`.

    Parameters
    ----------
    input_dir, cascade_parent, staging_dir
        Input FITS directory, per-time cascade output parent, and flat staging dir.
    discovery_freq_bin_hz, duplicate_resolver, group_metadata_source, time_key_source
        Shared discovery settings (same as ``convert``); see
        :class:`~ovro_lwa_portal.ingest.discovery.IngestDiscoveryConfig`.
    resume
        When True (default) and ``rebuild`` is False, skip time keys already in *out_zarr*.
    fixed_dir, chunk_lm, fix_headers_on_demand
        When *fixed_dir* is set, build a global LM reference from raw FITS (before the
        resume filter) and return it as the third tuple element for the downstream convert step.
    cascade_fn
        Callable with the same keyword interface as
        ``image_plane_correction.flow.flow_cascade73MHz``.
        Defaults to importing that function.
    cleaned, qa, use_best_pb_model, bright_source_flux_qa, write, target_size
        Forwarded to *cascade_fn* together with ``image_filenames`` and ``outroot``.
        *target_size* matches ``image_plane_correction`` dewarp entry points
        (e.g. ``calcflow``, ``flow_cascade73MHz``): side length in pixels for the
        output raster, or ``None`` for the library default.
    clear_staging
        If True (default), remove *staging_dir* before running. If False, caller manages
        staging (used when interleaving Zarr conversion after each time key).
    out_zarr
        Downstream Zarr path used with ``resume`` to skip completed observation times.
    rebuild
        When True, dewarp every discovered time key and ignore resume.

    Returns
    -------
    n_staged : int
        Number of FITS linked or copied into *staging_dir*.
    time_keys : list of str
        Sorted time keys that were processed.
    lm_reference_ds
        Global LM reference when *fixed_dir* was provided, else ``None``.

    Raises
    ------
    FileNotFoundError
        If discovery finds no groupable FITS.
    RuntimeError
        If a cascade step writes no ``*.fits`` under the per-time *outroot*.
    """
    cascade_fn = cascade_fn or import_flow_cascade73mhz()
    if clear_staging:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
    else:
        staging_dir.mkdir(parents=True, exist_ok=True)
    cascade_parent.mkdir(parents=True, exist_ok=True)

    discovery = IngestDiscoveryConfig(
        freq_bin_hz=discovery_freq_bin_hz,
        group_metadata_source=group_metadata_source,
        time_key_source=time_key_source,
    )
    by_time = discover_time_grouped_fits(
        input_dir, duplicate_resolver=duplicate_resolver, discovery=discovery
    )
    by_time = prepare_ingest_time_groups(
        by_time,
        rebuild=True,
        resume=False,
        require_73mhz=False,
        context="dewarp-convert",
    )
    if not by_time:
        msg = f"No groupable FITS files found in {input_dir}"
        raise FileNotFoundError(msg)

    by_time = prepare_ingest_time_groups(
        by_time,
        rebuild=True,
        resume=False,
        require_73mhz=True,
        context="dewarp-convert",
    )
    if not by_time:
        logger.warning(
            "No time group in %s contains a 73 MHz reference image; nothing to dewarp.",
            input_dir,
        )
        return 0, [], None

    lm_ref_ds = None
    if fixed_dir is not None:
        lm_ref_ds = _load_global_lm_reference_dataset(
            by_time,
            fixed_dir,
            chunk_lm=chunk_lm,
            fix_headers_on_demand=fix_headers_on_demand,
            target_size=target_size,
            group_metadata_source=group_metadata_source,
        ).copy(deep=True)

    by_time = prepare_ingest_time_groups(
        by_time,
        out_zarr=out_zarr,
        rebuild=rebuild,
        resume=resume,
        require_73mhz=False,
        context="dewarp-convert",
    )
    if not by_time:
        logger.info(
            "Nothing to dewarp: every discovered time key is already present in %s. "
            "Pass --rebuild to start over, or --no-resume to reprocess all times.",
            out_zarr,
        )
        return 0, [], lm_ref_ds

    time_keys_sorted = sorted(by_time.keys())
    n_staged = 0
    for tkey in time_keys_sorted:
        files = list(by_time[tkey])
        n_staged += run_cascade_for_time_key(
            tkey,
            files,
            cascade_parent,
            staging_dir,
            cascade_fn=cascade_fn,
            cleaned=cleaned,
            qa=qa,
            use_best_pb_model=use_best_pb_model,
            bright_source_flux_qa=bright_source_flux_qa,
            write=write,
            target_size=target_size,
        )
    return n_staged, time_keys_sorted, lm_ref_ds


def dewarp_and_convert_append_each_time(
    input_dir: Path,
    output_dir: Path,
    cascade_parent: Path,
    staging_dir: Path,
    fixed_dir: Path,
    *,
    zarr_name: str,
    chunk_lm: int,
    rebuild: bool,
    fix_headers_on_demand: bool,
    cleanup_fixed_fits: bool,
    discovery_freq_bin_hz: float,
    duplicate_resolver: Callable[[str, float, list[Path]], Path] | None,
    cleaned: bool = True,
    qa: bool = True,
    use_best_pb_model: bool = True,
    bright_source_flux_qa: bool = True,
    write: bool = True,
    target_size: int | None = None,
    cascade_fn: Callable[..., Any] | None = None,
    verbose: bool = False,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
    group_metadata_source: Literal["fits", "filename"] = "fits",
    time_key_source: Literal["header", "filename"] = "filename",
    resume: bool = True,
) -> tuple[int, list[str]]:
    """Dewarp each time group, append its Zarr slice, then clean staging/cascade for that time.

    Builds a single global LM reference from *input_dir* (raw FITS layout) once, then for
    each observation time: run the cascade, stage dewarped FITS, run
    :class:`~ovro_lwa_portal.ingest.core.FITSToZarrConverter` with ``time_keys_only`` set to
    that step only, then remove ``{tkey}__*.fits`` from *staging_dir* and delete
    ``cascade_parent / time_key`` after each successful Zarr append (same intent as
    ``cleanup_fixed_fits`` during convert: lower peak disk use for incremental runs).

    Pass ``group_metadata_source="filename"`` to align with
    :func:`run_cascade_per_time_group` when raw OVRO basenames carry ``-image-`` and
    ``_NNNMHz_`` tags so discovery avoids FITS header reads.

    Raw inputs are passed through
    :func:`ovro_lwa_portal.fits_to_zarr_xradio._filter_invalid_beam_files` before the
    cascade runs, so files whose primary header is missing or has a non-positive
    ``BMAJ``/``BMIN`` are dropped and never dewarped. The downstream convert step
    applies the same filter again to dewarped staging output, so any cascade run
    that emits an image without a valid synthesized beam leaves its
    ``(time, frequency)`` slot empty and the eventual Zarr write fills that cell
    with the float ``NaN`` fill value.

    After the beam filter, :func:`_filter_time_groups_without_cascade_reference`
    drops any time group whose files have no ``73MHz`` reference subband: the
    cascade is anchored at 73 MHz, so a group lacking that band cannot be
    cascaded. Those observation times do not appear in the resulting Zarr.

    When ``rebuild`` is False and ``output_dir / zarr_name`` already exists, time
    keys already present in the store are skipped via
    :func:`~ovro_lwa_portal.fits_to_zarr_xradio._filter_completed_time_keys`. An
    interrupted prior run can therefore be resumed by re-invoking with the same
    arguments; only the remaining observation times are dewarped and appended.
    The global LM reference is built **before** the resume filter from the full
    discovery set so the appended slices keep the same pixel grid as the times
    already in the store.
    """
    from ovro_lwa_portal.ingest.core import FITSToZarrConverter, ConversionConfig

    cascade_fn = cascade_fn or import_flow_cascade73mhz()
    staging_dir.mkdir(parents=True, exist_ok=True)
    cascade_parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_dir.mkdir(parents=True, exist_ok=True)

    discovery = IngestDiscoveryConfig(
        freq_bin_hz=discovery_freq_bin_hz,
        group_metadata_source=group_metadata_source,
        time_key_source=time_key_source,
    )
    by_time = discover_time_grouped_fits(
        input_dir, duplicate_resolver=duplicate_resolver, discovery=discovery
    )
    by_time = prepare_ingest_time_groups(
        by_time,
        rebuild=True,
        resume=False,
        require_73mhz=False,
        context="dewarp-convert",
    )
    if not by_time:
        msg = f"No groupable FITS files found in {input_dir}"
        raise FileNotFoundError(msg)

    by_time = prepare_ingest_time_groups(
        by_time,
        rebuild=True,
        resume=False,
        require_73mhz=True,
        context="dewarp-convert",
    )
    if not by_time:
        logger.warning(
            "No time group in %s contains a 73 MHz reference image; nothing to dewarp.",
            input_dir,
        )
        return 0, []

    lm_ref_ds = _load_global_lm_reference_dataset(
        by_time,
        fixed_dir,
        chunk_lm=chunk_lm,
        fix_headers_on_demand=fix_headers_on_demand,
        target_size=target_size,
        group_metadata_source=group_metadata_source,
    ).copy(deep=True)

    out_zarr = output_dir / zarr_name
    by_time = prepare_ingest_time_groups(
        by_time,
        out_zarr=out_zarr,
        rebuild=rebuild,
        resume=resume,
        require_73mhz=False,
        context="dewarp-convert",
    )
    if not by_time:
        logger.info(
            "Nothing to do: every discovered time key is already present in %s. "
            "Pass --rebuild to start over, or --no-resume to reprocess all times.",
            out_zarr,
        )
        _consolidate_zarr_metadata(out_zarr)
        return 0, []

    first_zarr_write = not (out_zarr.exists() and not rebuild)
    time_keys_sorted = sorted(by_time.keys())
    n_staged_total = 0
    total_steps = len(time_keys_sorted)

    for idx, tkey in enumerate(time_keys_sorted):
        remove_staged_files_for_time_key(staging_dir, tkey)
        n_staged_total += run_cascade_for_time_key(
            tkey,
            list(by_time[tkey]),
            cascade_parent,
            staging_dir,
            cascade_fn=cascade_fn,
            cleaned=cleaned,
            qa=qa,
            use_best_pb_model=use_best_pb_model,
            bright_source_flux_qa=bright_source_flux_qa,
            write=write,
            target_size=target_size,
        )
        config = ConversionConfig(
            input_dir=staging_dir,
            output_dir=output_dir,
            zarr_name=zarr_name,
            fixed_dir=fixed_dir,
            chunk_lm=chunk_lm,
            rebuild=first_zarr_write,
            resume=False,
            fix_headers_on_demand=fix_headers_on_demand,
            cleanup_fixed_fits=cleanup_fixed_fits,
            duplicate_resolver=duplicate_resolver,
            discovery_freq_bin_hz=discovery_freq_bin_hz,
            verbose=verbose,
            time_keys_only=(tkey,),
            lm_reference_ds=lm_ref_ds,
            group_metadata_source=group_metadata_source,
            discovery_time_key_source=time_key_source,
            lm_reference_target_size=target_size,
            consolidate_metadata_at_end=False,
        )
        FITSToZarrConverter(config, progress_callback=progress_callback).convert()
        first_zarr_write = False
        remove_staged_files_for_time_key(staging_dir, tkey)
        t_out = cascade_parent / tkey
        if t_out.exists():
            shutil.rmtree(t_out)
        if progress_callback:
            progress_callback(
                "dewarp_convert",
                idx + 1,
                total_steps,
                f"Completed dewarp+Zarr for time {idx + 1}/{total_steps} ({tkey})",
            )

    _consolidate_zarr_metadata(out_zarr)
    return n_staged_total, time_keys_sorted
