"""Command-line interface for FITS to Zarr conversion.

This module provides the CLI entry point for converting OVRO-LWA FITS files
to Zarr format using the Typer framework.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Iterator, Literal, Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Prompt

from ovro_lwa_portal.fits_to_zarr_xradio import (
    _DISCOVERY_FREQ_BIN_HZ,
    fix_fits_headers,
    repair_zarr_store,
    validate_zarr_store,
)
from ovro_lwa_portal.ingest.discovery import (
    IngestDiscoveryConfig,
    discover_time_grouped_fits,
)
from ovro_lwa_portal.ingest.core import ConversionConfig, FITSToZarrConverter
from ovro_lwa_portal.ingest.dewarp_convert import (
    dewarp_and_convert_append_each_time,
    run_cascade_per_time_group,
)
from ovro_lwa_portal.ingest.metadata_audit import audit_directory, print_audit_reports

__all__ = ["main", "app"]


def _cli_group_metadata_source(value: str) -> Literal["fits", "filename"]:
    if value not in ("fits", "filename"):
        raise typer.BadParameter(
            f'expected "fits" or "filename", got {value!r}',
            param_hint="--discovery-metadata-source",
        )
    return value  # type: ignore[return-value]


def _cli_time_key_source(value: str) -> Literal["header", "filename"]:
    if value not in ("header", "filename"):
        raise typer.BadParameter(
            f'expected "header" or "filename", got {value!r}',
            param_hint="--discovery-time-key-source",
        )
    return value  # type: ignore[return-value]


@contextmanager
def suppress_stderr() -> Iterator[None]:
    """Context manager to temporarily suppress stderr output.

    Some native dependencies still write to stderr outside Python logging;
    this keeps non-debug CLI runs quiet during conversion.
    """
    # Save the original stderr file descriptor
    stderr_fd = sys.stderr.fileno()
    old_stderr = os.dup(stderr_fd)

    try:
        # Redirect stderr to /dev/null
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        yield
    finally:
        # Restore original stderr
        os.dup2(old_stderr, stderr_fd)
        os.close(old_stderr)


console = Console()
app = typer.Typer(
    name="ovro-ingest",
    help="Convert OVRO-LWA FITS files to Zarr format",
    add_completion=False,
)


class LogLevel(str, Enum):
    """Supported logging levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


def _configure_logging(level: LogLevel) -> None:
    """Configure logging based on user preference.

    Parameters
    ----------
    level : LogLevel
        Desired logging level.
    """
    log_level_map = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.INFO: logging.INFO,
        LogLevel.WARNING: logging.WARNING,
        LogLevel.ERROR: logging.ERROR,
    }
    logging.basicConfig(
        level=log_level_map[level],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _execute_fits_to_zarr_conversion(
    config: ConversionConfig,
    *,
    log_level: LogLevel,
) -> Path:
    """Run FITS→Zarr with progress UI; raise :class:`typer.Exit` on failure."""
    verbose = log_level == LogLevel.DEBUG

    if log_level != LogLevel.DEBUG:
        logging.getLogger("ovro_lwa_portal.fits_to_zarr_xradio").setLevel(logging.WARNING)
        logging.getLogger("ovro_lwa_portal.ingest.core").setLevel(logging.WARNING)
        logging.getLogger("viperlog").setLevel(logging.ERROR)
        logging.getLogger("astropy").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=Warning, module="astropy")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting...", total=100)
        duplicate_prompt_context = {"active": False}

        def progress_callback(stage: str, current: int, total: int, message: str) -> None:
            if duplicate_prompt_context["active"]:
                return
            if total > 0:
                percentage = (current / total) * 100
                progress.update(task, completed=percentage, description=message)

        def duplicate_resolver(time_key: str, frequency_hz: float, candidates: list[Path]) -> Path:
            duplicate_prompt_context["active"] = True
            progress.stop()
            try:
                console.print(
                    "\n[bold yellow]Duplicate FITS candidates detected[/bold yellow] "
                    f"for time={time_key}, frequency={frequency_hz:.1f} Hz"
                )
                for idx, candidate in enumerate(candidates, start=1):
                    console.print(f"  {idx}. {candidate}")
                choices = [str(i) for i in range(1, len(candidates) + 1)]
                selection = Prompt.ask(
                    "Select which file to use",
                    choices=choices,
                    default="1",
                    console=console,
                )
                return candidates[int(selection) - 1]
            finally:
                progress.start()
                duplicate_prompt_context["active"] = False

        config.duplicate_resolver = duplicate_resolver
        converter = FITSToZarrConverter(config, progress_callback=progress_callback)

        try:
            if log_level != LogLevel.DEBUG:
                with suppress_stderr():
                    result = converter.convert()
            else:
                result = converter.convert()
            progress.update(task, completed=100, description="Conversion complete")
            console.print(f"\n[bold green]✓[/bold green] Successfully created: {result}")
            return result

        except FileNotFoundError as e:
            console.print(f"\n[bold red]✗[/bold red] Error: {e}", style="red")
            console.print(
                "\nNo matching FITS files found. Please check:\n"
                "  • The input directory contains FITS files\n"
                "  • FITS headers contain usable observation time/frequency metadata\n"
                "  • For the default header-based grouping, FITS headers include DATE-OBS and frequency metadata\n"
                "  • If you intend to use filename-based time parsing, enable the explicit filename time-key mode"
            )
            raise typer.Exit(code=1) from e

        except RuntimeError as e:
            console.print(f"\n[bold red]✗[/bold red] Conversion failed: {e}", style="red")
            if "lock" in str(e).lower():
                console.print(
                    "\nAnother process may be writing to the same output location.\n"
                    "Please ensure no other conversion processes are running."
                )
            raise typer.Exit(code=1) from e

        except Exception as e:
            console.print(
                f"\n[bold red]✗[/bold red] Unexpected error: {e}",
                style="red",
            )
            if verbose:
                console.print_exception()
            raise typer.Exit(code=1) from e


@app.command()
def convert(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing input FITS files",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Directory where the Zarr store will be written",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    zarr_name: str = typer.Option(
        "ovro_lwa_full_lm_only.zarr",
        "--zarr-name",
        "-z",
        help="Name of the output Zarr store",
    ),
    fixed_dir: Optional[Path] = typer.Option(
        None,
        "--fixed-dir",
        "-f",
        help="Directory for storing fixed FITS files (default: OUTPUT_DIR/fixed_fits)",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    chunk_lm: int = typer.Option(
        1024,
        "--chunk-lm",
        "-c",
        help="Chunk size for l and m spatial dimensions (0 to disable)",
        min=0,
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        "-r",
        help="Overwrite existing Zarr store instead of appending",
    ),
    no_resume: bool = typer.Option(
        False,
        "--no-resume",
        help=(
            "Reprocess every discovered time step even when the output Zarr already exists "
            "(default: skip times already present unless --rebuild)"
        ),
    ),
    target_size: Optional[int] = typer.Option(
        None,
        "--target-size",
        help=(
            "Reproject the global LM reference onto this square grid size (pixels) before "
            "combining subbands; use the same value as dewarp target_size when applicable"
        ),
        min=1,
    ),
    skip_header_fixing: bool = typer.Option(
        False,
        "--skip-header-fixing",
        "-s",
        help="Skip header fixing (assume headers already fixed with fix-headers command)",
    ),
    cleanup_fixed_fits: bool = typer.Option(
        False,
        "--cleanup-fixed-fits",
        help=(
            "Delete temporary *_fixed.fits files created during on-demand conversion "
            "after each time-step is written (reduces peak disk usage)"
        ),
    ),
    discovery_freq_bin_hz: float = typer.Option(
        _DISCOVERY_FREQ_BIN_HZ,
        "--discovery-freq-bin-hz",
        help=(
            "Treat header frequencies within this bin width (Hz) as one subband when "
            "grouping FITS (default: library default, 23 kHz)"
        ),
        min=1e-6,
    ),
    discovery_metadata_source: str = typer.Option(
        "fits",
        "--discovery-metadata-source",
        help=(
            'How to infer observation time and subband when grouping: "fits" reads FITS '
            'headers (with filename fallbacks; default). "filename" uses only basename '
            "``-image-YYYYMMDD_HHMMSS`` and ``_NNNMHz_`` tags, avoiding FITS header reads "
            "during discovery and frequency ordering."
        ),
    ),
    discovery_time_key_source: str = typer.Option(
        "filename",
        "--discovery-time-key-source",
        help=(
            'When --discovery-metadata-source is "fits", how to choose the observation time '
            'key: "filename" prefers ``-image-YYYYMMDD_HHMMSS`` (default); "header" uses '
            "DATE-OBS only."
        ),
    ),
    log_level: LogLevel = typer.Option(
        LogLevel.INFO,
        "--log-level",
        "-l",
        help="Logging verbosity level",
        case_sensitive=False,
    ),
) -> None:
    """Convert OVRO-LWA FITS files to a single Zarr store.

    This command processes FITS image files from the specified INPUT_DIR and
    combines them into an optimized Zarr store at OUTPUT_DIR. Files are grouped
    by observation time and frequency, using FITS headers first with filename
    parsing as a fallback when headers are incomplete.

    If inputs use different LM pixel shapes, the largest grid in the selection is
    chosen as the reference and smaller images are interpolated onto it so the
    output has a single consistent sky grid.

    \b
    Examples:
        # Basic conversion (fixes headers on-demand)
        ovro-ingest convert /data/fits /data/output

        # Rebuild existing store with verbose logging
        ovro-ingest convert /data/fits /data/output --rebuild --log-level debug

        # Custom Zarr name and chunk size
        ovro-ingest convert /data/fits /data/output \\
            --zarr-name my_data.zarr --chunk-lm 2048

        # Two-step workflow: fix headers first, then convert
        ovro-ingest fix-headers /data/fits /data/fixed_fits
        ovro-ingest convert /data/fits /data/output \\
            --fixed-dir /data/fixed_fits --skip-header-fixing
    """
    _configure_logging(log_level)

    verbose = log_level == LogLevel.DEBUG

    group_metadata_source = _cli_group_metadata_source(discovery_metadata_source)
    time_key_source = _cli_time_key_source(discovery_time_key_source)

    # Build configuration
    config = ConversionConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        zarr_name=zarr_name,
        fixed_dir=fixed_dir,
        chunk_lm=chunk_lm,
        rebuild=rebuild,
        resume=not no_resume,
        fix_headers_on_demand=not skip_header_fixing,  # Invert the flag
        cleanup_fixed_fits=cleanup_fixed_fits,
        duplicate_resolver=None,
        discovery_freq_bin_hz=discovery_freq_bin_hz,
        verbose=verbose,
        group_metadata_source=group_metadata_source,
        discovery_time_key_source=time_key_source,
        lm_reference_target_size=target_size,
    )

    # Display configuration
    console.print("\n[bold cyan]OVRO-LWA FITS → Zarr Conversion[/bold cyan]")
    console.print(f"  Input directory:  {input_dir}")
    console.print(f"  Output directory: {output_dir}")
    console.print(f"  Zarr store name:  {zarr_name}")
    console.print(f"  Fixed FITS dir:   {config.fixed_dir}")
    console.print(f"  Chunk size (l,m): {chunk_lm}")
    console.print(f"  Mode:             {'REBUILD' if rebuild else 'APPEND'}")
    console.print(f"  Resume mode:      {'OFF' if no_resume else 'ON'}")
    if target_size is not None:
        console.print(f"  Target size (px): {target_size}")
    console.print(f"  Fix headers:      {'ON-DEMAND' if not skip_header_fixing else 'SKIP (pre-fixed)'}")
    console.print(f"  Cleanup fixed:    {'YES' if cleanup_fixed_fits else 'NO'}")
    console.print(f"  Discovery meta:   {group_metadata_source}")
    console.print(f"  Time key source:  {time_key_source}")
    console.print(f"  Log level:        {log_level.value.upper()}\n")

    _execute_fits_to_zarr_conversion(config, log_level=log_level)


@app.command()
def dewarp_convert(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing raw FITS files (same discovery rules as convert)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    output_dir: Path = typer.Argument(
        ...,
        help="Directory where Zarr and intermediate cascade/staging dirs are written",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    zarr_name: str = typer.Option(
        "ovro_lwa_full_lm_only.zarr",
        "--zarr-name",
        "-z",
        help="Name of the output Zarr store",
    ),
    fixed_dir: Optional[Path] = typer.Option(
        None,
        "--fixed-dir",
        "-f",
        help="Directory for storing fixed FITS files (default: OUTPUT_DIR/fixed_fits)",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    chunk_lm: int = typer.Option(
        1024,
        "--chunk-lm",
        "-c",
        help="Chunk size for l and m spatial dimensions (0 to disable)",
        min=0,
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        "-r",
        help="Overwrite existing Zarr store instead of appending",
    ),
    no_resume: bool = typer.Option(
        False,
        "--no-resume",
        help=(
            "Reprocess every discovered time step even when the output Zarr already exists "
            "(default: skip times already present unless --rebuild)"
        ),
    ),
    skip_header_fixing: bool = typer.Option(
        False,
        "--skip-header-fixing",
        "-s",
        help="Skip header fixing (assume headers already fixed with fix-headers command)",
    ),
    cleanup_fixed_fits: bool = typer.Option(
        False,
        "--cleanup-fixed-fits",
        help=(
            "Delete temporary *_fixed.fits files created during on-demand conversion "
            "after each time-step is written (reduces peak disk usage)"
        ),
    ),
    append_after_each_time: bool = typer.Option(
        False,
        "--append-after-each-time",
        help=(
            "Dewarp each observation time, append that time slice to Zarr, then remove that "
            "time's staged FITS and cascade output directory before continuing (lowers peak "
            "dewarp staging disk use)"
        ),
    ),
    discovery_freq_bin_hz: float = typer.Option(
        _DISCOVERY_FREQ_BIN_HZ,
        "--discovery-freq-bin-hz",
        help=(
            "Treat header frequencies within this bin width (Hz) as one subband when "
            "grouping FITS (default: library default, 23 kHz)"
        ),
        min=1e-6,
    ),
    discovery_metadata_source: str = typer.Option(
        "fits",
        "--discovery-metadata-source",
        help=(
            'How to infer observation time and subband when grouping raw FITS: "fits" reads '
            'FITS headers (with filename fallbacks; default). "filename" uses only basename '
            "``-image-YYYYMMDD_HHMMSS`` and ``_NNNMHz_`` tags, avoiding FITS header reads "
            "during discovery and frequency ordering."
        ),
    ),
    discovery_time_key_source: str = typer.Option(
        "filename",
        "--discovery-time-key-source",
        help=(
            'When --discovery-metadata-source is "fits", how to choose the observation time '
            'key: "filename" prefers ``-image-YYYYMMDD_HHMMSS`` (default); "header" uses '
            "DATE-OBS only."
        ),
    ),
    cascade_parent: Optional[Path] = typer.Option(
        None,
        "--cascade-parent",
        help=(
            "Directory under which per-time cascade outputs are written "
            "(default: OUTPUT_DIR/cascade73MHz); from "
            "``image_plane_correction.flow.flow_cascade73MHz``"
        ),
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    staging_dir: Optional[Path] = typer.Option(
        None,
        "--staging-dir",
        help=(
            "Flat directory of dewarped FITS passed to Zarr conversion "
            "(default: OUTPUT_DIR/dewarped_fits_staging)"
        ),
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    cleaned: bool = typer.Option(
        True,
        "--cleaned/--no-cleaned",
        help="Forwarded to image_plane_correction.flow.flow_cascade73MHz",
    ),
    qa: bool = typer.Option(
        True,
        "--qa/--no-qa",
        help="Forwarded to image_plane_correction.flow.flow_cascade73MHz",
    ),
    use_best_pb_model: bool = typer.Option(
        True,
        "--use-best-pb-model/--no-use-best-pb-model",
        help="Forwarded to image_plane_correction.flow.flow_cascade73MHz",
    ),
    bright_source_flux_qa: bool = typer.Option(
        True,
        "--bright-source-flux-qa/--no-bright-source-flux-qa",
        help="Forwarded to image_plane_correction.flow.flow_cascade73MHz",
    ),
    write: bool = typer.Option(
        True,
        "--write/--no-write",
        help=(
            "Forwarded to image_plane_correction.flow.flow_cascade73MHz "
            "(keep True for file-based handoff)"
        ),
    ),
    target_size: Optional[int] = typer.Option(
        None,
        "--target-size",
        help=(
            "Output raster size in pixels (square side length), forwarded to "
            "image_plane_correction.flow.flow_cascade73MHz / calcflow; omit for library default"
        ),
        min=1,
    ),
    log_level: LogLevel = typer.Option(
        LogLevel.INFO,
        "--log-level",
        "-l",
        help="Logging verbosity level",
        case_sensitive=False,
    ),
) -> None:
    """Dewarp each time's subbands, then run Zarr convert.

    FITS in INPUT_DIR are grouped by observation time (same logic as ``convert``).
    For each time key, all subband files in that group are passed to
    ``image_plane_correction.flow.flow_cascade73MHz`` with a per-time ``outroot``
    under ``--cascade-parent``.
    Resulting ``*.fits`` are staged into ``--staging-dir``, then the usual
    ``ovro-ingest convert`` pipeline runs on that staging directory (unless
    ``--append-after-each-time`` is set; then Zarr is updated after each time group,
    and that time's staged FITS plus per-time cascade directory are removed before
    continuing).

    Requires the ``image_plane_correction`` package (submodule ``flow``).

    \b
    Example:
        ovro-ingest dewarp-convert /data/raw_fits /data/output --rebuild

    \b
    Incremental Zarr (lower peak staging disk; staging cleaned after each time):
        ovro-ingest dewarp-convert /data/raw /data/out --append-after-each-time \\
            --cleanup-fixed-fits
    """
    _configure_logging(log_level)
    verbose = log_level == LogLevel.DEBUG
    group_metadata_source = _cli_group_metadata_source(discovery_metadata_source)
    time_key_source = _cli_time_key_source(discovery_time_key_source)
    resume = not no_resume

    output_dir.mkdir(parents=True, exist_ok=True)
    cascade_root = cascade_parent or (output_dir / "cascade73MHz")
    staging = staging_dir or (output_dir / "dewarped_fits_staging")

    console.print("\n[bold cyan]OVRO-LWA dewarp-convert[/bold cyan]")
    console.print(f"  Input directory:     {input_dir}")
    console.print(f"  Output directory:    {output_dir}")
    console.print(f"  Cascade parent:      {cascade_root}")
    console.print(f"  Staging (→ Zarr):    {staging}")
    console.print(f"  Zarr store name:     {zarr_name}")
    if target_size is not None:
        console.print(f"  Target size (px):  {target_size}")
    if append_after_each_time:
        console.print("  Append after each time: YES (staging + per-time cascade cleaned after each append)")
    console.print(f"  Discovery meta:      {group_metadata_source}")
    console.print(f"  Time key source:     {time_key_source}")
    console.print(f"  Resume mode:         {'OFF' if no_resume else 'ON'}")
    console.print(f"  Log level:           {log_level.value.upper()}\n")

    fixed_resolved = fixed_dir or (output_dir / "fixed_fits")

    try:
        if append_after_each_time:
            n_staged, time_keys = dewarp_and_convert_append_each_time(
                input_dir,
                output_dir,
                cascade_root,
                staging,
                fixed_resolved,
                zarr_name=zarr_name,
                chunk_lm=chunk_lm,
                rebuild=rebuild,
                fix_headers_on_demand=not skip_header_fixing,
                cleanup_fixed_fits=cleanup_fixed_fits,
                discovery_freq_bin_hz=discovery_freq_bin_hz,
                duplicate_resolver=None,
                cleaned=cleaned,
                qa=qa,
                use_best_pb_model=use_best_pb_model,
                bright_source_flux_qa=bright_source_flux_qa,
                write=write,
                target_size=target_size,
                cascade_fn=None,
                verbose=verbose,
                progress_callback=None,
                group_metadata_source=group_metadata_source,
                time_key_source=time_key_source,
                resume=resume,
            )
        else:
            n_staged, time_keys, lm_ref_ds = run_cascade_per_time_group(
                input_dir,
                cascade_root,
                staging,
                discovery_freq_bin_hz=discovery_freq_bin_hz,
                duplicate_resolver=None,
                cleaned=cleaned,
                qa=qa,
                use_best_pb_model=use_best_pb_model,
                bright_source_flux_qa=bright_source_flux_qa,
                write=write,
                target_size=target_size,
                group_metadata_source=group_metadata_source,
                time_key_source=time_key_source,
                out_zarr=output_dir / zarr_name,
                rebuild=rebuild,
                resume=resume,
                fixed_dir=fixed_resolved,
                chunk_lm=chunk_lm,
                fix_headers_on_demand=not skip_header_fixing,
            )
    except ImportError as e:
        console.print(f"\n[bold red]✗[/bold red] {e}", style="red")
        raise typer.Exit(code=1) from e
    except FileNotFoundError as e:
        console.print(f"\n[bold red]✗[/bold red] {e}", style="red")
        raise typer.Exit(code=1) from e
    except RuntimeError as e:
        console.print(f"\n[bold red]✗[/bold red] Dewarp or Zarr step failed: {e}", style="red")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1) from e

    if append_after_each_time:
        if not time_keys:
            console.print(
                "\n[bold green]✓[/bold green] Nothing to do: every discovered time key "
                f"is already present in {output_dir / zarr_name}.\n"
                "Pass [bold]--rebuild[/bold] to overwrite the existing store.\n"
            )
            return
        console.print(
            f"\n[bold green]✓[/bold green] Dewarped and appended Zarr for {len(time_keys)} "
            f"time step(s); staged up to {n_staged} FITS per batch.\n"
        )
        console.print(
            f"[bold green]✓[/bold green] Zarr store: {output_dir / zarr_name}\n"
        )
        return

    if not time_keys:
        console.print(
            "\n[bold green]✓[/bold green] Nothing to do: every discovered time key "
            f"is already present in {output_dir / zarr_name}.\n"
            "Pass [bold]--rebuild[/bold] to overwrite the existing store.\n"
        )
        return

    console.print(
        f"[bold green]✓[/bold green] Staged {n_staged} FITS file(s) "
        f"from {len(time_keys)} time step(s) for conversion.\n"
    )

    out_zarr_path = output_dir / zarr_name
    convert_rebuild = rebuild and not out_zarr_path.exists()

    config = ConversionConfig(
        input_dir=staging,
        output_dir=output_dir,
        zarr_name=zarr_name,
        fixed_dir=fixed_dir,
        chunk_lm=chunk_lm,
        rebuild=convert_rebuild,
        resume=resume,
        fix_headers_on_demand=not skip_header_fixing,
        cleanup_fixed_fits=cleanup_fixed_fits,
        duplicate_resolver=None,
        discovery_freq_bin_hz=discovery_freq_bin_hz,
        verbose=verbose,
        group_metadata_source=group_metadata_source,
        discovery_time_key_source=time_key_source,
        lm_reference_ds=lm_ref_ds,
        lm_reference_target_size=target_size,
    )

    console.print("[bold cyan]OVRO-LWA FITS → Zarr Conversion[/bold cyan]")
    console.print(f"  Input directory:  {staging}")
    console.print(f"  Output directory: {output_dir}")
    console.print(f"  Zarr store name:  {zarr_name}")
    console.print(f"  Fixed FITS dir:   {config.fixed_dir}")
    console.print(f"  Chunk size (l,m): {chunk_lm}")
    console.print(f"  Mode:             {'REBUILD' if convert_rebuild else 'APPEND'}")
    console.print(f"  Resume mode:      {'OFF' if no_resume else 'ON'}")
    console.print(f"  Fix headers:      {'ON-DEMAND' if not skip_header_fixing else 'SKIP (pre-fixed)'}")
    console.print(f"  Cleanup fixed:    {'YES' if cleanup_fixed_fits else 'NO'}")
    console.print(f"  Discovery meta:   {group_metadata_source}")
    console.print(f"  Time key source:  {time_key_source}")
    console.print(f"  Log level:        {log_level.value.upper()}\n")

    _execute_fits_to_zarr_conversion(config, log_level=log_level)


@app.command("audit-metadata")
def audit_metadata(
    input_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Directory containing per-subband FITS (raw or dewarped staging layout)",
    ),
    time_key: Optional[list[str]] = typer.Option(
        None,
        "--time-key",
        "-t",
        help="Observation time key YYYYMMDD_HHMMSS (repeatable). Default: list keys only.",
    ),
    all_times: bool = typer.Option(
        False,
        "--all",
        help="Audit every discovered time group (can be slow for large directories)",
    ),
    staging_dir: Optional[Path] = typer.Option(
        None,
        "--staging-dir",
        help="Also audit ``{time_key}__*.fits`` dewarped staging files under this directory",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    fixed_dir: Optional[Path] = typer.Option(
        None,
        "--fixed-dir",
        help="Directory for ``*_fixed.fits`` when using --probe-combine",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    probe_combine: bool = typer.Option(
        False,
        "--probe-combine",
        help="Run one in-memory combine per group and report velocity/time dimensions",
    ),
    warn_time_delta_s: float = typer.Option(
        1.0,
        "--warn-time-delta-s",
        help="Warn when filename vs header time offset exceeds this many seconds",
        min=0.0,
    ),
    discovery_freq_bin_hz: float = typer.Option(
        _DISCOVERY_FREQ_BIN_HZ,
        "--discovery-freq-bin-hz",
        help="Frequency bin width (Hz) used when discovering time groups",
        min=1e-6,
    ),
    discovery_metadata_source: str = typer.Option(
        "fits",
        "--discovery-metadata-source",
        help='Discovery mode: "fits" (headers + filename fallbacks) or "filename"',
    ),
    discovery_time_key_source: str = typer.Option(
        "filename",
        "--discovery-time-key-source",
        help=(
            'When --discovery-metadata-source is "fits", observation time key from '
            '"filename" (default) or "header" (DATE-OBS only)'
        ),
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Exit with code 1 if any group has subband metadata issues",
    ),
    log_level: LogLevel = typer.Option(
        LogLevel.WARNING,
        "--log-level",
        "-l",
        help="Logging verbosity level",
        case_sensitive=False,
    ),
) -> None:
    """Audit FITS header metadata consistency across subbands per observation time.

    Checks that time keywords (``DATE-OBS``, ``MJD-OBS``, etc.) match across subbands,
    reports filename vs header time-key offsets, and summarizes spectral and image
    shape metadata. Use before long ``dewarp-convert`` runs to catch grouping issues.

    \b
    Examples:
        # List discovered time keys
        ovro-ingest audit-metadata /data/fits

        # Audit one observation (raw + optional staging)
        ovro-ingest audit-metadata /lustre/claw/symlinks5 -t 20250106_051855 \\
            --staging-dir /fast/claw/dewarped_fits_staging --probe-combine \\
            --fixed-dir /fast/claw

        # Audit all groups (verbose)
        ovro-ingest audit-metadata /data/fits --all --strict
    """
    _configure_logging(log_level)
    group_metadata_source = _cli_group_metadata_source(discovery_metadata_source)
    time_key_source = _cli_time_key_source(discovery_time_key_source)
    discovery = IngestDiscoveryConfig(
        freq_bin_hz=discovery_freq_bin_hz,
        group_metadata_source=group_metadata_source,
        time_key_source=time_key_source,
    )

    input_dir = Path(input_dir)
    by_time = discover_time_grouped_fits(input_dir, discovery=discovery)
    by_time_keys = sorted(by_time.keys())

    if not by_time_keys:
        console.print(f"[bold red]✗[/bold red] No FITS files found in {input_dir}", style="red")
        raise typer.Exit(code=1)

    if time_key is None and not all_times:
        console.print(f"\n[bold cyan]Discovered {len(by_time_keys)} time group(s) in {input_dir}[/bold cyan]")
        for k in by_time_keys:
            console.print(f"  {k}")
        console.print(
            "\nRe-run with [bold]--time-key YYYYMMDD_HHMMSS[/bold] or [bold]--all[/bold] to audit.\n"
        )
        return

    keys = by_time_keys if all_times else list(time_key or [])
    if not keys:
        console.print("[bold red]✗[/bold red] No time keys selected.", style="red")
        raise typer.Exit(code=1)

    console.print("\n[bold cyan]OVRO-LWA metadata audit[/bold cyan]")
    console.print(f"  Input directory:  {input_dir}")
    if staging_dir is not None:
        console.print(f"  Staging directory: {staging_dir}")
    console.print(f"  Time groups:      {len(keys)}")
    console.print(f"  Probe combine:    {'yes' if probe_combine else 'no'}\n")

    try:
        reports = audit_directory(
            input_dir,
            keys,
            staging_dir=staging_dir,
            group_metadata_source=group_metadata_source,
            time_key_source=time_key_source,
            discovery_freq_bin_hz=discovery_freq_bin_hz,
            probe_combine=probe_combine,
            fixed_dir=fixed_dir,
            warn_filename_header_delta_s=warn_time_delta_s,
        )
    except (FileNotFoundError, KeyError) as exc:
        console.print(f"[bold red]✗[/bold red] {exc}", style="red")
        raise typer.Exit(code=1) from exc

    print_audit_reports(reports, console=console)

    n_issues = sum(1 for r in reports if r.has_issues)
    if n_issues:
        console.print(
            f"\n[yellow]{n_issues} report(s) with subband metadata issues.[/yellow]\n"
        )
        if strict:
            raise typer.Exit(code=1)
    else:
        console.print("\n[bold green]✓[/bold green] No subband metadata issues detected.\n")


@app.command()
def fix_headers(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing input FITS files",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    fixed_dir: Path = typer.Argument(
        ...,
        help="Directory where fixed FITS files will be written",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--overwrite",
        help="Skip files that already have fixed versions",
    ),
    log_level: LogLevel = typer.Option(
        LogLevel.INFO,
        "--log-level",
        "-l",
        help="Logging verbosity level",
        case_sensitive=False,
    ),
) -> None:
    """Fix FITS headers ahead of time before conversion.

    This command processes FITS files to ensure they have the necessary headers
    for xradio conversion. It can be run as a separate step before the conversion
    to separate header fixing from the actual Zarr conversion process.

    \b
    Examples:
        # Fix headers for all FITS files
        ovro-ingest fix-headers /data/fits /data/fixed_fits

        # Overwrite existing fixed files
        ovro-ingest fix-headers /data/fits /data/fixed_fits --overwrite

        # Then convert using pre-fixed headers
        ovro-ingest convert /data/fits /data/output \\
            --fixed-dir /data/fixed_fits --skip-header-fixing
    """
    _configure_logging(log_level)

    console.print("\n[bold cyan]OVRO-LWA FITS Header Fixing[/bold cyan]")
    console.print(f"  Input directory:  {input_dir}")
    console.print(f"  Fixed FITS dir:   {fixed_dir}")
    console.print(f"  Skip existing:    {skip_existing}\n")

    # Discover all FITS files
    input_files = sorted(input_dir.glob("*.fits"))
    if not input_files:
        console.print(
            f"[bold red]✗[/bold red] No FITS files found in {input_dir}",
            style="red",
        )
        raise typer.Exit(code=1)

    console.print(f"Found {len(input_files)} FITS file(s)")

    # Temporarily suppress INFO-level logging during progress display
    # (unless user explicitly requested verbose output)
    if log_level != LogLevel.DEBUG:
        # Suppress our own loggers
        logging.getLogger("ovro_lwa_portal.fits_to_zarr_xradio").setLevel(logging.WARNING)

        # Suppress external library loggers
        logging.getLogger("viperlog").setLevel(logging.ERROR)  # xradio logging
        logging.getLogger("astropy").setLevel(logging.ERROR)   # astropy logging

        # Suppress astropy warnings
        warnings.filterwarnings("ignore", category=Warning, module="astropy")

    # Create progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Fixing headers...", total=len(input_files))

        try:
            # Process files one at a time to update progress
            fixed_dir.mkdir(parents=True, exist_ok=True)
            fixed_paths = []

            for i, f in enumerate(sorted(input_files, key=lambda p: p.name)):
                # Update progress with current file
                progress.update(
                    task,
                    completed=i,
                    description=f"Fixing {f.name}..."
                )

                # Fix this single file (suppress native stderr unless in debug mode)
                if log_level != LogLevel.DEBUG:
                    with suppress_stderr():
                        result = fix_fits_headers([f], fixed_dir, skip_existing=skip_existing)
                else:
                    result = fix_fits_headers([f], fixed_dir, skip_existing=skip_existing)
                fixed_paths.extend(result)

            progress.update(task, completed=len(input_files), description="Complete")
            console.print(
                f"\n[bold green]✓[/bold green] Successfully fixed {len(fixed_paths)} file(s)"
            )
            console.print(f"Fixed files written to: {fixed_dir}")

        except Exception as e:
            console.print(
                f"\n[bold red]✗[/bold red] Header fixing failed: {e}",
                style="red",
            )
            if log_level == LogLevel.DEBUG:
                console.print_exception()
            raise typer.Exit(code=1) from e


@app.command()
def validate(
    zarr_path: Path = typer.Argument(
        ...,
        help="Path to an existing Zarr store to validate",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
) -> None:
    """Validate time-axis consistency of an existing Zarr store."""
    try:
        report = validate_zarr_store(zarr_path)
    except Exception as e:
        console.print(f"[bold red]✗[/bold red] Validation failed: {e}", style="red")
        raise typer.Exit(code=1) from e

    console.print("\n[bold cyan]OVRO-LWA Zarr Validation[/bold cyan]")
    console.print(f"  Store: {report['store']}")
    buckets = report["time_length_buckets"]
    for tlen, names in buckets.items():
        console.print(f"  time={tlen}: {len(names)} array(s)")
    if report["consistent"]:
        console.print("\n[bold green]✓[/bold green] Store is time-axis consistent")
        return

    console.print(f"\n[bold red]✗[/bold red] {report['message']}", style="red")
    raise typer.Exit(code=1)


@app.command()
def repair(
    zarr_path: Path = typer.Argument(
        ...,
        help="Path to an existing Zarr store to repair",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    fits_dir: Optional[Path] = typer.Option(
        None,
        "--fits-dir",
        help="Optional FITS directory for rewriting wcs_header_str rows from source headers",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    backup_suffix: str = typer.Option(
        ".backup-before-repair",
        "--backup-suffix",
        help="Suffix to use when creating a backup copy before repair",
    ),
) -> None:
    """Repair interrupted-append time-axis inconsistencies in a Zarr store."""
    console.print("\n[bold cyan]OVRO-LWA Zarr Repair[/bold cyan]")
    console.print(f"  Store:         {zarr_path}")
    console.print(f"  Backup suffix: {backup_suffix}")
    if fits_dir:
        console.print(f"  FITS dir:      {fits_dir}")

    try:
        result = repair_zarr_store(zarr_path, fits_dir=fits_dir, backup_suffix=backup_suffix)
    except Exception as e:
        console.print(f"\n[bold red]✗[/bold red] Repair failed: {e}", style="red")
        raise typer.Exit(code=1) from e

    console.print("\n[bold green]✓[/bold green] Repair completed")
    console.print(f"  Backup:            {result['backup']}")
    console.print(f"  Repaired time len: {result['repaired_len']}")
    console.print(f"  Truncated arrays:  {len(result['truncated_arrays'])}")
    console.print(f"  Rewritten WCS rows:{result['rewritten_wcs_rows']}")
    if not result["post"]["consistent"]:
        console.print("\n[bold red]✗[/bold red] Store remains inconsistent after repair", style="red")
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Display version information."""
    try:
        from ovro_lwa_portal.version import __version__
    except ImportError:
        __version__ = "unknown"

    console.print(f"ovro-ingest version {__version__}")
    console.print("Part of the ovro_lwa_portal package")


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    version_flag: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
        is_eager=True,
    ),
) -> None:
    """OVRO-LWA FITS to Zarr conversion tool."""
    if version_flag:
        version()
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        console.print(
            "[yellow]No command specified. Use --help to see available commands.[/yellow]"
        )
        raise typer.Exit()


def main() -> None:
    """Main entry point for the CLI application."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Conversion interrupted by user[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
