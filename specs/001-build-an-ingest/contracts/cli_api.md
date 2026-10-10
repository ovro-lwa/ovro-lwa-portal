# API Contract: CLI Interface

**Module**: `ovro_lwa_portal.ingest.cli` **Purpose**: Typer-based command-line
interface **Requirements**: FR-001, FR-002, FR-003, FR-004, FR-026, FR-028,
FR-029, FR-030, FR-031, FR-032

## CLI Application

### Entry Point

**Command**: `ovro-lwa-ingest`

**Installation**: Registered in `pyproject.toml` as:

```toml
[project.scripts]
ovro-lwa-ingest = "ovro_lwa_portal.ingest.cli:app"
```

---

### Main Command

**Signature**:

```python
import typer
from pathlib import Path
from typing import Optional

app = typer.Typer(
    name="ovro-lwa-ingest",
    help="OVRO-LWA FITS to Zarr ingestion pipeline",
    add_completion=True,
)

@app.command()
def convert(
    input_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory containing OVRO-LWA FITS files",
    ),
    output_dir: Path = typer.Argument(
        ...,
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Directory for output Zarr store",
    ),
    zarr_name: str = typer.Option(
        "ovro_lwa_full_lm_only.zarr",
        "--zarr-name",
        "-n",
        help="Name of output Zarr store",
    ),
    fixed_dir: Path = typer.Option(
        Path("fixed_fits"),
        "--fixed-dir",
        "-f",
        help="Directory for corrected FITS files",
    ),
    chunk_lm: int = typer.Option(
        1024,
        "--chunk-lm",
        "-c",
        min=1,
        help="Chunk size for L/M spatial dimensions",
    ),
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        "-r",
        help="Rebuild Zarr store from scratch (overwrite existing)",
    ),
    use_prefect: bool = typer.Option(
        False,
        "--use-prefect",
        "-p",
        help="Use Prefect orchestration (if installed)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging (detailed per-file processing)",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress all output except errors",
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        "--log-file",
        "-l",
        help="Write logs to file (in addition to console)",
    ),
) -> None:
    """
    Convert OVRO-LWA FITS files to cloud-optimized Zarr format.

    Discovers FITS files in INPUT_DIR, groups by observation time,
    applies header corrections, and writes to a single Zarr store
    in OUTPUT_DIR.

    Examples:

        # Basic conversion
        ovro-lwa-ingest /data/fits /data/zarr

        # With custom Zarr name and rebuild
        ovro-lwa-ingest /data/fits /data/zarr -n my_data.zarr --rebuild

        # Verbose logging to file
        ovro-lwa-ingest /data/fits /data/zarr -v -l conversion.log

        # Use Prefect orchestration
        ovro-lwa-ingest /data/fits /data/zarr --use-prefect
    """
    ...
```

**Contract**:

- MUST validate required arguments (FR-002, FR-003)
- MUST provide sensible defaults (FR-004)
- MUST support rebuild flag (FR-016)
- MUST support Prefect toggle (FR-022)
- MUST accept `--sky-dtype {preserve,float32,float64}` (default `preserve`)
  and print `SKY dtype mode:` in the pre-run summary
- MUST configure logging based on verbose/quiet flags (FR-028, FR-029)
- MUST display help with `--help` flag (FR-031)
- MUST handle keyboard interrupts gracefully (SIGINT)
- MUST check for incomplete state and prompt user for resume/rebuild (FR-027,
  NFR-002)
- MUST show progress indicators for long operations (NFR-001)
- MUST display summary statistics on completion (FR-024)

---

### Version Command

**Signature**:

```python
@app.command()
def version() -> None:
    """Display version information."""
    from ovro_lwa_portal import __version__
    typer.echo(f"ovro-lwa-ingest version {__version__}")
```

**Contract**:

- MUST display version from package (FR-032)
- MUST use `ovro_lwa_portal.__version__` or `ovro_lwa_portal.version.version`

---

## CLI Behavior

### Resume/Rebuild Prompt

**Scenario**: Incomplete pipeline state detected

**Behavior**:

```python
from rich.prompt import Confirm

if zarr_store.has_incomplete_state():
    state = PipelineState.load(zarr_store.state_file_path)
    typer.echo(f"Incomplete conversion detected for {zarr_store.path}")
    typer.echo(f"Last processed: {state.current_time_step}")
    typer.echo(f"Completed: {len(state.completed_time_steps)} time steps")

    resume = Confirm.ask("Resume from last checkpoint?", default=True)

    if resume:
        typer.echo("Resuming conversion...")
        # Continue from state.completed_time_steps
    else:
        rebuild_confirm = Confirm.ask(
            "Rebuild from scratch (delete existing data)?",
            default=False
        )
        if rebuild_confirm:
            typer.echo("Rebuilding from scratch...")
            # Clear state and start over
        else:
            raise typer.Abort("Conversion cancelled by user")
```

**Contract**:

- MUST prompt user interactively (NFR-002)
- MUST show current state information
- MUST offer resume or rebuild options
- MUST handle user cancellation gracefully

---

### Progress Indicators

**Requirements**: NFR-001 (user-friendly output with progress indicators)

**Implementation**:

```python
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
) as progress:
    task = progress.add_task(
        "Converting FITS files...",
        total=len(time_step_groups)
    )

    for time_step in time_step_groups:
        progress.update(
            task,
            description=f"Processing {time_step.time_step_key}...",
            advance=1
        )
        # Perform conversion
```

**Contract**:

- MUST show progress for operations >5 seconds
- MUST display current time step being processed (FR-023)
- MUST update progress incrementally
- MUST handle terminal resize gracefully

---

### Error Messages

**Requirements**: NFR-005 (clear for non-Python astronomers)

**Examples**:

```python
# Concurrent access detected
typer.secho(
    "ERROR: Another process is currently writing to this Zarr store.",
    fg="red", err=True
)
typer.echo(
    f"Lock held by process: {lock_holder_pid}\n"
    f"Lock file: {zarr_store.lock_file_path}\n\n"
    "Suggested actions:\n"
    "  1. Wait for the other process to complete\n"
    "  2. If process crashed, manually remove the lock file:\n"
    f"     rm {zarr_store.lock_file_path}"
)

# Grid mismatch
typer.secho(
    "ERROR: Spatial grid mismatch detected.",
    fg="red", err=True
)
typer.echo(
    f"Expected grid shape: {expected_shape}\n"
    f"Got grid shape: {actual_shape}\n"
    f"Problematic file: {problematic_file}\n\n"
    "This usually indicates FITS files from different observations.\n"
    "Ensure all files in the input directory belong to the same dataset."
)
```

**Contract**:

- MUST include actionable suggested fixes (NFR-005)
- MUST use color coding (red for errors, yellow for warnings)
- MUST show relevant context (files, paths, PIDs)
- MUST avoid Python jargon and stack traces for user-facing errors

---

## Testing Contract

CLI tests MUST use subprocess invocation to test the actual entry point:

```python
import subprocess

def test_cli_help():
    """Test --help flag displays usage."""
    result = subprocess.run(
        ["ovro-lwa-ingest", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "OVRO-LWA FITS to Zarr" in result.stdout

def test_cli_version():
    """Test version command."""
    result = subprocess.run(
        ["ovro-lwa-ingest", "version"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "version" in result.stdout.lower()
```

**Contract**:

- MUST test CLI as subprocess (real entry point)
- MUST verify exit codes (0=success, non-zero=error)
- MUST capture stdout/stderr
- MUST test all flags and options
- MUST test error scenarios (missing args, invalid paths)
