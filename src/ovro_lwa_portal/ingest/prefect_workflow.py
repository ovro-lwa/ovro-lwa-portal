"""Optional Prefect workflow orchestration for FITS to Zarr conversion.

This module provides Prefect-based workflow orchestration as an optional
enhancement layer. The core conversion logic remains independent and can
be used without Prefect.

To use this module, install the 'prefect' optional dependency:
    pip install ovro_lwa_portal[prefect]
    # or
    uv pip install ovro_lwa_portal[prefect]
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NoReturn

if TYPE_CHECKING:
    from ovro_lwa_portal.ingest.core import ConversionConfig

__all__ = ["run_conversion_flow", "fits_to_zarr_flow"]


try:
    from prefect import flow, task
    from prefect.logging import get_run_logger

    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False

    def _raise_import_error() -> NoReturn:
        msg = (
            "Prefect is not installed. Install it with:\n"
            "  pip install ovro_lwa_portal[prefect]\n"
            "or\n"
            "  uv pip install ovro_lwa_portal[prefect]"
        )
        raise ImportError(msg)

    # Create dummy decorators for type checking
    def flow(func=None, *dargs, **dkwargs):  # type: ignore[no-untyped-def]
        # Support both @flow and @flow(...)
        def decorator(f):  # type: ignore[no-untyped-def]
            def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
                _raise_import_error()
            return wrapper
        if func is not None and callable(func):
            return decorator(func)
        return decorator

    def task(func=None, *dargs, **dkwargs):  # type: ignore[no-untyped-def]
        # Support both @task and @task(...)
        def decorator(f):  # type: ignore[no-untyped-def]
            def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
                _raise_import_error()
            return wrapper
        if func is not None and callable(func):
            return decorator(func)
        return decorator
if PREFECT_AVAILABLE:
    _DUP_FITS_ERR = "Duplicate FITS files detected"

    def _execute_conversion_should_retry(
        _task: object, _task_run: object, state: object
    ) -> bool:
        """Return False for duplicate-FITS RuntimeErrors: retries will not help."""
        try:
            state.result()  # type: ignore[union-attr]
        except RuntimeError as exc:
            if _DUP_FITS_ERR in str(exc):
                return False
        except Exception:
            pass
        return True

    @task(name="Validate Configuration")
    def validate_config_task(config: ConversionConfig) -> None:
        """Validate conversion configuration as a Prefect task.

        Parameters
        ----------
        config : ConversionConfig
            Configuration to validate.

        Raises
        ------
        FileNotFoundError
            If input directory doesn't exist.
        ValueError
            If parameters are invalid.
        """
        logger = get_run_logger()
        logger.info("Validating configuration")
        config.validate()
        logger.info("Configuration valid")

    @task(name="Prepare Directories")
    def prepare_directories_task(config: ConversionConfig) -> None:
        """Create necessary directories as a Prefect task.

        Parameters
        ----------
        config : ConversionConfig
            Configuration with directory paths.
        """
        logger = get_run_logger()
        logger.info("Creating output directories")
        config.output_dir.mkdir(parents=True, exist_ok=True)
        config.fixed_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {config.output_dir}")
        logger.info(f"Fixed FITS directory: {config.fixed_dir}")

    @task(
        name="Execute Conversion",
        retries=2,
        retry_delay_seconds=60,
        retry_condition_fn=_execute_conversion_should_retry,
    )
    def execute_conversion_task(config: ConversionConfig) -> Path:
        """Execute the FITS to Zarr conversion as a Prefect task.

        Parameters
        ----------
        config : ConversionConfig
            Conversion configuration.

        Returns
        -------
        Path
            Path to the output Zarr store.
        """
        from ovro_lwa_portal.ingest.core import FITSToZarrConverter

        logger = get_run_logger()

        def progress_callback(stage: str, current: int, total: int, message: str) -> None:
            """Log progress via Prefect logger."""
            if total > 0:
                percentage = (current / total) * 100
                logger.info(f"[{stage}] {message} ({percentage:.1f}%)")
            else:
                logger.info(f"[{stage}] {message}")

        logger.info("Starting FITS to Zarr conversion")
        converter = FITSToZarrConverter(config, progress_callback=progress_callback)
        result = converter.convert()
        logger.info(f"Conversion complete: {result}")
        return result

    @flow(name="FITS to Zarr Conversion")
    def fits_to_zarr_flow(
        input_dir: str | Path,
        output_dir: str | Path,
        zarr_name: str = "ovro_lwa_full_lm_only.zarr",
        fixed_dir: str | Path | None = None,
        chunk_lm: int = 1024,
        rebuild: bool = False,
        duplicate_resolver: Callable[[str, float, list[Path]], Path] | None = None,
        verbose: bool = False,
        group_metadata_source: Literal["fits", "filename"] = "fits",
        sky_dtype: Literal["preserve", "float32", "float64"] = "preserve",
    ) -> Path:
        """Prefect flow for FITS to Zarr conversion.

        This flow orchestrates the conversion process with Prefect's workflow
        management capabilities, including automatic retries, logging, and monitoring.

        Parameters
        ----------
        input_dir : str | Path
            Directory containing input FITS files.
        output_dir : str | Path
            Directory where the Zarr store will be written.
        zarr_name : str, optional
            Name of the output Zarr store. Defaults to "ovro_lwa_full_lm_only.zarr".
        fixed_dir : str | Path | None, optional
            Directory for storing fixed FITS files. If None, creates a "fixed_fits"
            subdirectory in output_dir.
        chunk_lm : int, optional
            Chunk size for l and m spatial dimensions. Defaults to 1024.
        rebuild : bool, optional
            If True, overwrite existing Zarr store. Defaults to False.
        duplicate_resolver : Callable[[str, float, list[Path]], Path] | None, optional
            When two FITS files share the same (time, frequency) group, this callback
            chooses which path to use. If omitted and duplicates exist, conversion
            raises ``RuntimeError`` at discovery (see ``ConversionConfig`` in
            ``ovro_lwa_portal.ingest.core``).
        verbose : bool, optional
            Enable verbose logging. Defaults to False.
        group_metadata_source : {"fits", "filename"}, optional
            Forwarded to :class:`~ovro_lwa_portal.ingest.core.ConversionConfig`.
        sky_dtype : {"preserve", "float32", "float64"}, optional
            On-disk ``SKY`` floating dtype policy. ``preserve`` (default) follows
            input FITS ``BITPIX`` (``-32``→float32, ``-64``→float64).

        Returns
        -------
        Path
            Path to the output Zarr store.

        Examples
        --------
        >>> from ovro_lwa_portal.ingest.prefect_workflow import fits_to_zarr_flow
        >>> result = fits_to_zarr_flow(
        ...     input_dir="/data/fits",
        ...     output_dir="/data/output",
        ...     rebuild=False,
        ... )
        """
        from ovro_lwa_portal.ingest.core import ConversionConfig

        logger = get_run_logger()
        logger.info("Starting FITS to Zarr conversion flow")

        # Build configuration
        config = ConversionConfig(
            input_dir=Path(input_dir),
            output_dir=Path(output_dir),
            zarr_name=zarr_name,
            fixed_dir=Path(fixed_dir) if fixed_dir else None,
            chunk_lm=chunk_lm,
            rebuild=rebuild,
            duplicate_resolver=duplicate_resolver,
            verbose=verbose,
            group_metadata_source=group_metadata_source,
            sky_dtype=sky_dtype,
        )

        # Execute tasks in sequence
        validate_config_task(config)
        prepare_directories_task(config)
        result = execute_conversion_task(config)

        logger.info(f"Flow complete. Output: {result}")
        return result

else:
    # Provide stub implementations when Prefect is not available
    def fits_to_zarr_flow(*args, **kwargs) -> Path:  # type: ignore[no-untyped-def]
        _raise_import_error()


def run_conversion_flow(
    input_dir: str | Path,
    output_dir: str | Path,
    zarr_name: str = "ovro_lwa_full_lm_only.zarr",
    fixed_dir: str | Path | None = None,
    chunk_lm: int = 1024,
    rebuild: bool = False,
    duplicate_resolver: Callable[[str, float, list[Path]], Path] | None = None,
    verbose: bool = False,
    group_metadata_source: Literal["fits", "filename"] = "fits",
    sky_dtype: Literal["preserve", "float32", "float64"] = "preserve",
) -> Path:
    """Run the FITS to Zarr conversion using Prefect orchestration.

    This is a convenience wrapper around the Prefect flow that checks for
    Prefect availability and provides helpful error messages.

    Parameters
    ----------
    input_dir : str | Path
        Directory containing input FITS files.
    output_dir : str | Path
        Directory where the Zarr store will be written.
    zarr_name : str, optional
        Name of the output Zarr store.
    fixed_dir : str | Path | None, optional
        Directory for storing fixed FITS files.
    chunk_lm : int, optional
        Chunk size for l and m spatial dimensions.
    rebuild : bool, optional
        If True, overwrite existing Zarr store.
    duplicate_resolver : Callable[[str, float, list[Path]], Path] | None, optional
        Resolves duplicate FITS files in the same (time, frequency) group; if ``None`` and
        duplicates exist, conversion raises at discovery time.
    verbose : bool, optional
        Enable verbose logging.
    sky_dtype : {"preserve", "float32", "float64"}, optional
        On-disk ``SKY`` floating dtype policy. ``preserve`` (default) follows
        input FITS ``BITPIX`` (``-32``→float32, ``-64``→float64).

    Returns
    -------
    Path
        Path to the output Zarr store.

    Raises
    ------
    ImportError
        If Prefect is not installed.
    RuntimeError
        If duplicate FITS are found and ``duplicate_resolver`` is not provided.
    """
    if not PREFECT_AVAILABLE:
        msg = (
            "Prefect is not installed. Install it with:\n"
            "  pip install 'ovro_lwa_portal[prefect]'\n"
            "or use the core conversion API directly:\n"
            "  from ovro_lwa_portal.ingest import FITSToZarrConverter"
        )
        raise ImportError(msg)

    return fits_to_zarr_flow(
        input_dir=input_dir,
        output_dir=output_dir,
        zarr_name=zarr_name,
        fixed_dir=fixed_dir,
        chunk_lm=chunk_lm,
        rebuild=rebuild,
        duplicate_resolver=duplicate_resolver,
        verbose=verbose,
        group_metadata_source=group_metadata_source,
        sky_dtype=sky_dtype,
    )
