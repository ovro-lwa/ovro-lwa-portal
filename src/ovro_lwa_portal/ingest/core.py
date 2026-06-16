"""Core FITS to Zarr conversion module.

This module provides the framework-independent conversion logic with
progress tracking, file locking, and robust error handling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence

import portalocker

from ovro_lwa_portal.fits_to_zarr_xradio import (
    _DISCOVERY_FREQ_BIN_HZ,
    convert_fits_dir_to_zarr,
)

__all__ = ["FITSToZarrConverter", "ConversionConfig", "ProgressCallback"]


logger = logging.getLogger(__name__)


class ProgressCallback(Protocol):
    """Protocol for progress reporting callbacks."""

    def __call__(self, stage: str, current: int, total: int, message: str) -> None:
        """Report progress to the caller.

        Parameters
        ----------
        stage : str
            The current stage of conversion (e.g., 'discovery', 'fixing', 'combining').
        current : int
            Current progress count.
        total : int
            Total items to process.
        message : str
            Human-readable progress message.
        """
        ...


class ConversionConfig:
    """Configuration for FITS to Zarr conversion.

    Parameters
    ----------
    input_dir : Path
        Directory containing input FITS files.
    output_dir : Path
        Directory where the Zarr store will be written.
    zarr_name : str, optional
        Name of the output Zarr store. Defaults to "ovro_lwa_full_lm_only.zarr".
    fixed_dir : Path | None, optional
        Directory for storing fixed FITS files. If None, creates a "fixed_fits"
        subdirectory in output_dir.
    chunk_lm : int, optional
        Chunk size for l and m spatial dimensions. Defaults to 1024.
    rebuild : bool, optional
        If True, overwrite existing Zarr store. If False, append new data. Defaults to False.
    fix_headers_on_demand : bool, optional
        If True, fix headers during conversion if they don't exist. If False, assume
        headers are already fixed. Defaults to True.
    resume : bool, optional
        If True, skip discovered FITS time steps that already exist in the output
        Zarr (when not rebuilding). Defaults to True.
    discovery_time_key_source : {"header", "filename"}, optional
        How to infer observation time when ``group_metadata_source`` is ``"fits"``.
        Defaults to ``"filename"`` (basename ``-image-`` stamp, else ``DATE-OBS``).
    lm_reference_target_size : int | None, optional
        When building the global LM reference, reproject onto this square grid size.
        Use the same value as dewarp ``target_size`` when combining dewarped FITS.
    cleanup_fixed_fits : bool, optional
        If True, delete temporary ``*_fixed.fits`` files created during on-demand
        conversion after each time-step is written. Defaults to False.
    discovery_freq_bin_hz : float, optional
        Bin width in Hz when grouping FITS by header frequency during discovery.
        Defaults to the library default (23~kHz).
    verbose : bool, optional
        Enable verbose logging. Defaults to False.
    time_keys_only : sequence of str | None, optional
        If set, only these observation time keys are converted (after discovery).
        Use with ``lm_reference_ds`` for incremental dewarp→Zarr workflows.
    lm_reference_ds : xarray.Dataset | None, optional
        Precomputed global LM reference. When set, conversion skips the reference
        scan over ``input_dir`` and uses this grid for each time step.
    group_metadata_source : {"fits", "filename"}, optional
        How to discover observation time / subband for grouping and frequency ordering.
        ``"fits"`` (default) reads FITS headers (with filename fallbacks).
        ``"filename"`` uses only basename ``-image-`` and ``_NNNMHz_`` tokens (no header
        reads during discovery). Match the value used when building ``lm_reference_ds``.
    discovery_filename_convention : {"image", "lst-color"}, optional
        Basename pattern for observation-time grouping. ``"image"`` (default) uses
        ``-image-YYYYMMDD_HHMMSS``. ``"lst-color"`` uses
        ``Blue_..._YYYYMMDD_LSTNNh_tXXXX.fits`` with subband frequency from FITS headers.
    consolidate_metadata_at_end : bool, optional
        When True (default), write consolidated ``.zmetadata`` after conversion.
        Set False for intermediate per-time Zarr appends (dewarp workflow).
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        zarr_name: str = "ovro_lwa_full_lm_only.zarr",
        fixed_dir: Path | None = None,
        chunk_lm: int = 1024,
        rebuild: bool = False,
        fix_headers_on_demand: bool = True,
        resume: bool = True,
        cleanup_fixed_fits: bool = False,
        duplicate_resolver: Callable[[str, float, list[Path]], Path] | None = None,
        discovery_freq_bin_hz: float = _DISCOVERY_FREQ_BIN_HZ,
        verbose: bool = False,
        time_keys_only: Sequence[str] | None = None,
        lm_reference_ds: Any | None = None,
        group_metadata_source: Literal["fits", "filename"] = "fits",
        discovery_time_key_source: Literal["header", "filename"] = "filename",
        discovery_filename_convention: Literal["image", "lst-color"] = "image",
        lm_reference_target_size: int | None = None,
        consolidate_metadata_at_end: bool = True,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.zarr_name = zarr_name
        self.fixed_dir = fixed_dir or (output_dir / "fixed_fits")
        self.chunk_lm = chunk_lm
        self.rebuild = rebuild
        self.fix_headers_on_demand = fix_headers_on_demand
        self.resume = resume
        self.cleanup_fixed_fits = cleanup_fixed_fits
        self.duplicate_resolver = duplicate_resolver
        self.discovery_freq_bin_hz = discovery_freq_bin_hz
        self.verbose = verbose
        self.time_keys_only = tuple(time_keys_only) if time_keys_only is not None else None
        self.lm_reference_ds = lm_reference_ds
        self.group_metadata_source = group_metadata_source
        self.discovery_time_key_source = discovery_time_key_source
        self.discovery_filename_convention = discovery_filename_convention
        self.lm_reference_target_size = lm_reference_target_size
        self.consolidate_metadata_at_end = consolidate_metadata_at_end

    @property
    def zarr_path(self) -> Path:
        """Full path to the output Zarr store."""
        return self.output_dir / self.zarr_name

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises
        ------
        FileNotFoundError
            If input directory doesn't exist.
        ValueError
            If parameters are invalid.
        """
        if not self.input_dir.exists():
            msg = f"Input directory does not exist: {self.input_dir}"
            raise FileNotFoundError(msg)

        if not self.input_dir.is_dir():
            msg = f"Input path is not a directory: {self.input_dir}"
            raise ValueError(msg)

        if self.chunk_lm < 0:
            msg = f"chunk_lm must be non-negative, got {self.chunk_lm}"
            raise ValueError(msg)

        if self.discovery_freq_bin_hz <= 0.0:
            msg = f"discovery_freq_bin_hz must be positive, got {self.discovery_freq_bin_hz}"
            raise ValueError(msg)


class FileLock:
    """Simple file-based lock for preventing concurrent writes.

    Parameters
    ----------
    lock_path : Path
        Path to the lock file.
    """

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self.lock_file: Any = None

    def __enter__(self) -> FileLock:
        """Acquire the lock."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file = open(self.lock_path, "w")
        try:
            portalocker.lock(self.lock_file, portalocker.LOCK_EX | portalocker.LOCK_NB)
        except portalocker.LockException as e:
            self.lock_file.close()
            msg = (
                f"Cannot acquire lock on {self.lock_path}. "
                "Another process may be writing to this Zarr store."
            )
            raise RuntimeError(msg) from e
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Release the lock."""
        if self.lock_file:
            portalocker.unlock(self.lock_file)
            self.lock_file.close()
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass


class FITSToZarrConverter:
    """Orchestrates FITS to Zarr conversion with progress tracking and locking.

    Parameters
    ----------
    config : ConversionConfig
        Conversion configuration.
    progress_callback : ProgressCallback | None, optional
        Optional callback for progress reporting.
    """

    def __init__(
        self,
        config: ConversionConfig,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.config = config
        self.progress_callback = progress_callback

        # Configure logging
        log_level = logging.DEBUG if config.verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def _report_progress(self, stage: str, current: int, total: int, message: str) -> None:
        """Report progress if callback is configured."""
        if self.progress_callback:
            self.progress_callback(stage, current, total, message)

    def convert(self) -> Path:
        """Execute the FITS to Zarr conversion.

        Returns
        -------
        Path
            Path to the output Zarr store.

        Raises
        ------
        FileNotFoundError
            If no matching FITS files are found.
        RuntimeError
            If conversion fails or another process is writing to the same output.
        """
        # Validate configuration
        self.config.validate()

        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Acquire lock to prevent concurrent writes
        lock_path = self.config.output_dir / f".{self.config.zarr_name}.lock"

        with FileLock(lock_path):
            logger.info("Starting FITS to Zarr conversion")
            logger.info(f"  Input: {self.config.input_dir}")
            logger.info(f"  Output: {self.config.zarr_path}")
            logger.info(f"  Mode: {'rebuild' if self.config.rebuild else 'append'}")

            self._report_progress("start", 0, 1, "Starting conversion")

            try:
                result = convert_fits_dir_to_zarr(
                    input_dir=self.config.input_dir,
                    out_dir=self.config.output_dir,
                    zarr_name=self.config.zarr_name,
                    fixed_dir=self.config.fixed_dir,
                    chunk_lm=self.config.chunk_lm,
                    rebuild=self.config.rebuild,
                    resume=self.config.resume,
                    fix_headers_on_demand=self.config.fix_headers_on_demand,
                    cleanup_fixed_fits=self.config.cleanup_fixed_fits,
                    progress_callback=self.progress_callback,
                    duplicate_resolver=self.config.duplicate_resolver,
                    discovery_freq_bin_hz=self.config.discovery_freq_bin_hz,
                    time_keys_only=self.config.time_keys_only,
                    lm_reference_ds=self.config.lm_reference_ds,
                    group_metadata_source=self.config.group_metadata_source,
                    time_key_source=self.config.discovery_time_key_source,
                    filename_convention=self.config.discovery_filename_convention,
                    lm_reference_target_size=self.config.lm_reference_target_size,
                    consolidate_metadata_at_end=self.config.consolidate_metadata_at_end,
                )

                self._report_progress("complete", 1, 1, "Conversion complete")
                logger.info(f"Conversion successful: {result}")
                return result

            except FileNotFoundError as e:
                logger.error(f"No matching FITS files found in {self.config.input_dir}")
                self._report_progress("error", 0, 1, f"Error: {e}")
                raise
            except RuntimeError as e:
                logger.error(f"Conversion failed: {e}")
                self._report_progress("error", 0, 1, f"Error: {e}")
                raise
            except Exception as e:
                logger.exception("Unexpected error during conversion")
                self._report_progress("error", 0, 1, f"Unexpected error: {e}")
                raise RuntimeError(f"Conversion failed: {e}") from e
