"""Tests for the core conversion module."""

from __future__ import annotations

import pytest
from pathlib import Path

from ovro_lwa_portal.fits_to_zarr_xradio import _DISCOVERY_FREQ_BIN_HZ
from ovro_lwa_portal.ingest.core import ConversionConfig, FileLock


class TestConversionConfig:
    """Tests for ConversionConfig class."""

    def test_default_initialization(self, tmp_path: Path) -> None:
        """Test default configuration initialization."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        config = ConversionConfig(input_dir=input_dir, output_dir=output_dir)

        assert config.input_dir == input_dir
        assert config.output_dir == output_dir
        assert config.zarr_name == "ovro_lwa_full_lm_only.zarr"
        assert config.fixed_dir == output_dir / "fixed_fits"
        assert config.chunk_lm == 1024
        assert config.rebuild is False
        assert config.verbose is False
        assert config.discovery_freq_bin_hz == _DISCOVERY_FREQ_BIN_HZ
        assert config.sky_dtype == "preserve"

    def test_custom_initialization(self, tmp_path: Path) -> None:
        """Test custom configuration initialization."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"
        fixed_dir = tmp_path / "fixed"

        config = ConversionConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            zarr_name="custom.zarr",
            fixed_dir=fixed_dir,
            chunk_lm=2048,
            rebuild=True,
            verbose=True,
        )

        assert config.zarr_name == "custom.zarr"
        assert config.fixed_dir == fixed_dir
        assert config.chunk_lm == 2048
        assert config.rebuild is True
        assert config.verbose is True

    def test_zarr_path_property(self, tmp_path: Path) -> None:
        """Test zarr_path property."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        config = ConversionConfig(input_dir=input_dir, output_dir=output_dir)

        assert config.zarr_path == output_dir / "ovro_lwa_full_lm_only.zarr"

    def test_validate_missing_input_dir(self, tmp_path: Path) -> None:
        """Test validation with missing input directory."""
        input_dir = tmp_path / "nonexistent"
        output_dir = tmp_path / "output"

        config = ConversionConfig(input_dir=input_dir, output_dir=output_dir)

        with pytest.raises(FileNotFoundError, match="Input directory does not exist"):
            config.validate()

    def test_validate_input_not_directory(self, tmp_path: Path) -> None:
        """Test validation when input is not a directory."""
        input_file = tmp_path / "input.txt"
        input_file.touch()
        output_dir = tmp_path / "output"

        config = ConversionConfig(input_dir=input_file, output_dir=output_dir)

        with pytest.raises(ValueError, match="Input path is not a directory"):
            config.validate()

    def test_validate_negative_chunk_lm(self, tmp_path: Path) -> None:
        """Test validation with negative chunk_lm."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        config = ConversionConfig(
            input_dir=input_dir, output_dir=output_dir, chunk_lm=-1
        )

        with pytest.raises(ValueError, match="chunk_lm must be non-negative"):
            config.validate()

    def test_validate_nonpositive_discovery_freq_bin(self, tmp_path: Path) -> None:
        """discovery_freq_bin_hz must be positive."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        config = ConversionConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            discovery_freq_bin_hz=0.0,
        )

        with pytest.raises(ValueError, match="discovery_freq_bin_hz must be positive"):
            config.validate()

    def test_validate_success(self, tmp_path: Path) -> None:
        """Test successful validation."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        config = ConversionConfig(input_dir=input_dir, output_dir=output_dir)

        # Should not raise
        config.validate()

    def test_validate_invalid_sky_dtype(self, tmp_path: Path) -> None:
        """sky_dtype must be preserve / float32 / float64."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        config = ConversionConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            sky_dtype="float16",  # type: ignore[arg-type]
        )

        with pytest.raises(ValueError, match="sky_dtype must be one of"):
            config.validate()

    def test_sky_dtype_float32_validates(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        config = ConversionConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            sky_dtype="float32",
        )
        config.validate()
        assert config.sky_dtype == "float32"


class TestFileLock:
    """Tests for FileLock class."""

    def test_lock_acquisition(self, tmp_path: Path) -> None:
        """Test successful lock acquisition and release."""
        lock_path = tmp_path / "test.lock"

        with FileLock(lock_path):
            assert lock_path.exists()

        # Lock file should be cleaned up
        assert not lock_path.exists()

    def test_concurrent_lock_fails(self, tmp_path: Path) -> None:
        """Test that concurrent lock acquisition fails."""
        lock_path = tmp_path / "test.lock"

        with FileLock(lock_path):
            # Try to acquire the same lock from another context
            with pytest.raises(RuntimeError, match="Cannot acquire lock"):
                with FileLock(lock_path):
                    pass
