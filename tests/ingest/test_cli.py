"""Tests for the CLI module."""

from __future__ import annotations

from unittest.mock import patch

import click
import numpy as np
import pytest
import zarr
from typer.testing import CliRunner

from ovro_lwa_portal.ingest.cli import app
from ovro_lwa_portal.ingest.core import ConversionConfig


runner = CliRunner()

# GitHub Actions sets FORCE_COLOR=3; disable color for stable help/substring checks.
_CI_PLAIN_ENV = {"NO_COLOR": "1", "FORCE_COLOR": "0"}


class TestCLI:
    """Tests for CLI commands."""

    def test_help(self) -> None:
        """Test CLI help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Convert OVRO-LWA FITS files to Zarr format" in result.stdout

    def test_version(self) -> None:
        """Test version command."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "ovro-ingest version" in result.stdout

    def test_version_flag(self) -> None:
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "ovro-ingest version" in result.stdout

    def test_convert_help(self) -> None:
        """Test convert command help."""
        result = runner.invoke(
            app,
            ["convert", "--help"],
            color=False,
            terminal_width=120,
            env=_CI_PLAIN_ENV,
        )
        plain_output = click.unstyle(result.stdout)
        assert result.exit_code == 0
        assert "Convert OVRO-LWA FITS files to a single Zarr store" in plain_output
        assert "largest grid" in plain_output
        assert "--no-resume" in plain_output
        assert "discovery-time-key" in plain_output
        assert "--target-size" in plain_output

    def test_validate_help(self) -> None:
        """Test validate command help."""
        result = runner.invoke(
            app,
            ["validate", "--help"],
            color=False,
            terminal_width=120,
            env=_CI_PLAIN_ENV,
        )
        assert result.exit_code == 0
        plain = click.unstyle(result.stdout)
        assert "Validate time-axis consistency" in plain

    def test_repair_help(self) -> None:
        """Test repair command help."""
        result = runner.invoke(
            app,
            ["repair", "--help"],
            color=False,
            terminal_width=120,
            env=_CI_PLAIN_ENV,
        )
        assert result.exit_code == 0
        plain = click.unstyle(result.stdout)
        assert "Repair interrupted-append time-axis inconsistencies" in plain

    def test_validate_consistent_store(self, tmp_path) -> None:
        """Validate command should pass for consistent test Zarr."""
        store = tmp_path / "ok.zarr"
        zg = zarr.open_group(str(store), mode="w")
        t = zg.create_dataset("time", data=np.arange(2), chunks=(2,))
        t.attrs["_ARRAY_DIMENSIONS"] = ["time"]
        v = zg.create_dataset("velocity", data=np.zeros((2, 3), dtype=np.float32), chunks=(1, 3))
        v.attrs["_ARRAY_DIMENSIONS"] = ["time", "frequency"]
        zarr.consolidate_metadata(str(store))

        result = runner.invoke(app, ["validate", str(store)])
        assert result.exit_code == 0
        assert "Store is time-axis consistent" in result.stdout

    def test_dewarp_convert_help(self) -> None:
        """dewarp-convert --help documents cascade + Zarr pipeline."""
        result = runner.invoke(
            app,
            ["dewarp-convert", "--help"],
            color=False,
            terminal_width=120,
        )
        assert result.exit_code == 0
        plain = click.unstyle(result.stdout)
        assert "flow_cascade73MHz" in plain
        assert "image_plane_correction" in plain
        assert "--cascade-parent" in plain
        assert "--target-size" in plain
        assert "--append-after-each-time" in plain
        assert "--cleanup-dewarp-staging" not in plain

    def test_audit_metadata_help(self) -> None:
        """audit-metadata --help documents subband header checks."""
        result = runner.invoke(
            app,
            ["audit-metadata", "--help"],
            color=False,
            terminal_width=120,
            env=_CI_PLAIN_ENV,
        )
        assert result.exit_code == 0
        plain = click.unstyle(result.stdout)
        assert "subband" in plain.lower()
        assert "probe-combine" in plain
        assert "staging-dir" in plain

    def test_convert_missing_args(self) -> None:
        """Test convert command with missing arguments."""
        result = runner.invoke(app, ["convert"])
        assert result.exit_code != 0
        # Typer outputs error messages, check for either stdout or that exit code is non-zero
        assert result.exit_code == 2  # Typer returns 2 for usage errors

    def test_convert_nonexistent_input(self, tmp_path) -> None:
        """Test convert command with nonexistent input directory."""
        nonexistent = tmp_path / "nonexistent"
        output = tmp_path / "output"

        result = runner.invoke(app, ["convert", str(nonexistent), str(output)])
        assert result.exit_code != 0

    def test_dewarp_convert_default_intermediate_paths_and_convert_input(
        self, tmp_path: Path
    ) -> None:
        """dewarp-convert passes default cascade/staging paths into cascade, then Zarr input=staging."""
        raw = tmp_path / "raw"
        raw.mkdir()
        out = tmp_path / "out"

        cascade_snap: list[tuple[Path, Path, Path]] = []
        convert_snap: list[Path] = []

        def fake_run_cascade(
            input_dir: Path,
            cascade_parent: Path,
            staging_dir: Path,
            **kwargs: object,
        ) -> tuple[int, list[str]]:
            cascade_snap.append((input_dir, cascade_parent, staging_dir))
            assert kwargs.get("discovery_freq_bin_hz") is not None
            staging_dir.mkdir(parents=True, exist_ok=True)
            (staging_dir / "placeholder.fits").touch()
            return (1, ["20240601_120000"], None)

        def fake_convert(config: ConversionConfig, *, log_level: Any) -> Path:
            convert_snap.append(config.input_dir)
            return out / "dummy.zarr"

        with (
            patch(
                "ovro_lwa_portal.ingest.cli.run_cascade_per_time_group",
                side_effect=fake_run_cascade,
            ),
            patch(
                "ovro_lwa_portal.ingest.cli._execute_fits_to_zarr_conversion",
                side_effect=fake_convert,
            ),
        ):
            result = runner.invoke(app, ["dewarp-convert", str(raw), str(out)])

        assert result.exit_code == 0, result.stdout + result.stderr
        assert len(cascade_snap) == 1
        inp, cascade_parent, staging_dir = cascade_snap[0]
        assert inp == raw.resolve()
        assert cascade_parent == (out / "cascade73MHz").resolve()
        assert staging_dir == (out / "dewarped_fits_staging").resolve()
        assert len(convert_snap) == 1
        assert convert_snap[0] == staging_dir

    def test_dewarp_convert_custom_cascade_and_staging_dirs(self, tmp_path: Path) -> None:
        """Explicit --cascade-parent and --staging-dir are forwarded to the cascade stage."""
        raw = tmp_path / "raw"
        raw.mkdir()
        out = tmp_path / "out"
        my_cascade = tmp_path / "my_cascade"
        my_staging = tmp_path / "my_staging"

        captured: list[tuple[Path, Path, Path]] = []

        def fake_run_cascade(
            input_dir: Path,
            cascade_parent: Path,
            staging_dir: Path,
            **_kw: object,
        ) -> tuple[int, list[str], None]:
            captured.append((input_dir, cascade_parent, staging_dir))
            staging_dir.mkdir(parents=True, exist_ok=True)
            (staging_dir / "x.fits").touch()
            return (1, ["tk"], None)

        with (
            patch(
                "ovro_lwa_portal.ingest.cli.run_cascade_per_time_group",
                side_effect=fake_run_cascade,
            ),
            patch(
                "ovro_lwa_portal.ingest.cli._execute_fits_to_zarr_conversion",
                return_value=out / "z.zarr",
            ),
        ):
            result = runner.invoke(
                app,
                [
                    "dewarp-convert",
                    str(raw),
                    str(out),
                    "--cascade-parent",
                    str(my_cascade),
                    "--staging-dir",
                    str(my_staging),
                ],
            )

        assert result.exit_code == 0, result.stdout + result.stderr
        assert captured[0][1] == my_cascade.resolve()
        assert captured[0][2] == my_staging.resolve()

    def test_dewarp_convert_forwards_target_size(self, tmp_path: Path) -> None:
        """--target-size is passed through to run_cascade_per_time_group."""
        raw = tmp_path / "raw"
        raw.mkdir()
        out = tmp_path / "out"
        seen: list[Any] = []

        def fake_run_cascade(
            _input_dir: Path,
            _cascade_parent: Path,
            staging_dir: Path,
            **kwargs: object,
        ) -> tuple[int, list[str], None]:
            seen.append(kwargs.get("target_size"))
            staging_dir.mkdir(parents=True, exist_ok=True)
            (staging_dir / "placeholder.fits").touch()
            return (1, ["20240601_120000"], None)

        with (
            patch(
                "ovro_lwa_portal.ingest.cli.run_cascade_per_time_group",
                side_effect=fake_run_cascade,
            ),
            patch(
                "ovro_lwa_portal.ingest.cli._execute_fits_to_zarr_conversion",
                return_value=out / "dummy.zarr",
            ),
        ):
            result = runner.invoke(
                app,
                ["dewarp-convert", str(raw), str(out), "--target-size", "2048"],
            )

        assert result.exit_code == 0, result.stdout + result.stderr
        assert seen == [2048]
