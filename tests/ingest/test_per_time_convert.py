"""Tests for per-time glob-driven FITS→Zarr conversion."""

from __future__ import annotations

from pathlib import Path

import pytest

from ovro_lwa_portal.ingest.discovery import discover_time_grouped_paths
from ovro_lwa_portal.ingest.per_time_convert import (
    PerTimeGlobConvertConfig,
    run_per_time_glob_convert,
    sources_need_funpack,
    stage_time_group_symlinks,
)


def _image_name(time_key: str, mhz: int, run: str = "RunA") -> str:
    date, hms = time_key.split("_")
    return f"{date}_{hms}_{mhz}MHz_averaged_{run}-I-image-{date}_{hms}.fits"


class TestStageTimeGroupSymlinks:
    def test_prefix_avoids_basename_collisions(self, tmp_path: Path) -> None:
        staging = tmp_path / "staging"
        src_a = tmp_path / "src_a"
        src_b = tmp_path / "src_b"
        src_a.mkdir()
        src_b.mkdir()
        name = _image_name("20250106_051855", 73)
        file_a = src_a / name
        file_b = src_b / name
        file_a.write_bytes(b"a")
        file_b.write_bytes(b"b")

        n = stage_time_group_symlinks(
            staging,
            "20250106_051855",
            [file_a, file_b],
        )
        assert n == 2
        staged = sorted(staging.glob("20250106_051855__*.fits"))
        assert len(staged) == 2
        assert all(p.is_symlink() for p in staged)
        resolved = {p.resolve() for p in staged}
        assert resolved == {file_a.resolve(), file_b.resolve()}


class TestDiscoverTimeGroupedPaths:
    def test_groups_paths_from_multiple_directories(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        tkey = "20250106_051855"
        f41 = dir_a / _image_name(tkey, 41)
        f55 = dir_b / _image_name(tkey, 55)
        f41.write_bytes(b"")
        f55.write_bytes(b"")

        by_time = discover_time_grouped_paths([f41, f55])
        assert set(by_time) == {tkey}
        assert {p.name for p in by_time[tkey]} == {f41.name, f55.name}


class TestSourcesNeedFunpack:
    def test_detects_fs_suffix(self, tmp_path: Path) -> None:
        fs_path = tmp_path / "x.fits.fs"
        fits_path = tmp_path / "x.fits"
        fs_path.write_bytes(b"")
        fits_path.write_bytes(b"")
        assert sources_need_funpack([fs_path]) is True
        assert sources_need_funpack([fits_path]) is False


class TestRunPerTimeGlobConvert:
    def test_invokes_converter_per_time_and_cleans_staging(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src_root = tmp_path / "pipeline"
        src_root.mkdir()
        tkey = "20250106_051855"
        fits_paths = []
        for mhz in (41, 55):
            p = src_root / _image_name(tkey, mhz, run=f"R{mhz}")
            p.write_bytes(b"")
            fits_paths.append(p)

        staging = tmp_path / "staging"
        output = tmp_path / "out"
        fixed = tmp_path / "fixed"
        convert_calls: list[str] = []

        def fake_glob(pattern: str) -> list[Path]:
            assert pattern == str(src_root / "*.fits")
            return fits_paths

        def fake_convert(self, progress_callback=None):  # noqa: ANN001
            convert_calls.append("called")
            return output / "store.zarr"

        monkeypatch.setattr(
            "ovro_lwa_portal.ingest.per_time_convert.collect_glob_sources",
            fake_glob,
        )
        monkeypatch.setattr(
            "ovro_lwa_portal.ingest.per_time_convert._load_global_lm_reference_dataset",
            lambda *a, **k: __import__("xarray").Dataset(),
        )
        monkeypatch.setattr(
            "ovro_lwa_portal.ingest.per_time_convert._global_frequency_coord_hz",
            lambda *a, **k: __import__("numpy").array([41e6, 55e6]),
        )
        monkeypatch.setattr(
            "ovro_lwa_portal.ingest.per_time_convert.prepare_ingest_time_groups",
            lambda by_time, **k: by_time,
        )
        monkeypatch.setattr(
            "ovro_lwa_portal.ingest.per_time_convert.FITSToZarrConverter.convert",
            fake_convert,
        )
        monkeypatch.setattr(
            "ovro_lwa_portal.ingest.per_time_convert._consolidate_zarr_metadata",
            lambda path: None,
        )

        config = PerTimeGlobConvertConfig(
            glob_pattern=str(src_root / "*.fits"),
            staging_dir=staging,
            output_dir=output,
            fixed_dir=fixed,
            zarr_name="store.zarr",
            funpack=False,
        )
        run_per_time_glob_convert(config)

        assert len(convert_calls) == 1
        assert list(staging.glob(f"{tkey}__*.fits")) == []
