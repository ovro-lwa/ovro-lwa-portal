"""Tests for shared ingest discovery helpers."""

from __future__ import annotations

from pathlib import Path

from ovro_lwa_portal.ingest.discovery import (
    IngestDiscoveryConfig,
    discover_time_grouped_fits,
    discover_time_grouped_paths,
    plan_convert_discovery,
    prepare_ingest_time_groups,
    summarize_time_grouped_fits,
)


def test_prepare_ingest_time_groups_resume_uses_completed_filter(
    monkeypatch, tmp_path: Path
) -> None:
    """Resume delegates to _filter_completed_time_keys when the Zarr exists."""
    out_zarr = tmp_path / "store.zarr"
    out_zarr.mkdir()
    by_time = {"20240601_120000": [tmp_path / "a.fits"], "20240602_120000": [tmp_path / "b.fits"]}

    monkeypatch.setattr(
        "ovro_lwa_portal.ingest.discovery._filter_invalid_beam_files",
        lambda groups: groups,
    )

    def fake_filter(
        groups: dict[str, list[Path]], path: Path, *, rebuild: bool, context: str
    ) -> dict[str, list[Path]]:
        assert path == out_zarr
        assert context == "convert"
        return {"20240602_120000": groups["20240602_120000"]}

    monkeypatch.setattr(
        "ovro_lwa_portal.ingest.discovery._filter_completed_time_keys",
        fake_filter,
    )
    remaining = prepare_ingest_time_groups(
        by_time,
        out_zarr=out_zarr,
        rebuild=False,
        resume=True,
        context="convert",
    )
    assert list(remaining.keys()) == ["20240602_120000"]


def test_prepare_ingest_time_groups_skips_beam_filter_when_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    """Per-time ingest can defer BMAJ/BMIN checks until after beam repair."""
    by_time = {"20240601_120000": [tmp_path / "a.fits"]}
    called = False

    def fake_beam_filter(groups: dict[str, list[Path]]) -> dict[str, list[Path]]:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(
        "ovro_lwa_portal.ingest.discovery._filter_invalid_beam_files",
        fake_beam_filter,
    )
    remaining = prepare_ingest_time_groups(by_time, filter_invalid_beam=False)
    assert not called
    assert remaining == by_time


def test_discover_time_grouped_fits_forwards_time_key_source(monkeypatch, tmp_path: Path) -> None:
    """IngestDiscoveryConfig.time_key_source is passed to _discover_groups."""
    seen: list[str] = []

    def fake_discover(*_a: object, **kw: object) -> dict[str, list[Path]]:
        seen.append(str(kw.get("time_key_source")))
        return {}

    monkeypatch.setattr(
        "ovro_lwa_portal.ingest.discovery._discover_groups",
        fake_discover,
    )
    discover_time_grouped_fits(
        tmp_path,
        discovery=IngestDiscoveryConfig(time_key_source="header"),
    )
    assert seen == ["header"]


def _image_name(time_key: str, mhz: int) -> str:
    date, hms = time_key.split("_")
    return f"{date}_{hms}_{mhz}MHz_averaged_Run-I-image-{date}_{hms}.fits"


def test_summarize_time_grouped_fits_counts_time_freq_pol(tmp_path: Path) -> None:
    """Summary reports input files and time/frequency/polarization groups."""
    tkey_a = "20250106_051855"
    tkey_b = "20250106_052955"
    paths = [
        tmp_path / _image_name(tkey_a, 41),
        tmp_path / _image_name(tkey_a, 55),
        tmp_path / _image_name(tkey_b, 41),
    ]
    for path in paths:
        path.write_bytes(b"")

    discovery = IngestDiscoveryConfig(group_metadata_source="filename")
    by_time = discover_time_grouped_paths(paths, discovery=discovery)
    summary = summarize_time_grouped_fits(by_time, discovery=discovery)

    assert summary.input_files == 3
    assert summary.time_groups == 2
    assert summary.frequency_groups == 2
    assert summary.polarization_groups == 1
    assert summary.polarization_labels == ("I",)
    assert summary.time_frequency_polarization_cells == 3
    assert summary.zarr_shape_hint == "(2, 2, 1)"


def test_plan_convert_discovery_splits_discovered_and_to_process(
    monkeypatch, tmp_path: Path
) -> None:
    """plan_convert_discovery returns separate summaries for resume filtering."""
    out_zarr = tmp_path / "store.zarr"
    out_zarr.mkdir()
    tkey_a = "20250106_051855"
    tkey_b = "20250106_052955"
    by_time = {
        tkey_a: [tmp_path / _image_name(tkey_a, 41)],
        tkey_b: [tmp_path / _image_name(tkey_b, 41)],
    }
    for files in by_time.values():
        files[0].write_bytes(b"")

    monkeypatch.setattr(
        "ovro_lwa_portal.ingest.discovery._filter_invalid_beam_files",
        lambda groups: groups,
    )
    monkeypatch.setattr(
        "ovro_lwa_portal.ingest.discovery._filter_completed_time_keys",
        lambda groups, path, *, rebuild, context: {tkey_b: groups[tkey_b]},
    )

    discovery = IngestDiscoveryConfig(group_metadata_source="filename")
    discovered, to_process = plan_convert_discovery(
        by_time,
        discovery=discovery,
        out_zarr=out_zarr,
        rebuild=False,
        resume=True,
    )
    assert discovered.time_groups == 2
    assert discovered.input_files == 2
    assert to_process.time_groups == 1
    assert to_process.input_files == 1
