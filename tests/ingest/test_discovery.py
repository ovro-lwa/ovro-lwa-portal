"""Tests for shared ingest discovery helpers."""

from __future__ import annotations

from pathlib import Path

from ovro_lwa_portal.ingest.discovery import (
    IngestDiscoveryConfig,
    discover_time_grouped_fits,
    prepare_ingest_time_groups,
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
