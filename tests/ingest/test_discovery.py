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
        lambda groups, **kwargs: groups,
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
        lambda groups, **kwargs: groups,
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


def test_summarize_time_grouped_fits_attaches_zarr_size_estimate(
    monkeypatch, tmp_path: Path
) -> None:
    """Discovery summary includes the per-file pixel size estimate when available."""
    tkey = "20250106_051855"
    by_time = {tkey: [tmp_path / _image_name(tkey, 41)]}
    by_time[tkey][0].write_bytes(b"")

    monkeypatch.setattr(
        "ovro_lwa_portal.ingest.discovery.estimate_zarr_store_bytes",
        lambda groups, **kwargs: 12_345,
    )
    summary = summarize_time_grouped_fits(
        by_time,
        discovery=IngestDiscoveryConfig(group_metadata_source="filename"),
    )
    assert summary.estimated_zarr_bytes == 12_345
    assert summary.estimated_zarr_size == "12.1 KiB"


def test_resolve_glob_convert_discovery_single_pass(tmp_path: Path, monkeypatch) -> None:
    """resolve_glob_convert_discovery groups once and fills cached metadata."""
    from astropy.io import fits

    from ovro_lwa_portal.ingest.discovery import (
        IngestDiscoveryConfig,
        resolve_glob_convert_discovery,
    )

    tkey = "20250106_051855"
    paths = [
        tmp_path / _image_name(tkey, 41),
        tmp_path / _image_name(tkey, 55),
    ]
    for mhz, path in zip((41, 55), paths, strict=True):
        fits.PrimaryHDU(
            data=[[1.0]],
            header=fits.Header(
                {"RESTFREQ": float(mhz * 1e6), "BMAJ": 0.25, "BMIN": 0.25}
            ),
        ).writeto(path)

    discover_calls = 0
    original = __import__(
        "ovro_lwa_portal.ingest.discovery", fromlist=["discover_time_grouped_paths"]
    ).discover_time_grouped_paths

    def counting_discover(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal discover_calls
        discover_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "ovro_lwa_portal.ingest.discovery.discover_time_grouped_paths",
        counting_discover,
    )

    plan = resolve_glob_convert_discovery(
        str(tmp_path / "*.fits"),
        discovery=IngestDiscoveryConfig(group_metadata_source="fits"),
        out_zarr=tmp_path / "store.zarr",
        rebuild=True,
        resume=False,
        funpack=False,
    )
    assert discover_calls == 1
    assert len(plan.discovery_metadata) == 2
    assert plan.discovered.input_files == 2
    assert plan.to_process_summary.input_files == 2


def test_summarize_time_grouped_fits_uses_cached_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    """Summarize reuses discovery metadata instead of re-reading FITS headers."""
    from ovro_lwa_portal.fits_to_zarr_xradio import _DiscoveryFileMetadata

    tkey = "20250106_051855"
    fp = tmp_path / _image_name(tkey, 41)
    fp.write_bytes(b"")
    by_time = {tkey: [fp]}
    metadata = {
        fp.resolve(): _DiscoveryFileMetadata(
            time_key=tkey,
            frequency_hz=41e6,
            notes=("time-from-filename",),
            stokes_key=1,
            ingest_header=None,
        )
    }

    def boom(*_a: object, **_k: object) -> None:
        pytest.fail("metadata extract should not run when cache is provided")

    monkeypatch.setattr(
        "ovro_lwa_portal.ingest.discovery._extract_group_metadata_for_discovery",
        boom,
    )
    summary = summarize_time_grouped_fits(
        by_time,
        discovery=IngestDiscoveryConfig(group_metadata_source="filename"),
        discovery_metadata=metadata,
    )
    assert summary.input_files == 1
    assert summary.frequency_groups == 1


def test_discovery_sidecar_path_for_zarr() -> None:
    """Sidecar name replaces ``.zarr`` with ``_metadata.json`` beside the store."""
    from ovro_lwa_portal.ingest.discovery_sidecar import discovery_sidecar_path_for_zarr

    out = Path("/fast/claw/IV-10min-Taper-Robust-0_30jun26.zarr")
    assert discovery_sidecar_path_for_zarr(out) == Path(
        "/fast/claw/IV-10min-Taper-Robust-0_30jun26_metadata.json"
    )


def test_resolve_glob_convert_discovery_uses_sidecar_on_rerun(
    tmp_path: Path, monkeypatch
) -> None:
    """A second convert run loads discovery from the sidecar instead of re-reading FITS."""
    from astropy.io import fits

    from ovro_lwa_portal.ingest.discovery import (
        IngestDiscoveryConfig,
        resolve_glob_convert_discovery,
    )

    tkey = "20250106_051855"
    paths = [
        tmp_path / _image_name(tkey, 41),
        tmp_path / _image_name(tkey, 55),
    ]
    for mhz, path in zip((41, 55), paths, strict=True):
        fits.PrimaryHDU(
            data=[[1.0]],
            header=fits.Header(
                {"RESTFREQ": float(mhz * 1e6), "BMAJ": 0.25, "BMIN": 0.25}
            ),
        ).writeto(path)

    discover_calls = 0
    original = __import__(
        "ovro_lwa_portal.ingest.discovery", fromlist=["discover_time_grouped_paths"]
    ).discover_time_grouped_paths

    def counting_discover(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal discover_calls
        discover_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "ovro_lwa_portal.ingest.discovery.discover_time_grouped_paths",
        counting_discover,
    )

    out_zarr = tmp_path / "store.zarr"
    glob_pattern = str(tmp_path / "*.fits")
    discovery = IngestDiscoveryConfig(group_metadata_source="fits")

    plan1 = resolve_glob_convert_discovery(
        glob_pattern,
        discovery=discovery,
        out_zarr=out_zarr,
        rebuild=True,
        resume=False,
        funpack=False,
    )
    assert discover_calls == 1
    assert (tmp_path / "store_metadata.json").is_file()

    plan2 = resolve_glob_convert_discovery(
        glob_pattern,
        discovery=discovery,
        out_zarr=out_zarr,
        rebuild=True,
        resume=False,
        funpack=False,
    )
    assert discover_calls == 1
    assert plan2.discovered.input_files == plan1.discovered.input_files
    assert len(plan2.discovery_metadata) == len(plan1.discovery_metadata)


def test_discovery_sidecar_invalidates_on_mtime_change(tmp_path: Path) -> None:
    """Sidecar is ignored when a source FITS file changes on disk."""
    from astropy.io import fits

    from ovro_lwa_portal.ingest.discovery import (
        IngestDiscoveryConfig,
        resolve_glob_convert_discovery,
    )

    tkey = "20250106_051855"
    path = tmp_path / _image_name(tkey, 41)
    fits.PrimaryHDU(
        data=[[1.0]],
        header=fits.Header({"RESTFREQ": 41e6, "BMAJ": 0.25, "BMIN": 0.25}),
    ).writeto(path)

    out_zarr = tmp_path / "store.zarr"
    glob_pattern = str(tmp_path / "*.fits")
    discovery = IngestDiscoveryConfig(group_metadata_source="fits")

    resolve_glob_convert_discovery(
        glob_pattern,
        discovery=discovery,
        out_zarr=out_zarr,
        rebuild=True,
        resume=False,
        funpack=False,
    )

    path.write_bytes(path.read_bytes() + b" ")

    from ovro_lwa_portal.ingest.discovery_sidecar import load_glob_discovery_sidecar

    sidecar = tmp_path / "store_metadata.json"
    loaded = load_glob_discovery_sidecar(
        sidecar,
        glob_pattern=glob_pattern,
        discovery=discovery,
        out_zarr=out_zarr,
        rebuild=True,
        resume=False,
        funpack=False,
    )
    assert loaded is None
