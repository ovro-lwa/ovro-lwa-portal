"""Tests for dewarp → staging helpers."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy.io import fits

from tests.conftest import skip_github_ci_without_image_plane_correction

import ovro_lwa_portal.ingest.dewarp_convert as dewarp_convert_mod
from ovro_lwa_portal.ingest.dewarp_convert import (
    collect_cascade_fits,
    dewarp_and_convert_append_each_time,
    remove_staged_files_for_time_key,
    run_cascade_per_time_group,
)


def _write_minimal_fits(path: Path, *, restfreq_hz: float, date_obs: str) -> None:
    data = np.zeros((4, 4), dtype=np.float32)
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["DATE-OBS"] = date_obs
    hdu.header["RESTFREQ"] = restfreq_hz
    # Synthesized beam keywords let the file survive ``_filter_invalid_beam_files``;
    # tests that specifically exercise the invalid-beam skip path build headers
    # without these (or override with zero) on their own.
    hdu.header["BMAJ"] = 0.1
    hdu.header["BMIN"] = 0.1
    hdu.writeto(path, overwrite=True)


def test_collect_cascade_fits_prefers_flat(tmp_path: Path) -> None:
    """Non-recursive *.fits in outroot are returned first."""
    (tmp_path / "a.fits").touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.fits").touch()
    found = collect_cascade_fits(tmp_path)
    assert [p.name for p in found] == ["a.fits"]


def test_collect_cascade_fits_recursive(tmp_path: Path) -> None:
    """When no top-level fits, rglob finds nested files."""
    sub = tmp_path / "nested"
    sub.mkdir(parents=True)
    (sub / "c.fits").touch()
    found = collect_cascade_fits(tmp_path)
    assert len(found) == 1
    assert found[0].name == "c.fits"


@skip_github_ci_without_image_plane_correction
@pytest.mark.dewarp
def test_run_cascade_per_time_group_fake_cascade(tmp_path: Path) -> None:
    """Staging receives one link per cascade output per time group."""

    def fake_cascade(
        image_filenames: list[str],
        outroot: str,
        write: bool = True,
        **_kwargs: Any,
    ) -> None:
        out = Path(outroot)
        out.mkdir(parents=True, exist_ok=True)
        for i, fp in enumerate(image_filenames):
            shutil.copy2(fp, out / f"dewarped_{i}.fits")

    raw = tmp_path / "raw"
    raw.mkdir()
    cascade_parent = tmp_path / "cascade"
    staging = tmp_path / "staging"
    # At least one file in the group must carry the ``73MHz`` cascade-reference
    # token so :func:`_filter_time_groups_without_cascade_reference` keeps the group.
    _write_minimal_fits(
        raw / "a_73MHz.fits", restfreq_hz=73e6, date_obs="2024-06-01T12:00:00.0"
    )
    _write_minimal_fits(
        raw / "b_82MHz.fits", restfreq_hz=82e6, date_obs="2024-06-01T12:00:00.0"
    )

    n, keys, _lm_ref = run_cascade_per_time_group(
        raw,
        cascade_parent,
        staging,
        discovery_freq_bin_hz=23e3,
        cascade_fn=fake_cascade,
    )
    assert len(keys) == 1
    assert n == 2
    staged = sorted(staging.glob("*.fits"))
    assert len(staged) == 2
    assert all("__dewarped_" in p.name for p in staged)


@skip_github_ci_without_image_plane_correction
@pytest.mark.dewarp
def test_run_cascade_per_time_group_raises_if_no_outputs(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_minimal_fits(
        raw / "only_73MHz.fits", restfreq_hz=73e6, date_obs="2024-06-02T12:00:00.0"
    )

    def noop_cascade(**_kwargs: Any) -> None:
        return None

    with pytest.raises(RuntimeError, match="produced no"):
        run_cascade_per_time_group(
            raw,
            tmp_path / "cascade",
            tmp_path / "staging",
            discovery_freq_bin_hz=23e3,
            cascade_fn=noop_cascade,
        )


def test_run_cascade_per_time_group_passes_absolute_paths_and_outroot(tmp_path: Path) -> None:
    """Cascade receives absolute ``image_filenames`` and ``outroot`` under cascade_parent/time_key."""
    recorded: list[dict[str, Any]] = []

    def recording_cascade(
        *,
        image_filenames: Sequence[str],
        outroot: str,
        cleaned: bool,
        qa: bool,
        use_best_pb_model: bool,
        bright_source_flux_qa: bool,
        write: bool,
        target_size: int | None = None,
    ) -> None:
        recorded.append(
            {
                "image_filenames": list(image_filenames),
                "outroot": outroot,
                "cleaned": cleaned,
                "qa": qa,
                "use_best_pb_model": use_best_pb_model,
                "bright_source_flux_qa": bright_source_flux_qa,
                "write": write,
                "target_size": target_size,
            }
        )
        out = Path(outroot)
        out.mkdir(parents=True, exist_ok=True)
        for i, fp in enumerate(image_filenames):
            shutil.copy2(fp, out / f"out_{i}.fits")

    raw = tmp_path / "raw"
    raw.mkdir()
    cascade_parent = tmp_path / "cascade"
    staging = tmp_path / "staging"
    _write_minimal_fits(
        raw / "low_70MHz.fits", restfreq_hz=70e6, date_obs="2024-06-01T12:00:00.0"
    )
    # ``high_73MHz`` is the cascade-reference subband; without a 73 MHz file the
    # whole group would now be filtered out before the cascade runs.
    _write_minimal_fits(
        raw / "high_73MHz.fits", restfreq_hz=73e6, date_obs="2024-06-01T12:00:00.0"
    )

    n, keys, _lm_ref = run_cascade_per_time_group(
        raw,
        cascade_parent,
        staging,
        discovery_freq_bin_hz=23e3,
        cascade_fn=recording_cascade,
        cleaned=False,
        qa=False,
        use_best_pb_model=False,
        bright_source_flux_qa=False,
        write=False,
    )

    assert keys == ["20240601_120000"]
    assert n == 2
    assert len(recorded) == 1
    call = recorded[0]
    assert call["cleaned"] is False
    assert call["qa"] is False
    assert call["use_best_pb_model"] is False
    assert call["bright_source_flux_qa"] is False
    assert call["write"] is False
    assert call["target_size"] is None

    expected_outroot = cascade_parent / "20240601_120000"
    assert Path(call["outroot"]) == expected_outroot
    assert Path(call["outroot"]).is_dir()

    # Frequency order: 70 MHz before 73 MHz (same as _discover_groups sort).
    paths = [Path(p) for p in call["image_filenames"]]
    assert all(p.is_absolute() for p in paths)
    assert [p.name for p in paths] == ["low_70MHz.fits", "high_73MHz.fits"]
    assert paths[0].resolve() == (raw / "low_70MHz.fits").resolve()
    assert paths[1].resolve() == (raw / "high_73MHz.fits").resolve()

    staged = sorted(staging.glob("*.fits"))
    assert [p.name for p in staged] == [
        "20240601_120000__out_0.fits",
        "20240601_120000__out_1.fits",
    ]


def test_run_cascade_per_time_group_multiple_time_keys(tmp_path: Path) -> None:
    """Each observation time gets its own outroot and staging name prefix."""
    calls: list[str] = []

    def fake_cascade(*, image_filenames: Sequence[str], outroot: str, **_kw: Any) -> None:
        calls.append(outroot)
        out = Path(outroot)
        out.mkdir(parents=True, exist_ok=True)
        tkey = Path(outroot).name
        shutil.copy2(image_filenames[0], out / f"{tkey}_band.fits")

    raw = tmp_path / "raw"
    raw.mkdir()
    # Both groups carry the ``73MHz`` cascade-reference token so the filter
    # introduced for missing 73 MHz files keeps them.
    _write_minimal_fits(
        raw / "day1_73MHz.fits", restfreq_hz=73e6, date_obs="2024-06-01T12:00:00.0"
    )
    _write_minimal_fits(
        raw / "day2_73MHz.fits", restfreq_hz=73e6, date_obs="2024-06-02T12:00:00.0"
    )

    n, keys, _lm_ref = run_cascade_per_time_group(
        raw,
        tmp_path / "cascade",
        tmp_path / "staging",
        discovery_freq_bin_hz=23e3,
        cascade_fn=fake_cascade,
    )

    assert keys == ["20240601_120000", "20240602_120000"]
    assert n == 2
    assert calls == [
        str(tmp_path / "cascade" / "20240601_120000"),
        str(tmp_path / "cascade" / "20240602_120000"),
    ]
    staged = sorted((tmp_path / "staging").glob("*.fits"))
    assert {p.name for p in staged} == {
        "20240601_120000__20240601_120000_band.fits",
        "20240602_120000__20240602_120000_band.fits",
    }


def test_run_cascade_per_time_group_clears_staging_and_per_time_outroot(tmp_path: Path) -> None:
    """Existing staging tree and per-time cascade dirs are removed before a run."""
    def fake_cascade(*, image_filenames: Sequence[str], outroot: str, **_kw: Any) -> None:
        out = Path(outroot)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_filenames[0], out / "new.fits")

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_minimal_fits(
        raw / "one_73MHz.fits", restfreq_hz=73e6, date_obs="2024-06-03T12:00:00.0"
    )

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "stale.fits").write_text("old")
    sub = staging / "nested"
    sub.mkdir()
    (sub / "more.fits").touch()

    tkey = "20240603_120000"
    outroot = tmp_path / "cascade" / tkey
    outroot.mkdir(parents=True)
    (outroot / "leftover.txt").write_text("stale")

    run_cascade_per_time_group(
        raw,
        tmp_path / "cascade",
        staging,
        discovery_freq_bin_hz=23e3,
        cascade_fn=fake_cascade,
    )

    assert not (staging / "stale.fits").exists()
    assert not (staging / "nested").exists()
    assert not (outroot / "leftover.txt").exists()
    assert (staging / f"{tkey}__new.fits").is_file()


def test_run_cascade_groups_by_basename_image_time_not_header(tmp_path: Path) -> None:
    """Dewarp discovery co-locates bands that share ``-image-…`` even when DATE-OBS disagrees."""
    calls: list[tuple[str, list[str]]] = []

    def fake_cascade(*, image_filenames: Sequence[str], outroot: str, **_kw: Any) -> None:
        calls.append((Path(outroot).name, sorted(Path(p).name for p in image_filenames)))
        out = Path(outroot)
        out.mkdir(parents=True, exist_ok=True)
        for i, fp in enumerate(image_filenames):
            shutil.copy2(fp, out / f"band_{i}.fits")

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_minimal_fits(
        raw / "18MHz-I-Deep-Taper-Robust-0-image-20241221_102109_a.fits",
        restfreq_hz=18e6,
        date_obs="2024-12-21T01:00:00.0",
    )
    _write_minimal_fits(
        raw / "73MHz-I-Deep-Taper-Robust-0-image-20241221_102109_b.fits",
        restfreq_hz=73e6,
        date_obs="2024-12-22T23:00:00.0",
    )

    n, keys, _lm_ref = run_cascade_per_time_group(
        raw,
        tmp_path / "cascade",
        tmp_path / "staging",
        discovery_freq_bin_hz=23e3,
        cascade_fn=fake_cascade,
    )

    assert keys == ["20241221_102109"]
    assert n == 2
    assert len(calls) == 1
    assert calls[0][0] == "20241221_102109"
    assert set(calls[0][1]) == {
        "18MHz-I-Deep-Taper-Robust-0-image-20241221_102109_a.fits",
        "73MHz-I-Deep-Taper-Robust-0-image-20241221_102109_b.fits",
    }


def test_remove_staged_files_for_time_key(tmp_path: Path) -> None:
    staging = tmp_path / "st"
    staging.mkdir()
    (staging / "20240101_000000__a.fits").touch()
    (staging / "20240101_000000__b.fits").touch()
    (staging / "20240102_000000__c.fits").touch()
    n = remove_staged_files_for_time_key(staging, "20240101_000000")
    assert n == 2
    assert sorted(p.name for p in staging.glob("*.fits")) == ["20240102_000000__c.fits"]


def test_dewarp_and_convert_append_each_time_calls_zarr_per_step(tmp_path: Path, monkeypatch):
    """Incremental mode runs FITS→Zarr once per observation time after staging that step."""
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    cascade = tmp_path / "cascade"
    staging = tmp_path / "staging"
    fixed = tmp_path / "fixed"
    for d in (raw, out, cascade, staging, fixed):
        d.mkdir()

    # ``_filter_invalid_beam_files`` opens the primary header of each discovered file,
    # so the paths returned by ``fake_discover`` must point at real FITS with valid
    # BMAJ/BMIN — otherwise the filter drops them and discovery yields nothing.
    # The basenames also need a ``73MHz`` token so the cascade-reference filter keeps
    # each single-file time group.
    _write_minimal_fits(raw / "a_73MHz.fits", restfreq_hz=73e6, date_obs="2024-06-01T12:00:00.0")
    _write_minimal_fits(raw / "b_73MHz.fits", restfreq_hz=73e6, date_obs="2024-06-02T12:00:00.0")

    def fake_discover(*_a: object, **_k: object) -> dict[str, list[Path]]:
        return {
            "20240601_120000": [raw / "a_73MHz.fits"],
            "20240602_120000": [raw / "b_73MHz.fits"],
        }

    monkeypatch.setattr(dewarp_convert_mod, "discover_time_grouped_fits", fake_discover)
    mref = MagicMock()
    mref.copy = MagicMock(return_value=mref)
    monkeypatch.setattr(
        dewarp_convert_mod,
        "_load_global_lm_reference_dataset",
        lambda *a, **k: mref,
    )

    cascade_calls: list[str] = []

    def fake_run_cascade(
        tkey: str,
        files: Sequence[Path],
        _cascade_parent: Path,
        staging_dir: Path,
        *,
        cascade_fn: Any,
        **_kw: Any,
    ) -> int:
        cascade_calls.append(tkey)
        assert len(files) == 1
        (staging_dir / f"{tkey}__out.fits").touch()
        return 1

    monkeypatch.setattr(dewarp_convert_mod, "run_cascade_for_time_key", fake_run_cascade)

    with patch("ovro_lwa_portal.ingest.core.FITSToZarrConverter") as conv_cls:
        inst = MagicMock()
        conv_cls.return_value = inst
        n_staged, keys = dewarp_and_convert_append_each_time(
            raw,
            out,
            cascade,
            staging,
            fixed,
            zarr_name="z.zarr",
            chunk_lm=64,
            rebuild=True,
            fix_headers_on_demand=False,
            cleanup_fixed_fits=False,
            discovery_freq_bin_hz=23e3,
            duplicate_resolver=None,
            cascade_fn=lambda **kw: None,
        )

    assert keys == ["20240601_120000", "20240602_120000"]
    assert n_staged == 2
    assert cascade_calls == keys
    assert inst.convert.call_count == 2
    cfg0 = conv_cls.call_args_list[0][0][0]
    cfg1 = conv_cls.call_args_list[1][0][0]
    assert cfg0.time_keys_only == ("20240601_120000",)
    assert cfg1.time_keys_only == ("20240602_120000",)
    assert cfg0.rebuild is True
    assert cfg1.rebuild is False


def test_run_cascade_per_time_group_skips_files_with_invalid_beam(tmp_path: Path) -> None:
    """Raw FITS with missing/zero BMAJ/BMIN must be dropped before the cascade runs."""

    cascade_inputs: list[list[str]] = []

    def recording_cascade(
        image_filenames: list[str],
        outroot: str,
        write: bool = True,
        **_kwargs: Any,
    ) -> None:
        cascade_inputs.append(list(image_filenames))
        out = Path(outroot)
        out.mkdir(parents=True, exist_ok=True)
        (out / "dewarped_0.fits").touch()

    raw = tmp_path / "raw"
    raw.mkdir()
    cascade_parent = tmp_path / "cascade"
    staging = tmp_path / "staging"

    # ``good`` doubles as the cascade-reference subband (73 MHz) so the
    # missing-73MHz filter doesn't drop the whole group while we're really
    # testing the invalid-beam skip.
    good = raw / "good_73MHz.fits"
    bad_missing = raw / "bad_missing_74MHz.fits"
    bad_zero = raw / "bad_zero_78MHz.fits"

    _write_minimal_fits(good, restfreq_hz=73e6, date_obs="2024-06-01T12:00:00.0")
    # Bad files have valid time/freq metadata but no usable beam.
    hdu = fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.float32))
    hdu.header["DATE-OBS"] = "2024-06-01T12:00:00.0"
    hdu.header["RESTFREQ"] = 74e6
    hdu.writeto(bad_missing, overwrite=True)

    hdu = fits.PrimaryHDU(data=np.zeros((4, 4), dtype=np.float32))
    hdu.header["DATE-OBS"] = "2024-06-01T12:00:00.0"
    hdu.header["RESTFREQ"] = 78e6
    hdu.header["BMAJ"] = 0.0
    hdu.header["BMIN"] = 0.0
    hdu.writeto(bad_zero, overwrite=True)

    n, keys, _lm_ref = run_cascade_per_time_group(
        raw,
        cascade_parent,
        staging,
        discovery_freq_bin_hz=23e3,
        cascade_fn=recording_cascade,
    )

    assert n == 1
    assert keys == ["20240601_120000"]
    assert len(cascade_inputs) == 1
    forwarded = {Path(p).name for p in cascade_inputs[0]}
    assert forwarded == {"good_73MHz.fits"}


def test_run_cascade_per_time_group_forwards_target_size(tmp_path: Path) -> None:
    recorded: list[int | None] = []

    def cascade_kw(
        *,
        image_filenames: Sequence[str],
        outroot: str,
        target_size: int | None = None,
        **_kw: Any,
    ) -> None:
        recorded.append(target_size)
        out = Path(outroot)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_filenames[0], out / "one.fits")

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_minimal_fits(
        raw / "one_73MHz.fits", restfreq_hz=73e6, date_obs="2024-06-01T12:00:00.0"
    )

    n, keys, _lm_ref = run_cascade_per_time_group(
        raw,
        tmp_path / "cascade",
        tmp_path / "staging",
        discovery_freq_bin_hz=23e3,
        cascade_fn=cascade_kw,
        target_size=3122,
    )
    assert n == 1
    assert keys == ["20240601_120000"]
    assert recorded == [3122]


def _seed_zarr_with_time_keys(out_zarr: Path, time_keys: list[str]) -> None:
    """Write a minimal Zarr store whose ``time`` coord matches the given keys.

    Each key uses the same ``%Y%m%d_%H%M%S`` UTC format the discovery layer
    emits, mirroring how ``xradio`` writes the ``time`` dimension as MJD floats.
    """
    import xarray as xr
    from astropy.time import Time

    iso = [
        f"{k[:4]}-{k[4:6]}-{k[6:8]}T{k[9:11]}:{k[11:13]}:{k[13:15]}" for k in time_keys
    ]
    mjd = np.array(
        [float(Time(t, format="isot", scale="utc").mjd) for t in iso],
        dtype=np.float64,
    )
    xr.Dataset(
        {"x": ("time", np.zeros(len(mjd), dtype=np.float32))},
        coords={"time": mjd},
    ).to_zarr(str(out_zarr), mode="w")


def test_run_cascade_per_time_group_skips_completed_time_keys(tmp_path: Path) -> None:
    """When ``out_zarr`` already contains a time key, the cascade must not run for it."""
    cascade_calls: list[str] = []

    def fake_cascade(*, image_filenames: Sequence[str], outroot: str, **_kw: Any) -> None:
        cascade_calls.append(Path(outroot).name)
        out = Path(outroot)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_filenames[0], out / "band.fits")

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_minimal_fits(
        raw / "done_73MHz.fits", restfreq_hz=73e6, date_obs="2024-06-01T12:00:00.0"
    )
    _write_minimal_fits(
        raw / "todo_73MHz.fits", restfreq_hz=73e6, date_obs="2024-06-02T12:00:00.0"
    )

    out_zarr = tmp_path / "out" / "store.zarr"
    out_zarr.parent.mkdir(parents=True, exist_ok=True)
    _seed_zarr_with_time_keys(out_zarr, ["20240601_120000"])

    n, keys, _lm_ref = run_cascade_per_time_group(
        raw,
        tmp_path / "cascade",
        tmp_path / "staging",
        discovery_freq_bin_hz=23e3,
        cascade_fn=fake_cascade,
        out_zarr=out_zarr,
        rebuild=False,
    )

    assert keys == ["20240602_120000"]
    assert n == 1
    assert cascade_calls == ["20240602_120000"]


def test_run_cascade_per_time_group_resume_full_is_noop(tmp_path: Path) -> None:
    """When every discovered key is already in the store, the cascade is not invoked."""
    def explode(**_kw: Any) -> None:
        raise AssertionError("cascade_fn must not run when every time key is already in zarr.")

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_minimal_fits(
        raw / "only_73MHz.fits", restfreq_hz=73e6, date_obs="2024-06-01T12:00:00.0"
    )

    out_zarr = tmp_path / "out" / "store.zarr"
    out_zarr.parent.mkdir(parents=True, exist_ok=True)
    _seed_zarr_with_time_keys(out_zarr, ["20240601_120000"])

    n, keys, _lm_ref = run_cascade_per_time_group(
        raw,
        tmp_path / "cascade",
        tmp_path / "staging",
        discovery_freq_bin_hz=23e3,
        cascade_fn=explode,
        out_zarr=out_zarr,
        rebuild=False,
    )

    assert n == 0
    assert keys == []


def test_run_cascade_per_time_group_rebuild_ignores_resume(tmp_path: Path) -> None:
    """``rebuild=True`` must bypass the resume filter so every time key is dewarped."""
    cascade_calls: list[str] = []

    def fake_cascade(*, image_filenames: Sequence[str], outroot: str, **_kw: Any) -> None:
        cascade_calls.append(Path(outroot).name)
        out = Path(outroot)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_filenames[0], out / "band.fits")

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_minimal_fits(
        raw / "one_73MHz.fits", restfreq_hz=73e6, date_obs="2024-06-01T12:00:00.0"
    )

    out_zarr = tmp_path / "out" / "store.zarr"
    out_zarr.parent.mkdir(parents=True, exist_ok=True)
    _seed_zarr_with_time_keys(out_zarr, ["20240601_120000"])

    n, keys, _lm_ref = run_cascade_per_time_group(
        raw,
        tmp_path / "cascade",
        tmp_path / "staging",
        discovery_freq_bin_hz=23e3,
        cascade_fn=fake_cascade,
        out_zarr=out_zarr,
        rebuild=True,
    )

    assert keys == ["20240601_120000"]
    assert n == 1
    assert cascade_calls == ["20240601_120000"]


def test_dewarp_and_convert_append_each_time_skips_completed_time_keys(
    tmp_path: Path, monkeypatch
) -> None:
    """Per-step append mode must skip time keys already in the existing Zarr."""
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    cascade = tmp_path / "cascade"
    staging = tmp_path / "staging"
    fixed = tmp_path / "fixed"
    for d in (raw, out, cascade, staging, fixed):
        d.mkdir()

    _write_minimal_fits(raw / "a_73MHz.fits", restfreq_hz=73e6, date_obs="2024-06-01T12:00:00.0")
    _write_minimal_fits(raw / "b_73MHz.fits", restfreq_hz=73e6, date_obs="2024-06-02T12:00:00.0")

    out_zarr = out / "z.zarr"
    _seed_zarr_with_time_keys(out_zarr, ["20240601_120000"])

    def fake_discover(*_a: object, **_k: object) -> dict[str, list[Path]]:
        return {
            "20240601_120000": [raw / "a_73MHz.fits"],
            "20240602_120000": [raw / "b_73MHz.fits"],
        }

    monkeypatch.setattr(dewarp_convert_mod, "discover_time_grouped_fits", fake_discover)
    mref = MagicMock()
    mref.copy = MagicMock(return_value=mref)
    monkeypatch.setattr(
        dewarp_convert_mod,
        "_load_global_lm_reference_dataset",
        lambda *a, **k: mref,
    )

    cascade_calls: list[str] = []

    def fake_run_cascade(
        tkey: str,
        files: Sequence[Path],
        _cascade_parent: Path,
        staging_dir: Path,
        *,
        cascade_fn: Any,
        **_kw: Any,
    ) -> int:
        cascade_calls.append(tkey)
        (staging_dir / f"{tkey}__out.fits").touch()
        return 1

    monkeypatch.setattr(dewarp_convert_mod, "run_cascade_for_time_key", fake_run_cascade)

    with patch("ovro_lwa_portal.ingest.core.FITSToZarrConverter") as conv_cls:
        inst = MagicMock()
        conv_cls.return_value = inst
        n_staged, keys = dewarp_and_convert_append_each_time(
            raw,
            out,
            cascade,
            staging,
            fixed,
            zarr_name="z.zarr",
            chunk_lm=64,
            rebuild=False,
            fix_headers_on_demand=False,
            cleanup_fixed_fits=False,
            discovery_freq_bin_hz=23e3,
            duplicate_resolver=None,
            cascade_fn=lambda **kw: None,
        )

    assert keys == ["20240602_120000"]
    assert n_staged == 1
    assert cascade_calls == ["20240602_120000"]
    assert inst.convert.call_count == 1
    cfg = conv_cls.call_args_list[0][0][0]
    assert cfg.time_keys_only == ("20240602_120000",)
    # The store already exists, so the converter must be told to append, not overwrite.
    assert cfg.rebuild is False


def test_dewarp_and_convert_append_each_time_resume_full_is_noop(
    tmp_path: Path, monkeypatch
) -> None:
    """If every discovered key is already in the Zarr, no cascade / convert runs."""
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    cascade = tmp_path / "cascade"
    staging = tmp_path / "staging"
    fixed = tmp_path / "fixed"
    for d in (raw, out, cascade, staging, fixed):
        d.mkdir()

    _write_minimal_fits(raw / "a_73MHz.fits", restfreq_hz=73e6, date_obs="2024-06-01T12:00:00.0")

    out_zarr = out / "z.zarr"
    _seed_zarr_with_time_keys(out_zarr, ["20240601_120000"])

    monkeypatch.setattr(
        dewarp_convert_mod,
        "discover_time_grouped_fits",
        lambda *a, **k: {"20240601_120000": [raw / "a_73MHz.fits"]},
    )
    mref = MagicMock()
    mref.copy = MagicMock(return_value=mref)
    monkeypatch.setattr(
        dewarp_convert_mod,
        "_load_global_lm_reference_dataset",
        lambda *a, **k: mref,
    )

    def explode_run_cascade(*_a: object, **_k: object) -> int:
        raise AssertionError("run_cascade_for_time_key must not run when nothing is to do.")

    monkeypatch.setattr(dewarp_convert_mod, "run_cascade_for_time_key", explode_run_cascade)

    with patch("ovro_lwa_portal.ingest.core.FITSToZarrConverter") as conv_cls:
        n_staged, keys = dewarp_and_convert_append_each_time(
            raw,
            out,
            cascade,
            staging,
            fixed,
            zarr_name="z.zarr",
            chunk_lm=64,
            rebuild=False,
            fix_headers_on_demand=False,
            cleanup_fixed_fits=False,
            discovery_freq_bin_hz=23e3,
            duplicate_resolver=None,
            cascade_fn=lambda **kw: None,
        )

    assert keys == []
    assert n_staged == 0
    conv_cls.assert_not_called()


# -------------------------------------------------------------------------
# Cascade-reference (73 MHz) filter
# -------------------------------------------------------------------------


def test_file_subband_mhz_parses_prefix_and_underscore_tokens(tmp_path: Path) -> None:
    """Both ``NNNMHz-...`` (prefix) and ``..._NNNMHz_...`` styles must parse."""
    fn = dewarp_convert_mod._file_subband_mhz
    assert fn(tmp_path / "73MHz-I-Deep-Taper-Robust-0-image-X.fits") == 73
    assert fn(tmp_path / "ovro_73MHz_averaged_iter.fits") == 73
    assert fn(tmp_path / "ovro-73MHz-image.fits") == 73
    assert fn(tmp_path / "stage__73MHz_image.fits") == 73
    assert fn(tmp_path / "18MHz_band.fits") == 18


def test_file_subband_mhz_rejects_non_boundary_match(tmp_path: Path) -> None:
    """``1473MHz`` and similar must not be misread as 73 MHz."""
    fn = dewarp_convert_mod._file_subband_mhz
    assert fn(tmp_path / "1473MHz-image.fits") == 1473
    # No leading separator; the regex requires ^ / _ / - before the digits,
    # so a token like ``foo73MHz_x.fits`` does not parse a subband.
    assert fn(tmp_path / "foo73MHz_x.fits") is None
    assert fn(tmp_path / "no_subband_tag.fits") is None


def test_has_cascade_reference_subband_positive_and_negative(tmp_path: Path) -> None:
    """At least one path must carry the ``73MHz`` token for the group to be cascadable."""
    fn = dewarp_convert_mod._has_cascade_reference_subband
    assert fn(
        [tmp_path / "73MHz-image.fits", tmp_path / "41MHz_other.fits"],
    )
    assert not fn(
        [tmp_path / "41MHz_a.fits", tmp_path / "82MHz_b.fits"],
    )
    # Custom target stays orthogonal to the default 73 MHz path.
    assert fn(
        [tmp_path / "41MHz_a.fits", tmp_path / "82MHz_b.fits"],
        target_mhz=82,
    )


def test_filter_time_groups_without_cascade_reference_drops_groups(
    tmp_path: Path, caplog
) -> None:
    """Groups missing a 73 MHz reference are dropped with a warning."""
    import logging

    fn = dewarp_convert_mod._filter_time_groups_without_cascade_reference
    by_time = {
        "20240601_120000": [
            tmp_path / "41MHz_a.fits",
            tmp_path / "73MHz_b.fits",
            tmp_path / "82MHz_c.fits",
        ],
        "20240602_120000": [
            tmp_path / "41MHz_a.fits",
            tmp_path / "82MHz_b.fits",
        ],
    }
    caplog.set_level(logging.WARNING, logger="ovro_lwa_portal.ingest.dewarp_convert")
    filtered = fn(by_time)

    assert "20240601_120000" in filtered
    assert "20240602_120000" not in filtered
    assert "no 73MHz reference image found" in caplog.text
    assert "[41, 82]" in caplog.text


def test_filter_time_groups_without_cascade_reference_returns_empty_when_no_groups_cascadable(
    tmp_path: Path,
) -> None:
    """If no group has the 73 MHz reference, the filter returns an empty dict."""
    fn = dewarp_convert_mod._filter_time_groups_without_cascade_reference
    by_time = {
        "20240601_120000": [tmp_path / "41MHz_a.fits"],
        "20240602_120000": [tmp_path / "82MHz_b.fits"],
    }
    assert fn(by_time) == {}


def test_run_cascade_per_time_group_skips_groups_without_73mhz(
    tmp_path: Path, caplog
) -> None:
    """Time groups whose files have no ``73MHz`` subband tag must be dropped pre-cascade."""
    import logging

    cascade_calls: list[str] = []

    def fake_cascade(*, image_filenames: Sequence[str], outroot: str, **_kw: Any) -> None:
        cascade_calls.append(Path(outroot).name)
        out = Path(outroot)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_filenames[0], out / "band.fits")

    raw = tmp_path / "raw"
    raw.mkdir()
    # Time A has a 73 MHz reference → keep.
    _write_minimal_fits(
        raw / "a_41MHz.fits", restfreq_hz=41e6, date_obs="2024-06-01T12:00:00.0"
    )
    _write_minimal_fits(
        raw / "a_73MHz.fits", restfreq_hz=73e6, date_obs="2024-06-01T12:00:00.0"
    )
    # Time B has no 73 MHz reference → drop.
    _write_minimal_fits(
        raw / "b_41MHz.fits", restfreq_hz=41e6, date_obs="2024-06-02T12:00:00.0"
    )
    _write_minimal_fits(
        raw / "b_82MHz.fits", restfreq_hz=82e6, date_obs="2024-06-02T12:00:00.0"
    )

    caplog.set_level(logging.WARNING, logger="ovro_lwa_portal.ingest.dewarp_convert")
    n, keys, _lm_ref = run_cascade_per_time_group(
        raw,
        tmp_path / "cascade",
        tmp_path / "staging",
        discovery_freq_bin_hz=23e3,
        cascade_fn=fake_cascade,
    )

    assert keys == ["20240601_120000"]
    assert n == 1
    assert cascade_calls == ["20240601_120000"]
    assert "Skipping time=20240602_120000" in caplog.text
    assert "no 73MHz reference image found" in caplog.text


def test_run_cascade_per_time_group_returns_empty_when_no_group_has_73mhz(
    tmp_path: Path, caplog
) -> None:
    """If every discovered group lacks a 73 MHz file, the cascade is not invoked."""
    import logging

    def explode(**_kw: Any) -> None:
        raise AssertionError("cascade_fn must not run when no group has a 73 MHz reference.")

    raw = tmp_path / "raw"
    raw.mkdir()
    _write_minimal_fits(
        raw / "a_41MHz.fits", restfreq_hz=41e6, date_obs="2024-06-01T12:00:00.0"
    )
    _write_minimal_fits(
        raw / "a_82MHz.fits", restfreq_hz=82e6, date_obs="2024-06-01T12:00:00.0"
    )

    caplog.set_level(logging.WARNING, logger="ovro_lwa_portal.ingest.dewarp_convert")
    n, keys, _lm_ref = run_cascade_per_time_group(
        raw,
        tmp_path / "cascade",
        tmp_path / "staging",
        discovery_freq_bin_hz=23e3,
        cascade_fn=explode,
    )

    assert n == 0
    assert keys == []
    assert "no 73MHz reference image found" in caplog.text


def test_dewarp_and_convert_append_each_time_skips_groups_without_73mhz(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """Per-step append mode must also skip time groups missing the 73 MHz reference."""
    import logging

    raw = tmp_path / "raw"
    out = tmp_path / "out"
    cascade = tmp_path / "cascade"
    staging = tmp_path / "staging"
    fixed = tmp_path / "fixed"
    for d in (raw, out, cascade, staging, fixed):
        d.mkdir()

    _write_minimal_fits(raw / "a_73MHz.fits", restfreq_hz=73e6, date_obs="2024-06-01T12:00:00.0")
    _write_minimal_fits(raw / "b_82MHz.fits", restfreq_hz=82e6, date_obs="2024-06-02T12:00:00.0")

    def fake_discover(*_a: object, **_k: object) -> dict[str, list[Path]]:
        return {
            "20240601_120000": [raw / "a_73MHz.fits"],   # keep (has 73 MHz)
            "20240602_120000": [raw / "b_82MHz.fits"],   # drop (no 73 MHz)
        }

    monkeypatch.setattr(dewarp_convert_mod, "discover_time_grouped_fits", fake_discover)
    mref = MagicMock()
    mref.copy = MagicMock(return_value=mref)
    monkeypatch.setattr(
        dewarp_convert_mod,
        "_load_global_lm_reference_dataset",
        lambda *a, **k: mref,
    )

    cascade_calls: list[str] = []

    def fake_run_cascade(
        tkey: str,
        files: Sequence[Path],
        _cascade_parent: Path,
        staging_dir: Path,
        *,
        cascade_fn: Any,
        **_kw: Any,
    ) -> int:
        cascade_calls.append(tkey)
        (staging_dir / f"{tkey}__out.fits").touch()
        return 1

    monkeypatch.setattr(dewarp_convert_mod, "run_cascade_for_time_key", fake_run_cascade)

    caplog.set_level(logging.WARNING, logger="ovro_lwa_portal.ingest.dewarp_convert")
    with patch("ovro_lwa_portal.ingest.core.FITSToZarrConverter") as conv_cls:
        inst = MagicMock()
        conv_cls.return_value = inst
        n_staged, keys = dewarp_and_convert_append_each_time(
            raw,
            out,
            cascade,
            staging,
            fixed,
            zarr_name="z.zarr",
            chunk_lm=64,
            rebuild=True,
            fix_headers_on_demand=False,
            cleanup_fixed_fits=False,
            discovery_freq_bin_hz=23e3,
            duplicate_resolver=None,
            cascade_fn=lambda **kw: None,
        )

    assert keys == ["20240601_120000"]
    assert n_staged == 1
    assert cascade_calls == ["20240601_120000"]
    assert inst.convert.call_count == 1
    cfg = conv_cls.call_args_list[0][0][0]
    assert cfg.time_keys_only == ("20240601_120000",)
    assert "Skipping time=20240602_120000" in caplog.text


def test_dewarp_and_convert_append_each_time_returns_when_no_group_has_73mhz(
    tmp_path: Path, monkeypatch
) -> None:
    """If no discovered group has a 73 MHz file, neither cascade nor convert runs."""
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    cascade = tmp_path / "cascade"
    staging = tmp_path / "staging"
    fixed = tmp_path / "fixed"
    for d in (raw, out, cascade, staging, fixed):
        d.mkdir()

    _write_minimal_fits(raw / "a_41MHz.fits", restfreq_hz=41e6, date_obs="2024-06-01T12:00:00.0")
    _write_minimal_fits(raw / "a_82MHz.fits", restfreq_hz=82e6, date_obs="2024-06-01T12:00:00.0")

    monkeypatch.setattr(
        dewarp_convert_mod,
        "discover_time_grouped_fits",
        lambda *a, **k: {
            "20240601_120000": [raw / "a_41MHz.fits", raw / "a_82MHz.fits"]
        },
    )

    def explode_lm_ref(*_a: object, **_k: object) -> Any:
        raise AssertionError("LM reference must not be built when no group is cascadable.")

    monkeypatch.setattr(
        dewarp_convert_mod, "_load_global_lm_reference_dataset", explode_lm_ref
    )

    def explode_run_cascade(*_a: object, **_k: object) -> int:
        raise AssertionError("run_cascade_for_time_key must not run.")

    monkeypatch.setattr(dewarp_convert_mod, "run_cascade_for_time_key", explode_run_cascade)

    with patch("ovro_lwa_portal.ingest.core.FITSToZarrConverter") as conv_cls:
        n_staged, keys = dewarp_and_convert_append_each_time(
            raw,
            out,
            cascade,
            staging,
            fixed,
            zarr_name="z.zarr",
            chunk_lm=64,
            rebuild=True,
            fix_headers_on_demand=False,
            cleanup_fixed_fits=False,
            discovery_freq_bin_hz=23e3,
            duplicate_resolver=None,
            cascade_fn=lambda **kw: None,
        )

    assert keys == []
    assert n_staged == 0
    conv_cls.assert_not_called()
