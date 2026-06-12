"""Tests for pipeline QA discovery, conversion helpers, and Panel app."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import pandas as pd
import param
import pytest

pn = pytest.importorskip("panel")
np = pytest.importorskip("numpy")
widgets = pytest.importorskip("ipywidgets")

from ovro_lwa_portal.viz import pipeline_qa as pq
from ovro_lwa_portal.viz.pipeline_qa_app import PipelineQAApp


def _mock_zenith_dataset(
    *,
    n_times: int = 4,
    n_freqs: int = 10,
    freq_hz: np.ndarray | None = None,
) -> Any:
    """Minimal dataset stand-in with time/frequency coordinates for zenith review tests."""
    if freq_hz is None:
        freq_hz = np.linspace(70e6, 90e6, n_freqs)
    time_mjd = np.linspace(60_000.0, 60_000.0 + 0.03 * max(n_times - 1, 0), n_times)

    class _Coord:
        def __init__(self, values: np.ndarray) -> None:
            self.values = values

    class _Coords:
        def __getitem__(self, key: str) -> _Coord:
            if key == "time":
                return _Coord(time_mjd)
            if key == "frequency":
                return _Coord(freq_hz)
            raise KeyError(key)

    class _Dataset:
        sizes = {"time": n_times, "frequency": n_freqs}
        coords = _Coords()
        frequency = _Coord(freq_hz)

    return _Dataset()


def _set_select_day(
    app: PipelineQAApp,
    day: str,
    *,
    days: list[str] | None = None,
) -> None:
    """Set select_day with valid Selector objects (avoids spurious load callbacks)."""
    options = days if days is not None else [day]
    with param.parameterized.batch_call_watchers(app):
        app._sync_day_selector(options, day)


def _write_qa_tree(root: Path, *, obs_date: str = "2024-12-28", hour: str = "08h") -> None:
    run_dir = root / hour / obs_date / "Run_20241228_120000"
    wideband = run_dir / "Wideband"
    wideband.mkdir(parents=True)
    (wideband / "thermal_noise_vs_subband.png").write_bytes(b"png")
    subband = run_dir / "82MHz" / "I" / "deep"
    subband.mkdir(parents=True)
    fits_name = "82MHz-I-Deep-Taper-Robust-0.75-image-20241228_120000.pbcorr.fits"
    (subband / fits_name).write_bytes(b"fits")


def _sample_coverage() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lst_hour": ["08h"],
            "obs_date": pd.to_datetime(["2024-12-28"]),
            "n_subbands": [1],
            "subbands": ["82MHz"],
            "latest_run": ["Run_20241228_120000"],
            "lst_hour_num": [8],
            "thermal_noise_png": ["/tmp/thermal.png"],
            "run_path": ["/tmp/run"],
        }
    )


def test_day_summary_table_columns(tmp_path: Path) -> None:
    png = tmp_path / "thermal.png"
    png.write_bytes(b"fakepng")
    coverage = _sample_coverage()
    coverage.loc[0, "thermal_noise_png"] = str(png)
    table = pq.day_summary_table("2024-12-28", coverage)
    assert list(table.columns) == [
        "lst_hour",
        "n_subbands",
        "thermal_noise_png",
    ]
    assert table.iloc[0]["thermal_noise_png"] == str(png)


def test_build_thermal_noise_grid(tmp_path: Path) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import build_thermal_noise_grid

    png = tmp_path / "thermal.png"
    png.write_bytes(b"fakepng")
    summary = pd.DataFrame(
        {
            "lst_hour": ["08h", "09h"],
            "n_subbands": [3, 1],
            "thermal_noise_png": [str(png), str(tmp_path / "missing.png")],
        }
    )
    grid = build_thermal_noise_grid(summary, "2024-12-28", n_cols=2)
    assert isinstance(grid, pn.Column)
    assert len(grid.objects) == 1
    row = grid.objects[0]
    assert isinstance(row, pn.Row)
    assert len(row.objects) == 2


def test_scan_coverage_uses_config_pipeline_root(tmp_path: Path) -> None:
    _write_qa_tree(tmp_path)
    cfg = pq.PipelineQAConfig(
        pipeline_root=tmp_path,
        symlink_root=tmp_path / "stage",
        zarr_root=tmp_path / "zarr",
        i_fits_glob=pq.I_FITS_GLOB,
        v_fits_glob=pq.V_FITS_GLOB,
    )
    coverage = pq.scan_coverage(config=cfg)
    assert coverage.iloc[0]["latest_run"] == "Run_20241228_120000"


def test_collect_pol_fits_uses_config_glob(tmp_path: Path) -> None:
    _write_qa_tree(tmp_path)
    cfg = pq.PipelineQAConfig(
        pipeline_root=tmp_path,
        symlink_root=tmp_path / "stage",
        zarr_root=tmp_path / "zarr",
        i_fits_glob="*Robust-0.75-image*.pbcorr.fits",
        v_fits_glob=pq.V_FITS_GLOB,
    )
    coverage = pq.scan_coverage(config=cfg)
    paths = pq.collect_pol_fits("2024-12-28", "I", coverage, config=cfg)
    assert len(paths) == 1
    assert paths[0].name.endswith(".pbcorr.fits")


def test_resolve_pipeline_qa_config_overrides_fields() -> None:
    custom_root = Path("/custom/root")
    custom_stage = Path("/custom/stage")
    custom_zarr = Path("/custom/zarr")
    cfg = pq.resolve_pipeline_qa_config(
        pipeline_root=custom_root,
        symlink_root=custom_stage,
        zarr_root=custom_zarr,
        i_fits_glob="*custom-I*.fits",
    )
    assert cfg.pipeline_root == custom_root
    assert cfg.symlink_root == custom_stage
    assert cfg.zarr_root == custom_zarr
    assert cfg.i_fits_glob == "*custom-I*.fits"
    assert cfg.v_fits_glob == pq.V_FITS_GLOB


def test_qa_zarr_path_uses_config_zarr_root(tmp_path: Path) -> None:
    cfg = pq.PipelineQAConfig(
        pipeline_root=tmp_path,
        symlink_root=tmp_path / "stage",
        zarr_root=tmp_path / "zarr",
        i_fits_glob=pq.I_FITS_GLOB,
        v_fits_glob=pq.V_FITS_GLOB,
    )
    path = pq.qa_zarr_path("I", "2024-12-28", config=cfg)
    assert path.parent == tmp_path / "zarr"
    assert path.name == "pipelineQA-I-Deep-Taper-Robust-0.75-20241228.zarr"


def test_scan_coverage_finds_wideband_runs(tmp_path: Path) -> None:
    _write_qa_tree(tmp_path)
    coverage = pq.scan_coverage(tmp_path)
    assert len(coverage) == 1
    assert coverage.iloc[0]["latest_run"] == "Run_20241228_120000"


def test_scan_coverage_defers_subband_listing(tmp_path: Path) -> None:
    _write_qa_tree(tmp_path)
    coverage = pq.scan_coverage(tmp_path)
    assert pd.isna(coverage.iloc[0]["n_subbands"])
    assert pd.isna(coverage.iloc[0]["subbands"])

    pq.populate_subbands_for_day(coverage, "2024-12-28")
    assert int(coverage.iloc[0]["n_subbands"]) == 1
    assert coverage.iloc[0]["subbands"] == "82MHz"

    table = pq.day_summary_table("2024-12-28", coverage)
    assert int(table.iloc[0]["n_subbands"]) == 1


def test_populate_subbands_for_day_only_touches_selected_day(tmp_path: Path) -> None:
    for obs_date in ("2024-12-27", "2024-12-28"):
        run_dir = tmp_path / "08h" / obs_date / f"Run_{obs_date.replace('-', '')}_120000"
        wideband = run_dir / "Wideband"
        wideband.mkdir(parents=True)
        (wideband / "thermal_noise_vs_subband.png").write_bytes(b"png")
        subband = run_dir / "82MHz" / "I" / "deep"
        subband.mkdir(parents=True)
        if obs_date == "2024-12-28":
            (run_dir / "23MHz").mkdir(parents=True)

    coverage = pq.scan_coverage(tmp_path)
    pq.populate_subbands_for_day(coverage, "2024-12-28")

    row27 = coverage.loc[coverage["obs_date"].dt.strftime("%Y-%m-%d") == "2024-12-27"].iloc[0]
    row28 = coverage.loc[coverage["obs_date"].dt.strftime("%Y-%m-%d") == "2024-12-28"].iloc[0]
    assert pd.isna(row27["n_subbands"])
    assert int(row28["n_subbands"]) == 2
    assert row28["subbands"] == "23MHz, 82MHz"


def test_default_select_day_prefers_earliest_with_i_zarr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pq, "ZARR_ROOT", tmp_path)
    coverage = pd.DataFrame(
        {
            "obs_date": pd.to_datetime(["2024-12-27", "2024-12-28"]),
            "latest_run": ["Run_a", "Run_b"],
            "lst_hour": ["08h", "08h"],
            "lst_hour_num": [8, 8],
        }
    )
    i_zarr = pq.qa_zarr_path("I", "2024-12-28")
    i_zarr.mkdir(parents=True)
    (i_zarr / ".zgroup").write_text("{}")
    assert pq.default_select_day(coverage) == "2024-12-28"


def test_zarr_status_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pq, "ZARR_ROOT", tmp_path)
    i_zarr = pq.qa_zarr_path("I", "2024-12-28")
    i_zarr.mkdir(parents=True)
    (i_zarr / ".zgroup").write_text("{}")
    monkeypatch.setattr(
        pq,
        "_zarr_store_exists",
        lambda path: path.name.endswith("20241228.zarr"),
    )
    status = pq.zarr_status("2024-12-28")
    assert status == {"I": True, "V": False}


def test_stage_symlinks_avoids_basename_collisions(tmp_path: Path) -> None:
    run_a = tmp_path / "08h" / "2024-12-28" / "Science_20241228_120000" / "82MHz" / "I" / "deep"
    run_b = tmp_path / "09h" / "2024-12-28" / "Science_20241228_130000" / "82MHz" / "I" / "deep"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    name = "82MHz-I-NoTaper-3581s-Robust-0-20241218_033402-image.pbcorr_dewarped.fits"
    file_a = run_a / name
    file_b = run_b / name
    file_a.write_bytes(b"fits-a")
    file_b.write_bytes(b"fits-b")

    staging = tmp_path / "stage"
    pq.stage_symlinks([file_a, file_b], staging)

    assert (staging / name).is_symlink()
    assert (staging / f"Science_20241228_130000__{name}").is_symlink()


def test_fits_group_key_strips_run_prefix() -> None:
    name = (
        "Science_20241228_130000__"
        "82MHz-I-NoTaper-3581s-Robust-0-20241218_033402-image.pbcorr_dewarped.fits"
    )
    assert pq.fits_group_key(Path(name)) == ("82MHz", "20241218_033402")


def test_convert_missing_zarr_cleans_up_staging_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_root = tmp_path / "stage"
    zarr_root = tmp_path / "zarr"
    cfg = pq.PipelineQAConfig(
        pipeline_root=tmp_path,
        symlink_root=stage_root,
        zarr_root=zarr_root,
        i_fits_glob=pq.I_FITS_GLOB,
        v_fits_glob=pq.V_FITS_GLOB,
    )
    _write_qa_tree(tmp_path)
    v_subband = (
        tmp_path
        / "08h"
        / "2024-12-28"
        / "Run_20241228_120000"
        / "82MHz"
        / "V"
        / "deep"
    )
    v_subband.mkdir(parents=True)
    v_name = "82MHz-V-Taper-Deep-image-20241228_120000.pbcorr.fits"
    (v_subband / v_name).write_bytes(b"fits")

    day_tag = "20241228"
    staging_i = stage_root / f"{cfg.i_qa_zarr_stem}-{day_tag}-fits"
    fixed_i = stage_root / f"{cfg.i_qa_zarr_stem}-{day_tag}-fixed"
    staging_v = stage_root / f"{cfg.v_qa_zarr_stem}-{day_tag}-fits"
    fixed_v = stage_root / f"{cfg.v_qa_zarr_stem}-{day_tag}-fixed"

    def _fake_convert(*, input_dir: Path, out_dir: Path, zarr_name: str, **kwargs: object) -> Path:
        out = out_dir / zarr_name
        out.mkdir(parents=True, exist_ok=True)
        (out / ".zgroup").write_text("{}")
        fixed_dir = kwargs.get("fixed_dir")
        if isinstance(fixed_dir, Path):
            fixed_dir.mkdir(parents=True, exist_ok=True)
            (fixed_dir / "stub_fixed.fits").write_bytes(b"fixed")
        return out

    def _fake_stage_symlinks(fits_paths: list[Path], staging_dir: Path) -> Path:
        staging_dir.mkdir(parents=True, exist_ok=True)
        for src in fits_paths:
            (staging_dir / src.name).write_bytes(b"fits")
        return staging_dir

    def _fake_stage_v(
        v_paths: list[Path],
        _i_paths: list[Path],
        staging_dir: Path,
    ) -> Path:
        staging_dir.mkdir(parents=True, exist_ok=True)
        for src in v_paths:
            (staging_dir / src.name).write_bytes(b"fits")
        return staging_dir

    monkeypatch.setattr(pq, "infer_target_size_from_82mhz", lambda *_args, **_kwargs: 64)
    monkeypatch.setattr(pq, "convert_fits_dir_to_zarr", _fake_convert)
    monkeypatch.setattr(pq, "stage_symlinks", _fake_stage_symlinks)
    monkeypatch.setattr(pq, "stage_v_fits_with_beam_from_i", _fake_stage_v)
    monkeypatch.setattr(pq, "_lm_reference_from_existing_zarr", lambda _path: object())

    calls: list[str] = []
    coverage = pq.scan_coverage(config=cfg)
    pq.convert_missing_zarr("2024-12-28", coverage, calls.append, config=cfg)

    for path in (staging_i, fixed_i, staging_v, fixed_v):
        assert not path.exists(), f"expected staging dir removed: {path}"
    assert any("Removed staging directory" in line for line in calls)


def test_convert_missing_zarr_skips_existing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pq, "ZARR_ROOT", tmp_path)
    i_zarr = pq.qa_zarr_path("I", "2024-12-28")
    v_zarr = pq.qa_zarr_path("V", "2024-12-28")
    i_zarr.mkdir(parents=True)
    v_zarr.mkdir(parents=True)
    (i_zarr / ".zgroup").write_text("{}")
    (v_zarr / ".zgroup").write_text("{}")

    calls: list[str] = []

    def _fail_convert(*_args, **_kwargs):
        raise AssertionError("convert_fits_dir_to_zarr should not be called")

    monkeypatch.setattr(pq, "convert_fits_dir_to_zarr", _fail_convert)
    paths = pq.convert_missing_zarr("2024-12-28", _sample_coverage(), calls.append)
    assert paths["I"] == i_zarr
    assert paths["V"] == v_zarr
    assert any("Using existing Zarr" in line for line in calls)


def test_pipeline_qa_app_panel_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        PipelineQAApp,
        "_start_initial_scan",
        lambda self, **kwargs: None,
    )
    app = PipelineQAApp()
    layout = app.panel()
    assert isinstance(layout, pn.Column)
    assert isinstance(app._zenith_review_row, pn.Row)
    assert set(app._zenith_section_content) == {"I", "V"}
    assert app._flux_ratio_grid in layout.objects
    assert app._stokes_review.heatmap_status_row in app._zenith_slot.objects
    assert app._stokes_review.zenith_footer in app._zenith_slot.objects
    zenith_objects = list(app._zenith_slot.objects)
    assert zenith_objects.index(app._stokes_review.heatmap_status_row) < zenith_objects.index(
        app._zenith_review_row
    )
    assert zenith_objects.index(app._zenith_review_row) < zenith_objects.index(
        app._stokes_review.zenith_footer
    )


def test_apply_day_payload_builds_flux_ratio_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import _DayLoadPayload

    _write_flux_check_tree(tmp_path)
    app = PipelineQAApp()
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)
    app._coverage = pq.scan_coverage(tmp_path)
    app._reset_zenith_sections = lambda **kwargs: None  # type: ignore[method-assign]
    app._push_panel_roots = lambda: None  # type: ignore[method-assign]
    app._execute = lambda callback: callback()  # type: ignore[method-assign]

    summary = pq.day_summary_table("2024-12-28", app._coverage)
    payload = _DayLoadPayload(select_day="2024-12-28", summary_df=summary)
    app._apply_day_payload(payload)

    assert len(app._flux_ratio_grid.objects) == 1
    grid = app._flux_ratio_grid.objects[0]
    assert isinstance(grid, pn.Column)
    assert len(grid.objects) >= 1


def test_display_pipeline_qa_app_displays_single_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import display_pipeline_qa_app

    displayed: list[Any] = []

    def _display(obj: Any, **_kwargs: Any) -> None:
        displayed.append(obj)

    monkeypatch.setattr("IPython.display.display", _display)
    monkeypatch.setattr(
        PipelineQAApp,
        "_start_initial_scan",
        lambda self, **kwargs: None,
    )

    display_pipeline_qa_app()

    assert len(displayed) == 1
    assert isinstance(displayed[0], pn.Column)


def test_format_activity_log_display_newest_first() -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import _format_activity_log_display

    text = "[12:00:01] first\n[12:00:02] second\n"
    assert _format_activity_log_display(text) == "[12:00:02] second\n[12:00:01] first"


def test_format_activity_log_html_is_scrollable() -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import (
        ACTIVITY_LOG_HEIGHT_PX,
        _format_activity_log_html,
    )

    html = _format_activity_log_html("[12:00:01] hello")
    assert f"height:{ACTIVITY_LOG_HEIGHT_PX}px" in html
    assert "overflow-y:auto" in html
    assert "hello" in html


def test_pipeline_qa_app_activity_log_is_fixed_height(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import ACTIVITY_LOG_HEIGHT_PX, PipelineQAApp

    app = PipelineQAApp()
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)

    assert isinstance(app._log_pane, pn.pane.HTML)
    assert app._log_pane.height == ACTIVITY_LOG_HEIGHT_PX


def test_convert_button_label_and_disabled() -> None:
    assert pq.convert_button_label({"I": True, "V": True}) == "Convert FITS → Zarr (complete)"
    assert pq.convert_button_label({"I": True, "V": False}) == "Convert Stokes V"
    assert pq.convert_button_disabled({"I": True, "V": True}, converting=False) is True
    assert pq.convert_button_disabled({"I": False, "V": False}, converting=False) is False


def test_refresh_convert_button_uses_primary_when_zarr_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = PipelineQAApp()
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)
    monkeypatch.setattr(app, "_begin_load_day", lambda: None)
    app.scanning = False
    app._coverage = pd.DataFrame({"obs_date": ["2024-12-27"]})
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.zarr_status",
        lambda _day, **kwargs: {"I": False, "V": False},
    )
    _set_select_day(app, "2024-12-27")

    app._sync_action_controls()

    assert app._convert_button.button_type == "primary"
    assert app._convert_button.disabled is False


def test_refresh_convert_button_uses_default_when_zarr_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = PipelineQAApp()
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)
    monkeypatch.setattr(app, "_begin_load_day", lambda: None)
    app.scanning = False
    app._coverage = pd.DataFrame({"obs_date": ["2024-12-27"]})
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.zarr_status",
        lambda _day, **kwargs: {"I": True, "V": True},
    )
    _set_select_day(app, "2024-12-27")

    app._sync_action_controls()

    assert app._convert_button.button_type == "default"
    assert app._convert_button.disabled is True


def test_initial_scan_does_not_auto_select_day(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    app = PipelineQAApp()
    coverage = _sample_coverage()
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.scan_coverage",
        lambda *args, **kwargs: coverage,
    )
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.qa_days",
        lambda _coverage: ["2024-12-28"],
    )
    monkeypatch.setattr(app, "_execute", lambda callback: callback())

    class _InlineThread:
        def __init__(self, *, target: Callable[[], None] | None = None, **_kwargs: object) -> None:
            self._target = target

        def start(self) -> None:
            if self._target is not None:
                self._target()

    monkeypatch.setattr(threading, "Thread", _InlineThread)

    load_calls: list[str] = []
    monkeypatch.setattr(app, "_begin_load_day", lambda: load_calls.append("load"))

    app._start_initial_scan()

    assert app.select_day is None
    assert "2024-12-28" in app.param.select_day.objects
    assert load_calls == []
    assert "Select a day" in app.log_text


def test_initial_scan_loads_day_selected_during_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    app = PipelineQAApp()
    coverage = _sample_coverage()
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.scan_coverage",
        lambda *args, **kwargs: coverage,
    )
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.qa_days",
        lambda _coverage: ["2024-12-28"],
    )
    monkeypatch.setattr(app, "_execute", lambda callback: callback())

    class _InlineThread:
        def __init__(self, *, target: Callable[[], None] | None = None, **_kwargs: object) -> None:
            self._target = target

        def start(self) -> None:
            if self._target is not None:
                self._target()

    monkeypatch.setattr(threading, "Thread", _InlineThread)

    load_calls: list[str] = []
    monkeypatch.setattr(app, "_begin_load_day", lambda: load_calls.append("load"))
    app.select_day = "2024-12-28"

    app._start_initial_scan()

    assert app.select_day == "2024-12-28"
    assert load_calls == ["load"]
    assert "Loading QA data for 2024-12-28" in app.log_text


def test_day_selector_triggers_load_day(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    app = PipelineQAApp()
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)
    app.scanning = False
    app._coverage = pd.DataFrame({"obs_date": ["2024-12-27", "2024-12-28"]})
    monkeypatch.setattr(
        app,
        "_start_day_load_thread",
        lambda select_day, load_seq: calls.append(select_day),
    )
    _set_select_day(app, "2024-12-27", days=["2024-12-27", "2024-12-28"])
    app._loaded_day = "2024-12-27"
    calls.clear()

    app._handle_day_selection("2024-12-28", previous="2024-12-27")

    assert app.select_day == "2024-12-28"
    assert calls == ["2024-12-28"]


def test_reselect_same_day_skips_when_already_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    app = PipelineQAApp()
    app.scanning = False
    app._coverage = _sample_coverage()
    monkeypatch.setattr(app, "_begin_load_day", lambda: calls.append("load"))
    _set_select_day(app, "2024-12-28", days=["2024-12-28"])
    app._loaded_day = "2024-12-28"
    calls.clear()

    app._handle_day_selection("2024-12-28", previous="2024-12-28")

    assert calls == []


def test_finish_zenith_load_skips_stale_load_seq() -> None:
    """Stale zenith finish must not clear ``loading_zenith`` owned by a newer load."""
    app = PipelineQAApp()
    app._load_seq = 2
    app.loading_zenith = True

    app._finish_zenith_load(load_seq=1)

    assert app.loading_zenith is True


def test_finish_zenith_load_clears_flag_for_current_seq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = PipelineQAApp()
    app._load_seq = 2
    app.loading_zenith = True
    monkeypatch.setattr(app, "_dispatch_ui", lambda callback: callback())

    app._finish_zenith_load(load_seq=2)

    assert app.loading_zenith is False


def test_day_change_during_zenith_load_starts_new_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calls: list[str] = []
    app = PipelineQAApp()
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)
    monkeypatch.setattr(
        app,
        "_start_day_load_thread",
        lambda select_day, load_seq: load_calls.append(select_day),
    )
    app.scanning = False
    app._coverage = _sample_coverage()
    app._loaded_day = "2024-12-19"
    app.loading_zenith = True
    _set_select_day(app, "2024-12-19", days=["2024-12-19", "2024-12-20"])
    load_calls.clear()

    app._handle_day_selection("2024-12-20", previous="2024-12-19")

    assert load_calls == ["2024-12-20"]
    assert app.loading_zenith is False


def test_day_selector_stays_enabled_during_zenith_load() -> None:
    app = PipelineQAApp()
    app.scanning = False
    app.converting = False
    app.loading_zenith = True
    app.loading_day = True

    app._sync_action_controls()

    assert app._day_selector.disabled is False


def test_stokes_review_holder_builds_both_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import _StokesReviewHolder

    holder = _StokesReviewHolder()
    stokes_order: list[str] = []

    def _record(spec, datasets, log, *, flush=None) -> pn.Column:
        stokes_order.append(spec.stokes)
        return pn.Column(pn.pane.Markdown(f"{spec.stokes} section"))

    monkeypatch.setattr(holder, "_build_section_content_for_spec", _record)
    column = holder.build_column({"I": object(), "V": object()}, lambda _message: None)  # type: ignore[arg-type]

    assert stokes_order == ["I", "V"]
    assert len(column.objects) == 1
    assert isinstance(column.objects[0], pn.Row)


def test_review_holder_builds_no_zarr_column() -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import _StokesReviewHolder

    holder = _StokesReviewHolder()
    column = holder.build_no_zarr_column()
    assert len(column.objects) == 1
    assert isinstance(column.objects[0], pn.Row)


def test_default_stat_slice_prefers_finite_preferred_frequency() -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import DEFAULT_FREQ_IDX, default_stat_slice

    stat_map = np.full((4, 10), np.nan)
    stat_map[2, DEFAULT_FREQ_IDX] = 0.5
    stat_map[0, 0] = 1.0

    class _Dataset:
        sizes = {"frequency": 10}

        class SKY:
            @staticmethod
            def isel(**_kwargs):
                raise AssertionError("fallback should not read SKY")

    assert default_stat_slice(stat_map, _Dataset()) == (2, DEFAULT_FREQ_IDX)


def test_default_stat_slice_falls_back_to_first_finite_cell() -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import default_stat_slice

    stat_map = np.full((3, 5), np.nan)
    stat_map[1, 2] = 0.2

    class _Dataset:
        sizes = {"frequency": 5}

        class SKY:
            @staticmethod
            def isel(**_kwargs):
                raise AssertionError("fallback should not read SKY")

    assert default_stat_slice(stat_map, _Dataset()) == (1, 2)


def test_heatmap_hover_source_includes_cell_values() -> None:
    from bokeh.models import HoverTool

    from ovro_lwa_portal.viz.pipeline_qa_app import (
        _ZenithHeatmapSelector,
        _build_heatmap_hover_source,
    )

    stat_map = np.array([[1.0, np.nan], [3.5, 4.0]])
    lst_hours = np.array([8.2, 10.7])
    freq_mhz = np.array([70.0, 80.0])
    source = _build_heatmap_hover_source(
        stat_map,
        lst_hours=lst_hours,
        freq_mhz=freq_mhz,
    )
    assert list(source.data["time_idx"]) == [0, 0, 1, 1]
    assert list(source.data["freq_idx"]) == [0, 1, 0, 1]
    assert source.data["lst_hour"][0] == "08h"
    assert source.data["lst_hour"][2] == "11h"
    assert source.data["freq_mhz"][1] == 80.0
    assert source.data["value"][0] == 1.0
    assert np.isnan(source.data["value"][1])

    heatmap = _ZenithHeatmapSelector(
        stat_map,
        metric_label="STD",
        lst_hours=lst_hours,
        freq_mhz=freq_mhz,
        on_select=lambda _t, _f: None,
    )
    assert any(isinstance(tool, HoverTool) for tool in heatmap._plot.tools)
    assert not hasattr(heatmap, "_marker")
    assert heatmap._plot.xaxis.axis_label == "LST hour"
    assert heatmap._plot.yaxis.axis_label == "Frequency (MHz)"
    assert heatmap._plot.xaxis.major_label_orientation == pytest.approx(np.pi / 4)


def test_heatmap_cell_center_and_index_from_coord() -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import (
        _heatmap_cell_center,
        _heatmap_index_from_coord,
    )

    assert _heatmap_cell_center(3) == 3.5
    assert _heatmap_index_from_coord(2.2, 10) == 2
    assert _heatmap_index_from_coord(2.9, 10) == 2
    assert _heatmap_index_from_coord(9.9, 10) == 9


def test_heatmap_axis_ticks_at_cell_centers() -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import (
        _format_freq_mhz_label,
        _format_lst_hour_label,
        _heatmap_axis_ticks,
    )

    ticks, labels = _heatmap_axis_ticks(5)
    assert ticks == [0.5, 1.5, 2.5, 3.5, 4.5]
    assert labels == {0.5: "0", 1.5: "1", 2.5: "2", 3.5: "3", 4.5: "4"}

    lst_hours = np.array([0.0, 2.4, 4.8, 7.2, 9.6])
    ticks, labels = _heatmap_axis_ticks(
        5,
        lst_hours,
        format_value=_format_lst_hour_label,
    )
    assert labels[0.5] == "00h"
    assert labels[2.5] == "05h"

    freq_mhz = np.array([70.0, 72.0, 74.0, 76.0, 78.0])
    _, labels = _heatmap_axis_ticks(
        5,
        freq_mhz,
        format_value=_format_freq_mhz_label,
    )
    assert labels[1.5] == "72.0"


def test_time_days_since_start_from_mjd() -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import _time_days_since_start

    mjd = np.array([60_000.0, 60_000.01, 60_000.03])
    days = _time_days_since_start(mjd)
    assert days.tolist() == pytest.approx([0.0, 0.01, 0.03])


def test_zenith_heatmap_lst_hours_from_mjd() -> None:
    from astropy import units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    from ovro_lwa_portal.viz.pipeline_qa_app import _zenith_heatmap_lst_hours

    dataset = _mock_zenith_dataset(n_times=2)
    lst_hours = _zenith_heatmap_lst_hours(dataset)
    observatory = EarthLocation(
        lat=37.2339 * u.deg, lon=-118.2817 * u.deg, height=1222 * u.m
    )
    expected = np.mod(
        np.asarray(
            Time(dataset.coords["time"].values, format="mjd", scale="utc")
            .sidereal_time("mean", longitude=observatory.lon)
            .deg,
            dtype=np.float64,
        )
        / 15.0,
        24.0,
    )
    assert lst_hours.tolist() == pytest.approx(expected.tolist())


def test_zenith_review_panel_slice_updates_status(monkeypatch: pytest.MonkeyPatch) -> None:
    import astropy.units as u
    import param

    from ovro_lwa_portal.viz.pipeline_qa_app import ZenithReviewPanel, ZenithSliceSelection

    class _Coord:
        ra = type("RA", (), {"to_string": lambda self, **kwargs: "12:00:00"})()
        dec = type("Dec", (), {"to_string": lambda self, **kwargs: "+00:00:00"})()

    _patch_zenith_status_center(monkeypatch)

    stat_map = np.ones((4, 10))
    freq_values = np.linspace(70e6, 90e6, 10)

    class _Radport:
        @staticmethod
        def _get_wcs(*, time_idx: int):
            class _WCS:
                @staticmethod
                def pixel_to_world(_l, _m):
                    class _PixelCoord:
                        ra = 0.0 * u.deg
                        dec = 0.0 * u.deg

                    return _PixelCoord()

            return _WCS()

        @staticmethod
        def nearest_lm_idx(_l, _m):
            return 0, 0

        @staticmethod
        def pixel_to_coords(l_idx, m_idx, *, time_idx):
            return 180.0, 45.0

    dataset = _mock_zenith_dataset(freq_hz=freq_values)
    dataset.radport = _Radport()

    scheduled: list[Callable[[], None]] = []
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app._schedule_ipython_main",
        lambda callback: scheduled.append(callback),
    )
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app._run_on_main_thread",
        lambda callback: callback(),
    )

    slice_selection = ZenithSliceSelection()
    slice_selection.configure(
        n_times=4,
        n_freqs=10,
        default_time=0,
        default_freq=8,
    )
    time_slider = pn.widgets.IntSlider.from_param(
        slice_selection.param.time_idx,
        name="Time index",
    )
    freq_slider = pn.widgets.IntSlider.from_param(
        slice_selection.param.freq_idx,
        name="Frequency index",
    )

    panel = ZenithReviewPanel(
        dataset,  # type: ignore[arg-type]
        stat_map,
        slice_selection=slice_selection,
        stokes_label="I",
        metric_label="STD",
    )

    def _run_scheduled() -> None:
        while scheduled:
            scheduled.pop(0)()

    _run_scheduled()
    assert panel.time_idx == 0 and panel.freq_idx == 8

    panel._select_slice(2, 5)
    _run_scheduled()

    assert slice_selection.time_idx == 2 and slice_selection.freq_idx == 5
    assert panel.time_idx == 2 and panel.freq_idx == 5
    assert "(2)" in panel._format_slice_status(2, 5)
    assert "(5)" in panel._format_slice_status(2, 5)
    assert " d (" in panel._format_slice_status(2, 5)
    assert " MHz (" in panel._format_slice_status(2, 5)
    assert "Stokes I" in panel._format_slice_status(2, 5)
    assert time_slider.value == 2
    assert freq_slider.value == 5

    panel._select_slice(3, 6)
    _run_scheduled()
    assert slice_selection.time_idx == 3 and slice_selection.freq_idx == 6
    assert time_slider.value == 3
    assert freq_slider.value == 6
    assert "(3)" in panel._format_slice_status(3, 6)

    with param.parameterized.batch_call_watchers(slice_selection):
        slice_selection.time_idx = 1
        slice_selection.freq_idx = 2
    assert "(1)" in panel._format_slice_status(1, 2)
    assert "(2)" in panel._format_slice_status(1, 2)

    pushed: list[int] = []
    slice_selection.set_push_root(lambda: pushed.append(1))
    slice_selection.param.watch(lambda *_events: slice_selection._push_ui(), ["time_idx", "freq_idx"])
    panel._select_slice(0, 3)
    _run_scheduled()
    assert pushed == [1]


def test_shared_zenith_slice_links_stokes_panels(monkeypatch: pytest.MonkeyPatch) -> None:
    import astropy.units as u

    from ovro_lwa_portal.viz.pipeline_qa_app import ZenithReviewPanel, ZenithSliceSelection

    class _Coord:
        ra = type("RA", (), {"to_string": lambda self, **kwargs: "12:00:00"})()
        dec = type("Dec", (), {"to_string": lambda self, **kwargs: "+00:00:00"})()

    _patch_zenith_status_center(monkeypatch)

    stat_map = np.ones((4, 10))
    freq_values = np.linspace(70e6, 90e6, 10)

    class _Radport:
        @staticmethod
        def _get_wcs(*, time_idx: int):
            class _WCS:
                @staticmethod
                def pixel_to_world(_l, _m):
                    class _PixelCoord:
                        ra = 0.0 * u.deg
                        dec = 0.0 * u.deg

                    return _PixelCoord()

            return _WCS()

        @staticmethod
        def nearest_lm_idx(_l, _m):
            return 0, 0

    dataset = _mock_zenith_dataset(freq_hz=freq_values)
    dataset.radport = _Radport()

    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app._run_on_main_thread",
        lambda callback: callback(),
    )

    slice_selection = ZenithSliceSelection()
    slice_selection.configure(
        n_times=4,
        n_freqs=10,
        default_time=0,
        default_freq=8,
    )
    panel_i = ZenithReviewPanel(
        dataset,  # type: ignore[arg-type]
        stat_map,
        slice_selection=slice_selection,
        stokes_label="I",
        metric_label="STD",
    )
    panel_v = ZenithReviewPanel(
        dataset,  # type: ignore[arg-type]
        stat_map,
        slice_selection=slice_selection,
        stokes_label="V",
        metric_label="STD",
    )

    panel_i._select_slice(2, 5)

    assert panel_i.time_idx == panel_v.time_idx == 2
    assert panel_i.freq_idx == panel_v.freq_idx == 5


def test_heatmap_click_sets_sky_stokes(monkeypatch: pytest.MonkeyPatch) -> None:
    import astropy.units as u

    from ovro_lwa_portal.viz.pipeline_qa_app import ZenithReviewPanel, _StokesReviewHolder

    monkeypatch.setattr(
        _StokesReviewHolder,
        "_request_sky_update",
        lambda self, **kwargs: None,
    )
    monkeypatch.setattr(
        _StokesReviewHolder,
        "_bind_sky_dataset",
        staticmethod(lambda widget, dataset: None),
    )

    class _Coord:
        ra = type("RA", (), {"to_string": lambda self, **kwargs: "12:00:00"})()
        dec = type("Dec", (), {"to_string": lambda self, **kwargs: "+00:00:00"})()

    _patch_zenith_status_center(monkeypatch)
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app._run_on_main_thread",
        lambda callback: callback(),
    )

    stat_map = np.ones((4, 10))
    dataset = _mock_zenith_dataset()

    holder = _StokesReviewHolder()
    holder.bind_datasets(
        {
            "I": dataset,  # type: ignore[arg-type]
            "V": dataset,  # type: ignore[arg-type]
        }
    )
    panel_i = ZenithReviewPanel(
        dataset,  # type: ignore[arg-type]
        stat_map,
        slice_selection=holder.slice_selection,
        stokes_label="I",
        metric_label="STD",
        on_heatmap_select=holder.select_slice_from_heatmap,
    )
    panel_v = ZenithReviewPanel(
        dataset,  # type: ignore[arg-type]
        stat_map,
        slice_selection=holder.slice_selection,
        stokes_label="V",
        metric_label="STD",
        on_heatmap_select=holder.select_slice_from_heatmap,
    )
    holder._panels["I"] = panel_i
    holder._panels["V"] = panel_v
    holder._configure_slice_selection()

    assert holder.sky_stokes == "I"
    panel_i._select_slice(1, 3)
    assert holder.sky_stokes == "I"
    assert holder.slice_selection.time_idx == 1
    assert holder.slice_selection.freq_idx == 3

    panel_v._select_slice(2, 4)
    assert holder.sky_stokes == "V"
    assert holder.slice_selection.time_idx == 2
    assert holder.slice_selection.freq_idx == 4


def test_sync_zenith_status_pushes_dashboard_root(monkeypatch: pytest.MonkeyPatch) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import ZenithReviewPanel, _StokesReviewHolder

    pushed: list[int] = []

    holder = _StokesReviewHolder()
    holder.set_push_root(lambda: pushed.append(1))
    holder._panels["I"] = None  # type: ignore[assignment]
    holder._sync_zenith_status_lines()
    assert pushed == []

    class _Dataset:
        sizes = {"time": 4, "frequency": 10}

    _patch_zenith_status_center(monkeypatch)
    holder.bind_datasets({"I": _Dataset()})  # type: ignore[arg-type]
    holder._panels["I"] = ZenithReviewPanel(
        _mock_zenith_dataset(n_times=4, n_freqs=10),  # type: ignore[arg-type]
        np.ones((4, 10)),
        slice_selection=holder.slice_selection,
        stokes_label="I",
        metric_label="STD",
    )
    holder._configure_slice_selection()
    holder.slice_selection.time_idx = 2
    assert pushed == [1]


def test_zenith_slice_syncs_status_on_slider_change(monkeypatch: pytest.MonkeyPatch) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import ZenithReviewPanel, _StokesReviewHolder

    pushed: list[int] = []
    monkeypatch.setattr(
        _StokesReviewHolder,
        "_request_sky_update",
        lambda self, **kwargs: None,
    )

    class _Coord:
        ra = type("RA", (), {"to_string": lambda self, **kwargs: "12:00:00"})()
        dec = type("Dec", (), {"to_string": lambda self, **kwargs: "+00:00:00"})()

    _patch_zenith_status_center(monkeypatch)

    stat_map = np.ones((4, 10))
    freq_values = np.linspace(70e6, 90e6, 10)
    dataset = _mock_zenith_dataset(freq_hz=freq_values)

    holder = _StokesReviewHolder()
    panel = ZenithReviewPanel(
        dataset,  # type: ignore[arg-type]
        stat_map,
        slice_selection=holder.slice_selection,
        stokes_label="I",
        metric_label="STD",
    )
    holder._panels["I"] = panel
    holder.set_push_root(lambda: pushed.append(1))
    holder._configure_slice_selection()

    holder._heatmap_status_row.visible = True
    holder.slice_selection.time_idx = 2
    holder.slice_selection.freq_idx = 4
    holder._refresh_heatmap_status_row()

    assert "(2)" in holder._heatmap_status_panes["I"].object
    assert "(4)" in holder._heatmap_status_panes["I"].object
    assert len(pushed) >= 1


def test_pipeline_qa_app_shows_error_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    app = PipelineQAApp()
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)

    app._log_error("Something failed")

    alert = app._error_alert_view()
    assert isinstance(alert, pn.pane.Alert)
    assert alert.object == "Something failed"


def test_pipeline_qa_app_zenith_loading_indicator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = PipelineQAApp()
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)

    app.loading_zenith = True
    app._sync_zenith_loading_indicator()

    assert app._zenith_loading_row.visible is True
    assert app._zenith_loading_spinner.value is True

    app.loading_zenith = False
    app._sync_zenith_loading_indicator()

    assert app._zenith_loading_row.visible is False
    assert app._zenith_loading_spinner.value is False


def test_finish_load_day_auto_starts_zenith(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    app = PipelineQAApp()
    monkeypatch.setattr(app, "_auto_load_zenith_if_ready", calls.append)
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)
    app._load_seq = 3

    app._finish_load_day(load_seq=3, auto_zenith=True)

    assert calls == [3]


def test_finish_load_day_skips_auto_zenith_on_stale_seq(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    app = PipelineQAApp()
    monkeypatch.setattr(app, "_auto_load_zenith_if_ready", calls.append)
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)
    app._load_seq = 2

    app._finish_load_day(load_seq=1, auto_zenith=True)

    assert calls == []


def test_auto_load_zenith_if_ready_requires_zarr(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, int]] = []
    app = PipelineQAApp()
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)
    monkeypatch.setattr(app, "_begin_load_day", lambda: None)
    app.scanning = False
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.zarr_status",
        lambda _day, **kwargs: {"I": False, "V": False},
    )
    monkeypatch.setattr(
        app,
        "_begin_zenith_load",
        lambda *, load_seq: calls.append({"load_seq": load_seq}),
    )
    app._load_seq = 4
    _set_select_day(app, "2024-12-27")
    app._loaded_day = "2024-12-27"

    app._auto_load_zenith_if_ready(4)

    assert calls == []


def test_auto_load_zenith_if_ready_starts_when_zarr_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, int]] = []
    app = PipelineQAApp()
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)
    monkeypatch.setattr(app, "_begin_load_day", lambda: None)
    app.scanning = False
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.zarr_status",
        lambda _day, **kwargs: {"I": True, "V": False},
    )
    monkeypatch.setattr(
        app,
        "_begin_zenith_load",
        lambda *, load_seq: calls.append({"load_seq": load_seq}),
    )
    app._load_seq = 5
    _set_select_day(app, "2024-12-27")
    app._loaded_day = "2024-12-27"

    app._auto_load_zenith_if_ready(5)

    assert calls == [{"load_seq": 5}]


def test_begin_zenith_load_rejects_stale_load_seq(monkeypatch: pytest.MonkeyPatch) -> None:
    app = PipelineQAApp()
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)
    monkeypatch.setattr(app, "_begin_load_day", lambda: None)
    app._load_seq = 2
    _set_select_day(app, "2024-12-27")
    app._loaded_day = "2024-12-27"

    app._begin_zenith_load(load_seq=1)

    assert app.loading_zenith is False


def test_begin_zenith_load_skipped_while_converting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Post-convert day refresh must clear converting before zenith auto-load runs."""
    app = PipelineQAApp()
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)
    monkeypatch.setattr(app, "_begin_load_day", lambda: None)
    _set_select_day(app, "2024-12-27")
    app._loaded_day = "2024-12-27"
    app.converting = True
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.zarr_status",
        lambda _day, **kwargs: {"I": True, "V": True},
    )

    app._begin_zenith_load()

    assert app.loading_zenith is False


def test_convert_success_schedules_refresh_after_clearing_converting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[bool] = []
    scheduled: list[Callable[[], None]] = []
    app = PipelineQAApp()
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)
    _set_select_day(app, "2024-12-27")
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.convert_missing_zarr",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app._schedule_ipython_main",
        lambda callback: scheduled.append(callback),
    )

    def _load_day(*, silent: bool) -> None:
        observed.append(app.converting)

    monkeypatch.setattr(app, "_load_day", _load_day)

    app._on_convert_click(None)
    while scheduled:
        scheduled.pop(0)()

    assert observed == [False]


def _configure_sky_holder_for_sync_tests(
    holder: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run sky updates on the main thread in unit tests."""
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app._run_on_main_thread",
        lambda callback: callback(),
    )


def _apply_fake_sky_slice(widget: object, time_idx: int, freq_idx: int, **kwargs: object) -> None:
    """Record a successful SkyWidget slice update for test doubles."""
    widget.time_idx = int(time_idx)  # type: ignore[attr-defined]
    widget.freq_idx = int(freq_idx)  # type: ignore[attr-defined]
    widget.image_revision = int(getattr(widget, "image_revision", 0)) + 1  # type: ignore[attr-defined]


def _mock_status_sky_coord() -> object:
    """Minimal coord for zenith status markdown (hourangle + deg strings)."""
    return type(
        "Coord",
        (),
        {
            "ra": type("RA", (), {"to_string": lambda self, **kwargs: "12:00:00"})(),
            "dec": type("Dec", (), {"to_string": lambda self, **kwargs: "+00:00:00"})(),
        },
    )()


def _patch_zenith_status_center(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock centers for heatmap status text and sky recentering in unit tests."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    def _coord(dataset: object, time_idx: int) -> object:
        return _mock_status_sky_coord()

    monkeypatch.setattr("ovro_lwa_portal.viz.pipeline_qa_app.sky_view_center", _coord)
    monkeypatch.setattr("ovro_lwa_portal.viz.pipeline_qa_app.zenith_lm_coord", _coord)


def _patch_sky_view_center(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock sky recentering for holder tests that use minimal dataset stubs."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.sky_view_center",
        lambda dataset, time_idx: SkyCoord(ra=180.0 * u.deg, dec=45.0 * u.deg, frame="fk5"),
    )


def test_slider_freq_change_keeps_sky_view(monkeypatch: pytest.MonkeyPatch) -> None:
    import astropy.units as u

    from ovro_lwa_portal.viz.pipeline_qa_app import _StokesReviewHolder

    class _FakeSky:
        def __init__(self) -> None:
            self.update_slice_calls: list[dict[str, object]] = []
            self.time_idx = 0
            self.freq_idx = 0
            self.image_revision = 0

        def set_dataset(self, dataset, **kwargs) -> None:
            return None

        def update_slice(self, time_idx, freq_idx, **kwargs) -> None:
            _apply_fake_sky_slice(self, time_idx, freq_idx, **kwargs)
            self.update_slice_calls.append(
                {"time_idx": time_idx, "freq_idx": freq_idx, **kwargs}
            )

    fake = _FakeSky()

    class _Dataset:
        sizes = {"time": 4, "frequency": 10}

    holder = _StokesReviewHolder()
    _configure_sky_holder_for_sync_tests(holder, monkeypatch)
    holder._sky_widget = fake  # type: ignore[assignment]
    holder._sky_bound_stokes = "I"
    holder.bind_datasets({"I": _Dataset()})  # type: ignore[arg-type]
    holder.slice_selection.configure(
        n_times=4,
        n_freqs=10,
        default_time=0,
        default_freq=0,
    )
    fake.update_slice_calls.clear()

    holder.slice_selection.freq_idx = 5

    assert len(fake.update_slice_calls) == 1
    call = fake.update_slice_calls[0]
    assert call["time_idx"] == 0
    assert call["freq_idx"] == 5
    assert "center" not in call
    assert "fov" not in call


def test_slider_time_change_recenters_sky_view(monkeypatch: pytest.MonkeyPatch) -> None:
    import astropy.units as u

    from ovro_lwa_portal.viz.pipeline_qa_app import _StokesReviewHolder

    class _FakeSky:
        def __init__(self) -> None:
            self.update_slice_calls: list[dict[str, object]] = []

        def set_dataset(self, dataset, **kwargs) -> None:
            return None

        def update_slice(self, time_idx, freq_idx, **kwargs) -> None:
            _apply_fake_sky_slice(self, time_idx, freq_idx, **kwargs)
            self.update_slice_calls.append(
                {"time_idx": time_idx, "freq_idx": freq_idx, **kwargs}
            )

    fake = _FakeSky()
    fake.time_idx = 0
    fake.freq_idx = 0
    fake.image_revision = 0

    class _Dataset:
        sizes = {"time": 4, "frequency": 10}

    holder = _StokesReviewHolder()
    _configure_sky_holder_for_sync_tests(holder, monkeypatch)
    holder._sky_widget = fake  # type: ignore[assignment]
    holder._sky_bound_stokes = "I"
    holder.bind_datasets({"I": _Dataset()})  # type: ignore[arg-type]
    holder.slice_selection.configure(
        n_times=4,
        n_freqs=10,
        default_time=0,
        default_freq=0,
    )
    fake.update_slice_calls.clear()
    _patch_sky_view_center(monkeypatch)

    holder.slice_selection.time_idx = 3

    assert len(fake.update_slice_calls) == 1
    assert fake.update_slice_calls[0]["fov"] == 25.0 * u.deg
    assert "center" in fake.update_slice_calls[0]


def test_stokes_toggle_updates_sky_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import astropy.units as u

    from ovro_lwa_portal.viz.pipeline_qa_app import _StokesReviewHolder

    class _FakeSky:
        def __init__(self) -> None:
            self.set_dataset_calls: list[dict[str, object]] = []
            self.update_slice_calls: list[dict[str, object]] = []
            self.time_idx = 0
            self.freq_idx = 0
            self.image_revision = 0

        def set_dataset(self, dataset, **kwargs) -> None:
            self.set_dataset_calls.append({"dataset": dataset, **kwargs})

        def update_slice(self, time_idx, freq_idx, **kwargs) -> None:
            _apply_fake_sky_slice(self, time_idx, freq_idx, **kwargs)
            self.update_slice_calls.append(
                {"time_idx": time_idx, "freq_idx": freq_idx, **kwargs}
            )

        def send_state(self) -> None:
            return None

    fake = _FakeSky()

    class _Dataset:
        sizes = {"time": 4, "frequency": 10}

    holder = _StokesReviewHolder()
    _configure_sky_holder_for_sync_tests(holder, monkeypatch)
    holder._sky_widget = fake  # type: ignore[assignment]
    holder.bind_datasets({"I": _Dataset(), "V": _Dataset()})  # type: ignore[arg-type]
    holder._configure_slice_selection()
    _patch_sky_view_center(monkeypatch)

    holder.sky_stokes = "V"

    assert len(fake.set_dataset_calls) == 1
    assert fake.set_dataset_calls[0]["defer_display"] is True
    assert len(fake.update_slice_calls) == 1
    assert fake.update_slice_calls[0]["fov"] == 25.0 * u.deg


def test_stokes_review_sky_loading_indicator(monkeypatch: pytest.MonkeyPatch) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import _StokesReviewHolder

    class _FakeSky:
        def set_dataset(self, dataset, **kwargs) -> None:
            return None

        def update_slice(self, time_idx, freq_idx, **kwargs) -> None:
            return None

    class _Dataset:
        sizes = {"time": 2, "frequency": 3}

    holder = _StokesReviewHolder()
    _configure_sky_holder_for_sync_tests(holder, monkeypatch)
    holder._sky_widget = _FakeSky()  # type: ignore[assignment]
    holder.bind_datasets({"I": _Dataset()})  # type: ignore[arg-type]
    holder._sky_bound_stokes = "I"
    holder.slice_selection.configure(
        n_times=2,
        n_freqs=3,
        default_time=0,
        default_freq=0,
    )

    holder.loading_sky = True
    holder._sync_sky_loading_indicator()

    assert holder._sky_status_spinner.visible is True
    assert holder._sky_status_spinner.value is True

    holder.loading_sky = False
    holder._sync_sky_loading_indicator()

    assert holder._sky_status_spinner.visible is False
    assert holder._sky_status_spinner.value is False


def test_sky_update_sets_ready_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import _StokesReviewHolder

    class _FakeSky:
        def __init__(self) -> None:
            self.time_idx = 0
            self.freq_idx = 0
            self.image_revision = 0

        def set_dataset(self, dataset, **kwargs) -> None:
            return None

        def update_slice(self, time_idx, freq_idx, **kwargs) -> None:
            _apply_fake_sky_slice(self, time_idx, freq_idx, **kwargs)

    class _Dataset:
        sizes = {"time": 2, "frequency": 3}

    holder = _StokesReviewHolder()
    _configure_sky_holder_for_sync_tests(holder, monkeypatch)
    holder._sky_widget = _FakeSky()  # type: ignore[assignment]
    holder._sky_bound_stokes = "I"
    holder.bind_datasets({"I": _Dataset()})  # type: ignore[arg-type]
    holder.slice_selection.configure(
        n_times=2,
        n_freqs=3,
        default_time=0,
        default_freq=0,
    )
    holder.slice_selection.freq_idx = 1

    assert "ready" in holder._sky_status_pane.object
    assert holder.loading_sky is False


def test_sky_update_shows_error_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import _StokesReviewHolder

    class _FakeSky:
        def __init__(self) -> None:
            self.time_idx = 0
            self.freq_idx = 0
            self.image_revision = 0

        def set_dataset(self, dataset, **kwargs) -> None:
            return None

        def update_slice(self, time_idx, freq_idx, **kwargs) -> None:
            msg = "slice failed"
            raise RuntimeError(msg)

    class _Dataset:
        sizes = {"time": 2, "frequency": 3}

    holder = _StokesReviewHolder()
    _configure_sky_holder_for_sync_tests(holder, monkeypatch)
    holder._sky_widget = _FakeSky()  # type: ignore[assignment]
    holder._sky_bound_stokes = "I"
    holder.bind_datasets({"I": _Dataset()})  # type: ignore[arg-type]
    holder.slice_selection.configure(
        n_times=2,
        n_freqs=3,
        default_time=0,
        default_freq=0,
    )
    holder._request_sky_update()

    assert holder._sky_error_alert.visible is True
    assert "slice failed" in holder._sky_error_alert.object
    assert "failed" in holder._sky_status_pane.object


def test_heatmap_select_high_time_updates_sky_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import astropy.units as u

    from ovro_lwa_portal.viz.pipeline_qa_app import _StokesReviewHolder

    class _FakeSky:
        def __init__(self) -> None:
            self.update_slice_calls: list[dict[str, object]] = []
            self.send_state_calls = 0

        def set_dataset(self, dataset, **kwargs) -> None:
            return None

        def update_slice(self, time_idx, freq_idx, **kwargs) -> None:
            _apply_fake_sky_slice(self, time_idx, freq_idx, **kwargs)
            self.update_slice_calls.append(
                {"time_idx": time_idx, "freq_idx": freq_idx, **kwargs}
            )

        def send_state(self) -> None:
            self.send_state_calls += 1

    fake = _FakeSky()
    fake.time_idx = 0
    fake.freq_idx = 0
    fake.image_revision = 0

    class _Dataset:
        sizes = {"time": 100, "frequency": 10}

    holder = _StokesReviewHolder()
    _configure_sky_holder_for_sync_tests(holder, monkeypatch)
    holder._sky_widget = fake  # type: ignore[assignment]
    holder._sky_bound_stokes = "I"
    holder.bind_datasets({"I": _Dataset()})  # type: ignore[arg-type]
    holder.slice_selection.configure(
        n_times=100,
        n_freqs=10,
        default_time=0,
        default_freq=0,
    )
    _patch_sky_view_center(monkeypatch)
    fake.update_slice_calls.clear()

    holder.select_slice_from_heatmap("I", 99, 5)

    assert len(fake.update_slice_calls) == 1
    assert fake.update_slice_calls[0]["time_idx"] == 99
    assert fake.update_slice_calls[0]["freq_idx"] == 5
    assert fake.update_slice_calls[0]["fov"] == 25.0 * u.deg
    assert fake.send_state_calls == 1
    assert "ready" in holder._sky_status_pane.object


def test_sky_update_missing_per_time_wcs_shows_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import xarray as xr

    from tests.test_fits_to_zarr import _make_sin_wcs_header_str

    from ovro_lwa_portal.viz import pipeline_qa_app as pqa_app
    from ovro_lwa_portal.viz.pipeline_qa_app import _StokesReviewHolder

    class _FakeSky:
        def set_dataset(self, dataset, **kwargs) -> None:
            return None

        def update_slice(self, time_idx, freq_idx, **kwargs) -> None:
            msg = "should not reach update_slice"
            raise RuntimeError(msg)

    n_times = 12
    hdr0 = _make_sin_wcs_header_str(nx=8, ny=8, crval1=180.0, crval2=45.0)
    enc0 = hdr0.encode("utf-8")
    wcs_per_time = np.array(
        [np.bytes_(enc0) if i < 9 else np.bytes_(b"") for i in range(n_times)],
        dtype=f"S{len(enc0)}",
    )
    ds = xr.Dataset(
        {
            "SKY": (
                ["time", "frequency", "polarization", "l", "m"],
                np.zeros((n_times, 2, 1, 8, 8), dtype=np.float32),
            ),
            "wcs_header_str": (["time"], wcs_per_time),
        },
        coords={
            "time": np.linspace(60_000.0, 60_000.0 + 0.11, n_times),
            "frequency": [70e6, 80e6],
            "polarization": [0],
            "l": np.linspace(-1, 1, 8),
            "m": np.linspace(-1, 1, 8),
        },
    )
    ds["SKY"].attrs["fits_wcs_header"] = hdr0

    holder = _StokesReviewHolder()
    _configure_sky_holder_for_sync_tests(holder, monkeypatch)
    holder._sky_widget = _FakeSky()  # type: ignore[assignment]
    holder._sky_bound_stokes = "I"
    holder.bind_datasets({"I": ds})
    holder.slice_selection.configure(
        n_times=n_times,
        n_freqs=2,
        default_time=0,
        default_freq=0,
    )
    holder._slice_selection.time_idx = 9

    assert holder._sky_error_alert.visible is True
    assert "time index 9" in holder._sky_error_alert.object
    assert "failed" in holder._sky_status_pane.object
    pqa_app._patch_astrowidget_get_wcs()


def test_format_slice_status_high_time_uses_sky_view_center(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Status line must not call zenith_lm_coord (fails for late timesteps on some days)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    from ovro_lwa_portal.viz.pipeline_qa_app import ZenithReviewPanel, ZenithSliceSelection

    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.sky_view_center",
        lambda dataset, time_idx: SkyCoord(
            ra=(180.0 + float(time_idx) * 15.0) * u.deg,
            dec=45.0 * u.deg,
            frame="fk5",
        ),
    )
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.zenith_lm_coord",
        lambda dataset, time_idx: (_ for _ in ()).throw(
            ValueError("zenith_lm_coord should not run")
        ),
    )

    dataset = _mock_zenith_dataset(n_times=12, n_freqs=2)
    stat_map = np.ones((12, 2))
    sel = ZenithSliceSelection()
    sel.configure(n_times=12, n_freqs=2, default_time=0, default_freq=0)
    panel = ZenithReviewPanel(
        dataset,  # type: ignore[arg-type]
        stat_map,
        slice_selection=sel,
        stokes_label="I",
        metric_label="STD",
    )
    text = panel._format_slice_status(9, 0)
    assert "(9)" in text
    assert "21h" in text


def test_heatmap_high_time_updates_status_and_sky(monkeypatch: pytest.MonkeyPatch) -> None:
    """Heatmap at high time_idx must not abort before sky update (regression 2024-12-28)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    from ovro_lwa_portal.viz.pipeline_qa_app import ZenithReviewPanel, _StokesReviewHolder

    class _FakeSky:
        def __init__(self) -> None:
            self.time_idx = 0
            self.freq_idx = 0
            self.image_revision = 0
            self.update_slice_calls: list[tuple[int, int]] = []

        def set_dataset(self, dataset, **kwargs) -> None:
            return None

        def update_slice(self, time_idx, freq_idx, **kwargs) -> None:
            _apply_fake_sky_slice(self, time_idx, freq_idx, **kwargs)
            self.update_slice_calls.append((int(time_idx), int(freq_idx)))

        def send_state(self) -> None:
            return None

    class _Dataset:
        sizes = {"time": 12, "frequency": 15}

        def __getitem__(self, key: str) -> object:
            if key == "wcs_header_str":
                return xr.DataArray(np.array([b"x"] * 12), dims=["time"])
            msg = f"unknown key {key}"
            raise KeyError(msg)

    import xarray as xr

    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app._has_per_time_wcs_header_str",
        lambda _ds: True,
    )
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.sky_view_center",
        lambda dataset, time_idx: SkyCoord(ra=(180.0 + time_idx * 15.0) * u.deg, dec=45.0 * u.deg, frame="fk5"),
    )
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.zenith_lm_coord",
        lambda dataset, time_idx: (_ for _ in ()).throw(ValueError("zenith_lm_coord should not run")),
    )

    fake = _FakeSky()
    holder = _StokesReviewHolder()
    _configure_sky_holder_for_sync_tests(holder, monkeypatch)
    holder._sky_widget = fake  # type: ignore[assignment]
    holder._sky_bound_stokes = "I"
    holder.bind_datasets({"I": _Dataset()})  # type: ignore[arg-type]
    holder.slice_selection.configure(
        n_times=12,
        n_freqs=15,
        default_time=0,
        default_freq=0,
    )

    holder._panels["I"] = ZenithReviewPanel(
        _mock_zenith_dataset(n_times=12, n_freqs=15),  # type: ignore[arg-type]
        np.ones((12, 15)),
        slice_selection=holder.slice_selection,
        stokes_label="I",
        metric_label="STD",
    )
    holder._configure_slice_selection()
    holder.select_slice_from_heatmap("I", 9, 8)

    assert holder._slice_selection.time_idx == 9
    assert fake.update_slice_calls[-1] == (9, 8)
    assert "(9)" in holder._heatmap_status_panes["I"].object
    assert "ready" in holder._sky_status_pane.object


def test_heatmap_high_time_recenters_when_tracker_ahead_of_widget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recenter when _sky_last_time_idx matches but the widget still shows an old slice."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    from ovro_lwa_portal.viz.pipeline_qa_app import _StokesReviewHolder

    centers: list[float] = []

    def _record_center(dataset, time_idx: int):
        ra = 180.0 + float(time_idx) * 15.0
        centers.append(ra)
        return SkyCoord(ra=ra * u.deg, dec=45.0 * u.deg, frame="fk5")

    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.sky_view_center",
        _record_center,
    )

    class _FakeSky:
        def __init__(self) -> None:
            self.time_idx = 0
            self.freq_idx = 0
            self.image_revision = 1
            self.update_slice_calls: list[dict[str, object]] = []

        def set_dataset(self, dataset, **kwargs) -> None:
            return None

        def update_slice(self, time_idx, freq_idx, **kwargs) -> None:
            _apply_fake_sky_slice(self, time_idx, freq_idx, **kwargs)
            self.update_slice_calls.append(
                {"time_idx": time_idx, "freq_idx": freq_idx, **kwargs}
            )

        def send_state(self) -> None:
            return None

    fake = _FakeSky()

    class _Dataset:
        sizes = {"time": 12, "frequency": 15}

    holder = _StokesReviewHolder()
    _configure_sky_holder_for_sync_tests(holder, monkeypatch)
    holder._sky_widget = fake  # type: ignore[assignment]
    holder._sky_bound_stokes = "I"
    holder._sky_last_time_idx = 9
    holder._sky_last_stokes = "I"
    holder.bind_datasets({"I": _Dataset()})  # type: ignore[arg-type]
    holder.slice_selection.configure(
        n_times=12,
        n_freqs=15,
        default_time=0,
        default_freq=0,
    )
    fake.update_slice_calls.clear()

    holder.select_slice_from_heatmap("I", 9, 8)

    assert len(fake.update_slice_calls) == 1
    call = fake.update_slice_calls[0]
    assert call["time_idx"] == 9
    assert "center" in call
    assert "fov" in call
    assert centers == [180.0 + 9.0 * 15.0]


def test_heatmap_select_clears_suppress_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import _StokesReviewHolder

    class _FakeSky:
        def __init__(self) -> None:
            self.time_idx = 0
            self.freq_idx = 0
            self.image_revision = 0

        def set_dataset(self, dataset, **kwargs) -> None:
            return None

        def update_slice(self, time_idx, freq_idx, **kwargs) -> None:
            self.time_idx = int(time_idx)
            self.freq_idx = int(freq_idx)
            self.image_revision += 1

        def send_state(self) -> None:
            return None

    holder = _StokesReviewHolder()
    _configure_sky_holder_for_sync_tests(holder, monkeypatch)
    _patch_sky_view_center(monkeypatch)
    fake = _FakeSky()
    holder._sky_widget = fake  # type: ignore[assignment]
    holder._sky_bound_stokes = "I"

    class _Dataset:
        sizes = {"time": 12, "frequency": 15}

    holder.bind_datasets({"I": _Dataset()})  # type: ignore[arg-type]
    holder.slice_selection.configure(
        n_times=12,
        n_freqs=15,
        default_time=0,
        default_freq=8,
    )

    rev_before = fake.image_revision
    holder.select_slice_from_heatmap("I", 9, 8)

    assert holder._suppress_post_heatmap_sky_watchers is False
    assert holder._ignore_slice_watcher_for_sky is False
    assert holder._slice_selection.time_idx == 9
    assert fake.time_idx == 9
    assert fake.image_revision > rev_before


def test_stale_sky_finish_does_not_mark_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import _StokesReviewHolder

    holder = _StokesReviewHolder()
    holder._sky_update_seq = 2
    holder.loading_sky = True
    holder._sky_status_pane.object = "loading slice"

    holder._finish_sky_update(1, None)

    assert holder.loading_sky is True
    assert holder._sky_status_pane.object == "loading slice"


def test_rapid_slider_updates_finish_at_latest_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each synchronous slider step runs to completion; the last slice wins."""
    from ovro_lwa_portal.viz.pipeline_qa_app import _StokesReviewHolder

    calls: list[int] = []

    class _FakeSky:
        def __init__(self) -> None:
            self.time_idx = 0
            self.freq_idx = 0
            self.image_revision = 0

        def set_dataset(self, dataset, **kwargs) -> None:
            return None

        def update_slice(self, time_idx, freq_idx, **kwargs) -> None:
            _apply_fake_sky_slice(self, time_idx, freq_idx, **kwargs)
            calls.append(int(freq_idx))

    class _Dataset:
        sizes = {"time": 4, "frequency": 4}

    holder = _StokesReviewHolder()
    _configure_sky_holder_for_sync_tests(holder, monkeypatch)
    _patch_sky_view_center(monkeypatch)
    holder._sky_widget = _FakeSky()  # type: ignore[assignment]
    holder._sky_bound_stokes = "I"
    holder.bind_datasets({"I": _Dataset()})  # type: ignore[arg-type]
    holder.slice_selection.configure(
        n_times=4,
        n_freqs=4,
        default_time=0,
        default_freq=0,
    )

    holder.slice_selection.freq_idx = 1
    holder.slice_selection.freq_idx = 2
    holder.slice_selection.freq_idx = 3

    assert calls == [1, 2, 3]
    assert holder._sky_widget.freq_idx == 3  # type: ignore[union-attr]
    assert "ready" in holder._sky_status_pane.object


def test_stokes_review_holder_mount_sky_updates_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import _StokesReviewHolder

    holder = _StokesReviewHolder()

    class _FakeSky(widgets.HTML):
        def __init__(self) -> None:
            super().__init__(value="sky")

    def _fake_sky_widget() -> _FakeSky:
        return _FakeSky()

    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.SkyWidget",
        _fake_sky_widget,
    )
    monkeypatch.setattr(
        _StokesReviewHolder,
        "_bind_sky_dataset",
        staticmethod(lambda widget, dataset: None),
    )
    monkeypatch.setattr(
        _StokesReviewHolder,
        "_request_sky_update",
        lambda self, **kwargs: None,
    )

    class _Dataset:
        sizes = {"time": 2, "frequency": 3}

    holder.bind_datasets({"I": _Dataset()})  # type: ignore[arg-type]
    holder.mount_sky()

    assert len(holder._sky_container.children) == 2
    assert holder._sky_status_row in holder._zenith_footer.objects
    assert isinstance(holder._sky_container.children[1], _FakeSky)
    assert holder.sky_widget is not None
    assert holder._sky_pane.width == 1048


def _write_flux_check_tree(root: Path, *, obs_date: str = "2024-12-28") -> None:
    run_dir = root / "08h" / obs_date / "Run_20241228_120000"
    for freq_mhz, ratio in ((32, 2.0), (46, 1.5)):
        qa_dir = run_dir / f"{freq_mhz}MHz" / "QA"
        qa_dir.mkdir(parents=True, exist_ok=True)
        (qa_dir / "flux_check_hybrid.csv").write_text(
            "imfit_flux,imfit_err,elevation,model_flux,source,freq\n"
            f"20.0,1.0,45.0,10.0,3C48,{freq_mhz}\n"
            f"{ratio * 6.0},1.0,50.0,6.0,3C147,{freq_mhz}\n"
        )
    run_dir2 = root / "09h" / obs_date / "Run_20241228_130000"
    qa_dir = run_dir2 / "32MHz" / "QA"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "flux_check_hybrid.csv").write_text(
        "imfit_flux,imfit_err,elevation,model_flux,source,freq\n"
        "30.0,1.0,45.0,15.0,3C48,32.0\n"
        "12.0,1.0,50.0,8.0,3C147,32.0\n"
    )
    wideband = run_dir / "Wideband"
    wideband.mkdir(parents=True, exist_ok=True)
    (wideband / "thermal_noise_vs_subband.png").write_bytes(b"png")
    wideband2 = run_dir2 / "Wideband"
    wideband2.mkdir(parents=True, exist_ok=True)
    (wideband2 / "thermal_noise_vs_subband.png").write_bytes(b"png")


def _write_phase2_qa_tree(root: Path, *, obs_date: str = "2025-01-11", hour: str = "05h") -> None:
    run_dir = root / hour / obs_date / "Science_20260527_173819"
    qa_dir = run_dir / "QA"
    qa_dir.mkdir(parents=True)
    (qa_dir / f"{obs_date}_{hour}_thermal_noise_vs_freq.png").write_bytes(b"png")
    (qa_dir / f"{obs_date}_{hour}_flux_check_hybrid.csv").write_text(
        "imfit_flux,imfit_err,elevation,model_flux,source,freq\n"
        "20.0,1.0,45.0,10.0,3C48,55.0\n"
        "15.0,1.0,45.0,7.5,3C147,55.0\n",
        encoding="utf-8",
    )
    (qa_dir / "20250111_05h_dewarp_summary.csv").write_text(
        "freq_mhz,median_shift_arcmin\n"
        "55,0.12\n"
        "82,0.34\n",
        encoding="utf-8",
    )
    subband = run_dir / "55MHz" / "I" / "deep"
    subband.mkdir(parents=True)
    fits_name = (
        "55MHz-I-NoTaper-3581s-Robust-0-20250111_055900-image.pbcorr_dewarped.fits"
    )
    (subband / fits_name).write_bytes(b"fits")
    v_subband = run_dir / "55MHz" / "V" / "deep"
    v_subband.mkdir(parents=True)
    v_name = "55MHz-V-Taper-3581s-Robust-0-20250111_055900-image.pbcorr_dewarped.fits"
    (v_subband / v_name).write_bytes(b"fits")


def test_scan_coverage_phase2_science_runs(tmp_path: Path) -> None:
    _write_phase2_qa_tree(tmp_path)
    cfg = pq.PipelineQAConfig.phase2_default()
    cfg = pq.PipelineQAConfig(
        pipeline_root=tmp_path,
        symlink_root=tmp_path / "stage",
        zarr_root=tmp_path / "zarr",
        i_fits_glob=cfg.i_fits_glob,
        v_fits_glob=cfg.v_fits_glob,
        run_dir_prefix=cfg.run_dir_prefix,
        run_dir_pattern=cfg.run_dir_pattern,
        qa_thermal_noise_glob=cfg.qa_thermal_noise_glob,
        flux_check_csv_glob=cfg.flux_check_csv_glob,
        flux_check_csv_per_run=cfg.flux_check_csv_per_run,
        i_qa_zarr_stem=cfg.i_qa_zarr_stem,
        v_qa_zarr_stem=cfg.v_qa_zarr_stem,
        thermal_noise_grid_cols=cfg.thermal_noise_grid_cols,
        thermal_noise_plot_name=cfg.thermal_noise_plot_name,
        qa_run_label=cfg.qa_run_label,
    )
    coverage = pq.scan_coverage(config=cfg)
    assert coverage.iloc[0]["latest_run"] == "Science_20260527_173819"
    png = coverage.iloc[0]["thermal_noise_png"]
    assert png.endswith("_thermal_noise_vs_freq.png")


def test_load_flux_check_hybrid_dataframe_phase2(tmp_path: Path) -> None:
    _write_phase2_qa_tree(tmp_path)
    cfg = pq.PipelineQAConfig(
        pipeline_root=tmp_path,
        symlink_root=tmp_path / "stage",
        zarr_root=tmp_path / "zarr",
        i_fits_glob=pq.I_FITS_GLOB_PHASE2,
        v_fits_glob=pq.V_FITS_GLOB_PHASE2,
        run_dir_prefix="Science_",
        run_dir_pattern=r"Science_(\d{8})_(\d{6})",
        qa_thermal_noise_glob="QA/*_thermal_noise_vs_freq.png",
        flux_check_csv_glob="QA/*_flux_check_hybrid.csv",
        flux_check_csv_per_run=True,
    )
    coverage = pq.scan_coverage(config=cfg)
    flux_df = pq.load_flux_check_hybrid_dataframe("2025-01-11", coverage, config=cfg)
    assert len(flux_df) == 2
    assert set(flux_df["source"]) == {"3C48", "3C147"}
    assert flux_df.loc[flux_df["source"] == "3C48", "flux_ratio"].iloc[0] == 2.0


def test_load_flux_check_hybrid_dataframe(tmp_path: Path) -> None:
    _write_flux_check_tree(tmp_path)
    coverage = pq.scan_coverage(tmp_path)
    flux_df = pq.load_flux_check_hybrid_dataframe("2024-12-28", coverage)

    assert len(flux_df) == 6
    assert set(flux_df["source"]) == {"3C48", "3C147"}
    assert set(flux_df["frequency_mhz"]) == {32.0, 46.0}
    assert set(flux_df["lst_hour"]) == {"08h", "09h"}
    assert flux_df.loc[flux_df["source"] == "3C48", "flux_ratio"].iloc[0] == 2.0


def test_flux_ratio_grids_and_figures(tmp_path: Path) -> None:
    from bokeh.models import ColorBar, HoverTool

    from ovro_lwa_portal.viz.flux_check_plots import (
        FLUX_RATIO_GRID_TOTAL_WIDTH,
        build_flux_ratio_figures,
        build_flux_ratio_panel_grid,
    )

    _write_flux_check_tree(tmp_path)
    coverage = pq.scan_coverage(tmp_path)
    flux_df = pq.load_flux_check_hybrid_dataframe("2024-12-28", coverage)
    grids = pq.flux_ratio_grids(flux_df)

    assert set(grids) == {"3C48", "3C147"}
    assert grids["3C48"].loc[8, 32.0] == 2.0
    assert grids["3C48"].loc[9, 32.0] == 2.0

    figures = build_flux_ratio_figures(flux_df)
    assert set(figures) == {"3C48", "3C147"}
    hover_tools = [tool for tool in figures["3C48"].tools if isinstance(tool, HoverTool)]
    assert len(hover_tools) == 1
    assert figures["3C48"].select_one({"type": ColorBar}) is not None
    assert figures["3C48"].xaxis.axis_label == "LST hour"
    assert figures["3C48"].yaxis.axis_label == "Frequency (MHz)"

    panel_grid = build_flux_ratio_panel_grid(figures, n_cols=2)
    assert isinstance(panel_grid, pn.Column)
    assert len(panel_grid.objects) == 1
    assert panel_grid.max_width == FLUX_RATIO_GRID_TOTAL_WIDTH


def _phase2_test_config(tmp_path: Path) -> pq.PipelineQAConfig:
    return pq.PipelineQAConfig(
        pipeline_root=tmp_path,
        symlink_root=tmp_path / "stage",
        zarr_root=tmp_path / "zarr",
        i_fits_glob=pq.I_FITS_GLOB_PHASE2,
        v_fits_glob=pq.V_FITS_GLOB_PHASE2,
        run_dir_prefix="Science_",
        run_dir_pattern=r"Science_(\d{8})_(\d{6})",
        qa_thermal_noise_glob="QA/*_thermal_noise_vs_freq.png",
        flux_check_csv_glob="QA/*_flux_check_hybrid.csv",
        flux_check_csv_per_run=True,
        dewarp_summary_csv_glob="QA/*dewarp_summary.csv",
    )


def test_load_dewarp_summary_dataframe_phase2(tmp_path: Path) -> None:
    _write_phase2_qa_tree(tmp_path)
    cfg = _phase2_test_config(tmp_path)
    coverage = pq.scan_coverage(config=cfg)
    dewarp_df = pq.load_dewarp_summary_dataframe("2025-01-11", coverage, config=cfg)

    assert len(dewarp_df) == 2
    assert set(dewarp_df["frequency_mhz"]) == {55.0, 82.0}
    assert dewarp_df["median_shift"].tolist() == [0.12, 0.34]
    assert dewarp_df["lst_hour"].iloc[0] == "05h"


def test_run_on_main_thread_from_worker_schedules_ipython_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ovro_lwa_portal.viz import pipeline_qa_app as app

    scheduled: list[Callable[[], None]] = []
    executed_inline: list[int] = []

    monkeypatch.setattr(
        app,
        "_schedule_ipython_main",
        lambda callback: scheduled.append(callback),
    )

    def _fail_execute(callback: Callable[[], None]) -> None:
        executed_inline.append(1)
        callback()

    monkeypatch.setattr(app.pn.state, "execute", _fail_execute)

    import threading

    worker_done = threading.Event()

    def _worker() -> None:
        app._run_on_main_thread(lambda: executed_inline.append(2))
        worker_done.set()

    threading.Thread(target=_worker).start()
    worker_done.wait(timeout=2.0)

    assert executed_inline == []
    assert len(scheduled) == 1
    scheduled[0]()
    assert executed_inline == [2]


def test_schedule_ipython_main_from_worker_never_runs_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ovro_lwa_portal.viz import pipeline_qa_app as app

    monkeypatch.setattr(app, "_IPYTHON_IO_LOOP", None)
    monkeypatch.setattr(app, "_resolve_ipython_event_loop", lambda: None)
    executed: list[int] = []

    import threading

    worker_done = threading.Event()

    def _worker() -> None:
        app._schedule_ipython_main(lambda: executed.append(1))
        worker_done.set()

    threading.Thread(target=_worker).start()
    worker_done.wait(timeout=2.0)

    assert executed == []
    assert len(app._PENDING_MAIN_CALLBACKS) >= 1
    app._flush_pending_main_callbacks()
    assert executed == [1]
    assert app._PENDING_MAIN_CALLBACKS == []


def test_dewarp_shift_grid_and_figure(tmp_path: Path) -> None:
    from bokeh.models import ColorBar, HoverTool

    from ovro_lwa_portal.viz.dewarp_summary_plots import (
        build_dewarp_shift_figure,
        build_dewarp_shift_panel,
    )

    _write_phase2_qa_tree(tmp_path)
    cfg = _phase2_test_config(tmp_path)
    coverage = pq.scan_coverage(config=cfg)
    dewarp_df = pq.load_dewarp_summary_dataframe("2025-01-11", coverage, config=cfg)
    grid = pq.dewarp_shift_grid(dewarp_df)

    lst_num = int(coverage.iloc[0]["lst_hour_num"])
    assert grid.loc[lst_num, 55.0] == 0.12
    assert grid.loc[lst_num, 82.0] == 0.34

    figure = build_dewarp_shift_figure(grid)
    assert figure.select_one({"type": ColorBar}) is not None
    hover_tools = [tool for tool in figure.tools if isinstance(tool, HoverTool)]
    assert len(hover_tools) == 1

    panel = build_dewarp_shift_panel(dewarp_df)
    assert isinstance(panel, pn.Column)
    assert len(panel.objects) == 1


class _FakeCallbacks:
    def __init__(self) -> None:
        self._hold: str | None = None

    @property
    def hold_value(self) -> str | None:
        return self._hold

    def hold(self, policy: str = "combine") -> None:
        self._hold = policy

    def unhold(self) -> None:
        self._hold = None


class _FakeDoc:
    def __init__(self) -> None:
        self.callbacks = _FakeCallbacks()

    def hold(self, policy: str = "combine") -> None:
        self.callbacks.hold(policy)

    def unhold(self) -> None:
        self.callbacks.unhold()


class _FakeRoot:
    tags: list[str] = []


class _FakeView:
    def __init__(self, ref: str, *, children: tuple["_FakeView", ...] = ()) -> None:
        self._models = {ref: object()}
        self.objects = children


def _install_fake_view(
    monkeypatch: pytest.MonkeyPatch, *, ref: str = "root-1"
) -> tuple[_FakeView, _FakeDoc, list[Any]]:
    """Register one fake Panel view/document in panel state and capture pushes."""
    from panel.io.state import state

    doc = _FakeDoc()
    comm = object()
    monkeypatch.setattr(state, "_views", {ref: (object(), _FakeRoot(), doc, comm)})

    pushed: list[Any] = []
    monkeypatch.setattr(
        "panel.io.notebook.push", lambda d, c, *a, **k: pushed.append((d, c))
    )
    return _FakeView(ref), doc, pushed


def test_notebook_doc_comms_walks_nested_viewables(monkeypatch: pytest.MonkeyPatch) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import _notebook_doc_comms
    from panel.io.state import state

    doc = _FakeDoc()
    comm = object()
    child = pn.pane.HTML("child")
    parent = pn.Column(child)
    child_ref = "child-ref"
    child._models = {child_ref: object()}  # type: ignore[attr-defined]
    monkeypatch.setattr(
        state,
        "_views",
        {
            child_ref: (child, _FakeRoot(), doc, comm),
        },
    )

    docs = _notebook_doc_comms(parent)
    assert len(docs) == 1
    assert docs[id(doc)][1] is comm


def test_hold_and_push_enters_panel_hold_with_comm(monkeypatch: pytest.MonkeyPatch) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import hold_and_push

    view, doc, pushed = _install_fake_view(monkeypatch)

    with hold_and_push(view):
        pass

    assert doc.callbacks.hold_value is None
    assert pushed == [(doc, pushed[0][1])]


def test_hold_and_push_noops_when_views_not_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import hold_and_push
    from panel.io.state import state

    monkeypatch.setattr(state, "_views", {})
    pushed: list[Any] = []
    monkeypatch.setattr(
        "panel.io.notebook.push",
        lambda *args, **kwargs: pushed.append(args),
    )

    with hold_and_push(_FakeView("missing")):
        pass

    assert pushed == []


def test_sync_pane_to_notebook_uses_layout_root_when_pane_ref_unregistered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import sync_pane_to_notebook
    from panel.io.state import state

    doc = _FakeDoc()
    comm = object()
    layout_ref = "layout-ref"
    pane_ref = "pane-ref"
    layout = _FakeView(layout_ref)
    pane = _FakeView(pane_ref)
    parent = object()
    root = _FakeRoot()
    pane._models = {pane_ref: (object(), parent)}  # type: ignore[attr-defined]
    updates: list[tuple[str, Any, Any, Any, Any, Any]] = []

    def _update_object(ref: str, doc_arg: Any, root: Any, parent: Any, comm_arg: Any) -> None:
        updates.append((ref, doc_arg, root, parent, comm_arg))

    pane._update_object = _update_object  # type: ignore[attr-defined]
    monkeypatch.setattr(
        state,
        "_views",
        {layout_ref: (layout, root, doc, comm)},
    )

    sync_pane_to_notebook(pane, layout)

    assert updates == [(pane_ref, doc, root, parent, comm)]


def test_dispatch_notebook_ui_runs_inside_hold_when_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ovro_lwa_portal.viz import pipeline_qa_app as pqa

    view, doc, _pushed = _install_fake_view(monkeypatch)
    ran: list[str] = []

    monkeypatch.setattr(pqa, "_schedule_ipython_main", lambda fn: fn())

    pqa.dispatch_notebook_ui(lambda: ran.append("done"), view)

    assert ran == ["done"]
    assert doc.callbacks.hold_value is None


def test_dispatch_notebook_ui_reentrant_runs_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ovro_lwa_portal.viz import pipeline_qa_app as pqa

    view, _doc, _pushed = _install_fake_view(monkeypatch)
    scheduled: list[Callable[[], None]] = []
    ran: list[str] = []

    monkeypatch.setattr(pqa, "_schedule_ipython_main", lambda fn: scheduled.append(fn))
    depth: ContextVar[int] = pqa._NOTEBOOK_UI_DEPTH
    token = depth.set(1)
    try:
        pqa.dispatch_notebook_ui(lambda: ran.append("inline"), view)
    finally:
        depth.reset(token)

    assert ran == ["inline"]
    assert scheduled == []


def test_hold_and_push_real_document_pushes_nested_html_pane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integration: real Bokeh doc + sync_pane_to_notebook inside hold."""
    import param

    from ovro_lwa_portal.viz.pipeline_qa_app import hold_and_push, sync_pane_to_notebook
    from tests.viz.panel_ui_testkit import PanelUITestHarness

    harness = PanelUITestHarness()
    log_pane = pn.pane.HTML("<p>before</p>")
    layout = pn.Column(log_pane)
    harness.mount_layout_only(layout)

    with harness.capture_notebook_pushes(monkeypatch) as pushed:
        with hold_and_push(layout, log_pane):
            with param.parameterized.discard_events(log_pane):
                log_pane.object = "<p>after</p>"
            sync_pane_to_notebook(log_pane, layout)

    assert pushed == [(harness.doc, harness.comm)]
    assert log_pane.object == "<p>after</p>"


def test_set_notebook_pane_object_replaces_bokeh_figure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bokeh pane figure swap inside hold uses discard_events + sync."""
    from bokeh.plotting import figure

    from ovro_lwa_portal.viz.pipeline_qa_app import hold_and_push, set_notebook_pane_object
    from tests.viz.panel_ui_testkit import PanelUITestHarness

    def _make_fig(values: np.ndarray):
        n_t, n_f = values.shape
        plot = figure(width=400, height=200, x_range=(0, n_t), y_range=(0, n_f))
        plot.image(image=[values.T], x=0, y=0, dw=n_t, dh=n_f)
        return plot

    harness = PanelUITestHarness()
    zeros = np.zeros((4, 6))
    pane = pn.pane.Bokeh(_make_fig(zeros), height=200, sizing_mode="stretch_width")
    layout = pn.Column(pane)
    harness.mount(layout)
    layout_ref = harness.layout_ref(layout)
    root = harness.doc.roots[0]

    with harness.capture_notebook_pushes(monkeypatch) as pushed:
        with hold_and_push(layout, pane):
            set_notebook_pane_object(
                pane, _make_fig(np.arange(24.0).reshape(4, 6)), layout
            )

    assert pushed == [(harness.doc, harness.comm)]
    new_model = pane._models[layout_ref][0]
    assert root.children[0] is new_model


def test_sync_widget_to_notebook_updates_autocomplete_inside_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Widgets need sync_widget_to_notebook; sync_pane alone is a no-op."""
    from ovro_lwa_portal.viz.pipeline_qa_app import hold_and_push, set_notebook_widget_params
    from tests.viz.panel_ui_testkit import PanelUITestHarness

    harness = PanelUITestHarness()
    coord = pn.widgets.AutocompleteInput(
        name="Coordinate", value="", value_input="", restrict=False
    )
    layout = pn.Column(coord)
    harness.mount(layout)

    with harness.capture_notebook_pushes(monkeypatch) as pushed:
        with hold_and_push(layout, coord):
            set_notebook_widget_params(
                coord,
                layout,
                value="123.4, 56.7",
                value_input="123.4, 56.7",
                options=[],
            )

    model = harness.bokeh_model(coord, layout)
    assert pushed == [(harness.doc, harness.comm)]
    assert model.value == "123.4, 56.7"
    assert model.value_input == "123.4, 56.7"


def test_set_notebook_widget_params_spinner_spins_inside_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LoadingSpinner needs synthetic events — empty events skip CSS ``spin`` class."""
    from ovro_lwa_portal.viz.pipeline_qa_app import hold_and_push, set_notebook_widget_params
    from tests.viz.panel_ui_testkit import PanelUITestHarness

    harness = PanelUITestHarness()
    spinner = pn.indicators.LoadingSpinner(value=False, size=24)
    layout = pn.Column(spinner)
    harness.mount(layout)

    with harness.capture_notebook_pushes(monkeypatch) as pushed:
        with hold_and_push(layout, spinner):
            set_notebook_widget_params(
                spinner,
                layout,
                value=True,
                visible=True,
            )

    model = harness.bokeh_model(spinner, layout)
    assert pushed == [(harness.doc, harness.comm)]
    assert spinner.value is True
    assert "spin" in model.css_classes


def test_defer_after_notebook_hold_runs_bokeh_publish_after_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bokeh figure assign must not run inside hold_and_push (Jupiter path)."""
    from bokeh.plotting import figure

    from ovro_lwa_portal.viz.pipeline_qa_app import hold_and_push, publish_bokeh_pane_to_notebook
    from ovro_lwa_portal.viz.source_review_app import _placeholder_heatmap_figure
    from tests.viz.panel_ui_testkit import PanelUITestHarness

    harness = PanelUITestHarness()
    pane = pn.pane.Bokeh(_placeholder_heatmap_figure(), height=420, sizing_mode="stretch_width")
    layout = pn.Column(pane)
    harness.mount(layout)

    new_fig = figure(width=100, height=100, title="AFTER HOLD")

    with hold_and_push(layout, pane):
        publish_bokeh_pane_to_notebook(pane, new_fig, layout)

    model = harness.bokeh_model(pane, layout)
    assert model.title.text == "AFTER HOLD"


def test_discard_events_blocks_bokeh_pane_push(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: discard_events + push leaves the browser on the placeholder."""
    import param
    from bokeh.plotting import figure

    from ovro_lwa_portal.viz import pipeline_qa_app as pqa
    from ovro_lwa_portal.viz.source_review_app import _placeholder_heatmap_figure
    from tests.viz.panel_ui_testkit import PanelUITestHarness

    harness = PanelUITestHarness()
    pane = pn.pane.Bokeh(_placeholder_heatmap_figure(), height=420, sizing_mode="stretch_width")
    layout = pn.Column(pane)
    harness.mount(layout)

    pushed: list[Any] = []

    def _capture_push(*args: Any, **kwargs: Any) -> None:
        pushed.append(args)

    monkeypatch.setattr(pqa, "_push_panel_layout", _capture_push)

    new_fig = figure(width=100, height=100, title="SHOULD NOT STICK")
    with param.parameterized.discard_events(pane):
        pane.object = new_fig
    pqa._push_panel_layout(layout, pane)

    model = harness.bokeh_model(pane, layout)
    assert pane.object.title.text == "SHOULD NOT STICK"
    assert model.title.text != "SHOULD NOT STICK"


def test_publish_panel_widget_to_notebook_updates_autocomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ovro_lwa_portal.viz import pipeline_qa_app as pqa
    from ovro_lwa_portal.viz.pipeline_qa_app import publish_panel_widget_to_notebook
    from tests.viz.panel_ui_testkit import PanelUITestHarness

    harness = PanelUITestHarness()
    coord = pn.widgets.AutocompleteInput(value="", value_input="", restrict=False)
    layout = pn.Column(coord)
    harness.mount(layout)

    pushed: list[Any] = []

    def _capture_push(*args: Any, **kwargs: Any) -> None:
        pushed.append(args)

    monkeypatch.setattr(pqa, "_push_panel_layout", _capture_push)

    publish_panel_widget_to_notebook(
        coord, layout, value="12.3, 45.6", value_input="12.3, 45.6", options=[]
    )

    model = harness.bokeh_model(coord, layout)
    assert len(pushed) == 1
    assert coord.value == "12.3, 45.6"
    assert model.value == "12.3, 45.6"
    assert model.value_input == "12.3, 45.6"


def test_stokes_review_controls_row_property(monkeypatch: pytest.MonkeyPatch) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import _StokesReviewHolder

    holder = _StokesReviewHolder()
    assert holder.controls_row is holder._controls_row


def test_notebook_ui_views_includes_zenith_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        PipelineQAApp,
        "_start_initial_scan",
        lambda self, **kwargs: None,
    )
    app = PipelineQAApp()
    app.panel()
    views = app._notebook_ui_views()
    assert app._layout in views
    assert app._zenith_slot in views
    assert app._stokes_review.controls_row in views


def test_mount_zenith_sections_skips_stale_load_seq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PipelineQAApp,
        "_start_initial_scan",
        lambda self, **kwargs: None,
    )
    app = PipelineQAApp()
    app.panel()
    app._load_seq = 2
    banner_before = app._zenith_banner.object
    content_before = list(app._zenith_section_content["I"].objects)
    stale = pn.pane.Markdown("STALE MOUNT")
    publish_calls: list[str] = []

    monkeypatch.setattr(app, "_publish_zenith_heatmaps", lambda: publish_calls.append("publish"))

    app._mount_zenith_sections(
        {"I": stale, "V": stale},
        banner="STALE BANNER",
        load_seq=1,
    )

    assert app._zenith_banner.object == banner_before
    assert list(app._zenith_section_content["I"].objects) == content_before
    assert publish_calls == []


def test_publish_zenith_heatmaps_uses_ui_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import ZenithReviewPanel

    monkeypatch.setattr(
        PipelineQAApp,
        "_start_initial_scan",
        lambda self, **kwargs: None,
    )
    app = PipelineQAApp()
    app.panel()
    dataset = _mock_zenith_dataset()
    stat_map = np.ones((dataset.sizes["time"], dataset.sizes["frequency"]))
    panel = ZenithReviewPanel(
        dataset,
        stat_map,
        slice_selection=app._stokes_review.slice_selection,
        stokes_label="I",
        metric_label="STD",
    )
    app._stokes_review._panels["I"] = panel
    published: list[tuple[Any, Any]] = []

    class _RecordingUI:
        def publish_bokeh_figure(self, pane: Any, figure: Any) -> None:
            published.append((pane, figure))

    app._ui_session = _RecordingUI()
    app._publish_zenith_heatmaps()

    assert len(published) == 1
    assert published[0][0] is panel._heatmap.pane
    assert published[0][1] is panel._heatmap._plot


def test_zenith_heatmap_set_data_publishes_with_root_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ovro_lwa_portal.viz import pipeline_qa_app as pqa

    stat_map = np.ones((3, 5))
    lst_hours = np.array([12.0, 13.0, 14.0])
    freq_mhz = np.linspace(70.0, 90.0, 5)
    published: list[tuple[Any, Any, tuple[Any, ...]]] = []

    monkeypatch.setattr(
        pqa,
        "publish_bokeh_pane_to_notebook",
        lambda pane, figure, *roots: published.append((pane, figure, roots)),
    )

    selector = pqa._ZenithHeatmapSelector(
        stat_map,
        metric_label="STD",
        lst_hours=lst_hours,
        freq_mhz=freq_mhz,
        on_select=lambda _t, _f: None,
    )
    layout = pn.Column(selector.pane)
    new_map = np.full((3, 5), 2.0)
    selector.set_data(new_map, lst_hours=lst_hours, freq_mhz=freq_mhz, root_views=(layout,))

    assert len(published) == 1
    assert published[0][0] is selector.pane
    assert published[0][2] == (layout,)


def test_configure_pipeline_qa_notebook_patches_wcs_and_io_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ovro_lwa_portal.viz import pipeline_qa_app as pqa

    calls: list[str] = []
    monkeypatch.setattr(pqa, "_patch_astrowidget_get_wcs", lambda: calls.append("wcs"))
    monkeypatch.setattr(pqa, "_capture_ipython_io_loop", lambda: calls.append("io_loop"))
    monkeypatch.setattr(pqa.pn, "extension", lambda *args, **kwargs: calls.append("ext"))

    pqa.configure_pipeline_qa_notebook()

    assert calls[0] == "wcs"
    assert "ext" in calls
    assert calls[-1] == "io_loop"
