"""Tests for pipeline QA discovery, conversion helpers, and Panel app."""

from __future__ import annotations

from collections.abc import Callable
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
    status = pq.zarr_status("2024-12-28")
    assert status == {"I": True, "V": False}


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
    assert app._stokes_review.zenith_footer in app._zenith_slot.objects
    zenith_objects = list(app._zenith_slot.objects)
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


def test_day_selector_triggers_load_day(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    app = PipelineQAApp()
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)
    monkeypatch.setattr(app, "_begin_load_day", lambda: None)
    app.scanning = False
    app._coverage = pd.DataFrame({"obs_date": ["2024-12-27", "2024-12-28"]})
    monkeypatch.setattr(app, "_begin_load_day", lambda: calls.append(app.select_day or ""))
    _set_select_day(app, "2024-12-27", days=["2024-12-27", "2024-12-28"])
    app._loaded_day = "2024-12-27"
    calls.clear()

    app.select_day = "2024-12-28"

    assert app.select_day == "2024-12-28"
    assert calls == ["2024-12-28"]


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

    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.zenith_lm_coord",
        lambda dataset, time_idx: _Coord(),
    )

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

    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.zenith_lm_coord",
        lambda dataset, time_idx: _Coord(),
    )

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

    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.zenith_lm_coord",
        lambda dataset, time_idx: _Coord(),
    )
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


def test_zenith_slice_syncs_status_on_slider_change(monkeypatch: pytest.MonkeyPatch) -> None:
    import param

    from ovro_lwa_portal.viz.pipeline_qa_app import ZenithReviewPanel, _StokesReviewHolder

    monkeypatch.setattr(
        _StokesReviewHolder,
        "_request_sky_update",
        lambda self, **kwargs: None,
    )

    class _Coord:
        ra = type("RA", (), {"to_string": lambda self, **kwargs: "12:00:00"})()
        dec = type("Dec", (), {"to_string": lambda self, **kwargs: "+00:00:00"})()

    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.zenith_lm_coord",
        lambda dataset, time_idx: _Coord(),
    )

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
    holder._configure_slice_selection()

    with param.parameterized.batch_call_watchers(holder.slice_selection):
        holder.slice_selection.time_idx = 2
        holder.slice_selection.freq_idx = 4

    assert "(2)" in panel._status_pane.object
    assert "(4)" in panel._status_pane.object


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
    """Run sky updates synchronously without slider debounce (unit tests)."""
    holder._sky_async = False  # type: ignore[attr-defined]
    holder._sky_slice_debounce_s = 0.0  # type: ignore[attr-defined]
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app._run_on_main_thread",
        lambda callback: callback(),
    )


def test_slider_freq_change_keeps_sky_view(monkeypatch: pytest.MonkeyPatch) -> None:
    import astropy.units as u

    from ovro_lwa_portal.viz.pipeline_qa_app import _StokesReviewHolder

    class _FakeSky:
        def __init__(self) -> None:
            self.update_slice_calls: list[dict[str, object]] = []

        def set_dataset(self, dataset, **kwargs) -> None:
            return None

        def update_slice(self, time_idx, freq_idx, **kwargs) -> None:
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

    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.zenith_lm_coord",
        lambda dataset, time_idx: type(
            "Coord",
            (),
            {
                "ra": type("RA", (), {"deg": 180.0})(),
                "dec": type("Dec", (), {"deg": 45.0})(),
                "icrs": property(lambda self: self),
            },
        )(),
    )

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

        def set_dataset(self, dataset, **kwargs) -> None:
            self.set_dataset_calls.append({"dataset": dataset, **kwargs})

        def update_slice(self, time_idx, freq_idx, **kwargs) -> None:
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
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.zenith_lm_coord",
        lambda dataset, time_idx: type(
            "Coord",
            (),
            {
                "ra": type("RA", (), {"deg": 180.0})(),
                "dec": type("Dec", (), {"deg": 45.0})(),
                "icrs": property(lambda self: self),
            },
        )(),
    )

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

    assert holder._sky_loading_row.visible is True
    assert holder._sky_loading_spinner.value is True

    holder.loading_sky = False
    holder._sync_sky_loading_indicator()

    assert holder._sky_loading_row.visible is False
    assert holder._sky_loading_spinner.value is False


def test_sky_update_sets_ready_status(monkeypatch: pytest.MonkeyPatch) -> None:
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
    holder._request_sky_update(debounce=False)

    assert holder._sky_error_alert.visible is True
    assert "slice failed" in holder._sky_error_alert.object
    assert "failed" in holder._sky_status_pane.object


def test_sky_slider_debounce_coalesces_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import _StokesReviewHolder

    calls: list[int] = []

    class _FakeSky:
        def set_dataset(self, dataset, **kwargs) -> None:
            return None

        def update_slice(self, time_idx, freq_idx, **kwargs) -> None:
            calls.append(int(freq_idx))

    class _Dataset:
        sizes = {"time": 4, "frequency": 4}

    holder = _StokesReviewHolder()
    holder._sky_async = False
    holder._sky_slice_debounce_s = 0.05
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app._run_on_main_thread",
        lambda callback: callback(),
    )
    holder._sky_widget = _FakeSky()  # type: ignore[assignment]
    holder._sky_bound_stokes = "I"
    holder.bind_datasets({"I": _Dataset()})  # type: ignore[arg-type]
    holder.slice_selection.configure(
        n_times=4,
        n_freqs=4,
        default_time=0,
        default_freq=0,
    )

    import time

    holder.slice_selection.freq_idx = 1
    holder.slice_selection.freq_idx = 2
    holder.slice_selection.freq_idx = 3
    time.sleep(0.12)

    assert calls == [3]
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
    assert holder._sky_status_pane in holder._zenith_footer.objects
    assert holder._sky_loading_row in holder._zenith_footer.objects
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

    panel_grid = build_flux_ratio_panel_grid(figures, n_cols=2)
    assert isinstance(panel_grid, pn.Column)
    assert len(panel_grid.objects) == 1
    assert panel_grid.max_width == FLUX_RATIO_GRID_TOTAL_WIDTH
