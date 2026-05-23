"""Tests for pipeline QA discovery, conversion helpers, and Panel app."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

pn = pytest.importorskip("panel")
np = pytest.importorskip("numpy")
widgets = pytest.importorskip("ipywidgets")

from ovro_lwa_portal.viz import pipeline_qa as pq
from ovro_lwa_portal.viz.pipeline_qa_app import PipelineQAApp


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
        lambda self: None,
    )
    app = PipelineQAApp()
    layout = app.panel()
    assert isinstance(layout, pn.Column)
    assert isinstance(app._zenith_review_row, pn.Row)
    assert set(app._zenith_section_content) == {"I", "V"}


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
        lambda self: None,
    )

    display_pipeline_qa_app()

    assert len(displayed) == 1
    assert isinstance(displayed[0], pn.Column)


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
    app.scanning = False
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.zarr_status",
        lambda _day: {"I": False, "V": False},
    )
    app.select_day = "2024-12-27"

    app._refresh_convert_button()

    assert app._convert_button.button_type == "primary"
    assert app._convert_button.disabled is False


def test_refresh_convert_button_uses_default_when_zarr_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = PipelineQAApp()
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)
    app.scanning = False
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.zarr_status",
        lambda _day: {"I": True, "V": True},
    )
    app.select_day = "2024-12-27"

    app._refresh_convert_button()

    assert app._convert_button.button_type == "default"
    assert app._convert_button.disabled is True


def test_day_selector_triggers_load_day(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    app = PipelineQAApp()
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)
    app.scanning = False
    monkeypatch.setattr(app, "_begin_load_day", lambda: calls.append(app.select_day or ""))
    app.select_day = "2024-12-27"
    app._loaded_day = None

    app._on_day_selector_changed(
        type("Event", (), {"new": "2024-12-28"})(),  # type: ignore[arg-type]
    )

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
    from ovro_lwa_portal.viz.pipeline_qa_app import _heatmap_axis_ticks

    ticks, labels = _heatmap_axis_ticks(5)
    assert ticks == [0.5, 1.5, 2.5, 3.5, 4.5]
    assert labels == {0.5: "0", 1.5: "1", 2.5: "2", 3.5: "3", 4.5: "4"}


def test_zenith_review_panel_slice_updates_status(monkeypatch: pytest.MonkeyPatch) -> None:
    import astropy.units as u
    import param

    from ovro_lwa_portal.viz.pipeline_qa_app import ZenithReviewPanel

    monkeypatch.setattr(
        ZenithReviewPanel,
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

    class _Dataset:
        sizes = {"time": 4, "frequency": 10}
        frequency = type("Freq", (), {"values": freq_values})()
        radport = _Radport()

    scheduled: list[Callable[[], None]] = []
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app._schedule_ipython_main",
        lambda callback: scheduled.append(callback),
    )
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app._run_on_main_thread",
        lambda callback: callback(),
    )

    panel = ZenithReviewPanel(
        _Dataset(),  # type: ignore[arg-type]
        stat_map,
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

    assert panel.time_idx == 2 and panel.freq_idx == 5
    assert panel._heatmap._time_idx == 2 and panel._heatmap._freq_idx == 5
    assert panel._heatmap._marker.data_source.data["x"] == [2.5]
    assert panel._heatmap._marker.data_source.data["y"] == [5.5]
    assert "time=2" in panel._slice_status()
    assert "freq=5" in panel._slice_status()
    assert "Stokes I" in panel._slice_status()
    assert panel._time_slider.value == 2
    assert panel._freq_slider.value == 5

    panel._select_slice(3, 6)
    _run_scheduled()
    assert panel.time_idx == 3 and panel.freq_idx == 6
    assert panel._time_slider.value == 3
    assert panel._freq_slider.value == 6
    assert "time=3" in panel._slice_status()

    with param.parameterized.batch_call_watchers(panel):
        panel.time_idx = 1
        panel.freq_idx = 2
    assert panel._heatmap._time_idx == 1 and panel._heatmap._freq_idx == 2
    assert "time=1" in panel._slice_status()

    pushed: list[int] = []
    panel.set_push_root(lambda: pushed.append(1))
    panel._select_slice(0, 3)
    _run_scheduled()
    assert pushed == [1]


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
    app.scanning = False
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.zarr_status",
        lambda _day: {"I": False, "V": False},
    )
    monkeypatch.setattr(
        app,
        "_begin_zenith_load",
        lambda *, load_seq: calls.append({"load_seq": load_seq}),
    )
    app._load_seq = 4
    app.select_day = "2024-12-27"
    app._loaded_day = "2024-12-27"

    app._auto_load_zenith_if_ready(4)

    assert calls == []


def test_auto_load_zenith_if_ready_starts_when_zarr_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, int]] = []
    app = PipelineQAApp()
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)
    app.scanning = False
    monkeypatch.setattr(
        "ovro_lwa_portal.viz.pipeline_qa_app.zarr_status",
        lambda _day: {"I": True, "V": False},
    )
    monkeypatch.setattr(
        app,
        "_begin_zenith_load",
        lambda *, load_seq: calls.append({"load_seq": load_seq}),
    )
    app._load_seq = 5
    app.select_day = "2024-12-27"
    app._loaded_day = "2024-12-27"

    app._auto_load_zenith_if_ready(5)

    assert calls == [{"load_seq": 5}]


def test_begin_zenith_load_rejects_stale_load_seq(monkeypatch: pytest.MonkeyPatch) -> None:
    app = PipelineQAApp()
    monkeypatch.setattr(app, "_start_initial_scan", lambda self: None)
    app._load_seq = 2
    app.select_day = "2024-12-27"
    app._loaded_day = "2024-12-27"

    app._begin_zenith_load(load_seq=1)

    assert app.loading_zenith is False


def test_sky_widget_host_mount_updates_container() -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import _SkyWidgetHost

    host = _SkyWidgetHost()

    class _FakeSky(widgets.HTML):
        def __init__(self) -> None:
            super().__init__(value="sky")

    class _FakePanel:
        def mount_sky(self) -> _FakeSky:
            return _FakeSky()

    host.mount({"I": _FakePanel(), "V": None})  # type: ignore[arg-type]
    assert len(host._containers["I"].children) == 2
    assert isinstance(host._containers["I"].children[1], _FakeSky)
    assert len(host._containers["V"].children) == 1
    assert isinstance(host.panel_row, pn.Row)
    assert host.panel_row.width == 1048
