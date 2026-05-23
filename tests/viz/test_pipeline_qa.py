"""Tests for pipeline QA discovery, conversion helpers, and Panel app."""

from __future__ import annotations

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


def test_convert_button_label_and_disabled() -> None:
    assert pq.convert_button_label({"I": True, "V": True}) == "Convert FITS → Zarr (complete)"
    assert pq.convert_button_label({"I": True, "V": False}) == "Convert Stokes V"
    assert pq.convert_button_disabled({"I": True, "V": True}, converting=False) is True
    assert pq.convert_button_disabled({"I": False, "V": False}, converting=False) is False


def test_stokes_review_holder_builds_both_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import _StokesReviewHolder

    holder = _StokesReviewHolder()
    stokes_order: list[str] = []

    def _record(spec, datasets, log, *, flush=None) -> pn.Column:
        stokes_order.append(spec.stokes)
        return pn.Column(pn.pane.Markdown(f"{spec.stokes} section"))

    monkeypatch.setattr(holder, "_build_section_for_spec", _record)
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


def test_zenith_review_panel_slice_updates_status(monkeypatch: pytest.MonkeyPatch) -> None:
    import astropy.units as u

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

    panel = ZenithReviewPanel(
        _Dataset(),  # type: ignore[arg-type]
        stat_map,
        stokes_label="I",
        metric_label="STD",
    )
    assert panel.time_idx == 0 and panel.freq_idx == 8

    panel._select_slice(2, 5)
    assert panel.time_idx == 2 and panel.freq_idx == 5
    assert "time=2" in panel._status_pane.object
    assert "freq=5" in panel._status_pane.object
    assert "Stokes I" in panel._status_pane.object


def test_sky_widget_host_mount_uses_display_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from ovro_lwa_portal.viz.pipeline_qa_app import _SKY_DISPLAY_ID, _SkyWidgetHost

    published: list[dict[str, Any]] = []

    def _display(obj: Any, *, display_id: str | None = None, update: bool = False, **_: Any) -> None:
        published.append(
            {
                "display_id": display_id,
                "update": update,
                "obj": obj,
            }
        )

    monkeypatch.setattr("IPython.display.display", _display)

    host = _SkyWidgetHost()
    host.mark_displayed()
    assert published[0]["display_id"] == _SKY_DISPLAY_ID
    assert published[0]["update"] is False

    class _FakeSky(widgets.HTML):
        def __init__(self) -> None:
            super().__init__(value="sky")

    class _FakePanel:
        def mount_sky(self) -> _FakeSky:
            return _FakeSky()

    host.mount({"I": _FakePanel(), "V": None})  # type: ignore[arg-type]
    assert published[-1]["display_id"] == _SKY_DISPLAY_ID
    assert published[-1]["update"] is True
    shell = published[-1]["obj"]
    assert isinstance(shell, widgets.VBox)
    assert isinstance(shell.children[0], widgets.HBox)
