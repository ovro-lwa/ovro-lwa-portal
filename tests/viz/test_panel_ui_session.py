"""Tests for PanelUISession backends and the shared test harness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pn = pytest.importorskip("panel")
from bokeh.plotting import figure

from ovro_lwa_portal.viz.source_review_app import SourceReview, SourceReviewConfig
from tests.viz.panel_ui_testkit import PanelUITestHarness


def test_harness_publishes_bokeh_figure_replaces_placeholder() -> None:
    from ovro_lwa_portal.viz.source_review_app import _placeholder_heatmap_figure

    harness = PanelUITestHarness()
    pane = pn.pane.Bokeh(_placeholder_heatmap_figure(), height=420)
    layout = pn.Column(pane)
    session = harness.mount(layout)

    def _make_fig(title: str) -> object:
        plot = figure(width=100, height=100, title=title)
        plot.image(image=[np.zeros((4, 4))], x=0, y=0, dw=4, dh=4)
        return plot

    session.publish_bokeh_figure(pane, _make_fig("GRID"))
    model = harness.bokeh_model(pane, layout)
    assert model.title.text == "GRID"

    session.publish_bokeh_figure(pane, _make_fig("GENERATED"))
    model = harness.bokeh_model(pane, layout)
    assert model.title.text == "GENERATED"


def test_harness_syncs_spinner_inside_dispatch() -> None:
    harness = PanelUITestHarness()
    spinner = pn.indicators.LoadingSpinner(value=False, size=24)
    layout = pn.Column(spinner)
    session = harness.mount(layout)

    def _run() -> None:
        session.sync_spinner(spinner, value=True, visible=True)

    harness.run_ui(session, _run)
    model = harness.bokeh_model(spinner, layout)
    assert spinner.value is True
    assert "spin" in model.css_classes


def test_harness_syncs_coordinate_after_dispatch_batch() -> None:
    harness = PanelUITestHarness()
    coord = pn.widgets.AutocompleteInput(value="", value_input="", restrict=False)
    status = pn.pane.Markdown("idle")
    layout = pn.Column(coord, status)
    session = harness.mount(layout)

    def _sky_click_batch() -> None:
        session.sync_status_pane(status, "clicked")
        session.sync_coordinate_field(
            coord,
            value="123.4, 56.7",
            value_input="123.4, 56.7",
        )

    harness.run_ui(session, _sky_click_batch)
    model = harness.bokeh_model(coord, layout)
    assert model.value == "123.4, 56.7"
    assert model.value_input == "123.4, 56.7"
    assert "clicked" in harness.bokeh_model(status, layout).text


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_source_review_accepts_inline_session(tmp_path: Path) -> None:
    zarr = tmp_path / "store.zarr"
    zarr.mkdir()
    harness = PanelUITestHarness()

    review = SourceReview(
        zarr,
        patch_scale=5.0,
        sky_fov_deg=8.0,
        patch_fit_max_reduced_chi_squared=10.0,
        config=SourceReviewConfig(
            hips_root=tmp_path,
            hips_background=tmp_path / "missing.hips",
        ),
        validate_zarr=False,
    )
    session = harness.mount(review._layout)
    review._ui_session_override = session
    review._ui_session = session

    def _run() -> None:
        review._sync_spinner(True)
        review._set_coordinate_field_from_text("10.0, 20.0")
        review._set_status("**ready**")

    harness.run_ui(session, _run)

    assert harness.bokeh_model(review._coord_input, review._layout).value == "10.0, 20.0"
    assert "spin" in harness.bokeh_model(review._spinner, review._layout).css_classes
    assert "ready" in harness.bokeh_model(review._status_pane, review._layout).text


def test_recording_session_logs_operations_without_document() -> None:
    from ovro_lwa_portal.viz.panel_ui_session import RecordingPanelUISession

    spinner = pn.indicators.LoadingSpinner(value=False, size=24)
    coord = pn.widgets.AutocompleteInput(value="", value_input="", restrict=False)
    status = pn.pane.Markdown("idle")
    pane = pn.pane.Bokeh(figure(width=10, height=10), height=10)
    recorder = RecordingPanelUISession()

    def _batch() -> None:
        recorder.sync_spinner(spinner, value=True, visible=True)
        recorder.sync_status_pane(status, "loading")
        recorder.sync_coordinate_field(coord, value="1, 2", value_input="1, 2")
        recorder.publish_bokeh_figure(pane, figure(width=10, height=10, title="GRID"))

    recorder.dispatch(_batch)

    assert [record.operation for record in recorder.records] == [
        "dispatch",
        "sync_spinner",
        "sync_status_pane",
        "sync_coordinate_field",
        "publish_bokeh_figure",
    ]
    assert recorder.records[0].payload == {}
    assert recorder.records[1].payload == {"value": True, "visible": True}
    assert recorder.records[3].payload["value"] == "1, 2"
    assert recorder.records[4].payload["title"] == "GRID"


def test_recording_session_delegates_to_inline_session() -> None:
    from ovro_lwa_portal.viz.panel_ui_session import RecordingPanelUISession

    harness = PanelUITestHarness()
    coord = pn.widgets.AutocompleteInput(value="", value_input="", restrict=False)
    layout = pn.Column(coord)
    inline = harness.mount(layout)
    recorder = RecordingPanelUISession(delegate=inline)

    recorder.sync_coordinate_field(coord, value="3, 4", value_input="3, 4")

    assert [record.operation for record in recorder.records] == ["sync_coordinate_field"]
    assert harness.bokeh_model(coord, layout).value == "3, 4"


def test_source_review_sky_click_records_ui_intent(tmp_path: Path) -> None:
    from ovro_lwa_portal.viz.panel_ui_session import RecordingPanelUISession

    zarr = tmp_path / "store.zarr"
    zarr.mkdir()
    recorder = RecordingPanelUISession()

    review = SourceReview(
        zarr,
        patch_scale=5.0,
        sky_fov_deg=8.0,
        patch_fit_max_reduced_chi_squared=10.0,
        config=SourceReviewConfig(
            hips_root=zarr,
            hips_background=zarr / "missing.hips",
        ),
        validate_zarr=False,
        ui_session=recorder,
    )

    review._dispatch(
        lambda: (
            review._set_coordinate_field_from_text("12.3, 45.6", log_prefix="Sky click →"),
            review._set_status("Sky click logged"),
        )
    )

    ops = [record.operation for record in recorder.records]
    assert "dispatch" in ops
    assert "sync_coordinate_field" in ops
    assert ops.count("sync_status_pane") >= 1


@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_jupyter_dispatch_batch_publishes_heatmap_on_next_io_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: heatmap publish must run on a fresh io_loop turn after dispatch."""
    from ovro_lwa_portal.viz import panel_ui_session as pus
    from ovro_lwa_portal.viz import pipeline_qa_app as pqa
    from ovro_lwa_portal.viz.panel_ui_session import JupyterPanelUISession
    from ovro_lwa_portal.viz.source_review_app import _placeholder_heatmap_figure
    from tests.viz.panel_ui_testkit import PanelUITestHarness, QueuedIOLoop

    harness = PanelUITestHarness()
    pane = pn.pane.Bokeh(_placeholder_heatmap_figure(), height=420)
    layout = pn.Column(pane)
    harness.mount(layout)
    loop = QueuedIOLoop()
    monkeypatch.setattr(pqa, "_IPYTHON_IO_LOOP", loop)
    monkeypatch.setattr(pqa, "_resolve_ipython_event_loop", lambda: loop)
    monkeypatch.setattr(pqa, "_is_jupyter_kernel_context", lambda: True)
    monkeypatch.setattr(pqa, "_schedule_ipython_main", loop.add_callback)
    monkeypatch.setattr(pqa, "notebook_views_registered", lambda *views: True)

    session = JupyterPanelUISession(lambda: (layout, pane))
    generated = figure(width=100, height=100, title="FL Cnc — tracked centre pixel")

    session.dispatch(lambda: session.publish_bokeh_figure(pane, generated))
    while loop.callbacks:
        loop.flush()

    assert harness.bokeh_model(pane, layout).title.text == "FL Cnc — tracked centre pixel"