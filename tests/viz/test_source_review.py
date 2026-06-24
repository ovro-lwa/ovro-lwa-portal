"""Regression tests for the source review **Center** decision logic.

These pin the recurring bug where Center recentered on the wrong sky position.
They replay the exact action sequences reported by users (pan -> type name ->
Center, name -> Generate, sky-click -> Center) against the pure decision function
in :mod:`ovro_lwa_portal.viz.source_review`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord

from ovro_lwa_portal.viz.source_review import (
    DatasetLoad,
    finalize_dataset_load,
    plan_center_action,
    run_dataset_load,
    should_build_heatmap_grid,
)

# Cas A and a clearly different target (Cyg A) for "different source" cases.
CAS_A = SkyCoord(ra=350.8500 * u.deg, dec=58.8150 * u.deg, frame="icrs")
CYG_A = SkyCoord(ra=299.8682 * u.deg, dec=40.7339 * u.deg, frame="icrs")


def _nudged(coord: SkyCoord, arcsec: float) -> SkyCoord:
    """Return ``coord`` shifted east by ``arcsec`` (sub-tolerance jitter)."""
    return SkyCoord(
        ra=coord.ra + (arcsec / 3600.0) * u.deg,
        dec=coord.dec,
        frame="icrs",
    )


class TestCenterUsesFieldCoordinate:
    def test_pan_then_name_center_no_overlay_centers_field(self):
        # User panned (no heatmap yet), typed Cas A, clicked Center.
        plan = plan_center_action(CAS_A, heatmap_coord=None, has_overlay=False)
        assert plan.goto_center is CAS_A
        assert plan.overlay_center is None  # nothing loaded to reproject yet
        assert plan.reason == "center_hips_only"

    def test_center_on_loaded_matching_target_centers_field(self):
        # Cas A heatmap loaded, user panned away, Center -> back to Cas A overlay.
        plan = plan_center_action(CAS_A, heatmap_coord=CAS_A, has_overlay=True)
        assert plan.goto_center is CAS_A
        assert plan.overlay_center is CAS_A
        assert plan.field_matches_heatmap is True
        assert plan.reason == "center_overlay_match"

    def test_center_within_tolerance_still_matches(self):
        nearby = _nudged(CAS_A, 0.5)  # < 1 arcsec
        plan = plan_center_action(nearby, heatmap_coord=CAS_A, has_overlay=True)
        assert plan.field_matches_heatmap is True
        assert plan.overlay_center is nearby

    def test_goto_and_overlay_centers_are_identical_when_overlay_present(self):
        # The invariant the bug violated: goto and overlay use the SAME center.
        plan = plan_center_action(CAS_A, heatmap_coord=CAS_A, has_overlay=True)
        assert plan.goto_center is plan.overlay_center


class TestCenterDifferentTarget:
    def test_far_target_keeps_and_reprojects_overlay(self):
        # Cyg A heatmap loaded, type/click Cas A, Center -> KEEP overlay, reproject
        # onto Cas A. The overlay is a radio image valid across its footprint, so
        # Center must never make it vanish (it renders transparent where empty).
        plan = plan_center_action(CAS_A, heatmap_coord=CYG_A, has_overlay=True)
        assert plan.overlay_center is CAS_A
        assert plan.drop_heatmap_state is False
        assert plan.field_matches_heatmap is False
        assert plan.reason == "center_overlay_field"

    def test_field_target_drives_goto_not_heatmap_target(self):
        plan = plan_center_action(CAS_A, heatmap_coord=CYG_A, has_overlay=True)
        assert plan.goto_center is CAS_A
        assert plan.goto_center.separation(CYG_A) > 1 * u.deg


class TestClickOverlaySourceThenCenter:
    """The reported UX: click a source visible in the overlay, then Center."""

    def test_clicked_overlay_source_stays_visible_after_center(self):
        # Heatmap loaded for Cas A; user clicks a nearby feature in the overlay
        # (a few arcmin away -> different from the catalog target) and centers.
        clicked = SkyCoord(
            ra=CAS_A.ra + 5 * u.arcmin, dec=CAS_A.dec + 3 * u.arcmin, frame="icrs"
        )
        plan = plan_center_action(clicked, heatmap_coord=CAS_A, has_overlay=True)
        # Overlay is kept and reprojected onto the clicked source (not cleared),
        # and HiPS recenters on the same position -> the source stays centered.
        assert plan.overlay_center is clicked
        assert plan.goto_center is clicked
        assert plan.goto_center is plan.overlay_center
        assert plan.drop_heatmap_state is False
        assert plan.reason == "center_overlay_field"


class TestCenterNoOverlay:
    def test_name_center_without_overlay_drops_mismatched_heatmap(self):
        plan = plan_center_action(CAS_A, heatmap_coord=CYG_A, has_overlay=False)
        assert plan.overlay_center is None
        assert plan.drop_heatmap_state is True
        assert plan.reason == "center_hips_only"

    def test_name_center_without_overlay_keeps_matching_heatmap_state(self):
        plan = plan_center_action(CAS_A, heatmap_coord=CAS_A, has_overlay=False)
        assert plan.drop_heatmap_state is False
        assert plan.field_matches_heatmap is True


@pytest.mark.parametrize("has_overlay", [True, False])
def test_goto_always_field_coord(has_overlay):
    """goto_center is always the field coordinate, regardless of state."""
    for heatmap_coord in (None, CAS_A, CYG_A):
        plan = plan_center_action(
            CAS_A, heatmap_coord=heatmap_coord, has_overlay=has_overlay
        )
        assert plan.goto_center is CAS_A


class _DeferredDispatcher:
    """Collects callbacks instead of running them (like a busy main loop).

    Records the thread each callback is *scheduled* from and *executed* on, so
    tests can prove UI work never runs on the worker thread.
    """

    def __init__(self) -> None:
        self.pending: list[Callable[[], None]] = []
        self.scheduled_from: list[threading.Thread] = []
        self.executed_on: list[threading.Thread] = []

    def __call__(self, callback: Callable[[], None]) -> None:
        self.scheduled_from.append(threading.current_thread())
        self.pending.append(callback)

    def drain(self) -> None:
        while self.pending:
            callback = self.pending.pop(0)
            self.executed_on.append(threading.current_thread())
            callback()


def _fake_load(n_times: int = 3, n_freqs: int = 4) -> DatasetLoad:
    return DatasetLoad(
        dataset=object(),
        default_time_idx=0,
        default_freq_idx=n_freqs // 2,
        lst_hours=list(range(n_times)),
        freq_mhz=list(range(n_freqs)),
    )


class TestRunDatasetLoadThreading:
    """Pin the recurring 'heatmap never loads / log frozen' threading bug."""

    def test_no_ui_callback_runs_on_worker_thread(self):
        dispatcher = _DeferredDispatcher()
        loaded: list[DatasetLoad] = []
        logs: list[str] = []
        open_threads: list[threading.Thread] = []

        def _open(report: Callable[[str], None]) -> DatasetLoad:
            open_threads.append(threading.current_thread())
            report("Zarr opened")
            report("Coordinates ready")
            return _fake_load()

        worker = threading.Thread(
            target=lambda: run_dataset_load(
                open_dataset=_open,
                dispatch=dispatcher,
                on_loaded=loaded.append,
                on_error=lambda exc: pytest.fail(f"unexpected error {exc!r}"),
                log=logs.append,
            )
        )
        worker.start()
        worker.join(timeout=5.0)
        assert not worker.is_alive()

        # The slow open ran off the test's main thread...
        assert open_threads and open_threads[0] is not threading.main_thread()
        # ...and NOTHING touched UI state during the worker run.
        assert loaded == []
        assert logs == []
        assert dispatcher.executed_on == []
        # Everything was scheduled from the worker, deferred for the main thread.
        assert all(t is worker for t in dispatcher.scheduled_from)
        assert len(dispatcher.pending) == 3  # 2 progress reports + on_loaded

        # Draining on the main thread delivers logs (in order) then the load.
        dispatcher.drain()
        assert logs == ["Zarr opened", "Coordinates ready"]
        assert len(loaded) == 1
        assert all(t is threading.main_thread() for t in dispatcher.executed_on)

    def test_success_calls_on_loaded_not_on_error(self):
        dispatcher = _DeferredDispatcher()
        loaded: list[DatasetLoad] = []
        errors: list[BaseException] = []

        run_dataset_load(
            open_dataset=lambda report: _fake_load(),
            dispatch=dispatcher,
            on_loaded=loaded.append,
            on_error=errors.append,
        )
        dispatcher.drain()

        assert len(loaded) == 1
        assert errors == []

    def test_error_path_dispatches_on_error_only(self):
        dispatcher = _DeferredDispatcher()
        loaded: list[DatasetLoad] = []
        errors: list[BaseException] = []
        boom = RuntimeError("zarr open failed")

        def _open(report: Callable[[str], None]) -> DatasetLoad:
            report("Opening…")
            raise boom

        run_dataset_load(
            open_dataset=_open,
            dispatch=dispatcher,
            on_loaded=loaded.append,
            on_error=errors.append,
            log=lambda _msg: None,
        )
        dispatcher.drain()

        assert loaded == []
        assert errors == [boom]

    def test_progress_reported_before_load_completes(self):
        dispatcher = _DeferredDispatcher()
        order: list[str] = []

        def _open(report: Callable[[str], None]) -> DatasetLoad:
            report("step-1")
            report("step-2")
            return _fake_load()

        run_dataset_load(
            open_dataset=_open,
            dispatch=dispatcher,
            on_loaded=lambda _load: order.append("loaded"),
            on_error=lambda exc: pytest.fail(f"unexpected error {exc!r}"),
            log=order.append,
        )
        dispatcher.drain()

        assert order == ["step-1", "step-2", "loaded"]

    def test_progress_log_can_use_separate_scheduler(self):
        ui_dispatch = _DeferredDispatcher()
        log_dispatch = _DeferredDispatcher()
        order: list[str] = []

        def _open(report: Callable[[str], None]) -> DatasetLoad:
            report("step-1")
            return _fake_load()

        run_dataset_load(
            open_dataset=_open,
            dispatch=ui_dispatch,
            log_dispatch=log_dispatch,
            on_loaded=lambda _load: order.append("loaded"),
            on_error=lambda exc: pytest.fail(f"unexpected error {exc!r}"),
            log=order.append,
        )
        assert order == []
        log_dispatch.drain()
        assert order == ["step-1"]
        ui_dispatch.drain()
        assert order == ["step-1", "loaded"]


class TestShouldBuildHeatmapGrid:
    def test_builds_when_no_existing_grid(self):
        assert should_build_heatmap_grid(False, force=False) is True

    def test_skips_when_grid_present_and_not_forced(self):
        assert should_build_heatmap_grid(True, force=False) is False

    def test_force_rebuilds_even_with_existing_grid(self):
        assert should_build_heatmap_grid(True, force=True) is True
        assert should_build_heatmap_grid(False, force=True) is True


class TestFinalizeDatasetLoad:
    """Guard the post-open tail: heatmap grid + loading state must always settle."""

    def test_happy_path_runs_steps_in_order(self):
        order: list[str] = []
        finalize_dataset_load(
            mount_sky=lambda: order.append("mount"),
            build_heatmap_grid=lambda: order.append("heatmap"),
            clear_loading=lambda: order.append("clear"),
            on_step_error=lambda name, exc: order.append(("error", name)),
        )
        assert order == ["mount", "heatmap", "clear"]

    def test_heatmap_grid_built_and_loading_cleared_when_mount_raises(self):
        # The exact reported symptom: SkyWidget mount fails, but the clickable
        # heatmap grid must still appear and the spinner must still clear.
        order: list[str] = []
        errors: list[tuple[str, BaseException]] = []
        boom = RuntimeError("SkyWidget/HiPS failed")

        def _mount() -> None:
            order.append("mount")
            raise boom

        finalize_dataset_load(
            mount_sky=_mount,
            build_heatmap_grid=lambda: order.append("heatmap"),
            clear_loading=lambda: order.append("clear"),
            on_step_error=lambda name, exc: errors.append((name, exc)),
        )

        assert order == ["mount", "heatmap", "clear"]
        assert errors == [("mount_sky", boom)]

    def test_loading_cleared_when_heatmap_build_raises(self):
        order: list[str] = []
        errors: list[tuple[str, BaseException]] = []
        boom = RuntimeError("bokeh figure failed")

        def _build() -> None:
            order.append("heatmap")
            raise boom

        finalize_dataset_load(
            mount_sky=lambda: order.append("mount"),
            build_heatmap_grid=_build,
            clear_loading=lambda: order.append("clear"),
            on_step_error=lambda name, exc: errors.append((name, exc)),
        )

        assert order == ["mount", "heatmap", "clear"]
        assert errors == [("build_heatmap_grid", boom)]

    def test_clear_loading_runs_even_without_error_sink(self):
        cleared: list[bool] = []
        finalize_dataset_load(
            mount_sky=lambda: (_ for _ in ()).throw(RuntimeError("x")),
            build_heatmap_grid=lambda: (_ for _ in ()).throw(RuntimeError("y")),
            clear_loading=lambda: cleared.append(True),
            on_step_error=None,
        )
        assert cleared == [True]
