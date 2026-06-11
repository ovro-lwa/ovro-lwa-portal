"""Regression tests for the source review **Center** decision logic.

These pin the recurring bug where Center recentered on the wrong sky position.
They replay the exact action sequences reported by users (pan -> type name ->
Center, name -> Generate, sky-click -> Center) against the pure decision function
in :mod:`ovro_lwa_portal.viz.source_review`.
"""

from __future__ import annotations

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord

from ovro_lwa_portal.viz.source_review import plan_center_action

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
