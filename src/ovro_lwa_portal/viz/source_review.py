"""Pure controller logic for the source review notebook.

The ``source_review.ipynb`` UI mixes Panel widgets, an ``astrowidget.SkyWidget``,
and WebGL rendering, none of which can be exercised headlessly. The *decisions*
that controller makes — which sky coordinate to recenter on, when to drop a stale
overlay — are pure functions of a handful of inputs, so they live here where they
can be unit tested. The notebook imports and calls these helpers instead of
duplicating the logic inline.

Splitting the logic this way lets tests pin the recurring "Center recenters on the
wrong position" bug to a concrete, reproducible decision. It does **not** verify
the WebGL reprojection itself; the notebook also logs the realized
``view_ra``/``view_dec``/``crval`` after each update so the rendering layer can be
checked against these decisions in a live session.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
from astropy.coordinates import SkyCoord

#: Coordinates within this separation are treated as the same sky target.
SAME_TARGET_TOLERANCE = 1 * u.arcsec


@dataclass(frozen=True)
class CenterPlan:
    """Decision for the **Center** button.

    Attributes
    ----------
    goto_center
        Sky position to pass to ``SkyWidget.goto`` (HiPS + view recenter).
    overlay_center
        Sky position to reproject the radio overlay onto, or ``None`` to leave
        the overlay untouched (e.g. nothing loaded yet).
    drop_heatmap_state
        Heatmap arrays no longer match the field target and should be discarded
        (only relevant when no overlay is loaded, so there is nothing to keep in
        sync with the dynamic-spectrum figure).
    field_matches_heatmap
        ``True`` when the field target is the same sky position as the loaded
        heatmap target.
    reason
        Short stable code for logging/testing.
    """

    goto_center: SkyCoord
    overlay_center: SkyCoord | None
    drop_heatmap_state: bool
    field_matches_heatmap: bool
    reason: str


def plan_center_action(
    field_coord: SkyCoord,
    heatmap_coord: SkyCoord | None,
    *,
    has_overlay: bool,
    same_target_tolerance: u.Quantity = SAME_TARGET_TOLERANCE,
) -> CenterPlan:
    """Decide what the **Center** button should do.

    The Center button always recenters HiPS and the view on the *coordinate
    field* (``field_coord``) — the resolved name or sky-clicked position — never
    on a stale heatmap target or a panned-away view.

    When a radio overlay is loaded it is **kept and reprojected** onto
    ``field_coord``. The overlay is a radio image valid across its whole
    footprint, so a source the user clicked in the overlay must stay visible and
    centered after Center; clicking a feature and centering on it should never
    make the overlay vanish. The overlay only changes when a new heatmap is
    generated (or it naturally renders transparent where it has no data). The
    catalog heatmap may then describe a different position than the new center;
    that is surfaced via ``field_matches_heatmap`` for messaging rather than by
    destroying the overlay.

    Parameters
    ----------
    field_coord
        Resolved coordinate currently in the coordinate field.
    heatmap_coord
        Sky target the loaded heatmap/overlay belongs to, or ``None`` when no
        heatmap has been generated.
    has_overlay
        Whether a radio overlay slice is currently displayed.
    same_target_tolerance
        Threshold for treating the field and heatmap as the same target
        (see module constant); exposed for testing.

    Returns
    -------
    CenterPlan
        Structured decision the notebook executes.
    """
    field_matches_heatmap = bool(
        heatmap_coord is not None
        and heatmap_coord.separation(field_coord) < same_target_tolerance
    )

    if has_overlay:
        return CenterPlan(
            goto_center=field_coord,
            overlay_center=field_coord,
            drop_heatmap_state=False,
            field_matches_heatmap=field_matches_heatmap,
            reason=(
                "center_overlay_match"
                if field_matches_heatmap
                else "center_overlay_field"
            ),
        )

    return CenterPlan(
        goto_center=field_coord,
        overlay_center=None,
        drop_heatmap_state=bool(not field_matches_heatmap),
        field_matches_heatmap=field_matches_heatmap,
        reason="center_hips_only",
    )
