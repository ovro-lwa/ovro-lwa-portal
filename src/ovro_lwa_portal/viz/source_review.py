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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import astropy.units as u
from astropy.coordinates import SkyCoord

#: Coordinates within this separation are treated as the same sky target.
SAME_TARGET_TOLERANCE = 1 * u.arcsec

_LoadT = TypeVar("_LoadT")


@dataclass
class DatasetLoad:
    """Result of opening a review Zarr store on a background thread.

    Pure data so the load orchestration can be unit tested without xarray or a
    live Zarr store; the notebook fills this in from ``open_dataset`` + the
    LST/frequency coordinate computation.
    """

    dataset: Any
    default_time_idx: int
    default_freq_idx: int
    lst_hours: Any
    freq_mhz: Any


def run_dataset_load(
    *,
    open_dataset: Callable[[Callable[[str], None]], _LoadT],
    dispatch: Callable[[Callable[[], None]], None],
    on_loaded: Callable[[_LoadT], None],
    on_error: Callable[[BaseException], None],
    log: Callable[[str], None] | None = None,
    log_dispatch: Callable[[Callable[[], None]], None] | None = None,
) -> None:
    """Run the slow Zarr open and marshal every UI update onto the main thread.

    This is the body that ``source_review.ipynb`` runs on a background thread. It
    exists here, separate from the notebook and Panel app, because the recurring "heatmap never
    loads / activity log freezes" bug was a **threading** bug: UI callbacks were
    executed inline on the worker thread, where Panel/Bokeh comm updates are
    silently dropped and never reach the browser. The fix is invariant and
    testable: *nothing* that touches UI state may run on the worker — it must go
    through ``dispatch`` (which marshals onto the notebook/IPython main thread).

    The function therefore never calls ``on_loaded``, ``on_error``, or ``log``
    directly; it only ever passes them to ``dispatch``. Tests inject a deferred
    ``dispatch`` to assert that no callback executes while the worker runs, and
    that the success/error sequencing is correct.

    Parameters
    ----------
    open_dataset
        Performs the slow work on the calling (worker) thread and returns a
        load result (e.g. :class:`DatasetLoad`). It receives a ``report``
        callback for progress messages; each ``report`` call is itself routed
        through ``dispatch`` so progress text reaches the UI in order.
    dispatch
        Schedules a zero-argument callback on the notebook main thread (in the
        notebook this is ``_schedule_ipython_main``). Must defer, not run inline,
        when called from a worker thread.
    on_loaded
        Called (via ``dispatch``) with the load result on success.
    on_error
        Called (via ``dispatch``) with the exception on failure.
    log
        Optional progress-message sink for each ``report`` call from
        ``open_dataset``.
    log_dispatch
        Optional scheduler for progress log lines (defaults to ``dispatch``).
        In Jupyter, pass ``_schedule_ipython_main`` so open progress reaches the
        ipywidgets activity log even when Panel layout comm registration lags.
    """
    schedule_log = log_dispatch if log_dispatch is not None else dispatch

    def _report(message: str) -> None:
        if log is not None:
            schedule_log(lambda: log(message))

    try:
        load = open_dataset(_report)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI via on_error
        # Bind ``exc`` as a default arg: ``except ... as`` clears the name when
        # the block exits, so a deferred ``lambda: on_error(exc)`` would fail.
        dispatch(lambda error=exc: on_error(error))
        return
    dispatch(lambda: on_loaded(load))


def should_build_heatmap_grid(has_existing_grid: bool, *, force: bool) -> bool:
    """Whether the zeros heatmap grid should be (re)built.

    Extracted so the "heatmap never appears" guard is unit-testable. A fresh open
    has no grid yet, so it must build one; ``force`` rebuilds even when a grid is
    already shown (used to discard a stale computed spectrum).
    """
    if force:
        return True
    return not has_existing_grid


def finalize_dataset_load(
    *,
    mount_sky: Callable[[], None],
    build_heatmap_grid: Callable[[], None],
    clear_loading: Callable[[], None],
    on_step_error: Callable[[str, BaseException], None] | None = None,
) -> None:
    """Run post-open UI setup so the heatmap grid and loading state always settle.

    This is the success tail of ``_finish_open`` in ``source_review.ipynb``,
    extracted so its **error containment** is testable. The recurring "sky widget
    shows but no heatmap, log frozen" symptom happens when one post-open step
    raises inside an io-loop callback: the exception is swallowed by Tornado, the
    later steps never run, and the loading spinner never clears — with no error in
    the activity log.

    The invariants pinned by tests:

    - ``build_heatmap_grid`` is attempted even if ``mount_sky`` raises (so the
      clickable grid appears regardless of a SkyWidget/HiPS failure).
    - ``clear_loading`` always runs, even if a step raises.
    - Each step failure is surfaced through ``on_step_error`` instead of being
      swallowed, so the activity log shows *why* a step failed.

    Parameters
    ----------
    mount_sky
        Build and display the ``SkyWidget`` (HiPS background, dataset cube).
    build_heatmap_grid
        Build and display the clickable zeros heatmap grid.
    clear_loading
        Clear the loading flag/spinner; always invoked.
    on_step_error
        Optional sink called as ``on_step_error(step_name, exc)`` when a step
        raises. When ``None``, step errors are suppressed (but later steps and
        ``clear_loading`` still run).
    """
    try:
        for name, step in (
            ("mount_sky", mount_sky),
            ("build_heatmap_grid", build_heatmap_grid),
        ):
            try:
                step()
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                if on_step_error is not None:
                    on_step_error(name, exc)
    finally:
        clear_loading()


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
        drop_heatmap_state=bool(
            heatmap_coord is not None and not field_matches_heatmap
        ),
        field_matches_heatmap=field_matches_heatmap,
        reason="center_hips_only",
    )


def should_reset_heatmap_on_center(plan: CenterPlan) -> bool:
    """True when Center should discard a computed heatmap for a new field target.

    The open-time zeros grid is not a computed spectrum — do not republish it on
    Center when ``heatmap_coord`` is still ``None``. When a radio overlay is
    loaded, ``plan_center_action`` keeps ``drop_heatmap_state`` false so Center
    reprojects the overlay without clobbering the heatmap pane.
    """
    return plan.drop_heatmap_state
