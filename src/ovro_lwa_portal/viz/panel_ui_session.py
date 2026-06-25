"""Panel/Bokeh notebook UI sync with pluggable session backends.

Production Jupyter apps and headless tests share the same update *intent* API
(``PanelUISession``) while each backend picks the comm strategy that actually
reaches the frontend (or the Bokeh model in tests).

Three sync tiers (do not collapse into one code path):

1. **Batch / dispatch** — spinner, status markdown inside :meth:`JupyterPanelUISession.dispatch`:
   assign on Python models, then one :func:`_push_panel_layout` sweep.
2. **Direct publish** — Bokeh heatmap figures: assign ``pane.object`` + push
   (Jupiter path; never inside ``doc.hold``, never ``discard_events``).
3. **Deferred publish** — Bokeh heatmap + coordinate field when a dispatch batch is
   active: ``publish_bokeh_pane_to_notebook`` (assign + sync + ``force_push``)
   runs before the batch layout push so the browser receives the new figure.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import logging

import panel as pn

logger = logging.getLogger(__name__)

from ovro_lwa_portal.viz.pipeline_qa_app import (
    DEFAULT_NOTEBOOK_UI_MAX_ATTEMPTS,
    _AFTER_HOLD_CALLBACKS,
    _NOTEBOOK_UI_DEPTH,
    _push_panel_layout,
    _schedule_ipython_main,
    _sync_all_notebook_views,
    defer_after_notebook_hold,
    hold_and_push,
    notebook_ui_hold_active,
    notebook_views_registered,
    publish_bokeh_pane_to_notebook,
    publish_panel_widget_to_notebook,
    set_notebook_pane_object,
    set_notebook_widget_params,
)

RootViewsFn = Callable[[], Sequence[pn.viewable.Viewable]]


@runtime_checkable
class PanelUISession(Protocol):
    """Sync Panel/Bokeh viewables to a live document (notebook or test harness)."""

    def dispatch(self, callback: Callable[[], None]) -> None:
        """Run ``callback`` inside a hold/push cycle on the UI thread."""
        ...

    def defer_dispatch(self, callback: Callable[[], None]) -> None:
        """Schedule a fresh hold/push cycle after the current one finishes."""
        ...

    def schedule(self, callback: Callable[[], None]) -> None:
        """Run ``callback`` on the next UI-thread turn (io_loop in Jupyter)."""
        ...

    def hold_active(self) -> bool:
        """True while inside :meth:`dispatch` (or equivalent batch)."""
        ...

    def sync_spinner(
        self,
        widget: pn.viewable.Viewable,
        *,
        value: bool,
        visible: bool,
    ) -> None:
        ...

    def sync_coordinate_field(
        self,
        widget: pn.viewable.Viewable,
        *,
        value: str,
        value_input: str,
    ) -> None:
        ...

    def sync_status_pane(self, pane: pn.viewable.Viewable, text: str) -> None:
        ...

    def publish_bokeh_figure(
        self,
        pane: pn.viewable.Viewable,
        figure: object,
        *,
        after_publish: Callable[[], None] | None = None,
    ) -> None:
        ...


class JupyterPanelUISession:
    """Production backend: Jupiter-style assign + ``_push_panel_layout``.

    ``hold_and_push`` / ``doc.hold('combine')`` does not reliably reach the browser
    for this layout in live Jupyter (tests pass with a synchronous io_loop and a
    fully registered view tree). Match ``jupiter_flux_review.ipynb``: mutate models
    on the kernel io_loop, push the layout comm, then run any deferred Bokeh/widget
    publishes in the same turn.
    """

    def __init__(self, root_views: RootViewsFn) -> None:
        self._root_views = root_views

    def _views(self) -> tuple[pn.viewable.Viewable, ...]:
        return tuple(self._root_views())

    def _push_views(self, *extra: pn.viewable.Viewable) -> None:
        seen: set[int] = set()
        ordered: list[pn.viewable.Viewable] = []
        for view in (*self._views(), *extra):
            key = id(view)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(view)
        _push_panel_layout(*ordered)

    def _run_dispatch_batch(self, callback: Callable[[], None]) -> None:
        """Run ``callback`` on the io_loop, push, then flush deferred publishes."""
        if not notebook_views_registered(*self._views()):
            callback()
            return

        after_batch: list[Callable[[], None]] = []
        depth_token = _NOTEBOOK_UI_DEPTH.set(_NOTEBOOK_UI_DEPTH.get() + 1)
        after_token = _AFTER_HOLD_CALLBACKS.set(after_batch)
        try:
            callback()
        finally:
            _AFTER_HOLD_CALLBACKS.reset(after_token)
            _NOTEBOOK_UI_DEPTH.reset(depth_token)
        # Assign deferred Bokeh/widget models before the batch push. Pushing first
        # leaves the browser on a stale zeros heatmap when publish runs second
        # (_push_panel_layout deduplicates by document id).
        for deferred in after_batch:
            deferred()
        # Nested ``pn.pane.Bokeh`` refs are often absent from ``state._views`` in
        # live Jupyter (layout root only). Sync before push or the browser keeps
        # the zeros placeholder even though ``pane.object`` updated in Python.
        _sync_all_notebook_views(*self._views())
        self._push_views()

    def dispatch(
        self,
        callback: Callable[[], None],
        *,
        _attempt: int = 0,
        max_attempts: int = DEFAULT_NOTEBOOK_UI_MAX_ATTEMPTS,
    ) -> None:
        if notebook_ui_hold_active():
            callback()
            return

        def _wrapped() -> None:
            if not notebook_views_registered(*self._views()):
                if _attempt < max_attempts:
                    self.dispatch(
                        callback,
                        _attempt=_attempt + 1,
                        max_attempts=max_attempts,
                    )
                else:
                    logger.error(
                        "Notebook Panel comm not registered after %d attempts; "
                        "skipped UI update callback %r",
                        max_attempts,
                        callback,
                    )
                return
            self._run_dispatch_batch(callback)

        _schedule_ipython_main(_wrapped)

    def defer_dispatch(self, callback: Callable[[], None]) -> None:
        if notebook_ui_hold_active():
            defer_after_notebook_hold(lambda: self.dispatch(callback))
            return

        def _wrapped() -> None:
            self.dispatch(callback)

        _schedule_ipython_main(_wrapped)

    def schedule(self, callback: Callable[[], None]) -> None:
        _schedule_ipython_main(callback)

    def hold_active(self) -> bool:
        return notebook_ui_hold_active()

    def sync_spinner(
        self,
        widget: pn.viewable.Viewable,
        *,
        value: bool,
        visible: bool,
    ) -> None:
        def _assign() -> None:
            widget.value = value
            widget.visible = visible

        def _push() -> None:
            # LoadingSpinner needs ``sync_widget_to_notebook`` for the ``spin`` CSS
            # class; a bare layout push often leaves the browser on a static icon.
            set_notebook_widget_params(
                widget,
                *self._views(),
                value=value,
                visible=visible,
            )
            _push_panel_layout(*self._views(), widget, _force=True)

        if notebook_ui_hold_active():
            defer_after_notebook_hold(_assign)
        defer_after_notebook_hold(_push)

    def sync_coordinate_field(
        self,
        widget: pn.viewable.Viewable,
        *,
        value: str,
        value_input: str,
    ) -> None:
        def _assign() -> None:
            widget.value = value
            widget.value_input = value_input
            widget.options = []

        def _push() -> None:
            publish_panel_widget_to_notebook(
                widget,
                *self._views(),
                value=value,
                value_input=value_input,
                options=[],
            )

        if notebook_ui_hold_active():
            defer_after_notebook_hold(_assign)
        else:
            defer_after_notebook_hold(_push)

    def sync_status_pane(self, pane: pn.viewable.Viewable, text: str) -> None:
        pane.object = text
        if self.hold_active():
            return
        _schedule_ipython_main(lambda: self._push_views(pane))

    def publish_bokeh_figure(
        self,
        pane: pn.viewable.Viewable,
        figure: object,
        *,
        after_publish: Callable[[], None] | None = None,
    ) -> None:
        def _publish() -> None:
            publish_bokeh_pane_to_notebook(
                pane,
                figure,
                *self._views(),
                force_push=True,
            )
            if after_publish is not None:
                after_publish()

        def _schedule_publish() -> None:
            _schedule_ipython_main(_publish)

        # Never publish inside the dispatch batch push: schedule a fresh io_loop
        # turn so assign + sync + force_push are not lost or overwritten.
        if notebook_ui_hold_active():
            defer_after_notebook_hold(_schedule_publish)
        else:
            _schedule_ipython_main(_publish)


class InlinePanelUISession:
    """Test / off-notebook backend: synchronous hold/push on a mounted document.

    Mount the layout with :class:`tests.viz.panel_ui_testkit.PanelUITestHarness`
    (or equivalent) so ``state._views`` is populated before calling app code.
    """

    def __init__(self, root_views: RootViewsFn) -> None:
        self._root_views = root_views

    def _views(self) -> tuple[pn.viewable.Viewable, ...]:
        return tuple(self._root_views())

    def dispatch(self, callback: Callable[[], None]) -> None:
        with hold_and_push(*self._views()):
            callback()

    def defer_dispatch(self, callback: Callable[[], None]) -> None:
        self.dispatch(callback)

    def schedule(self, callback: Callable[[], None]) -> None:
        defer_after_notebook_hold(callback)

    def hold_active(self) -> bool:
        return notebook_ui_hold_active()

    def sync_spinner(
        self,
        widget: pn.viewable.Viewable,
        *,
        value: bool,
        visible: bool,
    ) -> None:
        def _apply() -> None:
            set_notebook_widget_params(
                widget,
                *self._views(),
                value=value,
                visible=visible,
            )

        if self.hold_active():
            _apply()
        else:
            with hold_and_push(*self._views()):
                _apply()

    def sync_coordinate_field(
        self,
        widget: pn.viewable.Viewable,
        *,
        value: str,
        value_input: str,
    ) -> None:
        def _push() -> None:
            publish_panel_widget_to_notebook(
                widget,
                *self._views(),
                value=value,
                value_input=value_input,
                options=[],
            )

        defer_after_notebook_hold(_push)

    def sync_status_pane(self, pane: pn.viewable.Viewable, text: str) -> None:
        def _apply() -> None:
            set_notebook_pane_object(pane, text, *self._views())

        if self.hold_active():
            _apply()
        else:
            with hold_and_push(*self._views()):
                _apply()

    def publish_bokeh_figure(
        self,
        pane: pn.viewable.Viewable,
        figure: object,
        *,
        after_publish: Callable[[], None] | None = None,
    ) -> None:
        def _publish() -> None:
            publish_bokeh_pane_to_notebook(pane, figure, *self._views())
            if after_publish is not None:
                after_publish()

        defer_after_notebook_hold(_publish)


class ServedPanelUISession:
    """``panel serve`` backend: Bokeh server document updates, not Jupyter comm.

    Use from :mod:`scripts.serve_source_review` only. Mutating Panel models on the
    server event loop is enough; :func:`notebook_views_registered` is always false
    for embedded Bokeh server roots.
    """

    def __init__(self, root_views: RootViewsFn) -> None:
        self._root_views = root_views
        self._doc: Any | None = None
        self._pending: list[Callable[[], None]] = []

    def bind_document(self, doc: Any | None) -> None:
        """Capture the live Bokeh ``Document`` once ``panel.state.onload`` fires."""
        self._doc = doc
        if doc is None or doc.session_context is None:
            return
        pending = self._pending
        self._pending = []
        for callback in pending:
            self._schedule(callback)

    def _views(self) -> tuple[pn.viewable.Viewable, ...]:
        return tuple(self._root_views())

    def _schedule(self, callback: Callable[[], None]) -> None:
        if notebook_ui_hold_active():
            callback()
            return
        doc = self._doc or pn.state.curdoc
        if doc is not None and doc.session_context is not None:

            def _run() -> None:
                callback()

            doc.add_next_tick_callback(_run)
            return
        if threading.current_thread() is threading.main_thread():
            callback()
            return
        self._pending.append(callback)

    def dispatch(self, callback: Callable[[], None]) -> None:
        self._schedule(callback)

    def defer_dispatch(self, callback: Callable[[], None]) -> None:
        self._schedule(callback)

    def schedule(self, callback: Callable[[], None]) -> None:
        self._schedule(callback)

    def hold_active(self) -> bool:
        return notebook_ui_hold_active()

    def sync_spinner(
        self,
        widget: pn.viewable.Viewable,
        *,
        value: bool,
        visible: bool,
    ) -> None:
        set_notebook_widget_params(
            widget,
            *self._views(),
            value=value,
            visible=visible,
        )

    def sync_coordinate_field(
        self,
        widget: pn.viewable.Viewable,
        *,
        value: str,
        value_input: str,
    ) -> None:
        widget.value = value
        widget.value_input = value_input
        widget.options = []

    def sync_status_pane(self, pane: pn.viewable.Viewable, text: str) -> None:
        pane.object = text

    def publish_bokeh_figure(
        self,
        pane: pn.viewable.Viewable,
        figure: object,
        *,
        after_publish: Callable[[], None] | None = None,
    ) -> None:
        def _publish() -> None:
            pane.object = figure
            if after_publish is not None:
                after_publish()

        self._schedule(_publish)


class CallbackPanelUISession:
    """Legacy test hook: run ``dispatch`` callbacks inline without document sync."""

    def __init__(
        self,
        dispatch_fn: Callable[[Callable[[], None]], None],
        *,
        root_views: RootViewsFn | None = None,
    ) -> None:
        self._dispatch_fn = dispatch_fn
        self._root_views = root_views or (lambda: ())

    def _views(self) -> tuple[pn.viewable.Viewable, ...]:
        return tuple(self._root_views())

    def dispatch(self, callback: Callable[[], None]) -> None:
        self._dispatch_fn(callback)

    def defer_dispatch(self, callback: Callable[[], None]) -> None:
        self._dispatch_fn(callback)

    def schedule(self, callback: Callable[[], None]) -> None:
        callback()

    def hold_active(self) -> bool:
        return False

    def sync_spinner(
        self,
        widget: pn.viewable.Viewable,
        *,
        value: bool,
        visible: bool,
    ) -> None:
        widget.value = value
        widget.visible = visible

    def sync_coordinate_field(
        self,
        widget: pn.viewable.Viewable,
        *,
        value: str,
        value_input: str,
    ) -> None:
        widget.value = value
        widget.value_input = value_input
        widget.options = []

    def sync_status_pane(self, pane: pn.viewable.Viewable, text: str) -> None:
        pane.object = text

    def publish_bokeh_figure(
        self,
        pane: pn.viewable.Viewable,
        figure: object,
        *,
        after_publish: Callable[[], None] | None = None,
    ) -> None:
        pane.object = figure
        if after_publish is not None:
            after_publish()


@dataclass(frozen=True)
class UIRecord:
    """One recorded :class:`RecordingPanelUISession` operation."""

    operation: str
    payload: dict[str, Any] = field(default_factory=dict)


class RecordingPanelUISession:
    """Log UI sync intent for behavioral tests; optionally delegate model sync.

    Use without a delegate to assert call order from worker/io-loop code paths
    without mounting a Bokeh document. Wrap an :class:`InlinePanelUISession` to
    record **and** verify Bokeh models via :class:`tests.viz.panel_ui_testkit.PanelUITestHarness`.
    """

    def __init__(
        self,
        *,
        root_views: RootViewsFn | None = None,
        delegate: PanelUISession | None = None,
    ) -> None:
        self.records: list[UIRecord] = []
        self._root_views = root_views or (lambda: ())
        self._delegate = delegate
        self._hold_depth = 0

    def _record(self, operation: str, **payload: Any) -> None:
        self.records.append(UIRecord(operation, dict(payload)))

    def _views(self) -> tuple[pn.viewable.Viewable, ...]:
        return tuple(self._root_views())

    def dispatch(self, callback: Callable[[], None]) -> None:
        self._record("dispatch")
        if self._delegate is not None:
            self._delegate.dispatch(callback)
            return
        self._hold_depth += 1
        try:
            callback()
        finally:
            self._hold_depth -= 1

    def defer_dispatch(self, callback: Callable[[], None]) -> None:
        self._record("defer_dispatch")
        if self._delegate is not None:
            self._delegate.defer_dispatch(callback)
        else:
            self.dispatch(callback)

    def schedule(self, callback: Callable[[], None]) -> None:
        self._record("schedule")
        if self._delegate is not None:
            self._delegate.schedule(callback)
        else:
            callback()

    def hold_active(self) -> bool:
        if self._delegate is not None:
            return self._delegate.hold_active()
        return self._hold_depth > 0

    def sync_spinner(
        self,
        widget: pn.viewable.Viewable,
        *,
        value: bool,
        visible: bool,
    ) -> None:
        self._record("sync_spinner", value=value, visible=visible)
        if self._delegate is not None:
            self._delegate.sync_spinner(widget, value=value, visible=visible)

    def sync_coordinate_field(
        self,
        widget: pn.viewable.Viewable,
        *,
        value: str,
        value_input: str,
    ) -> None:
        self._record(
            "sync_coordinate_field",
            value=value,
            value_input=value_input,
        )
        if self._delegate is not None:
            self._delegate.sync_coordinate_field(
                widget,
                value=value,
                value_input=value_input,
            )

    def sync_status_pane(self, pane: pn.viewable.Viewable, text: str) -> None:
        self._record("sync_status_pane", text=text)
        if self._delegate is not None:
            self._delegate.sync_status_pane(pane, text)

    def publish_bokeh_figure(
        self,
        pane: pn.viewable.Viewable,
        figure: object,
        *,
        after_publish: Callable[[], None] | None = None,
    ) -> None:
        title = getattr(getattr(figure, "title", None), "text", None)
        self._record("publish_bokeh_figure", title=title)
        if self._delegate is not None:
            self._delegate.publish_bokeh_figure(pane, figure, after_publish=after_publish)
        else:
            pane.object = figure
            if after_publish is not None:
                after_publish()
