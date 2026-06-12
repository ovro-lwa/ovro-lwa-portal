"""Shared harness for headless Panel/Bokeh UI integration tests.

Registers a real :class:`bokeh.document.Document` in ``panel.io.state._views``
so :func:`ovro_lwa_portal.viz.pipeline_qa_app.hold_and_push` and the publish
helpers exercise the same code paths as Jupyter without monkeypatching ``push``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import panel as pn
import pytest
from bokeh.document import Document

from ovro_lwa_portal.viz.panel_ui_session import InlinePanelUISession, PanelUISession


class FakeNotebookComm:
    """Minimal comm stand-in accepted by ``panel.io.notebook.push``."""

    _comm = object()

    def send(self, *args: Any, **kwargs: Any) -> None:
        """No-op: headless tests only need push to run without a browser."""


class QueuedIOLoop:
    """Minimal Tornado-like loop for Jupyter comm regression tests."""

    def __init__(self) -> None:
        self.callbacks: list[Callable[[], None]] = []

    def add_callback(self, callback: Callable[[], None]) -> None:
        self.callbacks.append(callback)

    def flush(self) -> None:
        pending = self.callbacks[:]
        self.callbacks.clear()
        for callback in pending:
            callback()


class PanelUITestHarness:
    """Mount a Panel layout on a Bokeh document for comm integration tests."""

    def __init__(self) -> None:
        self.doc = Document()
        self.comm = FakeNotebookComm()
        self._mounted: dict[int, pn.viewable.Viewable] = {}

    def mount(self, layout: pn.viewable.Viewable) -> InlinePanelUISession:
        """Attach ``layout`` to the harness document and return a test session."""
        root = layout.get_root(self.doc)
        self.doc.add_root(root)
        self._register_view_tree(layout, root)
        session = InlinePanelUISession(lambda: (layout,))
        self._mounted[id(layout)] = layout
        return session

    def mount_layout_only(self, layout: pn.viewable.Viewable) -> InlinePanelUISession:
        """Register only the layout root ref (nested panes may be unregistered).

        Mirrors the notebook comm bug where nested ``pn.pane.HTML`` refs are absent
        from ``state._views`` even though the layout root is live.
        """
        root = layout.get_root(self.doc)
        self.doc.add_root(root)
        from panel.io.state import state

        layout_ref = next(iter(layout._models))
        state._views[layout_ref] = (layout, root, self.doc, self.comm)
        session = InlinePanelUISession(lambda: (layout,))
        self._mounted[id(layout)] = layout
        return session

    def session(self, layout: pn.viewable.Viewable) -> InlinePanelUISession:
        """Return an inline session for a previously mounted layout."""
        if id(layout) not in self._mounted:
            msg = "layout was not mounted; call mount(layout) first"
            raise RuntimeError(msg)
        return InlinePanelUISession(lambda: (layout,))

    def layout_ref(self, layout: pn.viewable.Viewable) -> str:
        """Primary ``state._views`` ref for ``layout``."""
        if not layout._models:
            msg = "layout has no Bokeh models; was it mounted?"
            raise RuntimeError(msg)
        return next(iter(layout._models))

    def bokeh_model(
        self,
        viewable: pn.viewable.Viewable,
        layout: pn.viewable.Viewable,
    ) -> Any:
        """Return the Bokeh model for ``viewable`` under ``layout``."""
        ref = self.layout_ref(layout)
        models = viewable._models.get(ref)
        if not models:
            msg = f"{viewable!r} has no model under layout ref {ref!r}"
            raise RuntimeError(msg)
        return models[0]

    def run_ui(
        self,
        session: PanelUISession,
        callback: Callable[[], None],
    ) -> None:
        """Execute a UI mutation through the session dispatch path."""
        session.dispatch(callback)

    @contextmanager
    def capture_notebook_pushes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> Iterator[list[tuple[Any, Any]]]:
        """Capture ``panel.io.notebook.push`` calls during a test block."""
        pushed: list[tuple[Any, Any]] = []

        def _capture(doc: Any, comm: Any, *args: Any, **kwargs: Any) -> None:
            pushed.append((doc, comm))

        monkeypatch.setattr("panel.io.notebook.push", _capture)
        monkeypatch.setattr("panel.pane.base.push", _capture)
        yield pushed

    def _register_view_tree(
        self,
        viewable: pn.viewable.Viewable,
        root: Any,
    ) -> None:
        from panel.io.state import state

        for panel_view in _iter_viewables(viewable):
            for ref in getattr(panel_view, "_models", {}) or {}:
                state._views[ref] = (panel_view, root, self.doc, self.comm)


def _iter_viewables(view: pn.viewable.Viewable) -> Sequence[pn.viewable.Viewable]:
    out: list[pn.viewable.Viewable] = [view]
    objects = getattr(view, "objects", None)
    if objects:
        for child in objects:
            if isinstance(child, pn.viewable.Viewable):
                out.extend(_iter_viewables(child))
    return out
