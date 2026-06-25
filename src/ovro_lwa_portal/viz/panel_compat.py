"""Panel version compatibility helpers."""

from __future__ import annotations

from typing import Any

import panel as pn

ButtonAppearance = str


def _supports_color(widget_or_class: Any) -> bool:
    param = getattr(widget_or_class, "param", None)
    return param is not None and "color" in param


def button_appearance_kwargs(
    appearance: ButtonAppearance,
    *,
    widget_class: type[Any] | None = None,
) -> dict[str, str]:
    """Kwargs for Panel buttons across 1.8 (``button_type``) and 1.9+ (``color``)."""
    cls = widget_class or pn.widgets.Button
    if _supports_color(cls):
        return {"color": appearance}
    return {"button_type": appearance}


def set_button_appearance(widget: Any, appearance: ButtonAppearance) -> None:
    """Set button appearance on an existing Panel button or toggle."""
    if _supports_color(widget):
        widget.color = appearance
        return
    widget.button_type = appearance


def button_appearance(widget: Any) -> str:
    """Read button appearance from a Panel button or toggle."""
    if _supports_color(widget):
        return str(widget.color)
    return str(widget.button_type)
