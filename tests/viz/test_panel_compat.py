"""Tests for Panel version compatibility helpers."""

from __future__ import annotations

import panel as pn

from ovro_lwa_portal.viz.panel_compat import (
    button_appearance,
    button_appearance_kwargs,
    set_button_appearance,
)


def test_button_appearance_round_trip() -> None:
    button = pn.widgets.Button(name="Test", **button_appearance_kwargs("primary"))
    assert button_appearance(button) == "primary"
    set_button_appearance(button, "default")
    assert button_appearance(button) == "default"


def test_radio_button_group_accepts_appearance_kwargs() -> None:
    group = pn.widgets.RadioButtonGroup(
        name="Stokes",
        options=["I", "V"],
        **button_appearance_kwargs("default", widget_class=pn.widgets.RadioButtonGroup),
    )
    assert button_appearance(group) == "default"
