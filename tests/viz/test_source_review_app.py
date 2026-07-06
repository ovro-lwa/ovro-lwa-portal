"""Tests for the extracted SourceReview Panel app."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pn = pytest.importorskip("panel")
pytest.importorskip("astrowidget")

import astropy.units as u
from astropy.coordinates import SkyCoord

from astrowidget import SkyWidget

from ovro_lwa_portal.viz.panel_ui_session import ServedPanelUISession
from ovro_lwa_portal.viz.source_review_app import SourceReview, SourceReviewConfig


def test_astrowidget_update_slice_supports_view_lock() -> None:
    """Editable ``../astrowidget`` is required for source review overlay view-lock."""
    import inspect

    assert "view_lock" in inspect.signature(SkyWidget.update_slice).parameters
    assert "overlay_view_lock" in SkyWidget.class_trait_names()


def test_serve_sky_widget_coerces_null_background_cuts() -> None:
    """JSON null from ipywidgets_bokeh must not raise on Float HiPS cut traits."""
    import math

    from ovro_lwa_portal.viz.source_review_app import ServeSkyWidget

    widget = ServeSkyWidget()
    widget.background_cut_max = None
    assert math.isnan(widget.background_cut_max)
    widget.background_cut_min = None
    assert math.isnan(widget.background_cut_min)


def test_serve_sky_push_coalesces_rapid_schedules(tmp_path: Path) -> None:
    """Only the latest scheduled serve bundle remount should run."""
    holder: dict[str, SourceReview] = {}
    review = SourceReview(
        tmp_path / "store.zarr",
        patch_scale=5.0,
        sky_fov_deg=8.0,
        patch_fit_max_reduced_chi_squared=10.0,
        config=SourceReviewConfig(
            hips_root=tmp_path,
            hips_background=tmp_path / "missing.hips",
        ),
        validate_zarr=False,
        ui_session=_served_session(holder),
    )
    holder["review"] = review

    remounts = 0
    widget = MagicMock()
    widget.image_revision = 1

    def _count_remount(_widget: object) -> None:
        nonlocal remounts
        remounts += 1

    review._remount_sky_ipywidget_model = _count_remount  # type: ignore[method-assign]
    scheduled: list[Callable[[], None]] = []
    review._ui.schedule = lambda callback: scheduled.append(callback)  # type: ignore[method-assign]

    review._schedule_sky_widget_push(widget, force=True)
    review._schedule_sky_widget_push(widget, force=True)
    for callback in scheduled:
        callback()

    assert remounts == 1


def test_serve_mode_remounts_ipywidget_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``panel serve`` re-embeds SkyWidget via a fresh ipywidgets_bokeh bundle."""
    holder: dict[str, SourceReview] = {}
    review = SourceReview(
        tmp_path / "store.zarr",
        patch_scale=5.0,
        sky_fov_deg=8.0,
        patch_fit_max_reduced_chi_squared=10.0,
        config=SourceReviewConfig(
            hips_root=tmp_path,
            hips_background=tmp_path / "missing.hips",
        ),
        validate_zarr=False,
        ui_session=_served_session(holder),
    )
    holder["review"] = review

    remounted: list[object] = []
    monkeypatch.setattr(
        review,
        "_remount_sky_ipywidget_model",
        lambda widget: remounted.append(widget),
    )
    widget = MagicMock()
    widget.image_revision = 1
    review._maybe_send_sky_widget_state(widget, force=True)

    assert remounted == [widget]


def _served_session(review_holder: dict[str, SourceReview]) -> ServedPanelUISession:
    def _root_views() -> tuple:
        review = review_holder["review"]
        return (
            review._layout,
            review._status_pane,
            review._heatmap_pane,
            review._coord_input,
        )

    return ServedPanelUISession(_root_views)


def test_panel_serve_mode_defers_sky_widget_until_session(tmp_path: Path) -> None:
    zarr = tmp_path / "store.zarr"
    zarr.mkdir()
    holder: dict[str, SourceReview] = {}
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
        ui_session=_served_session(holder),
    )
    holder["review"] = review

    assert review._sky_widget is None
    assert review._loading_widget is None
    assert review._log_widget is None
    assert isinstance(review._log_pane, pn.pane.HTML)
    assert isinstance(review._sky_pane, pn.pane.HTML)
    assert review._loading_pane.value is False


def test_source_review_builds_layout_without_zarr_validation(tmp_path: Path) -> None:
    zarr = tmp_path / "missing.zarr"
    zarr.mkdir()

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

    assert review.panel is review._layout
    assert review._log_pane is not None
    assert "Zarr:" in review.log_text


def test_source_review_log_updates_via_inline_dispatch(tmp_path: Path) -> None:
    zarr = tmp_path / "store.zarr"
    zarr.mkdir()

    def _inline_dispatch(callback) -> None:
        callback()

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
        dispatch_override=_inline_dispatch,
    )

    review._dispatch(lambda: review._log("dispatch test line"))
    assert "dispatch test line" in review.log_text


def test_ui_action_handlers_schedule_through_dispatch(tmp_path: Path) -> None:
    """Panel buttons and param actions must enter the notebook dispatch batch."""
    zarr = tmp_path / "store.zarr"
    zarr.mkdir()
    scheduled: list[str] = []

    def _recording_dispatch(callback) -> None:
        scheduled.append(callback.__name__ if hasattr(callback, "__name__") else "lambda")
        callback()

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
        dispatch_override=_recording_dispatch,
    )

    review._on_slew()
    assert scheduled and scheduled[-1] == "_on_slew_impl"

    scheduled.clear()
    review._on_generate_heatmap()
    assert scheduled and scheduled[-1] == "_on_generate_heatmap_impl"

    scheduled.clear()
    review._on_heatmap_method_change()
    assert scheduled and scheduled[-1] == "_on_heatmap_method_change_impl"

    scheduled.clear()
    review._on_overlay_toggle(MagicMock(new=True))
    assert scheduled and scheduled[-1] == "_run"


def test_fit_overlay_button_exists(tmp_path: Path) -> None:
    zarr = tmp_path / "store.zarr"
    zarr.mkdir()
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
    assert review._fit_overlay_button.name == "Fit overlay"
    assert review._fit_overlay_button.disabled is True


def test_fit_overlay_click_schedules_impl(tmp_path: Path) -> None:
    zarr = tmp_path / "store.zarr"
    zarr.mkdir()
    scheduled: list[str] = []

    def _recording_dispatch(callback) -> None:
        scheduled.append(callback.__name__ if hasattr(callback, "__name__") else "lambda")
        callback()

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
        dispatch_override=_recording_dispatch,
    )

    started: list[bool] = []
    review._load_overlay_fit = lambda: started.append(True)  # type: ignore[method-assign]
    review._fit_overlay_button.disabled = False

    review._on_fit_overlay(None)
    assert scheduled and scheduled[-1] == "_on_fit_overlay_impl"
    assert started == [True]


def test_fit_overlay_button_sync_is_nonblocking(
    tmp_path: Path,
    valid_ovro_dataset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fit overlay gating must not scan the full cube on the UI thread."""
    zarr = tmp_path / "store.zarr"
    zarr.mkdir()
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
    review._dataset = valid_ovro_dataset
    review._coord = SkyCoord(ra=350.85 * u.deg, dec=58.815 * u.deg, frame="icrs")
    review._time_idx = 0
    review._freq_idx = 0

    def _fail_full_cache(self, **_kwargs) -> None:
        msg = "ensure_patch_metadata_cache must not run for Fit overlay button gating"
        raise AssertionError(msg)

    monkeypatch.setattr(
        "ovro_lwa_portal.accessor.RadportAccessor.ensure_patch_metadata_cache",
        _fail_full_cache,
    )
    review._sync_fit_overlay_button()
    assert review._fit_overlay_button.disabled is True


def test_configure_sky_widget_display_applies_config(tmp_path: Path) -> None:
    """SkyWidget display config should set overlay and background traits."""
    zarr = tmp_path / "store.zarr"
    zarr.mkdir()
    widget = MagicMock()
    review = SourceReview(
        zarr,
        patch_scale=5.0,
        sky_fov_deg=8.0,
        patch_fit_max_reduced_chi_squared=10.0,
        config=SourceReviewConfig(
            hips_root=tmp_path,
            hips_background=tmp_path / "bg.hips",
            background_cut_min=-1.0,
            background_cut_max=42.0,
            background_opacity=0.85,
            overlay_colormap="viridis",
            overlay_stretch="sqrt",
            overlay_opacity=0.75,
        ),
        validate_zarr=False,
    )
    review._log = MagicMock()  # type: ignore[method-assign]
    review._configure_sky_widget_display(widget)

    assert widget.colormap == "viridis"
    assert widget.stretch == "sqrt"
    assert widget.opacity == 0.75
    assert widget.background_opacity == 0.85
    assert widget.background_cut_min == -1.0
    assert widget.background_cut_max == 42.0


def test_overlay_scale_kwargs_and_fixed_scale(tmp_path: Path) -> None:
    zarr = tmp_path / "store.zarr"
    zarr.mkdir()
    review = SourceReview(
        zarr,
        patch_scale=5.0,
        sky_fov_deg=8.0,
        patch_fit_max_reduced_chi_squared=10.0,
        config=SourceReviewConfig(
            hips_root=tmp_path,
            hips_background=tmp_path / "missing.hips",
            overlay_percentile_low=5.0,
            overlay_percentile_high=95.0,
            overlay_vmin=-2.0,
            overlay_vmax=20.0,
        ),
        validate_zarr=False,
    )
    assert review._overlay_scale_kwargs() == {
        "percentile_low": 5.0,
        "percentile_high": 95.0,
    }
    widget = MagicMock()
    widget.vmin = 0.0
    widget.vmax = 1.0
    review._apply_overlay_fixed_scale(widget)
    assert widget.vmin == -2.0
    assert widget.vmax == 20.0


def test_stokes_toggle_hidden_until_iv_store_opened(tmp_path: Path) -> None:
    zarr = tmp_path / "store.zarr"
    zarr.mkdir()
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
    assert review._stokes_toggle.visible is False
    assert review._stokes_toggle in review._layout.objects[0].objects


def test_configure_stokes_enables_toggle_for_iv_dataset(tmp_path: Path) -> None:
    import numpy as np
    import xarray as xr

    zarr = tmp_path / "store.zarr"
    zarr.mkdir()
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
    ds = xr.Dataset(
        data_vars={
            "SKY": (
                ["time", "frequency", "polarization", "l", "m"],
                np.ones((1, 1, 2, 4, 4)),
            ),
        },
        coords={
            "time": [60000.0],
            "frequency": [50e6],
            "polarization": [1, 4],
            "l": np.arange(4),
            "m": np.arange(4),
        },
    )
    review._dataset = ds
    review._configure_stokes_from_dataset(ds)
    assert review._stokes_toggle.visible is True
    assert review.param.stokes.objects == ["I", "V"]
    assert review._pol_idx() == 0
    review.stokes = "V"
    assert review._pol_idx() == 1


def test_stokes_change_clears_computed_heatmap(tmp_path: Path) -> None:
    zarr = tmp_path / "store.zarr"
    zarr.mkdir()

    def _inline_dispatch(callback) -> None:
        callback()

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
        dispatch_override=_inline_dispatch,
    )
    import numpy as np
    import xarray as xr

    ds = xr.Dataset(
        data_vars={
            "SKY": (
                ["time", "frequency", "polarization", "l", "m"],
                np.ones((2, 2, 2, 4, 4)),
            ),
        },
        coords={
            "time": [60000.0, 60000.1],
            "frequency": [46e6, 50e6],
            "polarization": [1, 4],
            "l": np.arange(4),
            "m": np.arange(4),
        },
    )
    review._dataset = ds
    review._lst_hours = np.array([12.0, 13.0])
    review._freq_mhz = np.array([46.0, 50.0])
    review._configure_stokes_from_dataset(ds)
    review._cache[("Cas A", "mad", "I")] = object()
    review._heatmap_coord = SkyCoord(ra=0.0 * u.deg, dec=0.0 * u.deg, frame="icrs")
    review._heatmap_values = np.zeros((2, 2))
    review._heatmap_grid_ready = True
    review.stokes = "V"
    review._on_stokes_change_impl()
    assert review._cache == {}
    assert review._heatmap_coord is None
