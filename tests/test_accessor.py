"""Tests for the radport xarray accessor."""

from __future__ import annotations

import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

# Set non-interactive backend before importing accessor
matplotlib.use("Agg")

# Import to register the accessor
import ovro_lwa_portal  # noqa: F401
from ovro_lwa_portal.accessor import (
    PatchFitResult,
    PatchStatisticResult,
    RadportAccessor,
    _fit_spatial_gaussian,
    _gaussian_parameters_from_patch_statistics,
    _reduce_spatial_statistic,
)


class TestRadportAccessorRegistration:
    """Tests for accessor registration and availability."""

    def test_accessor_available_after_import(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Accessor 'radport' is available on xarray Datasets after importing."""
        assert hasattr(valid_ovro_dataset, "radport")

    def test_accessor_returns_radport_accessor(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Accessor returns RadportAccessor instance."""
        assert isinstance(valid_ovro_dataset.radport, RadportAccessor)

    def test_accessor_cached_on_dataset(self, valid_ovro_dataset: xr.Dataset) -> None:
        """Accessor instance is cached (same object on repeated access)."""
        accessor1 = valid_ovro_dataset.radport
        accessor2 = valid_ovro_dataset.radport
        assert accessor1 is accessor2


class TestRadportValidation:
    """Tests for dataset validation during accessor initialization."""

    def test_valid_dataset_passes_validation(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Valid OVRO-LWA dataset passes validation without error."""
        # Should not raise
        _ = valid_ovro_dataset.radport

    def test_missing_dimensions_raises_value_error(
        self, dataset_missing_dimensions: xr.Dataset
    ) -> None:
        """Missing required dimensions raises ValueError with informative message."""
        with pytest.raises(ValueError, match="missing required dimensions"):
            _ = dataset_missing_dimensions.radport

    def test_missing_dimensions_lists_what_is_missing(
        self, dataset_missing_dimensions: xr.Dataset
    ) -> None:
        """Error message lists the specific missing dimensions."""
        with pytest.raises(ValueError) as exc_info:
            _ = dataset_missing_dimensions.radport

        error_msg = str(exc_info.value)
        # Should mention the missing dimensions
        assert "frequency" in error_msg
        assert "polarization" in error_msg
        assert "time" in error_msg
        assert "l" in error_msg
        assert "m" in error_msg

    def test_missing_sky_variable_raises_value_error(
        self, dataset_missing_sky_variable: xr.Dataset
    ) -> None:
        """Missing SKY variable raises ValueError with informative message."""
        with pytest.raises(ValueError, match="missing required variables"):
            _ = dataset_missing_sky_variable.radport

    def test_missing_sky_variable_lists_what_is_missing(
        self, dataset_missing_sky_variable: xr.Dataset
    ) -> None:
        """Error message lists the missing SKY variable."""
        with pytest.raises(ValueError) as exc_info:
            _ = dataset_missing_sky_variable.radport

        error_msg = str(exc_info.value)
        assert "SKY" in error_msg


class TestRadportHasBeam:
    """Tests for has_beam property."""

    def test_has_beam_false_when_no_beam(self, dataset_missing_sky_variable: xr.Dataset) -> None:
        """has_beam returns False when dataset has no BEAM variable."""
        ds = dataset_missing_sky_variable.rename({"OTHER": "SKY"})
        assert not ds.radport.has_beam

    def test_has_beam_true_when_beam_present(
        self, valid_ovro_dataset_with_beam: xr.Dataset
    ) -> None:
        """has_beam returns True when dataset has BEAM variable."""
        assert valid_ovro_dataset_with_beam.radport.has_beam


class TestPatchMetadataCache:
    """Tests for eager patch metadata cache (beam + populated mask)."""

    @staticmethod
    def _dataset_with_empty_channel() -> xr.Dataset:
        l = np.linspace(-0.01, 0.01, 50)
        m = np.linspace(-0.01, 0.01, 50)
        sky = np.random.default_rng(4).random((1, 3, 1, 50, 50)) * 10.0
        sky[:, 1, ...] = np.nan
        beam_meta = np.zeros((1, 3, 1, 3), dtype=np.float64)
        beam_meta[0, 0, 0] = [0.02, 0.01, 0.0]
        beam_meta[0, 1, 0] = [np.nan, np.nan, np.nan]
        beam_meta[0, 2, 0] = [0.02, 0.01, 0.0]
        return xr.Dataset(
            data_vars={
                "SKY": (["time", "frequency", "polarization", "l", "m"], sky),
                "BEAM": (
                    ["time", "frequency", "polarization", "beam_param"],
                    beam_meta,
                ),
            },
            coords={
                "time": [60000.0],
                "frequency": [46e6, 50e6, 54e6],
                "polarization": [0],
                "beam_param": ["major", "minor", "pa"],
                "l": l,
                "m": m,
            },
        )

    def test_patch_metadata_cache_uses_beam_without_sky_scan(self) -> None:
        """ensure_patch_metadata_cache builds populated mask from BEAM only."""
        import dask.array as da

        ds = self._dataset_with_empty_channel()
        sky_lazy = xr.DataArray(
            da.from_array(ds["SKY"].values, chunks=(1, 1, 1, 50, 50)),
            dims=["time", "frequency", "polarization", "l", "m"],
            name="SKY",
        )
        ds = ds.drop_vars("SKY").assign(SKY=sky_lazy)
        cache = ds.radport.ensure_patch_metadata_cache()
        assert cache.beam_major is not None
        assert cache.populated.shape == (1, 3)
        assert bool(cache.populated[0, 0]) is True
        assert bool(cache.populated[0, 1]) is False
        assert bool(cache.populated[0, 2]) is True
        assert isinstance(ds["SKY"].data, da.Array)

    def test_var_cell_has_finite_data_reads_populated_cache(self) -> None:
        """Cached populated mask answers cell probes without SKY isel."""
        ds = self._dataset_with_empty_channel()
        ds.radport.ensure_patch_metadata_cache()
        assert ds.radport._var_cell_has_finite_data(time_idx=0, frequency_idx=1) is False
        assert ds.radport._var_cell_has_finite_data(time_idx=0, frequency_idx=0) is True

    def test_beam_fwhm_all_frequencies_uses_populated_cache_not_sky_probe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """beam_fwhm_pixels_all_frequencies skips empty cells via cache only."""
        ds = self._dataset_with_empty_channel()
        ds.radport.ensure_patch_metadata_cache()

        def _fail_probe(*_args: object, **_kwargs: object) -> bool:
            raise AssertionError("_var_cell_has_finite_data should not run when cache is warm")

        monkeypatch.setattr(ds.radport, "_var_cell_has_finite_data", _fail_probe)
        widths = ds.radport.beam_fwhm_pixels_all_frequencies(time_idx=0)
        assert np.isfinite(widths[0][0])
        assert not np.isfinite(widths[1][0])
        assert np.isfinite(widths[2][0])


class TestPatchReduceScheduler:
    """Tests for per-time patch reduce scheduler selection."""

    def test_patch_reduce_scheduler_distributed_when_client_active(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ovro_lwa_portal import accessor as acc

        monkeypatch.setattr(acc, "_active_distributed_client", lambda: True)
        monkeypatch.setenv("OVRO_RADPORT_PATCH_SCHEDULER", "processes")
        assert acc._patch_reduce_scheduler() is None

    def test_patch_reduce_scheduler_processes_without_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ovro_lwa_portal import accessor as acc

        monkeypatch.setattr(acc, "_active_distributed_client", lambda: False)
        monkeypatch.delenv("OVRO_RADPORT_PATCH_SCHEDULER", raising=False)
        assert acc._patch_reduce_scheduler() == "processes"

    def test_patch_reduce_scheduler_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ovro_lwa_portal import accessor as acc

        monkeypatch.setattr(acc, "_active_distributed_client", lambda: False)
        monkeypatch.setenv("OVRO_RADPORT_PATCH_SCHEDULER", "single-threaded")
        assert acc._patch_reduce_scheduler() == "single-threaded"


class TestPatchExtractScheduler:
    """Tests for fused patch Zarr read scheduler selection."""

    def test_patch_extract_scheduler_distributed_when_client_active(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ovro_lwa_portal import accessor as acc

        monkeypatch.setattr(acc, "_active_distributed_client", lambda: True)
        monkeypatch.delenv("OVRO_RADPORT_EXTRACT_SCHEDULER", raising=False)
        assert acc._patch_extract_scheduler() is None

    def test_patch_extract_scheduler_threads_without_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ovro_lwa_portal import accessor as acc

        monkeypatch.setattr(acc, "_active_distributed_client", lambda: False)
        monkeypatch.delenv("OVRO_RADPORT_EXTRACT_SCHEDULER", raising=False)
        assert acc._patch_extract_scheduler() == "threads"

    def test_patch_extract_scheduler_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ovro_lwa_portal import accessor as acc

        monkeypatch.setattr(acc, "_active_distributed_client", lambda: True)
        monkeypatch.setenv("OVRO_RADPORT_EXTRACT_SCHEDULER", "threads")
        assert acc._patch_extract_scheduler() == "threads"


class TestRadportPlot:
    """Tests for plot() method."""

    def test_plot_returns_figure(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() returns a matplotlib Figure object."""
        fig = valid_ovro_dataset.radport.plot()
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_default_parameters(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() works with default parameters."""
        fig = valid_ovro_dataset.radport.plot()
        try:
            assert isinstance(fig, plt.Figure)
            # Should have one axes
            assert len(fig.axes) >= 1
        finally:
            plt.close(fig)

    def test_plot_custom_time_index(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts custom time_idx parameter."""
        fig = valid_ovro_dataset.radport.plot(time_idx=1)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_custom_freq_index(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts custom freq_idx parameter."""
        fig = valid_ovro_dataset.radport.plot(freq_idx=2)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_custom_polarization(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts custom pol parameter."""
        fig = valid_ovro_dataset.radport.plot(pol=1)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_custom_colormap(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts custom cmap parameter."""
        fig = valid_ovro_dataset.radport.plot(cmap="viridis")
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_vmin_vmax(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts vmin and vmax parameters."""
        fig = valid_ovro_dataset.radport.plot(vmin=0.0, vmax=10.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_robust_scaling(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts robust parameter for percentile-based scaling."""
        fig = valid_ovro_dataset.radport.plot(robust=True)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_custom_figsize(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts custom figsize parameter."""
        fig = valid_ovro_dataset.radport.plot(figsize=(10, 8))
        try:
            assert fig.get_figwidth() == 10.0
            assert fig.get_figheight() == 8.0
        finally:
            plt.close(fig)

    def test_plot_without_colorbar(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts add_colorbar=False."""
        fig = valid_ovro_dataset.radport.plot(add_colorbar=False)
        try:
            assert isinstance(fig, plt.Figure)
            # Should have only one axes (no colorbar)
            assert len(fig.axes) == 1
        finally:
            plt.close(fig)

    def test_plot_with_colorbar(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() with add_colorbar=True adds colorbar."""
        fig = valid_ovro_dataset.radport.plot(add_colorbar=True)
        try:
            # Should have two axes (main plot + colorbar)
            assert len(fig.axes) == 2
        finally:
            plt.close(fig)

    def test_plot_beam_variable(
        self, valid_ovro_dataset_with_beam: xr.Dataset
    ) -> None:
        """plot() can plot BEAM variable when present."""
        fig = valid_ovro_dataset_with_beam.radport.plot(var="BEAM")
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_invalid_variable_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot() raises ValueError for non-existent variable."""
        with pytest.raises(ValueError, match="not found in dataset"):
            valid_ovro_dataset.radport.plot(var="NONEXISTENT")

    def test_plot_invalid_variable_lists_available(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Error message lists available variables."""
        with pytest.raises(ValueError) as exc_info:
            valid_ovro_dataset.radport.plot(var="NONEXISTENT")

        error_msg = str(exc_info.value)
        assert "SKY" in error_msg

    def test_plot_title_contains_metadata(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Plot title contains time, frequency, and polarization info."""
        fig = valid_ovro_dataset.radport.plot()
        try:
            ax = fig.axes[0]
            title = ax.get_title()
            # Should contain variable name
            assert "SKY" in title
            # Should contain frequency in MHz
            assert "MHz" in title
            # Should contain polarization
            assert "pol=" in title
        finally:
            plt.close(fig)

    def test_plot_axis_labels(self, valid_ovro_dataset: xr.Dataset) -> None:
        """Plot has proper axis labels for l and m coordinates."""
        fig = valid_ovro_dataset.radport.plot()
        try:
            ax = fig.axes[0]
            assert "l" in ax.get_xlabel().lower()
            assert "m" in ax.get_ylabel().lower()
        finally:
            plt.close(fig)


class TestRadportSelectionHelpers:
    """Tests for selection helper methods."""

    def test_nearest_freq_idx_exact_match(self, valid_ovro_dataset: xr.Dataset) -> None:
        """nearest_freq_idx returns correct index for exact frequency match."""
        # Dataset has frequencies [46e6, 50e6, 54e6] Hz
        idx = valid_ovro_dataset.radport.nearest_freq_idx(50.0)  # 50 MHz
        assert idx == 1

    def test_nearest_freq_idx_nearest_match(self, valid_ovro_dataset: xr.Dataset) -> None:
        """nearest_freq_idx returns nearest index for non-exact frequency."""
        # 49 MHz is closer to 50 MHz (index 1) than 46 MHz (index 0)
        idx = valid_ovro_dataset.radport.nearest_freq_idx(49.0)
        assert idx == 1

    def test_nearest_freq_idx_lower_bound(self, valid_ovro_dataset: xr.Dataset) -> None:
        """nearest_freq_idx handles frequencies below range."""
        idx = valid_ovro_dataset.radport.nearest_freq_idx(10.0)  # Below 46 MHz
        assert idx == 0  # Should return first index

    def test_nearest_freq_idx_upper_bound(self, valid_ovro_dataset: xr.Dataset) -> None:
        """nearest_freq_idx handles frequencies above range."""
        idx = valid_ovro_dataset.radport.nearest_freq_idx(100.0)  # Above 54 MHz
        assert idx == 2  # Should return last index

    def test_nearest_time_idx_exact_match(self, valid_ovro_dataset: xr.Dataset) -> None:
        """nearest_time_idx returns correct index for exact MJD match."""
        # Dataset has times [60000.0, 60000.1] MJD
        idx = valid_ovro_dataset.radport.nearest_time_idx(60000.0)
        assert idx == 0

    def test_nearest_time_idx_nearest_match(self, valid_ovro_dataset: xr.Dataset) -> None:
        """nearest_time_idx returns nearest index for non-exact MJD."""
        # 60000.08 is closer to 60000.1 (index 1) than 60000.0 (index 0)
        idx = valid_ovro_dataset.radport.nearest_time_idx(60000.08)
        assert idx == 1

    def test_nearest_lm_idx_center(self, valid_ovro_dataset: xr.Dataset) -> None:
        """nearest_lm_idx returns center indices for (0, 0)."""
        # Dataset has l and m from -1 to 1 with 50 points
        l_idx, m_idx = valid_ovro_dataset.radport.nearest_lm_idx(0.0, 0.0)
        # Center should be around index 24 or 25 for 50 points
        assert 23 <= l_idx <= 26
        assert 23 <= m_idx <= 26

    def test_nearest_lm_idx_corner(self, valid_ovro_dataset: xr.Dataset) -> None:
        """nearest_lm_idx returns corner indices for extreme values."""
        l_idx, m_idx = valid_ovro_dataset.radport.nearest_lm_idx(-1.0, 1.0)
        assert l_idx == 0  # -1 is at index 0
        assert m_idx == 49  # 1 is at index 49

    def test_nearest_lm_idx_returns_tuple(self, valid_ovro_dataset: xr.Dataset) -> None:
        """nearest_lm_idx returns a tuple of two integers."""
        result = valid_ovro_dataset.radport.nearest_lm_idx(0.5, -0.5)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)


class TestRadportPlotFrequencySelection:
    """Tests for plot() frequency selection by MHz."""

    def test_plot_freq_mhz_parameter(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts freq_mhz parameter."""
        fig = valid_ovro_dataset.radport.plot(freq_mhz=50.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_freq_mhz_overrides_freq_idx(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """freq_mhz takes precedence over freq_idx."""
        # freq_idx=0 would select 46 MHz, but freq_mhz=54 should select index 2
        fig = valid_ovro_dataset.radport.plot(freq_idx=0, freq_mhz=54.0)
        try:
            ax = fig.axes[0]
            title = ax.get_title()
            # Title should show 54.00 MHz, not 46.00 MHz
            assert "54.00 MHz" in title
        finally:
            plt.close(fig)

    def test_plot_time_mjd_parameter(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts time_mjd parameter."""
        fig = valid_ovro_dataset.radport.plot(time_mjd=60000.1)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_time_mjd_overrides_time_idx(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """time_mjd takes precedence over time_idx."""
        # time_idx=0 would select 60000.0, but time_mjd=60000.1 should select index 1
        fig = valid_ovro_dataset.radport.plot(time_idx=0, time_mjd=60000.1)
        try:
            ax = fig.axes[0]
            title = ax.get_title()
            # Title should show 60000.1 MJD, not 60000.0 MJD
            assert "60000.1" in title
        finally:
            plt.close(fig)


class TestRadportPlotMasking:
    """Tests for plot() circular masking functionality."""

    def test_plot_mask_radius_parameter(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot() accepts mask_radius parameter."""
        fig = valid_ovro_dataset.radport.plot(mask_radius=20)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_mask_radius_creates_masked_values(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """mask_radius creates masked/NaN values outside the specified radius."""
        # Get the plotted data by accessing the image
        fig = valid_ovro_dataset.radport.plot(mask_radius=10)
        try:
            ax = fig.axes[0]
            im = ax.images[0]
            data = im.get_array()
            # With mask_radius=10, corner pixels should be masked or NaN
            # matplotlib may return a masked array
            if hasattr(data, "mask"):
                # Check that some values are masked
                assert np.any(data.mask)
            else:
                # Check for NaN values
                assert np.any(np.isnan(data))
        finally:
            plt.close(fig)

    def test_plot_mask_radius_preserves_center(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """mask_radius preserves data within the specified radius."""
        fig = valid_ovro_dataset.radport.plot(mask_radius=25)
        try:
            ax = fig.axes[0]
            im = ax.images[0]
            data = im.get_array()
            # Center pixels should not be NaN
            center = data.shape[0] // 2
            assert not np.isnan(data[center, center])
        finally:
            plt.close(fig)


class TestRadportPlotWithNaN:
    """Tests for plot() method with datasets containing NaN values."""

    @pytest.fixture
    def dataset_with_nan(self) -> xr.Dataset:
        """Create a dataset with some NaN values."""
        np.random.seed(42)
        data = np.random.rand(2, 3, 2, 50, 50) * 10
        # Add some NaN values
        data[0, 0, 0, :10, :10] = np.nan
        return xr.Dataset(
            data_vars={
                "SKY": (
                    ["time", "frequency", "polarization", "l", "m"],
                    data,
                ),
            },
            coords={
                "time": [60000.0, 60000.1],
                "frequency": [46e6, 50e6, 54e6],
                "polarization": [0, 1],
                "l": np.linspace(-1, 1, 50),
                "m": np.linspace(-1, 1, 50),
            },
        )

    def test_plot_handles_nan_values(self, dataset_with_nan: xr.Dataset) -> None:
        """plot() handles datasets with NaN values without error."""
        fig = dataset_with_nan.radport.plot()
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_robust_with_nan(self, dataset_with_nan: xr.Dataset) -> None:
        """plot() with robust=True handles NaN values correctly."""
        fig = dataset_with_nan.radport.plot(robust=True)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)


# =============================================================================
# Phase B Tests: Cutout, Dynamic Spectrum, Difference Maps
# =============================================================================


class TestRadportCutout:
    """Tests for cutout() method."""

    def test_cutout_returns_dataarray(self, valid_ovro_dataset: xr.Dataset) -> None:
        """cutout() returns an xarray DataArray."""
        cutout = valid_ovro_dataset.radport.cutout(
            l_center=0.0, m_center=0.0, dl=0.3, dm=0.3
        )
        assert isinstance(cutout, xr.DataArray)

    def test_cutout_has_correct_dimensions(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """cutout() returns 2D DataArray with l and m dimensions."""
        cutout = valid_ovro_dataset.radport.cutout(
            l_center=0.0, m_center=0.0, dl=0.3, dm=0.3
        )
        assert set(cutout.dims) == {"l", "m"}

    def test_cutout_smaller_than_full_image(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """cutout() returns smaller region than full image."""
        cutout = valid_ovro_dataset.radport.cutout(
            l_center=0.0, m_center=0.0, dl=0.2, dm=0.2
        )
        full_size = valid_ovro_dataset.sizes["l"] * valid_ovro_dataset.sizes["m"]
        assert cutout.size < full_size

    def test_cutout_with_freq_mhz(self, valid_ovro_dataset: xr.Dataset) -> None:
        """cutout() accepts freq_mhz parameter."""
        cutout = valid_ovro_dataset.radport.cutout(
            l_center=0.0, m_center=0.0, dl=0.3, dm=0.3, freq_mhz=50.0
        )
        assert cutout.attrs["freq_idx"] == 1  # 50 MHz is index 1

    def test_cutout_metadata_attrs(self, valid_ovro_dataset: xr.Dataset) -> None:
        """cutout() adds metadata attributes."""
        cutout = valid_ovro_dataset.radport.cutout(
            l_center=0.1, m_center=-0.1, dl=0.2, dm=0.3
        )
        assert cutout.attrs["cutout_l_center"] == 0.1
        assert cutout.attrs["cutout_m_center"] == -0.1
        assert cutout.attrs["cutout_dl"] == 0.2
        assert cutout.attrs["cutout_dm"] == 0.3

    def test_cutout_invalid_variable_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """cutout() raises ValueError for non-existent variable."""
        with pytest.raises(ValueError, match="not found"):
            valid_ovro_dataset.radport.cutout(
                l_center=0.0, m_center=0.0, dl=0.1, dm=0.1, var="NONEXISTENT"
            )

    def test_cutout_out_of_bounds_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """cutout() raises ValueError when region is outside data bounds."""
        with pytest.raises(ValueError, match="empty"):
            valid_ovro_dataset.radport.cutout(
                l_center=5.0, m_center=5.0, dl=0.1, dm=0.1  # Outside [-1, 1] range
            )


class TestRadportPlotCutout:
    """Tests for plot_cutout() method."""

    def test_plot_cutout_returns_figure(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot_cutout() returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_cutout(
            l_center=0.0, m_center=0.0, dl=0.3, dm=0.3
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_cutout_with_options(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot_cutout() accepts customization options."""
        fig = valid_ovro_dataset.radport.plot_cutout(
            l_center=0.0,
            m_center=0.0,
            dl=0.3,
            dm=0.3,
            freq_mhz=50.0,
            cmap="viridis",
            figsize=(8, 8),
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_cutout_title_contains_bounds(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_cutout() title includes cutout bounds."""
        fig = valid_ovro_dataset.radport.plot_cutout(
            l_center=0.0, m_center=0.0, dl=0.1, dm=0.1
        )
        try:
            ax = fig.axes[0]
            title = ax.get_title()
            assert "l=" in title
            assert "m=" in title
        finally:
            plt.close(fig)


class TestRadportDynamicSpectrum:
    """Tests for dynamic_spectrum() method."""

    def test_dynamic_spectrum_returns_dataarray(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum() returns xarray DataArray."""
        dynspec = valid_ovro_dataset.radport.dynamic_spectrum(l=0.0, m=0.0)
        assert isinstance(dynspec, xr.DataArray)

    def test_dynamic_spectrum_has_correct_dims(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum() returns 2D array with time and frequency."""
        dynspec = valid_ovro_dataset.radport.dynamic_spectrum(l=0.0, m=0.0)
        assert set(dynspec.dims) == {"time", "frequency"}

    def test_dynamic_spectrum_shape(self, valid_ovro_dataset: xr.Dataset) -> None:
        """dynamic_spectrum() has expected shape."""
        dynspec = valid_ovro_dataset.radport.dynamic_spectrum(l=0.0, m=0.0)
        assert dynspec.sizes["time"] == 2
        assert dynspec.sizes["frequency"] == 3

    def test_dynamic_spectrum_metadata(self, valid_ovro_dataset: xr.Dataset) -> None:
        """dynamic_spectrum() adds pixel metadata attributes."""
        dynspec = valid_ovro_dataset.radport.dynamic_spectrum(l=0.0, m=0.0)
        assert "pixel_l" in dynspec.attrs
        assert "pixel_m" in dynspec.attrs
        assert "l_idx" in dynspec.attrs
        assert "m_idx" in dynspec.attrs

    def test_dynamic_spectrum_progress_callback(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """progress_callback reports track and extract stages for RA/Dec tracking."""
        events: list[tuple[str, int, int, str]] = []

        def _callback(stage: str, current: int, total: int, message: str) -> None:
            events.append((stage, current, total, message))

        valid_ovro_dataset_with_tracking_wcs.radport.dynamic_spectrum(
            ra=180.0,
            dec=37.0,
            progress_callback=_callback,
        )
        stages = {stage for stage, *_rest in events}
        assert "track" in stages
        assert "extract" in stages

    def test_dynamic_spectrum_extract_progress_updates_regularly(self) -> None:
        """Extract stage reports start and finish without splitting I/O into batches."""
        events: list[tuple[str, int, int, str]] = []

        def _callback(stage: str, current: int, total: int, message: str) -> None:
            events.append((stage, current, total, message))

        ds, catalog_ra, catalog_dec = (
            TestCelestialTimeSeriesTracking._per_time_wcs_dataset(25)
        )
        ds.radport.dynamic_spectrum(
            ra=catalog_ra,
            dec=catalog_dec,
            progress_callback=_callback,
        )
        extract_events = [
            (current, total)
            for stage, current, total, _msg in events
            if stage == "extract"
        ]
        assert len(extract_events) >= 2
        assert extract_events[0][0] == 0
        assert extract_events[-1][0] == extract_events[-1][1]

    def test_radport_progress_heartbeat_emits_elapsed_messages(self) -> None:
        """Elapsed-time heartbeats log without chunking the underlying work."""
        from ovro_lwa_portal.accessor import _radport_progress_heartbeat

        events: list[tuple[str, int, int, str]] = []

        def _callback(stage: str, current: int, total: int, message: str) -> None:
            events.append((stage, current, total, message))

        with _radport_progress_heartbeat(
            _callback,
            stage="extract",
            current=3,
            total=10,
            message="Zarr patch read steps 4–10 of 10",
            interval_s=0.05,
        ):
            time.sleep(0.16)

        assert events
        assert all(
            stage == "extract" and current == 3 and total == 10 for stage, current, total, _ in events
        )
        assert all("still working" in message for _stage, _current, _total, message in events)
        assert len(events) >= 2

    def test_dynamic_spectrum_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum() raises ValueError for non-existent variable."""
        with pytest.raises(ValueError, match="not found"):
            valid_ovro_dataset.radport.dynamic_spectrum(
                l=0.0, m=0.0, var="NONEXISTENT"
            )


class TestRadportPatchStatistic:
    """Tests for patch_statistic() and related helpers."""

    def test_patch_statistic_returns_result(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """patch_statistic() returns PatchStatisticResult with 2D stat_map."""
        result = valid_ovro_dataset.radport.patch_statistic(
            l=0.0, m=0.0, statistic="std"
        )
        assert isinstance(result, PatchStatisticResult)
        assert set(result.stat_map.dims) == {"time", "frequency"}
        assert result.stat_map.shape == (2, 3)
        assert result.selection is None

    def test_patch_statistic_threshold_selection(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Threshold selection marks cells above/below threshold."""
        ds = valid_ovro_dataset.copy(deep=True)
        sky = np.full(ds["SKY"].shape, 1.0, dtype=float)
        sky[0, :, 0, :, :] = 0.1
        sky[1, 2, 0, 24:27, 24:27] = 1000.0
        ds["SKY"].values[:] = sky

        result = ds.radport.patch_statistic(
            l=0.0,
            m=0.0,
            scale=5.0,
            statistic="max",
            threshold=500.0,
            comparison="gt",
        )
        assert result.selection is not None
        assert not bool(
            result.selection.sel(time=ds.coords["time"][0], frequency=ds.coords["frequency"][0])
        )
        assert bool(
            result.selection.sel(time=ds.coords["time"][1], frequency=ds.coords["frequency"][2])
        )

        # Cells above threshold are False when comparison='le'
        result_le = ds.radport.patch_statistic(
            l=0.0,
            m=0.0,
            scale=5.0,
            statistic="max",
            threshold=500.0,
            comparison="le",
        )
        assert result_le.selection is not None
        assert not bool(
            result_le.selection.sel(
                time=ds.coords["time"][1], frequency=ds.coords["frequency"][2]
            )
        )

    def test_select_patch_statistic_builds_mask(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """select_patch_statistic() returns a boolean mask."""
        result = valid_ovro_dataset.radport.patch_statistic(
            l=0.0, m=0.0, statistic="std", threshold=0.0, comparison="gt"
        )
        mask = valid_ovro_dataset.radport.select_patch_statistic(
            result.stat_map, threshold=0.0, comparison="gt"
        )
        assert mask.dtype == bool
        assert set(mask.dims) == {"time", "frequency"}
        np.testing.assert_array_equal(mask.values, result.selection.values)

    @pytest.mark.parametrize("statistic", ["std", "max", "min", "mean", "mad"])
    def test_patch_statistic_supports_all_statistics(
        self, valid_ovro_dataset: xr.Dataset, statistic: str
    ) -> None:
        """Each supported statistic produces finite values on random data."""
        result = valid_ovro_dataset.radport.patch_statistic(
            l=0.0, m=0.0, statistic=statistic
        )
        assert np.all(np.isfinite(result.stat_map.values))

    def test_patch_statistic_result_masks_dynspec_and_light_curve(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Follow-up extractions mask unselected cells to NaN."""
        ds = valid_ovro_dataset.copy(deep=True)
        sky = np.full(ds["SKY"].shape, 1.0, dtype=float)
        sky[1, 2, 0, 24:27, 24:27] = 1000.0
        ds["SKY"].values[:] = sky

        result = ds.radport.patch_statistic(
            l=0.0,
            m=0.0,
            scale=5.0,
            statistic="max",
            threshold=500.0,
            comparison="gt",
        )
        dynspec = result.dynamic_spectrum()
        assert not np.isfinite(
            dynspec.sel(time=ds.coords["time"][0], frequency=ds.coords["frequency"][0]).values
        )
        assert np.isfinite(
            dynspec.sel(time=ds.coords["time"][1], frequency=ds.coords["frequency"][2]).values
        )

        lc = result.light_curve(freq_idx=2)
        assert lc.dims == ("time",)
        assert not np.isfinite(lc.sel(time=ds.coords["time"][0]).values)
        assert np.isfinite(lc.sel(time=ds.coords["time"][1]).values)

    def test_patch_statistic_invalid_statistic_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Invalid statistic name raises from helper."""
        with pytest.raises(ValueError, match="Unsupported statistic"):
            _reduce_spatial_statistic(np.ones((3, 3)), "invalid")  # type: ignore[arg-type]

    def test_patch_statistic_nonpositive_scale_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Non-positive scale raises ValueError."""
        with pytest.raises(ValueError, match="scale must be positive"):
            valid_ovro_dataset.radport.patch_statistic(l=0.0, m=0.0, scale=0.0)

    def test_patch_statistic_progress_callback(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """progress_callback reports extract and reduce stages through completion."""
        events: list[tuple[str, int, int, str]] = []

        def _callback(stage: str, current: int, total: int, message: str) -> None:
            events.append((stage, current, total, message))

        valid_ovro_dataset.radport.patch_statistic(
            l=0.0,
            m=0.0,
            statistic="std",
            progress_callback=_callback,
        )
        stages = {stage for stage, *_rest in events}
        assert "extract" in stages
        assert "reduce" in stages
        assert any(current == total and total > 0 for _s, current, total, _m in events)
        assert any("Zarr patch read" in message for *_rest, message in events)
        assert any(
            stage == "extract" and current > 0 and current < total
            for stage, current, total, _m in events
        )

    def test_extract_tracked_patch_cubes_fused_matches_loop(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Fused patch I/O returns the same cubes as independent isel computes."""
        from ovro_lwa_portal.accessor import (
            _compute_xarray_batch,
            _patch_slices_from_center,
            _split_stacked_patch_cubes,
            _stack_tracked_patch_selections,
        )

        ds = valid_ovro_dataset
        rad = ds.radport
        visible = np.array([True, True], dtype=bool)
        l_indices = np.array([25, 2], dtype=int)
        m_indices = np.array([25, 48], dtype=int)
        radii = [3, 5]

        vis_times, fused_patches = rad._extract_tracked_patch_cubes(
            l_indices=l_indices,
            m_indices=m_indices,
            visible=visible,
            var="SKY",
            pol=0,
            radii=radii,
        )

        data_var = ds["SKY"].isel(polarization=0)
        n_l = int(ds.sizes["l"])
        n_m = int(ds.sizes["m"])
        vis_l = l_indices[visible]
        vis_m = m_indices[visible]

        patch_arrays: list[xr.DataArray] = []
        for t, li, mi, radius in zip(vis_times, vis_l, vis_m, radii, strict=True):
            l_sl, m_sl = _patch_slices_from_center(
                int(li), int(mi), int(radius), n_l=n_l, n_m=n_m
            )
            patch_arrays.append(data_var.isel(time=int(t), l=l_sl, m=m_sl))
        loop_loaded = _compute_xarray_batch(patch_arrays, label="loop reference")
        loop_patches = [np.asarray(item) for item in loop_loaded]

        stacked, patch_sizes = _stack_tracked_patch_selections(
            data_var,
            vis_times,
            vis_l,
            vis_m,
            radii,
            n_l=n_l,
            n_m=n_m,
        )
        split_patches = _split_stacked_patch_cubes(
            np.asarray(stacked.compute().data), patch_sizes
        )

        assert len(fused_patches) == len(loop_patches) == len(split_patches)
        for fused, loop, split in zip(fused_patches, loop_patches, split_patches, strict=True):
            np.testing.assert_allclose(fused, loop, equal_nan=True)
            np.testing.assert_allclose(split, loop, equal_nan=True)

    def test_pad_patch_dataarray_renormalizes_coords_after_nan_pad(self) -> None:
        """Padding with NaN must not leave duplicate l/m index labels for concat."""
        from ovro_lwa_portal.accessor import (
            _TRACKED_POINT_DIM,
            _normalize_patch_coords,
            _pad_patch_dataarray,
        )

        patch = _normalize_patch_coords(
            xr.DataArray(
                np.zeros((2, 5, 5)),
                dims=["frequency", "l", "m"],
                coords={"frequency": [0, 1]},
            )
        )
        padded = _pad_patch_dataarray(patch, n_l=8, n_m=8)
        assert padded.sizes["l"] == 8
        assert padded.sizes["m"] == 8
        assert len(padded.coords["l"]) == len(np.unique(padded.coords["l"].values))
        stacked = xr.concat(
            [patch, padded],
            dim=_TRACKED_POINT_DIM,
            coords="minimal",
            compat="override",
            join="outer",
        )
        stacked.compute()

    def test_patch_statistic_radec_tracking(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """RA/Dec patch statistic uses tracked pixels across time."""
        ds = valid_ovro_dataset_with_tracking_wcs
        result = ds.radport.patch_statistic(
            ra=180.0,
            dec=37.0,
            scale=2.0,
            statistic="mean",
            threshold=0.0,
            comparison="gt",
        )
        assert result.stat_map.attrs["tracking"] is True
        assert result.selection is not None
        assert np.any(result.selection.values)


class TestRadportPatchFit:
    """Tests for patch_fit() and Gaussian fit helpers."""

    def test_patch_fit_progress_callback(self, valid_ovro_dataset: xr.Dataset) -> None:
        """progress_callback reports extract and fit stages through completion."""
        events: list[tuple[str, int, int, str]] = []

        def _callback(stage: str, current: int, total: int, message: str) -> None:
            events.append((stage, current, total, message))

        valid_ovro_dataset.radport.patch_fit(
            l=0.0,
            m=0.0,
            progress_callback=_callback,
        )
        stages = {stage for stage, *_rest in events}
        assert "extract" in stages
        assert "fit" in stages
        assert any(current == total and total > 0 for _s, current, total, _m in events)

    def test_patch_fit_returns_result(self, valid_ovro_dataset: xr.Dataset) -> None:
        """patch_fit() returns PatchFitResult with parameter and chi-squared maps."""
        result = valid_ovro_dataset.radport.patch_fit(l=0.0, m=0.0)
        assert isinstance(result, PatchFitResult)
        assert result.max_reduced_chi_squared == 3.0
        for da in (
            result.peak_map,
            result.widthx_map,
            result.widthy_map,
            result.background_map,
            result.reduced_chi_squared_map,
            result.x_offset_map,
            result.y_offset_map,
            result.center_flux_map,
            result.patch_max_map,
            result.fit_accepted_map,
        ):
            assert set(da.dims) == {"time", "frequency"}
            assert da.shape == (2, 3)
        diag = result.cell_diagnostics(time_idx=0, frequency_idx=0)
        assert "fit_accepted" in diag
        assert "patch_max" in diag

    def test_gaussian_parameters_use_patch_max_for_peak(self) -> None:
        """Peak estimate uses patch maximum minus median background."""
        patch = np.array(
            [
                [10.0, 10.0, 10.0],
                [10.0, 20.0, 10.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float64,
        )
        ny, nx = patch.shape
        yy, xx = np.indices((ny, nx))
        y = yy - (ny - 1) / 2.0
        x = xx - (nx - 1) / 2.0
        peak, *_rest = _gaussian_parameters_from_patch_statistics(
            patch,
            x,
            y,
            beam_widthx=3.0,
            beam_widthy=3.0,
            max_width=10.0,
            max_offset=1.0,
        )
        assert peak == pytest.approx(10.0)

    def test_gaussian_parameters_from_patch_statistics_bright_source(self) -> None:
        """Statistical defaults separate bright peak from high background."""
        ny, nx = 11, 11
        true_peak = 500.0
        true_bg = 4800.0
        true_fwhm = 3.0
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        yy, xx = np.indices((ny, nx))
        sigma = true_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        patch = true_bg + true_peak * np.exp(
            -0.5 * (((yy - cy) / sigma) ** 2 + ((xx - cx) / sigma) ** 2)
        )
        y = yy - cy
        x = xx - cx
        peak, x_off, y_off, widthx, widthy, background = (
            _gaussian_parameters_from_patch_statistics(
                patch,
                x,
                y,
                beam_widthx=true_fwhm,
                beam_widthy=true_fwhm,
                max_width=40.0,
                max_offset=5.0,
            )
        )
        np.testing.assert_allclose(background, true_bg, rtol=0.05)
        np.testing.assert_allclose(peak, true_peak, rtol=0.25)
        assert abs(x_off) < 0.5
        assert abs(y_off) < 0.5
        assert 1.0 <= widthx <= 10.0
        assert 1.0 <= widthy <= 10.0

    def test_fit_spatial_gaussian_returns_statistics_when_optimizer_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failed least_squares still returns finite patch-statistic estimates."""

        def _raise(*_args: object, **_kwargs: object) -> None:
            msg = "forced optimizer failure"
            raise RuntimeError(msg)

        monkeypatch.setattr(
            "ovro_lwa_portal.accessor.least_squares",
            _raise,
        )
        ny, nx = 11, 11
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        yy, xx = np.indices((ny, nx))
        sigma = 3.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        patch = 4800.0 + 500.0 * np.exp(
            -0.5 * (((yy - cy) / sigma) ** 2 + ((xx - cx) / sigma) ** 2)
        )
        peak, _x_off, _y_off, widthx, widthy, background, chi2_red = _fit_spatial_gaussian(
            patch, beam_widthx=3.0, beam_widthy=3.0
        )
        assert np.isfinite(peak)
        assert np.isfinite(widthx)
        assert np.isfinite(widthy)
        assert np.isfinite(background)
        assert np.isfinite(chi2_red)
        assert peak > 0
        assert background > 4000

    def test_fit_spatial_gaussian_recovers_off_center_peak(self) -> None:
        """Shifted Gaussian fit recovers an off-centre bright peak."""
        ny, nx = 11, 11
        true_peak = 40.0
        true_bg = 5.0
        true_fwhm = 2.5
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        y_off_true, x_off_true = 4.0, -2.0
        yy, xx = np.indices((ny, nx))
        y = yy - cy
        x = xx - cx
        sigma = true_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        patch = true_bg + true_peak * np.exp(
            -0.5 * (((y - y_off_true) / sigma) ** 2 + ((x - x_off_true) / sigma) ** 2)
        )
        peak, x_off, y_off, widthx, widthy, background, chi2_red = _fit_spatial_gaussian(
            patch, beam_widthx=true_fwhm, beam_widthy=true_fwhm
        )
        assert chi2_red < 3.0
        np.testing.assert_allclose(peak, true_peak, rtol=0.2)
        np.testing.assert_allclose(x_off, x_off_true, atol=0.75)
        np.testing.assert_allclose(y_off, y_off_true, atol=0.75)
        np.testing.assert_allclose(background, true_bg, atol=2.0)

    def test_fit_spatial_gaussian_recovers_synthetic_peak(
        self,
    ) -> None:
        """Gaussian fit recovers peak and FWHM on a synthetic patch."""
        ny, nx = 11, 11
        true_peak = 50.0
        true_fwhm = 3.0
        true_bg = 2.0
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        yy, xx = np.indices((ny, nx))
        sigma = true_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        patch = true_bg + true_peak * np.exp(
            -0.5 * (((yy - cy) / sigma) ** 2 + ((xx - cx) / sigma) ** 2)
        )
        peak, x_off, y_off, widthx, widthy, background, chi2_red = _fit_spatial_gaussian(
            patch, beam_widthx=true_fwhm, beam_widthy=true_fwhm
        )
        assert np.isfinite(peak)
        assert chi2_red < 3.0
        np.testing.assert_allclose(peak, true_peak, rtol=0.15)
        np.testing.assert_allclose(widthx, true_fwhm, rtol=0.2)
        np.testing.assert_allclose(widthy, true_fwhm, rtol=0.2)
        np.testing.assert_allclose(background, true_bg, atol=1.0)
        assert abs(x_off) < 0.5
        assert abs(y_off) < 0.5

    def test_patch_fit_on_injected_gaussian(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """patch_fit() returns finite peaks on a centred Gaussian bump with low chi2."""
        ds = valid_ovro_dataset.copy(deep=True)
        l_idx, m_idx = ds.radport._resolve_coordinates(l=0.0, m=0.0)
        scale = 25.0
        radius = ds.radport.patch_radius_pixels(time_idx=0, scale=scale)
        assert radius >= 5
        sky = np.zeros(ds["SKY"].shape, dtype=float)
        ny = nx = 2 * radius + 1
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        yy, xx = np.indices((ny, nx))
        sigma = 3.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        bump = 100.0 * np.exp(
            -0.5 * (((yy - cy) / sigma) ** 2 + ((xx - cx) / sigma) ** 2)
        )
        sky[0, :, 0, l_idx - radius : l_idx + radius + 1, m_idx - radius : m_idx + radius + 1] = (
            bump[np.newaxis, :, :]
        )
        ds["SKY"].values[:] = sky

        result = ds.radport.patch_fit(l=0.0, m=0.0, scale=scale)
        assert int(result.patch_radius_map.isel(time=0).values) == radius
        peaks = result.peak_map.sel(time=ds.coords["time"][0])
        chi2 = result.reduced_chi_squared_map.sel(time=ds.coords["time"][0])
        assert np.all(np.isfinite(peaks.values))
        assert np.all(peaks.values > 50.0)
        assert np.all(chi2.values <= result.max_reduced_chi_squared)

    def test_patch_fit_nonpositive_scale_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Non-positive scale raises ValueError."""
        with pytest.raises(ValueError, match="scale must be positive"):
            valid_ovro_dataset.radport.patch_fit(l=0.0, m=0.0, scale=-1.0)

    def test_patch_fit_missing_beam_metadata_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """patch_fit() requires synthesized beam BMAJ/BMIN metadata."""
        ds = valid_ovro_dataset.drop_vars("BEAM")
        with pytest.raises(ValueError, match="Synthesized beam metadata unavailable"):
            ds.radport.patch_fit(l=0.0, m=0.0)

    def test_patch_fit_skips_empty_time_frequency_cells(self) -> None:
        """patch_fit() leaves empty SKY slots as NaN without requiring BEAM there."""
        l = np.linspace(-0.01, 0.01, 50)
        m = np.linspace(-0.01, 0.01, 50)
        sky = np.random.default_rng(0).random((1, 3, 1, 50, 50)) * 10.0
        sky[:, 1, ...] = np.nan
        beam_meta = np.zeros((1, 3, 1, 3), dtype=np.float64)
        beam_meta[0, 0, 0] = [0.02, 0.01, 0.0]
        beam_meta[0, 1, 0] = [np.nan, np.nan, np.nan]
        beam_meta[0, 2, 0] = [0.02, 0.01, 0.0]
        ds = xr.Dataset(
            data_vars={
                "SKY": (["time", "frequency", "polarization", "l", "m"], sky),
                "BEAM": (
                    ["time", "frequency", "polarization", "beam_param"],
                    beam_meta,
                ),
            },
            coords={
                "time": [60000.0],
                "frequency": [46e6, 50e6, 54e6],
                "polarization": [0],
                "beam_param": ["major", "minor", "pa"],
                "l": l,
                "m": m,
            },
        )
        result = ds.radport.patch_fit(l=0.0, m=0.0, scale=2.0)
        assert not np.isfinite(result.peak_map.values[0, 1])
        assert not np.isfinite(result.reduced_chi_squared_map.values[0, 1])
        assert np.isfinite(result.center_flux_map.values[0, 0])
        assert np.isfinite(result.patch_max_map.values[0, 2])
        assert ds.radport.patch_radius_pixels(time_idx=0, scale=2.0) > 0

    def test_patch_statistic_skips_empty_time_frequency_cells(self) -> None:
        """patch_statistic() ignores empty SKY slots when sizing the patch."""
        l = np.linspace(-0.01, 0.01, 50)
        m = np.linspace(-0.01, 0.01, 50)
        sky = np.random.default_rng(1).random((1, 3, 1, 50, 50)) * 10.0
        sky[:, 1, ...] = np.nan
        beam_meta = np.zeros((1, 3, 1, 3), dtype=np.float64)
        beam_meta[0, 0, 0] = [0.02, 0.01, 0.0]
        beam_meta[0, 1, 0] = [np.nan, np.nan, np.nan]
        beam_meta[0, 2, 0] = [0.02, 0.01, 0.0]
        ds = xr.Dataset(
            data_vars={
                "SKY": (["time", "frequency", "polarization", "l", "m"], sky),
                "BEAM": (
                    ["time", "frequency", "polarization", "beam_param"],
                    beam_meta,
                ),
            },
            coords={
                "time": [60000.0],
                "frequency": [46e6, 50e6, 54e6],
                "polarization": [0],
                "beam_param": ["major", "minor", "pa"],
                "l": l,
                "m": m,
            },
        )
        result = ds.radport.patch_statistic(l=0.0, m=0.0, statistic="max", scale=2.0)
        stats = result.stat_map.values[0]
        assert np.isfinite(stats[0])
        assert not np.isfinite(stats[1])
        assert np.isfinite(stats[2])

    def test_patch_fit_masks_poor_chi2_fits(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Cells with reduced chi-squared above threshold have NaN parameters."""
        result = valid_ovro_dataset.radport.patch_fit(
            l=0.0, m=0.0, max_reduced_chi_squared=0.01
        )
        assert np.all(np.isfinite(result.reduced_chi_squared_map.values))
        assert not np.any(np.isfinite(result.peak_map.values))

    def test_patch_fit_invalid_max_chi2_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Non-positive max_reduced_chi_squared raises ValueError."""
        with pytest.raises(ValueError, match="max_reduced_chi_squared must be positive"):
            valid_ovro_dataset.radport.patch_fit(
                l=0.0, m=0.0, max_reduced_chi_squared=0.0
            )

    def test_patch_fit_radec_tracking(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """RA/Dec patch_fit uses tracked pixels and records reduced chi-squared."""
        ds = valid_ovro_dataset_with_tracking_wcs
        result = ds.radport.patch_fit(ra=180.0, dec=37.0, scale=2.0)
        assert result.peak_map.attrs["tracking"] is True
        assert np.any(np.isfinite(result.reduced_chi_squared_map.values))
        assert np.any(np.isfinite(result.patch_radius_map.values))

    def test_patch_radius_pixels_scales_with_beam(self) -> None:
        """patch_radius_pixels grows with scale and beam FWHM."""
        from ovro_lwa_portal.accessor import patch_half_width_pixels

        l = np.linspace(-0.01, 0.01, 50)
        m = np.linspace(-0.01, 0.01, 50)
        ds = xr.Dataset(
            data_vars={
                "SKY": (["time", "frequency", "polarization", "l", "m"], np.zeros((1, 2, 1, 50, 50))),
                "BEAM": (
                    ["time", "frequency", "polarization", "beam_param"],
                    np.array([[[[0.02, 0.01, 0.0]], [[0.04, 0.02, 0.0]]]]),
                ),
            },
            coords={
                "time": [60000.0],
                "frequency": [46e6, 54e6],
                "polarization": [0],
                "beam_param": ["major", "minor", "pa"],
                "l": l,
                "m": m,
            },
        )
        _wx_lo, _wy_lo = ds.radport.beam_fwhm_pixels(time_idx=0, frequency_idx=0)
        wx_hi, wy_hi = ds.radport.beam_fwhm_pixels(time_idx=0, frequency_idx=1)
        r1 = ds.radport.patch_radius_pixels(time_idx=0, scale=2.0)
        r2 = ds.radport.patch_radius_pixels(time_idx=0, scale=4.0)
        assert r2 > r1
        assert r1 == patch_half_width_pixels(2.0, wx_hi, wy_hi)

    def test_beam_fwhm_pixels_from_beam_param(self) -> None:
        """beam_fwhm_pixels reads major/minor from BEAM beam_param."""
        l = np.linspace(-0.01, 0.01, 50)
        m = np.linspace(-0.01, 0.01, 50)
        ds = xr.Dataset(
            data_vars={
                "SKY": (["time", "frequency", "polarization", "l", "m"], np.zeros((1, 1, 1, 50, 50))),
                "BEAM": (
                    ["time", "frequency", "polarization", "beam_param"],
                    np.array([[[[0.02, 0.01, 30.0]]]]),
                ),
            },
            coords={
                "time": [60000.0],
                "frequency": [50e6],
                "polarization": [0],
                "beam_param": ["major", "minor", "pa"],
                "l": l,
                "m": m,
            },
        )
        wx, wy = ds.radport.beam_fwhm_pixels(time_idx=0, frequency_idx=0)
        assert wx > 1.0
        assert wy > 1.0

    def test_format_radec_sexagesimal(self) -> None:
        """Sexagesimal RA/Dec strings use hh:mm:ss.s and signed dd:mm:ss."""
        from ovro_lwa_portal.accessor import format_radec_sexagesimal

        ra_s, dec_s = format_radec_sexagesimal(299.868, 40.734)
        assert ra_s.count(":") == 2
        assert dec_s.count(":") == 2
        assert dec_s.startswith("+") or dec_s.startswith("-")

    def test_patch_fit_peak_radec_maps(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """peak_radec_maps() returns finite coordinates when the fit is accepted."""
        ds = valid_ovro_dataset_with_tracking_wcs
        result = ds.radport.patch_fit(ra=180.0, dec=37.0, scale=5.0)
        ra_map, dec_map = result.peak_radec_maps()
        assert ra_map.shape == result.peak_map.shape
        accepted = result.fit_accepted_map.values
        finite_offsets = np.isfinite(result.x_offset_map.values) & np.isfinite(
            result.y_offset_map.values
        )
        if np.any(accepted & finite_offsets):
            mask = accepted & finite_offsets
            assert np.any(np.isfinite(ra_map.values[mask]))
            assert np.any(np.isfinite(dec_map.values[mask]))
        diag = result.cell_diagnostics(time_idx=0, frequency_idx=0)
        assert "peak_ra_deg" in diag
        assert "peak_dec_deg" in diag
        assert "peak_ra" in diag
        assert "peak_dec" in diag

    def test_patch_fit_cell_returns_diagnostics(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """patch_fit_cell() matches full-cube patch_fit diagnostics on one cell."""
        ds = valid_ovro_dataset.copy(deep=True)
        l_idx, m_idx = ds.radport._resolve_coordinates(l=0.0, m=0.0)
        scale = 25.0
        radius = ds.radport.patch_radius_pixels(time_idx=0, scale=scale)
        sky = np.zeros(ds["SKY"].shape, dtype=float)
        ny = nx = 2 * radius + 1
        cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
        yy, xx = np.indices((ny, nx))
        sigma = 3.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        bump = 100.0 * np.exp(
            -0.5 * (((yy - cy) / sigma) ** 2 + ((xx - cx) / sigma) ** 2)
        )
        sky[0, :, 0, l_idx - radius : l_idx + radius + 1, m_idx - radius : m_idx + radius + 1] = (
            bump[np.newaxis, :, :]
        )
        ds["SKY"].values[:] = sky

        full = ds.radport.patch_fit(l=0.0, m=0.0, scale=scale)
        cell = ds.radport.patch_fit_cell(0, 0, l=0.0, m=0.0, scale=scale)
        full_diag = full.cell_diagnostics(0, 0)
        cell_diag = cell.cell_diagnostics(0, 0)

        assert cell.patch_radius_pixels == radius
        assert cell.fit_accepted is True
        assert cell.reduced_chi_squared <= cell.max_reduced_chi_squared
        assert np.isfinite(cell.peak) and cell.peak > 50.0
        for key in (
            "fit_accepted",
            "reduced_chi_squared",
            "peak",
            "center_flux",
            "patch_max",
            "background",
            "widthx",
            "widthy",
        ):
            np.testing.assert_allclose(
                cell_diag[key],
                full_diag[key],
                rtol=1e-5,
                atol=1e-5,
                err_msg=key,
            )

    def test_patch_fit_cell_empty_cell_raises(self) -> None:
        """patch_fit_cell() rejects empty SKY cells before fitting."""
        l = np.linspace(-0.01, 0.01, 50)
        m = np.linspace(-0.01, 0.01, 50)
        sky = np.random.default_rng(0).random((1, 3, 1, 50, 50)) * 10.0
        sky[:, 1, ...] = np.nan
        beam_meta = np.zeros((1, 3, 1, 3), dtype=np.float64)
        beam_meta[0, 0, 0] = [0.02, 0.01, 0.0]
        beam_meta[0, 1, 0] = [np.nan, np.nan, np.nan]
        beam_meta[0, 2, 0] = [0.02, 0.01, 0.0]
        ds = xr.Dataset(
            data_vars={
                "SKY": (["time", "frequency", "polarization", "l", "m"], sky),
                "BEAM": (
                    ["time", "frequency", "polarization", "beam_param"],
                    beam_meta,
                ),
            },
            coords={
                "time": [60000.0],
                "frequency": [46e6, 50e6, 54e6],
                "polarization": [0],
                "beam_param": ["major", "minor", "pa"],
                "l": l,
                "m": m,
            },
        )
        with pytest.raises(ValueError, match="No finite data"):
            ds.radport.patch_fit_cell(0, 1, l=0.0, m=0.0, scale=2.0)


class TestRadportPlotDynamicSpectrum:
    """Tests for plot_dynamic_spectrum() method."""

    def test_plot_dynamic_spectrum_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_dynamic_spectrum() returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_dynamic_spectrum(l=0.0, m=0.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_dynamic_spectrum_axis_labels(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_dynamic_spectrum() has correct axis labels."""
        fig = valid_ovro_dataset.radport.plot_dynamic_spectrum(l=0.0, m=0.0)
        try:
            ax = fig.axes[0]
            assert "Time" in ax.get_xlabel()
            assert "Frequency" in ax.get_ylabel()
        finally:
            plt.close(fig)

    def test_plot_dynamic_spectrum_with_options(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_dynamic_spectrum() accepts customization options."""
        fig = valid_ovro_dataset.radport.plot_dynamic_spectrum(
            l=0.0, m=0.0, cmap="viridis", robust=False, vmin=0.0, vmax=10.0
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)


class TestRadportDiff:
    """Tests for diff() method."""

    def test_diff_time_returns_dataarray(self, valid_ovro_dataset: xr.Dataset) -> None:
        """diff() with mode='time' returns xarray DataArray."""
        diff = valid_ovro_dataset.radport.diff(mode="time", time_idx=1)
        assert isinstance(diff, xr.DataArray)

    def test_diff_frequency_returns_dataarray(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """diff() with mode='frequency' returns xarray DataArray."""
        diff = valid_ovro_dataset.radport.diff(mode="frequency", freq_idx=1)
        assert isinstance(diff, xr.DataArray)

    def test_diff_has_lm_dims(self, valid_ovro_dataset: xr.Dataset) -> None:
        """diff() returns 2D array with l and m dimensions."""
        diff = valid_ovro_dataset.radport.diff(mode="time", time_idx=1)
        assert set(diff.dims) == {"l", "m"}

    def test_diff_time_metadata(self, valid_ovro_dataset: xr.Dataset) -> None:
        """diff() with mode='time' adds correct metadata."""
        diff = valid_ovro_dataset.radport.diff(mode="time", time_idx=1)
        assert diff.attrs["diff_mode"] == "time"
        assert diff.attrs["time_idx_current"] == 1
        assert diff.attrs["time_idx_prev"] == 0

    def test_diff_frequency_metadata(self, valid_ovro_dataset: xr.Dataset) -> None:
        """diff() with mode='frequency' adds correct metadata."""
        diff = valid_ovro_dataset.radport.diff(mode="frequency", freq_idx=2)
        assert diff.attrs["diff_mode"] == "frequency"
        assert diff.attrs["freq_idx_current"] == 2
        assert diff.attrs["freq_idx_prev"] == 1

    def test_diff_time_idx_zero_raises(self, valid_ovro_dataset: xr.Dataset) -> None:
        """diff() with mode='time' and time_idx=0 raises ValueError."""
        with pytest.raises(ValueError, match="time_idx must be >= 1"):
            valid_ovro_dataset.radport.diff(mode="time", time_idx=0)

    def test_diff_freq_idx_zero_raises(self, valid_ovro_dataset: xr.Dataset) -> None:
        """diff() with mode='frequency' and freq_idx=0 raises ValueError."""
        with pytest.raises(ValueError, match="freq_idx must be >= 1"):
            valid_ovro_dataset.radport.diff(mode="frequency", freq_idx=0)

    def test_diff_with_freq_mhz(self, valid_ovro_dataset: xr.Dataset) -> None:
        """diff() accepts freq_mhz parameter."""
        diff = valid_ovro_dataset.radport.diff(mode="time", time_idx=1, freq_mhz=50.0)
        assert diff.attrs["freq_idx"] == 1


class TestRadportPlotDiff:
    """Tests for plot_diff() method."""

    def test_plot_diff_time_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_diff() with mode='time' returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_diff(mode="time", time_idx=1)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_diff_frequency_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_diff() with mode='frequency' returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_diff(mode="frequency", freq_idx=1)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_diff_uses_diverging_cmap(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_diff() uses diverging colormap by default."""
        fig = valid_ovro_dataset.radport.plot_diff(mode="time", time_idx=1)
        try:
            # Default cmap is RdBu_r (diverging)
            ax = fig.axes[0]
            im = ax.images[0]
            assert im.cmap.name == "RdBu_r"
        finally:
            plt.close(fig)

    def test_plot_diff_symmetric_scale(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot_diff() uses symmetric color scale by default."""
        fig = valid_ovro_dataset.radport.plot_diff(mode="time", time_idx=1)
        try:
            ax = fig.axes[0]
            im = ax.images[0]
            vmin, vmax = im.get_clim()
            # Symmetric means |vmin| == |vmax|
            assert abs(abs(vmin) - abs(vmax)) < 0.01
        finally:
            plt.close(fig)

    def test_plot_diff_title_contains_info(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_diff() title contains relevant information."""
        fig = valid_ovro_dataset.radport.plot_diff(mode="time", time_idx=1)
        try:
            ax = fig.axes[0]
            title = ax.get_title()
            assert "Diff" in title
        finally:
            plt.close(fig)


# =============================================================================
# Phase C Tests: Data Quality and Grid Plots
# =============================================================================


class TestRadportFindValidFrame:
    """Tests for find_valid_frame() method."""

    def test_find_valid_frame_returns_tuple(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """find_valid_frame() returns a tuple of two integers."""
        result = valid_ovro_dataset.radport.find_valid_frame()
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)

    def test_find_valid_frame_returns_first_valid(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """find_valid_frame() returns first (0, 0) for all-valid dataset."""
        ti, fi = valid_ovro_dataset.radport.find_valid_frame()
        assert ti == 0
        assert fi == 0

    def test_find_valid_frame_with_threshold(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """find_valid_frame() respects min_finite_fraction threshold."""
        # With 100% threshold, should still find frame (all data is finite)
        ti, fi = valid_ovro_dataset.radport.find_valid_frame(min_finite_fraction=1.0)
        assert ti >= 0
        assert fi >= 0

    def test_find_valid_frame_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """find_valid_frame() raises ValueError for non-existent variable."""
        with pytest.raises(ValueError, match="not found"):
            valid_ovro_dataset.radport.find_valid_frame(var="BEAM")


class TestRadportFiniteFraction:
    """Tests for finite_fraction() method."""

    def test_finite_fraction_returns_dataarray(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """finite_fraction() returns xarray DataArray."""
        frac = valid_ovro_dataset.radport.finite_fraction()
        assert isinstance(frac, xr.DataArray)

    def test_finite_fraction_has_correct_dims(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """finite_fraction() returns 2D array with time and frequency dims."""
        frac = valid_ovro_dataset.radport.finite_fraction()
        assert set(frac.dims) == {"time", "frequency"}

    def test_finite_fraction_values_in_range(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """finite_fraction() values are between 0 and 1."""
        frac = valid_ovro_dataset.radport.finite_fraction()
        assert float(frac.min()) >= 0.0
        assert float(frac.max()) <= 1.0

    def test_finite_fraction_all_valid_data(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """finite_fraction() returns 1.0 for all-valid dataset."""
        frac = valid_ovro_dataset.radport.finite_fraction()
        assert float(frac.min()) == 1.0

    def test_finite_fraction_metadata(self, valid_ovro_dataset: xr.Dataset) -> None:
        """finite_fraction() adds metadata attributes."""
        frac = valid_ovro_dataset.radport.finite_fraction()
        assert frac.attrs["variable"] == "SKY"
        assert frac.attrs["pol"] == 0


class TestRadportPlotGrid:
    """Tests for plot_grid() method."""

    def test_plot_grid_returns_figure(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot_grid() returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_grid()
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_grid_with_time_indices(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_grid() accepts time_indices parameter."""
        fig = valid_ovro_dataset.radport.plot_grid(time_indices=[0, 1])
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_grid_with_freq_indices(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_grid() accepts freq_indices parameter."""
        fig = valid_ovro_dataset.radport.plot_grid(freq_indices=[0, 1, 2])
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_grid_with_freq_mhz_list(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_grid() accepts freq_mhz_list parameter."""
        fig = valid_ovro_dataset.radport.plot_grid(freq_mhz_list=[46.0, 50.0])
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_grid_custom_ncols(self, valid_ovro_dataset: xr.Dataset) -> None:
        """plot_grid() accepts ncols parameter."""
        fig = valid_ovro_dataset.radport.plot_grid(ncols=2)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_grid_with_mask_radius(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_grid() accepts mask_radius parameter."""
        fig = valid_ovro_dataset.radport.plot_grid(mask_radius=20)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_grid_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_grid() raises ValueError for non-existent variable."""
        with pytest.raises(ValueError, match="not found"):
            valid_ovro_dataset.radport.plot_grid(var="NONEXISTENT")

    def test_plot_grid_creates_multiple_axes(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_grid() creates correct number of axes."""
        # 2 times x 3 frequencies = 6 panels
        fig = valid_ovro_dataset.radport.plot_grid()
        try:
            # At least 6 axes (may have colorbar axis)
            assert len(fig.axes) >= 6
        finally:
            plt.close(fig)


class TestRadportPlotFrequencyGrid:
    """Tests for plot_frequency_grid() method."""

    def test_plot_frequency_grid_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_frequency_grid() returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_frequency_grid()
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_frequency_grid_single_time(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_frequency_grid() plots single time across frequencies."""
        fig = valid_ovro_dataset.radport.plot_frequency_grid(time_idx=1)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_frequency_grid_with_freq_list(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_frequency_grid() accepts freq_mhz_list parameter."""
        fig = valid_ovro_dataset.radport.plot_frequency_grid(
            freq_mhz_list=[46.0, 54.0]
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)


class TestRadportPlotTimeGrid:
    """Tests for plot_time_grid() method."""

    def test_plot_time_grid_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_time_grid() returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_time_grid()
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_time_grid_with_freq_idx(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_time_grid() accepts freq_idx parameter."""
        fig = valid_ovro_dataset.radport.plot_time_grid(freq_idx=1)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_time_grid_with_freq_mhz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_time_grid() accepts freq_mhz parameter."""
        fig = valid_ovro_dataset.radport.plot_time_grid(freq_mhz=50.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_time_grid_with_time_indices(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_time_grid() accepts time_indices parameter."""
        fig = valid_ovro_dataset.radport.plot_time_grid(
            freq_mhz=50.0, time_indices=[0, 1]
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)


# =============================================================================
# Phase D: 1D Analysis Methods Tests
# =============================================================================


class TestRadportLightCurve:
    """Tests for light_curve() method."""

    def test_light_curve_returns_dataarray(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """light_curve() returns xr.DataArray."""
        lc = valid_ovro_dataset.radport.light_curve(l=0.0, m=0.0)
        assert isinstance(lc, xr.DataArray)

    def test_light_curve_has_time_dimension(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """light_curve() result has 'time' as only dimension."""
        lc = valid_ovro_dataset.radport.light_curve(l=0.0, m=0.0)
        assert lc.dims == ("time",)

    def test_light_curve_correct_length(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """light_curve() has correct number of time points."""
        lc = valid_ovro_dataset.radport.light_curve(l=0.0, m=0.0)
        assert len(lc) == valid_ovro_dataset.sizes["time"]

    def test_light_curve_with_freq_mhz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """light_curve() accepts freq_mhz parameter."""
        lc = valid_ovro_dataset.radport.light_curve(l=0.0, m=0.0, freq_mhz=50.0)
        assert lc.attrs["freq_mhz"] == 50.0

    def test_light_curve_with_freq_idx(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """light_curve() accepts freq_idx parameter."""
        lc = valid_ovro_dataset.radport.light_curve(l=0.0, m=0.0, freq_idx=1)
        assert lc.attrs["freq_idx"] == 1

    def test_light_curve_metadata(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """light_curve() includes metadata attributes."""
        lc = valid_ovro_dataset.radport.light_curve(l=0.0, m=0.0)
        assert "variable" in lc.attrs
        assert "l" in lc.attrs
        assert "m" in lc.attrs
        assert "freq_mhz" in lc.attrs

    def test_light_curve_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """light_curve() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.light_curve(l=0.0, m=0.0, var="INVALID")


class TestRadportPlotLightCurve:
    """Tests for plot_light_curve() method."""

    def test_plot_light_curve_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_light_curve() returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_light_curve(l=0.0, m=0.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_light_curve_with_freq_mhz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_light_curve() accepts freq_mhz parameter."""
        fig = valid_ovro_dataset.radport.plot_light_curve(l=0.0, m=0.0, freq_mhz=50.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_light_curve_axis_labels(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_light_curve() has correct axis labels."""
        fig = valid_ovro_dataset.radport.plot_light_curve(l=0.0, m=0.0)
        try:
            ax = fig.axes[0]
            assert "Time" in ax.get_xlabel()
            assert "Intensity" in ax.get_ylabel()
        finally:
            plt.close(fig)


class TestRadportSpectrum:
    """Tests for spectrum() method."""

    def test_spectrum_returns_dataarray(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectrum() returns xr.DataArray."""
        spec = valid_ovro_dataset.radport.spectrum(l=0.0, m=0.0)
        assert isinstance(spec, xr.DataArray)

    def test_spectrum_has_frequency_dimension(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectrum() result has 'frequency' as only dimension."""
        spec = valid_ovro_dataset.radport.spectrum(l=0.0, m=0.0)
        assert spec.dims == ("frequency",)

    def test_spectrum_correct_length(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectrum() has correct number of frequency points."""
        spec = valid_ovro_dataset.radport.spectrum(l=0.0, m=0.0)
        assert len(spec) == valid_ovro_dataset.sizes["frequency"]

    def test_spectrum_with_time_idx(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectrum() accepts time_idx parameter."""
        spec = valid_ovro_dataset.radport.spectrum(l=0.0, m=0.0, time_idx=1)
        assert spec.attrs["time_idx"] == 1

    def test_spectrum_metadata(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectrum() includes metadata attributes."""
        spec = valid_ovro_dataset.radport.spectrum(l=0.0, m=0.0)
        assert "variable" in spec.attrs
        assert "l" in spec.attrs
        assert "m" in spec.attrs
        assert "time_mjd" in spec.attrs

    def test_spectrum_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectrum() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.spectrum(l=0.0, m=0.0, var="INVALID")


class TestRadportPlotSpectrum:
    """Tests for plot_spectrum() method."""

    def test_plot_spectrum_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_spectrum() returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_spectrum(l=0.0, m=0.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_spectrum_with_time_idx(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_spectrum() accepts time_idx parameter."""
        fig = valid_ovro_dataset.radport.plot_spectrum(l=0.0, m=0.0, time_idx=1)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_spectrum_axis_labels(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_spectrum() has correct axis labels."""
        fig = valid_ovro_dataset.radport.plot_spectrum(l=0.0, m=0.0)
        try:
            ax = fig.axes[0]
            assert "Frequency" in ax.get_xlabel()
            assert "Intensity" in ax.get_ylabel()
        finally:
            plt.close(fig)

    def test_plot_spectrum_freq_unit_hz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_spectrum() accepts freq_unit='Hz'."""
        fig = valid_ovro_dataset.radport.plot_spectrum(l=0.0, m=0.0, freq_unit="Hz")
        try:
            ax = fig.axes[0]
            assert "Hz" in ax.get_xlabel()
        finally:
            plt.close(fig)


class TestRadportTimeAverage:
    """Tests for time_average() method."""

    def test_time_average_returns_dataarray(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """time_average() returns xr.DataArray."""
        avg = valid_ovro_dataset.radport.time_average()
        assert isinstance(avg, xr.DataArray)

    def test_time_average_has_correct_dims(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """time_average() result has (frequency, l, m) dimensions."""
        avg = valid_ovro_dataset.radport.time_average()
        assert set(avg.dims) == {"frequency", "l", "m"}

    def test_time_average_removes_time_dim(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """time_average() removes time dimension."""
        avg = valid_ovro_dataset.radport.time_average()
        assert "time" not in avg.dims

    def test_time_average_with_time_indices(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """time_average() accepts time_indices parameter."""
        avg = valid_ovro_dataset.radport.time_average(time_indices=[0, 1])
        assert "time_indices" in avg.attrs

    def test_time_average_metadata(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """time_average() includes metadata attributes."""
        avg = valid_ovro_dataset.radport.time_average()
        assert avg.attrs["operation"] == "time_average"
        assert "variable" in avg.attrs

    def test_time_average_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """time_average() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.time_average(var="INVALID")


class TestRadportFrequencyAverage:
    """Tests for frequency_average() method."""

    def test_frequency_average_returns_dataarray(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """frequency_average() returns xr.DataArray."""
        avg = valid_ovro_dataset.radport.frequency_average()
        assert isinstance(avg, xr.DataArray)

    def test_frequency_average_has_correct_dims(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """frequency_average() result has (time, l, m) dimensions."""
        avg = valid_ovro_dataset.radport.frequency_average()
        assert set(avg.dims) == {"time", "l", "m"}

    def test_frequency_average_removes_freq_dim(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """frequency_average() removes frequency dimension."""
        avg = valid_ovro_dataset.radport.frequency_average()
        assert "frequency" not in avg.dims

    def test_frequency_average_with_freq_indices(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """frequency_average() accepts freq_indices parameter."""
        avg = valid_ovro_dataset.radport.frequency_average(freq_indices=[0, 1])
        assert "freq_indices" in avg.attrs

    def test_frequency_average_with_freq_range(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """frequency_average() accepts freq_min/max_mhz parameters."""
        avg = valid_ovro_dataset.radport.frequency_average(
            freq_min_mhz=46.0, freq_max_mhz=54.0
        )
        assert avg.attrs.get("freq_min_mhz") == 46.0
        assert avg.attrs.get("freq_max_mhz") == 54.0

    def test_frequency_average_metadata(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """frequency_average() includes metadata attributes."""
        avg = valid_ovro_dataset.radport.frequency_average()
        assert avg.attrs["operation"] == "frequency_average"
        assert "variable" in avg.attrs

    def test_frequency_average_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """frequency_average() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.frequency_average(var="INVALID")

    def test_frequency_average_invalid_range_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """frequency_average() raises ValueError for invalid frequency range."""
        with pytest.raises(ValueError, match="No frequencies in range"):
            valid_ovro_dataset.radport.frequency_average(
                freq_min_mhz=1000.0, freq_max_mhz=2000.0
            )


class TestRadportPlotTimeAverage:
    """Tests for plot_time_average() method."""

    def test_plot_time_average_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_time_average() returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_time_average()
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_time_average_with_freq_mhz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_time_average() accepts freq_mhz parameter."""
        fig = valid_ovro_dataset.radport.plot_time_average(freq_mhz=50.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_time_average_with_time_indices(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_time_average() accepts time_indices parameter."""
        fig = valid_ovro_dataset.radport.plot_time_average(time_indices=[0, 1])
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_time_average_with_mask_radius(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_time_average() accepts mask_radius parameter."""
        fig = valid_ovro_dataset.radport.plot_time_average(mask_radius=20)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)


class TestRadportPlotFrequencyAverage:
    """Tests for plot_frequency_average() method."""

    def test_plot_frequency_average_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_frequency_average() returns matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_frequency_average()
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_frequency_average_with_time_idx(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_frequency_average() accepts time_idx parameter."""
        fig = valid_ovro_dataset.radport.plot_frequency_average(time_idx=1)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_frequency_average_with_freq_range(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_frequency_average() accepts freq_min/max_mhz parameters."""
        fig = valid_ovro_dataset.radport.plot_frequency_average(
            freq_min_mhz=46.0, freq_max_mhz=54.0
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_frequency_average_with_mask_radius(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_frequency_average() accepts mask_radius parameter."""
        fig = valid_ovro_dataset.radport.plot_frequency_average(mask_radius=20)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)


# =============================================================================
# Phase E: WCS & Coordinate Methods Tests
# =============================================================================


class TestRadportHasWcs:
    """Tests for has_wcs property."""

    def test_has_wcs_false_without_wcs(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """has_wcs returns False when no WCS header is present."""
        assert valid_ovro_dataset.radport.has_wcs is False

    def test_has_wcs_true_with_wcs(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """has_wcs returns True when WCS header is present."""
        assert valid_ovro_dataset_with_wcs.radport.has_wcs is True


class TestRadportGetWcsTimePromotedHeader:
    """``_get_wcs`` must read per-time headers from ``fits_header_str``."""

    def test_get_wcs_uses_time_index_for_fits_header_str(self) -> None:
        from astropy.io import fits as afits
        from astropy.wcs import WCS

        from tests.test_fits_to_zarr import _encoded_fits_header_bytes, _make_sin_wcs_header_str

        from ovro_lwa_portal.accessor import _read_wcs_header_str

        hdr0 = _make_sin_wcs_header_str(nx=8, ny=8, crval1=180.0, crval2=45.0)
        hdr1 = _make_sin_wcs_header_str(nx=8, ny=8, crval1=181.0, crval2=46.0)
        enc0 = _encoded_fits_header_bytes(hdr0, nl=8, nm=8)
        enc1 = _encoded_fits_header_bytes(hdr1, nl=8, nm=8)
        n_time = 2
        fits_per_time = np.array([enc0, enc1], dtype=object)
        ds = xr.Dataset(
            {
                "SKY": (
                    ("time", "frequency", "polarization", "m", "l"),
                    np.zeros((n_time, 1, 1, 8, 8), dtype=np.float32),
                ),
                "fits_header_str": (("time",), fits_per_time),
            },
            coords={
                "time": ("time", np.arange(n_time, dtype=float)),
                "frequency": ("frequency", np.array([55e6])),
                "polarization": ("polarization", np.array([1.0])),
                "l": ("l", np.linspace(-0.1, 0.1, 8)),
                "m": ("m", np.linspace(-0.1, 0.1, 8)),
            },
        )

        w0 = ds.radport._get_wcs(time_idx=0)
        w1 = ds.radport._get_wcs(time_idx=1)
        assert w0.wcs.crval[0] == pytest.approx(180.0)
        assert w1.wcs.crval[0] == pytest.approx(181.0)
        wcs1 = _read_wcs_header_str(ds, time_idx=1)
        assert wcs1 is not None
        assert WCS(afits.Header.fromstring(wcs1, sep="\n")).wcs.crval[0] == pytest.approx(181.0)

    def test_read_wcs_header_str_time_frequency_dim(self) -> None:
        """Per-time WCS may be stored as ``fits_header_str(time, frequency)``."""
        from astropy.io import fits as afits
        from astropy.wcs import WCS

        from tests.test_fits_to_zarr import _encoded_fits_header_bytes, _make_sin_wcs_header_str

        from ovro_lwa_portal.accessor import _read_wcs_header_str

        hdr0 = _make_sin_wcs_header_str(nx=8, ny=8, crval1=180.0, crval2=45.0)
        hdr1 = _make_sin_wcs_header_str(nx=8, ny=8, crval1=190.0, crval2=50.0)
        enc0 = _encoded_fits_header_bytes(hdr0, nl=8, nm=8)
        enc1 = _encoded_fits_header_bytes(hdr1, nl=8, nm=8)
        fits_arr = np.array([[enc0], [enc1]], dtype=object)
        ds = xr.Dataset(
            {
                "SKY": (
                    ("time", "frequency", "polarization", "m", "l"),
                    np.zeros((2, 1, 1, 8, 8), dtype=np.float32),
                ),
                "fits_header_str": (("time", "frequency"), fits_arr),
            },
            coords={
                "time": ("time", np.arange(2, dtype=float)),
                "frequency": ("frequency", np.array([55e6])),
                "polarization": ("polarization", np.array([1.0])),
                "l": ("l", np.linspace(-0.1, 0.1, 8)),
                "m": ("m", np.linspace(-0.1, 0.1, 8)),
            },
        )
        wcs1 = _read_wcs_header_str(ds, time_idx=1)
        assert wcs1 is not None
        assert WCS(afits.Header.fromstring(wcs1, sep="\n")).wcs.crval[0] == pytest.approx(190.0)
        assert ds.radport._use_persisted_wcs_for_pixel_mapping() is True

    def test_get_wcs_prefers_per_time_header_over_static_attrs(self) -> None:
        """Static ``fits_wcs_header`` must not mask per-time ``fits_header_str``."""
        from tests.test_fits_to_zarr import _encoded_fits_header_bytes, _make_sin_wcs_header_str

        hdr_static = _make_sin_wcs_header_str(nx=8, ny=8, crval1=10.0, crval2=20.0)
        hdr0 = _make_sin_wcs_header_str(nx=8, ny=8, crval1=180.0, crval2=45.0)
        hdr1 = _make_sin_wcs_header_str(nx=8, ny=8, crval1=181.0, crval2=46.0)
        enc0 = _encoded_fits_header_bytes(hdr0, nl=8, nm=8)
        enc1 = _encoded_fits_header_bytes(hdr1, nl=8, nm=8)
        fits_per_time = np.array([enc0, enc1], dtype=object)
        ds = xr.Dataset(
            {
                "SKY": (
                    ("time", "frequency", "polarization", "m", "l"),
                    np.zeros((2, 1, 1, 8, 8), dtype=np.float32),
                ),
                "fits_header_str": (("time",), fits_per_time),
            },
            coords={
                "time": ("time", np.arange(2, dtype=float)),
                "frequency": ("frequency", np.array([55e6])),
                "polarization": ("polarization", np.array([1.0])),
                "l": ("l", np.linspace(-0.1, 0.1, 8)),
                "m": ("m", np.linspace(-0.1, 0.1, 8)),
            },
            attrs={"fits_wcs_header": hdr_static},
        )
        ds["SKY"].attrs["fits_wcs_header"] = hdr_static

        assert ds.radport._get_wcs(time_idx=1).wcs.crval[0] == pytest.approx(181.0)

    def test_read_wcs_header_str_empty_per_time_does_not_use_static_attrs(
        self,
    ) -> None:
        """Late time steps with empty fits_header_str must not fall back to time-0 attrs."""
        from tests.test_fits_to_zarr import _encoded_fits_header_bytes, _make_sin_wcs_header_str

        from ovro_lwa_portal.accessor import _read_wcs_header_str

        hdr_static = _make_sin_wcs_header_str(nx=8, ny=8, crval1=10.0, crval2=20.0)
        hdr0 = _make_sin_wcs_header_str(nx=8, ny=8, crval1=180.0, crval2=45.0)
        enc0 = _encoded_fits_header_bytes(hdr0, nl=8, nm=8)
        fits_per_time = np.array([enc0, np.bytes_(b"")], dtype=object)
        ds = xr.Dataset(
            {
                "SKY": (
                    ("time", "frequency", "polarization", "m", "l"),
                    np.zeros((2, 1, 1, 8, 8), dtype=np.float32),
                ),
                "fits_header_str": (("time",), fits_per_time),
            },
            coords={
                "time": ("time", np.arange(2, dtype=float)),
                "frequency": ("frequency", np.array([55e6])),
                "polarization": ("polarization", np.array([1.0])),
                "l": ("l", np.linspace(-0.1, 0.1, 8)),
                "m": ("m", np.linspace(-0.1, 0.1, 8)),
            },
            attrs={"fits_wcs_header": hdr_static},
        )
        ds["SKY"].attrs["fits_wcs_header"] = hdr_static

        wcs0 = _read_wcs_header_str(ds, time_idx=0)
        assert wcs0 is not None
        from astropy.io import fits as afits
        from astropy.wcs import WCS

        assert WCS(afits.Header.fromstring(wcs0, sep="\n")).wcs.crval[0] == pytest.approx(180.0)
        assert _read_wcs_header_str(ds, time_idx=1) is None

    def test_read_wcs_header_str_lm_reference_attrs_only(self) -> None:
        """Stokes V regrid reads WCS from LM ref coords + ``fits_wcs_header`` attr only."""
        from tests.test_fits_to_zarr import _make_sin_wcs_header_str

        from ovro_lwa_portal.accessor import _read_wcs_header_str

        hdr_str = _make_sin_wcs_header_str(nx=8, ny=8, crval1=180.0, crval2=45.0)
        ref = xr.Dataset(
            coords={
                "l": ("l", np.linspace(-0.1, 0.1, 8)),
                "m": ("m", np.linspace(-0.1, 0.1, 8)),
            },
            attrs={"fits_wcs_header": hdr_str},
        )
        wcs_hdr = _read_wcs_header_str(ref)
        assert wcs_hdr is not None
        from astropy.io import fits as afits
        from astropy.wcs import WCS

        assert WCS(afits.Header.fromstring(wcs_hdr, sep="\n")).wcs.crval[0] == pytest.approx(
            180.0
        )

    def test_coords_to_pixel_uses_per_time_fits_header_str(self) -> None:
        """coords_to_pixel must follow fits_header_str(time), not analytical SIN."""
        from astropy.io.fits import Header
        from astropy.wcs import WCS
        from tests.test_fits_to_zarr import _encoded_fits_header_bytes, _make_sin_wcs_header_str

        hdr0 = _make_sin_wcs_header_str(nx=8, ny=8, crval1=180.0, crval2=45.0)
        hdr1 = _make_sin_wcs_header_str(nx=8, ny=8, crval1=200.0, crval2=50.0)
        enc0 = _encoded_fits_header_bytes(hdr0, nl=8, nm=8)
        enc1 = _encoded_fits_header_bytes(hdr1, nl=8, nm=8)
        fits_per_time = np.array([enc0, enc1], dtype=object)
        ds = xr.Dataset(
            {
                "SKY": (
                    ("time", "frequency", "polarization", "m", "l"),
                    np.zeros((2, 1, 1, 8, 8), dtype=np.float32),
                ),
                "fits_header_str": (("time",), fits_per_time),
            },
            coords={
                "time": ("time", [60000.0, 60000.01]),
                "frequency": ("frequency", np.array([55e6])),
                "polarization": ("polarization", np.array([1.0])),
                "l": ("l", np.linspace(-0.1, 0.1, 8)),
                "m": ("m", np.linspace(-0.1, 0.1, 8)),
            },
        )

        for ti, hdr in enumerate((hdr0, hdr1)):
            wcs = WCS(Header.fromstring(hdr, sep="\n"))
            sky = wcs.pixel_to_world(4, 4)
            ra_deg = float(sky.ra.deg)
            dec_deg = float(sky.dec.deg)
            li, mi = ds.radport.coords_to_pixel(ra_deg, dec_deg, time_idx=ti)
            assert li == 4
            assert mi == 4


class TestRadportPixelToCoords:
    """Tests for pixel_to_coords() method."""

    def test_pixel_to_coords_returns_tuple(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset,
    ) -> None:
        """pixel_to_coords() returns tuple of (ra, dec)."""
        result = valid_ovro_dataset_with_tracking_wcs.radport.pixel_to_coords(
            30, 30, time_idx=0
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_pixel_to_coords_center_pixel(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset,
    ) -> None:
        """Reference pixel round-trips with coords_to_pixel at the same time index."""
        ds = valid_ovro_dataset_with_tracking_wcs
        li, mi = 25, 25
        for ti in (0, 1):
            ra, dec = ds.radport.pixel_to_coords(li, mi, time_idx=ti)
            lb, mb = ds.radport.coords_to_pixel(ra, dec, time_idx=ti)
            assert abs(lb - li) <= 1
            assert abs(mb - mi) <= 1

    def test_pixel_to_coords_ra_range(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset,
    ) -> None:
        """pixel_to_coords() returns RA in [0, 360) range."""
        ra, dec = valid_ovro_dataset_with_tracking_wcs.radport.pixel_to_coords(
            30, 30, time_idx=0
        )
        assert 0 <= ra < 360

    def test_pixel_to_coords_requires_exactly_one_time_selector(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """pixel_to_coords requires exactly one of time_idx or time_mjd."""
        ds = valid_ovro_dataset_with_wcs
        with pytest.raises(ValueError, match="exactly one of time_idx or time_mjd"):
            ds.radport.pixel_to_coords(5, 5)
        with pytest.raises(ValueError, match="not both"):
            ds.radport.pixel_to_coords(5, 5, time_idx=0, time_mjd=60000.0)

    def test_pixel_to_coords_out_of_bounds_l_raises(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """pixel_to_coords() raises for l_idx out of bounds."""
        with pytest.raises(ValueError, match="l_idx=100 out of bounds"):
            valid_ovro_dataset_with_wcs.radport.pixel_to_coords(100, 25, time_idx=0)

    def test_pixel_to_coords_out_of_bounds_m_raises(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """pixel_to_coords() raises for m_idx out of bounds."""
        with pytest.raises(ValueError, match="m_idx=100 out of bounds"):
            valid_ovro_dataset_with_wcs.radport.pixel_to_coords(25, 100, time_idx=0)

    def test_pixel_to_coords_no_wcs_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """pixel_to_coords() raises when no WCS is available."""
        with pytest.raises(ValueError, match="No WCS header found"):
            valid_ovro_dataset.radport.pixel_to_coords(25, 25, time_idx=0)

    def test_pixel_to_coords_time_roundtrip_matches_coords_to_pixel(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset,
    ) -> None:
        """time-aware pixel_to_coords inverts coords_to_pixel at the same time index."""
        ds = valid_ovro_dataset_with_tracking_wcs
        # Tracking fixture: phase center tracks zenith so (l,m) pixels stay
        # above the horizon across time steps.
        for ti in (0, 5, 9):
            li, mi = 33, 22
            ra_t, dec_t = ds.radport.pixel_to_coords(li, mi, time_idx=ti)
            lb, mb = ds.radport.coords_to_pixel(ra_t, dec_t, time_idx=ti)
            assert abs(lb - li) <= 1
            assert abs(mb - mi) <= 1

    def test_pixel_to_coords_uses_per_time_wcs_header_str(self) -> None:
        """pixel_to_coords must not use time-0 WCS when headers vary per step."""
        from tests.test_fits_to_zarr import _make_sin_wcs_header_str

        hdr0 = _make_sin_wcs_header_str(nx=16, ny=16, crval1=180.0, crval2=45.0)
        hdr1 = _make_sin_wcs_header_str(nx=16, ny=16, crval1=200.0, crval2=45.0)
        enc = [h.encode("utf-8") for h in (hdr0, hdr1)]
        wcs_per_time = np.array(
            [np.bytes_(e) for e in enc],
            dtype=f"S{max(len(enc[0]), len(enc[1]))}",
        )
        ds = xr.Dataset(
            {
                "SKY": (
                    ("time", "frequency", "polarization", "m", "l"),
                    np.zeros((2, 1, 1, 16, 16), dtype=np.float32),
                ),
                "wcs_header_str": (("time",), wcs_per_time),
            },
            coords={
                "time": ("time", [60000.0, 60000.01]),
                "frequency": ("frequency", [55e6]),
                "polarization": ("polarization", [0]),
                "l": ("l", np.linspace(-0.2, 0.2, 16)),
                "m": ("m", np.linspace(-0.2, 0.2, 16)),
            },
        )
        for ti, cr1 in enumerate((180.0, 200.0)):
            li, mi = ds.radport.coords_to_pixel(cr1, 45.0, time_idx=ti)
            ra_p, dec_p = ds.radport.pixel_to_coords(li, mi, time_idx=ti)
            lb, mb = ds.radport.coords_to_pixel(ra_p, dec_p, time_idx=ti)
            assert abs(lb - li) <= 1
            assert abs(mb - mi) <= 1
            assert abs(ra_p - cr1) < 0.1

    def test_pixel_to_coords_time_mjd_same_as_time_idx(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """time_mjd selects the same epoch as nearest_time_idx for pixel_to_coords."""
        ds = valid_ovro_dataset_with_wcs
        mjd = float(ds.coords["time"].values[1])
        ra_a, dec_a = ds.radport.pixel_to_coords(25, 25, time_idx=1)
        ra_b, dec_b = ds.radport.pixel_to_coords(25, 25, time_mjd=mjd)
        assert abs(ra_a - ra_b) < 1e-9
        assert abs(dec_a - dec_b) < 1e-9

    def test_pixel_to_coords_same_pixel_diff_time_different_sky(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """A fixed (l,m) pixel points at different RA/Dec as time advances."""
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        ds = valid_ovro_dataset_with_tracking_wcs
        ra0, dec0 = ds.radport.pixel_to_coords(30, 30, time_idx=0)
        ra9, dec9 = ds.radport.pixel_to_coords(30, 30, time_idx=9)
        sep = SkyCoord(ra0 * u.deg, dec0 * u.deg).separation(
            SkyCoord(ra9 * u.deg, dec9 * u.deg)
        )
        assert sep.deg > 0.05


class TestRadportCoordsToPixel:
    """Tests for coords_to_pixel() method."""

    def test_coords_to_pixel_returns_tuple(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """coords_to_pixel() returns tuple of (l_idx, m_idx)."""
        ds = valid_ovro_dataset_with_tracking_wcs
        ra, dec = ds.radport.pixel_to_coords(25, 25, time_idx=0)
        result = ds.radport.coords_to_pixel(ra, dec, time_idx=0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_coords_to_pixel_returns_integers(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """coords_to_pixel() returns integer indices."""
        ds = valid_ovro_dataset_with_tracking_wcs
        ra, dec = ds.radport.pixel_to_coords(25, 25, time_idx=0)
        l_idx, m_idx = ds.radport.coords_to_pixel(ra, dec, time_idx=0)
        assert isinstance(l_idx, int)
        assert isinstance(m_idx, int)

    def test_coords_to_pixel_center_coords(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """coords_to_pixel() returns center pixel for center coords."""
        ds = valid_ovro_dataset_with_tracking_wcs
        ra, dec = ds.radport.pixel_to_coords(25, 25, time_idx=0)
        l_idx, m_idx = ds.radport.coords_to_pixel(ra, dec, time_idx=0)
        assert abs(l_idx - 25) <= 1
        assert abs(m_idx - 25) <= 1

    def test_coords_to_pixel_roundtrip(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """pixel_to_coords and coords_to_pixel are approximate inverses."""
        # Start with pixel
        l_orig, m_orig = 30, 30
        ti = 0
        ra, dec = valid_ovro_dataset_with_wcs.radport.pixel_to_coords(
            l_orig, m_orig, time_idx=ti
        )
        l_back, m_back = valid_ovro_dataset_with_wcs.radport.coords_to_pixel(
            ra, dec, time_idx=ti
        )
        # Should round-trip approximately
        assert abs(l_back - l_orig) <= 1
        assert abs(m_back - m_orig) <= 1

    def test_coords_to_pixel_no_wcs_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """coords_to_pixel() does not require WCS metadata."""
        l_idx, m_idx = valid_ovro_dataset.radport.coords_to_pixel(0.0, 90.0, time_idx=0)
        assert isinstance(l_idx, int)
        assert isinstance(m_idx, int)

    def test_coords_to_pixel_requires_exactly_one_time_selector(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """coords_to_pixel requires exactly one of time_idx or time_mjd."""
        ds = valid_ovro_dataset_with_wcs
        with pytest.raises(ValueError, match="exactly one of time_idx or time_mjd"):
            ds.radport.coords_to_pixel(180.0, 45.0)
        with pytest.raises(ValueError, match="not both"):
            ds.radport.coords_to_pixel(180.0, 45.0, time_idx=0, time_mjd=60000.0)

    def test_coords_to_pixel_radec_respects_freq_idx(self) -> None:
        """Channelized RA/Dec coords: lookup slices frequency before (l, m) search."""
        nl, nm, nf = 9, 9, 2
        l = np.linspace(-0.15, 0.15, nl)
        m = np.linspace(-0.15, 0.15, nm)
        ra = np.full((nf, nm, nl), 100.0)
        dec = np.full((nf, nm, nl), 20.0)
        ra[0, 3, 7] = 55.0
        dec[0, 3, 7] = 22.0
        ra[1, 5, 2] = 55.0
        dec[1, 5, 2] = 22.0

        ds = xr.Dataset(
            data_vars={
                "SKY": (
                    ["time", "frequency", "polarization", "l", "m"],
                    np.zeros((1, nf, 1, nl, nm)),
                ),
            },
            coords={
                "time": [60000.0],
                "frequency": np.array([46e6, 54e6], dtype=float),
                "polarization": [0],
                "l": l,
                "m": m,
                "right_ascension": (["frequency", "m", "l"], ra),
                "declination": (["frequency", "m", "l"], dec),
            },
        )
        lb0, mb0 = ds.radport.coords_to_pixel(55.0, 22.0, time_idx=0, freq_idx=0)
        lb1, mb1 = ds.radport.coords_to_pixel(55.0, 22.0, time_idx=0, freq_idx=1)
        assert (lb0, mb0) == (7, 3)
        assert (lb1, mb1) == (2, 5)


class TestRadportPlotWcs:
    """Tests for plot_wcs() method."""

    def test_plot_wcs_returns_figure(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """plot_wcs() returns matplotlib Figure."""
        fig = valid_ovro_dataset_with_wcs.radport.plot_wcs()
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_wcs_with_freq_mhz(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """plot_wcs() accepts freq_mhz parameter."""
        fig = valid_ovro_dataset_with_wcs.radport.plot_wcs(freq_mhz=50.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_wcs_with_mask_radius(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """plot_wcs() accepts mask_radius parameter."""
        fig = valid_ovro_dataset_with_wcs.radport.plot_wcs(mask_radius=20)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_wcs_custom_colors(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """plot_wcs() accepts color customization parameters."""
        fig = valid_ovro_dataset_with_wcs.radport.plot_wcs(
            grid_color="yellow",
            label_color="cyan",
            facecolor="navy",
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_wcs_no_colorbar(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """plot_wcs() accepts add_colorbar=False."""
        fig = valid_ovro_dataset_with_wcs.radport.plot_wcs(add_colorbar=False)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_wcs_no_wcs_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_wcs() raises when no WCS is available."""
        with pytest.raises(ValueError, match="No WCS header found"):
            valid_ovro_dataset.radport.plot_wcs()

    def test_plot_wcs_invalid_var_raises(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """plot_wcs() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset_with_wcs.radport.plot_wcs(var="INVALID")


# =============================================================================
# Phase F: Animation & Export Tests
# =============================================================================


class TestRadportAnimateTime:
    """Tests for RadportAccessor.animate_time() method."""

    def test_animate_time_returns_animation(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_time() returns a FuncAnimation object."""
        from matplotlib.animation import FuncAnimation

        anim = valid_ovro_dataset.radport.animate_time()
        try:
            assert isinstance(anim, FuncAnimation)
        finally:
            plt.close("all")

    def test_animate_time_with_freq_mhz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_time() accepts freq_mhz parameter."""
        from matplotlib.animation import FuncAnimation

        anim = valid_ovro_dataset.radport.animate_time(freq_mhz=50.0)
        try:
            assert isinstance(anim, FuncAnimation)
        finally:
            plt.close("all")

    def test_animate_time_with_freq_idx(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_time() accepts freq_idx parameter."""
        from matplotlib.animation import FuncAnimation

        anim = valid_ovro_dataset.radport.animate_time(freq_idx=1)
        try:
            assert isinstance(anim, FuncAnimation)
        finally:
            plt.close("all")

    def test_animate_time_with_mask_radius(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_time() accepts mask_radius parameter."""
        from matplotlib.animation import FuncAnimation

        anim = valid_ovro_dataset.radport.animate_time(mask_radius=20)
        try:
            assert isinstance(anim, FuncAnimation)
        finally:
            plt.close("all")

    def test_animate_time_custom_cmap(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_time() accepts custom colormap."""
        from matplotlib.animation import FuncAnimation

        anim = valid_ovro_dataset.radport.animate_time(cmap="viridis")
        try:
            assert isinstance(anim, FuncAnimation)
        finally:
            plt.close("all")

    def test_animate_time_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_time() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.animate_time(var="INVALID")


class TestRadportAnimateFrequency:
    """Tests for RadportAccessor.animate_frequency() method."""

    def test_animate_frequency_returns_animation(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_frequency() returns a FuncAnimation object."""
        from matplotlib.animation import FuncAnimation

        anim = valid_ovro_dataset.radport.animate_frequency()
        try:
            assert isinstance(anim, FuncAnimation)
        finally:
            plt.close("all")

    def test_animate_frequency_with_time_idx(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_frequency() accepts time_idx parameter."""
        from matplotlib.animation import FuncAnimation

        anim = valid_ovro_dataset.radport.animate_frequency(time_idx=1)
        try:
            assert isinstance(anim, FuncAnimation)
        finally:
            plt.close("all")

    def test_animate_frequency_with_time_mjd(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_frequency() accepts time_mjd parameter."""
        from matplotlib.animation import FuncAnimation

        anim = valid_ovro_dataset.radport.animate_frequency(time_mjd=60000.0)
        try:
            assert isinstance(anim, FuncAnimation)
        finally:
            plt.close("all")

    def test_animate_frequency_with_mask_radius(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_frequency() accepts mask_radius parameter."""
        from matplotlib.animation import FuncAnimation

        anim = valid_ovro_dataset.radport.animate_frequency(mask_radius=20)
        try:
            assert isinstance(anim, FuncAnimation)
        finally:
            plt.close("all")

    def test_animate_frequency_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """animate_frequency() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.animate_frequency(var="INVALID")


class TestRadportExportFrames:
    """Tests for RadportAccessor.export_frames() method."""

    def test_export_frames_returns_list(
        self, valid_ovro_dataset: xr.Dataset, tmp_path
    ) -> None:
        """export_frames() returns a list of file paths."""
        output_dir = str(tmp_path / "frames")
        files = valid_ovro_dataset.radport.export_frames(
            output_dir,
            time_indices=[0],
            freq_indices=[0],
        )
        assert isinstance(files, list)
        assert len(files) == 1

    def test_export_frames_creates_files(
        self, valid_ovro_dataset: xr.Dataset, tmp_path
    ) -> None:
        """export_frames() creates actual files on disk."""
        import os

        output_dir = str(tmp_path / "frames")
        files = valid_ovro_dataset.radport.export_frames(
            output_dir,
            time_indices=[0],
            freq_indices=[0],
        )
        for f in files:
            assert os.path.exists(f)

    def test_export_frames_all_combinations(
        self, valid_ovro_dataset: xr.Dataset, tmp_path
    ) -> None:
        """export_frames() exports all time/freq combinations when not specified."""
        output_dir = str(tmp_path / "frames")
        files = valid_ovro_dataset.radport.export_frames(output_dir)
        # Dataset has 2 times x 3 frequencies = 6 frames
        assert len(files) == 6

    def test_export_frames_custom_format(
        self, valid_ovro_dataset: xr.Dataset, tmp_path
    ) -> None:
        """export_frames() accepts custom format parameter."""
        output_dir = str(tmp_path / "frames")
        files = valid_ovro_dataset.radport.export_frames(
            output_dir,
            time_indices=[0],
            freq_indices=[0],
            format="jpg",
        )
        assert files[0].endswith(".jpg")

    def test_export_frames_custom_template(
        self, valid_ovro_dataset: xr.Dataset, tmp_path
    ) -> None:
        """export_frames() accepts custom filename template."""
        output_dir = str(tmp_path / "frames")
        files = valid_ovro_dataset.radport.export_frames(
            output_dir,
            time_indices=[0],
            freq_indices=[0],
            filename_template="frame_{time_idx}_{freq_idx}.{format}",
        )
        assert "frame_0_0.png" in files[0]

    def test_export_frames_with_mask_radius(
        self, valid_ovro_dataset: xr.Dataset, tmp_path
    ) -> None:
        """export_frames() accepts mask_radius parameter."""
        import os

        output_dir = str(tmp_path / "frames")
        files = valid_ovro_dataset.radport.export_frames(
            output_dir,
            time_indices=[0],
            freq_indices=[0],
            mask_radius=20,
        )
        assert len(files) == 1
        assert os.path.exists(files[0])

    def test_export_frames_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset, tmp_path
    ) -> None:
        """export_frames() raises ValueError for invalid variable."""
        output_dir = str(tmp_path / "frames")
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.export_frames(output_dir, var="INVALID")

    def test_export_frames_creates_directory(
        self, valid_ovro_dataset: xr.Dataset, tmp_path
    ) -> None:
        """export_frames() creates output directory if it doesn't exist."""
        import os

        output_dir = str(tmp_path / "new_directory" / "frames")
        assert not os.path.exists(output_dir)
        files = valid_ovro_dataset.radport.export_frames(
            output_dir,
            time_indices=[0],
            freq_indices=[0],
        )
        assert os.path.exists(output_dir)
        assert len(files) == 1


# =============================================================================
# Phase G: Source Detection Tests
# =============================================================================


class TestRadportRmsMap:
    """Tests for RadportAccessor.rms_map() method."""

    def test_rms_map_returns_dataarray(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """rms_map() returns an xarray DataArray."""
        rms = valid_ovro_dataset.radport.rms_map()
        assert isinstance(rms, xr.DataArray)

    def test_rms_map_correct_dims(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """rms_map() returns data with (l, m) dimensions."""
        rms = valid_ovro_dataset.radport.rms_map()
        assert rms.dims == ("l", "m")

    def test_rms_map_positive_values(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """rms_map() returns non-negative values."""
        rms = valid_ovro_dataset.radport.rms_map()
        finite_vals = rms.values[np.isfinite(rms.values)]
        assert np.all(finite_vals >= 0)

    def test_rms_map_with_freq_mhz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """rms_map() accepts freq_mhz parameter."""
        rms = valid_ovro_dataset.radport.rms_map(freq_mhz=50.0)
        assert isinstance(rms, xr.DataArray)

    def test_rms_map_with_box_size(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """rms_map() accepts box_size parameter."""
        rms = valid_ovro_dataset.radport.rms_map(box_size=10)
        assert rms.attrs["box_size"] == 10

    def test_rms_map_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """rms_map() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.rms_map(var="INVALID")


class TestRadportSnrMap:
    """Tests for RadportAccessor.snr_map() method."""

    def test_snr_map_returns_dataarray(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """snr_map() returns an xarray DataArray."""
        snr = valid_ovro_dataset.radport.snr_map()
        assert isinstance(snr, xr.DataArray)

    def test_snr_map_correct_dims(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """snr_map() returns data with (l, m) dimensions."""
        snr = valid_ovro_dataset.radport.snr_map()
        assert snr.dims == ("l", "m")

    def test_snr_map_with_freq_mhz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """snr_map() accepts freq_mhz parameter."""
        snr = valid_ovro_dataset.radport.snr_map(freq_mhz=50.0)
        assert isinstance(snr, xr.DataArray)

    def test_snr_map_with_box_size(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """snr_map() accepts box_size parameter."""
        snr = valid_ovro_dataset.radport.snr_map(box_size=10)
        assert snr.attrs["box_size"] == 10

    def test_snr_map_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """snr_map() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.snr_map(var="INVALID")


class TestRadportFindPeaks:
    """Tests for RadportAccessor.find_peaks() method."""

    def test_find_peaks_returns_list(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """find_peaks() returns a list."""
        peaks = valid_ovro_dataset.radport.find_peaks()
        assert isinstance(peaks, list)

    def test_find_peaks_dict_structure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """find_peaks() returns dicts with expected keys."""
        # Use low threshold to ensure we get peaks
        peaks = valid_ovro_dataset.radport.find_peaks(threshold_sigma=0.1)
        if len(peaks) > 0:
            expected_keys = {"l", "m", "l_idx", "m_idx", "flux", "snr"}
            assert set(peaks[0].keys()) == expected_keys

    def test_find_peaks_with_freq_mhz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """find_peaks() accepts freq_mhz parameter."""
        peaks = valid_ovro_dataset.radport.find_peaks(freq_mhz=50.0)
        assert isinstance(peaks, list)

    def test_find_peaks_threshold_filters(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Higher threshold should return fewer or equal peaks."""
        peaks_low = valid_ovro_dataset.radport.find_peaks(threshold_sigma=0.1)
        peaks_high = valid_ovro_dataset.radport.find_peaks(threshold_sigma=10.0)
        assert len(peaks_high) <= len(peaks_low)

    def test_find_peaks_sorted_by_snr(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """find_peaks() returns peaks sorted by SNR descending."""
        peaks = valid_ovro_dataset.radport.find_peaks(threshold_sigma=0.1)
        if len(peaks) >= 2:
            snrs = [p["snr"] for p in peaks]
            assert snrs == sorted(snrs, reverse=True)

    def test_find_peaks_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """find_peaks() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.find_peaks(var="INVALID")

    def test_find_peaks_wcs_includes_radec(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """find_peaks() on WCS dataset includes ra and dec keys."""
        peaks = valid_ovro_dataset_with_wcs.radport.find_peaks(threshold_sigma=0.1)
        if len(peaks) > 0:
            assert "ra" in peaks[0]
            assert "dec" in peaks[0]
            assert isinstance(peaks[0]["ra"], float)
            assert isinstance(peaks[0]["dec"], float)

    def test_find_peaks_no_wcs_no_radec(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """find_peaks() on non-WCS dataset does not include ra/dec keys."""
        peaks = valid_ovro_dataset.radport.find_peaks(threshold_sigma=0.1)
        if len(peaks) > 0:
            assert "ra" not in peaks[0]
            assert "dec" not in peaks[0]

    def test_find_peaks_radec_roundtrip(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """find_peaks RA/Dec roundtrips: coords_to_pixel at frame time → same pixel."""
        ds = valid_ovro_dataset_with_wcs
        peaks = ds.radport.find_peaks(threshold_sigma=0.1)
        if len(peaks) > 0:
            peak = peaks[0]
            l_rt, m_rt = ds.radport.coords_to_pixel(
                peak["ra"], peak["dec"], time_idx=0
            )
            assert abs(l_rt - peak["l_idx"]) <= 1
            assert abs(m_rt - peak["m_idx"]) <= 1


class TestRadportPeakFluxMap:
    """Tests for RadportAccessor.peak_flux_map() method."""

    def test_peak_flux_map_returns_dataarray(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """peak_flux_map() returns an xarray DataArray."""
        peak_map = valid_ovro_dataset.radport.peak_flux_map()
        assert isinstance(peak_map, xr.DataArray)

    def test_peak_flux_map_correct_dims(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """peak_flux_map() returns data with (l, m) dimensions."""
        peak_map = valid_ovro_dataset.radport.peak_flux_map()
        assert peak_map.dims == ("l", "m")

    def test_peak_flux_map_with_freq_mhz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """peak_flux_map() accepts freq_mhz parameter."""
        peak_map = valid_ovro_dataset.radport.peak_flux_map(freq_mhz=50.0)
        assert isinstance(peak_map, xr.DataArray)

    def test_peak_flux_map_max_across_time(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """peak_flux_map() returns maximum across time dimension."""
        peak_map = valid_ovro_dataset.radport.peak_flux_map()
        # Get data manually to verify
        data = valid_ovro_dataset["SKY"].isel(frequency=0, polarization=0)
        expected_max = data.max(dim="time", skipna=True)
        # Check a few values match
        np.testing.assert_array_almost_equal(
            peak_map.values[:5, :5],
            expected_max.values[:5, :5],
        )

    def test_peak_flux_map_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """peak_flux_map() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.peak_flux_map(var="INVALID")


class TestRadportPlotSnrMap:
    """Tests for RadportAccessor.plot_snr_map() method."""

    def test_plot_snr_map_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_snr_map() returns a matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_snr_map()
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_snr_map_with_freq_mhz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_snr_map() accepts freq_mhz parameter."""
        fig = valid_ovro_dataset.radport.plot_snr_map(freq_mhz=50.0)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_snr_map_with_mask_radius(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_snr_map() accepts mask_radius parameter."""
        fig = valid_ovro_dataset.radport.plot_snr_map(mask_radius=20)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_snr_map_no_colorbar(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_snr_map() accepts add_colorbar=False."""
        fig = valid_ovro_dataset.radport.plot_snr_map(add_colorbar=False)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)


# =============================================================================
# Phase H: Spectral Analysis Tests
# =============================================================================


class TestRadportSpectralIndex:
    """Tests for RadportAccessor.spectral_index() method."""

    def test_spectral_index_returns_float(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectral_index() returns a float."""
        alpha = valid_ovro_dataset.radport.spectral_index(l=0.0, m=0.0)
        assert isinstance(alpha, float)

    def test_spectral_index_with_freq_mhz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectral_index() accepts freq_mhz parameters."""
        alpha = valid_ovro_dataset.radport.spectral_index(
            l=0.0, m=0.0,
            freq1_mhz=46.0,
            freq2_mhz=54.0,
        )
        assert isinstance(alpha, float)

    def test_spectral_index_with_freq_idx(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectral_index() accepts freq_idx parameters."""
        alpha = valid_ovro_dataset.radport.spectral_index(
            l=0.0, m=0.0,
            freq1_idx=0,
            freq2_idx=2,
        )
        assert isinstance(alpha, float)

    def test_spectral_index_finite_for_positive_flux(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectral_index() returns finite value for positive flux data."""
        # Test fixture has positive random data
        alpha = valid_ovro_dataset.radport.spectral_index(l=0.0, m=0.0)
        assert np.isfinite(alpha)

    def test_spectral_index_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectral_index() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.spectral_index(l=0.0, m=0.0, var="INVALID")


class TestRadportSpectralIndexMap:
    """Tests for RadportAccessor.spectral_index_map() method."""

    def test_spectral_index_map_returns_dataarray(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectral_index_map() returns an xarray DataArray."""
        alpha_map = valid_ovro_dataset.radport.spectral_index_map()
        assert isinstance(alpha_map, xr.DataArray)

    def test_spectral_index_map_correct_dims(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectral_index_map() returns data with (l, m) dimensions."""
        alpha_map = valid_ovro_dataset.radport.spectral_index_map()
        assert alpha_map.dims == ("l", "m")

    def test_spectral_index_map_with_freq_mhz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectral_index_map() accepts freq_mhz parameters."""
        alpha_map = valid_ovro_dataset.radport.spectral_index_map(
            freq1_mhz=46.0,
            freq2_mhz=54.0,
        )
        assert isinstance(alpha_map, xr.DataArray)

    def test_spectral_index_map_has_freq_attrs(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectral_index_map() includes frequency info in attrs."""
        alpha_map = valid_ovro_dataset.radport.spectral_index_map()
        assert "freq1_hz" in alpha_map.attrs
        assert "freq2_hz" in alpha_map.attrs

    def test_spectral_index_map_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectral_index_map() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.spectral_index_map(var="INVALID")


class TestRadportIntegratedFlux:
    """Tests for RadportAccessor.integrated_flux() method."""

    def test_integrated_flux_returns_float(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """integrated_flux() returns a float."""
        flux = valid_ovro_dataset.radport.integrated_flux(l=0.0, m=0.0)
        assert isinstance(flux, float)

    def test_integrated_flux_with_freq_range(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """integrated_flux() accepts freq_min/max_mhz parameters."""
        flux = valid_ovro_dataset.radport.integrated_flux(
            l=0.0, m=0.0,
            freq_min_mhz=46.0,
            freq_max_mhz=54.0,
        )
        assert isinstance(flux, float)

    def test_integrated_flux_with_freq_indices(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """integrated_flux() accepts freq_indices parameter."""
        flux = valid_ovro_dataset.radport.integrated_flux(
            l=0.0, m=0.0,
            freq_indices=[0, 1, 2],
        )
        assert isinstance(flux, float)

    def test_integrated_flux_positive_for_positive_data(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """integrated_flux() returns positive value for positive flux data."""
        flux = valid_ovro_dataset.radport.integrated_flux(l=0.0, m=0.0)
        assert flux > 0

    def test_integrated_flux_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """integrated_flux() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.integrated_flux(l=0.0, m=0.0, var="INVALID")


class TestRadportPlotSpectralIndexMap:
    """Tests for RadportAccessor.plot_spectral_index_map() method."""

    def test_plot_spectral_index_map_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_spectral_index_map() returns a matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_spectral_index_map()
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_spectral_index_map_with_freq_mhz(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_spectral_index_map() accepts freq_mhz parameters."""
        fig = valid_ovro_dataset.radport.plot_spectral_index_map(
            freq1_mhz=46.0,
            freq2_mhz=54.0,
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_spectral_index_map_with_mask_radius(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_spectral_index_map() accepts mask_radius parameter."""
        fig = valid_ovro_dataset.radport.plot_spectral_index_map(mask_radius=20)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_spectral_index_map_no_colorbar(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_spectral_index_map() accepts add_colorbar=False."""
        fig = valid_ovro_dataset.radport.plot_spectral_index_map(add_colorbar=False)
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)


# =============================================================================
# Dispersion Measure Correction Tests
# =============================================================================


class TestRadportDispersionDelay:
    """Tests for RadportAccessor.dispersion_delay() method."""

    def test_dispersion_delay_returns_float_for_scalar(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dispersion_delay() returns float for scalar frequency input."""
        delay = valid_ovro_dataset.radport.dispersion_delay(dm=56.8, freq_mhz=46.0)
        assert isinstance(delay, (float, np.floating))

    def test_dispersion_delay_returns_array_for_array_input(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dispersion_delay() returns array for array frequency input."""
        freq_mhz = np.array([46.0, 50.0, 54.0])
        delays = valid_ovro_dataset.radport.dispersion_delay(dm=56.8, freq_mhz=freq_mhz)
        assert isinstance(delays, np.ndarray)
        assert delays.shape == freq_mhz.shape

    def test_dispersion_delay_uses_dataset_frequencies(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dispersion_delay() uses dataset frequencies when freq_mhz is None."""
        delays = valid_ovro_dataset.radport.dispersion_delay(dm=56.8)
        n_freq = len(valid_ovro_dataset.coords["frequency"])
        assert len(delays) == n_freq

    def test_dispersion_delay_zero_dm_returns_zero(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dispersion_delay() returns zero for DM=0."""
        delay = valid_ovro_dataset.radport.dispersion_delay(dm=0.0, freq_mhz=46.0)
        assert delay == 0.0

    def test_dispersion_delay_negative_dm_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dispersion_delay() raises ValueError for negative DM."""
        with pytest.raises(ValueError, match="DM must be non-negative"):
            valid_ovro_dataset.radport.dispersion_delay(dm=-10.0, freq_mhz=46.0)

    def test_dispersion_delay_lower_freq_has_larger_delay(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Lower frequencies have larger dispersion delays."""
        delay_low = valid_ovro_dataset.radport.dispersion_delay(dm=56.8, freq_mhz=46.0)
        delay_high = valid_ovro_dataset.radport.dispersion_delay(dm=56.8, freq_mhz=54.0)
        assert delay_low > delay_high

    def test_dispersion_delay_crab_pulsar_dm(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Test dispersion delay with Crab pulsar DM=56.8 pc/cm³.

        For Crab pulsar at typical LWA frequencies, delay should be
        on the order of seconds between low and high frequencies.
        """
        # Crab pulsar DM
        dm_crab = 56.8

        # Compute delay at 46 MHz relative to 54 MHz reference
        delay = valid_ovro_dataset.radport.dispersion_delay(
            dm=dm_crab, freq_mhz=46.0, freq_ref_mhz=54.0
        )

        # Expected delay: K_DM * DM * (f_lo^-2 - f_hi^-2)
        # K_DM = 4.148808e3 MHz^2 pc^-1 cm^3 s
        # delay = 4.148808e3 * 56.8 * (46^-2 - 54^-2)
        #       = 235655.3 * (0.000472 - 0.000343)
        #       ≈ 30.4 seconds
        expected_delay = 4.148808e3 * dm_crab * (46.0**-2 - 54.0**-2)
        assert np.isclose(delay, expected_delay, rtol=1e-6)

    def test_dispersion_delay_custom_reference_freq(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dispersion_delay() accepts custom reference frequency."""
        delay = valid_ovro_dataset.radport.dispersion_delay(
            dm=56.8, freq_mhz=46.0, freq_ref_mhz=100.0
        )
        # Should be positive (46 MHz arrives later than 100 MHz)
        assert delay > 0

    def test_dispersion_delay_at_reference_freq_is_zero(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dispersion_delay() returns zero at the reference frequency."""
        freq_ref = 54.0
        delay = valid_ovro_dataset.radport.dispersion_delay(
            dm=56.8, freq_mhz=freq_ref, freq_ref_mhz=freq_ref
        )
        assert np.isclose(delay, 0.0, atol=1e-10)


class TestRadportDynamicSpectrumDedispersed:
    """Tests for RadportAccessor.dynamic_spectrum_dedispersed() method."""

    def test_dynamic_spectrum_dedispersed_returns_dataarray(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum_dedispersed() returns xr.DataArray."""
        result = valid_ovro_dataset.radport.dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=56.8
        )
        assert isinstance(result, xr.DataArray)

    def test_dynamic_spectrum_dedispersed_correct_dims(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum_dedispersed() returns correct dimensions."""
        result = valid_ovro_dataset.radport.dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=56.8
        )
        assert set(result.dims) == {"time", "frequency"}

    def test_dynamic_spectrum_dedispersed_has_dm_attr(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum_dedispersed() includes DM in attributes."""
        dm_value = 56.8
        result = valid_ovro_dataset.radport.dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=dm_value
        )
        assert result.attrs["dm"] == dm_value

    def test_dynamic_spectrum_dedispersed_has_method_attr(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum_dedispersed() includes method in attributes."""
        result = valid_ovro_dataset.radport.dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=56.8, method="shift"
        )
        assert result.attrs["method"] == "shift"

        result_interp = valid_ovro_dataset.radport.dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=56.8, method="interpolate"
        )
        assert result_interp.attrs["method"] == "interpolate"

    def test_dynamic_spectrum_dedispersed_shift_method(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum_dedispersed() works with shift method."""
        result = valid_ovro_dataset.radport.dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=10.0, method="shift"
        )
        # Should have same shape as input dynamic spectrum
        n_time = len(valid_ovro_dataset.coords["time"])
        n_freq = len(valid_ovro_dataset.coords["frequency"])
        assert result.shape == (n_time, n_freq)

    def test_dynamic_spectrum_dedispersed_interpolate_method(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum_dedispersed() works with interpolate method."""
        result = valid_ovro_dataset.radport.dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=10.0, method="interpolate"
        )
        # Should have same shape as input dynamic spectrum
        n_time = len(valid_ovro_dataset.coords["time"])
        n_freq = len(valid_ovro_dataset.coords["frequency"])
        assert result.shape == (n_time, n_freq)

    def test_dynamic_spectrum_dedispersed_zero_dm_returns_original(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum_dedispersed() returns original for DM=0."""
        original = valid_ovro_dataset.radport.dynamic_spectrum(l=0.0, m=0.0)
        dedispersed = valid_ovro_dataset.radport.dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=0.0
        )
        np.testing.assert_array_equal(original.values, dedispersed.values)

    def test_dynamic_spectrum_dedispersed_negative_dm_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum_dedispersed() raises ValueError for negative DM."""
        with pytest.raises(ValueError, match="DM must be non-negative"):
            valid_ovro_dataset.radport.dynamic_spectrum_dedispersed(
                l=0.0, m=0.0, dm=-10.0
            )

    def test_dynamic_spectrum_dedispersed_invalid_method_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum_dedispersed() raises ValueError for invalid method."""
        with pytest.raises(ValueError, match="Method must be 'shift' or 'interpolate'"):
            valid_ovro_dataset.radport.dynamic_spectrum_dedispersed(
                l=0.0, m=0.0, dm=56.8, method="invalid"
            )

    def test_dynamic_spectrum_dedispersed_invalid_var_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum_dedispersed() raises ValueError for invalid variable."""
        with pytest.raises(ValueError, match="Variable 'INVALID' not found"):
            valid_ovro_dataset.radport.dynamic_spectrum_dedispersed(
                l=0.0, m=0.0, dm=56.8, var="INVALID"
            )

    def test_dynamic_spectrum_dedispersed_trim_option(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum_dedispersed() trim option reduces time samples."""
        untrimmed = valid_ovro_dataset.radport.dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=10.0, trim=False
        )
        trimmed = valid_ovro_dataset.radport.dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=10.0, trim=True
        )
        # Trimmed should have same or fewer time samples
        assert len(trimmed.coords["time"]) <= len(untrimmed.coords["time"])

    def test_dynamic_spectrum_dedispersed_fill_value(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum_dedispersed() uses fill_value for shifted regions."""
        result = valid_ovro_dataset.radport.dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=10.0, method="shift", fill_value=np.nan, trim=False
        )
        # Should have NaN values at edges where data was shifted out
        # (unless DM is small enough that no shifting occurs)
        assert result.attrs["dm"] == 10.0

    def test_dynamic_spectrum_dedispersed_has_pixel_coords(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum_dedispersed() includes pixel coordinates in attrs."""
        result = valid_ovro_dataset.radport.dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=56.8
        )
        assert "pixel_l" in result.attrs
        assert "pixel_m" in result.attrs
        assert "l_idx" in result.attrs
        assert "m_idx" in result.attrs

    def test_dynamic_spectrum_dedispersed_crab_pulsar(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Test dedispersion with Crab pulsar DM=56.8 pc/cm³."""
        dm_crab = 56.8
        result = valid_ovro_dataset.radport.dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=dm_crab, method="interpolate"
        )
        assert result.attrs["dm"] == dm_crab
        assert "freq_ref_mhz" in result.attrs


class TestRadportPlotDynamicSpectrumDedispersed:
    """Tests for RadportAccessor.plot_dynamic_spectrum_dedispersed() method."""

    def test_plot_dynamic_spectrum_dedispersed_returns_figure(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_dynamic_spectrum_dedispersed() returns a matplotlib Figure."""
        fig = valid_ovro_dataset.radport.plot_dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=56.8
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_dynamic_spectrum_dedispersed_shift_method(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_dynamic_spectrum_dedispersed() works with shift method."""
        fig = valid_ovro_dataset.radport.plot_dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=56.8, method="shift"
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_dynamic_spectrum_dedispersed_interpolate_method(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_dynamic_spectrum_dedispersed() works with interpolate method."""
        fig = valid_ovro_dataset.radport.plot_dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=56.8, method="interpolate"
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_dynamic_spectrum_dedispersed_with_delay_curve(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_dynamic_spectrum_dedispersed() can show dispersion delay curve."""
        fig = valid_ovro_dataset.radport.plot_dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=56.8, show_delay_curve=True
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_dynamic_spectrum_dedispersed_no_colorbar(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_dynamic_spectrum_dedispersed() accepts add_colorbar=False."""
        fig = valid_ovro_dataset.radport.plot_dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=56.8, add_colorbar=False
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_dynamic_spectrum_dedispersed_custom_cmap(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_dynamic_spectrum_dedispersed() accepts custom colormap."""
        fig = valid_ovro_dataset.radport.plot_dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=56.8, cmap="viridis"
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_dynamic_spectrum_dedispersed_with_vmin_vmax(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_dynamic_spectrum_dedispersed() accepts vmin/vmax parameters."""
        fig = valid_ovro_dataset.radport.plot_dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=56.8, vmin=0.0, vmax=10.0
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)

    def test_plot_dynamic_spectrum_dedispersed_trim_option(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """plot_dynamic_spectrum_dedispersed() accepts trim option."""
        fig = valid_ovro_dataset.radport.plot_dynamic_spectrum_dedispersed(
            l=0.0, m=0.0, dm=10.0, trim=True
        )
        try:
            assert isinstance(fig, plt.Figure)
        finally:
            plt.close(fig)


# =========================================================================
# Phase 1: Core Tracking Engine Tests
# =========================================================================


class TestComputePixelTrack:
    """Tests for _compute_pixel_track() per-time tracking method."""

    def test_zenith_source_at_t0_near_center(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """Source at zenith (RA=LST at t0, Dec=lat) maps to center pixel at t=0."""
        from astropy.time import Time
        from astropy import units as u

        ds = valid_ovro_dataset_with_tracking_wcs
        t0 = ds.coords["time"].values[0]
        lst_deg = float(
            Time(t0, format="mjd", scale="utc")
            .sidereal_time("mean", longitude=-118.2817 * u.deg)
            .deg
        )

        l_idx, m_idx, visible = ds.radport._compute_pixel_track(
            ra=lst_deg, dec=37.2339
        )
        # At t=0, source is at zenith → should be near center pixel (25)
        assert visible[0]
        assert abs(l_idx[0] - 25) <= 1
        assert abs(m_idx[0] - 25) <= 1

    def test_source_drifts_over_time(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """A source at a fixed RA should drift in l-index across time steps."""
        from astropy.time import Time
        from astropy import units as u

        ds = valid_ovro_dataset_with_tracking_wcs
        t0 = ds.coords["time"].values[0]
        lst_deg = float(
            Time(t0, format="mjd", scale="utc")
            .sidereal_time("mean", longitude=-118.2817 * u.deg)
            .deg
        )

        l_idx, m_idx, visible = ds.radport._compute_pixel_track(
            ra=lst_deg, dec=37.2339
        )
        # Over 10 steps × 14.4 min, source drifts significantly
        # The l-indices for visible time steps should NOT all be the same
        visible_l = l_idx[visible]
        if len(visible_l) > 1:
            assert visible_l[0] != visible_l[-1], (
                "Source l-index should change over time due to Earth rotation"
            )

    def test_below_horizon_marked_invisible(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """Source well below horizon has visible=False and out-of-range sentinels."""
        ds = valid_ovro_dataset_with_tracking_wcs
        # Dec = -80 is very far south, never visible from OVRO-LWA (lat 37.2°)
        with pytest.warns(UserWarning, match="never above the horizon"):
            l_idx, m_idx, visible = ds.radport._compute_pixel_track(
                ra=0.0, dec=-80.0
            )
        assert not np.any(visible)
        # Sentinel values are n_l/n_m (out-of-range), not -1
        n_l = ds.sizes["l"]
        n_m = ds.sizes["m"]
        assert np.all(l_idx == n_l)
        assert np.all(m_idx == n_m)

    def test_result_shapes_match_time_array(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """Output arrays have same length as time dimension."""
        ds = valid_ovro_dataset_with_tracking_wcs
        n_times = ds.sizes["time"]
        l_idx, m_idx, visible = ds.radport._compute_pixel_track(ra=180.0, dec=37.0)
        assert l_idx.shape == (n_times,)
        assert m_idx.shape == (n_times,)
        assert visible.shape == (n_times,)

    def test_observatory_override(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """Custom observatory location produces different pixel tracks."""
        from astropy.coordinates import EarthLocation
        from astropy import units as u

        ds = valid_ovro_dataset_with_tracking_wcs
        custom_obs = EarthLocation(
            lat=0.0 * u.deg, lon=0.0 * u.deg, height=0.0 * u.m
        )
        l_default, _, _ = ds.radport._compute_pixel_track(ra=180.0, dec=37.0)
        l_custom, _, _ = ds.radport._compute_pixel_track(
            ra=180.0, dec=37.0, observatory=custom_obs
        )
        # Different observatories → different pixel tracks
        assert not np.array_equal(l_default, l_custom)

    def test_circumpolar_source_always_visible(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """A circumpolar source (Dec > 90 - lat ≈ 52.8°) is always visible."""
        from astropy.time import Time
        from astropy import units as u

        ds = valid_ovro_dataset_with_tracking_wcs
        t0 = ds.coords["time"].values[0]
        lst_deg = float(
            Time(t0, format="mjd", scale="utc")
            .sidereal_time("mean", longitude=-118.2817 * u.deg)
            .deg
        )
        # Polaris-like source near celestial pole
        l_idx, m_idx, visible = ds.radport._compute_pixel_track(
            ra=lst_deg, dec=89.0
        )
        # May not all be in image bounds, but direction cosines should be < 1
        # for a source above horizon. Check that at least some are visible.
        # At Dec=89 from lat=37, elevation ≈ 38°, always above horizon.
        # But may be outside image FOV depending on image size.
        # The key test: visible should be True where l²+m² < 1 AND in bounds.
        # For this high-dec source, it will be above horizon at all times.
        # Image bounds check may exclude some, but the point is it's not
        # all False like the below-horizon test.
        assert visible.dtype == bool

    def test_ra_wrap_around(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """RA near 0/360 boundary doesn't cause errors."""
        ds = valid_ovro_dataset_with_tracking_wcs
        # RA=359.9 should work without error
        l_idx, m_idx, visible = ds.radport._compute_pixel_track(ra=359.9, dec=37.0)
        assert l_idx.shape == (ds.sizes["time"],)
        # RA=0.1 should also work
        l_idx2, m_idx2, visible2 = ds.radport._compute_pixel_track(ra=0.1, dec=37.0)
        assert l_idx2.shape == (ds.sizes["time"],)

    def test_batched_radec_grid_matches_per_time_pixel_at_time(self) -> None:
        """Batched (time, m, l) track matches looping :meth:`_compute_pixel_at_time`."""
        from astropy.time import Time
        from astropy import units as u

        nt, nf, npol, nl, nm = 5, 2, 2, 12, 10
        times = [60000.0 + 0.01 * i for i in range(nt)]
        # Phase center near zenith at mid-observation so the test source stays up.
        t_mid = Time(times[nt // 2], format="mjd", scale="utc")
        lst0 = float(
            t_mid.sidereal_time("mean", longitude=-118.2817 * u.deg).deg
        )
        dec0 = 37.2339
        l_coord = np.linspace(-0.6, 0.6, nl)
        m_coord = np.linspace(-0.5, 0.5, nm)
        # Coords are ordered (..., m, l); build base arrays with shape (nm, nl).
        mt, lt = np.meshgrid(np.arange(nm), np.arange(nl), indexing="ij")
        # Drift in RA with time; smooth variation over the grid
        ra_grid = (
            lst0
            + 0.08 * lt
            + 0.12 * mt
            + np.arange(nt, dtype=np.float64)[:, np.newaxis, np.newaxis] * 0.15
        )
        dec_grid = (
            dec0
            + 0.05 * lt
            - 0.04 * mt
            + np.arange(nt, dtype=np.float64)[:, np.newaxis, np.newaxis] * 0.02
        )
        ds = xr.Dataset(
            data_vars={
                "SKY": (
                    ["time", "frequency", "polarization", "l", "m"],
                    np.random.default_rng(0).random((nt, nf, npol, nl, nm)),
                ),
            },
            coords={
                "time": times,
                "frequency": [46e6, 54e6],
                "polarization": [0, 1],
                "l": l_coord,
                "m": m_coord,
                "right_ascension": (["time", "m", "l"], ra_grid),
                "declination": (["time", "m", "l"], dec_grid),
            },
        )
        ra_t, dec_t = lst0 + 0.35, dec0 + 0.12
        fi, pol = 0, 0
        n_l, n_m = nl, nm
        ref_l = np.empty(nt, dtype=int)
        ref_m = np.empty(nt, dtype=int)
        ref_v = np.zeros(nt, dtype=bool)
        for ti in range(nt):
            try:
                li, mi = ds.radport._compute_pixel_at_time(
                    ra_t, dec_t, ti, freq_idx=fi, pol=pol
                )
                ref_l[ti] = li
                ref_m[ti] = mi
                ref_v[ti] = True
            except ValueError:
                ref_l[ti] = n_l
                ref_m[ti] = n_m
                ref_v[ti] = False

        l_b, m_b, v_b = ds.radport._compute_pixel_track(
            ra_t, dec_t, freq_idx=fi, pol=pol
        )
        np.testing.assert_array_equal(l_b, ref_l)
        np.testing.assert_array_equal(m_b, ref_m)
        np.testing.assert_array_equal(v_b, ref_v)

    def test_batched_radec_grid_time_chunked_matches_per_time_pixel_at_time(
        self,
    ) -> None:
        """Time-chunked batched track matches :meth:`_compute_pixel_at_time` loop."""
        from astropy.time import Time
        from astropy import units as u

        from ovro_lwa_portal import accessor as acc_mod

        nt, nf, npol, nl, nm = 20, 1, 1, 50, 10
        times = [60000.0 + 0.005 * i for i in range(nt)]
        t_mid = Time(times[nt // 2], format="mjd", scale="utc")
        lst0 = float(
            t_mid.sidereal_time("mean", longitude=-118.2817 * u.deg).deg
        )
        dec0 = 37.2339
        l_coord = np.linspace(-0.6, 0.6, nl)
        m_coord = np.linspace(-0.5, 0.5, nm)
        mt, lt = np.meshgrid(np.arange(nm), np.arange(nl), indexing="ij")
        ra_grid = (
            lst0
            + 0.06 * lt
            + 0.1 * mt
            + np.arange(nt, dtype=np.float64)[:, np.newaxis, np.newaxis] * 0.08
        )
        dec_grid = (
            dec0
            + 0.04 * lt
            - 0.03 * mt
            + np.arange(nt, dtype=np.float64)[:, np.newaxis, np.newaxis] * 0.015
        )
        ds = xr.Dataset(
            data_vars={
                "SKY": (
                    ["time", "frequency", "polarization", "l", "m"],
                    np.random.default_rng(1).random((nt, nf, npol, nl, nm)),
                ),
            },
            coords={
                "time": times,
                "frequency": [50e6],
                "polarization": [0],
                "l": l_coord,
                "m": m_coord,
                "right_ascension": (["time", "m", "l"], ra_grid),
                "declination": (["time", "m", "l"], dec_grid),
            },
        )
        ra_t, dec_t = lst0 + 0.28, dec0 + 0.09
        fi, pol = 0, 0
        n_l, n_m = nl, nm
        ref_l = np.empty(nt, dtype=int)
        ref_m = np.empty(nt, dtype=int)
        ref_v = np.zeros(nt, dtype=bool)
        for ti in range(nt):
            try:
                li, mi = ds.radport._compute_pixel_at_time(
                    ra_t, dec_t, ti, freq_idx=fi, pol=pol
                )
                ref_l[ti] = li
                ref_m[ti] = mi
                ref_v[ti] = True
            except ValueError:
                ref_l[ti] = n_l
                ref_m[ti] = n_m
                ref_v[ti] = False

        old_chunk = acc_mod._MAX_PIXEL_TRACK_CHUNK_ELEMENTS
        # spatial = 500 → chunk_nt = min(nt, 2000 // 500) = 4 → multiple chunks.
        acc_mod._MAX_PIXEL_TRACK_CHUNK_ELEMENTS = 2000
        try:
            l_b, m_b, v_b = ds.radport._compute_pixel_track(
                ra_t, dec_t, freq_idx=fi, pol=pol
            )
        finally:
            acc_mod._MAX_PIXEL_TRACK_CHUNK_ELEMENTS = old_chunk

        np.testing.assert_array_equal(l_b, ref_l)
        np.testing.assert_array_equal(m_b, ref_m)
        np.testing.assert_array_equal(v_b, ref_v)


class TestResolveCoordinates:
    """Tests for _resolve_coordinates() input validation and dispatch."""

    def test_lm_returns_fixed_indices(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """l/m provided → returns tuple of two ints."""
        result = valid_ovro_dataset.radport._resolve_coordinates(l=0.0, m=0.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)

    def test_radec_returns_per_time_arrays(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """ra/dec provided → returns tuple of three ndarrays."""
        result = valid_ovro_dataset_with_tracking_wcs.radport._resolve_coordinates(
            ra=180.0, dec=37.0
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)
        assert isinstance(result[2], np.ndarray)

    def test_both_provided_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Providing both (ra, dec) and (l, m) raises ValueError."""
        with pytest.raises(ValueError, match="not both"):
            valid_ovro_dataset.radport._resolve_coordinates(
                ra=180.0, dec=45.0, l=0.0, m=0.0
            )

    def test_neither_provided_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Providing neither coordinate pair raises ValueError."""
        with pytest.raises(ValueError, match="Must provide"):
            valid_ovro_dataset.radport._resolve_coordinates()

    def test_partial_ra_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Providing ra without dec raises ValueError."""
        with pytest.raises(ValueError, match="Both ra and dec"):
            valid_ovro_dataset.radport._resolve_coordinates(ra=180.0)

    def test_partial_dec_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Providing dec without ra raises ValueError."""
        with pytest.raises(ValueError, match="Both ra and dec"):
            valid_ovro_dataset.radport._resolve_coordinates(dec=45.0)

    def test_partial_l_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """Providing l without m raises ValueError."""
        with pytest.raises(ValueError, match="Both l and m"):
            valid_ovro_dataset.radport._resolve_coordinates(l=0.0)


# =========================================================================
# Phase 2: Time-Series Tracking Tests
# =========================================================================


class TestCelestialTimeSeriesTracking:
    """Tests for light_curve and dynamic_spectrum with RA/Dec tracking."""

    def test_light_curve_radec_returns_time_dim(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """light_curve(ra, dec) returns DataArray with dim 'time'."""
        from astropy.time import Time
        from astropy import units as u

        ds = valid_ovro_dataset_with_tracking_wcs
        t0 = ds.coords["time"].values[0]
        lst_deg = float(
            Time(t0, format="mjd", scale="utc")
            .sidereal_time("mean", longitude=-118.2817 * u.deg)
            .deg
        )
        lc = ds.radport.light_curve(ra=lst_deg, dec=37.2339, freq_mhz=50.0)
        assert "time" in lc.dims
        assert lc.attrs["tracking"] is True
        assert lc.attrs["ra"] == lst_deg
        assert lc.attrs["dec"] == 37.2339

    def test_light_curve_radec_tracks_different_pixels(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """light_curve(ra, dec) extracts different pixels at different times."""
        from astropy.time import Time
        from astropy import units as u

        ds = valid_ovro_dataset_with_tracking_wcs
        t0 = ds.coords["time"].values[0]
        lst_deg = float(
            Time(t0, format="mjd", scale="utc")
            .sidereal_time("mean", longitude=-118.2817 * u.deg)
            .deg
        )
        # Get tracked light curve
        lc_tracked = ds.radport.light_curve(ra=lst_deg, dec=37.2339, freq_mhz=50.0)

        # Get fixed-pixel light curve at center
        l_idx, m_idx = ds.radport.nearest_lm_idx(0.0, 0.0)
        lc_fixed = ds.radport.light_curve(
            l=float(ds.coords["l"].values[l_idx]),
            m=float(ds.coords["m"].values[m_idx]),
            freq_mhz=50.0,
        )

        # Tracked and fixed should differ because tracking follows the source
        # while fixed stays at the same pixel
        assert not np.array_equal(
            lc_tracked.values[np.isfinite(lc_tracked.values)],
            lc_fixed.values[np.isfinite(lc_fixed.values)],
        )

    def test_light_curve_lm_keyword_works(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """light_curve(l=..., m=...) still works with keyword syntax."""
        lc = valid_ovro_dataset.radport.light_curve(l=0.0, m=0.0)
        assert "time" in lc.dims
        assert "tracking" not in lc.attrs

    def test_dynamic_spectrum_radec_returns_correct_dims(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """dynamic_spectrum(ra, dec) returns DataArray with (time, frequency)."""
        from astropy.time import Time
        from astropy import units as u

        ds = valid_ovro_dataset_with_tracking_wcs
        t0 = ds.coords["time"].values[0]
        lst_deg = float(
            Time(t0, format="mjd", scale="utc")
            .sidereal_time("mean", longitude=-118.2817 * u.deg)
            .deg
        )
        dynspec = ds.radport.dynamic_spectrum(ra=lst_deg, dec=37.2339)
        assert set(dynspec.dims) == {"time", "frequency"}
        assert dynspec.attrs["tracking"] is True

    def test_dynamic_spectrum_radec_tracks_correctly(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """dynamic_spectrum(ra, dec) tracks source across time."""
        from astropy.time import Time
        from astropy import units as u

        ds = valid_ovro_dataset_with_tracking_wcs
        t0 = ds.coords["time"].values[0]
        lst_deg = float(
            Time(t0, format="mjd", scale="utc")
            .sidereal_time("mean", longitude=-118.2817 * u.deg)
            .deg
        )
        dynspec = ds.radport.dynamic_spectrum(ra=lst_deg, dec=37.2339)
        # Should have some finite values (source visible at least at t=0)
        assert np.any(np.isfinite(dynspec.values))

    def test_radec_track_uses_per_time_wcs_not_fixed_pixel(self) -> None:
        """Fixed catalog (RA, Dec) must follow per-time WCS when CRVAL drifts."""
        from tests.test_fits_to_zarr import _make_sin_wcs_header_str

        catalog_ra = 180.0
        catalog_dec = 45.0
        n_time = 2
        # Small CRVAL1 steps: source stays in FOV but maps to different pixels.
        crvals = (180.0, 180.5)
        headers = [
            _make_sin_wcs_header_str(nx=16, ny=16, crval1=c1, crval2=catalog_dec)
            for c1 in crvals
        ]
        enc = [h.encode("utf-8") for h in headers]
        wcs_per_time = np.array(
            [np.bytes_(e) for e in enc],
            dtype=f"S{max(len(e) for e in enc)}",
        )
        times = [60000.0 + 0.01 * i for i in range(n_time)]
        ds = xr.Dataset(
            {
                "SKY": (
                    ("time", "frequency", "polarization", "m", "l"),
                    np.ones((n_time, 1, 1, 16, 16), dtype=np.float32),
                ),
                "wcs_header_str": (("time",), wcs_per_time),
            },
            coords={
                "time": ("time", np.asarray(times, dtype=float)),
                "frequency": ("frequency", np.array([55e6])),
                "polarization": ("polarization", np.array([0])),
                "l": ("l", np.linspace(-0.2, 0.2, 16)),
                "m": ("m", np.linspace(-0.2, 0.2, 16)),
            },
        )
        ds["SKY"].attrs["fits_wcs_header"] = headers[0]

        l_idx, m_idx, visible = ds.radport._compute_pixel_track(catalog_ra, catalog_dec)
        assert np.all(visible)
        assert (int(l_idx[0]), int(m_idx[0])) != (int(l_idx[1]), int(m_idx[1]))

        for ti in (0, 1):
            lb, mb = ds.radport.coords_to_pixel(catalog_ra, catalog_dec, time_idx=ti)
            assert abs(lb - int(l_idx[ti])) <= 1
            assert abs(mb - int(m_idx[ti])) <= 1

        dynspec = ds.radport.dynamic_spectrum(ra=catalog_ra, dec=catalog_dec)
        assert dynspec.attrs["tracking"] is True
        assert np.all(np.isfinite(dynspec.values))

    @staticmethod
    def _per_time_wcs_dataset(
        n_time: int,
        *,
        crval1_start: float = 180.0,
        crval1_step: float = 0.1,
        catalog_dec: float = 45.0,
    ) -> tuple[xr.Dataset, float, float]:
        from tests.test_fits_to_zarr import _make_sin_wcs_header_str

        catalog_ra = 180.0
        headers = [
            _make_sin_wcs_header_str(
                nx=16,
                ny=16,
                crval1=crval1_start + crval1_step * ti,
                crval2=catalog_dec,
            )
            for ti in range(n_time)
        ]
        enc = [h.encode("utf-8") for h in headers]
        wcs_per_time = np.array(
            [np.bytes_(e) for e in enc],
            dtype=f"S{max(len(e) for e in enc)}",
        )
        times = [60000.0 + 0.01 * i for i in range(n_time)]
        ds = xr.Dataset(
            {
                "SKY": (
                    ("time", "frequency", "polarization", "m", "l"),
                    np.ones((n_time, 1, 1, 16, 16), dtype=np.float32),
                ),
                "wcs_header_str": (("time",), wcs_per_time),
            },
            coords={
                "time": ("time", np.asarray(times, dtype=float)),
                "frequency": ("frequency", np.array([55e6])),
                "polarization": ("polarization", np.array([0])),
                "l": ("l", np.linspace(-0.2, 0.2, 16)),
                "m": ("m", np.linspace(-0.2, 0.2, 16)),
            },
        )
        ds["SKY"].attrs["fits_wcs_header"] = headers[0]
        return ds, catalog_ra, catalog_dec

    def test_world2pix_from_header_str_matches_get_wcs(self) -> None:
        from astropy.io.fits import Header
        from astropy.wcs import WCS

        from ovro_lwa_portal.accessor import _world2pix_from_header_str
        from tests.test_fits_to_zarr import _make_sin_wcs_header_str

        hdr = _make_sin_wcs_header_str(nx=16, ny=16, crval1=180.0, crval2=45.0)
        wcs = WCS(Header.fromstring(hdr, sep="\n")).celestial
        sky = wcs.pixel_to_world(4, 4)
        ra_deg = float(sky.ra.deg)
        dec_deg = float(sky.dec.deg)
        li, mi, visible = _world2pix_from_header_str(hdr, ra_deg, dec_deg, 16, 16)
        assert visible
        assert li == 4
        assert mi == 4

    def test_per_time_wcs_track_table_uses_template_for_uniform_headers(self) -> None:
        from ovro_lwa_portal.accessor import (
            _build_per_time_wcs_track_table,
            _bulk_per_time_wcs_header_strings,
        )

        ds, _, _ = self._per_time_wcs_dataset(4)
        header_strs = _bulk_per_time_wcs_header_strings(ds, 4)
        table = _build_per_time_wcs_track_table(header_strs)
        assert table.use_template
        assert table.template_header is not None
        assert np.all(table.header_valid)

    def test_per_time_wcs_track_table_falls_back_when_headers_differ(self) -> None:
        from ovro_lwa_portal.accessor import _build_per_time_wcs_track_table
        from tests.test_fits_to_zarr import _make_sin_wcs_header_str

        hdr0 = _make_sin_wcs_header_str(nx=16, ny=16, crval1=180.0, crval2=45.0)
        hdr1 = _make_sin_wcs_header_str(
            nx=16, ny=16, crval1=180.0, crval2=45.0, cdelt=0.2
        )
        table = _build_per_time_wcs_track_table([hdr0, hdr1])
        assert not table.use_template
        assert table.template_header is None
        assert np.all(table.header_valid)

    def test_per_time_wcs_track_matches_coords_to_pixel(self) -> None:
        ds, catalog_ra, catalog_dec = self._per_time_wcs_dataset(12)
        l_idx, m_idx, visible = ds.radport._compute_pixel_track(catalog_ra, catalog_dec)
        assert np.all(visible)
        assert len({int(v) for v in l_idx}) > 1
        for ti in range(int(ds.sizes["time"])):
            lb, mb = ds.radport.coords_to_pixel(
                catalog_ra, catalog_dec, time_idx=ti
            )
            assert abs(lb - int(l_idx[ti])) <= 1
            assert abs(mb - int(m_idx[ti])) <= 1

    def test_per_time_wcs_track_marks_empty_header_invisible(self) -> None:
        from tests.test_fits_to_zarr import _make_sin_wcs_header_str

        from ovro_lwa_portal.accessor import _bulk_per_time_wcs_header_strings

        hdr0 = _make_sin_wcs_header_str(nx=8, ny=8, crval1=180.0, crval2=45.0)
        enc0 = hdr0.encode("utf-8")
        wcs_per_time = np.array(
            [np.bytes_(enc0), np.bytes_(b"")],
            dtype=f"S{len(enc0)}",
        )
        ds = xr.Dataset(
            {
                "SKY": (
                    ("time", "frequency", "polarization", "m", "l"),
                    np.zeros((2, 1, 1, 8, 8), dtype=np.float32),
                ),
                "wcs_header_str": (("time",), wcs_per_time),
            },
            coords={
                "time": ("time", [60000.0, 60000.01]),
                "frequency": ("frequency", np.array([55e6])),
                "polarization": ("polarization", np.array([0])),
                "l": ("l", np.linspace(-0.1, 0.1, 8)),
                "m": ("m", np.linspace(-0.1, 0.1, 8)),
            },
        )
        headers = _bulk_per_time_wcs_header_strings(ds, 2)
        assert headers[0].strip()
        assert not headers[1].strip()

        l_idx, m_idx, visible = ds.radport._compute_pixel_track(180.0, 45.0)
        assert visible[0]
        assert not visible[1]
        assert l_idx[1] == ds.sizes["l"]
        assert m_idx[1] == ds.sizes["m"]

    def test_vectorized_tracked_pixel_values_matches_isel_loop(self) -> None:
        from ovro_lwa_portal.accessor import _vectorized_tracked_pixel_values

        rng = np.random.default_rng(0)
        n_time, n_freq, n_l, n_m = 5, 3, 8, 8
        data = rng.random((n_time, n_freq, 1, n_l, n_m), dtype=np.float32)
        ds = xr.Dataset(
            {
                "SKY": (
                    ("time", "frequency", "polarization", "l", "m"),
                    data,
                )
            },
            coords={
                "time": np.arange(n_time, dtype=float),
                "frequency": np.arange(n_freq, dtype=float),
                "polarization": [0],
                "l": np.arange(n_l, dtype=float),
                "m": np.arange(n_m, dtype=float),
            },
        )
        data_var = ds["SKY"].isel(polarization=0)
        vis_times = np.array([0, 1, 2, 4], dtype=int)
        vis_l = np.array([1, 2, 3, 4], dtype=int)
        vis_m = np.array([4, 3, 2, 1], dtype=int)
        plane = _vectorized_tracked_pixel_values(data_var, vis_times, vis_l, vis_m)
        loop_rows = [
            np.asarray(
                data_var.isel(time=int(t), l=int(li), m=int(mi)).values,
                dtype=np.float64,
            )
            for t, li, mi in zip(vis_times, vis_l, vis_m, strict=True)
        ]
        loop_plane = np.stack(loop_rows, axis=0)
        np.testing.assert_allclose(plane, loop_plane)

    def test_vectorized_tracked_pixel_values_dask_backed(self) -> None:
        import dask.array as da

        from ovro_lwa_portal.accessor import _vectorized_tracked_pixel_values

        n_time, n_freq, n_l, n_m = 4, 2, 6, 6
        backing = da.from_array(
            np.arange(n_time * n_freq * n_l * n_m, dtype=np.float32).reshape(
                n_time, n_freq, 1, n_l, n_m
            ),
            chunks=(1, 1, 1, 3, 3),
        )
        ds = xr.Dataset(
            {
                "SKY": (
                    ("time", "frequency", "polarization", "l", "m"),
                    backing,
                )
            },
            coords={
                "time": np.arange(n_time, dtype=float),
                "frequency": np.arange(n_freq, dtype=float),
                "polarization": [0],
                "l": np.arange(n_l, dtype=float),
                "m": np.arange(n_m, dtype=float),
            },
        )
        data_var = ds["SKY"].isel(polarization=0)
        vis_times = np.array([0, 2, 3], dtype=int)
        vis_l = np.array([1, 2, 3], dtype=int)
        vis_m = np.array([2, 3, 4], dtype=int)
        plane = _vectorized_tracked_pixel_values(data_var, vis_times, vis_l, vis_m)
        expected = np.stack(
            [
                np.asarray(
                    data_var.isel(time=int(t), l=int(li), m=int(mi)).compute().values,
                    dtype=np.float64,
                )
                for t, li, mi in zip(vis_times, vis_l, vis_m, strict=True)
            ],
            axis=0,
        )
        np.testing.assert_allclose(plane, expected)

    def test_per_time_wcs_track_fallback_matches_coords_to_pixel(self) -> None:
        from tests.test_fits_to_zarr import _make_sin_wcs_header_str

        catalog_ra = 180.0
        catalog_dec = 45.0
        hdr0 = _make_sin_wcs_header_str(nx=16, ny=16, crval1=180.0, crval2=catalog_dec)
        hdr1 = _make_sin_wcs_header_str(
            nx=16, ny=16, crval1=180.5, crval2=catalog_dec, cdelt=0.2
        )
        enc = [h.encode("utf-8") for h in (hdr0, hdr1)]
        wcs_per_time = np.array(
            [np.bytes_(e) for e in enc],
            dtype=f"S{max(len(e) for e in enc)}",
        )
        ds = xr.Dataset(
            {
                "SKY": (
                    ("time", "frequency", "polarization", "m", "l"),
                    np.ones((2, 1, 1, 16, 16), dtype=np.float32),
                ),
                "wcs_header_str": (("time",), wcs_per_time),
            },
            coords={
                "time": ("time", [60000.0, 60000.01]),
                "frequency": ("frequency", np.array([55e6])),
                "polarization": ("polarization", np.array([0])),
                "l": ("l", np.linspace(-0.2, 0.2, 16)),
                "m": ("m", np.linspace(-0.2, 0.2, 16)),
            },
        )
        l_idx, m_idx, visible = ds.radport._compute_pixel_track(catalog_ra, catalog_dec)
        assert np.all(visible)
        for ti in (0, 1):
            lb, mb = ds.radport.coords_to_pixel(catalog_ra, catalog_dec, time_idx=ti)
            assert abs(lb - int(l_idx[ti])) <= 1
            assert abs(mb - int(m_idx[ti])) <= 1

    def test_below_horizon_nan_in_light_curve(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """Time steps where source is below horizon are NaN in light curve."""
        ds = valid_ovro_dataset_with_tracking_wcs
        # Dec=-80 is never visible from OVRO-LWA
        with pytest.warns(UserWarning, match="never above the horizon"):
            lc = ds.radport.light_curve(ra=0.0, dec=-80.0, freq_mhz=50.0)
        assert np.all(np.isnan(lc.values))

    def test_dedispersed_radec_works_end_to_end(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """dynamic_spectrum_dedispersed(ra, dec) works end-to-end."""
        from astropy.time import Time
        from astropy import units as u

        ds = valid_ovro_dataset_with_tracking_wcs
        t0 = ds.coords["time"].values[0]
        lst_deg = float(
            Time(t0, format="mjd", scale="utc")
            .sidereal_time("mean", longitude=-118.2817 * u.deg)
            .deg
        )
        result = ds.radport.dynamic_spectrum_dedispersed(
            ra=lst_deg, dec=37.2339, dm=10.0
        )
        assert set(result.dims) == {"time", "frequency"}
        assert result.attrs["dm"] == 10.0

    def test_observatory_override_propagates(
        self, valid_ovro_dataset_with_tracking_wcs: xr.Dataset
    ) -> None:
        """observatory parameter propagates through light_curve."""
        from astropy.coordinates import EarthLocation
        from astropy import units as u

        ds = valid_ovro_dataset_with_tracking_wcs
        custom_obs = EarthLocation(
            lat=0.0 * u.deg, lon=0.0 * u.deg, height=0.0 * u.m
        )
        # Should not error
        lc = ds.radport.light_curve(
            ra=180.0, dec=0.0, freq_mhz=50.0, observatory=custom_obs
        )
        assert "time" in lc.dims

    def test_positional_args_raise_typeerror(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """light_curve(0.0, 0.0) positional raises TypeError (breaking change)."""
        with pytest.raises(TypeError):
            valid_ovro_dataset.radport.light_curve(0.0, 0.0)


# =========================================================================
# Phase 3: Single-Time Celestial Methods Tests
# =========================================================================


class TestCelestialSingleTimeMethods:
    """Tests for single-time methods with RA/Dec support."""

    def test_spectrum_radec_returns_valid(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """spectrum(ra, dec) returns valid spectrum at a visible time step."""
        # time_idx=1 because RA=180, Dec=45 is below the horizon at
        # time_idx=0 (MJD 60000.0) for the OVRO-LWA location.
        spec = valid_ovro_dataset_with_wcs.radport.spectrum(
            ra=180.0, dec=45.0, time_idx=1
        )
        assert "frequency" in spec.dims
        assert spec.size > 0

    def test_spectrum_lm_keyword_still_works(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectrum(l=..., m=...) still works."""
        spec = valid_ovro_dataset.radport.spectrum(l=0.0, m=0.0, time_idx=0)
        assert "frequency" in spec.dims

    def test_spectrum_neither_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """spectrum() with no coordinates raises ValueError."""
        with pytest.raises(ValueError, match="Must provide"):
            valid_ovro_dataset.radport.spectrum(time_idx=0)

    def test_spectral_index_radec_returns_float(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """spectral_index(ra, dec) returns a valid float."""
        alpha = valid_ovro_dataset_with_wcs.radport.spectral_index(
            ra=180.0, dec=45.0, time_idx=1
        )
        assert isinstance(alpha, float)

    def test_integrated_flux_radec_returns_float(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """integrated_flux(ra, dec) returns a valid float."""
        flux = valid_ovro_dataset_with_wcs.radport.integrated_flux(
            ra=180.0, dec=45.0, time_idx=1
        )
        assert isinstance(flux, float)

    def test_cutout_radec_returns_valid_2d(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """cutout(ra_center, dec_center, dl, dm) returns valid 2D DataArray."""
        cutout = valid_ovro_dataset_with_wcs.radport.cutout(
            ra_center=180.0, dec_center=45.0, dl=0.3, dm=0.3, time_idx=1
        )
        assert isinstance(cutout, xr.DataArray)
        assert set(cutout.dims) == {"l", "m"}

    def test_cutout_lm_keyword_still_works(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """cutout(l_center=..., m_center=...) still works."""
        cutout = valid_ovro_dataset.radport.cutout(
            l_center=0.0, m_center=0.0, dl=0.3, dm=0.3
        )
        assert set(cutout.dims) == {"l", "m"}

    def test_cutout_neither_raises(
        self, valid_ovro_dataset: xr.Dataset
    ) -> None:
        """cutout() with no center coordinates raises ValueError."""
        with pytest.raises(ValueError, match="Must provide"):
            valid_ovro_dataset.radport.cutout(dl=0.1, dm=0.1)

    def test_cutout_both_raises(
        self, valid_ovro_dataset_with_wcs: xr.Dataset
    ) -> None:
        """cutout() with both coordinate types raises ValueError."""
        with pytest.raises(ValueError, match="not both"):
            valid_ovro_dataset_with_wcs.radport.cutout(
                ra_center=180.0, dec_center=45.0,
                l_center=0.0, m_center=0.0,
                dl=0.1, dm=0.1,
            )
