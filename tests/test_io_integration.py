"""Integration tests for the io module with real data."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ovro_lwa_portal import open_dataset, resolve_source
from ovro_lwa_portal.io import DataSourceError, summarize_lm_chunks, warn_if_suboptimal_lm_chunks


@pytest.fixture
def sample_zarr_store(tmp_path: Path) -> Path:
    """Create a sample OVRO-LWA-like zarr store for testing.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory provided by pytest.

    Returns
    -------
    Path
        Path to the created zarr store.
    """
    zarr_path = tmp_path / "sample_observation.zarr"

    # Create a realistic OVRO-LWA dataset structure
    time = np.arange(10)
    frequency = np.linspace(40e6, 80e6, 20)  # 40-80 MHz
    l = np.arange(512)
    m = np.arange(512)

    ds = xr.Dataset(
        {
            "SKY": (
                ["time", "frequency", "polarization", "l", "m"],
                np.random.rand(10, 20, 1, 512, 512).astype(np.float32),
            ),
            "BEAM": (
                ["time", "frequency", "polarization", "l", "m"],
                np.random.rand(10, 20, 1, 512, 512).astype(np.float32),
            ),
        },
        coords={
            "time": time,
            "frequency": frequency,
            "polarization": [0],
            "l": l,
            "m": m,
            "right_ascension": (["m", "l"], np.random.rand(512, 512) * 360),
            "declination": (["m", "l"], np.random.rand(512, 512) * 180 - 90),
        },
        attrs={
            "instrument": "OVRO-LWA",
            "observation_date": "2024-05-24",
        },
    )

    # Add WCS header string (simplified)
    wcs_header = "WCSAXES = 2\nCTYPE1  = 'RA---SIN'\nCTYPE2  = 'DEC--SIN'\n"
    ds.attrs["fits_wcs_header"] = wcs_header

    # Save to zarr
    ds.to_zarr(zarr_path, mode="w")

    return zarr_path


@pytest.fixture
def small_chunk_zarr_store(tmp_path: Path) -> Path:
    """Zarr store with 128×128 on-disk l/m chunks for overlay I/O tests."""
    zarr_path = tmp_path / "small_chunk_observation.zarr"
    time = np.arange(2)
    frequency = np.linspace(50e6, 55e6, 4)
    l = np.arange(256)
    m = np.arange(256)
    ds = xr.Dataset(
        {
            "SKY": (
                ["time", "frequency", "polarization", "l", "m"],
                np.random.rand(2, 4, 1, 256, 256).astype(np.float32),
            ),
        },
        coords={
            "time": time,
            "frequency": frequency,
            "polarization": [0],
            "l": l,
            "m": m,
        },
    )
    ds.chunk({"time": 1, "frequency": 1, "polarization": 1, "l": 128, "m": 128}).to_zarr(
        zarr_path, mode="w"
    )
    return zarr_path


class TestLmChunkSummary:
    def test_summarize_lm_chunks(self, small_chunk_zarr_store: Path) -> None:
        summary = summarize_lm_chunks(small_chunk_zarr_store)
        assert summary["l_min"] == 128
        assert summary["m_min"] == 128
        assert summary["l_chunks"] == (128, 128)
        assert summary["m_chunks"] == (128, 128)

    def test_warn_if_suboptimal_lm_chunks(self, small_chunk_zarr_store: Path) -> None:
        with pytest.warns(UserWarning, match="small on-disk l/m chunks"):
            warn_if_suboptimal_lm_chunks(small_chunk_zarr_store)

    def test_open_dataset_warns_on_small_lm_chunks(
        self, small_chunk_zarr_store: Path
    ) -> None:
        with pytest.warns(UserWarning, match="small on-disk l/m chunks"):
            open_dataset(small_chunk_zarr_store)


class TestOpenDatasetIntegration:
    """Integration tests for open_dataset with realistic data."""

    def test_load_sample_dataset(self, sample_zarr_store: Path) -> None:
        """Test loading a sample OVRO-LWA dataset."""
        ds = open_dataset(sample_zarr_store)

        assert isinstance(ds, xr.Dataset)
        assert "SKY" in ds.data_vars
        assert "BEAM" in ds.data_vars
        assert set(ds.sizes.keys()) == {"time", "frequency", "polarization", "l", "m"}

    def test_load_with_validation(self, sample_zarr_store: Path) -> None:
        """Test loading with validation enabled."""
        ds = open_dataset(sample_zarr_store, validate=True)

        assert isinstance(ds, xr.Dataset)
        # Should pass validation without warnings for this dataset

    def test_load_without_validation(self, sample_zarr_store: Path) -> None:
        """Test loading without validation."""
        ds = open_dataset(sample_zarr_store, validate=False)

        assert isinstance(ds, xr.Dataset)

    def test_load_with_custom_chunks(self, sample_zarr_store: Path) -> None:
        """Test loading with custom chunking."""
        ds = open_dataset(
            sample_zarr_store,
            chunks={"time": 5, "frequency": 10, "l": 256, "m": 256},
        )

        assert isinstance(ds, xr.Dataset)
        # Verify chunking
        assert hasattr(ds["SKY"].data, "chunks")

    def test_load_with_auto_chunks(self, sample_zarr_store: Path) -> None:
        """Test loading with automatic chunking."""
        ds = open_dataset(sample_zarr_store, chunks="auto")

        assert isinstance(ds, xr.Dataset)
        assert hasattr(ds["SKY"].data, "chunks")

    def test_load_without_chunks(self, sample_zarr_store: Path) -> None:
        """Test loading entire dataset into memory."""
        ds = open_dataset(sample_zarr_store, chunks=None)

        assert isinstance(ds, xr.Dataset)
        # Data should be numpy array
        assert isinstance(ds["SKY"].data, np.ndarray)

    def test_dataset_attributes_preserved(self, sample_zarr_store: Path) -> None:
        """Test that dataset attributes are preserved."""
        ds = open_dataset(sample_zarr_store, validate=False)

        assert "instrument" in ds.attrs
        assert ds.attrs["instrument"] == "OVRO-LWA"
        assert "fits_wcs_header" in ds.attrs

    def test_open_dataset_strips_stale_fits_wcs_header_with_wcs_header_str(
        self, tmp_path: Path
    ) -> None:
        """Legacy Zarr rows must not keep time-0 ``fits_wcs_header`` on SKY in memory."""
        import numpy as np

        zarr_path = tmp_path / "per_time_wcs.zarr"
        hdr = np.bytes_(b"SIMPLE  = T\nCRVAL1  = 61.0\n")
        ds = xr.Dataset(
            {
                "SKY": (
                    ["time", "frequency", "polarization", "l", "m"],
                    np.zeros((2, 1, 1, 4, 4), dtype=np.float32),
                ),
                "wcs_header_str": (("time",), np.array([hdr, hdr])),
            },
            coords={
                "time": [60000.0, 60001.0],
                "frequency": [55e6],
                "polarization": [0],
                "l": np.arange(4),
                "m": np.arange(4),
            },
            attrs={"fits_wcs_header": "stale-global"},
        )
        ds["SKY"].attrs["fits_wcs_header"] = "stale-sky"
        ds.to_zarr(zarr_path, mode="w")

        loaded = open_dataset(zarr_path, validate=False)
        assert "fits_wcs_header" not in loaded.attrs
        assert "fits_wcs_header" not in loaded["SKY"].attrs
        assert "wcs_header_str" in loaded

    def test_coordinates_preserved(self, sample_zarr_store: Path) -> None:
        """Test that coordinates are preserved."""
        ds = open_dataset(sample_zarr_store, validate=False)

        assert "right_ascension" in ds.coords
        assert "declination" in ds.coords
        assert "time" in ds.coords
        assert "frequency" in ds.coords

    def test_data_values_preserved(self, sample_zarr_store: Path) -> None:
        """Test that data values are correctly loaded."""
        # Load original dataset
        ds_original = xr.open_zarr(sample_zarr_store)

        # Load via open_dataset
        ds_loaded = open_dataset(sample_zarr_store, validate=False, chunks=None)

        # Compare values
        np.testing.assert_array_equal(
            ds_original["SKY"].values,
            ds_loaded["SKY"].values,
        )

    def test_subset_selection(self, sample_zarr_store: Path) -> None:
        """Test selecting subsets of data."""
        ds = open_dataset(sample_zarr_store)

        # Select subset
        # Note: xarray's label-based slicing includes both endpoints for integer coordinates
        # slice(0, 5) gives [0, 1, 2, 3, 4, 5] = 6 elements
        subset = ds.sel(time=slice(0, 5), frequency=slice(40e6, 60e6))

        assert len(subset.time) == 6  # Includes both 0 and 5
        assert subset.frequency.min() >= 40e6
        assert subset.frequency.max() <= 60e6

    def test_computation_on_loaded_data(self, sample_zarr_store: Path) -> None:
        """Test performing computations on loaded data."""
        ds = open_dataset(sample_zarr_store)

        # Compute mean
        mean_sky = ds.SKY.mean(dim=["l", "m"]).compute()

        assert mean_sky.shape == (10, 20, 1)
        assert not np.isnan(mean_sky.values).any()

    def test_multiple_loads_same_store(self, sample_zarr_store: Path) -> None:
        """Test loading the same store multiple times."""
        ds1 = open_dataset(sample_zarr_store, validate=False)
        ds2 = open_dataset(sample_zarr_store, validate=False)

        # Both should load successfully
        assert isinstance(ds1, xr.Dataset)
        assert isinstance(ds2, xr.Dataset)

        # Values should be identical
        np.testing.assert_array_equal(
            ds1["SKY"].values,
            ds2["SKY"].values,
        )

    def test_pathlib_path_input(self, sample_zarr_store: Path) -> None:
        """Test that Path objects work as input."""
        # sample_zarr_store is already a Path object
        ds = open_dataset(sample_zarr_store)

        assert isinstance(ds, xr.Dataset)

    def test_string_path_input(self, sample_zarr_store: Path) -> None:
        """Test that string paths work as input."""
        ds = open_dataset(str(sample_zarr_store))

        assert isinstance(ds, xr.Dataset)


# Known test DOI for OVRO-LWA data on the DataCite test API
TEST_DOI = "10.33569/9wsys-h7b71"


@pytest.mark.network
class TestDOIResolutionNetwork:
    """Network tests for DOI resolution against real DataCite APIs.

    These tests require network access and are skipped by default.
    Run with: pytest -m network
    """

    def test_resolve_test_doi(self) -> None:
        """Test resolving a known test DOI via the DataCite test API."""
        result = resolve_source(TEST_DOI, production=False)

        assert result["source_type"] == "doi"
        assert result["resolved_url"] is not None
        assert result["resolved_url"].startswith("https://")

    def test_resolve_test_doi_returns_osn_url(self) -> None:
        """Test that the test DOI resolves to an OSN HTTPS URL."""
        result = resolve_source(TEST_DOI, production=False)

        assert ".osn.mghpcc.org" in result["resolved_url"]

    def test_resolve_test_doi_with_s3_conversion(self) -> None:
        """Test resolving a test DOI with S3 credentials triggers OSN conversion."""
        result = resolve_source(
            TEST_DOI,
            production=False,
            storage_options={"key": "test", "secret": "test"},
        )

        assert result["s3_url"] is not None
        assert result["s3_url"].startswith("s3://")
        assert result["endpoint"] is not None
        assert result["bucket"] is not None


@pytest.mark.network
@pytest.mark.osn
class TestOSNAccessNetwork:
    """Network tests for actual OSN S3 access.

    These tests require both network access and OSN credentials.
    Run with: pytest -m "network and osn"

    Required environment variables:
        OSN_KEY: OSN access key
        OSN_SECRET: OSN secret key
    """

    @pytest.fixture
    def osn_credentials(self) -> dict[str, str]:
        """Get OSN credentials from environment or skip."""
        key = os.environ.get("OSN_KEY")
        secret = os.environ.get("OSN_SECRET")
        if not key or not secret:
            pytest.skip("OSN credentials not available (set OSN_KEY and OSN_SECRET)")
        return {"key": key, "secret": secret}

    def test_resolve_and_access_test_doi(self, osn_credentials: dict[str, str]) -> None:
        """Test full DOI→URL→S3 resolution and bucket access."""
        result = resolve_source(
            TEST_DOI,
            production=False,
            storage_options=osn_credentials,
        )

        assert result["s3_url"] is not None
        assert result["endpoint"] is not None

        # Verify bucket is accessible using fsspec
        import fsspec

        fs = fsspec.filesystem(
            "s3",
            key=osn_credentials["key"],
            secret=osn_credentials["secret"],
            client_kwargs={"endpoint_url": result["endpoint"]},
        )

        # List the bucket — this is the actual access check
        bucket = result["bucket"]
        listing = fs.ls(bucket, detail=False)
        assert len(listing) > 0

    def test_open_dataset_from_test_doi(self, osn_credentials: dict[str, str]) -> None:
        """Test loading actual data from a test DOI end-to-end."""
        ds = open_dataset(
            TEST_DOI,
            production=False,
            storage_options=osn_credentials,
            validate=False,
        )

        assert isinstance(ds, xr.Dataset)
        # Should have at least one data variable
        assert len(ds.data_vars) > 0

    def test_invalid_bucket_gives_clear_error(self, osn_credentials: dict[str, str]) -> None:
        """Test that an invalid bucket produces a clear DataSourceError."""
        with pytest.raises(DataSourceError, match="Cannot access remote storage"):
            open_dataset(
                "s3://nonexistent-bucket-12345/data.zarr",
                storage_options={
                    **osn_credentials,
                    "client_kwargs": {
                        "endpoint_url": "https://caltech1.osn.mghpcc.org",
                    },
                },
                validate=False,
            )
