"""Tests for the io module (open_dataset function)."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import xarray as xr

from ovro_lwa_portal.io import (
    DataSourceError,
    _check_remote_access,
    _convert_osn_https_to_s3,
    _detect_source_type,
    _is_doi,
    _normalize_doi,
    _resolve_doi,
    _resolve_doi_from_metadata,
    _validate_dataset,
    open_dataset,
    resolve_source,
    validate_local_zarr_store,
)


class TestDOIDetection:
    """Tests for DOI detection and normalization."""

    def test_is_doi_with_prefix(self) -> None:
        """Test DOI detection with 'doi:' prefix."""
        assert _is_doi("doi:10.5281/zenodo.1234567")
        assert _is_doi("DOI:10.5281/zenodo.1234567")

    def test_is_doi_without_prefix(self) -> None:
        """Test DOI detection without prefix."""
        assert _is_doi("10.5281/zenodo.1234567")
        assert _is_doi("10.1234/abcd.efgh")

    def test_is_not_doi(self) -> None:
        """Test that non-DOI strings are not detected as DOIs."""
        assert not _is_doi("https://example.com")
        assert not _is_doi("/path/to/file")
        assert not _is_doi("s3://bucket/key")
        assert not _is_doi("not-a-doi")

    def test_normalize_doi_with_prefix(self) -> None:
        """Test DOI normalization removes prefix."""
        assert _normalize_doi("doi:10.5281/zenodo.1234567") == "10.5281/zenodo.1234567"
        assert _normalize_doi("DOI:10.5281/zenodo.1234567") == "10.5281/zenodo.1234567"

    def test_normalize_doi_without_prefix(self) -> None:
        """Test DOI normalization preserves DOI without prefix."""
        assert _normalize_doi("10.5281/zenodo.1234567") == "10.5281/zenodo.1234567"


class TestSourceTypeDetection:
    """Tests for source type detection."""

    def test_detect_local_path(self) -> None:
        """Test detection of local file paths."""
        source_type, normalized = _detect_source_type("/path/to/data.zarr")
        assert source_type == "local"
        assert normalized == "/path/to/data.zarr"

        source_type, normalized = _detect_source_type(Path("relative/path/data.zarr"))
        assert source_type == "local"

    def test_detect_remote_http(self) -> None:
        """Test detection of HTTP/HTTPS URLs."""
        source_type, normalized = _detect_source_type("https://example.com/data.zarr")
        assert source_type == "remote"
        assert normalized == "https://example.com/data.zarr"

        source_type, normalized = _detect_source_type("http://example.com/data.zarr")
        assert source_type == "remote"

    def test_detect_remote_s3(self) -> None:
        """Test detection of S3 URLs."""
        source_type, normalized = _detect_source_type("s3://bucket/data.zarr")
        assert source_type == "remote"
        assert normalized == "s3://bucket/data.zarr"

    def test_detect_remote_gcs(self) -> None:
        """Test detection of Google Cloud Storage URLs."""
        source_type, normalized = _detect_source_type("gs://bucket/data.zarr")
        assert source_type == "remote"

        source_type, normalized = _detect_source_type("gcs://bucket/data.zarr")
        assert source_type == "remote"

    def test_detect_doi(self) -> None:
        """Test detection of DOI identifiers."""
        source_type, normalized = _detect_source_type("doi:10.5281/zenodo.1234567")
        assert source_type == "doi"
        assert normalized == "10.5281/zenodo.1234567"

        source_type, normalized = _detect_source_type("10.5281/zenodo.1234567")
        assert source_type == "doi"
        assert normalized == "10.5281/zenodo.1234567"


class TestDatasetValidation:
    """Tests for dataset validation."""

    def test_validate_valid_ovro_dataset(self) -> None:
        """Test validation of a valid OVRO-LWA dataset."""
        ds = xr.Dataset(
            {
                "SKY": (["time", "frequency", "l", "m"], np.random.rand(2, 3, 10, 10)),
            },
            coords={
                "time": np.arange(2),
                "frequency": np.arange(3),
                "l": np.arange(10),
                "m": np.arange(10),
            },
        )

        # Should not raise or warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Turn warnings into errors
            _validate_dataset(ds)  # Should not raise

    def test_validate_missing_dimensions(self) -> None:
        """Test validation warns about missing expected dimensions."""
        ds = xr.Dataset(
            {
                "data": (["x", "y"], np.random.rand(10, 10)),
            },
            coords={
                "x": np.arange(10),
                "y": np.arange(10),
            },
        )

        # Should not raise but should warn
        with pytest.warns(UserWarning, match="may not be OVRO-LWA format"):
            _validate_dataset(ds)

    def test_validate_missing_variables(self) -> None:
        """Test validation warns about missing expected variables."""
        ds = xr.Dataset(
            {
                "other_var": (["time", "frequency"], np.random.rand(2, 3)),
            },
            coords={
                "time": np.arange(2),
                "frequency": np.arange(3),
            },
        )

        # Should not raise but should warn
        with pytest.warns(UserWarning, match="may not be OVRO-LWA format"):
            _validate_dataset(ds)


class TestOpenDataset:
    """Tests for open_dataset function."""

    def test_open_local_zarr(self, tmp_path: Path) -> None:
        """Test opening a local zarr store."""
        # Create a test zarr store
        zarr_path = tmp_path / "test.zarr"
        ds = xr.Dataset(
            {
                "SKY": (["time", "frequency", "l", "m"], np.random.rand(2, 3, 10, 10)),
            },
            coords={
                "time": np.arange(2),
                "frequency": np.arange(3),
                "l": np.arange(10),
                "m": np.arange(10),
            },
        )
        ds.to_zarr(zarr_path)

        # Load it back
        loaded_ds = open_dataset(zarr_path, validate=False)

        assert isinstance(loaded_ds, xr.Dataset)
        assert "SKY" in loaded_ds.data_vars
        assert set(loaded_ds.sizes.keys()) == {"time", "frequency", "l", "m"}

    def test_open_local_zarr_with_validation(self, tmp_path: Path) -> None:
        """Test opening a local zarr store with validation."""
        zarr_path = tmp_path / "test.zarr"
        ds = xr.Dataset(
            {
                "SKY": (["time", "frequency", "l", "m"], np.random.rand(2, 3, 10, 10)),
            },
            coords={
                "time": np.arange(2),
                "frequency": np.arange(3),
                "l": np.arange(10),
                "m": np.arange(10),
            },
        )
        ds.to_zarr(zarr_path)

        # Load with validation (default)
        loaded_ds = open_dataset(zarr_path)

        assert isinstance(loaded_ds, xr.Dataset)

    def test_open_nonexistent_local_path(self, tmp_path: Path) -> None:
        """Test opening a nonexistent local path raises FileNotFoundError."""
        nonexistent = tmp_path / "nonexistent.zarr"

        with pytest.raises(FileNotFoundError, match="Zarr store path does not exist"):
            open_dataset(nonexistent)

    def test_open_with_custom_chunks(self, tmp_path: Path) -> None:
        """Test opening with custom chunk specification."""
        import warnings

        zarr_path = tmp_path / "test.zarr"
        ds = xr.Dataset(
            {
                "SKY": (["time", "frequency", "l", "m"], np.random.rand(10, 20, 100, 100)),
            },
            coords={
                "time": np.arange(10),
                "frequency": np.arange(20),
                "l": np.arange(100),
                "m": np.arange(100),
            },
        )
        ds.to_zarr(zarr_path)

        # Load with custom chunks (may trigger performance warning from xarray)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            loaded_ds = open_dataset(
                zarr_path,
                chunks={"time": 5, "frequency": 10},
                validate=False,
            )

        assert isinstance(loaded_ds, xr.Dataset)
        # Check that data is chunked (dask array)
        assert hasattr(loaded_ds["SKY"].data, "chunks")

    def test_open_with_no_chunks(self, tmp_path: Path) -> None:
        """Test opening without chunking (load into memory)."""
        zarr_path = tmp_path / "test.zarr"
        ds = xr.Dataset(
            {
                "SKY": (["time", "frequency"], np.random.rand(2, 3)),
            },
            coords={
                "time": np.arange(2),
                "frequency": np.arange(3),
            },
        )
        ds.to_zarr(zarr_path)

        # Load without chunks
        loaded_ds = open_dataset(zarr_path, chunks=None, validate=False)

        assert isinstance(loaded_ds, xr.Dataset)
        # Data should be numpy array, not dask
        assert isinstance(loaded_ds["SKY"].data, np.ndarray)

    @patch("ovro_lwa_portal.io.xr.open_zarr")
    def test_open_remote_http(self, mock_open_zarr: Mock) -> None:
        """Test opening a remote HTTP URL."""
        mock_ds = xr.Dataset(
            {
                "SKY": (["time", "frequency"], np.random.rand(2, 3)),
            }
        )
        mock_open_zarr.return_value = mock_ds

        url = "https://example.com/data.zarr"
        loaded_ds = open_dataset(url, validate=False)

        mock_open_zarr.assert_called_once()
        # The implementation passes an FSMap object, not the URL string directly
        call_arg = mock_open_zarr.call_args[0][0]
        # Verify it's an fsspec mapper (FSMap or similar)
        assert hasattr(call_arg, "fs") or hasattr(call_arg, "__getitem__")
        assert isinstance(loaded_ds, xr.Dataset)

    def test_open_remote_s3_requires_s3fs(self) -> None:
        """Test that S3 URL detection works (actual loading requires s3fs)."""
        # We can't easily test S3 loading without s3fs installed
        # This test just verifies source type detection
        source_type, normalized = _detect_source_type("s3://bucket/data.zarr")
        assert source_type == "remote"
        assert normalized == "s3://bucket/data.zarr"

    @patch("ovro_lwa_portal.io._resolve_doi")
    @patch("ovro_lwa_portal.io.xr.open_zarr")
    def test_open_doi(self, mock_open_zarr: Mock, mock_resolve_doi: Mock) -> None:
        """Test opening a dataset via DOI."""
        mock_resolve_doi.return_value = "https://example.com/data.zarr"
        mock_ds = xr.Dataset(
            {
                "SKY": (["time", "frequency"], np.random.rand(2, 3)),
            }
        )
        mock_open_zarr.return_value = mock_ds

        doi = "doi:10.5281/zenodo.1234567"
        loaded_ds = open_dataset(doi, validate=False)

        mock_resolve_doi.assert_called_once_with("10.5281/zenodo.1234567", production=True)
        mock_open_zarr.assert_called_once()
        assert isinstance(loaded_ds, xr.Dataset)

    @patch("ovro_lwa_portal.io._resolve_doi")
    def test_open_doi_resolution_fails(self, mock_resolve_doi: Mock) -> None:
        """Test opening DOI when resolution fails."""
        mock_resolve_doi.side_effect = Exception("Resolution failed")

        doi = "doi:10.5281/zenodo.1234567"

        with pytest.raises(DataSourceError, match="Failed to resolve DOI"):
            open_dataset(doi, validate=False)

    def test_open_unsupported_engine(self, tmp_path: Path) -> None:
        """Test opening with unsupported engine raises error."""
        zarr_path = tmp_path / "test.zarr"
        ds = xr.Dataset({"data": (["x"], np.arange(10))})
        ds.to_zarr(zarr_path)

        with pytest.raises(DataSourceError, match="Unsupported engine"):
            open_dataset(zarr_path, engine="netcdf", validate=False)

    @patch("ovro_lwa_portal.io.xr.open_zarr")
    def test_open_dataset_load_failure(self, mock_open_zarr: Mock, tmp_path: Path) -> None:
        """Test that load failures are properly wrapped."""
        mock_open_zarr.side_effect = Exception("Load failed")

        zarr_path = tmp_path / "test.zarr"
        xr.Dataset({"SKY": (["time"], [1.0])}, coords={"time": [0]}).to_zarr(zarr_path)

        with pytest.raises(DataSourceError, match="Failed to load dataset"):
            open_dataset(zarr_path, validate=False)


class TestValidateLocalZarrStore:
    """Tests for validate_local_zarr_store (fast path checks)."""

    def test_valid_store_returns_resolved_path(self, tmp_path: Path) -> None:
        zarr_path = tmp_path / "obs.zarr"
        xr.Dataset({"SKY": (["time"], [1.0])}, coords={"time": [0]}).to_zarr(zarr_path)
        resolved = validate_local_zarr_store(zarr_path)
        assert resolved.is_dir()
        assert (resolved / ".zgroup").is_file()

    def test_missing_path_lists_siblings(self, tmp_path: Path) -> None:
        good = tmp_path / "I-Clean-Snapshot-5.zarr"
        xr.Dataset({"SKY": (["time"], [1.0])}, coords={"time": [0]}).to_zarr(good)
        bad = tmp_path / "I-Clean-Snapshot-6.zarr"

        with pytest.raises(FileNotFoundError, match="does not exist") as exc_info:
            validate_local_zarr_store(bad)

        message = str(exc_info.value)
        assert "Did you mean" in message
        assert "I-Clean-Snapshot-5.zarr" in message

    def test_existing_non_zarr_directory(self, tmp_path: Path) -> None:
        empty = tmp_path / "not-a-store"
        empty.mkdir()

        with pytest.raises(DataSourceError, match="not a Zarr group store"):
            validate_local_zarr_store(empty)

    def test_existing_file_raises(self, tmp_path: Path) -> None:
        file_path = tmp_path / "data.zarr"
        file_path.write_text("not zarr", encoding="utf-8")

        with pytest.raises(DataSourceError, match="is a file, not a directory"):
            validate_local_zarr_store(file_path)


class TestDOIResolution:
    """Tests for DOI resolution functionality."""

    def test_doi_resolution_requires_dependencies(self) -> None:
        """Test that DOI resolution requires requests library."""
        # DOI resolution requires optional dependencies
        # This test just verifies DOI detection works
        assert _is_doi("doi:10.5281/zenodo.1234567")
        assert _is_doi("10.5281/zenodo.1234567")

        # Verify normalization
        assert _normalize_doi("doi:10.5281/zenodo.1234567") == "10.5281/zenodo.1234567"

    @patch("requests.get")
    def test_resolve_doi_application_zarr(self, mock_get: Mock) -> None:
        """Test that application/zarr media type is matched."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [
                {
                    "attributes": {
                        "mediaType": "application/zarr",
                        "url": "https://example.com/data.zarr",
                    }
                }
            ]
        }
        mock_get.return_value = mock_response

        result = _resolve_doi("10.5281/test", production=True)
        assert result == "https://example.com/data.zarr"

    @patch("requests.get")
    def test_resolve_doi_application_x_zarr(self, mock_get: Mock) -> None:
        """Test that application/x-zarr media type is matched."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [
                {
                    "attributes": {
                        "mediaType": "application/x-zarr",
                        "url": "https://uma1.osn.mghpcc.org/bucket/data.zarr",
                    }
                }
            ]
        }
        mock_get.return_value = mock_response

        result = _resolve_doi("10.33569/test", production=False)
        assert result == "https://uma1.osn.mghpcc.org/bucket/data.zarr"

    @patch("requests.get")
    def test_resolve_doi_unknown_media_type_uses_first(self, mock_get: Mock) -> None:
        """Test that unknown media type falls back to first entry."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [
                {
                    "attributes": {
                        "mediaType": "application/octet-stream",
                        "url": "https://example.com/first.zarr",
                    }
                },
                {
                    "attributes": {
                        "mediaType": "text/plain",
                        "url": "https://example.com/second.txt",
                    }
                },
            ]
        }
        mock_get.return_value = mock_response

        result = _resolve_doi("10.5281/test", production=True)
        assert result == "https://example.com/first.zarr"

    @patch("ovro_lwa_portal.io._resolve_doi_from_metadata")
    @patch("requests.get")
    def test_resolve_doi_empty_media_falls_back_to_metadata(
        self, mock_get: Mock, mock_metadata: Mock
    ) -> None:
        """Test that empty media list triggers metadata fallback."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"data": []}
        mock_get.return_value = mock_response
        mock_metadata.return_value = "https://example.com/fallback.zarr"

        result = _resolve_doi("10.5281/test", production=True)
        assert result == "https://example.com/fallback.zarr"
        mock_metadata.assert_called_once()

    @patch("ovro_lwa_portal.io._resolve_doi_from_metadata")
    @patch("requests.get")
    def test_resolve_doi_404_falls_back_to_metadata(
        self, mock_get: Mock, mock_metadata: Mock
    ) -> None:
        """Test that 404 from media endpoint triggers metadata fallback."""
        import requests

        mock_response = Mock()
        mock_response.status_code = 404
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_get.return_value.raise_for_status.side_effect = http_error
        mock_metadata.return_value = "https://example.com/fallback.zarr"

        result = _resolve_doi("10.5281/test", production=True)
        assert result == "https://example.com/fallback.zarr"
        mock_metadata.assert_called_once()

    @patch("requests.get")
    def test_resolve_doi_uses_test_api(self, mock_get: Mock) -> None:
        """Test that production=False uses the test DataCite API."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": [
                {
                    "attributes": {
                        "mediaType": "application/x-zarr",
                        "url": "https://example.com/data.zarr",
                    }
                }
            ]
        }
        mock_get.return_value = mock_response

        _resolve_doi("10.33569/test", production=False)
        call_url = mock_get.call_args[0][0]
        assert "api.test.datacite.org" in call_url
        assert "api.datacite.org/dois" not in call_url


class TestDOIMetadataFallback:
    """Tests for _resolve_doi_from_metadata fallback."""

    @patch("requests.get")
    def test_resolve_from_metadata_success(self, mock_get: Mock) -> None:
        """Test successful URL extraction from DOI metadata."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": {
                "attributes": {
                    "url": "https://example.com/metadata-url.zarr",
                }
            }
        }
        mock_get.return_value = mock_response

        result = _resolve_doi_from_metadata(
            "10.5281/test", "https://api.datacite.org/dois"
        )
        assert result == "https://example.com/metadata-url.zarr"

    @patch("requests.get")
    def test_resolve_from_metadata_no_url(self, mock_get: Mock) -> None:
        """Test that missing URL in metadata raises DataSourceError."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "data": {"attributes": {"url": None}}
        }
        mock_get.return_value = mock_response

        with pytest.raises(DataSourceError, match="No download URL found"):
            _resolve_doi_from_metadata(
                "10.5281/test", "https://api.datacite.org/dois"
            )

    @patch("requests.get")
    def test_resolve_from_metadata_request_error(self, mock_get: Mock) -> None:
        """Test that request failure raises DataSourceError."""
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")

        with pytest.raises(DataSourceError, match="Failed to resolve DOI"):
            _resolve_doi_from_metadata(
                "10.5281/test", "https://api.datacite.org/dois"
            )


class TestOSNConversion:
    """Tests for _convert_osn_https_to_s3 URL conversion."""

    def test_convert_uma1_endpoint(self) -> None:
        """Test conversion of uma1 OSN path-style URL."""
        url = "https://uma1.osn.mghpcc.org/caltech-drill-core-temp/ovro-temp/test.zarr"
        opts = {"key": "ACCESS_KEY", "secret": "SECRET_KEY"}

        s3_url, updated_opts = _convert_osn_https_to_s3(url, opts)

        assert s3_url == "s3://caltech-drill-core-temp/ovro-temp/test.zarr"
        assert updated_opts["client_kwargs"]["endpoint_url"] == "https://uma1.osn.mghpcc.org"
        assert updated_opts["key"] == "ACCESS_KEY"
        assert updated_opts["secret"] == "SECRET_KEY"

    def test_convert_caltech1_endpoint(self) -> None:
        """Test conversion of caltech1 OSN URL (the bug from issue #88)."""
        url = "https://caltech1.osn.mghpcc.org/10.25800/all_subbands_2024-05-24_first10.zarr"
        opts = {"key": "ACCESS_KEY", "secret": "SECRET_KEY"}

        s3_url, updated_opts = _convert_osn_https_to_s3(url, opts)

        assert s3_url == "s3://10.25800/all_subbands_2024-05-24_first10.zarr"
        assert updated_opts["client_kwargs"]["endpoint_url"] == "https://caltech1.osn.mghpcc.org"

    def test_convert_unknown_endpoint(self) -> None:
        """Test that any *.osn.mghpcc.org URL converts correctly (path-style)."""
        url = "https://newsite99.osn.mghpcc.org/mybucket/path/to/data.zarr"
        opts = {"key": "K", "secret": "S"}

        s3_url, updated_opts = _convert_osn_https_to_s3(url, opts)

        assert s3_url == "s3://mybucket/path/to/data.zarr"
        assert updated_opts["client_kwargs"]["endpoint_url"] == "https://newsite99.osn.mghpcc.org"

    def test_non_osn_url_unchanged(self) -> None:
        """Test that non-OSN HTTPS URLs are returned unchanged."""
        url = "https://example.com/data.zarr"
        opts = {"key": "K", "secret": "S"}

        result_url, result_opts = _convert_osn_https_to_s3(url, opts)

        assert result_url == url
        assert result_opts is opts  # Same object, not modified

    def test_s3_url_unchanged(self) -> None:
        """Test that S3 URLs are returned unchanged."""
        url = "s3://bucket/data.zarr"
        opts = {"key": "K", "secret": "S"}

        result_url, result_opts = _convert_osn_https_to_s3(url, opts)

        assert result_url == url
        assert result_opts is opts

    def test_storage_options_preserved(self) -> None:
        """Test that existing storage_options keys are preserved."""
        url = "https://uma1.osn.mghpcc.org/bucket/data.zarr"
        opts = {
            "key": "ACCESS_KEY",
            "secret": "SECRET_KEY",
            "client_kwargs": {"region_name": "us-east-1"},
        }

        _, updated_opts = _convert_osn_https_to_s3(url, opts)

        assert updated_opts["key"] == "ACCESS_KEY"
        assert updated_opts["secret"] == "SECRET_KEY"
        assert updated_opts["client_kwargs"]["region_name"] == "us-east-1"
        assert updated_opts["client_kwargs"]["endpoint_url"] == "https://uma1.osn.mghpcc.org"
        # Original should not be mutated
        assert "endpoint_url" not in opts.get("client_kwargs", {})

    def test_bucket_only_no_path(self) -> None:
        """Test conversion when URL has only a bucket, no further path."""
        url = "https://uma1.osn.mghpcc.org/bucket"
        opts = {"key": "K", "secret": "S"}

        s3_url, updated_opts = _convert_osn_https_to_s3(url, opts)

        assert s3_url == "s3://bucket"
        assert updated_opts["client_kwargs"]["endpoint_url"] == "https://uma1.osn.mghpcc.org"


class TestResolveSource:
    """Tests for resolve_source() helper."""

    def test_resolve_local_path(self) -> None:
        """Test resolving a local path."""
        result = resolve_source("/path/to/data.zarr")

        assert result["source_type"] == "local"
        assert result["original_source"] == "/path/to/data.zarr"
        assert result["resolved_url"] == "/path/to/data.zarr"
        assert result["final_url"] == "/path/to/data.zarr"
        assert result["s3_url"] is None
        assert result["endpoint"] is None
        assert result["bucket"] is None
        assert result["path"] is None

    def test_resolve_remote_https(self) -> None:
        """Test resolving an HTTPS URL."""
        result = resolve_source("https://example.com/data.zarr")

        assert result["source_type"] == "remote"
        assert result["resolved_url"] == "https://example.com/data.zarr"
        assert result["final_url"] == "https://example.com/data.zarr"
        assert result["s3_url"] is None

    def test_resolve_remote_s3(self) -> None:
        """Test resolving an S3 URL."""
        result = resolve_source("s3://bucket/data.zarr")

        assert result["source_type"] == "remote"
        assert result["resolved_url"] == "s3://bucket/data.zarr"
        assert result["final_url"] == "s3://bucket/data.zarr"

    def test_resolve_osn_url_with_credentials(self) -> None:
        """Test resolving an OSN HTTPS URL with S3 credentials."""
        result = resolve_source(
            "https://caltech1.osn.mghpcc.org/mybucket/data.zarr",
            storage_options={"key": "K", "secret": "S"},
        )

        assert result["source_type"] == "remote"
        assert result["resolved_url"] == "https://caltech1.osn.mghpcc.org/mybucket/data.zarr"
        assert result["s3_url"] == "s3://mybucket/data.zarr"
        assert result["endpoint"] == "https://caltech1.osn.mghpcc.org"
        assert result["bucket"] == "mybucket"
        assert result["path"] == "data.zarr"
        assert result["final_url"] == "s3://mybucket/data.zarr"

    def test_resolve_osn_url_without_credentials(self) -> None:
        """Test resolving an OSN HTTPS URL without credentials returns HTTPS."""
        result = resolve_source(
            "https://caltech1.osn.mghpcc.org/mybucket/data.zarr"
        )

        assert result["s3_url"] is None
        assert result["final_url"] == "https://caltech1.osn.mghpcc.org/mybucket/data.zarr"

    @patch("ovro_lwa_portal.io._resolve_doi")
    def test_resolve_doi(self, mock_resolve_doi: Mock) -> None:
        """Test resolving a DOI."""
        mock_resolve_doi.return_value = "https://uma1.osn.mghpcc.org/bucket/data.zarr"

        result = resolve_source("doi:10.5281/zenodo.1234567", production=True)

        assert result["source_type"] == "doi"
        assert result["original_source"] == "doi:10.5281/zenodo.1234567"
        assert result["resolved_url"] == "https://uma1.osn.mghpcc.org/bucket/data.zarr"
        mock_resolve_doi.assert_called_once_with("10.5281/zenodo.1234567", production=True)

    @patch("ovro_lwa_portal.io._resolve_doi")
    def test_resolve_doi_with_osn_conversion(self, mock_resolve_doi: Mock) -> None:
        """Test resolving a DOI that points to an OSN URL with credentials."""
        mock_resolve_doi.return_value = (
            "https://caltech1.osn.mghpcc.org/10.25800/all_subbands.zarr"
        )

        result = resolve_source(
            "10.33569/test-doi",
            production=False,
            storage_options={"key": "K", "secret": "S"},
        )

        assert result["source_type"] == "doi"
        assert result["s3_url"] == "s3://10.25800/all_subbands.zarr"
        assert result["endpoint"] == "https://caltech1.osn.mghpcc.org"
        assert result["bucket"] == "10.25800"
        assert result["path"] == "all_subbands.zarr"

    @patch("ovro_lwa_portal.io._resolve_doi")
    def test_resolve_doi_failure(self, mock_resolve_doi: Mock) -> None:
        """Test that DOI resolution failure raises DataSourceError."""
        mock_resolve_doi.side_effect = Exception("Resolution failed")

        with pytest.raises(DataSourceError, match="Failed to resolve DOI"):
            resolve_source("doi:10.5281/zenodo.1234567")

    def test_resolve_pathlib_path(self) -> None:
        """Test resolving a pathlib.Path."""
        result = resolve_source(Path("/data/obs.zarr"))

        assert result["source_type"] == "local"
        assert result["original_source"] == "/data/obs.zarr"


class TestCheckRemoteAccess:
    """Tests for _check_remote_access pre-check."""

    def test_successful_access(self) -> None:
        """Test that successful ls() does not raise."""
        mock_fs = MagicMock()
        mock_fs.ls.return_value = [".zmetadata"]

        # Should not raise
        _check_remote_access(
            mock_fs,
            "bucket/data.zarr",
            "doi:10.1234/test",
            "s3://bucket/data.zarr",
            {"client_kwargs": {"endpoint_url": "https://endpoint.com"}},
        )

        mock_fs.ls.assert_called_once_with("bucket/data.zarr", detail=False)

    def test_no_such_bucket_error(self) -> None:
        """Test that NoSuchBucket produces an actionable error."""
        mock_fs = MagicMock()
        mock_fs.ls.side_effect = Exception("An error occurred (NoSuchBucket)")

        with pytest.raises(DataSourceError, match="Cannot access remote storage") as exc_info:
            _check_remote_access(
                mock_fs,
                "bad-bucket/data.zarr",
                "doi:10.1234/test",
                "s3://bad-bucket/data.zarr",
                {"client_kwargs": {"endpoint_url": "https://caltech1.osn.mghpcc.org"}},
            )

        error_msg = str(exc_info.value)
        assert "doi:10.1234/test" in error_msg
        assert "s3://bad-bucket/data.zarr" in error_msg
        assert "caltech1.osn.mghpcc.org" in error_msg
        assert "bad-bucket" in error_msg
        assert "Hint" in error_msg
        assert "does not exist" in error_msg

    def test_access_denied_error(self) -> None:
        """Test that AccessDenied warns but does not hard-fail pre-check."""
        mock_fs = MagicMock()
        mock_fs.ls.side_effect = Exception("AccessDenied: forbidden")

        with pytest.warns(UserWarning, match="Proceeding anyway"):
            _check_remote_access(
                mock_fs,
                "bucket/data.zarr",
                "s3://bucket/data.zarr",
                "s3://bucket/data.zarr",
                {"client_kwargs": {"endpoint_url": "https://endpoint.com"}},
            )

    def test_connection_error(self) -> None:
        """Test that connection errors produce an endpoint hint."""
        mock_fs = MagicMock()
        mock_fs.ls.side_effect = ConnectionError("Could not connect")

        with pytest.raises(DataSourceError, match="Cannot access remote storage") as exc_info:
            _check_remote_access(
                mock_fs,
                "bucket/data.zarr",
                "s3://bucket/data.zarr",
                "s3://bucket/data.zarr",
                {"client_kwargs": {"endpoint_url": "https://bad-endpoint.com"}},
            )

        error_msg = str(exc_info.value)
        assert "bad-endpoint.com" in error_msg

    def test_invalid_access_key_id_error(self) -> None:
        """Test InvalidAccessKeyId fails fast with auth hint."""
        mock_fs = MagicMock()
        mock_fs.ls.side_effect = Exception(
            "An error occurred (InvalidAccessKeyId) when calling the ListObjectsV2 operation: Unknown"
        )

        with pytest.raises(DataSourceError, match="Cannot access remote storage") as exc_info:
            _check_remote_access(
                mock_fs,
                "10.25800/all_subbands_2024-05-24_first10.zarr",
                "10.33569/4q7nb-ahq31",
                "s3://10.25800/all_subbands_2024-05-24_first10.zarr",
                {"client_kwargs": {"endpoint_url": "https://caltech1.osn.mghpcc.org"}},
            )

        error_msg = str(exc_info.value)
        assert "InvalidAccessKeyId" in error_msg
        assert "not valid for this endpoint" in error_msg
        assert "caltech1.osn.mghpcc.org" in error_msg

    def test_unknown_precheck_error_warns(self) -> None:
        """Test unknown pre-check errors warn and continue."""
        mock_fs = MagicMock()
        mock_fs.ls.side_effect = Exception("some error")

        with pytest.warns(UserWarning, match="Pre-check failed with a non-fatal error"):
            _check_remote_access(
                mock_fs,
                "bucket/path",
                "doi:10.33569/test",
                "s3://bucket/path",
                {"client_kwargs": {"endpoint_url": "https://ep.com"}},
            )

    def test_error_without_endpoint(self) -> None:
        """Test warning message when no endpoint_url is in storage_options."""
        mock_fs = MagicMock()
        mock_fs.ls.side_effect = Exception("some error")

        with pytest.warns(UserWarning, match="Cannot access remote storage"):
            _check_remote_access(
                mock_fs,
                "bucket/path",
                "s3://bucket/path",
                "s3://bucket/path",
                {},
            )


class TestOpenDatasetErrorMessages:
    """Tests for improved error messages in open_dataset."""

    @patch("ovro_lwa_portal.io._resolve_doi")
    @patch("ovro_lwa_portal.io.xr.open_zarr")
    def test_error_includes_original_doi(
        self, mock_open_zarr: Mock, mock_resolve_doi: Mock
    ) -> None:
        """Test that load errors include the original DOI source."""
        mock_resolve_doi.return_value = "https://example.com/data.zarr"
        mock_open_zarr.side_effect = Exception("zarr load failed")

        with pytest.raises(DataSourceError, match="doi:10.5281/test") as exc_info:
            open_dataset("doi:10.5281/test", validate=False)

        error_msg = str(exc_info.value)
        assert "resolved to:" in error_msg

    @patch("ovro_lwa_portal.io.xr.open_zarr")
    def test_error_includes_url(self, mock_open_zarr: Mock) -> None:
        """Test that load errors include the URL for non-DOI sources."""
        mock_open_zarr.side_effect = Exception("zarr load failed")

        with pytest.raises(DataSourceError, match="https://example.com/data.zarr"):
            open_dataset("https://example.com/data.zarr", validate=False)

    def test_s3_precheck_raises_datasource_error(self) -> None:
        """Test that S3 pre-check failure propagates as DataSourceError."""
        mock_fs = MagicMock()
        mock_fs.ls.side_effect = Exception("NoSuchBucket")

        with patch("fsspec.filesystem", return_value=mock_fs):
            with pytest.raises(DataSourceError, match="Cannot access remote storage"):
                open_dataset(
                    "s3://bad-bucket/data.zarr",
                    storage_options={"key": "K", "secret": "S"},
                    validate=False,
                )
