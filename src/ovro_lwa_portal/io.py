"""Data loading utilities for OVRO-LWA datasets.

This module provides a unified interface for loading OVRO-LWA data from
multiple sources including local paths, remote URLs, and DOI identifiers.
"""

from __future__ import annotations

import difflib
import json
import re
import warnings
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urlparse

import xarray as xr

__all__ = [
    "DataSourceError",
    "LmChunkSummary",
    "open_dataset",
    "resolve_source",
    "summarize_lm_chunks",
    "validate_local_zarr_store",
    "warn_if_suboptimal_lm_chunks",
]

# DOI pattern: matches both "doi:10.xxxx/xxxxx" and "10.xxxx/xxxxx"
DOI_PATTERN = re.compile(r"^(?:doi:)?(10\.\S+)$", re.IGNORECASE)

# DataCite API base URLs
DATACITE_API_PRODUCTION = "https://api.datacite.org/dois"
DATACITE_API_TEST = "https://api.test.datacite.org/dois"


class DataSourceError(Exception):
    """Exception raised for errors in data source handling."""

    pass


class LmChunkSummary(TypedDict):
    """On-disk ``l``/``m`` chunk layout for a Zarr data variable."""

    l_min: int
    m_min: int
    l_chunks: tuple[int, ...]
    m_chunks: tuple[int, ...]


_MIN_RECOMMENDED_LM_CHUNK = 512


def _is_doi(source: str) -> bool:
    """Check if source string is a DOI identifier.

    Parameters
    ----------
    source : str
        Source string to check.

    Returns
    -------
    bool
        True if source matches DOI pattern.
    """
    return bool(DOI_PATTERN.match(source))


def _normalize_doi(source: str) -> str:
    """Normalize DOI string by removing 'doi:' prefix if present.

    Parameters
    ----------
    source : str
        DOI string, possibly with 'doi:' prefix.

    Returns
    -------
    str
        Normalized DOI without prefix.
    """
    match = DOI_PATTERN.match(source)
    if match:
        return match.group(1)
    return source


def _resolve_doi(doi: str, production: bool = True) -> str:
    """Resolve DOI to zarr URL using DataCite Media API.

    Parameters
    ----------
    doi : str
        DOI identifier (e.g., "10.33569/9wsys-h7b71").
    production : bool, default True
        If True, use production DataCite API (api.datacite.org).
        If False, use test DataCite API (api.test.datacite.org).

    Returns
    -------
    str
        Resolved zarr URL.

    Raises
    ------
    DataSourceError
        If DOI resolution fails.
    """
    try:
        import requests
    except ImportError as e:
        msg = "requests is required for DOI resolution. Install with: pip install requests"
        raise ImportError(msg) from e

    api_base = DATACITE_API_PRODUCTION if production else DATACITE_API_TEST
    media_url = f"{api_base}/{doi}/media"

    try:
        r = requests.get(media_url, timeout=30)
        r.raise_for_status()
        data = r.json()["data"]

        # Prefer zarr media type, fallback to first URL
        zarr_media_types = {"application/zarr", "application/x-zarr"}
        for media in data:
            if media["attributes"]["mediaType"] in zarr_media_types:
                return media["attributes"]["url"]

        if data:
            return data[0]["attributes"]["url"]
    except requests.exceptions.HTTPError as e:
        # If media endpoint returns 404, fall back to DOI metadata
        if e.response.status_code == 404:
            return _resolve_doi_from_metadata(doi, api_base)
        msg = f"Failed to resolve DOI {doi}: {e}"
        raise DataSourceError(msg) from e
    except Exception as e:
        msg = f"Failed to resolve DOI {doi}: {e}"
        raise DataSourceError(msg) from e

    # No media entries found, fall back to DOI metadata
    return _resolve_doi_from_metadata(doi, api_base)


def _resolve_doi_from_metadata(doi: str, api_base: str) -> str:
    """Resolve DOI to URL from DOI metadata attributes.

    This is a fallback when the DataCite Media API endpoint returns no results
    or a 404. It queries the DOI metadata directly for a URL.

    Parameters
    ----------
    doi : str
        DOI identifier.
    api_base : str
        DataCite API base URL.

    Returns
    -------
    str
        Resolved URL.

    Raises
    ------
    DataSourceError
        If resolution fails.
    """
    import requests

    doi_url = f"{api_base}/{doi}"

    try:
        r = requests.get(doi_url, timeout=30)
        r.raise_for_status()
        doi_data = r.json()

        url = doi_data.get("data", {}).get("attributes", {}).get("url")
        if url:
            return url

        msg = f"No download URL found in metadata for DOI: {doi}"
        raise DataSourceError(msg)
    except requests.exceptions.RequestException as e:
        msg = f"Failed to resolve DOI {doi} from metadata: {e}"
        raise DataSourceError(msg) from e


def _convert_osn_https_to_s3(url: str, storage_options: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Convert OSN HTTPS URL to S3 URL when credentials are provided.

    OSN (Open Storage Network) provides both HTTPS and S3 access to the same data.
    When S3 credentials are provided, we need to use the S3 endpoint.

    Parameters
    ----------
    url : str
        HTTPS URL in path-style format:
        https://{endpoint}.osn.mghpcc.org/{bucket}/{path}
    storage_options : dict
        Storage options containing S3 credentials

    Returns
    -------
    tuple[str, dict]
        (s3_url, updated_storage_options) - S3 URL and storage options with endpoint
    """
    parsed = urlparse(url)
    if parsed.scheme == "https" and ".osn.mghpcc.org" in parsed.netloc:
        # OSN always uses path-style: https://{endpoint}.osn.mghpcc.org/{bucket}/{path}
        # The subdomain is the endpoint name (e.g., uma1, caltech1), not a bucket.
        path_parts = parsed.path.lstrip("/").split("/", 1)
        if len(path_parts) >= 1 and path_parts[0]:
            bucket = path_parts[0]
            path = path_parts[1] if len(path_parts) > 1 else ""
            endpoint = f"https://{parsed.netloc}"
        else:
            # Can't determine bucket, return unchanged
            return (url, storage_options)

        # Construct S3 URL
        s3_url = f"s3://{bucket}/{path}" if path else f"s3://{bucket}"

        # Add OSN S3 endpoint to storage_options (deep copy nested dicts)
        updated_options = storage_options.copy()
        if "client_kwargs" in updated_options:
            updated_options["client_kwargs"] = updated_options["client_kwargs"].copy()
        else:
            updated_options["client_kwargs"] = {}
        updated_options["client_kwargs"]["endpoint_url"] = endpoint

        return (s3_url, updated_options)

    return (url, storage_options)


def _detect_source_type(source: str | Path) -> tuple[str, str]:
    """Detect the type of data source.

    Parameters
    ----------
    source : str or Path
        Data source to analyze.

    Returns
    -------
    tuple[str, str]
        Tuple of (source_type, normalized_source) where source_type is one of:
        'local', 'remote', 'doi'.
    """
    source_str = str(source)

    # Check for DOI
    if _is_doi(source_str):
        normalized_doi = _normalize_doi(source_str)
        return ("doi", normalized_doi)

    # Check for remote URL
    parsed = urlparse(source_str)
    if parsed.scheme in ("http", "https", "s3", "gs", "gcs", "abfs", "az"):
        return ("remote", source_str)

    # Default to local path
    return ("local", source_str)


def _is_zarr_group_directory(path: Path) -> bool:
    """Return True if *path* looks like a Zarr v2 group or Zarr v3 store root."""
    return (path / ".zgroup").is_file() or (path / "zarr.json").is_file()


def _zarr_store_candidates_in_directory(directory: Path) -> list[Path]:
    """List child paths under *directory* that appear to be Zarr group roots."""
    if not directory.is_dir():
        return []
    candidates: list[Path] = []
    for child in sorted(directory.iterdir()):
        if child.is_dir() and _is_zarr_group_directory(child):
            candidates.append(child.resolve())
        elif child.is_file() and child.name.endswith(".zarr"):
            # Some deployments use a single-file store name; still report it.
            candidates.append(child.resolve())
    return candidates


def _format_zarr_path_suggestions(requested: Path, candidates: list[Path]) -> str:
    """Build a short hint listing likely alternate Zarr paths."""
    if not candidates:
        return ""
    names = [c.name for c in candidates]
    close_names = difflib.get_close_matches(requested.name, names, n=5, cutoff=0.4)
    if close_names:
        chosen = [c for c in candidates if c.name in close_names]
    else:
        chosen = candidates[:5]
    lines = ["Did you mean one of these Zarr stores?"]
    lines.extend(f"  {path}" for path in chosen)
    return "\n".join(lines)


def validate_local_zarr_store(source: str | Path) -> Path:
    """Validate that a local path exists and is a readable Zarr group root.

    Call this before expensive ``open_dataset`` work to fail fast on typos in
    ``ZARR_PATH`` or notebook configuration.

    Parameters
    ----------
    source : str or Path
        Path to a ``.zarr`` directory (trailing slash optional).

    Returns
    -------
    Path
        Resolved absolute path to the store.

    Raises
    ------
    FileNotFoundError
        If the path or an obvious parent is missing.
    DataSourceError
        If the path exists but is not a Zarr group (wrong directory, file, etc.).
    NotADirectoryError
        If the path exists but is a regular file.
    """
    requested = Path(source).expanduser()
    try:
        resolved = requested.resolve(strict=True)
    except FileNotFoundError:
        resolved = None

    if resolved is None or not resolved.exists():
        lines = [f"Zarr store path does not exist:\n  {requested}"]
        parent = requested.parent
        if parent.is_dir():
            hint = _format_zarr_path_suggestions(
                requested, _zarr_store_candidates_in_directory(parent)
            )
            if hint:
                lines.append(hint)
        elif parent.parent.is_dir():
            lines.append(f"Parent directory is also missing:\n  {parent}")
            hint = _format_zarr_path_suggestions(
                requested, _zarr_store_candidates_in_directory(parent.parent)
            )
            if hint:
                lines.append(f"In {parent.parent}:")
                lines.append(hint)
        msg = "\n".join(lines)
        raise FileNotFoundError(msg)

    if resolved.is_file():
        msg = (
            f"Zarr store path is a file, not a directory:\n  {resolved}\n"
            "Expected a directory ending in .zarr with a .zgroup marker inside."
        )
        raise DataSourceError(msg)

    if not resolved.is_dir():
        msg = f"Zarr store path is not a directory:\n  {resolved}"
        raise DataSourceError(msg)

    if not _is_zarr_group_directory(resolved):
        lines = [
            "Path exists but is not a Zarr group store "
            "(missing .zgroup or zarr.json at the store root):",
            f"  {resolved}",
        ]
        hint = _format_zarr_path_suggestions(
            requested, _zarr_store_candidates_in_directory(resolved.parent)
        )
        if hint:
            lines.append(hint)
        raise DataSourceError("\n".join(lines))

    return resolved


def _chunk_sizes_on_axis(chunk_size: int, axis_length: int) -> tuple[int, ...]:
    """Expand a uniform Zarr chunk length into per-chunk sizes along one axis."""
    size = int(chunk_size)
    if size <= 0:
        msg = f"chunk size must be positive, got {size}"
        raise ValueError(msg)
    n_full, remainder = divmod(int(axis_length), size)
    if remainder:
        return (size,) * n_full + (remainder,)
    return (size,) * max(n_full, 1)


def summarize_lm_chunks(
    zarr_path: str | Path,
    *,
    var: str = "SKY",
) -> LmChunkSummary:
    """Summarize on-disk ``l`` and ``m`` chunk sizes for a Zarr array.

    Reads ``{var}/.zarray`` and ``{var}/.zattrs`` from a local Zarr v2 store.
    """
    root = validate_local_zarr_store(zarr_path)
    zarray_path = root / var / ".zarray"
    zattrs_path = root / var / ".zattrs"
    if not zarray_path.is_file():
        msg = f"Variable {var!r} not found in Zarr store:\n  {root}"
        raise DataSourceError(msg)

    meta = json.loads(zarray_path.read_text(encoding="utf-8"))
    shape = tuple(int(x) for x in meta.get("shape", ()))
    chunks = meta.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        msg = f"Zarr array {var!r} is missing chunk metadata in {zarray_path}"
        raise DataSourceError(msg)

    dims: list[str] = []
    if zattrs_path.is_file():
        attrs = json.loads(zattrs_path.read_text(encoding="utf-8"))
        raw_dims = attrs.get("_ARRAY_DIMENSIONS")
        if isinstance(raw_dims, list):
            dims = [str(d) for d in raw_dims]

    if len(dims) != len(shape):
        msg = (
            f"Cannot map chunk layout for {var!r}: "
            f"shape rank {len(shape)} != _ARRAY_DIMENSIONS {dims!r}"
        )
        raise DataSourceError(msg)

    try:
        l_idx = dims.index("l")
        m_idx = dims.index("m")
    except ValueError as exc:
        msg = f"Variable {var!r} lacks l/m dimensions; found {dims!r}"
        raise DataSourceError(msg) from exc

    l_chunks = _chunk_sizes_on_axis(int(chunks[l_idx]), shape[l_idx])
    m_chunks = _chunk_sizes_on_axis(int(chunks[m_idx]), shape[m_idx])
    return {
        "l_min": min(l_chunks),
        "m_min": min(m_chunks),
        "l_chunks": l_chunks,
        "m_chunks": m_chunks,
    }


def warn_if_suboptimal_lm_chunks(
    zarr_path: str | Path,
    *,
    min_chunk: int = _MIN_RECOMMENDED_LM_CHUNK,
    var: str = "SKY",
) -> LmChunkSummary | None:
    """Warn when on-disk spatial chunks are smaller than recommended for overlay I/O."""
    try:
        summary = summarize_lm_chunks(zarr_path, var=var)
    except (DataSourceError, OSError, json.JSONDecodeError):
        return None

    l_extent = sum(summary["l_chunks"])
    m_extent = sum(summary["m_chunks"])
    smallest = min(summary["l_min"], summary["m_min"])
    max_achievable = min(l_extent, m_extent)

    if smallest >= int(min_chunk):
        return summary

    # Already chunked as large as this store geometry allows (e.g. 4×4 or 10×10 tests).
    if smallest >= max_achievable:
        return summary

    warnings.warn(
        "Zarr store "
        f"{Path(zarr_path).expanduser()} has small on-disk l/m chunks "
        f"(minimum={smallest}, recommended>={min_chunk}). "
        "Single-slice overlay reads may be slow. Re-ingest with "
        f"`ovro-ingest convert ... --chunk-lm {min_chunk}` or higher.",
        UserWarning,
        stacklevel=2,
    )
    return summary


def _validate_dataset(ds: xr.Dataset) -> None:
    """Validate that dataset conforms to OVRO-LWA data model.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to validate.

    Raises
    ------
    DataSourceError
        If dataset doesn't conform to expected structure.
    """
    # Check for required dimensions (at least some of these should exist)
    expected_dims = {"time", "frequency", "l", "m"}
    found_dims = set(ds.sizes.keys())

    if not found_dims.intersection(expected_dims):
        warnings.warn(
            f"Dataset may not be OVRO-LWA format. "
            f"Expected dimensions like {expected_dims}, found {found_dims}",
            UserWarning,
            stacklevel=2,
        )

    # Check for common OVRO-LWA data variables
    common_vars = {"SKY", "BEAM"}
    found_vars = set(ds.data_vars.keys())

    if not found_vars.intersection(common_vars):
        warnings.warn(
            f"Dataset may not be OVRO-LWA format. "
            f"Expected variables like {common_vars}, found {found_vars}",
            UserWarning,
            stacklevel=2,
        )


def _check_remote_access(
    fs: Any,
    path: str,
    original_source: str,
    normalized_source: str,
    storage_options: dict[str, Any],
) -> None:
    """Verify that a remote storage path is accessible before loading data.

    Performs a lightweight ``fs.ls()`` check to catch errors like NoSuchBucket
    or invalid credentials early, producing an actionable error message instead
    of letting them cascade through zarr/xarray into opaque tracebacks.

    Parameters
    ----------
    fs : fsspec filesystem
        The configured filesystem instance.
    path : str
        The path to check (e.g., "bucket/prefix/data.zarr").
    original_source : str
        The source string as originally provided by the user.
    normalized_source : str
        The URL after DOI resolution and OSN conversion.
    storage_options : dict
        Storage options (used to extract endpoint for error messages).
    """
    try:
        fs.ls(path, detail=False)
        return
    except Exception as e:
        error_name = type(e).__name__
        error_text = str(e)
        endpoint = storage_options.get("client_kwargs", {}).get("endpoint_url", "")

        parts = [
            f"Cannot access remote storage for '{original_source}'",
        ]
        if normalized_source != original_source:
            parts.append(f"Resolved URL: {normalized_source}")
        if endpoint:
            parts.append(f"S3 endpoint: {endpoint}")
        parts.append(f"Storage path: {path}")
        parts.append(f"Error ({error_name}): {error_text}")

        # Fatal auth mismatch: the key is not valid for this endpoint.
        if "InvalidAccessKeyId" in error_text:
            parts.append(
                "Hint: The provided access key is not valid for this endpoint. "
                "Check that your credentials match the resolved OSN endpoint."
            )
            msg = "\n  ".join(parts)
            raise DataSourceError(msg) from e

        # Least-privilege credentials may deny list operations but still allow direct reads.
        # Don't block loading in this case; continue and let open_zarr perform the real read.
        if "AccessDenied" in error_text or "Forbidden" in error_text:
            details = parts + [
                "Hint: List access is denied for this prefix. "
                "Proceeding anyway because direct object reads may still succeed.",
            ]
            warnings.warn("\n  ".join(details), UserWarning, stacklevel=2)
            return

        # Fail fast for clearly fatal remote-access errors.
        if "NoSuchBucket" in error_text or "NoSuchBucket" in error_name:
            bucket = path.split("/")[0] if "/" in path else path
            parts.append(
                f"Hint: The bucket '{bucket}' does not exist at this endpoint. "
                f"Check the URL or endpoint configuration."
            )
            msg = "\n  ".join(parts)
            raise DataSourceError(msg) from e

        if "EndpointConnectionError" in error_text or "ConnectionError" in error_name:
            parts.append(
                f"Hint: Could not connect to endpoint '{endpoint}'. "
                f"Check the endpoint URL and your network connection."
            )
            msg = "\n  ".join(parts)
            raise DataSourceError(msg) from e

        # Unknown pre-check error: keep behavior non-blocking and let actual read decide.
        details = parts + [
            "Hint: Pre-check failed with a non-fatal error. "
            "Continuing to attempt dataset load.",
        ]
        warnings.warn("\n  ".join(details), UserWarning, stacklevel=2)
        return


def resolve_source(
    source: str | Path,
    production: bool = True,
    storage_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a data source to its final URL without loading data.

    Performs DOI resolution and URL normalization, returning the full
    resolution chain. Useful for debugging DOI→URL→S3 resolution
    without actually loading any data.

    Parameters
    ----------
    source : str or Path
        Data source, can be:
        - Local file path (e.g., "/path/to/data.zarr")
        - Remote URL (e.g., "s3://bucket/data.zarr", "https://...")
        - DOI string (e.g., "doi:10.xxxx/xxxxx" or "10.xxxx/xxxxx")
    production : bool, default True
        Which DataCite API to use when resolving DOI identifiers.
    storage_options : dict, optional
        Options passed to the filesystem backend (e.g., S3 credentials).
        Used to determine if OSN HTTPS→S3 conversion should be applied.

    Returns
    -------
    dict[str, Any]
        Resolution details with keys:
        - ``source_type``: "local", "remote", or "doi"
        - ``original_source``: the source string as provided
        - ``resolved_url``: URL after DOI resolution (or original if not a DOI)
        - ``final_url``: final URL after OSN conversion (if applicable)
        - ``s3_url``: S3 URL if OSN conversion was applied, else None
        - ``endpoint``: S3 endpoint URL if applicable, else None
        - ``bucket``: S3 bucket name if applicable, else None
        - ``path``: path within bucket if applicable, else None

    Raises
    ------
    DataSourceError
        If DOI resolution fails.

    Examples
    --------
    Resolve a DOI to see the full chain:

    >>> from ovro_lwa_portal import resolve_source
    >>> info = resolve_source("10.33569/9wsys-h7b71", production=False)
    >>> info["source_type"]
    'doi'
    >>> info["resolved_url"]
    'https://caltech1.osn.mghpcc.org/...'

    Resolve with S3 credentials to see OSN conversion:

    >>> info = resolve_source(
    ...     "10.33569/9wsys-h7b71",
    ...     production=False,
    ...     storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"},
    ... )
    >>> info["s3_url"]
    's3://...'
    """
    original_source = str(source)
    source_type, normalized = _detect_source_type(source)

    result: dict[str, Any] = {
        "source_type": source_type,
        "original_source": original_source,
        "resolved_url": None,
        "final_url": None,
        "s3_url": None,
        "endpoint": None,
        "bucket": None,
        "path": None,
    }

    # Resolve DOI to actual data URL
    if source_type == "doi":
        try:
            resolved = _resolve_doi(normalized, production=production)
        except Exception as e:
            msg = f"Failed to resolve DOI {normalized}: {e}"
            raise DataSourceError(msg) from e
        result["resolved_url"] = resolved
    else:
        result["resolved_url"] = normalized
        resolved = normalized

    # Check for OSN HTTPS→S3 conversion
    if storage_options and source_type in ("doi", "remote"):
        converted_url, converted_opts = _convert_osn_https_to_s3(
            resolved, storage_options
        )
        if converted_url != resolved:
            # OSN conversion was applied
            result["s3_url"] = converted_url
            result["endpoint"] = converted_opts.get("client_kwargs", {}).get(
                "endpoint_url"
            )
            # Parse bucket and path from the S3 URL
            parsed_s3 = urlparse(converted_url)
            result["bucket"] = parsed_s3.netloc
            result["path"] = parsed_s3.path.lstrip("/") or None
            result["final_url"] = converted_url
        else:
            result["final_url"] = resolved
    else:
        result["final_url"] = resolved

    return result


def open_dataset(
    source: str | Path,
    chunks: dict[str, int] | str | None = "auto",
    production: bool = True,
    storage_options: dict[str, Any] | None = None,
    engine: str = "zarr",
    validate: bool = True,
    **kwargs: Any,
) -> xr.Dataset:
    """Load OVRO-LWA data as an xarray Dataset.

    This function provides a unified interface for loading OVRO-LWA data from
    multiple sources including local file paths, remote URLs, and DOI identifiers.

    Parameters
    ----------
    source : str or Path
        Data source, can be:
        - Local file path (e.g., "/path/to/data.zarr")
        - Remote URL (e.g., "s3://bucket/data.zarr", "https://...")
        - DOI string (e.g., "doi:10.xxxx/xxxxx" or "10.xxxx/xxxxx")
    chunks : dict, str, or None, default "auto"
        Chunking strategy for lazy loading:
        - dict: Explicit chunk sizes per dimension, e.g., {"time": 100, "frequency": 50}
        - "auto": Let xarray/dask determine optimal chunks
        - None: Load entire dataset into memory (not recommended for large data)
    production : bool, default True
        Which DataCite API to use when resolving DOI identifiers:
        - True: production API (api.datacite.org)
        - False: test API (api.test.datacite.org)
    storage_options : dict, optional
        Options passed to the filesystem backend (e.g., S3 credentials).
        Example: storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"}
    engine : str, default "zarr"
        Backend engine for loading data. Currently supports "zarr".
    validate : bool, default True
        If True, validate that loaded data conforms to OVRO-LWA data model.
    **kwargs
        Additional arguments passed to the underlying loader (xr.open_zarr, etc.).
        For very large local Zarr stores, ``consolidated=False`` avoids slow
        metadata walks when ``.zmetadata`` is missing or incomplete.

    Returns
    -------
    xr.Dataset
        OVRO-LWA dataset with standardized structure.

    Raises
    ------
    DataSourceError
        If source cannot be accessed or loaded.
    FileNotFoundError
        If local file path doesn't exist.
    ImportError
        If required dependencies for remote/DOI access are not installed.

    Examples
    --------
    Load from local zarr store:

    >>> import ovro_lwa_portal
    >>> ds = ovro_lwa_portal.open_dataset("/path/to/observation.zarr")

    Load from S3 bucket:

    >>> ds = ovro_lwa_portal.open_dataset("s3://ovro-lwa-data/obs_12345.zarr")

    Load from HTTP/HTTPS URL:

    >>> ds = ovro_lwa_portal.open_dataset("https://data.ovro.caltech.edu/obs_12345.zarr")

    Load via DOI (with or without prefix):

    >>> ds = ovro_lwa_portal.open_dataset("doi:10.5281/zenodo.1234567")
    >>> ds = ovro_lwa_portal.open_dataset("10.5281/zenodo.1234567")

    Load from test DataCite API with S3 credentials:

    >>> ds = ovro_lwa_portal.open_dataset(
    ...     "10.33569/4q7nb-ahq31",
    ...     production=False,
    ...     storage_options={"key": "ACCESS_KEY", "secret": "SECRET_KEY"}
    ... )

    Customize chunking:

    >>> ds = ovro_lwa_portal.open_dataset(
    ...     "path/to/data.zarr",
    ...     chunks={"time": 100, "frequency": 50}
    ... )

    Notes
    -----
    For remote data sources (S3, GCS), authentication is handled via environment
    variables or configuration files specific to each cloud provider:

    - AWS S3: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, or ~/.aws/credentials
    - Google Cloud Storage: GOOGLE_APPLICATION_CREDENTIALS
    - Azure: AZURE_STORAGE_ACCOUNT_NAME, AZURE_STORAGE_ACCOUNT_KEY

    For large datasets, lazy loading with dask is used by default (chunks="auto").
    This allows working with datasets larger than memory.
    """
    original_source = str(source)
    source_type, normalized_source = _detect_source_type(source)
    resolved_url: str | None = None  # Track DOI-resolved URL for error messages
    local_zarr_path: Path | None = None

    # Resolve DOI to actual data URL
    if source_type == "doi":
        try:
            normalized_source = _resolve_doi(normalized_source, production=production)
            resolved_url = normalized_source
            source_type = "remote"  # After resolution, treat as remote URL
        except Exception as e:
            msg = f"Failed to resolve DOI {normalized_source}: {e}"
            raise DataSourceError(msg) from e

    # Convert OSN HTTPS URLs to S3 when credentials are provided
    # OSN provides both HTTPS and S3 access to the same data
    if source_type == "remote" and storage_options:
        normalized_source, storage_options = _convert_osn_https_to_s3(
            normalized_source, storage_options
        )

    # Load data based on engine
    try:
        if engine == "zarr":
            # Use fsspec's universal pathlib for unified handling of local and remote paths
            try:
                from upath import UPath
            except ImportError as e:
                msg = (
                    "universal-pathlib is required for path handling. "
                    "Install with: pip install universal-pathlib"
                )
                raise ImportError(msg) from e

            # Create filesystem and mapper
            # When storage_options are provided, use fsspec directly for cloud storage
            parsed_url = urlparse(normalized_source)
            protocol = parsed_url.scheme if parsed_url.scheme else "file"

            if storage_options and protocol in ("s3", "gs", "gcs", "abfs", "az"):
                # Use fsspec directly for cloud storage with credentials
                try:
                    import fsspec
                except ImportError as e:
                    msg = (
                        "fsspec is required for remote storage access. "
                        "Install with: pip install fsspec"
                    )
                    raise ImportError(msg) from e

                # Create filesystem with storage options
                fs = fsspec.filesystem(protocol, **storage_options)

                # Get path without protocol (e.g., s3://bucket/path -> bucket/path)
                path = f"{parsed_url.netloc}/{parsed_url.path.lstrip('/')}"
                store = fs.get_mapper(path)

                # Early accessibility check for cloud storage
                _check_remote_access(
                    fs, path, original_source, normalized_source, storage_options
                )
            else:
                # For local files or HTTPS, use UPath without storage_options
                store_path = UPath(normalized_source)

                if store_path.protocol in ("", "file"):
                    local_store = validate_local_zarr_store(normalized_source)
                    local_zarr_path = local_store
                    store_path = UPath(str(local_store))

                # Build a Zarr store (fsspec mapper) from the UPath
                fs = store_path.fs
                store = fs.get_mapper(store_path.path)

            # Check if we need cloud storage backends for remote paths
            if source_type == "remote":
                parsed = urlparse(normalized_source)
                if parsed.scheme == "s3":
                    try:
                        import s3fs  # noqa: F401
                    except ImportError as e:
                        msg = (
                            "s3fs is required for S3 access. "
                            "Install with: pip install s3fs"
                        )
                        raise ImportError(msg) from e
                elif parsed.scheme in ("gs", "gcs"):
                    try:
                        import gcsfs  # noqa: F401
                    except ImportError as e:
                        msg = (
                            "gcsfs is required for Google Cloud Storage access. "
                            "Install with: pip install gcsfs"
                        )
                        raise ImportError(msg) from e

            # Open the zarr store using the UPath
            # xr.open_zarr can handle fsspec mappers directly
            with warnings.catch_warnings():
                warnings.simplefilter("default")
                ds = xr.open_zarr(store, chunks=chunks, **kwargs)

        else:
            msg = f"Unsupported engine: {engine}. Currently only 'zarr' is supported."
            raise DataSourceError(msg)

    except FileNotFoundError:
        raise
    except ImportError:
        raise
    except DataSourceError:
        raise
    except Exception as e:
        # Build a detailed error message including the resolution chain
        parts = [f"Failed to load dataset from '{original_source}'"]
        if resolved_url and resolved_url != original_source:
            parts.append(f"resolved to: {resolved_url}")
        if normalized_source != original_source and normalized_source != resolved_url:
            parts.append(f"final URL: {normalized_source}")
        if storage_options and "client_kwargs" in storage_options:
            endpoint = storage_options["client_kwargs"].get("endpoint_url")
            if endpoint:
                parts.append(f"S3 endpoint: {endpoint}")
        parts.append(str(e))
        msg = "\n  ".join(parts)
        raise DataSourceError(msg) from e

    # Validate dataset structure if requested
    if validate:
        _validate_dataset(ds)

    if local_zarr_path is not None:
        warn_if_suboptimal_lm_chunks(local_zarr_path)

    from ovro_lwa_portal.accessor import strip_redundant_fits_wcs_header_attrs

    return strip_redundant_fits_wcs_header_attrs(ds)
