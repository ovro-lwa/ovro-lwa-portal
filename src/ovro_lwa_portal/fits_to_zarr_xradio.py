"""xradio-powered FITS → Zarr conversion for OVRO-LWA.

This module converts per-time, per-subb and FITS images into a single LM-only Zarr store.
It uses `xradio` for FITS I/O (via Astropy-based helpers, not CASA-first ``read_image``)
and Zarr writing, and enforces deterministic ordering by
sorting on frequency and time. It also materializes FITS scaling (BSCALE/BZERO) and adds
a minimal set of header keywords so `xradio` can parse images reliably.

Library usage
-------------
    # Option 1: Fix headers on-demand during conversion (default behavior)
    from ovro_lwa_portal.ingest.fit_to_zarr_xradio import convert_fits_dir_to_zarr
    out = convert_fits_dir_to_zarr(
        input_dir="/path/to/fits",
        out_dir="zarr_out",
        zarr_name="ovro_lwa_full_lm_only.zarr",
        fixed_dir="fixed_fits",
        chunk_lm=1024,
        rebuild=False,
    )

    # Option 2: Fix headers ahead of time, then convert
    from pathlib import Path
    from ovro_lwa_portal.ingest.fit_to_zarr_xradio import fix_fits_headers, convert_fits_dir_to_zarr

    # Step 1: Fix all headers first
    input_files = list(Path("/path/to/fits").glob("*.fits"))
    fixed_dir = Path("fixed_fits")
    fix_fits_headers(input_files, fixed_dir)

    # Step 2: Convert using pre-fixed headers
    out = convert_fits_dir_to_zarr(
        input_dir="/path/to/fits",
        out_dir="zarr_out",
        zarr_name="ovro_lwa_full_lm_only.zarr",
        fixed_dir="fixed_fits",
        chunk_lm=1024,
        rebuild=False,
        fix_headers_on_demand=False,  # Skip fixing since already done
    )

Notes
-----
* Discovery groups files by observation time and by a **23~kHz binned** frequency key
  (from FITS headers when ``group_metadata_source`` is ``"fits"``, or from basename
  ``_NNNMHz_`` when ``group_metadata_source`` is ``"filename"``), so Hz-level jitter does
  not create extra ``frequency`` planes for one subband. Multiple files in the same
  (time, bin) without a duplicate resolver keep the first and skip the rest (with a warning).
  With ``"fits"``, filename parsing is a fallback when header metadata is missing.
* LM grids must match across time steps after global and per-step mixed-resolution normalization;
  a mismatch raises a RuntimeError.
* Within a single time step, mixed LM shapes are regridded onto the reference grid before combine.
  The reference contributes only the *pixel grid* (``CRPIX``/``CDELT``/projection); per-time
  ``CRVAL1``/``CRVAL2`` (FK5 zenith stamped by :func:`_fix_headers`) is taken from each source
  subband so all subbands at one time step share identical ``right_ascension``/``declination``.
* After stacking subbands along ``frequency``, ``right_ascension`` / ``declination`` are
  collapsed to a single ``(l, m)`` frame taken from the lowest-frequency slice. If sampled
  sky positions differ from that reference by more than ~one arcminute between slices, a
  warning is logged so any residual per-subband WCS inconsistency stays visible.
* Single-channel slices get a ``frequency`` coordinate from the basename ``_NNNMHz_`` token
  when present so dewarped products that share an identical spectral keyword in the FITS
  header still stack with unique labels (avoids pandas duplicate-index errors on ``sortby``).
* Before Zarr write, ``l``/``m`` are rechunked to uniform sizes so the store does not hit
  Dask/Zarr constraints on irregular spatial chunk boundaries after ``combine``/``concat``.
* On append, new time steps are written with ``xarray.Dataset.to_zarr(..., append_dim="time")``
  so the store can grow far larger than RAM without re-reading or re-writing prior times.
"""

from __future__ import annotations

import os
import logging
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
import zarr
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, FK5, SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from numpy.typing import NDArray

__all__ = [
    "InvalidBeamError",
    "convert_fits_dir_to_zarr",
    "fix_fits_headers",
    "repair_zarr_store",
    "validate_zarr_store",
]

logger = logging.getLogger(__name__)


class InvalidBeamError(ValueError):
    """Raised when a FITS primary header lacks a usable synthesized beam.

    The ingest pipeline refuses to invent a placeholder beam: images without a real
    ``BMAJ``/``BMIN`` (missing, zero, or non-positive) are excluded from processing.
    :func:`_filter_invalid_beam_files` drops them at discovery time;
    :func:`_fix_headers` raises this error if invoked directly on such a file and
    :func:`fix_fits_headers` catches it to skip the file with a warning.
    """


def _is_xradio_stokes_missing_valueerror(exc: BaseException) -> bool:
    """True when xradio failed building coords because ``CTYPE`` has no ``STOKES``."""
    if not isinstance(exc, ValueError):
        return False
    msg = str(exc).lower()
    return "stokes" in msg and "not in" in msg and "list" in msg


def _read_fits_via_xradio(
    path: str | Path,
    *,
    chunks: Optional[Dict] = None,
    verbose: bool = False,
    do_sky_coords: bool = False,
    compute_mask: bool = False,
) -> xr.Dataset:
    """Load a FITS image as an xradio-style :class:`xarray.Dataset` (FITS-only).

    :func:`xradio.image.read_image` tries a casacore/CASA image read on every path
    before falling back to FITS. OVRO-LWA ingest is FITS-only, so we call xradio's
    Astropy FITS helper directly to skip that probe and any associated stderr/log
    noise.

    Notes
    -----
    Uses the private ``_fits_image_to_xds`` from xradio; it may move between
    xradio releases—keep the dependency pin exercised in CI.

    xradio's FITS reader assumes a literal ``STOKES`` axis in ``CTYPE`` when it
    builds polarization coordinates. Legitimate 3D RA/DEC/FREQ cubes therefore
    raise ``ValueError: 'STOKES' is not in list``. On that specific failure we
    materialize a temporary copy with :func:`_fix_headers` and read again so raw
    paths and stale ``*_fixed.fits`` still work without a separate fix step.

    Pixel data are Dask-delayed reads of that temp path; we :meth:`xarray.Dataset.load`
    before unlinking so later ``compute()`` does not hit a missing file.
    """
    from xradio.image._util._fits.xds_from_fits import _fits_image_to_xds

    c: Dict = {} if chunks is None else chunks
    p = Path(os.path.expanduser(str(path)))
    try:
        return _fits_image_to_xds(str(p), c, verbose, do_sky_coords, compute_mask)
    except ValueError as exc:
        if not _is_xradio_stokes_missing_valueerror(exc):
            raise
        logger.info(
            "Re-reading FITS via temporary header fix copy: %s (%s)",
            p.name,
            exc,
        )
        tmp_dir: Optional[str] = None
        try:
            if p.parent.is_dir() and os.access(p.parent, os.W_OK):
                tmp_dir = str(p.parent)
        except OSError:
            tmp_dir = None
        fd, tmp_name = tempfile.mkstemp(
            suffix=".fits", prefix="ovro_xradio_stokes_", dir=tmp_dir
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            _fix_headers(p, tmp_path)
            xds = _fits_image_to_xds(str(tmp_path), c, verbose, do_sky_coords, compute_mask)
            # xradio wraps FITS pixels in dask.delayed reads that still reference
            # ``tmp_path``; unlinking in ``finally`` before compute would raise ENOENT.
            xds.load()
            return xds
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove temporary FITS %s", tmp_path)


# Grouping key for "same subband" when discovering FITS. Raw ``int(round(hz))`` can differ
# by 1--1000+ Hz between files for the same 55~MHz (etc.) product; that produced multiple
# ``frequency`` planes in the Zarr from one logical band. Bins of 23~kHz merge that jitter
# while keeping distinct LWA subbands (MHz-scale) separate.
_DISCOVERY_FREQ_BIN_HZ: float = 23_000.0

# Log a warning when any stacked frequency slice disagrees with the reference slice by
# more than this on-sky separation (sampled on the LM grid). Used after ``combine`` so
# per-channel WCS drift across a wideband time step is visible before collapsing coords.
_CELESTIAL_FRAME_WARN_MAX_SKY_SEP_ARCSEC: float = 60.0
_CELESTIAL_DRIFT_SAMPLE_MAX_POINTS: int = 65536
_CELESTIAL_DRIFT_SAMPLE_SEED: int = 0

# Default OVRO-LWA / OVRO geodetic site when FITS lacks ``OBSGEO-*`` (matches
# ``EarthLocation.of_site("ovro")`` in Astropy data; avoids network at import).
_OVRO_LWA_DEFAULT_LON_DEG = -118.28340511
_OVRO_LWA_DEFAULT_LAT_DEG = 37.23338698
_OVRO_LWA_DEFAULT_HEIGHT_M = 1188.6

# Cache expensive sky-coordinate transforms keyed by celestial WCS header + LM shape.
# Many OVRO-LWA subbands at one time share the same LM grid, so recomputing
# all_pix2world for every file dominates runtime.
_SKY_COORD_CACHE: Dict[Tuple[int, int, str], Tuple[NDArray[np.floating], NDArray[np.floating], str]] = {}


# Subband MHz in OVRO-style basenames, e.g. ``..._41MHz_...``, ``...__18MHz-I-...``
# (dewarp staging uses ``{time_key}__{original_name}``; product tags use a hyphen after MHz).
MHZ_RE = re.compile(r"_(\d+)MHz(?:_|-|\.|$)")

# OVRO-LWA ``...-image-YYYYMMDD_HHMMSS...`` segment (UTC wall-clock for the map).
_IMAGE_TIME_RE = re.compile(r"-image-(\d{8})_(\d{6})")


def _mhz_from_name(p: Path) -> int:
    """Extract the subband MHz from a filename; return a large sentinel if absent.

    Parameters
    ----------
    p : Path
        Path object with filename to extract MHz from.

    Returns
    -------
    int
        Subband frequency in MHz, or 10**9 if not found.
    """
    m = MHZ_RE.search(p.name)
    return int(m.group(1)) if m else 10**9


def _time_key_from_header(header: fits.Header) -> Optional[str]:
    """Extract observation time from FITS headers as ``YYYYMMDD_HHMMSS``.

    This project requires ``DATE-OBS`` to be present and parseable.
    ``TIME-OBS`` is used only when ``DATE-OBS`` is date-only (no ``T``).

    Returns ``None`` when no usable ``DATE-OBS`` timestamp is found.
    """
    date_obs = header.get("DATE-OBS")
    if date_obs:
        date_obs = str(date_obs).strip()
        time_obs = header.get("TIME-OBS")
        if time_obs and "T" not in date_obs:
            dt_value = f"{date_obs}T{str(time_obs).strip()}"
        else:
            dt_value = date_obs
        try:
            t = Time(dt_value, format="isot", scale="utc")
            return t.to_datetime().strftime("%Y%m%d_%H%M%S")
        except Exception:
            logger.debug(f"Could not parse DATE-OBS/TIME-OBS timestamp: {dt_value}")

    return None


def _frequency_hz_from_header(header: fits.Header) -> Optional[float]:
    """Extract frequency in Hz from FITS headers.

    Header precedence:
      1) ``RESTFREQ``
      2) ``RESTFRQ``
      3) ``CRVAL3``
      4) ``FREQ``
    """
    for key in ("RESTFREQ", "RESTFRQ", "CRVAL3", "FREQ"):
        value = header.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            logger.debug(f"Could not parse {key} frequency: {value}")
    return None


def _extract_group_metadata(
    fp: Path,
    *,
    time_key_source: Literal["header", "filename"] = "filename",
) -> Tuple[Optional[str], Optional[float], List[str]]:
    """Extract grouping metadata from FITS headers and optionally the basename.

    Returns
    -------
    Tuple[Optional[str], Optional[float], List[str]]
        Tuple of ``(time_key, frequency_hz, fallback_notes)``.

        ``frequency_hz`` is read from headers when possible with a filename fallback
        for frequency only.

        When *time_key_source* is ``"filename"`` (default), the basename pattern
        ``-image-YYYYMMDD_HHMMSS`` is used when present (see :func:`_time_key_from_filename`);
        otherwise ``time_key`` comes from ``DATE-OBS`` (see :func:`_time_key_from_header`).

        When *time_key_source* is ``"header"``, ``time_key`` comes from ``DATE-OBS`` only.
    """
    time_key: Optional[str] = None
    frequency_hz: Optional[float] = None
    notes: List[str] = []

    header: Optional[fits.Header] = None
    try:
        header = fits.getheader(fp, ext=0)
    except Exception as e:
        logger.warning(f"Could not read FITS header for {fp.name}: {e}")

    if header is not None:
        frequency_hz = _frequency_hz_from_header(header)
        if time_key_source == "header":
            time_key = _time_key_from_header(header)

    if time_key_source == "filename":
        tk_fn = _time_key_from_filename(fp)
        if tk_fn is not None:
            time_key = tk_fn
            notes.append("time-from-filename")
        elif header is not None and time_key is None:
            time_key = _time_key_from_header(header)

    if frequency_hz is None:
        mhz = _mhz_from_name(fp)
        if mhz != 10**9:
            frequency_hz = float(mhz * 1e6)
            notes.append("frequency-from-filename")

    return time_key, frequency_hz, notes


def _extract_group_metadata_filename_only(fp: Path) -> Tuple[Optional[str], Optional[float], List[str]]:
    """Time and frequency for discovery using only the basename (no FITS I/O).

    Observation time comes from :func:`_time_key_from_filename` (``-image-YYYYMMDD_HHMMSS``).
    Subband frequency is ``int(MHz) * 1e6`` from :func:`_mhz_from_name` when the
    ``_NNNMHz_`` / ``_NNNMHz-`` pattern is present.

    Returns
    -------
    Tuple[Optional[str], Optional[float], List[str]]
        ``(time_key, frequency_hz, notes)``. ``frequency_hz`` is ``None`` when no MHz
        token appears in the filename (same sentinel behavior as :func:`_mhz_from_name`).
    """
    notes: List[str] = []
    time_key = _time_key_from_filename(fp)
    mhz = _mhz_from_name(fp)
    frequency_hz: Optional[float] = None
    if mhz != 10**9:
        frequency_hz = float(mhz * 1_000_000)
    else:
        notes.append("frequency-from-filename-missing")
    return time_key, frequency_hz, notes


def _discovery_frequency_sort_tuple(
    fp: Path, *, group_metadata_source: Literal["fits", "filename"]
) -> Tuple[float, str]:
    """Sort key for deterministic frequency ordering (header+filename vs filename-only)."""
    if group_metadata_source == "filename":
        _, frequency_hz, _ = _extract_group_metadata_filename_only(fp)
    else:
        _, frequency_hz, _ = _extract_group_metadata(fp)
    if frequency_hz is None:
        return (float(10**15), fp.name)
    return (float(frequency_hz), fp.name)


def _canonical_stack_frequency_hz(
    fp: Path,
    *,
    group_metadata_source: Literal["fits", "filename"],
    time_key_source: Literal["header", "filename"] = "filename",
) -> Optional[float]:
    """Hz label to use when stacking single-channel slices along ``frequency``.

    Dewarped FITS often carry an identical spectral reference keyword in every
    product while the OVRO basename still encodes the true subband via
    ``_NNNMHz_``. :func:`xarray.combine_by_coords` and :meth:`xarray.Dataset.sortby`
    require unique ``frequency`` coordinates, so we prefer the MHz token when
    present, then fall back to the same discovery metadata used for ordering.
    """
    mhz = _mhz_from_name(fp)
    if mhz != 10**9:
        return float(mhz * 1_000_000)
    if group_metadata_source == "filename":
        _, hz, _ = _extract_group_metadata_filename_only(fp)
    else:
        _, hz, _ = _extract_group_metadata(fp, time_key_source=time_key_source)
    return hz


def _assign_canonical_frequency_for_stack(
    xds: xr.Dataset,
    fp: Path,
    *,
    group_metadata_source: Literal["fits", "filename"],
    time_key_source: Literal["header", "filename"] = "filename",
) -> xr.Dataset:
    """Replace a length-1 ``frequency`` index with :func:`_canonical_stack_frequency_hz` when known."""
    if "frequency" not in xds.dims or int(xds.sizes.get("frequency", 0)) != 1:
        return xds
    hz = _canonical_stack_frequency_hz(
        fp,
        group_metadata_source=group_metadata_source,
        time_key_source=time_key_source,
    )
    if hz is None:
        return xds
    return xds.assign_coords(
        frequency=xr.DataArray(np.asarray([hz], dtype=np.float64), dims=("frequency",))
    )


def _global_frequency_coord_hz(
    by_time: Dict[str, List[Path]],
    *,
    group_metadata_source: Literal["fits", "filename"],
) -> np.ndarray:
    """Union of canonical per-file Hz labels across all time groups (sorted ascending)."""
    hz_vals: set[float] = set()
    for files in by_time.values():
        for fp in files:
            hz = _canonical_stack_frequency_hz(
                fp,
                group_metadata_source=group_metadata_source,
                time_key_source="filename",
            )
            if hz is not None:
                hz_vals.add(float(hz))
    if not hz_vals:
        msg = (
            "Cannot determine a global frequency axis: no canonical Hz metadata on input FITS. "
            "Ensure `_NNNMHz_` basename tags or FITS spectral headers are present."
        )
        raise RuntimeError(msg)
    return np.array(sorted(hz_vals), dtype=np.float64)


def _frequency_coord_hz_from_zarr(out_zarr: Path) -> np.ndarray:
    """Read the store's ``frequency`` coordinate (must match appended time steps)."""
    with xr.open_zarr(str(out_zarr), consolidated=False) as ds:
        if "frequency" not in ds.coords:
            raise RuntimeError(
                f"Existing Zarr {out_zarr} has no frequency coordinate; cannot append incrementally."
            )
        return np.asarray(ds["frequency"].values, dtype=np.float64).copy()


def _align_time_step_to_frequency_grid(ds: xr.Dataset, freq_hz: np.ndarray) -> xr.Dataset:
    """Reindex one time step onto the fixed full-store ``frequency`` axis (NaN for missing subbands)."""
    if "frequency" not in ds.dims:
        return ds
    template = xr.DataArray(
        np.asarray(freq_hz, dtype=np.float64),
        dims=("frequency",),
        name="frequency",
    )
    return ds.reindex(frequency=template, fill_value=np.nan)


def _lm_reference_from_existing_zarr(out_zarr: Path) -> xr.Dataset:
    """Minimal LM + WCS reference from an on-disk Zarr (resume / append without re-scanning FITS)."""
    with xr.open_zarr(str(out_zarr), consolidated=False) as ex:
        if "l" not in ex.coords or "m" not in ex.coords:
            raise RuntimeError(f"Existing Zarr {out_zarr} is missing l/m coordinates; cannot resume.")
        hdr = _read_wcs_header_str(ex)
        if hdr is None:
            probe = ex
            for dim in ("time", "frequency"):
                if dim in probe.dims:
                    probe = probe.isel({dim: 0})
            for v in probe.data_vars:
                hdr = _read_wcs_header_str(probe[[v]])
                if hdr is not None:
                    break
        l_coord = np.asarray(ex["l"].values, dtype=np.float64)
        m_coord = np.asarray(ex["m"].values, dtype=np.float64)
    out = xr.Dataset(coords={"l": ("l", l_coord), "m": ("m", m_coord)})
    if hdr is not None:
        out.attrs["fits_wcs_header"] = hdr
    return out


def _normalize_time_key(value: object) -> Optional[str]:
    """Normalize mixed time representations to ``YYYYMMDD_HHMMSS`` in UTC.

    This helper is used to compare discovery keys against time values loaded
    from an existing Zarr store.
    """
    if value is None:
        return None

    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return None
        try:
            return Time(float(value), format="mjd", scale="utc").to_datetime().strftime("%Y%m%d_%H%M%S")
        except Exception:
            return None

    if isinstance(value, (bytes, np.bytes_)):
        text_value = value.decode("utf-8").strip()
    elif isinstance(value, str):
        text_value = value.strip()
    elif isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        dt64_s = value.astype("datetime64[s]")
        iso = np.datetime_as_string(dt64_s, unit="s", timezone="UTC")
        try:
            return Time(iso, format="isot", scale="utc").to_datetime().strftime("%Y%m%d_%H%M%S")
        except Exception:
            return None
    else:
        text_value = str(value).strip()

    if not text_value:
        return None

    if re.match(r"^\d{8}_\d{6}$", text_value):
        return text_value

    for fmt in ("isot", "fits"):
        try:
            return Time(text_value, format=fmt, scale="utc").to_datetime().strftime("%Y%m%d_%H%M%S")
        except Exception:
            continue

    try:
        dt64 = np.datetime64(text_value)
        if np.isnat(dt64):
            return None
        dt64_s = dt64.astype("datetime64[s]")
        iso = np.datetime_as_string(dt64_s, unit="s", timezone="UTC")
        return Time(iso, format="isot", scale="utc").to_datetime().strftime("%Y%m%d_%H%M%S")
    except Exception:
        return None


def _existing_time_keys_from_zarr(out_zarr: Path) -> set[str]:
    """Read and normalize timestep keys from an existing Zarr store."""
    try:
        xds = xr.open_zarr(str(out_zarr), consolidated=False)
    except Exception as exc:
        try:
            zg = zarr.open_group(str(out_zarr), mode="r")
            time_arr = zg["time"][:]
            fallback_keys: set[str] = set()
            for raw_value in np.atleast_1d(time_arr):
                key = _normalize_time_key(raw_value)
                if key is None:
                    msg = (
                        f"Could not normalize time value {raw_value!r} in existing Zarr store {out_zarr}; "
                        "cannot resume safely."
                    )
                    raise RuntimeError(msg)
                fallback_keys.add(key)
            return fallback_keys
        except Exception:
            msg = f"Could not open existing Zarr store {out_zarr}: {exc}"
            raise RuntimeError(msg) from exc

    try:
        if "time" not in xds.coords:
            msg = f"Existing Zarr store {out_zarr} has no 'time' coordinate; cannot resume safely."
            raise RuntimeError(msg)

        keys: set[str] = set()
        for raw_value in np.atleast_1d(xds["time"].values):
            key = _normalize_time_key(raw_value)
            if key is None:
                msg = (
                    f"Could not normalize time value {raw_value!r} in existing Zarr store {out_zarr}; "
                    "cannot resume safely."
                )
                raise RuntimeError(msg)
            keys.add(key)
        return keys
    finally:
        xds.close()


def _reindex_time_step_to_expected_frequencies(
    xds_t: xr.Dataset,
    expected_frequencies_hz: List[float],
) -> xr.Dataset:
    """Ensure each time-step has the full expected frequency axis.

    Missing subbands are introduced as NaN values in data variables.
    """
    if "frequency" not in xds_t.coords or not expected_frequencies_hz:
        return xds_t

    expected = np.asarray(expected_frequencies_hz, dtype=float)
    observed = np.asarray(np.atleast_1d(xds_t["frequency"].values), dtype=float)
    if observed.size == 0:
        return xds_t.reindex({"frequency": expected}, fill_value=np.nan)

    mapped = observed.copy()
    max_jitter_hz = _DISCOVERY_FREQ_BIN_HZ / 2.0
    for i, freq in enumerate(observed):
        nearest_idx = int(np.argmin(np.abs(expected - freq)))
        if abs(float(expected[nearest_idx] - freq)) <= max_jitter_hz:
            mapped[i] = expected[nearest_idx]

    xds_norm = xds_t.assign_coords(frequency=("frequency", mapped))

    _, first_indices = np.unique(mapped, return_index=True)
    if len(first_indices) != len(mapped):
        xds_norm = xds_norm.isel(frequency=np.sort(first_indices))

    xds_norm = xds_norm.sortby("frequency")
    return xds_norm.reindex({"frequency": expected}, fill_value=np.nan)


def _validate_time_axis_consistency_zarr(out_zarr: Path) -> None:
    """Ensure all Zarr arrays with a ``time`` dimension share one length."""
    try:
        zg = zarr.open_group(str(out_zarr), mode="r")
    except Exception:
        return
    buckets: Dict[int, List[str]] = {}
    for name in zg.array_keys():
        arr = zg[name]
        dims_attr = arr.attrs.get("_ARRAY_DIMENSIONS")
        if dims_attr is None:
            continue
        dims = [dims_attr] if isinstance(dims_attr, str) else [str(d) for d in dims_attr]
        if "time" not in dims:
            continue
        time_axis = dims.index("time")
        time_len = int(arr.shape[time_axis])
        buckets.setdefault(time_len, []).append(name)

    if len(buckets) <= 1:
        return

    details = "; ".join(f"time={k}: {sorted(v)}" for k, v in sorted(buckets.items()))
    msg = (
        f"Existing Zarr store {out_zarr} has inconsistent time-axis lengths across arrays ({details}). "
        "This usually indicates an interrupted append. Repair the store or rebuild before resuming."
    )
    raise RuntimeError(msg)


def _zarr_store_exists(out_zarr: Path) -> bool:
    """Return True when *out_zarr* is a readable Zarr group (not just an empty directory)."""
    if not out_zarr.exists():
        return False
    try:
        zarr.open_group(str(out_zarr), mode="r")
    except Exception:
        return False
    return True


def _time_axis_length_buckets(out_zarr: Path) -> Dict[int, List[str]]:
    """Return mapping of time-axis length -> array names."""
    zg = zarr.open_group(str(out_zarr), mode="r")
    buckets: Dict[int, List[str]] = {}
    for name in sorted(zg.array_keys()):
        arr = zg[name]
        dims_attr = arr.attrs.get("_ARRAY_DIMENSIONS")
        if dims_attr is None:
            continue
        dims = [dims_attr] if isinstance(dims_attr, str) else [str(d) for d in dims_attr]
        if "time" not in dims:
            continue
        time_axis = dims.index("time")
        time_len = int(arr.shape[time_axis])
        buckets.setdefault(time_len, []).append(name)
    return buckets


def validate_zarr_store(out_zarr: str | Path) -> Dict[str, object]:
    """Validate time-axis consistency for a Zarr store."""
    out_zarr = Path(out_zarr)
    if not out_zarr.exists():
        msg = f"Zarr store does not exist: {out_zarr}"
        raise FileNotFoundError(msg)

    buckets = _time_axis_length_buckets(out_zarr)
    consistent = len(buckets) <= 1
    report: Dict[str, object] = {
        "store": str(out_zarr),
        "consistent": consistent,
        "time_length_buckets": {k: sorted(v) for k, v in sorted(buckets.items())},
    }
    if not consistent:
        details = "; ".join(f"time={k}: {sorted(v)}" for k, v in sorted(buckets.items()))
        report["message"] = (
            f"Inconsistent time-axis lengths across arrays ({details}). "
            "This usually indicates an interrupted append."
        )
    return report


def repair_zarr_store(
    out_zarr: str | Path,
    *,
    fits_dir: str | Path | None = None,
    backup_suffix: str = ".backup-before-repair",
) -> Dict[str, object]:
    """Repair inconsistent time-axis lengths and optionally refresh WCS headers."""
    out_zarr = Path(out_zarr)
    if not out_zarr.exists():
        msg = f"Zarr store does not exist: {out_zarr}"
        raise FileNotFoundError(msg)

    backup_path = out_zarr.with_name(out_zarr.name + backup_suffix)
    if backup_path.exists():
        msg = f"Backup path already exists: {backup_path}"
        raise FileExistsError(msg)

    pre = validate_zarr_store(out_zarr)
    buckets = pre["time_length_buckets"]
    if not buckets:
        msg = f"No arrays with a time axis found in {out_zarr}"
        raise RuntimeError(msg)
    repaired_len = min(int(k) for k in buckets.keys())

    shutil.copytree(out_zarr, backup_path)
    zg = zarr.open_group(str(out_zarr), mode="a")

    truncated: List[str] = []
    for name in sorted(zg.array_keys()):
        arr = zg[name]
        dims_attr = arr.attrs.get("_ARRAY_DIMENSIONS")
        if dims_attr is None:
            continue
        dims = [dims_attr] if isinstance(dims_attr, str) else [str(d) for d in dims_attr]
        if "time" not in dims:
            continue
        time_axis = dims.index("time")
        if int(arr.shape[time_axis]) <= repaired_len:
            continue

        slicer = [slice(None)] * arr.ndim
        slicer[time_axis] = slice(0, repaired_len)
        data = arr[tuple(slicer)]
        attrs = dict(arr.attrs)
        chunks = arr.chunks
        dtype = arr.dtype
        del zg[name]
        new_arr = zg.create_dataset(
            name,
            data=data.astype(dtype, copy=False),
            chunks=chunks if len(chunks) == data.ndim else True,
            overwrite=True,
        )
        new_arr.attrs.update(attrs)
        truncated.append(name)

    rewritten_wcs_rows = 0
    if fits_dir is not None and "wcs_header_str" in zg and "time" in zg:
        fits_dir = Path(fits_dir)
        if fits_dir.exists():
            by_time = _discover_groups(fits_dir)
            z_time = np.atleast_1d(zg["time"][:])
            z_time_keys = [_normalize_time_key(v) for v in z_time[:repaired_len]]
            wcs_arr = zg["wcs_header_str"]
            n_freq = int(wcs_arr.shape[1]) if wcs_arr.ndim >= 2 else 0
            for ti, tkey in enumerate(z_time_keys):
                if tkey is None:
                    continue
                files = by_time.get(tkey, [])
                if not files:
                    continue
                files = sorted(files, key=_frequency_sort_tuple)
                row = wcs_arr[ti, :].copy()
                for fi, fp in enumerate(files[:n_freq]):
                    try:
                        with fits.open(str(fp), memmap=True) as hdul:
                            hdr = hdul[0].header
                        s = WCS(hdr).celestial.to_header().tostring(sep="\n")
                        row[fi] = np.bytes_(s.encode("utf-8"))
                    except Exception:
                        continue
                wcs_arr[ti, :] = row
                rewritten_wcs_rows += 1

    zarr.consolidate_metadata(str(out_zarr))
    post = validate_zarr_store(out_zarr)
    return {
        "store": str(out_zarr),
        "backup": str(backup_path),
        "repaired_len": repaired_len,
        "truncated_arrays": truncated,
        "rewritten_wcs_rows": rewritten_wcs_rows,
        "pre": pre,
        "post": post,
    }


def _frequency_sort_tuple(fp: Path) -> Tuple[float, str]:
    """Sort key for deterministic frequency ordering with fallback."""
    return _discovery_frequency_sort_tuple(fp, group_metadata_source="fits")


def _earth_location_from_header(hdr: fits.Header) -> EarthLocation:
    """Geodetic Earth location from FITS ``OBSGEO-*`` or OVRO-LWA defaults."""
    if "OBSGEO-L" in hdr and "OBSGEO-B" in hdr:
        height = float(hdr.get("OBSGEO-H", 0.0))
        return EarthLocation.from_geodetic(
            float(hdr["OBSGEO-L"]) * u.deg,
            float(hdr["OBSGEO-B"]) * u.deg,
            height * u.m,
        )
    return EarthLocation.from_geodetic(
        _OVRO_LWA_DEFAULT_LON_DEG * u.deg,
        _OVRO_LWA_DEFAULT_LAT_DEG * u.deg,
        _OVRO_LWA_DEFAULT_HEIGHT_M * u.m,
    )


def _obstime_from_fits_filename(path: Path) -> Optional[Time]:
    """Parse ``-image-YYYYMMDD_HHMMSS`` from a FITS basename as UTC :class:`~astropy.time.Time`.

    Returns ``None`` if the pattern is missing or digits are not a valid civil time.
    """
    m = _IMAGE_TIME_RE.search(path.name)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    try:
        naive = datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")
    except ValueError:
        return None
    utc = naive.replace(tzinfo=timezone.utc)
    return Time(utc)


def _time_key_from_filename(fp: Path) -> Optional[str]:
    """Observation time key from ``-image-YYYYMMDD_HHMMSS`` in the basename.

    Same ``%Y%m%d_%H%M%S`` formatting as :func:`_time_key_from_header`. Returns ``None``
    when the filename pattern is absent or not parseable.
    """
    obstime = _obstime_from_fits_filename(fp)
    if obstime is None:
        return None
    return obstime.to_datetime().strftime("%Y%m%d_%H%M%S")


def _strip_fits_ctype_cards(hdr: fits.Header) -> None:
    """Strip FITS ``CTYPEn`` string values of padding.

    xradio's FITS reader uses ``helpers['ctype'].index('STOKES')`` (exact match). Some
    writers leave trailing spaces in ``CTYPE`` values so the string is not equal to
    ``'STOKES'``, which raises ``ValueError: 'STOKES' is not in the list``.
    """
    nax = int(hdr.get("NAXIS", 0))
    for i in range(1, nax + 1):
        key = f"CTYPE{i}"
        if key not in hdr:
            continue
        val = hdr[key]
        if isinstance(val, str):
            hdr[key] = val.strip()


def _fits_axis_is_freq_like(ctype_u: str) -> bool:
    u = ctype_u.strip().upper()
    return u.startswith("FREQ") or u in ("VOPT", "VRAD")


def _header_has_exact_stokes_axis(hdr: fits.Header) -> bool:
    """True if some ``CTYPEn`` is exactly ``STOKES`` (after strip / case fold)."""
    nax = int(hdr.get("NAXIS", 0))
    return any(
        str(hdr.get(f"CTYPE{i}", "")).strip().upper() == "STOKES" for i in range(1, nax + 1)
    )


def _zenith_fk5_crvals_deg(hdr: fits.Header, obstime: Time) -> Optional[Tuple[float, float]]:
    """Return FK5 ``(RA, Dec)`` in degrees for the zenith at ``obstime``, or None if not applicable."""
    ctype1 = str(hdr.get("CTYPE1", "")).strip().upper()
    ctype2 = str(hdr.get("CTYPE2", "")).strip().upper()
    if "RA" not in ctype1 or "DEC" not in ctype2:
        return None

    loc = _earth_location_from_header(hdr)
    zen = SkyCoord(alt=90.0 * u.deg, az=0.0 * u.deg, frame=AltAz(obstime=obstime, location=loc))
    equinox = hdr.get("EQUINOX", 2000.0)
    try:
        eq_time = Time(equinox, format="jyear")
    except (ValueError, TypeError):
        eq_time = Time(2000.0, format="jyear")
    fk5 = zen.transform_to(FK5(equinox=eq_time))
    return (float(fk5.ra.deg), float(fk5.dec.deg))


def _fix_headers(path_in: Path, path_out: Path) -> None:
    """Write a *_fixed.fits with BSCALE/BZERO applied and minimal WCS/spectral keys.

    Adds/ensures:
      RESTFREQ/RESTFRQ when axis 3 is spectral (not after a synthetic Stokes axis),
      SPECSYS=LSRK, TIMESYS=UTC, RADESYS=FK5, LATPOLE=90,
      identity PC matrix for LM, BUNIT=Jy/beam. ``BMAJ``/``BMIN`` from the input are
      preserved verbatim — no placeholder beam is ever written. Inputs without a
      real (present, finite, strictly positive) ``BMAJ``/``BMIN`` raise
      :class:`InvalidBeamError`; the convert pipeline drops such files at discovery
      via :func:`_filter_invalid_beam_files` so this path is only reachable when
      callers invoke :func:`_fix_headers` (or :func:`fix_fits_headers`) directly on
      an unfiltered set.
      For celestial ``RA``/``DEC`` axes, ``CRVAL1``/``CRVAL2`` are set to the FK5
      zenith at the UTC instant parsed from ``-image-YYYYMMDD_HHMMSS`` in the input
      filename. If that pattern is absent, ``CRVAL1``/``CRVAL2`` are left unchanged.
      Geodetic position comes from ``OBSGEO-*`` when present, otherwise OVRO-LWA defaults.

    Parameters
    ----------
    path_in : Path
        Input FITS file path.
    path_out : Path
        Output fixed FITS file path.
    """
    with fits.open(path_in, memmap=True) as hdul:
        hdu = hdul[0]
        data = hdu.data
        hdr = hdu.header.copy()

        beam_reason = _invalid_beam_reason(hdr)
        if beam_reason is not None:
            raise InvalidBeamError(
                f"{path_in.name}: cannot fix headers because the input lacks a usable "
                f"synthesized beam ({beam_reason}); processing must exclude this file."
            )

        bscale = float(hdr.get("BSCALE", 1.0))
        bzero = float(hdr.get("BZERO", 0.0))
        if (bscale != 1.0) or (bzero != 0.0):
            data = data.astype(np.float32) * bscale + bzero
            for k in ("BSCALE", "BZERO"):
                if k in hdr:
                    del hdr[k]

        # xradio expects a STOKES axis in image metadata (see ``_get_pol_values`` in
        # ``xradio.image._util._fits.xds_from_fits``). OVRO-LWA FITS files may be
        # 2D (RA, DEC) only or 3D (RA, DEC, FREQ) without STOKES; promote so FITS parsing
        # succeeds. Pure 2D images become a 4D (RA, DEC, FREQ, STOKES) singleton cube so
        # xradio also builds ``helpers['frequency']`` (required for velocity coords).
        #
        # Strip ``CTYPE`` padding first: xradio uses ``helpers['ctype'].index('STOKES')``
        # (exact string match). Some writers leave trailing spaces so the value is not
        # equal to ``'STOKES'``, which raises ``ValueError: 'STOKES' is not in the list``.
        _strip_fits_ctype_cards(hdr)
        naxis_in = int(hdr.get("NAXIS", 0))
        if data is not None and not _header_has_exact_stokes_axis(hdr) and naxis_in == 4:
            # 4D cubes sometimes carry a length-1 axis mis-labeled (not RA/DEC/FREQ). xradio
            # still requires a literal ``STOKES`` axis name for polarization metadata.
            for i in range(1, 5):
                if int(hdr[f"NAXIS{i}"]) != 1:
                    continue
                axu = str(hdr.get(f"CTYPE{i}", "")).strip().upper()
                if axu.startswith("RA-") or axu.startswith("DEC-"):
                    continue
                if _fits_axis_is_freq_like(axu):
                    continue
                if axu == "STOKES":
                    continue
                hdr[f"CTYPE{i}"] = "STOKES"
                hdr[f"CRVAL{i}"] = 1.0
                hdr[f"CRPIX{i}"] = 1.0
                hdr[f"CDELT{i}"] = 1.0
                hdr.setdefault(f"CUNIT{i}", "")
                break
            _strip_fits_ctype_cards(hdr)

        naxis = int(hdr.get("NAXIS", 0))
        if data is not None and not _header_has_exact_stokes_axis(hdr):
            if naxis == 3:
                data = np.expand_dims(data, axis=0)
                hdr["NAXIS"] = 4
                hdr["CTYPE4"] = "STOKES"
                hdr["CRVAL4"] = 1.0
                hdr["CRPIX4"] = 1.0
                hdr["CDELT4"] = 1.0
                if "CUNIT4" not in hdr:
                    hdr["CUNIT4"] = ""
            elif naxis == 2:
                # Pure (RA, DEC) planes: xradio requires both FREQ (so ``helpers['frequency']``
                # exists) and STOKES. Promote to a 4D (RA, DEC, FREQ, STOKES) singleton cube
                # matching astropy's axis order NAXIS1=NAXIS2=spatial, NAXIS3=NAXIS4=1.
                spec_hz = float(hdr.get("RESTFREQ", 6e7))
                if spec_hz <= 0:
                    spec_hz = 6e7
                data = np.expand_dims(np.expand_dims(data, axis=0), axis=0)
                hdr["NAXIS"] = 4
                hdr["NAXIS3"] = 1
                hdr["CTYPE3"] = "FREQ"
                hdr["CRVAL3"] = spec_hz
                hdr["CRPIX3"] = 1.0
                hdr["CDELT3"] = 1.0
                hdr["CUNIT3"] = "Hz"
                hdr["NAXIS4"] = 1
                hdr["CTYPE4"] = "STOKES"
                hdr["CRVAL4"] = 1.0
                hdr["CRPIX4"] = 1.0
                hdr["CDELT4"] = 1.0
                if "CUNIT4" not in hdr:
                    hdr["CUNIT4"] = ""

        phdu = fits.PrimaryHDU(data=data, header=hdr)
        H = phdu.header

        obs_time = _obstime_from_fits_filename(path_in)
        if obs_time is None:
            logger.warning(
                "No -image-YYYYMMDD_HHMMSS timestamp in filename %s; "
                "leaving CRVAL1/CRVAL2 unchanged.",
                path_in.name,
            )

        # Spectral / frame basics (do not treat Stokes CRVAL4 as a rest frequency)
        ct3 = str(H.get("CTYPE3", "")).strip().upper()
        if "CRVAL3" in H and (
            ("FREQ" in ct3) or ct3.startswith("VOPT") or ct3.startswith("VRAD")
        ):
            H["RESTFREQ"] = H["CRVAL3"]
        H["RESTFRQ"] = (float(H.get("RESTFREQ", 1.0)), "Rest frequency in Hz")
        H["SPECSYS"] = "LSRK"
        H["TIMESYS"] = "UTC"
        H["RADESYS"] = "FK5"
        H["LATPOLE"] = 90.0

        if obs_time is not None:
            zc = _zenith_fk5_crvals_deg(H, obs_time)
            if zc is not None:
                cr1, cr2 = zc
                H["CRVAL1"] = cr1
                H["CRVAL2"] = cr2
                logger.debug(
                    "Set CRVAL1/CRVAL2 to FK5 zenith at filename time for %s: %.6f, %.6f deg",
                    path_in.name,
                    cr1,
                    cr2,
                )

        # Identity PC for LM axes
        H["PC1_1"] = 1.0
        H["PC1_2"] = 0.0
        H["PC2_1"] = 0.0
        H["PC2_2"] = 1.0

        # BMAJ/BMIN are preserved verbatim from the input; the early
        # ``_invalid_beam_reason`` check above guarantees they're present and positive.
        # No nominal/placeholder beam is ever stamped in: a file without a real
        # synthesized beam must be excluded from processing rather than annotated
        # with fabricated values.
        if "BPA" not in H:
            H["BPA"] = 0.0
        if "BUNIT" not in H:
            H["BUNIT"] = "Jy/beam"

        _strip_fits_ctype_cards(H)

        # Write via temporary file + atomic replace to avoid leaving a partial
        # output file when the underlying filesystem intermittently short-writes.
        # Retry short-write failures a few times because they can be transient.
        max_attempts = 3
        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            tmp_out = path_out.with_name(f"{path_out.name}.tmp.{os.getpid()}.{attempt}")
            try:
                phdu.writeto(tmp_out, overwrite=True)
                os.replace(tmp_out, path_out)
                return
            except OSError as exc:
                last_error = exc
                msg = str(exc)
                short_write = "requested" in msg and "written" in msg
                try:
                    if tmp_out.exists():
                        tmp_out.unlink()
                except OSError:
                    pass
                if short_write and attempt < max_attempts:
                    logger.warning(
                        "Short write while fixing %s (attempt %d/%d): %s; retrying",
                        path_in.name,
                        attempt,
                        max_attempts,
                        exc,
                    )
                    time.sleep(0.2 * attempt)
                    continue
                raise
            except Exception:
                try:
                    if tmp_out.exists():
                        tmp_out.unlink()
                except OSError:
                    pass
                raise

        if last_error is not None:
            raise last_error


def _get_fixed_paths(
    files: List[Path],
    fixed_dir: Path,
    *,
    group_metadata_source: Literal["fits", "filename"] = "fits",
) -> List[Path]:
    """Get paths to fixed FITS files, assuming they already exist.

    This function assumes that headers have already been fixed using
    :func:`fix_fits_headers` and simply returns the paths to the
    ``*_fixed.fits`` files.

    Parameters
    ----------
    files : List[Path]
        List of FITS file paths (may be original or already-fixed files).
    fixed_dir : Path
        Directory containing the ``*_fixed.fits`` files.
    group_metadata_source
        Same as :func:`_discover_groups`: ``"fits"`` uses header-backed frequency sort keys;
        ``"filename"`` uses basename MHz only (no FITS I/O).

    Returns
    -------
    List[Path]
        List of paths to fixed FITS files, sorted by frequency.
    """
    fixed_paths: List[Path] = []
    sort_key = lambda p: _discovery_frequency_sort_tuple(p, group_metadata_source=group_metadata_source)
    for f in sorted(files, key=sort_key):
        if f.name.endswith("_fixed.fits"):
            fixed_paths.append(f)
        else:
            fixed = fixed_dir / (f.stem + "_fixed.fits")
            fixed_paths.append(fixed)
    return fixed_paths


def fix_fits_headers(
    files: List[Path],
    fixed_dir: Path,
    *,
    skip_existing: bool = True,
    group_metadata_source: Literal["fits", "filename"] = "fits",
) -> List[Path]:
    """Fix FITS headers for a list of files, creating ``*_fixed.fits`` files.

    This function processes FITS files to ensure they have the necessary
    headers for xradio conversion. It can be run ahead of time before
    calling :func:`convert_fits_dir_to_zarr` to separate the header
    fixing step from the conversion process.

    Parameters
    ----------
    files : List[Path]
        List of FITS file paths to process.
    fixed_dir : Path
        Directory where ``*_fixed.fits`` files will be written.
    skip_existing : bool, optional
        If True, skip files that already have corresponding fixed versions.
        Default is True.
    group_metadata_source
        Frequency sort order for processing files (see :func:`_discover_groups`).
        Default ``"fits"`` uses header-backed keys; ``"filename"`` uses basename MHz only.

    Returns
    -------
    List[Path]
        List of paths to the fixed FITS files.

    Notes
    -----
    * Files already ending with ``_fixed.fits`` are considered already fixed
      and are returned as-is.
    * The :func:`_fix_headers` function applies BSCALE/BZERO and adds minimal
      WCS/spectral keywords required by xradio.
    * Files whose primary header lacks a real ``BMAJ``/``BMIN`` (missing or
      non-positive) raise :class:`InvalidBeamError` inside :func:`_fix_headers`;
      they are logged at WARNING level, omitted from the returned list, and any
      partially-written ``*_fixed.fits`` is removed so downstream consumers see
      only files with a real synthesized beam.

    Examples
    --------
    >>> from pathlib import Path
    >>> from ovro_lwa_portal.fits_to_zarr_xradio import fix_fits_headers
    >>> input_files = list(Path("input").glob("*.fits"))
    >>> fixed_dir = Path("fixed_fits")
    >>> fixed_dir.mkdir(exist_ok=True)
    >>> fixed_paths = fix_fits_headers(input_files, fixed_dir)
    >>> print(f"Fixed {len(fixed_paths)} files")
    """
    fixed_dir.mkdir(parents=True, exist_ok=True)
    fixed_paths: List[Path] = []

    sort_key = lambda p: _discovery_frequency_sort_tuple(p, group_metadata_source=group_metadata_source)
    for f in sorted(files, key=sort_key):
        if f.name.endswith("_fixed.fits"):
            # Already fixed, use as-is
            fixed_paths.append(f)
            logger.debug(f"Skipping already-fixed file: {f.name}")
        else:
            fixed = fixed_dir / (f.stem + "_fixed.fits")
            if skip_existing and fixed.exists():
                logger.debug(f"Skipping existing fixed file: {fixed.name}")
                fixed_paths.append(fixed)
            else:
                logger.info(f"Fixing headers: {f.name} -> {fixed.name}")
                try:
                    _fix_headers(f, fixed)
                except InvalidBeamError as exc:
                    logger.warning(
                        "Skipping %s: %s; excluded from the fixed-FITS set.",
                        f.name,
                        exc,
                    )
                    # Make sure no partial output is left behind for resume-style
                    # invocations (``skip_existing=True``) on a later run.
                    try:
                        fixed.unlink(missing_ok=True)
                    except OSError:
                        pass
                    continue
                fixed_paths.append(fixed)

    return fixed_paths


def _load_for_combine(fp: Path, *, chunk_lm: int = 1024) -> xr.Dataset:
    """
    Load a FITS image, attach *sky* coordinates from the FITS celestial WCS,
    and persist the exact WCS header for FITS-free WCSAxes plotting later.

    This function:
      • reads pixels via :func:`_read_fits_via_xradio` (FITS-only; sky coords off)
      • evaluates RA/Dec at pixel centers (origin=0) using the 2D celestial WCS
      • attaches 2D ``right_ascension``/``declination`` coordinates (deg; FK5/J2000)
      • stores the exact celestial WCS header string redundantly so it survives merges

    Parameters
    ----------
    fp : Path
        Path to the FITS image (original or ``*_fixed.fits``) to load.
    chunk_lm : int, optional
        Chunk size for the ``l`` and ``m`` dimensions. Set to ``0`` to disable
        chunking. Default is ``1024``.

    Returns
    -------
    xarray.Dataset
        Dataset with:
          • data vars from the input FITS (e.g., ``SKY``/``BEAM``)
          • 2D coords: ``right_ascension`` and ``declination`` in degrees
          • WCS header persisted in multiple locations (see Notes)

    Notes
    -----
    * RA/Dec are computed at pixel **centers** via
      ``WCS(header).celestial.all_pix2world(xx, yy, origin=0)`` and therefore
      exactly match the FITS celestial WCS.
    * The celestial WCS header is stored redundantly as:
        - ``xds.attrs['fits_wcs_header']`` (global attrs)
        - a 0-D variable ``wcs_header_str`` (robust across combines)
        - per-variable ``.attrs['fits_wcs_header']``
        - on the RA/Dec coord attrs
      This redundancy ensures at least one copy survives downstream combine/concat
      operations and writers that may drop attrs.
    * Uses :class:`numpy.bytes_` (NumPy ≥ 2.0) for the scalar variable payload.

    """
    # 1) Load image pixels via xradio FITS path only (no CASA probe)
    xds = _read_fits_via_xradio(fp, do_sky_coords=False, compute_mask=False)

    # 2) Open FITS header and extract 2D celestial WCS matching the image plane
    with fits.open(str(fp), memmap=True) as hdul:
        H = hdul[0].header
    w2d = WCS(H).celestial  # 2D (RA/Dec) WCS

    # 3) Compute RA/Dec at pixel centers (origin=0); FITS (NAXIS2, NAXIS1) = (m, l) sizes
    ny = int(xds.sizes["m"])
    nx = int(xds.sizes["l"])
    cel_hdr = w2d.to_header()
    hdr_str = cel_hdr.tostring(sep="\n")
    cache_key = (ny, nx, hdr_str)
    cached = _SKY_COORD_CACHE.get(cache_key)
    if cached is None:
        yy, xx = np.indices((ny, nx), dtype=float)
        ra2d, dec2d = w2d.all_pix2world(xx, yy, 0)  # degrees, shape (nm, nl) = (NAXIS2, NAXIS1)
        _SKY_COORD_CACHE[cache_key] = (ra2d, dec2d, hdr_str)
    else:
        ra2d, dec2d, hdr_str = cached

    # 4) Attach coords on the standard (l, m) grid (transpose of FITS row-major plane)
    ra_lm = np.transpose(ra2d)
    dec_lm = np.transpose(dec2d)
    xds = xds.assign_coords(
        right_ascension=(("l", "m"), ra_lm),
        declination=(("l", "m"), dec_lm),
    )
    xds["right_ascension"].attrs.update({"units": "deg", "frame": "fk5", "equinox": "J2000"})
    xds["declination"].attrs.update({"units": "deg", "frame": "fk5", "equinox": "J2000"})

    # 5) Persist the exact celestial WCS header so we can re-create WCSAxes later without FITS
    xds.attrs["fits_wcs_header"] = hdr_str

    # 6) Hygiene + optional LM chunking
    xds.attrs.pop("history", None)  # keep attrs minimal
    for v in xds.data_vars:
        xds[v].encoding = {}

    if chunk_lm and {"l", "m"} <= set(xds.dims):
        xds = xds.chunk({"l": chunk_lm, "m": chunk_lm})

    # ---- persist the exact celestial WCS header redundantly ----
    # 2) 0-D variable that always survives (NumPy ≥ 2.0: use np.bytes_)
    xds = xds.assign(wcs_header_str=((), np.bytes_(hdr_str.encode("utf-8"))))

    # 3) per-variable attrs (survive merges)
    for dv in xds.data_vars:
        xds[dv].attrs["fits_wcs_header"] = hdr_str

    # 4) also stash on coords for convenience
    xds["right_ascension"].attrs["fits_wcs_header"] = hdr_str
    xds["declination"].attrs["fits_wcs_header"] = hdr_str

    return xds


def _lm_shape(xds: xr.Dataset) -> Tuple[int, int]:
    """Return dataset LM shape as ``(l, m)`` pixel counts."""
    return int(xds.sizes["l"]), int(xds.sizes["m"])


def _select_reference_shape_index(shapes: List[Tuple[int, int]]) -> int:
    """Select deterministic reference index from LM shapes.

    Selection rule:
      1) largest pixel count (l * m)
      2) largest l
      3) largest m
      4) first occurrence on ties
    """
    if not shapes:
        msg = "Cannot select reference shape from empty list."
        raise RuntimeError(msg)

    best_idx = 0
    best_shape = shapes[0]
    best_score = (best_shape[0] * best_shape[1], best_shape[0], best_shape[1])

    for idx, shape in enumerate(shapes[1:], start=1):
        score = (shape[0] * shape[1], shape[0], shape[1])
        if score > best_score:
            best_idx = idx
            best_shape = shape
            best_score = score

    return best_idx


def _peek_lm_shape(fp: Path) -> Tuple[int, int]:
    """Return LM shape ``(l, m)`` from FITS header without loading pixel data.

    Uses ``NAXIS1`` × ``NAXIS2`` as ``(l, m)`` to match the on-disk FITS axis order.
    """
    header = fits.getheader(fp, ext=0)
    naxis = int(header.get("NAXIS", 0))
    if naxis < 2:
        msg = f"FITS file {fp} has NAXIS={naxis}; expected at least 2 for LM dimensions."
        raise RuntimeError(msg)

    return int(header["NAXIS1"]), int(header["NAXIS2"])


def _strip_axis_cards_above(header: fits.Header, *, max_axis: int = 2) -> None:
    """Remove ``NAXISn`` and per-axis WCS cards for axes ``> max_axis`` (in-place).

    Radio FITS often carry Stokes/Frequency on axes 3–4 while the image plane is 2D.
    When building a minimal temporary FITS for xradio, those extra cards must be removed
    so ``NAXIS=2`` matches the written array (same idea as image-plane-correction).
    """
    single_axis_re = re.compile(
        r"^(NAXIS|CTYPE|CRVAL|CRPIX|CDELT|CUNIT|CROTA|CNAME|CRDER|CSYER)(\d+)$"
    )
    matrix_re = re.compile(r"^(CD|PC)(\d+)_(\d+)$")
    pv_ps_re = re.compile(r"^(PV|PS)(\d+)_(\d+)$")
    to_delete: list[str] = []
    for key in header:
        m = single_axis_re.match(key)
        if m and int(m.group(2)) > max_axis:
            to_delete.append(key)
            continue
        m = matrix_re.match(key)
        if m and (int(m.group(2)) > max_axis or int(m.group(3)) > max_axis):
            to_delete.append(key)
            continue
        m = pv_ps_re.match(key)
        if m and int(m.group(2)) > max_axis:
            to_delete.append(key)
            continue
    for key in to_delete:
        try:
            del header[key]
        except KeyError:
            pass


def _make_target_wcs_scaled(seed_wcs: WCS, seed_n: int, target_size: int) -> WCS:
    """Scale a 2D celestial WCS to ``target_size × target_size`` preserving sky coverage.

    Uses the same ``CRPIX`` / ``CD`` (or ``CDELT``) rescaling as image-plane-correction
    ``calcflow`` so dewarp outputs and the portal LM reference stay on the same grid when
    both are driven by the same ``target_size``.
    """
    factor = float(seed_n) / float(target_size)
    target = seed_wcs.deepcopy()
    target.wcs.crpix = (np.asarray(seed_wcs.wcs.crpix, dtype=float) - 0.5) / factor + 0.5
    if seed_wcs.wcs.has_cd():
        target.wcs.cd = np.asarray(seed_wcs.wcs.cd, dtype=float) * factor
    else:
        target.wcs.cdelt = np.asarray(seed_wcs.wcs.cdelt, dtype=float) * factor
    target.pixel_shape = (target_size, target_size)
    return target


def _reproject_celestial_plane(
    plane: NDArray[np.number],
    w_src: WCS,
    w_tgt: WCS,
    shape_out: tuple[int, int],
) -> NDArray[np.floating]:
    from reproject import reproject_interp

    rep, _ = reproject_interp(
        (np.asarray(plane, dtype=np.float64), w_src), w_tgt, shape_out=shape_out
    )
    return np.asarray(rep)


def _resample_lm_reference_to_target_size(
    xds: xr.Dataset,
    ref_fp: Path,
    *,
    target_size: int,
    chunk_lm: int,
) -> xr.Dataset:
    """Reproject spatial planes onto a ``target_size`` square grid in standard ``(l, m)`` order.

    ``reproject`` expects a 2D array in FITS memory order ``(NAXIS2, NAXIS1)`` = ``(m, l)``.
    Each data variable that includes both ``l`` and ``m`` dimensions is transposed to
    ``(..., "l", "m")`` before slicing 2D planes so non-trailing layouts (e.g. another axis
    between ``l`` and ``m``) are still handled. Legacy ``(m, l)`` trailing pairs are
    normalized by that transpose. Output arrays are stored as ``(..., "l", "m")``.
    """
    if target_size <= 0:
        msg = f"target_size must be positive, got {target_size}"
        raise ValueError(msg)

    nl, nm = _lm_shape(xds)
    if nl != nm:
        msg = (
            f"Resampling the global LM reference to target_size={target_size} requires a "
            f"square sky grid; got (l,m)=({nl},{nm}) from {ref_fp}"
        )
        raise ValueError(msg)
    if nl == target_size:
        return xds

    with fits.open(str(ref_fp), memmap=True) as hdul:
        hdr = hdul[0].header.copy()
    w_src = WCS(hdr).celestial
    w_tgt = _make_target_wcs_scaled(w_src, nl, target_size)
    shape_out = (target_size, target_size)

    xds_mat = xds.load()
    replaced: dict[str, xr.DataArray] = {}
    for name, dv in xds_mat.data_vars.items():
        if name == "wcs_header_str":
            continue
        if dv.ndim < 2 or not ({"l", "m"} <= set(dv.dims)):
            continue
        try:
            dv_lm = dv.transpose(..., "l", "m")
        except ValueError:
            continue
        dims = tuple(dv_lm.dims)
        if dims[-2:] != ("l", "m"):
            continue
        arr = np.asarray(dv_lm.values)
        lead_shape = arr.shape[:-2]
        out_dims = dims[:-2] + ("l", "m")
        if lead_shape:
            out_dtype = np.result_type(arr.dtype, np.float32)
            out_arr = np.empty(lead_shape + shape_out, dtype=out_dtype)
            for idx in np.ndindex(lead_shape):
                plane = np.asarray(arr[idx], dtype=np.float64)
                plane_ml = plane.T
                rep_ml = _reproject_celestial_plane(plane_ml, w_src, w_tgt, shape_out)
                out_lm = np.asarray(rep_ml).T
                out_arr[idx] = out_lm.astype(out_dtype, copy=False)
        else:
            plane = np.asarray(arr, dtype=np.float64)
            plane_ml = plane.T
            rep_ml = _reproject_celestial_plane(plane_ml, w_src, w_tgt, shape_out)
            out_arr = np.asarray(rep_ml).T.astype(arr.dtype, copy=False)
        replaced[name] = xr.DataArray(out_arr, dims=out_dims, attrs=dict(dv.attrs))

    if not replaced:
        dv_dims = {k: tuple(v.dims) for k, v in xds_mat.data_vars.items()}
        msg = (
            f"No data variables with both 'l' and 'm' dimensions could be resampled "
            f"for global LM reference from {ref_fp}. data_vars (name -> dims): {dv_dims}"
        )
        raise RuntimeError(msg)

    _strip_axis_cards_above(hdr, max_axis=2)
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = target_size
    hdr["NAXIS2"] = target_size
    hdr["BITPIX"] = -32
    if "BSCALE" in hdr:
        hdr["BSCALE"] = 1.0
    if "BZERO" in hdr:
        hdr["BZERO"] = 0.0
    tw_hdr = w_tgt.to_header(relax=True)
    for key in tw_hdr:
        hdr[key] = tw_hdr[key]
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = target_size
    hdr["NAXIS2"] = target_size
    if "TELESCOP" not in hdr:
        hdr["TELESCOP"] = "UNKNOWN"

    fd, tmp_name = tempfile.mkstemp(suffix=".fits")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        fits.PrimaryHDU(
            data=np.zeros((target_size, target_size), dtype=np.float32),
            header=hdr,
        ).writeto(tmp_path, overwrite=True)
        shell = _read_fits_via_xradio(
            tmp_path, do_sky_coords=False, compute_mask=False
        ).load()
    finally:
        tmp_path.unlink(missing_ok=True)

    out = shell.copy(deep=False)
    for name, da_new in replaced.items():
        out[name] = da_new

    yy, xx = np.indices(shape_out, dtype=float)
    ra2d, dec2d = w_tgt.all_pix2world(xx, yy, 0)
    ra_lm = np.transpose(ra2d)
    dec_lm = np.transpose(dec2d)
    cel_hdr = w_tgt.to_header(relax=True)
    hdr_str = cel_hdr.tostring(sep="\n")
    out = out.assign_coords(
        right_ascension=(("l", "m"), ra_lm),
        declination=(("l", "m"), dec_lm),
    )
    out["right_ascension"].attrs.update({"units": "deg", "frame": "fk5", "equinox": "J2000"})
    out["declination"].attrs.update({"units": "deg", "frame": "fk5", "equinox": "J2000"})
    out.attrs["fits_wcs_header"] = hdr_str
    out.attrs.pop("history", None)
    out = out.assign(wcs_header_str=((), np.bytes_(hdr_str.encode("utf-8"))))
    for dv in out.data_vars:
        out[dv].encoding = {}
        out[dv].attrs["fits_wcs_header"] = hdr_str
    out["right_ascension"].attrs["fits_wcs_header"] = hdr_str
    out["declination"].attrs["fits_wcs_header"] = hdr_str

    if chunk_lm and {"l", "m"} <= set(out.dims):
        out = out.chunk({"l": chunk_lm, "m": chunk_lm})
    else:
        out = out.chunk({"l": -1, "m": -1})

    logger.info(
        "Global LM reference resampled from (l,m)=(%d,%d) to (%d,%d) (target_size=%d)",
        nl,
        nm,
        target_size,
        target_size,
        target_size,
    )
    return out


def _load_global_lm_reference_dataset(
    by_time: Dict[str, List[Path]],
    fixed_dir: Path,
    *,
    chunk_lm: int,
    fix_headers_on_demand: bool,
    target_size: int | None = None,
    group_metadata_source: Literal["fits", "filename"] = "fits",
) -> xr.Dataset:
    """Load the dataset whose LM grid has the largest shape across *all* time steps.

    Per-time reprojection alone is insufficient: different observation times can
    imply different in-step max shapes (e.g. only 3122² files in one step and
    mixed 4096²+3122² in another). A single global reference ensures every step
    normalizes to the same ``l``/``m`` so :func:`_assert_same_lm` can succeed.

    If ``target_size`` is set, the chosen reference is loaded at native resolution
    then reprojected onto a ``target_size`` square grid (same WCS scaling as
    image-plane-correction ``calcflow``) so the LM grid matches dewarped outputs.
    """
    candidates: List[Tuple[Path, Tuple[int, int]]] = []
    for tkey in sorted(by_time.keys()):
        files = by_time[tkey]
        # Avoid eagerly fixing every FITS file just to inspect LM shape.
        # Header LM dimensions (NAXIS1/NAXIS2) are sufficient for choosing
        # the global reference grid and dramatically reduce temporary disk use.
        if fix_headers_on_demand:
            sort_key = lambda p: _discovery_frequency_sort_tuple(
                p, group_metadata_source=group_metadata_source
            )
            shape_paths = sorted(files, key=sort_key)
        else:
            shape_paths = _get_fixed_paths(files, fixed_dir, group_metadata_source=group_metadata_source)
        for fp in shape_paths:
            candidates.append((fp, _peek_lm_shape(fp)))

    if not candidates:
        msg = "No FITS paths available to build a global LM reference."
        raise RuntimeError(msg)

    shapes = [sh for _, sh in candidates]
    win_idx = _select_reference_shape_index(shapes)
    ref_fp, ref_shape = candidates[win_idx]
    logger.info(
        "Global LM reference grid (l,m)=%s from %s",
        ref_shape,
        ref_fp.name,
    )
    if fix_headers_on_demand and not ref_fp.name.endswith("_fixed.fits"):
        ref_fp = fix_fits_headers([ref_fp], fixed_dir, skip_existing=True)[0]
    xds = _load_for_combine(ref_fp, chunk_lm=chunk_lm)
    if target_size is not None:
        xds = _resample_lm_reference_to_target_size(
            xds, ref_fp, target_size=target_size, chunk_lm=chunk_lm
        )
    return xds


def _lm_index_coords_match(
    ref: xr.Dataset,
    xds: xr.Dataset,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Return True if ``xds`` uses the same ``l`` / ``m`` index coordinates as ``ref``.

    Same-length ``l``/``m`` from different subbands can still differ numerically (e.g.
    different on-sky pixelization at 4096²). Skipping regrid in that case leaves
    :func:`xr.combine_by_coords` to align slices with an outer join on ``l``/``m``,
    inflating spatial dims (e.g. ``3 × 4096``) and breaking the global LM contract.
    """
    if {"l", "m"} > set(ref.coords) or {"l", "m"} > set(xds.coords):
        return False
    if _lm_shape(xds) != _lm_shape(ref):
        return False
    ref_l = np.asarray(ref["l"].values, dtype=np.float64)
    ref_m = np.asarray(ref["m"].values, dtype=np.float64)
    x_l = np.asarray(xds["l"].values, dtype=np.float64)
    x_m = np.asarray(xds["m"].values, dtype=np.float64)
    return bool(
        np.allclose(ref_l, x_l, rtol=rtol, atol=atol)
        and np.allclose(ref_m, x_m, rtol=rtol, atol=atol)
    )


def _read_wcs_header_str(ds: xr.Dataset) -> Optional[str]:
    """Return the persisted 2D celestial WCS header string from *ds*, if any.

    Looks first at ``ds.attrs['fits_wcs_header']`` (set by :func:`_load_for_combine`
    and :func:`_resample_lm_reference_to_target_size`), then falls back to the 0-D
    ``wcs_header_str`` data variable so the lookup also works after merges that may
    strip non-essential attrs.
    """
    s = ds.attrs.get("fits_wcs_header")
    if s is None and "wcs_header_str" in ds:
        raw_arr = np.asarray(ds["wcs_header_str"].values)
        if raw_arr.size == 0:
            return None
        # After incremental Zarr appends, ``wcs_header_str`` is promoted to ``(time,)``
        # (or ``(time, frequency)``); use any representative row for the LM grid WCS.
        raw = np.ravel(raw_arr)[0]
        s = raw.decode("utf-8") if isinstance(raw, (bytes, np.bytes_)) else str(raw)
    if s is None:
        return None
    return str(s)


def _wcs_header_from_ref_grid_and_source_crval(
    *,
    ref: xr.Dataset,
    xds: xr.Dataset,
    source_label: Optional[str],
) -> str:
    """Build a 2D celestial WCS header from ``ref``'s pixel grid + ``xds``'s CRVAL.

    The LM reference defines only the pixel grid (CRPIX/CDELT/CTYPE/projection); the
    per-time celestial reference (CRVAL1/CRVAL2, FK5 zenith at the source's obs time,
    stamped by :func:`_fix_headers`) lives on each source FITS. Combining them lets
    every subband at one time step emit identical ``right_ascension``/``declination``
    while keeping the LM reference's role as a grid-shape template across time steps.

    Raises
    ------
    RuntimeError
        If either ``ref`` or ``xds`` is missing a persisted ``fits_wcs_header``.
    """
    ref_hdr_str = _read_wcs_header_str(ref)
    src_hdr_str = _read_wcs_header_str(xds)
    if ref_hdr_str is None or src_hdr_str is None:
        who = f"{source_label}: " if source_label else ""
        msg = (
            f"{who}cannot derive per-time celestial WCS: missing fits_wcs_header on "
            f"ref={ref_hdr_str is None} or source={src_hdr_str is None}"
        )
        raise RuntimeError(msg)

    ref_hdr = fits.Header.fromstring(ref_hdr_str, sep="\n")
    src_hdr = fits.Header.fromstring(src_hdr_str, sep="\n")

    new_hdr = ref_hdr.copy()
    for k in ("CRVAL1", "CRVAL2", "RADESYS", "EQUINOX", "DATE-OBS", "MJD-OBS"):
        if k in src_hdr:
            new_hdr[k] = src_hdr[k]
    if "CRVAL2" in src_hdr:
        # Standard SIN: LATPOLE tracks CRVAL2 (latitude of native pole = ref dec).
        new_hdr["LATPOLE"] = float(src_hdr["CRVAL2"])

    return WCS(new_hdr).celestial.to_header(relax=True).tostring(sep="\n")


def _regrid_to_reference_lm(
    xds: xr.Dataset,
    ref: xr.Dataset,
    *,
    source_label: Optional[str] = None,
) -> xr.Dataset:
    """Interpolate ``xds`` onto ``ref``'s ``l`` / ``m`` coordinate grid.

    Uses linear interpolation in ``(l, m)``. The reference contributes **only the
    pixel grid** (``CRPIX``/``CDELT``/projection); per-time celestial reference
    (``CRVAL1``/``CRVAL2``, FK5 zenith stamped by :func:`_fix_headers`) is taken from
    ``xds`` so all subbands at one time step end up with identical
    ``right_ascension``/``declination``. No-op when ``xds`` already matches ``ref``'s
    LM shape **and** ``l``/``m`` index coordinates within tolerance — in that case the
    source's own (already per-time-correct) sky coords pass through unchanged.

    Parameters
    ----------
    xds
        Source dataset (e.g. from :func:`_load_for_combine`).
    ref
        Reference dataset whose ``l`` and ``m`` define the target pixel grid.
    source_label
        Optional filename or path for error messages when regridding fails.

    Returns
    -------
    xarray.Dataset
        Dataset on the reference LM grid with writer-safe encodings cleared.

    Raises
    ------
    RuntimeError
        If interpolation fails (e.g. incompatible coordinates) or the per-time
        celestial WCS cannot be derived (missing ``fits_wcs_header``).
    """
    if _lm_shape(xds) == _lm_shape(ref) and _lm_index_coords_match(ref, xds):
        return xds

    if "l" not in xds.coords or "m" not in xds.coords:
        who = f"{source_label}: " if source_label else ""
        msg = f"{who}cannot regrid: dataset is missing ``l`` and/or ``m`` coordinates."
        raise RuntimeError(msg)

    # Materialize for scipy-backed interp; reference coords may be lazy.
    xds = xds.load()
    target_l = ref["l"].load() if hasattr(ref["l"].data, "compute") else ref["l"]
    target_m = ref["m"].load() if hasattr(ref["m"].data, "compute") else ref["m"]

    try:
        regridded = xds.interp(l=target_l, m=target_m, method="linear")
    except Exception as exc:
        who = f"{source_label}: " if source_label else ""
        msg = f"{who}LM regridding onto reference grid failed: {exc}"
        raise RuntimeError(msg) from exc

    # Recompute sky coords on ref's pixel grid using the source's per-time CRVAL so
    # every subband at one time step shares identical RA/Dec regardless of which path
    # (regrid vs. no-op short-circuit) handled it.
    hdr_str = _wcs_header_from_ref_grid_and_source_crval(
        ref=ref, xds=xds, source_label=source_label
    )
    target_wcs = WCS(fits.Header.fromstring(hdr_str, sep="\n"))
    ny = int(regridded.sizes["m"])
    nx = int(regridded.sizes["l"])
    yy, xx = np.indices((ny, nx), dtype=float)
    ra2d, dec2d = target_wcs.all_pix2world(xx, yy, 0)
    ra_lm = np.transpose(ra2d)
    dec_lm = np.transpose(dec2d)

    regridded = regridded.assign_coords(
        right_ascension=(("l", "m"), ra_lm),
        declination=(("l", "m"), dec_lm),
    )
    regridded["right_ascension"].attrs.update(
        {"units": "deg", "frame": "fk5", "equinox": "J2000"}
    )
    regridded["declination"].attrs.update(
        {"units": "deg", "frame": "fk5", "equinox": "J2000"}
    )

    regridded.attrs.pop("history", None)
    regridded.attrs["fits_wcs_header"] = hdr_str
    regridded["right_ascension"].attrs["fits_wcs_header"] = hdr_str
    regridded["declination"].attrs["fits_wcs_header"] = hdr_str
    for dv in regridded.data_vars:
        if dv == "wcs_header_str":
            continue
        regridded[dv].attrs["fits_wcs_header"] = hdr_str

    regridded["wcs_header_str"] = xr.DataArray(
        np.bytes_(hdr_str.encode("utf-8")), dims=()
    )

    for v in regridded.data_vars:
        regridded[v].encoding = {}

    return regridded


def _sky_sep_max_vs_ref_arcsec(
    ra: NDArray[np.floating],
    dec: NDArray[np.floating],
    *,
    ref_idx: int,
    max_points: int = _CELESTIAL_DRIFT_SAMPLE_MAX_POINTS,
) -> float:
    """Worst-case on-sky separation (arcsec) of any non-ref channel from ``ref_idx``.

    Parameters
    ----------
    ra, dec
        Arrays shaped ``(n_freq, n_m, n_l)`` in degrees.
    ref_idx
        Reference frequency index.
    max_points
        Maximum number of (m, l) pixels sampled per comparison (subsampled if larger).
    """
    if ra.ndim != 3 or dec.ndim != 3:
        raise ValueError("ra/dec must be 3D arrays shaped (n_freq, n_m, n_l)")
    nf, nm, nl = int(ra.shape[0]), int(ra.shape[1]), int(ra.shape[2])
    if nf <= 1:
        return 0.0
    ri = int(np.clip(ref_idx, 0, nf - 1))
    total = nm * nl
    rng = np.random.default_rng(_CELESTIAL_DRIFT_SAMPLE_SEED)
    if total > max_points:
        flat_idx = rng.choice(total, size=max_points, replace=False)
    else:
        flat_idx = np.arange(total, dtype=np.intp)
    idx_m, idx_l = np.unravel_index(flat_idx, (nm, nl))
    ref_ra = ra[ri][idx_m, idx_l]
    ref_dec = dec[ri][idx_m, idx_l]
    worst = 0.0
    for fi in range(nf):
        if fi == ri:
            continue
        ra_i = ra[fi][idx_m, idx_l]
        dec_i = dec[fi][idx_m, idx_l]
        ok = np.isfinite(ref_ra) & np.isfinite(ref_dec) & np.isfinite(ra_i) & np.isfinite(dec_i)
        if not np.any(ok):
            continue
        c_a = SkyCoord(ra=ref_ra[ok] * u.deg, dec=ref_dec[ok] * u.deg, frame="fk5")
        c_b = SkyCoord(ra=ra_i[ok] * u.deg, dec=dec_i[ok] * u.deg, frame="fk5")
        worst = max(worst, float(c_a.separation(c_b).max().to(u.arcsec).value))
    return worst


def _harmonize_celestial_coords_independent_of_frequency(
    xds: xr.Dataset,
    *,
    ref_freq_idx: int = 0,
    warn_max_sep_arcsec: float = _CELESTIAL_FRAME_WARN_MAX_SKY_SEP_ARCSEC,
) -> xr.Dataset:
    """Collapse per-frequency RA/Dec coords to a single (m, l) celestial grid.

    Stacked wideband cubes can carry ``(frequency, m, l)`` right ascension and
    declination. Downstream code expects one celestial frame for the LM grid; this
    keeps the reference slice (default: lowest ``frequency`` index after sorting)
    and emits a warning if sampled pixels differ from that reference beyond
    ``warn_max_sep_arcsec``.
    """
    ra_c = xds.coords.get("right_ascension")
    dec_c = xds.coords.get("declination")
    if ra_c is None or dec_c is None:
        return xds
    if "frequency" not in ra_c.dims:
        return xds

    nf = int(xds.sizes["frequency"])
    ri = int(np.clip(ref_freq_idx, 0, nf - 1))

    ra_ord = ra_c.transpose("frequency", "m", "l")
    dec_ord = dec_c.transpose("frequency", "m", "l")
    if hasattr(ra_ord.data, "compute") or hasattr(dec_ord.data, "compute"):
        max_points = _CELESTIAL_DRIFT_SAMPLE_MAX_POINTS
        stacked_dims = ("m", "l")
        total = int(ra_ord.sizes["m"]) * int(ra_ord.sizes["l"])
        rng = np.random.default_rng(_CELESTIAL_DRIFT_SAMPLE_SEED)
        if total > max_points:
            flat_idx = rng.choice(total, size=max_points, replace=False)
        else:
            flat_idx = np.arange(total, dtype=np.intp)
        ra_sample = ra_ord.stack(pixel=stacked_dims).isel(pixel=flat_idx)
        dec_sample = dec_ord.stack(pixel=stacked_dims).isel(pixel=flat_idx)
        ra_np = np.asarray(ra_sample.data, dtype=np.float64)[:, np.newaxis, :]
        dec_np = np.asarray(dec_sample.data, dtype=np.float64)[:, np.newaxis, :]
    else:
        ra_np = np.asarray(ra_ord.data, dtype=np.float64)
        dec_np = np.asarray(dec_ord.data, dtype=np.float64)

    max_sep = _sky_sep_max_vs_ref_arcsec(ra_np, dec_np, ref_idx=ri)
    if max_sep > warn_max_sep_arcsec:
        freqs = xds.coords["frequency"].values
        logger.warning(
            "Celestial coordinate grids differ by up to %.1f arcsec across %d frequency "
            "slice(s) in this combined time step (threshold %.1f arcsec). "
            "Using the reference channel index %d (%.6g Hz) for a single "
            "right_ascension / declination frame shared by all frequencies; "
            "inspect per-FITS WCS or phase tracking if this is unexpected.",
            max_sep,
            nf,
            warn_max_sep_arcsec,
            ri,
            float(freqs[ri]),
        )

    ra_ref = xds.right_ascension.isel(frequency=ri, drop=True)
    dec_ref = xds.declination.isel(frequency=ri, drop=True)
    hdr_ref = ra_ref.attrs.get("fits_wcs_header")

    out = xds.assign_coords(right_ascension=ra_ref, declination=dec_ref)

    if hdr_ref is not None:
        out.right_ascension.attrs["fits_wcs_header"] = hdr_ref
        out.declination.attrs["fits_wcs_header"] = hdr_ref
        for dv in out.data_vars:
            out[dv].attrs["fits_wcs_header"] = hdr_ref

    if "wcs_header_str" in out.data_vars:
        wh = out["wcs_header_str"]
        if "frequency" in wh.dims:
            out = out.assign(wcs_header_str=wh.isel(frequency=ri, drop=True))

    return out


def _invalid_beam_reason(header: fits.Header) -> Optional[str]:
    """Return a short reason if the FITS header has no usable synthesized beam, else ``None``.

    A header is considered to have a usable beam when both ``BMAJ`` and ``BMIN`` are
    present, finite, and strictly positive. Common bad cases for OVRO-LWA dewarped
    products are:

    * the cascade flow failed to fit the synthesized beam and left ``BMAJ``/``BMIN``
      absent from the output FITS header;
    * the cascade emitted placeholder zeros (``BMAJ=0`` and/or ``BMIN=0``).

    These reasons are surfaced verbatim in the discovery skip warnings so log readers
    can correlate ingest decisions with upstream imaging issues.
    """
    missing = [k for k in ("BMAJ", "BMIN") if k not in header]
    if missing:
        return f"missing {'/'.join(missing)}"

    bad: List[str] = []
    for k in ("BMAJ", "BMIN"):
        try:
            v = float(header[k])
        except (TypeError, ValueError):
            bad.append(f"{k}=<non-numeric:{header[k]!r}>")
            continue
        if not np.isfinite(v):
            bad.append(f"{k}={v}")
        elif v <= 0.0:
            bad.append(f"{k}={v}")
    if bad:
        return "; ".join(bad)
    return None


def _min_primary_hdu_file_bytes(header: fits.Header) -> int:
    """Minimum on-disk bytes for an uncompressed primary image HDU (header blocks + data)."""
    header_size = ((len(header) * 80 + 2879) // 2880) * 2880
    bitpix = abs(int(header.get("BITPIX", 8)))
    naxis = int(header.get("NAXIS", 0))
    nelem = 1
    for i in range(1, naxis + 1):
        nelem *= int(header[f"NAXIS{i}"])
    bps = max(1, bitpix // 8)
    return header_size + nelem * bps


def _truncated_fits_reason(fp: Path) -> Optional[str]:
    """Return a short reason when *fp* is empty or shorter than its primary HDU data.

    Covers zero-byte files and partial copies where the on-disk size is less than the
    minimum required for the primary HDU header and uncompressed image array (typical for
    broken lustre symlinks). Without this check, combine fails later with opaque NumPy
    errors such as ``buffer is too small for requested array``.
    """
    try:
        file_size = fp.stat().st_size
    except OSError as exc:
        return f"cannot stat file ({exc})"
    if file_size == 0:
        return "empty file (0 bytes)"
    try:
        # Parse header from raw bytes so a truncated file does not leave astropy file
        # handles open (``getheader`` can leak on validation failure).
        with fp.open("rb") as f:
            raw = f.read(min(file_size, 2880 * 64))
        hdr = fits.Header.fromstring(raw, sep="")
        required = _min_primary_hdu_file_bytes(hdr)
    except Exception as exc:
        return f"truncated or unreadable FITS ({exc})"
    if file_size < required:
        return (
            f"truncated FITS ({file_size} bytes on disk, "
            f"{required} bytes required for primary HDU data)"
        )
    return None


def _filter_invalid_beam_files(
    by_time: Dict[str, List[Path]],
) -> Dict[str, List[Path]]:
    """Drop FITS files with truncated data or a missing/non-positive synthesized beam.

    Files surviving this filter are complete on disk and have a real, finite, strictly
    positive ``BMAJ`` and ``BMIN``. Skipped files are *omitted* from the returned grouping so downstream
    combine and Zarr append leave their ``(time, frequency)`` cells unwritten; on
    append, :func:`xarray.concat` outer-joins the frequency axis and fills missing
    cells with the float ``NaN`` fill value. Time keys whose entire bucket fails the
    check are dropped from the result (with a warning) so the outer convert loop
    does not attempt to combine an empty group.

    Parameters
    ----------
    by_time
        Discovery output mapping observation-time key → list of FITS file paths.

    Returns
    -------
    Dict[str, List[Path]]
        Same shape as ``by_time`` but with invalid-beam files removed; time keys
        whose remaining file list is empty are dropped entirely.
    """
    filtered: Dict[str, List[Path]] = {}
    n_dropped_files = 0
    for tkey, files in by_time.items():
        kept: List[Path] = []
        for fp in files:
            trunc_reason = _truncated_fits_reason(fp)
            if trunc_reason is not None:
                logger.warning(
                    "Skipping %s: %s; its (time=%s, frequency=*) slot will be filled "
                    "with NaN in Zarr.",
                    fp.name,
                    trunc_reason,
                    tkey,
                )
                n_dropped_files += 1
                continue
            try:
                hdr = fits.getheader(fp, ext=0)
            except Exception as exc:
                logger.warning(
                    "Skipping %s: could not read primary header to check beam (%s); "
                    "its (time=%s, frequency=*) slot will be filled with NaN in Zarr.",
                    fp.name,
                    exc,
                    tkey,
                )
                n_dropped_files += 1
                continue
            reason = _invalid_beam_reason(hdr)
            if reason is None:
                kept.append(fp)
            else:
                logger.warning(
                    "Skipping %s: invalid synthesized beam (%s); its (time=%s, "
                    "frequency=*) slot will be filled with NaN in Zarr.",
                    fp.name,
                    reason,
                    tkey,
                )
                n_dropped_files += 1
        if kept:
            filtered[tkey] = kept
        else:
            logger.warning(
                "All FITS files for time=%s dropped because none have a valid "
                "synthesized beam; this time step will not be written.",
                tkey,
            )
    if n_dropped_files:
        logger.info(
            "Discovery quality filter dropped %d file(s) (truncated or invalid beam) "
            "across %d remaining time step(s).",
            n_dropped_files,
            len(filtered),
        )
    return filtered


def _completed_time_keys_in_zarr(
    out_zarr: Path,
    *,
    rebuild: bool,
) -> set[str]:
    """Return the set of ``YYYYMMDD_HHMMSS`` time keys already present in *out_zarr*.

    Used to make ``convert`` / ``dewarp-convert`` re-runs naturally resumable: callers
    drop any time key already covered by the existing store so the expensive read /
    dewarp / append work is skipped on the second pass. The returned key strings use
    the same ``%Y%m%d_%H%M%S`` UTC formatting as :func:`_time_key_from_filename` and
    :func:`_time_key_from_header`, so they can be compared directly against the keys
    of :func:`_discover_groups`.

    Parameters
    ----------
    out_zarr
        Path to the candidate output Zarr store.
    rebuild
        When True, the caller has asked to overwrite the store; this function
        returns an empty set so every discovered time key is processed.

    Returns
    -------
    set[str]
        Sorted-by-time UTC keys already written. Empty when *rebuild* is True,
        when *out_zarr* does not exist, when it lacks a usable ``time`` coord,
        or when the store cannot be opened (a warning is logged in that case).

    Notes
    -----
    * ``time`` values are stored as MJD ``float64`` by ``xradio``; this helper
      converts each finite value back to UTC via :class:`astropy.time.Time` and
      truncates to one-second resolution. Sub-second timestamps in ``DATE-OBS``
      collapse to the same key as the filename ``-image-`` stamp, so resume
      decisions remain consistent across ``group_metadata_source`` values.
    * NaN ``time`` rows (which can appear if a future append step ever writes a
      placeholder row) are ignored rather than producing a spurious key.
    """
    if rebuild or not out_zarr.exists():
        return set()
    try:
        existing = xr.open_zarr(str(out_zarr), consolidated=False)
    except Exception as exc:
        logger.warning(
            "Could not open existing Zarr %s to check completed time keys (%s); "
            "treating all discovered times as new.",
            out_zarr,
            exc,
        )
        return set()
    try:
        if "time" not in existing.coords:
            return set()
        time_values = np.atleast_1d(np.asarray(existing["time"].values))
    finally:
        existing.close()

    keys: set[str] = set()
    for raw in time_values:
        try:
            mjd = float(raw)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(mjd):
            continue
        try:
            t = Time(mjd, format="mjd", scale="utc")
            keys.add(t.to_datetime().strftime("%Y%m%d_%H%M%S"))
        except Exception as exc:
            logger.debug(
                "Could not convert existing Zarr time=%r to a time key (%s); ignoring.",
                raw,
                exc,
            )
    return keys


def _header_time_key_for_files(files: Sequence[Path]) -> Optional[str]:
    """Return the ``DATE-OBS`` time key for the first file in a discovery group."""
    if not files:
        return None
    try:
        header = fits.getheader(files[0], ext=0)
    except OSError as exc:
        logger.debug(
            "Could not read header from %s for resume header-key lookup (%s).",
            files[0],
            exc,
        )
        return None
    return _time_key_from_header(header)


def _discovery_time_key_completed_in_zarr(
    discovery_key: str,
    files: Sequence[Path],
    completed: set[str],
) -> bool:
    """Return whether a discovered time key is already represented in *completed*.

    Discovery often groups by the filename ``-image-YYYYMMDD_HHMMSS`` stamp while
    ``xradio`` writes the Zarr ``time`` coordinate from ``DATE-OBS`` in the FITS
    header. Those strings differ by a systematic offset on OVRO-LWA products, so a
    straight set intersection would treat every row as new and re-dewarp already
    written observations. When the header key for the group is already in the store,
    treat the filename discovery key as complete as well.
    """
    if discovery_key in completed:
        return True
    header_key = _header_time_key_for_files(files)
    return bool(header_key and header_key in completed)


def _filter_completed_time_keys(
    by_time: Dict[str, List[Path]],
    out_zarr: Path,
    *,
    rebuild: bool,
    context: str,
) -> Dict[str, List[Path]]:
    """Drop time keys already present in *out_zarr* to make re-runs resumable.

    A thin wrapper around :func:`_completed_time_keys_in_zarr` that also emits
    consistent log messages so the convert and dewarp-convert call sites stay in
    sync. The original ordering of *by_time* is preserved for the remaining keys.

    When discovery keys come from filenames but the Zarr ``time`` coordinate was
    written from FITS ``DATE-OBS`` (the usual ``xradio`` path), a group is also
    treated as complete when :func:`_header_time_key_for_files` matches a key
    already in the store.

    Parameters
    ----------
    by_time
        Discovery output already passed through :func:`_filter_invalid_beam_files`.
    out_zarr
        Candidate output Zarr store (need not exist yet).
    rebuild
        When True, the caller has asked to overwrite the store; this short-circuits
        the filter and returns *by_time* unchanged.
    context
        Short label for log lines (e.g. ``"convert"``, ``"dewarp-convert"``); helps
        operators distinguish which subcommand reported the resume.

    Returns
    -------
    Dict[str, List[Path]]
        Same shape as *by_time* but with already-completed time keys removed.
    """
    completed = _completed_time_keys_in_zarr(out_zarr, rebuild=rebuild)
    if not completed:
        return by_time
    skipped_exact: list[str] = []
    skipped_header_alias: list[str] = []
    remaining: Dict[str, List[Path]] = {}
    for discovery_key, files in by_time.items():
        if not _discovery_time_key_completed_in_zarr(discovery_key, files, completed):
            remaining[discovery_key] = files
            continue
        if discovery_key in completed:
            skipped_exact.append(discovery_key)
        else:
            skipped_header_alias.append(discovery_key)
    skipped_total = len(skipped_exact) + len(skipped_header_alias)
    if not skipped_total:
        return by_time
    logger.info(
        "[%s] Resume from %s: %d time key(s) already present (%d by discovery key, "
        "%d by FITS header time), %d remaining to process.",
        context,
        out_zarr,
        skipped_total,
        len(skipped_exact),
        len(skipped_header_alias),
        len(remaining),
    )
    if skipped_exact:
        logger.debug(
            "[%s] Skipping discovery keys already in Zarr: %s",
            context,
            ", ".join(sorted(skipped_exact)),
        )
    if skipped_header_alias:
        logger.debug(
            "[%s] Skipping discovery keys whose DATE-OBS is already in Zarr: %s",
            context,
            ", ".join(sorted(skipped_header_alias)),
        )
    return remaining


def _discover_groups(
    in_dir: Path,
    duplicate_resolver: Optional[Callable[[str, float, List[Path]], Path]] = None,
    *,
    freq_bin_hz: float = _DISCOVERY_FREQ_BIN_HZ,
    time_key_source: Literal["header", "filename"] = "filename",
    group_metadata_source: Literal["fits", "filename"] = "fits",
) -> Dict[str, List[Path]]:
    """Group input FITS by observation time and frequency (filename time stamp, header fallback).

    Files are associated with a **coarse** frequency key (default 23~kHz bins) so small
    header differences in Hz (RESTFREQ, etc.) do not create extra ``frequency`` planes
    in the Zarr for the same physical subband. For multiple paths in the same
    (time, bin) without a ``duplicate_resolver``, the first file is kept and the rest
    are skipped (with a warning). Distinct subbands remain separate (e.g. 41~MHz vs 55~MHz).

    Parameters
    ----------
    in_dir : Path
        Directory containing input FITS files.
    duplicate_resolver
        Optional callback ``(time_key, frequency_hz, candidates) -> Path`` when multiple
        files share the same time and binned frequency group.
    freq_bin_hz
        Width in Hz for rounding header frequencies to a discovery key,
        ``int(round(frequency_hz / freq_bin_hz))``. Frequencies in the same bin are treated
        as one subband for grouping (up to ~``freq_bin_hz`` separation at bin edges).
    time_key_source
        Used only when ``group_metadata_source`` is ``"fits"``. ``"filename"`` (default):
        prefer the basename ``-image-YYYYMMDD_HHMMSS`` instant when present; otherwise use
        ``DATE-OBS``. ``"header"``: group by ``DATE-OBS`` time key only.
    group_metadata_source
        ``"fits"`` (default): read FITS headers (and filename fallbacks) via
        :func:`_extract_group_metadata`. ``"filename"``: derive time and frequency for
        grouping **only** from the basename (no FITS I/O); requires ``-image-`` time and
        ``_NNNMHz_`` / ``_NNNMHz-`` tokens when you need frequency-based bins. ``time_key_source``
        is ignored in ``"filename"`` mode.

    Returns
    -------
    Dict[str, List[Path]]
        Dictionary mapping time keys to lists of FITS file paths.
    """
    if freq_bin_hz <= 0.0:
        msg = f"freq_bin_hz must be positive, got {freq_bin_hz}"
        raise ValueError(msg)

    by_time: Dict[str, List[Path]] = {}
    by_time_freq: Dict[str, Dict[int, List[Path]]] = {}
    for f in sorted(in_dir.glob("*.fits")):
        if group_metadata_source == "filename":
            time_key, frequency_hz, notes = _extract_group_metadata_filename_only(f)
        else:
            time_key, frequency_hz, notes = _extract_group_metadata(f, time_key_source=time_key_source)
        if time_key is None:
            if group_metadata_source == "filename":
                t_hint = "-image-YYYYMMDD_HHMMSS in basename (filename-only grouping; no header fallback)"
            elif time_key_source == "filename":
                t_hint = "-image-YYYYMMDD_HHMMSS in basename or DATE-OBS"
            else:
                t_hint = "DATE-OBS"
            logger.warning(f"Skipping {f.name}: missing usable observation time ({t_hint}).")
            continue
        if frequency_hz is None:
            logger.warning(
                f"Could not determine frequency for {f.name}; duplicate detection disabled for this file."
            )
            by_time.setdefault(time_key, []).append(f)
            continue

        freq_key = int(round(frequency_hz / freq_bin_hz))
        time_freq_map = by_time_freq.setdefault(time_key, {})
        candidates = time_freq_map.setdefault(freq_key, [])
        candidates.append(f)

        if len(candidates) > 1:
            duplicate_names = [p.name for p in candidates]
            if duplicate_resolver is None:
                # Same time + same binned subband: stacking would be wrong; keep the first
                # file and drop the rest (typical: symlink pairs or header jitter in Hz).
                kept = candidates[0]
                logger.warning(
                    "Multiple FITS share time=%s and the same %g Hz frequency bin "
                    "(binned key=%s, ~%.3f MHz): %s. Using only %s. "
                    "Remove extras or pass duplicate_resolver to select a file.",
                    time_key,
                    freq_bin_hz,
                    freq_key,
                    frequency_hz / 1e6,
                    duplicate_names,
                    kept.name,
                )
                time_freq_map[freq_key] = [kept]
                continue

            if group_metadata_source == "filename":
                _, rep_hz, _ = _extract_group_metadata_filename_only(candidates[0])
            else:
                _, rep_hz, _ = _extract_group_metadata(candidates[0], time_key_source=time_key_source)
            resolver_hz = float(rep_hz) if rep_hz is not None else float(freq_key) * freq_bin_hz
            selected = duplicate_resolver(time_key, resolver_hz, candidates.copy())
            if selected not in candidates:
                msg = (
                    f"Duplicate resolver returned unknown file {selected} for "
                    f"time={time_key}, frequency_hz={resolver_hz}."
                )
                raise RuntimeError(msg)

            by_time.setdefault(time_key, [])
            by_time[time_key] = [p for p in by_time[time_key] if p not in candidates]
            by_time[time_key].append(selected)
            # Replace the pending duplicate bucket so the next file with this (time, freq)
            # starts from the chosen file only. Otherwise `candidates` never shrinks and
            # grows as [f1, f2, f3, ...], re-invoking the resolver with stale paths and
            # widening the filter that strips `by_time[time_key]`.
            time_freq_map[freq_key] = [selected]
            logger.warning(
                "Duplicate FITS files for time=%s, frequency_hz=%.1f. Selected: %s. Candidates: %s",
                time_key,
                resolver_hz,
                selected.name,
                duplicate_names,
            )
            continue

        if notes:
            logger.warning(f"Using fallback metadata for {f.name}: {', '.join(notes)}")
        by_time.setdefault(time_key, []).append(f)

    sort_key = lambda p: _discovery_frequency_sort_tuple(p, group_metadata_source=group_metadata_source)
    for time_key, files in by_time.items():
        by_time[time_key] = sorted(files, key=sort_key)
    return by_time


def _combine_time_step(
    files: List[Path],
    fixed_dir: Path,
    *,
    chunk_lm: int,
    fix_headers_on_demand: bool = True,
    lm_reference_ds: Optional[xr.Dataset] = None,
    group_metadata_source: Literal["fits", "filename"] = "fits",
) -> Tuple[xr.Dataset, List[float], List[Path]]:
    """Create a single-time dataset by combining frequency slices from subbands.

    Parameters
    ----------
    files : List[Path]
        List of FITS file paths for a single time step.
    fixed_dir : Path
        Directory to place generated ``*_fixed.fits`` files.
    chunk_lm : int
        LM chunk size for in-memory xarray datasets.
    fix_headers_on_demand : bool, optional
        If True, fix headers on-demand if they don't exist. If False,
        assume headers are already fixed. Default is True.
    lm_reference_ds : xarray.Dataset, optional
        If provided, regrid all slices to this dataset's ``l``/``m`` grid (used for
        a conversion-wide max-shape reference). If omitted, the largest slice in
        this time step is the reference.
    group_metadata_source
        Same as :func:`_discover_groups` for deterministic frequency ordering when fixing
        headers or resolving ``*_fixed.fits`` paths.

    Returns
    -------
    Tuple[xr.Dataset, List[float], List[Path]]
        Tuple of (combined dataset, sorted list of unique frequencies in Hz,
        newly-created ``*_fixed.fits`` paths for optional cleanup).
    """
    created_fixed_paths: List[Path] = []
    sort_key = lambda p: _discovery_frequency_sort_tuple(p, group_metadata_source=group_metadata_source)
    if fix_headers_on_demand:
        existed_before: Dict[Path, bool] = {}
        for f in sorted(files, key=sort_key):
            if f.name.endswith("_fixed.fits"):
                continue
            candidate = fixed_dir / (f.stem + "_fixed.fits")
            existed_before[candidate] = candidate.exists()
        # Fix headers if needed (skips existing fixed files)
        fixed_paths = fix_fits_headers(
            files, fixed_dir, skip_existing=True, group_metadata_source=group_metadata_source
        )
        for f in sorted(files, key=sort_key):
            if f.name.endswith("_fixed.fits"):
                continue
            candidate = fixed_dir / (f.stem + "_fixed.fits")
            if not existed_before.get(candidate, False) and candidate.exists():
                created_fixed_paths.append(candidate)
    else:
        # Just get the paths to already-fixed files
        fixed_paths = _get_fixed_paths(files, fixed_dir, group_metadata_source=group_metadata_source)

    xds_list: List[xr.Dataset] = []
    freqs_seen: List[float] = []
    for fp in fixed_paths:
        xds = _load_for_combine(fp, chunk_lm=chunk_lm)
        xds = _assign_canonical_frequency_for_stack(
            xds, fp, group_metadata_source=group_metadata_source
        )
        fvals = np.atleast_1d(xds.frequency.values)
        freqs_seen.extend([float(f) for f in fvals])
        xds_list.append(xds)

    lm_shapes = [_lm_shape(xds) for xds in xds_list]
    reference_idx = _select_reference_shape_index(lm_shapes)
    unique_shapes = sorted(set(lm_shapes))

    if lm_reference_ds is not None:
        ref_ds = lm_reference_ds
        if len(unique_shapes) > 1 or any(_lm_shape(xds) != _lm_shape(ref_ds) for xds in xds_list):
            logger.info(
                "Using global LM reference grid (l,m)=%s; this time step shapes: %s",
                _lm_shape(ref_ds),
                unique_shapes,
            )
    else:
        ref_ds = xds_list[reference_idx]
        if len(unique_shapes) > 1:
            logger.info(
                "Detected mixed LM shapes %s; selected reference shape %s from %s",
                unique_shapes,
                lm_shapes[reference_idx],
                fixed_paths[reference_idx].name,
            )
    for i, xds in enumerate(xds_list):
        if _lm_shape(xds) != _lm_shape(ref_ds):
            logger.info(
                "Regridding %s from LM shape %s onto reference %s",
                fixed_paths[i].name,
                _lm_shape(xds),
                _lm_shape(ref_ds),
            )
        xds_list[i] = _regrid_to_reference_lm(xds, ref_ds, source_label=str(fixed_paths[i]))

    try:
        xds_t = xr.combine_by_coords(
            xds_list,
            combine_attrs="drop",
            data_vars="minimal",
            coords="minimal",
            compat="no_conflicts",
        )
    except Exception:
        # Fallback requires each subband to have frequency size == 1
        for ds in xds_list:
            if "frequency" in ds.dims and ds.sizes["frequency"] != 1:
                msg = "A subband has frequency dimension != 1; cannot concat."
                raise RuntimeError(msg)
        xds_t = xr.concat(xds_list, dim="frequency")

    if "frequency" in xds_t.coords:
        fv = np.asarray(xds_t["frequency"].values, dtype=np.float64).ravel()
        if fv.size > 1 and len(np.unique(fv)) < len(fv):
            names = ", ".join(p.name for p in fixed_paths)
            msg = (
                "Duplicate ``frequency`` coordinate values after stacking subbands for one "
                "time step (``xarray`` cannot sort or write this). Usually dewarped FITS share "
                "the same RESTFREQ/CRVAL3 while basenames omit distinct ``_NNNMHz_`` tags. "
                f"Files: {names}"
            )
            raise RuntimeError(msg)
        xds_t = xds_t.sortby("frequency")
    if "time" in xds_t.coords:
        xds_t = xds_t.sortby("time")

    xds_t.attrs = {}
    for v in xds_t.data_vars:
        xds_t[v].encoding = {}

    xds_t = _harmonize_celestial_coords_independent_of_frequency(xds_t)

    xds_t = _rechunk_lm_for_zarr(xds_t, chunk_lm)

    return xds_t, sorted(set(freqs_seen)), created_fixed_paths


def _rechunk_nonuniform_aux_vars_for_zarr(xds: xr.Dataset) -> xr.Dataset:
    """Rechunk metadata-sized data vars so Dask chunks satisfy Zarr's uniformity rule.

    ``xr.concat`` / ``combine_by_coords`` can leave variables that only span
    ``frequency`` (or similar) with *irregular* Dask chunks (e.g. ``(2, 1, 1)``
    along one dimension). ``xarray``'s Zarr writer requires all non-final
    chunks to share the same size along each dimension; otherwise it raises
    (``wcs_header_str`` is a known case).

    Parameters
    ----------
    xds
        Dataset possibly containing small dask-backed aux variables.

    Returns
    -------
    xarray.Dataset
        Dataset with offending variables rechunks to one chunk per dimension.
    """
    out = xds
    for name in list(out.data_vars):
        v = out[name]
        da_ = v.data
        if not hasattr(da_, "chunks") or not da_.chunks:
            continue
        bad = False
        for dim_chunks in da_.chunks:
            if len(dim_chunks) > 1 and len(set(dim_chunks[:-1])) > 1:
                bad = True
                break
        if bad:
            chunk_arg = {d: -1 for d in v.dims}
            out = out.assign(**{name: v.chunk(chunk_arg)})
    return out


def _rechunk_nonuniform_coords_for_zarr(xds: xr.Dataset) -> xr.Dataset:
    """Rechunk coordinate arrays that have non-uniform Dask chunks.

    Coordinates like ``right_ascension``/``declination`` can gain a leading
    ``time`` dimension during append. If that axis chunks as ``(2, 1, 1)``,
    xarray's Zarr writer rejects it as non-uniform. Rechunk only the offending
    dimensions (e.g. ``time``) and keep already-uniform spatial chunks unchanged.
    """
    out = xds
    for name in list(out.coords):
        c = out[name]
        da_ = c.data
        if not hasattr(da_, "chunks") or not da_.chunks:
            continue
        bad_dims: list[str] = []
        for dim_name, dim_chunks in zip(c.dims, da_.chunks, strict=True):
            if len(dim_chunks) > 1 and len(set(dim_chunks[:-1])) > 1:
                bad_dims.append(dim_name)
        if bad_dims:
            chunk_arg = {d: -1 for d in bad_dims}
            out = out.assign_coords({name: c.chunk(chunk_arg)})
    return out


def _strip_encodings_for_zarr_write(xds: xr.Dataset) -> xr.Dataset:
    """Clear encodings on coordinates and data variables before Zarr write.

    After ``Dataset.chunk`` and ``concat``, xarray may attach ``encoding['chunks']``
    to coordinates (e.g. ``right_ascension``, ``declination``). If those encodings
    do not align with the current Dask chunk boundaries, ``to_zarr`` raises
    *Specified Zarr chunks … would overlap multiple Dask chunks*. Stripping
    encodings matches the pattern already used for data variables after
    :func:`_combine_time_step` and lets the writer derive chunks from the arrays.
    """
    for name in xds.coords:
        xds[name].encoding = {}
    for name in xds.data_vars:
        xds[name].encoding = {}
    return xds


def _rechunk_lm_for_zarr(xds: xr.Dataset, chunk_lm: int) -> xr.Dataset:
    """Rechunk ``l`` and ``m`` so Dask-backed arrays use uniform spatial chunk sizes.

    ``combine_by_coords`` / ``concat`` can fuse slices into irregular chunk
    boundaries along ``l``/``m``. Zarr encoding (via xradio) requires uniform
    chunk sizes per dimension except possibly the final chunk.

    Parameters
    ----------
    xds
        Dataset whose spatial dimensions are named ``l`` and ``m``.
    chunk_lm
        Target chunk length for each of ``l`` and ``m``. If zero, each spatial
        axis is stored as a single chunk (still uniform).
    """
    if {"l", "m"} <= set(xds.dims):
        if chunk_lm and chunk_lm > 0:
            xds = xds.chunk({"l": chunk_lm, "m": chunk_lm})
        else:
            xds = xds.chunk({"l": -1, "m": -1})
    xds = _rechunk_nonuniform_aux_vars_for_zarr(xds)
    xds = _rechunk_nonuniform_coords_for_zarr(xds)
    xds = _strip_encodings_for_zarr_write(xds)
    return xds


def _assert_same_lm(
    reference: Tuple[NDArray[np.floating], NDArray[np.floating]],
    current: Tuple[NDArray[np.floating], NDArray[np.floating]],
) -> None:
    """Ensure the LM grids match across time steps.

    Parameters
    ----------
    reference : Tuple[NDArray[np.floating], NDArray[np.floating]]
        Reference (l, m) grid arrays.
    current : Tuple[NDArray[np.floating], NDArray[np.floating]]
        Current (l, m) grid arrays to compare.

    Raises
    ------
    RuntimeError
        If l or m grids differ across time steps.
    """
    ref_l, ref_m = reference[0], reference[1]
    cur_l, cur_m = current[0], current[1]
    if ref_l.shape != cur_l.shape or ref_m.shape != cur_m.shape:
        msg = (
            "l/m coordinate length mismatch across time steps "
            f"(l: {ref_l.shape} vs {cur_l.shape}, m: {ref_m.shape} vs {cur_m.shape}). "
            "After mixed-resolution normalization, every time step must share one LM grid."
        )
        raise RuntimeError(msg)
    same_l = np.allclose(ref_l, cur_l)
    same_m = np.allclose(ref_m, cur_m)
    if not (same_l and same_m):
        raise RuntimeError(
            "l/m grids differ across times after normalization; aborting to avoid misalignment."
        )


def _ensure_append_friendly_time_chunks(xds: xr.Dataset) -> xr.Dataset:
    """Use one Zarr chunk per time index so ``append_dim='time'`` and region overwrites align."""
    if "time" not in xds.dims:
        return xds
    return xds.chunk({"time": 1})


def _time_coord_as_dimension(ds: xr.Dataset) -> xr.DataArray | None:
    """Return a length-1 ``time`` coordinate array when ``time`` is present on *ds*."""
    if "time" not in ds.coords:
        return None
    t = ds.coords["time"]
    if "time" in t.dims:
        return t
    return xr.DataArray(np.atleast_1d(np.asarray(t.values)), dims=("time",))


def _is_simple_index_coordinate(ds: xr.Dataset, name: str) -> bool:
    """True when *name* is a 1D dimension coordinate indexed by itself (e.g. ``frequency``, ``l``)."""
    if name not in ds.coords or name not in ds.dims:
        return False
    dims = ds.variables[name].dims
    return len(dims) == 1 and dims[0] == name


def _expand_leading_time_dim(
    da: xr.DataArray,
    *,
    time_coord: xr.DataArray,
    target_dims: tuple[str, ...],
) -> xr.DataArray:
    """Add a leading ``time`` dimension when *target_dims* expects one and *da* lacks it."""
    if "time" in da.dims or not target_dims or target_dims[0] != "time":
        return da
    if da.dims != target_dims[1:]:
        return da
    if "time" in time_coord.dims:
        return da.expand_dims(time=time_coord)
    return da.expand_dims(time=np.atleast_1d(np.asarray(time_coord.values)))


def _align_time_dimension_for_zarr_write(
    ds: xr.Dataset,
    *,
    schema: xr.Dataset | None = None,
) -> xr.Dataset:
    """Align per-time auxiliary variables with the Zarr store's ``time`` axis.

    xradio's ``velocity`` (and similar metadata) is often frequency-only
    ``(frequency,)`` even when image data carry a ``time`` coordinate. Earlier
    time steps in the same store may have been written with a leading ``time``
    dimension on those variables; ``to_zarr(..., append_dim='time')`` then rejects
    mismatched dimension names. When *schema* is given (append to an existing
    store), any variable overlapping the on-disk store (data vars or coords) is
    expanded to match *schema*. On the first write, frequency-only aux vars are
    promoted to ``(time, frequency)`` when ``time`` is
    a coordinate so later appends stay compatible. Scalar metadata such as
    ``wcs_header_str`` are promoted to ``(time,)`` so incremental appends do not
    leave stale ``encoding['chunks']`` on a length-1 slice that disagrees with the
    store.
    """
    time_coord = _time_coord_as_dimension(ds)
    if time_coord is None:
        return ds

    target_dims_by_name: dict[str, tuple[str, ...]] = {}
    if schema is not None:
        for name in ds.variables:
            if name not in schema.variables:
                continue
            incoming_dims = tuple(ds.variables[name].dims)
            if "time" in incoming_dims:
                continue
            schema_dims = tuple(schema.variables[name].dims)
            if schema_dims and schema_dims[0] == "time":
                target_dims_by_name[name] = schema_dims
    else:
        for name, da in ds.variables.items():
            if _is_simple_index_coordinate(ds, name):
                continue
            if "frequency" in da.dims and "time" not in da.dims:
                target_dims_by_name[name] = ("time", *tuple(da.dims))
            elif da.dims == ():
                target_dims_by_name[name] = ("time",)

    out = ds
    for name, target_dims in target_dims_by_name.items():
        da = out[name]
        if target_dims == ("time",) and "frequency" in da.dims:
            da = da.isel(frequency=0, drop=True)
        expanded = _expand_leading_time_dim(
            da, time_coord=time_coord, target_dims=target_dims
        )
        if expanded is not da:
            if name in out.data_vars:
                out = out.assign({name: expanded})
            else:
                out = out.assign_coords({name: expanded})
    return out


def _prepare_time_only_vars_for_zarr_write(xds: xr.Dataset) -> xr.Dataset:
    """Rechunk time-only metadata (e.g. ``wcs_header_str``) for safe Zarr append.

    After many ``append_dim='time'`` writes, a store can carry
    ``encoding['chunks']=(n_time,)`` while each new slice only spans one index.
    Materializing and rechunks to ``time=1`` avoids Dask/Zarr overlap errors.
    """
    out = xds
    for name in list(out.data_vars):
        da = out[name]
        if da.dims != ("time",):
            continue
        prepared = da.load()
        if hasattr(prepared.data, "rechunk"):
            prepared = prepared.chunk({"time": 1})
        prepared.encoding = {}
        out = out.assign({name: prepared})
    return out


def _write_or_append_zarr(
    xds_t: xr.Dataset,
    out_zarr: Path,
    first_write: bool,
    *,
    chunk_lm: int,
) -> None:
    """Write or append one time step to a Zarr store without re-writing existing times.

    First write replaces any existing store at *out_zarr* when *first_write* is True.
    Append uses ``to_zarr(..., mode='a', append_dim='time')``. If the new step's
    ``time`` already exists in the store (e.g. resume repair), that row is
    overwritten via a region write (``keep='last'`` semantics).
    """
    from shutil import rmtree

    if first_write or not out_zarr.exists():
        to_write = _align_time_dimension_for_zarr_write(xds_t)
    else:
        with xr.open_zarr(str(out_zarr), consolidated=False) as existing:
            to_write = _align_time_dimension_for_zarr_write(xds_t, schema=existing)

    to_write = _rechunk_lm_for_zarr(to_write, chunk_lm)
    to_write = _ensure_append_friendly_time_chunks(to_write)
    to_write = _prepare_time_only_vars_for_zarr_write(to_write)
    if "time" in to_write.coords:
        to_write = to_write.sortby("time")
    if "frequency" in to_write.coords:
        to_write = to_write.sortby("frequency")

    if first_write or not out_zarr.exists():
        if out_zarr.exists():
            rmtree(out_zarr)
        to_write.to_zarr(str(out_zarr), mode="w", consolidated=False)
        return

    with xr.open_zarr(str(out_zarr), consolidated=False) as existing:
        if "time" not in existing.coords:
            raise RuntimeError(
                f"Cannot append: existing Zarr {out_zarr} has no time coordinate."
            )
        old_times = np.atleast_1d(np.asarray(existing["time"].values, dtype=np.float64))
    new_times = np.atleast_1d(np.asarray(to_write["time"].values, dtype=np.float64))
    if new_times.size != 1:
        raise RuntimeError(
            f"Incremental append expects exactly one time index; got {new_times.size}."
        )
    new_t = float(new_times[0])
    dup_idx: int | None = None
    for i, old in enumerate(old_times):
        if not (np.isfinite(old) and np.isfinite(new_t)):
            continue
        if np.isclose(old, new_t, rtol=1e-12, atol=1e-9):
            dup_idx = i

    if dup_idx is not None:
        logger.info(
            "Overwriting existing time row %d (MJD %.8f) in %s; n_time will not grow.",
            dup_idx,
            new_t,
            out_zarr,
        )
        # ``to_zarr(..., region=...)`` rejects index coordinates that lack ``time``
        # (e.g. ``l``, ``m``, ``frequency``). Build a minimal dataset from only
        # arrays that span ``time`` (materialized) so the repair path matches the
        # on-disk layout without re-exporting static coords.
        data_map: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        for name, da in to_write.data_vars.items():
            if "time" not in da.dims:
                continue
            arr = da.load()
            data_map[name] = (tuple(arr.dims), np.asarray(arr.data))
        time_vals = np.asarray(to_write["time"].load().values)
        dup_write = xr.Dataset(data_map, coords={"time": ("time", time_vals)})
        dup_write = _ensure_append_friendly_time_chunks(dup_write)
        dup_write.to_zarr(
            str(out_zarr),
            mode="r+",
            region={"time": slice(dup_idx, dup_idx + 1)},
            consolidated=False,
            align_chunks=True,
        )
        return

    logger.info(
        "Appending new time row at MJD %.8f to %s (n_time will increase by 1).",
        new_t,
        out_zarr,
    )
    to_write.to_zarr(
        str(out_zarr),
        mode="a",
        append_dim="time",
        consolidated=False,
        align_chunks=True,
    )


def convert_fits_dir_to_zarr(
    input_dir: str | Path,
    out_dir: str | Path,
    zarr_name: str = "ovro_lwa_full_lm_only.zarr",
    fixed_dir: str | Path = "fixed_fits",
    chunk_lm: int = 1024,
    rebuild: bool = False,
    resume: bool = True,
    fix_headers_on_demand: bool = True,
    cleanup_fixed_fits: bool = False,
    progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    duplicate_resolver: Optional[Callable[[str, float, List[Path]], Path]] = None,
    discovery_freq_bin_hz: float = _DISCOVERY_FREQ_BIN_HZ,
    time_keys_only: Optional[Sequence[str]] = None,
    lm_reference_ds: Optional[xr.Dataset] = None,
    lm_reference_target_size: int | None = None,
    group_metadata_source: Literal["fits", "filename"] = "fits",
    time_key_source: Literal["header", "filename"] = "filename",
) -> Path:
    """Convert all matching FITS in a directory into a single LM-only Zarr store.

    Parameters
    ----------
    input_dir
        Directory containing input FITS files.
    out_dir
        Directory where the Zarr store will be written.
    zarr_name
        Name of the Zarr store directory (under ``out_dir``).
    fixed_dir
        Directory to place generated ``*_fixed.fits`` files.
    chunk_lm
        Optional LM chunk size for the in-memory xarray datasets (0 disables).
    rebuild
        If True, overwrite any existing Zarr; otherwise append to it.
    fix_headers_on_demand
        If True, fix FITS headers on-demand during conversion if they don't exist.
        If False, assume headers are already fixed using :func:`fix_fits_headers`.
        Default is True.
    cleanup_fixed_fits
        If True (and ``fix_headers_on_demand`` is enabled), delete temporary
        ``*_fixed.fits`` files created during each time-step after writing that
        step to Zarr. Use this to reduce peak disk usage.
    progress_callback
        Optional callback function for progress reporting. Should accept
        (stage: str, current: int, total: int, message: str).
    duplicate_resolver
        Optional callback to resolve duplicate files that map to the same
        time/frequency group. Signature: ``(time_key, frequency_hz, candidates) -> selected_path``.
    discovery_freq_bin_hz
        Bin width in Hz for treating header frequencies as the same subband during
        discovery (default 23~kHz). Must be positive.
    time_keys_only
        If set, only time keys in this collection are converted (after discovery).
        Used for incremental pipelines (e.g. dewarp one time step, then append Zarr).
    lm_reference_ds
        If provided, skip the global LM reference scan and use this dataset instead.
        Must match the grid chosen for the full run (typically built once from the
        same input layout before dewarping). Callers should pass a deep-copied dataset
        if the same object might be mutated elsewhere.
    lm_reference_target_size
        When building the global LM reference (``lm_reference_ds`` is None), reproject
        that reference onto this square grid size. Use the same value as dewarp
        ``target_size`` so staged FITS and the Zarr LM grid stay aligned.
    group_metadata_source
        ``"fits"`` (default): discover and sort by time/frequency using FITS headers
        (with filename fallbacks) as in :func:`_discover_groups`. ``"filename"``: group
        and order files using only basename ``-image-`` time and ``_NNNMHz_`` tokens,
        avoiding ``fits.getheader`` during discovery and frequency sorting.
    time_key_source
        Used when ``group_metadata_source`` is ``"fits"``. ``"filename"`` (default):
        prefer ``-image-YYYYMMDD_HHMMSS`` in the basename, else ``DATE-OBS``.
        ``"header"``: use ``DATE-OBS`` only.
    resume
        When True (default) and *rebuild* is False, skip time keys already present in
        an existing output Zarr via :func:`_filter_completed_time_keys`. Set False to
        reprocess every discovered time (overwriting rows with the same MJD).

    After discovery, files whose primary FITS header is missing or has a zero/negative
    ``BMAJ``/``BMIN`` are dropped via :func:`_filter_invalid_beam_files`. Their
    ``(time, frequency)`` cells are left unwritten so the Zarr append step's outer
    join fills them with the float ``NaN`` fill value instead of contaminating the
    store with placeholder zeros.

    When *resume* is True (default) and *rebuild* is False, time keys already present
    in the store are skipped via :func:`_filter_completed_time_keys`. Re-invoking
    with the same arguments therefore continues an interrupted run. Pass
    ``resume=False`` to reprocess every discovered time. Use ``rebuild=True`` to
    replace the entire Zarr store. If every discovered time key is already present,
    the function returns the existing Zarr path without writing.
    :func:`_write_or_append_zarr` overwrites an existing ``time`` row in-place when
    the same MJD timestamp is written again (``keep='last'`` semantics).

    Within each observation time step, after subbands are stacked along ``frequency``,
    ``right_ascension`` / ``declination`` are reduced to a single ``(l, m)`` celestial
    frame taken from the lowest-frequency slice. If sampled on-sky positions in other
    slices disagree with that reference by more than about one arcminute, a warning
    is logged.

    Mixed-resolution inputs (different ``l``/``m`` pixel shapes) are supported: the
    largest LM grid among all selected files becomes the conversion-wide reference,
    and smaller images are linearly interpolated onto that grid before combine. The
    same reference is used for every time step so output Zarr has one consistent
    sky pixel grid.

    Returns
    -------
    Path
        Path to the resulting Zarr store directory.

    Raises
    ------
    FileNotFoundError
        If no matching FITS files are found.
    RuntimeError
        If LM grids differ across time steps.
    """
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_dir = Path(fixed_dir)
    fixed_dir.mkdir(parents=True, exist_ok=True)
    out_zarr = out_dir / zarr_name

    if discovery_freq_bin_hz <= 0.0:
        msg = f"discovery_freq_bin_hz must be positive, got {discovery_freq_bin_hz}"
        raise ValueError(msg)
    if lm_reference_target_size is not None and int(lm_reference_target_size) <= 0:
        msg = f"lm_reference_target_size must be positive, got {lm_reference_target_size}"
        raise ValueError(msg)

    by_time = _discover_groups(
        input_dir,
        duplicate_resolver=duplicate_resolver,
        freq_bin_hz=discovery_freq_bin_hz,
        time_key_source=time_key_source,
        group_metadata_source=group_metadata_source,
    )
    if time_keys_only is not None:
        allowed = {str(k) for k in time_keys_only}
        by_time = {k: v for k, v in by_time.items() if k in allowed}
        missing = allowed - set(by_time.keys())
        if missing:
            logger.warning(
                "time_keys_only contained keys with no FITS in %s: %s",
                input_dir,
                ", ".join(sorted(missing)),
            )
    by_time = _filter_invalid_beam_files(by_time)
    total_files = sum(len(v) for v in by_time.values())
    logger.info(f"Discovered {total_files} FITS across {len(by_time)} time step(s).")
    for k, v in by_time.items():
        if group_metadata_source == "filename":
            freqs_hz = [_extract_group_metadata_filename_only(p)[1] for p in v]
        else:
            freqs_hz = [_extract_group_metadata(p)[1] for p in v]
        logger.info(f"  time {k}: {len(v)} file(s), frequencies (Hz): {freqs_hz}")

    if not by_time:
        raise FileNotFoundError(
            f"No matching FITS found in {input_dir} (none passed discovery and the "
            "BMAJ/BMIN beam-validity filter)."
        )

    by_time_for_global_freq = dict(by_time)

    if resume and not rebuild:
        by_time = _filter_completed_time_keys(
            by_time, out_zarr, rebuild=False, context="convert"
        )
        if not by_time:
            logger.info(
                "Nothing to do: every discovered time key is already present in %s. "
                "Pass rebuild=True to overwrite the store, or resume=False to reprocess all times.",
                out_zarr,
            )
            return out_zarr

    if _zarr_store_exists(out_zarr) and not rebuild:
        freq_coord_hz = _frequency_coord_hz_from_zarr(out_zarr)
        logger.info(
            "Using frequency axis from existing Zarr (%d channel(s)).",
            int(freq_coord_hz.size),
        )
    else:
        freq_coord_hz = _global_frequency_coord_hz(
            by_time_for_global_freq,
            group_metadata_source=group_metadata_source,
        )
        logger.info(
            "Built global frequency axis: %d channel(s), %.3f–%.3f MHz",
            int(freq_coord_hz.size),
            float(np.min(freq_coord_hz)) / 1e6,
            float(np.max(freq_coord_hz)) / 1e6,
        )

    if lm_reference_ds is not None:
        lm_ref_ds = lm_reference_ds
    elif _zarr_store_exists(out_zarr) and not rebuild:
        lm_ref_ds = _lm_reference_from_existing_zarr(out_zarr)
        logger.info("LM (l, m) reference grid loaded from existing Zarr for resume.")
    else:
        lm_ref_ds = _load_global_lm_reference_dataset(
            by_time,
            fixed_dir,
            chunk_lm=chunk_lm,
            fix_headers_on_demand=fix_headers_on_demand,
            target_size=lm_reference_target_size,
            group_metadata_source=group_metadata_source,
        )
    lm_reference = (lm_ref_ds["l"].values.copy(), lm_ref_ds["m"].values.copy())

    # Decide whether we write a fresh store or append to an existing one
    first_write = not (out_zarr.exists() and not rebuild)

    total_time_steps = len(by_time)
    for idx, tkey in enumerate(sorted(by_time.keys())):
        files = by_time[tkey]

        logger.info(f"[read/combine] time {tkey}")
        xds_t, freqs, created_fixed_paths = _combine_time_step(
            files,
            fixed_dir,
            chunk_lm=chunk_lm,
            fix_headers_on_demand=fix_headers_on_demand,
            lm_reference_ds=lm_ref_ds,
            group_metadata_source=group_metadata_source,
        )
        xds_t = _align_time_step_to_frequency_grid(xds_t, freq_coord_hz)
        logger.info(f"  combined dims: {dict(xds_t.sizes)}")
        logger.info(f"  combined freqs (Hz): {freqs[:8]}{' ...' if len(freqs) > 8 else ''}")

        lm_current = (xds_t["l"].values, xds_t["m"].values)
        _assert_same_lm(lm_reference, lm_current)
        logger.info("  l/m grid matches global reference")

        logger.info(f"[{'write new' if first_write else 'append'}] {out_zarr}")
        _write_or_append_zarr(xds_t, out_zarr, first_write=first_write, chunk_lm=chunk_lm)
        first_write = False
        if cleanup_fixed_fits and fix_headers_on_demand and created_fixed_paths:
            removed = 0
            for fixed_path in created_fixed_paths:
                try:
                    fixed_path.unlink(missing_ok=True)
                    removed += 1
                except OSError as exc:
                    logger.warning("Could not remove temporary fixed FITS %s: %s", fixed_path, exc)
            logger.info("Cleaned up %d temporary fixed FITS file(s) for time %s", removed, tkey)

        # Report progress after completing this time step
        if progress_callback:
            progress_callback(
                "converting",
                idx + 1,
                total_time_steps,
                f"Completed time step {idx + 1}/{total_time_steps}"
            )

    logger.info(f"[done] All times appended into: {out_zarr}")
    return out_zarr
