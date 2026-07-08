"""xarray accessor for OVRO-LWA radio astronomy datasets.

This module provides the `radport` accessor that extends xarray Datasets
with domain-specific methods for OVRO-LWA data visualization and analysis.

Example
-------
>>> import ovro_lwa_portal
>>> from ovro_lwa_portal import open_dataset
>>> ds = open_dataset("path/to/data.zarr")
>>> ds.radport.plot()  # Create default visualization
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import interpolate
from scipy.optimize import least_squares

if TYPE_CHECKING:
    from collections.abc import Generator

    from matplotlib.figure import Figure

RadportProgressCallback = Callable[[str, int, int, str], None]


# Results smaller than this are eagerly loaded to avoid dask graph
# scheduling overhead.  10 MB covers all realistic OVRO-LWA single-pixel
# extractions (e.g. 1000 times x 1000 freqs x 8 bytes = 8 MB) while
# keeping pathologically large results lazy.
_EAGER_LOAD_THRESHOLD = 10 * 1024 * 1024  # bytes

# :meth:`_compute_pixel_track_batched_radec_grid` loads RA/Dec in
# ``(time_chunk, l, m)`` blocks.  Limits:
#   * ``_MAX_PIXEL_TRACK_PLANE_ELEMENTS`` — max ``l * m`` for one time slice; if
#     larger, fall back to per-time :meth:`_compute_pixel_at_time`.
#   * ``_MAX_PIXEL_TRACK_CHUNK_ELEMENTS`` — max ``time_chunk * l * m`` per Dask
#     ``compute`` (per field); two fields are computed together per chunk.
_MAX_PIXEL_TRACK_PLANE_ELEMENTS = 72_000_000
_MAX_PIXEL_TRACK_CHUNK_ELEMENTS = 40_000_000


def _decode_wcs_header_bytes(raw: object) -> str:
    """Decode a scalar WCS header payload from Zarr (bytes or str)."""
    while isinstance(raw, np.ndarray):
        raw = raw.item()
    if raw is None:
        return ""
    if isinstance(raw, (float, np.floating)) and not np.isfinite(raw):
        # Frequency reindex used ``fill_value=np.nan`` on ``fits_header_str`` in older
        # ingests; treat as empty rather than the literal FITS card ``nan``.
        return ""
    if isinstance(raw, (bytes, bytearray)) or type(raw).__name__ == "bytes_":
        return raw.decode("utf-8", errors="replace").rstrip("\x00")
    text = str(raw).rstrip("\x00")
    if text.lower() == "nan":
        return ""
    return text


def _has_fits_header_str(ds: xr.Dataset) -> bool:
    """True when the dataset carries persisted ``fits_header_str`` metadata."""
    return "fits_header_str" in ds.data_vars


def _has_per_time_wcs_header_str(ds: xr.Dataset) -> bool:
    """True when per-time FITS headers are stored for portal WCS lookup."""
    for var_name in ("fits_header_str", "wcs_header_str"):
        if var_name not in ds:
            continue
        header_var = ds[var_name]
        if "time" not in header_var.dims or int(header_var.sizes.get("time", 0)) <= 0:
            continue
        if header_var.ndim == 1:
            return True
        if header_var.ndim >= 2 and "frequency" in header_var.dims:
            return True
    return False


def strip_redundant_fits_wcs_header_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Remove static ``fits_wcs_header`` attrs when per-slice headers are canonical.

    Incremental Zarr ingest stores one header per time in ``fits_header_str`` (or
    legacy ``wcs_header_str``) but array-level ``fits_wcs_header`` on ``SKY`` only
    reflects the first write. Drop those attrs on load and before Zarr export so
    consumers never prefer a frozen time-0 phase center over per-time headers.
    """
    if "fits_header_str" not in ds.data_vars and "wcs_header_str" not in ds.data_vars:
        return ds
    out = ds.copy(deep=False)
    out.attrs.pop("fits_wcs_header", None)
    for name in out.data_vars:
        out[name].attrs.pop("fits_wcs_header", None)
    for name in out.coords:
        out.coords[name].attrs.pop("fits_wcs_header", None)
    return out


def _read_fits_header_str(
    ds: xr.Dataset,
    *,
    time_idx: int = 0,
    freq_idx: int = 0,
    pol_idx: int = 0,
) -> str:
    """Return the persisted full FITS primary header string for one slice."""
    if not _has_fits_header_str(ds):
        msg = (
            "Dataset is missing fits_header_str. Re-ingest from original FITS files "
            "to enable FITS export and per-slice WCS."
        )
        raise ValueError(msg)

    fh = ds["fits_header_str"]
    sel = fh
    if "time" in fh.dims:
        sel = sel.isel(time=time_idx)
    if "frequency" in sel.dims:
        sel = sel.isel(frequency=freq_idx)
    if "polarization" in sel.dims:
        sel = sel.isel(polarization=pol_idx)
    hdr = _decode_wcs_header_bytes(sel.values)
    if not hdr.strip():
        msg = (
            f"fits_header_str is empty for time_idx={time_idx}, "
            f"freq_idx={freq_idx}, pol_idx={pol_idx}."
        )
        raise ValueError(msg)
    return hdr


def _read_wcs_header_str(
    ds: xr.Dataset,
    *,
    var: str = "SKY",
    time_idx: int = 0,
    freq_idx: int = 0,
    pol_idx: int = 0,
) -> str | None:
    """Return the celestial WCS header string derived from ``fits_header_str``.

    When ``fits_header_str`` is stored per ``time`` step (incremental Zarr),
    that header is preferred over static ``fits_wcs_header`` attrs so each
    slice keeps its own phase-center CRVAL. Empty per-time entries do **not**
    fall back to static attrs (that would mis-register late time slices).
    """
    if _has_fits_header_str(ds):
        try:
            from astropy.io.fits import Header
            from astropy.wcs import WCS

            full_hdr = _read_fits_header_str(
                ds, time_idx=time_idx, freq_idx=freq_idx, pol_idx=int(pol_idx)
            )
            from astropy.wcs import FITSFixedWarning

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FITSFixedWarning)
                wcs = WCS(Header.fromstring(full_hdr, sep="\n"))
            if not wcs.has_celestial:
                return None
            return wcs.celestial.to_header(relax=True).tostring(sep="\n")
        except ValueError:
            return None

    if _has_per_time_wcs_header_str(ds) and "wcs_header_str" in ds:
        wcs_var = ds["wcs_header_str"]
        sel = wcs_var.isel(time=time_idx)
        if "frequency" in sel.dims:
            sel = sel.isel(frequency=freq_idx)
        if "polarization" in sel.dims:
            sel = sel.isel(polarization=int(pol_idx))
        hdr = _decode_wcs_header_bytes(sel.values)
        if hdr.strip():
            return hdr
        return None

    hdr_str = None
    if var in ds.data_vars:
        hdr_str = ds[var].attrs.get("fits_wcs_header")
    if not hdr_str:
        hdr_str = ds.attrs.get("fits_wcs_header")
    if hdr_str is not None:
        return str(hdr_str)

    if "wcs_header_str" not in ds:
        return None

    wcs_var = ds["wcs_header_str"]
    if wcs_var.ndim == 0:
        return _decode_wcs_header_bytes(wcs_var.values)

    raw_arr = np.asarray(wcs_var.values)
    if raw_arr.size == 0:
        return None
    return _decode_wcs_header_bytes(np.ravel(raw_arr)[time_idx])


def _maybe_load(da: xr.DataArray) -> xr.DataArray:
    """Eagerly load a dask-backed DataArray if it is below the size threshold."""
    if hasattr(da, "nbytes") and da.nbytes < _EAGER_LOAD_THRESHOLD:
        if hasattr(da, "load"):
            da = da.load()
    return da


@contextlib.contextmanager
def _dask_progress(
    label: str = "Computing",
    *,
    quiet: bool = False,
) -> Generator[None, None, None]:
    """Show a dask progress bar when dask is available.

    When ``quiet`` is true, compute silently (used when a UI progress callback
    already reports the same work to an activity log).
    """
    message = f"{label}..."
    if quiet:
        yield
        return
    try:
        from dask.diagnostics import ProgressBar

        with ProgressBar(dt=1.0, minimum=2.0):
            print(message)  # noqa: T201
            yield
    except ImportError:
        yield


_RADPORT_PROGRESS_BATCH_TARGET = 16
_TRACKED_POINT_DIM = "_radport_track"
_EXTRACT_PROGRESS_HEARTBEAT_S = 5.0


def _radport_progress_batch_size(n_steps: int) -> int:
    """Batch size for phased progress reporting on long time axes."""
    if n_steps <= 1:
        return 1
    # About ten progress updates per axis; cap task batch size for Dask overhead.
    target = max(1, (n_steps + 9) // 10)
    return max(1, min(_RADPORT_PROGRESS_BATCH_TARGET, target))


@contextlib.contextmanager
def _radport_progress_heartbeat(
    progress_callback: RadportProgressCallback | None,
    *,
    stage: str,
    current: int,
    total: int,
    message: str,
    interval_s: float = _EXTRACT_PROGRESS_HEARTBEAT_S,
) -> Generator[None, None, None]:
    """Log elapsed-time pulses while a long I/O stage runs as one Dask compute."""
    if progress_callback is None or total <= 1 or interval_s <= 0:
        yield
        return

    stop = threading.Event()

    def _pulse() -> None:
        t0 = time.perf_counter()
        while not stop.wait(interval_s):
            elapsed = time.perf_counter() - t0
            _emit_radport_progress(
                progress_callback,
                stage,
                int(current),
                total,
                f"{message} — still working ({elapsed:.0f}s)",
            )

    thread = threading.Thread(target=_pulse, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=interval_s + 0.5)


def _emit_radport_progress(
    progress_callback: RadportProgressCallback | None,
    stage: str,
    current: int,
    total: int,
    message: str,
) -> None:
    """Invoke an optional UI/log callback for radport long-running work."""
    if progress_callback is not None:
        progress_callback(stage, int(current), int(total), message)


def _data_var_is_dask_backed(data_var: xr.DataArray) -> bool:
    return hasattr(data_var, "chunks") and data_var.chunks is not None


def _active_distributed_client() -> bool:
    """Return True when a dask.distributed ``Client`` is registered."""
    try:
        from dask.distributed import get_client

        get_client()
    except (ImportError, ValueError):
        return False
    return True


def _point_extract_scheduler() -> str | None:
    """Pick a local scheduler for many small Zarr reads under a distributed Client.

    Point extractions issue hundreds of tiny reads. Running them on process
    workers retains decompressed Zarr chunks until memory limits are hit; the
    threaded scheduler keeps I/O in the caller and avoids that growth.
    """
    if _active_distributed_client():
        return "threads"
    return None


def _io_batch_scheduler() -> str:
    """Scheduler for batched xarray point/patch reads from dask-backed Zarr."""
    return "threads"


def _patch_extract_scheduler() -> str | None:
    """Scheduler for fused patch-cube Zarr reads in :meth:`patch_statistic`.

    Returns ``None`` (distributed default) when a :class:`dask.distributed.Client`
    is active so fused reads can run on cluster workers.  Honors
    ``OVRO_RADPORT_EXTRACT_SCHEDULER`` when set.  Otherwise ``"threads"`` in the
    notebook kernel.
    """
    override = os.environ.get("OVRO_RADPORT_EXTRACT_SCHEDULER")
    if override:
        normalized = override.strip().lower()
        if normalized in {"distributed", "default", "none"}:
            return None
        return override
    if _active_distributed_client():
        return None
    return "threads"


def _compute_xarray_dataarray(
    da: xr.DataArray,
    *,
    label: str,
    progress_callback: RadportProgressCallback | None = None,
    quiet: bool | None = None,
    scheduler: str | None | object = ...,
) -> xr.DataArray:
    """Materialize one xarray selection with threaded/local/distributed scheduling."""
    if scheduler is ...:
        sched: str | None = _point_extract_scheduler()
        if sched is None and _data_var_is_dask_backed(da):
            sched = _io_batch_scheduler()
    else:
        sched = scheduler  # type: ignore[assignment]
    if quiet is None:
        quiet = progress_callback is not None
    if _data_var_is_dask_backed(da):
        with _dask_progress(label, quiet=quiet):
            if sched is None:
                return da.compute()
            return da.compute(scheduler=sched)
    return da.load()


def _vectorized_tracked_pixel_values(
    data_var: xr.DataArray,
    vis_times: np.ndarray,
    vis_l: np.ndarray,
    vis_m: np.ndarray,
    *,
    progress_callback: RadportProgressCallback | None = None,
    progress_message: str = "",
) -> np.ndarray:
    """Read values at per-time ``(l, m)`` pixels in one vectorized selection.

    Returns an array of shape ``(n_visible,)`` when ``frequency`` was already
    sliced away, otherwise ``(n_visible, n_frequency)``.
    """
    n_vis = int(np.asarray(vis_times).size)
    if n_vis == 0:
        return np.empty(0, dtype=np.float64)
    time_da = xr.DataArray(np.asarray(vis_times, dtype=np.intp), dims=_TRACKED_POINT_DIM)
    l_da = xr.DataArray(np.asarray(vis_l, dtype=np.intp), dims=_TRACKED_POINT_DIM)
    m_da = xr.DataArray(np.asarray(vis_m, dtype=np.intp), dims=_TRACKED_POINT_DIM)
    sel = data_var.isel(time=time_da, l=l_da, m=m_da)
    if not progress_message:
        progress_message = f"Extracting {n_vis} tracked pixels"
    with _radport_progress_heartbeat(
        progress_callback,
        stage="extract",
        current=0,
        total=n_vis,
        message=progress_message,
    ):
        loaded = _compute_xarray_dataarray(
            sel,
            label=f"Extracting {n_vis} tracked pixels",
            progress_callback=progress_callback,
        )
    values = np.asarray(loaded.data, dtype=np.float64)
    if values.ndim == 0:
        return values.reshape(1)
    if _TRACKED_POINT_DIM not in loaded.dims:
        return values.reshape(n_vis)
    track_axis = loaded.dims.index(_TRACKED_POINT_DIM)
    if track_axis != 0:
        values = np.moveaxis(values, track_axis, 0)
    if values.ndim == 1:
        return values.reshape(n_vis)
    return values


def _rows_from_tracked_pixel_plane(plane: np.ndarray) -> list[np.ndarray]:
    """Split a ``(n_visible[, n_frequency])`` plane into per-time rows."""
    if plane.ndim == 1:
        return [np.asarray(plane, dtype=np.float64)]
    return [np.asarray(plane[i], dtype=np.float64) for i in range(int(plane.shape[0]))]


def _patch_reduce_scheduler() -> str | None:
    """Scheduler for per-time patch reduce and Gaussian fit ``dask.delayed`` batches.

    Returns ``None`` when a distributed :class:`dask.distributed.Client` is
    active so ``dask.compute`` dispatches delayed tasks to cluster workers.
    Otherwise honors ``OVRO_RADPORT_PATCH_SCHEDULER`` when set, or ``"processes"``
    for a local multiprocessing pool in the notebook kernel.
    """
    if _active_distributed_client():
        return None
    override = os.environ.get("OVRO_RADPORT_PATCH_SCHEDULER")
    if override:
        return override
    return "processes"


def _compute_delayed_tasks(
    tasks: tuple[Any, ...],
    *,
    scheduler: str | None,
    label: str,
    quiet: bool,
) -> tuple[Any, ...]:
    """Run a batch of ``dask.delayed`` tasks with an optional local scheduler."""
    import dask

    with _dask_progress(label, quiet=quiet):
        if scheduler is None:
            return dask.compute(*tasks)
        return dask.compute(*tasks, scheduler=scheduler)


_WCS_TRACK_VARYING_KEYWORDS = frozenset({"CRVAL1", "CRVAL2", "LATPOLE"})


@dataclass(frozen=True)
class _PerTimeWcsTrackTable:
    """Parsed ``wcs_header_str(time)`` series for in-process pixel tracking."""

    crval1: np.ndarray
    crval2: np.ndarray
    latpole: np.ndarray | None
    header_valid: np.ndarray
    use_template: bool
    template_header: str | None
    header_strs: tuple[str, ...]


def _header_has_sin_celestial(hdr: object) -> bool:
    ctype1 = str(hdr.get("CTYPE1", "")).strip().upper()  # type: ignore[union-attr]
    ctype2 = str(hdr.get("CTYPE2", "")).strip().upper()  # type: ignore[union-attr]
    return ctype1.endswith("SIN") and ctype2.endswith("SIN")


def _parse_sin_celestial_keywords(hdr_str: str) -> dict[str, float | str] | None:
    """Return normalized celestial WCS keywords from one FITS header string."""
    if not hdr_str.strip():
        return None
    try:
        from astropy.io.fits import Header

        hdr = Header.fromstring(hdr_str, sep="\n")
    except (OSError, ValueError):
        return None
    if not _header_has_sin_celestial(hdr):
        return None
    kw: dict[str, float | str] = {
        "CTYPE1": str(hdr["CTYPE1"]).strip(),
        "CTYPE2": str(hdr["CTYPE2"]).strip(),
    }
    for key in (
        "NAXIS1",
        "NAXIS2",
        "CRPIX1",
        "CRPIX2",
        "CRVAL1",
        "CRVAL2",
        "CDELT1",
        "CDELT2",
        "CROTA2",
        "LATPOLE",
    ):
        if key in hdr:
            kw[key] = float(hdr[key])
    cd_entries: list[tuple[str, float]] = []
    for i in (1, 2):
        for j in (1, 2):
            name = f"CD{i}_{j}"
            if name in hdr:
                cd_entries.append((name, float(hdr[name])))
    if cd_entries:
        kw["cd_entries"] = tuple(cd_entries)
    pc_entries: list[tuple[str, float]] = []
    for i in (1, 2):
        for j in (1, 2):
            name = f"PC{i}_{j}"
            if name in hdr:
                pc_entries.append((name, float(hdr[name])))
    if pc_entries:
        kw["pc_entries"] = tuple(pc_entries)
    if "CRVAL1" not in kw or "CRVAL2" not in kw:
        return None
    return kw


def _sin_wcs_static_fingerprint(kw: dict[str, float | str]) -> tuple[tuple[str, float | str], ...]:
    """Hashable fingerprint of header keywords that must not vary across time."""
    items: list[tuple[str, float | str]] = []
    for key, value in sorted(kw.items()):
        if key in _WCS_TRACK_VARYING_KEYWORDS:
            continue
        items.append((key, value))
    return tuple(items)


def _build_per_time_wcs_track_table(header_strs: list[str]) -> _PerTimeWcsTrackTable:
    """Parse all per-time headers once; detect a shared SIN template when possible."""
    n_times = len(header_strs)
    crval1 = np.full(n_times, np.nan, dtype=np.float64)
    crval2 = np.full(n_times, np.nan, dtype=np.float64)
    latpole = np.full(n_times, np.nan, dtype=np.float64)
    header_valid = np.zeros(n_times, dtype=bool)
    latpole_seen = False
    fingerprints: list[tuple[int, tuple[tuple[str, float | str], ...]]] = []
    template_header: str | None = None

    for ti, hdr in enumerate(header_strs):
        kw = _parse_sin_celestial_keywords(hdr)
        if kw is None:
            continue
        header_valid[ti] = True
        crval1[ti] = float(kw["CRVAL1"])
        crval2[ti] = float(kw["CRVAL2"])
        if "LATPOLE" in kw:
            latpole_seen = True
            latpole[ti] = float(kw["LATPOLE"])
        fingerprints.append((ti, _sin_wcs_static_fingerprint(kw)))
        if template_header is None:
            template_header = hdr

    use_template = False
    if fingerprints and template_header is not None:
        ref_fp = fingerprints[0][1]
        use_template = all(fp == ref_fp for _, fp in fingerprints)

    latpole_out: np.ndarray | None
    if latpole_seen and use_template:
        latpole_out = latpole
    else:
        latpole_out = None

    return _PerTimeWcsTrackTable(
        crval1=crval1,
        crval2=crval2,
        latpole=latpole_out,
        header_valid=header_valid,
        use_template=use_template,
        template_header=template_header if use_template else None,
        header_strs=tuple(header_strs),
    )


def _world2pix_from_header_str(
    hdr_str: str,
    ra: float,
    dec: float,
    n_l: int,
    n_m: int,
) -> tuple[int, int, bool]:
    """Map one (RA, Dec) through a FITS header string.

    Returns ``(l_idx, m_idx, visible)``.  On missing/invalid WCS or out-of-bounds
    pixels, returns out-of-range sentinels ``(n_l, n_m)`` with ``visible=False``,
    matching :meth:`RadportAccessor._compute_pixel_track` error handling.
    """
    if not hdr_str.strip():
        return int(n_l), int(n_m), False
    try:
        from astropy.io.fits import Header
        from astropy.wcs import WCS
    except ImportError:
        raise
    try:
        wcs = WCS(Header.fromstring(hdr_str, sep="\n"))
        if not wcs.has_celestial:
            raise ValueError("WCS header has no celestial axes (RA/Dec)")
        celestial = wcs.celestial
        xp, yp = celestial.all_world2pix(float(ra), float(dec), 0)
        xp_f = float(np.asarray(xp).ravel()[0])
        yp_f = float(np.asarray(yp).ravel()[0])
        if not (np.isfinite(xp_f) and np.isfinite(yp_f)):
            raise ValueError(
                f"Source (RA={ra}, Dec={dec}) is outside the image footprint."
            )
        l_idx = int(np.round(xp_f))
        m_idx = int(np.round(yp_f))
        if not (0 <= l_idx < n_l and 0 <= m_idx < n_m):
            raise ValueError(
                f"Source (RA={ra}, Dec={dec}) maps outside the image FOV."
            )
    except ValueError:
        return int(n_l), int(n_m), False
    return l_idx, m_idx, True


def _bulk_per_time_wcs_header_strings(ds: xr.Dataset, n_times: int) -> list[str]:
    """Load decoded celestial WCS header strings for every time index in one pass."""
    if _has_fits_header_str(ds):
        return [
            _read_wcs_header_str(ds, time_idx=ti, freq_idx=0) or ""
            for ti in range(n_times)
        ]

    wcs_var = ds["wcs_header_str"]
    sel = wcs_var
    if wcs_var.ndim == 2 and "frequency" in wcs_var.dims:
        sel = wcs_var.isel(frequency=0)
    if hasattr(sel, "chunks") and sel.chunks is not None:
        sel = sel.load()
    raw = np.asarray(sel.values)
    if raw.shape[0] != n_times:
        msg = (
            f"wcs_header_str time length {raw.shape[0]} does not match "
            f"dataset n_times={n_times}"
        )
        raise ValueError(msg)
    return [_decode_wcs_header_bytes(raw[ti]) for ti in range(n_times)]


def _assign_world2pix_result(
    l_indices: np.ndarray,
    m_indices: np.ndarray,
    visible: np.ndarray,
    time_idx: int,
    li: int,
    mi: int,
    vis: bool,
) -> None:
    l_indices[time_idx] = int(li)
    m_indices[time_idx] = int(mi)
    visible[time_idx] = bool(vis)


def _track_pixels_from_wcs_table(
    table: _PerTimeWcsTrackTable,
    ra: float,
    dec: float,
    *,
    n_l: int,
    n_m: int,
    progress_callback: RadportProgressCallback | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map fixed (RA, Dec) through a parsed per-time WCS table in-process."""
    n_times = int(table.crval1.shape[0])
    l_indices = np.full(n_times, n_l, dtype=int)
    m_indices = np.full(n_times, n_m, dtype=int)
    visible = np.zeros(n_times, dtype=bool)
    progress_batch = _radport_progress_batch_size(n_times)

    _emit_radport_progress(
        progress_callback,
        "track",
        0,
        n_times,
        "Mapping RA/Dec to image pixels (per-time WCS)",
    )

    if table.use_template and table.template_header is not None:
        from astropy.io.fits import Header
        from astropy.wcs import WCS

        wcs = WCS(Header.fromstring(table.template_header, sep="\n")).celestial
        const_latpole: float | None = None
        if table.latpole is not None:
            valid_latpole = table.latpole[table.header_valid]
            if valid_latpole.size and np.all(valid_latpole == valid_latpole[0]):
                const_latpole = float(valid_latpole[0])
        for ti in range(n_times):
            if table.header_valid[ti]:
                wcs.wcs.crval = [float(table.crval1[ti]), float(table.crval2[ti])]
                if table.latpole is not None and const_latpole is None:
                    wcs.wcs.latpole = float(table.latpole[ti])
                elif const_latpole is not None:
                    wcs.wcs.latpole = const_latpole
                try:
                    xp, yp = wcs.all_world2pix(float(ra), float(dec), 0)
                    xp_f = float(np.asarray(xp).ravel()[0])
                    yp_f = float(np.asarray(yp).ravel()[0])
                    if not (np.isfinite(xp_f) and np.isfinite(yp_f)):
                        raise ValueError("non-finite world2pix")
                    li = int(np.round(xp_f))
                    mi = int(np.round(yp_f))
                    if not (0 <= li < n_l and 0 <= mi < n_m):
                        raise ValueError("out of bounds")
                    vis = True
                except ValueError:
                    li, mi, vis = n_l, n_m, False
                _assign_world2pix_result(l_indices, m_indices, visible, ti, li, mi, vis)
            if (ti + 1) % progress_batch == 0 or ti + 1 == n_times:
                _emit_radport_progress(
                    progress_callback,
                    "track",
                    ti + 1,
                    n_times,
                    "Mapping RA/Dec to image pixels (per-time WCS)",
                )
        return l_indices, m_indices, visible

    for ti, hdr in enumerate(table.header_strs):
        li, mi, vis = _world2pix_from_header_str(hdr, ra, dec, n_l, n_m)
        _assign_world2pix_result(l_indices, m_indices, visible, ti, li, mi, vis)
        if (ti + 1) % progress_batch == 0 or ti + 1 == n_times:
            _emit_radport_progress(
                progress_callback,
                "track",
                ti + 1,
                n_times,
                "Mapping RA/Dec to image pixels (per-time WCS)",
            )
    return l_indices, m_indices, visible


def _compute_xarray_batch(
    arrays: list[xr.DataArray],
    *,
    label: str,
    progress_callback: RadportProgressCallback | None = None,
) -> list[np.ndarray]:
    """Materialize a batch of xarray selections with bounded task-graph growth."""
    if not arrays:
        return []

    scheduler = _point_extract_scheduler()
    quiet = progress_callback is not None
    if _data_var_is_dask_backed(arrays[0]):
        import dask

        if scheduler is not None:
            with _dask_progress(label, quiet=quiet):
                computed = dask.compute(*arrays, scheduler=scheduler)
            return [np.asarray(item) for item in computed]

        with _dask_progress(label, quiet=quiet):
            computed = dask.compute(*arrays, scheduler=_io_batch_scheduler())
        return [np.asarray(item) for item in computed]

    return [np.asarray(arr.values) for arr in arrays]


def _patch_slices_from_center(
    li: int,
    mi: int,
    radius: int,
    *,
    n_l: int,
    n_m: int,
) -> tuple[slice, slice]:
    """Return ``(l, m)`` slice bounds for a square patch around ``(li, mi)``."""
    radius_i = int(radius)
    l_sl = slice(max(0, int(li) - radius_i), min(n_l, int(li) + radius_i + 1))
    m_sl = slice(max(0, int(mi) - radius_i), min(n_m, int(mi) + radius_i + 1))
    return l_sl, m_sl


def _pad_patch_dataarray(
    patch: xr.DataArray,
    *,
    n_l: int,
    n_m: int,
) -> xr.DataArray:
    """Pad a patch to ``(n_l, n_m)`` with NaN for fused concat."""
    cur_l = int(patch.sizes["l"])
    cur_m = int(patch.sizes["m"])
    if cur_l == n_l and cur_m == n_m:
        return patch
    pad_l = n_l - cur_l
    pad_m = n_m - cur_m
    if pad_l < 0 or pad_m < 0:
        msg = f"patch shape {(cur_l, cur_m)} exceeds pad target {(n_l, n_m)}"
        raise ValueError(msg)
    return _normalize_patch_coords(
        patch.pad(l=(0, pad_l), m=(0, pad_m), constant_values=np.nan)
    )


def _normalize_patch_coords(patch: xr.DataArray) -> xr.DataArray:
    """Use positional ``l``/``m`` coords so fused concat does not align sky indices."""
    nl = int(patch.sizes["l"])
    nm = int(patch.sizes["m"])
    return patch.assign_coords(l=np.arange(nl, dtype=np.intp), m=np.arange(nm, dtype=np.intp))


def _stack_tracked_patch_selections(
    data_var: xr.DataArray,
    vis_times: np.ndarray,
    vis_l: np.ndarray,
    vis_m: np.ndarray,
    radii: list[int] | np.ndarray,
    *,
    n_l: int,
    n_m: int,
) -> tuple[xr.DataArray, list[tuple[int, int]]]:
    """Stack per-time patch ``isel`` selections on ``_TRACKED_POINT_DIM``.

    Returns the stacked lazy array and ``(l_size, m_size)`` per track for splitting.
    """
    patch_list: list[xr.DataArray] = []
    patch_sizes: list[tuple[int, int]] = []
    for t, li, mi, radius in zip(vis_times, vis_l, vis_m, radii, strict=True):
        l_sl, m_sl = _patch_slices_from_center(
            int(li), int(mi), int(radius), n_l=n_l, n_m=n_m
        )
        patch = data_var.isel(time=int(t), l=l_sl, m=m_sl)
        patch_list.append(_normalize_patch_coords(patch))
        patch_sizes.append((int(patch.sizes["l"]), int(patch.sizes["m"])))

    if not patch_list:
        empty = xr.DataArray(np.empty(0, dtype=np.float64), dims=[_TRACKED_POINT_DIM])
        return empty, []

    unique_shapes = set(patch_sizes)
    if len(unique_shapes) == 1:
        to_stack = patch_list
    else:
        max_l = max(size[0] for size in patch_sizes)
        max_m = max(size[1] for size in patch_sizes)
        to_stack = [
            _pad_patch_dataarray(patch, n_l=max_l, n_m=max_m) for patch in patch_list
        ]

    stacked = xr.concat(
        to_stack,
        dim=_TRACKED_POINT_DIM,
        coords="minimal",
        compat="override",
        join="outer",
    )
    return stacked, patch_sizes


def _split_stacked_patch_cubes(
    stacked: np.ndarray,
    patch_sizes: list[tuple[int, int]],
) -> list[np.ndarray]:
    """Recover per-time ``(frequency, l, m)`` patches from a fused stack."""
    n_vis = len(patch_sizes)
    if n_vis == 0:
        return []
    arr = np.asarray(stacked, dtype=np.float64)
    if arr.shape[0] != n_vis:
        msg = (
            f"stacked patch axis 0 ({arr.shape[0]}) must match "
            f"visible tracks ({n_vis})"
        )
        raise ValueError(msg)
    return [arr[i, ..., :nl, :nm].copy() for i, (nl, nm) in enumerate(patch_sizes)]


def _process_patch_statistic_time(
    time_idx: int,
    patch: np.ndarray,
    statistic: PatchStatisticName,
) -> tuple[int, np.ndarray]:
    """Reduce one time step's patch cube to a per-frequency statistic vector."""
    return int(time_idx), _reduce_patch_cube_statistics(patch, statistic)


def _process_patch_fit_time(
    time_idx: int,
    patch: np.ndarray,
    *,
    allow_position_offset: bool,
    beam_widths_per_freq: list[tuple[float, float]],
    max_reduced_chi_squared: float,
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Fit Gaussians on one time step's patch cube and apply the chi-squared mask."""
    (
        peaks,
        x_offs,
        y_offs,
        widthxs,
        widthys,
        backgrounds,
        chi2_red,
        center_flux,
        patch_max,
    ) = _fit_patch_cube_gaussian(
        patch,
        allow_position_offset=allow_position_offset,
        beam_widths_per_freq=beam_widths_per_freq,
    )
    peaks, widthxs, widthys, backgrounds, x_offs, y_offs = _mask_patch_fit_by_chi2(
        peaks,
        widthxs,
        widthys,
        backgrounds,
        chi2_red,
        max_reduced_chi_squared=max_reduced_chi_squared,
        x_offsets=x_offs,
        y_offsets=y_offs,
    )
    return (
        int(time_idx),
        peaks,
        x_offs if x_offs is not None else np.full_like(peaks, np.nan),
        y_offs if y_offs is not None else np.full_like(peaks, np.nan),
        widthxs,
        widthys,
        backgrounds,
        chi2_red,
        center_flux,
        patch_max,
    )


def _run_batched_time_step_work(
    vis_times: np.ndarray,
    patches: list[np.ndarray],
    process_fn: Callable[..., Any],
    *,
    process_args: tuple[Any, ...] = (),
    progress_callback: RadportProgressCallback | None,
    stage: str,
    progress_label: str,
    parallel: bool,
) -> list[Any]:
    """Compute per-time work in batches with optional Dask parallelism.

    ``process_fn`` must be a picklable top-level callable invoked as
    ``process_fn(time_idx, patch, *process_args)``.
    """
    total = len(patches)
    if total <= 0:
        return []
    batch_size = _radport_progress_batch_size(total)
    _emit_radport_progress(
        progress_callback,
        stage,
        0,
        total,
        f"{progress_label} (0/{total} time steps)",
    )
    results: list[Any] = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_times = vis_times[start:end]
        batch_patches = patches[start:end]
        if parallel and len(batch_patches) > 1:
            import dask

            tasks = [
                dask.delayed(process_fn)(int(ti), patch, *process_args)
                for ti, patch in zip(batch_times, batch_patches, strict=True)
            ]
            label = f"{progress_label} ({end}/{total})"
            reduce_scheduler = _patch_reduce_scheduler()
            batch_results = _compute_delayed_tasks(
                tuple(tasks),
                scheduler=reduce_scheduler,
                label=label,
                quiet=progress_callback is not None,
            )
        else:
            batch_results = [
                process_fn(int(ti), patch, *process_args)
                for ti, patch in zip(batch_times, batch_patches, strict=True)
            ]
        results.extend(batch_results)
        _emit_radport_progress(
            progress_callback,
            stage,
            end,
            total,
            f"{progress_label} ({end}/{total} time steps)",
        )
    return results


def _run_batched_patch_fit_work(
    vis_times: np.ndarray,
    patches: list[np.ndarray],
    *,
    beam_widths_by_time: dict[int, list[tuple[float, float]]],
    allow_position_offset: bool,
    max_reduced_chi_squared: float,
    progress_callback: RadportProgressCallback | None,
    parallel: bool,
) -> list[
    tuple[
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
]:
    """Batched Gaussian patch fits with per-time beam metadata."""
    total = len(patches)
    if total <= 0:
        return []
    batch_size = _radport_progress_batch_size(total)
    _emit_radport_progress(
        progress_callback,
        "fit",
        0,
        total,
        f"Fitting Gaussian patches (0/{total} time steps)",
    )
    results: list[Any] = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_times = vis_times[start:end]
        batch_patches = patches[start:end]
        if parallel and len(batch_patches) > 1:
            import dask

            tasks = [
                dask.delayed(_process_patch_fit_time)(
                    int(ti),
                    patch,
                    allow_position_offset=allow_position_offset,
                    beam_widths_per_freq=beam_widths_by_time[int(ti)],
                    max_reduced_chi_squared=max_reduced_chi_squared,
                )
                for ti, patch in zip(batch_times, batch_patches, strict=True)
            ]
            label = f"Fitting Gaussian patches ({end}/{total})"
            reduce_scheduler = _patch_reduce_scheduler()
            batch_results = _compute_delayed_tasks(
                tuple(tasks),
                scheduler=reduce_scheduler,
                label=label,
                quiet=progress_callback is not None,
            )
        else:
            batch_results = [
                _process_patch_fit_time(
                    int(ti),
                    patch,
                    allow_position_offset=allow_position_offset,
                    beam_widths_per_freq=beam_widths_by_time[int(ti)],
                    max_reduced_chi_squared=max_reduced_chi_squared,
                )
                for ti, patch in zip(batch_times, batch_patches, strict=True)
            ]
        results.extend(batch_results)
        _emit_radport_progress(
            progress_callback,
            "fit",
            end,
            total,
            f"Fitting Gaussian patches ({end}/{total} time steps)",
        )
    return results


PatchStatisticName = Literal["std", "max", "min", "mean", "mad"]
PatchStatisticComparison = Literal["gt", "ge", "lt", "le"]


def _reduce_spatial_statistic(values: np.ndarray, statistic: PatchStatisticName) -> float:
    """Reduce a 2D spatial patch to one scalar."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return float("nan")
    if statistic == "std":
        return float(np.std(finite))
    if statistic == "max":
        return float(np.max(finite))
    if statistic == "min":
        return float(np.min(finite))
    if statistic == "mean":
        return float(np.mean(finite))
    if statistic == "mad":
        med = float(np.median(finite))
        return float(np.median(np.abs(finite - med)))
    msg = f"Unsupported statistic {statistic!r}"
    raise ValueError(msg)


def _reduce_patch_cube_statistics(
    patch: np.ndarray,
    statistic: PatchStatisticName,
) -> np.ndarray:
    """Apply a spatial statistic to each frequency plane in ``(frequency, l, m)``."""
    patch_arr = np.asarray(patch)
    if patch_arr.ndim != 3:
        msg = f"Expected patch with shape (frequency, l, m), got {patch_arr.shape}"
        raise ValueError(msg)
    return np.array(
        [_reduce_spatial_statistic(patch_arr[fi], statistic) for fi in range(patch_arr.shape[0])],
        dtype=np.float64,
    )


# FWHM (pixels) = _FWHM_TO_SIGMA * sigma for a 2D Gaussian profile.
_FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _gaussian_predict(
    params: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Evaluate a 2D Gaussian model on a pixel grid.

    ``params`` is either ``(peak, widthx, widthy, background)`` centred at the
    origin, or ``(peak, x_offset, y_offset, widthx, widthy, background)`` when
    the peak is offset from the patch centre.
    """
    if params.size == 4:
        peak, widthx, widthy, background = params
        x0 = 0.0
        y0 = 0.0
    else:
        peak, x0, y0, widthx, widthy, background = params
    sx = max(float(widthx) / _FWHM_TO_SIGMA, 1e-6)
    sy = max(float(widthy) / _FWHM_TO_SIGMA, 1e-6)
    return background + peak * np.exp(
        -0.5 * (((x - x0) / sx) ** 2 + ((y - y0) / sy) ** 2)
    )


def _reduced_chi_squared(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    n_params: int = 4,
) -> float:
    """Compute reduced chi-squared for a 2D patch fit."""
    obs = np.asarray(observed, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    finite = np.isfinite(obs) & np.isfinite(pred)
    if not np.any(finite):
        return float("nan")
    residuals = obs[finite] - pred[finite]
    chi2 = float(np.sum(residuals**2))
    dof = int(np.sum(finite)) - n_params
    if dof <= 0:
        return float("nan")
    return chi2 / dof


def _beam_fwhm_lm_pixels(
    bmaj_deg: float,
    bmin_deg: float,
    bpa_deg: float,
    dl_deg: float,
    dm_deg: float,
) -> tuple[float, float]:
    """Convert synthesized beam FWHM (degrees) to ``(widthx, widthy)`` in l/m pixels.

    ``widthx`` is along the ``m`` axis; ``widthy`` is along the ``l`` axis.
    ``BPA`` follows the FITS convention (degrees east of north).
    """
    pa = np.deg2rad(float(bpa_deg))
    cos_p = float(np.cos(pa))
    sin_p = float(np.sin(pa))
    bmaj = float(bmaj_deg)
    bmin = float(bmin_deg)
    width_m_deg = float(np.sqrt((bmaj * sin_p) ** 2 + (bmin * cos_p) ** 2))
    width_l_deg = float(np.sqrt((bmaj * cos_p) ** 2 + (bmin * sin_p) ** 2))
    dm = abs(float(dm_deg))
    dl = abs(float(dl_deg))
    widthx = width_m_deg / dm if dm > 0 else bmaj
    widthy = width_l_deg / dl if dl > 0 else bmin
    return max(widthx, 0.5), max(widthy, 0.5)


def patch_half_width_pixels(
    scale: float,
    beam_widthx: float,
    beam_widthy: float,
) -> int:
    """Patch half-width in pixels: ``scale`` times the larger beam FWHM."""
    if scale <= 0:
        msg = f"scale must be positive, got {scale}"
        raise ValueError(msg)
    beam_max = max(float(beam_widthx), float(beam_widthy))
    return max(1, int(np.ceil(scale * beam_max)))


def format_radec_sexagesimal(ra_deg: float, dec_deg: float) -> tuple[str, str]:
    """Format equatorial coordinates as ``hh:mm:ss.s`` and ``±dd:mm:ss.s``."""
    if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
        return "—", "—"
    from astropy import units as u
    from astropy.coordinates import Angle

    ra = Angle(float(ra_deg) * u.deg)
    dec = Angle(float(dec_deg) * u.deg)
    ra_s = ra.to_string(unit=u.hour, sep=":", precision=1, pad=True)
    dec_s = dec.to_string(unit=u.deg, sep=":", alwayssign=True, precision=1, pad=True)
    return str(ra_s), str(dec_s)


def _gaussian_parameters_from_patch_statistics(
    arr: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    beam_widthx: float,
    beam_widthy: float,
    max_width: float,
    max_offset: float,
) -> tuple[float, float, float, float, float, float]:
    """Estimate Gaussian parameters from patch pixel statistics.

    Returns ``(peak, x_offset, y_offset, widthx, widthy, background)`` where
    offsets are in patch-centred pixel coordinates.  ``peak`` uses the patch
    maximum minus the median background.  Beam widths are used when provided.
    """
    data = np.asarray(arr, dtype=np.float64)
    background = float(np.nanmedian(data))
    patch_max = float(np.nanmax(data))

    peak = patch_max - background
    if not np.isfinite(peak) or peak <= 0:
        peak = float(np.nanstd(data))
    if not np.isfinite(peak) or peak <= 0:
        peak = 1.0

    weight_map = np.where(np.isfinite(data), np.maximum(data - background, 0.0), 0.0)
    weight_sum = float(weight_map.sum())
    x_offset = 0.0
    y_offset = 0.0
    widthx = float(beam_widthx)
    widthy = float(beam_widthy)
    if weight_sum > 0:
        x_offset = float((weight_map * x).sum() / weight_sum)
        y_offset = float((weight_map * y).sum() / weight_sum)
        var_x = float((weight_map * (x - x_offset) ** 2).sum() / weight_sum)
        var_y = float((weight_map * (y - y_offset) ** 2).sum() / weight_sum)
        if var_x > 0:
            widthx = float(_FWHM_TO_SIGMA * np.sqrt(var_x))
        if var_y > 0:
            widthy = float(_FWHM_TO_SIGMA * np.sqrt(var_y))

    width_lo = 0.5
    width_hi = float(max_width)
    offset_hi = float(max_offset)
    widthx = float(np.clip(widthx, width_lo, width_hi))
    widthy = float(np.clip(widthy, width_lo, width_hi))
    x_offset = float(np.clip(x_offset, -offset_hi, offset_hi))
    y_offset = float(np.clip(y_offset, -offset_hi, offset_hi))
    return float(peak), x_offset, y_offset, widthx, widthy, background


def _fit_spatial_gaussian(
    values: np.ndarray,
    *,
    beam_widthx: float,
    beam_widthy: float,
    allow_position_offset: bool = True,
) -> tuple[float, float, float, float, float, float, float]:
    """Fit a 2D Gaussian to a spatial patch.

    Returns ``(peak, x_offset, y_offset, widthx, widthy, background, chi2_red)``
    where offsets are in patch-centred pixel coordinates.  When
    ``allow_position_offset`` is False the peak is fixed at the patch centre.

    Initial guesses and optimizer failures use
    :func:`_gaussian_parameters_from_patch_statistics`.  The patch is scaled by
    its maximum absolute value before optimization for numerical stability.
    """
    nan7 = (float("nan"),) * 7
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        msg = f"Expected 2D patch, got shape {arr.shape}"
        raise ValueError(msg)

    ny, nx = arr.shape
    if ny < 2 or nx < 2:
        return nan7

    if not np.any(np.isfinite(arr)):
        return nan7

    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    yy, xx = np.indices((ny, nx))
    y = yy - cy
    x = xx - cx

    max_width = max(nx, ny) * 4.0
    max_offset = float(min(nx, ny) // 2)
    widthx_fixed = float(beam_widthx)
    widthy_fixed = float(beam_widthy)

    def _return_statistical() -> tuple[float, float, float, float, float, float, float]:
        peak, x_off, y_off, widthx, widthy, background = (
            _gaussian_parameters_from_patch_statistics(
                arr,
                x,
                y,
                beam_widthx=widthx_fixed,
                beam_widthy=widthy_fixed,
                max_width=max_width,
                max_offset=max_offset,
            )
        )
        if not allow_position_offset:
            x_off = 0.0
            y_off = 0.0
        params = np.array(
            [peak, x_off, y_off, widthx, widthy, background],
            dtype=np.float64,
        )
        predicted = _gaussian_predict(params, x, y)
        n_params = 6 if allow_position_offset else 4
        chi2_red = _reduced_chi_squared(arr, predicted, n_params=n_params)
        return (
            float(peak),
            float(x_off),
            float(y_off),
            float(widthx),
            float(widthy),
            float(background),
            chi2_red,
        )

    peak0, x0_off, y0_off, widthx0, widthy0, bg0 = _gaussian_parameters_from_patch_statistics(
        arr,
        x,
        y,
        beam_widthx=widthx_fixed,
        beam_widthy=widthy_fixed,
        max_width=max_width,
        max_offset=max_offset,
    )
    widthx0 = widthx_fixed
    widthy0 = widthy_fixed
    if not allow_position_offset:
        x0_off = 0.0
        y0_off = 0.0

    scale = float(np.nanmax(np.abs(arr)))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    arr_scaled = arr / scale

    if allow_position_offset:

        def residual(params: np.ndarray) -> np.ndarray:
            peak_p, x_off, y_off, background_p = params
            full = np.array(
                [peak_p, x_off, y_off, widthx0, widthy0, background_p],
                dtype=np.float64,
            )
            return (_gaussian_predict(full, x, y) - arr_scaled).ravel()

        guess = np.array([peak0 / scale, x0_off, y0_off, bg0 / scale], dtype=np.float64)
        lower = np.array([0.0, -max_offset, -max_offset, -np.inf], dtype=np.float64)
        upper = np.array([np.inf, max_offset, max_offset, np.inf], dtype=np.float64)
        n_params = 4
    else:

        def residual(params: np.ndarray) -> np.ndarray:
            peak_p, background_p = params
            full = np.array(
                [peak_p, 0.0, 0.0, widthx0, widthy0, background_p],
                dtype=np.float64,
            )
            return (_gaussian_predict(full, x, y) - arr_scaled).ravel()

        guess = np.array([peak0 / scale, bg0 / scale], dtype=np.float64)
        lower = np.array([0.0, -np.inf], dtype=np.float64)
        upper = np.array([np.inf, np.inf], dtype=np.float64)
        n_params = 2

    try:
        res = least_squares(
            residual,
            x0=guess,
            bounds=(lower, upper),
            method="trf",
            ftol=1e-8,
            xtol=1e-8,
            max_nfev=500,
        )
    except (ValueError, RuntimeError):
        return _return_statistical()

    if not res.success:
        return _return_statistical()

    if allow_position_offset:
        peak, x_off, y_off, background = res.x
        peak = float(peak) * scale
        background = float(background) * scale
        params = np.array(
            [peak, float(x_off), float(y_off), widthx0, widthy0, background],
            dtype=np.float64,
        )
    else:
        peak, background = res.x
        peak = float(peak) * scale
        background = float(background) * scale
        params = np.array(
            [peak, 0.0, 0.0, widthx0, widthy0, background],
            dtype=np.float64,
        )
    predicted = _gaussian_predict(params, x, y)
    chi2_red = _reduced_chi_squared(arr, predicted, n_params=n_params)
    return (
        float(params[0]),
        float(params[1]),
        float(params[2]),
        float(params[3]),
        float(params[4]),
        float(params[5]),
        chi2_red,
    )


def _patch_plane_center_and_max(values: np.ndarray) -> tuple[float, float]:
    """Return (center_pixel, patch_max) for a 2D spatial patch."""
    arr = np.asarray(values, dtype=np.float64)
    center = float(arr[arr.shape[0] // 2, arr.shape[1] // 2])
    finite = arr[np.isfinite(arr)]
    patch_max = float(np.max(finite)) if finite.size else float("nan")
    return center, patch_max


def _fit_patch_cube_gaussian(
    patch: np.ndarray,
    *,
    allow_position_offset: bool = True,
    beam_widths_per_freq: list[tuple[float, float]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Fit a 2D Gaussian on each frequency plane in ``(frequency, l, m)``."""
    patch_arr = np.asarray(patch)
    if patch_arr.ndim != 3:
        msg = f"Expected patch with shape (frequency, l, m), got {patch_arr.shape}"
        raise ValueError(msg)

    peaks: list[float] = []
    x_offsets: list[float] = []
    y_offsets: list[float] = []
    widthxs: list[float] = []
    widthys: list[float] = []
    backgrounds: list[float] = []
    chi2_reds: list[float] = []
    center_fluxes: list[float] = []
    patch_maxes: list[float] = []
    for fi in range(patch_arr.shape[0]):
        if fi >= len(beam_widths_per_freq):
            msg = (
                f"beam_widths_per_freq length {len(beam_widths_per_freq)} "
                f"is less than number of frequency planes {patch_arr.shape[0]}"
            )
            raise ValueError(msg)
        if not np.any(np.isfinite(patch_arr[fi])):
            center_fluxes.append(float("nan"))
            patch_maxes.append(float("nan"))
            peaks.append(float("nan"))
            x_offsets.append(float("nan"))
            y_offsets.append(float("nan"))
            widthxs.append(float("nan"))
            widthys.append(float("nan"))
            backgrounds.append(float("nan"))
            chi2_reds.append(float("nan"))
            continue
        center_flux, patch_max = _patch_plane_center_and_max(patch_arr[fi])
        center_fluxes.append(center_flux)
        patch_maxes.append(patch_max)
        beam_wx, beam_wy = beam_widths_per_freq[fi]
        if (
            not np.isfinite(beam_wx)
            or not np.isfinite(beam_wy)
            or beam_wx <= 0
            or beam_wy <= 0
        ):
            peaks.append(float("nan"))
            x_offsets.append(float("nan"))
            y_offsets.append(float("nan"))
            widthxs.append(float("nan"))
            widthys.append(float("nan"))
            backgrounds.append(float("nan"))
            chi2_reds.append(float("nan"))
            continue
        peak, x_off, y_off, widthx, widthy, background, chi2_red = _fit_spatial_gaussian(
            patch_arr[fi],
            beam_widthx=beam_wx,
            beam_widthy=beam_wy,
            allow_position_offset=allow_position_offset,
        )
        peaks.append(peak)
        x_offsets.append(x_off)
        y_offsets.append(y_off)
        widthxs.append(widthx)
        widthys.append(widthy)
        backgrounds.append(background)
        chi2_reds.append(chi2_red)

    return (
        np.array(peaks, dtype=np.float64),
        np.array(x_offsets, dtype=np.float64),
        np.array(y_offsets, dtype=np.float64),
        np.array(widthxs, dtype=np.float64),
        np.array(widthys, dtype=np.float64),
        np.array(backgrounds, dtype=np.float64),
        np.array(chi2_reds, dtype=np.float64),
        np.array(center_fluxes, dtype=np.float64),
        np.array(patch_maxes, dtype=np.float64),
    )


def _mask_patch_fit_by_chi2(
    peaks: np.ndarray,
    widthxs: np.ndarray,
    widthys: np.ndarray,
    backgrounds: np.ndarray,
    chi2_red: np.ndarray,
    *,
    max_reduced_chi_squared: float,
    x_offsets: np.ndarray | None = None,
    y_offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Set fit parameters to NaN where reduced chi-squared exceeds the threshold."""
    bad = np.isfinite(chi2_red) & (chi2_red > max_reduced_chi_squared)
    peaks = np.asarray(peaks, dtype=np.float64).copy()
    widthxs = np.asarray(widthxs, dtype=np.float64).copy()
    widthys = np.asarray(widthys, dtype=np.float64).copy()
    backgrounds = np.asarray(backgrounds, dtype=np.float64).copy()
    peaks[bad] = np.nan
    widthxs[bad] = np.nan
    widthys[bad] = np.nan
    backgrounds[bad] = np.nan
    x_out: np.ndarray | None = None
    y_out: np.ndarray | None = None
    if x_offsets is not None:
        x_out = np.asarray(x_offsets, dtype=np.float64).copy()
        x_out[bad] = np.nan
    if y_offsets is not None:
        y_out = np.asarray(y_offsets, dtype=np.float64).copy()
        y_out[bad] = np.nan
    return peaks, widthxs, widthys, backgrounds, x_out, y_out


def _threshold_patch_selection(
    stat_values: np.ndarray,
    *,
    threshold: float,
    comparison: PatchStatisticComparison,
) -> np.ndarray:
    """Build a boolean ``(time, frequency)`` mask from a statistic map.

    Returns ``True`` where the statistic passes the threshold test and is
    finite; ``False`` elsewhere (including NaN cells).
    """
    if stat_values.ndim != 2:
        msg = f"stat_values must be 2D (time, frequency), got shape {stat_values.shape}"
        raise ValueError(msg)

    stats = np.asarray(stat_values, dtype=np.float64)
    finite = np.isfinite(stats)
    if comparison == "gt":
        passed = stats > threshold
    elif comparison == "ge":
        passed = stats >= threshold
    elif comparison == "lt":
        passed = stats < threshold
    elif comparison == "le":
        passed = stats <= threshold
    else:
        msg = f"Unsupported comparison {comparison!r}"
        raise ValueError(msg)
    return np.asarray(finite & passed, dtype=bool)


@dataclass
class PatchStatisticResult:
    """Statistic map and threshold selection for follow-up 1D/2D extractions.

    Returned by :meth:`RadportAccessor.patch_statistic`.  The ``selection``
    mask is ``True`` where the statistic passes the threshold test.
    Use :meth:`light_curve`, :meth:`dynamic_spectrum`, or :meth:`spectrum` to
    extract data with unselected cells masked to NaN.
    """

    stat_map: xr.DataArray
    selection: xr.DataArray | None
    threshold: float | None
    comparison: PatchStatisticComparison | None
    statistic: PatchStatisticName
    scale: float
    _accessor: Any
    _ra: float | None
    _dec: float | None
    _l: float | None
    _m: float | None
    _var: Literal["SKY", "BEAM"]
    _pol: int
    _track_freq_idx: int | None
    _track_freq_mhz: float | None
    _observatory: Any

    def _apply_selection_time(
        self,
        da: xr.DataArray,
        *,
        freq_idx: int,
        freq_mhz: float | None,
        apply_selection: bool,
    ) -> xr.DataArray:
        if not apply_selection or self.selection is None:
            return da
        if freq_mhz is not None:
            sel = self.selection.sel(frequency=freq_mhz, method="nearest")
        else:
            sel = self.selection.isel(frequency=int(freq_idx))
        return da.where(sel)

    def _apply_selection_frequency(
        self,
        da: xr.DataArray,
        *,
        time_idx: int,
        time_mjd: float | None,
        apply_selection: bool,
    ) -> xr.DataArray:
        if not apply_selection or self.selection is None:
            return da
        if time_mjd is not None:
            sel = self.selection.sel(time=time_mjd, method="nearest")
        else:
            sel = self.selection.isel(time=int(time_idx))
        return da.where(sel)

    def light_curve(
        self,
        *,
        freq_idx: int = 0,
        freq_mhz: float | None = None,
        apply_selection: bool = True,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Light curve at the tracked location, masked by ``selection``."""
        fi = int(self._accessor.nearest_freq_idx(freq_mhz)) if freq_mhz is not None else freq_idx
        lc = self._accessor.light_curve(
            ra=self._ra,
            dec=self._dec,
            l=self._l,
            m=self._m,
            freq_idx=fi,
            freq_mhz=freq_mhz,
            var=self._var,
            pol=self._pol,
            observatory=self._observatory,
            **kwargs,
        )
        return self._apply_selection_time(
            lc,
            freq_idx=fi,
            freq_mhz=freq_mhz,
            apply_selection=apply_selection,
        )

    def dynamic_spectrum(
        self,
        *,
        apply_selection: bool = True,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Time–frequency dynamic spectrum with unselected cells masked to NaN."""
        dynspec = self._accessor.dynamic_spectrum(
            ra=self._ra,
            dec=self._dec,
            l=self._l,
            m=self._m,
            var=self._var,
            pol=self._pol,
            freq_idx=self._track_freq_idx,
            freq_mhz=self._track_freq_mhz,
            observatory=self._observatory,
            **kwargs,
        )
        if apply_selection and self.selection is not None:
            dynspec = dynspec.where(self.selection)
        return dynspec

    def spectrum(
        self,
        *,
        time_idx: int = 0,
        time_mjd: float | None = None,
        apply_selection: bool = True,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Frequency spectrum at one time, masked by ``selection``."""
        ti = (
            int(self._accessor.nearest_time_idx(time_mjd))
            if time_mjd is not None
            else time_idx
        )
        spec = self._accessor.spectrum(
            ra=self._ra,
            dec=self._dec,
            l=self._l,
            m=self._m,
            time_idx=ti,
            time_mjd=time_mjd,
            var=self._var,
            pol=self._pol,
            freq_idx=self._track_freq_idx,
            freq_mhz=self._track_freq_mhz,
            **kwargs,
        )
        return self._apply_selection_frequency(
            spec,
            time_idx=ti,
            time_mjd=time_mjd,
            apply_selection=apply_selection,
        )


@dataclass(frozen=True)
class PatchFitCellResult:
    """Gaussian patch-fit diagnostics for one ``(time, frequency)`` cell.

    Returned by :meth:`RadportAccessor.patch_fit_cell`.  Implements
    :meth:`cell_diagnostics` with the same keys as
    :meth:`PatchFitResult.cell_diagnostics` for UI formatting.
    """

    time_idx: int
    frequency_idx: int
    fit_accepted: bool
    reduced_chi_squared: float
    peak: float
    peak_ra_deg: float
    peak_dec_deg: float
    peak_ra: str
    peak_dec: str
    x_offset_pixels: float
    y_offset_pixels: float
    peak_offset_pixels: float
    center_flux: float
    patch_max: float
    background: float
    widthx: float
    widthy: float
    scale: float
    max_reduced_chi_squared: float
    allow_position_offset: bool
    patch_radius_pixels: int

    def cell_diagnostics(
        self,
        time_idx: int,
        frequency_idx: int,
    ) -> dict[str, float | bool]:
        """Summarize patch-fit quality for this cell (duck-types :class:`PatchFitResult`)."""
        if int(time_idx) != self.time_idx or int(frequency_idx) != self.frequency_idx:
            msg = (
                f"Requested cell ({time_idx}, {frequency_idx}) does not match "
                f"({self.time_idx}, {self.frequency_idx})"
            )
            raise ValueError(msg)
        return {
            "fit_accepted": self.fit_accepted,
            "reduced_chi_squared": self.reduced_chi_squared,
            "peak": self.peak,
            "peak_ra_deg": self.peak_ra_deg,
            "peak_dec_deg": self.peak_dec_deg,
            "peak_ra": self.peak_ra,
            "peak_dec": self.peak_dec,
            "x_offset_pixels": self.x_offset_pixels,
            "y_offset_pixels": self.y_offset_pixels,
            "peak_offset_pixels": self.peak_offset_pixels,
            "center_flux": self.center_flux,
            "patch_max": self.patch_max,
            "background": self.background,
            "widthx": self.widthx,
            "widthy": self.widthy,
        }


@dataclass
class PatchFitResult:
    """Gaussian fit parameters on a tracked patch for each time/frequency cell.

    Returned by :meth:`RadportAccessor.patch_fit`.  Each parameter map has
    dimensions ``(time, frequency)``.  ``widthx`` and ``widthy`` are full width
    at half maximum in pixels.  ``x_offset_map`` and ``y_offset_map`` give the
    fitted peak offset from the tracked patch centre in pixels.

    Diagnostic maps ``center_flux_map`` and ``patch_max_map`` compare the
    tracked-centre pixel and the patch maximum.  ``fit_accepted_map`` is ``True``
    where reduced chi-squared is at or below ``max_reduced_chi_squared``.

    Fit parameter maps are NaN where the quality cut fails.
    """

    peak_map: xr.DataArray
    widthx_map: xr.DataArray
    widthy_map: xr.DataArray
    background_map: xr.DataArray
    reduced_chi_squared_map: xr.DataArray
    x_offset_map: xr.DataArray
    y_offset_map: xr.DataArray
    center_flux_map: xr.DataArray
    patch_max_map: xr.DataArray
    fit_accepted_map: xr.DataArray
    patch_radius_map: xr.DataArray
    scale: float
    max_reduced_chi_squared: float
    allow_position_offset: bool
    _accessor: Any
    _ra: float | None
    _dec: float | None
    _l: float | None
    _m: float | None
    _var: Literal["SKY", "BEAM"]
    _pol: int
    _track_freq_idx: int | None
    _track_freq_mhz: float | None
    _observatory: Any

    def _tracked_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(l_indices, m_indices, visible)`` used for patch fitting."""
        resolved = self._accessor._resolve_coordinates(
            ra=self._ra,
            dec=self._dec,
            l=self._l,
            m=self._m,
            observatory=self._observatory,
            freq_idx=self._track_freq_idx,
            freq_mhz=self._track_freq_mhz,
            pol=self._pol,
        )
        n_times = int(self.peak_map.sizes["time"])
        if isinstance(resolved, tuple) and len(resolved) == 2:
            l_idx, m_idx = resolved
            l_indices = np.full(n_times, int(l_idx), dtype=int)
            m_indices = np.full(n_times, int(m_idx), dtype=int)
            visible = np.ones(n_times, dtype=bool)
        else:
            l_indices, m_indices, visible = resolved
        return l_indices, m_indices, visible

    def _peak_radec_at(self, time_idx: int, frequency_idx: int) -> tuple[float, float]:
        """Peak RA/Dec (degrees) for one fitted cell."""
        ti = int(time_idx)
        fi = int(frequency_idx)
        dx = float(self.x_offset_map.isel(time=ti, frequency=fi).values)
        dy = float(self.y_offset_map.isel(time=ti, frequency=fi).values)
        if not (np.isfinite(dx) and np.isfinite(dy)):
            return float("nan"), float("nan")
        l_indices, m_indices, visible = self._tracked_indices()
        if not bool(visible[ti]):
            return float("nan"), float("nan")
        li = int(l_indices[ti])
        mi = int(m_indices[ti])
        n_l = int(self._accessor._obj.sizes["l"])
        n_m = int(self._accessor._obj.sizes["m"])
        l_peak = int(np.clip(int(round(li + dy)), 0, n_l - 1))
        m_peak = int(np.clip(int(round(mi + dx)), 0, n_m - 1))
        try:
            return self._accessor.pixel_to_coords(
                l_peak,
                m_peak,
                time_idx=ti,
                observatory=self._observatory,
            )
        except ValueError:
            return float("nan"), float("nan")

    def peak_radec_maps(self) -> tuple[xr.DataArray, xr.DataArray]:
        """Peak position in RA/Dec (degrees) from fitted patch offsets.

        Offsets are added to the tracked patch centre pixel at each time step,
        then converted with :meth:`RadportAccessor.pixel_to_coords`.  Cells with
        non-finite offsets or failed coordinate conversion are NaN.
        """
        n_times = int(self.peak_map.sizes["time"])
        n_freqs = int(self.peak_map.sizes["frequency"])
        ra_values = np.full((n_times, n_freqs), np.nan, dtype=np.float64)
        dec_values = np.full((n_times, n_freqs), np.nan, dtype=np.float64)
        l_indices, m_indices, visible = self._tracked_indices()
        x_off = np.asarray(self.x_offset_map.values, dtype=np.float64)
        y_off = np.asarray(self.y_offset_map.values, dtype=np.float64)
        n_l = int(self._accessor._obj.sizes["l"])
        n_m = int(self._accessor._obj.sizes["m"])

        for ti in range(n_times):
            if not bool(visible[ti]):
                continue
            li = int(l_indices[ti])
            mi = int(m_indices[ti])
            for fi in range(n_freqs):
                dx = float(x_off[ti, fi])
                dy = float(y_off[ti, fi])
                if not (np.isfinite(dx) and np.isfinite(dy)):
                    continue
                l_peak = int(np.clip(int(round(li + dy)), 0, n_l - 1))
                m_peak = int(np.clip(int(round(mi + dx)), 0, n_m - 1))
                try:
                    ra_deg, dec_deg = self._accessor.pixel_to_coords(
                        l_peak,
                        m_peak,
                        time_idx=ti,
                        observatory=self._observatory,
                    )
                except ValueError:
                    continue
                ra_values[ti, fi] = ra_deg
                dec_values[ti, fi] = dec_deg

        coords = {
            "time": self.peak_map.coords["time"].values,
            "frequency": self.peak_map.coords["frequency"].values,
        }
        attrs = dict(self.peak_map.attrs)
        ra_map = xr.DataArray(
            ra_values,
            dims=["time", "frequency"],
            coords=coords,
            attrs=attrs,
            name=f"{attrs.get('variable', 'SKY')}_patch_peak_ra",
        )
        dec_map = xr.DataArray(
            dec_values,
            dims=["time", "frequency"],
            coords=coords,
            attrs=attrs,
            name=f"{attrs.get('variable', 'SKY')}_patch_peak_dec",
        )
        return ra_map, dec_map

    def cell_diagnostics(self, time_idx: int, frequency_idx: int) -> dict[str, float | bool]:
        """Summarize patch-fit quality and comparison fluxes for one cell."""
        ti = int(time_idx)
        fi = int(frequency_idx)
        chi2 = float(self.reduced_chi_squared_map.isel(time=ti, frequency=fi).values)
        accepted = bool(self.fit_accepted_map.isel(time=ti, frequency=fi).values)
        peak = float(self.peak_map.isel(time=ti, frequency=fi).values)
        x_off = float(self.x_offset_map.isel(time=ti, frequency=fi).values)
        y_off = float(self.y_offset_map.isel(time=ti, frequency=fi).values)
        center = float(self.center_flux_map.isel(time=ti, frequency=fi).values)
        patch_max = float(self.patch_max_map.isel(time=ti, frequency=fi).values)
        peak_ra, peak_dec = self._peak_radec_at(ti, fi)
        peak_ra_str, peak_dec_str = format_radec_sexagesimal(peak_ra, peak_dec)
        return {
            "fit_accepted": accepted,
            "reduced_chi_squared": chi2,
            "peak": peak,
            "peak_ra_deg": peak_ra,
            "peak_dec_deg": peak_dec,
            "peak_ra": peak_ra_str,
            "peak_dec": peak_dec_str,
            "x_offset_pixels": x_off,
            "y_offset_pixels": y_off,
            "peak_offset_pixels": float(np.hypot(x_off, y_off)) if np.isfinite(x_off) else float("nan"),
            "center_flux": center,
            "patch_max": patch_max,
            "background": float(self.background_map.isel(time=ti, frequency=fi).values),
            "widthx": float(self.widthx_map.isel(time=ti, frequency=fi).values),
            "widthy": float(self.widthy_map.isel(time=ti, frequency=fi).values),
        }


@dataclass(frozen=True)
class _PatchMetadataCache:
    """Eager beam planes and populated-cell mask for patch sizing."""

    pol: int
    var: Literal["SKY", "BEAM"]
    populated: np.ndarray
    beam_major: np.ndarray | None
    beam_minor: np.ndarray | None
    beam_pa: np.ndarray | None


def _beam_param_plane(beam: xr.DataArray, *, pol: int, param: str) -> np.ndarray:
    """Return a ``(time, frequency)`` plane for one ``beam_param`` name."""
    sel = beam.isel(polarization=int(pol))
    for name in np.asarray(sel.coords["beam_param"].values):
        if str(name).lower() == param.lower():
            plane = sel.sel(beam_param=name)
            if _data_var_is_dask_backed(plane):
                plane = _compute_xarray_dataarray(
                    plane,
                    label=f"BEAM {param}",
                    quiet=True,
                )
            return np.asarray(plane.values, dtype=np.float64)
    msg = f"BEAM variable is missing beam_param {param!r}."
    raise KeyError(msg)


def _materialize_populated_mask(da: xr.DataArray, *, label: str) -> np.ndarray:
    """Eager bool ``(time, frequency)`` mask from a lazy ``notnull().any`` reduction."""
    if _data_var_is_dask_backed(da):
        da = _compute_xarray_dataarray(da, label=label, quiet=True)
    return np.asarray(da.values, dtype=bool)


@xr.register_dataset_accessor("radport")
class RadportAccessor:
    """xarray accessor for OVRO-LWA radio astronomy datasets.

    This accessor provides domain-specific methods for working with
    OVRO-LWA data, including visualization and validation utilities.

    The accessor is automatically available on any xarray Dataset after
    importing the `ovro_lwa_portal` package.

    Parameters
    ----------
    xarray_obj : xr.Dataset
        The xarray Dataset to extend with accessor methods.

    Raises
    ------
    ValueError
        If the dataset is missing required dimensions or variables
        for OVRO-LWA data.

    Example
    -------
    >>> import ovro_lwa_portal
    >>> from ovro_lwa_portal import open_dataset
    >>> ds = open_dataset("path/to/data.zarr")
    >>> ds.radport.plot()
    """

    # Required dimensions for OVRO-LWA datasets
    _required_dims: frozenset[str] = frozenset(
        {"time", "frequency", "polarization", "l", "m"}
    )

    # Required data variables
    _required_vars: frozenset[str] = frozenset({"SKY"})

    # Optional data variables
    _optional_vars: frozenset[str] = frozenset({"BEAM"})

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        """Initialize the RadportAccessor.

        Parameters
        ----------
        xarray_obj : xr.Dataset
            The xarray Dataset to extend.

        Raises
        ------
        ValueError
            If the dataset structure is invalid for OVRO-LWA data.
        """
        self._obj = xarray_obj
        self._lst_cache: dict[tuple, np.ndarray] = {}
        self._wcs_cache: dict[tuple[int, int], Any] = {}
        self._patch_metadata_cache: _PatchMetadataCache | None = None
        self._validate_structure()

    def _validate_structure(self) -> None:
        """Validate that the dataset has required dimensions and variables.

        Raises
        ------
        ValueError
            If required dimensions or variables are missing, with an
            informative error message listing what is missing.
        """
        # Check for required dimensions
        missing_dims = self._required_dims - set(self._obj.dims)
        if missing_dims:
            raise ValueError(
                f"Dataset is missing required dimensions for OVRO-LWA data: "
                f"{sorted(missing_dims)}. "
                f"Expected dimensions: {sorted(self._required_dims)}. "
                f"Found dimensions: {sorted(self._obj.dims)}."
            )

        # Check for required data variables
        missing_vars = self._required_vars - set(self._obj.data_vars)
        if missing_vars:
            raise ValueError(
                f"Dataset is missing required variables for OVRO-LWA data: "
                f"{sorted(missing_vars)}. "
                f"Expected variables: {sorted(self._required_vars)}. "
                f"Found variables: {sorted(self._obj.data_vars)}."
            )

    @property
    def has_beam(self) -> bool:
        """Check if the dataset contains BEAM data.

        Returns
        -------
        bool
            True if the dataset contains a BEAM variable.
        """
        return "BEAM" in self._obj.data_vars

    def _patch_metadata_cache_matches(
        self,
        *,
        pol: int,
        var: Literal["SKY", "BEAM"],
    ) -> bool:
        cache = self._patch_metadata_cache
        return cache is not None and cache.pol == int(pol) and cache.var == var

    def ensure_patch_metadata_cache(
        self,
        *,
        pol: int = 0,
        var: Literal["SKY", "BEAM"] = "SKY",
    ) -> _PatchMetadataCache:
        """Load beam metadata and a populated ``(time, frequency)`` mask into memory.

        When ``BEAM`` has ``beam_param`` major/minor, the populated mask is derived
        from finite positive beam sizes (no full SKY scan). Otherwise one
        ``notnull().any`` reduction over ``l``/``m`` is computed for *var*.
        """
        if self._patch_metadata_cache_matches(pol=pol, var=var):
            return self._patch_metadata_cache  # type: ignore[return-value]

        beam_major: np.ndarray | None = None
        beam_minor: np.ndarray | None = None
        beam_pa: np.ndarray | None = None
        populated: np.ndarray

        if self.has_beam:
            beam = self._obj["BEAM"]
            if "beam_param" in beam.dims:
                beam_major = _beam_param_plane(beam, pol=pol, param="major")
                beam_minor = _beam_param_plane(beam, pol=pol, param="minor")
                with contextlib.suppress(KeyError):
                    beam_pa = _beam_param_plane(beam, pol=pol, param="pa")
                if beam_pa is None:
                    beam_pa = np.zeros_like(beam_major, dtype=np.float64)
                populated = (
                    np.isfinite(beam_major)
                    & (beam_major > 0)
                    & np.isfinite(beam_minor)
                    & (beam_minor > 0)
                )
            else:
                plane = beam.isel(polarization=int(pol))
                populated = _materialize_populated_mask(
                    plane.notnull().any(dim=list(plane.dims)),
                    label="BEAM populated mask",
                )
        else:
            plane = self._obj[var].isel(polarization=int(pol))
            populated = _materialize_populated_mask(
                plane.notnull().any(dim=["l", "m"]),
                label=f"{var} populated mask",
            )

        cache = _PatchMetadataCache(
            pol=int(pol),
            var=var,
            populated=populated,
            beam_major=beam_major,
            beam_minor=beam_minor,
            beam_pa=beam_pa,
        )
        self._patch_metadata_cache = cache
        return cache

    def _var_cell_has_finite_data(
        self,
        *,
        time_idx: int,
        frequency_idx: int,
        pol: int = 0,
        var: Literal["SKY", "BEAM"] = "SKY",
    ) -> bool:
        """Return whether a ``(time, frequency)`` cell contains any finite data."""
        if self._patch_metadata_cache_matches(pol=pol, var=var):
            cache = self._patch_metadata_cache
            assert cache is not None
            return bool(cache.populated[int(time_idx), int(frequency_idx)])

        if var not in self._obj.data_vars:
            return False
        plane = self._obj[var].isel(
            time=int(time_idx),
            frequency=int(frequency_idx),
            polarization=int(pol),
        )
        if "beam_param" in plane.dims:
            vals = np.asarray(plane.values, dtype=np.float64)
            return bool(np.any(np.isfinite(vals)))
        if plane.ndim == 0:
            return bool(np.isfinite(plane.values))
        populated = plane.notnull().any(dim=list(plane.dims))
        return bool(np.asarray(populated.values).any())

    def beam_fwhm_pixels(
        self,
        *,
        time_idx: int,
        frequency_idx: int,
        pol: int = 0,
        var: Literal["SKY", "BEAM"] = "SKY",
    ) -> tuple[float, float]:
        """Return synthesized beam FWHM in ``(widthx, widthy)`` patch pixels.

        Reads ``BMAJ``/``BMIN``/``BPA`` from the per-channel ``BEAM`` variable
        (``beam_param`` of ``major``, ``minor``, ``pa``) when present, otherwise
        from the FITS WCS header at the requested time.

        ``widthx`` is along the ``m`` axis; ``widthy`` is along the ``l`` axis.

        Raises
        ------
        ValueError
            When beam metadata or pixel scales are unavailable.
        """
        bmaj: float | None = None
        bmin: float | None = None
        bpa = 0.0
        ti = int(time_idx)
        fi = int(frequency_idx)

        cache = self._patch_metadata_cache
        if cache is not None and cache.pol == int(pol) and cache.beam_major is not None:
            bmaj = float(cache.beam_major[ti, fi])
            bmin = float(cache.beam_minor[ti, fi])  # type: ignore[index]
            bpa = float(cache.beam_pa[ti, fi]) if cache.beam_pa is not None else 0.0  # type: ignore[index]

        if (bmaj is None or bmin is None) and self.has_beam:
            beam = self._obj["BEAM"]
            if "beam_param" in beam.dims:
                beam_sel = beam.isel(
                    time=ti,
                    frequency=fi,
                    polarization=int(pol),
                )
                params = {
                    str(name).lower(): float(beam_sel.sel(beam_param=name).values)
                    for name in np.asarray(beam.coords["beam_param"].values)
                }
                if "major" in params and "minor" in params:
                    bmaj = params["major"]
                    bmin = params["minor"]
                    bpa = params.get("pa", 0.0)

        if bmaj is None or bmin is None:
            hdr_str = _read_wcs_header_str(self._obj, var=var, time_idx=int(time_idx))
            if hdr_str:
                try:
                    from astropy.io.fits import Header

                    header = Header.fromstring(hdr_str, sep="\n")
                    if "BMAJ" in header and "BMIN" in header:
                        bmaj = float(header["BMAJ"])
                        bmin = float(header["BMIN"])
                        bpa = float(header.get("BPA", 0.0))
                except (TypeError, ValueError):
                    bmaj = None

        if (
            bmaj is None
            or bmin is None
            or not np.isfinite(bmaj)
            or not np.isfinite(bmin)
            or bmaj <= 0
            or bmin <= 0
        ):
            msg = (
                "Synthesized beam metadata unavailable: need BEAM major/minor "
                f"(time_idx={time_idx}, frequency_idx={frequency_idx}) or "
                "FITS BMAJ/BMIN in the WCS header."
            )
            raise ValueError(msg)

        l_vals = np.asarray(self._obj.coords["l"].values, dtype=np.float64)
        m_vals = np.asarray(self._obj.coords["m"].values, dtype=np.float64)
        if l_vals.size < 2 or m_vals.size < 2:
            msg = "l and m coordinates must have at least two points to convert beam FWHM to pixels."
            raise ValueError(msg)
        dl = float(abs(l_vals[1] - l_vals[0]))
        dm = float(abs(m_vals[1] - m_vals[0]))
        return _beam_fwhm_lm_pixels(bmaj, bmin, bpa, dl, dm)

    def beam_fwhm_pixels_all_frequencies(
        self,
        *,
        time_idx: int,
        pol: int = 0,
        var: Literal["SKY", "BEAM"] = "SKY",
        skip_empty_cells: bool = True,
        data_var: Literal["SKY", "BEAM"] | None = None,
    ) -> list[tuple[float, float]]:
        """Synthesized beam FWHM in pixels for every frequency at one time step.

        When *skip_empty_cells* is True (default), ``(time, frequency)`` cells
        with no finite data in *data_var* (defaults to *var*) return
        ``(nan, nan)`` instead of requiring beam metadata.
        """
        n_freqs = int(self._obj.sizes["frequency"])
        empty_var = data_var if data_var is not None else var
        use_populated_cache = self._patch_metadata_cache_matches(pol=pol, var=empty_var)
        populated_mask = (
            self._patch_metadata_cache.populated if use_populated_cache else None
        )
        widths: list[tuple[float, float]] = []
        for fi in range(n_freqs):
            try:
                widths.append(
                    self.beam_fwhm_pixels(
                        time_idx=int(time_idx),
                        frequency_idx=fi,
                        pol=pol,
                        var=var,
                    )
                )
            except ValueError:
                if skip_empty_cells:
                    if populated_mask is not None and not populated_mask[int(time_idx), fi]:
                        widths.append((float("nan"), float("nan")))
                        continue
                    if not self._var_cell_has_finite_data(
                        time_idx=int(time_idx),
                        frequency_idx=fi,
                        pol=pol,
                        var=empty_var,
                    ):
                        widths.append((float("nan"), float("nan")))
                        continue
                raise
        return widths

    def patch_radius_pixels(
        self,
        *,
        time_idx: int,
        scale: float,
        pol: int = 0,
        var: Literal["SKY", "BEAM"] = "SKY",
    ) -> int:
        """Patch half-width at one time step from ``scale`` and max beam FWHM over frequency.

        Only populated ``(time, frequency)`` cells in *var* contribute to the
        maximum beam size. Empty grid slots are ignored.
        """
        beam_widths = self.beam_fwhm_pixels_all_frequencies(
            time_idx=time_idx,
            pol=pol,
            var=var,
            skip_empty_cells=True,
            data_var=var,
        )
        finite_widths = [
            (wx, wy)
            for wx, wy in beam_widths
            if np.isfinite(wx) and np.isfinite(wy) and wx > 0 and wy > 0
        ]
        if not finite_widths:
            return 0
        max_wx = max(wx for wx, _wy in finite_widths)
        max_wy = max(wy for _wx, wy in finite_widths)
        return patch_half_width_pixels(scale, max_wx, max_wy)

    # =========================================================================
    # Selection Helper Methods
    # =========================================================================

    def nearest_freq_idx(self, freq_mhz: float) -> int:
        """Find the index of the frequency nearest to the given value in MHz.

        Parameters
        ----------
        freq_mhz : float
            Target frequency in MHz.

        Returns
        -------
        int
            Index of the nearest frequency in the dataset.

        Examples
        --------
        >>> idx = ds.radport.nearest_freq_idx(50.0)  # Find index nearest to 50 MHz
        >>> ds.radport.plot(freq_idx=idx)
        """
        freq_hz = freq_mhz * 1e6
        freq_values = self._obj.coords["frequency"].values
        return int(np.argmin(np.abs(freq_values - freq_hz)))

    def nearest_time_idx(self, mjd: float) -> int:
        """Find the index of the time nearest to the given MJD value.

        Parameters
        ----------
        mjd : float
            Target time in Modified Julian Date (MJD).

        Returns
        -------
        int
            Index of the nearest time in the dataset.

        Examples
        --------
        >>> idx = ds.radport.nearest_time_idx(60000.5)  # Find index nearest to MJD
        >>> ds.radport.plot(time_idx=idx)
        """
        time_values = self._obj.coords["time"].values
        return int(np.argmin(np.abs(time_values - mjd)))

    def nearest_lm_idx(self, l: float, m: float) -> tuple[int, int]:
        """Find the indices of the (l, m) pixel nearest to the given coordinates.

        Parameters
        ----------
        l : float
            Target l direction cosine coordinate.
        m : float
            Target m direction cosine coordinate.

        Returns
        -------
        tuple of int
            (l_idx, m_idx) indices of the nearest pixel.

        Examples
        --------
        >>> l_idx, m_idx = ds.radport.nearest_lm_idx(0.0, 0.0)  # Find center pixel
        """
        l_values = self._obj.coords["l"].values
        m_values = self._obj.coords["m"].values
        l_idx = int(np.argmin(np.abs(l_values - l)))
        m_idx = int(np.argmin(np.abs(m_values - m)))
        return l_idx, m_idx

    def _lst_deg_for_time_index(self, time_idx: int, observatory: Any = None) -> float:
        """Mean sidereal longitude in degrees at one dataset ``time`` index.

        Uses the same caching and IERS policy as :meth:`_compute_pixel_at_time`.
        """
        from astropy.coordinates import EarthLocation

        if observatory is None:
            from astropy import units as u

            observatory = EarthLocation(
                lat=37.2339 * u.deg, lon=-118.2817 * u.deg, height=1222 * u.m
            )

        mjd = float(self._obj.coords["time"].values[time_idx])
        lon_deg = float(observatory.lon.deg)

        all_mjd = np.asarray(self._obj.coords["time"].values, dtype=np.float64)
        all_mjd = np.atleast_1d(all_mjd)
        full_key = (all_mjd.tobytes(), lon_deg)

        if full_key in self._lst_cache:
            lst_arr = np.asarray(self._lst_cache[full_key], dtype=np.float64).ravel()
            return float(lst_arr[time_idx])

        single_key = (np.float64(mjd).tobytes(), lon_deg)
        if single_key in self._lst_cache:
            return float(np.asarray(self._lst_cache[single_key], dtype=np.float64).ravel()[0])

        from astropy.time import Time
        from astropy.utils.iers import conf as iers_conf

        orig = iers_conf.auto_download
        try:
            iers_conf.auto_download = False
            t = Time(mjd, format="mjd", scale="utc")
            lst_deg = float(t.sidereal_time("mean", longitude=observatory.lon).deg)
        finally:
            iers_conf.auto_download = orig
        self._lst_cache[single_key] = np.atleast_1d(np.asarray(lst_deg, dtype=np.float64))
        return float(np.asarray(lst_deg, dtype=np.float64).ravel()[0])

    def _ensure_lst_deg_vector_cached(self, observatory: Any) -> None:
        """Compute and cache mean LST (deg) for every ``time`` coordinate at once.

        :meth:`_lst_deg_for_time_index` reads this via the ``(all_mjd, lon)``
        cache key. Calling this before a per-time :meth:`coords_to_pixel` loop
        replaces one Astropy sidereal-time evaluation per index with a single
        vectorized call over all MJDs.
        """
        from astropy.coordinates import EarthLocation
        from astropy.time import Time
        from astropy.utils.iers import conf as iers_conf

        if observatory is None:
            from astropy import units as u

            observatory = EarthLocation(
                lat=37.2339 * u.deg, lon=-118.2817 * u.deg, height=1222 * u.m
            )

        all_mjd = np.asarray(self._obj.coords["time"].values, dtype=np.float64)
        all_mjd = np.atleast_1d(all_mjd)
        lon_deg = float(observatory.lon.deg)
        full_key = (all_mjd.tobytes(), lon_deg)
        if full_key in self._lst_cache:
            return

        orig = iers_conf.auto_download
        try:
            iers_conf.auto_download = False
            t = Time(all_mjd, format="mjd", scale="utc")
            lst_vec = t.sidereal_time("mean", longitude=observatory.lon).deg
            lst_arr = np.atleast_1d(np.asarray(lst_vec, dtype=np.float64)).ravel()
        finally:
            iers_conf.auto_download = orig
        self._lst_cache[full_key] = lst_arr

    def _pixel_track_can_batch_time_radec_grids(self) -> bool:
        """True when per-time RA/Dec grids can be argmin'd in one array pass."""
        obj = self._obj
        ra_c = obj.coords.get("right_ascension")
        dec_c = obj.coords.get("declination")
        n_times = int(obj.sizes.get("time", 1))
        if ra_c is None or dec_c is None:
            return False
        if "time" not in obj.coords or n_times < 1:
            return False
        if not ({"l", "m"} <= set(ra_c.dims) and {"l", "m"} <= set(dec_c.dims)):
            return False
        if "time" not in ra_c.dims or "time" not in dec_c.dims:
            return False
        return True

    def _compute_pixel_track_batched_radec_grid(
        self,
        ra: float,
        dec: float,
        observatory: Any,
        *,
        fi: int,
        pol: int,
        l_coords: np.ndarray,
        m_coords: np.ndarray,
        n_l: int,
        n_m: int,
        n_times: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Batched (time, m, l) nearest-pixel search; ``None`` to fall back to per-time loop.

        Loads RA/Dec in ``(time_chunk, l, m)`` blocks.  Each chunk has at most
        ``_MAX_PIXEL_TRACK_CHUNK_ELEMENTS`` samples per field; a single ``(l, m)``
        plane may be at most ``_MAX_PIXEL_TRACK_PLANE_ELEMENTS`` samples (else
        ``None``).  See module constants for defaults.
        """
        from astropy.coordinates import EarthLocation

        if observatory is None:
            from astropy import units as u

            observatory = EarthLocation(
                lat=37.2339 * u.deg, lon=-118.2817 * u.deg, height=1222 * u.m
            )

        obj = self._obj
        all_mjd = np.asarray(obj.coords["time"].values, dtype=np.float64)
        all_mjd = np.atleast_1d(all_mjd)
        lon_deg = float(observatory.lon.deg)
        full_key = (all_mjd.tobytes(), lon_deg)
        lst_cached = self._lst_cache.get(full_key)
        if lst_cached is None:
            return None
        lst_arr = np.asarray(lst_cached, dtype=np.float64).ravel()
        if lst_arr.shape[0] != n_times:
            return None

        # Horizon (identical to _compute_pixel_at_time, vectorized).
        ha_rad = np.deg2rad(lst_arr - ra)
        dec_rad = np.deg2rad(dec)
        lat_rad = np.deg2rad(observatory.lat.deg)
        sin_alt = np.sin(dec_rad) * np.sin(lat_rad) + np.cos(dec_rad) * np.cos(
            lat_rad
        ) * np.cos(ha_rad)
        horiz_ok = sin_alt > 0

        ra_coord = obj.coords["right_ascension"]
        dec_coord = obj.coords["declination"]
        ra_b = ra_coord
        dec_b = dec_coord
        for dim_name, idx_raw in (
            ("frequency", fi),
            ("polarization", pol),
        ):
            if dim_name in ra_b.dims:
                n_d = int(ra_b.sizes[dim_name])
                ii = int(np.clip(int(idx_raw), 0, n_d - 1))
                ra_b = ra_b.isel({dim_name: ii})
            if dim_name in dec_b.dims:
                n_d = int(dec_b.sizes[dim_name])
                ii = int(np.clip(int(idx_raw), 0, n_d - 1))
                dec_b = dec_b.isel({dim_name: ii})

        if "time" not in ra_b.dims or "time" not in dec_b.dims:
            return None

        try:
            ra_bt = ra_b.transpose("time", "l", "m")
            dec_bt = dec_b.transpose("time", "l", "m")
        except (KeyError, ValueError):
            return None

        nt_sz = int(ra_bt.sizes["time"])
        nl_sz = int(ra_bt.sizes["l"])
        nm_sz = int(ra_bt.sizes["m"])
        spatial = nl_sz * nm_sz
        if spatial == 0 or spatial > _MAX_PIXEL_TRACK_PLANE_ELEMENTS:
            return None
        if nt_sz != n_times:
            return None

        chunk_nt = max(1, min(nt_sz, _MAX_PIXEL_TRACK_CHUNK_ELEMENTS // spatial))

        l_indices = np.full(nt_sz, n_l, dtype=int)
        m_indices = np.full(nt_sz, n_m, dtype=int)
        visible = np.zeros(nt_sz, dtype=bool)

        for t0 in range(0, nt_sz, chunk_nt):
            t1 = min(t0 + chunk_nt, nt_sz)
            ra_sl = ra_bt.isel(time=slice(t0, t1))
            dec_sl = dec_bt.isel(time=slice(t0, t1))

            ra_da = ra_sl.data
            dec_da = dec_sl.data
            ra_lazy = hasattr(ra_da, "compute")
            dec_lazy = hasattr(dec_da, "compute")
            if ra_lazy or dec_lazy:
                from dask import compute as dask_compute

                ra_raw, dec_raw = dask_compute(ra_da, dec_da)
                ra_blk = np.asarray(ra_raw, dtype=np.float64)
                dec_blk = np.asarray(dec_raw, dtype=np.float64)
            else:
                ra_blk = np.asarray(ra_da, dtype=np.float64)
                dec_blk = np.asarray(dec_da, dtype=np.float64)

            if ra_blk.shape != dec_blk.shape:
                dec_blk = np.broadcast_to(dec_blk, ra_blk.shape)
            nchunk = int(ra_blk.shape[0])
            if nchunk != t1 - t0:
                return None
            if int(ra_blk.shape[1]) != nl_sz or int(ra_blk.shape[2]) != nm_sz:
                return None

            horiz_chunk = horiz_ok[t0:t1]

            dra = (ra_blk - float(ra) + 180.0) % 360.0 - 180.0
            ddec = dec_blk - float(dec)
            cos_dec = np.cos(np.deg2rad(dec_blk))
            dist2 = dra * dra * cos_dec * cos_dec + ddec * ddec
            dist2 = np.where(
                np.isfinite(ra_blk) & np.isfinite(dec_blk), dist2, np.inf
            )
            dist2[~horiz_chunk, :, :] = np.inf

            nl_b, nm_b = int(ra_blk.shape[1]), int(ra_blk.shape[2])
            flat = np.argmin(dist2.reshape(nchunk, nl_b * nm_b), axis=1)
            li = (flat // nm_b).astype(np.intp)
            mi = (flat % nm_b).astype(np.intp)
            row_min = dist2.reshape(nchunk, nl_b * nm_b)[np.arange(nchunk), flat]
            in_bounds = (li >= 0) & (li < n_l) & (mi >= 0) & (mi < n_m)
            vis_blk = horiz_chunk & np.isfinite(row_min) & in_bounds

            l_sub = np.full(nchunk, n_l, dtype=int)
            m_sub = np.full(nchunk, n_m, dtype=int)
            l_sub[vis_blk] = li[vis_blk]
            m_sub[vis_blk] = mi[vis_blk]
            l_indices[t0:t1] = l_sub
            m_indices[t0:t1] = m_sub
            visible[t0:t1] = vis_blk

        return l_indices, m_indices, visible

    def _compute_pixel_track_via_per_time_wcs(
        self,
        ra: float,
        dec: float,
        *,
        n_l: int,
        n_m: int,
        n_times: int,
        progress_callback: RadportProgressCallback | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map fixed (RA, Dec) through ``wcs_header_str(time)`` in-process."""
        header_strs = _bulk_per_time_wcs_header_strings(self._obj, n_times)
        table = _build_per_time_wcs_track_table(header_strs)
        return _track_pixels_from_wcs_table(
            table,
            ra,
            dec,
            n_l=n_l,
            n_m=n_m,
            progress_callback=progress_callback,
        )

    def _compute_pixel_track(
        self,
        ra: float,
        dec: float,
        observatory: Any = None,
        *,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        pol: int = 0,
        progress_callback: RadportProgressCallback | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-time-step pixel indices for a fixed (RA, Dec).

        For each dataset time index, uses :meth:`coords_to_pixel` (via
        :meth:`_compute_pixel_at_time`) so pixel selection matches all other
        extraction APIs, except when per-time ``wcs_header_str`` is stored
        (bulk header parse + in-process ``world2pix`` via
        :meth:`_compute_pixel_track_via_per_time_wcs`) or when per-time
        ``right_ascension`` / ``declination`` grids allow a time-chunked batched
        argmin on a ``(time, l, m)`` grid (same horizon and distance rules as
        :meth:`_compute_pixel_at_time`).
        Before the loop, LST for all times is computed once (see
        :meth:`_ensure_lst_deg_vector_cached`) so the analytical branch does not
        repeat Astropy work per index.

        Parameters
        ----------
        ra : float
            Right Ascension in degrees (FK5/J2000).
        dec : float
            Declination in degrees (FK5/J2000).
        observatory : astropy.coordinates.EarthLocation, optional
            Passed to :meth:`coords_to_pixel`. Defaults to OVRO-LWA.
        freq_idx : int, optional
            Frequency index for channelized ``right_ascension`` / ``declination``
            coordinates. Defaults to channel 0 when neither this nor ``freq_mhz``
            is set (see :meth:`coords_to_pixel`).
        freq_mhz : float, optional
            Select frequency by MHz for the same purpose; overrides ``freq_idx``
            when provided.
        pol : int, default 0
            Polarization index for slicing sky coordinates when present.

        Returns
        -------
        l_indices : np.ndarray of int, shape (n_times,)
            Pixel l-index at each time step.
        m_indices : np.ndarray of int, shape (n_times,)
            Pixel m-index at each time step.
        visible : np.ndarray of bool, shape (n_times,)
            True where :meth:`coords_to_pixel` succeeded at that time (source
            above horizon and in-bounds). Otherwise indices are set to
            ``n_l`` / ``n_m`` out-of-range sentinels.
        """
        l_coords = self._obj.coords["l"].values
        m_coords = self._obj.coords["m"].values
        n_l = len(l_coords)
        n_m = len(m_coords)
        n_times = int(self._obj.sizes["time"])

        l_indices = np.empty(n_times, dtype=int)
        m_indices = np.empty(n_times, dtype=int)
        visible = np.zeros(n_times, dtype=bool)

        use_per_time_wcs = _has_per_time_wcs_header_str(self._obj)
        if not use_per_time_wcs:
            self._ensure_lst_deg_vector_cached(observatory)

        if freq_mhz is not None:
            fi = int(self.nearest_freq_idx(freq_mhz))
        elif freq_idx is not None:
            fi = int(freq_idx)
        else:
            fi = 0

        # Per-time FITS WCS (zenith-tracking CRVAL): bulk headers + parallel world2pix.
        if use_per_time_wcs:
            l_indices, m_indices, visible = self._compute_pixel_track_via_per_time_wcs(
                ra,
                dec,
                n_l=n_l,
                n_m=n_m,
                n_times=n_times,
                progress_callback=progress_callback,
            )
            if not np.any(visible):
                warnings.warn(
                    f"Source (RA={ra}°, Dec={dec}°) is never above the horizon "
                    f"during this observation. All output values will be NaN.",
                    stacklevel=3,
                )
            return l_indices, m_indices, visible
        if self._pixel_track_can_batch_time_radec_grids():
            batched = self._compute_pixel_track_batched_radec_grid(
                ra,
                dec,
                observatory,
                fi=fi,
                pol=pol,
                l_coords=l_coords,
                m_coords=m_coords,
                n_l=n_l,
                n_m=n_m,
                n_times=n_times,
            )
            if batched is not None:
                l_indices, m_indices, visible = batched
                _emit_radport_progress(
                    progress_callback,
                    "track",
                    n_times,
                    n_times,
                    "Mapped RA/Dec to image pixels (batched grid)",
                )
                if not np.any(visible):
                    warnings.warn(
                        f"Source (RA={ra}°, Dec={dec}°) is never above the horizon "
                        f"during this observation. All output values will be NaN.",
                        stacklevel=3,
                    )
                return l_indices, m_indices, visible

        _emit_radport_progress(
            progress_callback,
            "track",
            0,
            n_times,
            "Mapping RA/Dec to image pixels",
        )
        batch_size = _radport_progress_batch_size(n_times)
        for start in range(0, n_times, batch_size):
            end = min(start + batch_size, n_times)
            for ti in range(start, end):
                try:
                    li, mi = self.coords_to_pixel(
                        ra,
                        dec,
                        time_idx=ti,
                        observatory=observatory,
                        freq_idx=freq_idx,
                        freq_mhz=freq_mhz,
                        pol=pol,
                    )
                    l_indices[ti] = li
                    m_indices[ti] = mi
                    visible[ti] = True
                except ValueError:
                    l_indices[ti] = n_l
                    m_indices[ti] = n_m
                    visible[ti] = False
            _emit_radport_progress(
                progress_callback,
                "track",
                end,
                n_times,
                "Mapping RA/Dec to image pixels",
            )

        if not np.any(visible):
            warnings.warn(
                f"Source (RA={ra}°, Dec={dec}°) is never above the horizon "
                f"during this observation. All output values will be NaN.",
                stacklevel=3,
            )

        return l_indices, m_indices, visible

    def _use_persisted_wcs_for_pixel_mapping(self) -> bool:
        """True when RA/Dec should map via stored FITS WCS rather than SIN drift.

        Incremental OVRO-LWA Zarr stores one header per ``time`` step (zenith
        ``CRVAL1``/``CRVAL2`` drift). The analytical LST+SIN fallback does not
        follow that phase center and will mis-track fixed-sky sources.
        """
        if _has_per_time_wcs_header_str(self._obj):
            return True
        n_times = int(self._obj.sizes.get("time", 1))
        return n_times <= 1 and _read_wcs_header_str(self._obj, time_idx=0) is not None

    def _coords_to_pixel_via_wcs(
        self,
        ra: float,
        dec: float,
        time_idx: int,
        *,
        var: Literal["SKY", "BEAM"] = "SKY",
    ) -> tuple[int, int] | None:
        """Map (RA, Dec) with the persisted FITS WCS at *time_idx*, if available.

        Matches astrowidget ``get_wcs(ds, time_idx)`` / ``all_world2pix`` on the
        full-resolution image grid. Returns ``None`` when no WCS header is stored.
        """
        if _read_wcs_header_str(self._obj, var=var, time_idx=time_idx) is None:
            return None

        wcs = self._get_wcs(var=var, time_idx=time_idx)
        xp, yp = wcs.all_world2pix(float(ra), float(dec), 0)
        xp_f = float(np.asarray(xp).ravel()[0])
        yp_f = float(np.asarray(yp).ravel()[0])
        if not (np.isfinite(xp_f) and np.isfinite(yp_f)):
            raise ValueError(
                f"Source (RA={ra}, Dec={dec}) is outside the image footprint "
                f"at time index {time_idx}."
            )

        n_l = int(self._obj.sizes["l"])
        n_m = int(self._obj.sizes["m"])
        l_idx = int(np.round(xp_f))
        m_idx = int(np.round(yp_f))
        if not (0 <= l_idx < n_l and 0 <= m_idx < n_m):
            raise ValueError(
                f"Source (RA={ra}, Dec={dec}) maps outside the image FOV "
                f"at time index {time_idx}."
            )
        return l_idx, m_idx

    def _compute_pixel_at_time(
        self,
        ra: float,
        dec: float,
        time_idx: int,
        observatory: Any = None,
        *,
        freq_idx: int = 0,
        pol: int = 0,
    ) -> tuple[int, int]:
        """Compute the pixel index for (RA, Dec) at a single time step.

        When a FITS WCS header is stored (``fits_wcs_header`` or per-time
        ``wcs_header_str``), this method uses :meth:`_get_wcs` and
        ``all_world2pix`` so pixel indices match astrowidget and
        :meth:`plot_wcs`. Otherwise, when the dataset provides
        ``right_ascension`` and ``declination`` coordinates that include a
        ``time`` dimension (or the cube has only one time step), it finds the
        nearest sky pixel by minimizing angular distance on those grids. If those
        coordinates also carry ``frequency`` and/or ``polarization`` dimensions,
        the slice at ``freq_idx`` / ``pol`` is taken first. Otherwise it falls
        back to the closed-form SIN projection using mean sidereal time.

        Parameters
        ----------
        ra, dec : float
            Source coordinates in degrees (FK5/J2000).
        time_idx : int
            Index into the dataset's time dimension.
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location. Used for the horizon check and for the
            analytical fallback. Defaults to OVRO-LWA.
        freq_idx : int, default 0
            Frequency index for slicing per-channel ``right_ascension`` /
            ``declination`` grids when present. Ignored for the analytical path.
        pol : int, default 0
            Polarization index for the same slicing when those coords carry a
            ``polarization`` dimension.

        Returns
        -------
        tuple[int, int]
            ``(l_idx, m_idx)`` pixel indices.

        Raises
        ------
        ValueError
            If the source is below the horizon or outside the image FOV
            at the requested time step.
        """
        from astropy.coordinates import EarthLocation

        if observatory is None:
            from astropy import units as u

            observatory = EarthLocation(
                lat=37.2339 * u.deg, lon=-118.2817 * u.deg, height=1222 * u.m
            )

        if self._use_persisted_wcs_for_pixel_mapping():
            wcs_pixel = self._coords_to_pixel_via_wcs(ra, dec, time_idx)
            if wcs_pixel is not None:
                return wcs_pixel
            if _has_per_time_wcs_header_str(self._obj):
                n_time = int(self._obj.sizes.get("time", 0))
                msg = (
                    f"Missing or invalid WCS metadata for time index {time_idx} "
                    f"(dataset has {n_time} per-time wcs_header_str steps). "
                    "Cannot map fixed (RA, Dec) without the slice WCS."
                )
                raise ValueError(msg)

        lst_deg = self._lst_deg_for_time_index(time_idx, observatory=observatory)

        # SIN projection for horizon / analytical fallback
        ha_rad = np.deg2rad(lst_deg - ra)
        dec_rad = np.deg2rad(dec)
        lat_rad = np.deg2rad(observatory.lat.deg)

        l_val = -np.cos(dec_rad) * np.sin(ha_rad)
        m_val = np.sin(dec_rad) * np.cos(lat_rad) - np.cos(dec_rad) * np.sin(
            lat_rad
        ) * np.cos(ha_rad)

        sin_alt = np.sin(dec_rad) * np.sin(lat_rad) + np.cos(
            dec_rad
        ) * np.cos(lat_rad) * np.cos(ha_rad)
        if sin_alt <= 0:
            raise ValueError(
                f"Source (RA={ra}, Dec={dec}) is below the horizon "
                f"at time index {time_idx}."
            )

        obj = self._obj
        ra_c = obj.coords.get("right_ascension")
        dec_c = obj.coords.get("declination")
        n_times = int(obj.sizes.get("time", 1))
        use_radec_grid = (
            ra_c is not None
            and dec_c is not None
            and {"l", "m"} <= set(ra_c.dims)
            and {"l", "m"} <= set(dec_c.dims)
            and (
                ("time" in ra_c.dims and "time" in obj.coords)
                or (n_times <= 1 and "time" not in ra_c.dims)
            )
        )
        if (
            use_radec_grid
            and "time" in obj.coords
            and ra_c is not None
            and "time" not in ra_c.dims
            and n_times > 1
        ):
            use_radec_grid = False

        if use_radec_grid:
            ra_coord = obj.coords["right_ascension"]
            dec_coord = obj.coords["declination"]
            if "time" in ra_coord.dims:
                ra_sel = ra_coord.isel(time=time_idx)
            else:
                ra_sel = ra_coord
            if "time" in dec_coord.dims:
                dec_sel = dec_coord.isel(time=time_idx)
            else:
                dec_sel = dec_coord

            for dim_name, idx_raw in (
                ("frequency", freq_idx),
                ("polarization", pol),
            ):
                if dim_name in ra_sel.dims:
                    n_d = int(ra_sel.sizes[dim_name])
                    ii = int(np.clip(int(idx_raw), 0, n_d - 1))
                    ra_sel = ra_sel.isel({dim_name: ii})
                if dim_name in dec_sel.dims:
                    n_d = int(dec_sel.sizes[dim_name])
                    ii = int(np.clip(int(idx_raw), 0, n_d - 1))
                    dec_sel = dec_sel.isel({dim_name: ii})

            ra_da = ra_sel.data
            dec_da = dec_sel.data
            if hasattr(ra_da, "compute"):
                ra_arr = np.asarray(ra_da.compute(), dtype=np.float64)
            else:
                ra_arr = np.asarray(ra_da, dtype=np.float64)
            if hasattr(dec_da, "compute"):
                dec_arr = np.asarray(dec_da.compute(), dtype=np.float64)
            else:
                dec_arr = np.asarray(dec_da, dtype=np.float64)
            if ra_arr.shape != dec_arr.shape:
                dec_arr = np.broadcast_to(dec_arr, ra_arr.shape)

            dra = (ra_arr - float(ra) + 180.0) % 360.0 - 180.0
            ddec = dec_arr - float(dec)
            cos_dec = np.cos(np.deg2rad(dec_arr))
            dist2 = dra * dra * cos_dec * cos_dec + ddec * ddec
            dist2 = np.where(np.isfinite(ra_arr) & np.isfinite(dec_arr), dist2, np.inf)

            flat = int(np.argmin(dist2))
            idx_mv = np.unravel_index(flat, dist2.shape)
            dim_map = {d: int(idx_mv[i]) for i, d in enumerate(ra_sel.dims)}
            l_idx = dim_map["l"]
            m_idx = dim_map["m"]

            l_coords = self._obj.coords["l"].values
            m_coords = self._obj.coords["m"].values
            if not (0 <= l_idx < len(l_coords) and 0 <= m_idx < len(m_coords)):
                raise ValueError(
                    f"Source (RA={ra}, Dec={dec}) has no valid sky pixel at "
                    f"time index {time_idx}."
                )
            return l_idx, m_idx

        # Analytical (l, m) + nearest index on l/m axes
        l_coords = self._obj.coords["l"].values
        m_coords = self._obj.coords["m"].values

        l_idx = int(np.argmin(np.abs(l_coords - l_val)))
        m_idx = int(np.argmin(np.abs(m_coords - m_val)))

        if not (l_coords.min() <= l_val <= l_coords.max()):
            raise ValueError(
                f"Source (RA={ra}, Dec={dec}) maps to l={l_val:.4f} which "
                f"is outside the image FOV at time index {time_idx}."
            )
        if not (m_coords.min() <= m_val <= m_coords.max()):
            raise ValueError(
                f"Source (RA={ra}, Dec={dec}) maps to m={m_val:.4f} which "
                f"is outside the image FOV at time index {time_idx}."
            )

        return l_idx, m_idx

    def _resolve_coordinates(
        self,
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        observatory: Any = None,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        pol: int = 0,
        progress_callback: RadportProgressCallback | None = None,
    ) -> (
        tuple[int, int]
        | tuple[np.ndarray, np.ndarray, np.ndarray]
    ):
        """Validate coordinate input and return pixel indices.

        Dispatches to either the fixed-pixel path (l/m) or the per-time
        tracking path (ra/dec) built from repeated :meth:`coords_to_pixel` calls.

        Parameters
        ----------
        ra : float, optional
            Right Ascension in degrees.
        dec : float, optional
            Declination in degrees.
        l : float, optional
            Direction cosine l coordinate.
        m : float, optional
            Direction cosine m coordinate.
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location for RA/Dec tracking. Defaults to OVRO-LWA.
        freq_idx : int, optional
            Passed to :meth:`_compute_pixel_track` for RA/Dec (channelized sky
            coordinates). Default is channel 0 when unset.
        freq_mhz : float, optional
            Select that channel by MHz; overrides ``freq_idx`` when provided.
        pol : int, default 0
            Polarization index for RA/Dec coordinate slicing when present.

        Returns
        -------
        tuple[int, int]
            Fixed pixel indices ``(l_idx, m_idx)`` when l/m provided.
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Per-time ``(l_indices, m_indices, visible)`` when ra/dec provided.

        Raises
        ------
        ValueError
            If input is ambiguous (both pairs, neither pair, or partial pair).
        """
        has_radec = ra is not None or dec is not None
        has_lm = l is not None or m is not None

        if has_radec and has_lm:
            raise ValueError("Provide either (ra, dec) or (l, m), not both.")

        if not has_radec and not has_lm:
            raise ValueError("Must provide either (ra, dec) or (l, m) coordinates.")

        if has_radec:
            if ra is None or dec is None:
                raise ValueError("Both ra and dec must be provided together.")
            return self._compute_pixel_track(
                ra,
                dec,
                observatory=observatory,
                freq_idx=freq_idx,
                freq_mhz=freq_mhz,
                pol=pol,
                progress_callback=progress_callback,
            )

        # l/m path
        if l is None or m is None:
            raise ValueError("Both l and m must be provided together.")
        return self.nearest_lm_idx(l, m)

    # =========================================================================
    # Plotting Methods
    # =========================================================================

    def plot(
        self,
        var: Literal["SKY", "BEAM"] = "SKY",
        time_idx: int | None = None,
        freq_idx: int | None = None,
        pol: int = 0,
        freq_mhz: float | None = None,
        time_mjd: float | None = None,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = False,
        mask_radius: int | None = None,
        figsize: tuple[float, float] = (8, 6),
        add_colorbar: bool = True,
        **kwargs: Any,
    ) -> Figure:
        """Create a visualization of radio data as a 2D image.

        Plots a single snapshot of the data at the specified time, frequency,
        and polarization indices. The resulting image shows intensity values
        in the (l, m) direction cosine coordinate system.

        Parameters
        ----------
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot. Use 'BEAM' only if the dataset contains
            beam data (check with `ds.radport.has_beam`).
        time_idx : int, optional
            Index along the time dimension for the snapshot. Default is 0.
            Ignored if `time_mjd` is provided.
        freq_idx : int, optional
            Index along the frequency dimension for the snapshot. Default is 0.
            Ignored if `freq_mhz` is provided.
        pol : int, default 0
            Index along the polarization dimension.
        freq_mhz : float, optional
            Select frequency by value in MHz. If provided, overrides `freq_idx`.
            Uses the nearest available frequency.
        time_mjd : float, optional
            Select time by MJD value. If provided, overrides `time_idx`.
            Uses the nearest available time.
        cmap : str, default 'inferno'
            Matplotlib colormap name for the image.
        vmin : float, optional
            Minimum value for the color scale. If None and robust=False,
            uses the data minimum.
        vmax : float, optional
            Maximum value for the color scale. If None and robust=False,
            uses the data maximum.
        robust : bool, default False
            If True, compute vmin/vmax using the 2nd and 98th percentiles
            of the data, which is useful for data with outliers.
        mask_radius : int, optional
            If provided, mask pixels outside this radius (in pixels) from
            the image center. Useful for all-sky images where edge pixels
            are invalid. Masked pixels are shown as NaN (transparent).
        figsize : tuple of float, default (8, 6)
            Figure size in inches as (width, height).
        add_colorbar : bool, default True
            Whether to add a colorbar to the plot.
        **kwargs : dict
            Additional keyword arguments passed to `matplotlib.pyplot.imshow`.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib Figure object containing the plot.

        Raises
        ------
        ValueError
            If the requested variable does not exist in the dataset.

        Examples
        --------
        >>> import ovro_lwa_portal
        >>> ds = ovro_lwa_portal.open_dataset("path/to/data.zarr")

        Plot with default settings (first time, frequency, polarization):

        >>> fig = ds.radport.plot()

        Plot a specific time and frequency with custom colormap:

        >>> fig = ds.radport.plot(time_idx=5, freq_idx=10, cmap='viridis')

        Plot by selecting frequency in MHz (more intuitive):

        >>> fig = ds.radport.plot(freq_mhz=50.0)

        Plot with fixed color scale:

        >>> fig = ds.radport.plot(vmin=-1.0, vmax=16.0)

        Plot with robust color scaling for data with outliers:

        >>> fig = ds.radport.plot(robust=True)

        Plot with circular mask to hide invalid edge pixels:

        >>> fig = ds.radport.plot(mask_radius=1800)
        """
        # Validate the requested variable exists
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            msg = f"Variable '{var}' not found in dataset. Available variables: {available}"
            raise ValueError(msg)

        # Resolve frequency selection: freq_mhz takes precedence over freq_idx
        if freq_mhz is not None:
            freq_idx = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is None:
            freq_idx = 0

        # Resolve time selection: time_mjd takes precedence over time_idx
        if time_mjd is not None:
            time_idx = self.nearest_time_idx(time_mjd)
        elif time_idx is None:
            time_idx = 0

        # Extract the 2D slice for plotting
        da = self._obj[var].isel(
            time=time_idx,
            frequency=freq_idx,
            polarization=pol,
        )

        # Build title with metadata
        title = self._build_plot_title(var, time_idx, freq_idx, pol)

        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Compute data for plotting (triggers dask computation if needed)
        data = da.values.copy()  # Copy to allow modification for masking

        # Apply circular mask if requested
        if mask_radius is not None:
            ny, nx = data.shape
            cy, cx = ny // 2, nx // 2
            yy, xx = np.ogrid[:ny, :nx]
            distance_from_center = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            data[distance_from_center > mask_radius] = np.nan

        # Handle robust scaling (after masking, so we only consider valid pixels)
        if robust and vmin is None and vmax is None:
            finite_data = data[np.isfinite(data)]
            if finite_data.size > 0:
                vmin = float(np.percentile(finite_data, 2))
                vmax = float(np.percentile(finite_data, 98))

        # Get coordinate extents for proper axis labeling
        l_vals = da.coords["l"].values
        m_vals = da.coords["m"].values
        extent = [float(l_vals[0]), float(l_vals[-1]),
                  float(m_vals[0]), float(m_vals[-1])]

        # Plot the image.
        # Transpose: xarray dims are (l, m) where l=NAXIS1 (RA/x) and
        # m=NAXIS2 (Dec/y).  imshow maps axis 0→y, axis 1→x, so .T
        # puts l on x and m on y.
        im = ax.imshow(
            data.T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="equal",
            **kwargs,
        )

        # Add colorbar
        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Jy/beam")

        # Set labels and title
        ax.set_xlabel("l (direction cosine)")
        ax.set_ylabel("m (direction cosine)")
        ax.set_title(title)

        fig.tight_layout()

        return fig

    def _build_plot_title(
        self,
        var: str,
        time_idx: int,
        freq_idx: int,
        pol: int,
    ) -> str:
        """Build an informative title for the plot.

        Parameters
        ----------
        var : str
            The variable being plotted.
        time_idx : int
            Time index.
        freq_idx : int
            Frequency index.
        pol : int
            Polarization index.

        Returns
        -------
        str
            Formatted title string.
        """
        # Get time value
        time_val = self._obj.coords["time"].values[time_idx]
        try:
            time_str = f"{float(time_val):.6f} MJD"
        except (TypeError, ValueError):
            time_str = str(time_val)

        # Get frequency value in MHz
        freq_val = self._obj.coords["frequency"].values[freq_idx]
        freq_mhz = float(freq_val) / 1e6

        return f"{var}: t={time_str}, f={freq_mhz:.2f} MHz, pol={pol}"

    # =========================================================================
    # Spatial Cutout Methods
    # =========================================================================

    def cutout(
        self,
        *,
        ra_center: float | None = None,
        dec_center: float | None = None,
        l_center: float | None = None,
        m_center: float | None = None,
        dl: float | None = None,
        dm: float | None = None,
        dra: float | None = None,
        ddec: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        time_idx: int | None = None,
        freq_idx: int | None = None,
        pol: int = 0,
        freq_mhz: float | None = None,
        time_mjd: float | None = None,
    ) -> xr.DataArray:
        """Extract a spatial cutout (rectangular region) from the data.

        Returns a 2D DataArray containing data within the specified
        bounding box for a given time, frequency, and polarization.

        Parameters
        ----------
        ra_center : float, optional
            Center Right Ascension in degrees. Requires ``dec_center``.
            Converted to l/m at the specified time step using WCS.
        dec_center : float, optional
            Center Declination in degrees. Requires ``ra_center``.
        l_center : float, optional
            Center l coordinate of the cutout region. Requires ``m_center``.
        m_center : float, optional
            Center m coordinate of the cutout region. Requires ``l_center``.
        dl : float, optional
            Half-width of the cutout in the l direction (direction cosine).
            The cutout spans [l_center - dl, l_center + dl].
            Use with l_center/m_center.
        dm : float, optional
            Half-width of the cutout in the m direction (direction cosine).
            The cutout spans [m_center - dm, m_center + dm].
            Use with l_center/m_center.
        dra : float, optional
            Half-width of the cutout in RA in degrees.
            Use with ra_center/dec_center. Converted to dl internally
            using the SIN projection at the center declination.
        ddec : float, optional
            Half-width of the cutout in Dec in degrees.
            Use with ra_center/dec_center. Converted to dm internally.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to extract.
        time_idx : int, optional
            Time index. Default is 0. Ignored if `time_mjd` is provided.
        freq_idx : int, optional
            Frequency index. Default is 0. Ignored if `freq_mhz` is provided.
        pol : int, default 0
            Polarization index.
        freq_mhz : float, optional
            Select frequency by value in MHz (overrides `freq_idx`).
        time_mjd : float, optional
            Select time by MJD value (overrides `time_idx`).

        Returns
        -------
        xr.DataArray
            2D DataArray with dimensions (l, m) containing the cutout data.
            Includes metadata attributes: cutout_l_center, cutout_m_center,
            cutout_dl, cutout_dm. When using RA/Dec, also includes
            cutout_ra_center, cutout_dec_center, cutout_dra, cutout_ddec.

        Raises
        ------
        ValueError
            If the requested variable does not exist or cutout is empty.

        Examples
        --------
        >>> # Extract 0.2 x 0.2 region centered at (0, 0) in direction cosines
        >>> cutout = ds.radport.cutout(l_center=0.0, m_center=0.0, dl=0.1, dm=0.1)

        >>> # Extract using celestial coordinates with extent in degrees
        >>> cutout = ds.radport.cutout(ra_center=180.0, dec_center=45.0, dra=5.0, ddec=5.0)
        """
        # Validate variable
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(
                f"Variable '{var}' not found. Available: {available}"
            )

        # Resolve indices
        if freq_mhz is not None:
            freq_idx = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is None:
            freq_idx = 0

        if time_mjd is not None:
            time_idx = self.nearest_time_idx(time_mjd)
        elif time_idx is None:
            time_idx = 0

        # Resolve center coordinates
        has_radec = ra_center is not None or dec_center is not None
        has_lm = l_center is not None or m_center is not None

        if has_radec and has_lm:
            raise ValueError(
                "Provide either (ra_center, dec_center) or (l_center, m_center), not both."
            )
        if not has_radec and not has_lm:
            raise ValueError(
                "Must provide either (ra_center, dec_center) or (l_center, m_center)."
            )

        # Track whether we're in celestial mode for metadata/plotting
        celestial_mode = False

        if has_radec:
            if ra_center is None or dec_center is None:
                raise ValueError(
                    "Both ra_center and dec_center must be provided together."
                )
            celestial_mode = True

            # Convert dra/ddec (degrees) to dl/dm (direction cosines)
            if dra is not None or ddec is not None:
                if dra is None or ddec is None:
                    raise ValueError(
                        "Both dra and ddec must be provided together."
                    )
                dec_rad = np.deg2rad(dec_center)
                cos_dec = np.cos(dec_rad)

                # Near the celestial poles, RA extent maps to ~zero
                # extent in l. The pixel scale sets the minimum
                # meaningful dl.
                l_coords = self._obj.coords["l"].values
                pixel_scale_l = float(np.abs(np.median(np.diff(np.sort(l_coords)))))
                dl = np.deg2rad(dra) * cos_dec
                if dl < pixel_scale_l:
                    raise ValueError(
                        f"At Dec={dec_center}°, the requested dra={dra}° "
                        f"maps to dl={dl:.6f} in direction cosines, which "
                        f"is smaller than the pixel scale ({pixel_scale_l:.6f}). "
                        f"Near the celestial poles, RA extent degenerates. "
                        f"Use dl/dm directly for cutouts near the poles."
                    )
                dm = np.deg2rad(ddec)
            elif dl is None or dm is None:
                raise ValueError(
                    "Must provide cutout extent: either (dra, ddec) in degrees "
                    "or (dl, dm) in direction cosines."
                )

            # Convert RA/Dec to pixel at the resolved time step so the
            # cutout is centred on the correct frame.
            pix_l, pix_m = self.coords_to_pixel(
                ra_center,
                dec_center,
                time_idx=time_idx,
                freq_idx=freq_idx,
                pol=pol,
            )
            l_center = float(self._obj.coords["l"].values[pix_l])
            m_center = float(self._obj.coords["m"].values[pix_m])
        else:
            if l_center is None or m_center is None:
                raise ValueError(
                    "Both l_center and m_center must be provided together."
                )
            if dl is None or dm is None:
                raise ValueError(
                    "Both dl and dm must be provided when using l_center/m_center."
                )

        # Extract 2D slice
        da = self._obj[var].isel(
            time=time_idx,
            frequency=freq_idx,
            polarization=pol,
        )

        # Compute l/m bounds
        l_min, l_max = l_center - dl, l_center + dl
        m_min, m_max = m_center - dm, m_center + dm

        # Handle coordinate ordering (ascending or descending)
        l_coords = da.coords["l"]
        m_coords = da.coords["m"]

        # Determine slice direction based on coordinate ordering
        if float(l_coords[0]) <= float(l_coords[-1]):
            l_slice = slice(l_min, l_max)
        else:
            l_slice = slice(l_max, l_min)

        if float(m_coords[0]) <= float(m_coords[-1]):
            m_slice = slice(m_min, m_max)
        else:
            m_slice = slice(m_max, m_min)

        # Select the cutout region
        cutout = da.sel(l=l_slice, m=m_slice)

        # Check if cutout is empty
        if cutout.size == 0:
            raise ValueError(
                f"Cutout region is empty. Requested l=[{l_min:.3f}, {l_max:.3f}], "
                f"m=[{m_min:.3f}, {m_max:.3f}]. "
                f"Dataset l range: [{float(l_coords.min()):.3f}, {float(l_coords.max()):.3f}], "
                f"m range: [{float(m_coords.min()):.3f}, {float(m_coords.max()):.3f}]."
            )

        # Add metadata attributes
        cutout.attrs["cutout_l_center"] = l_center
        cutout.attrs["cutout_m_center"] = m_center
        cutout.attrs["cutout_dl"] = dl
        cutout.attrs["cutout_dm"] = dm
        cutout.attrs["time_idx"] = time_idx
        cutout.attrs["freq_idx"] = freq_idx
        cutout.attrs["pol"] = pol
        cutout.attrs["celestial_mode"] = celestial_mode

        if celestial_mode:
            cutout.attrs["cutout_ra_center"] = ra_center
            cutout.attrs["cutout_dec_center"] = dec_center
            if dra is not None:
                cutout.attrs["cutout_dra"] = dra
                cutout.attrs["cutout_ddec"] = ddec

        return cutout

    def plot_cutout(
        self,
        *,
        ra_center: float | None = None,
        dec_center: float | None = None,
        l_center: float | None = None,
        m_center: float | None = None,
        dl: float | None = None,
        dm: float | None = None,
        dra: float | None = None,
        ddec: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        time_idx: int | None = None,
        freq_idx: int | None = None,
        pol: int = 0,
        freq_mhz: float | None = None,
        time_mjd: float | None = None,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        figsize: tuple[float, float] = (6, 5),
        add_colorbar: bool = True,
        **kwargs: Any,
    ) -> Figure:
        """Extract and plot a spatial cutout.

        Convenience method that combines `cutout()` with plotting.

        Parameters
        ----------
        ra_center : float, optional
            Center Right Ascension in degrees. Requires ``dec_center``.
        dec_center : float, optional
            Center Declination in degrees. Requires ``ra_center``.
        l_center : float, optional
            Center l coordinate. Requires ``m_center``.
        m_center : float, optional
            Center m coordinate. Requires ``l_center``.
        dl, dm : float, optional
            Half-widths of the cutout in l and m directions (direction cosines).
        dra, ddec : float, optional
            Half-widths of the cutout in degrees.
            Use with ra_center/dec_center for consistent units.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        time_idx : int, optional
            Time index. Default is 0.
        freq_idx : int, optional
            Frequency index. Default is 0.
        pol : int, default 0
            Polarization index.
        freq_mhz : float, optional
            Select frequency by value in MHz.
        time_mjd : float, optional
            Select time by MJD value.
        cmap : str, default 'inferno'
            Matplotlib colormap.
        vmin, vmax : float, optional
            Color scale limits.
        robust : bool, default True
            Use percentile-based color scaling.
        figsize : tuple, default (6, 5)
            Figure size in inches.
        add_colorbar : bool, default True
            Whether to add a colorbar.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the cutout plot.

        Examples
        --------
        >>> fig = ds.radport.plot_cutout(l_center=0.0, m_center=0.0, dl=0.1, dm=0.1, freq_mhz=50.0)
        >>> fig = ds.radport.plot_cutout(ra_center=180.0, dec_center=45.0, dra=5.0, ddec=5.0)
        """
        # Get cutout data
        cutout = self.cutout(
            ra_center=ra_center,
            dec_center=dec_center,
            l_center=l_center,
            m_center=m_center,
            dl=dl,
            dm=dm,
            dra=dra,
            ddec=ddec,
            var=var,
            time_idx=time_idx,
            freq_idx=freq_idx,
            pol=pol,
            freq_mhz=freq_mhz,
            time_mjd=time_mjd,
        )

        # Resolve actual indices for title
        actual_time_idx = cutout.attrs["time_idx"]
        actual_freq_idx = cutout.attrs["freq_idx"]
        celestial_mode = cutout.attrs.get("celestial_mode", False)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Compute data
        data = cutout.values

        # Handle robust scaling
        if robust and vmin is None and vmax is None:
            finite_data = data[np.isfinite(data)]
            if finite_data.size > 0:
                vmin = float(np.percentile(finite_data, 2))
                vmax = float(np.percentile(finite_data, 98))

        # Get coordinate extents and set axis labels
        l_vals = cutout.coords["l"].values
        m_vals = cutout.coords["m"].values

        if celestial_mode and "cutout_dra" in cutout.attrs:
            # Use the requested RA/Dec extent directly — this avoids
            # non-linear SIN projection artifacts when converting l/m
            # pixel corners back to RA/Dec.
            ra_c = cutout.attrs["cutout_ra_center"]
            dec_c = cutout.attrs["cutout_dec_center"]
            dra_val = cutout.attrs["cutout_dra"]
            ddec_val = cutout.attrs["cutout_ddec"]

            # Clamp Dec to physical range and wrap RA to [0, 360)
            ra_min = (ra_c - dra_val) % 360.0
            ra_max = (ra_c + dra_val) % 360.0
            dec_min = max(-90.0, dec_c - ddec_val)
            dec_max = min(90.0, dec_c + ddec_val)

            # If RA wraps across 0°/360°, keep the raw values for
            # a continuous axis (matplotlib handles negative/360+ fine)
            if ra_min > ra_max:
                ra_min = ra_c - dra_val
                ra_max = ra_c + dra_val

            # RA increases to the left on sky images (reversed x-axis)
            extent = [ra_max, ra_min, dec_min, dec_max]

            ax.set_xlabel("RA (degrees)")
            ax.set_ylabel("Dec (degrees)")
        elif celestial_mode:
            # Celestial mode with dl/dm (no dra/ddec) — display in l/m
            # since we can't reliably invert the SIN projection for
            # display extents.
            extent = [
                float(l_vals[0]), float(l_vals[-1]),
                float(m_vals[0]), float(m_vals[-1]),
            ]
            ax.set_xlabel("l (direction cosine)")
            ax.set_ylabel("m (direction cosine)")
        else:
            extent = [
                float(l_vals[0]), float(l_vals[-1]),
                float(m_vals[0]), float(m_vals[-1]),
            ]
            ax.set_xlabel("l (direction cosine)")
            ax.set_ylabel("m (direction cosine)")

        # Plot — transpose (l, m) to put l on x-axis and m on y-axis
        im = ax.imshow(
            data.T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="equal",
            **kwargs,
        )

        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Jy/beam")

        # Build title
        title = self._build_plot_title(var, actual_time_idx, actual_freq_idx, pol)
        if celestial_mode:
            ra_c = cutout.attrs["cutout_ra_center"]
            dec_c = cutout.attrs["cutout_dec_center"]
            title += f"\nRA={ra_c:.4f}°, Dec={dec_c:.4f}°"
        else:
            resolved_l = cutout.attrs["cutout_l_center"]
            resolved_m = cutout.attrs["cutout_m_center"]
            dl_val = cutout.attrs["cutout_dl"]
            dm_val = cutout.attrs["cutout_dm"]
            title += (
                f"\nl=[{resolved_l - dl_val:+.2f},{resolved_l + dl_val:+.2f}], "
                f"m=[{resolved_m - dm_val:+.2f},{resolved_m + dm_val:+.2f}]"
            )

        ax.set_title(title)

        fig.tight_layout()
        return fig

    # =========================================================================
    # Dynamic Spectrum Methods
    # =========================================================================

    def dynamic_spectrum(
        self,
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        observatory: Any = None,
        progress_callback: RadportProgressCallback | None = None,
    ) -> xr.DataArray:
        """Extract a dynamic spectrum (time vs frequency) for a single pixel.

        Returns a 2D DataArray showing how intensity varies across time
        and frequency at the pixel nearest to the specified location.

        When celestial coordinates (ra, dec) are provided, the pixel is
        tracked across time steps as the source drifts due to Earth rotation.
        Time steps where the source is below the horizon are NaN-filled.

        Parameters
        ----------
        ra : float, optional
            Right Ascension in degrees (FK5/J2000). Requires ``dec``.
        dec : float, optional
            Declination in degrees (FK5/J2000). Requires ``ra``.
        l : float, optional
            Target l direction cosine coordinate. Requires ``m``.
        m : float, optional
            Target m direction cosine coordinate. Requires ``l``.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to extract.
        pol : int, default 0
            Polarization index.
        freq_idx : int, optional
            Frequency index for RA/Dec → pixel mapping when sky coordinates
            vary with frequency (passed to :meth:`coords_to_pixel`). Defaults to
            channel 0 when neither ``freq_idx`` nor ``freq_mhz`` is set. Ignored
            for the ``l``/``m`` path.
        freq_mhz : float, optional
            Select frequency by MHz for the same purpose; overrides ``freq_idx``.
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location for RA/Dec tracking. Defaults to OVRO-LWA.
        progress_callback : callable, optional
            ``callback(stage, current, total, message)`` for UI progress.
            Stages are ``track`` (RA/Dec → pixel mapping) and ``extract``
            (per-time pixel I/O).  ``current`` and ``total`` count time steps.

        Returns
        -------
        xr.DataArray
            2D DataArray with dimensions (time, frequency).
            Includes metadata: pixel_l, pixel_m, l_idx, m_idx, pol.
            When ra/dec is used, also includes ra, dec, and tracking=True.

        Examples
        --------
        >>> # Get dynamic spectrum at image center (direction cosines)
        >>> dynspec = ds.radport.dynamic_spectrum(l=0.0, m=0.0)

        >>> # Track a celestial source across time
        >>> dynspec = ds.radport.dynamic_spectrum(ra=180.0, dec=45.0)
        """
        # Validate variable
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(
                f"Variable '{var}' not found. Available: {available}"
            )

        result = self._resolve_coordinates(
            ra=ra,
            dec=dec,
            l=l,
            m=m,
            observatory=observatory,
            freq_idx=freq_idx,
            freq_mhz=freq_mhz,
            pol=pol,
            progress_callback=progress_callback,
        )

        if isinstance(result, tuple) and len(result) == 2:
            # Fixed pixel path (l/m) — single pixel across all times/freqs.
            # Eagerly load small results to avoid dask graph overhead.
            l_idx, m_idx = result
            _emit_radport_progress(
                progress_callback,
                "extract",
                0,
                1,
                "Reading tracked pixel (all times and frequencies)",
            )
            da = _maybe_load(
                self._obj[var].isel(l=l_idx, m=m_idx, polarization=pol)
            )
            _emit_radport_progress(
                progress_callback,
                "extract",
                1,
                1,
                "Reading tracked pixel (all times and frequencies)",
            )

            if "time" in da.dims:
                da = da.sortby("time")
            if "frequency" in da.dims:
                da = da.sortby("frequency")

            da.attrs["pixel_l"] = float(self._obj.coords["l"].values[l_idx])
            da.attrs["pixel_m"] = float(self._obj.coords["m"].values[m_idx])
            da.attrs["l_idx"] = l_idx
            da.attrs["m_idx"] = m_idx
            da.attrs["pol"] = pol

            return da

        # Per-time tracking path (ra/dec)
        # Do NOT sortby("time") here — per-time indices match the dataset order.
        # Sorting data_var would misalign positional indices with data.
        l_indices, m_indices, visible = result
        data_var = self._obj[var].isel(polarization=pol)

        n_times = int(self._obj.sizes["time"])
        n_freqs = int(self._obj.sizes["frequency"])

        time_coords = self._obj.coords["time"].values
        freq_coords = self._obj.coords["frequency"].values

        # Build output array, NaN-filled
        out = np.full((n_times, n_freqs), np.nan)

        vis_times, pixel_rows = self._extract_tracked_pixel_vectors(
            data_var,
            l_indices=l_indices,
            m_indices=m_indices,
            visible=visible,
            progress_callback=progress_callback,
        )
        for ti, row in zip(vis_times, pixel_rows, strict=True):
            out[int(ti)] = np.asarray(row)

        da = xr.DataArray(
            out,
            dims=["time", "frequency"],
            coords={"time": time_coords, "frequency": freq_coords},
            attrs={
                "pixel_l": "tracked",
                "pixel_m": "tracked",
                "l_idx": "tracked",
                "m_idx": "tracked",
                "pol": pol,
                "ra": ra,
                "dec": dec,
                "tracking": True,
            },
        )

        return da

    def _extract_tracked_pixel_vectors(
        self,
        data_var: xr.DataArray,
        *,
        l_indices: np.ndarray,
        m_indices: np.ndarray,
        visible: np.ndarray,
        progress_callback: RadportProgressCallback | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Load one ``(frequency,)`` vector per visible tracked time step."""
        vis_times = np.asarray(np.where(visible)[0], dtype=int)
        vis_l = np.asarray(l_indices, dtype=int)[visible]
        vis_m = np.asarray(m_indices, dtype=int)[visible]

        if vis_times.size == 0:
            return vis_times, []

        total = int(vis_times.size)
        extract_message = f"Extracting {total} tracked pixels"
        _emit_radport_progress(
            progress_callback,
            "extract",
            0,
            total,
            extract_message,
        )

        # One sky pixel for all visible times: single read instead of N graphs.
        if np.all(vis_l == vis_l[0]) and np.all(vis_m == vis_m[0]):
            li = int(vis_l[0])
            mi = int(vis_m[0])
            sel = data_var.isel(l=li, m=mi).isel(time=vis_times)
            with _radport_progress_heartbeat(
                progress_callback,
                stage="extract",
                current=0,
                total=total,
                message=extract_message,
            ):
                loaded = _compute_xarray_dataarray(
                    sel,
                    label=extract_message,
                    progress_callback=progress_callback,
                )
            plane = np.asarray(loaded.data, dtype=np.float64)
            rows_out = _rows_from_tracked_pixel_plane(plane)
            _emit_radport_progress(
                progress_callback,
                "extract",
                total,
                total,
                extract_message,
            )
            return vis_times, rows_out

        plane = _vectorized_tracked_pixel_values(
            data_var,
            vis_times,
            vis_l,
            vis_m,
            progress_callback=progress_callback,
            progress_message=extract_message,
        )
        rows_out = _rows_from_tracked_pixel_plane(plane)
        _emit_radport_progress(
            progress_callback,
            "extract",
            total,
            total,
            extract_message,
        )
        return vis_times, rows_out

    # =========================================================================
    # Patch Statistic Methods
    # =========================================================================

    def _extract_tracked_patch_cubes(
        self,
        *,
        l_indices: np.ndarray,
        m_indices: np.ndarray,
        visible: np.ndarray,
        var: str,
        pol: int,
        radii: list[int],
        progress_callback: RadportProgressCallback | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Load ``(frequency, l, m)`` patches for visible tracked time steps.

        ``radii`` gives the patch half-width in pixels for each visible time step
        (one entry per visible time, in visitation order).

        Patch reads are issued in batches so progress callbacks can report I/O
        through long time axes on large datasets.
        """
        data_var = self._obj[var].isel(polarization=pol)
        n_l = int(self._obj.sizes["l"])
        n_m = int(self._obj.sizes["m"])

        vis_times = np.asarray(np.where(visible)[0], dtype=int)
        vis_l = np.asarray(l_indices, dtype=int)[visible]
        vis_m = np.asarray(m_indices, dtype=int)[visible]

        if len(radii) != len(vis_times):
            msg = (
                f"radii length {len(radii)} must match visible time steps "
                f"{len(vis_times)}"
            )
            raise ValueError(msg)

        if vis_times.size == 0:
            return vis_times, []

        total = int(vis_times.size)
        on_workers = _patch_extract_scheduler() is None
        extract_where = "Dask workers" if on_workers else "notebook kernel"
        extract_message = (
            f"Reading {total} tracked patches from Zarr on {extract_where} "
            f"({'distributed' if on_workers else 'threaded'} fused I/O, "
            "then per-time statistics)"
        )
        _emit_radport_progress(
            progress_callback,
            "extract",
            0,
            total,
            extract_message,
        )

        batch_size = _radport_progress_batch_size(total)
        patches_out: list[np.ndarray] = []

        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_times = vis_times[batch_start:batch_end]
            batch_l = vis_l[batch_start:batch_end]
            batch_m = vis_m[batch_start:batch_end]
            batch_radii = radii[batch_start:batch_end]
            n_batch = int(batch_end - batch_start)
            fused_message = (
                f"Zarr patch read steps {batch_start + 1}–{batch_end} of {total} "
                f"({n_batch} times, one fused compute)"
            )

            stacked, patch_sizes = _stack_tracked_patch_selections(
                data_var,
                batch_times,
                batch_l,
                batch_m,
                batch_radii,
                n_l=n_l,
                n_m=n_m,
            )
            with _radport_progress_heartbeat(
                progress_callback,
                stage="extract",
                current=batch_start,
                total=total,
                message=fused_message,
            ):
                loaded = _compute_xarray_dataarray(
                    stacked,
                    label=fused_message,
                    progress_callback=progress_callback,
                    scheduler=_patch_extract_scheduler(),
                )
            patches_out.extend(
                _split_stacked_patch_cubes(np.asarray(loaded.data), patch_sizes)
            )
            _emit_radport_progress(
                progress_callback,
                "extract",
                batch_end,
                total,
                f"finished Zarr patch read steps {batch_start + 1}–{batch_end} of {total}",
            )

        return vis_times, patches_out

    def patch_statistic(
        self,
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        scale: float = 3.0,
        statistic: PatchStatisticName = "std",
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        observatory: Any = None,
        threshold: float | None = None,
        comparison: PatchStatisticComparison = "gt",
        progress_callback: RadportProgressCallback | None = None,
    ) -> PatchStatisticResult:
        """Compute a spatial statistic on a tracked patch for each time/frequency cell.

        For each time step the patch is centred on the pixel nearest to the
        given celestial or direction-cosine coordinates.  A statistic is
        applied to all pixels in the patch independently for every frequency
        channel, producing a 2D ``(time, frequency)`` map.

        When ``threshold`` is set, a boolean ``selection`` mask marks cells
        where the statistic passes the comparison test.  Unselected cells are
        ``False``; NaN statistic values are always ``False``.  Use
        :meth:`PatchStatisticResult.light_curve`,
        :meth:`PatchStatisticResult.dynamic_spectrum`, or
        :meth:`PatchStatisticResult.spectrum` to extract data with unselected
        cells masked to NaN.

        Parameters
        ----------
        ra, dec : float, optional
            Celestial coordinates in degrees (FK5/J2000).  Requires tracking
            across time when both are provided.
        l, m : float, optional
            Direction-cosine coordinates for a fixed patch centre.
        scale : float, default 3.0
            Patch half-width is ``ceil(scale * max(beam FWHM))`` pixels at each
            time step, using the largest synthesized beam over populated
            frequency channels (empty ``(time, frequency)`` slots are skipped).
        statistic : {'std', 'max', 'min', 'mean', 'mad'}, default 'std'
            Spatial reducer applied within each patch.  ``mad`` is the
            median absolute deviation from the patch median.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to analyse.
        pol : int, default 0
            Polarization index.
        freq_idx, freq_mhz : optional
            Passed to coordinate tracking for RA/Dec pixel mapping.
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location for RA/Dec tracking.
        threshold : float, optional
            Statistic threshold for the ``selection`` mask.  When omitted,
            ``selection`` is ``None``.
        comparison : {'gt', 'ge', 'lt', 'le'}, default 'gt'
            Comparison applied at each ``(time, frequency)`` cell.  The
            cell is selected (``True``) when the statistic satisfies the
            comparison against ``threshold``.  For example, ``comparison='le'``
            selects cells with statistic less than or equal to ``threshold``;
            cells above the threshold are ``False``.
        progress_callback : callable, optional
            ``callback(stage, current, total, message)`` for UI progress.
            Stages are ``extract`` (patch I/O) and ``reduce`` (per-time
            statistics).  ``current`` and ``total`` count visible time steps.

        Notes
        -----
        Patch I/O uses :func:`_patch_extract_scheduler` — distributed workers when a
        :class:`dask.distributed.Client` is active, otherwise threaded reads in the
        notebook kernel.  Per-time statistics use :func:`_patch_reduce_scheduler` —
        distributed workers when a Client is active, otherwise a local process pool.
        Set ``OVRO_RADPORT_EXTRACT_SCHEDULER`` or ``OVRO_RADPORT_PATCH_SCHEDULER`` to
        override schedulers when no Client is active.

        Returns
        -------
        PatchStatisticResult
            Container with ``stat_map``, optional ``selection`` mask, and
            convenience methods for downstream extractions.

        Examples
        --------
        >>> result = ds.radport.patch_statistic(
        ...     ra=299.868, dec=40.734, statistic="std", threshold=0.5, comparison="gt"
        ... )
        >>> dynspec = result.dynamic_spectrum()
        >>> lc = result.light_curve(freq_idx=0)
        """
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")
        if scale <= 0:
            msg = f"scale must be positive, got {scale}"
            raise ValueError(msg)

        self.ensure_patch_metadata_cache(pol=pol, var=var)

        track_freq_idx = freq_idx
        if freq_mhz is not None:
            track_freq_idx = int(self.nearest_freq_idx(freq_mhz))

        resolved = self._resolve_coordinates(
            ra=ra,
            dec=dec,
            l=l,
            m=m,
            observatory=observatory,
            freq_idx=track_freq_idx,
            freq_mhz=freq_mhz,
            pol=pol,
        )

        n_times = int(self._obj.sizes["time"])
        n_freqs = int(self._obj.sizes["frequency"])
        stat_values = np.full((n_times, n_freqs), np.nan, dtype=np.float64)
        tracking = ra is not None and dec is not None

        if isinstance(resolved, tuple) and len(resolved) == 2:
            l_idx, m_idx = resolved
            visible = np.ones(n_times, dtype=bool)
            l_indices = np.full(n_times, int(l_idx), dtype=int)
            m_indices = np.full(n_times, int(m_idx), dtype=int)
        else:
            l_indices, m_indices, visible = resolved

        vis_times = np.asarray(np.where(visible)[0], dtype=int)
        radii = [
            self.patch_radius_pixels(time_idx=int(ti), scale=scale, pol=pol, var=var)
            for ti in vis_times
        ]
        vis_times, patches = self._extract_tracked_patch_cubes(
            l_indices=l_indices,
            m_indices=m_indices,
            visible=visible,
            var=var,
            pol=pol,
            radii=radii,
            progress_callback=progress_callback,
        )
        n_visible = len(patches)
        parallel_reduce = n_visible > 1

        for ti_int, row in _run_batched_time_step_work(
            vis_times,
            patches,
            _process_patch_statistic_time,
            process_args=(statistic,),
            progress_callback=progress_callback,
            stage="reduce",
            progress_label=f"Computing patch {statistic}",
            parallel=parallel_reduce,
        ):
            stat_values[int(ti_int)] = row

        selection_array: np.ndarray | None
        if threshold is not None:
            selection_array = _threshold_patch_selection(
                stat_values,
                threshold=threshold,
                comparison=comparison,
            )
        else:
            selection_array = None

        attrs: dict[str, Any] = {
            "statistic": statistic,
            "scale": scale,
            "variable": var,
            "pol": pol,
            "tracking": tracking,
        }
        if threshold is not None:
            attrs["threshold"] = threshold
            attrs["comparison"] = comparison
        if ra is not None:
            attrs["ra"] = ra
            attrs["dec"] = dec
        if l is not None:
            attrs["l"] = l
            attrs["m"] = m

        stat_map = xr.DataArray(
            stat_values,
            dims=["time", "frequency"],
            coords={
                "time": self._obj.coords["time"].values,
                "frequency": self._obj.coords["frequency"].values,
            },
            attrs=attrs,
            name=f"{var}_patch_{statistic}",
        )

        selection: xr.DataArray | None = None
        if selection_array is not None:
            selection = xr.DataArray(
                selection_array,
                dims=["time", "frequency"],
                coords={
                    "time": self._obj.coords["time"].values,
                    "frequency": self._obj.coords["frequency"].values,
                },
                attrs={
                    "threshold": threshold,
                    "comparison": comparison,
                },
                name="selection",
            )

        return PatchStatisticResult(
            stat_map=stat_map,
            selection=selection,
            threshold=threshold,
            comparison=comparison if threshold is not None else None,
            statistic=statistic,
            scale=scale,
            _accessor=self,
            _ra=ra,
            _dec=dec,
            _l=l,
            _m=m,
            _var=var,
            _pol=pol,
            _track_freq_idx=track_freq_idx,
            _track_freq_mhz=freq_mhz,
            _observatory=observatory,
        )

    def patch_fit(
        self,
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        scale: float = 3.0,
        max_reduced_chi_squared: float = 3.0,
        allow_position_offset: bool = True,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        observatory: Any = None,
        progress_callback: RadportProgressCallback | None = None,
    ) -> PatchFitResult:
        """Fit a 2D Gaussian to a tracked patch for each time/frequency cell.

        For each time step the patch is centred on the pixel nearest to the
        given celestial or direction-cosine coordinates.  A Gaussian with
        parameters ``peak``, ``widthx``, ``widthy``, and ``background`` is fit
        on every frequency channel.  When ``allow_position_offset`` is ``True``
        (default), the peak may shift within the patch; offsets are recorded in
        ``x_offset_map`` and ``y_offset_map``.

        Initial peak guesses use the patch maximum minus median background.
        Gaussian width is fixed from synthesized beam ``BMAJ``/``BMIN`` (``BEAM``
        variable or FITS header) on populated cells; empty ``(time, frequency)``
        slots and cells without beam metadata are left as NaN in the output maps.
        When the nonlinear fit does not converge, statistical estimates are
        returned instead of NaN.

        Diagnostic maps ``center_flux_map`` and ``patch_max_map`` compare the
        tracked-centre pixel with the patch maximum.  Use
        :meth:`PatchFitResult.cell_diagnostics` for a single-cell summary.

        When reduced chi-squared exceeds ``max_reduced_chi_squared`` (default
        3), fit parameters are set to NaN but diagnostics and chi-squared are
        retained.

        Parameters
        ----------
        ra, dec : float, optional
            Celestial coordinates in degrees (FK5/J2000).  Requires tracking
            across time when both are provided.
        l, m : float, optional
            Direction-cosine coordinates for a fixed patch centre.
        scale : float, default 3.0
            Patch half-width is ``ceil(scale * max(beam FWHM))`` pixels at each
            time step, using the largest synthesized beam over populated
            frequency channels (empty ``(time, frequency)`` slots are skipped).
        max_reduced_chi_squared : float, default 3.0
            Maximum acceptable reduced chi-squared.  Cells above this threshold
            have fit parameters set to NaN.
        allow_position_offset : bool, default True
            When ``True``, fit the Gaussian peak position within the patch.
            When ``False``, force the peak to remain at the tracked centre.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to analyse.
        pol : int, default 0
            Polarization index.
        freq_idx, freq_mhz : optional
            Passed to coordinate tracking for RA/Dec pixel mapping.
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location for RA/Dec tracking.
        progress_callback : callable, optional
            ``callback(stage, current, total, message)`` for UI progress.
            Stages are ``extract`` (patch I/O) and ``fit`` (per-time Gaussian
            fits).  ``current`` and ``total`` count visible time steps.

        Notes
        -----
        Patch I/O uses :func:`_patch_extract_scheduler` — distributed workers when a
        :class:`dask.distributed.Client` is active, otherwise threaded reads in the
        notebook kernel.  Per-time Gaussian fits use :func:`_patch_reduce_scheduler`
        — distributed workers when a Client is active, otherwise a local process pool.
        Set ``OVRO_RADPORT_EXTRACT_SCHEDULER`` or ``OVRO_RADPORT_PATCH_SCHEDULER`` to
        override schedulers when no Client is active.

        Returns
        -------
        PatchFitResult
            Parameter, diagnostic, and quality maps for the patch fit.

        Examples
        --------
        >>> result = ds.radport.patch_fit(ra=299.868, dec=40.734, scale=3.0)
        >>> peaks = result.peak_map
        >>> result.cell_diagnostics(time_idx=0, frequency_idx=0)
        """
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")
        if scale <= 0:
            msg = f"scale must be positive, got {scale}"
            raise ValueError(msg)
        if max_reduced_chi_squared <= 0:
            msg = f"max_reduced_chi_squared must be positive, got {max_reduced_chi_squared}"
            raise ValueError(msg)

        self.ensure_patch_metadata_cache(pol=pol, var=var)

        track_freq_idx = freq_idx
        if freq_mhz is not None:
            track_freq_idx = int(self.nearest_freq_idx(freq_mhz))

        resolved = self._resolve_coordinates(
            ra=ra,
            dec=dec,
            l=l,
            m=m,
            observatory=observatory,
            freq_idx=track_freq_idx,
            freq_mhz=freq_mhz,
            pol=pol,
        )

        n_times = int(self._obj.sizes["time"])
        n_freqs = int(self._obj.sizes["frequency"])
        peak_values = np.full((n_times, n_freqs), np.nan, dtype=np.float64)
        x_offset_values = np.full((n_times, n_freqs), np.nan, dtype=np.float64)
        y_offset_values = np.full((n_times, n_freqs), np.nan, dtype=np.float64)
        widthx_values = np.full((n_times, n_freqs), np.nan, dtype=np.float64)
        widthy_values = np.full((n_times, n_freqs), np.nan, dtype=np.float64)
        background_values = np.full((n_times, n_freqs), np.nan, dtype=np.float64)
        chi2_values = np.full((n_times, n_freqs), np.nan, dtype=np.float64)
        center_flux_values = np.full((n_times, n_freqs), np.nan, dtype=np.float64)
        patch_max_values = np.full((n_times, n_freqs), np.nan, dtype=np.float64)
        tracking = ra is not None and dec is not None

        if isinstance(resolved, tuple) and len(resolved) == 2:
            l_idx, m_idx = resolved
            visible = np.ones(n_times, dtype=bool)
            l_indices = np.full(n_times, int(l_idx), dtype=int)
            m_indices = np.full(n_times, int(m_idx), dtype=int)
        else:
            l_indices, m_indices, visible = resolved

        vis_times = np.asarray(np.where(visible)[0], dtype=int)
        radii = [
            self.patch_radius_pixels(time_idx=int(ti), scale=scale, pol=pol, var=var)
            for ti in vis_times
        ]
        patch_radius_values = np.full(n_times, np.nan, dtype=np.float64)
        for ti, radius_px in zip(vis_times, radii, strict=True):
            patch_radius_values[int(ti)] = float(radius_px)

        beam_widths_by_time = {
            int(ti): self.beam_fwhm_pixels_all_frequencies(
                time_idx=int(ti),
                pol=pol,
                var=var,
                skip_empty_cells=True,
                data_var=var,
            )
            for ti in vis_times
        }
        vis_times, patches = self._extract_tracked_patch_cubes(
            l_indices=l_indices,
            m_indices=m_indices,
            visible=visible,
            var=var,
            pol=pol,
            radii=radii,
            progress_callback=progress_callback,
        )
        n_visible = len(patches)
        parallel_fit = n_visible > 1

        for (
            ti_int,
            peaks,
            x_offs,
            y_offs,
            widthxs,
            widthys,
            backgrounds,
            chi2_red,
            center_flux,
            patch_max,
        ) in _run_batched_patch_fit_work(
            vis_times,
            patches,
            beam_widths_by_time=beam_widths_by_time,
            allow_position_offset=allow_position_offset,
            max_reduced_chi_squared=max_reduced_chi_squared,
            progress_callback=progress_callback,
            parallel=parallel_fit,
        ):
            peak_values[ti_int] = peaks
            x_offset_values[ti_int] = x_offs
            y_offset_values[ti_int] = y_offs
            widthx_values[ti_int] = widthxs
            widthy_values[ti_int] = widthys
            background_values[ti_int] = backgrounds
            chi2_values[ti_int] = chi2_red
            center_flux_values[ti_int] = center_flux
            patch_max_values[ti_int] = patch_max

        fit_accepted_values = np.isfinite(chi2_values) & (
            chi2_values <= max_reduced_chi_squared
        )

        attrs: dict[str, Any] = {
            "scale": scale,
            "max_reduced_chi_squared": max_reduced_chi_squared,
            "allow_position_offset": allow_position_offset,
            "variable": var,
            "pol": pol,
            "tracking": tracking,
        }
        if ra is not None:
            attrs["ra"] = ra
            attrs["dec"] = dec
        if l is not None:
            attrs["l"] = l
            attrs["m"] = m

        coords = {
            "time": self._obj.coords["time"].values,
            "frequency": self._obj.coords["frequency"].values,
        }

        peak_map = xr.DataArray(
            peak_values,
            dims=["time", "frequency"],
            coords=coords,
            attrs=attrs,
            name=f"{var}_patch_peak",
        )
        widthx_map = xr.DataArray(
            widthx_values,
            dims=["time", "frequency"],
            coords=coords,
            attrs=attrs,
            name=f"{var}_patch_widthx",
        )
        widthy_map = xr.DataArray(
            widthy_values,
            dims=["time", "frequency"],
            coords=coords,
            attrs=attrs,
            name=f"{var}_patch_widthy",
        )
        background_map = xr.DataArray(
            background_values,
            dims=["time", "frequency"],
            coords=coords,
            attrs=attrs,
            name=f"{var}_patch_background",
        )
        reduced_chi_squared_map = xr.DataArray(
            chi2_values,
            dims=["time", "frequency"],
            coords=coords,
            attrs=attrs,
            name=f"{var}_patch_reduced_chi_squared",
        )
        x_offset_map = xr.DataArray(
            x_offset_values,
            dims=["time", "frequency"],
            coords=coords,
            attrs=attrs,
            name=f"{var}_patch_x_offset",
        )
        y_offset_map = xr.DataArray(
            y_offset_values,
            dims=["time", "frequency"],
            coords=coords,
            attrs=attrs,
            name=f"{var}_patch_y_offset",
        )
        center_flux_map = xr.DataArray(
            center_flux_values,
            dims=["time", "frequency"],
            coords=coords,
            attrs=attrs,
            name=f"{var}_patch_center_flux",
        )
        patch_max_map = xr.DataArray(
            patch_max_values,
            dims=["time", "frequency"],
            coords=coords,
            attrs=attrs,
            name=f"{var}_patch_max",
        )
        fit_accepted_map = xr.DataArray(
            fit_accepted_values,
            dims=["time", "frequency"],
            coords=coords,
            attrs=attrs,
            name=f"{var}_patch_fit_accepted",
        )
        patch_radius_map = xr.DataArray(
            patch_radius_values,
            dims=["time"],
            coords={"time": self._obj.coords["time"].values},
            attrs=attrs,
            name=f"{var}_patch_radius",
        )

        return PatchFitResult(
            peak_map=peak_map,
            widthx_map=widthx_map,
            widthy_map=widthy_map,
            background_map=background_map,
            reduced_chi_squared_map=reduced_chi_squared_map,
            x_offset_map=x_offset_map,
            y_offset_map=y_offset_map,
            center_flux_map=center_flux_map,
            patch_max_map=patch_max_map,
            fit_accepted_map=fit_accepted_map,
            patch_radius_map=patch_radius_map,
            scale=scale,
            max_reduced_chi_squared=max_reduced_chi_squared,
            allow_position_offset=allow_position_offset,
            _accessor=self,
            _ra=ra,
            _dec=dec,
            _l=l,
            _m=m,
            _var=var,
            _pol=pol,
            _track_freq_idx=track_freq_idx,
            _track_freq_mhz=freq_mhz,
            _observatory=observatory,
        )

    def patch_fit_cell(
        self,
        time_idx: int,
        frequency_idx: int,
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        scale: float = 3.0,
        max_reduced_chi_squared: float = 3.0,
        allow_position_offset: bool = True,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        observatory: Any = None,
    ) -> PatchFitCellResult:
        """Fit a 2D Gaussian patch on one ``(time, frequency)`` overlay cell.

        Resolves the patch centre at ``time_idx``, extracts the spatial plane
        for ``frequency_idx``, and returns diagnostics suitable for the source
        review **Fit overlay** workflow.

        Parameters
        ----------
        time_idx, frequency_idx : int
            Dataset indices for the overlay slice to fit.
        ra, dec : float, optional
            Celestial coordinates in degrees.  Requires both.
        l, m : float, optional
            Fixed direction-cosine patch centre.  Requires both.
        scale : float, default 3.0
            Patch half-width multiplier (see :meth:`patch_radius_pixels`).
        max_reduced_chi_squared : float, default 3.0
            Fit parameters are masked when reduced chi-squared exceeds this value.
        allow_position_offset : bool, default True
            When ``False``, fix the Gaussian peak at the tracked patch centre.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to analyse.
        pol : int, default 0
            Polarization index.
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location for RA/Dec pixel mapping.

        Returns
        -------
        PatchFitCellResult
            Single-cell fit diagnostics.

        Raises
        ------
        ValueError
            When coordinates are ambiguous, the cell has no finite data, or beam
            metadata is unavailable for the requested cell.
        """
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")
        if scale <= 0:
            msg = f"scale must be positive, got {scale}"
            raise ValueError(msg)
        if max_reduced_chi_squared <= 0:
            msg = f"max_reduced_chi_squared must be positive, got {max_reduced_chi_squared}"
            raise ValueError(msg)

        ti = int(time_idx)
        fi = int(frequency_idx)
        self.ensure_patch_metadata_cache(pol=pol, var=var)

        if not self._var_cell_has_finite_data(
            time_idx=ti,
            frequency_idx=fi,
            pol=pol,
            var=var,
        ):
            msg = f"No finite data in {var} at time_idx={ti}, frequency_idx={fi}"
            raise ValueError(msg)

        has_radec = ra is not None or dec is not None
        has_lm = l is not None or m is not None
        if has_radec and has_lm:
            raise ValueError("Provide either (ra, dec) or (l, m), not both.")
        if not has_radec and not has_lm:
            raise ValueError("Must provide either (ra, dec) or (l, m) coordinates.")
        if has_radec:
            if ra is None or dec is None:
                raise ValueError("Both ra and dec must be provided together.")
            l_idx, m_idx = self.coords_to_pixel(
                float(ra),
                float(dec),
                time_idx=ti,
                observatory=observatory,
                freq_idx=fi,
                pol=pol,
            )
        else:
            assert l is not None and m is not None
            resolved = self._resolve_coordinates(l=l, m=m, pol=pol)
            if not isinstance(resolved, tuple) or len(resolved) != 2:
                msg = "patch_fit_cell with (l, m) requires fixed sky coordinates"
                raise ValueError(msg)
            l_idx, m_idx = (int(resolved[0]), int(resolved[1]))

        radius = int(self.patch_radius_pixels(time_idx=ti, scale=scale, pol=pol, var=var))
        if radius <= 0:
            msg = (
                f"Patch radius is zero at time_idx={ti} "
                "(no populated frequency channels with beam metadata)"
            )
            raise ValueError(msg)

        n_l = int(self._obj.sizes["l"])
        n_m = int(self._obj.sizes["m"])
        l_sl, m_sl = _patch_slices_from_center(l_idx, m_idx, radius, n_l=n_l, n_m=n_m)
        data_var = self._obj[var].isel(polarization=int(pol))
        plane = data_var.isel(time=ti, frequency=fi, l=l_sl, m=m_sl)
        if _data_var_is_dask_backed(plane):
            plane = _compute_xarray_dataarray(plane, label="patch_fit_cell", quiet=True)
        patch_2d = np.asarray(plane.values, dtype=np.float64)
        if patch_2d.ndim != 2:
            msg = f"Expected 2D spatial patch, got shape {patch_2d.shape}"
            raise ValueError(msg)

        beam_wx, beam_wy = self.beam_fwhm_pixels(
            time_idx=ti,
            frequency_idx=fi,
            pol=pol,
            var=var,
        )
        center_flux, patch_max = _patch_plane_center_and_max(patch_2d)

        if (
            not np.isfinite(beam_wx)
            or not np.isfinite(beam_wy)
            or beam_wx <= 0
            or beam_wy <= 0
        ):
            msg = (
                f"Synthesized beam metadata unavailable at time_idx={ti}, "
                f"frequency_idx={fi}"
            )
            raise ValueError(msg)

        peak, x_off, y_off, widthx, widthy, background, chi2_red = _fit_spatial_gaussian(
            patch_2d,
            beam_widthx=beam_wx,
            beam_widthy=beam_wy,
            allow_position_offset=allow_position_offset,
        )
        (
            peak_arr,
            widthx_arr,
            widthy_arr,
            background_arr,
            x_off_arr,
            y_off_arr,
        ) = _mask_patch_fit_by_chi2(
            np.array([peak], dtype=np.float64),
            np.array([widthx], dtype=np.float64),
            np.array([widthy], dtype=np.float64),
            np.array([background], dtype=np.float64),
            np.array([chi2_red], dtype=np.float64),
            max_reduced_chi_squared=max_reduced_chi_squared,
            x_offsets=np.array([x_off], dtype=np.float64),
            y_offsets=np.array([y_off], dtype=np.float64),
        )
        peak = float(peak_arr[0])
        x_off = float(x_off_arr[0]) if x_off_arr is not None else float("nan")
        y_off = float(y_off_arr[0]) if y_off_arr is not None else float("nan")
        widthx = float(widthx_arr[0])
        widthy = float(widthy_arr[0])
        background = float(background_arr[0])
        fit_accepted = bool(np.isfinite(chi2_red) and chi2_red <= max_reduced_chi_squared)

        if np.isfinite(x_off) and np.isfinite(y_off):
            l_peak = int(np.clip(int(round(l_idx + y_off)), 0, n_l - 1))
            m_peak = int(np.clip(int(round(m_idx + x_off)), 0, n_m - 1))
            try:
                peak_ra_deg, peak_dec_deg = self.pixel_to_coords(
                    l_peak,
                    m_peak,
                    time_idx=ti,
                    observatory=observatory,
                )
            except ValueError:
                peak_ra_deg, peak_dec_deg = float("nan"), float("nan")
        else:
            peak_ra_deg, peak_dec_deg = float("nan"), float("nan")
        peak_ra_str, peak_dec_str = format_radec_sexagesimal(peak_ra_deg, peak_dec_deg)
        peak_offset = (
            float(np.hypot(x_off, y_off)) if np.isfinite(x_off) and np.isfinite(y_off) else float("nan")
        )

        return PatchFitCellResult(
            time_idx=ti,
            frequency_idx=fi,
            fit_accepted=fit_accepted,
            reduced_chi_squared=float(chi2_red),
            peak=peak,
            peak_ra_deg=peak_ra_deg,
            peak_dec_deg=peak_dec_deg,
            peak_ra=peak_ra_str,
            peak_dec=peak_dec_str,
            x_offset_pixels=x_off,
            y_offset_pixels=y_off,
            peak_offset_pixels=peak_offset,
            center_flux=center_flux,
            patch_max=patch_max,
            background=background,
            widthx=widthx,
            widthy=widthy,
            scale=float(scale),
            max_reduced_chi_squared=float(max_reduced_chi_squared),
            allow_position_offset=bool(allow_position_offset),
            patch_radius_pixels=radius,
        )

    def select_patch_statistic(
        self,
        stat_map: xr.DataArray,
        *,
        threshold: float,
        comparison: PatchStatisticComparison = "gt",
    ) -> xr.DataArray:
        """Build a boolean ``(time, frequency)`` selection mask from a statistic map.

        Parameters
        ----------
        stat_map : xr.DataArray
            2D ``(time, frequency)`` array, e.g. from
            :meth:`patch_statistic` ``.stat_map``.
        threshold : float
            Statistic threshold.
        comparison : {'gt', 'ge', 'lt', 'le'}, default 'gt'
            Comparison for selection.  Cells where the statistic satisfies
            the comparison are ``True``; others are ``False``.

        Returns
        -------
        xr.DataArray
            Boolean mask with dimensions ``(time, frequency)``.
        """
        if set(stat_map.dims) != {"time", "frequency"}:
            msg = f"stat_map must have dims ('time', 'frequency'), got {stat_map.dims}"
            raise ValueError(msg)
        mask = _threshold_patch_selection(
            np.asarray(stat_map.values, dtype=np.float64),
            threshold=threshold,
            comparison=comparison,
        )
        return xr.DataArray(
            mask,
            dims=["time", "frequency"],
            coords={
                "time": stat_map.coords["time"].values,
                "frequency": stat_map.coords["frequency"].values,
            },
            attrs={"threshold": threshold, "comparison": comparison},
            name="selection",
        )

    def plot_dynamic_spectrum(
        self,
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        figsize: tuple[float, float] = (8, 5),
        add_colorbar: bool = True,
        observatory: Any = None,
        **kwargs: Any,
    ) -> Figure:
        """Plot a dynamic spectrum (time vs frequency) for a single pixel.

        Creates a 2D visualization showing intensity variations across
        time and frequency at a specified location.

        Parameters
        ----------
        ra : float, optional
            Right Ascension in degrees (FK5/J2000). Requires ``dec``.
        dec : float, optional
            Declination in degrees (FK5/J2000). Requires ``ra``.
        l : float, optional
            Target l direction cosine coordinate. Requires ``m``.
        m : float, optional
            Target m direction cosine coordinate. Requires ``l``.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        pol : int, default 0
            Polarization index.
        freq_idx : int, optional
            Passed to :meth:`dynamic_spectrum` for RA/Dec pixel selection.
        freq_mhz : float, optional
            Passed to :meth:`dynamic_spectrum`; overrides ``freq_idx``.
        cmap : str, default 'inferno'
            Matplotlib colormap.
        vmin, vmax : float, optional
            Color scale limits.
        robust : bool, default True
            Use percentile-based color scaling.
        figsize : tuple, default (8, 5)
            Figure size in inches.
        add_colorbar : bool, default True
            Whether to add a colorbar.
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location for RA/Dec tracking. Defaults to OVRO-LWA.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the dynamic spectrum plot.

        Examples
        --------
        >>> fig = ds.radport.plot_dynamic_spectrum(l=0.0, m=0.0)
        >>> fig = ds.radport.plot_dynamic_spectrum(ra=180.0, dec=45.0)
        """
        # Get dynamic spectrum
        dynspec = self.dynamic_spectrum(
            ra=ra,
            dec=dec,
            l=l,
            m=m,
            var=var,
            pol=pol,
            freq_idx=freq_idx,
            freq_mhz=freq_mhz,
            observatory=observatory,
        )

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Compute data
        data = dynspec.values

        # Handle robust scaling
        if robust and vmin is None and vmax is None:
            finite_data = data[np.isfinite(data)]
            if finite_data.size > 0:
                vmin = float(np.percentile(finite_data, 2))
                vmax = float(np.percentile(finite_data, 98))

        # Get coordinate values
        time_vals = dynspec.coords["time"].values
        freq_vals = dynspec.coords["frequency"].values / 1e6  # Convert to MHz

        # Compute extent for imshow
        # extent = [xmin, xmax, ymin, ymax]
        extent = [
            float(time_vals.min()), float(time_vals.max()),
            float(freq_vals.min()), float(freq_vals.max()),
        ]

        # Plot - transpose so time is x-axis and frequency is y-axis
        im = ax.imshow(
            data.T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="auto",
            **kwargs,
        )

        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Jy/beam")

        # Labels and title
        ax.set_xlabel("Time (MJD)")
        ax.set_ylabel("Frequency (MHz)")

        if dynspec.attrs.get("tracking"):
            ra_val = dynspec.attrs["ra"]
            dec_val = dynspec.attrs["dec"]
            ax.set_title(
                f"{var} Dynamic Spectrum (tracked) at RA={ra_val:.4f}°, "
                f"Dec={dec_val:.4f}°, pol={pol}"
            )
        else:
            pixel_l = dynspec.attrs["pixel_l"]
            pixel_m = dynspec.attrs["pixel_m"]
            ax.set_title(
                f"{var} Dynamic Spectrum at l={pixel_l:+.4f}, m={pixel_m:+.4f}, pol={pol}"
            )

        fig.tight_layout()
        return fig

    # =========================================================================
    # Difference Map Methods
    # =========================================================================

    def diff(
        self,
        mode: Literal["time", "frequency"] = "time",
        var: Literal["SKY", "BEAM"] = "SKY",
        time_idx: int | None = None,
        freq_idx: int | None = None,
        pol: int = 0,
        freq_mhz: float | None = None,
        time_mjd: float | None = None,
    ) -> xr.DataArray:
        """Compute a difference map between adjacent time or frequency slices.

        Useful for identifying transient sources or spectral features by
        subtracting consecutive frames.

        Parameters
        ----------
        mode : {'time', 'frequency'}, default 'time'
            Difference mode:
            - 'time': Subtract previous time step from current (at fixed freq)
            - 'frequency': Subtract previous frequency from current (at fixed time)
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to difference.
        time_idx : int, optional
            Time index for the "current" frame. Default is 1.
            For mode='time', differences frame[time_idx] - frame[time_idx-1].
        freq_idx : int, optional
            Frequency index for the "current" frame. Default is 1.
            For mode='frequency', differences frame[freq_idx] - frame[freq_idx-1].
        pol : int, default 0
            Polarization index.
        freq_mhz : float, optional
            Select frequency by value in MHz.
        time_mjd : float, optional
            Select time by MJD value.

        Returns
        -------
        xr.DataArray
            2D DataArray with dimensions (l, m) containing the difference.
            Includes metadata: diff_mode, idx1, idx2.

        Raises
        ------
        ValueError
            If indices are out of bounds for differencing.

        Examples
        --------
        >>> # Time difference at fixed frequency
        >>> diff = ds.radport.diff(mode='time', time_idx=5, freq_mhz=50.0)

        >>> # Frequency difference at fixed time
        >>> diff = ds.radport.diff(mode='frequency', freq_idx=10, time_idx=0)
        """
        # Validate variable
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(
                f"Variable '{var}' not found. Available: {available}"
            )

        # Resolve indices
        if freq_mhz is not None:
            freq_idx = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is None:
            freq_idx = 1 if mode == "frequency" else 0

        if time_mjd is not None:
            time_idx = self.nearest_time_idx(time_mjd)
        elif time_idx is None:
            time_idx = 1 if mode == "time" else 0

        # Compute difference based on mode
        if mode == "time":
            if time_idx < 1:
                raise ValueError(
                    f"time_idx must be >= 1 for time differencing. Got {time_idx}."
                )
            n_times = self._obj.sizes["time"]
            if time_idx >= n_times:
                raise ValueError(
                    f"time_idx {time_idx} out of bounds (dataset has {n_times} times)."
                )

            frame_current = self._obj[var].isel(
                time=time_idx, frequency=freq_idx, polarization=pol
            )
            frame_prev = self._obj[var].isel(
                time=time_idx - 1, frequency=freq_idx, polarization=pol
            )
            diff = frame_current - frame_prev

            diff.attrs["diff_mode"] = "time"
            diff.attrs["time_idx_current"] = time_idx
            diff.attrs["time_idx_prev"] = time_idx - 1
            diff.attrs["freq_idx"] = freq_idx

        else:  # mode == "frequency"
            if freq_idx < 1:
                raise ValueError(
                    f"freq_idx must be >= 1 for frequency differencing. Got {freq_idx}."
                )
            n_freqs = self._obj.sizes["frequency"]
            if freq_idx >= n_freqs:
                raise ValueError(
                    f"freq_idx {freq_idx} out of bounds (dataset has {n_freqs} frequencies)."
                )

            frame_current = self._obj[var].isel(
                time=time_idx, frequency=freq_idx, polarization=pol
            )
            frame_prev = self._obj[var].isel(
                time=time_idx, frequency=freq_idx - 1, polarization=pol
            )
            diff = frame_current - frame_prev

            diff.attrs["diff_mode"] = "frequency"
            diff.attrs["freq_idx_current"] = freq_idx
            diff.attrs["freq_idx_prev"] = freq_idx - 1
            diff.attrs["time_idx"] = time_idx

        diff.attrs["pol"] = pol
        return diff

    def plot_diff(
        self,
        mode: Literal["time", "frequency"] = "time",
        var: Literal["SKY", "BEAM"] = "SKY",
        time_idx: int | None = None,
        freq_idx: int | None = None,
        pol: int = 0,
        freq_mhz: float | None = None,
        time_mjd: float | None = None,
        cmap: str = "RdBu_r",
        vmin: float | None = None,
        vmax: float | None = None,
        symmetric: bool = True,
        figsize: tuple[float, float] = (8, 6),
        add_colorbar: bool = True,
        **kwargs: Any,
    ) -> Figure:
        """Plot a difference map between adjacent time or frequency slices.

        Parameters
        ----------
        mode : {'time', 'frequency'}, default 'time'
            Difference mode.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to difference.
        time_idx : int, optional
            Time index for the "current" frame.
        freq_idx : int, optional
            Frequency index for the "current" frame.
        pol : int, default 0
            Polarization index.
        freq_mhz : float, optional
            Select frequency by value in MHz.
        time_mjd : float, optional
            Select time by MJD value.
        cmap : str, default 'RdBu_r'
            Colormap (diverging colormaps work well for differences).
        vmin, vmax : float, optional
            Color scale limits.
        symmetric : bool, default True
            If True and vmin/vmax not specified, use symmetric color scale
            centered on zero.
        figsize : tuple, default (8, 6)
            Figure size in inches.
        add_colorbar : bool, default True
            Whether to add a colorbar.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the difference plot.

        Examples
        --------
        >>> fig = ds.radport.plot_diff(mode='time', time_idx=5, freq_mhz=50.0)
        """
        # Get difference data
        diff = self.diff(
            mode=mode,
            var=var,
            time_idx=time_idx,
            freq_idx=freq_idx,
            pol=pol,
            freq_mhz=freq_mhz,
            time_mjd=time_mjd,
        )

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Compute data
        data = diff.values

        # Handle symmetric scaling
        if symmetric and vmin is None and vmax is None:
            finite_data = data[np.isfinite(data)]
            if finite_data.size > 0:
                max_abs = float(np.percentile(np.abs(finite_data), 98))
                vmin, vmax = -max_abs, max_abs

        # Get coordinate extents
        l_vals = diff.coords["l"].values
        m_vals = diff.coords["m"].values
        extent = [
            float(l_vals[0]), float(l_vals[-1]),
            float(m_vals[0]), float(m_vals[-1]),
        ]

        # Plot — transpose (l, m) to put l on x-axis and m on y-axis
        im = ax.imshow(
            data.T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="equal",
            **kwargs,
        )

        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("ΔJy/beam")

        # Build title
        if mode == "time":
            t_curr = diff.attrs["time_idx_current"]
            t_prev = diff.attrs["time_idx_prev"]
            f_idx = diff.attrs["freq_idx"]
            freq_mhz_val = float(self._obj.coords["frequency"].values[f_idx]) / 1e6
            title = f"{var} Time Diff (t{t_curr} - t{t_prev}) at f={freq_mhz_val:.2f} MHz"
        else:
            f_curr = diff.attrs["freq_idx_current"]
            f_prev = diff.attrs["freq_idx_prev"]
            t_idx = diff.attrs["time_idx"]
            freq_curr_mhz = float(self._obj.coords["frequency"].values[f_curr]) / 1e6
            freq_prev_mhz = float(self._obj.coords["frequency"].values[f_prev]) / 1e6
            time_val = self._obj.coords["time"].values[t_idx]
            title = f"{var} Freq Diff ({freq_curr_mhz:.1f} - {freq_prev_mhz:.1f} MHz) at t={float(time_val):.6f}"

        ax.set_xlabel("l (direction cosine)")
        ax.set_ylabel("m (direction cosine)")
        ax.set_title(title)

        fig.tight_layout()
        return fig

    # =========================================================================
    # Data Quality Methods
    # =========================================================================

    def find_valid_frame(
        self,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        min_finite_fraction: float = 0.1,
    ) -> tuple[int, int]:
        """Find the first (time, freq) frame with sufficient finite data.

        Searches through time and frequency indices to find a frame where
        at least `min_finite_fraction` of pixels contain finite (non-NaN) values.
        Useful for automatically selecting a valid frame for visualization.

        Parameters
        ----------
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to check.
        pol : int, default 0
            Polarization index.
        min_finite_fraction : float, default 0.1
            Minimum fraction of finite pixels required (0 to 1).

        Returns
        -------
        tuple of int
            (time_idx, freq_idx) of the first valid frame.

        Raises
        ------
        ValueError
            If no valid frame is found.

        Examples
        --------
        >>> time_idx, freq_idx = ds.radport.find_valid_frame()
        >>> fig = ds.radport.plot(time_idx=time_idx, freq_idx=freq_idx)
        """
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")

        da = self._obj[var].isel(polarization=pol)

        # Compute fraction of finite values for each (time, freq) plane
        finite_frac = np.isfinite(da).mean(dim=("l", "m"))

        # If data is lazy (dask), compute it
        if hasattr(finite_frac, "compute"):
            finite_frac = finite_frac.compute()

        arr = finite_frac.values

        # Search for first valid frame
        for ti in range(arr.shape[0]):
            for fi in range(arr.shape[1]):
                if arr[ti, fi] >= min_finite_fraction:
                    return ti, fi

        raise ValueError(
            f"No valid frame found with at least {min_finite_fraction:.0%} finite pixels. "
            f"Dataset may contain all NaN values."
        )

    def finite_fraction(
        self,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
    ) -> xr.DataArray:
        """Compute the fraction of finite (non-NaN) pixels for each (time, freq).

        Returns a 2D DataArray showing data availability across all
        time and frequency combinations. Useful for identifying which
        frames contain valid data.

        Parameters
        ----------
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to check.
        pol : int, default 0
            Polarization index.

        Returns
        -------
        xr.DataArray
            2D array with dimensions (time, frequency) containing fractions
            from 0 (all NaN) to 1 (all finite).

        Examples
        --------
        >>> frac = ds.radport.finite_fraction()
        >>> frac.plot()  # Visualize data availability
        """
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")

        da = self._obj[var].isel(polarization=pol)
        finite_frac = np.isfinite(da).mean(dim=("l", "m"))

        finite_frac.attrs["variable"] = var
        finite_frac.attrs["pol"] = pol

        return finite_frac

    # =========================================================================
    # Grid Plot Methods
    # =========================================================================

    def plot_grid(
        self,
        time_indices: list[int] | None = None,
        freq_indices: list[int] | None = None,
        freq_mhz_list: list[float] | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        ncols: int = 4,
        panel_size: tuple[float, float] = (3.0, 2.6),
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        mask_radius: int | None = None,
        share_colorbar: bool = True,
        **kwargs: Any,
    ) -> Figure:
        """Create a grid of plots showing multiple time/frequency combinations.

        Useful for comparing observations across time or frequency in a
        single figure with consistent scaling.

        Parameters
        ----------
        time_indices : list of int, optional
            Time indices to plot. If None, uses all available times.
        freq_indices : list of int, optional
            Frequency indices to plot. If None, uses all available frequencies.
            Ignored if `freq_mhz_list` is provided.
        freq_mhz_list : list of float, optional
            Frequencies in MHz to plot. Overrides `freq_indices`.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        pol : int, default 0
            Polarization index.
        ncols : int, default 4
            Number of columns in the grid.
        panel_size : tuple of float, default (3.0, 2.6)
            Size of each panel in inches as (width, height).
        cmap : str, default 'inferno'
            Matplotlib colormap.
        vmin, vmax : float, optional
            Color scale limits. Applied to all panels.
        robust : bool, default True
            If True and vmin/vmax not specified, compute global percentile
            scaling across all panels.
        mask_radius : int, optional
            Circular mask radius in pixels.
        share_colorbar : bool, default True
            If True, show a single shared colorbar for all panels.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the grid of plots.

        Examples
        --------
        >>> # Plot all times at a single frequency
        >>> fig = ds.radport.plot_grid(freq_mhz_list=[50.0])

        >>> # Plot specific times and frequencies
        >>> fig = ds.radport.plot_grid(
        ...     time_indices=[0, 1, 2],
        ...     freq_mhz_list=[46.0, 50.0, 54.0],
        ... )

        >>> # Plot first 4 times at all frequencies
        >>> fig = ds.radport.plot_grid(time_indices=[0, 1, 2, 3])
        """
        # Validate variable
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")

        # Resolve time indices
        if time_indices is None:
            time_indices = list(range(self._obj.sizes["time"]))

        # Resolve frequency indices
        if freq_mhz_list is not None:
            freq_indices = [self.nearest_freq_idx(f) for f in freq_mhz_list]
        elif freq_indices is None:
            freq_indices = list(range(self._obj.sizes["frequency"]))

        # Calculate grid dimensions
        n_panels = len(time_indices) * len(freq_indices)
        if n_panels == 0:
            raise ValueError("No panels to plot. Check time_indices and freq_indices.")

        nrows = int(np.ceil(n_panels / ncols))

        # Create figure
        fig_width = panel_size[0] * ncols
        fig_height = panel_size[1] * nrows
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(fig_width, fig_height),
            squeeze=False,
        )

        # Collect all data for global scaling if robust=True
        all_data = []
        panel_data = []

        for ti in time_indices:
            for fi in freq_indices:
                da = self._obj[var].isel(
                    time=ti, frequency=fi, polarization=pol
                )
                data = da.values.copy()

                # Apply mask if requested
                if mask_radius is not None:
                    ny, nx = data.shape
                    cy, cx = ny // 2, nx // 2
                    yy, xx = np.ogrid[:ny, :nx]
                    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
                    data[dist > mask_radius] = np.nan

                panel_data.append((ti, fi, data, da))
                if robust and vmin is None and vmax is None:
                    finite = data[np.isfinite(data)]
                    if finite.size > 0:
                        all_data.append(finite)

        # Compute global vmin/vmax if robust
        if robust and vmin is None and vmax is None and all_data:
            all_finite = np.concatenate(all_data)
            vmin = float(np.percentile(all_finite, 2))
            vmax = float(np.percentile(all_finite, 98))

        # Plot each panel
        im = None
        for idx, (ti, fi, data, da) in enumerate(panel_data):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]

            # Get coordinate extents
            l_vals = da.coords["l"].values
            m_vals = da.coords["m"].values
            extent = [
                float(l_vals[0]), float(l_vals[-1]),
                float(m_vals[0]), float(m_vals[-1]),
            ]

            # Check if panel has data
            has_data = np.any(np.isfinite(data))

            if has_data:
                im = ax.imshow(
                    data.T,
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    extent=extent,
                    aspect="equal",
                    **kwargs,
                )
            else:
                ax.text(
                    0.5, 0.5, "No Data",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])

            # Build panel title
            time_val = self._obj.coords["time"].values[ti]
            freq_val = self._obj.coords["frequency"].values[fi] / 1e6
            try:
                time_str = f"{float(time_val):.6f}"
            except (TypeError, ValueError):
                time_str = str(time_val)

            ax.set_title(f"t={time_str}\nf={freq_val:.2f} MHz", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide unused panels
        for idx in range(n_panels, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].axis("off")

        # Add shared colorbar
        if share_colorbar and im is not None:
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label("Jy/beam")

        fig.suptitle(f"{var} Grid (pol={pol})", fontsize=12, y=1.02)

        return fig

    def plot_frequency_grid(
        self,
        time_idx: int = 0,
        freq_mhz_list: list[float] | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        ncols: int = 4,
        panel_size: tuple[float, float] = (3.0, 2.6),
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        mask_radius: int | None = None,
        **kwargs: Any,
    ) -> Figure:
        """Create a grid showing all frequencies at a fixed time.

        Convenience method for comparing across frequency channels.

        Parameters
        ----------
        time_idx : int, default 0
            Time index for all panels.
        freq_mhz_list : list of float, optional
            Specific frequencies to plot. If None, plots all frequencies.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        pol : int, default 0
            Polarization index.
        ncols : int, default 4
            Number of columns.
        panel_size : tuple, default (3.0, 2.6)
            Size of each panel.
        cmap : str, default 'inferno'
            Colormap.
        vmin, vmax : float, optional
            Color scale limits.
        robust : bool, default True
            Use percentile-based scaling.
        mask_radius : int, optional
            Circular mask radius.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> fig = ds.radport.plot_frequency_grid(time_idx=0)
        """
        return self.plot_grid(
            time_indices=[time_idx],
            freq_mhz_list=freq_mhz_list,
            var=var,
            pol=pol,
            ncols=ncols,
            panel_size=panel_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            robust=robust,
            mask_radius=mask_radius,
            **kwargs,
        )

    def plot_time_grid(
        self,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        time_indices: list[int] | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        ncols: int = 4,
        panel_size: tuple[float, float] = (3.0, 2.6),
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        mask_radius: int | None = None,
        **kwargs: Any,
    ) -> Figure:
        """Create a grid showing all times at a fixed frequency.

        Convenience method for comparing across time steps (time evolution).

        Parameters
        ----------
        freq_idx : int, optional
            Frequency index. Default is 0. Ignored if `freq_mhz` is provided.
        freq_mhz : float, optional
            Frequency in MHz (overrides freq_idx).
        time_indices : list of int, optional
            Specific time indices to plot. If None, plots all times.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        pol : int, default 0
            Polarization index.
        ncols : int, default 4
            Number of columns.
        panel_size : tuple, default (3.0, 2.6)
            Size of each panel.
        cmap : str, default 'inferno'
            Colormap.
        vmin, vmax : float, optional
            Color scale limits.
        robust : bool, default True
            Use percentile-based scaling.
        mask_radius : int, optional
            Circular mask radius.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> fig = ds.radport.plot_time_grid(freq_mhz=50.0)
        """
        # Resolve frequency
        if freq_mhz is not None:
            freq_indices = [self.nearest_freq_idx(freq_mhz)]
        elif freq_idx is not None:
            freq_indices = [freq_idx]
        else:
            freq_indices = [0]

        return self.plot_grid(
            time_indices=time_indices,
            freq_indices=freq_indices,
            var=var,
            pol=pol,
            ncols=ncols,
            panel_size=panel_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            robust=robust,
            mask_radius=mask_radius,
            **kwargs,
        )

    # =========================================================================
    # 1D Analysis Methods
    # =========================================================================

    def light_curve(
        self,
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        observatory: Any = None,
    ) -> xr.DataArray:
        """Extract a light curve (time series) at a specific spatial location.

        Returns intensity as a function of time at the pixel nearest to
        the specified coordinates and frequency.

        When celestial coordinates (ra, dec) are provided, the pixel is
        tracked across time steps as the source drifts due to Earth rotation.
        Time steps where the source is below the horizon are NaN-filled.

        Parameters
        ----------
        ra : float, optional
            Right Ascension in degrees (FK5/J2000). Requires ``dec``.
        dec : float, optional
            Declination in degrees (FK5/J2000). Requires ``ra``.
        l : float, optional
            Direction cosine l coordinate. Requires ``m``.
        m : float, optional
            Direction cosine m coordinate. Requires ``l``.
        freq_idx : int, optional
            Frequency index. Default is 0. Ignored if `freq_mhz` is provided.
        freq_mhz : float, optional
            Frequency in MHz (overrides freq_idx).
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to extract.
        pol : int, default 0
            Polarization index.
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location for RA/Dec tracking. Defaults to OVRO-LWA.

        Returns
        -------
        xr.DataArray
            1D array with dimension 'time' containing the light curve.

        Examples
        --------
        >>> lc = ds.radport.light_curve(l=0.0, m=0.0, freq_mhz=50.0)
        >>> lc = ds.radport.light_curve(ra=180.0, dec=45.0, freq_mhz=50.0)
        """
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")

        # Resolve frequency index
        if freq_mhz is not None:
            fi = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is not None:
            fi = freq_idx
        else:
            fi = 0

        result = self._resolve_coordinates(
            ra=ra,
            dec=dec,
            l=l,
            m=m,
            observatory=observatory,
            freq_idx=fi,
            pol=pol,
        )

        freq_hz = float(self._obj.coords["frequency"].values[fi])

        if isinstance(result, tuple) and len(result) == 2:
            # Fixed pixel path (l/m) — eagerly load small results
            l_idx, m_idx = result
            lc = _maybe_load(
                self._obj[var].isel(
                    frequency=fi, polarization=pol, l=l_idx, m=m_idx
                )
            )

            l_val = float(self._obj.coords["l"].values[l_idx])
            m_val = float(self._obj.coords["m"].values[m_idx])

            lc.attrs["variable"] = var
            lc.attrs["freq_idx"] = fi
            lc.attrs["freq_mhz"] = freq_hz / 1e6
            lc.attrs["pol"] = pol
            lc.attrs["l"] = l_val
            lc.attrs["m"] = m_val
            lc.attrs["l_idx"] = l_idx
            lc.attrs["m_idx"] = m_idx

            return lc

        # Per-time tracking path (ra/dec)
        l_indices, m_indices, visible = result
        data_var = self._obj[var].isel(frequency=fi, polarization=pol)

        n_times = self._obj.sizes["time"]
        out = np.full(n_times, np.nan)

        vis_mask = visible
        if np.any(vis_mask):
            vis_times = np.asarray(np.where(vis_mask)[0], dtype=int)
            vis_l = np.asarray(l_indices[vis_mask], dtype=int)
            vis_m = np.asarray(m_indices[vis_mask], dtype=int)
            plane = _vectorized_tracked_pixel_values(data_var, vis_times, vis_l, vis_m)
            for i, ti in enumerate(vis_times):
                out[int(ti)] = float(np.asarray(plane[i]).ravel()[0])

        time_coords = self._obj.coords["time"].values
        lc = xr.DataArray(
            out,
            dims=["time"],
            coords={"time": time_coords},
            attrs={
                "variable": var,
                "freq_idx": fi,
                "freq_mhz": freq_hz / 1e6,
                "pol": pol,
                "l": "tracked",
                "m": "tracked",
                "l_idx": "tracked",
                "m_idx": "tracked",
                "ra": ra,
                "dec": dec,
                "tracking": True,
            },
        )

        return lc

    def plot_light_curve(
        self,
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        figsize: tuple[float, float] = (10, 4),
        marker: str = "o",
        linestyle: str = "-",
        observatory: Any = None,
        **kwargs: Any,
    ) -> Figure:
        """Plot a light curve (time series) at a specific spatial location.

        Parameters
        ----------
        ra : float, optional
            Right Ascension in degrees (FK5/J2000). Requires ``dec``.
        dec : float, optional
            Declination in degrees (FK5/J2000). Requires ``ra``.
        l : float, optional
            Direction cosine l coordinate. Requires ``m``.
        m : float, optional
            Direction cosine m coordinate. Requires ``l``.
        freq_idx : int, optional
            Frequency index. Default is 0. Ignored if `freq_mhz` is provided.
        freq_mhz : float, optional
            Frequency in MHz (overrides freq_idx).
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        pol : int, default 0
            Polarization index.
        figsize : tuple, default (10, 4)
            Figure size in inches.
        marker : str, default 'o'
            Marker style for data points.
        linestyle : str, default '-'
            Line style connecting points.
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location for RA/Dec tracking. Defaults to OVRO-LWA.
        **kwargs : dict
            Additional arguments passed to plt.plot.

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> fig = ds.radport.plot_light_curve(l=0.0, m=0.0, freq_mhz=50.0)
        >>> fig = ds.radport.plot_light_curve(ra=180.0, dec=45.0, freq_mhz=50.0)
        """
        lc = self.light_curve(
            ra=ra, dec=dec, l=l, m=m,
            freq_idx=freq_idx, freq_mhz=freq_mhz, var=var, pol=pol,
            observatory=observatory,
        )

        fig, ax = plt.subplots(figsize=figsize)

        time_vals = lc.coords["time"].values
        ax.plot(time_vals, lc.values, marker=marker, linestyle=linestyle, **kwargs)

        ax.set_xlabel("Time (MJD)")
        ax.set_ylabel(f"{var} Intensity (Jy/beam)")

        freq_mhz_val = lc.attrs["freq_mhz"]
        if lc.attrs.get("tracking"):
            ra_val = lc.attrs["ra"]
            dec_val = lc.attrs["dec"]
            ax.set_title(
                f"{var} Light Curve (tracked) at RA={ra_val:.4f}°, "
                f"Dec={dec_val:.4f}°, f={freq_mhz_val:.2f} MHz, pol={pol}"
            )
        else:
            l_val = lc.attrs["l"]
            m_val = lc.attrs["m"]
            ax.set_title(
                f"{var} Light Curve at (l={l_val:.3f}, m={m_val:.3f}), "
                f"f={freq_mhz_val:.2f} MHz, pol={pol}"
            )

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return fig

    def spectrum(
        self,
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        time_idx: int | None = None,
        time_mjd: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
    ) -> xr.DataArray:
        """Extract a frequency spectrum at a specific spatial location and time.

        Returns intensity as a function of frequency at the pixel nearest to
        the specified coordinates and time.

        Parameters
        ----------
        ra : float, optional
            Right Ascension in degrees (FK5/J2000). Requires ``dec``.
        dec : float, optional
            Declination in degrees (FK5/J2000). Requires ``ra``.
        l : float, optional
            Direction cosine l coordinate. Requires ``m``.
        m : float, optional
            Direction cosine m coordinate. Requires ``l``.
        time_idx : int, optional
            Time index. Default is 0. Ignored if `time_mjd` is provided.
        time_mjd : float, optional
            Time in MJD (overrides time_idx).
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to extract.
        pol : int, default 0
            Polarization index.
        freq_idx : int, optional
            Frequency index for RA/Dec → pixel mapping (see :meth:`coords_to_pixel`).
            Defaults to channel 0 when neither this nor ``freq_mhz`` is set.
        freq_mhz : float, optional
            Select that channel by MHz; overrides ``freq_idx``.

        Returns
        -------
        xr.DataArray
            1D array with dimension 'frequency' containing the spectrum.

        Examples
        --------
        >>> spec = ds.radport.spectrum(l=0.0, m=0.0, time_idx=0)
        >>> spec = ds.radport.spectrum(ra=180.0, dec=45.0, time_idx=0)
        """
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")

        # Resolve time index
        if time_mjd is not None:
            ti = self.nearest_time_idx(time_mjd)
        elif time_idx is not None:
            ti = time_idx
        else:
            ti = 0

        # Resolve coordinates
        if ra is not None or dec is not None:
            if ra is None or dec is None:
                raise ValueError("Both ra and dec must be provided together.")
            l_idx, m_idx = self.coords_to_pixel(
                ra, dec, time_idx=ti, freq_idx=freq_idx, freq_mhz=freq_mhz, pol=pol
            )
        elif l is not None or m is not None:
            if l is None or m is None:
                raise ValueError("Both l and m must be provided together.")
            l_idx, m_idx = self.nearest_lm_idx(l, m)
        else:
            raise ValueError("Must provide either (ra, dec) or (l, m) coordinates.")

        # Extract spectrum
        spec = self._obj[var].isel(
            time=ti,
            polarization=pol,
            l=l_idx,
            m=m_idx,
        )

        # Add metadata
        time_val = float(self._obj.coords["time"].values[ti])
        l_val = float(self._obj.coords["l"].values[l_idx])
        m_val = float(self._obj.coords["m"].values[m_idx])

        spec.attrs["variable"] = var
        spec.attrs["time_idx"] = ti
        spec.attrs["time_mjd"] = time_val
        spec.attrs["pol"] = pol
        spec.attrs["l"] = l_val
        spec.attrs["m"] = m_val
        spec.attrs["l_idx"] = l_idx
        spec.attrs["m_idx"] = m_idx

        return spec

    def plot_spectrum(
        self,
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        time_idx: int | None = None,
        time_mjd: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        figsize: tuple[float, float] = (10, 4),
        marker: str = "o",
        linestyle: str = "-",
        freq_unit: Literal["Hz", "MHz"] = "MHz",
        **kwargs: Any,
    ) -> Figure:
        """Plot a frequency spectrum at a specific spatial location and time.

        Parameters
        ----------
        ra : float, optional
            Right Ascension in degrees (FK5/J2000). Requires ``dec``.
        dec : float, optional
            Declination in degrees (FK5/J2000). Requires ``ra``.
        l : float, optional
            Direction cosine l coordinate. Requires ``m``.
        m : float, optional
            Direction cosine m coordinate. Requires ``l``.
        time_idx : int, optional
            Time index. Default is 0. Ignored if `time_mjd` is provided.
        time_mjd : float, optional
            Time in MJD (overrides time_idx).
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        pol : int, default 0
            Polarization index.
        freq_idx : int, optional
            Passed to :meth:`spectrum` for RA/Dec pixel selection.
        freq_mhz : float, optional
            Passed to :meth:`spectrum`; overrides ``freq_idx``.
        figsize : tuple, default (10, 4)
            Figure size in inches.
        marker : str, default 'o'
            Marker style for data points.
        linestyle : str, default '-'
            Line style connecting points.
        freq_unit : {'Hz', 'MHz'}, default 'MHz'
            Unit for frequency axis.
        **kwargs : dict
            Additional arguments passed to plt.plot.

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> fig = ds.radport.plot_spectrum(l=0.0, m=0.0, time_idx=0)
        >>> fig = ds.radport.plot_spectrum(ra=180.0, dec=45.0, time_idx=0)
        """
        spec = self.spectrum(
            ra=ra,
            dec=dec,
            l=l,
            m=m,
            time_idx=time_idx,
            time_mjd=time_mjd,
            var=var,
            pol=pol,
            freq_idx=freq_idx,
            freq_mhz=freq_mhz,
        )

        fig, ax = plt.subplots(figsize=figsize)

        freq_vals = spec.coords["frequency"].values
        if freq_unit == "MHz":
            freq_vals = freq_vals / 1e6
            xlabel = "Frequency (MHz)"
        else:
            xlabel = "Frequency (Hz)"

        ax.plot(freq_vals, spec.values, marker=marker, linestyle=linestyle, **kwargs)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"{var} Intensity (Jy/beam)")

        time_mjd_val = spec.attrs["time_mjd"]
        l_val = spec.attrs["l"]
        m_val = spec.attrs["m"]
        ax.set_title(
            f"{var} Spectrum at (l={l_val:.3f}, m={m_val:.3f}), "
            f"t={time_mjd_val:.6f} MJD, pol={pol}"
        )

        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return fig

    def time_average(
        self,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        time_indices: list[int] | None = None,
    ) -> xr.DataArray:
        """Compute the time-averaged image.

        Averages the data across the time dimension, returning a 3D array
        with dimensions (frequency, l, m).

        Parameters
        ----------
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to average.
        pol : int, default 0
            Polarization index.
        time_indices : list of int, optional
            Specific time indices to include in the average.
            If None, averages over all times.

        Returns
        -------
        xr.DataArray
            3D array with dimensions (frequency, l, m).

        Examples
        --------
        >>> avg = ds.radport.time_average()
        >>> avg.isel(frequency=0).plot()  # Plot mean image at first frequency
        """
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")

        da = self._obj[var].isel(polarization=pol)

        if time_indices is not None:
            da = da.isel(time=time_indices)

        avg = da.mean(dim="time")

        avg.attrs["variable"] = var
        avg.attrs["pol"] = pol
        avg.attrs["operation"] = "time_average"
        if time_indices is not None:
            avg.attrs["time_indices"] = time_indices
        else:
            avg.attrs["n_times"] = self._obj.sizes["time"]

        return avg

    def frequency_average(
        self,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        freq_indices: list[int] | None = None,
        freq_min_mhz: float | None = None,
        freq_max_mhz: float | None = None,
    ) -> xr.DataArray:
        """Compute the frequency-averaged image.

        Averages the data across the frequency dimension, returning a 3D array
        with dimensions (time, l, m).

        Parameters
        ----------
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to average.
        pol : int, default 0
            Polarization index.
        freq_indices : list of int, optional
            Specific frequency indices to include in the average.
            If None (and freq_min/max not set), averages over all frequencies.
        freq_min_mhz : float, optional
            Minimum frequency in MHz for averaging band.
        freq_max_mhz : float, optional
            Maximum frequency in MHz for averaging band.

        Returns
        -------
        xr.DataArray
            3D array with dimensions (time, l, m).

        Examples
        --------
        >>> avg = ds.radport.frequency_average()
        >>> avg.isel(time=0).plot()  # Plot mean image at first time

        >>> # Average only 45-55 MHz band
        >>> band_avg = ds.radport.frequency_average(freq_min_mhz=45.0, freq_max_mhz=55.0)
        """
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")

        da = self._obj[var].isel(polarization=pol)

        # Handle frequency selection
        if freq_min_mhz is not None or freq_max_mhz is not None:
            freq_hz = self._obj.coords["frequency"].values
            freq_mhz = freq_hz / 1e6

            if freq_min_mhz is None:
                freq_min_mhz = freq_mhz.min()
            if freq_max_mhz is None:
                freq_max_mhz = freq_mhz.max()

            mask = (freq_mhz >= freq_min_mhz) & (freq_mhz <= freq_max_mhz)
            freq_indices = list(np.where(mask)[0])

            if len(freq_indices) == 0:
                raise ValueError(
                    f"No frequencies in range [{freq_min_mhz}, {freq_max_mhz}] MHz. "
                    f"Available range: [{freq_mhz.min():.2f}, {freq_mhz.max():.2f}] MHz"
                )

        if freq_indices is not None:
            da = da.isel(frequency=freq_indices)

        avg = da.mean(dim="frequency")

        avg.attrs["variable"] = var
        avg.attrs["pol"] = pol
        avg.attrs["operation"] = "frequency_average"
        if freq_indices is not None:
            avg.attrs["freq_indices"] = freq_indices
        else:
            avg.attrs["n_frequencies"] = self._obj.sizes["frequency"]
        if freq_min_mhz is not None:
            avg.attrs["freq_min_mhz"] = freq_min_mhz
        if freq_max_mhz is not None:
            avg.attrs["freq_max_mhz"] = freq_max_mhz

        return avg

    def plot_time_average(
        self,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        time_indices: list[int] | None = None,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        mask_radius: int | None = None,
        figsize: tuple[float, float] = (8, 6),
        add_colorbar: bool = True,
        **kwargs: Any,
    ) -> Figure:
        """Plot the time-averaged image at a specific frequency.

        Parameters
        ----------
        freq_idx : int, optional
            Frequency index. Default is 0. Ignored if `freq_mhz` is provided.
        freq_mhz : float, optional
            Frequency in MHz (overrides freq_idx).
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        pol : int, default 0
            Polarization index.
        time_indices : list of int, optional
            Specific time indices to include in the average.
        cmap : str, default 'inferno'
            Colormap.
        vmin, vmax : float, optional
            Color scale limits.
        robust : bool, default True
            Use 2nd/98th percentile for scaling.
        mask_radius : int, optional
            Circular mask radius in pixels.
        figsize : tuple, default (8, 6)
            Figure size in inches.
        add_colorbar : bool, default True
            Whether to add colorbar.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> fig = ds.radport.plot_time_average(freq_mhz=50.0)
        """
        avg = self.time_average(var=var, pol=pol, time_indices=time_indices)

        # Resolve frequency index
        if freq_mhz is not None:
            fi = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is not None:
            fi = freq_idx
        else:
            fi = 0

        # Select frequency slice
        data = avg.isel(frequency=fi).values.copy()

        # Apply mask if requested
        if mask_radius is not None:
            ny, nx = data.shape
            cy, cx = ny // 2, nx // 2
            yy, xx = np.ogrid[:ny, :nx]
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            data[dist > mask_radius] = np.nan

        # Compute vmin/vmax if robust
        if robust and vmin is None and vmax is None:
            finite = data[np.isfinite(data)]
            if finite.size > 0:
                vmin = float(np.percentile(finite, 2))
                vmax = float(np.percentile(finite, 98))

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        l_vals = avg.coords["l"].values
        m_vals = avg.coords["m"].values
        extent = [
            float(l_vals[0]), float(l_vals[-1]),
            float(m_vals[0]), float(m_vals[-1]),
        ]

        # Transpose (l, m) to put l on x-axis and m on y-axis
        im = ax.imshow(
            data.T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="equal",
            **kwargs,
        )

        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Jy/beam")

        freq_hz = float(self._obj.coords["frequency"].values[fi])
        n_times = len(time_indices) if time_indices else self._obj.sizes["time"]
        ax.set_xlabel("l (direction cosine)")
        ax.set_ylabel("m (direction cosine)")
        ax.set_title(
            f"{var} Time Average ({n_times} frames) at f={freq_hz/1e6:.2f} MHz, pol={pol}"
        )

        fig.tight_layout()
        return fig

    def plot_frequency_average(
        self,
        time_idx: int | None = None,
        time_mjd: float | None = None,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        freq_indices: list[int] | None = None,
        freq_min_mhz: float | None = None,
        freq_max_mhz: float | None = None,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        mask_radius: int | None = None,
        figsize: tuple[float, float] = (8, 6),
        add_colorbar: bool = True,
        **kwargs: Any,
    ) -> Figure:
        """Plot the frequency-averaged image at a specific time.

        Parameters
        ----------
        time_idx : int, optional
            Time index. Default is 0. Ignored if `time_mjd` is provided.
        time_mjd : float, optional
            Time in MJD (overrides time_idx).
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        pol : int, default 0
            Polarization index.
        freq_indices : list of int, optional
            Specific frequency indices to include in the average.
        freq_min_mhz : float, optional
            Minimum frequency in MHz for averaging band.
        freq_max_mhz : float, optional
            Maximum frequency in MHz for averaging band.
        cmap : str, default 'inferno'
            Colormap.
        vmin, vmax : float, optional
            Color scale limits.
        robust : bool, default True
            Use 2nd/98th percentile for scaling.
        mask_radius : int, optional
            Circular mask radius in pixels.
        figsize : tuple, default (8, 6)
            Figure size in inches.
        add_colorbar : bool, default True
            Whether to add colorbar.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> fig = ds.radport.plot_frequency_average(time_idx=0)

        >>> # Average 45-55 MHz band
        >>> fig = ds.radport.plot_frequency_average(
        ...     time_idx=0, freq_min_mhz=45.0, freq_max_mhz=55.0
        ... )
        """
        avg = self.frequency_average(
            var=var,
            pol=pol,
            freq_indices=freq_indices,
            freq_min_mhz=freq_min_mhz,
            freq_max_mhz=freq_max_mhz,
        )

        # Resolve time index
        if time_mjd is not None:
            ti = self.nearest_time_idx(time_mjd)
        elif time_idx is not None:
            ti = time_idx
        else:
            ti = 0

        # Select time slice
        data = avg.isel(time=ti).values.copy()

        # Apply mask if requested
        if mask_radius is not None:
            ny, nx = data.shape
            cy, cx = ny // 2, nx // 2
            yy, xx = np.ogrid[:ny, :nx]
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            data[dist > mask_radius] = np.nan

        # Compute vmin/vmax if robust
        if robust and vmin is None and vmax is None:
            finite = data[np.isfinite(data)]
            if finite.size > 0:
                vmin = float(np.percentile(finite, 2))
                vmax = float(np.percentile(finite, 98))

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        l_vals = avg.coords["l"].values
        m_vals = avg.coords["m"].values
        extent = [
            float(l_vals[0]), float(l_vals[-1]),
            float(m_vals[0]), float(m_vals[-1]),
        ]

        # Transpose (l, m) to put l on x-axis and m on y-axis
        im = ax.imshow(
            data.T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="equal",
            **kwargs,
        )

        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Jy/beam")

        time_val = float(self._obj.coords["time"].values[ti])

        # Build title with frequency info
        if freq_min_mhz is not None and freq_max_mhz is not None:
            freq_info = f"{freq_min_mhz:.1f}-{freq_max_mhz:.1f} MHz"
        elif freq_indices is not None:
            freq_info = f"{len(freq_indices)} channels"
        else:
            freq_info = f"{self._obj.sizes['frequency']} channels"

        ax.set_xlabel("l (direction cosine)")
        ax.set_ylabel("m (direction cosine)")
        ax.set_title(
            f"{var} Frequency Average ({freq_info}) at t={time_val:.6f} MJD, pol={pol}"
        )

        fig.tight_layout()
        return fig

    # =========================================================================
    # WCS & Coordinate Methods
    # =========================================================================

    def _get_wcs(
        self,
        var: Literal["SKY", "BEAM"] = "SKY",
        *,
        time_idx: int = 0,
        freq_idx: int = 0,
    ):
        """Get WCS object from the dataset.

        Parameters
        ----------
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to get WCS from (checks attrs first).
        time_idx : int, default 0
            Time index when ``fits_header_str`` is stored per time step (common
            after incremental Zarr writes).
        freq_idx : int, default 0
            Frequency index for per-subband celestial WCS when stored in
            ``fits_header_str(time, frequency, polarization)``.

        Returns
        -------
        astropy.wcs.WCS
            The 2D celestial (RA/Dec) WCS, matching astrowidget ``get_wcs``.

        Raises
        ------
        ImportError
            If astropy is not installed.
        ValueError
            If no WCS header is found in the dataset, or the header has no
            celestial axes.
        """
        cache_key = (int(time_idx), int(freq_idx))
        cached = self._wcs_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            from astropy.io.fits import Header
            from astropy.wcs import WCS
        except ImportError as e:
            raise ImportError(
                "astropy is required for WCS functionality. "
                "Install with: pip install astropy"
            ) from e

        hdr_str = _read_wcs_header_str(
            self._obj, var=var, time_idx=time_idx, freq_idx=freq_idx
        )
        if not hdr_str:
            raise ValueError(
                "No WCS header found in dataset. Expected 'fits_header_str' "
                "(or legacy 'wcs_header_str') with a celestial subset."
            )

        wcs = WCS(Header.fromstring(hdr_str, sep="\n"))
        if not wcs.has_celestial:
            raise ValueError("WCS header has no celestial axes (RA/Dec)")
        celestial = wcs.celestial
        self._wcs_cache[cache_key] = celestial
        return celestial

    @property
    def has_wcs(self) -> bool:
        """Check if WCS coordinate information is available.

        Returns
        -------
        bool
            True if WCS header is available in the dataset.

        Example
        -------
        >>> if ds.radport.has_wcs:
        ...     fig = ds.radport.plot_wcs()
        """
        try:
            self._get_wcs()
            return True
        except (ImportError, ValueError):
            return False

    def pixel_to_coords(
        self,
        l_idx: int,
        m_idx: int,
        *,
        time_idx: int | None = None,
        time_mjd: float | None = None,
        observatory: Any = None,
    ) -> tuple[float, float]:
        """Convert pixel indices to celestial coordinates (RA, Dec).

        When the dataset has per-time ``wcs_header_str`` (incremental Zarr), uses
        the same slice WCS as :meth:`coords_to_pixel` via ``pixel_to_world``.
        Otherwise inverts the analytical SIN + LST model to match legacy datasets
        without per-time headers.

        You must pass **exactly one** of ``time_idx`` or ``time_mjd``.

        Parameters
        ----------
        l_idx : int
            Index along the l dimension.
        m_idx : int
            Index along the m dimension.
        time_idx : int, optional
            Index into the dataset ``time`` coordinate. Mutually exclusive with
            ``time_mjd``.
        time_mjd : float, optional
            MJD timestamp; the nearest ``time`` index is used. Mutually
            exclusive with ``time_idx``.
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location for sidereal time. Defaults to OVRO-LWA.

        Returns
        -------
        tuple of float
            ``(ra, dec)`` in degrees. RA is in ``[0, 360)``.

        Raises
        ------
        ValueError
            If neither or both of ``time_idx`` and ``time_mjd`` are given, the
            dataset has no ``time`` coordinate, WCS metadata is unavailable,
            indices are out of bounds, the inverse SIN fit fails to converge, or
            the direction is below the horizon at the requested time.

        Example
        -------
        >>> ra, dec = ds.radport.pixel_to_coords(100, 100, time_idx=0)
        >>> ra, dec = ds.radport.pixel_to_coords(100, 100, time_mjd=60000.0)
        """
        n_l = self._obj.sizes["l"]
        n_m = self._obj.sizes["m"]
        if not (0 <= l_idx < n_l):
            raise ValueError(f"l_idx={l_idx} out of bounds [0, {n_l})")
        if not (0 <= m_idx < n_m):
            raise ValueError(f"m_idx={m_idx} out of bounds [0, {n_m})")

        if time_idx is not None and time_mjd is not None:
            raise ValueError("Provide exactly one of time_idx or time_mjd, not both.")
        if time_idx is None and time_mjd is None:
            raise ValueError(
                "pixel_to_coords requires exactly one of time_idx or time_mjd."
            )

        if "time" not in self._obj.coords:
            raise ValueError(
                "pixel_to_coords requires a dataset ``time`` coordinate "
                "(pass time_idx or time_mjd)."
            )

        from astropy.coordinates import EarthLocation

        if time_mjd is not None:
            ti = self.nearest_time_idx(time_mjd)
        else:
            ti = int(time_idx)

        n_time = self._obj.sizes["time"]
        if not (0 <= ti < n_time):
            raise ValueError(f"time index {ti} out of bounds [0, {n_time})")

        if self._use_persisted_wcs_for_pixel_mapping():
            from astropy import units as u

            wcs_t = self._get_wcs(time_idx=ti)
            world = wcs_t.pixel_to_world(l_idx, m_idx)
            ra_wrapped = float(world.ra.wrap_at(360 * u.deg).deg)
            dec_sol = float(world.dec.deg)
            return ra_wrapped, dec_sol

        if observatory is None:
            from astropy import units as u

            observatory = EarthLocation(
                lat=37.2339 * u.deg, lon=-118.2817 * u.deg, height=1222 * u.m
            )

        l_val = float(self._obj.coords["l"].values[l_idx])
        m_val = float(self._obj.coords["m"].values[m_idx])
        lst_deg = self._lst_deg_for_time_index(ti, observatory=observatory)
        lat_deg = float(observatory.lat.deg)
        lat_rad = np.deg2rad(lat_deg)

        wcs = self._get_wcs(time_idx=ti)
        coord = wcs.pixel_to_world(l_idx, m_idx)
        ra0 = float(coord.ra.wrap_at("360d").deg)
        dec0 = float(coord.dec.deg)
        if not (np.isfinite(ra0) and np.isfinite(dec0)):
            ra0 = float(wcs.wcs.crval[0])
            dec0 = float(wcs.wcs.crval[1])

        def sin_altitude(ra_deg: float, dec_deg: float) -> float:
            dec_rad = np.deg2rad(dec_deg)
            ha_rad = np.deg2rad(lst_deg - ra_deg)
            return float(
                np.sin(dec_rad) * np.sin(lat_rad)
                + np.cos(dec_rad) * np.cos(lat_rad) * np.cos(ha_rad)
            )

        def residual(vec: np.ndarray) -> np.ndarray:
            ra_deg, dec_deg = float(vec[0]), float(vec[1])
            ha_rad = np.deg2rad(lst_deg - ra_deg)
            dec_rad = np.deg2rad(dec_deg)
            lf = -np.cos(dec_rad) * np.sin(ha_rad)
            mf = (
                np.sin(dec_rad) * np.cos(lat_rad)
                - np.cos(dec_rad) * np.sin(lat_rad) * np.cos(ha_rad)
            )
            sa = sin_altitude(ra_deg, dec_deg)
            horizon_penalty = 0.0 if sa > 0.01 else 5.0 * (0.01 - sa)
            return np.array(
                [lf - l_val, mf - m_val, horizon_penalty],
                dtype=float,
            )

        # Unwrap RA near the WCS seed and search locally so the optimiser does
        # not lock onto a below-horizon branch of the inverse.
        ra_unwrap = float(ra0)
        if ra_unwrap > 180.0:
            ra_unwrap -= 360.0
        lo_ra, hi_ra = ra_unwrap - 120.0, ra_unwrap + 120.0
        lo_dec, hi_dec = max(-90.0, dec0 - 75.0), min(90.0, dec0 + 75.0)

        res = least_squares(
            residual,
            x0=np.array([ra_unwrap, dec0], dtype=float),
            bounds=([lo_ra, lo_dec], [hi_ra, hi_dec]),
            method="trf",
            ftol=1e-12,
            xtol=1e-10,
            max_nfev=300,
        )
        lm_res = float(np.linalg.norm(res.fun[:2]))
        if not res.success or lm_res > 1e-5:
            raise ValueError(
                f"Could not invert (l, m)=({l_val:.6g}, {m_val:.6g}) to RA/Dec at "
                f"time_idx={ti} (lm residual norm {lm_res:.3g}; message: {res.message})."
            )

        ra_wrapped = float(res.x[0]) % 360.0
        if ra_wrapped < 0:
            ra_wrapped += 360.0
        dec_sol = float(res.x[1])

        if sin_altitude(ra_wrapped, dec_sol) <= 0:
            raise ValueError(
                f"Sky at pixel (l_idx={l_idx}, m_idx={m_idx}) is below the horizon "
                f"at time index {ti}."
            )

        return ra_wrapped, dec_sol

    def coords_to_pixel(
        self,
        ra: float,
        dec: float,
        *,
        time_idx: int | None = None,
        time_mjd: float | None = None,
        observatory: Any = None,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        pol: int = 0,
    ) -> tuple[int, int]:
        """Convert celestial coordinates (RA, Dec) to pixel indices.

        A fixed (RA, Dec) maps to different pixel positions at different
        times due to Earth rotation. The epoch must be given via ``time_idx``
        or ``time_mjd``. When the dataset stores per-pixel ``right_ascension``
        and ``declination`` coordinates, this method minimizes angular distance
        on that grid (after optional ``frequency`` / ``polarization`` slicing);
        otherwise it uses the time-aware SIN + mean sidereal time model.

        Parameters
        ----------
        ra : float
            Right Ascension in degrees (FK5/J2000).
        dec : float
            Declination in degrees (FK5/J2000).
        time_idx : int, optional
            Index into the dataset ``time`` coordinate. Mutually exclusive with
            ``time_mjd``.
        time_mjd : float, optional
            MJD timestamp; the nearest ``time`` index is used. Mutually
            exclusive with ``time_idx``.
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location. Defaults to OVRO-LWA.
        freq_idx : int, optional
            Frequency index for slicing channelized RA/Dec coordinate arrays
            when present. Defaults to 0 if neither ``freq_idx`` nor ``freq_mhz``
            is given. Ignored when ``freq_mhz`` is set.
        freq_mhz : float, optional
            Select ``freq_idx`` via :meth:`nearest_freq_idx`. Overrides
            ``freq_idx`` when provided.
        pol : int, default 0
            Polarization index for slicing RA/Dec coords that include a
            ``polarization`` dimension.

        Returns
        -------
        tuple of int
            (l_idx, m_idx) pixel indices (rounded to nearest integer).

        Raises
        ------
        ValueError
            If neither or both of ``time_idx`` and ``time_mjd`` are given, the
            dataset has no ``time`` coordinate, coordinates are outside the
            image, or the source is below the horizon at the given time.

        Example
        -------
        >>> l_idx, m_idx = ds.radport.coords_to_pixel(180.0, 45.0, time_idx=10)
        """
        if time_idx is not None and time_mjd is not None:
            raise ValueError("Provide exactly one of time_idx or time_mjd, not both.")
        if time_idx is None and time_mjd is None:
            raise ValueError("coords_to_pixel requires exactly one of time_idx or time_mjd.")
        if "time" not in self._obj.coords:
            raise ValueError(
                "coords_to_pixel requires a dataset ``time`` coordinate "
                "(pass time_idx or time_mjd)."
            )

        ti = self.nearest_time_idx(time_mjd) if time_mjd is not None else int(time_idx)
        if freq_mhz is not None:
            fi = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is not None:
            fi = int(freq_idx)
        else:
            fi = 0
        return self._compute_pixel_at_time(
            ra, dec, ti, observatory=observatory, freq_idx=fi, pol=int(pol)
        )

    def plot_wcs(
        self,
        var: Literal["SKY", "BEAM"] = "SKY",
        time_idx: int = 0,
        freq_idx: int = 0,
        freq_mhz: float | None = None,
        pol: int = 0,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        mask_radius: int | None = None,
        figsize: tuple[float, float] = (10, 10),
        add_colorbar: bool = True,
        grid_color: str = "white",
        grid_alpha: float = 0.6,
        grid_linestyle: str = ":",
        label_color: str = "white",
        facecolor: str = "black",
        **kwargs: Any,
    ) -> Figure:
        """Plot with WCS projection and celestial coordinate grid.

        Creates a publication-quality plot with RA/Dec coordinate axes
        and optional grid overlay.

        Parameters
        ----------
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        time_idx : int, default 0
            Time index.
        freq_idx : int, default 0
            Frequency index. Ignored if `freq_mhz` is provided.
        freq_mhz : float, optional
            Frequency in MHz (overrides freq_idx).
        pol : int, default 0
            Polarization index.
        cmap : str, default 'inferno'
            Colormap name.
        vmin, vmax : float, optional
            Color scale limits.
        robust : bool, default True
            Use 2nd/98th percentile for scaling.
        mask_radius : int, optional
            Circular mask radius in pixels.
        figsize : tuple, default (10, 10)
            Figure size in inches.
        add_colorbar : bool, default True
            Whether to add colorbar.
        grid_color : str, default 'white'
            Color of coordinate grid lines.
        grid_alpha : float, default 0.6
            Transparency of grid lines.
        grid_linestyle : str, default ':'
            Line style for grid.
        label_color : str, default 'white'
            Color for axis labels and ticks.
        facecolor : str, default 'black'
            Background color for the plot.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            If WCS is not available in the dataset.

        Example
        -------
        >>> fig = ds.radport.plot_wcs(freq_mhz=50.0, mask_radius=1800)
        """
        try:
            from astropy import units as u
        except ImportError as e:
            raise ImportError(
                "astropy is required for WCS plotting."
            ) from e

        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")

        wcs = self._get_wcs(var)

        # Resolve frequency index
        if freq_mhz is not None:
            fi = self.nearest_freq_idx(freq_mhz)
        else:
            fi = freq_idx

        # Extract data
        da = self._obj[var].isel(
            time=time_idx, frequency=fi, polarization=pol
        )

        # imshow uses array rows/cols; put declination-like m on the first axis
        if set(da.dims) == {"m", "l"}:
            da = da.transpose("m", "l")

        data = da.values.astype(float).copy()

        # Apply mask if requested
        if mask_radius is not None:
            ny, nx = data.shape
            cy, cx = ny // 2, nx // 2
            yy, xx = np.ogrid[:ny, :nx]
            dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            data[dist > mask_radius] = np.nan

        # Compute vmin/vmax if robust
        if robust and vmin is None and vmax is None:
            finite = data[np.isfinite(data)]
            if finite.size > 0:
                vmin = float(np.percentile(finite, 2))
                vmax = float(np.percentile(finite, 98))

        # Set up colormap with bad values as black
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(facecolor, 1.0)

        # Create figure with WCS projection
        fig = plt.figure(figsize=figsize, facecolor=facecolor)
        ax = fig.add_subplot(111, projection=wcs, facecolor=facecolor)

        # Plot image
        im = ax.imshow(
            data,
            origin="lower",
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

        # Configure axes
        ax.set_xlabel("RA", color=label_color, fontsize=12)
        ax.set_ylabel("Dec", color=label_color, fontsize=12)

        # Check if RA needs to be inverted (increases to left in sky)
        try:
            cdelt1 = float(wcs.wcs.cdelt[0])
            if np.isfinite(cdelt1) and cdelt1 > 0:
                ax.invert_xaxis()
        except (AttributeError, IndexError):
            pass

        # Add coordinate grid
        overlay = ax.get_coords_overlay("fk5")
        overlay.grid(color=grid_color, ls=grid_linestyle, lw=1.0, alpha=grid_alpha)

        # Configure tick labels
        for coord in overlay:
            coord.set_ticklabel_visible(True)
            coord.set_ticklabel(color=label_color, size=10)
            coord.tick_params(width=1, color=label_color)

        # Add colorbar
        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Jy/beam", color=label_color, fontsize=11)
            cbar.ax.tick_params(color=label_color, labelcolor=label_color)
            cbar.outline.set_edgecolor(label_color)

        # Add title
        freq_hz = float(self._obj.coords["frequency"].values[fi])
        time_val = self._obj.coords["time"].values[time_idx]
        try:
            time_str = f"{float(time_val):.6f}"
        except (TypeError, ValueError):
            time_str = str(time_val)

        ax.set_title(
            f"{var} at t={time_str} MJD, f={freq_hz/1e6:.2f} MHz, pol={pol}",
            color=label_color,
            fontsize=12,
            pad=10,
        )

        return fig

    # =========================================================================
    # Phase F: Animation & Export Methods
    # =========================================================================

    def animate_time(
        self,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        var: str = "SKY",
        pol: int = 0,
        output_file: str | None = None,
        fps: int = 5,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        mask_radius: int | None = None,
        figsize: tuple[float, float] = (8, 6),
        dpi: int = 100,
        **kwargs: Any,
    ) -> Any:
        """Create an animation showing time evolution at a fixed frequency.

        Parameters
        ----------
        freq_idx : int, optional
            Frequency index to animate. Defaults to 0 if neither freq_idx
            nor freq_mhz is provided.
        freq_mhz : float, optional
            Select frequency by value in MHz. Overrides freq_idx if provided.
        var : str, default "SKY"
            Data variable to animate ("SKY" or "BEAM").
        pol : int, default 0
            Polarization index.
        output_file : str, optional
            Path to save the animation. Supported formats: .mp4, .gif.
            If None, returns the animation object for display in notebooks.
        fps : int, default 5
            Frames per second for the animation.
        cmap : str, default "inferno"
            Matplotlib colormap name.
        vmin : float, optional
            Minimum value for color scaling. If None and robust=True,
            uses 2nd percentile across all frames.
        vmax : float, optional
            Maximum value for color scaling. If None and robust=True,
            uses 98th percentile across all frames.
        robust : bool, default True
            Use percentile-based color scaling across all frames.
        mask_radius : int, optional
            Apply circular mask with this radius in pixels.
        figsize : tuple, default (8, 6)
            Figure size in inches.
        dpi : int, default 100
            Resolution for saved animation.
        **kwargs
            Additional arguments passed to FuncAnimation.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            Animation object. Can be displayed in notebooks with HTML(anim.to_jshtml())
            or saved to file.

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> # Create animation and save to file
        >>> anim = ds.radport.animate_time(freq_mhz=50.0, output_file="time_evolution.mp4")
        >>>
        >>> # Display in Jupyter notebook
        >>> from IPython.display import HTML
        >>> anim = ds.radport.animate_time(freq_mhz=50.0)
        >>> HTML(anim.to_jshtml())
        """
        from matplotlib.animation import FuncAnimation

        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Resolve frequency index
        if freq_mhz is not None:
            fi = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is not None:
            fi = freq_idx
        else:
            fi = 0

        # Get data for all time steps
        data = self._obj[var].isel(frequency=fi, polarization=pol)
        n_times = len(self._obj.coords["time"])

        # Compute global color scale from all frames
        if vmin is None or vmax is None:
            all_values = data.values.ravel()
            finite_values = all_values[np.isfinite(all_values)]
            if len(finite_values) > 0:
                if robust:
                    computed_vmin = np.percentile(finite_values, 2)
                    computed_vmax = np.percentile(finite_values, 98)
                else:
                    computed_vmin = np.nanmin(finite_values)
                    computed_vmax = np.nanmax(finite_values)
            else:
                computed_vmin, computed_vmax = 0, 1

            if vmin is None:
                vmin = computed_vmin
            if vmax is None:
                vmax = computed_vmax

        # Create mask if requested
        mask = None
        if mask_radius is not None:
            nl = len(self._obj.coords["l"])
            nm = len(self._obj.coords["m"])
            center_l, center_m = nl // 2, nm // 2
            l_idx, m_idx = np.ogrid[:nl, :nm]
            dist = np.sqrt((l_idx - center_l) ** 2 + (m_idx - center_m) ** 2)
            mask = dist > mask_radius

        # Create figure and initial plot
        fig, ax = plt.subplots(figsize=figsize)

        frame_data = data.isel(time=0).values.copy()
        if mask is not None:
            frame_data[mask] = np.nan

        im = ax.imshow(
            frame_data.T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Jy/beam", fontsize=11)

        # Labels
        ax.set_xlabel("l index", fontsize=11)
        ax.set_ylabel("m index", fontsize=11)

        freq_hz = float(self._obj.coords["frequency"].values[fi])

        def update(frame: int) -> tuple:
            """Update function for animation."""
            frame_data = data.isel(time=frame).values.copy()
            if mask is not None:
                frame_data[mask] = np.nan
            im.set_array(frame_data.T)

            time_val = self._obj.coords["time"].values[frame]
            try:
                time_str = f"{float(time_val):.6f}"
            except (TypeError, ValueError):
                time_str = str(time_val)

            ax.set_title(
                f"{var} at f={freq_hz/1e6:.2f} MHz, pol={pol}\n"
                f"Time: {time_str} MJD (frame {frame + 1}/{n_times})",
                fontsize=11,
            )
            return (im,)

        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            frames=n_times,
            interval=1000 // fps,
            blit=True,
            **kwargs,
        )

        # Save if output file specified
        if output_file is not None:
            if output_file.endswith(".gif"):
                anim.save(output_file, writer="pillow", fps=fps, dpi=dpi)
            else:
                anim.save(output_file, writer="ffmpeg", fps=fps, dpi=dpi)
            plt.close(fig)

        return anim

    def animate_frequency(
        self,
        time_idx: int | None = None,
        time_mjd: float | None = None,
        var: str = "SKY",
        pol: int = 0,
        output_file: str | None = None,
        fps: int = 5,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        mask_radius: int | None = None,
        figsize: tuple[float, float] = (8, 6),
        dpi: int = 100,
        **kwargs: Any,
    ) -> Any:
        """Create an animation showing frequency sweep at a fixed time.

        Parameters
        ----------
        time_idx : int, optional
            Time index to animate. Defaults to 0 if neither time_idx
            nor time_mjd is provided.
        time_mjd : float, optional
            Select time by MJD value. Overrides time_idx if provided.
        var : str, default "SKY"
            Data variable to animate ("SKY" or "BEAM").
        pol : int, default 0
            Polarization index.
        output_file : str, optional
            Path to save the animation. Supported formats: .mp4, .gif.
            If None, returns the animation object for display in notebooks.
        fps : int, default 5
            Frames per second for the animation.
        cmap : str, default "inferno"
            Matplotlib colormap name.
        vmin : float, optional
            Minimum value for color scaling. If None and robust=True,
            uses 2nd percentile across all frames.
        vmax : float, optional
            Maximum value for color scaling. If None and robust=True,
            uses 98th percentile across all frames.
        robust : bool, default True
            Use percentile-based color scaling across all frames.
        mask_radius : int, optional
            Apply circular mask with this radius in pixels.
        figsize : tuple, default (8, 6)
            Figure size in inches.
        dpi : int, default 100
            Resolution for saved animation.
        **kwargs
            Additional arguments passed to FuncAnimation.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            Animation object. Can be displayed in notebooks with HTML(anim.to_jshtml())
            or saved to file.

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> # Create animation and save to file
        >>> anim = ds.radport.animate_frequency(time_idx=0, output_file="freq_sweep.gif")
        >>>
        >>> # Display in Jupyter notebook
        >>> from IPython.display import HTML
        >>> anim = ds.radport.animate_frequency(time_idx=0)
        >>> HTML(anim.to_jshtml())
        """
        from matplotlib.animation import FuncAnimation

        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Resolve time index
        if time_mjd is not None:
            ti = self.nearest_time_idx(time_mjd)
        elif time_idx is not None:
            ti = time_idx
        else:
            ti = 0

        # Get data for all frequencies
        data = self._obj[var].isel(time=ti, polarization=pol)
        n_freqs = len(self._obj.coords["frequency"])
        freqs_hz = self._obj.coords["frequency"].values

        # Compute global color scale from all frames
        if vmin is None or vmax is None:
            all_values = data.values.ravel()
            finite_values = all_values[np.isfinite(all_values)]
            if len(finite_values) > 0:
                if robust:
                    computed_vmin = np.percentile(finite_values, 2)
                    computed_vmax = np.percentile(finite_values, 98)
                else:
                    computed_vmin = np.nanmin(finite_values)
                    computed_vmax = np.nanmax(finite_values)
            else:
                computed_vmin, computed_vmax = 0, 1

            if vmin is None:
                vmin = computed_vmin
            if vmax is None:
                vmax = computed_vmax

        # Create mask if requested
        mask = None
        if mask_radius is not None:
            nl = len(self._obj.coords["l"])
            nm = len(self._obj.coords["m"])
            center_l, center_m = nl // 2, nm // 2
            l_idx, m_idx = np.ogrid[:nl, :nm]
            dist = np.sqrt((l_idx - center_l) ** 2 + (m_idx - center_m) ** 2)
            mask = dist > mask_radius

        # Create figure and initial plot
        fig, ax = plt.subplots(figsize=figsize)

        frame_data = data.isel(frequency=0).values.copy()
        if mask is not None:
            frame_data[mask] = np.nan

        im = ax.imshow(
            frame_data.T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Jy/beam", fontsize=11)

        # Labels
        ax.set_xlabel("l index", fontsize=11)
        ax.set_ylabel("m index", fontsize=11)

        time_val = self._obj.coords["time"].values[ti]
        try:
            time_str = f"{float(time_val):.6f}"
        except (TypeError, ValueError):
            time_str = str(time_val)

        def update(frame: int) -> tuple:
            """Update function for animation."""
            frame_data = data.isel(frequency=frame).values.copy()
            if mask is not None:
                frame_data[mask] = np.nan
            im.set_array(frame_data.T)

            freq_hz = float(freqs_hz[frame])
            ax.set_title(
                f"{var} at t={time_str} MJD, pol={pol}\n"
                f"Frequency: {freq_hz/1e6:.2f} MHz (channel {frame + 1}/{n_freqs})",
                fontsize=11,
            )
            return (im,)

        # Create animation
        anim = FuncAnimation(
            fig,
            update,
            frames=n_freqs,
            interval=1000 // fps,
            blit=True,
            **kwargs,
        )

        # Save if output file specified
        if output_file is not None:
            if output_file.endswith(".gif"):
                anim.save(output_file, writer="pillow", fps=fps, dpi=dpi)
            else:
                anim.save(output_file, writer="ffmpeg", fps=fps, dpi=dpi)
            plt.close(fig)

        return anim

    def export_frames(
        self,
        output_dir: str,
        var: str = "SKY",
        pol: int = 0,
        time_indices: list[int] | None = None,
        freq_indices: list[int] | None = None,
        format: str = "png",
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        mask_radius: int | None = None,
        figsize: tuple[float, float] = (8, 6),
        dpi: int = 150,
        filename_template: str = "{var}_t{time_idx:04d}_f{freq_idx:04d}.{format}",
    ) -> list[str]:
        """Export all (time, freq) frames as individual image files.

        Parameters
        ----------
        output_dir : str
            Directory to save the image files. Will be created if it doesn't exist.
        var : str, default "SKY"
            Data variable to export ("SKY" or "BEAM").
        pol : int, default 0
            Polarization index.
        time_indices : list of int, optional
            Time indices to export. If None, exports all times.
        freq_indices : list of int, optional
            Frequency indices to export. If None, exports all frequencies.
        format : str, default "png"
            Image format (e.g., "png", "jpg", "pdf").
        cmap : str, default "inferno"
            Matplotlib colormap name.
        vmin : float, optional
            Minimum value for color scaling. If None and robust=True,
            uses 2nd percentile across all exported frames.
        vmax : float, optional
            Maximum value for color scaling. If None and robust=True,
            uses 98th percentile across all exported frames.
        robust : bool, default True
            Use percentile-based color scaling across all exported frames.
        mask_radius : int, optional
            Apply circular mask with this radius in pixels.
        figsize : tuple, default (8, 6)
            Figure size in inches.
        dpi : int, default 150
            Resolution for saved images.
        filename_template : str, default "{var}_t{time_idx:04d}_f{freq_idx:04d}.{format}"
            Template for filenames. Available placeholders: {var}, {time_idx},
            {freq_idx}, {time_mjd}, {freq_mhz}, {format}.

        Returns
        -------
        list of str
            List of paths to the saved image files.

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> # Export all frames
        >>> files = ds.radport.export_frames("./frames")
        >>> print(f"Exported {len(files)} frames")
        >>>
        >>> # Export specific time/frequency combinations
        >>> files = ds.radport.export_frames(
        ...     "./frames",
        ...     time_indices=[0, 1, 2],
        ...     freq_indices=[0, 5, 10],
        ... )
        """
        import os

        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get indices to export
        if time_indices is None:
            time_indices = list(range(len(self._obj.coords["time"])))
        if freq_indices is None:
            freq_indices = list(range(len(self._obj.coords["frequency"])))

        # Get coordinate values for labels
        time_values = self._obj.coords["time"].values
        freq_values = self._obj.coords["frequency"].values

        # Compute global color scale from all frames to export
        if vmin is None or vmax is None:
            all_values = []
            for ti in time_indices:
                for fi in freq_indices:
                    frame_data = self._obj[var].isel(
                        time=ti, frequency=fi, polarization=pol
                    ).values
                    all_values.extend(frame_data.ravel())

            all_values = np.array(all_values)
            finite_values = all_values[np.isfinite(all_values)]
            if len(finite_values) > 0:
                if robust:
                    computed_vmin = np.percentile(finite_values, 2)
                    computed_vmax = np.percentile(finite_values, 98)
                else:
                    computed_vmin = np.nanmin(finite_values)
                    computed_vmax = np.nanmax(finite_values)
            else:
                computed_vmin, computed_vmax = 0, 1

            if vmin is None:
                vmin = computed_vmin
            if vmax is None:
                vmax = computed_vmax

        # Create mask if requested
        mask = None
        if mask_radius is not None:
            nl = len(self._obj.coords["l"])
            nm = len(self._obj.coords["m"])
            center_l, center_m = nl // 2, nm // 2
            l_idx, m_idx = np.ogrid[:nl, :nm]
            dist = np.sqrt((l_idx - center_l) ** 2 + (m_idx - center_m) ** 2)
            mask = dist > mask_radius

        # Export frames
        exported_files = []

        for ti in time_indices:
            for fi in freq_indices:
                # Get frame data
                frame_data = self._obj[var].isel(
                    time=ti, frequency=fi, polarization=pol
                ).values.copy()

                if mask is not None:
                    frame_data[mask] = np.nan

                # Create figure
                fig, ax = plt.subplots(figsize=figsize)

                im = ax.imshow(
                    frame_data.T,
                    origin="lower",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    aspect="equal",
                )

                # Add colorbar
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Jy/beam", fontsize=11)

                # Labels
                ax.set_xlabel("l index", fontsize=11)
                ax.set_ylabel("m index", fontsize=11)

                # Title
                time_val = time_values[ti]
                freq_hz = float(freq_values[fi])
                try:
                    time_str = f"{float(time_val):.6f}"
                except (TypeError, ValueError):
                    time_str = str(time_val)

                ax.set_title(
                    f"{var} at t={time_str} MJD, f={freq_hz/1e6:.2f} MHz, pol={pol}",
                    fontsize=11,
                )

                # Generate filename
                try:
                    time_mjd = float(time_val)
                except (TypeError, ValueError):
                    time_mjd = 0.0

                filename = filename_template.format(
                    var=var,
                    time_idx=ti,
                    freq_idx=fi,
                    time_mjd=time_mjd,
                    freq_mhz=freq_hz / 1e6,
                    format=format,
                )
                filepath = os.path.join(output_dir, filename)

                # Save figure
                fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
                plt.close(fig)

                exported_files.append(filepath)

        return exported_files

    def build_fits_hdu(
        self,
        *,
        time_idx: int = 0,
        freq_idx: int = 0,
        pol_idx: int = 0,
        var: str = "SKY",
    ):
        """Build a 4D singleton-axis ``PrimaryHDU`` for one exported ``SKY`` slice.

        Delegates to :func:`ovro_lwa_portal.export_fits.build_fits_hdu`.
        """
        from ovro_lwa_portal import export_fits as export_fits_module

        return export_fits_module.build_fits_hdu(
            self._obj,
            time_idx=time_idx,
            freq_idx=freq_idx,
            pol_idx=pol_idx,
            var=var,
        )

    def export_fits(
        self,
        output_dir: str | Path,
        *,
        var: str = "SKY",
        pol_idx: int = 0,
        pol_indices: list[int] | None = None,
        time_indices: list[int] | None = None,
        freq_indices: list[int] | None = None,
        filename_template: str = "image_t{time_idx:04d}_f{freq_mhz:.3f}MHz_s{stokes}.fits",
        overwrite: bool = False,
    ) -> list[str]:
        """Export ``SKY`` slices as standalone 4D singleton-axis FITS files.

        Writes one FITS file per ``(time_idx, freq_idx, pol_idx)`` combination.
        Each file has ``NAXIS=4`` with ``NAXIS3=NAXIS4=1`` (singleton FREQ and
        Stokes axes). Requires persisted ``fits_header_str`` on the dataset.

        Parameters
        ----------
        output_dir : str or Path
            Directory for exported FITS files (created if missing).
        var : str, default "SKY"
            Data variable to export (``SKY`` only in practice).
        pol_idx : int, default 0
            Polarization index used when *pol_indices* is ``None``.
        pol_indices : list of int, optional
            Polarization indices to export. Defaults to ``[pol_idx]``.
        time_indices : list of int, optional
            Time indices to export. Defaults to all times.
        freq_indices : list of int, optional
            Frequency indices to export. Defaults to all frequencies.
        filename_template : str
            Template for output filenames. Placeholders: ``{time_idx}``,
            ``{freq_idx}``, ``{pol_idx}``, ``{freq_mhz}``, ``{stokes}``.
        overwrite : bool, default False
            Overwrite existing files.

        Returns
        -------
        list of str
            Paths to written FITS files.
        """
        from pathlib import Path

        from ovro_lwa_portal import export_fits as export_fits_module
        from ovro_lwa_portal.fits_to_zarr_xradio import _fits_stokes_from_polarization_coord

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if var not in self._obj.data_vars:
            msg = (
                f"Variable {var!r} not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )
            raise ValueError(msg)

        if time_indices is None:
            time_indices = list(range(int(self._obj.sizes.get("time", 1))))
        if freq_indices is None:
            freq_indices = list(range(int(self._obj.sizes.get("frequency", 1))))
        if pol_indices is None:
            pol_indices = [pol_idx]

        freq_values = np.asarray(self._obj.coords["frequency"].values).ravel()
        pol_values = np.asarray(self._obj.coords["polarization"].values).ravel()

        exported: list[str] = []
        for ti in time_indices:
            for fi in freq_indices:
                freq_hz = float(freq_values[fi])
                freq_mhz = freq_hz / 1e6
                for pi in pol_indices:
                    stokes_raw = pol_values[pi]
                    mapped = _fits_stokes_from_polarization_coord(stokes_raw)
                    stokes_label = (
                        f"{mapped:.0f}" if mapped is not None else str(stokes_raw)
                    )
                    filename = filename_template.format(
                        time_idx=ti,
                        freq_idx=fi,
                        pol_idx=pi,
                        freq_mhz=freq_mhz,
                        stokes=stokes_label,
                    )
                    filepath = out_dir / filename
                    export_fits_module.write_fits_slice(
                        self._obj,
                        filepath,
                        time_idx=ti,
                        freq_idx=fi,
                        pol_idx=pi,
                        var=var,
                        overwrite=overwrite,
                    )
                    exported.append(str(filepath))
        return exported

    # =========================================================================
    # Phase G: Source Detection Methods
    # =========================================================================

    def rms_map(
        self,
        time_idx: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        var: str = "SKY",
        pol: int = 0,
        box_size: int = 50,
    ) -> xr.DataArray:
        """Compute local RMS noise estimate map using a sliding box.

        The RMS is computed using a uniform filter approach where each pixel's
        RMS is estimated from the surrounding box_size x box_size region.

        Parameters
        ----------
        time_idx : int, default 0
            Time index for the frame.
        freq_idx : int, optional
            Frequency index for the frame. Defaults to 0 if neither freq_idx
            nor freq_mhz is provided.
        freq_mhz : float, optional
            Select frequency by value in MHz. Overrides freq_idx if provided.
        var : str, default "SKY"
            Data variable to analyze ("SKY" or "BEAM").
        pol : int, default 0
            Polarization index.
        box_size : int, default 50
            Size of the sliding box for local RMS computation.

        Returns
        -------
        xr.DataArray
            2D array of local RMS values with dimensions (l, m).

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> rms = ds.radport.rms_map(freq_mhz=50.0, box_size=100)
        >>> rms.plot()
        """
        from scipy.ndimage import uniform_filter

        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Resolve frequency index
        if freq_mhz is not None:
            fi = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is not None:
            fi = freq_idx
        else:
            fi = 0

        # Get frame data
        data = self._obj[var].isel(
            time=time_idx, frequency=fi, polarization=pol
        ).values.astype(float)

        # Replace NaN with 0 for filtering (we'll handle NaN regions later)
        nan_mask = ~np.isfinite(data)
        data_filled = np.where(nan_mask, 0.0, data)

        # Compute local mean and mean of squares
        local_mean = uniform_filter(data_filled, size=box_size, mode="constant")
        local_mean_sq = uniform_filter(data_filled**2, size=box_size, mode="constant")

        # Count valid pixels in each box
        valid_count = uniform_filter(
            (~nan_mask).astype(float), size=box_size, mode="constant"
        )
        valid_count = np.maximum(valid_count, 1e-10)  # Avoid division by zero

        # Correct for the fact that we filled NaN with 0
        local_mean = local_mean / valid_count * (box_size**2)
        local_mean_sq = local_mean_sq / valid_count * (box_size**2)

        # Compute variance: E[X^2] - E[X]^2
        local_var = local_mean_sq - local_mean**2
        local_var = np.maximum(local_var, 0.0)  # Ensure non-negative

        # RMS is sqrt of variance
        rms = np.sqrt(local_var)

        # Restore NaN where original was NaN
        rms[nan_mask] = np.nan

        # Create DataArray with coordinates
        return xr.DataArray(
            rms,
            dims=["l", "m"],
            coords={
                "l": self._obj.coords["l"],
                "m": self._obj.coords["m"],
            },
            name="rms",
            attrs={
                "long_name": "Local RMS noise estimate",
                "units": "Jy/beam",
                "box_size": box_size,
            },
        )

    def snr_map(
        self,
        time_idx: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        var: str = "SKY",
        pol: int = 0,
        box_size: int = 50,
    ) -> xr.DataArray:
        """Compute signal-to-noise ratio map.

        The SNR is computed as the signal divided by the local RMS noise
        estimate from a sliding box.

        Parameters
        ----------
        time_idx : int, default 0
            Time index for the frame.
        freq_idx : int, optional
            Frequency index for the frame. Defaults to 0 if neither freq_idx
            nor freq_mhz is provided.
        freq_mhz : float, optional
            Select frequency by value in MHz. Overrides freq_idx if provided.
        var : str, default "SKY"
            Data variable to analyze ("SKY" or "BEAM").
        pol : int, default 0
            Polarization index.
        box_size : int, default 50
            Size of the sliding box for local RMS computation.

        Returns
        -------
        xr.DataArray
            2D array of SNR values with dimensions (l, m).

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> snr = ds.radport.snr_map(freq_mhz=50.0)
        >>> # Find pixels with SNR > 5
        >>> significant = snr.where(snr > 5)
        """
        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Resolve frequency index
        if freq_mhz is not None:
            fi = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is not None:
            fi = freq_idx
        else:
            fi = 0

        # Get signal
        signal = self._obj[var].isel(
            time=time_idx, frequency=fi, polarization=pol
        ).values.astype(float)

        # Get RMS map
        rms = self.rms_map(
            time_idx=time_idx,
            freq_idx=fi,
            var=var,
            pol=pol,
            box_size=box_size,
        ).values

        # Compute SNR (avoiding division by zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = signal / rms
            snr[~np.isfinite(snr)] = np.nan

        # Create DataArray with coordinates
        return xr.DataArray(
            snr,
            dims=["l", "m"],
            coords={
                "l": self._obj.coords["l"],
                "m": self._obj.coords["m"],
            },
            name="snr",
            attrs={
                "long_name": "Signal-to-noise ratio",
                "units": "",
                "box_size": box_size,
            },
        )

    def find_peaks(
        self,
        time_idx: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        var: str = "SKY",
        pol: int = 0,
        threshold_sigma: float = 5.0,
        box_size: int = 50,
        min_separation: int = 5,
    ) -> list[dict]:
        """Find peaks above threshold in the image.

        Identifies local maxima that exceed the specified SNR threshold.
        Uses local maximum detection with minimum separation between peaks.

        Parameters
        ----------
        time_idx : int, default 0
            Time index for the frame.
        freq_idx : int, optional
            Frequency index for the frame. Defaults to 0 if neither freq_idx
            nor freq_mhz is provided.
        freq_mhz : float, optional
            Select frequency by value in MHz. Overrides freq_idx if provided.
        var : str, default "SKY"
            Data variable to analyze ("SKY" or "BEAM").
        pol : int, default 0
            Polarization index.
        threshold_sigma : float, default 5.0
            Minimum SNR threshold for peak detection.
        box_size : int, default 50
            Size of the sliding box for local RMS computation.
        min_separation : int, default 5
            Minimum separation between peaks in pixels.

        Returns
        -------
        list of dict
            List of detected peaks, each with keys:
            - l: l coordinate value
            - m: m coordinate value
            - l_idx: l pixel index
            - m_idx: m pixel index
            - flux: peak flux value (Jy/beam)
            - snr: signal-to-noise ratio
            - ra: Right Ascension in degrees (None if WCS unavailable
              or pixel is outside the projection domain)
            - dec: Declination in degrees (None if WCS unavailable)

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> peaks = ds.radport.find_peaks(freq_mhz=50.0, threshold_sigma=5.0)
        >>> print(f"Found {len(peaks)} peaks")
        >>> for p in peaks[:5]:
        ...     print(f"  l={p['l']:.3f}, m={p['m']:.3f}, flux={p['flux']:.2f}, SNR={p['snr']:.1f}")
        """
        from scipy.ndimage import maximum_filter

        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Resolve frequency index
        if freq_mhz is not None:
            fi = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is not None:
            fi = freq_idx
        else:
            fi = 0

        # Get signal and SNR maps
        signal = self._obj[var].isel(
            time=time_idx, frequency=fi, polarization=pol
        ).values.astype(float)

        snr = self.snr_map(
            time_idx=time_idx,
            freq_idx=fi,
            var=var,
            pol=pol,
            box_size=box_size,
        ).values

        # Find local maxima using maximum filter
        # A pixel is a local max if it equals the max in its neighborhood
        local_max = maximum_filter(signal, size=min_separation * 2 + 1)
        is_local_max = (signal == local_max) & np.isfinite(signal)

        # Apply SNR threshold
        is_peak = is_local_max & (snr >= threshold_sigma)

        # Get peak locations
        l_indices, m_indices = np.where(is_peak)

        # Get coordinate values
        l_coords = self._obj.coords["l"].values
        m_coords = self._obj.coords["m"].values

        # Build list of peaks sorted by SNR (descending)
        peaks = []
        for l_idx, m_idx in zip(l_indices, m_indices):
            peak: dict[str, Any] = {
                "l": float(l_coords[l_idx]),
                "m": float(m_coords[m_idx]),
                "l_idx": int(l_idx),
                "m_idx": int(m_idx),
                "flux": float(signal[l_idx, m_idx]),
                "snr": float(snr[l_idx, m_idx]),
            }
            # Always include ra/dec keys for consistent caller interface.
            # Set to None when WCS is unavailable or conversion fails
            # (e.g., edge pixels outside the SIN projection domain).
            peak["ra"] = None
            peak["dec"] = None
            if self.has_wcs:
                try:
                    ra_val, dec_val = self.pixel_to_coords(
                        int(l_idx), int(m_idx), time_idx=time_idx
                    )
                    # WCS returns NaN for pixels outside the SIN
                    # projection domain (near l²+m²≈1).  Keep None.
                    if np.isfinite(ra_val) and np.isfinite(dec_val):
                        peak["ra"] = ra_val
                        peak["dec"] = dec_val
                except (ValueError, ImportError):
                    pass
            peaks.append(peak)

        # Sort by SNR descending
        peaks.sort(key=lambda p: p["snr"], reverse=True)

        return peaks

    def peak_flux_map(
        self,
        var: str = "SKY",
        pol: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
    ) -> xr.DataArray:
        """Compute peak flux at each pixel across all times.

        For each (l, m) pixel, finds the maximum flux value across
        all time steps at the specified frequency.

        Parameters
        ----------
        var : str, default "SKY"
            Data variable to analyze ("SKY" or "BEAM").
        pol : int, default 0
            Polarization index.
        freq_idx : int, optional
            Frequency index. Defaults to 0 if neither freq_idx
            nor freq_mhz is provided.
        freq_mhz : float, optional
            Select frequency by value in MHz. Overrides freq_idx if provided.

        Returns
        -------
        xr.DataArray
            2D array of peak flux values with dimensions (l, m).

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> # Find brightest emission at each pixel across all times
        >>> peak_map = ds.radport.peak_flux_map(freq_mhz=50.0)
        >>> peak_map.plot()
        """
        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Resolve frequency index
        if freq_mhz is not None:
            fi = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is not None:
            fi = freq_idx
        else:
            fi = 0

        # Get data for all times at this frequency
        data = self._obj[var].isel(frequency=fi, polarization=pol)

        # Compute max across time dimension
        peak_flux = data.max(dim="time", skipna=True)

        # Update attributes
        peak_flux.name = "peak_flux"
        peak_flux.attrs = {
            "long_name": "Peak flux across time",
            "units": "Jy/beam",
        }

        return peak_flux

    def plot_snr_map(
        self,
        time_idx: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        var: str = "SKY",
        pol: int = 0,
        box_size: int = 50,
        cmap: str = "RdBu_r",
        vmin: float | None = None,
        vmax: float | None = None,
        mask_radius: int | None = None,
        figsize: tuple[float, float] = (8, 6),
        add_colorbar: bool = True,
        symmetric: bool = True,
    ) -> "Figure":
        """Plot the signal-to-noise ratio map.

        Parameters
        ----------
        time_idx : int, default 0
            Time index for the frame.
        freq_idx : int, optional
            Frequency index for the frame.
        freq_mhz : float, optional
            Select frequency by value in MHz.
        var : str, default "SKY"
            Data variable to analyze.
        pol : int, default 0
            Polarization index.
        box_size : int, default 50
            Size of the sliding box for local RMS computation.
        cmap : str, default "RdBu_r"
            Colormap (diverging recommended for SNR).
        vmin : float, optional
            Minimum value for color scaling.
        vmax : float, optional
            Maximum value for color scaling.
        mask_radius : int, optional
            Apply circular mask with this radius in pixels.
        figsize : tuple, default (8, 6)
            Figure size in inches.
        add_colorbar : bool, default True
            Whether to add a colorbar.
        symmetric : bool, default True
            Use symmetric color scale centered at zero.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.

        Example
        -------
        >>> fig = ds.radport.plot_snr_map(freq_mhz=50.0, mask_radius=1800)
        """
        # Get SNR map
        snr = self.snr_map(
            time_idx=time_idx,
            freq_idx=freq_idx,
            freq_mhz=freq_mhz,
            var=var,
            pol=pol,
            box_size=box_size,
        )

        snr_values = snr.values.copy()

        # Apply mask if requested
        if mask_radius is not None:
            nl = len(self._obj.coords["l"])
            nm = len(self._obj.coords["m"])
            center_l, center_m = nl // 2, nm // 2
            l_idx, m_idx = np.ogrid[:nl, :nm]
            dist = np.sqrt((l_idx - center_l) ** 2 + (m_idx - center_m) ** 2)
            mask = dist > mask_radius
            snr_values[mask] = np.nan

        # Compute color scale
        if symmetric and vmin is None and vmax is None:
            finite_vals = snr_values[np.isfinite(snr_values)]
            if len(finite_vals) > 0:
                max_abs = np.percentile(np.abs(finite_vals), 98)
                vmin = -max_abs
                vmax = max_abs

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            snr_values.T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        # Add colorbar
        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("SNR (σ)", fontsize=11)

        # Labels
        ax.set_xlabel("l index", fontsize=11)
        ax.set_ylabel("m index", fontsize=11)

        # Get frequency for title
        if freq_mhz is not None:
            fi = self.nearest_freq_idx(freq_mhz)
        elif freq_idx is not None:
            fi = freq_idx
        else:
            fi = 0

        freq_hz = float(self._obj.coords["frequency"].values[fi])
        time_val = self._obj.coords["time"].values[time_idx]
        try:
            time_str = f"{float(time_val):.6f}"
        except (TypeError, ValueError):
            time_str = str(time_val)

        ax.set_title(
            f"SNR Map at t={time_str} MJD, f={freq_hz/1e6:.2f} MHz\n"
            f"(box_size={box_size})",
            fontsize=11,
        )

        return fig

    # =========================================================================
    # Phase H: Spectral Analysis Methods
    # =========================================================================

    def spectral_index(
        self,
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        time_idx: int = 0,
        pol: int = 0,
        freq1_mhz: float | None = None,
        freq2_mhz: float | None = None,
        freq1_idx: int | None = None,
        freq2_idx: int | None = None,
        var: str = "SKY",
    ) -> float:
        """Compute spectral index (power-law slope) between two frequencies.

        The spectral index α is defined by the power-law relationship S ∝ ν^α,
        computed as: α = log(S2/S1) / log(ν2/ν1)

        Parameters
        ----------
        ra : float, optional
            Right Ascension in degrees (FK5/J2000). Requires ``dec``.
        dec : float, optional
            Declination in degrees (FK5/J2000). Requires ``ra``.
        l : float, optional
            The l coordinate value for the pixel location. Requires ``m``.
        m : float, optional
            The m coordinate value for the pixel location. Requires ``l``.
        time_idx : int, default 0
            Time index for the measurement.
        pol : int, default 0
            Polarization index.
        freq1_mhz : float, optional
            First frequency in MHz. If not provided, uses freq1_idx or first channel.
        freq2_mhz : float, optional
            Second frequency in MHz. If not provided, uses freq2_idx or last channel.
        freq1_idx : int, optional
            First frequency index. Overridden by freq1_mhz if provided.
        freq2_idx : int, optional
            Second frequency index. Overridden by freq2_mhz if provided.
        var : str, default "SKY"
            Data variable to analyze.

        Returns
        -------
        float
            Spectral index α where S ∝ ν^α. Returns NaN if calculation
            is not possible (e.g., non-positive flux values).

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> # Compute spectral index at image center between 46 and 54 MHz
        >>> alpha = ds.radport.spectral_index(
        ...     l=0.0, m=0.0,
        ...     freq1_mhz=46.0,
        ...     freq2_mhz=54.0,
        ... )
        >>> print(f"Spectral index: {alpha:.2f}")

        Notes
        -----
        - Assumes power-law spectrum: S ∝ ν^α
        - Returns NaN for non-positive flux values (cannot take log)
        - Typical radio sources have α ≈ -0.7 (synchrotron emission)
        """
        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Resolve frequency indices (also selects RA/Dec grid channel when used)
        if freq1_mhz is not None:
            fi1 = self.nearest_freq_idx(freq1_mhz)
        elif freq1_idx is not None:
            fi1 = freq1_idx
        else:
            fi1 = 0

        if freq2_mhz is not None:
            fi2 = self.nearest_freq_idx(freq2_mhz)
        elif freq2_idx is not None:
            fi2 = freq2_idx
        else:
            fi2 = len(self._obj.coords["frequency"]) - 1

        # Resolve coordinates
        if ra is not None or dec is not None:
            if ra is None or dec is None:
                raise ValueError("Both ra and dec must be provided together.")
            l_idx, m_idx = self.coords_to_pixel(
                ra, dec, time_idx=time_idx, freq_idx=fi1, pol=pol
            )
        elif l is not None or m is not None:
            if l is None or m is None:
                raise ValueError("Both l and m must be provided together.")
            l_idx, m_idx = self.nearest_lm_idx(l, m)
        else:
            raise ValueError("Must provide either (ra, dec) or (l, m) coordinates.")

        # Get flux values at both frequencies
        s1 = float(
            self._obj[var]
            .isel(time=time_idx, frequency=fi1, polarization=pol, l=l_idx, m=m_idx)
            .values
        )
        s2 = float(
            self._obj[var]
            .isel(time=time_idx, frequency=fi2, polarization=pol, l=l_idx, m=m_idx)
            .values
        )

        # Get frequency values in Hz
        nu1 = float(self._obj.coords["frequency"].values[fi1])
        nu2 = float(self._obj.coords["frequency"].values[fi2])

        # Compute spectral index: α = log(S2/S1) / log(ν2/ν1)
        # Handle non-positive flux values
        if s1 <= 0 or s2 <= 0 or nu1 <= 0 or nu2 <= 0 or nu1 == nu2:
            return float("nan")

        alpha = np.log(s2 / s1) / np.log(nu2 / nu1)
        return float(alpha)

    def spectral_index_map(
        self,
        time_idx: int = 0,
        pol: int = 0,
        freq1_mhz: float | None = None,
        freq2_mhz: float | None = None,
        freq1_idx: int | None = None,
        freq2_idx: int | None = None,
        var: str = "SKY",
    ) -> xr.DataArray:
        """Compute spectral index map across the image.

        Computes the spectral index α at each pixel, where S ∝ ν^α.

        Parameters
        ----------
        time_idx : int, default 0
            Time index for the measurement.
        pol : int, default 0
            Polarization index.
        freq1_mhz : float, optional
            First frequency in MHz. If not provided, uses freq1_idx or first channel.
        freq2_mhz : float, optional
            Second frequency in MHz. If not provided, uses freq2_idx or last channel.
        freq1_idx : int, optional
            First frequency index. Overridden by freq1_mhz if provided.
        freq2_idx : int, optional
            Second frequency index. Overridden by freq2_mhz if provided.
        var : str, default "SKY"
            Data variable to analyze.

        Returns
        -------
        xr.DataArray
            2D array of spectral index values with dimensions (l, m).
            NaN values indicate pixels where the calculation was not possible.

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> # Compute spectral index map between first and last frequency
        >>> alpha_map = ds.radport.spectral_index_map()
        >>> alpha_map.plot(vmin=-3, vmax=1, cmap="RdBu_r")
        >>>
        >>> # Compute between specific frequencies
        >>> alpha_map = ds.radport.spectral_index_map(
        ...     freq1_mhz=46.0,
        ...     freq2_mhz=54.0,
        ... )
        """
        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Resolve frequency indices
        if freq1_mhz is not None:
            fi1 = self.nearest_freq_idx(freq1_mhz)
        elif freq1_idx is not None:
            fi1 = freq1_idx
        else:
            fi1 = 0

        if freq2_mhz is not None:
            fi2 = self.nearest_freq_idx(freq2_mhz)
        elif freq2_idx is not None:
            fi2 = freq2_idx
        else:
            fi2 = len(self._obj.coords["frequency"]) - 1

        # Get flux arrays at both frequencies
        s1 = self._obj[var].isel(
            time=time_idx, frequency=fi1, polarization=pol
        ).values.astype(float)
        s2 = self._obj[var].isel(
            time=time_idx, frequency=fi2, polarization=pol
        ).values.astype(float)

        # Get frequency values in Hz
        nu1 = float(self._obj.coords["frequency"].values[fi1])
        nu2 = float(self._obj.coords["frequency"].values[fi2])

        # Compute spectral index: α = log(S2/S1) / log(ν2/ν1)
        with np.errstate(divide="ignore", invalid="ignore"):
            # Mask non-positive values
            valid_mask = (s1 > 0) & (s2 > 0)
            alpha = np.full_like(s1, np.nan)
            alpha[valid_mask] = (
                np.log(s2[valid_mask] / s1[valid_mask]) / np.log(nu2 / nu1)
            )

        # Create DataArray with coordinates
        return xr.DataArray(
            alpha,
            dims=["l", "m"],
            coords={
                "l": self._obj.coords["l"],
                "m": self._obj.coords["m"],
            },
            name="spectral_index",
            attrs={
                "long_name": "Spectral index",
                "units": "",
                "freq1_hz": nu1,
                "freq2_hz": nu2,
                "freq1_mhz": nu1 / 1e6,
                "freq2_mhz": nu2 / 1e6,
            },
        )

    def integrated_flux(
        self,
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        time_idx: int = 0,
        pol: int = 0,
        freq_min_mhz: float | None = None,
        freq_max_mhz: float | None = None,
        freq_indices: list[int] | None = None,
        var: str = "SKY",
    ) -> float:
        """Compute integrated flux density over a frequency band.

        Integrates the flux density across the specified frequency range
        using the trapezoidal rule.

        Parameters
        ----------
        ra : float, optional
            Right Ascension in degrees (FK5/J2000). Requires ``dec``.
        dec : float, optional
            Declination in degrees (FK5/J2000). Requires ``ra``.
        l : float, optional
            The l coordinate value for the pixel location. Requires ``m``.
        m : float, optional
            The m coordinate value for the pixel location. Requires ``l``.
        time_idx : int, default 0
            Time index for the measurement.
        pol : int, default 0
            Polarization index.
        freq_min_mhz : float, optional
            Minimum frequency in MHz. If not provided, uses full range.
        freq_max_mhz : float, optional
            Maximum frequency in MHz. If not provided, uses full range.
        freq_indices : list of int, optional
            Specific frequency indices to include. Overrides freq_min/max_mhz.
        var : str, default "SKY"
            Data variable to analyze.

        Returns
        -------
        float
            Integrated flux density in Jy·Hz. Divide by bandwidth to get
            average flux density.

        Raises
        ------
        ValueError
            If the specified variable doesn't exist in the dataset.

        Example
        -------
        >>> # Compute integrated flux at image center across all frequencies
        >>> flux = ds.radport.integrated_flux(l=0.0, m=0.0)
        >>> print(f"Integrated flux: {flux:.2e} Jy·Hz")
        >>>
        >>> # Compute over specific band
        >>> flux = ds.radport.integrated_flux(
        ...     l=0.0, m=0.0,
        ...     freq_min_mhz=45.0,
        ...     freq_max_mhz=55.0,
        ... )

        Notes
        -----
        Uses trapezoidal integration over the frequency axis.
        """
        # Validate variable
        if var not in self._obj.data_vars:
            raise ValueError(
                f"Variable '{var}' not found in dataset. "
                f"Available variables: {list(self._obj.data_vars)}."
            )

        # Reference channel for RA/Dec coordinate grids that vary with frequency.
        fi_for_coord = 0
        if freq_indices is not None and len(freq_indices) > 0:
            fi_for_coord = int(min(freq_indices))
        elif freq_min_mhz is not None:
            fi_for_coord = self.nearest_freq_idx(freq_min_mhz)

        # Resolve coordinates
        if ra is not None or dec is not None:
            if ra is None or dec is None:
                raise ValueError("Both ra and dec must be provided together.")
            l_idx, m_idx = self.coords_to_pixel(
                ra, dec, time_idx=time_idx, freq_idx=fi_for_coord, pol=pol
            )
        elif l is not None or m is not None:
            if l is None or m is None:
                raise ValueError("Both l and m must be provided together.")
            l_idx, m_idx = self.nearest_lm_idx(l, m)
        else:
            raise ValueError("Must provide either (ra, dec) or (l, m) coordinates.")

        # Get all frequency values
        freq_hz = self._obj.coords["frequency"].values

        # Determine which frequencies to include
        if freq_indices is not None:
            indices = freq_indices
        else:
            if freq_min_mhz is not None:
                min_idx = self.nearest_freq_idx(freq_min_mhz)
            else:
                min_idx = 0

            if freq_max_mhz is not None:
                max_idx = self.nearest_freq_idx(freq_max_mhz)
            else:
                max_idx = len(freq_hz) - 1

            # Ensure proper ordering
            if min_idx > max_idx:
                min_idx, max_idx = max_idx, min_idx

            indices = list(range(min_idx, max_idx + 1))

        if len(indices) < 2:
            # Need at least 2 points for integration
            if len(indices) == 1:
                # Return single point value (no integration possible)
                return float(
                    self._obj[var]
                    .isel(
                        time=time_idx,
                        frequency=indices[0],
                        polarization=pol,
                        l=l_idx,
                        m=m_idx,
                    )
                    .values
                )
            return 0.0

        # Get flux values at selected frequencies
        flux_values = []
        freq_values = []
        for fi in indices:
            flux = float(
                self._obj[var]
                .isel(time=time_idx, frequency=fi, polarization=pol, l=l_idx, m=m_idx)
                .values
            )
            flux_values.append(flux)
            freq_values.append(float(freq_hz[fi]))

        flux_values = np.array(flux_values)
        freq_values = np.array(freq_values)

        # Integrate using trapezoidal rule
        integrated = np.trapezoid(flux_values, freq_values)

        return float(integrated)

    def plot_spectral_index_map(
        self,
        time_idx: int = 0,
        pol: int = 0,
        freq1_mhz: float | None = None,
        freq2_mhz: float | None = None,
        freq1_idx: int | None = None,
        freq2_idx: int | None = None,
        var: str = "SKY",
        cmap: str = "RdBu_r",
        vmin: float | None = -3.0,
        vmax: float | None = 1.0,
        mask_radius: int | None = None,
        figsize: tuple[float, float] = (8, 6),
        add_colorbar: bool = True,
    ) -> "Figure":
        """Plot the spectral index map.

        Parameters
        ----------
        time_idx : int, default 0
            Time index for the measurement.
        pol : int, default 0
            Polarization index.
        freq1_mhz : float, optional
            First frequency in MHz.
        freq2_mhz : float, optional
            Second frequency in MHz.
        freq1_idx : int, optional
            First frequency index.
        freq2_idx : int, optional
            Second frequency index.
        var : str, default "SKY"
            Data variable to analyze.
        cmap : str, default "RdBu_r"
            Colormap (diverging recommended for spectral index).
        vmin : float, default -3.0
            Minimum value for color scaling.
        vmax : float, default 1.0
            Maximum value for color scaling.
        mask_radius : int, optional
            Apply circular mask with this radius in pixels.
        figsize : tuple, default (8, 6)
            Figure size in inches.
        add_colorbar : bool, default True
            Whether to add a colorbar.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.

        Example
        -------
        >>> fig = ds.radport.plot_spectral_index_map(
        ...     freq1_mhz=46.0,
        ...     freq2_mhz=54.0,
        ...     mask_radius=1800,
        ... )
        """
        # Get spectral index map
        alpha_map = self.spectral_index_map(
            time_idx=time_idx,
            pol=pol,
            freq1_mhz=freq1_mhz,
            freq2_mhz=freq2_mhz,
            freq1_idx=freq1_idx,
            freq2_idx=freq2_idx,
            var=var,
        )

        alpha_values = alpha_map.values.copy()

        # Apply mask if requested
        if mask_radius is not None:
            nl = len(self._obj.coords["l"])
            nm = len(self._obj.coords["m"])
            center_l, center_m = nl // 2, nm // 2
            l_idx, m_idx = np.ogrid[:nl, :nm]
            dist = np.sqrt((l_idx - center_l) ** 2 + (m_idx - center_m) ** 2)
            mask = dist > mask_radius
            alpha_values[mask] = np.nan

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            alpha_values.T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )

        # Add colorbar
        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Spectral Index (α)", fontsize=11)

        # Labels
        ax.set_xlabel("l index", fontsize=11)
        ax.set_ylabel("m index", fontsize=11)

        # Title
        freq1_hz = alpha_map.attrs.get("freq1_hz", 0)
        freq2_hz = alpha_map.attrs.get("freq2_hz", 0)
        time_val = self._obj.coords["time"].values[time_idx]
        try:
            time_str = f"{float(time_val):.6f}"
        except (TypeError, ValueError):
            time_str = str(time_val)

        ax.set_title(
            f"Spectral Index Map at t={time_str} MJD\n"
            f"({freq1_hz/1e6:.1f} - {freq2_hz/1e6:.1f} MHz)",
            fontsize=11,
        )

        return fig

    # =========================================================================
    # Dispersion Measure Correction Methods
    # =========================================================================

    # Dispersion constant in MHz^2 pc^-1 cm^3 s
    # Reference: Lorimer & Kramer (2004), Handbook of Pulsar Astronomy
    K_DM = 4.148808e3  # MHz^2 pc^-1 cm^3 s

    def dispersion_delay(
        self,
        dm: float,
        freq_mhz: float | np.ndarray | None = None,
        freq_ref_mhz: float | None = None,
    ) -> float | np.ndarray:
        """Calculate dispersion delay for a given DM and frequency.

        Radio signals experience frequency-dependent delays when propagating
        through the ionized interstellar medium. Lower frequencies arrive
        later than higher frequencies. This method computes the time delay
        using the cold plasma dispersion relation.

        Parameters
        ----------
        dm : float
            Dispersion measure in pc cm^-3. Must be non-negative.
        freq_mhz : float or np.ndarray, optional
            Frequency or array of frequencies in MHz at which to compute
            delays. If None, uses all frequencies in the dataset.
        freq_ref_mhz : float, optional
            Reference frequency in MHz (typically the highest frequency).
            Delays are computed relative to this frequency.
            If None, uses the highest frequency in the dataset.

        Returns
        -------
        float or np.ndarray
            Time delay(s) in seconds. Positive values indicate the signal
            arrives later at lower frequencies. Returns the same shape as
            freq_mhz input.

        Raises
        ------
        ValueError
            If dm is negative.

        Notes
        -----
        The dispersion delay is computed using:

            Δt = K_DM × DM × (f_lo^-2 - f_hi^-2)

        where:
        - K_DM = 4.148808 × 10^3 MHz^2 pc^-1 cm^3 s (dispersion constant)
        - DM is the dispersion measure in pc cm^-3
        - f_lo, f_hi are frequencies in MHz

        Example
        -------
        >>> # Crab pulsar DM = 56.8 pc cm^-3
        >>> dm = 56.8
        >>> delay = ds.radport.dispersion_delay(dm=dm, freq_mhz=46.0)
        >>> print(f"Delay at 46 MHz: {delay:.3f} seconds")

        >>> # Get delays at all dataset frequencies
        >>> delays = ds.radport.dispersion_delay(dm=56.8)

        References
        ----------
        .. [1] Lorimer & Kramer (2004), "Handbook of Pulsar Astronomy"
        """
        # Validate DM
        if dm < 0:
            raise ValueError(f"DM must be non-negative, got {dm}")

        # Get frequencies
        if freq_mhz is None:
            freq_mhz = self._obj.coords["frequency"].values / 1e6

        freq_mhz = np.asarray(freq_mhz)

        # Get reference frequency (highest frequency by default)
        if freq_ref_mhz is None:
            freq_ref_mhz = float(self._obj.coords["frequency"].values.max() / 1e6)

        # Validate reference frequency
        if freq_ref_mhz <= 0:
            raise ValueError(f"Reference frequency must be positive, got {freq_ref_mhz}")

        # Validate input frequencies
        if np.any(freq_mhz <= 0):
            raise ValueError("All frequencies must be positive")

        # Compute delay: Δt = K_DM × DM × (f^-2 - f_ref^-2)
        delay = self.K_DM * dm * (freq_mhz**-2 - freq_ref_mhz**-2)

        return delay

    def dynamic_spectrum_dedispersed(
        self,
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        dm: float,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        method: Literal["shift", "interpolate"] = "shift",
        fill_value: float = np.nan,
        trim: bool = False,
        observatory: Any = None,
    ) -> xr.DataArray:
        """Extract a dedispersed dynamic spectrum for a single pixel.

        Corrects for interstellar dispersion by shifting or interpolating
        frequency channels according to the dispersion delay. This is essential
        for analyzing dispersed radio transients like pulsars and FRBs.

        Parameters
        ----------
        ra : float, optional
            Right Ascension in degrees (FK5/J2000). Requires ``dec``.
        dec : float, optional
            Declination in degrees (FK5/J2000). Requires ``ra``.
        l : float, optional
            Target l direction cosine coordinate. Requires ``m``.
        m : float, optional
            Target m direction cosine coordinate. Requires ``l``.
        dm : float
            Dispersion measure in pc cm^-3. Must be non-negative.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to extract.
        pol : int, default 0
            Polarization index.
        freq_idx : int, optional
            Passed to :meth:`dynamic_spectrum` for RA/Dec pixel selection.
        freq_mhz : float, optional
            Passed to :meth:`dynamic_spectrum`; overrides ``freq_idx``.
        method : {'shift', 'interpolate'}, default 'shift'
            Dedispersion method:
            - 'shift': Fast integer-sample shifting (approximate).
              Rounds delays to nearest time sample.
            - 'interpolate': Slower but precise sub-sample interpolation.
              Uses linear interpolation for accurate delay correction.
        fill_value : float, default np.nan
            Value to use for samples shifted outside the time range.
        trim : bool, default False
            If True, trim the time axis to only include valid data
            (removes NaN edges from shifting). If False, returns full
            time axis with NaN-filled edges.

        Returns
        -------
        xr.DataArray
            2D DataArray with dimensions (time, frequency) containing
            the dedispersed dynamic spectrum. Time axis represents
            arrival time at the reference frequency.
            Includes metadata: pixel_l, pixel_m, dm, method, freq_ref_mhz.

        Raises
        ------
        ValueError
            If dm is negative, variable doesn't exist, or method is invalid.

        Warns
        -----
        UserWarning
            If the maximum dispersion shift exceeds 50% of the time span.

        Notes
        -----
        The dedispersion aligns all frequency channels to a common reference
        time (typically the highest frequency). Lower frequency channels are
        shifted backwards in time to compensate for the dispersion delay.

        For the 'shift' method, delays are rounded to the nearest integer
        number of time samples, which introduces quantization error. For
        precise analysis, use 'interpolate'.

        Example
        -------
        >>> # Dedisperse at Crab pulsar DM
        >>> dynspec = ds.radport.dynamic_spectrum_dedispersed(
        ...     l=0.0, m=0.0, dm=56.8, method="interpolate"
        ... )

        >>> # Fast approximate dedispersion
        >>> dynspec_fast = ds.radport.dynamic_spectrum_dedispersed(
        ...     l=0.0, m=0.0, dm=56.8, method="shift", trim=True
        ... )

        See Also
        --------
        dispersion_delay : Compute dispersion delays for given frequencies.
        dynamic_spectrum : Extract uncorrected dynamic spectrum.
        plot_dynamic_spectrum : Plot dynamic spectrum with optional dedispersion.
        """
        # Validate inputs
        if dm < 0:
            raise ValueError(f"DM must be non-negative, got {dm}")

        if method not in ("shift", "interpolate"):
            raise ValueError(
                f"Method must be 'shift' or 'interpolate', got '{method}'"
            )

        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(
                f"Variable '{var}' not found. Available: {available}"
            )

        # Get the uncorrected dynamic spectrum
        dynspec = self.dynamic_spectrum(
            ra=ra,
            dec=dec,
            l=l,
            m=m,
            var=var,
            pol=pol,
            freq_idx=freq_idx,
            freq_mhz=freq_mhz,
            observatory=observatory,
        )

        # If DM is zero, return the original spectrum
        if dm == 0:
            dynspec.attrs["dm"] = 0.0
            dynspec.attrs["method"] = method
            dynspec.attrs["freq_ref_mhz"] = float(
                self._obj.coords["frequency"].values.max() / 1e6
            )
            return dynspec

        # Get coordinates
        time_vals = dynspec.coords["time"].values
        freq_vals = dynspec.coords["frequency"].values  # Hz
        freq_mhz = freq_vals / 1e6

        # Compute reference frequency (highest)
        freq_ref_mhz = float(freq_mhz.max())

        # Compute dispersion delays for each frequency channel
        delays = self.dispersion_delay(dm=dm, freq_mhz=freq_mhz, freq_ref_mhz=freq_ref_mhz)

        # Get time resolution
        if len(time_vals) < 2:
            raise ValueError("Need at least 2 time samples for dedispersion")

        dt = float(time_vals[1] - time_vals[0])  # Time resolution in MJD
        dt_seconds = dt * 86400.0  # Convert to seconds

        # Convert delays to time samples
        delay_samples = delays / dt_seconds

        # Check for excessive delays
        max_delay_samples = np.abs(delay_samples).max()
        if max_delay_samples > 0.5 * len(time_vals):
            warnings.warn(
                f"Maximum dispersion shift ({max_delay_samples:.1f} samples) "
                f"exceeds 50% of time span ({len(time_vals)} samples). "
                "Consider using a smaller DM or longer observation.",
                UserWarning,
                stacklevel=2,
            )

        # Get data values
        data = dynspec.values.copy()  # Shape: (time, frequency)
        n_time, n_freq = data.shape

        # Create output array
        dedispersed = np.full_like(data, fill_value)

        if method == "shift":
            # Integer sample shifting (fast, approximate)
            for i_freq in range(n_freq):
                shift = int(np.round(delay_samples[i_freq]))

                if shift == 0:
                    dedispersed[:, i_freq] = data[:, i_freq]
                elif shift > 0:
                    # Signal arrives later at lower freq, shift backwards
                    if shift < n_time:
                        dedispersed[:-shift, i_freq] = data[shift:, i_freq]
                else:
                    # Negative shift (shouldn't happen for positive DM)
                    shift = abs(shift)
                    if shift < n_time:
                        dedispersed[shift:, i_freq] = data[:-shift, i_freq]

        else:  # method == "interpolate"
            # Sub-sample interpolation (slower, precise)
            for i_freq in range(n_freq):
                delay_mjd = delays[i_freq] / 86400.0  # Convert to MJD

                # Create interpolator for this frequency channel
                interp_func = interpolate.interp1d(
                    time_vals,
                    data[:, i_freq],
                    kind="linear",
                    bounds_error=False,
                    fill_value=fill_value,
                )

                # Interpolate at shifted times
                # To correct for dispersion, we sample at time + delay
                shifted_times = time_vals + delay_mjd
                dedispersed[:, i_freq] = interp_func(shifted_times)

        # Trim if requested
        if trim:
            # Find valid time range (where all frequencies have data)
            valid_mask = ~np.all(np.isnan(dedispersed), axis=1)
            if np.any(valid_mask):
                first_valid = np.argmax(valid_mask)
                last_valid = len(valid_mask) - np.argmax(valid_mask[::-1]) - 1
                dedispersed = dedispersed[first_valid:last_valid + 1, :]
                time_vals = time_vals[first_valid:last_valid + 1]

        # Create output DataArray
        result = xr.DataArray(
            dedispersed,
            dims=["time", "frequency"],
            coords={
                "time": time_vals,
                "frequency": freq_vals,
            },
            name=f"{var}_dedispersed",
            attrs={
                "pixel_l": dynspec.attrs["pixel_l"],
                "pixel_m": dynspec.attrs["pixel_m"],
                "l_idx": dynspec.attrs["l_idx"],
                "m_idx": dynspec.attrs["m_idx"],
                "pol": pol,
                "dm": dm,
                "method": method,
                "freq_ref_mhz": freq_ref_mhz,
                "long_name": f"Dedispersed {var} (DM={dm:.2f} pc/cm³)",
                "units": "Jy/beam",
            },
        )

        return result

    def plot_dynamic_spectrum_dedispersed(
        self,
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        dm: float,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        method: Literal["shift", "interpolate"] = "shift",
        trim: bool = False,
        cmap: str = "inferno",
        vmin: float | None = None,
        vmax: float | None = None,
        robust: bool = True,
        figsize: tuple[float, float] = (10, 5),
        add_colorbar: bool = True,
        show_delay_curve: bool = False,
        observatory: Any = None,
        **kwargs: Any,
    ) -> "Figure":
        """Plot a dedispersed dynamic spectrum for a single pixel.

        Creates a 2D visualization showing intensity variations across
        time and frequency after correcting for interstellar dispersion.

        Parameters
        ----------
        l : float
            Target l coordinate for pixel selection.
        m : float
            Target m coordinate for pixel selection.
        dm : float
            Dispersion measure in pc cm^-3.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to plot.
        pol : int, default 0
            Polarization index.
        freq_idx : int, optional
            Passed to :meth:`dynamic_spectrum_dedispersed` for RA/Dec pixel selection.
        freq_mhz : float, optional
            Passed to :meth:`dynamic_spectrum_dedispersed`; overrides ``freq_idx``.
        method : {'shift', 'interpolate'}, default 'shift'
            Dedispersion method ('shift' for fast, 'interpolate' for precise).
        trim : bool, default False
            If True, trim time axis to valid data only.
        cmap : str, default 'inferno'
            Matplotlib colormap.
        vmin, vmax : float, optional
            Color scale limits.
        robust : bool, default True
            Use percentile-based color scaling.
        figsize : tuple, default (10, 5)
            Figure size in inches.
        add_colorbar : bool, default True
            Whether to add a colorbar.
        show_delay_curve : bool, default False
            If True, overlay the dispersion delay curve on the plot.
        **kwargs : dict
            Additional arguments passed to imshow.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the dedispersed dynamic spectrum plot.

        Example
        -------
        >>> fig = ds.radport.plot_dynamic_spectrum_dedispersed(
        ...     l=0.0, m=0.0, dm=56.8, method="interpolate"
        ... )
        """
        # Get dedispersed dynamic spectrum
        dynspec = self.dynamic_spectrum_dedispersed(
            ra=ra,
            dec=dec,
            l=l,
            m=m,
            dm=dm,
            var=var,
            pol=pol,
            freq_idx=freq_idx,
            freq_mhz=freq_mhz,
            method=method,
            trim=trim,
            observatory=observatory,
        )

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Compute data
        data = dynspec.values

        # Handle robust scaling
        if robust and vmin is None and vmax is None:
            finite_data = data[np.isfinite(data)]
            if finite_data.size > 0:
                vmin = float(np.percentile(finite_data, 2))
                vmax = float(np.percentile(finite_data, 98))

        # Get coordinate values
        time_vals = dynspec.coords["time"].values
        freq_vals = dynspec.coords["frequency"].values / 1e6  # Convert to MHz

        # Compute extent for imshow
        extent = [
            float(time_vals.min()), float(time_vals.max()),
            float(freq_vals.min()), float(freq_vals.max()),
        ]

        # Plot - transpose so time is x-axis and frequency is y-axis
        im = ax.imshow(
            data.T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent,
            aspect="auto",
            **kwargs,
        )

        # Optionally show dispersion delay curve
        if show_delay_curve and dm > 0:
            delays = self.dispersion_delay(dm=dm, freq_mhz=freq_vals)
            # Convert delays to MJD offset from reference
            delay_mjd = delays / 86400.0
            # Plot as time offset from center of time range
            t_center = (time_vals.min() + time_vals.max()) / 2
            ax.plot(
                t_center - delay_mjd,
                freq_vals,
                "w--",
                linewidth=1.5,
                alpha=0.7,
                label="Dispersion curve",
            )
            ax.legend(loc="upper right")

        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Jy/beam")

        # Labels and title
        ax.set_xlabel("Time (MJD)")
        ax.set_ylabel("Frequency (MHz)")

        if dynspec.attrs.get("tracking"):
            ra_val = dynspec.attrs["ra"]
            dec_val = dynspec.attrs["dec"]
            ax.set_title(
                f"{var} Dedispersed Dynamic Spectrum (tracked)\n"
                f"RA={ra_val:.4f}°, Dec={dec_val:.4f}°, pol={pol}, "
                f"DM={dm:.2f} pc/cm³ ({method})"
            )
        else:
            pixel_l = dynspec.attrs["pixel_l"]
            pixel_m = dynspec.attrs["pixel_m"]
            ax.set_title(
                f"{var} Dedispersed Dynamic Spectrum\n"
                f"l={pixel_l:+.4f}, m={pixel_m:+.4f}, pol={pol}, "
                f"DM={dm:.2f} pc/cm³ ({method})"
            )

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Interactive visualization (requires optional [visualization] deps)
    # ------------------------------------------------------------------

    def explore(self, *, max_size: int = 512) -> Any:
        """Launch an interactive exploration dashboard.

        Combines image, dynamic spectrum, and cutout explorers in a
        tabbed Panel layout. Requires the ``[visualization]`` optional
        dependencies (Panel, HoloViews, Bokeh).

        Install with::

            pip install 'ovro_lwa_portal[visualization]'

        Parameters
        ----------
        max_size : int, optional
            Maximum pixels per spatial side after downsampling. Lower
            values are faster but coarser. Default 512.

        Returns
        -------
        panel.Tabs
            Interactive tabbed dashboard.
        """
        from ovro_lwa_portal.viz import create_exploration_dashboard

        return create_exploration_dashboard(self._obj, max_size=max_size)

    def explore_image(self, *, max_size: int = 512, **kwargs: Any) -> Any:
        """Launch an interactive image explorer.

        Provides sliders for time, frequency, and polarization with
        live image updates. Requires ``[visualization]`` dependencies.

        Parameters
        ----------
        max_size : int, optional
            Maximum pixels per spatial side after downsampling. Lower
            values are faster but coarser. Default 512.
        **kwargs
            Passed to :class:`~ovro_lwa_portal.viz.explorers.ImageExplorer`.

        Returns
        -------
        panel.viewable.Viewable
            Interactive Panel layout.
        """
        from ovro_lwa_portal.viz import ImageExplorer

        return ImageExplorer(self._obj, max_size=max_size, **kwargs).panel()

    def explore_dynamic_spectrum(self, **kwargs: Any) -> Any:
        """Launch an interactive dynamic spectrum explorer.

        Displays a time-frequency waterfall with linked spectrum and
        light curve views. Requires ``[visualization]`` dependencies.

        Parameters
        ----------
        **kwargs
            Passed to :class:`~ovro_lwa_portal.viz.explorers.DynamicSpectrumExplorer`.

        Returns
        -------
        panel.viewable.Viewable
            Interactive Panel layout with linked views.
        """
        from ovro_lwa_portal.viz import DynamicSpectrumExplorer

        return DynamicSpectrumExplorer(self._obj, **kwargs).panel()

    def explore_sky(self, **kwargs: Any) -> Any:
        """Launch an interactive sky viewer with Aladin Lite.

        Overlays OVRO-LWA data on astronomical survey backgrounds
        (DSS, WISE, Planck, etc.) with real-time panning, zooming,
        and coordinate exploration. Requires ``[visualization]``
        dependencies and a WCS header (``fits_wcs_header`` attribute)
        in the dataset.

        Parameters
        ----------
        **kwargs
            Passed to :class:`~ovro_lwa_portal.viz.sky_viewer.SkyViewer`.

        Returns
        -------
        panel.viewable.Viewable
            Interactive Panel layout with Aladin sky viewer.

        Raises
        ------
        ValueError
            If the dataset does not contain a WCS header.
        """
        if not self.has_wcs:
            msg = (
                "explore_sky() requires a WCS header in the dataset "
                "(fits_wcs_header attribute). Use explore_image() for "
                "datasets without celestial coordinates."
            )
            raise ValueError(msg)
        from ovro_lwa_portal.viz import SkyViewer

        return SkyViewer(self._obj, **kwargs).panel()
