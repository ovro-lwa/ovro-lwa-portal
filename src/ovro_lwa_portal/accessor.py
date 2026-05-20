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
import warnings
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


def _read_wcs_header_str(
    ds: xr.Dataset,
    *,
    var: str = "SKY",
    time_idx: int = 0,
) -> str | None:
    """Return the persisted FITS WCS header string from *ds*, if any.

    Checks ``fits_wcs_header`` on the data variable and dataset attrs, then
    ``wcs_header_str`` (0-D scalar or 1-D per ``time`` from incremental Zarr).
    """
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
        raw = wcs_var.values
        if isinstance(raw, np.ndarray):
            raw = raw.item()
    elif wcs_var.ndim == 1 and "time" in wcs_var.dims:
        raw = wcs_var.isel(time=time_idx).values
        if isinstance(raw, np.ndarray) and raw.ndim == 0:
            raw = raw.item()
    else:
        raw_arr = np.asarray(wcs_var.values)
        if raw_arr.size == 0:
            return None
        raw = np.ravel(raw_arr)[time_idx]

    if isinstance(raw, (bytes, bytearray)) or type(raw).__name__ == "bytes_":
        return raw.decode("utf-8", errors="replace").rstrip("\x00")
    return str(raw).rstrip("\x00")


def _maybe_load(da: xr.DataArray) -> xr.DataArray:
    """Eagerly load a dask-backed DataArray if it is below the size threshold."""
    if hasattr(da, "nbytes") and da.nbytes < _EAGER_LOAD_THRESHOLD:
        if hasattr(da, "load"):
            da = da.load()
    return da


@contextlib.contextmanager
def _dask_progress(label: str = "Computing") -> Generator[None, None, None]:
    """Show a dask progress bar when dask is available.

    Falls back to a simple print when dask.diagnostics is unavailable.
    """
    try:
        from dask.diagnostics import ProgressBar

        with ProgressBar(dt=1.0, minimum=2.0):
            print(f"{label}...")  # noqa: T201
            yield
    except ImportError:
        yield


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
    """Evaluate a centred 2D Gaussian model on a pixel grid."""
    peak, widthx, widthy, background = params
    sx = max(float(widthx) / _FWHM_TO_SIGMA, 1e-6)
    sy = max(float(widthy) / _FWHM_TO_SIGMA, 1e-6)
    return background + peak * np.exp(-0.5 * ((x / sx) ** 2 + (y / sy) ** 2))


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


def _fit_spatial_gaussian(
    values: np.ndarray,
    *,
    default_fwhm: float = 3.0,
) -> tuple[float, float, float, float, float]:
    """Fit a centred 2D Gaussian to a spatial patch.

    The model is ``background + peak * exp(-0.5 * ((x/sx)**2 + (y/sy)**2))`` with
    the peak centred on the patch centre.  ``widthx`` and ``widthy`` are returned
    as full width at half maximum (FWHM) in pixels.

    Initial guesses use ``peak = max(patch)``, ``widthx = widthy = default_fwhm``,
    and ``background = min(patch)``.
    """
    nan5 = (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        msg = f"Expected 2D patch, got shape {arr.shape}"
        raise ValueError(msg)

    ny, nx = arr.shape
    if ny < 2 or nx < 2:
        return nan5

    if not np.any(np.isfinite(arr)):
        return nan5

    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    yy, xx = np.indices((ny, nx))
    y = yy - cy
    x = xx - cx

    peak0 = float(np.nanmax(arr))
    bg0 = float(np.nanmin(arr))
    width0 = float(default_fwhm)
    if not np.isfinite(peak0):
        return nan5

    def residual(params: np.ndarray) -> np.ndarray:
        return (_gaussian_predict(params, x, y) - arr).ravel()

    x0 = np.array([peak0, width0, width0, bg0], dtype=np.float64)
    lower = np.array([0.0, 0.5, 0.5, -np.inf], dtype=np.float64)
    upper = np.array([np.inf, max(nx, ny) * 4.0, max(nx, ny) * 4.0, np.inf], dtype=np.float64)

    try:
        res = least_squares(
            residual,
            x0=x0,
            bounds=(lower, upper),
            method="trf",
            ftol=1e-8,
            xtol=1e-8,
            max_nfev=300,
        )
    except (ValueError, RuntimeError):
        return nan5

    if not res.success:
        return nan5

    peak, widthx, widthy, background = res.x
    predicted = _gaussian_predict(res.x, x, y)
    chi2_red = _reduced_chi_squared(arr, predicted)
    return float(peak), float(widthx), float(widthy), float(background), chi2_red


def _fit_patch_cube_gaussian(
    patch: np.ndarray,
    *,
    default_fwhm: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a 2D Gaussian on each frequency plane in ``(frequency, l, m)``."""
    patch_arr = np.asarray(patch)
    if patch_arr.ndim != 3:
        msg = f"Expected patch with shape (frequency, l, m), got {patch_arr.shape}"
        raise ValueError(msg)

    peaks = []
    widthxs = []
    widthys = []
    backgrounds = []
    chi2_reds = []
    for fi in range(patch_arr.shape[0]):
        peak, widthx, widthy, background, chi2_red = _fit_spatial_gaussian(
            patch_arr[fi],
            default_fwhm=default_fwhm,
        )
        peaks.append(peak)
        widthxs.append(widthx)
        widthys.append(widthy)
        backgrounds.append(background)
        chi2_reds.append(chi2_red)

    return (
        np.array(peaks, dtype=np.float64),
        np.array(widthxs, dtype=np.float64),
        np.array(widthys, dtype=np.float64),
        np.array(backgrounds, dtype=np.float64),
        np.array(chi2_reds, dtype=np.float64),
    )


def _mask_patch_fit_by_chi2(
    peaks: np.ndarray,
    widthxs: np.ndarray,
    widthys: np.ndarray,
    backgrounds: np.ndarray,
    chi2_red: np.ndarray,
    *,
    max_reduced_chi_squared: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return peaks, widthxs, widthys, backgrounds


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
    radius: int
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


@dataclass
class PatchFitResult:
    """Gaussian fit parameters on a tracked patch for each time/frequency cell.

    Returned by :meth:`RadportAccessor.patch_fit`.  Each parameter map has
    dimensions ``(time, frequency)``.  ``widthx`` and ``widthy`` are full width
    at half maximum in pixels.  ``reduced_chi_squared_map`` stores the fit
    quality; parameter maps are NaN where reduced chi-squared exceeds
    ``max_reduced_chi_squared``.
    """

    peak_map: xr.DataArray
    widthx_map: xr.DataArray
    widthy_map: xr.DataArray
    background_map: xr.DataArray
    reduced_chi_squared_map: xr.DataArray
    radius: int
    default_fwhm: float
    max_reduced_chi_squared: float
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

    def _compute_pixel_track(
        self,
        ra: float,
        dec: float,
        observatory: Any = None,
        *,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        pol: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-time-step pixel indices for a fixed (RA, Dec).

        For each dataset time index, uses :meth:`coords_to_pixel` (via
        :meth:`_compute_pixel_at_time`) so pixel selection matches all other
        extraction APIs, except when per-time ``right_ascension`` /
        ``declination`` grids allow a time-chunked batched argmin on a
        ``(time, l, m)`` grid (same horizon and distance rules as
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

        self._ensure_lst_deg_vector_cached(observatory)

        if freq_mhz is not None:
            fi = int(self.nearest_freq_idx(freq_mhz))
        elif freq_idx is not None:
            fi = int(freq_idx)
        else:
            fi = 0

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
                if not np.any(visible):
                    warnings.warn(
                        f"Source (RA={ra}°, Dec={dec}°) is never above the horizon "
                        f"during this observation. All output values will be NaN.",
                        stacklevel=3,
                    )
                return l_indices, m_indices, visible

        for ti in range(n_times):
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

        if not np.any(visible):
            warnings.warn(
                f"Source (RA={ra}°, Dec={dec}°) is never above the horizon "
                f"during this observation. All output values will be NaN.",
                stacklevel=3,
            )

        return l_indices, m_indices, visible

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

        When the dataset provides ``right_ascension`` and ``declination``
        coordinates that include a ``time`` dimension (or the cube has only one
        time step), this method finds the nearest sky pixel by minimizing
        angular distance on the stored RA/Dec grids. If those coordinates also
        carry ``frequency`` and/or ``polarization`` dimensions, the slice at
        ``freq_idx`` / ``pol`` is taken first so the minimization is only over
        ``(l, m)``. Otherwise it falls back to the closed-form SIN projection
        using mean sidereal time at the dataset timestamp (the analytical branch
        used when :meth:`coords_to_pixel` does not use stored RA/Dec grids).

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
        )

        if isinstance(result, tuple) and len(result) == 2:
            # Fixed pixel path (l/m) — single pixel across all times/freqs.
            # Eagerly load small results to avoid dask graph overhead.
            l_idx, m_idx = result
            da = _maybe_load(
                self._obj[var].isel(l=l_idx, m=m_idx, polarization=pol)
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

        n_times = self._obj.sizes["time"]
        n_freqs = self._obj.sizes["frequency"]

        time_coords = self._obj.coords["time"].values
        freq_coords = self._obj.coords["frequency"].values

        # Build output array, NaN-filled
        out = np.full((n_times, n_freqs), np.nan)

        # Extract per-time pixel values.
        # Each time step may map to a different (l, m) pixel, so we
        # select the exact pixel per time step rather than loading the
        # full spatial grid (which would be e.g. 4096x4096 per step).
        vis_mask = visible
        if np.any(vis_mask):
            vis_times = np.where(vis_mask)[0]
            vis_l = l_indices[vis_mask]
            vis_m = m_indices[vis_mask]

            # Build per-time-step pixel selections (each is shape n_freqs)
            pixel_arrays = [
                data_var.isel(time=int(t), l=int(li), m=int(mi))
                for t, li, mi in zip(vis_times, vis_l, vis_m)
            ]

            # Compute all pixels in one pass — dask deduplicates
            # shared chunk reads automatically.
            if hasattr(data_var, "chunks") and data_var.chunks is not None:
                import dask

                with _dask_progress("Extracting tracked pixels"):
                    results = dask.compute(*pixel_arrays)
            else:
                results = [p.values for p in pixel_arrays]

            for i, ti in enumerate(vis_times):
                out[ti] = np.asarray(results[i])

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
        radius: int,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Load ``(frequency, l, m)`` patches for visible tracked time steps.

        Uses one batched ``dask.compute`` call to avoid per-iteration scheduler
        overhead on large time series.
        """
        data_var = self._obj[var].isel(polarization=pol)
        n_l = int(self._obj.sizes["l"])
        n_m = int(self._obj.sizes["m"])

        vis_times = np.asarray(np.where(visible)[0], dtype=int)
        vis_l = np.asarray(l_indices, dtype=int)[visible]
        vis_m = np.asarray(m_indices, dtype=int)[visible]

        patch_arrays = []
        for t, li, mi in zip(vis_times, vis_l, vis_m, strict=True):
            l_sl = slice(max(0, int(li) - radius), min(n_l, int(li) + radius + 1))
            m_sl = slice(max(0, int(mi) - radius), min(n_m, int(mi) + radius + 1))
            patch_arrays.append(data_var.isel(time=int(t), l=l_sl, m=m_sl))

        if not patch_arrays:
            return vis_times, []

        if hasattr(data_var, "chunks") and data_var.chunks is not None:
            import dask

            with _dask_progress("Extracting tracked patches"):
                results = dask.compute(*patch_arrays)
        else:
            results = [np.asarray(p.values) for p in patch_arrays]

        return vis_times, [np.asarray(r) for r in results]

    def patch_statistic(
        self,
        *,
        ra: float | None = None,
        dec: float | None = None,
        l: float | None = None,
        m: float | None = None,
        radius: int = 5,
        statistic: PatchStatisticName = "std",
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        observatory: Any = None,
        threshold: float | None = None,
        comparison: PatchStatisticComparison = "gt",
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
        radius : int, default 5
            Patch half-width in pixels.  The patch spans
            ``[center - radius, center + radius]`` along ``l`` and ``m``.
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
        if radius < 0:
            msg = f"radius must be non-negative, got {radius}"
            raise ValueError(msg)

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

        vis_times, patches = self._extract_tracked_patch_cubes(
            l_indices=l_indices,
            m_indices=m_indices,
            visible=visible,
            var=var,
            pol=pol,
            radius=radius,
        )
        for ti, patch in zip(vis_times, patches, strict=True):
            stat_values[int(ti)] = _reduce_patch_cube_statistics(patch, statistic)

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
            "radius": radius,
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
            radius=radius,
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
        radius: int = 5,
        default_fwhm: float = 3.0,
        max_reduced_chi_squared: float = 3.0,
        var: Literal["SKY", "BEAM"] = "SKY",
        pol: int = 0,
        freq_idx: int | None = None,
        freq_mhz: float | None = None,
        observatory: Any = None,
    ) -> PatchFitResult:
        """Fit a 2D Gaussian to a tracked patch for each time/frequency cell.

        For each time step the patch is centred on the pixel nearest to the
        given celestial or direction-cosine coordinates.  A centred Gaussian
        with parameters ``peak``, ``widthx``, ``widthy``, and ``background`` is
        fit independently on every frequency channel, producing four
        ``(time, frequency)`` maps plus a reduced chi-squared quality map.

        The Gaussian is centred on the patch centre.  ``widthx`` and ``widthy``
        are full width at half maximum in pixels.  Initial guesses use
        ``peak = max(patch)``, ``widthx = widthy = default_fwhm`` (3 pixels by
        default), and ``background = min(patch)``.

        When reduced chi-squared exceeds ``max_reduced_chi_squared`` (default
        3), the fit parameters for that cell are set to NaN.  The reduced
        chi-squared value is always retained in ``reduced_chi_squared_map``.

        Parameters
        ----------
        ra, dec : float, optional
            Celestial coordinates in degrees (FK5/J2000).  Requires tracking
            across time when both are provided.
        l, m : float, optional
            Direction-cosine coordinates for a fixed patch centre.
        radius : int, default 5
            Patch half-width in pixels.  The patch spans
            ``[center - radius, center + radius]`` along ``l`` and ``m``.
        default_fwhm : float, default 3.0
            Initial full width at half maximum in pixels for ``widthx`` and
            ``widthy``.
        max_reduced_chi_squared : float, default 3.0
            Maximum acceptable reduced chi-squared.  Cells above this threshold
            have ``peak``, ``widthx``, ``widthy``, and ``background`` set to
            NaN.
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to analyse.
        pol : int, default 0
            Polarization index.
        freq_idx, freq_mhz : optional
            Passed to coordinate tracking for RA/Dec pixel mapping.
        observatory : astropy.coordinates.EarthLocation, optional
            Observatory location for RA/Dec tracking.

        Returns
        -------
        PatchFitResult
            Container with parameter maps, ``reduced_chi_squared_map``, and
            ``max_reduced_chi_squared``.

        Examples
        --------
        >>> result = ds.radport.patch_fit(ra=299.868, dec=40.734, radius=5)
        >>> peaks = result.peak_map
        """
        if var not in self._obj.data_vars:
            available = sorted(self._obj.data_vars)
            raise ValueError(f"Variable '{var}' not found. Available: {available}")
        if radius < 0:
            msg = f"radius must be non-negative, got {radius}"
            raise ValueError(msg)
        if default_fwhm <= 0:
            msg = f"default_fwhm must be positive, got {default_fwhm}"
            raise ValueError(msg)
        if max_reduced_chi_squared <= 0:
            msg = f"max_reduced_chi_squared must be positive, got {max_reduced_chi_squared}"
            raise ValueError(msg)

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
        widthx_values = np.full((n_times, n_freqs), np.nan, dtype=np.float64)
        widthy_values = np.full((n_times, n_freqs), np.nan, dtype=np.float64)
        background_values = np.full((n_times, n_freqs), np.nan, dtype=np.float64)
        chi2_values = np.full((n_times, n_freqs), np.nan, dtype=np.float64)
        tracking = ra is not None and dec is not None

        if isinstance(resolved, tuple) and len(resolved) == 2:
            l_idx, m_idx = resolved
            visible = np.ones(n_times, dtype=bool)
            l_indices = np.full(n_times, int(l_idx), dtype=int)
            m_indices = np.full(n_times, int(m_idx), dtype=int)
        else:
            l_indices, m_indices, visible = resolved

        vis_times, patches = self._extract_tracked_patch_cubes(
            l_indices=l_indices,
            m_indices=m_indices,
            visible=visible,
            var=var,
            pol=pol,
            radius=radius,
        )
        for ti, patch in zip(vis_times, patches, strict=True):
            peaks, widthxs, widthys, backgrounds, chi2_red = _fit_patch_cube_gaussian(
                patch,
                default_fwhm=default_fwhm,
            )
            peaks, widthxs, widthys, backgrounds = _mask_patch_fit_by_chi2(
                peaks,
                widthxs,
                widthys,
                backgrounds,
                chi2_red,
                max_reduced_chi_squared=max_reduced_chi_squared,
            )
            ti_int = int(ti)
            peak_values[ti_int] = peaks
            widthx_values[ti_int] = widthxs
            widthy_values[ti_int] = widthys
            background_values[ti_int] = backgrounds
            chi2_values[ti_int] = chi2_red

        attrs: dict[str, Any] = {
            "radius": radius,
            "default_fwhm": default_fwhm,
            "max_reduced_chi_squared": max_reduced_chi_squared,
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

        return PatchFitResult(
            peak_map=peak_map,
            widthx_map=widthx_map,
            widthy_map=widthy_map,
            background_map=background_map,
            reduced_chi_squared_map=reduced_chi_squared_map,
            radius=radius,
            default_fwhm=default_fwhm,
            max_reduced_chi_squared=max_reduced_chi_squared,
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
            vis_times = np.where(vis_mask)[0]
            vis_l = l_indices[vis_mask]
            vis_m = m_indices[vis_mask]

            # Select the exact pixel per time step rather than loading
            # the full spatial grid (which can be e.g. 4096x4096).
            pixel_arrays = [
                data_var.isel(time=int(t), l=int(li), m=int(mi))
                for t, li, mi in zip(vis_times, vis_l, vis_m)
            ]

            if hasattr(data_var, "chunks") and data_var.chunks is not None:
                import dask

                with _dask_progress("Extracting tracked pixels"):
                    results = dask.compute(*pixel_arrays)
            else:
                results = [p.values for p in pixel_arrays]

            for i, ti in enumerate(vis_times):
                out[ti] = float(results[i])

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
    ):
        """Get WCS object from the dataset.

        Parameters
        ----------
        var : {'SKY', 'BEAM'}, default 'SKY'
            Data variable to get WCS from (checks attrs first).
        time_idx : int, default 0
            Time index when ``wcs_header_str`` is stored per time step (common
            after incremental Zarr writes).

        Returns
        -------
        astropy.wcs.WCS
            The WCS object for coordinate transformations.

        Raises
        ------
        ImportError
            If astropy is not installed.
        ValueError
            If no WCS header is found in the dataset.
        """
        try:
            from astropy.io.fits import Header
            from astropy.wcs import WCS
        except ImportError as e:
            raise ImportError(
                "astropy is required for WCS functionality. "
                "Install with: pip install astropy"
            ) from e

        hdr_str = _read_wcs_header_str(self._obj, var=var, time_idx=time_idx)
        if not hdr_str:
            raise ValueError(
                "No WCS header found in dataset. Expected 'fits_wcs_header' "
                "attribute on variable/dataset or 'wcs_header_str' variable."
            )

        return WCS(Header.fromstring(hdr_str, sep="\n"))

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

        Returns the FK5/J2000 position whose direction cosines ``(l, m)`` at
        that pixel match the same mean sidereal time and SIN geometry as
        :meth:`coords_to_pixel` — i.e. the sky coordinate of that pixel at the
        chosen dataset time as the field rotates. A ``fits_wcs_header`` is only
        used as a numerical seed for the inverse solve, not as the returned
        static WCS solution.

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

        wcs = self._get_wcs()

        from astropy.coordinates import EarthLocation

        if time_mjd is not None:
            ti = self.nearest_time_idx(time_mjd)
        else:
            ti = int(time_idx)

        n_time = self._obj.sizes["time"]
        if not (0 <= ti < n_time):
            raise ValueError(f"time index {ti} out of bounds [0, {n_time})")

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
