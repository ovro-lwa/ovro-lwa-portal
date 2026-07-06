"""Data helpers for the source review Panel app.

Pure functions and heatmap computation extracted from ``source_review.ipynb``
so they can be unit tested without Panel, Bokeh, or a live notebook comm.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from astropy.coordinates import SkyCoord

from ovro_lwa_portal.accessor import PatchFitCellResult, format_radec_sexagesimal
from bokeh.models import LinearColorMapper
from bokeh.palettes import Magma256

HEATMAP_METHOD_OPTIONS = [
    "dynamic_spectrum",
    "patch_max",
    "mad",
    "std",
    "mean",
    "min",
]

_STOKES_FITS_VALUE: dict[str, int] = {"I": 1, "V": 4}


def available_stokes_labels(dataset: xr.Dataset) -> list[str]:
    """Return ``I`` / ``V`` labels present on the dataset ``polarization`` coordinate."""
    if "polarization" not in dataset.dims:
        return []
    pol_values = np.asarray(dataset.coords["polarization"].values).ravel().astype(int)
    labels: list[str] = []
    for val in pol_values:
        for label, fits_val in _STOKES_FITS_VALUE.items():
            if int(val) == int(fits_val) and label not in labels:
                labels.append(label)
    return labels


def polarization_index_for_stokes(dataset: xr.Dataset, stokes: str) -> int:
    """Map a Stokes label to the dataset ``polarization`` dimension index."""
    label = str(stokes).strip().upper()
    if label not in _STOKES_FITS_VALUE:
        msg = f"Unknown Stokes label {stokes!r}; expected one of {sorted(_STOKES_FITS_VALUE)}"
        raise ValueError(msg)
    target = int(_STOKES_FITS_VALUE[label])
    pol_values = np.asarray(dataset.coords["polarization"].values).ravel().astype(int)
    matches = np.flatnonzero(pol_values == target)
    if matches.size == 0:
        msg = (
            f"Dataset polarization coordinate {list(pol_values)} "
            f"does not include Stokes {label} (FITS value {target})."
        )
        raise ValueError(msg)
    return int(matches[0])

def filter_known_source_names(text: str, names: list[str]) -> list[str]:
    """``includes`` match on known names when input starts with a letter."""
    stripped = text.lstrip()
    if not stripped or not stripped[0].isalpha():
        return []
    query = stripped.casefold()
    return [name for name in names if query in name.casefold()]


def resolve_known_sources_path(path: Path) -> Path | None:
    """Find ``known_sources.yaml`` relative to cwd or ``notebooks/``."""
    for candidate in (path, Path.cwd() / path, Path.cwd() / "notebooks" / path.name):
        if candidate.is_file():
            return candidate.resolve()
    return None


def load_known_sources(path: Path) -> list[str]:
    """Load source name completion labels from a YAML file."""
    resolved = resolve_known_sources_path(path)
    if resolved is None:
        return []

    with resolved.open(encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}

    names: list[str] = []
    for entry in payload.get("sources") or []:
        if isinstance(entry, str):
            name = entry.strip()
        else:
            name = str(entry.get("name", "")).strip()
        if name:
            names.append(name)
    names.sort(key=str.casefold)
    return names


def build_source_from_coordinate(label: str, coord: SkyCoord) -> dict:
    """Build a source record from a resolved sky position."""
    gal = coord.galactic
    name = label if len(label) <= 48 else f"{label[:45]}…"
    return {
        "name": name,
        "l": float(gal.l.deg),
        "b": float(gal.b.deg),
        "ra": float(coord.ra.deg),
        "dec": float(coord.dec.deg),
    }


def lst_hours_for_dataset(ds: xr.Dataset) -> np.ndarray:
    """Mean local sidereal time (hours) for each dataset time sample."""
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    from astropy.utils.iers import conf as iers_conf

    observatory = EarthLocation.of_site("ovro")
    mjd = np.asarray(ds.coords["time"].values, dtype=np.float64)
    orig = iers_conf.auto_download
    try:
        iers_conf.auto_download = False
        times = Time(mjd, format="mjd", scale="utc")
        lst_deg = np.asarray(times.sidereal_time("mean", longitude=observatory.lon).deg)
    finally:
        iers_conf.auto_download = orig
    return np.mod(lst_deg / 15.0, 24.0)


def first_valid_sky_slice(
    dataset: xr.Dataset,
    freq_idx: int | None = None,
    *,
    pol: int = 0,
) -> tuple[int, int]:
    """First time index with finite SKY at the image centre."""
    fi = dataset.sizes["frequency"] // 2 if freq_idx is None else int(freq_idx)
    center = dataset.sizes["l"] // 2
    ts = dataset["SKY"].isel(polarization=int(pol), frequency=fi, l=center, m=center)
    data = ts.data
    vals = np.asarray(data.compute() if hasattr(data, "compute") else data)
    valid = np.flatnonzero(np.isfinite(vals))
    if valid.size == 0:
        raise ValueError("No finite SKY data found at the image center.")
    return int(valid[0]), fi


@dataclass
class HeatmapLoad:
    """Values and optional accessor results for one source/method pair."""

    values: np.ndarray
    patch_fit_result: object | None = None
    patch_stat_result: object | None = None


_PROGRESS_STAGE_LABELS = {
    "track": "Pixel track",
    "extract": "Pixel I/O",
    "reduce": "Statistics",
    "fit": "Patch fit",
}


def compute_source_heatmap(
    dataset: xr.Dataset,
    src: dict,
    *,
    method: str,
    scale: float,
    patch_fit_max_reduced_chi_squared: float,
    pol: int = 0,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> HeatmapLoad:
    """Build the (time, frequency) array used to fill the heatmap."""
    ra = float(src["ra"])
    dec = float(src["dec"])
    if method == "dynamic_spectrum":
        da = dataset.radport.dynamic_spectrum(
            ra=ra,
            dec=dec,
            pol=int(pol),
            progress_callback=progress_callback,
        )
        return HeatmapLoad(np.asarray(da.values, dtype=np.float64))
    if method == "patch_fit":
        msg = (
            "Full-cube patch_fit is no longer a heatmap method; use the Fit overlay "
            "button in source review or dataset.radport.patch_fit_cell() / patch_fit()."
        )
        raise ValueError(msg)
    if method == "patch_max":
        result = dataset.radport.patch_statistic(
            ra=ra,
            dec=dec,
            statistic="max",
            scale=scale,
            pol=int(pol),
            progress_callback=progress_callback,
        )
        return HeatmapLoad(np.asarray(result.stat_map.values, dtype=np.float64), patch_stat_result=result)
    if method in ("mad", "std", "mean", "min"):
        result = dataset.radport.patch_statistic(
            ra=ra,
            dec=dec,
            statistic=method,
            scale=scale,
            pol=int(pol),
            progress_callback=progress_callback,
        )
        return HeatmapLoad(np.asarray(result.stat_map.values, dtype=np.float64), patch_stat_result=result)
    msg = f"Unknown heatmap method {method!r}; expected one of {HEATMAP_METHOD_OPTIONS}"
    raise ValueError(msg)


def compute_overlay_patch_fit(
    dataset: xr.Dataset,
    src: dict,
    *,
    time_idx: int,
    freq_idx: int,
    scale: float,
    patch_fit_max_reduced_chi_squared: float,
    pol: int = 0,
) -> PatchFitCellResult:
    """Fit a Gaussian patch on the overlay cell at ``(time_idx, freq_idx)``."""
    return dataset.radport.patch_fit_cell(
        time_idx=int(time_idx),
        frequency_idx=int(freq_idx),
        ra=float(src["ra"]),
        dec=float(src["dec"]),
        scale=float(scale),
        max_reduced_chi_squared=float(patch_fit_max_reduced_chi_squared),
        allow_position_offset=True,
        pol=int(pol),
    )


def diagnose_heatmap_coverage(
    dataset: xr.Dataset,
    src: dict,
    values: np.ndarray,
    *,
    method: str,
    patch_fit_max_reduced_chi_squared: float,
) -> str:
    """Explain missing (NaN) heatmap cells for logging in the review UI."""
    ra = float(src["ra"])
    dec = float(src["dec"])
    n_times, n_freqs = values.shape
    finite = np.isfinite(values)
    n_finite = int(finite.sum())
    if n_finite == finite.size:
        return ""

    lines: list[str] = []
    try:
        _li, _mi, visible = dataset.radport._compute_pixel_track(ra, dec)
        n_visible = int(np.sum(visible))
        lines.append(
            f"Sky footprint: source visible on {n_visible}/{n_times} time steps "
            f"in this dataset (gray = not in FOV or no finite data at the tracked pixel/patch)."
        )
        if n_visible == 0:
            lines.append(
                "The target position never falls inside this snapshot's image — "
                "common when the Zarr pointing differs from the requested sky position."
            )
        elif n_visible < n_times:
            lines.append(
                f"{n_times - n_visible} time step(s) are off-field; those rows stay gray."
            )
    except Exception as exc:
        lines.append(f"Could not evaluate sky footprint: {exc}")

    if n_finite and n_finite < finite.size:
        lines.append(
            f"Finite heatmap cells: {n_finite}/{finite.size} "
            f"({100.0 * n_finite / finite.size:.1f}%)."
        )
    elif n_finite == 0:
        lines.append("No finite heatmap cells.")

    return " ".join(lines)


def _heatmap_index_from_coord(coord: float, n: int) -> int:
    if n <= 0:
        return 0
    return int(np.clip(int(np.floor(float(coord))), 0, n - 1))


def _format_lst_hour_label(lst_hour: float) -> str:
    hour = int(round(float(lst_hour))) % 24
    return f"{hour:02d}h"


def calendar_mmdd_labels_for_time_coord(time_values: np.ndarray) -> np.ndarray:
    """UTC calendar ``MM-DD`` label for each sample in a dataset ``time`` coordinate."""
    from astropy.time import Time

    tv = np.asarray(time_values)
    if np.issubdtype(tv.dtype, np.datetime64):
        times = Time(tv, format="datetime64")
    else:
        times = Time(np.asarray(tv, dtype=np.float64), format="mjd", scale="utc")
    return np.array([t.isot[5:10] for t in times])


def format_heatmap_time_axis_label(
    time_values: np.ndarray,
    time_idx: int,
    lst_hours: np.ndarray,
    *,
    day_labels: np.ndarray | None = None,
) -> str:
    """Compact heatmap tick: calendar day + LST hour (disambiguates multi-day stores)."""
    lst = _format_lst_hour_label(float(lst_hours[int(time_idx)]))
    if day_labels is not None:
        day = str(day_labels[int(time_idx)])
    else:
        day = calendar_mmdd_labels_for_time_coord(time_values)[int(time_idx)]
    return f"{day} {lst}"


def _format_scalar_hover(value: float, *, fmt: str = ".3g") -> str:
    if np.isfinite(value):
        return format(float(value), fmt)
    return "n/a"


def _color_mapper(values: np.ndarray, *, palette=Magma256) -> LinearColorMapper:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return LinearColorMapper(palette=palette, low=0.0, high=1.0, nan_color="#9e9e9e")
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    return LinearColorMapper(
        palette=palette,
        low=float(lo),
        high=float(hi),
        nan_color="#9e9e9e",
    )


def _row_hover(arr: np.ndarray) -> list[str]:
    return [_format_scalar_hover(float(v)) for v in arr.ravel()]


def _patch_fit_hover_columns(fit: object) -> dict[str, list[str]]:
    """Pre-formatted Bokeh hover fields for patch-fit (same as Jupiter notebook)."""
    chi2 = np.asarray(fit.reduced_chi_squared_map.values, dtype=np.float64)
    peak = np.asarray(fit.peak_map.values, dtype=np.float64)
    x_off = np.asarray(fit.x_offset_map.values, dtype=np.float64)
    y_off = np.asarray(fit.y_offset_map.values, dtype=np.float64)
    pmax = np.asarray(fit.patch_max_map.values, dtype=np.float64)
    accepted = np.asarray(fit.fit_accepted_map.values, dtype=bool)
    peak_ra, peak_dec = fit.peak_radec_maps()
    ra = np.asarray(peak_ra.values, dtype=np.float64)
    dec = np.asarray(peak_dec.values, dtype=np.float64)

    peak_ra_display: list[str] = []
    peak_dec_display: list[str] = []
    for r, d in zip(ra.ravel(), dec.ravel(), strict=True):
        ra_s, dec_s = format_radec_sexagesimal(float(r), float(d))
        peak_ra_display.append(ra_s)
        peak_dec_display.append(dec_s)

    return {
        "chi2_display": _row_hover(chi2),
        "peak_ra_display": peak_ra_display,
        "peak_dec_display": peak_dec_display,
        "offset_display": [
            (
                f"({x:.2f}, {y:.2f})"
                if np.isfinite(x) and np.isfinite(y)
                else "n/a"
            )
            for x, y in zip(x_off.ravel(), y_off.ravel(), strict=True)
        ],
        "fit_accepted_display": ["yes" if a else "no" for a in accepted.ravel()],
        "patch_max_display": _row_hover(pmax),
        "peak_flux_display": [
            f"{v:.3g} (masked)" if not np.isfinite(v) else f"{v:.3g}"
            for v in peak.ravel()
        ],
    }


def _format_patch_fit_diagnostics(fit: object, time_idx: int, freq_idx: int) -> str:
    diag = fit.cell_diagnostics(time_idx=time_idx, frequency_idx=freq_idx)
    accepted = "yes" if diag["fit_accepted"] else "no (χ² above cut)"
    peak = diag["peak"]
    peak_s = f"{peak:.3g}" if np.isfinite(peak) else "n/a (masked)"
    return (
        f"**Fit overlay** t={time_idx} f={freq_idx}: accepted={accepted}, "
        f"χ²_red={diag['reduced_chi_squared']:.3g}, peak={peak_s} Jy, "
        f"peak RA/Dec=({diag['peak_ra']}, {diag['peak_dec']}), "
        f"offset=({diag['x_offset_pixels']:.2f}, {diag['y_offset_pixels']:.2f}) px, "
        f"patch_max={diag['patch_max']:.3g} Jy"
    )
