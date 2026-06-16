"""Jupiter flux review helpers for phase2 QA Zarr stores."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
from astropy.coordinates import SkyCoord, get_body
from astropy.time import Time
from astropy.utils.iers import conf as iers_conf
from bokeh.models import LinearColorMapper
from bokeh.palettes import Inferno256

from ovro_lwa_portal.viz.pipeline_qa import PipelineQAConfig
from ovro_lwa_portal.viz.source_review_data import (
    _PROGRESS_STAGE_LABELS,
    lst_hours_for_dataset,
)

__all__ = [
    "JupiterLoad",
    "format_flux_hover",
    "jupiter_at_observation_start",
    "jupiter_color_mapper",
    "jupiter_flux_map",
    "list_phase2_i_qa_zarrs",
    "zarr_path_to_day",
]


@dataclass(frozen=True)
class JupiterLoad:
    """Result of opening one Jupiter QA Zarr and extracting a flux map."""

    dataset: xr.Dataset
    dynspec: xr.DataArray
    jupiter: SkyCoord
    lst_hours: np.ndarray
    freq_mhz: np.ndarray
    patch_fit_result: object | None


def list_phase2_i_qa_zarrs(config: PipelineQAConfig) -> list[Path]:
    """Return sorted Stokes I phase2 QA Zarr paths under ``config.zarr_root``."""
    pattern = f"{config.i_qa_zarr_stem}-*.zarr"
    return sorted(config.zarr_root.glob(pattern))


def zarr_path_to_day(path: Path, *, stem: str) -> str:
    """Parse ``YYYY-MM-DD`` from a QA Zarr directory name."""
    day_tag = path.name.removeprefix(f"{stem}-").removesuffix(".zarr")
    if len(day_tag) != 8 or not day_tag.isdigit():
        return path.name
    return f"{day_tag[:4]}-{day_tag[4:6]}-{day_tag[6:8]}"


def jupiter_at_observation_start(ds: xr.Dataset) -> SkyCoord:
    """Jupiter FK5 coordinates at the first time sample in the dataset."""
    mjd = float(np.asarray(ds.coords["time"].values, dtype=np.float64)[0])
    orig = iers_conf.auto_download
    try:
        iers_conf.auto_download = False
        t0 = Time(mjd, format="mjd", scale="utc")
        return get_body("jupiter", t0)
    finally:
        iers_conf.auto_download = orig


def jupiter_flux_map(
    ds: xr.Dataset,
    jupiter: SkyCoord,
    *,
    method: str = "dynamic_spectrum",
    patch_fit_scale: float = 3.0,
    patch_fit_max_reduced_chi_squared: float = 3.0,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> tuple[xr.DataArray, object | None]:
    """Flux map and optional :class:`~ovro_lwa_portal.accessor.PatchFitResult`."""
    ra = float(jupiter.ra.deg)
    dec = float(jupiter.dec.deg)
    if method == "dynamic_spectrum":
        flux = ds.radport.dynamic_spectrum(
            ra=ra, dec=dec, progress_callback=progress_callback
        )
        flux.attrs["flux_method"] = "dynamic_spectrum"
        return flux, None
    if method == "patch_max":
        stat = ds.radport.patch_statistic(
            ra=ra,
            dec=dec,
            statistic="max",
            scale=patch_fit_scale,
            progress_callback=progress_callback,
        )
        flux = stat.stat_map
        flux.name = "flux"
        flux.attrs["flux_method"] = "patch_max"
        return flux, None
    if method == "patch_fit":
        fit = ds.radport.patch_fit(
            ra=ra,
            dec=dec,
            scale=patch_fit_scale,
            max_reduced_chi_squared=patch_fit_max_reduced_chi_squared,
            allow_position_offset=True,
            progress_callback=progress_callback,
        )
        flux = fit.peak_map
        flux.name = "flux"
        flux.attrs["flux_method"] = "patch_fit"
        return flux, fit
    msg = (
        f"Unknown flux method {method!r}; expected "
        "'dynamic_spectrum', 'patch_fit', or 'patch_max'"
    )
    raise ValueError(msg)


def jupiter_color_mapper(values: np.ndarray) -> LinearColorMapper:
    """Inferno colormap with percentile stretch (Jupiter dynspec convention)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return LinearColorMapper(palette=Inferno256, low=0.0, high=1.0)
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    return LinearColorMapper(
        palette=Inferno256,
        low=float(lo),
        high=float(hi),
        nan_color="#9e9e9e",
    )


def format_flux_hover(values: np.ndarray) -> list[str]:
    """One hover label per (time, freq) cell; non-finite flux reads as n/a."""
    return [f"{float(v):.3g}" if np.isfinite(v) else "n/a" for v in values.ravel()]
