"""Shared pytest fixtures for ovro_lwa_portal tests."""

from __future__ import annotations

import importlib.util
import os
from typing import Final

import numpy as np
import pytest
import xarray as xr


def image_plane_correction_available() -> bool:
    """Return True if the optional ``image_plane_correction`` package is importable."""
    return importlib.util.find_spec("image_plane_correction") is not None


_SKIP_DEWARP_ORCHESTRATION_ON_CI: Final[bool] = (
    os.environ.get("GITHUB_ACTIONS") == "true" and not image_plane_correction_available()
)


skip_github_ci_without_image_plane_correction = pytest.mark.skipif(
    _SKIP_DEWARP_ORCHESTRATION_ON_CI,
    reason=(
        "GitHub Actions omits the optional image_plane_correction checkout; "
        "install the dewarp env locally to exercise these tests."
    ),
)


@pytest.fixture
def valid_ovro_dataset() -> xr.Dataset:
    """Create a valid OVRO-LWA dataset with all required dimensions and variables.

    Returns
    -------
    xr.Dataset
        A dataset with SKY variable and all required dimensions:
        time, frequency, polarization, l, m.
    """
    np.random.seed(42)
    beam_meta = np.empty((2, 3, 2, 3), dtype=np.float64)
    beam_meta[..., 0] = 0.02  # major (deg)
    beam_meta[..., 1] = 0.01  # minor (deg)
    beam_meta[..., 2] = 0.0  # position angle (deg)
    return xr.Dataset(
        data_vars={
            "SKY": (
                ["time", "frequency", "polarization", "l", "m"],
                np.random.rand(2, 3, 2, 50, 50) * 10,
            ),
            "BEAM": (
                ["time", "frequency", "polarization", "beam_param"],
                beam_meta,
            ),
        },
        coords={
            "time": [60000.0, 60000.1],  # MJD values
            "frequency": [46e6, 50e6, 54e6],  # Hz
            "polarization": [0, 1],
            "beam_param": ["major", "minor", "pa"],
            "l": np.linspace(-1, 1, 50),
            "m": np.linspace(-1, 1, 50),
        },
    )


@pytest.fixture
def valid_ovro_dataset_with_beam() -> xr.Dataset:
    """Create a valid OVRO-LWA dataset with both SKY and BEAM variables.

    Returns
    -------
    xr.Dataset
        A dataset with SKY and BEAM variables and all required dimensions.
    """
    np.random.seed(42)
    shape = (2, 3, 2, 50, 50)
    dims = ["time", "frequency", "polarization", "l", "m"]
    return xr.Dataset(
        data_vars={
            "SKY": (dims, np.random.rand(*shape) * 10),
            "BEAM": (dims, np.random.rand(*shape)),
        },
        coords={
            "time": [60000.0, 60000.1],
            "frequency": [46e6, 50e6, 54e6],
            "polarization": [0, 1],
            "l": np.linspace(-1, 1, 50),
            "m": np.linspace(-1, 1, 50),
        },
    )


@pytest.fixture
def dataset_missing_dimensions() -> xr.Dataset:
    """Create a dataset missing required OVRO-LWA dimensions.

    Returns
    -------
    xr.Dataset
        A dataset with SKY variable but wrong dimensions (x, y instead of l, m).
    """
    return xr.Dataset(
        data_vars={
            "SKY": (["x", "y"], np.random.rand(10, 10)),
        },
        coords={
            "x": np.arange(10),
            "y": np.arange(10),
        },
    )


@pytest.fixture
def dataset_missing_sky_variable() -> xr.Dataset:
    """Create a dataset with correct dimensions but missing SKY variable.

    Returns
    -------
    xr.Dataset
        A dataset with all required dimensions but OTHER variable instead of SKY.
    """
    return xr.Dataset(
        data_vars={
            "OTHER": (
                ["time", "frequency", "polarization", "l", "m"],
                np.random.rand(2, 3, 2, 10, 10),
            ),
        },
        coords={
            "time": [0, 1],
            "frequency": [1e6, 2e6, 3e6],
            "polarization": [0, 1],
            "l": np.linspace(-1, 1, 10),
            "m": np.linspace(-1, 1, 10),
        },
    )


@pytest.fixture
def valid_ovro_dataset_with_wcs() -> xr.Dataset:
    """Create a valid OVRO-LWA dataset with WCS coordinate information.

    Returns
    -------
    xr.Dataset
        A dataset with SKY variable and WCS header for coordinate transforms.
    """
    np.random.seed(42)

    # Simple FITS WCS header for a 50x50 image centered at RA=180, Dec=45
    wcs_header = """NAXIS   =                    2
NAXIS1  =                   50
NAXIS2  =                   50
CTYPE1  = 'RA---SIN'
CTYPE2  = 'DEC--SIN'
CRPIX1  =                 25.0
CRPIX2  =                 25.0
CRVAL1  =                180.0
CRVAL2  =                 45.0
CDELT1  =                 -1.0
CDELT2  =                  1.0
CUNIT1  = 'deg'
CUNIT2  = 'deg'
RADESYS = 'FK5'
EQUINOX =               2000.0"""

    ds = xr.Dataset(
        data_vars={
            "SKY": (
                ["time", "frequency", "polarization", "l", "m"],
                np.random.rand(2, 3, 2, 50, 50) * 10,
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

    # Add WCS header to variable attrs
    ds["SKY"].attrs["fits_wcs_header"] = wcs_header

    return ds


@pytest.fixture
def valid_ovro_dataset_with_tracking_wcs() -> xr.Dataset:
    """Create a dataset with WCS for per-time celestial coordinate tracking.

    The WCS phase center is at OVRO-LWA zenith (Dec=37.2339) with CRVAL1
    set to the LST at MJD 60000.0 at the OVRO-LWA longitude. The dataset
    has 10 time steps separated by 0.01 MJD (~14.4 minutes) and a 50x50 image
    to produce measurable pixel drift when tracking a source.
    """
    np.random.seed(42)

    n_times = 10
    t0 = 60000.0
    dt = 0.01  # ~14.4 minutes
    times = [t0 + i * dt for i in range(n_times)]

    # Compute LST at t0 for OVRO-LWA longitude to set CRVAL1
    # This makes the WCS phase center match the zenith at t0
    from astropy.time import Time
    from astropy import units as u

    t_ref = Time(t0, format="mjd", scale="utc")
    lst_deg = float(t_ref.sidereal_time("mean", longitude=-118.2817 * u.deg).deg)

    # SIN projection WCS with phase center at zenith
    wcs_header = f"""NAXIS   =                    2
NAXIS1  =                   50
NAXIS2  =                   50
CTYPE1  = 'RA---SIN'
CTYPE2  = 'DEC--SIN'
CRPIX1  =                 25.0
CRPIX2  =                 25.0
CRVAL1  =  {lst_deg:18.10f}
CRVAL2  =           37.2339000
CDELT1  =                 -3.6
CDELT2  =                  3.6
CUNIT1  = 'deg'
CUNIT2  = 'deg'
RADESYS = 'FK5'
EQUINOX =               2000.0"""

    beam_meta = np.empty((n_times, 3, 2, 3), dtype=np.float64)
    beam_meta[..., 0] = 0.02
    beam_meta[..., 1] = 0.01
    beam_meta[..., 2] = 0.0
    ds = xr.Dataset(
        data_vars={
            "SKY": (
                ["time", "frequency", "polarization", "l", "m"],
                np.random.rand(n_times, 3, 2, 50, 50) * 10,
            ),
            "BEAM": (
                ["time", "frequency", "polarization", "beam_param"],
                beam_meta,
            ),
        },
        coords={
            "time": times,
            "frequency": [46e6, 50e6, 54e6],
            "polarization": [0, 1],
            "beam_param": ["major", "minor", "pa"],
            "l": np.linspace(-1, 1, 50),
            "m": np.linspace(-1, 1, 50),
        },
    )

    ds["SKY"].attrs["fits_wcs_header"] = wcs_header

    return ds
