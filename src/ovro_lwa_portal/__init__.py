"""OVRO-LWA Portal: Radio astronomy data processing and visualization library.

This package provides tools for processing radio astronomy data from the
Owens Valley Radio Observatory - Long Wavelength Array (OVRO-LWA),
including FITS to Zarr conversion and visualization components.
"""

from __future__ import annotations

try:
    from .version import version as __version__  # type: ignore[import-not-found]
except ImportError:
    __version__ = "0.0.0+unknown"

from . import accessor  # Register xarray accessor on import
from . import fits_to_zarr_xradio
from .io import open_dataset, resolve_source, validate_local_zarr_store
from .name_resolution import CoordinateResolution, resolve_coordinate_string

__all__ = [
    "__version__",
    "accessor",
    "CoordinateResolution",
    "fits_to_zarr_xradio",
    "open_dataset",
    "resolve_coordinate_string",
    "resolve_source",
    "validate_local_zarr_store",
]
