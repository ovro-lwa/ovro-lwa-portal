"""Export OVRO-LWA Zarr ``SKY`` slices to standalone 4D singleton-axis FITS files.

Archival model
--------------

OVRO-LWA delivers **separate FITS files** per ``(time, frequency, Stokes)`` cell.
Ingest combines them into one Zarr store; this module recovers any slice as a
**pixel-faithful** standalone FITS file for external tools (pybdsf, CASA, DS9,
``ovro-ingest convert``, xradio) without portal-specific knowledge.

Zarr is the consolidated archive; export is the round-trip path back to per-slice
FITS. Metadata comes from ``fits_header_str(time, frequency, polarization)``
(the sole on-disk header array on new ingests). Portal WCS helpers derive a
celestial subset in memory — export reads the **full** stored header.

Export layout
-------------

Each call selects exactly one ``(time_idx, freq_idx, pol_idx)`` and writes one
``PrimaryHDU`` with the OVRO/xradio 4D singleton-axis layout:

- ``NAXIS=4``, ``NAXIS3=NAXIS4=1`` (singleton FREQ and Stokes axes)
- Data shape ``(1, 1, n_m, n_l)`` — FITS axis order 4→3→2→1
- ``CTYPE3='FREQ'``, ``CRVAL3`` from ``ds.frequency[freq_idx]``
- ``CTYPE4='STOKES'``, ``CRVAL4`` from ``ds.polarization[pol_idx]`` (FITS Stokes
  code, e.g. ``1`` = I, ``4`` = V)
- Spatial WCS from stored ``fits_header_str`` (post-regrid, pixel-faithful)

**Not supported:** multi-frequency, multi-Stokes, or time cubes in one FITS file
(``NAXIS3>1`` or ``NAXIS4>1``). Combined I+V Zarr stores export **two** files
(one per ``pol_idx``).

Legacy stores
-------------

Stores with only ``wcs_header_str`` (no ``fits_header_str``) **hard-fail** on
export with :class:`ValueError` and re-ingest guidance. Re-ingest from original
FITS to populate ``fits_header_str``.

Examples
--------

Single-slice export::

    from ovro_lwa_portal import open_dataset, write_fits_slice

    ds = open_dataset("/path/to/store.zarr")
    write_fits_slice(ds, "image.fits", time_idx=0, freq_idx=0, pol_idx=0)

Batch export via accessor::

    paths = ds.radport.export_fits("/out/dir", time_indices=[0, 1], freq_indices=[0])

Build an HDU for pybdsf or xradio::

    from ovro_lwa_portal import build_fits_hdu

    hdu = build_fits_hdu(ds, time_idx=0, freq_idx=0, pol_idx=0)
    hdu.writeto("image.fits", overwrite=True)

    import bdsf

    bdsf.process_image("image.fits")  # expects NAXIS=4 singleton FREQ/Stokes

``BEAM`` data variables are **not** exported — beam keywords remain in the FITS
header when present in ``fits_header_str``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy.io.fits import HDUList, Header, PrimaryHDU

from ovro_lwa_portal.accessor import _read_fits_header_str
from ovro_lwa_portal.fits_to_zarr_xradio import (
    _fits_stokes_from_polarization_coord,
    _promote_singleton_freq_stokes_cards,
)

if TYPE_CHECKING:
    import xarray as xr

_EXPORT_VAR = "SKY"


def _freq_hz_at_index(ds: xr.Dataset, freq_idx: int) -> float:
    if "frequency" not in ds.coords:
        msg = "Dataset is missing a frequency coordinate required for FITS export."
        raise ValueError(msg)
    values = np.asarray(ds.coords["frequency"].values).ravel()
    if freq_idx < 0 or freq_idx >= values.size:
        msg = f"freq_idx={freq_idx} out of range for frequency size {values.size}."
        raise IndexError(msg)
    return float(values[freq_idx])


def _stokes_at_index(ds: xr.Dataset, pol_idx: int) -> float:
    if "polarization" not in ds.coords:
        msg = "Dataset is missing a polarization coordinate required for FITS export."
        raise ValueError(msg)
    values = np.asarray(ds.coords["polarization"].values).ravel()
    if pol_idx < 0 or pol_idx >= values.size:
        msg = f"pol_idx={pol_idx} out of range for polarization size {values.size}."
        raise IndexError(msg)
    mapped = _fits_stokes_from_polarization_coord(values[pol_idx])
    if mapped is not None:
        return mapped
    msg = (
        f"polarization coordinate value {values[pol_idx]!r} at pol_idx={pol_idx} "
        "is not a FITS Stokes code."
    )
    raise ValueError(msg)


def _validate_slice_indices(
    ds: xr.Dataset,
    *,
    time_idx: int,
    freq_idx: int,
    pol_idx: int,
    var: str = _EXPORT_VAR,
) -> None:
    if var not in ds.data_vars:
        msg = f"Dataset is missing required variable {var!r} for FITS export."
        raise ValueError(msg)
    da = ds[var]
    for dim, idx, name in (
        ("time", time_idx, "time_idx"),
        ("frequency", freq_idx, "freq_idx"),
        ("polarization", pol_idx, "pol_idx"),
    ):
        if dim not in da.dims:
            continue
        size = int(da.sizes[dim])
        if idx < 0 or idx >= size:
            msg = f"{name}={idx} out of range for {dim} size {size}."
            raise IndexError(msg)
    if "l" not in da.dims or "m" not in da.dims:
        msg = f"{var} must include l and m dimensions for FITS export."
        raise ValueError(msg)


def build_fits_header(
    ds: xr.Dataset,
    *,
    time_idx: int = 0,
    freq_idx: int = 0,
    pol_idx: int = 0,
    var: str = _EXPORT_VAR,
) -> Header:
    """Build a 4D singleton-axis FITS header for one ``SKY`` slice.

    Starts from persisted ``fits_header_str``, patches spatial ``NAXIS1/2`` to
    match the dataset grid, and aligns singleton FREQ/Stokes cards with live
    ``frequency`` / ``polarization`` coordinates.
    """
    _validate_slice_indices(
        ds, time_idx=time_idx, freq_idx=freq_idx, pol_idx=pol_idx, var=var
    )
    hdr_str = _read_fits_header_str(
        ds, time_idx=time_idx, freq_idx=freq_idx, pol_idx=pol_idx
    )
    header = Header.fromstring(hdr_str, sep="\n")
    nl = int(ds.sizes["l"])
    nm = int(ds.sizes["m"])
    header["NAXIS1"] = nl
    header["NAXIS2"] = nm
    freq_hz = _freq_hz_at_index(ds, freq_idx)
    stokes = _stokes_at_index(ds, pol_idx)
    _promote_singleton_freq_stokes_cards(header, freq_hz=freq_hz, stokes=stokes)
    header["BITPIX"] = -32
    for key in ("BSCALE", "BZERO"):
        if key in header:
            del header[key]
    return header


def build_fits_data_array(
    ds: xr.Dataset,
    *,
    time_idx: int = 0,
    freq_idx: int = 0,
    pol_idx: int = 0,
    var: str = _EXPORT_VAR,
) -> np.ndarray:
    """Return ``SKY`` slice data as ``(1, 1, n_m, n_l)`` float32 for FITS export."""
    _validate_slice_indices(
        ds, time_idx=time_idx, freq_idx=freq_idx, pol_idx=pol_idx, var=var
    )
    da = ds[var]
    sel = da
    if "time" in da.dims:
        sel = sel.isel(time=time_idx)
    if "frequency" in sel.dims:
        sel = sel.isel(frequency=freq_idx)
    if "polarization" in sel.dims:
        sel = sel.isel(polarization=pol_idx)
    plane = np.asarray(sel.load().values, dtype=np.float32)
    if plane.ndim != 2:
        msg = f"Expected 2D (m, l) slice after index selection; got shape {plane.shape}."
        raise ValueError(msg)
    return plane[np.newaxis, np.newaxis, :, :]


def build_fits_hdu(
    ds: xr.Dataset,
    *,
    time_idx: int = 0,
    freq_idx: int = 0,
    pol_idx: int = 0,
    var: str = _EXPORT_VAR,
) -> PrimaryHDU:
    """Build a ``PrimaryHDU`` for one exported ``SKY`` slice.

    The returned HDU has ``NAXIS=4`` with singleton FREQ/Stokes axes
    (``NAXIS3=NAXIS4=1``), ``data.shape == (1, 1, n_m, n_l)``, and a header
    rebuilt from ``fits_header_str`` plus live ``frequency`` / ``polarization``
    coordinates. Readable by Astropy, xradio, pybdsf, and ``ovro-ingest convert``.

    Raises
    ------
    ValueError
        If ``fits_header_str`` is missing (legacy store — re-ingest required) or
        required coordinates/variables are absent.
    IndexError
        If any slice index is out of range.
    """
    header = build_fits_header(
        ds, time_idx=time_idx, freq_idx=freq_idx, pol_idx=pol_idx, var=var
    )
    data = build_fits_data_array(
        ds, time_idx=time_idx, freq_idx=freq_idx, pol_idx=pol_idx, var=var
    )
    return PrimaryHDU(data=data, header=header)


def build_fits_hdulist(
    ds: xr.Dataset,
    *,
    time_idx: int = 0,
    freq_idx: int = 0,
    pol_idx: int = 0,
    var: str = _EXPORT_VAR,
) -> HDUList:
    """Build an ``HDUList`` containing a single exported ``PrimaryHDU``."""
    return HDUList(
        [
            build_fits_hdu(
                ds,
                time_idx=time_idx,
                freq_idx=freq_idx,
                pol_idx=pol_idx,
                var=var,
            )
        ]
    )


def write_fits_slice(
    ds: xr.Dataset,
    path: str | Path,
    *,
    time_idx: int = 0,
    freq_idx: int = 0,
    pol_idx: int = 0,
    var: str = _EXPORT_VAR,
    overwrite: bool = False,
) -> Path:
    """Write one exported ``SKY`` slice to a standalone FITS file."""
    out = Path(path)
    hdu = build_fits_hdu(
        ds, time_idx=time_idx, freq_idx=freq_idx, pol_idx=pol_idx, var=var
    )
    hdu.writeto(out, overwrite=overwrite)
    return out
