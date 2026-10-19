#!/usr/bin/env python3
"""Split a 4D Stokes×FREQ×sky FITS cube into per-channel planes for ``ovro-ingest convert``.

``ovro-ingest convert`` expects one FITS file per ``(time, frequency, Stokes)`` cell
with a usable primary-header beam (``BMAJ``/``BMIN``). Multi-channel cubes that store
the beam only in a CASA ``BEAMS`` binary table (``CASAMBM=True``) are skipped.

This script writes one singleton 4D primary HDU per spectral channel:

* spatial WCS and Stokes axis copied from the cube
* ``NAXIS3=NAXIS4=1`` with ``CRVAL3`` / ``RESTFREQ`` / ``RESTFRQ`` set to that channel
* ``BMAJ``/``BMIN``/``BPA`` copied from the matching ``BEAMS`` row (arcsec → degrees)

Provisional output names (rename later if needed)::

    {stem}_ch{NN}_{freq_mhz:.6f}MHz.fits

Usage::

    pixi run python scripts/split_spectral_cube_for_ingest.py \\
        /fast/claw/mockfm_nl_cube.fits /fast/claw/mockfm_nl_planes

    # Then:
    ovro-ingest convert /fast/claw/mockfm_nl_planes /fast/claw/mockfm_nl_zarr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


def _axis_index(header: fits.Header, *prefixes: str) -> int:
    """1-based FITS axis index whose ``CTYPE`` starts with one of *prefixes*."""
    naxis = int(header.get("NAXIS", 0))
    wanted = tuple(p.upper() for p in prefixes)
    for i in range(1, naxis + 1):
        ctype = str(header.get(f"CTYPE{i}", "")).strip().upper()
        if any(ctype.startswith(p) for p in wanted):
            return i
    msg = f"No axis with CTYPE starting {wanted} in {dict(header)}"
    raise ValueError(msg)


def _channel_frequency_hz(header: fits.Header, freq_axis: int, chan: int) -> float:
    """World frequency for 0-based channel index along *freq_axis*."""
    crval = float(header[f"CRVAL{freq_axis}"])
    cdelt = float(header[f"CDELT{freq_axis}"])
    crpix = float(header.get(f"CRPIX{freq_axis}", 1.0))
    return crval + (float(chan) + 1.0 - crpix) * cdelt


def _beam_row_for_channel(beams: fits.FITS_rec, chan: int, pol: int) -> fits.FITS_record:
    """Return the ``BEAMS`` table row for ``(CHAN, POL)``."""
    match = (beams["CHAN"] == chan) & (beams["POL"] == pol)
    idx = np.flatnonzero(match)
    if idx.size == 0:
        msg = f"BEAMS table has no row for CHAN={chan}, POL={pol}"
        raise ValueError(msg)
    if idx.size > 1:
        msg = f"BEAMS table has {idx.size} rows for CHAN={chan}, POL={pol}"
        raise ValueError(msg)
    return beams[int(idx[0])]


def _beams_units_are_arcsec(beams_hdu: fits.BinTableHDU) -> bool:
    """True when ``BMAJ``/``BMIN`` ``TUNIT`` are arcsec (CASA multi-beam default)."""
    cols = beams_hdu.columns
    for name in ("BMAJ", "BMIN"):
        unit = str(cols[name].unit or "").strip().lower()
        if unit not in {"arcsec", "arcsecs", '"', "as"}:
            return False
    return True


def _stamp_singleton_spectral_axis(header: fits.Header, freq_axis: int, freq_hz: float) -> None:
    """Collapse channel axis to length 1 and stamp the channel frequency."""
    header[f"NAXIS{freq_axis}"] = 1
    header[f"CRVAL{freq_axis}"] = float(freq_hz)
    header[f"CRPIX{freq_axis}"] = 1.0
    # Keep CDELT so WCS remains well-formed; singleton axis length is 1.
    if f"CUNIT{freq_axis}" not in header:
        header[f"CUNIT{freq_axis}"] = "Hz"
    # Discovery prefers RESTFREQ/RESTFRQ over CRVAL3; must match the channel,
    # not a cube-level rest frequency (e.g. HI line).
    header["RESTFREQ"] = float(freq_hz)
    header["RESTFRQ"] = (float(freq_hz), "Rest Frequency (Hz)")


def _stamp_beam_from_table_row(
    header: fits.Header,
    row: fits.FITS_record,
    *,
    bmaj_bmin_in_arcsec: bool,
) -> None:
    """Write primary-header beam cards (degrees) from a ``BEAMS`` row."""
    bmaj = float(row["BMAJ"])
    bmin = float(row["BMIN"])
    bpa = float(row["BPA"])
    if bmaj_bmin_in_arcsec:
        bmaj /= 3600.0
        bmin /= 3600.0
    if not np.isfinite(bmaj) or not np.isfinite(bmin) or bmaj <= 0.0 or bmin <= 0.0:
        msg = f"Invalid beam in BEAMS row: BMAJ={bmaj}, BMIN={bmin}"
        raise ValueError(msg)
    header["BMAJ"] = (bmaj, "Beam major axis (deg)")
    header["BMIN"] = (bmin, "Beam minor axis (deg)")
    header["BPA"] = (bpa, "Beam position angle (deg)")


def _ensure_convert_friendly_cards(header: fits.Header) -> None:
    """Fill cards ``ovro-ingest convert`` / xradio commonly require."""
    naxis = int(header.get("NAXIS", 0))
    for i in range(1, naxis + 1):
        key = f"CUNIT{i}"
        if key in header:
            continue
        ctype = str(header.get(f"CTYPE{i}", "")).strip().upper()
        if ctype.startswith("RA") or ctype.startswith("DEC"):
            header[key] = "deg"
        elif ctype.startswith("FREQ") or ctype in {"VOPT", "VRAD"}:
            header[key] = "Hz"
        else:
            header[key] = ""

    # Identity PC for celestial axes (xradio indexes these without .get).
    header.setdefault("PC1_1", 1.0)
    header.setdefault("PC1_2", 0.0)
    header.setdefault("PC2_1", 0.0)
    header.setdefault("PC2_2", 1.0)

    if "LONPOLE" not in header:
        header["LONPOLE"] = 180.0
    if "EQUINOX" not in header and str(header.get("RADESYS", "")).strip().upper() != "ICRS":
        header["EQUINOX"] = 2000.0
    if "BUNIT" not in header:
        header["BUNIT"] = "Jy/beam"

    # Beam now lives on the primary header; drop the multi-beam flag.
    if "CASAMBM" in header:
        del header["CASAMBM"]


def split_spectral_cube(
    cube_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
) -> list[Path]:
    """Write one convert-friendly FITS plane per frequency channel.

    Parameters
    ----------
    cube_path
        Input 4D FITS (Stokes × frequency × sky) with optional ``BEAMS`` table.
    output_dir
        Directory for output planes (created if missing).
    overwrite
        Replace existing output files.
    dry_run
        Parse and report outputs without writing.

    Returns
    -------
    list[Path]
        Paths that were (or would be) written, in channel order.
    """
    cube_path = cube_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    with fits.open(cube_path, memmap=True) as hdul:
        image = hdul[0]
        header0 = image.header
        data = image.data
        if data is None:
            msg = f"{cube_path}: primary HDU has no image data"
            raise ValueError(msg)
        if data.ndim != 4:
            msg = f"{cube_path}: expected 4D primary data, got shape {data.shape}"
            raise ValueError(msg)

        freq_axis_fits = _axis_index(header0, "FREQ", "VOPT", "VRAD")
        stokes_axis_fits = _axis_index(header0, "STOKES")
        # Output planes use CASA-style axis order:
        # NAXIS1=RA, NAXIS2=Dec, NAXIS3=FREQ, NAXIS4=STOKES.
        if freq_axis_fits != 3 or stokes_axis_fits != 4:
            msg = (
                f"{cube_path}: expected CTYPE3=FREQ and CTYPE4=STOKES "
                f"(got FREQ on axis {freq_axis_fits}, STOKES on axis {stokes_axis_fits})"
            )
            raise ValueError(msg)
        # Numpy axis order is reversed vs FITS NAXISn (NAXIS4 is axis 0).
        freq_axis_np = data.ndim - freq_axis_fits
        stokes_axis_np = data.ndim - stokes_axis_fits
        n_freq = int(data.shape[freq_axis_np])
        n_stokes = int(data.shape[stokes_axis_np])
        if n_stokes != 1:
            msg = (
                f"{cube_path}: expected a single Stokes plane for convert "
                f"(got NAXIS{stokes_axis_fits}={n_stokes}); split Stokes first"
            )
            raise ValueError(msg)

        if "BEAMS" not in hdul:
            msg = f"{cube_path}: missing BEAMS binary table (needed for BMAJ/BMIN)"
            raise ValueError(msg)
        beams_hdu = hdul["BEAMS"]
        beams = beams_hdu.data
        if beams is None:
            msg = f"{cube_path}: BEAMS extension has no rows"
            raise ValueError(msg)
        arcsec = _beams_units_are_arcsec(beams_hdu)
        if not arcsec:
            units = {
                name: beams_hdu.columns[name].unit
                for name in ("BMAJ", "BMIN")
                if name in beams_hdu.columns.names
            }
            msg = (
                f"{cube_path}: BEAMS BMAJ/BMIN units must be arcsec for this script "
                f"(got {units}); refusing to guess"
            )
            raise ValueError(msg)

        # Stokes CRVAL for POL index in BEAMS (0-based table index for pol plane 0).
        pol_table = 0
        stem = cube_path.stem
        written: list[Path] = []

        for chan in range(n_freq):
            freq_hz = _channel_frequency_hz(header0, freq_axis_fits, chan)
            freq_mhz = freq_hz / 1e6
            out_name = f"{stem}_ch{chan:02d}_{freq_mhz:.6f}MHz.fits"
            out_path = output_dir / out_name

            # Isolate (stokes=0, freq=chan) → (1, 1, ny, nx). Numpy axis order is
            # (NAXIS4, NAXIS3, NAXIS2, NAXIS1) == (Stokes, FREQ, Dec, RA).
            slicer: list[object] = [slice(None)] * data.ndim
            slicer[stokes_axis_np] = 0
            slicer[freq_axis_np] = chan
            plane2d = np.asarray(data[tuple(slicer)], dtype=np.float32)
            if plane2d.ndim != 2:
                msg = f"Channel slice is not 2D: shape={plane2d.shape}"
                raise RuntimeError(msg)
            cube1 = plane2d[np.newaxis, np.newaxis, :, :]

            hdr = header0.copy()
            hdr["NAXIS"] = 4
            hdr["NAXIS1"] = int(cube1.shape[-1])
            hdr["NAXIS2"] = int(cube1.shape[-2])
            hdr["NAXIS3"] = 1
            hdr["NAXIS4"] = 1
            _stamp_singleton_spectral_axis(hdr, 3, freq_hz)
            hdr["CRPIX4"] = float(hdr.get("CRPIX4", 1.0))
            if "CUNIT4" not in hdr:
                hdr["CUNIT4"] = ""

            row = _beam_row_for_channel(beams, chan, pol_table)
            _stamp_beam_from_table_row(hdr, row, bmaj_bmin_in_arcsec=True)
            _ensure_convert_friendly_cards(hdr)

            while "HISTORY" in hdr:
                del hdr["HISTORY"]

            written.append(out_path)
            if dry_run:
                print(
                    f"would write {out_path.name}: "
                    f"freq={freq_hz:.3f} Hz, "
                    f"BMAJ={float(hdr['BMAJ']):.8f} deg, "
                    f"BMIN={float(hdr['BMIN']):.8f} deg, "
                    f"shape={cube1.shape}"
                )
                continue

            if out_path.exists() and not overwrite:
                msg = f"Refusing to overwrite existing file: {out_path} (pass --overwrite)"
                raise FileExistsError(msg)

            fits.PrimaryHDU(data=cube1, header=hdr).writeto(out_path, overwrite=overwrite)
            print(f"wrote {out_path}")

        return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Split a 4D spectral FITS cube into per-channel planes suitable for "
            "ovro-ingest convert (primary BMAJ/BMIN + singleton FREQ/STOKES)."
        )
    )
    parser.add_argument(
        "cube",
        type=Path,
        help="Input 4D FITS cube (Stokes × FREQ × sky) with BEAMS table",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for per-channel FITS outputs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs without writing",
    )
    args = parser.parse_args(argv)

    try:
        paths = split_spectral_cube(
            args.cube,
            args.output_dir,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{'would write' if args.dry_run else 'wrote'} {len(paths)} file(s) → {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
