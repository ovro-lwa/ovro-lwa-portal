"""HiPS URL helpers for local OVRO-LWA calibration sky backgrounds."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DEFAULT_HIPS_ROOT = Path("/lustre/pipeline/calibration/hips")
DEFAULT_HIPS_HTTP_PREFIX = "/calibration/hips"


def resolve_hips_root() -> Path:
    """Root directory containing ``*.hips`` survey folders."""
    return Path(os.environ.get("OVRO_HIPS_ROOT", str(DEFAULT_HIPS_ROOT)))


def resolve_hips_http_prefix() -> str:
    """URL path prefix where HiPS is served (no trailing slash)."""
    prefix = os.environ.get("OVRO_HIPS_HTTP_PREFIX", DEFAULT_HIPS_HTTP_PREFIX)
    prefix = os.environ.get("OVRO_HIPS_HTTP_BASE", prefix)
    prefix = prefix.rstrip("/")
    if not prefix.startswith("/"):
        prefix = f"/{prefix}"
    return prefix


def hips_background_survey_url(
    hips_dir: Path | str,
    *,
    hips_root: Path | str | None = None,
    http_prefix: str | None = None,
) -> str:
    """HTTP URL for an on-disk HiPS directory (trailing slash, for Aladin Lite).

    Parameters
    ----------
    hips_dir
        Path to a ``*.hips`` directory or a survey name under ``hips_root``.
    hips_root
        Directory containing HiPS surveys. Defaults to :func:`resolve_hips_root`.
    http_prefix
        URL prefix where the Jupyter HiPS handler is mounted. Defaults to
        :func:`resolve_hips_http_prefix`.
    """
    hips_path = Path(hips_dir).expanduser()
    if not hips_path.is_absolute():
        root = Path(hips_root) if hips_root is not None else resolve_hips_root()
        hips_path = (root / hips_path).resolve()
    else:
        hips_path = hips_path.resolve()

    root = Path(hips_root).resolve() if hips_root is not None else resolve_hips_root().resolve()
    try:
        rel = hips_path.relative_to(root)
    except ValueError:
        rel = Path(hips_path.name)

    base = (http_prefix if http_prefix is not None else resolve_hips_http_prefix()).rstrip("/")
    if base.startswith(("http://", "https://")):
        return f"{base}/{rel.as_posix()}/"
    if not base.startswith("/"):
        base = f"/{base}"
    return f"{base}/{rel.as_posix()}/"


def compute_hips_percentile_cuts(
    hips_dir: Path | str,
    *,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    max_pixels_per_tile: int = 50_000,
) -> tuple[float, float]:
    """Global min/max display cuts from HiPS FITS tile pixel values.

    Samples finite pixels from every ``*.fits`` tile under ``hips_dir`` and
    returns the requested percentiles (default 2nd and 98th), matching the
    radio overlay scaling in ``source_review.ipynb``.
    """
    from astropy.io import fits

    hips_path = Path(hips_dir).expanduser().resolve()
    tile_paths = sorted(hips_path.rglob("*.fits"))
    if not tile_paths:
        msg = f"No FITS tiles found under HiPS directory {hips_path}"
        raise FileNotFoundError(msg)

    samples: list[np.ndarray] = []
    for tile_path in tile_paths:
        with fits.open(tile_path, memmap=True) as hdul:
            data = hdul[0].data
            if data is None:
                continue
            finite = np.asarray(data, dtype=np.float64).ravel()
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                continue
            if finite.size > max_pixels_per_tile:
                stride = max(1, finite.size // max_pixels_per_tile)
                finite = finite[::stride][:max_pixels_per_tile]
            samples.append(finite)

    if not samples:
        msg = f"No finite pixel data in HiPS tiles under {hips_path}"
        raise ValueError(msg)

    merged = np.concatenate(samples)
    lo, hi = np.percentile(merged, [percentile_low, percentile_high])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)
