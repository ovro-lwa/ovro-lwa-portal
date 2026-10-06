"""Offline IERS policy for LST / ephemeris calls.

With ``auto_download=False`` (avoids notebook network stalls), times that need
IERS-A *predictive* values older than ``auto_max_age`` (default 30 days) raise:

    ValueError: interpolating from IERS_Auto using predictive values that are more
    than 30.0 days old...

Astropy's recommended suppression is ``auto_max_age = None``. At OVRO-LWA beam
scales (~arcmin) that accuracy loss is negligible, so :func:`offline_iers` keeps
downloads off and clears the age gate (and ignores degraded-accuracy errors).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def offline_iers() -> Iterator[None]:
    """Disable IERS auto-download and allow stale predictive values."""
    from astropy.utils.iers import conf as iers_conf

    orig_download = iers_conf.auto_download
    orig_max_age = iers_conf.auto_max_age
    orig_degraded = iers_conf.iers_degraded_accuracy
    try:
        iers_conf.auto_download = False
        iers_conf.auto_max_age = None
        iers_conf.iers_degraded_accuracy = "ignore"
        yield
    finally:
        iers_conf.auto_download = orig_download
        iers_conf.auto_max_age = orig_max_age
        iers_conf.iers_degraded_accuracy = orig_degraded
