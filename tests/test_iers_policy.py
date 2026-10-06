"""Regression: offline IERS must not raise on stale predictive values."""

from __future__ import annotations

import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy.utils.iers import conf as iers_conf

from ovro_lwa_portal._iers import offline_iers


def test_offline_iers_allows_stale_predictive_values() -> None:
    """Stale IERS-A predictive values must not raise under offline_iers."""
    # Defaults that trigger ValueError without our context manager.
    iers_conf.auto_download = False
    iers_conf.auto_max_age = 30.0
    iers_conf.iers_degraded_accuracy = "error"
    obs = EarthLocation.of_site("ovro")
    # Past predictive_mjd with table age > auto_max_age (see Astropy IERS_Auto).
    t = Time(np.array([Time.now().mjd + 90.0]), format="mjd", scale="utc")

    with offline_iers():
        lst = t.sidereal_time("mean", longitude=obs.lon)

    assert np.isfinite(float(lst.deg[0]))
    # Context restores caller policy.
    assert iers_conf.auto_download is False
    assert iers_conf.auto_max_age == 30.0
    assert iers_conf.iers_degraded_accuracy == "error"
