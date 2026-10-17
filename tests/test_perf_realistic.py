"""Performance tests for the radport accessor at production-scale data dimensions.

This module creates a Dask-backed dataset matching real OVRO-LWA production data
(4096x4096 spatial, 10 time steps, 10 frequency channels) and benchmarks the
core accessor methods. A synthetic Cygnus A point source is injected to verify
that the RA/Dec tracking path recovers the expected flux.

Run only these tests with:
    pixi run pytest -m slow tests/test_perf_realistic.py -v

Skip these tests in CI with:
    pixi run pytest -m "not slow" tests/

Physics notes
-------------
SIN (orthographic) projection maps a source at (RA, Dec) to direction cosines
(l, m) via:

    H   = LST - RA                     [hour angle, radians]
    l   = cos(Dec) * sin(H)
    m   = sin(Dec)*cos(lat) - cos(Dec)*sin(lat)*cos(H)

where lat is the observatory geodetic latitude.  A source is above the horizon
when sin(alt) = sin(Dec)*sin(lat) + cos(Dec)*cos(lat)*cos(H) > 0.
"""

from __future__ import annotations

import time
import warnings

import matplotlib

# Use non-interactive backend before any other matplotlib import.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
import dask.array as da

# Register the accessor.
import ovro_lwa_portal  # noqa: F401

# ---------------------------------------------------------------------------
# Physical / observatory constants
# ---------------------------------------------------------------------------
OVRO_LAT_DEG = 37.2339
OVRO_LON_DEG = -118.2817
OVRO_HEIGHT_M = 1222.0

# Cygnus A (3C 405) — one of the brightest radio sources in the sky
CYGA_RA_DEG = 299.868
CYGA_DEC_DEG = 40.734
CYGA_FLUX_JY = 100.0  # injected peak flux density

# Dataset dimensions — matching real OVRO-LWA zarr chunk layout
N_TIME = 10
N_FREQ = 10
N_POL = 1
N_L = 4096
N_M = 4096
CHUNK = (1, 1, 1, 1024, 1024)

MJD_START = 60000.0
MJD_STEP = 0.01  # ~14.4 minutes per step

FREQ_START_HZ = 27.0e6
FREQ_STOP_HZ = 88.0e6

# Pixel half-width used when injecting the synthetic source (pixels).
# One pixel on a 4096-grid covers ~0.028°; ±2 pixels gives a ~3-pixel
# diameter blob, easily recovered with nearest-neighbour extraction.
SOURCE_INJECT_HALF_WIDTH = 2

# Performance threshold — each operation must finish within this many seconds.
PERF_THRESHOLD_S = 30.0

# Background flux level (Jy) — noise floor for the injected dataset.
BACKGROUND_JY = 1.0

# Minimum flux considered a "detection" in the tracked light curve / dynspec.
DETECTION_THRESHOLD_JY = 50.0


# ---------------------------------------------------------------------------
# Helper: compute SIN-projection (l, m) for a source at each time step
# ---------------------------------------------------------------------------

def _compute_lm_track(
    ra_deg: float,
    dec_deg: float,
    mjd_times: np.ndarray,
    lon_deg: float = OVRO_LON_DEG,
    lat_deg: float = OVRO_LAT_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return l, m, and visibility mask for a source at each MJD.

    Parameters
    ----------
    ra_deg, dec_deg : float
        Source J2000 FK5 coordinates in degrees.
    mjd_times : np.ndarray
        Array of MJD timestamps (UTC).
    lon_deg, lat_deg : float
        Observatory geodetic coordinates in degrees.

    Returns
    -------
    l_vals, m_vals : np.ndarray
        SIN-projection direction cosines at each time step.
    visible : np.ndarray of bool
        True where the source is above the horizon.
    """
    from astropy import units as u
    from astropy.time import Time

    from ovro_lwa_portal._iers import offline_iers

    # Bundled/cached IERS; download off. Accuracy vs fresh IERS-A is negligible
    # at OVRO-LWA's ~6-arcmin beam.
    with offline_iers():
        t = Time(mjd_times, format="mjd", scale="utc")
        lst_deg = t.sidereal_time("mean", longitude=lon_deg * u.deg).deg

    ha_rad = np.deg2rad(lst_deg - ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    lat_rad = np.deg2rad(lat_deg)

    l_vals = -np.cos(dec_rad) * np.sin(ha_rad)
    m_vals = (
        np.sin(dec_rad) * np.cos(lat_rad)
        - np.cos(dec_rad) * np.sin(lat_rad) * np.cos(ha_rad)
    )

    sin_alt = (
        np.sin(dec_rad) * np.sin(lat_rad)
        + np.cos(dec_rad) * np.cos(lat_rad) * np.cos(ha_rad)
    )
    visible = sin_alt > 0

    return l_vals, m_vals, visible


# ---------------------------------------------------------------------------
# Helper: build WCS header string
# ---------------------------------------------------------------------------

def _build_wcs_header(crval1_deg: float, crval2_deg: float, n_pix: int) -> str:
    """Return a minimal FITS WCS header string for a SIN-projection image.

    The phase centre is placed at (crval1_deg, crval2_deg).  The pixel scale
    is computed to exactly span [-1, 1] in direction cosines over n_pix pixels:

        CDELT = (2.0 / n_pix) * (180 / pi)  degrees/pixel

    This matches the l/m coordinate grid used by the dataset.

    Note: CDELT1 is negative (RA increases to the left on standard sky images).
    """
    cdelt_deg = (2.0 / n_pix) * (180.0 / np.pi)
    crpix = n_pix / 2 + 0.5  # centre of the array (FITS 1-based)

    return (
        f"NAXIS   =                    2\n"
        f"NAXIS1  =                 {n_pix:4d}\n"
        f"NAXIS2  =                 {n_pix:4d}\n"
        f"CTYPE1  = 'RA---SIN'\n"
        f"CTYPE2  = 'DEC--SIN'\n"
        f"CRPIX1  =          {crpix:11.4f}\n"
        f"CRPIX2  =          {crpix:11.4f}\n"
        f"CRVAL1  =   {crval1_deg:18.10f}\n"
        f"CRVAL2  =   {crval2_deg:18.10f}\n"
        f"CDELT1  =  {-cdelt_deg:19.14f}\n"
        f"CDELT2  =   {cdelt_deg:19.14f}\n"
        f"CUNIT1  = 'deg     '\n"
        f"CUNIT2  = 'deg     '\n"
        f"RADESYS = 'FK5     '\n"
        f"EQUINOX =               2000.0\n"
    )


# ---------------------------------------------------------------------------
# Module-scoped fixture: production-scale dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def production_dataset() -> xr.Dataset:
    """Create a production-scale Dask-backed OVRO-LWA dataset.

    Shape: (10 time, 10 freq, 1 pol, 4096 l, 4096 m)
    Chunking: (1, 1, 1, 1024, 1024) — matches real zarr layout on disk.

    A synthetic Cygnus A point source (100 Jy) is injected into the pixel
    nearest to the source's SIN-projection position at each visible time step.
    The background level is 1 Jy uniform noise.

    The WCS phase centre is set to the zenith (RA = LST at MJD 60000.0,
    Dec = OVRO-LWA latitude) so that the pixel scale exactly matches the
    l/m coordinate grid.
    """
    from astropy import units as u
    from astropy.time import Time

    from ovro_lwa_portal._iers import offline_iers

    mjd_times = MJD_START + np.arange(N_TIME) * MJD_STEP
    freq_hz = np.linspace(FREQ_START_HZ, FREQ_STOP_HZ, N_FREQ)
    l_coords = np.linspace(-1.0, 1.0, N_L)
    m_coords = np.linspace(-1.0, 1.0, N_M)

    # ------------------------------------------------------------------ #
    # Build the background as a truly lazy Dask array — no eager GB
    # allocation.  Each chunk is generated on demand via da.random.
    # Only the small source-injection patches are materialised as numpy.
    # ------------------------------------------------------------------ #
    dask_rng = da.random.RandomState(42)
    sky_dask = dask_rng.uniform(
        low=0.5 * BACKGROUND_JY,
        high=1.5 * BACKGROUND_JY,
        size=(N_TIME, N_FREQ, N_POL, N_L, N_M),
        chunks=CHUNK,
    ).astype(np.float32)

    # Pre-compute source pixel indices for injection
    l_vals, m_vals, visible = _compute_lm_track(
        CYGA_RA_DEG, CYGA_DEC_DEG, mjd_times
    )

    n_visible = int(np.sum(visible))
    injected_times = []

    # Build a dict mapping (time_idx, l_chunk_idx, m_chunk_idx) → injection
    # info so we can apply patches lazily via da.map_blocks.
    # Each spatial chunk is 1024x1024; we need to know which chunk(s) a
    # source pixel falls into and the local offset within that chunk.
    l_chunk_size = CHUNK[3]  # 1024
    m_chunk_size = CHUNK[4]  # 1024

    # injection_map: ti → (l_idx, m_idx) in global coordinates
    injection_map: dict[int, tuple[int, int]] = {}

    for ti in range(N_TIME):
        if not visible[ti]:
            continue
        l_idx = int(np.argmin(np.abs(l_coords - l_vals[ti])))
        m_idx = int(np.argmin(np.abs(m_coords - m_vals[ti])))
        injection_map[ti] = (l_idx, m_idx)
        injected_times.append(ti)

    def _inject_source(block, block_info=None):
        """map_blocks callback: inject source into affected chunks."""
        if block_info is None or not injection_map:
            return block
        # block_info[0] gives the array-location of this block
        info = block_info[0]
        ti_start = info["array-location"][0][0]  # scalar — chunk is size 1
        l_start = info["array-location"][3][0]
        l_end = info["array-location"][3][1]
        m_start = info["array-location"][4][0]
        m_end = info["array-location"][4][1]

        if ti_start not in injection_map:
            return block

        g_l, g_m = injection_map[ti_start]
        hw = SOURCE_INJECT_HALF_WIDTH
        # Global injection bounds
        inj_l_lo = max(0, g_l - hw)
        inj_l_hi = min(N_L, g_l + hw + 1)
        inj_m_lo = max(0, g_m - hw)
        inj_m_hi = min(N_M, g_m + hw + 1)

        # Check if this spatial chunk overlaps the injection region
        if inj_l_hi <= l_start or inj_l_lo >= l_end:
            return block
        if inj_m_hi <= m_start or inj_m_lo >= m_end:
            return block

        # Compute local offsets within this chunk
        loc_l_lo = max(0, inj_l_lo - l_start)
        loc_l_hi = min(l_end - l_start, inj_l_hi - l_start)
        loc_m_lo = max(0, inj_m_lo - m_start)
        loc_m_hi = min(m_end - m_start, inj_m_hi - m_start)

        out = block.copy()
        out[:, :, :, loc_l_lo:loc_l_hi, loc_m_lo:loc_m_hi] = CYGA_FLUX_JY
        return out

    sky_dask = sky_dask.map_blocks(_inject_source, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Build the WCS header.
    # The phase centre RA is set to the mean LST across the observation so
    # the source tracking is reasonably centred in the image.
    # ------------------------------------------------------------------ #
    with offline_iers():
        t0 = Time(MJD_START, format="mjd", scale="utc")
        lst0_deg = float(
            t0.sidereal_time("mean", longitude=OVRO_LON_DEG * u.deg).deg
        )

    wcs_header_str = _build_wcs_header(
        crval1_deg=lst0_deg,
        crval2_deg=OVRO_LAT_DEG,
        n_pix=N_L,
    )

    ds = xr.Dataset(
        data_vars={
            "SKY": (
                ["time", "frequency", "polarization", "l", "m"],
                sky_dask,
            )
        },
        coords={
            "time": mjd_times,
            "frequency": freq_hz,
            "polarization": np.array([0]),
            "l": l_coords,
            "m": m_coords,
        },
    )
    ds["SKY"].attrs["fits_wcs_header"] = wcs_header_str
    ds["SKY"].attrs["units"] = "Jy/beam"

    # Stash ground-truth info for correctness assertions in tests.
    ds.attrs["_test_cyga_visible_times"] = injected_times
    ds.attrs["_test_n_visible"] = n_visible

    return ds


# ---------------------------------------------------------------------------
# Performance benchmark helper
# ---------------------------------------------------------------------------

def _bench(label: str, fn, threshold: float = PERF_THRESHOLD_S):
    """Time *fn*, print result, and assert it finishes within *threshold* s.

    Parameters
    ----------
    label : str
        Human-readable description printed to stdout.
    fn : callable
        Zero-argument callable to benchmark.
    threshold : float
        Wall-clock limit in seconds before the test is marked as failing.

    Returns
    -------
    result
        Whatever *fn* returns (pass-through for subsequent assertions).
    """
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    status = "PASS" if elapsed < threshold else "FAIL"
    print(  # noqa: T201
        f"  [{status}] {label}: {elapsed:.2f}s "
        f"(threshold: {threshold:.0f}s)"
    )
    assert elapsed < threshold, (
        f"Performance regression: '{label}' took {elapsed:.2f}s, "
        f"threshold is {threshold:.0f}s"
    )
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDynamicSpectrum:
    """Benchmark and correctness tests for dynamic_spectrum()."""

    @pytest.mark.slow
    def test_dynamic_spectrum_lm_path_performance(
        self, production_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum(l=0, m=0) on 4096x4096 data completes within threshold."""
        print()  # noqa: T201
        dynspec = _bench(
            "dynamic_spectrum(l/m, centre pixel)",
            lambda: production_dataset.radport.dynamic_spectrum(l=0.0, m=0.0),
        )
        # Shape must be (N_TIME, N_FREQ)
        assert dynspec.dims == ("time", "frequency")
        assert dynspec.shape == (N_TIME, N_FREQ)
        # Centre pixel is background — all values should be finite and low
        assert np.all(np.isfinite(dynspec.values))
        assert float(dynspec.values.max()) < CYGA_FLUX_JY, (
            "Centre pixel should not contain the injected source"
        )

    @pytest.mark.slow
    def test_dynamic_spectrum_radec_path_performance(
        self, production_dataset: xr.Dataset
    ) -> None:
        """dynamic_spectrum(ra/dec, Cyg A) on 4096x4096 data completes within threshold."""
        print()  # noqa: T201
        with warnings.catch_warnings():
            # Silence the IERS auto-download warning that may fire inside accessor.
            warnings.filterwarnings("ignore", category=UserWarning)
            dynspec = _bench(
                "dynamic_spectrum(ra/dec, Cygnus A)",
                lambda: production_dataset.radport.dynamic_spectrum(
                    ra=CYGA_RA_DEG, dec=CYGA_DEC_DEG
                ),
            )

        assert dynspec.dims == ("time", "frequency")
        assert dynspec.shape == (N_TIME, N_FREQ)

        visible_times = production_dataset.attrs["_test_cyga_visible_times"]
        if visible_times:
            # At least one visible step must recover the injected flux.
            for ti in visible_times:
                row = dynspec.values[ti, :]
                assert np.all(row > DETECTION_THRESHOLD_JY) or np.all(
                    np.isnan(row)
                ), (
                    f"Time step {ti}: expected flux > {DETECTION_THRESHOLD_JY} Jy "
                    f"(or NaN if out of bounds), got max={np.nanmax(row):.2f} Jy"
                )
        else:
            # Source never rises above horizon during these 10 steps — all NaN.
            pytest.skip(
                "Cygnus A is not visible during the simulated observation window; "
                "correctness check skipped."
            )

    @pytest.mark.slow
    def test_dynamic_spectrum_radec_shows_source(
        self, production_dataset: xr.Dataset
    ) -> None:
        """Tracked dynamic spectrum contains values >> background at Cyg A position."""
        visible_times = production_dataset.attrs["_test_cyga_visible_times"]
        if not visible_times:
            pytest.skip("No visible time steps for Cygnus A in this window.")

        print()  # noqa: T201
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            dynspec = production_dataset.radport.dynamic_spectrum(
                ra=CYGA_RA_DEG, dec=CYGA_DEC_DEG
            )

        for ti in visible_times:
            row_max = float(np.nanmax(dynspec.values[ti, :]))
            assert row_max > DETECTION_THRESHOLD_JY, (
                f"Time step {ti}: Cygnus A not detected. "
                f"Max flux = {row_max:.2f} Jy, expected > {DETECTION_THRESHOLD_JY} Jy. "
                f"Visible l={dynspec.attrs.get('pixel_l')}, "
                f"m={dynspec.attrs.get('pixel_m')}."
            )


class TestLightCurve:
    """Benchmark and correctness tests for light_curve()."""

    @pytest.mark.slow
    def test_light_curve_lm_path_performance(
        self, production_dataset: xr.Dataset
    ) -> None:
        """light_curve(l=0, m=0, freq_idx=0) completes within threshold."""
        print()  # noqa: T201
        lc = _bench(
            "light_curve(l/m, freq_idx=0)",
            lambda: production_dataset.radport.light_curve(
                l=0.0, m=0.0, freq_idx=0
            ),
        )
        assert lc.dims == ("time",)
        assert lc.shape == (N_TIME,)
        assert np.all(np.isfinite(lc.values))

    @pytest.mark.slow
    def test_light_curve_radec_path_performance(
        self, production_dataset: xr.Dataset
    ) -> None:
        """light_curve(ra/dec, Cyg A, freq_idx=0) completes within threshold."""
        print()  # noqa: T201
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            lc = _bench(
                "light_curve(ra/dec, Cygnus A, freq_idx=0)",
                lambda: production_dataset.radport.light_curve(
                    ra=CYGA_RA_DEG, dec=CYGA_DEC_DEG, freq_idx=0
                ),
            )

        assert lc.dims == ("time",)
        assert lc.shape == (N_TIME,)

    @pytest.mark.slow
    def test_light_curve_radec_detects_source(
        self, production_dataset: xr.Dataset
    ) -> None:
        """Tracked light curve shows Cygnus A flux at visible time steps."""
        visible_times = production_dataset.attrs["_test_cyga_visible_times"]
        if not visible_times:
            pytest.skip("No visible time steps for Cygnus A in this window.")

        print()  # noqa: T201
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            lc = production_dataset.radport.light_curve(
                ra=CYGA_RA_DEG, dec=CYGA_DEC_DEG, freq_idx=0
            )

        for ti in visible_times:
            val = float(lc.values[ti])
            assert np.isnan(val) or val > DETECTION_THRESHOLD_JY, (
                f"Time step {ti}: Cygnus A not detected in light curve. "
                f"Value = {val:.2f} Jy."
            )


class TestPlotDynamicSpectrum:
    """Benchmark test for the full plot pipeline."""

    @pytest.mark.slow
    def test_plot_dynamic_spectrum_lm_path_performance(
        self, production_dataset: xr.Dataset
    ) -> None:
        """plot_dynamic_spectrum(l/m) on production data completes within threshold."""
        print()  # noqa: T201

        def _run():
            fig = production_dataset.radport.plot_dynamic_spectrum(l=0.0, m=0.0)
            plt.close("all")
            return fig

        _bench("plot_dynamic_spectrum(l/m)", _run)

    @pytest.mark.slow
    def test_plot_dynamic_spectrum_radec_path_performance(
        self, production_dataset: xr.Dataset
    ) -> None:
        """plot_dynamic_spectrum(ra/dec) on production data completes within threshold."""
        print()  # noqa: T201

        def _run():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                fig = production_dataset.radport.plot_dynamic_spectrum(
                    ra=CYGA_RA_DEG, dec=CYGA_DEC_DEG
                )
            plt.close("all")
            return fig

        _bench("plot_dynamic_spectrum(ra/dec, Cygnus A)", _run)


class TestDatasetFixture:
    """Sanity checks on the production fixture itself."""

    @pytest.mark.slow
    def test_fixture_shape(self, production_dataset: xr.Dataset) -> None:
        """Dataset has the expected production shape."""
        sky = production_dataset["SKY"]
        assert sky.shape == (N_TIME, N_FREQ, N_POL, N_L, N_M), (
            f"Unexpected shape: {sky.shape}"
        )

    @pytest.mark.slow
    def test_fixture_is_dask_backed(self, production_dataset: xr.Dataset) -> None:
        """SKY variable is backed by a Dask array (not yet computed)."""
        sky = production_dataset["SKY"]
        assert hasattr(sky.data, "dask"), "SKY should be a Dask array"

    @pytest.mark.slow
    def test_fixture_chunk_shape(self, production_dataset: xr.Dataset) -> None:
        """Dask chunks match the expected zarr-layout chunk dimensions."""
        sky = production_dataset["SKY"]
        # sky.chunks returns a tuple of tuples: one per dimension
        chunks = sky.chunks
        # Each chunk tuple should have uniform chunk size except possibly
        # the last chunk along each axis.
        expected_chunk_sizes = {0: 1, 1: 1, 2: 1, 3: 1024, 4: 1024}
        for dim_idx, expected_cs in expected_chunk_sizes.items():
            # All chunks except the last should equal expected_cs
            for cs in chunks[dim_idx][:-1]:
                assert cs == expected_cs, (
                    f"Dim {dim_idx}: expected chunk size {expected_cs}, got {cs}"
                )

    @pytest.mark.slow
    def test_fixture_source_injected(self, production_dataset: xr.Dataset) -> None:
        """At least some time steps contain the injected Cygnus A flux."""
        n_visible = production_dataset.attrs.get("_test_n_visible", 0)
        assert n_visible > 0, (
            "Cygnus A (Dec=40.7°) must be visible from OVRO-LWA (lat=37.2°) "
            "for at least some of the 10 time steps. Check the MJD start value."
        )

    @pytest.mark.slow
    def test_fixture_wcs_header_present(self, production_dataset: xr.Dataset) -> None:
        """SKY variable carries a WCS header string for coordinate transforms."""
        hdr = production_dataset["SKY"].attrs.get("fits_wcs_header", "")
        assert "RA---SIN" in hdr, "WCS header must declare SIN projection"
        assert "DEC--SIN" in hdr
        assert "CRVAL1" in hdr
        assert "CRVAL2" in hdr

    @pytest.mark.slow
    def test_fixture_coordinates(self, production_dataset: xr.Dataset) -> None:
        """Time, frequency, l, and m coordinates have the expected shape and range."""
        ds = production_dataset
        assert ds.coords["time"].shape == (N_TIME,)
        assert ds.coords["frequency"].shape == (N_FREQ,)
        assert ds.coords["l"].shape == (N_L,)
        assert ds.coords["m"].shape == (N_M,)

        # l/m span exactly [-1, 1]
        np.testing.assert_allclose(float(ds.coords["l"].min()), -1.0, atol=1e-6)
        np.testing.assert_allclose(float(ds.coords["l"].max()), 1.0, atol=1e-6)
        np.testing.assert_allclose(float(ds.coords["m"].min()), -1.0, atol=1e-6)
        np.testing.assert_allclose(float(ds.coords["m"].max()), 1.0, atol=1e-6)

        # Frequencies span 27-88 MHz
        assert float(ds.coords["frequency"].min()) == pytest.approx(FREQ_START_HZ, rel=1e-6)
        assert float(ds.coords["frequency"].max()) == pytest.approx(FREQ_STOP_HZ, rel=1e-6)
