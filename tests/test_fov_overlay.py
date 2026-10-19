"""Unit tests for FOV cutout helpers (no large Zarr required)."""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import xarray as xr
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from ovro_lwa_portal import fov_overlay
from ovro_lwa_portal.fov_overlay import (
    CachedCutout,
    FovCutoutSpec,
    FovOverlaySession,
    adjust_wcs_for_cutout,
    celestial_wcs_for_time,
    chunk_align_bbox,
    fov_pixel_bbox,
    read_sky_tile_parallel,
)


def _sin_wcs(*, n_l: int = 200, n_m: int = 200, cdelt_deg: float = 0.05) -> WCS:
    """Simple SIN WCS: axis 0 ↔ l, axis 1 ↔ m."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    w.wcs.crval = [180.0, 40.0]
    w.wcs.cdelt = [-float(cdelt_deg), float(cdelt_deg)]
    w.wcs.crpix = [n_l / 2 + 0.5, n_m / 2 + 0.5]
    w.wcs.cunit = ["deg", "deg"]
    return w


class _Sky:
    def __init__(self) -> None:
        self.opacity = 1.0
        self.view_ra = 180.0
        self.view_dec = 40.0
        self.view_fov = 8.0
        self.time_idx = 0
        self.freq_idx = 0
        self.click_tick = 0
        self.clicked_coord = (0.0, 0.0)
        self.crosshair_ra = -999.0
        self.crosshair_dec = -999.0
        self.pushes: list[tuple[np.ndarray, WCS, dict[str, object]]] = []
        self.sent = 0

    def view_center_skycoord(self) -> SkyCoord:
        return SkyCoord(self.view_ra, self.view_dec, unit="deg", frame="icrs")

    def push_shader_frame(self, data: np.ndarray, wcs: WCS, **kwargs: object) -> None:
        self.pushes.append((np.asarray(data), wcs, kwargs))

    def send_state(self) -> None:
        self.sent += 1


class _Cube:
    def __init__(self, data: np.ndarray, chunk: int = 10) -> None:
        self._data = data
        self.shape = data.shape
        self.chunks = (1, 1, 1, chunk, chunk)

    def __getitem__(self, key: object) -> np.ndarray:
        return self._data[key]


def test_newline_separated_header_keeps_crval(monkeypatch: pytest.MonkeyPatch) -> None:
    """80-column cards plus newlines must not be read as a fixed 80-byte grid."""
    header = _sin_wcs().to_header()
    header["CRVAL1"] = 172.5
    header["CRVAL2"] = 37.215849166667
    ds = xr.Dataset({"fits_header_str": ("time", [header.tostring(sep="\n")])})
    monkeypatch.setattr(fov_overlay, "_wcs_from_astrowidget", lambda *_a, **_k: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wcs = celestial_wcs_for_time(ds, 0)
    assert wcs is not None
    assert wcs.wcs.crval[0] == pytest.approx(172.5)
    assert wcs.wcs.crval[1] == pytest.approx(37.215849166667)
    assert not any("unrecognized non-standard convention" in str(item.message) for item in caught)


def test_fov_pixel_bbox_contains_center() -> None:
    n = 400
    w = _sin_wcs(n_l=n, n_m=n)
    l0, l1, m0, m1 = fov_pixel_bbox(w, 180.0, 40.0, 2.0, n, n)
    mid = n // 2
    assert l0 < mid < l1
    assert m0 < mid < m1


def test_chunk_align_expands_across_a_boundary() -> None:
    spec = chunk_align_bbox(9, 11, 0, 5, 100, 100, 10, 10)
    assert spec.l_slice == slice(0, 20)
    assert spec.n_chunks_l == 2
    inside = chunk_align_bbox(10, 11, 0, 5, 100, 100, 10, 10)
    assert inside.l_slice == slice(10, 20)
    assert inside.n_chunks_l == 1


def test_adjust_wcs_for_cutout_shifts_crpix() -> None:
    w = _sin_wcs()
    w.wcs.crpix = [100.0, 200.0]
    out = adjust_wcs_for_cutout(w, 10, 20)
    assert list(out.wcs.crpix) == [90.0, 180.0]
    assert list(w.wcs.crpix) == [100.0, 200.0]


def test_parallel_read_matches_the_plane() -> None:
    plane = np.arange(20 * 20, dtype=np.float32).reshape(1, 1, 1, 20, 20)
    cube = _Cube(plane, chunk=10)
    with ThreadPoolExecutor(max_workers=2) as pool:
        got = read_sky_tile_parallel(cube, 0, 0, 0, slice(0, 20), slice(0, 20), pool)
    assert got.shape == (20, 20)
    assert np.allclose(got, plane[0, 0, 0])


def test_push_warps_at_phase_center_and_does_not_pass_a_view_center(monkeypatch) -> None:
    cut = _sin_wcs(n_l=8, n_m=8)
    center = SkyCoord(10.0, -5.0, unit="deg", frame="icrs")
    recorded: dict[str, float] = {}

    def _project(data: np.ndarray, cut_wcs: WCS) -> tuple[np.ndarray, WCS, float]:
        recorded["ra"] = float(cut_wcs.wcs.crval[0])
        recorded["dec"] = float(cut_wcs.wcs.crval[1])
        return data, cut_wcs, 1.0

    monkeypatch.setattr(fov_overlay, "_project_native_sin", _project)
    sky = _Sky()
    sky.view_ra = 1.0
    sky.view_dec = 2.0
    sky.view_fov = 3.0
    data = np.ones((4, 4), dtype=np.float32)
    fov_overlay._push_cutout_frame(
        sky,
        data,
        cut,
        center,
        8.0,
        time_idx=4,
        freq_idx=2,
        update_view=False,
    )
    assert recorded["ra"] == 180.0
    assert recorded["dec"] == 40.0
    assert sky.view_ra == 1.0
    assert sky.view_dec == 2.0
    assert sky.view_fov == 3.0
    _data, _wcs, kwargs = sky.pushes[0]
    assert kwargs["time_idx"] == 4
    assert kwargs["freq_idx"] == 2
    assert "center" not in kwargs
    assert "fov" not in kwargs


def test_crosshair_skycoord_prefers_click_over_trait_and_view() -> None:
    sky = _Sky()
    sky.view_ra = 9.0
    sky.view_dec = 9.0
    assert fov_overlay.crosshair_skycoord(sky) is None

    sky.crosshair_ra = 5.0
    sky.crosshair_dec = 6.0
    trait = fov_overlay.crosshair_skycoord(sky)
    assert trait is not None
    assert trait.ra.deg == 5.0
    assert trait.dec.deg == 6.0

    sky.click_tick = 1
    sky.clicked_coord = (12.0, 34.0)
    clicked = fov_overlay.crosshair_skycoord(sky)
    assert clicked is not None
    assert clicked.ra.deg == 12.0
    assert clicked.dec.deg == 34.0


def _session(data: np.ndarray) -> tuple[FovOverlaySession, _Sky, list[str]]:
    sky = _Sky()
    logs: list[str] = []
    session = FovOverlaySession(
        sky,
        ds=None,  # type: ignore[arg-type]
        sky_arr=_Cube(data),
        debounce_s=0.0,
        log=logs.append,
    )
    return session, sky, logs


def test_reload_time_freq_fades_then_restores_without_moving_the_view(monkeypatch) -> None:
    monkeypatch.setattr(fov_overlay, "celestial_wcs_for_time", lambda *_a, **_k: _sin_wcs(n_l=80, n_m=80))
    monkeypatch.setattr(
        fov_overlay,
        "_project_native_sin",
        lambda data, cut_wcs: (data, cut_wcs, 0.5),
    )
    seen: list[float] = []
    real = fov_overlay.extract_fov_cutout

    def _extract(*args: object, **kwargs: object) -> np.ndarray:
        seen.append(session.sky.opacity)
        return real(*args, **kwargs)

    monkeypatch.setattr(fov_overlay, "extract_fov_cutout", _extract)
    data = np.ones((2, 2, 1, 80, 80), dtype=np.float32)
    session, sky, logs = _session(data)
    ra, dec, fov = sky.view_ra, sky.view_dec, sky.view_fov
    session.reload_time_freq(1, 0)
    assert seen == [0.35]
    assert sky.opacity == 1.0
    assert (sky.view_ra, sky.view_dec, sky.view_fov) == (ra, dec, fov)
    assert len(sky.pushes) == 1
    assert any("[slice-fade]" in line for line in logs)
    assert any("[slice]" in line for line in logs)


def test_empty_slice_read_keeps_the_previous_frame(monkeypatch) -> None:
    monkeypatch.setattr(fov_overlay, "celestial_wcs_for_time", lambda *_a, **_k: _sin_wcs(n_l=80, n_m=80))
    data = np.full((2, 2, 1, 80, 80), np.nan, dtype=np.float32)
    session, sky, logs = _session(data)
    previous = np.full((4, 4), 7.0, dtype=np.float32)
    session.cache = CachedCutout(
        data=previous,
        cut_wcs=_sin_wcs(n_l=4, n_m=4),
        spec=FovCutoutSpec(slice(0, 4), slice(0, 4), 4, 4, 1, 1),
        time_idx=0,
        freq_idx=0,
        pol_idx=0,
    )
    session.reload_time_freq(1, 0)
    assert session.cache is not None
    assert np.all(session.cache.data == 7.0)
    assert sky.pushes == []
    assert sky.opacity == 1.0
    assert any("empty" in line for line in logs)
