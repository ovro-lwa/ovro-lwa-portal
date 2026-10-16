"""FOV-native SkyWidget overlays for Source Review.

Ported from the ``async-test`` experiment in ``notebooks/async_cube_slice``.
A cutout is read at native resolution, warped once to naive SIN at that slice's
phase center, then uploaded with :meth:`astrowidget.SkyWidget.push_shader_frame`
so the widget does not warp again onto the view.

Typical flow
------------
1. Per-time celestial WCS for the full plane.
2. ``fov_pixel_bbox`` + ``chunk_align_bbox`` → ``(l_slice, m_slice)``.
3. ``extract_fov_cutout`` at native resolution.
4. ``adjust_wcs_for_cutout`` so CRPIX matches the cutout array.
5. One naive-SIN warp at the slice phase center, then ``push_shader_frame``.

Pan / zoom feedback (:class:`FovOverlaySession`)
-----------------------------------------------
* Zoom or pan that stays inside the cached cutout does not touch the overlay.
  The shader already samples that texture at the new view.
* A pan whose center leaves the cache fades the overlay until a new cutout
  arrives. A zoom that keeps the center inside the cache stays at full opacity.
* Leaving the cache debounces, then reads a new FOV cutout once.
* An empty read never replaces a cutout that already has data.
"""

from __future__ import annotations

import contextlib
import itertools
import threading
import time
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np
import xarray as xr
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from ovro_lwa_portal.accessor import _read_wcs_header_str

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True)
class FovCutoutSpec:
    """Chunk-aligned native cutout for one ``(t, f, p)`` plane."""

    l_slice: slice
    m_slice: slice
    n_l: int
    n_m: int
    n_chunks_l: int
    n_chunks_m: int

    @property
    def n_chunks(self) -> int:
        return int(self.n_chunks_l * self.n_chunks_m)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_l, self.n_m)


@dataclass
class CachedCutout:
    """Last native cutout. Views inside it are drawn from the current texture."""

    data: np.ndarray
    cut_wcs: WCS
    spec: FovCutoutSpec
    time_idx: int
    freq_idx: int
    pol_idx: int


def fov_pixel_bbox(
    wcs: WCS,
    ra_deg: float,
    dec_deg: float,
    fov_deg: float,
    n_l: int,
    n_m: int,
    *,
    pad_frac: float = 0.05,
) -> tuple[int, int, int, int]:
    """Return half-open ``(l0, l1, m0, m1)`` covering a square FOV around a sky point.

    Uses four cardinal offsets at ``fov_deg / 2`` plus the center. WCS axis 0
    corresponds to ``l`` and axis 1 to ``m``.

    Out-of-sky / SIN-invalid corners are ignored when ``world_to_pixel`` yields
    non-finite values. If nothing maps, falls back to a 1x1 box at the array
    center so callers always get a valid region.
    """
    center = SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg, frame="icrs")
    half = 0.5 * float(fov_deg) * u.deg
    bearings = [0.0, 90.0, 180.0, 270.0] * u.deg
    samples = [center.directional_offset_by(b, half) for b in bearings]
    samples.append(center)

    ra = np.array([float(c.icrs.ra.deg) for c in samples], dtype=np.float64)
    dec = np.array([float(c.icrs.dec.deg) for c in samples], dtype=np.float64)
    pix_l, pix_m = wcs.world_to_pixel_values(ra, dec)
    pix_l = np.asarray(pix_l, dtype=np.float64)
    pix_m = np.asarray(pix_m, dtype=np.float64)
    ok = np.isfinite(pix_l) & np.isfinite(pix_m)
    if not np.any(ok):
        mid_l = n_l // 2
        mid_m = n_m // 2
        return mid_l, mid_l + 1, mid_m, mid_m + 1

    l_min = float(np.min(pix_l[ok]))
    l_max = float(np.max(pix_l[ok]))
    m_min = float(np.min(pix_m[ok]))
    m_max = float(np.max(pix_m[ok]))
    if l_max <= l_min:
        l_max = l_min + 1.0
    if m_max <= m_min:
        m_max = m_min + 1.0
    pad_l = pad_frac * (l_max - l_min)
    pad_m = pad_frac * (m_max - m_min)
    l0 = int(np.floor(l_min - pad_l))
    l1 = int(np.ceil(l_max + pad_l))
    m0 = int(np.floor(m_min - pad_m))
    m1 = int(np.ceil(m_max + pad_m))
    l0 = max(0, min(l0, n_l - 1))
    m0 = max(0, min(m0, n_m - 1))
    l1 = max(l0 + 1, min(l1, n_l))
    m1 = max(m0 + 1, min(m1, n_m))
    return l0, l1, m0, m1


def chunk_align_bbox(
    l0: int,
    l1: int,
    m0: int,
    m1: int,
    n_l: int,
    n_m: int,
    chunk_l: int,
    chunk_m: int,
) -> FovCutoutSpec:
    """Expand a pixel bbox to on-disk chunk boundaries (clipped to the array)."""
    cl = max(1, int(chunk_l))
    cm = max(1, int(chunk_m))
    l0a = max(0, (int(l0) // cl) * cl)
    m0a = max(0, (int(m0) // cm) * cm)
    l1a = min(int(n_l), ((int(l1) + cl - 1) // cl) * cl)
    m1a = min(int(n_m), ((int(m1) + cm - 1) // cm) * cm)
    if l1a <= l0a:
        l1a = min(n_l, l0a + cl)
    if m1a <= m0a:
        m1a = min(n_m, m0a + cm)
    n_chunks_l = max(1, (l1a - l0a + cl - 1) // cl)
    n_chunks_m = max(1, (m1a - m0a + cm - 1) // cm)
    return FovCutoutSpec(
        l_slice=slice(l0a, l1a),
        m_slice=slice(m0a, m1a),
        n_l=l1a - l0a,
        n_m=m1a - m0a,
        n_chunks_l=n_chunks_l,
        n_chunks_m=n_chunks_m,
    )


def plan_fov_cutout(
    wcs: WCS,
    ra_deg: float,
    dec_deg: float,
    fov_deg: float,
    *,
    n_l: int,
    n_m: int,
    chunk_l: int,
    chunk_m: int,
    pad_frac: float = 0.05,
) -> FovCutoutSpec:
    """FOV → chunk-aligned :class:`FovCutoutSpec`."""
    l0, l1, m0, m1 = fov_pixel_bbox(wcs, ra_deg, dec_deg, fov_deg, n_l, n_m, pad_frac=pad_frac)
    return chunk_align_bbox(l0, l1, m0, m1, n_l, n_m, chunk_l, chunk_m)


def cutout_covers_view(
    cut_wcs: WCS,
    n_l: int,
    n_m: int,
    ra_deg: float,
    dec_deg: float,
    fov_deg: float,
    *,
    margin: float = 1.02,
) -> bool:
    """True when ``cut_wcs`` already contains the view center and FOV corners."""
    center = SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg, frame="icrs")
    half = 0.5 * float(fov_deg) * float(margin) * u.deg
    bearings = [0.0, 90.0, 180.0, 270.0] * u.deg
    samples = [center]
    samples.extend(center.directional_offset_by(b, half) for b in bearings)

    ra = np.array([float(c.icrs.ra.deg) for c in samples], dtype=np.float64)
    dec = np.array([float(c.icrs.dec.deg) for c in samples], dtype=np.float64)
    pix_l, pix_m = cut_wcs.world_to_pixel_values(ra, dec)
    pix_l = np.asarray(pix_l, dtype=np.float64)
    pix_m = np.asarray(pix_m, dtype=np.float64)
    inside = (
        np.isfinite(pix_l)
        & np.isfinite(pix_m)
        & (pix_l >= 0.0)
        & (pix_l <= float(n_l - 1))
        & (pix_m >= 0.0)
        & (pix_m <= float(n_m - 1))
    )
    return bool(np.all(inside))


def adjust_wcs_for_cutout(wcs: WCS, l0: int, m0: int) -> WCS:
    """Return a WCS describing ``data[l0:, m0:]`` (same CDELT/CRVAL, shifted CRPIX).

    0-based cutout origin ``(l0, m0)`` maps full-plane FITS pixel
    ``P = i + 1`` to cutout FITS pixel ``P' = (i - l0) + 1``, so
    ``CRPIX' = CRPIX - origin``.
    """
    out = wcs.deepcopy()
    out.wcs.crpix = [
        float(wcs.wcs.crpix[0]) - float(l0),
        float(wcs.wcs.crpix[1]) - float(m0),
    ]
    return out


def read_sky_tile_zarr(
    array: Any,
    t: int,
    f: int,
    p: int,
    l_slice: slice,
    m_slice: slice,
) -> np.ndarray:
    """Read one ``(l, m)`` tile from a 5-D ``(time, frequency, pol, l, m)`` array."""
    return np.asarray(array[t, f, p, l_slice, m_slice], dtype=np.float32)


def _chunk_aligned_ranges(start: int, stop: int, chunk: int) -> list[tuple[int, int]]:
    edges = [start, *range((start // chunk + 1) * chunk, stop, chunk), stop]
    return list(itertools.pairwise(edges))


def read_sky_tile_parallel(
    array: Any,
    t: int,
    f: int,
    p: int,
    l_slice: slice,
    m_slice: slice,
    executor: Executor,
) -> np.ndarray:
    """Read a sky tile as one task per spatial chunk (zarr v2 reads chunks serially)."""
    if l_slice.start is None or l_slice.stop is None or m_slice.start is None or m_slice.stop is None:
        msg = "sky tile slices must be bounded"
        raise ValueError(msg)
    l_ranges = _chunk_aligned_ranges(l_slice.start, l_slice.stop, int(array.chunks[3]))
    m_ranges = _chunk_aligned_ranges(m_slice.start, m_slice.stop, int(array.chunks[4]))
    if len(l_ranges) * len(m_ranges) == 1:
        return executor.submit(read_sky_tile_zarr, array, t, f, p, l_slice, m_slice).result()
    out = np.empty((l_slice.stop - l_slice.start, m_slice.stop - m_slice.start), dtype=np.float32)

    def _read(lr: tuple[int, int], mr: tuple[int, int]) -> None:
        l_dst = slice(lr[0] - l_slice.start, lr[1] - l_slice.start)
        m_dst = slice(mr[0] - m_slice.start, mr[1] - m_slice.start)
        out[l_dst, m_dst] = array[t, f, p, lr[0] : lr[1], mr[0] : mr[1]]

    futures = [executor.submit(_read, lr, mr) for lr in l_ranges for mr in m_ranges]
    for fut in futures:
        fut.result()
    return out


def _axis_chunk(sky: xr.DataArray, dim: str) -> int:
    """On-disk chunk length along ``dim``, or the full axis when the array is one piece."""
    size = int(sky.sizes[dim])
    chunks = getattr(sky.data, "chunks", None)
    if not chunks:
        return size
    axis = int(sky.get_axis_num(dim))
    if axis >= len(chunks):
        return size
    block = chunks[axis]
    if isinstance(block, tuple):
        return max(1, int(block[0])) if block else size
    return max(1, int(block))


class SkyPlaneReader:
    """5-D ``(time, frequency, pol, l, m)`` view of a SKY DataArray.

    Datasets without a polarization axis are exposed as one Stokes plane so
    tile reads keep a single index order.
    """

    def __init__(self, sky: xr.DataArray) -> None:
        self._sky = sky
        self._has_pol = "polarization" in sky.dims
        n_pol = int(sky.sizes["polarization"]) if self._has_pol else 1
        self.shape = (
            int(sky.sizes["time"]),
            int(sky.sizes["frequency"]),
            n_pol,
            int(sky.sizes["l"]),
            int(sky.sizes["m"]),
        )
        self.chunks = (1, 1, 1, _axis_chunk(sky, "l"), _axis_chunk(sky, "m"))

    def __getitem__(self, key: tuple[Any, ...]) -> np.ndarray:
        t, f, p, l_slice, m_slice = key
        sel: dict[str, Any] = {"time": t, "frequency": f, "l": l_slice, "m": m_slice}
        if self._has_pol:
            sel["polarization"] = p
        return np.asarray(self._sky.isel(sel).values, dtype=np.float32)


def extract_fov_cutout(
    array: Any,
    t: int,
    f: int,
    p: int,
    l_slice: slice,
    m_slice: slice,
    *,
    executor: Executor | None = None,
) -> np.ndarray:
    """Read a native ``(l, m)`` cutout; fan out per spatial chunk when ``executor`` set."""
    if executor is None:
        return read_sky_tile_zarr(array, t, f, p, l_slice, m_slice)
    return read_sky_tile_parallel(array, t, f, p, l_slice, m_slice, executor)


def celestial_wcs_for_time(ds: xr.Dataset, time_idx: int, *, freq_idx: int = 0) -> WCS | None:
    """Build a 2-D celestial WCS for ``time_idx``.

    Prefers the portal-patched astrowidget lookup. An empty per-time header
    returns ``None`` rather than falling back to time 0.
    """
    wcs = _wcs_from_astrowidget(ds, int(time_idx))
    if wcs is not None:
        return wcs
    header_str = _read_wcs_header_str(ds, time_idx=int(time_idx), freq_idx=int(freq_idx))
    if not header_str:
        return None
    header = fits.Header.fromstring(header_str)
    return WCS(header).celestial


def _wcs_from_astrowidget(ds: xr.Dataset, time_idx: int) -> WCS | None:
    try:
        from astrowidget.wcs import get_wcs  # noqa: PLC0415
    except ImportError:
        return None
    try:
        wcs = get_wcs(ds, var="SKY", time_idx=int(time_idx))
    except Exception:  # noqa: BLE001 — a missing widget WCS falls through to the portal header
        return None
    if isinstance(wcs, WCS):
        return wcs
    return None


def crosshair_skycoord(sky: Any) -> SkyCoord | None:
    """Return the placed crosshair, or ``None`` when no marker is set.

    A canvas click stores the on-screen marker in ``clicked_coord`` (``click_tick``
    advances). ``set_crosshair`` stores it on ``crosshair_ra`` / ``crosshair_dec``
    (values ``<= -900`` hide the marker). A click wins, because that is the marker
    currently drawn. This is not the view center.
    """
    if int(getattr(sky, "click_tick", 0) or 0) > 0:
        clicked = getattr(sky, "clicked_coord", None)
        if clicked is not None and len(clicked) >= 2:
            ra, dec = float(clicked[0]), float(clicked[1])
            if np.isfinite(ra) and np.isfinite(dec):
                return SkyCoord(ra, dec, unit="deg", frame="icrs")
    ra = float(getattr(sky, "crosshair_ra", -999.0))
    dec = float(getattr(sky, "crosshair_dec", -999.0))
    if ra <= -900.0 or dec <= -900.0 or not np.isfinite(ra) or not np.isfinite(dec):
        return None
    return SkyCoord(ra, dec, unit="deg", frame="icrs")


def _set_opacity(sky: Any, opacity: float) -> None:
    if hasattr(sky, "opacity"):
        sky.opacity = float(opacity)
    if hasattr(sky, "send_state"):
        sky.send_state()


def _project_native_sin(data: np.ndarray, cut_wcs: WCS) -> tuple[np.ndarray, WCS, float]:
    """One naive-SIN warp at the slice phase center (not the current view)."""
    from astrowidget.wcs import reproject_for_shader_display  # noqa: PLC0415

    phase_ra = float(cut_wcs.wcs.crval[0])
    phase_dec = float(cut_wcs.wcs.crval[1])
    t0 = time.perf_counter()
    projected, naive_wcs = reproject_for_shader_display(
        data,
        cut_wcs,
        crval_ra=phase_ra,
        crval_dec=phase_dec,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return np.asarray(projected, dtype=np.float32), naive_wcs, elapsed_ms


def _push_cutout_frame(
    sky: Any,
    data: np.ndarray,
    cut_wcs: WCS,
    center: SkyCoord,
    fov_deg: float,
    *,
    time_idx: int | None = None,
    freq_idx: int | None = None,
    update_view: bool = True,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> dict[str, float | bool]:
    """Push a cutout after one phase-center SIN warp.

    :meth:`~astrowidget.SkyWidget.push_shader_frame` does not take a view
    center, so it does not warp again. The shader samples this texture for
    later zoom and slew. ``update_view`` only moves ``view_ra`` / ``view_dec``
    / ``view_fov``.
    """
    t0 = time.perf_counter()
    projected, naive_wcs, reproject_ms = _project_native_sin(data, cut_wcs)
    sky.push_shader_frame(
        projected,
        naive_wcs,
        time_idx=time_idx,
        freq_idx=freq_idx,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )
    if update_view:
        sky.view_ra = float(center.icrs.ra.deg)
        sky.view_dec = float(center.icrs.dec.deg)
        sky.view_fov = float(fov_deg)

    profile: dict[str, float | bool] = {
        "push_ms": (time.perf_counter() - t0) * 1000.0,
        "reproject_ms": float(reproject_ms),
        "skipped_reproject": False,
    }
    last = getattr(sky, "_profile_last_push", None)
    if isinstance(last, dict) and "trait_ms" in last:
        profile["trait_ms"] = float(last["trait_ms"])
    return profile


def push_fov_overlay(
    sky: Any,
    ds: xr.Dataset,
    sky_arr: Any,
    time_idx: int,
    freq_idx: int,
    center: SkyCoord,
    fov_deg: float,
    *,
    pol_idx: int = 0,
    executor: Executor | None = None,
    pad_frac: float = 0.05,
    update_view: bool = True,
) -> dict[str, Any]:
    """Read an FOV cutout and push one phase-center SIN texture.

    The warp uses the slice ``CRVAL``, not the current view. Does not call
    ``PreloadedCube.image`` (no full-plane stride).

    Set ``update_view=False`` when reloading after a user pan/zoom so existing
    view traits are not overwritten. The returned dict includes ``_cache``.
    """
    n_l = int(sky_arr.shape[3])
    n_m = int(sky_arr.shape[4])
    chunk_l = int(sky_arr.chunks[3])
    chunk_m = int(sky_arr.chunks[4])

    t_all = time.perf_counter()
    wcs = celestial_wcs_for_time(ds, int(time_idx), freq_idx=int(freq_idx))
    if wcs is None:
        msg = f"No WCS for time_idx={time_idx}"
        raise RuntimeError(msg)

    ra = float(center.icrs.ra.deg)
    dec = float(center.icrs.dec.deg)
    spec = plan_fov_cutout(
        wcs,
        ra,
        dec,
        float(fov_deg),
        n_l=n_l,
        n_m=n_m,
        chunk_l=chunk_l,
        chunk_m=chunk_m,
        pad_frac=pad_frac,
    )

    t_zarr = time.perf_counter()
    data = extract_fov_cutout(
        sky_arr,
        int(time_idx),
        int(freq_idx),
        int(pol_idx),
        spec.l_slice,
        spec.m_slice,
        executor=executor,
    )
    zarr_ms = (time.perf_counter() - t_zarr) * 1000.0

    cut_wcs = adjust_wcs_for_cutout(wcs, int(spec.l_slice.start or 0), int(spec.m_slice.start or 0))
    push_prof = _push_cutout_frame(
        sky,
        data,
        cut_wcs,
        center,
        float(fov_deg),
        time_idx=int(time_idx),
        freq_idx=int(freq_idx),
        update_view=bool(update_view),
    )

    profile: dict[str, Any] = {
        "shape": spec.shape,
        "n_chunks": spec.n_chunks,
        "nbytes": int(data.nbytes),
        "zarr_ms": float(zarr_ms),
        "total_ms": (time.perf_counter() - t_all) * 1000.0,
        "l0": int(spec.l_slice.start or 0),
        "m0": int(spec.m_slice.start or 0),
        "_cache": CachedCutout(
            data=data,
            cut_wcs=cut_wcs,
            spec=spec,
            time_idx=int(time_idx),
            freq_idx=int(freq_idx),
            pol_idx=int(pol_idx),
        ),
    }
    profile.update(push_prof)
    return profile


def attach_fov_pan_reload(sky: Any, on_view_changed: Callable[[SkyCoord, float], None]) -> Any:
    """Reload FOV tiles when the user finishes a pan/zoom on ``sky``.

    JS only increments ``view_gesture_revision`` when ``overlay_view_lock`` is
    true. The built-in SkyWidget handler then reprojects the full-plane
    ``PreloadedCube`` slice, so this helper removes that handler and installs
    ``on_view_changed(center, fov_deg)`` instead.
    """
    sky.overlay_view_lock = True
    with contextlib.suppress(ValueError, AttributeError):
        sky.unobserve(sky._on_view_gesture_revision, names=["view_gesture_revision"])

    def _handler(change: dict[str, Any]) -> None:
        if change.get("name") != "view_gesture_revision":
            return
        if getattr(sky, "_suppress_gesture_observer", False):
            return
        center = sky.view_center_skycoord()
        fov_deg = float(sky.view_fov)
        on_view_changed(center, fov_deg)

    sky.observe(_handler, names="view_gesture_revision")
    sky._fov_pan_reload_handler = _handler
    return _handler


class FovOverlaySession:
    """FOV overlay that reloads only when the view leaves the cached cutout.

    * **slice / catalog push** — Zarr FOV read, cache the native cutout.
    * **pan / zoom inside the cache** — no reproject and no Zarr read.
    * **pan off the cached view** — fade immediately, then one debounced read.
      This includes a pan whose center is still on the padded cutout when the
      new field no longer fits. Opacity returns when the new cutout is shown.
    * **zoom out past the cache** — center is still valid, so no fade; one
      debounced read extends the cutout.
    * Empty results are dropped so a good overlay is never replaced by a blank one.

    ``executor`` runs one read job per cutout. ``tile_executor`` runs that job's
    per-chunk reads and must be a different pool, since the job waits on them.
    """

    def __init__(
        self,
        sky: Any,
        ds: xr.Dataset,
        sky_arr: Any,
        *,
        pol_idx: int = 0,
        executor: Executor | None = None,
        tile_executor: Executor | None = None,
        pad_frac: float = 0.05,
        pan_pad_frac: float = 0.5,
        debounce_s: float = 0.3,
        fade_opacity: float = 0.35,
        schedule_main: Callable[[Callable[[], None]], None] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.sky = sky
        self.ds = ds
        self.sky_arr = sky_arr
        self.pol_idx = int(pol_idx)
        self.executor = executor
        self.tile_executor = tile_executor
        self.pad_frac = float(pad_frac)
        self.pan_pad_frac = float(pan_pad_frac)
        self.debounce_s = float(debounce_s)
        self.fade_opacity = float(fade_opacity)
        self.schedule_main = schedule_main or (lambda cb: cb())
        self.log = log or (lambda _msg: None)
        self.cache: CachedCutout | None = None
        self._tf = {"t": 0, "f": 0}
        self._gen = 0
        self._pending: Future[Any] | None = None
        self._timer: threading.Timer | None = None
        self._logged_skip = False
        self._faded = False
        self._anchor: SkyCoord | None = None
        self.on_settled: Callable[[], None] | None = None
        self.percentile_low = 1.0
        self.percentile_high = 99.0

    def _notify_settled(self) -> None:
        callback = self.on_settled
        if callback is not None:
            callback()

    def set_slice_indices(self, t: int, f: int) -> None:
        self._tf["t"], self._tf["f"] = int(t), int(f)

    def push_slice(
        self,
        center: SkyCoord,
        fov_deg: float,
        *,
        t: int | None = None,
        f: int | None = None,
        update_view: bool = True,
        reason: str = "slice",
    ) -> dict[str, Any]:
        """Catalog / slice load. Updates the cached cutout."""
        if t is not None:
            self._tf["t"] = int(t)
        if f is not None:
            self._tf["f"] = int(f)
        t_idx, f_idx = self._tf["t"], self._tf["f"]
        t0 = time.perf_counter()
        prof = push_fov_overlay(
            self.sky,
            self.ds,
            self.sky_arr,
            t_idx,
            f_idx,
            center,
            float(fov_deg),
            pol_idx=self.pol_idx,
            executor=self.tile_executor or self.executor,
            pad_frac=self.pan_pad_frac,
            update_view=update_view,
        )
        cache = prof.pop("_cache", None)
        if isinstance(cache, CachedCutout):
            self.cache = cache
            self._logged_skip = False
        self._anchor = center
        self._restore_opacity()
        if update_view and hasattr(self.sky, "set_crosshair"):
            self.sky.set_crosshair(center)
        if hasattr(self.sky, "send_state"):
            self.sky.send_state()
        self.log(
            f"FOV overlay [{reason}] t={t_idx} f={f_idx} "
            f"view=({center.icrs.ra.deg:.3f},{center.icrs.dec.deg:.3f}) "
            f"fov={fov_deg:.2f}° shape={prof['shape']} chunks={prof['n_chunks']} "
            f"zarr={prof['zarr_ms']:.0f}ms reproj={prof.get('reproject_ms', float('nan')):.0f}ms "
            f"total={time.perf_counter() - t0:.3f}s"
        )
        return prof

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _center_on_cache(self, center: SkyCoord) -> bool:
        """True when the view center still lands on the cached cutout."""
        cache = self.cache
        if cache is None:
            return False
        return cutout_covers_view(
            cache.cut_wcs,
            int(cache.data.shape[0]),
            int(cache.data.shape[1]),
            float(center.icrs.ra.deg),
            float(center.icrs.dec.deg),
            0.0,
        )

    def _cache_covers(self, center: SkyCoord, fov_deg: float) -> bool:
        cache = self.cache
        if cache is None:
            return False
        return cutout_covers_view(
            cache.cut_wcs,
            int(cache.data.shape[0]),
            int(cache.data.shape[1]),
            float(center.icrs.ra.deg),
            float(center.icrs.dec.deg),
            float(fov_deg),
        )

    def _center_moved(self, center: SkyCoord, *, tol_deg: float = 0.05) -> bool:
        """True when this gesture is a pan, not a zoom around the same center."""
        anchor = self._anchor
        if anchor is None:
            return False
        return float(center.separation(anchor).deg) > float(tol_deg)

    def _fade_for_invalid_pan(self, center: SkyCoord, fov_deg: float, gen: int) -> None:
        if self._faded:
            return
        self._faded = True
        _set_opacity(self.sky, self.fade_opacity)
        self.log(
            f"FOV overlay [pan-fade] gen={gen} "
            f"view=({center.icrs.ra.deg:.3f},{center.icrs.dec.deg:.3f}) "
            f"fov={fov_deg:.2f}° pan left cached view"
        )

    def _restore_opacity(self) -> None:
        if not self._faded:
            return
        self._faded = False
        _set_opacity(self.sky, 1.0)

    def on_pan(self, center: SkyCoord, fov_deg: float) -> None:
        """Gesture end: keep a valid texture; fade only a pan off the cache."""
        self._gen += 1
        gen = self._gen
        self._cancel_timer()

        if self._cache_covers(center, float(fov_deg)):
            self._restore_opacity()
            if not self._logged_skip:
                self.log(
                    f"FOV overlay [pan-skip] gen={gen} "
                    f"view=({center.icrs.ra.deg:.3f},{center.icrs.dec.deg:.3f}) "
                    f"fov={fov_deg:.2f}° inside cache"
                )
                self._logged_skip = True
            return

        self._logged_skip = False
        if self.cache is not None and (
            self._center_moved(center) or not self._center_on_cache(center)
        ):
            self._fade_for_invalid_pan(center, float(fov_deg), gen)

        def _fire() -> None:
            self.schedule_main(lambda: self._load_view(gen, center, float(fov_deg)))

        if self.debounce_s <= 0:
            _fire()
            return
        timer = threading.Timer(self.debounce_s, _fire)
        timer.daemon = True
        self._timer = timer
        timer.start()

    def reload_time_freq(self, t: int, f: int) -> None:
        """Load ``(t, f)`` at the current view. Does not move the view or the crosshair.

        The overlay fades immediately. The cutout read runs on the executor so the
        fade can reach the browser before the new texture is pushed.
        """
        self.set_slice_indices(int(t), int(f))
        center = self.sky.view_center_skycoord()
        fov_deg = float(self.sky.view_fov)
        self._gen += 1
        gen = self._gen
        self._cancel_timer()
        if not self._faded:
            self._faded = True
            _set_opacity(self.sky, self.fade_opacity)
        self.log(
            f"FOV overlay [slice-fade] t={int(t)} f={int(f)} "
            f"view=({center.icrs.ra.deg:.3f},{center.icrs.dec.deg:.3f}) fov={fov_deg:.2f}°"
        )
        self._load_view(gen, center, fov_deg, reason="slice")

    def _load_view(
        self,
        gen: int,
        center: SkyCoord,
        fov_deg: float,
        *,
        reason: str = "pan",
    ) -> None:
        """Async FOV read. Apply on the io_loop. ``reason`` is ``pan`` or ``slice``."""
        if gen != self._gen:
            return
        t_idx, f_idx = self._tf["t"], self._tf["f"]

        def _read_only() -> tuple[int, np.ndarray, WCS, FovCutoutSpec, float]:
            t_z0 = time.perf_counter()
            wcs = celestial_wcs_for_time(self.ds, t_idx, freq_idx=f_idx)
            if wcs is None:
                msg = f"No WCS for time_idx={t_idx}"
                raise RuntimeError(msg)
            spec = plan_fov_cutout(
                wcs,
                float(center.icrs.ra.deg),
                float(center.icrs.dec.deg),
                float(fov_deg),
                n_l=int(self.sky_arr.shape[3]),
                n_m=int(self.sky_arr.shape[4]),
                chunk_l=int(self.sky_arr.chunks[3]),
                chunk_m=int(self.sky_arr.chunks[4]),
                pad_frac=self.pan_pad_frac,
            )
            data = extract_fov_cutout(
                self.sky_arr,
                t_idx,
                f_idx,
                self.pol_idx,
                spec.l_slice,
                spec.m_slice,
                executor=self.tile_executor,
            )
            l0 = int(spec.l_slice.start or 0)
            m0 = int(spec.m_slice.start or 0)
            cut_wcs = adjust_wcs_for_cutout(wcs, l0, m0)
            zarr_ms = (time.perf_counter() - t_z0) * 1000.0
            return gen, data, cut_wcs, spec, zarr_ms

        def _apply(result: tuple[int, np.ndarray, WCS, FovCutoutSpec, float]) -> None:
            res_gen, data, cut_wcs, spec, zarr_ms = result
            if res_gen != self._gen:
                self.log(f"FOV overlay [pan stale] gen={res_gen} (current={self._gen})")
                return
            if not np.any(np.isfinite(data)):
                self.log(
                    f"FOV overlay [{reason} empty] gen={res_gen} shape={spec.shape} "
                    "kept previous overlay"
                )
                if reason == "slice":
                    self._restore_opacity()
                    self._notify_settled()
                return
            t_b = time.perf_counter()
            push_prof = _push_cutout_frame(
                self.sky,
                data,
                cut_wcs,
                center,
                float(fov_deg),
                time_idx=t_idx,
                freq_idx=f_idx,
                update_view=False,
                percentile_low=self.percentile_low,
                percentile_high=self.percentile_high,
            )
            self.cache = CachedCutout(
                data=data,
                cut_wcs=cut_wcs,
                spec=spec,
                time_idx=t_idx,
                freq_idx=f_idx,
                pol_idx=self.pol_idx,
            )
            self._anchor = center
            self._restore_opacity()
            if hasattr(self.sky, "send_state"):
                self.sky.send_state()
            apply_ms = (time.perf_counter() - t_b) * 1000
            reproj_ms = push_prof.get("reproject_ms", float("nan"))
            self.log(
                f"FOV overlay [{reason}] gen={res_gen} shape={spec.shape} "
                f"chunks={spec.n_chunks} zarr={zarr_ms:.0f}ms "
                f"reproj={reproj_ms:.0f}ms apply={apply_ms:.0f}ms"
            )
            if reason == "slice":
                self._notify_settled()

        pool = self.executor
        if pool is None:
            _apply(_read_only())
            return

        fut = pool.submit(_read_only)

        def _done(done: Future[tuple[int, np.ndarray, WCS, FovCutoutSpec, float]]) -> None:
            try:
                result = done.result()
            except Exception as exc:  # noqa: BLE001 — worker errors stay on the log
                err = exc

                def _fail() -> None:
                    self.log(f"FOV overlay [{reason} error] {err}")
                    if reason == "slice":
                        self._restore_opacity()
                        self._notify_settled()

                self.schedule_main(_fail)
                return
            self.schedule_main(lambda: _apply(result))

        fut.add_done_callback(_done)
        self._pending = fut
