"""FOV-driven native-resolution spatial cutouts for SkyWidget overlay (Approach 1).

Experiment code on ``async-test``. Does not change production ``PreloadedCube`` /
``update_slice``. Reuses :func:`read_sky_tile_zarr` / :func:`read_sky_tile_parallel`.

Typical flow
------------
1. Per-time celestial WCS for the full plane.
2. ``fov_pixel_bbox`` + ``chunk_align_bbox`` → ``(l_slice, m_slice)``.
3. ``extract_fov_cutout`` at native resolution.
4. ``adjust_wcs_for_cutout`` so CRPIX matches the cutout array.
5. One naive-SIN warp at the slice phase center, then ``SkyWidget._push_image_frame``
   without a second view-centered warp.

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

import threading
import time
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from dynspec_async import read_sky_tile_parallel, read_sky_tile_zarr

if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS as AstropyWCS
    import xarray as xr


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
    cut_wcs: Any
    spec: FovCutoutSpec
    time_idx: int
    freq_idx: int
    pol_idx: int


def fov_pixel_bbox(
    wcs: AstropyWCS,
    ra_deg: float,
    dec_deg: float,
    fov_deg: float,
    n_l: int,
    n_m: int,
    *,
    pad_frac: float = 0.05,
) -> tuple[int, int, int, int]:
    """Return half-open ``(l0, l1, m0, m1)`` covering a square FOV around a sky point.

    Uses four cardinal offsets at ``fov_deg / 2`` plus the center. WCS axis 0 ↔
    ``l`` and axis 1 ↔ ``m`` (same convention as
    ``astrowidget.wcs.adjust_wcs_for_array_stride``).

    Out-of-sky / SIN-invalid corners are ignored when ``world_to_pixel`` yields
    non-finite values. If nothing maps, falls back to a 1×1 box at the array
    center so callers always get a valid region.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    center = SkyCoord(float(ra_deg) * u.deg, float(dec_deg) * u.deg, frame="icrs")
    half = 0.5 * float(fov_deg) * u.deg
    bearings = [0.0, 90.0, 180.0, 270.0] * u.deg
    samples = [center.directional_offset_by(b, half) for b in bearings]
    samples.append(center)

    ra = np.array([float(c.icrs.ra.deg) for c in samples], dtype=np.float64)
    dec = np.array([float(c.icrs.dec.deg) for c in samples], dtype=np.float64)
    # world_to_pixel_values: (NAXIS1, NAXIS2) = (l, m) for portal/astrowidget WCS.
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
    # Ensure a non-empty box even if all samples collapse to one pixel.
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
    wcs: AstropyWCS,
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
    l0, l1, m0, m1 = fov_pixel_bbox(
        wcs, ra_deg, dec_deg, fov_deg, n_l, n_m, pad_frac=pad_frac
    )
    return chunk_align_bbox(l0, l1, m0, m1, n_l, n_m, chunk_l, chunk_m)


def cutout_covers_view(
    cut_wcs: AstropyWCS,
    n_l: int,
    n_m: int,
    ra_deg: float,
    dec_deg: float,
    fov_deg: float,
    *,
    margin: float = 1.02,
) -> bool:
    """True when ``cut_wcs`` already contains the view center and FOV corners.

    A zoom-in (or a small pan) inside this footprint does not need another
    Zarr read or shader reproject: the texture on the widget still maps those
    sky coordinates.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord

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


def adjust_wcs_for_cutout(wcs: AstropyWCS, l0: int, m0: int) -> AstropyWCS:
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


def extract_fov_cutout(
    array,
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


def celestial_wcs_for_time(ds: xr.Dataset, time_idx: int, *, freq_idx: int = 0):
    """Build a 2-D celestial WCS for ``time_idx`` (portal-patched path preferred)."""
    try:
        from astrowidget.wcs import get_wcs

        wcs = get_wcs(ds, var="SKY", time_idx=int(time_idx))
        if wcs is not None:
            return wcs
    except Exception:
        pass
    # Fallback: portal header → astropy WCS (celestial only).
    from astropy.io import fits
    from astropy.wcs import WCS

    from ovro_lwa_portal.accessor import _read_wcs_header_str

    header_str = _read_wcs_header_str(ds, time_idx=int(time_idx), freq_idx=int(freq_idx))
    if not header_str:
        return None
    header = fits.Header.fromstring(header_str)
    return WCS(header).celestial


def crosshair_skycoord(sky: Any) -> SkyCoord | None:
    """Return the placed crosshair, or ``None`` when no marker is set.

    A canvas click stores the on-screen marker in ``clicked_coord`` (``click_tick``
    advances). ``set_crosshair`` stores it on ``crosshair_ra`` / ``crosshair_dec``
    (values ``<= -900`` hide the marker). A click wins, because that is the marker
    currently drawn. This is not the view center.
    """
    from astropy.coordinates import SkyCoord

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


def _project_native_sin(data: np.ndarray, cut_wcs: Any) -> tuple[np.ndarray, Any, float]:
    """One naive-SIN warp at the slice phase center (not the current view)."""
    from astrowidget.wcs import reproject_for_shader_display

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
    return projected, naive_wcs, elapsed_ms


def _push_cutout_frame(
    sky: Any,
    data: np.ndarray,
    cut_wcs: Any,
    center: SkyCoord,
    fov_deg: float,
    *,
    time_idx: int | None = None,
    freq_idx: int | None = None,
    update_view: bool = True,
) -> dict[str, float | bool]:
    """Push a cutout after one phase-center SIN warp.

    ``_push_image_frame`` is called with ``center=None`` so it does not warp
    again onto the view. The shader samples this texture for later zoom and
    slew. ``update_view`` only moves ``view_ra`` / ``view_dec`` / ``view_fov``.
    """
    suppress = getattr(sky, "_suppress_slice_observer", False)
    sky._suppress_slice_observer = True
    sky._suppress_gesture_observer = True
    t0 = time.perf_counter()
    reproject_ms = float("nan")
    try:
        projected, naive_wcs, reproject_ms = _project_native_sin(data, cut_wcs)
        if time_idx is not None and hasattr(sky, "time_idx"):
            sky.time_idx = int(time_idx)
        if freq_idx is not None and hasattr(sky, "freq_idx"):
            sky.freq_idx = int(freq_idx)
        sky._push_image_frame(
            projected,
            naive_wcs,
            center=None,
            fov=None,
            update_view=False,
        )
        if update_view:
            sky.view_ra = float(center.icrs.ra.deg)
            sky.view_dec = float(center.icrs.dec.deg)
            sky.view_fov = float(fov_deg)
    finally:
        sky._suppress_slice_observer = suppress
        sky._suppress_gesture_observer = False

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
    sky_arr,
    time_idx: int,
    freq_idx: int,
    center: SkyCoord,
    fov_deg: float,
    *,
    pol_idx: int = 0,
    executor: Executor | None = None,
    pad_frac: float = 0.05,
    update_view: bool = True,
) -> dict[str, float | int | tuple[int, int]]:
    """Read an FOV cutout and push one phase-center SIN texture.

    The warp uses the slice ``CRVAL``, not the current view, so a later zoom
    or slew inside the cutout does not reproject. Does **not** call
    ``PreloadedCube.image`` (no full-plane stride).

    Set ``update_view=False`` when reloading after a user pan/zoom so existing
    ``view_ra`` / ``view_dec`` / ``view_fov`` traits are not overwritten.

    Returns a small timing / shape profile dict (includes ``cached`` cutout bits
    via ``_cache`` key when used by :class:`FovOverlaySession`).
    """
    n_l = int(sky_arr.shape[3])
    n_m = int(sky_arr.shape[4])
    chunk_l = int(sky_arr.chunks[3])
    chunk_m = int(sky_arr.chunks[4])

    t_all = time.perf_counter()
    wcs = celestial_wcs_for_time(ds, int(time_idx), freq_idx=int(freq_idx))
    if wcs is None:
        raise RuntimeError(f"No WCS for time_idx={time_idx}")

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

    cut_wcs = adjust_wcs_for_cutout(wcs, spec.l_slice.start, spec.m_slice.start)
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

    profile: dict[str, float | int | tuple[int, int] | CachedCutout] = {
        "shape": spec.shape,
        "n_chunks": spec.n_chunks,
        "nbytes": int(data.nbytes),
        "zarr_ms": float(zarr_ms),
        "total_ms": (time.perf_counter() - t_all) * 1000.0,
        "l0": int(spec.l_slice.start),
        "m0": int(spec.m_slice.start),
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
    return profile  # type: ignore[return-value]


def attach_fov_pan_reload(sky: Any, on_view_changed) -> Any:
    """Reload FOV tiles when the user finishes a pan/zoom on ``sky``.

    JS only increments ``view_gesture_revision`` when ``overlay_view_lock`` is
    true. The built-in SkyWidget handler then reprojects the **full-plane**
    ``PreloadedCube`` slice — wrong for Approach 1 — so this helper removes
    that handler and installs ``on_view_changed(center, fov_deg)`` instead.

    Pan/zoom updates fire on gesture **end** (not every drag frame); that matches
    astrowidget's debounce and keeps Zarr load rate reasonable.

    Parameters
    ----------
    sky
        ``SkyWidget`` instance.
    on_view_changed
        ``callable(center: SkyCoord, fov_deg: float)`` invoked on the trait
        thread after each finished gesture (caller should schedule heavy work).

    Returns
    -------
    The installed observer callback (for ``unobserve``).
    """
    sky.overlay_view_lock = True
    # Drop the default full-plane reproject-at-view path.
    try:
        sky.unobserve(sky._on_view_gesture_revision, names=["view_gesture_revision"])
    except (ValueError, AttributeError):
        pass

    def _handler(change) -> None:
        if change.get("name") != "view_gesture_revision":
            return
        if getattr(sky, "_suppress_gesture_observer", False):
            return
        center = sky.view_center_skycoord()
        fov_deg = float(sky.view_fov)
        on_view_changed(center, fov_deg)

    sky.observe(_handler, names="view_gesture_revision")
    sky._fov_pan_reload_handler = _handler  # type: ignore[attr-defined]
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

    Parameters
    ----------
    sky, ds, sky_arr
        Widget + portal dataset + consolidated zarr ``SKY`` array.
    pol_idx, executor
        Stokes index and optional thread pool for spatial reads.
    pad_frac
        Padding for catalog/slice loads.
    pan_pad_frac
        Larger padding for pan reads (working-set margin).
    debounce_s
        Wait after the last gesture before starting a Zarr read. ``0`` runs
        the read immediately (tests).
    fade_opacity
        Opacity while a pan has moved the view center off the cached cutout.
    schedule_main
        ``callable(cb)`` to run the apply step on the kernel io_loop.
    log
        Optional ``callable(str)`` for status lines (e.g. ``view.log.append``).
    """

    def __init__(
        self,
        sky: Any,
        ds: xr.Dataset,
        sky_arr,
        *,
        pol_idx: int = 0,
        executor: Executor | None = None,
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
        self.pad_frac = float(pad_frac)
        self.pan_pad_frac = float(pan_pad_frac)
        self.debounce_s = float(debounce_s)
        self.fade_opacity = float(fade_opacity)
        self.schedule_main = schedule_main or (lambda cb: cb())
        self.log = log or (lambda _msg: None)
        self.cache: CachedCutout | None = None
        self._tf = {"t": 0, "f": 0}
        self._gen = 0
        self._pending: Future | None = None
        self._timer: threading.Timer | None = None
        self._logged_skip = False
        self._faded = False
        self._anchor: SkyCoord | None = None

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
    ) -> dict[str, float | int | tuple[int, int]]:
        """Catalog / dynspec-driven load (Phase B only; updates cache)."""
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
            executor=self.executor,
            pad_frac=self.pan_pad_frac,  # generous working set for later pans
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
        return prof  # type: ignore[return-value]

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
        # Fade when a pan no longer fits in the cache. The center can still lie
        # on the padded cutout; that reload used to stay at full opacity.
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
        self, gen: int, center: SkyCoord, fov_deg: float, *, reason: str = "pan"
    ) -> None:
        """Async FOV read. Apply on the io_loop. ``reason`` is ``pan`` or ``slice``."""
        if gen != self._gen:
            return
        t_idx, f_idx = self._tf["t"], self._tf["f"]

        def _read_only() -> tuple[int, np.ndarray, Any, FovCutoutSpec, float]:
            t_z0 = time.perf_counter()
            wcs = celestial_wcs_for_time(self.ds, t_idx, freq_idx=f_idx)
            if wcs is None:
                raise RuntimeError(f"No WCS for time_idx={t_idx}")
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
                executor=None,  # already on a worker; avoid nested pool waits
            )
            cut_wcs = adjust_wcs_for_cutout(wcs, spec.l_slice.start, spec.m_slice.start)
            zarr_ms = (time.perf_counter() - t_z0) * 1000.0
            return gen, data, cut_wcs, spec, zarr_ms

        def _apply(result: tuple[int, np.ndarray, Any, FovCutoutSpec, float]) -> None:
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
            self.log(
                f"FOV overlay [{reason}] gen={res_gen} shape={spec.shape} "
                f"chunks={spec.n_chunks} zarr={zarr_ms:.0f}ms "
                f"reproj={push_prof.get('reproject_ms', float('nan')):.0f}ms "
                f"apply={(time.perf_counter() - t_b) * 1000:.0f}ms"
            )

        pool = self.executor
        if pool is None:
            _apply(_read_only())
            return

        fut = pool.submit(_read_only)

        def _done(f: Future) -> None:
            try:
                result = f.result()
            except Exception as exc:  # noqa: BLE001 — worker errors stay on the log
                err = exc

                def _fail() -> None:
                    self.log(f"FOV overlay [{reason} error] {err}")
                    if reason == "slice":
                        self._restore_opacity()

                self.schedule_main(_fail)
                return
            self.schedule_main(lambda: _apply(result))

        fut.add_done_callback(_done)
        self._pending = fut
