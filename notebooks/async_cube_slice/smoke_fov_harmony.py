"""Headless harmony smoke: progressive dynspec fill + FOV overlay on shared pool.

Exits 0 on success. Does not need Jupyter / a browser.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("ASTROWIDGET_PROFILE_PUSH", "1")

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def main() -> int:
    store = os.environ.get(
        "FOV_SMOKE_STORE", "/fast/claw/IV_1hMHz_Taper_shflux_coadd.zarr"
    )
    if not Path(store).exists():
        print(f"SKIP: store missing {store}")
        return 0

    import numpy as np
    import ovro_lwa_portal as ovro
    import zarr
    from astropy.coordinates import SkyCoord
    from astrowidget import SkyWidget
    from ovro_lwa_portal.accessor import _parse_sin_celestial_keywords, _read_wcs_header_str
    from ovro_lwa_portal.viz.pipeline_qa_app import bind_sky_widget_dataset

    from fov_cutout import push_fov_overlay
    from linked_ui import LinkedAsyncView

    ds = ovro.open_dataset(store, chunks={}, validate=False)
    arr = zarr.open_consolidated(store, mode="r")["SKY"]
    tile = slice(1173, 1564)
    pixel = 1368
    pol = 0
    hdr = _read_wcs_header_str(ds, time_idx=0, freq_idx=0)
    kw = _parse_sin_celestial_keywords(hdr)
    center = SkyCoord(float(kw["CRVAL1"]), float(kw["CRVAL2"]), unit="deg", frame="fk5")

    # Fake schedule: run callbacks inline (headless).
    def schedule(cb):
        cb()

    pushes: list[str] = []

    def push(*_views):
        pushes.append("push")

    view = LinkedAsyncView(
        store,
        l_slice=tile,
        m_slice=tile,
        l_idx=pixel,
        m_idx=pixel,
        pol_idx=pol,
        max_workers=4,
        schedule_main=schedule,
        push=push,
        start_periodic=lambda cb, ms: _FakePeriodic(cb),
    )

    sky = SkyWidget()
    bind_sky_widget_dataset(sky, ds, max_size=256)
    overlay_profiles: list[dict] = []

    def on_slice(t: int, f: int) -> None:
        overlay_profiles.append(
            push_fov_overlay(
                sky, ds, arr, t, f, center, 8.0, pol_idx=pol, executor=view.pool
            )
        )

    view.on_slice = on_slice

    # Start progressive fill (also requests tile → on_slice).
    view.start(t=0, f=0)
    # Drain until complete (fake periodic ticks).
    t_deadline = time.perf_counter() + 30.0
    while view._remaining > 0 and time.perf_counter() < t_deadline:
        view._drain()
        time.sleep(0.02)
    view._drain()

    finite = float(np.isfinite(view.grid).mean())
    assert view._remaining == 0, f"dynspec incomplete remaining={view._remaining}"
    assert finite > 0.5, f"unexpected finite fraction {finite}"
    assert overlay_profiles, "FOV overlay never ran"
    assert overlay_profiles[0]["n_chunks"] >= 1
    assert overlay_profiles[0]["shape"][0] < arr.shape[3]

    # Second overlay while pool is idle (link tap). Wait for worker → schedule.
    n_before = len(overlay_profiles)
    view.request_tile(10, 3)
    t_wait = time.perf_counter() + 10.0
    while len(overlay_profiles) <= n_before and time.perf_counter() < t_wait:
        time.sleep(0.02)
    assert len(overlay_profiles) > n_before, (
        f"second FOV overlay did not run (have {len(overlay_profiles)})"
    )

    print(
        f"OK harmony: dynspec finite={finite:.3f} sum={np.nansum(view.grid):.1f} "
        f"overlays={len(overlay_profiles)} "
        f"last_shape={overlay_profiles[-1]['shape']} "
        f"last_zarr_ms={overlay_profiles[-1]['zarr_ms']:.1f} "
        f"pushes={len(pushes)}"
    )
    view.close()
    return 0


class _FakePeriodic:
    def __init__(self, callback):
        self.callback = callback
        self._stopped = False

    def stop(self):
        self._stopped = True


if __name__ == "__main__":
    raise SystemExit(main())
