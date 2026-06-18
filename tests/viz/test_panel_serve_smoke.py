"""Smoke tests that SourceReview layouts serve over HTTP."""

from __future__ import annotations

import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pn = pytest.importorskip("panel")
pytest.importorskip("astrowidget")

from ovro_lwa_portal.viz.source_review_app import SourceReview, SourceReviewConfig


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


@pytest.mark.network
@pytest.mark.filterwarnings("ignore:There is no current event loop:DeprecationWarning")
def test_source_review_layout_serves_http(tmp_path: Path) -> None:
    """Panel server can render the SourceReview layout (browser-level smoke)."""
    zarr = tmp_path / "store.zarr"
    zarr.mkdir()
    review = SourceReview(
        zarr,
        patch_scale=5.0,
        sky_fov_deg=8.0,
        patch_fit_max_reduced_chi_squared=10.0,
        config=SourceReviewConfig(
            hips_root=tmp_path,
            hips_background=tmp_path / "missing.hips",
        ),
        validate_zarr=False,
    )
    port = _free_port()
    server = pn.serve(
        {"review": review._layout},
        port=port,
        address="127.0.0.1",
        show=False,
        threaded=True,
        verbose=False,
    )
    try:
        deadline = time.monotonic() + 10.0
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/review",
                    timeout=2.0,
                ) as response:
                    body = response.read()
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                last_error = exc
                time.sleep(0.2)
        else:
            raise AssertionError(f"Panel server did not respond: {last_error}") from last_error

        assert response.status == 200
        assert len(body) > 1000
        assert b"bokeh" in body.lower() or b"panel" in body.lower()
    finally:
        server.stop()
