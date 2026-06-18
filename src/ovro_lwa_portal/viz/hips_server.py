"""Jupyter Server extension: serve local OVRO-LWA calibration HiPS over HTTP."""

from __future__ import annotations

import logging
from pathlib import Path

from jupyter_server.utils import url_path_join
from tornado.web import StaticFileHandler

from ovro_lwa_portal.viz.hips import resolve_hips_http_prefix, resolve_hips_root

logger = logging.getLogger(__name__)


def _register_hips_handlers(web_app, hips_root: Path, url_prefix: str) -> None:
    base_url = web_app.settings.get("base_url", "/")
    prefix = url_prefix.lstrip("/")
    pattern = url_path_join(base_url, prefix, "(.*)")
    web_app.add_handlers(
        ".*$",
        [(pattern, StaticFileHandler, {"path": str(hips_root)})],
    )
    logger.info("Serving OVRO HiPS from %s at %s", hips_root, pattern)


def _load_jupyter_server_extension(serverapp) -> None:
    hips_root = resolve_hips_root()
    url_prefix = resolve_hips_http_prefix()
    if not hips_root.is_dir():
        serverapp.log.warning(
            "OVRO HiPS root not found (%s); %s will 404 until OVRO_HIPS_ROOT is set.",
            hips_root,
            url_prefix,
        )
        return
    _register_hips_handlers(serverapp.web_app, hips_root, url_prefix)
    serverapp.log.info(
        "OVRO-LWA HiPS server extension enabled: %s -> %s",
        url_prefix,
        hips_root,
    )


load_jupyter_server_extension = _load_jupyter_server_extension
