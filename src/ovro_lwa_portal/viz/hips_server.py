"""Jupyter Server extension: serve local OVRO-LWA calibration HiPS over HTTP."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from jupyter_server.utils import url_path_join
from tornado.web import StaticFileHandler

from ovro_lwa_portal.viz.hips import resolve_hips_http_prefix, resolve_hips_root

logger = logging.getLogger(__name__)

_HIPS_PANEL_PATTERN: str | None = None


def register_hips_static_handlers(
    web_app: object,
    hips_root: Path,
    url_prefix: str,
    *,
    base_url: str = "/",
) -> None:
    """Register Tornado static handlers on a running web application."""
    prefix = url_prefix.lstrip("/")
    if prefix.startswith(("http://", "https://")):
        return
    pattern = url_path_join(base_url, prefix, "(.*)")
    web_app.add_handlers(  # type: ignore[attr-defined]
        ".*$",
        [(pattern, StaticFileHandler, {"path": str(hips_root.resolve())})],
    )
    logger.info("Serving OVRO HiPS from %s at %s", hips_root, pattern)


def register_hips_panel_serve(hips_root: Path, url_prefix: str) -> None:
    """Mount HiPS tiles on Bokeh ``toplevel_patterns`` for ``panel serve``.

    Relative URLs such as ``/calibration/hips/Survey.hips/`` then resolve from the
    same host/port as the Panel app (mirrors the Jupyter server extension).
    """
    global _HIPS_PANEL_PATTERN  # noqa: PLW0603

    prefix = url_prefix.strip().rstrip("/")
    if prefix.startswith(("http://", "https://")):
        return
    if not hips_root.is_dir():
        logger.warning(
            "HiPS root not found (%s); %s will 404 until OVRO_HIPS_ROOT is set.",
            hips_root,
            prefix,
        )
        return

    from bokeh.server.urls import toplevel_patterns

    rel_prefix = prefix.lstrip("/")
    pattern = rf"/{re.escape(rel_prefix)}/(.*)"
    if _HIPS_PANEL_PATTERN == pattern:
        return
    for entry in toplevel_patterns:
        existing_pattern = entry[0]
        if existing_pattern == pattern:
            _HIPS_PANEL_PATTERN = pattern
            return

    toplevel_patterns.append(
        (pattern, StaticFileHandler, {"path": str(hips_root.resolve())}),
    )
    _HIPS_PANEL_PATTERN = pattern
    logger.info("Panel serve HiPS: %s -> %s", pattern, hips_root)


def _register_hips_handlers(web_app, hips_root: Path, url_prefix: str) -> None:
    base_url = web_app.settings.get("base_url", "/")
    register_hips_static_handlers(web_app, hips_root, url_prefix, base_url=base_url)


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
