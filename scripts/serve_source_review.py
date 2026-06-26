#!/usr/bin/env python3
"""Serve :class:`~ovro_lwa_portal.viz.source_review_app.SourceReview` with Panel.

Uses :class:`ServedPanelUISession` (Bokeh server comm) and
:func:`configure_source_review_serve` (includes the ipywidgets bridge). The
notebook path keeps :class:`JupyterPanelUISession` and must not load
``pn.extension('ipywidgets')`` in JupyterLab.

Usage::

    export OVRO_SOURCE_REVIEW_ZARR=/path/to/store.zarr
    pixi run panel serve scripts/serve_source_review.py --show --autoreload

Or pass the store with Panel's ``--args`` (not ``--``; Panel does not forward bare
``--zarr`` after the script path)::

    pixi run panel serve scripts/serve_source_review.py --show --autoreload \\
        --args /path/to/store.zarr

    pixi run panel serve scripts/serve_source_review.py --show --autoreload \\
        --args --zarr /path/to/store.zarr

Optional environment variables (same as ``notebooks/source_review.ipynb``)::

    OVRO_SOURCE_REVIEW_ZARR   Zarr store path
    OVRO_HIPS_HTTP_BASE       HiPS URL prefix (default ``/calibration/hips``)
    OVRO_HIPS_ROOT            HiPS files on disk

HiPS note: this script registers a static route on the Panel server at
``OVRO_HIPS_HTTP_BASE`` (default ``/calibration/hips``) when ``--hips-root`` exists
on disk. Alternatively serve tiles with ``python -m http.server`` and set
``OVRO_HIPS_HTTP_BASE`` to the full URL (e.g. ``http://localhost:3005``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import ovro_lwa_portal as ovro

from ovro_lwa_portal.viz.hips_server import register_hips_panel_serve
from ovro_lwa_portal.viz.panel_ui_session import ServedPanelUISession
from ovro_lwa_portal.viz.source_review_app import (
    SourceReview,
    SourceReviewConfig,
    configure_source_review_serve,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_KNOWN_SOURCES = _REPO_ROOT / "notebooks" / "known_sources.yaml"
_DEFAULT_HIPS_ROOT = Path("/lustre/pipeline/calibration/hips")
_DEFAULT_HIPS_BACKGROUND = _DEFAULT_HIPS_ROOT / "Blue_I_deep_Taper_Robust-0.75_Jan25.hips"


def _resolve_zarr_path(args: argparse.Namespace) -> Path | None:
    if args.zarr is not None:
        return Path(args.zarr)
    if args.zarr_path is not None:
        return Path(args.zarr_path)
    env = os.environ.get("OVRO_SOURCE_REVIEW_ZARR")
    return Path(env) if env else None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SourceReview Panel app")
    parser.add_argument(
        "zarr_path",
        nargs="?",
        type=Path,
        default=None,
        help="OVRO-LWA Zarr store (positional; use with panel serve --args)",
    )
    parser.add_argument(
        "--zarr",
        type=Path,
        default=None,
        help="OVRO-LWA Zarr store (or set OVRO_SOURCE_REVIEW_ZARR)",
    )
    parser.add_argument(
        "--coordinate",
        default=os.environ.get("OVRO_SOURCE_REVIEW_COORDINATE", ""),
        help="Optional default coordinate field value",
    )
    parser.add_argument(
        "--known-sources",
        type=Path,
        default=Path(os.environ.get("OVRO_SOURCE_REVIEW_KNOWN_SOURCES", _DEFAULT_KNOWN_SOURCES)),
        help="known_sources.yaml for autocomplete",
    )
    parser.add_argument("--patch-scale", type=float, default=5.0)
    parser.add_argument("--sky-fov-deg", type=float, default=8.0)
    parser.add_argument("--heatmap-method", default="dynamic_spectrum")
    parser.add_argument(
        "--hips-root",
        type=Path,
        default=Path(os.environ.get("OVRO_HIPS_ROOT", _DEFAULT_HIPS_ROOT)),
    )
    parser.add_argument(
        "--hips-background",
        type=Path,
        default=Path(
            os.environ.get("OVRO_HIPS_BACKGROUND", str(_DEFAULT_HIPS_BACKGROUND)),
        ),
    )
    parser.add_argument(
        "--hips-http-prefix",
        default=os.environ.get("OVRO_HIPS_HTTP_BASE", "/calibration/hips"),
    )
    parser.add_argument("--zarr-lm-chunk", type=int, default=512)
    return parser.parse_known_args(argv)[0]


def _build_review(args: argparse.Namespace) -> SourceReview:
    zarr_raw = _resolve_zarr_path(args)
    if zarr_raw is None:
        raise SystemExit(
            "Zarr path required: panel serve ... --args /path/to/store.zarr, "
            "or --args --zarr /path, or set OVRO_SOURCE_REVIEW_ZARR",
        )
    zarr_path = zarr_raw.expanduser()
    ovro.validate_local_zarr_store(zarr_path)
    review_holder: dict[str, SourceReview] = {}

    def _root_views() -> tuple:
        review = review_holder["review"]
        return (
            review._layout,
            review._status_pane,
            review._heatmap_pane,
            review._coord_input,
        )

    ui_session = ServedPanelUISession(_root_views)
    review = SourceReview(
        zarr_path,
        coordinate_string=args.coordinate,
        known_sources_path=args.known_sources,
        patch_scale=args.patch_scale,
        sky_fov_deg=args.sky_fov_deg,
        patch_fit_max_reduced_chi_squared=10.0,
        heatmap_method=args.heatmap_method,
        config=SourceReviewConfig(
            zarr_lm_chunk=args.zarr_lm_chunk,
            hips_root=args.hips_root,
            hips_background=args.hips_background,
            hips_http_prefix=args.hips_http_prefix,
        ),
        validate_zarr=False,
        ui_session=ui_session,
    )
    review_holder["review"] = review
    return review


_args = _parse_args()
if _resolve_zarr_path(_args) is None:
    raise SystemExit(
        "Zarr path required: panel serve ... --args /path/to/store.zarr, "
        "or --args --zarr /path, or set OVRO_SOURCE_REVIEW_ZARR",
    )

configure_source_review_serve()

register_hips_panel_serve(_args.hips_root, _args.hips_http_prefix)

_review = _build_review(_args)
_review.panel.servable(title="Source review")

if __name__ == "__main__":
    print(
        "Launch with:\n"
        "  panel serve scripts/serve_source_review.py --show --autoreload "
        "--args /path/to/store.zarr\n"
        "Or set OVRO_SOURCE_REVIEW_ZARR (no --args needed).",
        file=sys.stderr,
    )
