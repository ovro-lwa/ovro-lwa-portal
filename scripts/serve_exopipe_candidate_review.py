#!/usr/bin/env python3
"""Serve ExoPipe candidate review with Panel.

Same ``SourceReview`` backend as :mod:`scripts.serve_source_review`, plus the
candidate table from :func:`ovro_lwa_portal.viz.exopipe_catalog.attach_candidate_catalog`.
Launch on port 5010 so this app does not share Source Review's port 5006 or the
ExoPipe indexes on 8766 and 8768.

Usage::

    export OVRO_SOURCE_REVIEW_ZARR=/path/to/store.zarr
    pixi run panel serve scripts/serve_exopipe_candidate_review.py \\
        --port 5010 --show --autoreload

    pixi run panel serve scripts/serve_exopipe_candidate_review.py \\
        --port 5010 --show --autoreload --args /path/to/store.zarr

Optional environment variables::

    OVRO_SOURCE_REVIEW_ZARR   Zarr store path (local path or ``s3://`` URL)
    OVRO_EXOPIPE_V_URL        Stokes V index (default ``http://127.0.0.1:8766``)
    OVRO_EXOPIPE_I_URL        Stokes I index (default ``http://127.0.0.1:8768``)
    OVRO_HIPS_HTTP_BASE       HiPS URL (default ``http://localhost:3005``)
    OVRO_HIPS_ROOT            HiPS survey root (default ``/lustre/pipeline/calibration/hips``)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import ovro_lwa_portal as ovro

from ovro_lwa_portal.viz.exopipe_catalog import (
    URLError,
    attach_candidate_catalog,
    load_candidate_table,
)
from ovro_lwa_portal.viz.hips import list_hips_surveys
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
_DEFAULT_HIPS_BACKGROUND = _DEFAULT_HIPS_ROOT / "healpix_78MHz_nside2048.hips"
_DEFAULT_HIPS_HTTP = "http://localhost:3005"
_FORWARD = """\
ssh -N -f \\
  -L 127.0.0.1:8766:127.0.0.1:8766 \\
  -L 127.0.0.1:8768:127.0.0.1:8768 \\
  lwacalim07"""
_ZARR_MISSING = (
    "Zarr path required: panel serve scripts/serve_exopipe_candidate_review.py "
    "--port 5010 --args /path/to/store.zarr, or set OVRO_SOURCE_REVIEW_ZARR"
)


def _resolve_zarr_path(args: argparse.Namespace) -> str | None:
    for candidate in (args.zarr, args.zarr_path, os.environ.get("OVRO_SOURCE_REVIEW_ZARR")):
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return None


def _parse_queues(items: list[str] | None) -> dict[str, str]:
    if not items:
        return {
            "V": os.environ.get("OVRO_EXOPIPE_V_URL", "http://127.0.0.1:8766"),
            "I": os.environ.get("OVRO_EXOPIPE_I_URL", "http://127.0.0.1:8768"),
        }
    queues: dict[str, str] = {}
    for item in items:
        stokes, separator, url = item.partition("=")
        if not separator or not stokes.strip() or not url.strip():
            raise SystemExit(f"Expected STOKES=URL for --queue, got {item!r}")
        queues[stokes.strip()] = url.strip()
    return queues


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ExoPipe candidate review Panel app")
    parser.add_argument(
        "zarr_path",
        nargs="?",
        default=None,
        help="OVRO-LWA Zarr store (positional; use with panel serve --args)",
    )
    parser.add_argument(
        "--zarr",
        default=None,
        help="OVRO-LWA Zarr store (or set OVRO_SOURCE_REVIEW_ZARR)",
    )
    parser.add_argument(
        "--queue",
        action="append",
        default=None,
        help="Stokes=URL candidate index. Repeat for each queue.",
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
    parser.add_argument(
        "--hips-root",
        type=Path,
        default=Path(os.environ.get("OVRO_HIPS_ROOT", _DEFAULT_HIPS_ROOT)),
    )
    parser.add_argument(
        "--hips-background",
        type=Path,
        default=Path(os.environ.get("OVRO_HIPS_BACKGROUND", str(_DEFAULT_HIPS_BACKGROUND))),
    )
    parser.add_argument(
        "--hips-http-prefix",
        default=os.environ.get("OVRO_HIPS_HTTP_BASE", _DEFAULT_HIPS_HTTP),
    )
    parser.add_argument("--zarr-lm-chunk", type=int, default=512)
    return parser.parse_known_args(argv)[0]


def _build_app(args: argparse.Namespace) -> SourceReview:
    zarr_raw = _resolve_zarr_path(args)
    if zarr_raw is None:
        raise SystemExit(_ZARR_MISSING)
    resolved = ovro.resolve_zarr_source(zarr_raw)
    try:
        candidates = load_candidate_table(_parse_queues(args.queue))
    except URLError as exc:
        raise SystemExit(
            "ExoPipe indexes are not reachable from this process.\n"
            "On the Panel host, start:\n"
            f"{_FORWARD}\n"
            f"{exc}"
        ) from exc
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
        resolved.open_source,
        coordinate_string=args.coordinate,
        known_sources_path=args.known_sources,
        patch_scale=args.patch_scale,
        sky_fov_deg=args.sky_fov_deg,
        patch_fit_max_reduced_chi_squared=10.0,
        heatmap_method="dynamic_spectrum",
        storage_options=resolved.storage_options,
        config=SourceReviewConfig(
            zarr_lm_chunk=args.zarr_lm_chunk,
            hips_root=args.hips_root,
            hips_background=args.hips_background,
            hips_http_prefix=args.hips_http_prefix,
            hips_backgrounds=tuple(list_hips_surveys(args.hips_root)),
            show_heatmap_method=False,
        ),
        validate_zarr=False,
        ui_session=ui_session,
    )
    review_holder["review"] = review
    attach_candidate_catalog(review, candidates)
    return review


_args = _parse_args()
if _resolve_zarr_path(_args) is None:
    raise SystemExit(_ZARR_MISSING)

configure_source_review_serve()
register_hips_panel_serve(_args.hips_root, _args.hips_http_prefix)
_review = _build_app(_args)
_review.panel.servable(title="ExoPipe candidate review")

if __name__ == "__main__":
    print(
        "Launch with:\n"
        "  panel serve scripts/serve_exopipe_candidate_review.py "
        "--port 5010 --show --autoreload --args /path/to/store.zarr\n"
        "Or set OVRO_SOURCE_REVIEW_ZARR (no --args needed).",
        file=sys.stderr,
    )
