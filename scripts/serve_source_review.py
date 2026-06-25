#!/usr/bin/env python3
"""Serve :class:`~ovro_lwa_portal.viz.source_review_app.SourceReview` with Panel.

Thin wrapper around the notebook launch cell — same ``SourceReview`` backend
(``JupyterPanelUISession``), no app refactor.

Usage::

    export OVRO_SOURCE_REVIEW_ZARR=/path/to/store.zarr
    pixi run panel serve scripts/serve_source_review.py --show --autoreload

Or pass the store on the command line (``--`` separates Panel flags from script args)::

    pixi run panel serve scripts/serve_source_review.py --show --autoreload -- \\
        --zarr /path/to/store.zarr

Optional environment variables (same as ``notebooks/source_review.ipynb``)::

    OVRO_SOURCE_REVIEW_ZARR   Zarr store path
    OVRO_HIPS_HTTP_BASE       HiPS URL prefix (default ``/calibration/hips``)
    OVRO_HIPS_ROOT            HiPS files on disk

HiPS note: ``panel serve`` does not install the Jupyter HiPS extension. For a
standalone server, serve tiles with ``python -m http.server`` from the HiPS root
and point ``OVRO_HIPS_HTTP_BASE`` at that URL, or run behind a reverse proxy that
maps ``/calibration/hips``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import ovro_lwa_portal as ovro
import panel as pn

from ovro_lwa_portal.viz.source_review_app import (
    SourceReview,
    SourceReviewConfig,
    configure_source_review_notebook,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_KNOWN_SOURCES = _REPO_ROOT / "notebooks" / "known_sources.yaml"
_DEFAULT_HIPS_ROOT = Path("/lustre/pipeline/calibration/hips")
_DEFAULT_HIPS_BACKGROUND = _DEFAULT_HIPS_ROOT / "Blue_I_deep_Taper_Robust-0.75_Jan25.hips"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SourceReview Panel app")
    parser.add_argument(
        "--zarr",
        type=Path,
        default=os.environ.get("OVRO_SOURCE_REVIEW_ZARR"),
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
    if args.zarr is None:
        raise SystemExit(
            "Zarr path required: pass --zarr or set OVRO_SOURCE_REVIEW_ZARR",
        )
    zarr_path = Path(args.zarr).expanduser()
    ovro.validate_local_zarr_store(zarr_path)
    return SourceReview(
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
    )


_args = _parse_args()
if _args.zarr is None:
    raise SystemExit(
        "Zarr path required: pass --zarr or set OVRO_SOURCE_REVIEW_ZARR",
    )

configure_source_review_notebook()
try:
    pn.extension("ipywidgets")
except Exception:  # noqa: BLE001
    pass

_review = _build_review(_args)
_review.panel.servable(title="Source review")

if __name__ == "__main__":
    print(
        "Launch with: panel serve scripts/serve_source_review.py --show --autoreload",
        file=sys.stderr,
    )
