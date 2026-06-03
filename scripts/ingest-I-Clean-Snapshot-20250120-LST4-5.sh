#!/usr/bin/env bash
# Ingest OVRO-LWA I clean snapshots (2025-01-20, LST 4–5 h) to Zarr.
#
# Pipeline (low peak disk — one observation time at a time):
#   1. Discover all *.fits.fs with a Python glob; group by time (filename metadata)
#   2. For each time group:
#        - symlink each .fs as .fz, funpack the symlink, then convert that group
#        - stage {time_key}__*.fits, ovro-ingest convert (append one time bin)
#        - delete work dir, staged FITS, and temporary fixed FITS before continuing
#
# Prerequisites: funpack (CFITSIO), python3 with ovro_lwa_portal installed
#
# Usage:
#   pixi run bash scripts/ingest-I-Clean-Snapshot-20250120-LST4-5.sh
#   REBUILD=1 pixi run bash scripts/ingest-I-Clean-Snapshot-20250120-LST4-5.sh
#
# Do not run from CI without access to /lustre and /fast.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

readonly GLOB_PATTERN='/lustre/pipeline/exopipe/phase1/0[45]h/2025-01-20/Run_20260[56]*/*MHz/snapshots_clean/*image.fits.fs'
readonly WORK_ROOT='/lustre/claw/I-Clean-Snapshot-20250120-LST4-5/work'
readonly STAGING_DIR='/lustre/claw/I-Clean-Snapshot-20250120-LST4-5/fits_staging'
readonly FIXED_ROOT='/lustre/claw/I-Clean-Snapshot-20250120-LST4-5/fixed_fits'
readonly OUTPUT_DIR='/fast/claw/I-Clean-Snapshot-20250120-LST4-5'
readonly ZARR_NAME='I-Clean-Snapshot-20250120-LST4-5.zarr'

REBUILD="${REBUILD:-0}"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "ERROR: required command not found: $1"
    exit 1
  fi
}

require_cmd python3
require_cmd funpack

mkdir -p "${WORK_ROOT}" "${STAGING_DIR}" "${FIXED_ROOT}" "${OUTPUT_DIR}"

convert_args=(
  "${SCRIPT_DIR}/ingest_per_time_convert.py"
  --glob-pattern "${GLOB_PATTERN}"
  --work-root "${WORK_ROOT}"
  --staging-dir "${STAGING_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --fixed-dir "${FIXED_ROOT}"
  --zarr-name "${ZARR_NAME}"
  --chunk-lm 1024
  --log-level info
)

if [[ "${REBUILD}" == 1 ]]; then
  convert_args+=(--rebuild)
  log "REBUILD=1: first time write will replace existing Zarr if present"
else
  log "Resume mode: time keys already in Zarr are skipped"
fi

log "Starting per-time ingest -> ${OUTPUT_DIR}/${ZARR_NAME}"
python3 "${convert_args[@]}"
log "Done. Zarr store: ${OUTPUT_DIR}/${ZARR_NAME}"
