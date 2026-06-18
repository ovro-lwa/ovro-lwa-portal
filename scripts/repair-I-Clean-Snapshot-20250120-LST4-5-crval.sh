#!/usr/bin/env bash
# Repair wcs_header_str CRVAL in an existing I-Clean-Snapshot Zarr store.
#
# Reads native CRVAL1/CRVAL2 from the same pipeline *.fits.fs sources used by
# scripts/ingest-I-Clean-Snapshot-20250120-LST4-5.sh and patches each per-time
# WCS row in place (LM pixel grid unchanged).
#
# Zarr ``time`` uses FITS DATE-OBS while filenames use ``YYYYMMDD_HHMMSS-image``;
# the repair pairs them by sorted index (typically ~5 s offset), not exact strings.
#
# Run a dry-run first:
#   DRY_RUN=1 pixi run bash scripts/repair-I-Clean-Snapshot-20250120-LST4-5-crval.sh
#
# Apply in-place (patches wcs_header_str only; no full-store backup on /fast):
#   pixi run bash scripts/repair-I-Clean-Snapshot-20250120-LST4-5-crval.sh
#
# If a prior run failed with "No space left on device", remove the partial backup:
#   rm -rf /fast/claw/I-Clean-Snapshot-20250120-LST4-5/I-Clean-Snapshot-20250120-LST4-5.zarr.backup-before-crval-repair
#
# Prerequisites: funpack (CFITSIO), python3 with ovro_lwa_portal installed
#
# Do not run from CI without access to /lustre and /fast.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

readonly GLOB_PATTERN='/lustre/pipeline/exopipe/phase1/0[45]h/2025-01-20/Run_20260[56]*/*MHz/snapshots_clean/*image.fits.fs'
readonly WORK_ROOT='/lustre/claw/I-Clean-Snapshot-20250120-LST4-5/crval_repair_work'
readonly ZARR_PATH='/fast/claw/I-Clean-Snapshot-20250120-LST4-5/I-Clean-Snapshot-20250120-LST4-5.zarr'

DRY_RUN="${DRY_RUN:-0}"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "ERROR: required command not found: $1"
    exit 1
  fi
}

require_cmd python3
require_cmd funpack

repair_args=(
  "${SCRIPT_DIR}/repair_zarr_crval_from_fits.py"
  --glob-pattern "${GLOB_PATTERN}"
  --zarr-path "${ZARR_PATH}"
  --work-root "${WORK_ROOT}"
  --skip-backup
  --log-level info
)

if [[ "${DRY_RUN}" == 1 ]]; then
  repair_args+=(--dry-run)
  log "DRY_RUN=1: reporting CRVAL deltas only"
else
  log "Applying CRVAL repair in place (--skip-backup; wcs_header_str only)"
fi

log "Repairing wcs_header_str CRVAL -> ${ZARR_PATH}"
python3 "${repair_args[@]}"
log "Done."
