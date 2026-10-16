#!/usr/bin/env bash
# Run the linked-UI benchmark matrix. Usage: run_benchmarks.sh stage1|stage2 [A_WORKERS]
set -euo pipefail
cd "$(dirname "$0")/../.."

STAGE="${1:?stage1 or stage2}"
A_WORKERS="${2:-8}"
FULL=/fast/claw/IV_1hMHz_Taper_shflux_coadd.zarr
SUB=/fast/claw/async-test
OUT=notebooks/async_cube_slice/results
mkdir -p "$OUT"
bench() { pixi run python notebooks/async_cube_slice/bench_linked.py "$@" 2>&1 | grep -E '^ *[A-Za-z0-9-]+ +[A-Z]' || true; }

if [[ "$STAGE" == stage1 ]]; then
  R="$OUT/stage1.jsonl"; : > "$R"
  for w in 1 4 8 16; do bench --store "$FULL" --approach A --workers "$w" --label full --out "$R"; done
  for a in A-blit A-bokeh A-xr; do bench --store "$FULL" --approach "$a" --workers 8 --label full --out "$R"; done
  bench --store "$FULL" --approach AD --workers 8 --label full --out "$R"
  for a in C C-native; do bench --store "$FULL" --approach "$a" --label full --out "$R"; done
else
  R="$OUT/stage2.jsonl"; : > "$R"
  for s in R0 R1 R2 R3; do
    for a in A A-blit A-bokeh; do bench --store "$SUB/$s.zarr" --approach "$a" --workers "$A_WORKERS" --label "$s" --out "$R"; done
    for a in C C-native; do bench --store "$SUB/$s.zarr" --approach "$a" --no-validate --label "$s" --out "$R"; done
  done
  for a in A A-blit A-bokeh; do
    bench --sky-store "$SUB/R0.zarr" --dyn-store "$SUB/R3.zarr" --approach "$a" --workers "$A_WORKERS" --label R4 --out "$R"
  done
  for a in C C-native; do
    bench --sky-store "$SUB/R0.zarr" --dyn-store "$SUB/R3.zarr" --approach "$a" --no-validate --label R4 --out "$R"
  done
fi
