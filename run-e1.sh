#!/usr/bin/env bash
# Experiment 1 end to end: generate cases (f64 oracle), solve each on the GPU in f32, score against f64, summarize.
#   ./run-e1.sh [webgpu|vgpu] [adapter]
#   webgpu adapters: hardware | software      vgpu adapters: auto | hardware | software
#   CASES="tension dynamic slack" SIZES="33 65 129" REPEATS=3 ./run-e1.sh webgpu hardware
#   REPLACE_EVERY=50 REPLACE_ABOVE=1e-2 ./run-e1.sh ...   residual replacement in the GPU solve and its CPU f32 emulation
set -euo pipefail
cd "$(dirname "$0")"

BACKEND="${1:-webgpu}"
ADAPTER="${2:-hardware}"
CASES="${CASES:-tension dynamic slack}"
SIZES="${SIZES:-33 65 129}"
REPEATS="${REPEATS:-3}"
REPLACE_EVERY="${REPLACE_EVERY:-0}"
REPLACE_ABOVE="${REPLACE_ABOVE:-0}"
ORACLE=./oracle/target/release/oracle
RUNNER=$([ "$BACKEND" = vgpu ] && echo src/node/run-vgpu.ts || echo src/node/run-browser.ts)

cargo build --release --manifest-path oracle/Cargo.toml
for kind in $CASES; do
  for n in $SIZES; do
    dir="cases/$kind-n$n"
    [ -f "$dir/reference.bin" ] || "$ORACLE" generate --case "$kind" --n "$n" --out "$dir" --replace-every "$REPLACE_EVERY" --replace-above "$REPLACE_ABOVE"
    (cd gpu && node "$RUNNER" --case "../$dir" --adapter "$ADAPTER" --repeats "$REPEATS")
    "$ORACLE" compare --dir "$dir" > /dev/null
  done
done
"$ORACLE" summarize --dir cases --out "results/experiment1-$BACKEND-$ADAPTER.json"
