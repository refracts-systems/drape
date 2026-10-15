#!/usr/bin/env bash
# Experiment 1 end to end: generate cases (f64 oracle), solve each on the GPU in f32, score against f64, summarize.
#   ./run-e1.sh [webgpu|vgpu] [adapter]
#   webgpu adapters: hardware | software      vgpu adapters: auto | hardware | software
#   CASES="tension dynamic slack" SIZES="33 65 129" REPEATS=3 ./run-e1.sh webgpu hardware
set -euo pipefail
cd "$(dirname "$0")"

BACKEND="${1:-webgpu}"
ADAPTER="${2:-hardware}"
CASES="${CASES:-tension dynamic slack}"
SIZES="${SIZES:-33 65 129}"
REPEATS="${REPEATS:-3}"
ORACLE=./oracle/target/release/oracle
RUNNER=$([ "$BACKEND" = vgpu ] && echo src/node/run-vgpu.ts || echo src/node/run-browser.ts)

cargo build --release --manifest-path oracle/Cargo.toml
for kind in $CASES; do
  for n in $SIZES; do
    dir="cases/$kind-n$n"
    [ -f "$dir/reference.bin" ] || "$ORACLE" generate --case "$kind" --n "$n" --out "$dir"
    (cd gpu && node "$RUNNER" --case "../$dir" --adapter "$ADAPTER" --repeats "$REPEATS")
    "$ORACLE" compare --dir "$dir" > /dev/null
  done
done
"$ORACLE" summarize --dir cases --out "results/experiment1-$BACKEND-$ADAPTER.json"
