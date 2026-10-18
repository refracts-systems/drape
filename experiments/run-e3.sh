#!/usr/bin/env bash
# Experiment 3 end to end: f64 backward-Euler drape oracle, f32 WGSL drape (membrane + hinge + inertia) with the E1 PCG on
# the GPU (vgpu), score, summarize.
#   ./run-e3.sh [adapter]                       vgpu adapters: auto | hardware | software
#   SIZES="33 65" GPU_ONLY="129" ./run-e3.sh hardware     GPU_ONLY sizes get no f64 trajectory (timing and PCG growth only)
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd -W 2>/dev/null || pwd)"
ENGINE="$ROOT/../engine"

ADAPTER="${1:-hardware}"
SIZES="${SIZES:-33 65}"
GPU_ONLY="${GPU_ONLY:-129}"
ORACLE="$ENGINE/oracle/target/release/oracle"

cargo build --release --manifest-path "$ENGINE/oracle/Cargo.toml"
for n in $SIZES $GPU_ONLY; do
  dir="drape/drape-n$n"
  reference=$([[ " $GPU_ONLY " == *" $n "* ]] && echo false || echo true)
  [ -f "$dir/problem.bin" ] || "$ORACLE" e3-generate --n "$n" --reference "$reference" --out "$dir"
  (cd "$ENGINE/gpu" && node src/node/run-e3.ts --case "$ROOT/$dir" --adapter "$ADAPTER")
  "$ORACLE" e3-compare --dir "$dir" > /dev/null
done
"$ORACLE" e3-summarize --dir drape --out "results/experiment3-vgpu-$ADAPTER.json"
