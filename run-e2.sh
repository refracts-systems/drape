#!/usr/bin/env bash
# Experiment 2 end to end: f64 Newton oracle per coupon, f32 WGSL Newton + E1 PCG on the GPU (vgpu), score, summarize.
#   ./run-e2.sh [adapter]                       vgpu adapters: auto | hardware | software
#   PATHS="patch uniaxial biaxial" SIZES="33 65" ./run-e2.sh hardware
set -euo pipefail
cd "$(dirname "$0")"

ADAPTER="${1:-hardware}"
PATHS="${PATHS:-patch uniaxial biaxial}"
SIZES="${SIZES:-33 65}"
ORACLE=./oracle/target/release/oracle

cargo build --release --manifest-path oracle/Cargo.toml
for path in $PATHS; do
  for n in $SIZES; do
    dir="coupon/$path-n$n"
    [ -f "$dir/reference.bin" ] || "$ORACLE" e2-generate --path "$path" --n "$n" --out "$dir"
    (cd gpu && node src/node/run-e2.ts --case "../$dir" --adapter "$ADAPTER")
    "$ORACLE" e2-compare --dir "$dir" > /dev/null
  done
done
"$ORACLE" e2-summarize --dir coupon --out "results/experiment2-vgpu-$ADAPTER.json"
