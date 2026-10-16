#!/usr/bin/env bash
# Experiment 4 end to end: f64 drape oracle with the regularised exact tangent, the robust f32 Newton on the GPU (vgpu)
# for one or more ablation rungs, score, summarize.
#   ./run-e4.sh                                  RUNGS="A B C D" SIZES="33 65" GPU_ONLY="129"
#   RUNGS="D" SIZES="33" GPU_ONLY="" ./run-e4.sh
set -euo pipefail
cd "$(dirname "$0")"

RUNGS="${RUNGS:-A B C D}"
SIZES="${SIZES:-33 65}"
GPU_ONLY="${GPU_ONLY:-129}"
ORACLE=./oracle/target/release/oracle

cargo build --release --manifest-path oracle/Cargo.toml
for n in $SIZES $GPU_ONLY; do
  dir="drape4/drape-n$n"
  reference=$([[ " $GPU_ONLY " == *" $n "* ]] && echo false || echo true)
  [ -f "$dir/problem.bin" ] || "$ORACLE" e4-generate --n "$n" --reference "$reference" --out "$dir"
done
for rung in $RUNGS; do
  for n in $SIZES $GPU_ONLY; do
    dir="drape4/drape-n$n"
    (cd gpu && node src/node/run-e4.ts --case "../$dir" --rung "$rung")
    "$ORACLE" e4-compare --dir "$dir" --rung "$rung" > /dev/null
  done
done
"$ORACLE" e4-summarize --dir drape4 --out results/experiment4-vgpu-hardware.json
