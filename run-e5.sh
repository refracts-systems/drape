#!/usr/bin/env bash
# Experiment 5 end to end.
#   Part 0: E4-final = rung D with the f32-resolution update stop, the f64-floor oracle convergence and an oracle cap of 200
#           (drape4-final/), plus the diagnosis of n65 step 89 of the E4 reference (drape4/, when present).
#   Part 1: the free sheet falling onto a sphere resting on a floor, IPC barrier, frictionless (drape5/).
# Every problem is generated before any GPU run is timed.
#   ./run-e5.sh                          SIZES="33 65" GPU_ONLY="129"
#   SIZES="33" GPU_ONLY="" ./run-e5.sh
set -euo pipefail
cd "$(dirname "$0")"

SIZES="${SIZES:-33 65}"
GPU_ONLY="${GPU_ONLY:-129}"
ORACLE=./oracle/target/release/oracle

reference_flag() { [[ " $GPU_ONLY " == *" $1 "* ]] && echo false || echo true; }

cargo build --release --manifest-path oracle/Cargo.toml
if [ -f drape4/drape-n65/reference.bin ]; then
  "$ORACLE" e4-diagnose --dir drape4/drape-n65 --step 89 --out results/e4-step89-diagnosis.json > /dev/null
fi
for n in $SIZES $GPU_ONLY; do
  [ -f "drape4-final/drape-n$n/problem.bin" ] || "$ORACLE" e4-generate --final true --n "$n" --reference "$(reference_flag "$n")" --out "drape4-final/drape-n$n"
  [ -f "drape5/drape-n$n/problem.bin" ] || "$ORACLE" e5-generate --n "$n" --reference "$(reference_flag "$n")" --out "drape5/drape-n$n"
done
for n in $SIZES $GPU_ONLY; do
  (cd gpu && node src/node/run-e4.ts --case "../drape4-final/drape-n$n" --rung D)
  "$ORACLE" e4-compare --dir "drape4-final/drape-n$n" --rung D > /dev/null
done
"$ORACLE" e4-summarize --dir drape4-final --out results/experiment4-final.json
for n in $SIZES $GPU_ONLY; do
  (cd gpu && node src/node/run-e5.ts --case "../drape5/drape-n$n")
  "$ORACLE" e5-compare --dir "drape5/drape-n$n" > /dev/null
done
"$ORACLE" e5-summarize --dir drape5 --baseline results/experiment4-final.json --out results/experiment5-vgpu-hardware.json
