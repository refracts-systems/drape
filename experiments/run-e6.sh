#!/usr/bin/env bash
# Experiment 6 end to end.
#   Part 0: re-score the stored E4-final and E5 GPU outputs with the Experiment 6 measurement rules (no GPU runs, no
#           oracle generation): e6-rescore.
#   Part 1: the E5 scene with the ablation ladder A (E5) / B (+ noise-aware Armijo) / C (+ GPU reductions) / D (+ GPU minima),
#           drape5/drape-nN/e6-rung-X/, scored and summarised.
# The oracle references of drape4-final/ and drape5/ are never regenerated here (about 3 h of CPU at n65); the script stops
# when they are missing. Run it as a detached process: the oracle scoring steps are CPU work and must not overlap a timed GPU run.
# Timing: each size runs REPEATS rounds of the whole ladder (rungs interleaved, so a warm laptop GPU hits every rung alike) and
# the run with the median ms/step is kept (merge-runs.ts). The outputs themselves are identical between rounds.
#   ./run-e6.sh                              RUNGS="A B C D" SIZES="33 65" GPU_ONLY="129" REPEATS=3 REPEATS_LARGE=2
#   RUNGS="A" SIZES="33" GPU_ONLY="" REPEATS=1 ./run-e6.sh
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd -W 2>/dev/null || pwd)"
ENGINE="$ROOT/../engine"

RUNGS="${RUNGS:-A B C D}"
SIZES="${SIZES:-33 65}"
GPU_ONLY="${GPU_ONLY:-129}"
REPEATS="${REPEATS:-3}"
REPEATS_LARGE="${REPEATS_LARGE:-2}"
ORACLE="$ENGINE/oracle/target/release/oracle"

for n in $SIZES; do
  for file in "drape4-final/drape-n$n/reference.bin" "drape5/drape-n$n/reference.bin"; do
    [ -f "$file" ] || { echo "missing $file: generate it with run-e5.sh first (e6 never regenerates oracle references)" >&2; exit 1; }
  done
done
for n in $GPU_ONLY; do
  [ -f "drape5/drape-n$n/problem.bin" ] || { echo "missing drape5/drape-n$n/problem.bin: generate it with run-e5.sh first" >&2; exit 1; }
done

cargo build --release --manifest-path "$ENGINE/oracle/Cargo.toml"
"$ORACLE" e6-rescore --root . --out results/experiment6-part0-rescore.json > /dev/null
for n in $SIZES $GPU_ONLY; do
  rounds=$([[ " $GPU_ONLY " == *" $n "* ]] && echo "$REPEATS_LARGE" || echo "$REPEATS")
  for round in $(seq 1 "$rounds"); do
    for rung in $RUNGS; do
      (cd "$ENGINE/gpu" && node src/node/run-e6.ts --case "$ROOT/drape5/drape-n$n" --rung "$rung")
      cp "drape5/drape-n$n/e6-rung-$rung/gpu-result.json" "drape5/drape-n$n/e6-rung-$rung/gpu-result-run$round.json"
    done
  done
  for rung in $RUNGS; do (cd "$ENGINE/gpu" && node src/node/merge-runs.ts --dir "$ROOT/drape5/drape-n$n/e6-rung-$rung"); done
done
for rung in $RUNGS; do
  for n in $SIZES $GPU_ONLY; do
    "$ORACLE" e6-compare --dir "drape5/drape-n$n" --rung "$rung" > /dev/null
  done
done
"$ORACLE" e6-summarize --dir drape5 --out results/experiment6-vgpu-hardware.json
