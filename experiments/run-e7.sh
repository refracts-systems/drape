#!/usr/bin/env bash
# Experiment 7 end to end: quasi-static drape by pseudo-transient continuation.
#   Stage oracle: oracle-e7.sh (f64 references at n17 and n33, problem files at n65 and n129). Heavy CPU work; run it alone.
#   Stage gpu:    the GPU solves in interleaved rounds (laptop GPU throttles, so every case sees the same machine state), the run
#                 with the median ms/step is kept (merge-runs.ts). Nothing else heavy may run meanwhile.
#   Stage score:  e7-compare per case and mode, then e7-summarize into results/experiment7-vgpu-hardware.json.
# Per size: Part 1 base cold; Part 2 edits w1..w3 cold and warm (warm starts from the GPU's own base equilibrium);
# Part 3 prolongation of the previous size's GPU base equilibrium (n17 -> n33 -> n65 -> n129).
# Rounds: Part 1 REPEATS (3) at n17/n33/n65 and REPEATS_LARGE (2) at n129; Parts 2-3 REPEATS_EDITS (2) and 1 at n129.
#   ./run-e7.sh                     STAGES="oracle gpu score" SIZES="17 33" GPU_ONLY="65 129"
#   FIRST_ROUND=2 resumes at a later round; PREVIOUS=65 names the coarse size of the first size; BUDGET_LARGE=1200 is the wall-clock
#   limit in seconds of every n129 solve (stop reason time_budget, reported as such; the cap of 400 steps costs hours at n129).
#   STAGES="gpu" SIZES="17" GPU_ONLY="" REPEATS=1 REPEATS_EDITS=1 ./run-e7.sh
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd -W 2>/dev/null || pwd)"
ENGINE="$ROOT/../engine"

STAGES="${STAGES:-oracle gpu score}"
SIZES="${SIZES-17 33}"
GPU_ONLY="${GPU_ONLY-65 129}"
REPEATS="${REPEATS:-3}"
REPEATS_LARGE="${REPEATS_LARGE:-2}"
REPEATS_EDITS="${REPEATS_EDITS:-2}"
REPEATS_EDITS_LARGE="${REPEATS_EDITS_LARGE:-1}"
EDITS="w1 w2 w3"
ORACLE="$ENGINE/oracle/target/release/oracle"
ALL="$SIZES $GPU_ONLY"

large() { [[ " $GPU_ONLY " == *" $1 "* && "$1" == "129" ]]; }

gpu_run() { # case dir, mode, round, extra args...
  local dir="$1" mode="$2" round="$3"; shift 3
  local budget=""
  [[ "$dir" == *drape-n129* ]] && budget="${BUDGET_LARGE:-1200}"
  (cd "$ENGINE/gpu" && env ${budget:+E7_BUDGET_S=$budget} node src/node/run-e7.ts --case "$ROOT/$dir" --mode "$mode" "$@")
  cp "$dir/gpu-$mode/gpu-result.json" "$dir/gpu-$mode/gpu-result-run$round.json"
}

round_for_size() { # size, round
  local n="$1" round="$2" base="drape7/drape-n$1/base" cold_rounds edit_rounds checks=()
  cold_rounds=$(large "$n" && echo "$REPEATS_LARGE" || echo "$REPEATS")
  edit_rounds=$(large "$n" && echo "$REPEATS_EDITS_LARGE" || echo "$REPEATS_EDITS")
  [ "$round" -eq 1 ] && [ "$n" -le 33 ] && checks=(--checks)
  if [ "$round" -le "$cold_rounds" ]; then gpu_run "$base" cold "$round" "${checks[@]}"; fi
  [ "$round" -le "$edit_rounds" ] || return 0
  for edit in $EDITS; do
    gpu_run "drape7/drape-n$n/$edit" cold "$round"
    gpu_run "drape7/drape-n$n/$edit" warm "$round" --start "$ROOT/$base/gpu-cold/gpu-equilibrium"
  done
  if [ -n "${previous:-}" ]; then
    "$ORACLE" e7-prolong --coarse "drape7/drape-n$previous/base/gpu-cold/gpu-equilibrium" --coarse-n "$previous" --fine-dir "$base" --out "$base/start-prolong-gpu" > /dev/null
    gpu_run "$base" prolong "$round" --start "$ROOT/$base/start-prolong-gpu"
  fi
}

merge() { # case dir, mode
  [ -f "$1/gpu-$2/gpu-result-run1.json" ] && (cd "$ENGINE/gpu" && node src/node/merge-runs.ts --dir "$ROOT/$1/gpu-$2") || true
}

stage_gpu() {
  local rounds=$REPEATS
  [ "$REPEATS_EDITS" -gt "$rounds" ] && rounds=$REPEATS_EDITS
  for round in $(seq "${FIRST_ROUND:-1}" "$rounds"); do
    previous="${PREVIOUS:-}"
    for n in $ALL; do
      round_for_size "$n" "$round"
      previous="$n"
    done
  done
  for n in $ALL; do
    merge "drape7/drape-n$n/base" cold
    merge "drape7/drape-n$n/base" prolong
    for edit in $EDITS; do merge "drape7/drape-n$n/$edit" cold; merge "drape7/drape-n$n/$edit" warm; done
  done
}

stage_score() {
  for n in $ALL; do
    for mode in cold prolong; do
      [ -d "drape7/drape-n$n/base/gpu-$mode" ] && "$ORACLE" e7-compare --dir "drape7/drape-n$n/base" --mode "$mode" > /dev/null
    done
    for edit in $EDITS; do
      for mode in cold warm; do "$ORACLE" e7-compare --dir "drape7/drape-n$n/$edit" --mode "$mode" > /dev/null; done
    done
  done
  "$ORACLE" e7-summarize --dir drape7 --out results/experiment7-vgpu-hardware.json
}

cargo build --release --manifest-path "$ENGINE/oracle/Cargo.toml"
for stage in $STAGES; do
  case "$stage" in
    oracle) SIZES="$SIZES" GPU_ONLY="$GPU_ONLY" ./oracle-e7.sh ;;
    gpu) stage_gpu ;;
    score) stage_score ;;
    *) echo "unknown stage $stage" >&2; exit 1 ;;
  esac
done
