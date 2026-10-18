#!/usr/bin/env bash
# Experiment 7 oracle references (f64), one process per independent solve, run in parallel; never overlaps a timed GPU run.
#   SIZES="17 33"   oracle + GPU meshes;  GPU_ONLY="65 129"   meshes that only get problem files (no f64 reference).
# Layout: drape7/drape-n<N>/<edit>/ with problem.*, reference-{cold,warm,prolong}.*, equilibrium-*, probes.* (base cold).
# Stored references are never regenerated. Progress of every solve is written to <dir>/<mode>.log.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd -W 2>/dev/null || pwd)"
ENGINE="$ROOT/../engine"

SIZES="${SIZES-17 33}"
GPU_ONLY="${GPU_ONLY-65 129}"
EDITS="base w1 w2 w3"
ORACLE="$ENGINE/oracle/target/release/oracle"

generate() {
  for n in "$@"; do
    for edit in $EDITS; do
      dir="drape7/drape-n$n/$edit"
      [ -f "$dir/problem.bin" ] || "$ORACLE" e7-generate --n "$n" --edit "$edit" --out "$dir"
    done
  done
}

solve() { # dir mode [start] [probes]
  local dir="$1" mode="$2" start="${3:-}" probes="${4:-false}"
  [ -f "$dir/reference-$mode.json" ] && return 0
  local args=(e7-solve --dir "$dir" --mode "$mode" --probes "$probes")
  [ -n "$start" ] && args+=(--start "$start")
  "$ORACLE" "${args[@]}" > "$dir/$mode.out" 2> "$dir/$mode.log" || echo "solve failed: $dir $mode (see $dir/$mode.log)" >&2
}

size_chain() { # one size: base cold, then the three edits cold and warm in parallel
  local n="$1" base="drape7/drape-n$1/base"
  solve "$base" cold "" true
  for edit in w1 w2 w3; do
    ( solve "drape7/drape-n$n/$edit" cold; solve "drape7/drape-n$n/$edit" warm "$base/equilibrium-cold" ) &
  done
  wait
}

cargo build --release --manifest-path "$ENGINE/oracle/Cargo.toml"
generate $SIZES $GPU_ONLY
for n in $SIZES; do size_chain "$n" & done
wait
previous=""
for n in $SIZES; do
  if [ -n "$previous" ]; then
    fine="drape7/drape-n$n/base"
    "$ORACLE" e7-prolong --coarse "drape7/drape-n$previous/base/equilibrium-cold" --coarse-n "$previous" --fine-dir "$fine" --out "$fine/start-prolong-oracle"
    solve "$fine" prolong "$fine/start-prolong-oracle"
  fi
  previous="$n"
done
echo ORACLE-E7-DONE
