# AGENTS.md: context for the next thread or model

Read this first. It is the hand-off of the work so far (state on 2026-10-18, last experiment commit `7427e1c`, repo pushed to
`https://github.com/refracts-systems/drape`, branch `main`). Verify anything below against `git log` and `experiments/results/*.json`.

## Goal

A simulation-based **Jersey Editor** close to real life, usable from small local garment shops up to club-level designers (Barcelona / Real Madrid),
embeddable in web shops. The editor needs the **final drape** (static equilibrium of the cloth on a body), interactively.

Business plan (decided): the engine here stays **Apache-2.0**; the editor is a **hosted product** (subscription with cloud simulation first, then an
embeddable widget/SDK for web shops). Editor code lives in the **private** repo `refracts-systems/drape-studio`, never in this public repo (history stays
readable). Editor licence is undecided (proprietary vs BSL/FSL). No outside contributions until a CLA exists. `NOTICE` copyright holder is a placeholder.

## Layout

```
engine/oracle/   Rust, zero dependencies, f64 everywhere: material, contact, Newton, PTC drape, scoring commands (main.rs lists them)
engine/gpu/      WGSL kernels + TypeScript runner (Node >= 22.18 runs .ts directly; vgpu/Dawn for the hardware adapter)
experiments/     run-e1..e7.sh, oracle-e7.sh and the per-experiment data: cases, coupon, drape, drape4-final, drape5, drape7, results
docs/            Indonesian step-by-step walkthrough of every experiment (docs/README.md is the index, step_1 … step_7)
editor/          placeholder README only
```

Scripts run from `experiments/` and find the engine through `../engine`. Run data (`*.bin`, ~4 GB) is not in git. Old text in `docs/step_*` still uses
the pre-restructure paths (`oracle/`, `gpu/`, `drape5/`).

## The experiment ladder (what was learned)

- E1–E2: f32 block-Jacobi PCG and nonlinear membrane Newton on GPU against the f64 oracle.
- E3–E4: dynamic drape; E4-final rung D = 106 ms/step at n65 (real-time budget 4.17 ms/step).
- E5: frictionless IPC contact with analytic colliders (sphere, floor); never penetrates.
- E6: honest contact Newton at the f32 floor; GPU reductions cut readback 3–5x, timing change is inside run-to-run noise.
- E7 (`docs/step_7`, `experiments/results/experiment7-vgpu-hardware.json`): quasi-static drape by pseudo-transient continuation (PTC).
  Cold equilibrium on an RTX 3050 Laptop: n17 19 s, n33 53 s, n65 45 s (f32 ends by stagnation; the f32 representation floor of r/L is 1e-2 / 4e-2 / 0.4),
  n129 does NOT converge (20 min wall budget `time_budget`; PCG up to 200k iterations per step). Warm start after an edit is ~1.4–5x faster in wall time up to
  n65 (n33 warm/cold Newton 0.38 / 0.23 / 0.28, gate 0.25 mostly missed). Two-tier n33 = 0.51x fine cold (gate 0.5, just missed). Accuracy gates fail at
  n17 base cold, n33 base cold (probe one-step) and n33 prolong, and two μ-cap oracle cases. E1–E6 n33 outputs are bit-identical after E7.
- **Open problem / next step:** PCG per Newton step grows 10.8x from n65 to n129 with the block-Jacobi preconditioner. A multigrid (or hierarchical) preconditioner
  is the next lever, then a browser vertical slice (two sewn panels on an analytic body) and real fabric test data. The interactive target (10 s at n65,
  30 s at n129) is not met.

## Rules that must be kept

- The f64 oracle is the reference. **Never loosen a tolerance to obtain a pass**; report the failing gate.
- Hardware must be the NVIDIA GeForce RTX 3050 Laptop GPU (D3D12 via vgpu, driver 32.0.16.1714). The AMD iGPU and software adapters are not acceptable; never
  fall back silently (`assertTargetAdapter`).
- Do not change E1–E6 behaviour (outputs bit-identical except timing); do not move the Newton loop onto the GPU; do not change the E1 PCG kernels.
- Do **not** regenerate `drape4-final/` or `drape5/` oracle references (about 3 h of CPU at n65).
- Do not run heavy CPU work (oracle generation or scoring) while timing GPU runs.
- Functions stay short (< ~20 lines); match surrounding style; no comments that restate code.
- Regression check before claiming no change: run the n33 case of E1 (run-vgpu.ts), E2 (run-e2.ts), E3 (run-e3.ts), E4-final rung D (run-e4.ts), E5 (run-e5.ts), E6 rung D
  (run-e6.ts) and compare `md5sum` of the `*.bin` outputs before/after; then `git checkout` the rewritten `gpu-result.json` timing files and rescore.

## Commands

```
cd engine/oracle && cargo build --release && cargo test --release     # 46 tests
cd engine/gpu && npm install && npm run typecheck
cd experiments && STAGES="gpu score" ./run-e7.sh                        # see the header comment for SIZES, REPEATS, FIRST_ROUND, PREVIOUS, BUDGET_LARGE
```

Single GPU case: `cd engine/gpu && node src/node/run-e7.ts --case <abs path>/drape7/drape-n33/base --mode cold` (`E7_PROGRESS=1` prints a line per step,
`E7_BUDGET_S=1200` sets a wall-clock stop). Score: `oracle e7-compare --dir <case> --mode cold`, summary: `oracle e7-summarize --dir drape7 --out results/...`.

## Gotchas learned the hard way

- Windows + Git Bash: Node needs `C:/dev/...` paths, not `/c/dev/...` (scripts use `pwd -W`). Use the Write tool for large files; heredocs with apostrophes break.
- `${VAR:-default}` treats an empty string as unset: use `${VAR-default}` when `VAR=` must mean empty (this once re-ran n17 and overwrote timings).
- `--start` paths are resolved from `engine/gpu/`, so pass absolute paths.
- The laptop GPU throttles (same case up to 28% slower later), so timings use interleaved rounds + median (`merge-runs.ts`). Run-to-run spread is in `ms_per_step_runs`.
- A very slow n129 solve looks like a hang; use `E7_PROGRESS=1`. Background shells have time limits; detach long runs and poll a marker file.
- After `cargo test` rebuild with `cargo build --release` (a stale binary was used once).
- PCG and the PTC path are chaotic at the step level in f32; one-step errors against the oracle from probe states can be large, which is a finding and not a bug to hide.

## Working style with the user

The user writes Indonesian; answer in Indonesian. Commit messages are English. Do not push or create remote objects unless the user asks (they usually say "push").
Commits end with the attribution line given by the harness at the time. Be honest about failed gates; do not describe a pass that did not happen.
Pending housekeeping: replace the placeholder copyright holder in `NOTICE`, set up a real CLA, update stale paths in `docs/step_*`, decide the editor licence.
