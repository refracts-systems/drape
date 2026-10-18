# Experiment 1 verdict: FAIL on `speedup_ok` only

**Decision:** Experiment 1 is recorded as **FAIL on `speedup_ok` only**. Every accuracy gate passes at n257 (195,075 unknowns) with residual replacement enabled.
`SPEEDUP_LIMIT` (10× over a one-core f64 PCG) and all other gates stay as defined in `oracle/src/criteria.rs`.
Speedup remains an open question for stronger hardware.

## Hardware

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB GDDR6, 128-bit, 176 GB/s rated, 50 W, driver 32.0.16.1714 |
| Backend | D3D12. Raw WebGPU runs in Playwright Chromium 141 (Dawn, FXC shader compiler); vgpu 0.6 runs under Node (Dawn). |
| Adapter strings | `nvidia / ampere / 0x25a2 / NVIDIA GeForce RTX 3050 Laptop GPU / D3D12 / D3D12 driver version 32.0.16.1714`; vgpu: `D3D12 driver version 32.0.16.1714 (gpu)` |
| Also present, not used | AMD Radeon(TM) Graphics iGPU, driver 31.0.21925.1001. Both runners reject it and any software adapter. |
| CPU (f64 bar) | AMD Ryzen 5 5600H, one core, on AC power, Windows Balanced plan |
| OS | Windows 11 Home Single Language 10.0.26300 |

## Configuration

- Jacobi-PCG in f32 with tolerance 1e-7 (recursive residual).
- Residual replacement r = b − Ax every 50 iterations, applied only while ‖r‖/‖r₀‖ > 1e-2.
- Without the 1e-2 gate, every replacement resets r up to the f32 rounding floor of b − Ax, so PCG never reaches 1e-7. Gates of 1e-3 and 3.45e-4 make CG diverge.

## Results at n257 (raw WebGPU, with replacement)

| | tension | dynamic |
|---|---|---|
| iterations: GPU / CPU f32 / CPU f64 | 1545 / 1608 / 1386 | 1495 / 1495 / 1493 |
| true relative residual, evaluated in f64 | **7.84e-3** (1.25e-2 without replacement) | 4.67e-3 (5.94e-3 without) |
| relative error vs f64 / f32 storage error | 8.4e-4 / 8.5e-4 | 9.8e-5 / 9.8e-5 |
| warm solve / GPU timestamp time | 1226 / 1091 ms | 1158 / 1051 ms |
| effective bandwidth | 58.3 GB/s (33% of rated) | 58.6 GB/s (33% of rated) |
| one-core f64 PCG | 8.06 s | 9.98 s |
| speedup, raw WebGPU / vgpu | **6.57× / 5.93×** | **8.62× / 7.53×** |

## Gates

| gate | result |
|---|---|
| converged | pass |
| `forcing_term_ok` (true residual ≤ 1e-2) | pass, with residual replacement |
| `f32_iterations_ok` (≤ 1.5× f64) | pass (1.11×, 1.00×) |
| `faithful_to_cpu_f32_emulation` | pass |
| `deterministic` | pass |
| **`speedup_ok` (≥ 10×)** | **FAIL**: tension 6.57×, dynamic 8.62× |
| iteration growth per mesh halving < 3× | pass: tension 2.01 / 2.08 / 2.45, dynamic 2.54 / 2.12 / 2.15 |
| vgpu overhead ≤ 1.15× raw WebGPU | pass: 1.11× tension, 1.14× dynamic |
| storage error at n257 ≤ 1e-3 | pass: 8.5e-4 tension, 9.8e-5 dynamic |
| effective bandwidth ≥ 25% of rated | pass: 33% from GPU timestamps, so sliced BSR is not triggered |

`slack` is the known-fail diagnostic. Its κ reaches 8e8, its iteration growth is about 4×, and it diverges once replacement is enabled.

## Context for the speedup result

- **The f64 CPU bar is unstable.** The same 1386-iteration f64 tension solve took **8.06 s** in one oracle run and **12.2 s** in another, so speedup measurements range from 6.6× to 9.8× for tension and 8.6× to 13.1× for dynamic.
- **Tension never reached 10× in any of the four measurements** (best: 9.82×).
- **The GPU side is stable**, at about 0.79 ms per iteration wall time and 0.70 ms per iteration from GPU timestamps.
- **The solve is bandwidth bound.** It reaches 33% of the 176 GB/s rated bandwidth of a 50 W, 4 GB laptop part. 10× needs either more effective bandwidth or a slower bar; neither is changed here.
- **No architecture change is forced.** Neither defined trigger fired: iteration growth stayed below 3× (no multigrid) and bandwidth stayed at or above 25% (no sliced BSR).

## Observations

- **Remaining residual source.** With replacement, tension's true residual (7.8e-3) still sits about 10× above the f32 storage floor (7.6e-4, the exact solution of the f32-rounded system). The remainder comes from f32 arithmetic inside the iteration.
- **Bitwise agreement with the CPU emulation.** The D3D compilers contract a·b + c into FMA, which WGSL permits and offers no way to disable. GPU results therefore agree bitwise with the CPU f32 emulation in only 0.4–10% of entries, while iteration counts agree.
- **DXC is unavailable in Playwright Chromium.** Its build lacks `dxil.dll`, so the raw WebGPU path runs Dawn with `--disable-dawn-features=use_dxc` (FXC).

## Evidence

- `results/experiment1-webgpu-hardware.json` and `results/experiment1-vgpu-hardware.json`: with replacement (`REPLACE_EVERY=50 REPLACE_ABOVE=1e-2`).
- `results/experiment1-webgpu-hardware-baseline.json` and `results/experiment1-vgpu-hardware-baseline.json`: without replacement.
- `cases/`, `cases-baseline/`: per-case `problem.json`, `reference.json`, `gpu-result.json`, `result.json`.
- `results/experiment1-webgpu-software.json`: SwiftShader. These numbers are not hardware evidence.
