# Experiment 1: f32 Jacobi-PCG on WebGPU against an f64 oracle

Question: can TypeScript + WGSL (raw WebGPU and vgpu) solve the linear systems of an implicit thin-shell FEM
in f32 accurately enough, deterministically, and fast enough? Everything else in the jersey programme waits on this.

```
jersey-feasibility/
├── run-e1.sh                 generate → GPU solve → score → summarize (all cases)
├── oracle/                   Rust, zero dependencies, f64 everywhere
│   └── src/
│       ├── main.rs           CLI: generate | compare | summarize
│       ├── material.rs       orthotropic St.Venant–Kirchhoff knit membrane + bending constants
│       ├── membrane.rs       CST membrane gradient/Hessian (clamped-stress PSD projection)
│       ├── hinge.rs          discrete-hinge bending, Gauss–Newton Hessian
│       ├── mesh.rs           clamped square sheet, alternating diagonals, hinges
│       ├── cases.rs          tension | slack | dynamic matrices and RHS
│       ├── bsr.rs            3×3 block CSR (the GPU format)
│       ├── band.rs           f64 band Cholesky: the exact solution
│       ├── pcg.rs            Jacobi-PCG generic over f32/f64, same arithmetic order as the WGSL
│       ├── lanczos.rs        κ of the preconditioned operator from CG coefficients
│       ├── metrics.rs        true residual, error vs f64, in-/out-of-plane split
│       ├── generate.rs       writes problem.{json,bin} and reference.{json,bin}
│       ├── compare.rs        scores gpu-x.bin → result.json
│       ├── criteria.rs       per-case pass/fail thresholds
│       ├── summarize.rs      results/experiment1-*.json + iteration-growth check
│       └── jsonscan.rs, io.rs, linalg.rs, testing.rs
├── gpu/                      TypeScript (Node ≥ 22.18 runs .ts directly), WGSL
│   ├── kernels/              state, jacobi_invert, cg_init, spmv_dot, reduce, update_dot, direction, residual
│   └── src/
│       ├── backend.ts        the 6-method GPU interface the solver needs
│       ├── solver.ts         buffers, dispatch order, chunked convergence check
│       ├── experiment.ts     cold/warm/repeat runs → e1-gpu/1 report
│       ├── webgpu-backend.ts raw WebGPU (timestamp-query when available)
│       ├── browser-main.ts   page entry for the raw-WebGPU path
│       ├── problem.ts, kernels.ts
│       └── node/             run-browser.ts (Playwright Chromium), run-vgpu.ts, vgpu-backend.ts, cli.ts, files.ts
└── cases/<case>-n<N>/        problem.*, reference.*, gpu-x.bin, gpu-result.json, result.json
```

## Run

Prerequisites: Rust stable, Node ≥ 22.18, a GPU with WebGPU support.

```bash
cd jersey-feasibility/gpu
npm install
npx playwright install chromium
cd ..
./run-e1.sh webgpu hardware            # raw WebGPU in headless Chromium on the real GPU
```

vgpu path (same kernels, same dispatch order, Dawn in Node):

```bash
cd gpu
npm install vgpu @vgpu/wgsl
npx vgpu doctor --no-render            # if it reports missing Dawn: npx vgpu install-dawn
npm run typecheck                      # checks vgpu-backend.ts against the real vgpu types
cd ..
./run-e1.sh vgpu hardware
```

One case by hand:

```bash
./oracle/target/release/oracle generate --case dynamic --n 129 --out cases/dynamic-n129
(cd gpu && node src/node/run-browser.ts --case ../cases/dynamic-n129 --adapter hardware --repeats 3 --chunk 32)
./oracle/target/release/oracle compare --dir cases/dynamic-n129
./oracle/target/release/oracle summarize --dir cases --out results/experiment1-webgpu-hardware.json
```

Jersey-scale run (≈ 195k unknowns, 2 mm elements): the deciding measurement for f32. The oracle needs ~2.5 GB RAM and ~5 min for the band Cholesky.

```bash
CASES="tension dynamic" SIZES="257" ./run-e1.sh webgpu hardware
```

No GPU: `--adapter software` (SwiftShader). Numbers are valid, timings are not.

## result.json (schema e1-result/1)

| field | meaning |
|---|---|
| `unknowns` | 3 × free vertices |
| `iterations` | GPU PCG iterations to ‖r‖ ≤ 1e-7‖b‖ (recursive residual) |
| `iterations_cpu_f32`, `iterations_cpu_f64` | the same PCG in Rust f32 (GPU arithmetic order) and f64 |
| `relative_residual` | ‖b − A x_gpu‖ / ‖b‖, evaluated in f64 against the f64 matrix |
| `recursive_relative_residual` | the GPU's own √(rᵀr / r₀ᵀr₀); always looks converged |
| `gpu_f32_true_relative_residual` | the GPU recomputing b − A x in f32 (drift check available at runtime) |
| `residual_floor_f32_rounded_exact` | residual of the exact solution rounded to f32: no f32 solve can beat it |
| `relative_error_vs_f64` | ‖x_gpu − x_f64‖ / ‖x_f64‖ (plus `_in_plane`, `_out_of_plane`) |
| `storage_error_f32_system` | error of the exact solution of the f32-rounded matrix: the precision floor |
| `condition_jacobi_preconditioned` | κ(D⁻¹A) from the f64 CG–Lanczos coefficients |
| `bitwise_equal_fraction_vs_cpu_f32_pcg` | fraction of x bits identical to the Rust f32 emulation |
| `deterministic_repeat` | repeated GPU solves are bitwise identical |
| `gpu_time_ms` | warm solve, GPU timestamps summed over passes (null without timestamp-query) |
| `solve_time_ms` | warm solve wall time: setup dispatches → convergence → true residual |
| `total_time_ms` | cold: upload + pipeline compile + first solve + x readback |
| `ms_per_iteration`, `effective_bandwidth_gb_s` | warm cost; compulsory traffic 40 B/block + 196 B/row per iteration |
| `cpu_f64_pcg_single_thread_ms` | the Rust f64 PCG on one core, the speed bar the GPU must clear |
| `speedup_vs_cpu_f64_single_thread` | `cpu_f64_pcg_single_thread_ms / solve_time_ms` |
| `verdict` | `converged`; `forcing_term_ok` (`relative_residual` ≤ 1e-2: Newton still gains two digits per iteration); `f32_iterations_ok` (≤ 1.5 × f64 iterations); `faithful_to_cpu_f32_emulation`; `deterministic`; `speedup_ok` (≥ 10×) |
