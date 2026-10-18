# drape

Physically-based cloth simulation for a browser jersey editor. The GPU solver (WebGPU/WGSL, f32) is validated at every step
against a dependency-free Rust f64 oracle, and no tolerance is ever loosened to obtain a pass.

Goal: a simulation-based Jersey Editor close to real life, from small garment shops up to club-level designers, embeddable in web shops.

```
engine/
  oracle/        Rust, f64 reference: material, contact, Newton, pseudo-transient drape, scoring commands
  gpu/           WGSL kernels + TypeScript runner (Node, vgpu, WebGPU)
editor/          placeholder only: the editor lives in the private repository refracts-systems/drape-studio (see editor/README.md)
experiments/     one folder per experiment: inputs, references, scored results, run scripts
  run-e1.sh … run-e7.sh, oracle-e7.sh
  cases/ coupon/ drape/ drape4-final/ drape5/ drape7/ results/
docs/            step-by-step walkthrough of every experiment (Indonesian), docs/README.md is the index
```

## Status

Experiments 1–7 are done (f32 PCG, membrane Newton, dynamic drape, IPC contact on a sphere, honest Newton at the f32 floor,
quasi-static drape by pseudo-transient continuation). Latest results are in `experiments/results/` and `docs/step_7/`.
Open problem: PCG with a block-Jacobi preconditioner does not scale to n129 (10.8x more PCG iterations per Newton step than at n65);
a multigrid preconditioner is the next step.

## Run

Requirements: Rust (stable), Node >= 22.18, a hardware GPU adapter (software adapters are rejected on purpose).

```
cd engine/oracle && cargo build --release && cargo test --release
cd engine/gpu && npm install && npm run typecheck
cd experiments && ./run-e7.sh          # STAGES="oracle gpu score"; see the header of each script
```

Scripts run from `experiments/` and find the engine through `../engine`. Run data (`*.bin`, several GB) is not in git; regenerate it with
the `oracle` stage of the scripts. The oracle reference of drape4-final and drape5 takes hours of CPU, so those stored references are not regenerated casually.

## License

The engine, experiments and docs are licensed under the Apache License 2.0 (see `LICENSE` and `NOTICE`).
The editor is developed in a separate private repository and will be licensed separately; the decision is open (see `editor/README.md`). External contributions wait for a CLA (see `CONTRIBUTING.md`).
