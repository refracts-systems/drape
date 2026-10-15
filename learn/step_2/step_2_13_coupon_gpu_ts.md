# Step 2.13: `gpu/src/coupon-gpu.ts`, Menyambungkan Kernel Membran ke Solver E1

> File kunci yang memenuhi syarat "reuse the existing E1 solver; do not redesign the architecture". Ia membuat solver E1 **tanpa mengubahnya**,
> lalu mengikat kernel perakitan ke buffer solver itu.

---

## 1. Antarmuka hasil

```ts
/** Membrane kernels plus the unchanged E1 PCG, whose vals/rhs the assembly kernels write in place. */
export interface CouponGpu<A, K> {
  readonly backend: Backend<A, K>;
  readonly solver: PreparedSolver<A, K>;
  readonly positions: A;
  readonly direction: A;
  readonly step: A;
  readonly elemGrad: A;
  readonly elemHess: A;
  readonly elemDelta: A;
  readonly element: readonly Dispatch<K>[];
  readonly evaluate: readonly Dispatch<K>[];
  readonly energyChange: readonly Dispatch<K>[];
}
```

Tiga daftar dispatch siap pakai:

| Daftar | Isi | Dipakai untuk |
|---|---|---|
| `element` | kernel elemen saja | cek kernel WGSL vs f64 |
| `evaluate` | elemen → rakit matriks → rakit gaya | setiap iterasi Newton |
| `energyChange` | perubahan energi per elemen | setiap percobaan line search |

Generik `<A, K>` sama seperti `Backend`: file ini tidak tahu apakah di bawahnya WebGPU atau vgpu.

## 2. "Problem" sintetis untuk solver E1

```ts
/** The E1 solver's view of the Newton system: the coupon's BSR pattern with zero values until assembled. */
function linearProblem(problem: CouponProblem): Problem {
  const m = problem.manifest;
  return {
    manifest: { case: m.case, grid_columns: m.grid_columns, rows: m.rows, unknowns: m.unknowns, blocks: m.blocks,
      replace_every: m.pcg.replace_every, replace_above: m.pcg.replace_above, arrays: [] },
    rowPtr: problem.u32.row_ptr, colIdx: problem.u32.col_idx, diagIdx: problem.u32.diag_idx,
    vals: new Float32Array(9 * m.blocks), rhs: new Float32Array(m.unknowns),
  };
}
```

`prepareSolver` (E1) mengharapkan `Problem` dari Step 1. Kita membuatnya dengan:

- pola BSR kupon (`row_ptr`, `col_idx`, `diag_idx`),
- `vals` dan `rhs` **nol**, karena nanti diisi kernel perakitan,
- pengaturan replacement dari `problem.json` (50 / 10⁻²).

Ini pola **adapter**: bentuk data Step 2 disesuaikan dengan antarmuka Step 1, tanpa mengubah Step 1.

## 3. Buffer milik Step 2

```ts
function meshArrays<A, K>(backend: Backend<A, K>, problem: CouponProblem): MeshArrays<A> {
  const scratch = (label: string, length: number): A => backend.array(label, new Float32Array(length), "read-write");
  return {
    positions: backend.array("positions", Float32Array.from(f32.initial_positions), "read-write"),
    direction: scratch("direction", 3 * manifest.vertices), step: scratch("step", 1),
    elemVertices: ..., elemGeom: ...,
    elemGrad: scratch("elem_grad", 9 * manifest.triangles), elemHess: scratch("elem_hess", 81 * manifest.triangles),
    elemDelta: scratch("elem_delta", manifest.triangles), dofMask: ...,
    blockPtr: ..., blockIdx: ..., forcePtr: ..., forceIdx: ...,
  };
}
```

Buffer "scratch" (kerja) dibuat kosong dan read-write. Buffer data tetap (geometri, peta) dibuat `read`.

## 4. Mengikat ke buffer solver

```ts
matrix: { label: "assemble_matrix", source: sources.assemble_matrix, constants: {},
  bindings: { row_ptr: s.rowPtr, col_idx: s.colIdx, dof_mask: a.dofMask, block_contrib_ptr: a.blockPtr, block_contrib_idx: a.blockIdx, elem_hess: a.elemHess, vals: s.vals } },
force: { label: "assemble_force", source: sources.assemble_force, constants: {},
  bindings: { dof_mask: a.dofMask, force_contrib_ptr: a.forcePtr, force_contrib_idx: a.forceIdx, elem_grad: a.elemGrad, rhs: s.rhs } },
```

`s = solver.arrays`. `vals: s.vals` dan `rhs: s.rhs` adalah **buffer yang sama** yang dibaca kernel PCG. Tidak ada salinan, tidak ada upload ulang dari host.

### Satu-satunya perubahan di solver E1

`solver.ts` membuat `vals` dan `rhs` dengan akses `"read"`. vgpu **menolak** kernel yang menulis ke buffer `read` (dibuktikan dengan probe di sesi
kerja). Perubahan minimalnya: kedua buffer dibuat `"read-write"`. Kernel E1 tetap mendeklarasikan `var<storage, read>`, dan probe kedua membuktikan
vgpu menerima buffer read-write diikat sebagai read. Aritmetika E1 tidak berubah (regresi tension-n33: 151 iterasi, residual identik).

## 5. Persiapan

```ts
export async function prepareCouponGpu<A, K>(backend: Backend<A, K>, problem: CouponProblem, sources: KernelSources): Promise<CouponGpu<A, K>> {
  const solver = await prepareSolver(backend, linearProblem(problem), sources);
  const arrays = meshArrays(backend, problem);
  const specs = stageSpecs(sources, problem, arrays, solver);
  const [element, energy, matrix, force] = await Promise.all([specs.element, specs.energy, specs.matrix, specs.force].map((spec) => backend.kernel(spec)));
  const elements = Math.ceil(problem.manifest.triangles / WORKGROUP);
  const rows = Math.ceil(problem.manifest.rows / WORKGROUP);
  return {
    ...,
    evaluate: [{ kernel: element, workgroups: elements }, { kernel: matrix, workgroups: rows }, { kernel: force, workgroups: rows }],
    energyChange: [{ kernel: energy, workgroups: elements }],
  };
}
```

Jumlah workgroup berbeda per kernel: kernel elemen per segitiga (2048 → 8 workgroup di n33), kernel perakitan per baris (1023 → 4 workgroup).

Solver dibuat **sekali**; setiap iterasi Newton hanya mengisi ulang `vals`/`rhs` lalu memanggil `solveOnce`. `setup` di `solveOnce` menjalankan
`jacobi_invert` ulang, sehingga preconditioner selalu cocok dengan matriks terbaru.

## 6. Diagram buffer

```
                   ┌─────────────── milik Step 2 ────────────────┐
positions ───────► membrane_element ──► elem_grad ──► assemble_force ──► rhs ──┐
elem_geom ───────►                  └─► elem_hess ──► assemble_matrix ──► vals ─┤ milik solver E1
                                                                                ▼
                                          jacobi_invert, cg_init, spmv_dot, ... (E1, tidak diubah)
                                                                                │
                                                                     x (= arah Newton d)
```

## 7. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Problem sintetis | pola kupon + nilai nol → solver E1 apa adanya |
| Buffer bersama | perakitan menulis ke `vals`/`rhs` solver |
| Perubahan E1 | hanya akses `read-write` pada vals/rhs |
| Tiga daftar dispatch | element, evaluate, energyChange |
| Preconditioner | dihitung ulang di setiap solve |
