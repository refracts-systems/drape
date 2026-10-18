# Step 2.14: `gpu/src/newton.ts`, Loop Newton f32 di Host

> Kembaran `newton.rs`, tetapi berjalan di TypeScript dengan fisika dan aljabar linear di GPU. Struktur dan urutannya sengaja **sama**
> dengan versi Rust supaya perbandingannya adil. Perbedaannya: langkah linear memakai PCG f32 (inexact), dan semua skalar dihitung dalam f32.

---

## 1. Catatan per iterasi

```ts
/** One Newton iterate: f32 residual ‖g‖/scale, the PCG solve and line search taken from it (NaN on the last iterate). */
export interface IterationRecord {
  readonly step: number;
  readonly iteration: number;
  readonly relativeResidual: number;
  readonly pcgIterations: number;
  readonly pcgStatus: number;
  readonly linearResidual: number;
  readonly stepLength: number;
  readonly halvings: number;
  readonly status: StepStatus;
}

export const STEP_STATUS = { running: 0, converged: 1, iteration_limit: 2, line_search_failed: 3 } as const;
```

Setiap iterate dicatat beserta posisinya (`iterates`). Ini memungkinkan oracle menghitung residual **f64** untuk setiap iterate f32 (dokumen 2.16).
`linearResidual` = residual sejati f32 PCG = **forcing term η terukur**.

Status sebagai objek `as const` (bukan `enum`): kode tetap lolos aturan `erasableSyntaxOnly`, sehingga bisa dijalankan Node tanpa transpiler.

## 2. Aritmetika f32 di host

```ts
const norm32 = (values: Float32Array): number => Math.fround(Math.sqrt(values.reduce((sum, v) => Math.fround(sum + Math.fround(v * v)), 0)));
const dot32 = (a: Float32Array, b: Float32Array): number => a.reduce((sum, v, i) => Math.fround(sum + Math.fround(v * b[i])), 0);
```

`number` JavaScript adalah f64. `Math.fround(x)` membulatkan x ke f32 terdekat. Dengan membulatkan setiap operasi, norma dan dot product di host
benar-benar aritmetika f32, sehingga "Newton f32" tidak diam-diam dibantu f64.

## 3. Merakit: satu panggilan

```ts
/** Assembles tangent and −g at `positions` into the solver's vals/rhs; returns −g. */
async function assemble<A, K>(gpu: CouponGpu<A, K>, positions: Float32Array<ArrayBuffer>): Promise<Float32Array<ArrayBuffer>> {
  gpu.backend.write(gpu.positions, positions);
  await gpu.backend.run(gpu.evaluate);
  return new Float32Array(await gpu.backend.read(gpu.solver.arrays.rhs));
}
```

Upload posisi → jalankan tiga kernel → baca `rhs` (= −g). Sesudahnya `vals` dan `rhs` solver sudah berisi sistem Newton.

## 4. Line search

```ts
async function lineSearch<A, K>(context: Context<A, K>, slope: number): Promise<LineSearch | null> {
  const { armijo, max_halvings } = context.problem.manifest.newton;
  if (!(slope < 0)) return null;
  for (let halvings = 0; halvings <= max_halvings; halvings += 1) {
    const stepLength = 0.5 ** halvings;
    if (await energyChange(context.gpu, stepLength) <= Math.fround(armijo * stepLength * slope)) return { stepLength, halvings };
  }
  return null;
}
```

- `!(slope < 0)` (bukan `slope >= 0`) juga menangkap **NaN**: `NaN >= 0` bernilai false, tetapi `!(NaN < 0)` bernilai true. Arah yang rusak ditolak.
- `energyChange` = kernel 2.9 + penjumlahan f32.

## 5. Menyebar dan memajukan

```ts
function scatter(problem: CouponProblem, rows: Float32Array): Float32Array<ArrayBuffer> {
  const vertices = new Float32Array(3 * problem.manifest.vertices);
  problem.u32.row_vertex.forEach((vertex, row) => vertices.set(rows.subarray(3 * row, 3 * row + 3), 3 * vertex));
  return vertices;
}

function advance(problem: CouponProblem, positions: Float32Array, direction: Float32Array, stepLength: number): void {
  problem.u32.row_vertex.forEach((vertex, row) => {
    [0, 1, 2].forEach((d) => { positions[3 * vertex + d] += stepLength * direction[3 * row + d]; });
  });
}
```

- Solusi PCG per-baris (titik bebas) → per-titik (kernel energy-change butuh per-titik). Titik jepit tetap 0.
- `positions` adalah `Float32Array`: menulis ke dalamnya otomatis membulatkan ke f32. **Posisi disimpan sebagai koordinat absolut f32.** Fakta inilah
  yang menentukan lantai residual (dokumen 2.17).

## 6. Prediktor dan penjepit

```ts
/** Load-step predictor: adds (F(to) − F(from))·X to every vertex, as in the oracle. */
function predict(problem: CouponProblem, positions: Float32Array, step: number): void {
  const [from, to] = [stepGradient(problem, step - 1), stepGradient(problem, step)];
  ...positions[3v + r] += (to[2r] − from[2r]) · X + (to[2r+1] − from[2r+1]) · Y ...
}

function applyBoundary(problem: CouponProblem, positions: Float32Array, step: number): void {
  const { fixed_vertex: fixed } = problem.u32;
  const offset = 3 * step * fixed.length;
  fixed.forEach((vertex, i) => positions.set(problem.f32.fixed_positions.subarray(offset + 3 * i, offset + 3 * i + 3), 3 * vertex));
}
```

`stepGradient(problem, -1)` = identitas (F(0) = I untuk semua jalur), jadi langkah pertama bergerak dari rest. Persis `Coupon::predict` dan
`apply_boundary`.

## 7. Satu update Newton

```ts
/** One Newton update from `positions`: E1 PCG on the assembled system, then the Armijo line search. */
async function update<A, K>(context, positions, rhs, where): Promise<boolean> {
  const run = await solveOnce(context.gpu.solver, pcgSettings(context.problem));
  const direction = run.solution;
  context.gpu.backend.write(context.gpu.direction, scatter(context.problem, direction));
  const search = await lineSearch(context, -dot32(rhs, direction));
  const status = search === null ? STEP_STATUS.line_search_failed : STEP_STATUS.running;
  record(context, { ..., pcgIterations: run.iterations, pcgStatus: linearStatus(run),
    linearResidual: run.gpuTrueRelativeResidual, stepLength: search?.stepLength ?? Number.NaN, ... }, positions);
  if (search !== null) advance(context.problem, positions, direction, search.stepLength);
  return search !== null;
}
```

1. **PCG E1** (`solveOnce`) menyelesaikan `K d = −g` dari buffer yang baru dirakit. Ini satu-satunya titik pemanggilan solver Step 1.
2. Upload arah ke GPU.
3. `slope = gᵀd = −(rhsᵀd)`.
4. Line search.
5. Catat, lalu majukan posisi.

## 8. Loop per langkah dan per jalur

```ts
async function solveStep<A, K>(context, positions, scale, step): Promise<void> {
  const { tolerance, max_iterations } = context.problem.manifest.newton;
  for (let iteration = 0; ; iteration += 1) {
    const rhs = await assemble(context.gpu, positions);
    const relative = Math.fround(norm32(rhs) / scale);
    if (relative <= tolerance || iteration === max_iterations) {
      record(context, terminal(step, iteration, relative, relative <= tolerance ? STEP_STATUS.converged : STEP_STATUS.iteration_limit), positions);
      return;
    }
    if (!(await update(context, positions, rhs, { step, iteration, relative }))) return;
  }
}

export async function solvePath<A, K>(gpu: CouponGpu<A, K>, problem: CouponProblem): Promise<PathResult> {
  const context: Context<A, K> = { gpu, problem, records: [], iterates: [] };
  const positions = Float32Array.from(problem.f32.initial_positions);
  const started = performance.now();
  for (let step = 0; step < problem.manifest.steps; step += 1) {
    const scale = await stepScale(context, positions, step);
    predict(problem, positions, step);
    applyBoundary(problem, positions, step);
    await solveStep(context, positions, scale, step);
  }
  return { records: context.records, iterates: context.iterates, solveMs: performance.now() - started };
}
```

Urutan identik dengan `solve_path` Rust: skala (dari kesetimbangan lama + penjepit baru), prediksi, penjepit, lalu Newton.

`for (let iteration = 0; ; ...)` tanpa kondisi: loop berakhir lewat `return` (konvergen, batas iterasi, atau line search gagal). Aturan "const over let"
tetap dipenuhi: `let` hanya untuk penghitung loop.

## 9. Cek kernel

```ts
export async function evaluateKernelState<A, K>(gpu: CouponGpu<A, K>, problem: CouponProblem): Promise<Float32Array<ArrayBuffer>> {
  gpu.backend.write(gpu.positions, problem.f32.kernel_state);
  await gpu.backend.run(gpu.element);
  const [gradient, tangent] = await Promise.all([gpu.backend.read(gpu.elemGrad), gpu.backend.read(gpu.elemHess)]);
  ... gabungkan jadi satu Float32Array ...
}
```

Dijalankan sekali sebelum jalur: kernel elemen pada keadaan uji dari oracle, untuk perbandingan per elemen.

## 10. Perilaku nyata

Uniaxial-n33 langkah pertama:

```
it 0 res 2.046e-3 pcg 123 lin 1.89e-5 alpha 1
it 1 res 6.441e-5 pcg 146 lin 8.41e-6 alpha 1
it 2 res 5.068e-6 pcg 153 lin 3.11e-7 alpha 1
it 3 res 5.118e-6 ...      ← lantai
...
it 30 res 5.114e-6 status 2 (iteration_limit)
```

- Iterasi 0–2 mengikuti f64 (2,05×10⁻³ → 6,41×10⁻⁵ → …).
- PCG konvergen setiap kali; η (lin) ≪ 10⁻².
- Setelah mencapai ~5×10⁻⁶, residual hanya berfluktuasi di level derau sampai batas 30 iterasi. Tidak ada exit "stagnan", sesuai instruksi
  untuk tidak menambah atau mengubah kriteria.

## 11. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Struktur = Rust | perbandingan adil |
| `Math.fround` | skalar host tetap f32 |
| `!(slope < 0)` | juga menolak NaN |
| Posisi `Float32Array` | koordinat absolut f32 → sumber lantai |
| `solveOnce` | satu-satunya panggilan solver E1 |
| Semua iterate disimpan | oracle bisa mengevaluasinya dalam f64 |
