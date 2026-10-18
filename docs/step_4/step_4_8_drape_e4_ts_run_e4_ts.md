# Step 4.8: `drape-e4.ts` dan `node/run-e4.ts`, Langkah Waktu Robust, Probe, Cek Inersia, dan Runner Ablation

> Dua file yang merangkai Newton robust (dokumen 4.7) menjadi simulasi drape lengkap, lalu menjalankan satu rung ablation per pemanggilan.

---

## 1. `drape-e4.ts`: langkah waktu

```ts
async function timeStep<A, K>(ctx: RobustContext<A, K>, gpu: DrapeGpu<A, K>, problem: DrapeProblem, state: State, step: number): Promise<State> {
  const xHat = prediction(problem, state);
  gpu.backend.write(gpu.xHat, xHat);
  gpu.backend.write(gpu.regularization, Float32Array.of(0));
  const scale = norm32(await inPhase(ctx.clock, "assembly", () => assemble(gpu, state.positions)));
  const next = Float32Array.from(xHat);
  await solveStepRobust(ctx, next, scale, step, state.positions);
  return { positions: next, velocity: velocityOf(next, state.positions, ...) };
}
```

Sama dengan E3 (dokumen 3.16), dengan dua perbedaan:

- μ direset ke 0 sebelum mengukur skala. Skala adalah ‖∇Φ(x_n)‖ dan tidak bergantung pada μ, tetapi matriks yang ikut dirakit sebaiknya bersih.
- `solveStepRobust` dengan `origin = state.positions` (x_n).

## 2. Rincian waktu per langkah

```ts
/** Mean per-step wall clock: GPU submits per phase, host I/O, and the remainder ("other"). */
function phaseMeans(clock: PhaseClock, totalMs: number, steps: number) {
  const { assembly, pcg, line_search } = clock.runMs;
  const other = totalMs - assembly - pcg - line_search - clock.ioMs;
  return { assembly: assembly / steps, pcg: pcg / steps, line_search: line_search / steps, io: clock.ioMs / steps, other: other / steps };
}
```

Jam direset (`resetClock`) sebelum lintasan, sehingga cek kernel dan probe tidak tercampur.

## 3. Cek kernel dengan inersia

```ts
/** The E3 element-kernel check plus the inertia kernel at x = kernel state, x̂ = rest, μ = kernel_mu: [gradient 3R | Hessian 9R]. */
export async function evaluateKernelsWithInertia<A, K>(gpu, problem, rules): Promise<Float32Array<ArrayBuffer>> {
  const elements = await evaluateDrapeKernels(gpu, problem);
  ...
  gpu.backend.write(gpu.positions, problem.f32.kernel_state);
  gpu.backend.write(gpu.xHat, problem.f32.rest_positions);
  gpu.backend.write(gpu.regularization, Float32Array.of(rules.kernel_mu));
  await gpu.backend.run(gpu.inertia);
  ...
}
```

Output = output cek E3 (membran + hinge, eksak + proyeksi) **ditambah** bagian inersia di belakang. Penilai E3 membaca bagian depan dengan offset yang sama dan
mengabaikan sisanya, sehingga format tetap kompatibel.

`new Float32Array(buffer, byteOffset, length)` membuat **view** ke bagian buffer tanpa menyalin. `byteOffset` dalam byte, jadi dikali 4.

## 4. `run-e4.ts`: rung

```ts
/** A: E3 configuration. B: + floor-aware stop. C: + Eisenstat–Walker forcing. D: + regularised exact tangent. */
const RUNGS = {
  A: { projected: true, settings: { floorStop: false, forcing: false, regularize: false } },
  B: { projected: true, settings: { floorStop: true, forcing: false, regularize: false } },
  C: { projected: true, settings: { floorStop: true, forcing: true, regularize: false } },
  D: { projected: false, settings: { floorStop: true, forcing: true, regularize: true } },
} as const;
type Rung = keyof typeof RUNGS;
```

Tabel konfigurasi sebagai data (`as const`), dan tipe `Rung` = `"A" | "B" | "C" | "D"` diturunkan darinya. Validasi argumen memakai `find` atas kunci, sehingga
nilai yang salah ditolak tanpa cast.

## 5. Backend yang diukur

```ts
const clock = createPhaseClock();
const inner = await createVgpuBackend("hardware");
...
assertTargetAdapter("hardware", inner.adapter);
const gpu = await prepareDrapeGpu(timeBackend(inner, clock), problem, sources, RUNGS[rung].projected);
```

Solver, kernel, dan Newton semuanya menerima backend yang **dibungkus** decorator waktu (dokumen 4.6). Penjaga adapter tetap memeriksa adapter asli.

## 6. Output per rung

Semua ke `drape4/drape-n<N>/rung-<X>/`:

| File | Isi |
|---|---|
| `gpu-kernel.bin` | cek kernel E3 + inersia |
| `gpu-trajectory.bin` | posisi 241 keadaan |
| `gpu-iterations.bin`, `gpu-robust.bin` | catatan iterate lintasan (9 kolom) + catatan robust (5 kolom) |
| `gpu-probe-iterations.bin`, `gpu-probe-robust.bin`, `gpu-probe-iterates.bin` | sama, untuk probe |
| `gpu-result.json` | rung, saklar, ms/langkah, **rincian fase**, info host |

```ts
const table = <T>(rows: readonly T[], fields: readonly (keyof T)[]): Float64Array<ArrayBuffer> =>
  Float64Array.from(rows.flatMap((row) => fields.map((field) => Number(row[field]))));
```

Satu fungsi generik menulis tabel f64 untuk kedua jenis catatan. `keyof T` menjamin nama kolom ada di tipe.

## 7. Ringkasan cetak

```
drape-n33 rung D: 4.53 Newton/step, 46.2 PCG/solve, 0 residual / 238 update stops, 70.8 ms/step [assembly 12.4, pcg 44.2, line_search 2.3, io 7.5, other 4.5]
```

## 8. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Langkah waktu | E3 + Newton robust + origin x_n |
| Rincian fase | jam direset sebelum lintasan |
| Cek inersia | μ uji 0,37; format kompatibel dengan penilai E3 |
| Rung sebagai data | `RUNGS as const` + tipe turunan |
| Output per rung | folder `rung-X/` |
