# Step 3.16: `gpu/src/drape-step.ts`, Langkah Waktu f32, Lintasan, Probe, dan Cek Kernel

> Logika waktu di host: prediksi x̂, skala, Newton (inti E2), kecepatan baru. Plus dua mode khusus: **probe** (satu langkah dari keadaan f64) dan cek kernel.

---

## 1. Prediksi dalam aritmetika f32

```ts
/** x̂ = x + h·v + h²·g on free vertices in f32 arithmetic (same order as the oracle); pinned vertices stay. */
function prediction(problem: DrapeProblem, state: State): Float32Array<ArrayBuffer> {
  const h = Math.fround(problem.manifest.time_step_s);
  const fall = Math.fround(Math.fround(h * h) * Math.fround(-problem.manifest.gravity_m_per_s2));
  const predicted = Float32Array.from(state.positions);
  problem.u32.row_vertex.forEach((vertex) => {
    [0, 1, 2].forEach((d) => {
      const i = 3 * vertex + d;
      const drift = Math.fround(state.positions[i] + Math.fround(h * state.velocity[i]));
      predicted[i] = d === 2 ? drift + fall : drift;
    });
  });
  return predicted;
}
```

- Oracle: `x + h·v + h²·g`, dievaluasi kiri ke kanan. Di sini setiap operasi dibulatkan ke f32 (`Math.fround`) dengan urutan yang sama.
- Menulis ke `Float32Array` (`predicted[i] = drift + fall`) otomatis membulatkan sekali lagi ke f32.
- Hanya titik bebas (`row_vertex`) yang diprediksi; titik dipaku tetap.

## 2. Satu langkah waktu

```ts
/** One time step: write x̂, scale by ‖∇Φ(x_n)‖, Newton on Φ from x̂. */
async function timeStep<A, K>(context: NewtonContext<A, K>, gpu: DrapeGpu<A, K>, problem: DrapeProblem, state: State, step: number): Promise<State> {
  const xHat = prediction(problem, state);
  gpu.backend.write(gpu.xHat, xHat);
  const scale = norm32(await assemble(gpu, state.positions));
  const next = Float32Array.from(xHat);
  await solveStep(context, next, scale, step);
  return { positions: next, velocity: velocityOf(next, state.positions, Math.fround(problem.manifest.time_step_s)) };
}
```

1. Tulis x̂ ke GPU. Kernel inersia membacanya.
2. **Skala** = ‖∇Φ(x_n)‖: rakit di posisi lama (dengan x̂ baru), lalu ambil norma rhs.
3. Newton mulai dari x̂ (salinan). `solveStep` (inti E2) memperbaruinya di tempat.
4. Kecepatan baru `(x_{n+1} − x_n)/h`.

Seluruh langkah ini sama dengan `time_step` di `drape.rs`.

## 3. Lintasan penuh

```ts
export async function simulateDrape<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem): Promise<TrajectoryResult> {
  const context: NewtonContext<A, K> = { gpu, model: drapeModel(problem), records: [], iterates: null };
  ...
  for (let step = 0, state: State = { positions: rest, velocity: new Float32Array(rest.length) }; step < problem.manifest.steps; step += 1) {
    state = await timeStep(context, gpu, problem, state, step + 1);
    positions.push(state.positions);
  }
  return { records: context.records, positions, ms: performance.now() - started };
}
```

- `iterates: null`: lintasan tidak menyimpan iterate Newton (dokumen 3.13), hanya posisi akhir setiap langkah (241 keadaan).
- Waktu total → ms/langkah.

## 4. Probe: satu langkah dari keadaan f64

```ts
/** One GPU step from each f64 probe state (state k−1 of the oracle trajectory), keeping every iterate. */
export async function runProbes<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem): Promise<ProbeResult> {
  const context: NewtonContext<A, K> = { gpu, model: drapeModel(problem), records: [], iterates: [] };
  for (let probe = 0; probe < problem.manifest.probes; probe += 1) {
    await timeStep(context, gpu, problem, probeState(problem, probe), problem.u32.probe_steps[probe]);
  }
  ...
}
```

**Kenapa probe adalah perbandingan penentu?** Kain yang berkerut bersifat **chaos**: perbedaan sekecil pembulatan bisa membuat kerutan muncul di tempat lain
setelah ratusan langkah. Membandingkan lintasan f32 dan f64 titik demi titik di t = 1 s tidak bisa membedakan "solver salah" dari "fisika sensitif".
Probe memulai GPU dari **keadaan f64 yang persis sama** (dibulatkan f32), menjalankan satu langkah, lalu membandingkannya dengan langkah f64 berikutnya.
Itu mengukur galat **solver**, bukan galat **lintasan**.

Analogi: membandingkan dua implementasi service dengan *replay* request yang sama dari snapshot yang sama, bukan membandingkan state database setelah sebulan
traffic yang berbeda.

Probe menyimpan semua iterate supaya oracle bisa mengevaluasi residual f64 setiap iterate dan menghitung lantai representasi.

## 5. Cek kernel

```ts
/** Membrane and hinge gradients and Hessians at the oracle's kernel state: exact kernels, then projected ones. */
export async function evaluateDrapeKernels<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem): Promise<Float32Array<ArrayBuffer>> {
  const parts = [...await elementArrays(gpu, problem, true), ...await elementArrays(gpu, problem, false)];
  ...
}
```

Hasilnya `[gradien | Hessian]` versi eksak, lalu `[gradien | Hessian]` versi proyeksi. Hanya bagian membran + hinge yang diambil (`new Float32Array(buffer, 0, panjang)`
adalah *view* ke awal buffer tanpa menyalin).

## 6. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| x̂ dalam f32 | `Math.fround` per operasi, urutan oracle |
| Skala | ‖∇Φ(x_n)‖ dengan x̂ baru |
| Lintasan | 240 langkah, posisi disimpan, iterate tidak |
| Probe | satu langkah dari keadaan f64 → galat solver, bukan chaos |
| Cek kernel | eksak dan proyeksi di keadaan terlipat |
