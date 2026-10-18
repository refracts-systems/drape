# Step 3.17: `gpu/src/node/run-e3.ts`, Runner Step 3 dan Format Output

> Titik masuk GPU Step 3, dengan struktur yang sama seperti `run-e2.ts`: penjaga adapter, persiapan, kerja, output biner untuk oracle.

---

## 1. Alur `main`

```ts
async function main(): Promise<void> {
  const { caseDir, adapter } = options();
  const [problem, sources] = await Promise.all([loadDrape(caseDir), loadKernels()]);
  const backend = await createVgpuBackend(adapter);
  try {
    assertTargetAdapter(adapter, backend.adapter);
    const gpu = await prepareDrapeGpu(backend, problem, sources);
    const kernel = await evaluateDrapeKernels(gpu, problem);
    const probes = await runProbes(gpu, problem);
    const trajectory = await simulateDrape(gpu, problem);
    const out: Outputs = { adapter: backend.adapter, kernel, trajectory, probes };
    await writeOutputs(caseDir, problem, out);
    printSummary(problem, out);
  } finally {
    backend.dispose();
  }
}
```

Urutan kerja: cek kernel → probe (5 langkah) → lintasan (240 langkah). Probe dijalankan lebih dulu supaya waktu lintasan tidak tercampur dengan pemanasan pipeline.

## 2. Output

| File | Isi |
|---|---|
| `gpu-kernel.bin` | f32: gradien + Hessian membran/hinge, versi eksak lalu proyeksi |
| `gpu-trajectory.bin` | f32: posisi semua titik, keadaan 0..240 |
| `gpu-iterations.bin` | f64: satu baris 9 kolom per iterate Newton lintasan (format E2) |
| `gpu-probe-iterations.bin` | f64: iterate probe |
| `gpu-probe-iterates.bin` | f32: posisi setiap iterate probe |
| `gpu-result.json` | backend, adapter, tangen, ms/langkah, info host |

```ts
const report = { schema: "e3-gpu/1", backend: "vgpu-node-dawn", adapter: out.adapter, tangent: problem.manifest.tangent, record_fields: RECORD_FIELDS,
  steps: problem.manifest.steps, trajectory_ms: out.trajectory.ms, ms_per_step: out.trajectory.ms / problem.manifest.steps, ...hostInfo() };
```

`ms_per_step` mencakup **semua** biaya per langkah: perakitan, semua PCG, line search, dan round-trip host. Ini angka yang dibandingkan dengan anggaran
real-time 4,17 ms (= 1/240 s).

## 3. Ringkasan cetak

```
drape-n33 f32 projected: 7200 Newton iterations over 240 steps, 240 not at 1e-6, worst final 2.73e-2, 1338.9 ms/step
```

## 4. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Sama dengan E2 | penjaga → persiapan → kerja → output → ringkasan |
| Urutan | kernel → probe → lintasan |
| ms/langkah | biaya end-to-end per langkah waktu |
