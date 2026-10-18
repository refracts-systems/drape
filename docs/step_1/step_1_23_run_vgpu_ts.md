# Step 1.23: `gpu/src/node/run-vgpu.ts`, Runner vgpu

> Runner kedua, jauh lebih sederhana daripada versi browser karena semuanya terjadi di satu proses Node.

---

## 1. Isi lengkap

```ts
// Experiment 1 through vgpu/node: identical WGSL, problem files and dispatch order as run-browser.ts.
import { runExperiment } from "../experiment.ts";
import { parseCommon } from "./cli.ts";
import { loadKernels, loadProblem, printSummary, writeOutputs } from "./files.ts";
import { assertTargetAdapter } from "./hardware.ts";
import { createVgpuBackend } from "./vgpu-backend.ts";

const ADAPTERS = ["auto", "hardware", "software"] as const;

async function main(): Promise<void> {
  const options = parseCommon(process.argv.slice(2), ADAPTERS, "hardware");
  const [problem, sources] = await Promise.all([loadProblem(options.caseDir), loadKernels()]);
  const backend = await createVgpuBackend(options.adapter);
  try {
    assertTargetAdapter(options.adapter, backend.adapter);
    const result = await runExperiment(backend, problem, sources, options.settings);
    await writeOutputs(options.caseDir, result.report, result.solution);
    printSummary(result.report);
  } finally {
    backend.dispose();
  }
}

await main();
```

## 2. Hal yang patut diperhatikan

- **Top-level `await`**: file adalah modul ES (`"type": "module"` di `package.json`), jadi `await main()` bisa langsung di tingkat atas.
- Node ≥ 22.18 menjalankan `.ts` langsung (type stripping), tanpa `tsc` atau `ts-node`: `node src/node/run-vgpu.ts ...`.
- Soal dan kernel dibaca **paralel** (`Promise.all`).
- Penjaga adapter dipanggil **sebelum** solve (beda dengan browser, yang baru tahu adapter setelah halaman berjalan).
- `try/finally` memastikan device GPU dilepas walau solve gagal. Ini seperti menutup koneksi DB di `finally`.

## 3. Satu `solver.ts`, dua runner

| | `run-browser.ts` | `run-vgpu.ts` |
|---|---|---|
| Proses | Node + Chromium (multi-proses) | Node saja |
| Backend | `webgpu-backend.ts` (di halaman) | `vgpu-backend.ts` |
| Data ke GPU | lewat HTTP lokal | langsung dari file |
| Hasil kembali | serialisasi `page.evaluate` | langsung |
| Timestamp GPU | ada | tidak ada |
| `solver.ts`, `experiment.ts`, kernel | **sama** | **sama** |

## 4. Contoh output nyata

```
tension-n257 N=195075 converged it=1545 rec=9.76e-8 true32=7.82e-3 warm=1359.8ms gpu=n/ams total=1576.5ms det=true [D3D12 driver version 32.0.16.1714 (gpu)]
```

`gpu=n/a` karena vgpu tanpa timestamp. Driver 32.0.16.1714 = RTX 3050.

## 5. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Satu proses | lebih sederhana dan lebih cepat di ukuran kecil |
| Penjaga dulu | adapter dicek sebelum kerja dimulai |
| `finally` | device selalu dilepas |
