# Step 1.20: `gpu/src/experiment.ts`, Cold Run, Warm Run, Determinisme, dan Laporan Waktu

> Mengukur performa GPU itu mudah salah. File ini menerapkan disiplin benchmarking: pisahkan biaya sekali jalan dari
> biaya per solve, dan buktikan hasilnya bisa diulang bit-per-bit.

---

## 1. Alur

```ts
export async function runExperiment<A, K>(backend: Backend<A, K>, problem: Problem, sources: KernelSources, settings: ExperimentSettings): Promise<ExperimentResult> {
  const started = performance.now();
  const solver = await prepareSolver(backend, problem, sources);
  const prepareMs = performance.now() - started;
  const runs: SolveRun[] = [];
  for (let repeat = 0; repeat < Math.max(1, settings.repeats); repeat += 1) {
    runs.push(await solveOnce(solver, settings));
  }
  return { report: report(backend, problem, settings, runs, prepareMs), solution: runs[0].solution };
}
```

1. **Prepare** sekali: upload matriks (~30 MB di n257) dan kompilasi 12 pipeline.
2. **Solve** diulang `repeats` kali (default 3) dengan buffer dan pipeline yang sama.

## 2. Cold vs warm

| Ukuran | Isi | Untuk apa |
|---|---|---|
| `prepare_ms` | upload + kompilasi | biaya sekali per matriks baru |
| `first_solve_ms` | solve pertama | termasuk efek "pemanasan" (alokasi malas driver, cache dingin) |
| `warm_solve_ms` | solve **tercepat** setelah yang pertama | biaya per solve dalam simulasi berjalan |
| `total_ms` | prepare + first solve + readback | pengalaman "satu kali pakai" |

```ts
/** Fastest repeat after the first; the first run carries lazy pipeline and allocation costs. */
const warmest = (runs: readonly SolveRun[]): SolveRun =>
  runs.slice(1).reduce((best, run) => (run.solveMs < best.solveMs ? run : best), runs[runs.length > 1 ? 1 : 0]);
```

Gate speedup memakai **warm** solve, karena di simulasi sungguhan solver dipanggil ribuan kali dengan pipeline yang sudah siap.
Memakai solve tercepat dari beberapa run adalah praktik standar untuk mengurangi noise OS; ia tidak memihak GPU karena patokan CPU juga
diukur sebagai satu run penuh.

Analogi: latency endpoint diukur setelah JIT/cache hangat, terpisah dari cold start.

## 3. Determinisme

```ts
const bitwiseEqual = (a: Float32Array<ArrayBuffer>, b: Float32Array<ArrayBuffer>): boolean => {
  const left = new Uint32Array(a.buffer);
  const right = new Uint32Array(b.buffer);
  return left.length === right.length && left.every((word, index) => word === right[index]);
};

const deterministic = (runs: readonly SolveRun[]): boolean =>
  runs.every((run) => run.iterations === runs[0].iterations && bitwiseEqual(run.solution, runs[0].solution));
```

Membandingkan float sebagai **bit** (lewat `Uint32Array`), bukan `===` pada angka. Alasannya:

- `0.0 === -0.0` bernilai true di JS, padahal bitnya beda.
- `NaN === NaN` bernilai false.
- Yang ingin dibuktikan adalah **bit identik**, bukan "kira-kira sama".

Kenapa determinisme penting? Simulasi yang tidak deterministik tidak bisa di-debug ("bug hanya muncul kadang-kadang") dan tidak bisa
direproduksi untuk validasi. Desain reduksi pohon tanpa atomik (dokumen 1.8) adalah yang membuat ini mungkin. Hasil: deterministic = true
di **semua** kasus, kedua backend.

## 4. Laporan `e1-gpu/1`

```ts
return {
  schema: "e1-gpu/1", case: manifest.case, grid_columns: manifest.grid_columns, backend: backend.name, adapter: backend.adapter,
  rows: manifest.rows, unknowns: manifest.unknowns, blocks: manifest.blocks,
  tolerance: settings.tolerance, max_iterations: settings.maxIterations, chunk_iterations: settings.chunkIterations,
  replace_every: manifest.replace_every,
  replace_above: manifest.replace_above,
  status: first.status, iterations: first.iterations,
  recursive_relative_residual: first.recursiveRelativeResidual,
  gpu_f32_true_relative_residual: first.gpuTrueRelativeResidual,
  repeats: runs.length, deterministic_repeat: deterministic(runs),
  timing: timing(prepareMs, runs, backend.gpuTiming),
};
```

Field `schema` adalah versi format. Kalau format berubah, pembaca (`compare.rs`) bisa mengenalinya. `replace_every` dan `replace_above`
ditambahkan di sesi ini supaya setiap hasil mencatat pengaturan yang dipakai.

## 5. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Prepare terpisah | upload + kompilasi tidak dicampur dengan waktu solve |
| Warm = tercepat setelah pertama | biaya realistis per solve |
| Determinisme bit | bandingkan `Uint32Array`, bukan angka float |
| `schema` | versi format laporan |
