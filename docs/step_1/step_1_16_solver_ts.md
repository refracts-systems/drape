# Step 1.16: `gpu/src/solver.ts`, Orkestrasi PCG di Host

> Kalau kernel WGSL adalah "worker", `solver.ts` adalah **job scheduler**-nya: membuat buffer, mengompilasi kernel,
> menyusun urutan dispatch, dan memutuskan kapan berhenti. File ini tidak menghitung satu angka pun.

---

## 1. Buffer yang dibuat

```ts
interface Arrays<A> {
  readonly rowPtr: A; readonly colIdx: A; readonly vals: A; readonly diagIdx: A; readonly rhs: A;
  readonly dinv: A; readonly x: A; readonly r: A; readonly z: A; readonly p: A; readonly q: A;
  readonly partials: A; readonly state: A;
}

function createArrays<A, K>(backend: Backend<A, K>, problem: Problem): Arrays<A> {
  const rows = problem.manifest.rows;
  const vector = (label: string): A => backend.array(label, new Float32Array(3 * rows), "read-write");
  return {
    rowPtr: backend.array("row_ptr", problem.rowPtr, "read"),
    colIdx: backend.array("col_idx", problem.colIdx, "read"),
    vals: backend.array("vals", problem.vals, "read-write"),
    diagIdx: backend.array("diag_idx", problem.diagIdx, "read"),
    rhs: backend.array("rhs", problem.rhs, "read-write"),
    dinv: backend.array("dinv", new Float32Array(9 * rows), "read-write"),
    x: vector("x"), r: vector("r"), z: vector("z"), p: vector("p"), q: vector("q"),
    partials: backend.array("partials", new Float32Array(2 * workgroupsFor(rows)), "read-write"),
    state: backend.array("state", new Uint32Array(16), "read-write"),
  };
}
```

| Buffer | Ukuran (n257) | Isi |
|---|---|---|
| `rowPtr`, `colIdx`, `diagIdx` | 0,3 / 2,8 / 0,3 MB | pola BSR, read-only |
| `vals`, `rhs` | 25,6 / 0,8 MB | matriks dan beban; **read-write sejak Step 2** |
| `dinv` | 2,3 MB | invers blok diagonal |
| `x, r, z, p, q` | 0,8 MB masing-masing | vektor CG |
| `partials` | 2 × 255 f32 | hasil reduksi per workgroup |
| `state` | 64 byte | skalar CG |

`vals` dan `rhs` dibuat `"read-write"` di Step 2 agar kernel perakitan (`assemble_matrix`, `assemble_force`) bisa menulis matriks
Newton langsung ke buffer solver. Kernel E1 tetap mendeklarasikannya `read`; probe di sesi kerja membuktikan vgpu menerima kombinasi
itu (dokumen 1.6 bagian 8).

## 2. Stage dan spesifikasi kernel

```ts
type Stage = "invert" | "init" | "start" | "spmv" | "alpha" | "update" | "beta" | "direction" | "residual" | "measure" | "replace" | "replaced";
```

12 stage dari 7 file WGSL: `reduce.wgsl` menjadi 5 stage (MODE 0–4), `residual.wgsl` menjadi 2 (REPLACE false/true).

```ts
async function createStages<A, K>(backend: Backend<A, K>, specs: Record<Stage, KernelSpec<A>>): Promise<Record<Stage, K>> {
  const stages = Object.keys(specs) as Stage[];
  const kernels = await Promise.all(stages.map((stage) => backend.kernel(specs[stage])));
  return Object.fromEntries(stages.map((stage, index) => [stage, kernels[index]])) as Record<Stage, K>;
}
```

Semua pipeline dikompilasi **paralel** (`Promise.all`). Kompilasi shader bisa makan puluhan milidetik masing-masing; ini termasuk
dalam `prepare_ms` di laporan.

## 3. Urutan dispatch

```ts
return {
  backend,
  arrays,
  setup: [at(k.invert, rows), at(k.init, rows), at(k.start, 1)],
  iteration: [at(k.spmv, rows), at(k.alpha, 1), at(k.update, rows), at(k.beta, 1), at(k.direction, rows)],
  finish: [at(k.residual, rows), at(k.measure, 1)],
  replacement: [at(k.replace, rows), at(k.replaced, 1)],
  replaceEvery: problem.manifest.replace_every,
  replaceAbove: problem.manifest.replace_above,
};
```

Diagram satu solve:

```
setup:      jacobi_invert ─► cg_init ─► reduce(0)
iteration:  spmv_dot ─► reduce(1) ─► update_dot ─► reduce(2) ─► direction        (× 50, atau × 32 tanpa replacement)
replacement:                                    residual(replace) ─► reduce(4)    (setiap akhir chunk 50)
finish:     residual(measure) ─► reduce(3)
```

`rows` di sini adalah jumlah workgroup (255), bukan jumlah baris.

## 4. Satu solve: `solveOnce`

```ts
export async function solveOnce<A, K>(solver: PreparedSolver<A, K>, settings: SolveSettings): Promise<SolveRun> {
  const { backend, arrays } = solver;
  const started = performance.now();
  const timings: (number | null)[] = [];
  backend.write(arrays.state, initialState(settings, solver.replaceAbove));
  timings.push(await backend.run(solver.setup));
  const chunks = await iterate(solver, settings, timings);
  timings.push(await backend.run(solver.finish));
  const state = decodeState(await backend.read(arrays.state));
  const solved = performance.now();
  const solution = new Float32Array(await backend.read(arrays.x));
  return summarize(state, solution, timings, { started, solved, chunks });
}
```

1. Tulis `State` awal (toleransi, batas iterasi, gate replacement).
2. Setup.
3. Iterasi per chunk sampai GPU menyatakan berhenti.
4. Finish: ukur residual sejati.
5. Baca `State` dan `x`.

`solveMs` diukur sampai sebelum `x` dibaca, karena membaca `x` (0,8 MB) adalah biaya "keluar dari GPU", bukan bagian solve.

## 5. Chunk: kompromi latensi vs pemborosan

```ts
async function iterate<A, K>(solver: PreparedSolver<A, K>, settings: SolveSettings, timings: (number | null)[]): Promise<number> {
  const { chunk, iterations } = chunkDispatches(solver, settings);
  const chunkLimit = Math.ceil(settings.maxIterations / iterations) + 1;
  for (let chunks = 1; chunks <= chunkLimit; chunks += 1) {
    timings.push(await solver.backend.run(chunk));
    const state = decodeState(await solver.backend.read(solver.arrays.state));
    if (state.status !== 0) {
      return chunks;
    }
  }
  throw new Error("GPU never raised a stop status: the iteration counter or reduction is broken");
}
```

- Host mengirim **satu chunk** (32 atau 50 iterasi = 160 atau 250 dispatch), lalu membaca 64 byte `State` untuk melihat status.
- Kalau GPU konvergen di tengah chunk, sisa dispatch menjadi no-op karena setiap kernel mengecek `status` (dokumen 1.9 dan 1.12).

Trade-off:

| Chunk kecil | Chunk besar |
|---|---|
| banyak round-trip host↔GPU (lambat) | sedikit round-trip |
| sedikit dispatch sia-sia setelah konvergen | sampai (chunk−1) iterasi no-op |

Iterasi no-op tetap murah tetapi tidak gratis: setiap workgroup tetap diluncurkan lalu langsung keluar.

### Pengaman

`chunkLimit` = batas iterasi / chunk + 1. Kalau GPU tidak pernah menyatakan berhenti (misalnya counter rusak), host melempar error
alih-alih loop selamanya. Pola **circuit breaker** sederhana.

## 6. Chunk dengan replacement

```ts
/** With replacement on, a chunk is exactly one replacement period, so replacements land after iterations K, 2K, … */
function chunkDispatches<A, K>(solver, settings) {
  const iterations = solver.replaceEvery > 0 ? solver.replaceEvery : settings.chunkIterations;
  const tail = solver.replaceEvery > 0 ? solver.replacement : [];
  return { chunk: [...Array.from({ length: iterations }, () => solver.iteration).flat(), ...tail], iterations };
}
```

Desain yang dipilih: **chunk = periode replacement**. Alternatif "kernel replacement dijalankan setiap iterasi dan mengecek
`iteration % 50` sendiri" akan menambah dispatch di setiap iterasi. Dengan menyamakan chunk dan periode, replacement cukup ditempel
di akhir chunk, dan host tidak perlu tahu nomor iterasi.

## 7. Laporan per solve

```ts
return {
  status: STATUS[state.status] ?? "breakdown",
  iterations: state.iteration,
  recursiveRelativeResidual: Math.sqrt(state.rr / state.rr0),
  gpuTrueRelativeResidual: Math.sqrt(state.trueRr / state.rr0),
  solution,
  gpuMs: sumTimings(timings),
  ...
};
```

- `recursiveRelativeResidual`: yang dipercaya CG, selalu tampak "konvergen".
- `gpuTrueRelativeResidual`: GPU menghitung ulang `b − Ax` dalam f32. Ini peringatan dini drift, dan dipakai sebagai "η" (forcing term
  terukur) di Step 2.
- `gpuMs`: jumlah timestamp GPU semua submit, atau `null` kalau backend tidak punya timer.

## 8. Dipakai ulang di Step 2

Di Step 2, `prepareSolver` dipanggil dengan `Problem` sintetis (pola BSR kupon, nilai nol). Lalu kernel perakitan mengisi `vals` dan
`rhs` sebelum **setiap** `solveOnce`. Karena `setup` selalu menjalankan `jacobi_invert` ulang, preconditioner otomatis diperbarui untuk
matriks baru. Tidak ada satu baris pun `solver.ts` yang perlu diubah untuk Step 2 selain akses `read-write` pada `vals`/`rhs`.

## 9. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Stage | 7 file WGSL → 12 pipeline (lewat `override`) |
| Satu solve | setup → chunk × N → finish → baca hasil |
| Chunk | 32/50 iterasi per submit; cek status 64 byte per chunk |
| Pengaman | batas jumlah chunk mencegah loop tak berujung |
| Chunk = periode replacement | replacement jatuh di iterasi 50k tanpa modulo di GPU |
| Reuse Step 2 | vals/rhs diisi kernel lain, setup menghitung ulang preconditioner |
