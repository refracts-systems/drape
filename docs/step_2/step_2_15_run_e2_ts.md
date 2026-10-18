# Step 2.15: `gpu/src/node/run-e2.ts`, Runner Step 2 dan Format Output Biner

> Titik masuk GPU Step 2. Ia merangkai semuanya, lalu menulis tiga file biner dan satu JSON untuk dinilai oracle.

---

## 1. Alur `main`

```ts
async function main(): Promise<void> {
  const { caseDir, adapter } = options();
  const [problem, sources] = await Promise.all([loadCoupon(caseDir), loadKernels()]);
  const backend = await createVgpuBackend(adapter);
  try {
    assertTargetAdapter(adapter, backend.adapter);
    const gpu = await prepareCouponGpu(backend, problem, sources);
    const kernel = await evaluateKernelState(gpu, problem);
    const path = await solvePath(gpu, problem);
    await writeOutputs(caseDir, backend.adapter, path, kernel);
    printSummary(problem, path);
  } finally {
    backend.dispose();
  }
}
```

1. Baca soal kupon dan semua kernel (paralel).
2. Buka vgpu dan **tolak** iGPU/software (penjaga yang sama dengan E1).
3. Siapkan kernel membran + solver E1.
4. Cek kernel elemen pada keadaan uji.
5. Jalankan seluruh jalur beban.
6. Tulis output, cetak ringkasan.

Step 2 hanya memakai jalur vgpu: lebih sederhana (tanpa browser), dan vgpu sudah lulus gate overhead di E1.

## 2. Opsi CLI

```ts
function options(): { readonly caseDir: string; readonly adapter: VgpuAdapter } {
  const { values } = parseArgs({ options: { case: { type: "string" }, adapter: { type: "string", default: "hardware" } } });
  const adapter = ADAPTERS.find((name) => name === values.adapter);
  if (values.case === undefined || adapter === undefined) throw new Error(`usage: --case <coupon dir> [--adapter ...]`);
  return { caseDir: values.case, adapter };
}
```

`ADAPTERS.find(name => name === values.adapter)` sekaligus memvalidasi dan **mempersempit tipe** dari `string` ke `VgpuAdapter`, tanpa cast.

## 3. Format output

| File | Isi | Pembaca |
|---|---|---|
| `gpu-kernel.bin` | f32: semua `elem_grad` (T×9), lalu semua `elem_hess` (T×81) | `e2-compare`: cek kernel |
| `gpu-iterations.bin` | f64: satu baris 9 angka per iterate | `e2-compare`: riwayat Newton |
| `gpu-iterates.bin` | f32: posisi semua titik (V×3) per iterate, urutan sama | `e2-compare`: residual f64 per iterate |
| `gpu-result.json` | metadata: backend, adapter, host, jumlah record, waktu | `e2-compare` |

### Tabel iterasi sebagai f64

```ts
const RECORD_FIELDS = ["step", "iteration", "relativeResidual", "pcgIterations", "pcgStatus", "linearResidual", "stepLength", "halvings", "status"] as const;

/** gpu-iterations.bin: one row of little-endian f64 per Newton iterate, columns RECORD_FIELDS. */
const recordTable = (records: readonly IterationRecord[]): Float64Array<ArrayBuffer> =>
  Float64Array.from(records.flatMap((entry) => RECORD_FIELDS.map((key) => entry[key])));
```

Kenapa biner, bukan JSON? Oracle membaca JSON dengan `jsonscan.rs` yang hanya bisa mencari kunci datar, bukan array objek bersarang. Tabel f64 dengan
kolom tetap jauh lebih mudah dibaca di Rust (`chunks_exact(9)`) dan tetap presisi penuh. NaN (untuk iterate terakhir yang tidak punya PCG/line search)
terwakili dengan sendirinya di f64.

`RECORD_FIELDS` juga ditulis ke `gpu-result.json` sebagai dokumentasi kolom yang ikut dengan datanya.

### Menggabungkan iterate

```ts
function joinIterates(iterates: readonly Float32Array<ArrayBuffer>[]): Float32Array<ArrayBuffer> {
  const joined = new Float32Array(iterates.reduce((length, x) => length + x.length, 0));
  iterates.reduce((offset, x) => (joined.set(x, offset), offset + x.length), 0);
  return joined;
}
```

Semua posisi iterate dirangkai jadi satu blob. Untuk uniaxial-n65: 20 langkah × 31 iterate × 4225 titik × 3 × 4 byte ≈ 30 MB. Karena itu
`coupon/*/*.bin` di-gitignore; yang di-commit hanya JSON.

## 4. Ringkasan cetak

```
uniaxial-n33 f32 Newton iterations per step [30 30 ...] worst final 4.43e-5 unconverged=20 11668ms
patch-n33 f32 Newton iterations per step [7] worst final 6.30e-7 unconverged=0 128ms
```

`unconverged=20` berarti semua 20 langkah uniaxial berhenti di batas iterasi (lantai f32), bukan konvergen ke 10⁻⁶.

## 5. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Penjaga dulu | adapter dicek sebelum kerja |
| Cek kernel sebelum jalur | per-elemen WGSL vs f64 |
| Tabel f64 | riwayat Newton mudah dibaca Rust, presisi penuh, NaN terwakili |
| Iterate disimpan | oracle menghitung residual f64 setiap iterate f32 |
| Bin di-gitignore | puluhan MB; JSON yang di-commit |
