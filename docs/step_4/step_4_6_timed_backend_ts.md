# Step 4.6: `gpu/src/timed-backend.ts`, Mengukur ke Mana Waktu Pergi

> vgpu tidak punya timestamp GPU, padahal prompt meminta rincian waktu per langkah (perakitan, PCG, line search, I/O host, sisanya). File ini adalah
> **decorator** di atas `Backend`: semua panggilan diteruskan apa adanya, sambil dicatat waktunya.

---

## 1. Jam fase

```ts
export type Phase = "assembly" | "pcg" | "line_search" | "other";

/** Wall-clock time of GPU submits per Newton phase, and of host readbacks/writes (vgpu has no GPU timestamps). */
export interface PhaseClock {
  current: Phase;
  readonly runMs: Record<Phase, number>;
  ioMs: number;
}
```

`current` adalah fase yang sedang berjalan. `runMs` mengakumulasi waktu `run()` (submit + tunggu GPU) per fase. `ioMs` mengakumulasi `read()` dan `write()`.

## 2. Decorator

```ts
/** The same backend, with every run() charged to the clock's current phase and every read()/write() to host I/O. */
export function timeBackend<A, K>(inner: Backend<A, K>, clock: PhaseClock): Backend<A, K> {
  return {
    ...
    write: (target, data) => {
      const started = performance.now();
      inner.write(target, data);
      clock.ioMs += performance.now() - started;
    },
    run: async (dispatches) => {
      const started = performance.now();
      const gpuMs = await inner.run(dispatches);
      clock.runMs[clock.current] += performance.now() - started;
      return gpuMs;
    },
    read: async (source) => { ... clock.ioMs += ... },
    ...
  };
}
```

Objek baru yang memenuhi interface `Backend` yang sama. Solver E1, kernel E2/E3, dan Newton E4 tidak tahu sedang diukur.

Analogi backend: **middleware tracing** (seperti OpenTelemetry) yang membungkus client database. Kode bisnis tidak berubah, tetapi setiap query tercatat
durasinya.

## 3. Menandai fase

```ts
/** Runs `work` with the clock's phase set, restoring "other" afterwards. */
export async function inPhase<T>(clock: PhaseClock, phase: Phase, work: () => Promise<T>): Promise<T> {
  clock.current = phase;
  try {
    return await work();
  } finally {
    clock.current = "other";
  }
}
```

Dipakai di Newton E4: `inPhase(clock, "pcg", () => solveOnce(...))`. `try/finally` menjamin fase kembali ke `"other"` walau terjadi error.

## 4. Cara membaca hasilnya

Per langkah (rata-rata):

| Kategori | Isi |
|---|---|
| `assembly` | submit kernel elemen + hinge + inersia + rakit (termasuk menunggu GPU) |
| `pcg` | semua submit PCG (setup, chunk iterasi, finish) |
| `line_search` | submit kernel perubahan energi |
| `io` | semua `read()` (salin GPU → host, map) dan `write()` (upload) |
| `other` | waktu total − semua di atas: kode JavaScript host (norma f32, prediksi, scatter, loop) |

Kategori `run` sudah termasuk **menunggu GPU selesai**, karena `run` menunggu `onSubmittedWorkDone`. Jadi angka ini adalah waktu dinding, bukan waktu
komputasi GPU murni. Angka yang dibutuhkan adalah biaya end-to-end per fase.

## 5. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Decorator | Backend yang sama, plus pencatatan waktu |
| Fase | ditandai oleh `inPhase` di sekitar tiap bagian Newton |
| io terpisah | baca/tulis host dihitung sendiri |
| other | sisa waktu: kode JavaScript host |
