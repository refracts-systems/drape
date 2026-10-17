# Step 6.3: `ccd.wgsl` (jarak per baris) dan `timed-backend.ts` (fase `reduce`, penghitung byte)

> Dua perubahan kecil yang mendukung pengukuran Step 6: kernel CCD sekarang juga menulis jarak terkecil tiap vertex ke collider, dan jam fase
> menghitung **byte yang dibaca dari GPU** serta fase baru `reduce`.

---

## 1. `ccd.wgsl`: keluaran kedua

```wgsl
@group(0) @binding(5) var<storage, read_write> distances: array<f32>;
...
  var alpha = 1.0;
  var nearest = 3.0e38;
  for (var k = 0u; k < collider_count(); k = k + 1u) {
    alpha = min(alpha, collider_step(k, x, p));
    nearest = min(nearest, proximity(k, x).distance);
  }
  bounds[row] = alpha;
  distances[row] = nearest;
```

Selain batas langkah α per baris (`bounds`), kernel menulis jarak terkecil vertex ke semua collider (`distances`). Rung D mereduksi array itu dengan
`argmin.wgsl` (dokumen 6.2) untuk mendapat jarak minimum seluruh kain tanpa membaca array-nya.

Rung A (E5) tetap menjalankan kernel yang sama dan membaca `bounds` saja. Tambahan tulisan `distances` tidak mengubah angka α mana pun: hasil E5 identik
bit-per-bit (dicek, dokumen 6.9).

Konstanta `3.0e38` adalah "hampir tak hingga" untuk f32 (maksimum f32 ≈ 3,4×10³⁸). WGSL tidak punya literal `infinity`.

## 2. `timed-backend.ts`

### Fase `reduce`

```ts
export type Phase = "assembly" | "pcg" | "line_search" | "ccd" | "reduce" | "other";
```

Waktu run kernel reduksi yang berdiri sendiri (misalnya reduksi kemiringan setelah PCG) dicatat di fase `reduce`.

### Penghitung byte

```ts
read: async (source) => {
  const started = performance.now();
  const bytes = await inner.read(source);
  clock.readBytes += bytes.byteLength;
  ...
```

Setiap pembacaan buffer menambah `readBytes` sebesar ukuran nyata hasilnya. Tidak ada perkiraan: "bytes per iterasi Newton" = total byte baca satu
lintasan ÷ jumlah iterasi Newton. Penghitung ikut membaca yang dilakukan solver E1 (status per chunk 64 byte, solusi PCG), karena dekorator
membungkus backend yang dipakai solver.

### `inPhase` memulihkan fase sebelumnya

Sebelumnya `inPhase` selalu mengembalikan fase ke `other` setelah selesai. Sekarang ia **memulihkan fase yang berlaku sebelumnya**:

```ts
export async function inPhase<T>(clock: PhaseClock, phase: Phase, work: () => Promise<T>): Promise<T> {
  const previous = clock.current;
  clock.current = phase;
  try {
    return await work();
  } finally {
    clock.current = previous;
  }
}
```

Perubahan ini diperlukan karena fase kini bersarang (reduksi di dalam line search). Panggilan lama selalu di tingkat atas (fase sebelumnya `other`), jadi
perilakunya sama. Analogi: `finally` yang mengembalikan context logging ke nilai sebelumnya, bukan ke nilai default.
