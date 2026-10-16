# Step 5.10: `robust-newton.ts`, `contact-host.ts`, `drape-e4.ts`, dan `drape-gpu.ts`, Newton Host dengan Kontak

> Newton tetap di host (TypeScript), dan solver PCG E1 tetap tidak tersentuh. Yang berubah: aturan berhenti E4-final, α_max dari kernel CCD sebagai awal
> line search, titik awal yang feasible, dan catatan tambahan untuk replay oracle.

---

## 1. `robust-newton.ts`

### Aturan berhenti dengan resolusi f32

```ts
/** Spacing of f32 numbers at max |x_i| over the unknowns: 2^(exponent − 23), from the f32 exponent bits. */
function resolutionOf(context: ..., positions: Float32Array): number {
  const largest = ...;
  const bits = new Uint32Array(Float32Array.of(largest).buffer)[0];
  return 2 ** (((bits >>> 23) & 0xff) - 150);
}

const updateIsSmall = (rules: RobustRules, extra: RobustRecord): boolean =>
  extra.updateInf <= Math.max(rules.update_stop * extra.stepInf, rules.resolution_factor * extra.resolution);
```

- `Float32Array.of(x).buffer` lalu `Uint32Array` adalah cara JavaScript membaca bit mentah f32. Kembaran `to_bits()` di Rust.
- 150 = 127 (bias eksponen) + 23 (bit mantisa).
- Keputusan memakai **record yang sama** yang ditulis ke file (`extra`), sehingga yang diputar ulang oracle persis input keputusan GPU.

### Awal line search

```ts
/** Writes p_k and returns the line-search start: α_max of the colliders (timed as "ccd"), or 1 without colliders. */
async function startStep<A, K>(context, positions, direction) {
  const scattered = scatter(context.model, direction);
  context.gpu.backend.write(context.gpu.direction, scattered);
  context.directions?.push(Float32Array.from(positions), scattered);
  const contact = context.contact;
  return contact === null ? NO_LIMIT : inPhase(context.clock, "ccd", () => contact.maxStep());
}
```

- `context.contact` bernilai `null` tanpa collider (E3/E4), sehingga α₀ = 1 dan tidak ada dispatch tambahan.
- `context.directions?.push(...)`: hanya probe yang menyimpan x_k dan p_k (operator `?.` melewati pemanggilan kalau `null`). Analogi: debug log yang hanya
  aktif di request tertentu.
- `lineSearch(context, slope, limit.alpha)` di `newton.ts` mendapat parameter `initial = 1`: α = initial·0,5^h. Dengan initial 1, hasilnya bit-identik
  untuk E2/E3/E4.

### Catatan robust baru

| Kolom | Isi |
|---|---|
| `resolution` | res(x) setelah update |
| `alphaMax` | α₀ line search |
| `limitingVertex` | vertex yang membatasi α₀ (−1 kalau tidak ada) |
| `minDistance` | jarak minimum ke collider, dihitung host dalam f64 dari posisi f32 |

### Antarmuka kontak

```ts
export interface ContactHooks {
  readonly maxStep: () => Promise<{ readonly alpha: number; readonly vertex: number }>;
  readonly minDistance: (positions: Float32Array) => number;
}
```

`robust-newton.ts` tidak tahu apa itu bola atau lantai. Ia hanya menerima dua fungsi. Ini **dependency inversion**: Newton generik, kontak dipasang dari
luar.

## 2. `contact-host.ts`

| Fungsi | Tugas |
|---|---|
| `minDistance` | jarak minimum f64 atas vertex dan collider (`Math.hypot` untuk bola) |
| `maxStep` | jalankan kernel CCD, baca batas, ambil minimum |
| `contactHooks` | `null` tanpa collider, selain itu dua fungsi di atas |
| `withoutColliders` | jalankan pekerjaan dengan jumlah collider 0, lalu pulihkan (`try/finally`) |
| `evaluateContactKernel` | cek kernel kontak: x̂ = x, μ = 0, massa baris 0 |

### Mengisolasi suku kontak untuk cek kernel

```ts
[gpu.positions, gpu.xHat].forEach((target) => gpu.backend.write(target, state));
gpu.backend.write(gpu.regularization, Float32Array.of(0));
gpu.backend.write(gpu.rowMass, new Float32Array(rows));
await gpu.backend.run(gpu.inertia);
gpu.backend.write(gpu.rowMass, problem.f32.row_mass);
```

Kernel inersia menulis inersia **ditambah** kontak. Supaya yang dibandingkan dengan oracle hanya kontak:

- x̂ = x membuat gradien inersia (M/h²)(x − x̂) = 0;
- massa baris 0 membuat blok inersia 0 (kalau hanya dikurangkan, pembulatan penjumlahan f32 dengan blok inersia yang besar ikut masuk);
- setelah selesai, massa baris dipulihkan.

### Collider dimatikan saat cek inersia E4

Cek inersia E4 memakai keadaan uji `kernel_state` (lipatan di sekitar z ≈ 0) yang sebagian di bawah lantai. Dengan collider aktif, ln(d) dengan d ≤ 0
menghasilkan NaN. `withoutColliders` menjalankan cek itu dengan jumlah collider 0, persis seperti di E4.

## 3. `drape-e4.ts`: titik awal yang feasible

```ts
/** x̂ without colliders; with colliders x_n + α_max·(x̂ − x_n), α_max from the CCD kernel at x_n (already on the GPU). */
async function newtonStart<A, K>(ctx, gpu, origin, xHat) {
  const contact = ctx.contact;
  if (contact === null) return Float32Array.from(xHat);
  const towards = Float32Array.from(xHat, (value, i) => value - origin[i]);
  gpu.backend.write(gpu.direction, towards);
  const { alpha } = await inPhase(ctx.clock, "ccd", () => contact.maxStep());
  return Float32Array.from(origin, (value, i) => value + alpha * towards[i]);
}
```

Kembaran `feasible_start` oracle (dokumen 5.5). Buffer posisi GPU sudah berisi x_n, karena perakitan untuk skala ‖∇Φ(x_n)‖ baru saja menulisnya.

Fase waktu baru `ccd` ditambahkan ke `PhaseClock` dan ke rata-rata per langkah.

## 4. `drape-gpu.ts` dan `drape.ts`

- Buffer `colliders` (read) di-bind ke kernel inersia dan perubahan energi inersia. Isinya `[0, 0, 0, 0]` untuk problem tanpa array `colliders`.
- Buffer `ccd_bounds` (satu float per baris) dan kernel `ccd` (dengan `KEEP` dari `problem.json`). Kalau ada collider tetapi `keep_fraction` tidak ada,
  host melempar error alih-alih memakai default diam-diam.
- `DrapeProblem.contact = { colliders, state, count }` diparse dari array opsional.
- `DrapeGpu.rowMass` diekspos untuk cek kernel kontak.
