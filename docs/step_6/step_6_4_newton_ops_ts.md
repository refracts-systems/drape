# Step 6.4: `newton-ops.ts`, Tiga Pengukuran Newton dan Line Search Sadar Noise

> Dalam loop Newton ada tiga pengukuran yang selama ini membaca buffer besar ke host: **norma residual**, **kemiringan gᵀd**, dan **perubahan energi**
> setiap percobaan α. File ini memisahkannya dari loop dalam satu antarmuka `NewtonOps`, dengan dua implementasi (host dan GPU), dan memuat line search baru.

---

## 1. Antarmuka

```ts
export interface NewtonOps {
  readonly residual: (positions) => Promise<Residual>;
  readonly slope: (solution) => Promise<Slope>;
  readonly energy: (stepLength) => Promise<EnergySums>;
}
```

| Pengukuran | Dipakai untuk | Hasil |
|---|---|---|
| `residual(x)` | merakit sistem di x dan mengukur ‖g‖ | norma f32 |
| `slope(d)` | kemiringan gᵀd dari solusi PCG | negatif = arah turun |
| `energy(α)` | perubahan Φ untuk langkah α | Σδ dan Σ\|δ\| |

Pola ini disebut **strategi**: loop Newton hanya memanggil `ops.residual(...)` dan tidak peduli apakah hasilnya dari bacaan penuh atau reduksi GPU.
`robust-newton.ts` tidak lagi memanggil `assemble` atau `dot32` langsung.

## 2. `hostOps`: perilaku E3–E5

```ts
residual: async (positions) => {
  rhs = await inPhase(clock, "assembly", () => assemble(gpu, positions));
  return { norm: norm32(rhs), ... };
},
slope: async (solution) => ({ value: -dot32(rhs, solution), ... }),
energy: async (stepLength) => hostEnergy(gpu, stepLength),
```

- Menyimpan `rhs` di closure (variabel `let`) supaya `slope` memakai gradien terakhir yang dirakit. Penyimpanan itu satu-satunya keadaan yang berubah, dan sengaja
  hanya di sini.
- `hostEnergy` membaca seluruh `elem_delta` dan menjumlahkan berurutan di f32. Jumlah Σδ **sama persis** dengan `energyChange` lama, jadi E5 tetap identik
  bit-per-bit. Σ|δ| dijumlahkan sekalian.

## 3. `gpuOps`: rung C dan D

```ts
residual: async (positions) => {
  backend.write(gpu.positions, positions);
  await inPhase(clock, "assembly", () => backend.run([...gpu.evaluate, ...reducers.rhsStats]));
  const normSquared = (await readReduced(backend, reducers.reduced))[SLOT.rhsNormSquared];
  ...
  return { norm: f32(Math.sqrt(normSquared)), ... };
},
```

- **Satu submit**: kernel perakitan dan reduksi dikirim bersama dalam satu `run`. Pertama kali saya kirim reduksi sebagai submit sendiri, dan rung C
  jadi lebih **lambat** dari A (134 vs 117 ms di n33): di jalur vgpu, setiap submit memiliki latensi tetap yang lebih mahal daripada membaca 50 KB.
  Menggabungkan submit mengembalikan jumlah submit seperti rung A.
- Host membaca 32 byte (`reduced`) alih-alih seluruh `rhs`.
- Kemiringan: setelah `solveOnce`, buffer `x` berisi solusi dan `rhs` masih berisi −g, jadi `rhsStats` menghasilkan Σ rhs·x, dan kemiringan = −Σ rhs·x.
- `energy(α)`: `energyChange` dan `energySums` dalam satu submit, lalu baca 32 byte alih-alih seluruh `elem_delta` (~100 KB di n65).

### Mode `verify` (hanya probe)

Dengan `verify` (probe), `gpuOps` juga membaca `rhs` dan solusi ke host dan menjumlahkan di f64 (`sumOf`). Hasilnya dicatat di samping nilai GPU, supaya oracle
memeriksa akurasi reduksi pada nilai-nilai yang sama persis. Pembacaan tambahan ini hanya terjadi di probe, sebelum lintasan yang diukur waktunya.

## 4. Line search sadar noise: `searchStep`

Prompt mendefinisikan penerimaan langkah baru untuk rung B:

> terima α jika Σδ ≤ c·α·gᵀd + 4·ε32·Σ|δ|, dan tetap wajib gᵀd < 0 dan α ≤ α_max.

```ts
const plain = sum <= armijoThreshold(armijo, alpha, slope);
const accepted = plain || (options.noiseAware && sum <= armijoThreshold(armijo, alpha, slope) + NOISE_FACTOR * EPSILON32 * abs);
options.onTrial?.({ alpha, sum, abs, slope, accepted });
if (accepted) return { stepLength: alpha, halvings, noiseOnly: !plain, sum, abs };
```

- Dengan `noiseAware` mati, ini **persis** line search E3–E5 (rung A): `armijoThreshold` = `Math.fround(armijo·α·slope)`.
- `noiseOnly` = diterima hanya berkat suku noise (tidak lolos Armijo biasa).
- `onTrial` (hanya probe) mencatat setiap α yang dicoba beserta Σδ, Σ|δ|, kemiringan, dan keputusan. Oracle memutar ulang setiap keputusan di f64 (dokumen 6.7).

### Hasil terukur: suku noise tidak menolong

Di n33 **tidak ada satu pun** update yang diterima hanya berkat suku noise, sehingga rung B identik dengan A (21 line search gagal tetap gagal). Selisih
terkecil antara Σδ dan ambang Armijo, untuk percobaan yang ditolak di probe, adalah **612×** batas noise 4·ε32·Σ|δ|. Jadi kegagalan di lantai f32 **bukan**
noise penjumlahan: ΔΦ(α) yang dihitung naik secara sistematis (dokumen 5.11, ukuran di n17), dengan besar jauh di atas ε32·Σ|δ|. Penyebabnya
kuantisasi **posisi** (gerak ~30 ulp), bukan galat jumlah. Aturan noise dari prompt mengukur hal yang salah, dan itu temuan eksperimen ini.
