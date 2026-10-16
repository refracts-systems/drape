# Step 3.15: `gpu/src/drape-gpu.ts`, Buffer Bersama dan Dua Belas Pipeline

> Kembaran `coupon-gpu.ts` untuk Step 3. Ia membuat satu set buffer lokal yang ditulis bersama oleh kernel membran, hinge, dan inersia, lalu menyambungkan
> kernel perakitan E2 (tanpa perubahan) ke solver E1 (tanpa perubahan).

---

## 1. Tata letak buffer bersama

```ts
/** Offsets (in floats) of the hinge and inertia sections behind the membrane entries of the shared local arrays. */
function layout(problem: DrapeProblem): Layout {
  const { triangles: t, hinges: h } = problem.manifest;
  return { hingeGrad: 9 * t, inertiaGrad: 9 * t + 12 * h, hingeHess: 81 * t, inertiaHess: 81 * t + 144 * h, hingeDelta: t, inertiaDelta: t + h };
}
```

```
elem_grad  : [ membran 9T | hinge 12H  | inersia 3R ]
elem_hess  : [ membran 81T | hinge 144H | inersia 9R ]
elem_delta : [ membran T  | hinge H    | inersia R  ]
```

Offset-offset ini dikirim ke kernel sebagai **override** (`GRAD_OFFSET`, `HESS_OFFSET`, `DELTA_OFFSET`). Mereka cocok dengan indeks peta gather dari oracle
(dokumen 3.6). Untuk n129, `elem_hess` berukuran ~10 juta float (40 MB), masih jauh di bawah batas binding 128 MB.

## 2. Buffer

```ts
function drapeArrays<A, K>(backend: Backend<A, K>, problem: DrapeProblem): DrapeArrays<A> {
  const scratch = (label: string, length: number): A => backend.array(label, new Float32Array(length), "read-write");
  const fixed = <T extends Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>>(label: string, data: T): A => backend.array(label, data, "read");
  return {
    positions: ..., xHat: scratch("x_hat", 3 * m.vertices), direction: ..., step: scratch("step", 1),
    ... data tetap (geometri, hinge, massa, peta) sebagai "read" ...
    elemGrad: scratch("elem_grad", 9 * m.triangles + 12 * m.hinges + 3 * m.rows),
    elemHess: scratch("elem_hess", 81 * m.triangles + 144 * m.hinges + 9 * m.rows),
    elemDelta: scratch("elem_delta", m.triangles + m.hinges + m.rows),
  };
}
```

Buffer baru dibanding E2: `x_hat` (prediksi backward Euler, ditulis host setiap langkah), `hinge_vertices`, `hinge_stiffness`, dan `row_mass`.

## 3. Spesifikasi kernel

Fungsi kecil per jenis kernel:

| Fungsi | Kernel | Override |
|---|---|---|
| `membraneSpec(..., projected)` | `membrane_element` | modulus + `PROJECTED` |
| `hingeSpec(..., exact)` | `hinge_element` | `GRAD_OFFSET`, `HESS_OFFSET`, `EXACT` |
| `inertiaSpec` | `inertia` | `GRAD_OFFSET`, `HESS_OFFSET` |
| `assemblySpecs` | `assemble_matrix`, `assemble_force` | — (E2 apa adanya) |
| `energySpecs` | tiga kernel perubahan energi | `DELTA_OFFSET` |

`Number(projected)` mengubah boolean menjadi 1/0 untuk override `bool`.

## 4. Dua belas pipeline

```ts
const projected = problem.manifest.tangent === "projected";
const specs = [membraneSpec(..., projected), hingeSpec(..., !projected), inertiaSpec(...),
  ...assemblySpecs(...), ...energySpecs(...),
  membraneSpec(..., false), hingeSpec(..., true), membraneSpec(..., true), hingeSpec(..., false)];
const [membrane, hinge, inertia, matrix, force, membraneDelta, hingeDelta, inertiaDelta, membraneExact, hingeExact, membraneProjected, hingeGaussNewton] =
  await Promise.all(specs.map((spec) => backend.kernel(spec)));
```

- 8 pipeline untuk simulasi (varian dipilih dari `tangent` di manifest).
- 4 pipeline untuk cek kernel: membran eksak, hinge eksak, membran proyeksi, hinge Gauss–Newton. Kedua varian diperiksa terhadap f64, sehingga gate kernel
  mencakup semua kode yang ditulis, bukan hanya yang dipakai simulasi.

Semua dikompilasi **paralel** dengan `Promise.all`. Destrukturisasi array memberi nama yang jelas untuk setiap pipeline.

## 5. Daftar dispatch

```ts
evaluate: [membran, hinge, inersia, rakit matriks, rakit gaya],
energyChange: [membranDelta, hingeDelta, inersiaDelta],
exactElements: [membranEksak, hingeEksak],
projectedElements: [membranProyeksi, hingeGN],
```

Jumlah workgroup per jenis: per elemen, per hinge, per baris (`workgroups()`).

`evaluate` dan `energyChange` adalah satu-satunya yang dibutuhkan inti Newton (`NewtonGpu`). Dengan menyusun kernel tambahan ke dalam kedua daftar ini,
`solveStep` dari E2 menyelesaikan potensial inkremental Φ tanpa tahu apa-apa soal hinge atau inersia.

## 6. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Buffer bersama | tiga jenis kontribusi, satu set buffer, offset per bagian |
| Kernel E2 utuh | perakitan dan membran (default) tidak diubah |
| 12 pipeline | 8 simulasi + 4 cek kernel (eksak dan proyeksi) |
| Reuse Newton | kernel baru cukup masuk ke `evaluate` dan `energyChange` |
