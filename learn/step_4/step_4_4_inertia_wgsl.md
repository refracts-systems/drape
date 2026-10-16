# Step 4.4: `gpu/kernels/inertia.wgsl` (Perubahan Step 4), μ di Blok Diagonal

> Satu binding baru dan satu perkalian. Begitulah regularisasi K + μ·M/h² diwujudkan di GPU tanpa menyentuh kernel lain.

---

## 1. Perubahan

```wgsl
// Per unknown row: inertia gradient (M/h²)(x − x̂) and diagonal block (1 + μ)(M/h²)·I into the shared local arrays.
// regularization[0] = μ of the K + μ·M/h² shift (Experiment 4); always 0 in Experiment 3.
...
@group(0) @binding(6) var<storage, read> regularization: array<f32>;
...
    elem_hess[HESS_OFFSET + 9u * row + i] = select(0.0, mass * (1.0 + regularization[0]), i % 4u == 0u);
```

- **Gradien tidak berubah** (`mass * (x − x̂)`). Regularisasi hanya mengubah **matriks**, bukan soal yang diselesaikan. Jawaban akhir Newton tetap minimum Φ
  yang sama.
- **Blok diagonal**: `(1 + μ)·m/h²`.

## 2. Kenapa μ di buffer, bukan `override`

μ berubah di tengah langkah waktu (naik saat PCG breakdown, turun setelah langkah penuh). `override` hanya bisa diubah dengan membuat pipeline baru
(puluhan milidetik). Buffer satu float cukup ditulis host (4 byte) sebelum perakitan. Pola yang sama dengan α di line search (dokumen 2.9).

## 3. E3 tidak berubah

E3 tidak pernah menulis buffer ini, sehingga isinya tetap nol (buffer dibuat dari `new Float32Array(1)`). `mass * (1.0 + 0.0)` = `mass` **tepat**, karena
1 + 0 = 1 eksak dan x·1 = x eksak dalam IEEE-754. Regresi E3 membuktikannya: hasil identik bit-per-bit.

## 4. Diperiksa

`e4-compare` menjalankan kernel ini dengan x = keadaan uji, x̂ = posisi rest, dan μ = 0,37 (`KERNEL_MU`), lalu membandingkan gradien dan blok diagonal dengan f64:
galat 2,2×10⁻⁹.

## 5. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Hanya matriks | gradien tetap → jawaban Newton tetap |
| Buffer μ | ditulis host, tanpa kompilasi ulang |
| μ = 0 | identik dengan E3 secara bit |
