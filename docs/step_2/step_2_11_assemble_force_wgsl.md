# Step 2.11: `gpu/kernels/assemble_force.wgsl`, Merakit Vektor Gaya (Right-hand Side Newton)

> Pasangan `assemble_matrix` untuk vektor. Hasilnya `−g`, ditulis langsung ke buffer `rhs` solver E1.

---

## 1. Kode lengkap

```wgsl
// Newton right-hand side −∂Π/∂x on the unknowns, one row per thread: element gradients summed in element
// order; constrained components are zero.
@group(0) @binding(0) var<storage, read> dof_mask: array<u32>;
@group(0) @binding(1) var<storage, read> force_contrib_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> force_contrib_idx: array<u32>;
@group(0) @binding(3) var<storage, read> elem_grad: array<f32>;
@group(0) @binding(4) var<storage, read_write> rhs: array<f32>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  if (row >= arrayLength(&dof_mask)) {
    return;
  }
  for (var d = 0u; d < 3u; d = d + 1u) {
    var sum = 0.0;
    for (var j = force_contrib_ptr[row]; j < force_contrib_ptr[row + 1u]; j = j + 1u) {
      sum = sum + elem_grad[3u * force_contrib_idx[j] + d];
    }
    rhs[3u * row + d] = select(0.0, -sum, ((dof_mask[row] >> d) & 1u) == 1u);
  }
}
```

## 2. Penjelasan

- Satu thread per baris (titik bebas).
- Untuk setiap komponen d (x, y, z): jumlahkan gaya dari setiap elemen yang memuat titik ini. Indeks `3e + k` dari peta menunjuk gaya titik lokal k elemen e
  di `elem_grad[9e + 3k + d]` (= `3 × (3e + k) + d`).
- Tulis **minus** jumlahnya: Newton menyelesaikan `K d = −g`.
- Komponen terkunci (z) → 0.

`arrayLength(&dof_mask)` = jumlah baris (satu mask per baris).

## 3. `rhs` dipakai dua kali

1. Sebagai **right-hand side PCG**: `solveOnce` memakai `solver.arrays.rhs` di `cg_init` (`r = b`).
2. Sebagai **residual nonlinear**: host membacanya (`assemble()` di `newton.ts`) untuk menghitung `‖g‖` (karena ‖−g‖ = ‖g‖) dan `gᵀd`.

Satu kernel, dua kegunaan. Itulah kenapa `assemble()` di host selalu menjalankan `[element, matrix, force]` sekaligus, bahkan saat hanya butuh residual
(misalnya untuk skala langkah). Matriks ikut dirakit "sia-sia", tetapi biayanya kecil dan kodenya lebih sederhana.

## 4. Urutan dispatch

```ts
evaluate: [{ kernel: element, ... }, { kernel: matrix, ... }, { kernel: force, ... }],
```

Ketiganya dalam **satu compute pass**. Batas antar dispatch menjamin `matrix` dan `force` melihat `elem_hess` dan `elem_grad` yang sudah lengkap ditulis
`element` (dokumen 1.8 bagian 9).

## 5. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Gather vektor | per baris, kontributor dari peta oracle |
| Tanda minus | rhs = −g untuk `K d = −g` |
| Dua kegunaan | rhs PCG dan residual Newton |
| Satu pass | element → matrix → force |
