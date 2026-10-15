# Step 2.10: `gpu/kernels/assemble_matrix.wgsl`, Merakit Matriks Newton ke Format BSR E1

> Kernel ini menjembatani fisika Step 2 dengan solver Step 1. Ia menulis tangen global **langsung ke buffer `vals` milik solver E1**,
> dengan pola gather yang deterministik.

---

## 1. Bindings

```wgsl
@group(0) @binding(0) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read> col_idx: array<u32>;
@group(0) @binding(2) var<storage, read> dof_mask: array<u32>;
@group(0) @binding(3) var<storage, read> block_contrib_ptr: array<u32>;
@group(0) @binding(4) var<storage, read> block_contrib_idx: array<u32>;
@group(0) @binding(5) var<storage, read> elem_hess: array<f32>;
@group(0) @binding(6) var<storage, read_write> vals: array<f32>;
```

`row_ptr`, `col_idx`, dan `vals` adalah **buffer yang sama** dengan milik solver E1 (`solver.arrays.rowPtr`, `colIdx`, `vals`). Tidak ada salinan.
Setelah kernel ini berjalan, PCG E1 langsung menyelesaikan matriks Newton yang baru dirakit.

7 storage buffer, di bawah default 8.

## 2. Kode

```wgsl
fn is_free(mask: u32, component: u32) -> bool {
  return ((mask >> component) & 1u) == 1u;
}

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  if (row >= arrayLength(&row_ptr) - 1u) {
    return;
  }
  for (var k = row_ptr[row]; k < row_ptr[row + 1u]; k = k + 1u) {
    let column = col_idx[k];
    for (var i = 0u; i < 9u; i = i + 1u) {
      var sum = 0.0;
      for (var j = block_contrib_ptr[k]; j < block_contrib_ptr[k + 1u]; j = j + 1u) {
        sum = sum + elem_hess[9u * block_contrib_idx[j] + i];
      }
      let r = i / 3u;
      let c = i % 3u;
      let free = is_free(dof_mask[row], r) && is_free(dof_mask[column], c);
      vals[9u * k + i] = select(select(0.0, 1.0, column == row && r == c), sum, free);
    }
  }
}
```

## 3. Tiga loop

1. **Blok baris ini** (`k`): rentang BSR, sama seperti SpMV.
2. **9 entri blok** (`i`): `r = i / 3`, `c = i % 3` (row-major).
3. **Kontributor** (`j`): dari peta `block_contrib_*` (dokumen 2.6). Setiap kontributor menunjuk ke blok lokal elemen di `elem_hess`.

Satu thread per baris, dan setiap thread hanya menulis blok-blok barisnya sendiri. Tidak ada thread lain yang menulis blok yang sama, sehingga
**tanpa race dan tanpa atomik**. Urutan kontributor = urutan elemen = urutan oracle, sehingga deterministik.

Analogi: daripada setiap order (elemen) meng-*increment* saldo akun (blok) secara bersamaan dengan lock, setiap akun menjalankan
`SELECT SUM(amount) WHERE account = me` atas daftar order yang sudah diindeks.

## 4. Masking

```wgsl
let free = is_free(dof_mask[row], r) && is_free(dof_mask[column], c);
vals[9u * k + i] = select(select(0.0, 1.0, column == row && r == c), sum, free);
```

- `dof_mask[row] >> r & 1`: apakah komponen r titik ini bebas? (bit 0 = x, 1 = y, 2 = z; kupon: `0b011`).
- Kalau baris **dan** kolom bebas, tulis jumlahnya.
- Kalau tidak: 1 di diagonal blok diagonal, 0 di tempat lain.

`select` bersarang dibaca dari dalam: `inner = (diagonal) ? 1 : 0`, lalu `free ? sum : inner`. Sama dengan `mask_matrix` di `coupon.rs`.

## 5. Hubungan dengan preconditioner

Setelah `vals` dirakit, `solveOnce` menjalankan `setup` → `jacobi_invert` membalik blok diagonal baru. Blok diagonal yang dimasking punya bentuk:

```
| a b 0 |
| c d 0 |
| 0 0 1 |
```

Inversnya terdefinisi dengan baik (sub-blok 2×2 ditambah 1). Tanpa angka 1 itu, determinannya nol dan `jacobi_invert` akan membagi dengan nol.

## 6. Biaya

Untuk n65: 4095 baris, ~28 ribu blok, setiap blok dengan 1–6 kontributor × 9 entri. Kecil dibanding PCG (ratusan iterasi), sehingga perakitan bukan
hambatan.

## 7. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Buffer bersama | menulis langsung ke `vals` solver E1 |
| Gather | per baris, kontributor dari peta oracle |
| Deterministik | urutan elemen tetap |
| Masking | komponen terkunci → baris/kolom identitas |
| Preconditioner aman | 1 di diagonal menjaga blok bisa dibalik |
