# Step 3.6: `oracle/src/e3_generate.rs`, File Soal Drape, Lintasan Referensi, dan Peta Gather Gabungan

> Kontrak Rust↔GPU Step 3. Bagian paling cerdik adalah **peta gather gabungan**: blok lokal membran, hinge, dan inersia ditaruh di satu array
> berurutan, sehingga kernel perakitan E2 bisa dipakai **tanpa diubah**.

---

## 1. Perintah

```bash
oracle e3-generate --n 33 --out drape/drape-n33                       # dengan lintasan f64 (≈ 4 menit)
oracle e3-generate --n 129 --reference false --out drape/drape-n129   # GPU saja (tanpa lintasan f64)
```

n129 tanpa referensi karena lintasan f64-nya butuh berjam-jam (Cholesky 48 ribu unknown × ribuan iterasi Newton). Sesuai prompt, n129 hanya untuk waktu
dan pertumbuhan iterasi.

## 2. Konstanta

```rust
pub const NEWTON_TOLERANCE: f64 = 1e-6;            // GPU f32
const REFERENCE_TOLERANCE: f64 = 1e-10;            // oracle f64
const MAX_NEWTON_ITERATIONS: usize = 30;
const ARMIJO: f64 = 1e-4;
const MAX_HALVINGS: usize = 30;
const PCG_TOLERANCE: f64 = 1e-7;                   // solver E1 tidak berubah
const REPLACE_EVERY: usize = 50;
const REPLACE_ABOVE: f64 = 1e-2;
/// Probe k is the step from state k−1 to state k of the f64 trajectory.
pub const PROBE_STEPS: [usize; 5] = [1, 10, 60, 120, 240];
/// Evidence-triggered: the exact tangent lost positive definiteness at step 87 of the n33 drape.
pub const TANGENT: TangentKind = TangentKind::Projected;
```

**Interpretasi probe**: prompt meminta "di langkah 1, 10, 60, 120, 240 lintasan f64, jalankan satu langkah GPU dari keadaan f64 dan bandingkan dengan keadaan
f64 berikutnya". Langkah 240 adalah langkah terakhir; tidak ada keadaan 241. Maka probe k diartikan sebagai **langkah dari keadaan k−1 ke keadaan k**, sehingga
kelima probe berada di dalam 240 langkah.

## 3. Isi `problem.bin`

| Array | Isi |
|---|---|
| `rest_positions` | posisi awal (datar) |
| `elem_vertices`, `elem_geom` | membran, sama dengan E2 |
| `hinge_vertices`, `hinge_stiffness` | **baru**: 4 titik + k per hinge |
| `row_vertex`, `dof_mask` | mask `0b111` (semua komponen bebas) |
| `row_mass` | **baru**: m/h² per baris |
| `row_ptr`, `col_idx`, `diag_idx` | pola BSR (segitiga + hinge) |
| `block_contrib_*`, `force_contrib_*` | peta gather gabungan |
| `fixed_vertex` | dua sudut |
| `kernel_state` | keadaan terlipat untuk cek kernel |
| `probe_steps`, `probe_positions`, `probe_velocities` | **baru**: keadaan k−1 dari lintasan f64 (kosong untuk n129) |

## 4. Peta gather gabungan

Ingat E2: `assemble_matrix.wgsl` menjumlahkan blok dari array `elem_hess` menurut daftar indeks (satuan 9 float per blok). Kuncinya: kernel itu **tidak tahu**
dari mana blok berasal. Jadi cukup taruh semua blok lokal di **satu** array, berurutan:

```
elem_hess (satuan blok 9 float):
[ membran: 9 blok × T elemen | hinge: 16 blok × H hinge | massa: 1 blok × R baris ]
  indeks 9e + 3k + l          9T + 16h + 4k + l            9T + 16H + row
```

```rust
/// Local 3×3 blocks in units of 9 floats: mass 9T + 16H + row, then membrane 9e + 3k + l, then hinge 9T + 16h + 4k + l.
fn block_contributions(drape: &Drape) -> Vec<Vec<u32>> {
    ...
    (0..drape.rows()).for_each(|row| lists[position(row, row)].push((9 * t + 16 * h + row) as u32));
    for (e, element) in drape.membranes.iter().enumerate() {
        add_pairs(drape, &mut lists, &element.vertices, |k, l| 9 * e + 3 * k + l);
    }
    for (i, hinge) in drape.hinges.iter().enumerate() {
        add_pairs(drape, &mut lists, &hinge.vertices, |k, l| 9 * t + 16 * i + 4 * k + l);
    }
    lists
}
```

Urutan dalam daftar: massa, lalu membran, lalu hinge. Itu sama dengan urutan perakitan tangen di oracle (`DrapeStep::tangent`).

Gaya memakai trik yang sama (satuan 3 float): inersia `3T + 4H + row`, membran `3e + k`, hinge `3T + 4h + k`.

Hasilnya, **tidak satu baris pun** `assemble_matrix.wgsl` atau `assemble_force.wgsl` diubah. Kernel hinge dan inersia hanya perlu menulis ke bagian buffer mereka
sendiri, lewat override offset (dokumen 3.8 dan 3.10).

Analogi backend: tiga service menulis ke tabel event yang sama dengan rentang ID berbeda, dan satu consumer yang sudah ada membaca semuanya tanpa tahu
asal-usulnya.

## 5. Referensi

`reference.bin` berisi:

- posisi f64 semua keadaan 0..240,
- per langkah: `newton_iterations`, `final_residual`, `converged`, `halvings`, `scale`,
- semua residual setiap iterasi (dirangkai), plus `residual_offsets`.

Data per langkah disimpan dalam **biner**, bukan hanya JSON, karena `jsonscan` di compare hanya bisa membaca kunci datar. Ini perubahan yang dilakukan setelah
run pertama: proses generate dihentikan, array ditambahkan, lalu generate dijalankan ulang.

`reference.json` mencatat tangen, **teks trigger** (langkah 87, pivot −1,21×10⁻¹), cek FD, energi, dan riwayat per langkah.

## 6. Ringkasan cetak

```
drape n=33 rows=1087 | f64 Newton 3896 iterations over 240 steps, 44 not at 1e-10, worst final 1.13e-2
```

## 7. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| n129 tanpa referensi | lintasan f64 terlalu mahal; GPU saja |
| Probe k | langkah k−1 → k |
| Gather gabungan | membran + hinge + massa dalam satu array; kernel perakitan E2 tanpa perubahan |
| Referensi biner | data per langkah bisa dibaca compare |
