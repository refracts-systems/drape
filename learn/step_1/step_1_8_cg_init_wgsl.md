# Step 1.8: `gpu/kernels/cg_init.wgsl`, Dasar WGSL Bagian 3 (Memori Workgroup, Barrier, Reduksi Pohon)

> Kernel ini memperkenalkan masalah tersulit dalam pemrograman GPU: **menjumlahkan** nilai dari ribuan thread
> menjadi satu angka. Polanya dipakai ulang di `spmv_dot`, `update_dot`, `residual`, dan `reduce`.

---

## 1. Tugas kernel

Inisialisasi PCG:

```
x = 0, r = b, z = D⁻¹ r, p = z
dan partial sum per workgroup dari rᵀz dan rᵀr
```

Bagian pertama (inisialisasi vektor) sama mudahnya dengan `jacobi_invert`: satu thread per baris. Bagian kedua,
menjumlahkan `rᵀz` dari **semua** baris, adalah tantangannya.

## 2. Kenapa menjumlahkan itu sulit di GPU?

Di CPU: `let mut sum = 0; for i in 0..n { sum += a[i]*b[i]; }`. Satu thread, satu variabel.

Di GPU ada 65 ribu thread. Kalau semuanya melakukan `sum += nilaiku` ke satu variabel di storage buffer, terjadi
**race condition**: dua thread membaca `sum` lama yang sama, masing-masing menambah, lalu menulis; satu tambahan hilang.
Sama persis dengan dua request web yang meng-update saldo tanpa transaksi atau lock.

Solusi umum di CPU adalah *atomic add*. Tetapi **WGSL tidak punya atomic untuk f32** (hanya untuk i32/u32). Dan atomik juga membuat urutan
penjumlahan tidak tentu, sehingga hasilnya tidak deterministik, yang melanggar salah satu gate.

Solusinya adalah **reduksi pohon**, dalam dua tingkat:

1. Di dalam setiap workgroup: 256 thread bekerja sama menjumlahkan 256 nilai → 1 **partial**. Kernel ini melakukannya.
2. Lintas workgroup: 255 partial dijumlahkan oleh satu workgroup khusus. `reduce.wgsl` melakukannya (dokumen 1.10).

## 3. Isi file

```wgsl
// x = 0, r = b, z = D⁻¹r, p = z; per-workgroup partial sums of rᵀz and rᵀr.
@group(0) @binding(0) var<storage, read> rhs: array<f32>;
@group(0) @binding(1) var<storage, read_write> dinv: array<f32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> r: array<f32>;
@group(0) @binding(4) var<storage, read_write> z: array<f32>;
@group(0) @binding(5) var<storage, read_write> p: array<f32>;
@group(0) @binding(6) var<storage, read_write> partials: array<f32>;

var<workgroup> lane_a: array<f32, 256>;
var<workgroup> lane_b: array<f32, 256>;

@compute @workgroup_size(256)
fn cs_main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u,
) {
  let row = gid.x;
  var rz = 0.0;
  var rr = 0.0;
  if (row < arrayLength(&dinv) / 9u) {
    ... inisialisasi x, r, z, p ...
    rz = r0 * z0 + r1 * z1 + r2 * z2;
    rr = r0 * r0 + r1 * r1 + r2 * r2;
  }
  lane_a[lid.x] = rz;
  lane_b[lid.x] = rr;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (lid.x < stride) {
      lane_a[lid.x] = lane_a[lid.x] + lane_a[lid.x + stride];
      lane_b[lid.x] = lane_b[lid.x] + lane_b[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    partials[2u * wid.x] = lane_a[0];
    partials[2u * wid.x + 1u] = lane_b[0];
  }
}
```

## 4. Bagian 1: kerja per baris

```wgsl
if (row < arrayLength(&dinv) / 9u) {
  let i = 3u * row;
  let d = 9u * row;
  let r0 = rhs[i]; let r1 = rhs[i + 1u]; let r2 = rhs[i + 2u];
  let z0 = dinv[d] * r0 + dinv[d + 1u] * r1 + dinv[d + 2u] * r2;
  ...
  x[i] = 0.0; ...
  r[i] = r0; ...
  z[i] = z0; ...
  p[i] = z0; ...
```

- Jumlah baris dihitung dari `dinv` (9 angka per baris).
- `z = D⁻¹ r` adalah perkalian matriks 3×3 dengan vektor 3, ditulis terang-terangan.
- Perhatikan perbedaan dengan `jacobi_invert`: tidak ada `return` di awal untuk thread kelebihan. Thread kelebihan **tetap hidup**
  dengan `rz = rr = 0`, karena semua 256 thread **wajib** ikut reduksi di bawah (lihat bagian 6).

## 5. Memori workgroup: papan tulis tim

```wgsl
var<workgroup> lane_a: array<f32, 256>;
var<workgroup> lane_b: array<f32, 256>;
```

Variabel `workgroup` adalah memori kecil on-chip yang **dibagi oleh 256 thread satu workgroup** dan hanya hidup selama workgroup itu.
Workgroup lain punya salinan sendiri.

Analogi: setiap tim 256 orang punya **papan tulis** sendiri dengan 256 kotak. Setiap orang menulis angkanya di kotak bernomor
`lid.x` miliknya.

```wgsl
lane_a[lid.x] = rz;
lane_b[lid.x] = rr;
```

Dua papan karena dua penjumlahan dilakukan sekaligus (`rᵀz` dan `rᵀr`). Satu lintasan, dua hasil, lebih hemat.

## 6. `workgroupBarrier()`: "tunggu semua"

```wgsl
workgroupBarrier();
```

Thread dalam satu workgroup **tidak berjalan serentak sempurna**. Thread 0 bisa sudah sampai baris berikutnya sementara thread 200
belum menulis ke papan. Kalau thread 0 langsung membaca `lane_a[128]`, ia mungkin membaca nilai lama.

`workgroupBarrier()` adalah titik sinkronisasi: **tidak ada thread yang melewati barrier sampai semua 256 thread workgroup itu
sampai di barrier**, dan semua tulisan ke memori workgroup sebelum barrier terlihat oleh semua thread setelahnya.

Analogi: `await Promise.all(tim)` sebelum lanjut, atau *CyclicBarrier* di Java.

**Aturan keras**: semua thread workgroup **harus** mencapai barrier yang sama. Kalau sebagian thread sudah `return` atau
masuk cabang `if` yang tidak memanggil barrier, perilakunya tidak terdefinisi (deadlock atau hasil acak). Itulah kenapa
thread kelebihan di kernel ini tidak boleh `return` lebih awal.

## 7. Reduksi pohon

```wgsl
for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
  if (lid.x < stride) {
    lane_a[lid.x] = lane_a[lid.x] + lane_a[lid.x + stride];
    ...
  }
  workgroupBarrier();
}
```

`stride >> 1u` = bagi 2 (geser bit). Ilustrasi dengan 8 kotak, bukan 256:

```
awal:     [a0 a1 a2 a3 a4 a5 a6 a7]
stride 4: thread 0..3 menjumlahkan kotak i dan i+4
          [a0+a4, a1+a5, a2+a6, a3+a7, ...]
stride 2: thread 0..1
          [(a0+a4)+(a2+a6), (a1+a5)+(a3+a7), ...]
stride 1: thread 0
          [total, ...]
```

Untuk 256: stride 128, 64, 32, 16, 8, 4, 2, 1 = **8 langkah** (log₂ 256). Setiap langkah separuh thread bekerja, lalu barrier.
Dibanding loop 255 penjumlahan berurutan, ini jauh lebih cepat, dan **urutan penjumlahannya tetap** (deterministik).

Barrier di **dalam** loop penting: langkah stride 64 harus melihat hasil langkah stride 128 yang lengkap.

Perhatikan barrier ada **di luar** `if (lid.x < stride)`: semua thread memanggilnya, termasuk yang tidak menjumlahkan.

## 8. Menulis partial

```wgsl
if (lid.x == 0u) {
  partials[2u * wid.x] = lane_a[0];
  partials[2u * wid.x + 1u] = lane_b[0];
}
```

Hanya thread 0 setiap workgroup yang menulis hasil tim ke storage buffer. `partials` berisi pasangan `(a, b)` per workgroup:
`[rz_wg0, rr_wg0, rz_wg1, rr_wg1, ...]`, panjang `2 × 255`.

`wid.x` (nomor workgroup) menentukan posisinya, jadi tidak ada dua thread yang menulis ke tempat sama: **tanpa race**.

## 9. Lalu siapa yang menjumlahkan 255 partial?

Dispatch berikutnya: `reduce.wgsl` dengan MODE 0, dijalankan dengan **1 workgroup** saja. Ia membaca semua partial, menjumlahkannya
dengan pola yang sama, lalu menulis `rz`, `rr`, `rr0` ke `State`.

Kenapa dispatch terpisah? Karena **tidak ada barrier lintas workgroup** di WGSL. Workgroup 3 tidak bisa menunggu workgroup 200
di dalam satu dispatch. Satu-satunya titik sinkronisasi global adalah **batas antar dispatch**: WebGPU menjamin dispatch kedua
melihat semua tulisan dispatch pertama.

Analogi: satu dispatch = satu fase *map*. Batas dispatch = *shuffle barrier*. `reduce` = fase *reduce*. Ini MapReduce mini.

## 10. Kesetiaan pada emulasi f32

Pohon di sini (`stride = 128 … 1`) **identik** dengan `tree()` di `pcg.rs`, dan baris di luar batas mengisi 0 seperti
`row_or_zero`. Karena itu emulasi CPU bisa menghitung angka yang sama bit-per-bit, selama kompiler GPU tidak mengubah urutan
operasinya. Kompiler GPU **memang** kadang menggabungkan `a*b + c` jadi satu instruksi FMA (dibahas di dokumen 1.24 dan 1.26).

## 11. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Race condition | banyak thread menulis satu lokasi → tambahan hilang |
| Tidak ada atomic f32 | reduksi harus dirancang sebagai pohon |
| Memori workgroup | papan tulis bersama 256 thread, cepat, kecil |
| `workgroupBarrier()` | semua thread workgroup harus sampai sebelum lanjut |
| Jangan `return` sebelum barrier | semua thread wajib mencapai barrier yang sama |
| Reduksi pohon | log₂(256) = 8 langkah, deterministik |
| Partial | satu hasil per workgroup; dijumlahkan dispatch berikutnya |
| Batas dispatch | satu-satunya sinkronisasi global |
