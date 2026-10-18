# Step 1.7: `gpu/kernels/jacobi_invert.wgsl`, Dasar WGSL Bagian 2 (Thread, Dispatch, Satu Thread per Baris)

> Kernel pertama yang benar-benar menghitung sesuatu. Kita pakai untuk memahami **model eksekusi GPU**:
> siapa yang menjalankan fungsi, berapa kali, dan bagaimana setiap pemanggilan tahu bagiannya.

---

## 1. Isi file

```wgsl
// D⁻¹: inverse of each 3×3 diagonal block (adjugate / determinant), same order as oracle pcg::invert3.
@group(0) @binding(0) var<storage, read> vals: array<f32>;
@group(0) @binding(1) var<storage, read> diag_idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> dinv: array<f32>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  if (row >= arrayLength(&diag_idx)) {
    return;
  }
  let o = 9u * diag_idx[row];
  let a = vals[o]; let b = vals[o + 1u]; let c = vals[o + 2u];
  let d = vals[o + 3u]; let e = vals[o + 4u]; let f = vals[o + 5u];
  let g = vals[o + 6u]; let h = vals[o + 7u]; let i = vals[o + 8u];
  let c00 = e * i - f * h;
  ...
  let det = a * c00 + b * c10 + c * c20;
  let w = 9u * row;
  dinv[w] = c00 / det; ...
}
```

Tugasnya: untuk setiap baris, ambil blok diagonal 3×3 dari matriks dan simpan **inversnya** ke `dinv`. Ini preconditioner
block-Jacobi `M⁻¹` dari dokumen 1.4.

## 2. Bindings: "parameter" kernel

```wgsl
@group(0) @binding(0) var<storage, read> vals: array<f32>;
```

Kernel tidak punya parameter fungsi seperti `fn f(a, b)`. Ia menerima data lewat **binding**: slot bernomor yang diisi host
dengan buffer sebelum kernel dijalankan.

- `@group(0)`: kelompok binding ke-0. Proyek ini hanya memakai satu grup.
- `@binding(0)`: slot ke-0 dalam grup.
- `var<storage, read> vals: array<f32>`: slot itu berisi storage buffer read-only berisi f32, diberi nama `vals` di kode ini.

Analogi: seperti **named parameter** di query SQL (`:vals`, `:diag_idx`), dengan host yang mengikat nilainya.
Di host, `KernelSpec.bindings` adalah objek `{ vals: <buffer>, diag_idx: <buffer>, dinv: <buffer> }`. Backend mencocokkan
**nama** ke nomor `@binding` (raw WebGPU mem-parse nomornya dengan regex; vgpu melakukannya sendiri).

## 3. Entry point dan `@workgroup_size`

```wgsl
@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
```

- `@compute`: ini **compute shader** (bukan untuk menggambar piksel).
- `fn cs_main`: nama fungsi masuk. Host menyebut nama ini saat membuat pipeline (`entryPoint: "cs_main"`).
- `@workgroup_size(256)`: thread dikelompokkan per **256**. Satu kelompok = satu **workgroup**.

## 4. Model eksekusi: dispatch dan workgroup

Host menjalankan kernel dengan perintah **dispatch**: "jalankan `W` workgroup". Total thread = `W × 256`.

Untuk n257, ada 65.025 baris. Host menghitung:

```ts
const workgroupsFor = (rows: number): number => Math.ceil(rows / WORKGROUP);   // ceil(65025 / 256) = 255
```

255 workgroup × 256 thread = **65.280 thread**. Setiap thread mengerjakan satu baris. GPU menjadwalkan workgroup ke
*streaming multiprocessor* (SM) yang tersedia. RTX 3050 Laptop punya 16 SM, dan setiap SM menjalankan beberapa workgroup bergantian.

Analogi: kamu submit **255 batch job**, masing-masing berisi 256 task. Cluster punya 16 node; scheduler membagi batch ke node.
Kamu tidak mengontrol urutan batch mana duluan.

## 5. `global_invocation_id`: nomor tiket setiap thread

```wgsl
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
```

Setiap thread menerima **nomor unik global**: `gid.x = workgroup_id.x × 256 + local_invocation_id.x`.

- Thread pertama workgroup 0: `gid.x = 0`.
- Thread terakhir workgroup 254: `gid.x = 254 × 256 + 255 = 65279`.

`vec3u` karena dispatch bisa 3 dimensi (x, y, z), cocok untuk gambar atau volume. Di sini hanya `x` yang dipakai.

Ada tiga builtin indeks yang akan sering muncul:

| Builtin | Arti | Rentang di proyek |
|---|---|---|
| `global_invocation_id` | nomor thread global | 0 … 65279 |
| `local_invocation_id` | nomor thread di dalam workgroup | 0 … 255 |
| `workgroup_id` | nomor workgroup | 0 … 254 |

## 6. Pemeriksaan batas: wajib

```wgsl
if (row >= arrayLength(&diag_idx)) {
  return;
}
```

Ada 65.280 thread tetapi hanya 65.025 baris. **255 thread terakhir tidak punya pekerjaan.** Tanpa cek ini mereka membaca di luar
array. Di WebGPU itu tidak *crash* (akses di luar batas dijepit atau dikembalikan nol oleh implementasi), tetapi hasilnya
sampah dan bisa **menulis** ke posisi yang salah.

`arrayLength(&diag_idx)` memberi panjang runtime array (dari ukuran buffer yang diikat). `&` di sini membuat **referensi**
ke variabel. WGSL butuh referensi untuk fungsi ini, bukan salinan nilainya.

Ini pola yang muncul di **setiap** kernel proyek ini: hitung indeks dari `gid`, keluar kalau melewati batas, lalu kerjakan.

## 7. `let` vs `var`

```wgsl
let o = 9u * diag_idx[row];
var a0 = 0.0;
```

- `let`: konstanta (tidak bisa diubah setelah diisi). Seperti `const` di TypeScript.
- `var`: variabel yang bisa diubah. Seperti `let` di TypeScript.

Perhatikan **pembalikan nama** yang membingungkan: `let` WGSL = `const` TS, dan `var` WGSL = `let` TS.

## 8. Membalik matriks 3×3

```wgsl
let o = 9u * diag_idx[row];
let a = vals[o]; let b = vals[o + 1u]; let c = vals[o + 2u];
...
let c00 = e * i - f * h;
let c01 = c * h - b * i;
...
let det = a * c00 + b * c10 + c * c20;
dinv[w] = c00 / det; ...
```

- `diag_idx[row]` adalah nomor blok diagonal baris ini di array BSR (dari `bsr.rs::diagonal_positions`). Tanpa array ini,
  setiap thread harus mencari kolom = row di daftar bloknya.
- 9 angka blok dibaca ke variabel `a`…`i`.
- Invers dengan **adjugate**: matriks kofaktor ditranspose, dibagi determinan. Untuk 3×3 ini rumus tertutup tanpa loop.
- Hasil ditulis ke `dinv[9*row .. 9*row+9]`.

Komentar "same order as oracle pcg::invert3" penting: rumus kofaktor ditulis dengan urutan operasi **persis sama**
dengan versi Rust, supaya emulasi f32 bisa mereproduksi hasilnya.

## 9. Kenapa ini cocok untuk GPU?

- Setiap baris **independen**: thread 7 tidak perlu tahu apa pun tentang thread 8.
- Tidak ada penjumlahan lintas thread, tidak butuh sinkronisasi.
- Ini disebut **embarrassingly parallel**: pola ideal GPU.

Kernel ini dijalankan **sekali per solve** (di dispatch `setup`), bukan tiap iterasi.

## 10. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Binding | parameter kernel berupa buffer bernomor, diikat oleh host |
| `@workgroup_size(256)` | thread dikelompokkan 256 per workgroup |
| Dispatch | host bilang "jalankan W workgroup"; W = ceil(baris/256) |
| `global_invocation_id` | nomor unik thread → nomor baris |
| Cek batas | thread kelebihan harus `return` |
| `let` / `var` | konstan / bisa diubah (kebalikan dari nama TS) |
| Embarrassingly parallel | setiap thread independen, tanpa sinkronisasi |
