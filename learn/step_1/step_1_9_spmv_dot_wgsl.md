# Step 1.9: `gpu/kernels/spmv_dot.wgsl`, Perkalian Matriks Jarang × Vektor + Dot Product

> Kernel termahal per iterasi dan inti performa Experiment 1. Ia juga memperkenalkan dua konsep baru:
> **berhenti seragam satu workgroup** (`workgroupUniformLoad`) dan **fusi kernel** (dua tugas dalam satu lintasan memori).

---

## 1. Tugas

```
q = A p              (SpMV, BSR blok 3×3)
partial pᵀq          per workgroup
```

## 2. Isi file

```wgsl
@group(0) @binding(0) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read> col_idx: array<u32>;
@group(0) @binding(2) var<storage, read> vals: array<f32>;
@group(0) @binding(3) var<storage, read_write> p: array<f32>;
@group(0) @binding(4) var<storage, read_write> q: array<f32>;
@group(0) @binding(5) var<storage, read_write> partials: array<f32>;
@group(0) @binding(6) var<storage, read_write> state: State;

var<workgroup> lane: array<f32, 256>;
var<workgroup> stop: u32;

@compute @workgroup_size(256)
fn cs_main(...) {
  if (lid.x == 0u) {
    stop = state.status;
  }
  if (workgroupUniformLoad(&stop) != 0u) {
    return;
  }
  let row = gid.x;
  var pq = 0.0;
  if (row < arrayLength(&row_ptr) - 1u) {
    var a0 = 0.0; var a1 = 0.0; var a2 = 0.0;
    for (var k = row_ptr[row]; k < row_ptr[row + 1u]; k = k + 1u) {
      let c = 3u * col_idx[k];
      let o = 9u * k;
      let p0 = p[c]; let p1 = p[c + 1u]; let p2 = p[c + 2u];
      a0 = a0 + (vals[o] * p0 + vals[o + 1u] * p1 + vals[o + 2u] * p2);
      a1 = a1 + (vals[o + 3u] * p0 + vals[o + 4u] * p1 + vals[o + 5u] * p2);
      a2 = a2 + (vals[o + 6u] * p0 + vals[o + 7u] * p1 + vals[o + 8u] * p2);
    }
    let i = 3u * row;
    q[i] = a0; q[i + 1u] = a1; q[i + 2u] = a2;
    pq = p[i] * a0 + p[i + 1u] * a1 + p[i + 2u] * a2;
  }
  ... reduksi pohon lane → partials[2*wid] = pq_wg, partials[2*wid+1] = 0
}
```

## 3. Berhenti kalau solve sudah selesai

Host mengirim **32 iterasi sekaligus** dalam satu submit (dokumen 1.16). Bagaimana kalau CG sudah konvergen di iterasi ke-10
dari batch itu? 22 iterasi sisanya harus jadi **no-op**, kalau tidak `x` terus diubah dan iterasi terhitung salah.

Setiap kernel iterasi memeriksa `state.status`. Kalau bukan 0 (running), keluar.

### Kenapa tidak cukup `if (state.status != 0u) { return; }`?

Karena kernel ini punya `workgroupBarrier()` di reduksinya. Ingat aturan dokumen 1.8: semua thread workgroup harus mencapai barrier
yang sama. Keputusan "keluar atau tidak" harus **seragam** di seluruh workgroup.

Masalahnya, `state` ada di storage buffer, dan secara teori thread berbeda bisa membaca nilai berbeda (walau di sini nilainya tidak
berubah selama dispatch). Kompiler WGSL melakukan **uniformity analysis**: ia menolak kode yang memanggil barrier di dalam cabang
yang *mungkin* tidak seragam. Membaca storage buffer dianggap tidak seragam.

Solusinya:

```wgsl
if (lid.x == 0u) {
  stop = state.status;            // hanya thread 0 membaca status
}
if (workgroupUniformLoad(&stop) != 0u) {   // semua thread membaca nilai yang sama, dijamin seragam
  return;
}
```

`workgroupUniformLoad` melakukan barrier lalu membaca variabel workgroup, dan **kompiler tahu** hasilnya sama untuk semua thread.
Sekarang `return` di sini aman: seluruh workgroup keluar bersama, sehingga tidak ada yang tertinggal di barrier.

Analogi: daripada setiap anggota tim mengecek status proyek sendiri-sendiri (dan mungkin melihat versi berbeda), ketua tim membaca
status sekali, menulisnya di papan, lalu semua membaca papan yang sama.

## 4. SpMV per baris

```wgsl
for (var k = row_ptr[row]; k < row_ptr[row + 1u]; k = k + 1u) {
  let c = 3u * col_idx[k];
  let o = 9u * k;
```

Persis seperti `Bsr::multiply` di Rust (dokumen 1.1):

- `row_ptr[row] .. row_ptr[row+1]`: rentang blok baris ini.
- `col_idx[k]`: titik tetangga. `c = 3 × tetangga` = posisi 3 komponen tetangga di vektor `p`.
- `o = 9k`: posisi 9 angka blok di `vals`.
- Tiga akumulator `a0`, `a1`, `a2`: tiga komponen hasil baris ini.

Tanda kurung `a0 + (v0*p0 + v1*p1 + v2*p2)` sengaja sama dengan Rust: jumlah per blok dulu, baru ditambah ke akumulator.

### Pola akses memori

Setiap thread membaca ~11 blok × 9 float = ~400 byte dari `vals`, plus `p` dari ~11 tetangga. Thread yang bersebelahan
memproses baris yang bersebelahan, jadi blok-bloknya **berdekatan di memori** (karena BSR diurutkan per baris). Itu bagus untuk
**coalescing**: GPU menggabungkan permintaan memori dari thread bertetangga menjadi transaksi besar.

Ada ketimpangan: thread 0 membaca `vals[0..]`, thread 1 membaca `vals[99..]` (setelah blok-blok baris 0), sehingga aksesnya
tidak berdampingan sempurna. Format **sliced BSR** (yang disebut di gate bandwidth) menyusun ulang data supaya thread bertetangga
membaca alamat bertetangga. Gate bandwidth mengukur apakah ini perlu: hasilnya 33% dari rated, di atas batas 25%,
jadi tidak perlu.

## 5. Fusi: SpMV + dot dalam satu lintasan

```wgsl
let i = 3u * row;
q[i] = a0; q[i + 1u] = a1; q[i + 2u] = a2;
pq = p[i] * a0 + p[i + 1u] * a1 + p[i + 2u] * a2;
```

Setelah `q` baris ini dihitung, `pᵀq` untuk baris ini langsung dihitung selagi `a0..a2` masih di register dan `p[i]` baru saja dibaca.
Kalau dot product dikerjakan kernel terpisah, GPU harus **membaca ulang** `p` dan `q` dari VRAM.

Untuk solver yang dibatasi bandwidth memori (*memory-bound*), mengurangi lintasan memori = mengurangi waktu. Seluruh desain kernel
proyek ini mengikuti prinsip itu: `update_dot` juga menggabungkan update dan dua dot product.

## 6. Reduksi

Sama dengan `cg_init`, hanya satu nilai (`lane`, bukan dua). Slot kedua partial diisi 0:

```wgsl
partials[2u * wid.x] = lane[0];
partials[2u * wid.x + 1u] = 0.0;
```

Format `partials` selalu berpasangan supaya `reduce.wgsl` bisa membaca dengan pola sama di semua mode.

## 7. Biaya per iterasi di n257

| Data | Ukuran |
|---|---|
| `vals` | 25,6 MB |
| `col_idx` | 2,8 MB |
| `p` (dibaca ~11× per baris, sebagian dari cache) | 0,8 MB |
| `q` ditulis | 0,8 MB |

SpMV menyumbang sebagian besar dari ~41 MB lalu lintas "wajib" per iterasi. Dengan 0,70 ms/iterasi (timestamp GPU), itu sekitar
58 GB/s, atau 33% dari 176 GB/s rated.

## 8. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| SpMV | satu thread per baris, loop atas blok baris, 3 akumulator |
| Stop seragam | thread 0 baca status → `workgroupUniformLoad` → semua keluar bersama |
| Uniformity analysis | kompiler menolak barrier di cabang yang mungkin tidak seragam |
| Fusi | dot product dihitung saat data masih di register |
| Memory-bound | waktu ditentukan byte yang dipindahkan, bukan operasi aritmetika |
