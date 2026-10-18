# Step 6.2: `sums.wgsl` dan `argmin.wgsl`, Reduksi di GPU untuk Loop Newton

> Newton tetap di host, tetapi beberapa angka kecil yang selama ini diambil dengan **membaca seluruh buffer** (norma residual, kemiringan gᵀd, jumlah
> perubahan energi, minimum batas α) sekarang bisa dihitung di GPU dan dibaca sebagai 8 byte. Dua kernel baru melakukannya. Kernel E1 tidak disentuh.

---

## 1. Apa itu reduksi, dan kenapa dua lintasan

**Reduksi** = menggabungkan jutaan angka menjadi satu (jumlah, minimum). Di CPU itu `reduce()` berurutan. Di GPU 256 thread bekerja bersama dalam satu
workgroup, tetapi satu workgroup hanya bisa bekerja pada sebagian data, dan workgroup berbeda **tidak bisa saling menunggu** di dalam satu dispatch.
Karena itu reduksi besar dilakukan dua lintasan:

1. **Lintasan 0**: sejumlah workgroup (maks 128), tiap workgroup menjumlahkan sebagian data dan menulis hasil sebagian (partial) ke buffer.
2. **Lintasan 1**: satu workgroup menjumlahkan semua partial.

Analogi backend: **map-reduce**. Lintasan 0 = mapper yang masing-masing meringkas satu shard, lintasan 1 = reducer tunggal.

## 2. `sums.wgsl`

```wgsl
override MODE: u32 = 0u;
override PASS: u32 = 0u;
override OUT_OFFSET: u32 = 0u;
override GROUPS: u32 = 1u;
```

`override` adalah konstanta yang diisi **saat pipeline dibuat** (bukan saat shader jalan), mirip variabel environment saat deployment. Satu file shader
menjadi empat pipeline:

| Pipeline | MODE | PASS | Menghitung | Ditulis ke |
|---|---|---|---|---|
| sums_rhs_partial | 0 | 0 | Σ rhs², Σ rhs·x per workgroup | `partials` |
| sums_rhs_final | 0 | 1 | menjumlahkan partial | `out[0], out[1]` |
| sums_delta_partial | 1 | 0 | Σ δ, Σ \|δ\| per workgroup | `partials` |
| sums_delta_final | 1 | 1 | menjumlahkan partial | `out[2], out[3]` |

```wgsl
fn strided_sums(lane: u32, group: u32, groups: u32) -> vec2f {
  var sums = vec2f(0.0, 0.0);
  let stride = groups * 256u;
  for (var i = group * 256u + lane; i < arrayLength(&a); i = i + stride) {
    let v = a[i];
    let w = b[i];
    sums = sums + select(vec2f(v, abs(v)), vec2f(v * v, v * w), MODE == 0u);
  }
  return sums;
}
```

- **Grid-stride loop**: thread nomor *t* di workgroup *g* memproses elemen `g·256 + t`, lalu melompat sejauh `groups·256`, dan seterusnya. Dengan begitu
  jumlah workgroup tetap kecil (≤ 128) berapa pun panjang datanya, dan akses memori berurutan antar thread (cepat).
- `vec2f` menampung dua jumlah sekaligus. `select(a, b, kondisi)` adalah ternary tanpa percabangan.
- `b` selalu direferensikan (dibaca) walau MODE 1 tidak memakainya. Alasannya praktis: kalau sebuah binding tidak dipakai pipeline tertentu, layout
  otomatis dapat membuangnya dan `bind group` yang menyertakannya error. Di MODE 1, `b` di-bind ke buffer yang sama dengan `a`.

```wgsl
var<workgroup> first: array<f32, 256>;
var<workgroup> second: array<f32, 256>;
...
for (var span = 128u; span > 0u; span = span >> 1u) {
  if (lid.x < span) {
    first[lid.x] = first[lid.x] + first[lid.x + span];
    second[lid.x] = second[lid.x] + second[lid.x + span];
  }
  workgroupBarrier();
}
```

Reduksi pohon dalam workgroup (dokumen 1.8): setiap putaran separuh thread menambahkan pasangannya, `workgroupBarrier()` menjamin semua penjumlahan putaran
ini selesai sebelum putaran berikutnya.

## 3. Konsekuensi: urutan penjumlahan berubah

Host menjumlahkan f32 **berurutan** (`for` dengan `Math.fround`). GPU menjumlahkan sebagai **pohon**. Penjumlahan f32 tidak asosiatif, sehingga hasilnya
berbeda di bit terakhir. Karena itu rung C dan D tidak harus identik bit-per-bit dengan A dan B, dan aturan prompt menyebutnya begitu. Yang diperiksa:

- galat reduksi vs jumlah f64 dari nilai f32 yang sama: ‖rhs‖² 1,3×10⁻⁷, gᵀd 1,7×10⁻⁷ (batas 10⁻⁵);
- semua gate akurasi tetap lulus; dan di n33 alasan berhenti setiap langkah identik antar rung.

## 4. `argmin.wgsl`

Minimum nilai **dan indeksnya**, supaya host tahu vertex mana yang membatasi α.

```wgsl
fn strided_min(lane: u32, group: u32, groups: u32) -> vec2u {
  var value = 3.0e38;
  var index = NONE;
  ...
    if (values[i] < value) { value = values[i]; index = i; }
  return vec2u(bitcast<u32>(value), index);
}

fn better(value: f32, index: u32, other_value: f32, other_index: u32) -> bool {
  return other_value < value || (other_value == value && other_index < index);
}
```

- `bitcast<u32>(value)` menafsirkan 32 bit f32 sebagai bilangan bulat tanpa mengubah bitnya, supaya nilai dan indeks bisa dibawa dalam satu `vec2u`.
  Hasil akhirnya disimpan di buffer f32, jadi indeks di-bitcast lagi ke f32 dan host menafsirkannya kembali sebagai u32 (`decodeReduced`).
- Nilai NaN diabaikan (`NaN < x` selalu salah).
- Tie-break ke indeks **lebih kecil** membuat hasil deterministik walau urutan workgroup tidak.

Hasilnya: `out[4]` = α_max minimum, `out[5]` = barisnya, `out[6]` = jarak minimum, `out[7]` = barisnya (buffer `reduced`, 8 float = 32 byte).

## 5. `reductions.ts`

Menyiapkan 8 pipeline (4 jumlah + 4 minimum), buffer `reduced` dan `partials`, serta daftar dispatch siap pakai:

| Daftar | Isi |
|---|---|
| `rhsStats` | sums_rhs_partial + sums_rhs_final |
| `energySums` | sums_delta_partial + sums_delta_final |
| `boundMin` | argmin batas α |
| `distanceMin` | argmin jarak |

`SLOT` memberi nama posisi hasil. Batas kernel: tiap pipeline memakai ≤ 4 storage buffer.

## 6. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Dua lintasan | map-reduce: partial per workgroup, lalu satu workgroup menjumlahkannya |
| `override` | konstanta waktu-pembuatan-pipeline: satu shader jadi empat pipeline |
| Grid-stride loop | jumlah workgroup tetap, thread melompat sejauh total thread |
| Pohon vs berurutan | f32 tidak asosiatif: hasil GPU berbeda di bit terakhir dari host |
| argmin dengan bitcast | membawa nilai dan indeks dalam satu vektor, tie-break deterministik |
