# Step 1.10: `gpu/kernels/reduce.wgsl`, Reduksi Akhir dan Mesin Status CG di GPU

> Satu kernel, lima "mode". Inilah tempat semua keputusan skalar CG dibuat: panjang langkah α, koefisien β,
> deteksi konvergensi, deteksi kerusakan numerik, dan residual replacement. Ia juga memperkenalkan **override constant**.

---

## 1. Lima mode

```wgsl
// One workgroup: strided sum of per-workgroup partials, a fixed tree, then the CG scalar update.
// MODE 0 = INIT (rᵀz, rᵀr), 1 = ALPHA (pᵀq), 2 = BETA (rᵀz, rᵀr, convergence), 3 = TRUE (‖b − Ax‖²),
// 4 = REPLACE (rᵀz, rᵀr of the replaced residual).
override MODE: u32 = 0u;
```

| Mode | Dipanggil setelah | Menjumlahkan | Menulis ke `State` |
|---|---|---|---|
| 0 INIT | `cg_init` | rᵀz, rᵀr | `rz`, `rr`, `rr0`, `iteration=0`, status |
| 1 ALPHA | `spmv_dot` | pᵀq | `pq`, `alpha`, status breakdown |
| 2 BETA | `update_dot` | rᵀz, rᵀr | `beta`, `rz`, `rr`, `iteration++`, status konvergen/batas |
| 3 TRUE | `residual` (ukur) | ‖b−Ax‖² | `true_rr` |
| 4 REPLACE | `residual` (replace) | rᵀz, rᵀr | `rz`, `rr` |

## 2. `override`: konstanta yang diisi saat membuat pipeline

```wgsl
override MODE: u32 = 0u;
```

`override` adalah konstanta yang nilainya **ditentukan host saat pipeline dibuat**, bukan saat kode ditulis dan bukan saat runtime.
Kompiler kemudian bisa menghapus cabang `switch MODE` yang tidak terpakai, sehingga tiap mode menjadi kernel ramping sendiri.

Host membuat **lima pipeline** dari satu sumber:

```ts
function reduceSpec<A>(sources: KernelSources, arrays: Arrays<A>, mode: number): KernelSpec<A> {
  const label = `reduce_mode_${mode}`;
  return { label, source: sources.reduce, constants: { MODE: mode }, bindings: { partials: arrays.partials, state: arrays.state } };
}
```

Analogi: seperti **feature flag yang di-bake saat build**: satu codebase, beberapa binary dengan flag berbeda.
Bandingkan dengan membaca mode dari buffer saat runtime: itu bisa, tetapi setiap thread harus membaca dan bercabang, dan kompiler tidak
bisa menghapus kode mati.

## 3. Satu workgroup saja

Kernel ini selalu di-dispatch dengan **1 workgroup** (256 thread):

```ts
setup: [at(k.invert, rows), at(k.init, rows), at(k.start, 1)],
iteration: [at(k.spmv, rows), at(k.alpha, 1), at(k.update, rows), at(k.beta, 1), at(k.direction, rows)],
```

`at(k.alpha, 1)` = satu workgroup. Karena hanya satu workgroup, barrier di dalamnya cukup untuk menyinkronkan **semua** thread
yang terlibat, sehingga hasilnya bisa ditulis ke `State` tanpa race.

## 4. Kapan berhenti

```wgsl
if (lid.x == 0u) {
  stop = select(state.status, 0u, MODE == 0u || MODE == 3u);
  if (MODE == 4u && state.rr <= state.replace_above2 * state.rr0) {
    stop = 1u;
  }
}
if (workgroupUniformLoad(&stop) != 0u) {
  return;
}
```

`select(a, b, cond)` di WGSL = `cond ? b : a`. **Urutannya terbalik** dari operator ternary, dan ini sumber bug klasik.

- MODE 0 (inisialisasi) dan 3 (ukur residual akhir) **selalu** berjalan, walau status ≠ 0. Mode 3 dijalankan setelah solve selesai,
  saat status sudah 1.
- MODE 1, 2: berhenti kalau solve sudah selesai.
- MODE 4: berhenti kalau solve selesai **atau** residual sudah di bawah gate replacement. Kondisi ini **sama persis** dengan yang dicek
  `residual.wgsl` dalam mode replace, jadi kalau kernel residual melewatkan replacement, reduce juga melewatkannya.

## 5. Jumlah strided lalu pohon

```wgsl
let count = arrayLength(&partials) / 2u;
var a = 0.0;
var b = 0.0;
for (var j = lid.x; j < count; j = j + 256u) {
  a = a + partials[2u * j];
  b = b + partials[2u * j + 1u];
}
lane_a[lid.x] = a;
lane_b[lid.x] = b;
workgroupBarrier();
... pohon 128 → 1 ...
if (lid.x == 0u) {
  finish(lane_a[0], lane_b[0]);
}
```

Ada 255 partial di n257 (lebih sedikit dari 256 thread), tetapi kode ini benar untuk berapa pun jumlahnya: thread `t` menjumlahkan
partial `t`, `t+256`, `t+512`, dan seterusnya. Ini **strided sum** yang ditiru `strided_sum` di `pcg.rs`.

## 6. Keputusan skalar: `finish`

```wgsl
fn finish(sum_a: f32, sum_b: f32) {
  switch MODE {
    case 0u: { start(sum_a, sum_b); }
    case 1u: { step_length(sum_a); }
    case 2u: { advance(sum_a, sum_b); }
    case 4u: { replace(sum_a, sum_b); }
    default: { state.true_rr = sum_a; }
  }
}
```

WGSL punya fungsi biasa (`fn`). Fungsi ini dipanggil hanya oleh thread 0, yang menulis ke `State`.

### MODE 0

```wgsl
fn start(rz: f32, rr: f32) {
  state.rz = rz;
  state.rr = rr;
  state.rr0 = rr;
  state.iteration = 0u;
  state.status = select(0u, 1u, rr <= 0.0);
}
```

Kalau `b = 0`, langsung dianggap konvergen (jawabannya nol).

### MODE 1: α dan deteksi breakdown

```wgsl
// Bit test, because WGSL lets compilers assume NaN and Inf never occur.
fn is_finite(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn step_length(pq: f32) {
  let alpha = state.rz / pq;
  state.pq = pq;
  state.alpha = alpha;
  state.status = select(3u, 0u, pq > 0.0 && is_finite(alpha));
}
```

Untuk matriks positif definit, `pᵀAp` selalu > 0. Kalau ≤ 0 atau α tak hingga/NaN, terjadi **breakdown** (status 3).

**Detail penting**: kenapa tidak `isNan(alpha)`? Spesifikasi WGSL mengizinkan kompiler **menganggap NaN dan Inf tidak pernah terjadi**
(demi optimasi), sehingga `alpha != alpha` (trik deteksi NaN klasik) bisa dioptimasi menjadi `false`. Solusinya adalah memeriksa
**bit mentah**: `bitcast<u32>(value)` menafsirkan ulang 32 bit float sebagai integer tanpa konversi. Dalam format IEEE-754,
float tak hingga atau NaN punya 8 bit eksponen yang semuanya 1 (`0x7f800000`). Kompiler tidak bisa "mengoptimasi" operasi bit.

### MODE 2: β, iterasi, dan konvergensi

```wgsl
fn advance(rz: f32, rr: f32) {
  state.beta = rz / state.rz;
  state.rz = rz;
  state.rr = rr;
  state.iteration = state.iteration + 1u;
  if (rr <= state.tol2 * state.rr0) {
    state.status = 1u;
  } else if (state.iteration >= state.max_iterations) {
    state.status = 2u;
  }
}
```

Kriteria berhenti: `‖r‖² ≤ tol² · ‖r₀‖²`, ekuivalen dengan `‖r‖/‖r₀‖ ≤ 10⁻⁷` tanpa akar kuadrat. Ini **residual rekursif**.

### MODE 4: setelah residual replacement

```wgsl
fn replace(rz: f32, rr: f32) {
  state.rz = rz;
  state.rr = rr;
}
```

`rz` baru dipakai untuk α di iterasi berikutnya (`α = rz/pq`) dan untuk β (`β = rz'/rz`). Sama dengan `replace_residual` di Rust.

### MODE 3 (default)

`state.true_rr = sum_a;`: residual sejati ‖b − Ax‖² yang dihitung GPU sendiri dalam f32, sebagai pemeriksaan drift.

## 7. Kenapa semuanya di GPU?

Lihat dokumen 1.6 bagian 9: seluruh iterasi berjalan di GPU tanpa host. Konsekuensinya, logika kontrol (if konvergen, if breakdown)
harus ditulis di GPU juga. Kernel ini pada dasarnya adalah **state machine** CG.

## 8. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| `override` | konstanta diisi saat membuat pipeline; satu sumber, lima pipeline |
| Satu workgroup | barrier cukup untuk sinkron global → aman menulis `State` |
| `select(a, b, c)` | = `c ? b : a` (urutan terbalik!) |
| Breakdown | `pᵀq ≤ 0` atau α tidak hingga → status 3 |
| `bitcast` | cek NaN/Inf lewat bit, karena kompiler boleh menganggap NaN tidak ada |
| Konvergensi | `rr ≤ tol²·rr0` pada residual rekursif |
