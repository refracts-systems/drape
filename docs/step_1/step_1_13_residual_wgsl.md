# Step 1.13: `gpu/kernels/residual.wgsl`, Residual Sejati, Residual Replacement, dan Batas 10 Storage Buffer

> Kernel ini diubah di sesi kerja. Awalnya hanya mengukur `‖b − Ax‖²` di akhir solve. Sekarang ia juga menjalankan
> **residual replacement** di tengah solve. Dokumen ini membahas desainnya, termasuk dua masalah nyata yang muncul.

---

## 1. Dua mode lewat `override bool`

```wgsl
// True residual s = b − A x in f32.
// REPLACE = false: per-workgroup partial sum of sᵀs (the GPU's own drift check).
// REPLACE = true: residual replacement r = s, z = D⁻¹s, partial sums of rᵀz and rᵀr; skipped once the solve has stopped
// or the recursive residual is below the replacement gate.
override REPLACE: bool = false;
```

Seperti `MODE` di `reduce.wgsl`, satu sumber menghasilkan dua pipeline:

```ts
function residualSpec<A>(sources: KernelSources, a: Arrays<A>, replace: boolean): KernelSpec<A> {
  const bindings = { row_ptr: a.rowPtr, col_idx: a.colIdx, vals: a.vals, rhs: a.rhs, x: a.x, partials: a.partials, dinv: a.dinv, r: a.r, z: a.z, state: a.state };
  return { label: replace ? "residual_replace" : "residual", source: sources.residual, constants: { REPLACE: replace ? 1 : 0 }, bindings };
}
```

Konstanta `REPLACE` dikirim sebagai angka 1/0; WebGPU mengonversinya ke `bool`.

## 2. Bindings: 10 buffer

```wgsl
@group(0) @binding(0) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read> col_idx: array<u32>;
@group(0) @binding(2) var<storage, read> vals: array<f32>;
@group(0) @binding(3) var<storage, read> rhs: array<f32>;
@group(0) @binding(4) var<storage, read_write> x: array<f32>;
@group(0) @binding(5) var<storage, read_write> partials: array<f32>;
@group(0) @binding(6) var<storage, read_write> dinv: array<f32>;
@group(0) @binding(7) var<storage, read_write> r: array<f32>;
@group(0) @binding(8) var<storage, read_write> z: array<f32>;
@group(0) @binding(9) var<storage, read_write> state: State;
```

Mode ukur hanya butuh 6 buffer pertama, tetapi binding yang muncul di kode (walau di cabang `if (!REPLACE)` yang mati) tetap dianggap
**dipakai statis**, jadi kedua pipeline punya 10 binding.

### Masalah nyata #1: batas 8 storage buffer

Run pertama setelah perubahan ini gagal:

```
GPUPipelineError: The number of storage buffers (10) in the Compute stage exceeds the maximum per-stage limit (8).
This adapter supports a higher maxStorageBuffersPerShaderStage of 10, which can be specified in requiredLimits ...
```

WebGPU punya **limit default** yang dijamin di semua perangkat. Untuk storage buffer per stage, defaultnya 8.
Perangkat yang lebih kuat bisa lebih, tetapi kode harus **memintanya** saat membuat device.

Perbaikan minimal (dokumen 1.15, 1.17, 1.18):

```ts
/** residual.wgsl in REPLACE mode binds 10 storage buffers; the WebGPU default per-stage limit is 8. */
export const REQUIRED_STORAGE_BUFFERS = 10;
...
requiredLimits: { maxStorageBuffersPerShaderStage: REQUIRED_STORAGE_BUFFERS },
```

Analogi: seperti `max_connections` di database. Default-nya aman untuk semua server, tetapi kalau butuh lebih, kamu menaikkannya
secara eksplisit di konfigurasi, dan server yang tidak sanggup menolak dengan error yang jelas.

## 3. Berhenti: dua alasan

```wgsl
if (lid.x == 0u) {
  stop = select(0u, state.status | u32(state.rr <= state.replace_above2 * state.rr0), REPLACE);
}
if (workgroupUniformLoad(&stop) != 0u) {
  return;
}
```

- Mode ukur (`REPLACE = false`): `stop = 0`, selalu jalan. Mode ini dijalankan setelah solve selesai.
- Mode replace: berhenti kalau `status ≠ 0` **atau** `rr ≤ replace_above² · rr0`.
  `u32(bool)` mengubah true/false jadi 1/0, lalu `|` (OR bit) menggabungkannya dengan status.

Gate disimpan di `State.replace_above2` (diisi host), **bukan** sebagai `override`. Alasannya: dua kernel (residual replace dan
reduce MODE 4) harus mengambil keputusan yang **sama persis**. Kalau residual melewatkan replacement tetapi reduce tidak,
reduce akan menjumlahkan `partials` basi. Dengan keduanya membaca nilai yang sama dari `State` dan menghitung ekspresi yang sama,
keputusannya selalu konsisten.

## 4. Kerja per baris: fungsi `residual_row`

```wgsl
// (sᵀs, 0) when measuring; (rᵀz, rᵀr) after writing r and z when replacing.
fn residual_row(row: u32) -> vec2f {
  var a0 = 0.0; var a1 = 0.0; var a2 = 0.0;
  for (var k = row_ptr[row]; k < row_ptr[row + 1u]; k = k + 1u) {
    ... a = A x (sama dengan SpMV) ...
  }
  let i = 3u * row;
  let s0 = rhs[i] - a0; ...
  let ss = s0 * s0 + s1 * s1 + s2 * s2;
  if (!REPLACE) {
    return vec2f(ss, 0.0);
  }
  let d = 9u * row;
  let z0 = dinv[d] * s0 + ...;
  r[i] = s0; ...
  z[i] = z0; ...
  return vec2f(s0 * z0 + s1 * z1 + s2 * z2, ss);
}
```

- Fungsi WGSL bisa mengembalikan vektor: `vec2f` = dua f32. Ini cara rapi mengembalikan dua nilai sekaligus.
- `s = b − A x` adalah **residual sejati**, dihitung dari nol, bukan rekursif.
- Mode replace menulis `r = s` dan `z = D⁻¹ s`, lalu mengembalikan `(rᵀz, rᵀr)` untuk direduksi `reduce` MODE 4.

Urutan aritmetika sama dengan `replace_residual` di Rust: `multiply_row`, `rhs − ax`, `precondition_row`, `row_dot`.

## 5. Di mana kernel ini dipanggil

```ts
finish: [at(k.residual, rows), at(k.measure, 1)],               // akhir solve: ukur ‖b − Ax‖²
replacement: [at(k.replace, rows), at(k.replaced, 1)],          // setiap 50 iterasi
```

Dan di `solver.ts`, replacement ditempel di akhir setiap chunk:

```ts
const iterations = solver.replaceEvery > 0 ? solver.replaceEvery : settings.chunkIterations;
const tail = solver.replaceEvery > 0 ? solver.replacement : [];
return { chunk: [...Array.from({ length: iterations }, () => solver.iteration).flat(), ...tail], iterations };
```

Dengan replacement aktif, satu chunk = tepat 50 iterasi + replacement. Karena chunk selalu dimulai dari iterasi 0, replacement jatuh
tepat setelah iterasi 50, 100, 150, dan seterusnya. Tidak perlu menghitung modulo di GPU.

## 6. Masalah nyata #2: replacement tanpa gate tidak pernah konvergen

Sudah dibahas di dokumen 1.4 bagian 8. Ringkasnya: f32 tidak bisa menghitung `b − Ax` lebih teliti dari lantai pembulatannya.
Setiap replacement mereset residual ke lantai itu, sehingga toleransi 10⁻⁷ tidak pernah tercapai. Gate `> 10⁻²` membatasi
replacement ke fase awal, saat residual masih besar.

## 7. Hasil di GPU

| n257 | tanpa replacement | dengan replacement |
|---|---|---|
| tension: iterasi | 1524 | 1545 |
| tension: residual sejati (f64) | 1,25×10⁻² | **7,84×10⁻³** ✓ |
| dynamic: residual sejati | 5,94×10⁻³ | 4,67×10⁻³ |

## 8. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| `override bool` | satu sumber → pipeline ukur dan pipeline replace |
| Binding statis | binding di cabang mati tetap dihitung |
| Limit device | default 8 storage buffer per stage; minta 10 lewat `requiredLimits` |
| Gate di `State` | dua kernel membaca keputusan yang sama → konsisten |
| `vec2f` return | cara rapi mengembalikan dua nilai |
| Chunk = periode | replacement jatuh tepat di iterasi 50, 100, … |
