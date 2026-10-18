# Step 2.9: `gpu/kernels/membrane_energy_change.wgsl`, Perubahan Energi Tanpa Pembatalan Angka

> Kernel untuk line search Armijo di GPU. Ia ada karena **f32 tidak bisa mengurangkan dua energi yang hampir sama**.
> Dokumen ini juga menunjukkan dua kesalahan sintaks WGSL nyata dari sesi kerja.

---

## 1. Masalah yang dipecahkan

Armijo bertanya: "kalau melangkah α sepanjang arah d, berapa energi turun?". Cara naif:

```
ΔΠ = Π(x + αd) − Π(x)
```

Di dekat jawaban, Π ≈ 0,05 J dan ΔΠ ≈ 10⁻¹⁰ J. f32 hanya menyimpan ~7 digit, sehingga kedua energi dibulatkan ke angka yang sama dan ΔΠ jadi 0 atau
derau acak. Armijo akan menerima atau menolak secara acak.

Analogi: mengukur berat satu lembar kertas dengan menimbang truk bermuatan kertas, lalu truk tanpa satu lembar itu. Timbangannya tidak cukup teliti.
Timbang kertasnya langsung.

## 2. Rumus eksak (dokumen 2.2)

```
ΔF = α · columns(d)
ΔE_ww = F_w·ΔF_w + ½|ΔF_w|²
ΔE_cc = F_c·ΔF_c + ½|ΔF_c|²
ΔE_wc = ½(F_w·ΔF_c + ΔF_w·F_c + ΔF_w·ΔF_c)
ΔW    = S(E):ΔE + W(ΔE)            ← eksak untuk StVK
ΔΠ_e  = luas · ΔW
```

Semua suku kecil dihitung **langsung**, sehingga tidak ada pengurangan dua angka besar.

## 3. Bindings

```wgsl
@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> direction: array<f32>;
@group(0) @binding(2) var<storage, read> elem_vertices: array<u32>;
@group(0) @binding(3) var<storage, read> elem_geom: array<f32>;
@group(0) @binding(4) var<storage, read> step: array<f32>;
@group(0) @binding(5) var<storage, read_write> elem_delta: array<f32>;
```

- `direction`: arah Newton `d` **per titik** (host menyebarkan solusi PCG per-baris ke array per-titik).
- `step`: buffer satu angka berisi α.

### Kenapa α di buffer, bukan `override`?

α berubah setiap percobaan line search (1, ½, ¼, …). `override` hanya bisa diisi saat membuat pipeline; mengganti nilainya berarti **kompilasi ulang**
(puluhan milidetik). Buffer satu angka cukup di-*write* (4 byte) sebelum setiap dispatch.

| Nilai | Berubah | Mekanisme |
|---|---|---|
| Modulus material | tidak pernah | `override` |
| α | setiap percobaan | buffer `step` |
| Posisi, arah | setiap iterasi | buffer besar |

## 4. Kode utama

```wgsl
@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let e = gid.x;
  if (e >= arrayLength(&elem_vertices) / 3u) {
    return;
  }
  let v0 = elem_vertices[3u * e];
  let v1 = elem_vertices[3u * e + 1u];
  let v2 = elem_vertices[3u * e + 2u];
  let g1 = shape_gradient(e, 1u);
  let g2 = shape_gradient(e, 2u);
  let f = columns_of(position(v1) - position(v0), position(v2) - position(v0), g1, g2);
  let d = columns_of(displacement(v1) - displacement(v0), displacement(v2) - displacement(v0), g1, g2);
  let dw = d.wale * step[0];
  let dc = d.course * step[0];
  let change = vec3f(
    dot(f.wale, dw) + 0.5 * dot(dw, dw),
    dot(f.course, dc) + 0.5 * dot(dc, dc),
    0.5 * (dot(f.wale, dc) + dot(dw, f.course) + dot(dw, dc)),
  );
  let s = stress_of(green_strain(f));
  let first = s.x * change.x + s.y * change.y + 2.0 * s.z * change.z;
  elem_delta[e] = elem_geom[7u * e + 6u] * (first + energy_density(change));
}
```

- `columns_of` dipanggil **dua kali**: untuk posisi (F) dan arah (ΔF per satuan α). Karena F linear dalam posisi, rumus yang sama berlaku untuk arah.
- `2.0 * s.z * change.z`: faktor 2 geser (dokumen 2.1).
- Output: satu angka per elemen. Host menjumlahkannya.

## 5. Dua kesalahan nyata dari sesi kerja

### WGSL tidak punya destrukturisasi tuple

Versi pertama:

```wgsl
let (v0, v1, v2) = (elem_vertices[3u * e], elem_vertices[3u * e + 1u], elem_vertices[3u * e + 2u]);   // ✗ bukan WGSL
```

Itu sintaks Rust. WGSL tidak punya tuple, sehingga harus tiga `let` terpisah. Kesalahan yang sama muncul di `assemble_matrix.wgsl` (`let (r, c) = ...`).
Keduanya diperbaiki sebelum dijalankan.

Pelajaran untuk developer Rust/TS: WGSL **mirip** Rust, tetapi jauh lebih kecil. Tuple, closure, pattern matching, generik, dan trait tidak ada.

### Trailing comma

`vec3f(a, b, c,)` dengan koma di akhir **boleh** di WGSL modern. Kode di atas memakainya di konstruktor `change`.

## 6. Penjumlahan di host

```ts
async function energyChange<A, K>(gpu: CouponGpu<A, K>, stepLength: number): Promise<number> {
  gpu.backend.write(gpu.step, Float32Array.of(stepLength));
  await gpu.backend.run(gpu.energyChange);
  const deltas = new Float32Array(await gpu.backend.read(gpu.elemDelta));
  return deltas.reduce((sum, v) => Math.fround(sum + v), 0);
}
```

Untuk kupon (≤ 8192 elemen), membaca semua `elem_delta` dan menjumlahkannya di host lebih sederhana daripada kernel reduksi baru. `Math.fround`
membulatkan setiap jumlah parsial ke f32, sehingga penjumlahannya tetap aritmetika f32 sungguhan, bukan f64 JavaScript diam-diam.

## 7. Hasil

Identitas `energy_change` vs selisih langsung (diuji dalam f64): ~10⁻¹⁵. Di GPU, line search tidak pernah perlu memotong langkah, karena prediktor
sudah menaruh Newton di daerah kuadratik. Semua α = 1, sama dengan f64.

## 8. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Pembatalan | f32 tidak bisa mengurangkan dua energi yang hampir sama |
| Rumus eksak StVK | ΔW = S:ΔE + W(ΔE), tanpa pengurangan besar |
| α di buffer | override butuh kompilasi ulang; buffer cukup ditulis |
| Tanpa tuple | WGSL bukan Rust: tiga `let` terpisah |
| `Math.fround` | penjumlahan host tetap f32 |
