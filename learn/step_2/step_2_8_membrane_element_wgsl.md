# Step 2.8: `gpu/kernels/membrane_element.wgsl`, Kernel per Elemen untuk Gaya dan Tangen

> Kernel fisika utama Step 2. Satu thread per **segitiga**: hitung F, E, S, lalu tulis 9 angka gaya dan 81 angka tangen.
> Ia memperkenalkan loop bersarang, struct lokal, dan indeks komponen vektor dinamis.

---

## 1. Bindings

```wgsl
@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> elem_vertices: array<u32>;
@group(0) @binding(2) var<storage, read> elem_geom: array<f32>;
@group(0) @binding(3) var<storage, read_write> elem_grad: array<f32>;
@group(0) @binding(4) var<storage, read_write> elem_hess: array<f32>;
```

| Buffer | Isi | Ukuran (n33: 2048 segitiga) |
|---|---|---|
| `positions` | x,y,z semua titik | 1089 × 3 |
| `elem_vertices` | 3 titik per segitiga | 2048 × 3 |
| `elem_geom` | g0x g0y g1x g1y g2x g2y luas | 2048 × 7 |
| `elem_grad` | output: gaya 3 titik × 3 komponen | 2048 × 9 |
| `elem_hess` | output: 9 blok × 9 angka | 2048 × 81 |

## 2. Fungsi pembaca

```wgsl
fn position(v: u32) -> vec3f {
  return vec3f(positions[3u * v], positions[3u * v + 1u], positions[3u * v + 2u]);
}

fn shape_gradient(e: u32, k: u32) -> vec2f {
  return vec2f(elem_geom[7u * e + 2u * k], elem_geom[7u * e + 2u * k + 1u]);
}
```

Fungsi kecil yang membaca buffer global (boleh, karena buffer adalah variabel modul). Mereka membuat kode utama terbaca seperti rumus.

**Kenapa `array<f32>` dan bukan `array<vec3f>`?** `vec3f` di storage buffer punya **alignment 16 byte**, jadi `array<vec3f>` memakan 16 byte per elemen
(4 byte terbuang) dan tidak cocok dengan layout padat dari Rust (12 byte). Menyimpan sebagai f32 datar dan merakit `vec3f` manual menghindari jebakan
layout itu.

## 3. Entry point

```wgsl
@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let e = gid.x;
  if (e >= arrayLength(&elem_vertices) / 3u) {
    return;
  }
  let x0 = position(elem_vertices[3u * e]);
  let f = columns_of(position(elem_vertices[3u * e + 1u]) - x0, position(elem_vertices[3u * e + 2u]) - x0, shape_gradient(e, 1u), shape_gradient(e, 2u));
  let stress = stress_of(green_strain(f));
  let area = elem_geom[7u * e + 6u];
  for (var k = 0u; k < 3u; k = k + 1u) {
    write_gradient(e, k, f, stress, area);
    for (var l = 0u; l < 3u; l = l + 1u) {
      write_block(e, k, l, f, stress, area);
    }
  }
}
```

- Satu thread = satu elemen. Tidak ada barrier, jadi `return` dini aman.
- F, E, S dihitung **sekali** per elemen, lalu dipakai untuk 3 gaya dan 9 blok.
- Loop `for` dengan `var` biasa. Loop berukuran tetap kecil seperti ini biasanya di-*unroll* kompiler.

## 4. Gaya

```wgsl
fn write_gradient(e: u32, k: u32, f: Columns, s: vec3f, area: f32) {
  let g = shape_gradient(e, k);
  let pw = f.wale * s.x + f.course * s.z;
  let pc = f.wale * s.z + f.course * s.y;
  let value = (pw * g.x + pc * g.y) * area;
  let o = 9u * e + 3u * k;
  elem_grad[o] = value.x; elem_grad[o + 1u] = value.y; elem_grad[o + 2u] = value.z;
}
```

Persis `MembraneElement::gradient`: `P = F·S` (dua kolom `pw`, `pc`), lalu `(P · grad_n[k]) × area`. Ingat `s.z` = `S_wc`.

## 5. Tangen

```wgsl
struct StrainRows {
  wale: vec3f,
  course: vec3f,
  shear: vec3f,
}

fn strain_rows(g: vec2f, f: Columns) -> StrainRows {
  return StrainRows(f.wale * g.x, f.course * g.y, (f.wale * g.y + f.course * g.x) * 0.5);
}

fn write_block(e: u32, k: u32, l: u32, f: Columns, s: vec3f, area: f32) {
  let gk = shape_gradient(e, k);
  let gl = shape_gradient(e, l);
  let a = strain_rows(gk, f);
  let b = strain_rows(gl, f);
  let geometric = gk.x * (s.x * gl.x + s.z * gl.y) + gk.y * (s.z * gl.x + s.y * gl.y);
  let o = 81u * e + 9u * (3u * k + l);
  for (var r = 0u; r < 3u; r = r + 1u) {
    for (var c = 0u; c < 3u; c = c + 1u) {
      var value = select(0.0, geometric, r == c);
      value = value + a.wale[r] * b.wale[c] * WALE;
      value = value + a.course[r] * b.course[c] * COURSE;
      value = value + a.wale[r] * b.course[c] * COUPLING;
      value = value + a.course[r] * b.wale[c] * COUPLING;
      value = value + a.shear[r] * b.shear[c] * (4.0 * SHEAR);
      elem_hess[o + 3u * r + c] = value * area;
    }
  }
}
```

- `StrainRows`: turunan (E_ww, E_cc, E_wc) terhadap posisi titik k. Sama dengan `strain_vectors` di Rust.
- `geometric`: suku kekakuan geometrik `gkᵀ S gl`, hanya di diagonal (`select(0.0, geometric, r == c)` = `r == c ? geometric : 0`).
- Lima suku material: `outer(a, b)[r][c] = a[r] * b[c]`, dikali modulus.
- `a.wale[r]`: indeks komponen vektor dengan variabel. WGSL mengizinkan indeks dinamis pada nilai vektor.
- **Tegangan tidak dipotong**: ini tangen **eksak**, sama dengan `MembraneElement::tangent`, bukan `hessian` E1.

### Layout output

`elem_hess[81e + 9(3k + l) + 3r + c]`: elemen e, blok (k, l), baris r, kolom c. Kernel `assemble_matrix` membaca blok (k, l) lewat indeks `9e + 3k + l`
× 9, cocok dengan peta dari `e2_generate.rs`.

## 6. Kenapa per elemen, bukan per titik?

Satu thread per elemen menghitung F/E/S sekali dan menulis ke slot **miliknya sendiri** (`elem_grad[9e..]`, `elem_hess[81e..]`). Tidak ada dua thread
menulis tempat yang sama, sehingga tanpa race. Penjumlahan ke titik/blok global dilakukan kernel lain dengan pola gather.

Ini pola dua fase klasik GPU FEM: **hitung lokal** (per elemen) → **rakit** (per baris). Harganya memori perantara (`elem_hess`: 81 float per elemen),
tetapi hasilnya deterministik tanpa atomik.

## 7. Diverifikasi

`run-e2.ts` menjalankan kernel ini sekali pada `kernel_state` (keadaan acak teregang dari oracle), membaca `elem_grad` dan `elem_hess`, lalu
`e2-compare` membandingkannya dengan `gradient` dan `tangent` f64 pada posisi yang sama (dibulatkan f32):

| | gaya | tangen |
|---|---|---|
| galat relatif | ≤ 5,1×10⁻⁷ | ≤ 1,9×10⁻⁷ |
| batas | 10⁻⁵ | 10⁻⁵ |

## 8. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Satu thread per elemen | F/E/S sekali, 9 gaya + 81 tangen |
| `array<f32>` datar | menghindari alignment 16 byte `vec3f` |
| Struct lokal | `Columns`, `StrainRows` sebagai nilai |
| Indeks komponen dinamis | `a.wale[r]` |
| Tangen eksak | tegangan apa adanya, tidak dipotong |
| Dua fase | lokal per elemen, lalu gather per baris |
