# Step 3.8: `gpu/kernels/hinge_element.wgsl`, Gradien dan Hessian Hinge per Thread

> Satu thread per hinge. Ia menulis 12 angka gaya dan 144 angka Hessian (16 blok 3×3) ke bagian hinge dari buffer lokal bersama.
> Override `EXACT` memilih Hessian eksak (12 lintasan dual) atau Gauss–Newton.

---

## 1. Override

```wgsl
override GRAD_OFFSET: u32 = 0u;
override HESS_OFFSET: u32 = 0u;
override EXACT: bool = true;
```

- `GRAD_OFFSET` = 9T dan `HESS_OFFSET` = 81T (dalam satuan float): awal bagian hinge, tepat setelah bagian membran (dokumen 3.6).
  Override dipakai karena nilainya konstan untuk satu kupon/drape.
- `EXACT`: dua pipeline dari satu sumber. Simulasi memakai `false` (Gauss–Newton, efek trigger); cek kernel memakai keduanya.

## 2. Bindings: 5 buffer

```wgsl
@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> hinge_vertices: array<u32>;
@group(0) @binding(2) var<storage, read> hinge_stiffness: array<f32>;
@group(0) @binding(3) var<storage, read_write> elem_grad: array<f32>;
@group(0) @binding(4) var<storage, read_write> elem_hess: array<f32>;
```

`elem_grad` dan `elem_hess` adalah **buffer yang sama** yang ditulis kernel membran E2. Setiap kernel menulis bagiannya sendiri, sehingga tidak ada tumpang tindih.

## 3. Alur per thread

```wgsl
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let h = gid.x;
  if (h >= arrayLength(&hinge_stiffness)) { return; }
  var x: array<vec3f, 4>;
  for (var k = 0u; k < 4u; k = k + 1u) {
    x[k] = position(hinge_vertices[4u * h + k]);
  }
  let theta = hinge_angle(x[0], x[1], x[2], x[3]);
  let factor = 2.0 * hinge_stiffness[h];
  var g = angle_gradient_dual(D3(x[0], vec3f(0.0)), ...);         // nilai ∇θ (turunan nol)
  ... gradien: g[k].v * (factor * theta) ...
  for (var column = 0u; column < 12u; column = column + 1u) {
    write_column(h, column, x, g, theta, factor);
  }
}
```

1. Ambil 4 posisi ke array **`var`**. WGSL hanya mengizinkan indeks runtime (`x[k]`) dengan aman pada variabel memori (`var`); pada nilai (`let`, parameter)
   kompiler bisa menolaknya. Ini juga alasan `seeded_gradient` menyalin parameternya ke `var source = x`.
2. θ (dengan `atan2_accurate`) dan `factor = 2k`.
3. `∇θ` dihitung sekali (dual dengan turunan nol = evaluasi biasa).
4. Gaya = `2kθ∇θ` ke `elem_grad[GRAD_OFFSET + 12h + 3k + d]`.
5. 12 kolom Hessian.

## 4. Satu kolom Hessian

```wgsl
fn write_column(h: u32, column: u32, x: array<vec3f, 4>, g: array<D3, 4>, theta: f32, factor: f32) {
  let l = column / 3u;
  let c = column % 3u;
  var curvature: array<D3, 4>;
  if (EXACT) {
    curvature = seeded_gradient(x, l, c);
  }
  var gradient = g;
  for (var k = 0u; k < 4u; k = k + 1u) {
    for (var r = 0u; r < 3u; r = r + 1u) {
      let second = select(0.0, theta * curvature[k].t[r], EXACT);
      elem_hess[HESS_OFFSET + 144u * h + 9u * (4u * k + l) + 3u * r + c] = factor * (gradient[k].v[r] * gradient[l].v[c] + second);
    }
  }
}
```

- Kolom (l, c) = titik l, komponen c.
- **`if (EXACT)`**: lintasan dual hanya dijalankan kalau eksak. Karena `EXACT` override, cabang ini seragam dan kompiler bisa menghapusnya di pipeline
  Gauss–Newton, yang jadi ~12× lebih murah.
- `var curvature` tanpa nilai awal otomatis diisi **nol** di WGSL (semua variabel diinisialisasi nol), jadi aman dipakai walau `EXACT = false`.
- Nilai = `2k(∂θ/∂x_k,r · ∂θ/∂x_l,c + θ · ∂²θ/∂x_k,r∂x_l,c)`, ditulis ke blok (k, l), baris r, kolom c.

## 5. Benih dual

```wgsl
fn seeded_gradient(x: array<vec3f, 4>, l: u32, c: u32) -> array<D3, 4> {
  var source = x;
  var points: array<D3, 4>;
  for (var k = 0u; k < 4u; k = k + 1u) {
    points[k] = D3(source[k], select(vec3f(0.0), vec3f(f32(c == 0u), f32(c == 1u), f32(c == 2u)), k == l));
  }
  return angle_gradient_dual(points[0], points[1], points[2], points[3]);
}
```

Titik l mendapat turunan = vektor satuan arah c; titik lain mendapat nol. `f32(c == 0u)` mengubah boolean menjadi 1.0/0.0.

## 6. Biaya

Per hinge eksak: 1 + 12 evaluasi gradien dual (masing-masing puluhan operasi vektor). Untuk n65 (~12 ribu hinge) itu ringan dibanding PCG. Simulasi E3
memakai Gauss–Newton, jadi hanya 1 evaluasi.

## 7. Hasil cek (n33)

| | Galat vs f64 |
|---|---|
| gradien hinge | 1,7×10⁻⁷ |
| Hessian eksak | 1,4×10⁻⁷ |
| Hessian Gauss–Newton | 1,3×10⁻⁷ |

## 8. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Offset override | menulis ke bagian hinge buffer bersama |
| Indeks runtime | hanya pada `var`; parameter disalin dulu |
| `EXACT` | cabang seragam; GN melewati 12 lintasan dual |
| Inisialisasi nol | variabel WGSL tanpa nilai awal = nol |
