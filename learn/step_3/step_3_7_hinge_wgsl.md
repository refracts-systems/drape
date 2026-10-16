# Step 3.7: `gpu/kernels/hinge.wgsl`, Bilangan Dual di WGSL dan Kisah `atan2` yang Kurang Akurat

> Header yang ditempel ke dua kernel hinge. Dokumen ini membahas dua hal: bagaimana bilangan dual ditulis di bahasa yang tidak punya generik maupun
> operator overloading, dan **bug presisi nyata** yang ditemukan lewat gate kernel: fungsi bawaan `atan2` WGSL hanya dijamin akurat sampai 4096 ULP.

---

## 1. Bilangan dual tanpa overloading

```wgsl
struct D1 {
  v: f32,
  t: f32,
}

struct D3 {
  v: vec3f,
  t: vec3f,
}

fn d1_div(a: D1, b: D1) -> D1 {
  return D1(a.v / b.v, (a.t * b.v - a.v * b.t) / (b.v * b.v));
}

fn d3_cross(a: D3, b: D3) -> D3 {
  return D3(cross(a.v, b.v), cross(a.t, b.v) + cross(a.v, b.t));
}
```

Di Rust, `a * b` pada `Dual` otomatis memanggil `mul`. WGSL tidak bisa begitu, jadi setiap operasi adalah fungsi bernama: `d3_sub`, `d3_add`, `d3_scale`, `d3_dot`,
`d3_cross`, `d1_div`, `d1_neg`, `d1_sqrt`. `D3` menyimpan vektor nilai dan vektor turunan sekaligus, sehingga turunan cross product cukup satu baris
(aturan perkalian untuk cross: `a' × b + a × b'`).

## 2. Gradien sudut dalam dual

```wgsl
fn angle_gradient_dual(x1: D3, x2: D3, x3: D3, x4: D3) -> array<D3, 4> {
  let edge = d3_sub(x4, x3);
  let edge_length = d1_sqrt(d3_dot(edge, edge));
  let n1 = d3_cross(d3_sub(x1, x3), d3_sub(x1, x4));
  ...
  return array<D3, 4>(d3_scale(u1, flip), d3_scale(u2, flip), d3_scale(u3, flip), d3_scale(u4, flip));
}
```

Terjemahan baris-demi-baris `angle_gradient_of` Rust. Fungsi WGSL bisa mengembalikan **array** (`array<D3, 4>`), karena array dengan elemen struct yang bisa
dibangun adalah tipe yang bisa dibangun.

Nama `edge_length` (bukan `length`) dipakai karena `length` adalah fungsi bawaan WGSL. Variabel lokal bernama sama akan menutupi fungsi itu.

## 3. Bug presisi: `atan2` hanya akurat 4096 ULP

### Gejala

Run GPU n33 pertama. Gate kernel (WGSL vs f64 ≤ 10⁻⁵):

| Kernel | Galat |
|---|---|
| gradien membran | 2,7×10⁻⁷ ✓ |
| tangen membran eksak | 8,0×10⁻⁸ ✓ |
| **gradien hinge** | **5,2×10⁻⁵ ✗** |
| **Hessian hinge eksak** | **7,3×10⁻⁶** |
| Hessian hinge Gauss–Newton | 1,3×10⁻⁷ ✓ |

Petunjuk penting: Gauss–Newton (`2k∇θ∇θᵀ`, **tanpa θ**) akurat, sedangkan yang memakai **θ** (gradien `2kθ∇θ` dan suku `θ∇²θ`) tidak.

### Investigasi

Tes sementara di Rust membaca output GPU dan mengurutkan hinge berdasarkan galat:

```
HINGE err 4.177e-8 hinge 1348 theta -1.060e-1 norm 4.064e-4
HINGE err 4.168e-8 hinge 365  theta -1.160e-1 norm 4.297e-4
...
```

Bukan beberapa hinge pencilan. **Semua** hinge salah ~10⁻⁴ relatif secara seragam (4×10⁻⁸ / 4×10⁻⁴). Galat sistematik sebesar itu terlalu besar untuk
pembulatan f32 biasa (~6×10⁻⁸), dan bentuknya cocok dengan satu fungsi yang tidak akurat: θ.

### Penyebab

Spesifikasi WGSL menetapkan akurasi **`atan` dan `atan2`: 4096 ULP**. Satu ULP f32 ≈ 6×10⁻⁸ relatif, jadi 4096 ULP ≈ **2,4×10⁻⁴** relatif. Implementasi GPU
boleh (dan sering) memakai aproksimasi cepat yang hanya seakurat itu. Bandingkan dengan `+`, `*`, `/` yang dijamin tepat sampai ~1 ULP.

Analogi backend: library yang menjanjikan "eventually consistent dalam 4 detik" padahal kamu butuh konsistensi milidetik. Kontraknya memang begitu; yang salah
adalah mengandalkannya.

### Perbaikan: hanya bagian yang rusak

```wgsl
const PI: f32 = 3.14159265358979;

// Cephes atanf (about 2 ulp). The WGSL builtin atan/atan2 is only required to be within 4096 ulp (≈ 2.4e-4 relative),
// which put every hinge angle, force and Hessian 1e-4 off the f64 values: too coarse for the 1e-5 kernel gate.
fn atan_accurate(value: f32) -> f32 {
  let x = abs(value);
  var offset = 0.0;
  var t = x;
  if (x > 2.414213562373095) {
    offset = 0.5 * PI;
    t = -1.0 / x;
  } else if (x > 0.4142135623730950) {
    offset = 0.25 * PI;
    t = (x - 1.0) / (x + 1.0);
  }
  let z = t * t;
  let y = offset + ((((0.0805374449538 * z - 0.138776856032) * z + 0.199777106478) * z - 0.333329491539) * z * t + t);
  return select(y, -y, value < 0.0);
}

fn atan2_accurate(y: f32, x: f32) -> f32 {
  if (x > 0.0) { return atan_accurate(y / x); }
  if (x < 0.0) { return atan_accurate(y / x) + select(-PI, PI, y >= 0.0); }
  return select(-0.5 * PI, 0.5 * PI, y >= 0.0) * f32(y != 0.0);
}
```

`atanf` dari library Cephes (klasik, akurasi ~2 ULP):

1. **Reduksi argumen**: x > tan(3π/8) → π/2 − atan(1/x); x > tan(π/8) → π/4 + atan((x−1)/(x+1)). Setelah reduksi |t| ≤ tan(π/8) ≈ 0,414.
2. **Polinom minimax** derajat rendah dalam z = t² untuk interval kecil itu.
3. Tanda dikembalikan.

`atan2_accurate` menangani empat kuadran: untuk x < 0 tambahkan ±π.

`var` dipakai di sini karena WGSL tidak punya ekspresi `if` yang mengembalikan nilai. Aturan "const over let" di proyek berlaku untuk TypeScript.

### Hasil

| Kernel | Sebelum | Sesudah |
|---|---|---|
| gradien hinge | 5,2×10⁻⁵ | **1,7×10⁻⁷** |
| Hessian hinge eksak | 7,3×10⁻⁶ | **1,4×10⁻⁷** |

Gate tidak dilonggarkan; yang diperbaiki adalah sumber galatnya. Fungsi bawaan lain yang dipakai proyek (`sqrt`, `length`, `cross`, `dot`) tidak bermasalah.

## 4. Sudut

```wgsl
fn hinge_angle(x1: vec3f, x2: vec3f, x3: vec3f, x4: vec3f) -> f32 {
  let edge = x4 - x3;
  let n1 = cross(x1 - x3, x1 - x4);
  let n2 = cross(x2 - x4, x2 - x3);
  let u1 = n1 * (1.0 / length(n1));
  let u2 = n2 * (1.0 / length(n2));
  let sine = dot(cross(u1, u2), edge * (1.0 / length(edge)));
  return atan2_accurate(sine, dot(u1, u2));
}
```

Sama dengan `HingeElement::angle` di Rust.

## 5. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Dual di WGSL | struct D1/D3 + fungsi bernama untuk setiap operasi |
| Mengembalikan array | `array<D3, 4>` dari fungsi |
| Akurasi bawaan | `atan`/`atan2` WGSL hanya 4096 ULP |
| Diagnosis | galat seragam di semua hinge + GN (tanpa θ) akurat → θ tersangka |
| Perbaikan | `atanf` Cephes ~2 ULP; gate tidak diubah |
