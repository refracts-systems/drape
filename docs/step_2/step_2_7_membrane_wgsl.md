# Step 2.7: `gpu/kernels/membrane.wgsl`, Header WGSL Bersama untuk Hukum Material

> Seperti `state.wgsl` di Step 1, file ini bukan kernel. Ia potongan kode yang **ditempel** di depan dua kernel membran.
> Ia memperkenalkan dua fitur WGSL baru: **struct sebagai nilai kembali** dan **override f32**.

---

## 1. Isi

```wgsl
// Constant-strain triangle and orthotropic St. Venant–Kirchhoff law in material axes (wale = x, course = y),
// prepended by the host to the membrane kernels. Same formulas and order as oracle membrane.rs / material.rs.
override WALE: f32 = 0.0;
override COURSE: f32 = 0.0;
override COUPLING: f32 = 0.0;
override SHEAR: f32 = 0.0;

struct Columns {
  wale: vec3f,
  course: vec3f,
}

fn columns_of(e1: vec3f, e2: vec3f, g1: vec2f, g2: vec2f) -> Columns {
  return Columns(e1 * g1.x + e2 * g2.x, e1 * g1.y + e2 * g2.y);
}

fn green_strain(f: Columns) -> vec3f {
  return vec3f(0.5 * (dot(f.wale, f.wale) - 1.0), 0.5 * (dot(f.course, f.course) - 1.0), 0.5 * dot(f.wale, f.course));
}

fn stress_of(e: vec3f) -> vec3f {
  return vec3f(WALE * e.x + COUPLING * e.y, COURSE * e.y + COUPLING * e.x, 2.0 * SHEAR * e.z);
}

fn energy_density(e: vec3f) -> f32 {
  return 0.5 * WALE * e.x * e.x + 0.5 * COURSE * e.y * e.y + COUPLING * e.x * e.y + 2.0 * SHEAR * e.z * e.z;
}
```

## 2. Modulus material sebagai `override`

```wgsl
override WALE: f32 = 0.0;
```

Nilai diisi host saat membuat pipeline (dokumen 1.10), dari `problem.json` → `materialConstants()` di `coupon-gpu.ts`:

```ts
return { WALE: m.wale_N_per_m, COURSE: m.course_N_per_m, COUPLING: m.coupling_N_per_m, SHEAR: m.shear_N_per_m };
```

Kenapa override, bukan buffer? Modulus **konstan** sepanjang simulasi. Sebagai override, kompiler memperlakukannya seperti konstanta: bisa melipat
perkalian (misalnya `2.0 * SHEAR` dihitung sekali) dan tidak perlu membaca memori di setiap thread.

Default `0.0` sengaja: kalau host lupa mengisi, semua gaya nol, dan tes kernel langsung gagal keras. Itu lebih baik daripada angka material diam-diam
salah.

**Jebakan WebGPU**: kunci di `constants` harus override yang **ada** di modul. Kedua kernel membran memakai keempat modulus, sehingga keempat kunci
selalu valid.

## 3. Struct sebagai nilai

```wgsl
struct Columns { wale: vec3f, course: vec3f }

fn columns_of(...) -> Columns {
  return Columns(e1 * g1.x + e2 * g2.x, e1 * g1.y + e2 * g2.y);
}
```

WGSL tidak bisa mengembalikan tuple. Untuk mengembalikan dua vektor, definisikan struct. `Columns(a, b)` adalah konstruktor (urutan field).
Berbeda dengan `State` di Step 1 (yang hidup di storage buffer), `Columns` adalah **nilai** di register, gratis.

### Operasi vektor bawaan

- `e1 * g1.x`: vektor × skalar (setiap komponen).
- `e1 * g1.x + e2 * g2.x`: penjumlahan vektor.
- `dot(a, b)`: dot product.
- `.x`, `.y`, `.z`: akses komponen.

Rust menulis `add(scale(e1, g1[0]), scale(e2, g2[0]))`, sedangkan WGSL cukup `e1 * g1.x + e2 * g2.x`. Aritmetika vektor adalah bawaan bahasa GPU.

## 4. Pemetaan ke Rust

| WGSL | Rust | Dokumen |
|---|---|---|
| `columns_of(e1, e2, g1, g2)` | `MembraneElement::columns` | 2.2 |
| `green_strain(f)` | `MembraneElement::strain` | 2.2 |
| `stress_of(e)` | `Material::stress` | 2.1 |
| `energy_density(e)` | `Material::energy_density` | 2.1 |

`columns_of` menerima rusuk dan gradien sebagai **argumen**, bukan membaca buffer. Alasannya: header dipakai dua kernel dengan nama binding berbeda
(`positions` vs `direction`). Kernel energy-change memanggilnya dua kali: sekali untuk posisi (F), sekali untuk arah (ΔF per satuan α). Fungsi murni
tanpa akses buffer bisa dipakai ulang di kedua tempat.

WGSL juga tidak mudah menerima pointer ke storage buffer sebagai parameter fungsi (butuh fitur bahasa tambahan), jadi fungsi murni adalah pilihan aman.

## 5. Urutan operasi

Setiap ekspresi meniru urutan Rust: `WALE * e.x + COUPLING * e.y` (bukan `COUPLING * e.y + WALE * e.x`). Kompiler GPU masih boleh memakai FMA, tetapi
selebihnya hasil setia. Pengukuran: kernel elemen WGSL cocok dengan f64 sampai ~5×10⁻⁷ relatif, kira-kira beberapa ε f32.

## 6. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Header | potongan kode bersama, ditempel host |
| `override f32` | modulus material diisi saat pipeline dibuat |
| Struct nilai | cara mengembalikan dua vektor |
| Fungsi murni | dipakai ulang untuk posisi dan arah |
| Aritmetika vektor | bawaan WGSL (`*`, `+`, `dot`) |
