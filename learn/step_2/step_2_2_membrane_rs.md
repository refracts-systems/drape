# Step 2.2: `oracle/src/membrane.rs`, Elemen Segitiga (F, Green Strain, Gaya, Tangen, Perubahan Energi)

> Jantung fisika Step 2. Setiap rumus di file ini punya kembaran WGSL di `membrane.wgsl` dan `membrane_element.wgsl`.
> Step 2 menambahkan dua metode: `tangent` (Hessian eksak) dan `energy_change` (perubahan energi tanpa pembatalan angka).

---

## 1. Elemen CST

**CST** = *Constant Strain Triangle*. Kain dipotong jadi segitiga kecil, dan di dalam satu segitiga regangan dianggap **konstan**.
Semua perhitungan dilakukan per segitiga, lalu dijumlahkan ke titik-titiknya.

```rust
pub struct MembraneElement {
    pub vertices: [usize; 3],
    pub grad_n: [[f64; 2]; 3],
    pub area: f64,
}
```

- `vertices`: nomor tiga titik.
- `grad_n[k]`: gradien **fungsi bentuk** titik k terhadap koordinat rest `(∂N_k/∂X_wale, ∂N_k/∂X_course)`.
- `area`: luas segitiga saat rest.

### Fungsi bentuk dalam satu kalimat

Di dalam segitiga, posisi titik mana pun adalah rata-rata tertimbang tiga sudutnya. Bobotnya (N₀, N₁, N₂) berubah linear di dalam segitiga.
`grad_n` memberi tahu seberapa cepat bobot itu berubah saat kita bergerak di arah wale atau course. Nilainya **hanya bergantung pada geometri
rest**, jadi dihitung sekali dan disimpan (di GPU: `elem_geom`, 7 angka per elemen).

```rust
pub fn new(vertices: [usize; 3], rest: &[[f64; 2]]) -> MembraneElement {
    let [p0, p1, p2] = vertices.map(|v| rest[v]);
    let dm = [[p1[0] - p0[0], p2[0] - p0[0]], [p1[1] - p0[1], p2[1] - p0[1]]];
    let det = dm[0][0] * dm[1][1] - dm[0][1] * dm[1][0];
    let g1 = [dm[1][1] / det, -dm[0][1] / det];
    let g2 = [-dm[1][0] / det, dm[0][0] / det];
    let g0 = [-g1[0] - g2[0], -g1[1] - g2[1]];
    MembraneElement { vertices, grad_n: [g0, g1, g2], area: 0.5 * det.abs() }
}
```

`dm` adalah matriks dua rusuk rest; `g1, g2` adalah baris-baris inversnya. `g0 = −g1 − g2` karena bobot selalu berjumlah 1.

## 2. Deformation gradient F

```rust
/// Deformation gradient columns (F_wale, F_course) from edge differences.
pub fn columns(&self, x: &[Vec3]) -> (Vec3, Vec3) {
    let [v0, v1, v2] = self.vertices;
    let (e1, e2) = (sub(x[v1], x[v0]), sub(x[v2], x[v0]));
    let [g1, g2] = [self.grad_n[1], self.grad_n[2]];
    let wale = add(scale(e1, g1[0]), scale(e2, g2[0]));
    let course = add(scale(e1, g1[1]), scale(e2, g2[1]));
    (wale, course)
}
```

F adalah matriks 3×2: dua kolom vektor 3D.

- `F_wale`: ke mana arah wale rest dipetakan sekarang (panjang dan arah).
- `F_course`: sama untuk course.

Kain belum ditarik → `F_wale = (1,0,0)`, `F_course = (0,1,0)`. Ditarik 20% di wale → `F_wale = (1,2, 0, 0)`.

Dihitung dari dua rusuk segitiga **sekarang** (`e1`, `e2`) dikombinasikan dengan `grad_n`. Rumus ini **linear** dalam posisi, sebuah fakta
penting di `energy_change`: `columns(d)` untuk vektor arah `d` memberi "perubahan F" per satuan langkah.

## 3. Green strain E

```rust
pub fn strain(&self, x: &[Vec3]) -> [f64; 3] {
    let (fw, fc) = self.columns(x);
    [0.5 * (dot(fw, fw) - 1.0), 0.5 * (dot(fc, fc) - 1.0), 0.5 * dot(fw, fc)]
}
```

`E = ½(FᵀF − I)` dalam tiga komponen:

- `E_ww = ½(|F_wale|² − 1)`: perubahan panjang kuadrat arah wale,
- `E_cc = ½(|F_course|² − 1)`,
- `E_wc = ½ F_wale·F_course`: perubahan sudut (0 kalau masih tegak lurus).

Rotasi tidak mengubah panjang maupun sudut, jadi E = 0. Tes `rigid_rotation_is_stress_free` memutar segitiga 53° plus translasi dan memeriksa E < 10⁻¹³.

## 4. Energi dan gaya

```rust
pub fn energy(&self, x: &[Vec3], material: &Material) -> f64 {
    self.area * material.energy_density(self.strain(x))
}

pub fn gradient(&self, x: &[Vec3], material: &Material) -> [Vec3; 3] {
    let (fw, fc) = self.columns(x);
    let [sww, scc, swc] = material.stress(self.strain(x));
    let pw = add(scale(fw, sww), scale(fc, swc));
    let pc = add(scale(fw, swc), scale(fc, scc));
    self.grad_n.map(|g| scale(add(scale(pw, g[0]), scale(pc, g[1])), self.area))
}
```

- Energi elemen = luas × energi per luas.
- Gradien = gaya dalam pada tiga titik. `P = F·S` (*first Piola–Kirchhoff stress*, kolom `pw`, `pc`) adalah gaya per satuan panjang rest.
  Gaya pada titik k = `(P · grad_n[k]) × area`.

Hasilnya `[Vec3; 3]`: satu vektor gaya 3D per titik. Jumlah ketiganya = 0 (gaya dalam elemen saling meniadakan; tidak ada gaya bersih dari
dalam).

## 5. Tangen eksak (baru di Step 2)

```rust
/// PSD-projected Hessian (negative stress clamped): the E1 linear-system matrices.
pub fn hessian(&self, x: &[Vec3], material: &Material) -> ElementHessian {
    self.hessian_with_stress(x, material, clamp_negative_stress(material.stress(self.strain(x))))
}

/// Exact tangent ∂²W/∂x², the matrix Newton needs for quadratic convergence.
pub fn tangent(&self, x: &[Vec3], material: &Material) -> ElementHessian {
    self.hessian_with_stress(x, material, material.stress(self.strain(x)))
}
```

Dua versi Hessian, satu implementasi:

| | `hessian` (Step 1) | `tangent` (Step 2) |
|---|---|---|
| Tegangan negatif (kompresi) | dipotong ke nol (proyeksi PSD) | dipakai apa adanya |
| Sifat | selalu positif semidefinit | turunan kedua **eksak** |
| Dipakai untuk | matriks uji E1 | Newton E2 |

Kenapa Newton butuh yang eksak? Konvergensi kuadratik hanya terjadi kalau `K` adalah turunan sebenarnya dari `g`. Hessian yang dimodifikasi
memberi konvergensi **linear** di dekat jawaban.

Refactor-nya: kode lama `hessian` dipindah ke `hessian_with_stress(x, m, stress)`, dan dua metode publik hanya berbeda di tegangan yang dikirim.

### Isi satu blok (k, l)

```rust
fn block(&self, k: usize, l: usize, rows: &[[Vec3; 3]; 3], stress: [f64; 3], m: &Material) -> Mat3 {
    let ([wk, ck, sk], [wl, cl, sl]) = (rows[k], rows[l]);
    let mut block = identity_scaled(geometric_factor(self.grad_n[k], self.grad_n[l], stress));
    mat_add_scaled(&mut block, &outer(wk, wl), m.wale);
    mat_add_scaled(&mut block, &outer(ck, cl), m.course);
    mat_add_scaled(&mut block, &outer(wk, cl), m.coupling);
    mat_add_scaled(&mut block, &outer(ck, wl), m.coupling);
    mat_add_scaled(&mut block, &outer(sk, sl), 4.0 * m.shear);
    block.map(|row| row.map(|v| v * self.area))
}
```

Dua bagian:

1. **Kekakuan geometrik** `(gkᵀ S gl) · I`: tali yang tegang menolak dibelokkan. Bisa **negatif** kalau S negatif (kompresi).
2. **Kekakuan material**: `bᵢᵀ C bⱼ`, dengan `strain_vectors` (`wk, ck, sk`) = turunan E terhadap posisi titik k.

Total 9 blok 3×3 per elemen = 81 angka. Di GPU, `membrane_element.wgsl` menulis 81 angka ini per elemen ke `elem_hess`.

## 6. Perubahan energi tanpa pembatalan (baru di Step 2)

```rust
/// W(x + αd) − W(x), exact for St. Venant–Kirchhoff (W is quadratic in E) and free of the
/// cancellation in subtracting two large energies: S(E):ΔE + W(ΔE).
pub fn energy_change(&self, x: &[Vec3], d: &[Vec3], alpha: f64, material: &Material) -> f64 {
    let (fw, fc) = self.columns(x);
    let (dw, dc) = self.columns(d);
    let (dw, dc) = (scale(dw, alpha), scale(dc, alpha));
    let change = [
        dot(fw, dw) + 0.5 * dot(dw, dw),
        dot(fc, dc) + 0.5 * dot(dc, dc),
        0.5 * (dot(fw, dc) + dot(dw, fc) + dot(dw, dc)),
    ];
    let [sww, scc, swc] = material.stress(self.strain(x));
    let first = sww * change[0] + scc * change[1] + 2.0 * swc * change[2];
    self.area * (first + material.energy_density(change))
}
```

### Masalah yang dipecahkan

Line search Armijo butuh `Π(x + αd) − Π(x)`. Cara naif menghitung dua energi lalu mengurangkan. Di dekat jawaban, energi total misalnya 0,05 J,
sedangkan perubahannya 10⁻¹² J. Dalam f32 (7 digit), `0,0500000 − 0,0500000 = 0`. Perubahannya **tenggelam dalam pembulatan**, sehingga
Armijo memutuskan secara acak.

Analogi: menghitung selisih saldo dua akun miliaran rupiah dengan kalkulator 7 digit. Hitung **selisih transaksinya** langsung, jangan selisih
saldonya.

### Derivasi singkat

Karena F linear dalam x: `F(x + αd) = F + ΔF` dengan `ΔF = α·columns(d)`.

```
ΔE_ww = ½(|F_w + ΔF_w|² − |F_w|²) = F_w·ΔF_w + ½|ΔF_w|²
ΔE_cc = F_c·ΔF_c + ½|ΔF_c|²
ΔE_wc = ½(F_w·ΔF_c + ΔF_w·F_c + ΔF_w·ΔF_c)
```

Lalu identitas eksak StVK: `ΔW = S(E):ΔE + W(ΔE)`, dengan `S:ΔE = S_ww ΔE_ww + S_cc ΔE_cc + 2 S_wc ΔE_wc` (faktor 2 geser dari dokumen 2.1).

Ini **bukan aproksimasi**: untuk StVK hasilnya sama persis dengan selisih langsung dalam aritmetika eksak. Bedanya, tidak ada pengurangan
dua angka besar yang hampir sama.

## 7. Tes

| Tes | Memeriksa | Toleransi |
|---|---|---|
| `rest_state_is_stress_free` | gaya nol saat rest | 10⁻¹⁴ |
| `gradient_matches_energy_finite_difference` | gradien = turunan numerik energi | 10⁻⁶ relatif |
| `hessian_matches_gradient_finite_difference_in_tension` | Hessian = turunan numerik gradien (tarik) | 10⁻⁵ |
| `exact_tangent_matches_gradient_finite_difference_in_compression` **(baru)** | tangen eksak juga benar saat **kompresi** (0,93×), kondisi di mana Hessian terpotong akan salah | 10⁻⁵ |
| `energy_change_equals_energy_difference` **(baru)** | rumus ΔW = selisih langsung dalam f64 | 10⁻¹² |
| `rigid_rotation_is_stress_free` | E = 0 untuk rotasi | 10⁻¹³ |

"Finite difference" (beda hingga): geser titik sedikit (h = 10⁻⁸) ke depan dan ke belakang, lalu ukur perubahan fungsi: `(f(x+h) − f(x−h)) / 2h`.
Kalau turunan analitis benar, keduanya cocok sampai galat pembulatan. Ini cara standar memverifikasi turunan yang ditulis tangan.

## 8. Ringkasan

| Metode | Rumus | Kembaran GPU |
|---|---|---|
| `columns` | F dari rusuk × grad_n | `columns_of` |
| `strain` | E = ½(FᵀF − I) | `green_strain` |
| `gradient` | (F S · grad_n) × area | `write_gradient` |
| `tangent` | geometrik + material, tegangan eksak | `write_block` |
| `energy_change` | S:ΔE + W(ΔE) | `membrane_energy_change.wgsl` |
