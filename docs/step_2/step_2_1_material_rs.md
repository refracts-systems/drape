# Step 2.1: `oracle/src/material.rs`, Hukum Material Ortotropik St. Venant–Kirchhoff

> File ini sudah ada sejak Step 1 dan tidak diubah di Step 2. Ia dipakai ulang sesuai permintaan ("reuse material.rs").
> Dokumen ini menjelaskan apa arti setiap angka dan rumusnya.

---

## 1. Isi

```rust
/// Orthotropic St. Venant–Kirchhoff membrane in material axes (wale = x, course = y),
/// plus directional hinge bending. Units: N/m for membrane, N·m for bending, kg/m² density.
pub struct Material {
    pub wale: f64,
    pub course: f64,
    pub coupling: f64,
    pub shear: f64,
    pub bend_wale: f64,
    pub bend_course: f64,
    pub density: f64,
}

pub fn knit_baseline() -> Material {
    Material { wale: 1000.0, course: 300.0, coupling: 100.0, shear: 50.0, bend_wale: 2.0e-6, bend_course: 1.0e-6, density: 0.13 }
}
```

## 2. Ortotropik: kain punya arah

Kain rajut tidak sama kaku ke semua arah:

- **Wale**: arah "kolom" jeratan rajut (sumbu x di sini). Paling kaku: 1000 N/m.
- **Course**: arah "baris" jeratan (sumbu y). Lebih lunak: 300 N/m.
- **Shear**: tahanan terhadap geser (persegi jadi jajaran genjang): 50 N/m, sangat lunak, ciri khas kain rajut.
- **Coupling**: menarik satu arah juga memengaruhi arah lain (seperti efek Poisson): 100 N/m.

Satuan N/m karena ini **membran**: kekakuan per satuan lebar kain, bukan per satuan luas penampang (kain dianggap sangat tipis).

"Ortotropik" = sifatnya berbeda sepanjang dua arah tegak lurus. Untuk backend developer: seperti tabel dengan dua index yang punya
karakteristik performa berbeda.

## 3. Hukum material: dari regangan ke tegangan

```rust
/// Second Piola–Kirchhoff stress (S_ww, S_cc, S_wc) from Green strain (E_ww, E_cc, E_wc).
pub fn stress(&self, strain: [f64; 3]) -> [f64; 3] {
    let [eww, ecc, ewc] = strain;
    [
        self.wale * eww + self.coupling * ecc,
        self.course * ecc + self.coupling * eww,
        2.0 * self.shear * ewc,
    ]
}
```

Dalam bentuk matriks:

```
| S_ww |   | wale      coupling   0       | | E_ww |
| S_cc | = | coupling  course     0       | | E_cc |
| S_wc |   | 0         0          2·shear | | E_wc |
```

**St. Venant–Kirchhoff (StVK)**: tegangan **linear** terhadap Green strain. Ini model nonlinear paling sederhana: nonlinearitasnya hanya
datang dari geometri (E kuadratik terhadap posisi), bukan dari materialnya. Cocok untuk eksperimen "simple elastic" sesuai permintaan.

Notasi tiga komponen (Voigt):

- `E_ww`: regangan arah wale,
- `E_cc`: regangan arah course,
- `E_wc`: regangan geser (komponen tensor; ia muncul dua kali di tensor 2×2 yang simetris).

## 4. Energi

```rust
pub fn energy_density(&self, strain: [f64; 3]) -> f64 {
    let [eww, ecc, ewc] = strain;
    0.5 * self.wale * eww * eww
        + 0.5 * self.course * ecc * ecc
        + self.coupling * eww * ecc
        + 2.0 * self.shear * ewc * ewc
}
```

Energi per satuan luas rest = `½ E : C : E` (kuadratik dalam E). Tegangan adalah turunannya: `∂W/∂E_ww = wale·E_ww + coupling·E_cc = S_ww`.

**Detail yang penting di Step 2**: `∂W/∂E_wc = 4·shear·E_wc = 2·S_wc`. Faktor 2 ini muncul karena komponen geser muncul dua kali di tensor.
Ia muncul lagi di rumus perubahan energi (`2.0 * swc * change[2]`), dan kalau terlewat, rumus perubahan energi salah.

Karena W **kuadratik tepat** dalam E, berlaku identitas eksak:

```
W(E + ΔE) = W(E) + S(E):ΔE + W(ΔE)
```

Identitas inilah dasar `energy_change` yang bebas pembatalan angka (dokumen 2.2 dan 2.9).

## 5. Konveksitas

```rust
pub fn is_convex(&self) -> bool {
    self.coupling * self.coupling < self.wale * self.course && self.shear > 0.0
}
```

Matriks material positif definit ⇔ coupling² < wale × course (100² = 10⁴ < 3×10⁵ ✓) dan shear > 0. Tanpa ini, ada arah regangan
dengan energi negatif dan material tidak stabil secara fisik.

Catatan: StVK konveks dalam **E**, tetapi **tidak** konveks dalam posisi `x` di bawah tekanan (kompresi) besar. Ini alasan tangen pertama
di Step 2 bisa tidak positif definit kalau elemen terkompresi/terbalik (dokumen 2.3).

## 6. Tekukan dan massa

`bend_wale`, `bend_course`, `hinge_bending`, dan `density` dipakai di Step 1 (engsel dan inersia). Kupon Step 2 adalah membran **di bidang**
(z dikunci), jadi tekukan dan massa tidak dipakai.

## 7. Versi GPU

`membrane.wgsl` (dokumen 2.7) menulis ulang `stress_of` dan `energy_density` dalam f32 dengan urutan operasi yang sama. Modulus dikirim sebagai
`override` (`WALE`, `COURSE`, `COUPLING`, `SHEAR`) dari `problem.json`, sehingga satu-satunya sumber angka material tetap `knit_baseline()`.

## 8. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Ortotropik | kekakuan berbeda di arah wale dan course |
| StVK | S linear terhadap E; nonlinear hanya karena geometri |
| Energi kuadratik | identitas W(E+ΔE) eksak → perubahan energi tanpa pembatalan |
| Faktor 2 geser | ∂W/∂E_wc = 2·S_wc |
| Konveks | dalam E ya; dalam x tidak selalu (kompresi) |
