# Step 3.4: `oracle/src/drape.rs`, Kain yang Jatuh: Backward Euler sebagai Minimisasi Energi

> File inti fisika Step 3. Ia mendefinisikan benchmark (kain 0,5 m dipaku di dua sudut atas, jatuh karena gravitasi), langkah waktu implisit,
> dan mencatat **trigger berbasis bukti** yang terjadi: tangen eksak kehilangan positif definit, sehingga dipakai tangen proyeksi.

---

## 1. Benchmark

```rust
pub fn build(columns: usize, tangent_kind: TangentKind) -> Drape {
    let mesh = Mesh::grid(columns, columns, SHEET_SIZE, SHEET_SIZE);            // 0,5 m × 0,5 m, datar di bidang xy
    let material = Material::knit_baseline();                                   // wale = x
    let fixed = vec![(mesh.rows - 1) * mesh.columns, mesh.rows * mesh.columns - 1];   // dua sudut atas (y maksimum)
    ...
    let hinges = hinge_elements(&mesh, &material);                              // dipakai ulang dari cases.rs (Step 1)
    let groups = ...segitiga dan hinge...;
    let pattern = Bsr::with_pattern(free.len(), &groups);
    let mass = mesh.vertex_areas().iter().map(|a| material.density * a).collect();   // massa tergumpal
    ...
}
```

- Kain mula-mula **horizontal** dan diam, lalu dilepas. Gravitasi menariknya ke −z. Kain berayun ke bawah sampai menggantung, berkerut, dan bergoyang.
- **Semua tiga komponen** setiap titik bebas adalah unknown (tidak ada masking z seperti kupon E2).
- `hinge_elements` di `cases.rs` diubah dari privat ke `pub`. Itu satu-satunya perubahan di file Step 1 itu.
- **Massa tergumpal** (*lumped mass*): setiap titik membawa massa = densitas × luas yang diwakilinya. Matriks massa jadi diagonal.

## 2. Backward Euler sebagai minimisasi

Langkah waktu implisit (backward Euler) dengan h = 1/240 s bisa ditulis sebagai **mencari minimum** satu fungsi energi per langkah:

```
Φ(x) = 1/(2h²) · (x − x̂)ᵀ M (x − x̂) + Π_membran(x) + Π_hinge(x)
x̂   = x_n + h·v_n + h²·g        (posisi "seandainya tidak ada gaya dalam")
v_{n+1} = (x_{n+1} − x_n) / h
```

Intuisinya: suku pertama adalah **inersia**. Ia "menghukum" penyimpangan dari lintasan bebas `x̂` (titik melanjutkan kecepatannya dan jatuh karena gravitasi).
Suku kedua dan ketiga adalah energi elastis. Minimum Φ adalah kompromi antara "ingin terus bergerak" dan "ingin tidak teregang".

Keuntungannya, Newton + Armijo dari Step 2 bisa dipakai apa adanya: setiap langkah waktu adalah satu soal minimisasi.

```rust
/// x̂ = x + h·v + h²·g on free vertices; pinned vertices stay where they are.
pub fn prediction(&self, x: &[Vec3], v: &[Vec3]) -> Vec<Vec3> { ... }
```

## 3. `DrapeStep`: satu langkah sebagai `NewtonSystem`

```rust
pub struct DrapeStep<'a> {
    pub drape: &'a Drape,
    pub x_hat: Vec<Vec3>,
}
```

`'a` adalah **lifetime** Rust: `DrapeStep` meminjam `Drape` (tidak memilikinya) dan tidak boleh hidup lebih lama darinya. Analogi: request context yang
memegang referensi ke config global.

### Residual

```rust
fn residual(&self, x: &[Vec3]) -> Vec<f64> {
    let gradient = self.drape.potential_gradient(x);
    ... m/h² · (x − x̂) + ∇Π, per komponen titik bebas ...
}
```

### Tangen: kenapa ada dua jenis

```rust
/// Newton tangent. `Projected` (clamped membrane stress + Gauss–Newton hinge) replaced `Exact` after the band Cholesky
/// reported a non-positive-definite exact tangent at step 87 of the n33 drape: the evidence-triggered change.
pub enum TangentKind {
    Exact,
    Projected,
}
```

```rust
fn tangent(&self, x: &[Vec3]) -> Bsr {
    ... M/h² di diagonal ...
    drape.membranes.iter().for_each(|e| drape.add_blocks(&mut matrix, &e.vertices, &drape.membrane_hessian(e, x)));
    drape.hinges.iter().for_each(|h| drape.add_blocks(&mut matrix, &h.vertices, &drape.hinge_hessian(h, x)));
    matrix
}
```

**Kronologi trigger**:

1. Prompt meminta tangen = M/h² + tangen membran eksak + Hessian hinge eksak.
2. Lintasan f64 n33 dengan tangen eksak berjalan baik sampai langkah 87, lalu band Cholesky melaporkan:
   `matrix not positive definite at row 47 (pivot -1.2085600732858737e-1)`.
   Kain sudah jatuh ~0,5 m dan mulai terkompresi. StVK di bawah kompresi punya kekakuan geometrik negatif, dan suku M/h² tidak cukup besar untuk menutupinya.
3. Prompt sudah menetapkan aturan di depan: kalau ini terjadi, pakai Hessian membran **terjepit** (`MembraneElement::hessian`, tegangan negatif dipotong)
   ditambah Hessian hinge **Gauss–Newton**, untuk oracle **dan** GPU.
4. Efek yang terukur: Newton tidak lagi kuadratik. Di fase kompresi banyak langkah mentok di 30 iterasi, dengan residual sampai 1,1×10⁻².

`add_blocks` memakai **const generics** (`const N: usize`) supaya satu fungsi melayani blok membran 3×3 (N = 3) dan hinge 4×4 (N = 4).

### Perubahan energi

```rust
/// Inertia (x − x̂)ᵀM(αd) + ½α²dᵀMd, plus per-element membrane and hinge changes.
fn energy_change(&self, x: &[Vec3], d: &[Vec3], alpha: f64) -> f64 { ... }
```

Ketiga bagian dihitung tanpa mengurangkan dua energi global. Inersia: ½m|x + αd − x̂|² − ½m|x − x̂|² = m((x−x̂)·αd + ½|αd|²), eksak.

## 4. Energi total untuk diagnosis

```rust
/// Kinetic + elastic + gravitational energy (z up, pinned vertices at rest).
pub fn total_energy(&self, x: &[Vec3], v: &[Vec3]) -> f64 { ½Σm|v|² + Σ m g z + Π(x) }
```

Backward Euler bersifat **disipatif**: energi total tidak boleh naik dari langkah ke langkah. Ini gate fisika di Step 3.

## 5. Simulasi

```rust
pub fn time_step(drape: &Drape, x: &[Vec3], v: &[Vec3], settings: &NewtonSettings) -> Result<TimeStep, String> {
    let step = DrapeStep { drape, x_hat: drape.prediction(x, v) };
    let scale = vector_norm(&step.residual(x));
    let solution = solve_step(&step, step.x_hat.clone(), scale, settings)?;
    let velocity = ... (x_{n+1} − x_n)/h ...;
    Ok(TimeStep { solution, scale, velocity })
}
```

- Skala residual = ‖∇Φ(x_n)‖: gaya tak seimbang **sebelum** langkah, sesuai prompt.
- Newton mulai dari x̂, sesuai prompt.
- `simulate` mengulang 240 kali dan menyimpan posisi, kecepatan, dan catatan Newton setiap langkah.

## 6. Tes

| Tes | Memeriksa |
|---|---|
| `residual_is_the_gradient_of_the_incremental_potential` | residual = turunan FD dari Φ (inersia + membran + hinge) |
| `first_step_falls_by_h_squared_g_and_energy_decreases` | langkah pertama: titik tengah jatuh tepat h²g; energi turun di 3 langkah pertama |

## 7. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Benchmark | kain 0,5 m, dua sudut atas dipaku, jatuh dari horizontal |
| Backward Euler | minimisasi Φ = inersia + elastis per langkah |
| Massa tergumpal | matriks massa diagonal |
| Trigger | tangen eksak tidak PD di langkah 87 → tangen proyeksi |
| Efek trigger | Newton tidak kuadratik; banyak langkah mentok 30 iterasi |
| Energi total | harus tidak naik (disipatif) |
