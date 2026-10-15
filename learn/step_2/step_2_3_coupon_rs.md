# Step 2.3: `oracle/src/coupon.rs`, Kupon Uji, Penjepit, Prediktor, dan Skala Residual

> File baru Step 2. Ia mendefinisikan **soal**: benda uji, cara menjepit, jalur beban, dan semua operasi global
> (energi total, gaya total, tangen global). File ini juga mencatat tiga keputusan desain yang dipaksa oleh hasil pengukuran.

---

## 1. Konstanta

```rust
pub const COUPON_SIZE: f64 = 0.1;
/// In-plane coupon: every vertex keeps z = 0, so only the first two components are unknowns.
pub const FREE_COMPONENTS: [bool; 3] = [true, true, false];
const LOAD_STEPS: usize = 10;
const PATCH_JITTER: f64 = 0.2;
const PATCH_START_PERTURBATION: f64 = 0.1;
```

- Kupon 10 cm.
- **Di bidang**: z dikunci nol. Membran tanpa tekukan tidak punya kekakuan keluar bidang saat datar; tanpa mengunci z, sistemnya singular.
- Format BSR tetap 3×3 (supaya solver E1 dipakai apa adanya); komponen z diperlakukan sebagai "terkunci" lewat **masking** (bagian 6).

## 2. Jalur beban

```rust
pub fn boundary_gradient(&self, t: f64) -> [[f64; 2]; 2] {
    match self {
        LoadPath::Uniaxial => [[1.0 + 0.2 * t, 0.0], [0.0, 1.0]],
        LoadPath::Biaxial => [[1.0 + 0.15 * t, 0.0], [0.0, 1.0 + 0.075 * t]],
        LoadPath::Patch => [[1.0 + 0.1 * t, 0.05 * t], [0.02 * t, 1.0 - 0.05 * t]],
    }
}

pub fn load_factors(&self) -> Vec<f64> {
    match self {
        LoadPath::Patch => vec![1.0],
        _ => (1..=LOAD_STEPS).chain((0..LOAD_STEPS).rev()).map(|k| k as f64 / LOAD_STEPS as f64).collect(),
    }
}
```

- Faktor beban `t` dari 0 ke 1. Setiap jalur punya **gradien deformasi batas** 2×2 `F(t)`: posisi penjepit = `F(t) · X_rest`.
- Uniaxial: 1,0 → 1,2 di x.
- Biaxial: 1,15 × 1,075.
- Patch: termasuk geser (komponen di luar diagonal), di arah course **dikompresi** 5%.
- Faktor: `0.1, 0.2, …, 1.0, 0.9, …, 0.0` = 20 langkah (naik lalu turun ke rest). Patch: satu langkah.

## 3. Membangun kupon

```rust
pub fn build(path: LoadPath, columns: usize) -> Coupon {
    let mut mesh = Mesh::grid(columns, columns, COUPON_SIZE, COUPON_SIZE);
    if path == LoadPath::Patch {
        jitter_interior(&mut mesh);
    }
    let fixed: Vec<usize> = (0..mesh.rest.len()).filter(|&v| is_fixed(&mesh, path, v)).collect();
    let free: Vec<usize> = (0..mesh.rest.len()).filter(|v| !fixed.contains(v)).collect();
    let mut row_of = vec![None; mesh.rest.len()];
    free.iter().enumerate().for_each(|(row, &v)| row_of[v] = Some(row));
    let elements = ...MembraneElement::new...;
    let groups = ...segitiga → baris...;
    let pattern = Bsr::with_pattern(free.len(), &groups);
    Coupon { path, mesh, material: Material::knit_baseline(), elements, fixed, row_of, free, pattern }
}
```

- `fixed`: titik penjepit (posisinya ditentukan, bukan dicari).
- `free`: titik yang dicari, masing-masing jadi satu baris BSR.
- `row_of`: lookup `vertex → row`.
- `pattern`: pola BSR dari segitiga (tanpa engsel, karena tidak ada tekukan).

### Patch: mesh diacak

Patch test klasik memakai mesh **tidak beraturan** supaya kecocokan bukan karena kebetulan simetri. Titik interior digeser acak ±20% jarak grid
(deterministik dengan `Lcg::new(2024)`).

## 4. Penjepit

```rust
fn is_fixed(mesh: &Mesh, path: LoadPath, vertex: usize) -> bool {
    let (column, row) = (vertex % mesh.columns, vertex / mesh.columns);
    match path {
        LoadPath::Uniaxial => column == 0 || column == mesh.columns - 1,
        LoadPath::Biaxial => on_side_jaw(mesh, column, row),
        LoadPath::Patch => mesh.is_boundary(vertex),
    }
}

/// A vertex on an edge lies under a jaw when its position along that edge is in the middle half.
fn on_side_jaw(mesh: &Mesh, column: usize, row: usize) -> bool {
    let middle = |i: usize, n: usize| 4 * i >= n - 1 && 4 * i <= 3 * (n - 1);
    ...
}
```

| Jalur | Dijepit | Bebas |
|---|---|---|
| Uniaxial | seluruh tepi kiri dan kanan | tepi atas/bawah dan interior |
| Biaxial | **setengah tengah** setiap tepi (rahang) | ujung-ujung tepi dan interior |
| Patch | seluruh tepi | interior |

### Keputusan desain #1: biaxial dengan rahang

Rencana awal: biaxial dijepit di **semua** tepi. Masalahnya, batas afin di semua tepi → solusi eksaknya peta afin itu sendiri (itulah patch test!).
Medannya homogen, dan dengan prediktor apa pun Newton tidak punya pekerjaan. Biaxial akan jadi duplikat patch test.

Solusi: rahang hanya di tengah setiap tepi, seperti alat uji biaxial sungguhan yang memakai beberapa klem dengan celah. Medannya jadi tidak homogen.

`middle(i, n)`: `n/4 ≤ i ≤ 3n/4` ditulis dengan integer (`4i ≥ n−1` dst.) untuk menghindari float.

## 5. Operasi global

```rust
pub fn energy(&self, x: &[Vec3]) -> f64 { Σ element.energy }
pub fn energy_change(&self, x, d, alpha) -> f64 { Σ element.energy_change }

/// ∂Π/∂x for every vertex, fixed ones included (their entries are the grip reactions).
pub fn vertex_gradient(&self, x: &[Vec3]) -> Vec<Vec3> {
    let mut gradient = vec![[0.0; 3]; x.len()];
    for element in &self.elements {
        let g = element.gradient(x, &self.material);
        for (k, &v) in element.vertices.iter().enumerate() {
            (0..3).for_each(|d| gradient[v][d] += g[k][d]);
        }
    }
    gradient
}

/// Nonlinear residual on the unknowns: ∂Π/∂x of free vertices with constrained components zeroed.
pub fn free_gradient(&self, x: &[Vec3]) -> Vec<f64> { ... mask_vector(gradient[v]) ... }
```

**Perakitan** (*assembly*): setiap elemen menyumbang gaya ke tiga titiknya (`+=`). Gaya di titik **bebas** adalah **residual nonlinear** (harus 0 saat
setimbang). Gaya di titik **jepit** adalah **reaksi**: gaya yang harus diberikan penjepit, dan itulah yang diukur mesin uji tarik.

## 6. Masking komponen z

```rust
fn mask_matrix(matrix: &mut Bsr) {
    for row in 0..matrix.rows() {
        for k in matrix.row_ptr[row]..matrix.row_ptr[row + 1] {
            let diagonal = matrix.col_idx[k] == row;
            for (i, j) in ... {
                if !(FREE_COMPONENTS[i] && FREE_COMPONENTS[j]) {
                    matrix.vals[9 * k + 3 * i + j] = if diagonal && i == j { 1.0 } else { 0.0 };
                }
            }
        }
    }
}
```

Untuk setiap entri yang menyentuh komponen terkunci: nol, kecuali di diagonal yang diberi **1**. Hasilnya, persamaan z menjadi `1 · d_z = 0`, terlepas
dari yang lain. Gradien z juga dinolkan. PCG lalu otomatis menghasilkan `d_z = 0` persis.

Trik standar FEM untuk syarat batas: lebih sederhana daripada membuang baris/kolom (yang akan mengubah format BSR 3×3).

## 7. Tegangan penjepit

```rust
/// Nominal (first Piola) grip stresses in N/m: right-edge x reaction per rest height, top-edge y reaction per rest width.
pub fn grip_stress(&self, x: &[Vec3]) -> [f64; 2] { ... }
```

Jumlah reaksi x di tepi kanan dibagi tinggi rest = tegangan nominal `P_xx` (N/m), angka yang dicatat mesin uji untuk kurva tegangan-regangan.

## 8. Keputusan desain #2: prediktor

**Gejala** saat pertama dijalankan:

```
error: matrix not positive definite at row 466 (pivot -2.731863400838309e2)
```

Cholesky menolak tangen pertama. Penyebab: awalnya setiap langkah hanya **memindahkan penjepit**, sedangkan interior tetap di posisi lama.
Pada uniaxial langkah pertama, penjepit kanan bergeser 2 mm, sedangkan elemen berukuran 3,1 mm. Elemen di sebelah penjepit melar ~64%,
bahkan bisa terbalik pada patch. StVK di bawah kompresi/inversi besar punya tangen tidak positif definit.

**Solusi**:

```rust
/// Load-step predictor: adds the affine increment (F(to) − F(from))·X to every vertex, grips exact.
pub fn predict(&self, x: &mut [Vec3], from: f64, to: f64) {
    let (a, b) = (self.path.boundary_gradient(from), self.path.boundary_gradient(to));
    for (p, rest) in x.iter_mut().zip(&self.mesh.rest) {
        (0..2).for_each(|r| p[r] += (b[r][0] - a[r][0]) * rest[0] + (b[r][1] - a[r][1]) * rest[1]);
    }
    self.apply_boundary(x, to);
}
```

Setiap langkah dimulai dari kesetimbangan sebelumnya **ditambah** kenaikan regangan afin pada **semua** titik, sehingga seluruh kupon ikut meregang
sedikit, bukan hanya tepinya. Setelah ini, f64 Newton konvergen di semua langkah, dan line search tidak pernah perlu memotong langkah.

## 9. Keputusan desain #3: skala residual

```rust
/// Out-of-balance force created by moving only the grips: the scale of a load step's nonlinear residual.
pub fn step_scale(&self, previous: &[Vec3], t: f64) -> f64 {
    let mut gripped = previous.to_vec();
    self.apply_boundary(&mut gripped, t);
    crate::linalg::vector_norm(&self.free_gradient(&gripped))
}
```

"Residual relatif 10⁻⁶" harus relatif terhadap **sesuatu**. Pilihan yang ditolak:

- `‖g(x₀)‖` setelah prediktor: untuk medan yang hampir afin, prediktor hampir tepat, sehingga ‖g(x₀)‖ ≈ derau pembulatan dan pembaginya tidak bermakna.
- Gaya total / reaksi: nol di langkah terakhir (rest).

Yang dipilih: gaya tak seimbang yang muncul kalau **hanya penjepit** digeser dari kesetimbangan sebelumnya. Ini peran `‖b‖` di sistem linear:
"seberapa besar beban langkah ini". Tidak pernah nol (selama beban berubah) dan punya makna fisik.

Ini definisi yang **ketat**: kenaikan beban per langkah kecil (2% regangan), sehingga skalanya kecil dibanding gaya dalam total. Konsekuensinya
dibahas di dokumen 2.17.

## 10. Patch: titik awal diganggu

```rust
/// State before the first load step: the rest state, with the interior perturbed for the patch test.
pub fn initial_positions(&self) -> Vec<Vec3> { ... PATCH_START_PERTURBATION * spacing * rng.uniform() ... }
```

Dengan prediktor, patch test akan trivial (prediktor = peta afin = jawaban). Karena itu interiornya **diganggu** ±10% jarak grid sebelum langkah.
Newton harus bekerja untuk kembali ke peta afin: 9 iterasi di f64.

## 11. Ringkasan

| Bagian | Keputusan |
|---|---|
| Di bidang | z dikunci lewat masking, format BSR 3×3 tetap |
| Jalur | 10 naik + 10 turun; patch 1 langkah dengan geser |
| Biaxial | rahang setengah tengah tepi (non-homogen) |
| Prediktor | kenaikan afin ke semua titik (mencegah tangen tak definit) |
| Skala | gaya tak seimbang dari pergeseran penjepit |
| Patch | mesh diacak + titik awal diganggu |
