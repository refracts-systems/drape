# Step 5.5: `oracle/src/drape.rs`, Scene, Barrier di Φ, dan Titik Awal Newton yang Aman

> `Drape` dari Step 3 sekarang bisa menjadi dua **scene**: kain yang digantung di dua sudut (E3/E4), atau kain bebas yang dijatuhkan ke bola di atas
> lantai (E5). Barrier masuk ke energi langkah Φ, dan Newton mendapat titik awal yang dijamin tidak menembus collider.

---

## 1. Scene

```rust
pub enum Scene {
    PinnedCorners,
    SphereOnFloor,
}
```

| | PinnedCorners (E3/E4) | SphereOnFloor (E5) |
|---|---|---|
| vertex terkunci | 2 sudut atas | tidak ada |
| collider | tidak ada | lantai z = 0 + bola |
| tinggi awal (`lift`) | 0 | 2r + 0,02·L = 0,26 m |

Ukuran E5, semuanya dalam satuan sisi kain L = 0,5 m:

```rust
const SPHERE_RADIUS: f64 = 0.25 * SHEET_SIZE;                        // r = 0,125 m
pub const ACTIVATION_DISTANCE: f64 = 1e-3 * SHEET_SIZE;              // d̂ = 0,5 mm
const RELEASE_HEIGHT: f64 = 2.0 * SPHERE_RADIUS + 0.02 * SHEET_SIZE; // 0,26 m
```

Pusat bola (0,25; 0,25; 0,125): di bawah pusat kain, menempel di lantai. Ketiga koordinatnya **tepat terwakili di f32**, jadi GPU dan oracle memakai
bola yang persis sama.

`Scene::parse` membaca kunci `"scene"` dari `problem.json`, dan file lama yang tidak punya kunci itu dianggap `PinnedCorners`. Dengan
`Drape::from_problem`, semua penilai (E3, E4, E5) membangun drape yang benar hanya dari `problem.json`.

## 2. `for_scene` dan `unknown_layout`

`Drape::build(columns, tangent)` (dipakai E3/E4) sekarang hanya memanggil `for_scene(…, PinnedCorners)`. Pembangunannya dipecah supaya tiap fungsi di
bawah 20 baris:

- `unknown_layout(mesh, fixed)`: vertex bebas, peta vertex → baris unknown, dan pola BSR;
- `scene_contact(scene, mass)`: collider, barrier (κ dari massa vertex terberat), dan tinggi awal.

## 3. Barrier di setiap besaran Φ

Φ(x) = 1/(2h²)(x − x̂)ᵀM(x − x̂) + Π_membran + Π_hinge + **Σ_vertex Σ_collider b(d)**.

| Besaran | Tambahan kontak | Dijaga oleh |
|---|---|---|
| `potential` | `barrier_energy(x)` | `if self.has_contact()` |
| `potential_gradient` | `contact.gradient(x_i)` per vertex | `if self.has_contact()` |
| `tangent` | `contact.hessian(x_i)` di blok diagonal | `if drape.has_contact()` |
| `energy_change` | `barrier_change(x, d, α)` | `if drape.has_contact()` |
| `max_step` | min α_max atas vertex | tanpa collider = 1 |
| `absolute_contributions` | \|gradien barrier\| | tanpa collider = +0 |

Penjaga `has_contact()` memastikan E3/E4 menjalankan **kode yang persis sama** seperti sebelumnya, bukan sekadar menambah nol.

### Perubahan energi barrier

```rust
/// Σ over free vertices of b(d(x + αd)) − b(d(x)), per vertex as the WGSL kernel forms it.
pub fn barrier_change(&self, x: &[Vec3], d: &[Vec3], alpha: f64) -> f64 {
    let contact = &self.drape.contact;
    self.drape.free.iter().map(|&i| contact.energy(std::array::from_fn(|k| x[i][k] + alpha * d[i][k])) - contact.energy(x[i])).sum()
}
```

Untuk membran dan hinge, Step 2–3 menulis perubahan energi dalam bentuk yang bebas cancellation (selisih dihitung langsung dari regangan). Untuk barrier,
prompt menetapkan bentuk sederhana b(baru) − b(lama) per vertex, dihitung sama di oracle dan di kernel. Akibatnya di f32 dibahas di dokumen 5.11.

## 4. Titik awal Newton yang aman

Di uji coba pertama (n17) muncul masalah nyata. Newton E3/E4 mulai dari prediksi x̂ = x + h·v + h²·g. Tetapi x̂ tidak tahu ada bola: kalau kain jatuh
cukup cepat, sebagian vertex x̂ sudah **di dalam** bola. Di sana ln(d/d̂) dengan d ≤ 0 menghasilkan NaN, sehingga Newton berhenti di iterasi 0 dan kain
"menembus" bola.

Metode barrier mensyaratkan titik awal yang **feasible** (di luar semua collider). Solusinya memakai aturan yang sama dengan line search:

```rust
/// With colliders, Newton starts from x_n + α_max·(x̂ − x_n): the prediction cut back by the line-search rule so every
/// vertex keeps at least 0.1 of its distance (x̂ itself may lie inside a collider).
pub fn feasible_start(step: &DrapeStep, x: &[Vec3]) -> Vec<Vec3> {
    let towards: Vec<Vec3> = step.x_hat.iter().zip(x).map(|(p, q)| std::array::from_fn(|k| p[k] - q[k])).collect();
    let alpha = step.max_step(x, &towards);
    x.iter().zip(&towards).map(|(q, d)| std::array::from_fn(|k| q[k] + alpha * d[k])).collect()
}
```

x_n (keadaan akhir langkah sebelumnya) selalu feasible. Dari sana kita bergerak ke arah x̂ sejauh yang diizinkan α_max. Tanpa collider, Newton tetap mulai
dari x̂ persis (bukan x_n + 1·(x̂ − x_n), yang bisa berbeda pembulatannya), sehingga E3/E4 tetap bit-identik.

Ini **keputusan desain** yang tidak tertulis di prompt, tetapi wajib supaya barrier valid. Keputusan ini dilaporkan di laporan akhir. GPU melakukan hal
yang sama (dokumen 5.10). Tes `feasible_start_keeps_a_tenth_of_every_collider_distance` menguncinya: kain di 0,1 mm di atas bola, bergerak turun 2 m/s,
dan titik awalnya menyisakan tepat 10% jarak.

## 5. `Solver`: cara menyelesaikan satu langkah

```rust
pub struct Solver {
    pub regularized: bool,
    pub floor_factor: Option<f64>,
}
```

Sebelumnya `simulate(..., regularized: bool)`. Sekarang satu struct kecil yang juga membawa faktor lantai f64 (dokumen 5.1). Skema di `e3_generate.rs`
(dokumen 5.6) memilih nilainya.
