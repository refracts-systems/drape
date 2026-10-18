# Step 4.3: `drape.rs`, `e3_generate.rs`, `e3_compare.rs` (Perubahan Step 4)

> Tiga file E3 mendapat perubahan kecil supaya E4 bisa memakai ulang semuanya: drape bisa diregularisasi, generator bisa membuat referensi dengan skema
> berbeda, dan penilai bisa membaca output GPU dari folder per rung.

---

## 1. `drape.rs`: drape sebagai `RegularizedSystem`

```rust
impl RegularizedSystem for DrapeStep<'_> {
    /// K + μ·M/h²: the tangent of the drape's kind with μ times the inertia block added on the diagonal.
    fn regularized_tangent(&self, x: &[Vec3], mu: f64) -> Bsr {
        let mut matrix = self.tangent(x);
        self.drape.free.iter().enumerate().for_each(|(row, &i)| matrix.add(row, row, &identity_scaled(mu * self.drape.mass_over_h2(i))));
        matrix
    }
}
```

Tangen biasa, ditambah μ·m/h² di setiap blok diagonal. Hasilnya tangen total menjadi `(1 + μ)·M/h² + K_elastis`.

`time_step` dan `simulate` mendapat parameter `regularized: bool` yang memilih `solve_step` (E3) atau `solve_step_regularized` (E4). Pemanggil E3 mengirim `false`.

## 2. `e3_generate.rs`: skema

```rust
/// How a drape experiment builds its reference: tangent kind, plain or regularised Newton, and the note recorded with it.
pub struct Scheme {
    pub name: &'static str,
    pub tangent: TangentKind,
    pub regularized: bool,
    pub note: &'static str,
}

pub const E3_SCHEME: Scheme = Scheme { name: "e3", tangent: TANGENT, regularized: false, note: "band Cholesky: exact tangent not positive definite at step 87 ..." };
pub const E4_SCHEME: Scheme = Scheme { name: "e4", tangent: TangentKind::Exact, regularized: true, note: "exact tangent + adaptive K + mu*M/h^2 ..." };

pub fn run(columns: usize, with_reference: bool, dir: &Path) -> Result<(), String> {
    generate(columns, with_reference, dir, &E3_SCHEME)
}
```

Satu generator, dua konfigurasi. Perintah `e3-generate` tetap memanggil `run` (skema E3); perintah baru `e4-generate` memanggil `generate(..., &E4_SCHEME)`.
Isi `problem.bin` identik (geometri, peta gather, probe dari lintasan referensi masing-masing), sedangkan `problem.json`/`reference.json` mencatat skema, tangen,
dan flag regularisasi.

`reference.bin` mendapat tiga array baru per langkah: `max_mu`, `regularization_retries`, `mu_cap_hit` (nol semua untuk E3).

`problem.json` mendapat bagian `"robust"` (konstanta aturan E4). Bagian ini juga ikut di soal E3 yang dibuat ulang, tetapi GPU E3 mengabaikannya.

Pola ini disebut **strategy object**: perilaku yang berbeda dikumpulkan sebagai data (`Scheme`), bukan disalin menjadi fungsi kembar.

## 3. `e3_compare.rs`: folder soal vs folder GPU

Di E3, soal, referensi, dan output GPU ada di satu folder. Di E4, satu soal punya empat set output GPU (rung A–D), jadi fungsi-fungsi dipisah:

```rust
fn load_run(problem_dir: &Path, gpu_dir: &Path) -> Result<Run, String>
fn result_json(run: &Run, problem_dir: &Path, gpu_dir: &Path) -> Result<String, String>

/// The e3-result/1 JSON for a GPU run whose outputs live in `gpu_dir` and whose problem and reference live in `problem_dir`.
pub fn score(problem_dir: &Path, gpu_dir: &Path) -> Result<String, String>

pub fn run(dir: &Path) -> Result<(), String> {
    let json = score(dir, dir)?;   // E3: kedua folder sama
    ...
}
```

Plus ringkasan probe untuk E4:

```rust
pub struct ProbeSummary { pub step: usize, pub error: f64, pub f32_floor: f64, pub representation_floor: f64 }
pub fn probe_summaries(problem_dir: &Path, gpu_dir: &Path) -> Result<Vec<ProbeSummary>, String>
```

Hasilnya, semua cek E3 (FD, kernel, probe, energi, bentuk akhir) dijalankan **tanpa perubahan logika** untuk setiap rung E4.

## 4. Bukti tidak ada perubahan perilaku

Regresi `drape-n33` dengan konfigurasi E3: `result.json` **identik** dengan versi commit E3, kecuali `ms_per_step`.

## 5. Ringkasan

| File | Perubahan | Tujuan |
|---|---|---|
| `drape.rs` | `RegularizedSystem`, flag `regularized` | oracle E4 |
| `e3_generate.rs` | `Scheme`, `generate`, array μ | satu generator untuk E3 dan E4 |
| `e3_compare.rs` | folder soal ≠ folder GPU, `score`, `probe_summaries` | cek E3 per rung |
