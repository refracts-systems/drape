# Step 4.1: `oracle/src/newton.rs`, Newton dengan Regularisasi Levenberg–Marquardt

> E3 menunjukkan dua hal sekaligus: tangen **eksak** memberi Newton yang cepat (3–4 iterasi) tetapi bisa **tidak positif definit** saat kain
> terkompresi, sedangkan tangen **proyeksi** selalu aman tetapi lambat (sampai 30 iterasi dan sering tidak konvergen). Step 4 memakai cara klasik
> untuk mendapatkan keduanya: tangen eksak yang **digeser** sedikit hanya ketika perlu.

---

## 1. Idenya

Kalau matriks tangen `K` tidak positif definit, tambahkan sesuatu yang positif sampai positif definit:

```
K + μ·M/h²
```

- `M/h²` adalah blok inersia, yang sudah ada di tangen backward Euler dan selalu positif.
- `μ` adalah "seberapa banyak inersia ekstra" yang ditambahkan.
- μ = 0: Newton murni, eksak, kuadratik.
- μ besar: mendekati langkah "gradient descent berbobot massa", yang lambat tetapi pasti turun.

Analogi backend: seperti **circuit breaker dengan backoff adaptif**. Jalur cepat dipakai selama aman; begitu gagal, mundur ke jalur konservatif yang makin
konservatif kalau terus gagal, lalu pelan-pelan kembali ke jalur cepat setelah berhasil.

## 2. Jadwal μ

```rust
/// Smallest non-zero μ and the cap of the K + μ·M/h² schedule.
pub const MU_START: f64 = 1e-3;
pub const MU_CAP: f64 = 1e3;

/// μ after a non-positive-definite factorisation or a non-descent direction: max(10μ, 1e-3), capped.
pub fn raise_mu(mu: f64) -> f64 {
    (10.0 * mu).max(MU_START).min(MU_CAP)
}

/// μ after an accepted full Armijo step: μ/10, and 0 once it falls below 1e-3.
pub fn relax_mu(mu: f64) -> f64 {
    let next = mu / 10.0;
    if next < MU_START { 0.0 } else { next }
}
```

| Kejadian | μ baru |
|---|---|
| Awal setiap langkah waktu | 0 |
| Cholesky gagal (tidak PD) atau arah bukan turun | max(10μ, 10⁻³), maksimum 10³ |
| Line search menerima langkah **penuh** (α = 1) | μ/10, atau 0 kalau < 10⁻³ |
| Line search harus memotong | tetap |

Persis seperti yang ditetapkan prompt. Tes `regularization_schedule_rises_tenfold_from_start_and_relaxes_to_zero` mengunci setiap baris tabel.

## 3. Trait tambahan

```rust
/// A system whose tangent can be shifted by μ times its inertia (M/h² for the drape).
pub trait RegularizedSystem: NewtonSystem {
    fn regularized_tangent(&self, x: &[Vec3], mu: f64) -> Bsr;
}
```

`RegularizedSystem: NewtonSystem` artinya "sistem yang teregularisasi juga harus sistem Newton biasa" (*supertrait*). Hanya drape yang mengimplementasikannya;
kupon E2 tidak perlu.

## 4. Newton teregularisasi

```rust
pub fn solve_step_regularized<S: RegularizedSystem>(system: &S, start: Vec<Vec3>, scale: f64, settings: &NewtonSettings) -> Result<StepSolution, String> {
    ...
    let mut mu = 0.0;
    loop {
        ... residual, cek konvergen, cek batas iterasi (sama dengan solve_step) ...
        let Some((direction, used)) = regularized_direction(system, &step.positions, &gradient, mu, &mut step.regularization) else {
            step.status = StepStatus::RegularizationCap;
            return Ok(step);
        };
        let Some(search) = line_search(...) else { ... };
        mu = if search.halvings == 0 { relax_mu(used) } else { used };
        advance(...);
    }
}
```

`solve_step` lama (dipakai E2 dan E3) **tidak disentuh**. Fungsi baru ini berdampingan, sehingga regresi E2/E3 dijamin tidak terpengaruh.

## 5. Mencari μ yang cukup

```rust
fn regularized_direction<S: RegularizedSystem>(system: &S, x: &[Vec3], gradient: &[f64], start: f64, record: &mut Regularization) -> Option<(Vec<Vec3>, f64)> {
    let negative: Vec<f64> = gradient.iter().map(|g| -g).collect();
    let mut mu = start;
    loop {
        record.max_mu = record.max_mu.max(mu);
        if let Ok(factor) = BandCholesky::factor(&system.regularized_tangent(x, mu)) {
            let direction = system.scatter(&factor.solve(&negative));
            if free_dot(gradient, &system.gather(&direction)) < 0.0 {
                return Some((direction, mu));
            }
        }
        if mu >= MU_CAP {
            record.cap_hit = true;
            return None;
        }
        mu = raise_mu(mu);
        record.retries += 1;
    }
}
```

Dua syarat untuk menerima μ:

1. Cholesky berhasil (matriks positif definit).
2. Arah yang dihasilkan benar-benar **turun** (`gᵀd < 0`).

Kalau salah satu gagal, μ dinaikkan dan **iterasi Newton yang sama** dicoba lagi. Kalau sudah di batas dan masih gagal, langkah berakhir dengan status
`RegularizationCap`. Gate E4: **tidak boleh ada** langkah yang mencapai batas.

## 6. Catatan per langkah

```rust
/// Levenberg–Marquardt bookkeeping of one step: largest μ used, failed factorisations or non-descent retries, cap reached.
#[derive(Clone, Copy, Default, Debug)]
pub struct Regularization {
    pub max_mu: f64,
    pub retries: usize,
    pub cap_hit: bool,
}
```

Field baru di `StepSolution`. `#[derive(Default)]` membuat nilai awal nol otomatis, sehingga `solve_step` lama cukup menulis `Regularization::default()`.
Data ini disimpan ke `reference.bin` (`max_mu`, `regularization_retries`, `mu_cap_hit`).

## 7. Hasil di oracle

| | E3 (proyeksi) | E4 (eksak + μ) |
|---|---|---|
| n33: rata-rata Newton/langkah | 16,2 | **5,83** |
| n33: langkah tidak konvergen ke 10⁻¹⁰ | 43 | **0** |
| n33: langkah memakai μ > 0 | – | 6 (μ maks 1) |
| n33: langkah mencapai batas μ | – | 0 |
| n33: waktu oracle | 4 min 22 s | 1 min 18 s |

## 8. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| K + μM/h² | geser tangen eksak dengan inersia ekstra hanya saat perlu |
| Naik ×10 | saat Cholesky gagal atau arah tidak turun |
| Turun ÷10 | setelah langkah penuh diterima |
| Batas | 10³; mencapai batas = gagal gate |
| Fungsi baru berdampingan | `solve_step` E2/E3 tidak berubah |
