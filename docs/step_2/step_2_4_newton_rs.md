# Step 2.4: `oracle/src/newton.rs`, Newton f64 dengan Line Search Armijo

> Solver nonlinear referensi. Langkah linearnya **pasti** (Cholesky pita dari Step 1). Ini kunci jawaban untuk Newton GPU f32,
> yang langkah linearnya tidak pasti (PCG f32).

---

## 1. Pengaturan

```rust
pub struct NewtonSettings {
    pub tolerance: f64,
    pub max_iterations: usize,
    pub armijo: f64,
    pub max_halvings: usize,
}
```

| Pengaturan | Oracle f64 | GPU f32 |
|---|---|---|
| `tolerance` | 10⁻¹² (sampai lantai f64) | 10⁻⁶ |
| `max_iterations` | 30 | 30 |
| `armijo` (c) | 10⁻⁴ | 10⁻⁴ |
| `max_halvings` | 30 | 30 |

Oracle dijalankan jauh lebih ketat (10⁻¹²) supaya solusinya menjadi **referensi** untuk solusi f32. Untuk membandingkan jumlah iterasi, dihitung
"iterasi sampai ≤ 10⁻⁶" dari riwayat f64.

## 2. Status langkah

```rust
pub enum StepStatus {
    Converged,
    IterationLimit,
    LineSearchFailed,
}
```

Tiga cara sebuah langkah beban berakhir. "Gagal line search" berarti arah Newton bukan arah turun, atau 30 kali dipotong pun energi tidak turun cukup.

## 3. Loop Newton

```rust
pub fn solve_step(coupon: &Coupon, start: Vec<Vec3>, scale: f64, settings: &NewtonSettings) -> Result<StepSolution, String> {
    let mut step = StepSolution { positions: start, residuals: Vec::new(), searches: Vec::new(), status: StepStatus::IterationLimit };
    loop {
        let gradient = coupon.free_gradient(&step.positions);
        let relative = vector_norm(&gradient) / scale;
        step.residuals.push(relative);
        if relative <= settings.tolerance {
            step.status = StepStatus::Converged;
            return Ok(step);
        }
        if step.searches.len() == settings.max_iterations {
            return Ok(step);
        }
        let direction = newton_direction(coupon, &step.positions, &gradient)?;
        let Some(search) = line_search(coupon, &step.positions, &gradient, &direction, settings) else {
            step.status = StepStatus::LineSearchFailed;
            return Ok(step);
        };
        advance(&mut step.positions, &direction, search.step_length);
        step.searches.push(search);
    }
}
```

Langkah demi langkah:

1. Hitung residual `g` (gaya di titik bebas) dan normanya relatif terhadap skala langkah.
2. Catat. Kalau ≤ toleransi: selesai.
3. Kalau sudah 30 iterasi: berhenti dengan status `IterationLimit`.
4. Selesaikan `K d = −g` (arah Newton).
5. Line search menentukan panjang langkah α.
6. `x ← x + α d`.

`let Some(x) = ... else { ... }` adalah sintaks Rust *let-else*: kalau hasilnya `None`, jalankan blok else (yang harus keluar dari fungsi).
Seperti *guard clause* di TypeScript.

`residuals[k]` = residual iterasi ke-k; `residuals[0]` = residual setelah prediktor. `searches[k]` = line search dari iterasi k ke k+1.

## 4. Arah Newton: langkah linear pasti

```rust
fn newton_direction(coupon: &Coupon, x: &[Vec3], gradient: &[f64]) -> Result<Vec<Vec3>, String> {
    let negative: Vec<f64> = gradient.iter().map(|g| -g).collect();
    let rows = BandCholesky::factor(&coupon.tangent(x))?.solve(&negative);
    Ok(coupon.scatter(&rows))
}
```

- Rakit tangen global eksak (dengan masking z).
- Faktorisasi Cholesky pita, lalu selesaikan.
- `scatter`: ubah vektor per-baris (hanya titik bebas) menjadi vektor per-titik (titik jepit = 0).

Kalau tangen tidak positif definit, `?` meneruskan error Cholesky, dan itulah pesan yang memicu prediktor (dokumen 2.3).

## 5. Line search Armijo

```rust
/// Backtracking from α = 1: accept when ΔΠ ≤ c·α·gᵀd. None when d is not a descent direction or halving runs out.
pub fn line_search(coupon: &Coupon, x: &[Vec3], gradient: &[f64], direction: &[Vec3], settings: &NewtonSettings) -> Option<LineSearch> {
    let rows: Vec<f64> = coupon.free.iter().flat_map(|&v| direction[v]).collect();
    let slope = free_dot(gradient, &rows);
    if slope >= 0.0 {
        return None;
    }
    (0..=settings.max_halvings)
        .map(|halvings| LineSearch { step_length: 0.5f64.powi(halvings as i32), halvings })
        .find(|s| coupon.energy_change(x, direction, s.step_length) <= settings.armijo * s.step_length * slope)
}
```

**Syarat Armijo**: `Π(x + αd) − Π(x) ≤ c · α · gᵀd`

- `gᵀd` (*slope*) adalah laju perubahan energi di awal langkah. Untuk arah turun, nilainya negatif.
- Syaratnya: energi turun **minimal** sebagian kecil (c = 10⁻⁴) dari yang diprediksi garis lurus.
- Coba α = 1, ½, ¼, … sampai syarat terpenuhi.

Kode memakai iterator Rust (`map` + `find`) yang **malas** (*lazy*): berhenti di α pertama yang lolos, tidak menghitung semuanya.
Mirip `Array.prototype.find` di atas generator.

`energy_change` dipakai, **bukan** `energy(x+αd) − energy(x)`. Dokumen 2.2 menjelaskan kenapa.

## 6. Satu jalur penuh

```rust
/// Runs every load step of the coupon's path from the previous equilibrium plus the affine predictor.
pub fn solve_path(coupon: &Coupon, settings: &NewtonSettings) -> Result<Vec<StepSolution>, String> {
    let mut current = coupon.initial_positions();
    let (mut steps, mut previous_factor) = (Vec::new(), 0.0);
    for t in coupon.path.load_factors() {
        let scale = coupon.step_scale(&current, t);
        coupon.predict(&mut current, previous_factor, t);
        let step = solve_step(coupon, current.clone(), scale, settings)?;
        previous_factor = t;
        current = step.positions.clone();
        steps.push(step);
    }
    Ok(steps)
}
```

Untuk setiap faktor beban: hitung skala, prediksi, selesaikan, lalu bawa kesetimbangan ke langkah berikutnya. Urutan "skala dulu, baru prediksi" penting:
skala diukur dari kesetimbangan lama + penjepit baru, sebelum interior digeser prediktor. Kode TypeScript memakai urutan yang sama.

## 7. Hasil f64 nyata

```
patch    n=33 | f64 Newton iterations per step [9]
uniaxial n=33 | [3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3]
biaxial  n=33 | [4 4 4 ... ]
```

Contoh riwayat uniaxial langkah pertama:

```
2,05×10⁻³ → 6,41×10⁻⁵ → 2,04×10⁻⁸ → 1,04×10⁻¹⁴
```

Pangkatnya kira-kira berlipat dua setiap iterasi (−3 → −4,2 → −7,7 → −14): **konvergensi kuadratik** klasik. Semua langkah dengan α = 1 dan nol pemotongan,
karena prediktor sudah menaruh Newton di daerah kuadratiknya.

Di uniaxial-n65, dua langkah mencapai **lantai f64** (2,1×10⁻¹² > 10⁻¹²) dan berhenti di 30 iterasi. Itu bukan kegagalan; f64 pun punya lantai pembulatan.

## 8. Tes (di file ini)

| Tes | Memeriksa |
|---|---|
| `armijo_backtracks_an_overlong_newton_step` | arah Newton dikali 4 → line search **harus** memotong ≥ 1 kali, dan langkah yang diterima memenuhi Armijo |
| `patch_test_reproduces_the_affine_map` | n9: perpindahan < 10⁻¹², regangan < 10⁻¹⁰ |
| `assembled_derivatives_match_finite_differences` | cek FD global di biaxial n9 |
| `unloading_retraces_loading` | uniaxial n9: celah tegangan < 10⁻⁹, kembali ke rest < 10⁻¹² |

Tes Armijo ada karena di kupon sungguhan line search **tidak pernah** memotong langkah. Tanpa tes paksa, logika backtracking tidak pernah terbukti berjalan.

Tes patch awalnya memakai batas regangan 10⁻¹² dan gagal (4,8×10⁻¹² di n9). Penyebabnya bukan fisika: Newton berhenti di residual 10⁻¹², dan galat regangan
memperbesar galat posisi sisa sebesar 1/(ukuran elemen × regangan). Batasnya diubah ke 10⁻¹⁰ untuk regangan, dan tetap 10⁻¹² untuk perpindahan. Batas ini
dibuat sendiri di Step 2, bukan gate yang ditetapkan pengguna.

## 9. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Newton | selesaikan `K d = −g`, melangkah, ulangi |
| Pasti vs inexact | oracle: Cholesky; GPU: PCG f32 |
| Armijo | potong α setengah sampai energi turun cukup |
| `energy_change` | perubahan energi tanpa pembatalan angka |
| Kuadratik | eksponen residual berlipat dua per iterasi |
| Lantai f64 | sekitar 10⁻¹² sampai 10⁻¹⁴ |
