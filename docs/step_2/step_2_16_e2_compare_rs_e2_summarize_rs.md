# Step 2.16: `oracle/src/e2_compare.rs` dan `e2_summarize.rs`, Menilai Newton f32 terhadap f64

> Penilaian Step 2. Konsep paling penting di sini adalah **representation floor** (lantai representasi): ukuran yang memisahkan
> "solver f32 kurang bagus" dari "f32 memang tidak bisa lebih baik".

---

## 1. Batas yang dipakai

```rust
/// WGSL element gradient/tangent vs the f64 element routines at the same f32 input (≈ 100 ε_f32).
pub const KERNEL_LIMIT: f64 = 1e-5;
/// f32 solution vs f64 solution, relative to the peak displacement / peak grip stress: E1's storage-error bar.
pub const SOLUTION_LIMIT: f64 = 1e-3;
/// While both are above the f32 floor, f32 Newton residuals must stay within this factor of f64 Newton's.
pub const TRACKING_FACTOR: f64 = 2.0;
const FD_GRADIENT_LIMIT: f64 = 1e-6;
const FD_TANGENT_LIMIT: f64 = 1e-5;
const F64_EXACT_LIMIT: f64 = 1e-12;
/// Element strains amplify the leftover position error of a 1e-12 Newton stop by 1/(element size · strain).
const F64_PATCH_STRAIN_LIMIT: f64 = 1e-10;
const F64_PATH_LIMIT: f64 = 1e-9;
```

Asal-usulnya dicatat di komentar: `SOLUTION_LIMIT` meminjam bar storage error E1, dan batas FD meminjam batas tes elemen yang sudah ada.
`F64_PATCH_STRAIN_LIMIT` adalah satu-satunya batas yang ditetapkan sendiri setelah melihat tes unit (dokumen 2.4 bagian 8).

## 2. Memuat sebuah run

```rust
struct Run {
    coupon: Coupon,
    reference: Vec<StepSolution>,
    records: Vec<Record>,
    iterates: Vec<Vec<Vec3>>,
}

fn load(dir: &Path) -> Result<Run, String> {
    let problem = fs::read_to_string(dir.join("problem.json"))?;
    let coupon = Coupon::build(LoadPath::parse(&string(&problem, "", "path"))?, number(&problem, "", "grid_columns") as usize);
    let reference = solve_path(&coupon, &reference_settings())?;
    ...
}
```

Kupon dibangun ulang dari jalur + ukuran (deterministik), dan referensi f64 **dihitung ulang** alih-alih diparse dari JSON. Untuk n33/n65 hanya butuh
beberapa detik, dan itu menghindari parsing array bersarang dengan `jsonscan`.

## 3. Skor per langkah

```rust
struct StepScore {
    f32_residuals: Vec<f64>,                       // residual menurut GPU sendiri (f32)
    f64_residuals_of_f32_iterates: Vec<f64>,       // residual iterate GPU dihitung ulang dalam f64
    reference_residuals: Vec<f64>,                 // riwayat Newton f64
    representation_floor: f64,                     // residual f64 dari solusi f64 yang dibulatkan ke f32
    displacement_error: f64,
    stress_error: f64,
    tracking: f64,
    ...
}
```

### Tiga sudut pandang residual

| Ukuran | Dihitung oleh | Menjawab |
|---|---|---|
| `f32_residuals` | GPU, dalam f32 | apa yang dilihat Newton f32 |
| `f64_residuals_of_f32_iterates` | oracle, f64, pada posisi GPU | seberapa setimbang sebenarnya posisi itu |
| `representation_floor` | oracle, f64, pada `round_f32(x_ref)` | residual terbaik yang **mungkin** dengan posisi f32 |

```rust
fn round_f32(x: &[Vec3]) -> Vec<Vec3> {
    x.iter().map(|p| p.map(|v| v as f32 as f64)).collect()
}
...
representation_floor: relative(&round_f32(&reference.positions), reference_scale),
```

**Representation floor** adalah analog `residual_floor_f32_rounded_exact` di E1: ambil jawaban f64 yang (hampir) sempurna, bulatkan setiap koordinat
ke f32, lalu ukur residualnya dalam f64. Kalau Newton f32 mencapai lantai ini, ia sudah mendapat **jawaban f32 terbaik yang ada**. Tidak ada solver,
algoritma, atau trik yang bisa lebih baik selama posisi disimpan sebagai f32.

### Skala yang konsisten

```rust
let (gpu_scale, reference_scale) = (coupon.step_scale(previous_gpu, t), coupon.step_scale(previous_reference, t));
```

Iterate GPU dinormalisasi dengan skala dari kesetimbangan GPU sebelumnya; referensi dengan kesetimbangan f64 sebelumnya. Masing-masing dinilai dari titik
awalnya sendiri.

### Galat solusi

- `displacement_error`: max |x_gpu − x_ref| / perpindahan puncak sepanjang jalur. Dinormalisasi dengan **puncak** jalur, karena di langkah terakhir (rest)
  perpindahannya nol dan pembagi lokal akan nol.
- `stress_error`: tegangan grip (dihitung f64 dari posisi GPU) vs referensi, relatif terhadap tegangan puncak.

### Tracking

```rust
/// Largest ratio between f32 and f64 Newton residuals at equal iteration count, while f64 is above the f32 floor
/// (taken as the smallest f32 residual of the step times 10).
fn tracking(f32_residuals: &[f64], f64_residuals: &[f64]) -> f64 { ... }
```

Bandingkan residual f32 dan f64 pada nomor iterasi yang sama, **selama** f64 masih di atas 10× lantai f32. Rasio ≈ 1 berarti Newton f32 berperilaku
sama dengan Newton f64, termasuk sifat kuadratiknya, sampai lantai. Margin 10× ini konservatif, sehingga di uniaxial hanya iterate pertama yang diuji
secara formal; riwayat lengkap di `result.json` menunjukkan kecocokan sampai iterate kedua.

## 4. Cek kernel

```rust
fn kernel_errors(run: &Run, dir: &Path) -> Result<(f64, f64), String> {
    let gpu = read_f32_file(&dir.join("gpu-kernel.bin"))?;
    let state: Vec<Vec3> = kernel_state(coupon).chunks_exact(3).map(|p| [p[0] as f32 as f64, ...]).collect();
    let gradient: Vec<f64> = ...element.gradient(&state)...;
    let tangent: Vec<f64> = ...element.tangent(&state)...;
    ...
}
```

Posisi uji dibulatkan ke f32 **dulu**, supaya f64 menghitung pada input yang persis sama dengan GPU. Galat yang tersisa murni dari aritmetika kernel.

## 5. Verdict

```rust
fn verdict_json(verdict: &Verdict, scores: &[StepScore]) -> Json {
    Json::new()
        .flag("fd_gradient_ok", d.gradient_error <= FD_GRADIENT_LIMIT)
        .flag("fd_tangent_ok", d.tangent_error <= FD_TANGENT_LIMIT)
        .flag("energy_change_identity_ok", d.energy_change_error <= F64_EXACT_LIMIT)
        .flag("f64_exactness_ok", verdict.f64_exact)
        .flag("wgsl_kernel_ok", verdict.kernel.0.max(verdict.kernel.1) <= KERNEL_LIMIT)
        .flag("f32_reaches_newton_tolerance", reached)
        .flag("f32_tracks_f64_newton", worst(scores, |s| s.tracking) <= TRACKING_FACTOR)
        .flag("f32_displacement_ok", worst(scores, |s| s.displacement_error) <= SOLUTION_LIMIT)
        .flag("f32_stress_ok", worst(scores, |s| s.stress_error) <= SOLUTION_LIMIT)
}
```

`f64_exactness_ok`: patch test untuk jalur patch, bolak-balik untuk lainnya.

`worst(scores, |s| ...)`: ambil nilai terburuk atas semua langkah. Closure Rust di sini mirip arrow function TS.

Saat ditulis pertama kali, `Verdict` menyimpan pointer fungsi `worst` sebagai field. Itu indireksi tanpa guna, dan dihapus sebelum build.

## 6. `e2_summarize.rs`

```rust
const CHECKS: [&str; 9] = [
    "fd_gradient_ok", "fd_tangent_ok", "energy_change_identity_ok", "f64_exactness_ok", "wgsl_kernel_ok",
    "f32_reaches_newton_tolerance", "f32_tracks_f64_newton", "f32_displacement_ok", "f32_stress_ok",
];
```

Mengumpulkan semua `coupon/*/result.json`, mencetak satu baris per kupon:

```
uniaxial-n33 N=3069 f32floor=4.43e-5 f64(f32)=4.43e-5 repfloor=4.43e-5 disp=3.55e-7 stress=3.66e-6 track=1.00e0 kernel=1.92e-7/9.74e-8 ...=Y/Y/Y/Y/Y/n/Y/Y/Y
```

Tiga kolom pertama setelah N adalah inti temuan Step 2: **f32floor = f64(f32) = repfloor**.

## 7. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Referensi dihitung ulang | menghindari parsing JSON bersarang |
| Tiga residual | f32 sendiri, f64 pada posisi f32, lantai representasi |
| Representation floor | residual terbaik yang mungkin dengan posisi f32 |
| Normalisasi puncak | perpindahan/tegangan relatif terhadap puncak jalur |
| Tracking | f32 vs f64 per iterasi selama di atas lantai |
| Input kernel identik | posisi uji dibulatkan f32 sebelum f64 menghitung |
