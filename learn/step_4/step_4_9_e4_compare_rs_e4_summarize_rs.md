# Step 4.9: `e4_compare.rs` dan `e4_summarize.rs`, Menilai Setiap Rung dan Memilih Konfigurasi Final

> Penilaian Step 4 membungkus penilaian E3. Untuk setiap rung: jalankan semua cek E3, lalu tambahkan cek E4 (oracle konvergen, replay aturan, alasan berhenti
> di probe, iterasi Newton, ms/langkah).

---

## 1. Alur `run`

```rust
pub fn run(dir: &Path, rung: &str) -> Result<(), String> {
    let gpu_dir = dir.join(format!("rung-{rung}"));
    ...
    let base = score(dir, &gpu_dir)?;                      // semua cek E3, apa adanya
    let groups = by_step(&read_iterates(&gpu_dir, "")?);   // catatan iterate + catatan robust
    let stats = trajectory_stats(&groups);
    let inertia = inertia_error(&drape, &gpu_dir)?;
    let replay = forcing_replay(&groups, forcing, pcg_tolerance);
    let terminations = termination_replay(&groups, floor_stop);
    ...
}
```

`rung-X/result.json` berisi: verdict E4 (paling depan, supaya `jsonscan` menemukannya lebih dulu), statistik robust, data oracle, probe, dan seluruh JSON E3
di `e3_checks`.

## 2. Menggabungkan dua file catatan

```rust
fn read_iterates(dir: &Path, prefix: &str) -> Result<Vec<Iterate>, String> {
    let records = read_f64_rows(&dir.join(format!("gpu-{prefix}iterations.bin")), RECORD_FIELDS)?;
    let extras = read_f64_rows(&dir.join(format!("gpu-{prefix}robust.bin")), EXTRA_FIELDS)?;
    Ok(records.iter().zip(&extras).map(|(r, e)| Iterate { ... }).collect())
}
```

`zip` memasangkan baris ke-i kedua file (keduanya ditulis bersamaan oleh `pushRecord`). `by_step` mengelompokkan iterate per langkah waktu.

## 3. Replay forcing

```rust
/// Largest relative gap between the GPU's recorded η and the f64 rule replayed on its recorded residuals.
fn forcing_replay(groups: &[Vec<&Iterate>], forcing: bool, fixed: f64) -> f64 {
    ... untuk setiap langkah: norma = residual relatif setiap update; harapan = forcing_sequence(norma) ...
}
```

GPU mencatat η yang dipakai dan residual setiap iterasi. Oracle menjalankan `forcing_sequence` (dokumen 4.2) pada residual yang sama, lalu membandingkan.
Batas 10⁻⁵ (presisi f32). Hasil n33 rung D: 3×10⁻⁷. Untuk rung tanpa forcing, η yang tercatat harus tepat 10⁻⁷.

Rasio norma = rasio residual relatif, karena skalanya sama dalam satu langkah.

## 4. Replay aturan berhenti

```rust
/// Every update-rule stop is justified by its last update, and with the floor stop on, no update the rule would stop on was followed.
fn termination_replay(groups: &[Vec<&Iterate>], floor_stop: bool) -> bool {
    ...
    let last_ok = !stopped || updates.last().is_some_and(|it| update_is_small(it.update_inf, it.step_inf));
    let early = updates.iter().rev().skip(usize::from(stopped)).any(|it| update_is_small(it.update_inf, it.step_inf));
    last_ok && !(floor_stop && early)
}
```

Dua arah:

1. Setiap berhenti karena update **memang** memenuhi aturan.
2. Tidak ada update yang memenuhi aturan tetapi GPU **terus jalan** (kalau aturan aktif).

Ini pengujian **dua sisi** (tidak ada false positive, tidak ada false negative). Analogi: rekonsiliasi audit log dengan aturan bisnis.

## 5. Gate probe

```rust
/// A probe stops properly when it reaches the residual rule, or the update rule within 2x of the representation floor.
fn probe_stops_ok(probes: &[ProbeSummary], groups: &[Vec<&Iterate>]) -> bool { ... status == 1 || (status == 4 && f32_floor <= 2·rep) ... }
```

Sesuai prompt: berhenti lewat aturan residual, **atau** lewat aturan update dengan lantai f32 ≤ 2× lantai representasi. Berhenti karena batas iterasi **tidak**
memenuhi gate ini, walaupun lantainya sudah benar.

## 6. Gate lain

| Gate | Sumber |
|---|---|
| `f64_converged_every_step` | `reference.bin`: semua langkah konvergen, nol langkah di batas μ |
| `wgsl_kernels_ok` | gate E3 **dan** galat inersia ≤ 10⁻⁵ |
| `rules_replayed_ok` | replay forcing ≤ 10⁻⁵ dan replay berhenti lulus |
| `newton_iterations_ok` | rata-rata Newton GPU ≤ rata-rata oracle + 2 |
| `ms_per_step_ok` | hanya n65: ≤ 174 ms |
| FD, energi, konsistensi satu langkah | dari verdict E3 |

## 7. `e4_summarize.rs`: tabel ablation dan rung final

```rust
/// Accuracy gates decide the final rung; ms_per_step_ok and newton_iterations_ok are the performance gates.
const ACCURACY: [&str; 10] = [ "wgsl_kernels_ok", "fd_gradient_ok", "fd_hessian_ok", "hinge_energy_change_identity_ok", "energy_f32_ok", "rules_replayed_ok",
    "f64_converged_every_step", "energy_f64_ok", "one_step_ok", "f32_stops_ok_at_probes" ];
const PERFORMANCE: [&str; 2] = ["newton_iterations_ok", "ms_per_step_ok"];
```

Pembagian **ditetapkan sebelum melihat hasil semua rung**. Gate akurasi menentukan rung final ("rung terakhir yang lolos semua gate akurasi"); gate performa
dilaporkan.

```rust
let passing: Vec<&str> = RUNGS.iter().copied().filter(|r| results.iter().filter(|t| string(t, "", "rung") == *r).all(|t| accuracy_ok(t))).collect();
let final_rung = passing.last().copied().unwrap_or("none");
```

`-` (tidak dinilai, misalnya n129 tanpa referensi) tidak dihitung gagal. Pertumbuhan PCG dihitung per rung antar ukuran mesh.

## 8. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Membungkus E3 | semua cek E3 per rung, tanpa perubahan |
| Replay | oracle memeriksa η dan keputusan berhenti GPU |
| Uji dua sisi | tidak berhenti salah, tidak lupa berhenti |
| Akurasi vs performa | pembagian ditetapkan sebelum hasil |
| Rung final | terakhir yang lolos semua gate akurasi di semua ukuran |
