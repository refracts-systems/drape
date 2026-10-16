# Step 4.2: `oracle/src/e4_rules.rs`, Aturan Forcing dan Berhenti sebagai Fungsi Murni

> File kecil dengan peran ganda. (1) Satu sumber kebenaran untuk angka-angka aturan E4, yang ditulis ke `problem.json` dan dibaca GPU.
> (2) Implementasi referensi f64 untuk aturan yang dijalankan GPU, sehingga oracle bisa **memutar ulang** setiap keputusan GPU dan memeriksanya.

---

## 1. Forcing Eisenstat–Walker (choice 2)

### Masalahnya

E3 menyelesaikan setiap sistem Newton sampai residual linear 10⁻⁷, butuh ratusan iterasi PCG. Padahal di awal Newton (masih jauh dari jawaban), solusi linear
yang sangat akurat **terbuang**: langkah Newton itu sendiri masih kasar. Akurasi linear baru penting di akhir, saat Newton hampir konvergen.

### Aturannya

```rust
pub const ETA_START: f64 = 1e-2;
pub const ETA_MIN: f64 = 1e-7;
pub const ETA_MAX: f64 = 1e-2;
const GAMMA: f64 = 0.9;
const SAFEGUARD_THRESHOLD: f64 = 0.1;

/// η_k = 0.9·(‖g_k‖/‖g_{k−1}‖)², safeguarded by 0.9·η_{k−1}² when that exceeds 0.1, clamped to [1e-7, 1e-2].
pub fn next_forcing(previous: f64, residual_ratio: f64) -> f64 {
    let eta = GAMMA * residual_ratio * residual_ratio;
    let safeguard = GAMMA * previous * previous;
    let guarded = if safeguard > SAFEGUARD_THRESHOLD { eta.max(safeguard) } else { eta };
    guarded.clamp(ETA_MIN, ETA_MAX)
}
```

- η adalah **toleransi PCG** untuk iterasi Newton ini.
- Kalau residual Newton turun cepat (rasio kecil), η jadi kecil dan PCG diminta lebih akurat, karena kita sudah dekat jawaban.
- Kalau residual turun lambat, η tetap besar; tidak ada gunanya akurat kalau Newton-nya sendiri belum mendekat.
- Rentang [10⁻⁷, 10⁻²]: atas sama dengan forcing term E1, bawah sama dengan toleransi E1 lama.

Analogi backend: **adaptive timeout/retry budget**. Ketika upstream masih lambat, jangan habiskan banyak waktu untuk optimasi kecil; ketika sudah hampir
selesai, baru teliti.

Catatan: dengan η_max = 10⁻², safeguard (0,9·η² > 0,1) tidak pernah aktif. Ia tetap ditulis sesuai spesifikasi prompt.

### Barisan

```rust
pub fn forcing_sequence(norms: &[f64]) -> Vec<f64> {
    norms.iter().enumerate().scan(ETA_START, |eta, (k, &norm)| {
        if k > 0 {
            *eta = next_forcing(*eta, norm / norms[k - 1]);
        }
        Some(*eta)
    }).collect()
}
```

`scan` adalah fold yang menghasilkan setiap nilai antara (seperti reduce yang mengeluarkan running state). Dipakai `e4-compare` untuk memutar ulang η GPU.

## 2. Aturan berhenti berbasis perpindahan

```rust
/// Stop once the accepted update is this small relative to the step taken so far (‖αd‖∞ ≤ 1e-4·‖x − x_n‖∞).
pub const UPDATE_STOP: f64 = 1e-4;

pub fn update_is_small(update_inf: f64, step_inf: f64) -> bool {
    update_inf <= UPDATE_STOP * step_inf
}
```

**Masalah E3**: toleransi residual 10⁻⁶ berada di bawah lantai f32, sehingga GPU selalu menghabiskan 30 iterasi, padahal posisinya sudah tidak berubah berarti
setelah 2–3 iterasi.

**Aturan baru**: berhenti kalau koreksi terakhir sudah < 10⁻⁴ dari perpindahan total langkah ini. Satuannya **meter**, bukan residual, jadi tidak tergantung pada
skala residual yang bermasalah di E3.

## 3. Pengaturan untuk GPU

```rust
/// μ used by the WGSL inertia-kernel check: the regularised diagonal block must equal (1 + μ)·m/h².
pub const KERNEL_MU: f64 = 0.37;

/// The rule constants the GPU host loop reads from problem.json (one source of truth with the oracle).
pub fn settings_json() -> Json { ... eta_start, eta_min, eta_max, eta_gamma, eta_safeguard, update_stop, mu_start, mu_cap, kernel_mu ... }
```

Bagian `"robust"` di `problem.json`. GPU tidak punya konstanta aturan sendiri.

## 4. Tes

| Tes | Memeriksa |
|---|---|
| `forcing_starts_at_its_maximum_and_follows_the_squared_residual_ratio` | η₀ = 10⁻², lalu 0,9·rasio² (9×10⁻³, 9×10⁻⁵, 9×10⁻⁷) |
| `forcing_is_clamped_to_its_interval` | rasio besar → 10⁻²; rasio kecil → 10⁻⁷ |
| `update_rule_compares_with_the_step_taken` | batas 10⁻⁴ |

## 5. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Eisenstat–Walker | toleransi PCG mengikuti laju Newton |
| Aturan update | berhenti saat koreksi < 10⁻⁴ dari langkah, dalam meter |
| Satu sumber | konstanta ditulis ke problem.json |
| Replay | oracle memeriksa keputusan GPU dengan fungsi yang sama |
