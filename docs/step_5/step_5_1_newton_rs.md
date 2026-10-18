# Step 5.1: `oracle/src/newton.rs`, Trace per Iterasi, Konvergensi di Lantai f64, dan Awal Line Search

> Step 5 menyentuh Newton oracle di tiga tempat kecil. Ketiganya tidak mengubah perilaku E2, E3, atau E4. Yang ditambahkan: **catatan per iterasi**
> untuk diagnosis, **cabang konvergensi kedua** (berhenti di lantai pembulatan f64 yang diukur), dan **titik awal line search** yang bisa lebih kecil
> dari 1 (dibutuhkan kontak di Part 1).

---

## 1. `max_step`: line search tidak selalu mulai dari α = 1

```rust
pub trait NewtonSystem {
    ...
    /// Largest step length the line search may start from (1 unless a barrier must stay positive).
    fn max_step(&self, _x: &[Vec3], _d: &[Vec3]) -> f64 {
        1.0
    }
}
```

Ini **default method** pada trait: semua sistem lama (kupon E2, drape E3/E4) otomatis mengembalikan 1. Analogi backend: menambah kolom baru dengan nilai
default di tabel, sehingga query lama tetap berjalan tanpa diubah.

Line search memakainya sebagai titik awal:

```rust
let start = system.max_step(x, direction);
(0..=settings.max_halvings)
    .map(|halvings| LineSearch { step_length: start * 0.5f64.powi(halvings as i32), halvings })
```

Untuk sistem lama, `start = 1.0` dan `1.0 * 0.5^h` **identik bit per bit** dengan `0.5^h`. Jadi E2–E4 tidak berubah, dan ini dicek dengan tes regresi
(dokumen 5.11).

Di Part 1, drape dengan collider mengembalikan α_max: langkah terbesar yang menjaga setiap vertex tetap sejauh ≥ 10% jarak awalnya dari collider
(dokumen 5.4).

## 2. Cabang konvergensi kedua: berhenti di lantai f64

E4 mendapati lima langkah oracle n65 **macet** di residual 1–4×10⁻¹⁰, sedikit di atas toleransi 10⁻¹⁰. Prompt Step 5 menetapkan aturan baru sebelum run:

> Konvergen ⇔ residual relatif ≤ 10⁻¹⁰, **atau** ≤ 2·floor_k, dengan floor_k = ε64·‖Σ_c |g_c|‖₂ / scale_k.

```rust
/// Converged at the tolerance, or (with `floor_factor`) within `floor_factor` times the measured f64 floor.
fn convergence(relative: f64, floor: f64, settings: &NewtonSettings, floor_factor: Option<f64>) -> Option<StepStatus> {
    if relative <= settings.tolerance {
        return Some(StepStatus::Converged);
    }
    floor_factor.filter(|factor| relative <= factor * floor).map(|_| StepStatus::ConvergedAtFloor)
}
```

- `Option<f64>`: E4 memberi `None` (cabang kedua mati), E4-final dan E5 memberi `Some(2.0)`.
- `filter(...).map(...)` adalah cara Rust menulis "kalau ada faktor **dan** syaratnya terpenuhi, hasilkan status ini". Analogi: `Optional.filter().map()` di Java.
- Status baru `ConvergedAtFloor` dibedakan dari `Converged`, supaya laporan bisa menghitung berapa langkah lulus lewat cabang mana.

```rust
pub fn converged(&self) -> bool {
    matches!(self, StepStatus::Converged | StepStatus::ConvergedAtFloor)
}
```

### Apa itu Σ|g_c|?

Gradien Φ dirakit dari banyak **kontribusi** (inersia per vertex, gaya membran per segitiga, gaya hinge per pasangan segitiga, dan di E5 gaya barrier per
vertex). Setiap penjumlahan f64 membawa galat sekitar ε64 × besar suku yang dijumlahkan. Jadi "lantai" yang wajar untuk ‖∇Φ‖ adalah ε64 dikali jumlah
**nilai mutlak** kontribusi. Kalau semua suku saling meniadakan sampai tersisa angka kecil, galat pembulatannya tetap sebesar suku-suku aslinya.

Hasil yang diukur (dokumen 5.3) menunjukkan rumus ini **terlalu kecil** untuk kelima langkah tadi, sehingga cabang kedua tidak menolong. Hasil ini
dilaporkan apa adanya, tanpa mengubah konstanta.

## 3. `IterateTrace`: catatan per iterasi untuk diagnosis

```rust
/// One regularised update, for diagnosis: μ, step length, slope gᵀd, accepted ΔΦ, ‖αd‖∞, retries and the line-search start.
pub struct IterateTrace {
    pub mu: f64,
    pub step_length: f64,
    pub slope: f64,
    pub energy_change: f64,
    pub update_inf: f64,
    pub retries: usize,
    pub max_step: f64,
}
```

Prompt meminta diagnosis langkah 89 per iterasi: μ, α, gᵀd, ΔΦ, residual relatif, ‖αd‖∞, dan berapa kali Cholesky gagal. `StepSolution` sekarang
membawa `trace: Vec<IterateTrace>`, yang diisi hanya oleh Newton teregularisasi. Analogi backend: **structured logging** per request, tetapi hanya di
endpoint yang sedang diselidiki.

## 4. `solve_step_regularized` dipecah

Fungsi lama lebih dari 20 baris. Sekarang loop utamanya hanya: hitung residual, cek konvergensi (dua cabang), cek batas iterasi, lalu panggil
`regularized_update`:

```rust
match regularized_update(system, &mut step, &gradient, mu, settings) {
    Some(next) => mu = next,
    None => return Ok(step),
}
```

`regularized_update` mencari arah (dengan μ), menjalankan line search, mencatat trace, lalu maju. Ia mengembalikan μ berikutnya, atau `None` kalau langkah
waktu berakhir (status sudah diisi). Urutan operasinya sama dengan E4, sehingga 30 iterasi pertama langkah 89 **sama persis** dengan referensi E4
(diukur: selisih relatif 0).

## 5. Ringkasan

| Perubahan | Efek pada E2–E4 | Dipakai oleh |
|---|---|---|
| `max_step` default 1 | tidak ada (bit-identik) | kontak E5 |
| `ConvergedAtFloor` + `floor_factor` | tidak ada (`None`) | E4-final, E5 |
| `IterateTrace` | hanya catatan tambahan | `e4-diagnose` |
| batas iterasi lewat `Scheme` | E3/E4 tetap 30 | E4-final, E5: 200 (oracle saja) |
