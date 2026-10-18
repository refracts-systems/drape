# Step 3.18: `oracle/src/e3_compare.rs` dan `e3_summarize.rs`, Menilai Drape GPU terhadap f64

> Penilaian Step 3. Konsep baru: **probe satu langkah**, **energi relatif max |E|**, dan definisi tertulis untuk "oracle berhenti di lantainya sendiri".

---

## 1. Gate, ditulis sebelum mengukur

```rust
const KERNEL_LIMIT: f64 = 1e-5;
const FD_GRADIENT_LIMIT: f64 = 1e-6;
const FD_HESSIAN_LIMIT: f64 = 1e-5;
const HINGE_IDENTITY_LIMIT: f64 = 1e-12;
/// One GPU step from an f64 state vs the f64 next state, relative to ‖x_{n+1} − x_n‖.
const ONE_STEP_LIMIT: f64 = 1e-3;
/// f32 Newton may stop above 1e-6 only at its floor: within 2x of the representation floor.
const FLOOR_FACTOR: f64 = 2.0;
/// Energy increase per step relative to max |E|: f64 rounding allowance, and the f32 bar.
const F64_ENERGY_LIMIT: f64 = 1e-12;
const F32_ENERGY_LIMIT: f64 = 1e-6;
/// An f64 step "stops at its own floor" when it ends within 10x of the 1e-10 tolerance without decreasing over its last 10 iterates.
const F64_FLOOR_BOUND: f64 = 1e-9;
const FLOOR_WINDOW: usize = 10;
```

Semua angka gate berasal dari prompt. Dua interpretasi ditambahkan dan ditulis eksplisit sebelum hasil terlihat:

- **Energi relatif max |E|**, karena energi awal = 0 (dokumen 3.5).
- **"Berhenti di lantai sendiri"** untuk f64: berakhir ≤ 10× toleransi (≤ 10⁻⁹) **dan** stagnan selama 10 iterasi terakhir. Langkah yang berhenti di 10⁻² karena batas
  iterasi bukan "lantai".

## 2. Membaca referensi biner

```rust
struct Reference {
    positions: Vec<Vec<Vec3>>,
    iterations: Vec<f64>,
    final_residual: Vec<f64>,
    converged: Vec<f64>,
    residual_offsets: Vec<f64>,
    residuals: Vec<f64>,
}
```

Dari `reference.bin` (dokumen 3.6). Lintasan f64 **tidak dihitung ulang** (n65 butuh 72 menit). Riwayat residual per langkah diambil lewat offset:

```rust
fn residual_history(&self, step: usize) -> &[f64] {
    &self.residuals[self.residual_offsets[step - 1] as usize..self.residual_offsets[step] as usize]
}
```

## 3. Cek kernel: enam angka

```rust
fn kernel_errors(drape: &Drape, dir: &Path) -> Result<[f64; 6], String>
```

| Angka | WGSL | f64 |
|---|---|---|
| gradien membran | kedua varian | `MembraneElement::gradient` |
| gradien hinge | kedua varian | `HingeElement::gradient` |
| tangen membran eksak | `PROJECTED = false` | `tangent` |
| Hessian hinge eksak | `EXACT = true` | `tangent` (dual) |
| Hessian membran proyeksi | `PROJECTED = true` | `hessian` |
| Hessian hinge Gauss–Newton | `EXACT = false` | `hessian` |

Posisi uji dibulatkan ke f32 dulu, supaya input identik.

## 4. Probe

```rust
fn probe(drape: &Drape, reference: &Reference, step: usize, records: &[&Record], iterates: &[Vec<Vec3>]) -> Probe {
    let (previous, next) = (&reference.positions[step - 1], &reference.positions[step]);
    let start_velocity = if step == 1 { nol } else { velocity(&reference.positions[step - 2], previous) };
    let gpu_step = DrapeStep { drape, x_hat: drape.prediction(&round_f32(previous), &round_f32(&start_velocity)) };
    let exact_step = DrapeStep { drape, x_hat: drape.prediction(previous, &start_velocity) };
    ...
    Probe {
        error: ‖x_gpu − x_ref(k)‖ / ‖x_ref(k) − x_ref(k−1)‖,
        f32_residuals: residual menurut GPU,
        f64_residuals_of_f32_iterates: residual iterate GPU dievaluasi f64 (dengan x̂ dari input f32 yang sama dengan GPU),
        reference_residuals: riwayat Newton f64 langkah itu,
        representation_floor: residual f64 dari round_f32(x_ref(k)), dengan x̂ f64,
    }
}
```

Tiga sudut pandang residual (seperti E2) ditambah galat satu langkah. Gate Newton f32 di probe: mencapai 10⁻⁶ **atau** lantainya ≤ 2× lantai representasi.

**Interpretasi**: lantai representasi hanya terdefinisi di probe, di mana keadaan f64 yang tepat tersedia. Untuk 240 langkah lintasan, lintasan f32 sudah menyimpang
(chaos), sehingga tidak ada keadaan f64 pembanding. Maka gate "f32 Newton" dinilai di probe, sedangkan lintasan dilaporkan (residual akhir per langkah).

## 5. Statistik lintasan

```rust
struct StepRecords {
    iterations: Vec<f64>,
    final_residual: Vec<f64>,
    converged: Vec<bool>,
    pcg_per_solve: f64,
    worst_linear_residual: f64,
    halvings: f64,
}
```

`pcg_per_solve` (rata-rata iterasi PCG per solve Newton) adalah dasar gate pertumbuhan iterasi.

## 6. Bentuk akhir dan energi

- `shape_difference`: perbedaan maksimum dan RMS antara kain f32 dan f64 di t = 1 s, relatif terhadap ukuran lembar. Ini **diagnostik**, bukan gate, karena chaos.
- `energy_history` untuk kedua lintasan.

## 7. `e3_summarize.rs`

```rust
/// PCG iterations per Newton solve between consecutive mesh sizes (the E1 iteration-growth gate).
fn growth(results: &[String]) -> Vec<Json> { ... ratio < ITERATION_GROWTH_LIMIT ... }
```

Hasil diurutkan menurut jumlah unknown (n33 → n65 → n129), lalu rasio PCG per solve dihitung antar ukuran bertetangga. `ITERATION_GROWTH_LIMIT` diimpor langsung
dari `criteria.rs` E1, bukan disalin.

Kolom yang tidak berlaku (n129 tanpa referensi) dicetak sebagai `-`, bukan `n`, supaya "tidak dinilai" tidak terbaca sebagai "gagal".

## 8. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Gate dari prompt | plus dua interpretasi tertulis sebelum mengukur |
| Referensi biner | lintasan f64 tidak dihitung ulang |
| Probe | galat solver dari keadaan identik + tiga sudut pandang residual |
| Lantai f64 | ≤ 10⁻⁹ dan stagnan; batas iterasi bukan lantai |
| Pertumbuhan PCG | gate E1 dipakai ulang lintas ukuran |
