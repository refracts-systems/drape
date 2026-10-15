# Step 1.24: `oracle/src/compare.rs` dan `criteria.rs`, Menilai Hasil GPU

> Setelah GPU selesai, oracle mengambil alih lagi. `compare.rs` menghitung semua metrik; `criteria.rs` memutuskan lulus/gagal
> per kasus. Di sinilah jawaban GPU benar-benar "diadili" dalam presisi f64.

---

## 1. Perintah

```bash
./oracle/target/release/oracle compare --dir cases/tension-n257
```

Membaca `problem.json`, `reference.json`, `reference.bin`, `gpu-result.json`, `gpu-x.bin`, lalu menulis `result.json` (schema `e1-result/1`).

## 2. Memuat semuanya: `CaseFiles`

```rust
pub struct CaseFiles {
    problem: String,
    pub reference: String,
    pub gpu: String,
    matrix: Bsr,
    rhs: Vec<f64>,
    exact: Vec<f64>,
    exact_rounded_system: Vec<f64>,
    cpu_f32: Vec<f32>,
}
```

JSON disimpan sebagai **teks** dan dibaca dengan `jsonscan.rs`, sebuah pembaca kunci datar yang sangat kecil (oracle tidak punya dependensi
sama sekali, termasuk serde). Data besar dibaca dari `reference.bin` memakai manifest, seperti di TypeScript.

## 3. Metrik: `score`

```rust
fn score(case: &CaseFiles, gpu32: &[f32]) -> Scores {
    let gpu: Vec<f64> = gpu32.iter().map(|&v| v as f64).collect();
    ...
    Scores {
        metrics: evaluate(&case.matrix, &case.rhs, &case.exact, &gpu),
        error_vs_f32_system: relative_difference(&gpu, &case.exact_rounded_system),
        difference_vs_cpu_f32: relative_difference(&gpu, &cpu),
        bitwise_vs_cpu_f32: bitwise_equal_fraction(gpu32, &case.cpu_f32),
    }
}
```

Jawaban GPU (f32) dinaikkan ke f64 **tanpa** mengubah nilainya, lalu:

| Metrik | Rumus | Arti |
|---|---|---|
| `relative_residual` | ‖b − A x_gpu‖ / ‖b‖ dalam f64 dengan matriks f64 | **residual sejati**; dasar gate forcing |
| `relative_error_vs_f64` | ‖x_gpu − x_exact‖ / ‖x_exact‖ | galat terhadap jawaban pasti |
| `_in_plane` / `_out_of_plane` | galat komponen x,y vs z | arah mana yang paling menderita |
| `relative_error_vs_f32_system` | galat terhadap solusi pasti matriks f32 | seberapa dekat ke lantai f32 |
| `relative_difference_vs_cpu_f32_pcg` | ‖x_gpu − x_emulasi‖ / ‖x_emulasi‖ | kesetiaan pada emulasi |
| `bitwise_equal_fraction_vs_cpu_f32_pcg` | fraksi bit identik | bukti FMA/urutan berbeda |

`evaluate` ada di `metrics.rs`; `bitwise_equal_fraction` membandingkan `to_bits()` dari dua f32.

## 4. Waktu dan bandwidth

```rust
/// Compulsory memory traffic of one PCG iteration: BSR values + column indices, plus every vector touched.
const BYTES_PER_BLOCK: f64 = 40.0;
const BYTES_PER_ROW: f64 = 196.0;

fn effective_bandwidth_gb_s(case: &CaseFiles) -> f64 {
    let bytes = BYTES_PER_BLOCK * blocks + BYTES_PER_ROW * rows;
    let gpu_ms = number(&case.gpu, "timing", "gpu_ms");
    let ms = if gpu_ms.is_finite() { gpu_ms } else { number(&case.gpu, "timing", "warm_solve_ms") };
    bytes * iterations / (ms * 1e6)
}
```

**Lalu lintas wajib** = byte minimum yang harus dibaca/ditulis per iterasi:

- per blok: 36 byte nilai (9 × f32) + 4 byte `col_idx` = 40,
- per baris: semua vektor yang disentuh (x, r, z, p, q, dinv, …) = 196 byte.

Di n257: 40 × 711.201 + 196 × 65.025 ≈ **41,2 MB/iterasi**. Dibagi waktu per iterasi, hasilnya bandwidth efektif. Pakai timestamp GPU kalau
ada (WebGPU mentah), jam dinding kalau tidak (vgpu).

```rust
pub fn speedup(case: &CaseFiles) -> f64 {
    1e3 * number(&case.reference, "pcg_f64_cpu", "seconds") / number(&case.gpu, "timing", "warm_solve_ms")
}
```

Speedup = waktu PCG f64 satu inti (diukur saat `generate`) / warm solve GPU.

## 5. Info host di hasil (perubahan sesi ini)

```rust
.text("adapter", &string(g, "", "adapter"))
.text("host_gpus", &string(g, "", "host_gpus"))
.text("host_os", &string(g, "", "host_os"))
.integer("replace_every", number(p, "", "replace_every") as usize)
.number("replace_above", number(p, "", "replace_above"))
```

Setiap `result.json` kini membawa identitas hardware dan pengaturan replacement-nya sendiri.

## 6. `criteria.rs`: gate per kasus

```rust
pub const FORCING_LIMIT: f64 = 1e-2;
pub const F32_ITERATION_RATIO_LIMIT: f64 = 1.5;
pub const ITERATION_GROWTH_LIMIT: f64 = 3.0;
pub const SPEEDUP_LIMIT: f64 = 10.0;
const EMULATION_ITERATION_SLACK: f64 = 0.05;

pub fn verdict(case: &CaseFiles, scores: &Scores) -> Verdict {
    let gpu_iterations = number(gpu, "", "iterations");
    let emulated = number(reference, "pcg_f32_cpu", "iterations");
    let same_iterations = (gpu_iterations - emulated).abs() <= 2.0 + EMULATION_ITERATION_SLACK * emulated;
    let m = &scores.metrics;
    Verdict {
        converged: string(gpu, "", "status") == "converged",
        forcing_ok: m.true_relative_residual <= FORCING_LIMIT,
        f32_iterations_ok: gpu_iterations <= F32_ITERATION_RATIO_LIMIT * number(reference, "pcg_f64_cpu", "iterations"),
        faithful: same_iterations && scores.difference_vs_cpu_f32 <= (2.0 * m.relative_error).max(1e-6),
        deterministic: flag(gpu, "", "deterministic_repeat"),
        speedup_ok: speedup(case) >= SPEEDUP_LIMIT,
    }
}
```

Komentar di file menjelaskan **asal-usul** setiap batas, bukan sekadar angkanya:

- `FORCING_LIMIT`: Newton tetap mendapat 2 digit per iterasi.
- `F32_ITERATION_RATIO_LIMIT`: lebih dari 1,5× berarti ortogonalitas CG rusak.
- `ITERATION_GROWTH_LIMIT`: ~2 untuk membran (orde 2), ~4 untuk tekukan (orde 4).
- `SPEEDUP_LIMIT`: kira-kira titik impas dengan solver CPU f64 multi-thread yang baik.

### `faithful`: dua syarat

1. Iterasi GPU dan emulasi berbeda paling banyak 2 + 5%. GPU 1545 vs emulasi 1608: selisih 63 ≤ 2 + 80,4. Lulus.
2. Selisih solusi GPU vs emulasi ≤ max(2 × galat-vs-f64, 10⁻⁶). Artinya "GPU tidak lebih jauh dari emulasi daripada keduanya dari jawaban benar".

Di baseline, dynamic-n129 gagal tipis (5,9×10⁻⁶ > 3,7×10⁻⁶) karena FMA. Dengan replacement, semua lulus.

### Aturan yang dipegang

Selama sesi kerja, **tidak satu konstanta pun di file ini diubah**, walau speedup gagal. Itu instruksi eksplisit: *"Do not loosen tolerances
just to obtain a pass"* dan *"Do not change SPEEDUP_LIMIT or any gate"*. Gate yang bisa digeser setelah melihat hasil bukan gate.

## 7. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Residual sejati f64 | GPU dinilai dengan matriks f64, bukan oleh dirinya sendiri |
| Lalu lintas wajib | 40 B/blok + 196 B/baris → bandwidth efektif |
| Speedup | PCG f64 1 inti / warm solve GPU |
| Gate dengan asal-usul | setiap batas punya alasan tertulis |
| Tidak digeser | konstanta gate tetap walau ada yang gagal |
