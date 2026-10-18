# Step 5.3: `oracle/src/e4_diagnose.rs`, Membedah Langkah 89 dan Lantai f64

> Sebelum mengubah apa pun pada oracle, prompt meminta **diagnosis dulu**: apa yang terjadi di langkah 89 n65, iterasi demi iterasi? Setelah itu baru
> batas iterasi oracle dinaikkan dari 30 ke 200. File ini adalah alat diagnosis itu.

---

## 1. Mengulang satu langkah dari lintasan tersimpan

```rust
/// The time step `step` of a stored f64 trajectory: x_{k−1}, v_{k−1} and the step's energy.
fn step_from<'a>(drape: &'a Drape, reference: &Reference, step: usize) -> DrapeStep<'a> {
    let x = &reference.positions[step - 1];
    let v: Vec<Vec3> = if step == 1 { vec![[0.0; 3]; x.len()] } else {
        x.iter().zip(&reference.positions[step - 2]).map(|(p, q)| std::array::from_fn(|d| (p[d] - q[d]) / TIME_STEP)).collect()
    };
    DrapeStep { drape, x_hat: drape.prediction(x, &v) }
}
```

`reference.bin` E4 menyimpan semua posisi f64. Kecepatan dihitung ulang dengan rumus yang **sama persis** dengan simulasi aslinya, (x_k − x_{k−1})/h,
sehingga x̂ langkah 89 identik bit per bit. Kita tidak perlu mensimulasikan ulang 88 langkah (yang memakan menit), cukup "melompat" ke langkah 89.

Analogi backend: **replay** satu request dari log produksi, dengan state database yang dipulihkan dari snapshot tepat sebelum request itu.

`<'a>` adalah anotasi lifetime Rust: `DrapeStep` meminjam `drape`, dan kompiler perlu tahu bahwa hasilnya tidak boleh hidup lebih lama dari `drape`.

## 2. Bukti bahwa replay-nya benar

```rust
let original = reference.residual_history(step);
let reproduced = original.iter().zip(&solution.residuals).map(|(a, b)| (a - b).abs() / a).fold(0.0, f64::max);
```

30 residual pertama hasil replay dibandingkan dengan yang tersimpan di referensi E4. Hasil: **selisih relatif 0**. Replay memakai skema E4-final (cap
200), tetapi 30 iterasi pertamanya identik dengan E4. Itu juga membuktikan perombakan `solve_step_regularized` (dokumen 5.1) tidak mengubah perilaku.

## 3. Hasil diagnosis langkah 89

| Iterasi | Residual relatif | μ | α | Catatan |
|---|---|---|---|---|
| 0 | 4,1×10⁻¹ | 10 | 1 | 5 kali Cholesky gagal sebelum μ cukup besar |
| 1–14 | 7×10⁻² → 3,3×10⁻⁶ | 1 | 1 | turun ~2× per iterasi (linear, bukan kuadratik) |
| 15–28 | 2,6×10⁻⁶ → 6,5×10⁻⁵ | 1 | 1 | **residual naik** sementara energi terus turun |
| 29 | 7,4×10⁻⁵ | 0,1 | 0,25 | line search memotong langkah |
| 30 | 8,2×10⁻⁴ | 0,1 | 1 | di sinilah E4 berhenti (batas 30) |
| 31–33 | 1,9×10⁻⁴ → 3,9×10⁻⁹ | 0,01 → 0 | 1 | konvergensi kuadratik |
| 34 | 8,4×10⁻¹² | – | – | konvergen |

Cara membaca:

- Di setiap iterasi 1–28 tercatat **satu retry**. μ diturunkan ke 0,1 setelah langkah penuh, tangen K + 0,1·M/h² ternyata tidak positif-definit
  (Cholesky gagal), jadi μ naik lagi ke 1. Tangen eksak di daerah ini **indefinit**: kain sedang melewati konfigurasi yang tidak stabil (lipatan berbalik).
- Dengan μ = 1, Newton menjadi Newton "teredam": arahnya aman, tetapi lajunya linear.
- Residual sempat naik selama energi Φ turun. Ini sah untuk minimisasi non-konveks: Newton meninggalkan daerah pelana menuju lembah dengan energi lebih
  rendah. Gradien di lereng antara dua lembah memang bisa membesar.
- Begitu sampai di lembah yang positif-definit, μ turun ke 0 dan konvergensi kuadratik kembali (4 iterasi dari 10⁻⁴ ke 10⁻¹¹).

**Kesimpulan:** bukan bug dan bukan kegagalan aturan. Langkah 89 hanya butuh 34 iterasi, lebih dari batas 30. Dengan batas 200, langkah ini konvergen di
iterasi 34, persis seperti replay, baik di diagnosis maupun di referensi E4-final yang dibuat ulang.

## 4. Lantai f64 di langkah yang macet

```rust
/// Final residual and measured f64 floor (both relative) at the stored final state of every unconverged step.
fn floors_json(drape: &Drape, reference: &Reference) -> String { ... system.residual_floor(&reference.positions[s]) / scale ... }
```

| Langkah | Residual akhir | floor_k (rumus prompt) | Rasio |
|---|---|---|---|
| 1 | 4,05×10⁻¹⁰ | 2,3×10⁻¹⁷ | 1,8×10⁷ |
| 2 | 1,96×10⁻¹⁰ | 8,0×10⁻¹⁷ | 2,4×10⁶ |
| 3 | 1,32×10⁻¹⁰ | 1,8×10⁻¹⁶ | 7,5×10⁵ |
| 4 | 1,03×10⁻¹⁰ | 2,9×10⁻¹⁶ | 3,5×10⁵ |
| 111 | 3,11×10⁻¹⁰ | 2,3×10⁻¹⁵ | 1,3×10⁵ |

Residual langkah 1 diam di 4,0535×10⁻¹⁰ selama 26 iterasi berturut-turut, jadi itu benar-benar lantai. Tetapi rumus ε64·‖Σ|g_c|‖ meramalkan lantai
**10⁵–10⁷ kali lebih rendah**.

Penyebabnya: rumus itu hanya menghitung galat **penjumlahan** kontribusi gradien. Lantai yang sebenarnya didominasi oleh pembulatan **posisi**: setiap
koordinat x ≈ 0,5 m membawa galat ε64·|x|, dan galat itu dikalikan kekakuan membran K saat dijadikan gaya (≈ ε64·‖K‖·|x|). Di langkah awal, gaya
elemen masih sangat kecil (kain baru mulai melengkung), sehingga Σ|g_c| kecil, padahal ‖K‖·|x| tidak.

Prompt berkata: *"if they do not, report it. Do not tune the constant."* Jadi rumusnya tidak diubah dan hasilnya dilaporkan. Setelah oracle dibuat ulang
dengan cap 200:

- langkah 89 dan 111 konvergen;
- langkah 1–4 tetap di 1,0–4,1×10⁻¹⁰ (langkah 1 berhenti karena line search gagal, 2–4 karena batas 200 iterasi).

Gate `f64_converged_every_step` di n65 tetap gagal, dan kini hanya karena empat langkah ini.

## 5. Menjalankan

```bash
oracle e4-diagnose --dir drape4/drape-n65 --step 89 --out results/e4-step89-diagnosis.json
```
