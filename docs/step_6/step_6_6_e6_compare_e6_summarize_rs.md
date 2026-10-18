# Step 6.6: `e6_compare.rs` dan `e6_summarize.rs`, Menilai Satu Rung dan Memilih Rung Final

> Penilai E6 membungkus penilai E5 (semua cek E3/E4/E5 dengan aturan Part 0) dan menambah tiga cek khusus rung: replay keputusan penerimaan, akurasi
> reduksi GPU, dan bukti keamanan untuk penerimaan karena noise. Ringkasan lalu menerapkan aturan pemilihan rung final dan trigger.

---

## 1. `e6_compare.rs`

### Replay keputusan penerimaan

```rust
fn replayed_decision(trial: &Trial, noise_aware: bool) -> bool {
    let plain = (ARMIJO * trial.alpha * trial.slope) as f32 as f64;
    trial.sum <= plain || (noise_aware && trial.sum <= plain + NOISE_FACTOR * EPSILON32 * trial.abs)
}
```

Untuk setiap α yang dicoba di probe, oracle menghitung ulang di f64 keputusan terima/tolak dari angka yang dicatat GPU (Σδ, Σ|δ|, gᵀd, α) lalu
membandingkannya dengan keputusan GPU. Gate `acceptance_replay_ok` mensyaratkan **setiap halving** sama. `as f32 as f64` meniru pembulatan
`Math.fround` di sisi TypeScript.

Replay ini hanya memeriksa **konsistensi aturan**, bukan apakah aturannya berguna. Untuk itu ada angka lain:

```rust
/// the smallest (Σδ − c·α·gᵀd) / (4·ε32·Σ|δ|) over its rejected trials
fn rejected_excess(trials: &[Trial]) -> f64
```

Selisih terkecil antara energi yang diukur dan ambang Armijo, dinyatakan dalam satuan batas noise. Nilai di atas 1 berarti suku noise terlalu kecil untuk
menjelaskan penolakan. Hasilnya **612** di n33: bahkan percobaan yang paling dekat diterima melebihi batas noise 612 kali.

### Akurasi reduksi

```rust
fn reduction_errors(iterates: &[Iterate]) -> (f64, f64, usize)
```

Hanya untuk rung C dan D: galat relatif ‖rhs‖² dan gᵀd dari reduksi GPU terhadap jumlah f64 dari **nilai f32 yang sama** (dibaca host di probe), batas 10⁻⁵.
Hasil: 1,3×10⁻⁷ dan 1,7×10⁻⁷.

### Bukti keamanan noise

```rust
fn noise_safety(score: &CaseScore, gpu_dir: &Path, reference: &Reference) -> Result<NoiseSafety, String>
```

Untuk setiap update yang diterima **hanya** karena suku noise (probe dan lintasan), oracle mengevaluasi Φ di f64 sebelum dan sesudah, pada x̂ langkah itu, dan
melaporkan kenaikan Φ dalam satuan batas noise. Kalau ada kenaikan lebih dari 2× batas, rung B dibuang (trigger). Di n33 tidak ada update seperti itu,
jadi tidak ada pelanggaran.

x̂ dihitung dari keadaan yang dibulatkan ke f32 (seperti yang dipegang GPU): untuk probe dari referensi, untuk lintasan dari keadaan GPU sebelumnya.

### Verdict

Verdict ditulis **pertama** (penilai `jsonscan` membaca kemunculan pertama), dengan urutan: cek E6, lalu verdict E5. `acceptance_replay_ok` selalu ada,
`reduction_accuracy_ok` hanya untuk rung dengan `gpuReductions`, dan `noise_safety_ok` hanya untuk rung dengan `noiseAware`. Gate yang tidak berlaku
ditandai `-` di ringkasan, bukan dihitung gagal.

## 2. `e6_summarize.rs`

### Gate akurasi

17 gate: 14 gate E5 dengan aturan Part 0, ditambah `acceptance_replay_ok`, `reduction_accuracy_ok`, dan `noise_safety_ok`. Kasus yang diperiksa untuk memilih
rung final adalah n33 dan n65 (n129 tidak punya oracle, jadi hanya gate yang berlaku tanpa oracle yang dinilai).

### Aturan rung final

> Rung final = rung terakhir yang semua gate akurasinya lulus di n33 dan n65 **dan tidak dibuang trigger**.

```rust
let usable: Vec<&str> = RUNGS.iter().copied().filter(|r| !dropped || *r == "A").collect();
let passing: Vec<&str> = usable.iter().copied().filter(|r| !of_rung(&runs, r).is_empty() && of_rung(&runs, r).iter().all(|x| accuracy_ok(&x.text))).collect();
let final_rung = passing.last().copied().unwrap_or("none");
```

Syarat `!of_rung(...).is_empty()` mencegah bug vacuous truth yang sempat muncul di E4 (rung tanpa hasil dianggap lulus).

### Trigger

| Trigger | Implementasi |
|---|---|
| rung B dibuang | `rung_b_dropped`: ada update noise-only yang menaikkan Φ > 2× batas noise, **atau** error satu langkah probe B lebih dari 2× A |
| C atau D mengubah alasan berhenti | `stop_change`: fraksi langkah n33 dengan alasan berhenti berbeda antara dua rung berurutan; batas 5% |

`step_statuses` (di `e6_compare.rs`) mengambil status terakhir tiap langkah dari catatan iterasi, dibaca langsung dari file rung.

### Gate performa

```rust
fn performance(runs: &[Run], rung: &str) -> Json
```

Untuk setiap rung, terhadap rung A pada ukuran yang sama:

| Gate | Batas |
|---|---|
| ms/langkah n65 | ≤ 0,8× rung A |
| byte baca per iterasi Newton, n65 | ≤ 0,1× rung A |
| Newton ≤ oracle + 2 | semua kasus |
| pertumbuhan PCG n33→n65→n129 | < 3× |

Ringkasan menghitungnya untuk **setiap** rung (bukan hanya yang final), supaya terlihat apa yang dihasilkan tiap perubahan.
