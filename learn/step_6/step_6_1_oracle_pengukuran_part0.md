# Step 6.1: Oracle, Aturan Pengukuran Baru (Part 0)

> Part 0 Step 6 tidak mengubah fisika atau solver. Ia memperbaiki **cara mengukur**: empat aturan yang ditetapkan di prompt sebelum run, diterapkan ke
> data yang sudah tersimpan. Dokumen ini membahas file oracle yang berubah: `drape.rs`, `contact.rs`, `e5_checks.rs`, `e5_compare.rs`,
> `e6_rescore.rs`, dan `e4_compare.rs`.

---

## 1. Kenapa mengukur ulang, bukan menjalankan ulang

Oracle E5 di n65 butuh sekitar 3 jam CPU (iterasi Newton f64 dengan Cholesky pita ±0,9 detik per iterasi). Semua gate Part 0 hanya bergantung pada
**keadaan akhir yang tersimpan** (`reference.bin`) dan keluaran GPU yang sudah ada. Karena itu Part 0 adalah *re-scoring*: skor dihitung ulang dari file,
tanpa menyalakan GPU dan tanpa membuat ulang referensi.

Analogi backend: mengubah definisi SLO dan menghitung ulang dashboard dari log yang sudah ada, bukan menjalankan ulang load test.

## 2. Aturan 0.1: toleransi replay α_max yang sadar kondisi

Replay α_max membandingkan α_max GPU (f32) dengan α_max f64 dari input yang sama. Di E5 batasnya 10⁻⁵ relatif, dan semua selisih di atasnya berasal dari
bola, karena jarak ke bola dihitung sebagai |x − c| − r, selisih dua angka ≈ 0,125 m.

```rust
/// For every probe update: the GPU's α_max against the f64 rule ..., with the tolerance max(1e-5, 8·ε32·cond_α),
/// cond_α = |q|/d0 of the limiting sphere (1 for the plane).
fn alpha_replay(drape: &Drape, gpu_dir: &Path, iterates: &[Iterate]) -> Result<AlphaReplay, String>
```

Toleransi per update: max(10⁻⁵, 8·ε32·cond_α) dengan ε32 = 2⁻²³. cond_α = |q|/d0 adalah faktor pembesaran galat: galat absolut |q| (≈ ε32·0,125)
dibagi jarak d0 yang kecil. Ini analisis kondisi yang sama dengan toleransi kernel kontak di Step 5, kini diterapkan ke α_max.

Untuk mengetahui sphere mana dan vertex mana yang membatasi α_max di f64, ada dua method baru:

- `Contact::max_step_limited(x, p)` mengembalikan α dan indeks collider yang membatasinya;
- `Contact::distance_condition(k, x)` mengembalikan |x − c|/d untuk bola dan 1 untuk lantai;
- `DrapeStep::max_step_detail(x, d)` mengembalikan α_max, vertex pembatas, dan colliderny.

Laporan memuat berapa update yang membutuhkan cabang kondisi (galat > 10⁻⁵) dan rasio galat terhadap toleransi terburuk.

## 3. Aturan 0.2: FD yang tidak tumbuh dengan N

Di E5, FD gagal di mesh besar karena keadaan uji menaruh **setiap** vertex di dalam barrier, sehingga jumlah energi global dibulatkan di f64 dan galat
relatif FD tumbuh seiring jumlah vertex. Aturan baru: arah perturbasi hanya di **patch 16 vertex berurutan** (6 patch acak, seed tetap).

```rust
struct Patch { vertices: Vec<usize>, direction: Vec<Vec3> }
```

Agar selisih energi juga tidak membawa pembulatan jumlah global, Φ dihitung **lokal**:

```rust
/// Φ restricted to what a perturbation of the vertices in `patch` can change: their inertia and barrier terms and the
/// membranes and hinges that touch them.
pub fn local_value(&self, x: &[Vec3], patch: &[usize]) -> f64
```

Hanya elemen yang menyentuh patch yang dijumlahkan. Selisih Φ(x + hv) − Φ(x − hv) sama persis dengan selisih energi global (elemen lain tidak berubah),
tetapi tanpa pembulatan jumlah ribuan suku yang tidak berubah. Barrier-only FD memakai langkah 10⁻³·d_min patch (aturan E5), dan identitas energi
barrier dihitung dari selisih dua jumlah barrier lokal.

Angka dengan arah global tetap dilaporkan sebagai `global_no_gate`.

### Apa yang terjadi, dan satu diagnostik yang tidak mengubah gate

FD Φ memakai langkah 10⁻⁷ m (aturan E3, tidak disebut berubah). Dengan patch, sebagian patch memuat vertex yang jaraknya ke collider hanya
~2,5×10⁻⁵ m, sehingga galat truncation (h/d)² ≈ 1,6×10⁻⁵ muncul lagi:

| | FD Φ gradien (patch, h = 10⁻⁷) | dengan h = 10⁻³·d_min (**tanpa gate**) |
|---|---|---|
| n33 | **4,5×10⁻⁶** ✗ (batas 10⁻⁶) | 4,7×10⁻⁷ |
| n65 | 8,0×10⁻⁷ ✓ | 5,6×10⁻⁸ |
| n129 | **4,6×10⁻⁶** ✗ | 3,2×10⁻⁷ |

Gate mengikuti aturan prompt apa adanya (jadi `contact_fd_ok` di n33 berubah dari ✓ menjadi ✗, karena kebetulan patch acak yang sekarang terambil
memuat vertex yang sangat dekat), dan varian dengan langkah yang diskalakan dilaporkan terpisah sebagai bukti bahwa penyebabnya langkah FD, bukan turunan.

## 4. Aturan 0.3: lantai f64 yang menghitung pembulatan posisi

E5 mendapati rumus ε64·‖Σ|g_c|‖ **10⁵–10⁷ kali** terlalu kecil. Alasannya: ia hanya menghitung galat **penjumlahan** kontribusi gradien. Padahal
setiap koordinat x ≈ 0,5 m sendiri sudah dibulatkan (galat ε64·|x|), dan galat itu dikalikan kekakuan K saat menjadi gaya.

```rust
fn residual_floor(&self, x: &[Vec3]) -> f64 {
    f64::EPSILON * (vector_norm(&self.absolute_contributions(x)) + vector_norm(&self.stiffness_times_positions(x)))
}

/// |K|·|x|: the entrywise absolute tangent (μ = 0) times the absolute unknown positions.
pub fn stiffness_times_positions(&self, x: &[Vec3]) -> Vec<f64> {
    let mut tangent = self.tangent(x);
    tangent.vals.iter_mut().for_each(|v| *v = v.abs());
    tangent.multiply(&self.gather(x).iter().map(|v| v.abs()).collect::<Vec<f64>>())
}
```

Rumus prompt: floor_k = ε64·(‖Σ|g_c|‖₂ + ‖|K|·|x|‖₂)/scale_k. Konvergen jika residual ≤ 10⁻¹⁰ **atau** ≤ 2·floor_k. Konstanta 2 tidak disetel.

### `e6_rescore.rs`

```rust
pub fn classify(dir: &Path) -> Result<Convergence, String>
```

Untuk setiap langkah referensi yang tidak mencapai 10⁻¹⁰, `classify` membangun kembali `DrapeStep` langkah itu dari lintasan tersimpan (`step_from`, dokumen 5.3),
menghitung floor_k pada **keadaan akhir tersimpan**, dan membandingkan. Hasil:

| | oleh toleransi | oleh lantai | gagal |
|---|---|---|---|
| E4-final n33 | 240 | 0 | 0 |
| E4-final n65 | 236 | **4** | **0** |
| E5 n33 | 240 | 0 | 0 |
| E5 n65 | 192 | **47** | **1 (langkah 127)** |

Kelima puluh langkah itu kini masuk dengan margin lebar: rasio residual/lantai ≈ 0,08–0,12. Langkah 127 tetap gagal (rasio 225), karena memang macet di
1,75×10⁻⁷ (bukan lantai pembulatan).

`e4_compare::oracle_stats` memakai klasifikasi ini **hanya** untuk referensi E4-final dan E5 (`problem.json` punya `robust.resolution_factor > 0`).
Referensi E4 asli (`drape4/`) tetap dinilai dengan flag tersimpan, jadi tidak ada hasil E3 atau E4 lama yang berubah.

Aturan lantai baru juga dipakai `solve_step_regularized` untuk generasi **mendatang** (lewat `residual_floor`), tetapi tidak dijalankan di eksperimen ini.

## 5. Aturan 0.4: `oracle e6-rescore`

```bash
oracle e6-rescore --root . --out results/experiment6-part0-rescore.json
```

Menjalankan ulang `e4-compare` (rung D, drape4-final) dan `e5-compare` di semua ukuran, membandingkan setiap gate dengan nilai tersimpan, dan mencetak
yang berubah:

```
e4-final n65: f64_converged_every_step n -> Y
e5 n33: contact_fd_ok Y -> n
e5 n33: rules_replayed_ok n -> Y
e5 n65: contact_fd_ok n -> Y
e5 n65: rules_replayed_ok n -> Y
e5 n129: barrier_energy_change_identity_ok n -> Y
```

Pola ini, yaitu menyimpan nilai sebelum dan sesudah lalu menampilkan selisihnya, mirip diff skema database sebelum migrasi.

## 6. Refaktor `e5_compare.rs`

`run` sekarang memanggil `score_case(dir, gpu_dir)` yang mengembalikan `CaseScore` (drape, iterate, probe, verdict, dll.) dan `write_result`. E6 memakai
keduanya dengan folder GPU `e6-rung-X`. Dua fungsi baru, `rules` dan `alpha_json`, memecah fungsi yang terlalu panjang.
