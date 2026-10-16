# Step 5.7: `e4_compare.rs`, `e5_compare.rs`, dan `e5_summarize.rs`, Menilai Kontak

> Penilai E5 **membungkus** penilai E3 dan E4 (semua cek lama tetap jalan), lalu menambah cek kontak: kernel kontak, turunan barrier, bebas penetrasi,
> replay α_max, dan diagnostik (jumlah kontak, drift pusat massa).

---

## 1. Perubahan kecil di `e4_compare.rs`

### Membaca catatan robust berdasarkan nama kolom

```rust
/// Robust records are read by the field names listed in gpu-result.json, so E4 runs (5 fields) and later runs score alike.
pub fn read_iterates(dir: &Path, prefix: &str) -> Result<Vec<Iterate>, String> {
    let fields = names(&gpu, "robust_fields");
    ...
    let pick = |e: &[f64], name: &str| fields.iter().position(|f| f == name).map_or(f64::NAN, |i| e[i]);
```

E4 menulis 5 kolom, E4-final 6 (`resolution`), E5 9 (`alphaMax`, `limitingVertex`, `minDistance`). Daripada lebar tetap, penilai membaca daftar nama
kolom dari `gpu-result.json` lalu mengambil kolom berdasarkan nama. Kolom yang tidak ada menjadi NaN. Analogi backend: membaca CSV lewat header, bukan
posisi kolom, sehingga menambah kolom tidak merusak pembaca lama.

### Replay aturan berhenti memakai resolusi tercatat

```rust
update_is_small(it.update_inf, it.step_inf, resolution_of(it))
```

Run tanpa kolom `resolution` (E4 lama) diputar ulang dengan 0, yaitu aturan E4 persis.

### Cabang konvergensi di laporan oracle

`OracleStats` sekarang juga menghitung `converged_at_floor_steps`, berapa langkah f64 yang lulus lewat cabang lantai.

### Perbaikan `e4_summarize.rs`

Pemilihan rung final memakai `.all(...)` atas hasil per rung. Untuk folder `drape4-final/` yang hanya punya rung D, rung A–C tidak punya hasil sama sekali,
dan `.all` atas daftar kosong bernilai `true` (vacuous truth), sehingga ringkasan sempat menyebut "rung C". Sekarang rung harus **punya hasil** dulu
(`ran(r) && …`). Untuk E4 asli (semua rung ada), hasilnya tidak berubah.

## 2. `e5_compare.rs`: cek kontak

### Kernel kontak dengan toleransi berbasis kondisi

```rust
/// cond = max over active colliders of (|x − c|/d)·max(1, |b″·d/b′|); the plane's |x − c|/d is 1.
fn condition(drape: &Drape, x: Vec3) -> f64
```

Toleransi per entri, ditetapkan di prompt sebelum run: **max(10⁻⁵, 8·ε32·cond)**.

Kenapa tidak cukup 10⁻⁵ seperti kernel lain? Jarak ke bola dihitung di f32 sebagai |x − c| − r, yaitu selisih dua angka ≈ 0,125 m. Galat pembulatannya
≈ ε32 × 0,125 m ≈ 10⁻⁸ m. Itu kecil dibanding 0,125, tetapi besar dibanding d = 2,5×10⁻⁵ m. Rasio |x − c|/d mengukur seberapa besar galat relatif itu
diperbesar. Faktor kedua mengukur kepekaan b′ terhadap d. Ini **analisis kondisi** klasik: toleransi mengikuti seberapa sulit soalnya, bukan dipilih
setelah melihat hasil.

Setiap entri dinilai relatif terhadap entri terbesar di kelompoknya (3 entri gradien atau 9 entri blok Hessian per vertex), supaya entri off-diagonal
yang hampir nol tidak menghasilkan galat relatif tak hingga. Laporan menyebut berapa entri yang **butuh** cabang cond (galat > 10⁻⁵) dan berapa yang gagal.

### Bebas penetrasi, tiga sumber

```rust
/// Intersection checks: f32 trajectory states in f64, the GPU's f64 record of every accepted iterate, probe iterates in f64.
fn distances(...) -> Result<[f64; 3], String>
```

1. Setiap keadaan lintasan f32 (241 keadaan) dievaluasi ulang di f64 oleh oracle.
2. Setiap iterasi yang diterima: GPU mencatat jarak minimum yang dihitung host dalam f64 dari posisi f32. Lintasan tidak menyimpan semua iterasi (terlalu
   besar), jadi ini satu-satunya cara memeriksa iterasi di tengah langkah.
3. Setiap iterasi probe dievaluasi ulang oleh oracle.

Gate lulus kalau ketiganya > 0. Laporan menyebut min d/d̂ masing-masing.

### Replay α_max

```rust
/// Largest relative gap between the GPU's α_max and the f64 rule at the recorded probe iterate x_k and direction p_k.
fn alpha_replay(...)
```

Di probe, GPU menyimpan x_k dan p_k setiap update (`gpu-probe-directions.bin`). Oracle menghitung α_max di f64 dari input f32 yang sama lalu membandingkan,
dengan batas prompt 10⁻⁵ relatif. Hasilnya di dokumen 5.11.

### Replay resolusi

```rust
/// The f32 spacing the GPU recorded for each probe update equals f32_spacing(max |x_i|) of the iterate it was recorded at.
fn resolution_replay(...)
```

Angka `res` yang dipakai GPU untuk memutuskan berhenti juga diperiksa: oracle menghitungnya sendiri dari posisi probe dan menuntut kesamaan **eksak**
(bilangan pangkat dua, tidak ada pembulatan).

### Diagnostik (tanpa gate)

- `gpu_active_contacts` / `f64_active_contacts`: jumlah vertex dalam d̂ per langkah.
- `gpu_com_drift_max_m`: jarak horizontal pusat massa kain dari sumbu bola. Kain simetris di atas bola tanpa gesekan adalah **kesetimbangan tidak
  stabil** (seperti tali di atas katrol licin). Gangguan sekecil pembulatan bisa membuatnya bergeser lambat. Karena cek kita satu-langkah, drift ini
  dilaporkan tetapi tidak di-gate.

## 3. `e5_summarize.rs`

| Gate akurasi (14) | Gate performa |
|---|---|
| kernel WGSL, kernel kontak, FD (E3 + kontak), identitas hinge dan barrier, bebas penetrasi, oracle konvergen, konsistensi satu langkah, berhenti benar di probe, energi f64 dan f32, replay aturan | Newton ≤ oracle + 2; ms/langkah n65 ≤ 2× E4-final n65; pertumbuhan PCG < 3× |

```rust
/// E4-final n65 ms/step from its summary (the `robust` block that follows the drape-n65 case).
fn baseline_ms(baseline: &Path) -> Result<f64, String>
```

Patokan performa dibaca dari `results/experiment4-final.json` (hasil Part 0), tidak ditulis tangan.

## 4. Menyusun JSON: `Json::extend`

```rust
/// The fields of `self` followed by those of `other`.
pub fn extend(mut self, other: Json) -> Json
```

Penulis JSON minimal kita (`io.rs`) mendapat satu metode baru untuk menggabungkan dua objek. Urutan dijaga: `verdict` E5 ditulis **pertama**, sebelum
JSON E3 yang juga punya `verdict`, karena `jsonscan` mencari kemunculan pertama.
