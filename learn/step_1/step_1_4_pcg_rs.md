# Step 1.4: `oracle/src/pcg.rs`, Algoritma PCG, Emulasi f32, dan Residual Replacement

> Ini dokumen algoritma paling penting di Step 1. Semua kernel WGSL di dokumen 1.7–1.13 adalah **terjemahan paralel**
> dari kode di file ini. Kalau kamu paham `pcg.rs`, kernel GPU tinggal soal "bagaimana membagi pekerjaan ke ribuan thread".

---

## 1. Ide CG dalam bahasa sehari-hari

Kita mencari `x` yang memenuhi `A x = b`. Untuk `A` simetris positif definit, ini sama dengan mencari titik terendah dari
"mangkuk" energi `½ xᵀA x − bᵀx`.

**Conjugate Gradient (CG)** turun ke dasar mangkuk dengan cara pintar:

1. Mulai dari tebakan `x = 0`. Hitung **residual** `r = b − A x` (seberapa jauh dari benar; awalnya `r = b`).
2. Pilih **arah** `p` untuk melangkah (awalnya searah residual).
3. Hitung **panjang langkah optimal** `α` sepanjang `p`.
4. Melangkah: `x ← x + α p`. Perbarui residual: `r ← r − α A p` (tanpa menghitung ulang `A x`).
5. Pilih arah baru yang **"konjugat"** dengan semua arah sebelumnya: `p ← r + β p`. "Konjugat" kira-kira berarti
   langkah baru tidak merusak kemajuan yang sudah dicapai di arah lama.
6. Ulangi sampai `‖r‖` cukup kecil.

Dalam aritmetika sempurna, CG selesai dalam paling banyak N langkah. Dalam praktik, kita berhenti jauh lebih awal saat
residual sudah turun 10⁷ kali (`tolerance = 1e-7`).

## 2. "P" di PCG: preconditioner

Kecepatan CG bergantung pada **bilangan kondisi** κ: perbandingan antara arah tercuram dan terlandai di mangkuk.
Mangkuk lonjong (κ besar) butuh banyak iterasi. Jumlah iterasi kira-kira sebanding dengan √κ.

**Preconditioner** `M ≈ A` yang mudah dibalik mengubah soal supaya mangkuknya lebih bulat. Di sini dipakai yang paling sederhana:
**block-Jacobi**, yaitu `M` = blok-blok 3×3 diagonal dari `A`. Membaliknya cukup membalik 65 ribu matriks 3×3 kecil, sepenuhnya paralel.

Dengan preconditioner, muncul vektor `z = M⁻¹ r` ("residual yang sudah dinormalkan") dan rumus α, β memakai `rᵀz`.

## 3. Algoritma lengkap (seperti di kode)

```
x = 0, r = b, z = M⁻¹ r, p = z
rz = rᵀz, rr = rᵀr, rr0 = rr
loop:
    q    = A p                       ← SpMV, operasi termahal
    pq   = pᵀq
    α    = rz / pq
    x   += α p
    r   -= α q
    z    = M⁻¹ r
    rz'  = rᵀz,  rr = rᵀr
    β    = rz' / rz,  rz = rz'
    stop kalau rr ≤ tol² · rr0
    p    = z + β p
```

Ada tiga **reduksi** (penjumlahan seluruh vektor jadi satu angka) per iterasi: `pᵀq`, `rᵀz`, dan `rᵀr`.
Di GPU, reduksi adalah bagian yang paling rumit (dokumen 1.8 dan 1.10).

## 4. Generik atas f32 dan f64

```rust
pub trait Real: Copy + PartialOrd + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> {
    fn zero() -> Self;
    fn of(v: f64) -> Self;
    fn wide(self) -> f64;
}
impl Real for f32 { ... }
impl Real for f64 { ... }

pub fn solve<T: Real>(matrix: &Bsr, rhs: &[f64], settings: &PcgSettings) -> PcgReport
```

Satu implementasi, dua tipe. `solve::<f64>` adalah PCG presisi tinggi; `solve::<f32>` adalah **emulasi GPU**: setiap
penjumlahan dan perkalian dibulatkan ke f32 tepat di tempat yang sama dengan kernel WGSL.

Analogi: seperti menulis fungsi uang yang generik atas `Decimal` dan `float`, lalu menjalankan keduanya untuk melihat
seberapa besar galat pembulatan `float`.

## 5. Bagian terpenting: urutan operasi

Penjumlahan float **tidak asosiatif**: `(a + b) + c` bisa berbeda bit dari `a + (b + c)`. Kalau emulasi menjumlahkan dalam urutan
lain dari GPU, hasilnya akan berbeda sedikit dan kita tidak bisa membedakan "GPU salah" dari "urutannya saja beda".

Karena itu reduksi di Rust meniru **struktur pohon GPU** secara persis:

```rust
pub const WORKGROUP: usize = 256;

pub fn reduce_rows<T: Real>(rows: usize, value: impl Fn(usize) -> T) -> T {
    let groups = rows.div_ceil(WORKGROUP);
    let partials: Vec<T> = (0..groups)
        .map(|g| {
            let mut lane: Vec<T> = (0..WORKGROUP).map(|t| row_or_zero(g * WORKGROUP + t, rows, &value)).collect();
            tree(&mut lane)
        })
        .collect();
    let mut lane: Vec<T> = (0..WORKGROUP).map(|t| strided_sum(&partials, t)).collect();
    tree(&mut lane)
}
```

Tiga tahap, sama persis dengan GPU:

1. **Per workgroup** (256 baris): setiap baris menghitung nilai 3-komponennya (`row_dot`), lalu 256 nilai dijumlahkan dengan
   **pohon** (`tree`): 128 pasangan, lalu 64, 32, …, 1. Hasilnya satu angka "partial" per workgroup.
2. **Strided sum**: 256 "lane" (jalur) menjumlahkan partial dengan lompatan 256: lane `t` menjumlahkan partial `t`, `t+256`,
   `t+512`, …
3. **Pohon terakhir** atas 256 lane.

```rust
fn tree<T: Real>(lane: &mut [T]) -> T {
    let mut stride = WORKGROUP / 2;
    while stride > 0 {
        for t in 0..stride {
            lane[t] = lane[t] + lane[t + stride];
        }
        stride /= 2;
    }
    lane[0]
}
```

Bahkan `row_dot` menjumlahkan tiga komponen dalam urutan `a0*b0 + a1*b1 + a2*b2` yang sama dengan WGSL,
dan perkalian SpMV memakai tanda kurung yang sama (`acc + (v0*p0 + v1*p1 + v2*p2)`).

Pembalikan 3×3 (`invert3`) juga mengikuti urutan *adjugate* yang sama dengan `jacobi_invert.wgsl`.

## 6. Satu iterasi: `State::step`

```rust
fn step(&mut self, system: &System<T>) -> (f64, f64) {
    self.q = (0..system.rows()).flat_map(|row| system.multiply_row(row, &self.p)).collect();
    let pq = reduce_rows(system.rows(), |row| row_dot(&self.p, &self.q, row));
    let alpha = self.rz / pq;
    for i in 0..self.x.len() {
        self.x[i] = self.x[i] + alpha * self.p[i];
        self.r[i] = self.r[i] - alpha * self.q[i];
    }
    self.z = (0..system.rows()).flat_map(|row| system.precondition_row(row, &self.r)).collect();
    let rz_new = reduce_rows(system.rows(), |row| row_dot(&self.r, &self.z, row));
    self.rr = reduce_rows(system.rows(), |row| row_dot(&self.r, &self.r, row));
    let beta = rz_new / self.rz;
    self.rz = rz_new;
    (alpha.wide(), beta.wide())
}
```

Peta ke kernel GPU:

| Baris Rust | Kernel WGSL |
|---|---|
| `q = A p`, partial `pᵀq` | `spmv_dot.wgsl` |
| `α = rz / pq` | `reduce.wgsl` MODE 1 |
| `x += αp`, `r −= αq`, `z = M⁻¹r`, partial `rᵀz`, `rᵀr` | `update_dot.wgsl` |
| `β`, cek konvergensi | `reduce.wgsl` MODE 2 |
| `p = z + βp` | `direction.wgsl` |

## 7. Residual rekursif vs residual sejati

Perhatikan bahwa `r` **tidak pernah dihitung ulang** sebagai `b − A x`. Ia diperbarui secara rekursif: `r ← r − α q`.
Secara matematika keduanya sama. Dalam f32, galat pembulatan membuat keduanya **pelan-pelan berpisah**.

Akibatnya PCG f32 bisa melaporkan "konvergen, residual 10⁻⁷" padahal residual sejati `‖b − A x‖/‖b‖` sudah melayang ke 10⁻².
Inilah temuan utama Experiment 1:

| n257 tension | residual rekursif | residual sejati (f64) |
|---|---|---|
| f32 tanpa replacement | 9,97×10⁻⁸ | **1,25×10⁻²** (gagal gate 10⁻²) |

## 8. Residual replacement (ditambahkan di sesi ini)

Obatnya klasik (van der Vorst & Ye): sesekali **ganti** residual rekursif dengan residual sejati.

```rust
pub struct PcgSettings {
    pub tolerance: f64,
    pub max_iterations: usize,
    pub history_stride: usize,
    /// Residual replacement r = b − Ax every this many iterations (0 = never), as residual.wgsl in REPLACE mode.
    pub replace_every: usize,
    /// Replacement only while the recursive relative residual is above this (keeps the f32 rounding floor from resetting r forever).
    pub replace_above: f64,
}
```

Di loop utama:

```rust
if !converged {
    state.update_direction(beta);
}
if !converged && replaces(settings, report.iterations, &state) {
    state.replace_residual(&system);
}
```

```rust
fn replaces<T: Real>(settings: &PcgSettings, iterations: usize, state: &State<T>) -> bool {
    let due = settings.replace_every > 0 && iterations % settings.replace_every == 0;
    due && state.rr > T::of(settings.replace_above * settings.replace_above) * state.rr0
}

fn replace_residual(&mut self, system: &System<T>) {
    self.r = ... rhs − A x per baris ...;
    self.z = ... M⁻¹ r ...;
    self.rz = reduce_rows(... rᵀz ...);
    self.rr = reduce_rows(... rᵀr ...);
}
```

Arah `p` **dipertahankan**; hanya `r`, `z`, `rz`, dan `rr` yang disegarkan.

### Kenapa harus ada "gate" `replace_above`?

Percobaan pertama: replacement **setiap 50 iterasi tanpa syarat**. Hasilnya tidak pernah konvergen. Alasannya:

- `b − A x` dihitung dalam f32 punya **lantai pembulatan**, sekitar 10⁻⁵ relatif di n33 dan 4×10⁻⁴ di n257.
- Setiap replacement mereset residual rekursif ke lantai itu.
- 50 iterasi tidak cukup untuk turun dari lantai ke 10⁻⁷ sebelum replacement berikutnya mereset lagi.

Solusinya: replacement hanya **selama residual masih besar** (> 10⁻², sama dengan forcing term). Galat pemisahan
residual sebanding dengan residual terbesar sejak replacement terakhir. Jadi setelah replacement terakhir di level 10⁻²,
galat sisanya jauh lebih kecil.

Hasil sapuan parameter di CPU (emulasi f32 identik GPU), n257:

| Gate | tension: iterasi / residual sejati | dynamic: iterasi / residual sejati |
|---|---|---|
| mati | 1527 / 1,25×10⁻² | 1494 / 6,0×10⁻³ |
| tiap 50 selama > 10⁻² | 1608 / **7,8×10⁻³** | 1495 / 4,8×10⁻³ |
| tiap 50 selama > 10⁻³ | divergen (NaN) | divergen (NaN) |

Gate yang lebih rendah merusak sifat konjugat CG dan memicu divergensi. Nilai 10⁻² dipilih dari **definisi gate-nya sendiri**
(forcing level), bukan dari "coba-coba sampai lulus".

### Tes unit

```rust
#[test]
fn residual_replacement_keeps_f32_pcg_on_the_true_residual() {
    let problem = build(CaseKind::Tension, 33);
    ...
    let replaced = solve::<f32>(&problem.matrix, &problem.rhs, &settings(10));
    assert!(replaced.converged);
    assert!(true_residual(&replaced) < true_residual(&plain), ...);
}
```

Periode 10 dipakai di tes karena di n33 residual sudah turun di bawah 10⁻² sebelum iterasi ke-50, sehingga dengan
periode 50 replacement tidak pernah terjadi dan tes membandingkan dua run yang identik.

## 9. Laporan yang dihasilkan

```rust
pub struct PcgReport {
    pub iterations: usize,
    pub converged: bool,
    pub recursive_relative_residual: f64,
    pub solution: Vec<f64>,
    pub alphas: Vec<f64>,
    pub betas: Vec<f64>,
    pub history: Vec<(usize, f64)>,
    pub seconds: f64,
}
```

- `alphas`, `betas` dipakai `lanczos.rs` untuk **memperkirakan κ** tanpa menghitung eigenvalue secara langsung
  (koefisien CG secara matematis membentuk matriks tridiagonal Lanczos).
- `seconds` dari PCG **f64** adalah patokan kecepatan "1 inti CPU" untuk gate `speedup_ok`.

## 10. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| CG | turun ke dasar mangkuk energi dengan arah-arah konjugat |
| Preconditioner Jacobi blok | balik blok diagonal 3×3, paralel sempurna |
| Emulasi f32 | PCG yang sama dengan urutan operasi persis seperti GPU |
| Reduksi pohon | 256-lebar per workgroup, strided, lalu pohon terakhir |
| Residual rekursif | murah, tetapi menyimpang dari residual sejati dalam f32 |
| Residual replacement | segarkan `r = b − Ax` tiap 50 iterasi, **hanya selama** residual > 10⁻² |
