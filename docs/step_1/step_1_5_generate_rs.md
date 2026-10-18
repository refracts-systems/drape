# Step 1.5: `oracle/src/generate.rs`, Membuat Soal dan Kunci Jawaban

> `oracle generate` adalah langkah pertama setiap kasus. Ia menghasilkan empat file yang menjadi **kontrak** antara Rust
> dan GPU. Dokumen ini juga menjelaskan format biner yang dipakai (`io.rs`) dan perkiraan κ (`lanczos.rs`).

---

## 1. Perintahnya

```bash
./oracle/target/release/oracle generate --case tension --n 257 --out cases/tension-n257 \
    --replace-every 50 --replace-above 1e-2
```

Opsi dibaca `main.rs` (`generate_options`): `--case`, `--n` (kolom grid), `--tol` (default 1e-7), `--max-iter` (default 20000),
dan dua opsi baru dari sesi ini, `--replace-every` dan `--replace-above` (default 0 = mati).

## 2. Alur `run`

```rust
pub fn run(options: &GenerateOptions, dir: &Path) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let problem = build(options.kind, options.columns);
    check_symmetry(&problem)?;
    let problem_arrays = problem_arrays(&problem);
    problem_arrays.write(&dir.join("problem.bin"))?;
    write_text(&dir.join("problem.json"), &problem_json(&problem, options, &problem_arrays.manifest()))?;
    let reference = Reference::compute(&problem, options)?;
    let reference_arrays = reference.arrays(&problem);
    reference_arrays.write(&dir.join("reference.bin"))?;
    write_text(&dir.join("reference.json"), &reference.json(&problem, &reference_arrays.manifest()))?;
    println!("{}", reference.summary(&problem));
    Ok(())
}
```

1. Bangun soal (`cases.rs`).
2. Tolak kalau tidak simetris.
3. Tulis **soal untuk GPU**: `problem.bin` (data) + `problem.json` (metadata + daftar isi biner).
4. Hitung **referensi** (Cholesky dua kali, PCG f64, PCG f32).
5. Tulis **kunci jawaban**: `reference.bin` + `reference.json`.

## 3. Format biner: `BinaryArrays` (io.rs)

Daripada JSON raksasa (array 6 juta angka dalam teks itu lambat dan boros), data besar disimpan sebagai **byte mentah
little-endian**, dan JSON hanya menyimpan "daftar isi"-nya.

```rust
pub fn push_f32(&mut self, name: &str, values: impl Iterator<Item = f32>) {
    let offset = self.align();
    ...
    self.describe(name, "f32", (self.bytes.len() - before) / 4, offset);
}

fn align(&mut self) -> usize {
    while self.bytes.len() % 8 != 0 { self.bytes.push(0); }
    self.bytes.len()
}
```

Setiap array dicatat sebagai `{"name":"vals","dtype":"f32","length":6400809,"offset":...}`.
Offset di-align ke kelipatan 8 byte supaya array f64 pun bisa dibaca langsung tanpa salin ulang.

Analogi: seperti format **Parquet** atau file **tar**: header kecil berisi indeks, lalu blob besar.
Di sisi TypeScript, `problem.ts` memakai offset ini untuk membuat `Float32Array` langsung dari potongan buffer.

### Isi `problem.bin` (untuk GPU)

```rust
arrays.push_u32("row_ptr", ...);
arrays.push_u32("col_idx", ...);
arrays.push_u32("diag_idx", ...);
arrays.push_f32("vals", m.vals.iter().map(|&v| v as f32));
arrays.push_f32("rhs", problem.rhs.iter().map(|&v| v as f32));
```

Perhatikan `as f32`: **pembulatan ke f32 terjadi di sini**, sekali, di sisi oracle. GPU dan emulasi f32 sama-sama
menerima angka yang identik bit-per-bit.

### Isi `problem.json`

Metadata: kasus, ukuran grid, jumlah baris/unknown/blok, regangan, material, dan sejak sesi ini juga
`replace_every` dan `replace_above`. GPU membaca dua nilai terakhir dari sini, **bukan** dari argumen CLI-nya sendiri.
Dengan begitu emulasi CPU dan GPU dijamin memakai pengaturan replacement yang sama: satu sumber kebenaran.

## 4. Referensi: `Reference::compute`

```rust
let factor = BandCholesky::factor(&problem.matrix)?;
let exact = factor.solve(&problem.rhs);
...
let exact_of_rounded_system = BandCholesky::factor(&problem.matrix.rounded_to_f32())?.solve(&rounded_rhs);
let plain = PcgSettings { tolerance: ..., max_iterations: ..., history_stride: 16, replace_every: 0, replace_above: 0.0 };
let pcg64 = solve::<f64>(&problem.matrix, &problem.rhs, &plain);
let pcg32 = solve::<f32>(&problem.matrix, &problem.rhs, &PcgSettings { replace_every: options.replace_every, replace_above: options.replace_above, ..plain });
```

| Hasil | Arti | Dipakai untuk |
|---|---|---|
| `exact` | solusi pasti f64 | galat GPU, residual |
| `exact_of_rounded_system` | solusi pasti dari matriks **yang sudah dibulatkan ke f32** | *storage error*: lantai akurasi f32 |
| `pcg64` | PCG f64 **tanpa** replacement | jumlah iterasi patokan + waktu 1 inti CPU |
| `pcg32` | emulasi GPU **dengan** replacement yang sama | cek kesetiaan GPU |

Keputusan desain penting: PCG f64 **tidak** memakai replacement. f64 tidak membutuhkannya, dan patokan kecepatan
harus PCG f64 "polos" satu inti.

Sintaks `..plain` adalah *struct update* Rust: salin semua field dari `plain` kecuali yang disebut. Mirip spread
`{ ...plain, replace_every: 50 }` di TypeScript.

## 5. Isi `reference.json`

- `cholesky_f64`: waktu, lebar pita, residual Cholesky (bukti referensinya benar).
- `storage_error_f32_system`: `‖x_exact_rounded − x_exact‖ / ‖x_exact‖`.
- `residual_floor_f32_rounded_exact`: residual dari `x_exact` yang **hanya dibulatkan** ke f32. Tidak ada solver f32 yang bisa
  menghasilkan residual lebih kecil dari ini, karena jawaban sempurna pun, kalau disimpan dalam f32, sudah sebesar ini.
- `pcg_f64_cpu`, `pcg_f32_cpu`: iterasi, konvergen, residual sejati, galat, waktu, riwayat.
- `spectrum_jacobi_preconditioned`: κ perkiraan.

Konsep "lantai f32" ini akan muncul lagi di Step 2 dalam bentuk **representation floor**.

## 6. Perkiraan κ: `lanczos.rs`

Menghitung eigenvalue matriks 195 ribu × 195 ribu itu mahal. Tetapi CG secara diam-diam menjalankan algoritma **Lanczos**:
koefisien α dan β membentuk matriks **tridiagonal** kecil yang eigenvalue ekstremnya mendekati eigenvalue ekstrem `M⁻¹A`.

```rust
fn tridiagonal(alphas: &[f64], betas: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let diagonal = ... 1/α_j + β_{j-1}/α_{j-1} ...;
    let offdiagonal = ... √β_j / α_j ...;
}
```

Lalu eigenvalue terkecil dan terbesar dicari dengan **bisection Sturm sequence** (menghitung berapa eigenvalue di bawah suatu
angka, lalu bagi dua intervalnya). κ = terbesar / terkecil. Ini gratis karena α, β sudah dicatat PCG f64.

## 7. Ringkasan cetak

```
tension  unknowns  195075 | cholesky 211.62s | pcg f64 it   1386 res 9.82e-8 | pcg f32 it   1527 res 1.25e-2 err 8.03e-4
```

Baris ini muncul di sesi kerja dan langsung memberi tahu masalahnya **sebelum GPU dijalankan**: emulasi f32 (yang setia
pada GPU) mencapai residual sejati 1,25×10⁻², di atas batas 10⁻². Prediksi ini kemudian dikonfirmasi oleh GPU sungguhan
(1,24×10⁻²).

## 8. Ringkasan

| File | Penulis | Pembaca | Isi |
|---|---|---|---|
| `problem.bin` | generate.rs | GPU | BSR + rhs dalam f32/u32 |
| `problem.json` | generate.rs | GPU, compare.rs | metadata, manifest, pengaturan replacement |
| `reference.bin` | generate.rs | compare.rs | matriks f64, `x_exact`, solusi emulasi f32 |
| `reference.json` | generate.rs | compare.rs, summarize | statistik referensi |
