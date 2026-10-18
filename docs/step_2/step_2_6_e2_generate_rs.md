# Step 2.6: `oracle/src/e2_generate.rs`, File Soal Step 2 dan Peta "Gather" untuk Perakitan Tanpa Atomik

> Seperti `generate.rs` di Step 1, file ini membuat kontrak Rust↔GPU. Bagian paling menarik adalah **peta kontribusi**:
> cara memungkinkan GPU merakit matriks global secara deterministik tanpa operasi atomik.

---

## 1. Konstanta: semua pengaturan di satu tempat

```rust
/// GPU f32 Newton: stop at ‖g‖/‖g₀‖ ≤ 1e-6 per load step.
pub const NEWTON_TOLERANCE: f64 = 1e-6;
/// The f64 oracle runs to its own floor so its solution is the reference for the f32 one.
const REFERENCE_TOLERANCE: f64 = 1e-12;
const MAX_NEWTON_ITERATIONS: usize = 30;
const ARMIJO: f64 = 1e-4;
const MAX_HALVINGS: usize = 30;
/// The E1 solver unchanged: PCG to 1e-7 with gated residual replacement.
const PCG_TOLERANCE: f64 = 1e-7;
const PCG_MAX_ITERATIONS: usize = 20000;
const REPLACE_EVERY: usize = 50;
const REPLACE_ABOVE: f64 = 1e-2;
```

Pengaturan PCG **persis** pengaturan akhir E1 ("existing E1 solver, residual replacement, forcing term 1e-2"). Semua nilai ditulis ke `problem.json`,
sehingga GPU tidak punya angka ajaib sendiri.

## 2. Alur

```rust
pub fn run(path: LoadPath, columns: usize, dir: &Path) -> Result<(), String> {
    let coupon = Coupon::build(path, columns);
    let steps = solve_path(&coupon, &reference_settings())?;
    let problem = problem_arrays(&coupon);
    problem.write(&dir.join("problem.bin"))?;
    write_text(&dir.join("problem.json"), &problem_json(&coupon, &problem.manifest()))?;
    ... reference.bin: posisi f64 setiap langkah ...
    write_text(&dir.join("reference.json"), &reference_json(&coupon, &steps, &reference.manifest()))?;
    println!("{}", summary(&coupon, &steps));
    Ok(())
}
```

## 3. Isi `problem.bin`

```rust
arrays.push_f32("rest_positions", ...);       // V×3
arrays.push_f32("initial_positions", ...);    // V×3 (patch: interior diganggu)
arrays.push_f32("step_gradients", ...);       // S×4: F(t) per langkah untuk prediktor
arrays.push_u32("elem_vertices", ...);        // T×3
arrays.push_f32("elem_geom", ...);            // T×7: grad_n (6) + luas (1)
arrays.push_u32("row_vertex", ...);           // baris → titik
arrays.push_u32("dof_mask", ...);             // per baris: bit komponen bebas (0b011)
push_pattern(...);                            // row_ptr, col_idx, diag_idx
push_contributions(..., "block", ...);        // peta perakitan matriks
push_contributions(..., "force", ...);        // peta perakitan gaya
arrays.push_u32("fixed_vertex", ...);         // titik jepit
arrays.push_f32("fixed_positions", ...);      // S×F×3: posisi jepit per langkah
arrays.push_f32("kernel_state", ...);         // keadaan uji kernel
```

| Array | Kenapa ada |
|---|---|
| `elem_geom` | grad_n dan luas hanya bergantung pada geometri rest → dihitung sekali dalam f64, dikirim sebagai f32 |
| `dof_mask` | bit per komponen: x=1, y=2, z=4. Kupon: `0b011` (z terkunci). Umum, supaya kelak bisa mengunci komponen lain |
| `step_gradients` | GPU menjalankan prediktor yang sama dengan oracle |
| `fixed_positions` | posisi jepit dihitung oracle dalam f64 lalu dibulatkan, bukan dihitung ulang di GPU |
| `kernel_state` | keadaan acak teregang untuk membandingkan kernel elemen WGSL vs f64 |

## 4. Masalah perakitan di GPU

Di CPU, perakitan = loop elemen dengan `+=` ke matriks global. Di GPU, satu thread per elemen akan menulis ke blok global yang sama dengan thread elemen
tetangga. Ini **race condition** (dokumen 1.8). WGSL tidak punya atomic add untuk f32. Atomik juga membuat urutan penjumlahan acak, sehingga tidak
deterministik.

**Solusinya: balik arahnya.** Bukan "setiap elemen *mendorong* (scatter) ke blok", tetapi "setiap blok *menarik* (gather) dari elemen-elemen yang
menyumbang padanya". Satu thread per **baris**, dan setiap thread hanya menulis blok-blok miliknya sendiri, sehingga tanpa race.

Supaya bisa menarik, thread harus tahu **siapa saja** penyumbangnya. Peta itu dibuat oracle sekali, di CPU.

### Peta blok

```rust
/// For each BSR block, the element Hessian blocks (index 9e + 3k + l) that sum into it, in element order.
fn block_contributions(coupon: &Coupon) -> Vec<Vec<u32>> {
    let mut lists = vec![Vec::new(); coupon.pattern.blocks()];
    for (e, element) in coupon.elements.iter().enumerate() {
        for (k, l) in (0..3).flat_map(|k| (0..3).map(move |l| (k, l))) {
            if let (Some(r), Some(c)) = (coupon.row_of[element.vertices[k]], coupon.row_of[element.vertices[l]]) {
                lists[coupon.pattern.position(r, c)].push((9 * e + 3 * k + l) as u32);
            }
        }
    }
    lists
}
```

Untuk blok global ke-b, daftarnya berisi indeks blok lokal elemen `9e + 3k + l` ("elemen e, blok (k,l) dari Hessian 3×3-nya"). Di GPU, Hessian elemen
disimpan di `elem_hess[81e + 9(3k+l) + …]`, jadi indeks ini langsung menunjuk ke sana.

**Urutan elemen**: daftar dibangun dengan loop elemen naik, sehingga urutan penjumlahan di GPU **sama** dengan urutan `Bsr::add` di oracle. Ini membuat
hasil deterministik dan sebanding.

### Peta gaya

Sama, untuk vektor: setiap baris punya daftar `3e + k` ("elemen e, titik lokal k").

### Format CSR untuk daftar

```rust
fn push_contributions(arrays: &mut BinaryArrays, name: &str, lists: &[Vec<u32>]) {
    let offsets = std::iter::once(0).chain(lists.iter().scan(0u32, |sum, l| { *sum += l.len() as u32; Some(*sum) }));
    arrays.push_u32(&format!("{name}_contrib_ptr"), offsets);
    arrays.push_u32(&format!("{name}_contrib_idx"), lists.iter().flatten().copied());
}
```

Daftar-daftar dengan panjang berbeda disimpan dengan trik yang sama seperti `row_ptr`/`col_idx` BSR: satu array offset (panjang+1) dan satu array isi
yang dirangkai. `scan` adalah prefix sum (jumlah kumulatif).

Analogi: ini **materialized view** atau **inverted index**. Mahal dibangun sekali, membuat query (perakitan) per iterasi murah dan tanpa lock.

## 5. `problem.json`

Berisi kasus, jalur, ukuran (titik, segitiga, baris, unknown, blok, jepit, langkah), faktor beban, material (4 modulus), pengaturan Newton dan PCG, plus
manifest array. Contoh untuk uniaxial-n33: 1089 titik, 2048 segitiga, 1023 baris, 3069 unknown, 20 langkah.

## 6. `reference.json` dan `reference.bin`

- `reference.bin`: posisi f64 semua titik di akhir setiap langkah.
- `reference.json`: per langkah: faktor beban, regangan, tegangan grip, status, iterasi Newton, iterasi sampai 10⁻⁶, riwayat residual, panjang langkah,
  jumlah pemotongan. Ditambah `derivative_check` dan `checks` (patch untuk jalur patch, bolak-balik untuk lainnya).

## 7. Ringkasan cetak

```
uniaxial n=33 rows=1023 | f64 Newton iterations per step [3 3 3 ...] | worst final residual 6.77e-14
```

## 8. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Satu sumber pengaturan | semua toleransi ditulis ke problem.json |
| Geometri rest di oracle | grad_n dan luas dihitung f64, dikirim f32 |
| Scatter → gather | setiap baris menarik kontribusinya → tanpa race, tanpa atomik |
| Urutan elemen | sama dengan oracle → deterministik |
| Peta CSR | offset + isi, seperti row_ptr/col_idx |
