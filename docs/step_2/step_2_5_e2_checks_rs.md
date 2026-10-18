# Step 2.5: `oracle/src/e2_checks.rs`, Patch Test, Cek Beda Hingga, dan Jalur Bolak-balik

> Tiga cara independen untuk membuktikan bahwa fisika yang ditulis tangan itu benar, sebelum membandingkannya dengan GPU.
> Fungsi-fungsi ini dipakai dua kali: pada solusi f64 (oleh generate/compare) dan pada solusi GPU f32 (oleh compare).

---

## 1. Patch test

### Idenya

Kalau batas kain digeser mengikuti peta **afin** (peregangan + geser seragam), jawaban fisik yang benar adalah **seluruh** kain mengikuti peta itu,
dengan regangan konstan di setiap elemen, walaupun mesh-nya acak. Elemen yang tidak lolos patch test tidak konvergen ke jawaban benar saat mesh
diperhalus. Ini salah satu tes paling tua di FEM.

Analogi backend: *invariant test*. Kalau input "seragam", output harus seragam juga, terlepas dari cara data dipartisi.

### Kode

```rust
pub fn patch_check(coupon: &Coupon, x: &[Vec3]) -> PatchCheck {
    let f = coupon.path.boundary_gradient(1.0);
    let affine = coupon.affine(1.0);
    let displacement = ...max |x − affine|...;
    let exact_columns = ([f[0][0], f[1][0], 0.0], [f[0][1], f[1][1], 0.0]);
    let exact_strain = green_strain(exact_columns);
    ...
    for element in &coupon.elements {
        let columns = element.columns(x);
        gradient_error = gradient_error.max(max_difference(&[columns.0, columns.1], &[exact_columns.0, exact_columns.1]));
        strain_error = strain_error.max(max_difference(&[element.strain(x)], &[exact_strain]) / scale);
    }
    PatchCheck { displacement_error: displacement / COUPON_SIZE, deformation_gradient_error: gradient_error, strain_error }
}
```

Tiga ukuran:

| Ukuran | Arti |
|---|---|
| `displacement_error_per_coupon_size` | posisi terjauh dari peta afin, dibagi ukuran kupon |
| `deformation_gradient_error` | F setiap elemen vs F afin (mencakup item 1: F) |
| `relative_strain_error` | E setiap elemen vs E afin, relatif (mencakup item 1: Green strain) |

### Hasil

| | n33 | n65 |
|---|---|---|
| f64: perpindahan / F / regangan | 3,5×10⁻¹⁶ / 3,4×10⁻¹⁴ / 2,3×10⁻¹³ | 1,4×10⁻¹⁶ / 1,4×10⁻¹⁴ / 1,5×10⁻¹³ |
| f32 GPU: perpindahan | 3,2×10⁻⁶ | 2,2×10⁻⁷ |

f64 tepat sampai pembulatan mesin. f32 berhenti di residual 10⁻⁶ (toleransinya), sehingga galatnya lebih besar, tetapi tetap jauh di bawah batas 10⁻³.

## 2. Cek beda hingga (finite difference) global

Tes unit di `membrane.rs` memeriksa **satu elemen**. Cek ini memeriksa **sistem rakitan**: perakitan, masking, dan pemetaan baris semuanya ikut diuji.

### Keadaan acak

```rust
fn random_state(coupon: &Coupon, rng: &mut Lcg) -> Vec<Vec3> {
    let spacing = COUPON_SIZE / (coupon.mesh.columns - 1) as f64;
    let mut x = coupon.affine(0.5);
    coupon.free.iter().for_each(|&v| x[v] = mask_vector([x[v][0] + 0.05 * spacing * rng.uniform(), ...]));
    x
}
```

Setengah beban + derau acak 5% jarak grid. Keadaan yang "tidak istimewa" supaya tidak ada suku yang kebetulan nol.

`Lcg` (generator acak deterministik) dipindah dari modul tes ke `rng.rs` di Step 2, karena cek ini berjalan di binary oracle, bukan hanya di `cargo test`.

### Tiga cek

```rust
fn gradient_error(coupon: &Coupon, x: &[Vec3], v: &[Vec3]) -> f64 {
    let fd = (coupon.energy(&shifted(x, v, FD_STEP)) - coupon.energy(&shifted(x, v, -FD_STEP))) / (2.0 * FD_STEP);
    let exact = free_dot(&coupon.free_gradient(x), &rows_of(coupon, v));
    (fd - exact).abs() / exact.abs()
}
```

- **Gradien**: turunan berarah energi `(Π(x+hv) − Π(x−hv)) / 2h` vs `g·v`.
- **Tangen**: `(g(x+hv) − g(x−hv)) / 2h` vs `K v`, sebagai norma relatif vektor.
- **Perubahan energi**: `Π(x + αv) − Π(x)` langsung vs `energy_change`. Ini membuktikan identitas StVK pada sistem penuh.

Diulang untuk 6 arah acak; dilaporkan galat terbesar. `FD_STEP = 10⁻⁷` m: cukup kecil supaya galat pemotongan O(h²) kecil, cukup besar supaya galat
pembulatan (ε/h) tidak mendominasi.

### Hasil (batas: gradien 10⁻⁶, tangen 10⁻⁵, identitas 10⁻¹²)

| Kupon | gradien | tangen | identitas energi |
|---|---|---|---|
| patch-n33 | 8,3×10⁻⁸ | 3,0×10⁻⁹ | 1,7×10⁻¹⁵ |
| uniaxial-n33 | 2,0×10⁻⁸ | 1,8×10⁻⁹ | 5,7×10⁻¹⁵ |
| biaxial-n33 | 1,1×10⁻⁸ | 1,9×10⁻⁹ | 3,4×10⁻¹⁵ |

## 3. Jalur bolak-balik (loading/unloading)

### Idenya

StVK adalah material **hiperelastik**: tegangan hanya bergantung pada regangan sekarang, bukan riwayat. Jadi saat dilepas, kurva tegangan-regangan harus
**menelusuri ulang** kurva saat ditarik, tanpa histeresis, dan berakhir tepat di rest. Ini pemeriksaan fisika sekaligus pemeriksaan solver
(solver yang tidak konvergen akan meninggalkan "jejak").

```rust
pub fn path_independence(coupon: &Coupon, positions: &[Vec<Vec3>]) -> PathCheck {
    let factors = coupon.path.load_factors();
    let stresses: Vec<[f64; 2]> = positions.iter().map(|x| coupon.grip_stress(x)).collect();
    let peak = ...;
    let gap = (half..factors.len()).map(|u| hysteresis(&factors, &stresses, u)).fold(0.0, f64::max);
    let final_displacement = max_difference(&positions[factors.len() - 1], &rest) / COUPON_SIZE;
    PathCheck { peak_stress: peak, stress_gap: gap / peak, final_displacement }
}

fn hysteresis(factors: &[f64], stresses: &[[f64; 2]], unload: usize) -> f64 {
    let matching = (0..unload).find(|&l| (factors[l] - factors[unload]).abs() < 1e-12);
    let load = matching.map(|l| stresses[l]).unwrap_or([0.0, 0.0]);
    ...
}
```

Untuk setiap langkah turun (misalnya t = 0,6), cari langkah naik dengan t yang sama, lalu bandingkan tegangannya. Langkah terakhir t = 0 dibandingkan
dengan tegangan nol (rest).

Fungsi ini mula-mula menerima `&[StepSolution]`; di-refactor menjadi `&[Vec<Vec3>]` (hanya posisi) supaya bisa dipakai untuk posisi GPU juga.

### Hasil

| | puncak P | celah turun-vs-naik | kembali ke rest |
|---|---|---|---|
| uniaxial-n33 f64 | 260,1 N/m | 5,7×10⁻¹⁵ | 1,9×10⁻¹⁵ |
| uniaxial-n33 f32 | 260,1 N/m | 3,7×10⁻⁶ | 4,5×10⁻⁸ |
| biaxial-n33 f64 | 121,3 N/m | 1,0×10⁻¹⁴ | 5,6×10⁻¹⁵ |
| biaxial-n33 f32 | 121,3 N/m | 4,8×10⁻⁶ | 4,5×10⁻⁸ |

## 4. Ringkasan

| Cek | Membuktikan | Item permintaan |
|---|---|---|
| Patch | F dan E direproduksi tepat untuk batas afin | 1, 5 |
| FD gradien | gaya = turunan energi | 3, 6 |
| FD tangen | tangen = turunan gaya | 3, 6 |
| Identitas energi | rumus line search eksak | 4 |
| Bolak-balik | hiperelastik + solver konvergen | 7 |
