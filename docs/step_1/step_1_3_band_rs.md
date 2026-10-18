# Step 1.3: `oracle/src/band.rs`, Cholesky Pita sebagai Jawaban Pasti

> Untuk menilai solver GPU, kita butuh jawaban yang **tidak diragukan**. `band.rs` adalah penyelesai langsung (direct solver)
> f64 yang sama sekali tidak memakai kode iteratif, jadi bug di PCG tidak bisa ikut "menyalin" diri ke kunci jawaban.

---

## 1. Kenapa tidak pakai PCG f64 saja sebagai referensi?

Karena itu seperti menguji fungsi dengan dirinya sendiri. Kalau ada kesalahan konsep di PCG (misalnya preconditioner salah),
PCG f64 dan PCG f32 bisa sama-sama salah ke arah yang sama, dan perbandingan akan tampak "cocok".

Prinsip yang dipakai: **referensi harus berasal dari jalur kode yang independen**. Komentar di file ini menyatakannya:
*"Independent of any Krylov code path."* (Krylov = keluarga metode iteratif tempat CG/PCG berada.)

## 2. Apa itu Cholesky?

Untuk matriks simetris positif definit `A`, selalu ada matriks segitiga bawah `L` sehingga:

```
A = L · Lᵀ
```

Setelah `L` diketahui, `A x = b` diselesaikan dalam dua sapuan murah:

1. **Substitusi maju**: selesaikan `L y = b` dari atas ke bawah.
2. **Substitusi mundur**: selesaikan `Lᵀ x = y` dari bawah ke atas.

Analogi: memfaktorkan `A` itu seperti membangun **index** di database. Mahal sekali di awal, tetapi setelahnya setiap
"query" (penyelesaian dengan `b` berbeda) murah.

## 3. Kenapa "pita" (band)?

Penomoran titik pada grid adalah baris demi baris. Titik `v` hanya terhubung ke titik yang nomornya paling jauh
"satu baris grid + 1" darinya. Jadi semua entri bukan nol di matriks berada dalam **pita** selebar `w` di sekitar diagonal.

Sifat penting: faktor Cholesky dari matriks pita **juga** pita dengan lebar yang sama. Isi di luar pita tetap nol.
Kita cukup menyimpan pita itu:

```rust
pub struct BandCholesky {
    size: usize,        // jumlah unknown (3 × baris)
    bandwidth: usize,   // lebar pita
    lower: Vec<f64>,    // L, disimpan per baris: bandwidth+1 angka per baris
}
```

Memori = `size × (bandwidth + 1) × 8 byte`. Di n257: 195.075 × ~780 × 8 ≈ **1,2 GB**, dan menghitungnya butuh
beberapa menit. Inilah alasan README menyebut oracle n257 butuh ~2,5 GB RAM dan ~5 menit: Cholesky dijalankan
**dua kali** (matriks asli dan matriks yang dibulatkan ke f32).

## 4. Menghitung lebar pita

```rust
fn band_of(matrix: &Bsr) -> usize {
    let mut widest = 0;
    for row in 0..matrix.rows() {
        for k in matrix.row_ptr[row]..matrix.row_ptr[row + 1] {
            let column = matrix.col_idx[k];
            if column <= row {
                widest = widest.max(3 * (row - column) + 2);
            }
        }
    }
    widest
}
```

Jarak blok terjauh dari diagonal dalam satuan blok, dikali 3 (karena tiap blok 3 skalar), plus 2 (posisi terjauh di dalam blok).

## 5. Layout memori pita

```rust
fn index(&self, i: usize, j: usize) -> usize {
    i * (self.bandwidth + 1) + (j + self.bandwidth - i)
}
```

Entri `L[i][j]` (dengan `i − bandwidth ≤ j ≤ i`) disimpan di baris `i` pada kolom lokal `j − i + bandwidth`.
Diagonal `L[i][i]` berada di posisi terakhir tiap baris. Ini trik layout klasik untuk menghemat memori tanpa hash map.

## 6. Faktorisasi baris demi baris

```rust
fn factor_row(&mut self, i: usize) -> Result<(), String> {
    for j in self.first(i)..=i {
        let start = self.first(i).max(self.first(j));
        ...
        let product: f64 = (0..overlap).map(|k| self.lower[ri + k] * self.lower[rj + k]).sum();
        let value = self.lower[self.index(i, j)] - product;
        self.lower[at] = if i == j { positive_sqrt(value, i)? } else { value / self.lower[self.index(j, j)] };
    }
    Ok(())
}
```

Rumus Cholesky standar:

- `L[i][j] = (A[i][j] − Σ L[i][k]·L[j][k]) / L[j][j]` untuk `j < i`
- `L[i][i] = √(A[i][i] − Σ L[i][k]²)`

Jumlah Σ hanya menjalar di bagian yang tumpang tindih antara pita baris `i` dan baris `j`.

Kalau angka di bawah akar ≤ 0, matriks **tidak positif definit**, dan `positive_sqrt` mengembalikan error dengan nomor barisnya.
Pesan inilah yang muncul di Step 2 saat tangen pertama tidak positif definit, dan yang memaksa adanya **prediktor**
(dokumen 2.3).

## 7. Substitusi

```rust
pub fn solve(&self, rhs: &[f64]) -> Vec<f64> {
    let mut y = rhs.to_vec();
    for i in 0..self.size { ... y[i] = (y[i] - s) / L[i][i]; }        // maju
    for i in (0..self.size).rev() { ... y[i] = (y[i] - s) / L[i][i]; } // mundur
    y
}
```

Vektor yang sama dipakai ulang di tempat (*in place*) untuk dua sapuan.

## 8. Seberapa "pasti"?

Tes:

```rust
#[test]
fn cholesky_solves_tension_case() {
    ...
    assert!(residual < 1e-12, "{residual}");
}
```

Di n257, `reference.json` mencatat `cholesky_f64.true_relative_residual` sekitar 10⁻¹⁴ sampai 10⁻¹⁵. Itu 12 orde
besaran lebih kecil daripada gate solver GPU (10⁻²). Jadi saat kita bilang "galat GPU vs f64 = 8,4×10⁻⁴", kita yakin
angka itu adalah galat GPU, bukan galat referensi.

## 9. Dipakai lagi di Step 2

Di Step 2, Newton f64 oracle memakai `BandCholesky` sebagai langkah linear **pasti** di setiap iterasi Newton.
Kontrasnya dengan GPU, yang memakai PCG f32 tidak pasti (*inexact*). Perbandingan keduanya menjawab pertanyaan
"apakah Newton tetap konvergen kuadratik walau langkah linearnya tidak pasti?".

## 10. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Direct solver | jawaban dalam jumlah langkah tetap, tanpa iterasi |
| Cholesky | `A = L Lᵀ`, lalu dua substitusi |
| Pita | entri bukan nol dekat diagonal; faktor juga pita |
| Kenapa independen | referensi tidak boleh berbagi kode dengan yang diuji |
| Biaya | memori dan waktu besar; cocok untuk oracle, tidak untuk produksi |
