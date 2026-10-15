# Step 1.1: `oracle/src/bsr.rs`, Format Matriks Jarang Blok 3×3

> File ini kecil, tetapi ia **bahasa bersama** antara oracle (Rust) dan GPU (WGSL). Setiap kernel yang menyentuh matriks
> membaca format yang didefinisikan di sini. Paham BSR = paham separuh kernel di Step 1.

---

## 1. Masalah: matriks 195 ribu × 195 ribu

Kalau matriks `A` di n257 disimpan penuh (dense): 195.075 × 195.075 angka × 4 byte ≈ **152 GB**. Tidak mungkin.

Untungnya hampir semua isinya **nol**. Titik kain hanya "merasakan" tetangganya yang terhubung lewat segitiga atau engsel
(hinge). Satu titik punya sekitar 10–11 tetangga, jadi tiap baris hanya punya ~11 entri bukan nol dari 65.025 kemungkinan.

Matriks seperti ini disebut **sparse** (jarang). Kita hanya menyimpan yang bukan nol, plus informasi di mana letaknya.

## 2. Kenapa "blok 3×3"?

Setiap titik kain punya 3 koordinat (x, y, z). Hubungan antara titik `i` dan titik `j` bukan satu angka, tetapi
**matriks 3×3**: "kalau titik j bergeser di arah x, berapa gaya yang dirasakan titik i di arah x, y, dan z?".

Daripada menyimpan 9 entri skalar terpisah, kita simpan **satu blok 3×3** per pasangan titik. Inilah **BSR**
(*Block Sparse Row*, atau *Block CSR*).

Analogi database: bayangkan tabel relasi `neighbors(row, col, v00..v22)`, satu baris per pasangan titik yang terhubung,
diurutkan berdasarkan `row`, lalu di-*index* supaya kita bisa langsung lompat ke semua tetangga satu titik.

## 3. Tiga array BSR

```rust
pub struct Bsr {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub vals: Vec<f64>,
}
```

Contoh kecil: 3 titik, titik 0 terhubung dengan 0 dan 1, titik 1 dengan 0, 1, 2, titik 2 dengan 1 dan 2.

```
row 0: blok (0,0), (0,1)
row 1: blok (1,0), (1,1), (1,2)
row 2: blok (2,1), (2,2)

col_idx = [0, 1,   0, 1, 2,   1, 2]        ← kolom tiap blok, diurutkan per baris
row_ptr = [0, 2, 5, 7]                      ← blok baris r ada di col_idx[row_ptr[r] .. row_ptr[r+1]]
vals    = [ 9 angka blok ke-0 | 9 angka blok ke-1 | ... ]   ← panjang 9 × jumlah blok
```

- `row_ptr[r]` sampai `row_ptr[r+1]` adalah **rentang** indeks blok milik baris `r`. Persis seperti offset pagination:
  "baris 1 dimulai di posisi 2, berakhir sebelum posisi 5".
- `col_idx[k]` adalah kolom (titik tetangga) dari blok ke-`k`.
- `vals[9k .. 9k+9]` adalah isi blok ke-`k`, **row-major**: `v00 v01 v02 v10 v11 v12 v20 v21 v22`.

Panjang `row_ptr` = jumlah baris + 1. Trik "+1" membuat rumus rentang selalu `row_ptr[r]..row_ptr[r+1]` tanpa kasus khusus
untuk baris terakhir.

## 4. Membangun pola: `with_pattern`

```rust
pub fn with_pattern(rows: usize, groups: &[Vec<usize>]) -> Bsr {
    let mut columns: Vec<Vec<usize>> = (0..rows).map(|r| vec![r]).collect();
    for group in groups {
        for &r in group {
            columns[r].extend(group.iter().copied());
        }
    }
    ...
        row.sort_unstable();
        row.dedup();
    ...
    let vals = vec![0.0; 9 * col_idx.len()];
```

- `groups` adalah daftar "kelompok titik yang saling memengaruhi". Satu segitiga = kelompok 3 titik. Satu engsel = 4 titik.
- Setiap baris pasti punya **blok diagonal** (titik terhadap dirinya sendiri): `vec![r]`.
- Untuk setiap kelompok, setiap anggota mendapat semua anggota lain sebagai tetangga.
- `sort_unstable` + `dedup`: urutkan kolom, buang duplikat (dua segitiga yang berbagi rusuk menyumbang pasangan yang sama).
- `vals` diisi nol. Ini **pola** (*pattern*): kita tahu di mana yang bukan nol, tetapi nilainya belum diisi.

Memisahkan pola dan nilai itu penting. Di Step 2, pola dibuat sekali, lalu **nilai** diisi ulang setiap iterasi Newton
oleh kernel GPU.

## 5. Mencari posisi blok: `position`

```rust
pub fn position(&self, row: usize, column: usize) -> usize {
    let (start, end) = (self.row_ptr[row], self.row_ptr[row + 1]);
    let offset = self.col_idx[start..end].binary_search(&column).expect("block outside sparsity pattern");
    start + offset
}
```

Karena kolom tiap baris sudah diurutkan, kita bisa **binary search**: "di baris `row`, di indeks berapa kolom `column`?".
Kalau tidak ketemu, program panik dengan pesan jelas. Menulis ke blok yang tidak ada di pola adalah bug perakitan,
dan lebih baik meledak keras daripada diam-diam salah.

## 6. Menambah blok: `add`

```rust
pub fn add(&mut self, row: usize, column: usize, block: &Mat3) {
    let base = 9 * self.position(row, column);
    for r in 0..3 { for c in 0..3 {
        self.vals[base + 3 * r + c] += block[r][c];
    }}
}
```

**Perakitan** (*assembly*) FEM: setiap elemen (segitiga) menghitung matriks lokal kecilnya, lalu **menjumlahkan** (`+=`)
potongannya ke matriks global. Satu blok global bisa menerima sumbangan dari beberapa segitiga yang berbagi titik.

Indeks `base + 3*r + c` adalah layout row-major yang sama yang dibaca kernel WGSL: `vals[o + 3u*r + c]`.

## 7. Perkalian matriks-vektor: `multiply`

```rust
pub fn multiply(&self, x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; x.len()];
    for row in 0..self.rows() {
        for k in self.row_ptr[row]..self.row_ptr[row + 1] {
            let column = self.col_idx[k];
            for r in 0..3 {
                let v = &self.vals[9 * k + 3 * r..9 * k + 3 * r + 3];
                y[3 * row + r] += v[0] * x[3 * column] + v[1] * x[3 * column + 1] + v[2] * x[3 * column + 2];
            }
        }
    }
    y
}
```

Ini **SpMV** (*Sparse Matrix-Vector product*), operasi paling penting di seluruh Step 1:

- Untuk setiap baris (titik) `row`,
- untuk setiap tetangga `column` di baris itu,
- ambil blok 3×3 dan kalikan dengan 3 koordinat titik tetangga di `x`,
- jumlahkan ke 3 komponen hasil `y` untuk titik `row`.

Kernel `spmv_dot.wgsl` melakukan hal yang **persis sama**, hanya saja setiap baris dikerjakan oleh satu thread GPU
secara paralel, bukan satu per satu dalam loop.

Oracle memakai `multiply` versi f64 ini untuk menghitung **residual sejati** `‖b − A x_gpu‖ / ‖b‖` dalam presisi penuh.

## 8. Fungsi pendukung

- `diagonal_positions()`: indeks blok diagonal tiap baris. Dikirim ke GPU sebagai `diag_idx`, dipakai preconditioner Jacobi
  (dokumen 1.7) untuk langsung lompat ke blok diagonal tanpa mencari.
- `asymmetry()`: memeriksa `A` simetris (`A_ij = A_ji`). PCG **hanya benar** untuk matriks simetris positif definit.
  `generate.rs` menolak soal kalau asimetri > 10⁻¹².
- `rounded_to_f32()`: salinan matriks dengan setiap nilai dibulatkan ke f32 lalu kembali ke f64. Dipakai untuk mengukur
  **storage error**: seberapa banyak jawaban berubah **hanya karena** matriksnya disimpan dalam f32, sebelum satu pun
  operasi f32 dilakukan. Ini lantai akurasi yang tidak bisa dikalahkan solver f32 mana pun.

## 9. Berapa ukurannya di n257?

- baris (titik bebas): 65.025
- blok: 711.201 (~10,9 per baris)
- `vals`: 711.201 × 9 × 4 byte (f32) ≈ 25,6 MB
- `col_idx`: 711.201 × 4 byte ≈ 2,8 MB

Setiap iterasi PCG membaca semua `vals` dan `col_idx` sekali (di SpMV). Inilah kenapa `compare.rs` menghitung lalu lintas
memori "40 byte per blok" (36 byte nilai + 4 byte indeks) "plus 196 byte per baris" (vektor-vektor), dan dari situ
**bandwidth efektif**.

## 10. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Sparse | simpan hanya yang bukan nol |
| BSR | unit penyimpanannya blok 3×3 per pasangan titik |
| `row_ptr` | offset awal-akhir blok tiap baris (panjang baris+1) |
| `col_idx` | kolom tiap blok, terurut per baris |
| `vals` | 9 angka per blok, row-major |
| Pola vs nilai | pola dibuat sekali, nilai bisa diisi ulang |
| SpMV | operasi inti PCG; GPU: satu thread per baris |
