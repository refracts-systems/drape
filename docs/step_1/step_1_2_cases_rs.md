# Step 1.2: `oracle/src/cases.rs`, Tiga Soal Uji dan Asal-usul Matriksnya

> Dokumen ini menjelaskan **dari mana** matriks `A` dan vektor `b` datang. Kamu tidak harus menguasai mekanikanya
> untuk Step 1, tetapi kamu perlu tahu kenapa tiga kasus ini dipilih dan kenapa satu di antaranya memang diharapkan gagal.
> Mekanika membran dibahas tuntas di Step 2 (dokumen 2.1 dan 2.2).

---

## 1. Komponen yang dirakit

`cases.rs` memakai beberapa modul lain:

| Modul | Isi | Analogi |
|---|---|---|
| `mesh.rs` | grid persegi 0,5 m × 0,5 m, dipecah jadi segitiga, plus daftar engsel | skema tabel: titik, segitiga, rusuk |
| `material.rs` | kekakuan kain rajut: arah wale 1000 N/m, course 300 N/m, dst. | konfigurasi |
| `membrane.rs` | gaya regang (membran) per segitiga | aturan bisnis per baris |
| `hinge.rs` | gaya tekuk (bending) per engsel (dua segitiga yang berbagi rusuk) | aturan bisnis per relasi |
| `bsr.rs` | matriks global | tabel hasil agregasi |

### Mesh

```rust
let mesh = Mesh::grid(columns, columns, SHEET_SIZE, SHEET_SIZE);
```

Grid `n × n` titik pada lembar 0,5 m. Setiap sel persegi dipotong jadi dua segitiga; arah diagonalnya **berselang-seling**
(`(i + j) % 2`) supaya mesh tidak punya arah dominan (bias yang membuat kain "lebih kaku" ke satu diagonal).

Engsel (*hinge*): setiap rusuk interior yang dimiliki dua segitiga. Engsel memberi kain kekakuan terhadap **tekukan**.

### Titik tepi dikunci

```rust
fn free_rows(mesh: &Mesh) -> (Vec<Option<usize>>, Vec<usize>) {
    let free: Vec<usize> = (0..mesh.rest.len()).filter(|&v| !mesh.is_boundary(v)).collect();
    ...
}
```

Titik di tepi lembar dianggap **dijepit** (tidak boleh bergerak), jadi tidak masuk ke sistem persamaan. Hanya titik interior
yang menjadi baris matriks. `row_of[v]` memetakan nomor titik ke nomor baris (atau `None` kalau dijepit). Ini seperti
tabel lookup `vertex_id → row_id`.

Untuk n257: 257² = 66.049 titik, 255² = 65.025 interior → 65.025 baris → 195.075 unknown.

## 2. Matriks = Hessian energi

Kalimat kuncinya:

> `A` adalah **Hessian** (turunan kedua) dari energi total kain terhadap posisi titik.

Analogi: energi itu seperti "biaya" sebuah konfigurasi. Gaya = turunan pertama ("ke arah mana biaya turun").
Hessian = turunan kedua ("seberapa curam lembah biayanya ke setiap arah"). Langkah Newton menyelesaikan
`Hessian · Δx = −gaya`. Itulah `A x = b`.

Perakitan:

```rust
let mut matrix = Bsr::with_pattern(free_vertices.len(), &coupling_groups(&mesh, &row_of));
assemble_membranes(&mut matrix, &membranes, &positions, &material, &row_of);
assemble_hinges(&mut matrix, &hinges, &positions, &row_of);
if kind == CaseKind::Dynamic {
    add_inertia(&mut matrix, &mesh, &material, &free_vertices);
}
```

1. Pola BSR dibuat dari "kelompok titik": tiap segitiga (3 titik) dan tiap engsel (4 titik).
2. Setiap segitiga menyumbang Hessian 9 blok (3×3 titik), hanya untuk pasangan yang dua-duanya bebas.
3. Setiap engsel menyumbang 16 blok (4×4 titik).
4. Kasus dynamic menambahkan **massa / h²** di diagonal (dibahas di bawah).

## 3. Tiga kasus

```rust
pub fn stretch(&self) -> (f64, f64) {
    match self {
        CaseKind::Tension => (1.03, 1.08),
        _ => (1.0, 1.0),
    }
}
```

### Tension: kain ditarik

Lembar diregangkan 3% di arah wale dan 8% di arah course, lalu Hessian dihitung di keadaan itu. Kain yang tegang punya
kekakuan **geometrik** tambahan, seperti tali gitar yang ditarik: makin tegang, makin keras menolak dibelokkan ke samping.
Akibatnya arah tegak lurus kain (z) mendapat kekakuan dari tegangan, bukan hanya dari tekukan.

Matriksnya "jinak". Bilangan kondisi Jacobi κ ≈ 4,7×10⁴ di n257.

### Slack: kain kendur

Tanpa regangan. Kekakuan arah z (keluar bidang) **hanya** dari tekukan (bending), yang sangat lemah:
2×10⁻⁶ N·m dibanding membran 1000 N/m. Perbedaan skala yang ekstrem → matriks **sangat buruk**: κ sampai 8×10⁸.

Ini kasus **diagnostik**: kita tahu f32 PCG Jacobi akan gagal, dan memang gagal. Kasus ini menunjukkan
**batas** metodenya, bukan target lulus.

### Dynamic: kain kendur + inersia

```rust
fn add_inertia(m: &mut Bsr, mesh: &Mesh, material: &Material, free: &[usize]) {
    let areas = mesh.vertex_areas();
    for (row, &v) in free.iter().enumerate() {
        let mass = material.density * areas[v];
        m.add(row, row, &identity_scaled(mass / (TIME_STEP * TIME_STEP)));
    }
}
```

Simulasi dinamis dengan *backward Euler* dan langkah waktu h = 1/240 detik menambahkan **M/h²** ke diagonal. Dengan h kecil,
suku ini besar dan membuat matriks jauh lebih "dominan diagonal", jadi jauh lebih jinak (κ ≈ 4×10⁴ di n257). Ini sebabnya
simulasi kain real-time praktis hampir selalu memakai langkah dinamis: inersia "menolong" solver.

## 4. Vektor `b`: beban

```rust
fn load_vector(...) -> Vec<f64> {
    ...
    let shape = (PI * px / w).sin() * (PI * py / h).sin();
    let skew = (2.0 * PI * px / w).sin() * (PI * py / h).sin();
    [area * IN_PLANE_LOAD * shape, 0.5 * area * IN_PLANE_LOAD * skew, -material.density * area * GRAVITY]
}
```

- Komponen z: **gravitasi** (massa × 9,81) ke bawah.
- Komponen x dan y: beban halus berbentuk gelombang sinus. Ini sengaja dibuat supaya `b` punya komponen di semua arah
  dan tidak simetris sempurna, sehingga solver tidak "kebetulan" mudah.

Setiap komponen dikalikan **luas titik** (`vertex_areas`): titik yang mewakili area kain lebih besar menerima beban lebih besar.
Ini cara FEM membagi beban terdistribusi ke titik-titik.

## 5. Tes di file ini

```rust
#[test]
fn assembled_matrices_are_symmetric() { ... asymmetry() < 1e-12 ... }

#[test]
fn uniform_prestretch_has_no_interior_force() { ... }

#[test]
fn material_is_convex() { ... }
```

- **Simetri**: PCG butuh `A` simetris. Kalau perakitan salah (misalnya blok (i,j) tidak sama dengan transpose (j,i)),
  tes ini menangkapnya.
- **Prategang seragam tanpa gaya interior**: regangan merata harus seimbang di semua titik interior (setiap titik ditarik sama
  kuat dari semua sisi). Kalau tidak, ada bug di gradien membran.
- **Material konveks**: syarat matematis supaya energi punya lembah tunggal (coupling² < wale × course).

## 6. Hubungan ke GPU

GPU **tidak pernah tahu** soal kain, segitiga, atau gravitasi. Yang diterima GPU hanya:

- `row_ptr`, `col_idx`, `diag_idx` (u32),
- `vals` dan `rhs` (f32, dibulatkan dari f64).

Pemisahan ini sengaja: Experiment 1 menguji **solver linear murni**. Kalau solver gagal, kita tahu masalahnya di aljabar linear
f32, bukan di fisika. Fisikanya dipindahkan ke GPU baru di Step 2.

## 7. Ringkasan

| Pertanyaan | Jawaban |
|---|---|
| Apa itu `A`? | Hessian energi kain (membran + tekukan, + inersia untuk dynamic) |
| Apa itu `b`? | gravitasi + beban sinus di bidang, dikali luas titik |
| Kenapa 3 kasus? | dua kasus produksi (tension, dynamic) + satu diagnostik batas (slack) |
| Kenapa tepi dikunci? | supaya sistem punya solusi tunggal (tidak ada gerak benda kaku) |
| Apa yang dikirim ke GPU? | hanya array BSR + rhs dalam f32 |
