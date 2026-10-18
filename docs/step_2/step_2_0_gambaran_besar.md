# Step 2.0: Gambaran Besar Experiment 2, dari Soal Linear ke Mekanika Nonlinear

> Step 1 menjawab "bisakah GPU f32 menyelesaikan `A x = b`?". Step 2 menjawab pertanyaan berikutnya: "bisakah GPU f32
> menyelesaikan **fisika kain yang sebenarnya** (nonlinear), dengan solver Step 1 di dalamnya?".
> Dokumen ini membangun intuisi mekanika yang dibutuhkan, tanpa kalkulus berat.

---

## 1. Kenapa nonlinear?

Di Step 1, matriks `A` sudah jadi; GPU hanya menyelesaikannya. Di dunia nyata, kain melar 10–20%. Pada regangan sebesar itu,
**gaya tidak sebanding dengan perpindahan**: tarik 2× lebih jauh tidak menghasilkan gaya tepat 2×. Hubungannya melengkung (nonlinear).

Akibatnya tidak ada satu `A` yang bisa langsung diselesaikan. Kita harus **mencari** posisi di mana semua gaya seimbang, secara iteratif.

## 2. Konsep fisika dalam bahasa backend

| Konsep | Arti | Analogi backend |
|---|---|---|
| Posisi `x` | koordinat setiap titik kain sekarang | state |
| Posisi rest `X` | koordinat titik saat kain belum ditarik | state awal / baseline |
| Deformation gradient **F** | "bagaimana potongan kecil kain teregang dan berputar" | diff antara state sekarang dan baseline, per elemen |
| Green strain **E** | regangan murni, tanpa rotasi | diff yang dinormalisasi supaya rotasi tidak dihitung sebagai perubahan |
| Stress **S** | gaya dalam per satuan panjang | "tekanan" akibat diff |
| Energi **Π** | "biaya" konfigurasi; kain ingin energi minimum | fungsi cost yang ingin diminimalkan |
| Gaya dalam **g = ∂Π/∂x** | arah energi naik paling cepat, per titik | gradien cost |
| Tangent/Hessian **K = ∂²Π/∂x²** | seberapa cepat gaya berubah saat titik digeser | kelengkungan cost |
| Kesetimbangan | `g = 0` di semua titik bebas | cost minimum lokal |

### Kenapa Green strain, bukan "panjang baru − panjang lama"?

Kalau sepotong kain hanya **diputar** tanpa ditarik, ia tidak boleh merasa tegang. Regangan naif (selisih koordinat) akan menganggap
rotasi sebagai regangan. Green strain `E = ½(FᵀF − I)` menghapus rotasi secara matematis: FᵀF hanya mengukur perubahan **panjang dan
sudut**. Ada tes khusus untuk ini: `rigid_rotation_is_stress_free`.

## 3. Newton: mencari g = 0

Analogi: kamu mencari titik terendah di lembah dalam kabut, tetapi punya alat yang memberi kemiringan (`g`) dan kelengkungan (`K`)
di tempatmu berdiri.

1. Di posisi `x`, ukur `g` dan `K`.
2. Anggap lembahnya parabola sempurna. Titik terendah parabola itu ada di `x + d`, dengan `K d = −g`. **Ini sistem linear: solver Step 1.**
3. Melangkah ke sana (mungkin tidak penuh, lihat line search).
4. Ulangi sampai `‖g‖` cukup kecil.

Kalau sudah dekat jawaban, Newton konvergen **kuadratik**: jumlah digit benar kira-kira **berlipat dua** setiap iterasi
(10⁻² → 10⁻⁴ → 10⁻⁸ → 10⁻¹⁶). Ini yang dicek di Step 2.

### Line search Armijo

Kalau masih jauh dari jawaban, parabola bisa menipu dan langkah penuh justru **menaikkan** energi. Line search Armijo memeriksa:
"apakah energi benar-benar turun cukup?". Kalau tidak, langkahnya dipotong setengah, lalu dicek lagi. Seperti exponential backoff, tetapi
untuk panjang langkah.

### Inexact Newton

Langkah 2 tidak harus diselesaikan sempurna. Kalau solver linear hanya menurunkan residual 100× (forcing term 10⁻²), Newton tetap
konvergen dengan baik. Inilah alasan gate forcing 10⁻² di Step 1. Di Step 2, solver GPU f32 Step 1 dipakai sebagai langkah linear yang
"tidak pasti" ini, sedangkan oracle f64 memakai Cholesky pasti.

## 4. Kupon: benda uji

Bukan jersey, melainkan **kupon**: persegi kain 10 cm × 10 cm, seperti spesimen di laboratorium uji material.

| Jalur beban | Penjepit | Beban |
|---|---|---|
| `uniaxial` | tepi kiri dan kanan dijepit penuh | ditarik ke 1,20× (20%) arah wale, lalu dilepas |
| `biaxial` | "rahang" di tengah setiap tepi | 1,15× wale × 1,075× course, lalu dilepas |
| `patch` | semua tepi mengikuti peta afin (termasuk geser) | satu langkah; jawaban pasti diketahui |

Masing-masing di grid 33×33 dan 65×65 titik. Uniaxial dan biaxial terdiri dari 10 langkah naik + 10 langkah turun.

## 5. Pembagian kerja

```
┌──────────── oracle (Rust, f64) ────────────┐   ┌──────────── GPU (WGSL f32) + host (TS) ───────────┐
│ material.rs   hukum material StVK           │   │ membrane.wgsl            hukum yang sama dalam f32 │
│ membrane.rs   F, E, g, K, ΔΠ per segitiga   │   │ membrane_element.wgsl    g, K per segitiga         │
│ coupon.rs     kupon, penjepit, prediktor    │   │ membrane_energy_change   ΔΠ per segitiga            │
│ newton.rs     Newton + Armijo, Cholesky     │   │ assemble_matrix/force    rakit ke BSR E1           │
│ e2_checks.rs  patch test, FD, bolak-balik   │   │ solver.ts (E1)           PCG f32 = langkah linear  │
│ e2_generate   file soal + referensi         │   │ newton.ts                loop Newton f32           │
│ e2_compare    nilai GPU vs f64              │   │ run-e2.ts                runner                     │
└─────────────────────────────────────────────┘   └───────────────────────────────────────────────────┘
```

Prinsip yang sama dengan Step 1: oracle f64 = kunci jawaban, GPU f32 = yang diuji, keduanya memakai rumus dan urutan operasi yang sama.

## 6. Delapan hal yang diminta dan diperiksa

1. Deformation gradient F, Green strain E
2. Hukum material ortotropik St. Venant–Kirchhoff (memakai ulang `material.rs`)
3. Gaya dalam dan tangent
4. Newton + Armijo (GPU: PCG f32 inexact; oracle: Cholesky pasti)
5. Patch test (regangan konstan direproduksi tepat)
6. Cek beda hingga (finite difference) untuk gradien dan Hessian
7. Kupon beban/lepas (uniaxial dan biaxial) dengan kurva tegangan-regangan
8. Perbandingan WGSL f32 vs Rust f64: galat perpindahan, residual per iterasi Newton, jumlah iterasi, langkah line search

## 7. Hasil singkat (detail di dokumen 2.17)

- Semua cek f64 lulus dengan margin besar (FD ~10⁻⁸, patch ~10⁻¹⁶, bolak-balik ~10⁻¹⁴).
- Kernel WGSL cocok dengan f64 per elemen sampai ~5×10⁻⁷.
- Newton f32 **mengikuti** Newton f64 iterasi demi iterasi (~3 digit sama) **sampai mencapai lantai f32**.
- Target residual 10⁻⁶: tercapai di patch; tidak tercapai di uniaxial/biaxial. Lantainya 5×10⁻⁶ sampai 1,1×10⁻³, dan lantai itu **sama persis**
  dengan residual solusi f64 yang hanya dibulatkan ke f32. Tidak ada solver f32 yang bisa lebih baik dengan posisi absolut f32.
- Secara fisik jawaban f32 sangat dekat: perpindahan ~4×10⁻⁷ dan tegangan ~8×10⁻⁶ relatif terhadap puncaknya.

## 8. Cara menjalankan

```bash
./run-e2.sh hardware                         # patch, uniaxial, biaxial × n33, n65
PATHS="uniaxial" SIZES="33" ./run-e2.sh hardware
```
