# Step 1.26: `results/E1-VERDICT.md`, Membaca Hasil Experiment 1

> Dokumen terakhir Step 1: apa yang diukur, apa artinya, dan kenapa keputusan akhirnya "GAGAL hanya di speedup".
> Ini juga pelajaran tentang cara melaporkan eksperimen dengan jujur.

---

## 1. Hardware yang terbukti dipakai

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB GDDR6, 128-bit, 176 GB/s, 50 W, driver 32.0.16.1714 |
| Backend | D3D12 (Chromium 141 dengan FXC; vgpu 0.6) |
| Ditolak | iGPU AMD 31.0.21925.1001 dan semua software adapter |
| CPU patokan | AMD Ryzen 5 5600H, satu inti, colokan listrik, Balanced |

**Bandwidth rated 176 GB/s** dihitung dari `nvidia-smi`: clock memori maksimum 5501 MHz. Untuk GDDR6, angka itu adalah setengah data rate,
jadi 11 Gbps per pin × 128 pin ÷ 8 = 176 GB/s. 25% darinya = 44 GB/s (batas gate bandwidth).

## 2. Kronologi temuan

### 2.1 Prediksi dari emulasi, sebelum GPU dijalankan

Saat `generate` n257 tension:

```
pcg f64 it 1386 res 9.82e-8 | pcg f32 it 1527 res 1.25e-2 err 8.03e-4
```

Emulasi f32 (bit-setia GPU) mencapai residual sejati **1,25×10⁻²**, di atas batas 10⁻².

### 2.2 Apakah itu lantai f32?

Pemeriksaan pertama: `residual_floor_f32_rounded_exact` = 4,3×10⁻⁴, jadi jawaban sempurna yang hanya dibulatkan ke f32 jauh lebih baik.
Kesimpulan: ini **drift residual rekursif**, bukan batas f32. Obatnya residual replacement.

### 2.3 Baseline GPU mengonfirmasi

| n257 | iterasi | residual sejati | speedup |
|---|---|---|---|
| tension | 1524 | 1,25×10⁻² ✗ | 9,82× ✗ |
| dynamic | 1494 | 5,94×10⁻³ ✓ | 13,1× ✓ |

### 2.4 Replacement: perlu gate

- Tiap 50 tanpa syarat → tidak pernah konvergen (dokumen 1.4).
- Tiap 50 selama > 10⁻² → tension **7,84×10⁻³** ✓.
- Gate lebih rendah → divergen.

### 2.5 Hipotesis yang ditolak

Hipotesis: "sisa residual 7,8×10⁻³ adalah lantai penyimpanan f32 (matriks dibulatkan)". Diuji dengan menghitung residual dari solusi pasti
sistem f32: 7,6×10⁻⁴, sepuluh kali lebih kecil. **Hipotesis ditolak**: sisa galat berasal dari aritmetika f32 di dalam iterasi, bukan dari
penyimpanan. Mencatat hipotesis yang ditolak sama pentingnya dengan mencatat yang diterima.

## 3. Hasil akhir n257 (WebGPU mentah, dengan replacement)

| | tension | dynamic |
|---|---|---|
| iterasi GPU / emulasi f32 / f64 | 1545 / 1608 / 1386 | 1495 / 1495 / 1493 |
| residual sejati (f64) | 7,84×10⁻³ | 4,67×10⁻³ |
| galat vs f64 / storage error | 8,4×10⁻⁴ / 8,5×10⁻⁴ | 9,8×10⁻⁵ / 9,8×10⁻⁵ |
| warm solve / timestamp GPU | 1226 / 1091 ms | 1158 / 1051 ms |
| bandwidth efektif | 58,3 GB/s (33%) | 58,6 GB/s (33%) |
| PCG f64 1 inti | 8,06 s | 9,98 s |
| speedup WebGPU / vgpu | **6,57× / 5,93×** | **8,62× / 7,53×** |

Perhatikan: galat vs f64 (8,4×10⁻⁴) hampir sama dengan storage error (8,5×10⁻⁴). Artinya solusi GPU **sedekat mungkin** ke jawaban
benar, mengingat matriksnya disimpan dalam f32.

## 4. Gate

| Gate | Hasil |
|---|---|
| converged | lulus |
| forcing_term_ok | lulus (dengan replacement) |
| f32_iterations_ok | lulus (1,11× dan 1,00×) |
| faithful | lulus |
| deterministic | lulus |
| **speedup_ok** | **GAGAL** |
| pertumbuhan iterasi < 3× | lulus |
| overhead vgpu ≤ 1,15× | lulus (1,11×, 1,14×) |
| storage error ≤ 10⁻³ | lulus |
| bandwidth ≥ 25% | lulus (33%), sliced BSR tidak dipicu |

## 5. Kenapa speedup gagal, dan kenapa tidak "diperbaiki"

### Patokan CPU tidak stabil

Solve f64 yang sama (1386 iterasi tension) tercatat **8,06 s** di satu run oracle dan **12,2 s** di run lain. Penyebabnya boost clock dan
suhu laptop. Speedup tension berkisar 6,6×–9,8×, tetapi **tidak pernah mencapai 10×** di empat pengukuran.

### GPU dibatasi bandwidth

0,70 ms/iterasi pada 41,2 MB/iterasi = 58 GB/s = 33% dari 176 GB/s. Batas atas teoretis pada 100% bandwidth sekitar 0,23 ms/iterasi,
atau ~25× speedup. Untuk 10× dibutuhkan ~40% bandwidth efektif, atau GPU dengan bandwidth lebih besar.

### Keputusan

Pengguna memutuskan: **E1 GAGAL hanya di `speedup_ok`**. Semua gate akurasi lulus. `SPEEDUP_LIMIT` tidak diubah. Pertanyaan speedup tetap
terbuka untuk hardware yang lebih kuat. Tidak ada perubahan arsitektur yang dipaksa: pemicu multigrid (pertumbuhan ≥ 3×) dan sliced BSR
(bandwidth < 25%) tidak terpenuhi.

## 6. Pelajaran metodologi

1. **Gate ditulis sebelum mengukur**, dan tidak digeser sesudahnya.
2. **Emulasi bit-setia** memungkinkan prediksi sebelum menjalankan GPU, dan membedakan "GPU salah" dari "f32 memang begini".
3. **Lantai referensi** (rounded exact, storage floor) memisahkan batas fisik f32 dari kelemahan algoritma.
4. **Penjaga hardware** mencegah data dari GPU yang salah.
5. **Hipotesis yang ditolak dicatat.**
6. **Ketidakpastian patokan dilaporkan**, bukan disembunyikan.

## 7. File bukti

- `results/experiment1-webgpu-hardware.json`, `experiment1-vgpu-hardware.json`: dengan replacement.
- `results/experiment1-*-baseline.json`: tanpa replacement.
- `cases/`, `cases-baseline/`: per kasus.
- `results/experiment1-webgpu-software.json`: SwiftShader, **bukan bukti hardware**.
