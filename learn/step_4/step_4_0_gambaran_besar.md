# Step 4.0: Gambaran Besar Experiment 4, Newton yang Robust dan Hemat

> Step 3 membuktikan bahwa GPU f32 **akurat** dan **stabil** untuk kain jatuh yang berkerut, tetapi lambat (900–7200 ms per langkah) dan Newton-nya sering
> tidak konvergen. Step 4 tidak menambah fisika baru. Ia memperbaiki **cara menyelesaikan** fisika yang sama, lalu mengukur efek setiap perbaikan secara terpisah.

---

## 1. Masalah dari Step 3

| Masalah | Bukti E3 | Akibat |
|---|---|---|
| Tangen proyeksi membuat Newton linear | oracle f64 gagal konvergen di 43/240 (n33) dan 70/240 (n65) langkah | referensi tidak bersih; GPU lambat |
| Toleransi 10⁻⁶ di bawah lantai f32 | GPU selalu 30 iterasi Newton | 25+ iterasi sia-sia per langkah |
| PCG selalu ke 10⁻⁷ | 216 / 463 / 799 iterasi per solve | sebagian besar waktu habis di PCG |

## 2. Tiga perbaikan, satu per satu

Analogi backend: optimasi performa yang benar adalah **satu perubahan per eksperimen**, dengan pengukuran sebelum dan sesudah. Kalau tiga perubahan digabung sekaligus,
kita tidak tahu mana yang membantu dan mana yang merugikan. Karena itu E4 adalah **ablation ladder**:

| Rung | Perubahan | Pertanyaan |
|---|---|---|
| A | konfigurasi E3 | patokan |
| B | + berhenti saat koreksi < 10⁻⁴ dari langkah (meter) | berapa iterasi Newton yang sia-sia? |
| C | + toleransi PCG adaptif (Eisenstat–Walker) | berapa iterasi PCG yang sia-sia? |
| D | + tangen eksak dengan regularisasi μ | apakah tangen eksak yang aman lebih baik daripada proyeksi? |

### Aturan berhenti berbasis update (dokumen 4.2)

Berhenti kalau koreksi Newton terakhir ‖αd‖∞ ≤ 10⁻⁴·‖x − x_n‖∞. Satuannya meter, jadi kebal terhadap skala residual relatif yang bermasalah di E3.

### Forcing Eisenstat–Walker (dokumen 4.2)

Toleransi PCG mengikuti kemajuan Newton: longgar (10⁻²) di awal, ketat hanya saat Newton hampir selesai. Rumus: η = 0,9·(‖g_k‖/‖g_{k−1}‖)², dijepit ke [10⁻⁷, 10⁻²].

### Regularisasi Levenberg–Marquardt (dokumen 4.1)

Tangen eksak `K` (Newton cepat) ditambah `μ·M/h²` hanya kalau perlu (Cholesky gagal atau PCG breakdown). μ naik ×10 saat gagal, turun ÷10 setelah langkah penuh.
Dipakai di oracle **dan** GPU (rung D).

## 3. Mengukur ke mana waktu pergi

`timed-backend.ts` (dokumen 4.6) membungkus backend GPU dan membagi waktu per langkah menjadi: perakitan, PCG, line search, I/O host, dan sisanya. Tanpa ini,
kita hanya tahu "lambat", bukan "lambat di mana".

## 4. Tidak boleh mengorbankan akurasi

Setiap rung menjalankan **semua** cek E3 lagi (FD, kernel, konsistensi satu langkah, lantai, energi), ditambah:

- oracle f64 harus konvergen di setiap langkah;
- keputusan GPU (η, berhenti) diputar ulang dan diperiksa oleh oracle;
- kernel inersia dengan μ diperiksa terhadap f64.

Aturannya: **konfigurasi final = rung terakhir yang lolos semua gate akurasi**.

## 5. Hasil singkat (detail di dokumen 4.10)

| n65 | A | B | C | D |
|---|---|---|---|---|
| Newton / langkah | 30 | 8,03 | 7,51 | **4,82** |
| PCG / solve | 463 | 554 | 76 | 91 |
| ms / langkah | 1815 | 563 | 164 | **112** |

- Rung D di n65: **112 ms/langkah, 15,5× lebih cepat dari E3** (gate ≤ 174 ms lulus). n129: 7486 → 374 ms (20×).
- Semua cek akurasi E3 tetap lulus di semua rung: kernel, FD, energi, dan konsistensi satu langkah (≤ 1,2×10⁻⁴).
- **Tetapi tidak ada rung yang menjadi konfigurasi final**, karena dua gate gagal di **semua** rung dengan alasan di luar pilihan rung:
  1. Oracle f64 di n65 tidak konvergen di 6 langkah. Lima di antaranya berhenti di lantai f64 (1–4×10⁻¹⁰), dan satu (langkah 89) benar-benar macet di 8×10⁻⁴.
  2. Probe 1 tidak berhenti lewat aturan residual atau aturan update. Di langkah pertama, ambang aturan update (1,7×10⁻⁸ m) lebih kecil dari resolusi f32 posisi
     (~3×10⁻⁸ m pada koordinat 0,5 m).

## 6. Cara menjalankan

```bash
./run-e4.sh                                    # semua rung, n33 + n65 (oracle) + n129
RUNGS="D" SIZES="33" GPU_ONLY="" ./run-e4.sh   # satu rung, satu ukuran
```
