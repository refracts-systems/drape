# Step 4.10: `run-e4.sh` dan Hasil Experiment 4

> Dokumen penutup Step 4: skrip, tabel ablation lengkap, gate per rung, dan cara membaca kesimpulan "tidak ada konfigurasi final" dengan jujur.

---

## 1. Skrip

```bash
RUNGS="${RUNGS:-A B C D}"
SIZES="${SIZES:-33 65}"
GPU_ONLY="${GPU_ONLY:-129}"
...
for n in $SIZES $GPU_ONLY; do      # 1) generate semua soal dulu (CPU berat)
  ...e4-generate...
done
for rung in $RUNGS; do             # 2) baru jalankan GPU (tanpa beban CPU lain)
  for n in $SIZES $GPU_ONLY; do
    ...run-e4.ts --rung "$rung"...
    ...e4-compare --rung "$rung"...
  done
done
...e4-summarize...
```

Perbedaan dari `run-e3.sh`: generate dipisah ke loop pertama. Prompt meminta *"Do not run heavy CPU work while timing GPU runs"*, dan dengan urutan ini semua oracle
selesai sebelum GPU pertama diukur.

## 2. Oracle E4

| | n33 | n65 |
|---|---|---|
| rata-rata Newton/langkah | 5,83 | 6,34 |
| langkah konvergen ke 10⁻¹⁰ | **240/240** | **234/240** |
| langkah dengan μ > 0 / μ maks | 6 / 1 | 4 / 10 |
| langkah di batas μ | 0 | 0 |
| waktu oracle | 1 min 18 s | 22 min |

Enam langkah n65 yang tidak konvergen:

- Langkah 1, 2, 3, 4, dan 111 stagnan di 1,0–4,1×10⁻¹⁰, tepat di atas toleransi. Itu lantai pembulatan f64 untuk langkah dengan skala residual kecil.
- Langkah 89 memakai 30 iterasi dengan μ sampai 10; residual sempat 3,9×10⁻⁵ lalu naik ke 8,2×10⁻⁴. Ini kegagalan Newton sungguhan (Armijo menurunkan
  energi, bukan residual).

## 3. Tabel ablation

| Kasus | Rung | Newton/langkah | PCG/solve | ms/langkah | perakitan | PCG | line search | I/O | lain |
|---|---|---|---|---|---|---|---|---|---|
| n33 | A | 30,00 | 216 | 886 | 45 | 746 | 11 | 67 | 18 |
| n33 | B | 6,93 | 237 | 222 | 13 | 185 | 2 | 16 | 6 |
| n33 | C | 6,84 | 41 | 90 | 17 | 56 | 3 | 10 | 5 |
| n33 | D | **4,53** | 46 | **64** | 12 | 38 | 2 | 8 | 4 |
| n65 | A | 30,00 | 463 | 1815 | 85 | 1541 | 14 | 112 | 63 |
| n65 | B | 8,03 | 554 | 563 | 27 | 477 | 4 | 33 | 23 |
| n65 | C | 7,51 | 76 | 164 | 25 | 96 | 6 | 16 | 21 |
| n65 | D | **4,82** | 91 | **112** | 20 | 66 | 2 | 10 | 14 |
| n129 | A | 30,00 | 799 | 7486 | 362 | 6580 | 37 | 234 | 272 |
| n129 | B | 6,88 | 1076 | 2344 | 107 | 2053 | 13 | 76 | 95 |
| n129 | C | 5,72 | 130 | 391 | 85 | 218 | 8 | 19 | 62 |
| n129 | D | **4,66** | 152 | **374** | 85 | 210 | 7 | 19 | 53 |

(Waktu dalam ms per langkah waktu; anggaran real-time 4,17 ms.)

### Membaca tabelnya

- **A → B**: aturan update memangkas Newton dari 30 ke ~7 iterasi (3,2–4×). PCG per solve sedikit **naik**, karena iterasi yang tersisa adalah yang "sulit"
  (iterasi gampang di lantai sudah dibuang).
- **B → C**: forcing memangkas PCG per solve 5,7–8,3× (237→41, 554→76, 1076→130). Ini perbaikan terbesar.
- **C → D**: tangen eksak teregularisasi memangkas Newton lagi (6,8→4,5 di n33, 7,5→4,8 di n65). PCG per solve sedikit naik, karena tangen eksak sedikit lebih sulit
  dibanding proyeksi, tetapi totalnya tetap lebih cepat.
- **Ke mana waktu pergi (D, n65)**: PCG 59%, perakitan 18%, lain-lain (JS host) 12%, I/O 9%, line search 2%.

## 4. Gate per rung

`Y` lulus, `n` gagal, `-` tidak dinilai (n129 tanpa referensi atau bukan n65).

| Gate | A | B | C | D |
|---|---|---|---|---|
| kernel WGSL (termasuk inersia μ) | Y | Y | Y | Y |
| FD gradien / Hessian Φ | Y | Y | Y | Y |
| identitas energi hinge | Y | Y | Y | Y |
| energi f32 tidak naik | Y | Y | Y | Y |
| replay aturan (η, berhenti) | Y | Y | Y | Y |
| **oracle f64 konvergen setiap langkah** | Y n33 / **n n65** | sama | sama | sama |
| energi f64 tidak naik | Y | Y | Y | Y |
| konsistensi satu langkah ≤ 10⁻³ | Y | Y | Y | Y |
| **berhenti benar di probe** | **n** | **n** | **n** | **n** |
| iterasi Newton ≤ oracle + 2 | n | Y | Y | Y |
| ms/langkah n65 ≤ 174 | n | n | Y | Y |
| pertumbuhan PCG < 3× | Y | Y | Y | Y |

Detail probe rung D (galat satu langkah / lantai f32 ÷ lantai representasi / alasan berhenti):

| Probe | n33 | n65 |
|---|---|---|
| 1 | 5,3×10⁻⁵ / 0,93× / **batas iterasi** | 1,2×10⁻⁴ / 0,96× / **batas iterasi** |
| 10 | 6,4×10⁻⁶ / 0,99× / update | 3,0×10⁻⁵ / 0,98× / update |
| 60 | 2,2×10⁻⁶ / 1,00× / update | 3,0×10⁻⁶ / 0,99× / update |
| 120 | 4,0×10⁻⁶ / 1,01× / update | 9,9×10⁻⁶ / 1,00× / update |
| 240 | 3,5×10⁻⁶ / 0,97× / update | 6,5×10⁻⁶ / 0,97× / update |

## 5. Kenapa "tidak ada konfigurasi final"

Aturan dari prompt: rung final = rung terakhir yang lolos **semua** gate akurasi. Dua gate gagal di semua rung:

1. **Oracle f64 n65.** Gate ini menilai referensi, bukan GPU, jadi sama untuk semua rung. Lima langkah berada di lantai f64 dan satu langkah (89) benar-benar macet.
2. **Berhenti di probe 1.** Langkah pertama memindahkan kain hanya h²g ≈ 1,7×10⁻⁴ m. Ambang aturan update = 10⁻⁴ × itu = 1,7×10⁻⁸ m, **lebih kecil** dari jarak antar
   angka f32 di sekitar 0,5 m (~3×10⁻⁸ m). Aturan itu secara fisik tidak bisa terpenuhi di langkah ini, sehingga Newton berhenti di batas iterasi. Residualnya
   sendiri sudah di lantai representasi (0,93–0,96×), dan galat posisinya 5×10⁻⁵ sampai 1,2×10⁻⁴.

Pembagian gate akurasi vs performa ditetapkan **sebelum** semua rung dijalankan (dokumen 4.9), dan tidak diubah setelah melihat hasil.

**Yang bisa disimpulkan**: secara performa, rung D adalah kandidat terbaik (15,5× lebih cepat dari E3 di n65 dengan semua cek akurasi E3 tetap lulus). Kedua kegagalan
punya penyebab yang terukur dan spesifik, dan keputusan berikutnya ada di tangan pengguna.

## 6. Trigger berbasis bukti

| Trigger | Terpicu? | Bukti |
|---|---|---|
| Regularisasi GPU mencapai batas μ → langkah itu pakai tangen proyeksi | tidak | 0 langkah di batas (μ maks 10) |
| Forcing merusak konsistensi satu langkah → kembali ke 10⁻⁷ | tidak | konsistensi C dan D ≤ 1,2×10⁻⁴ |
| Pertumbuhan iterasi ≥ 3× → multigrid | tidak | 1,67–2,34× |

Tidak ada perubahan arsitektur yang dipaksa bukti. Newton tetap di host, dan kernel PCG E1 tidak diubah.

## 7. Regresi

E1 tension-n33, E2 uniaxial-n33, dan E3 drape-n33 dijalankan ulang setelah semua perubahan: hasil identik dengan commit sebelumnya kecuali waktu.

## 8. Ringkasan

| Pertanyaan | Jawaban |
|---|---|
| Apakah lebih cepat? | ya: 15,5× (n65), 20× (n129) di rung D |
| Apa yang paling membantu? | forcing (PCG 6–8× lebih sedikit), lalu aturan update (Newton 4×), lalu tangen eksak (Newton 1,5×) |
| Apakah tetap akurat? | semua cek E3 lulus di semua rung |
| Konfigurasi final? | tidak ada, menurut aturan; dua gate gagal di semua rung karena oracle n65 dan probe 1 |
| Di mana waktu sekarang? | PCG ~60%, perakitan ~20% |
