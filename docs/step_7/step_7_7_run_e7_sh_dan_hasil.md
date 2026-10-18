# Step 7.7: `run-e7.sh`, `oracle-e7.sh`, dan Semua Hasil

> Skrip yang merangkai seluruh Experiment 7, jebakan yang ditemui saat menjalankannya, dan angka lengkap yang dihasilkan.

---

## 1. Skrip

**`oracle-e7.sh`**: membangun oracle, membuat berkas soal untuk base/w1/w2/w3 di n17, n33 (dan n65, n129 hanya soal), lalu per ukuran (paralel) base cold
dengan probe, kemudian w1–w3 cold dan warm, terakhir prolongasi n17 → n33. Progres tiap solve ke `<dir>/<mode>.log`. Ini kerja CPU berat; jangan
bersamaan dengan run GPU yang diukur waktunya.

**`run-e7.sh`** dengan tahap `STAGES="oracle gpu score"`:

- `gpu`: putaran diselang-seling, per ukuran: base cold (`--checks` hanya di putaran 1 untuk n ≤ 33), edit cold + warm (warm mulai dari keadaan
  setimbang GPU sendiri), lalu prolongasi dari base GPU ukuran sebelumnya. Putaran: Part 1 tiga kali di n17/n33/n65 dan dua di n129; Part 2–3 dua kali dan satu.
  Median ms/langkah dari `merge-runs.ts`.
- `score`: `e7-compare` per kasus dan mode, lalu `e7-summarize`.
- Variabel: `SIZES`, `GPU_ONLY`, `REPEATS*`, `FIRST_ROUND`, `PREVIOUS`, `BUDGET_LARGE`.

## 2. Jebakan

| Masalah | Penyebab | Perbaikan |
|---|---|---|
| `--start` tidak ditemukan | runner dijalankan dari `gpu/`, path relatif akar repo | awali dengan `../` |
| `SIZES=` kosong menjalankan n17 | `${SIZES:-…}` menganggap kosong = tidak diset | pakai `${SIZES-…}` |
| n129 tampak macet 8,9 jam | bukan hang; PCG sampai 200.000 iterasi per langkah | log progres + batas dinding 20 menit |
| run n129 mati diam-diam | proses hilang tanpa pesan (kemungkinan penyebab sistem) | diulang lewat skrip ekor |
| waiter latar belakang dibunuh | batas waktu perintah latar belakang | pasang ulang, jangan mengandalkannya sebagai penanda |

## 3. Hasil lengkap (RTX 3050 Laptop, D3D12, median)

### Part 1: kesetimbangan cold

| n | langkah | ditolak | Newton | PCG | detik | stop | r/L | lantai |
|---|---|---|---|---|---|---|---|---|
| 17 | 50 | 0 | 397 | 65.586 | 19,4 | stagnasi | 3,27e-2 | 1,14e-2 |
| 33 | 92 | 0 | 715 | 243.092 | 53,1 | stagnasi | 5,55e-2 | 4,27e-2 |
| 65 | 79 | 0 | 656 | 178.957 | 45,3 | stagnasi | 0,203 | – |
| 129 | 79 | 0 | 643 | 1.891.354 | ≥1210,8 | time_budget | 0,730 | – |

Fase waktu didominasi PCG: 77% di n17, 80% di n33, 75% di n65, 92% di n129. Pembacaan balik: 2,1 / 14,2 / 49,6 / 160,6 MiB.

### Part 2: edit (Newton cold → warm, rasio)

| n | w1 | w2 | w3 |
|---|---|---|---|
| 17 | 627 → 176 (0,28) | 446 → 221 (0,50) | 418 → 224 (0,54) |
| 33 | 506 → 193 (0,38) | 723 → 165 (0,23) | 792 → 219 (0,28) |
| 65 | 700 → 179 (0,26) | 673 → 218 (0,32) | 750 → 406 (0,54) |
| 129 | 446 → 1949 (4,4) | 745 → 1983 (2,7) | 917 → 3351 (3,7) |

Detik warm terburuk: 8,5 (n17), 21,5 (n33), 32,7 (n65), ≥1200 (n129). Di n129 semua cold dan warm berhenti di `time_budget` (satu di `cap`).

### Part 3: prolongasi

| Kasar → halus | prolong (detik) | total / halus cold (ms) | Newton prolong / cold | layak |
|---|---|---|---|---|
| 17 → 33 | 7,8 | 0,51 | 0,32 | ya |
| 33 → 65 | 33,5 | 1,91 | 0,60 | ya |
| 65 → 129 | ≥1201 | 1,03 | 4,2 | ya |

Residual awal hasil prolongasi r/L ≈ 5,5e2 sampai 5,8e2 (jauh dari setimbang); 187 simpul diproyeksikan keluar ke d = d̂ di n17 → n33.

### Gerbang dan pemicu

- Gerbang akurasi gagal: n17 base cold (residual akhir, satu-langkah), n33 base cold (satu-langkah), n33 prolong (residual akhir), n17 w1 warm dan n33 w2 cold (langkah μ-cap).
- Gerbang performa: warm ≤0,25 di n33 (1 dari 3 lolos), dua tingkat ≤0,5 (0,51, gagal tipis), cold n65 ≤10 s (45 s), cold n129 ≤30 s (≥1210 s).
- Pemicu menyala: pertumbuhan PCG 65 → 129 sebesar 10,8× (≥3×) → multigrid berikutnya; warm/cold > 0,5 di n17 w3, n65 w3, dan seluruh n129.
- Pemicu tidak menyala: solve cold dengan r/L jauh di atas lantai (tidak ada), prolongasi tak layak (tidak ada).
- Regresi: keluaran E1–E6 n33 identik byte demi byte, scoring E3–E6 tidak berubah.

## 4. Implikasi untuk editor

Dengan PCG block-Jacobi, drape akhir bisa dicapai dalam puluhan detik hingga n65 dan sekitar 1,4–5× lebih cepat (waktu dinding) setelah edit kecil, tetapi bukan interaktif.
Di n129 pengkondisi tidak lagi skala. Pekerjaan berikutnya yang paling berdampak adalah pengkondisi multigrid (atau geometris-hierarkis) untuk PCG;
warm start dan prolongasi sudah menunjukkan arah yang benar, tetapi tidak bisa menutup selisih ini sendirian.
