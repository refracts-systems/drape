# Step 6.7: `run-e6.sh` dan Hasil Experiment 6

> Dokumen penutup Step 6: skrip, semua angka, gate per rung, trigger, regresi, dan apa artinya dengan jujur. Hasil intinya negatif pada dua dari tiga hal
> yang diuji, dan itu informasi yang berguna.

---

## 1. Skrip

```bash
./run-e6.sh                                   # RUNGS="A B C D" SIZES="33 65" GPU_ONLY="129" REPEATS=3 REPEATS_LARGE=2
RUNGS="A" SIZES="33" GPU_ONLY="" REPEATS=1 ./run-e6.sh
```

1. Menolak berjalan kalau referensi oracle hilang (`drape4-final/`, `drape5/`), karena E6 tidak boleh membuat ulang referensi 3 jam.
2. `oracle e6-rescore`: Part 0, dari data tersimpan.
3. Untuk tiap ukuran: beberapa **putaran** tangga A–D berselang-seling; tiap run menyimpan `gpu-result-run<k>.json`; `merge-runs.ts` memilih median.
4. `e6-compare` per rung dan ukuran, lalu `e6-summarize`.

Seluruhnya memakan sekitar 2 jam, jadi dijalankan sebagai proses terpisah dari sesi (`Start-Process`), dan tidak ada kerja CPU berat selama run GPU
(aturan prompt).

## 2. Hardware

RTX 3050 Laptop GPU, driver 32.0.16.1714, D3D12 lewat vgpu, Windows 11. iGPU AMD dan renderer perangkat lunak ditolak oleh penjaga adapter.

## 3. Part 0: apa yang berubah

| Gate | Sebelum (hasil E5/E4-final) | Sesudah aturan Part 0 |
|---|---|---|
| E4-final n65 `f64_converged_every_step` | ✗ | ✓ (4 langkah konvergen lewat lantai) |
| E5 n33 `rules_replayed_ok` | ✗ | ✓ |
| E5 n65 `rules_replayed_ok` | ✗ | ✓ |
| E5 n65 `contact_fd_ok` | ✗ | ✓ |
| E5 n129 `barrier_energy_change_identity_ok` | ✗ | ✓ |
| E5 n33 `contact_fd_ok` | ✓ | **✗** |
| E5 n129 `contact_fd_ok` | ✗ | ✗ |
| E5 n65 `f64_converged_every_step` | ✗ | ✗ (langkah 127) |

Konvergensi oracle (0.3):

| | oleh toleransi | oleh lantai | gagal |
|---|---|---|---|
| E4-final n33 / n65 | 240 / 236 | 0 / 4 | 0 / 0 |
| E5 n33 / n65 | 240 / 192 | 0 / 47 | 0 / **1 (langkah 127)** |

Langkah 127 macet di 1,75×10⁻⁷ setelah 200 iterasi (rasio residual/lantai 225), bukan lantai pembulatan.

Replay α_max (0.1), dengan tolerans max(10⁻⁵, 8·ε32·cond): n33 membutuhkan cabang kondisi pada 6 dari 56 update, n65 pada 24 dari 67. Semua lulus
(rasio galat/toleransi terburuk 0,09 dan 0,10).

FD per patch (0.2), batas 10⁻⁶ untuk gradien, 10⁻⁵ untuk Hessian:

| | Φ gradien / Hessian | barrier gradien / Hessian | identitas |
|---|---|---|---|
| n33 | **4,5×10⁻⁶** / 3,2×10⁻⁷ | 4,2×10⁻⁷ / 7,1×10⁻⁷ | 2,8×10⁻¹⁵ |
| n65 | 8,0×10⁻⁷ / 2,3×10⁻⁷ | 7,9×10⁻⁷ / 7,7×10⁻⁷ | 2,7×10⁻¹⁵ |
| n129 | **4,6×10⁻⁶** / 3,1×10⁻⁸ | **2,7×10⁻⁶** / 6,9×10⁻⁷ | 1,8×10⁻¹⁴ |

Gate gagal di n33 dan n129 karena patch acak yang terambil memuat vertex yang jaraknya ke collider hanya ~2,5×10⁻⁵ m, dan langkah FD 10⁻⁷ m dari E3
punya galat truncation (h/d)² sendiri. Diagnostik tanpa gate dengan langkah Φ yang diskalakan menunjukkan turunan benar: gradien Φ 4,7×10⁻⁷ (n33),
5,6×10⁻⁸ (n65), 3,2×10⁻⁷ (n129). Angka itu **tidak** dipakai untuk meloloskan gate.

## 4. Part 1: tangga A–D

### Median ms/langkah (n run), byte baca per iterasi Newton

| Ukuran | Rung | ms/langkah | run (urut naik) | KiB/iterasi | rincian: perakitan / ccd / reduce / PCG / line search / I/O / lain |
|---|---|---|---|---|---|
| n33 | A | 123,7 | 120 / 124 / 124 | 71,1 | 21,3 / 2,4 / 0 / 69,1 / 4,1 / 14,4 / 12,3 |
| n33 | B | 118,9 | 116 / 119 / 120 | 71,1 | 21,6 / 2,9 / 0 / 65,4 / 3,9 / 13,4 / 11,7 |
| n33 | C | 120,6 | 120 / 121 / 123 | 18,5 | 19,9 / 2,7 / 2,1 / 65,3 / 4,1 / 16,3 / 10,2 |
| n33 | D | 118,8 | 118 / 119 / 121 | 13,7 | 16,5 / 2,9 / 2,0 / 66,2 / 3,8 / 17,3 / 10,0 |
| n65 | A | 383,1 | 380 / 383 / 462 | 270,8 | 47,6 / 4,8 / 0 / 226,4 / 5,8 / 30,8 / 67,6 |
| n65 | B | 391,0 | 363 / 391 / 492 | 270,8 | 49,5 / 6,8 / 0 / 230,6 / 5,3 / 29,5 / 69,3 |
| n65 | C | 370,3 | 370 / 370 / 437 | 81,6 | 46,2 / 4,1 / 3,7 / 225,0 / 5,9 / 32,4 / 52,9 |
| n65 | D | 405,5 | 361 / 406 / 417 | 63,6 | 47,8 / 8,1 / 4,0 / 263,3 / 4,9 / 38,2 / 39,0 |
| n129 | A | 1420,6 | 1388 / 1421 | 1020,1 | 191,9 / 11,4 / 0 / 908,1 / 6,8 / 50,3 / 252,1 |
| n129 | B | 1434,7 | 1415 / 1435 | 1020,1 | 191,0 / 10,9 / 0 / 916,9 / 5,3 / 49,5 / 261,1 |
| n129 | C | 1399,0 | 1372 / 1399 | 307,0 | 182,6 / 11,1 / 4,0 / 936,6 / 6,3 / 51,9 / 206,6 |
| n129 | D | 1316,5 | 1282 / 1317 | 235,9 | 186,5 / 11,9 / 3,8 / 939,9 / 5,1 / 56,7 / 112,5 |

Newton per langkah dan PCG per solve **identik di semua rung**: 7,21 / 50,1 (n33), 10,34 / 115,7 (n65), 10,46–10,47 / 291,4–291,5 (n129).

### Membaca tabelnya

- **Trajektori numerik identik antar rung.** Setiap probe (galat satu langkah, alasan berhenti, lantai f32) dan setiap jarak minimum bernilai sama di A, B, C, D.
  Fraksi langkah n33 yang alasan berhentinya berubah antar rung berurutan: 0% (batas pemicu 5%).
- **Byte baca turun, waktu hampir tidak.** KiB/iterasi di n65: 270,8 → 81,6 (C) → 63,6 (D), yaitu 0,30× dan 0,23× dari A; batas 0,1× tidak tercapai. Sisa
  pembacaan adalah solusi PCG (3n float), yang dibaca oleh `solveOnce` (kode E1) dan dibutuhkan host untuk memajukan posisi. Menurunkannya ke 0,1× akan
  menuntut posisi dan penyebaran arah tinggal di GPU, yaitu perubahan arsitektur yang di luar aturan eksperimen ini.
- **Waktu di n33 dan n65: tidak terbedakan dari noise.** Selisih antar rung (−3% sampai +6%) lebih kecil dari sebaran antar run pada rung yang sama (sampai 28%,
  misalnya A n65: 380 / 383 / 462). Fase PCG sendiri, yang kodenya tidak berubah, bervariasi 226–263 ms antar rung.
- **Satu efek nyata: fase "lain" di rung D.** Di n129 turun dari 252 menjadi 113 ms (−139 ms), karena evaluasi jarak minimum f64 di host (16.641 vertex × 2 collider
  per iterasi) hilang dari loop. Total hanya turun 7% karena PCG tidak berubah. Di n65 efek yang sama (68 → 39 ms) tertutup oleh PCG yang lebih lambat 37 ms di run D.
- **Submit lebih mahal dari baca.** Percobaan pertama memisahkan reduksi menjadi submit sendiri, dan rung C jadi lebih lambat dari A (134 vs 117 ms di n33): di
  jalur vgpu setiap submit punya latensi tetap yang lebih besar daripada membaca 50 KB. Menggabungkan reduksi ke submit kernel sebelumnya menghilangkan efek itu.

### Gate performa (vs rung A)

| Rung | n65 ms (≤ 0,8×) | n65 byte (≤ 0,1×) | Newton ≤ oracle + 2 | PCG growth < 3× |
|---|---|---|---|---|
| A | 1,00× | 1,00× | ✓ | ✓ |
| B | 1,02× ✗ | 1,00× ✗ | ✓ | ✓ |
| C | 0,97× ✗ | 0,30× ✗ | ✓ | ✓ |
| D | 1,06× ✗ | 0,23× ✗ | ✓ | ✓ |

## 5. Rung B: suku noise tidak menolong

| | n33 | n65 |
|---|---|---|
| percobaan α di probe / ditolak | 98 / 43 | 75 / 8 |
| update trajektori yang diterima hanya berkat noise | **0** | **0** |
| ekses terkecil percobaan yang ditolak, dalam satuan 4·ε32·Σ\|δ\| | **612** | **1.225** |
| update noise-only yang dievaluasi di f64 | 0 | 0 |
| pelanggaran keamanan (ΔΦ > 2× batas) | 0 | 0 |

Rung B tidak melanggar apa pun (tidak ada update noise-only untuk dilanggar), dan trigger "buang rung B" **tidak menyala**. Tetapi ia juga tidak mengubah satu
keputusan pun: trajektorinya identik dengan A. Percobaan yang paling dekat diterima masih melebihi batas noise 612 kali, jadi kegagalan di lantai f32
**bukan noise penjumlahan Σδ**. Penyebabnya, sesuai ukuran di n17 (dokumen 5.11 dan 6.4), adalah kuantisasi posisi: vertex bergerak ~30 ulp f32, dan
perubahan energi barrier b(x + Δ) − b(x) dari posisi yang dibulatkan tidak konsisten dengan gradien pada tingkat 0,2%. Aturan 4·ε32·Σ|δ| mengukur galat yang salah.

## 6. Gate akurasi per rung

17 gate; `-` = tidak berlaku. Urutan: kernel, kernel kontak, FD gradien, FD Hessian, FD kontak, identitas hinge, identitas barrier, bebas penetrasi, oracle
konvergen, satu langkah, berhenti di probe, energi f64, energi f32, replay aturan, replay penerimaan, akurasi reduksi, keamanan noise.

| | A | B | C | D |
|---|---|---|---|---|
| n33 | YYYY**n**YYYYY**n**YYYY-- | YYYY**n**YYYYY**n**YYYY-Y | YYYY**n**YYYYY**n**YYYYYY | YYYY**n**YYYYY**n**YYYYYY |
| n65 | YYYYYYYY**n**YYYYYY-- | YYYYYYYY**n**YYYYYY-Y | YYYYYYYY**n**YYYYYYYY | YYYYYYYY**n**YYYYYYYY |

Setiap rung gagal pada gate yang **sama**: n33 gagal di `contact_fd_ok` (langkah FD, bagian 3) dan `f32_stops_ok_at_probes` (probe 60 berhenti di batas iterasi,
probe 240 karena line search gagal, keduanya di 1,20× lantai representasi); n65 gagal di `f64_converged_every_step` (langkah 127). Karena itu **konfigurasi
final = tidak ada** menurut aturan. Gate khusus E6 lulus di semua rung yang memilikinya: replay penerimaan, akurasi reduksi (1,3–1,8×10⁻⁷, batas 10⁻⁵), dan keamanan noise.

Probe n33 (alasan berhenti / lantai f32 ÷ lantai representasi): langkah 11 update 0,99×, langkah 54 update 1,02×, langkah 60 batas iterasi 1,20×,
langkah 240 line search gagal 1,20×. Probe n65: semua berhenti lewat aturan update, 0,98–1,05×.

Jarak minimum ke collider: n33 1,5×10⁻³·d̂, n65 1,0×10⁻³·d̂, n129 6,0×10⁻⁴·d̂ (tidak pernah menembus). Di rung D, jarak itu diukur di f32 pada tiap iterate yang
diterima (1,01×10⁻³·d̂ di n65), dan pemeriksaan f64 tetap dilakukan pada 241 keadaan lintasan dan iterate probe. Drift pusat massa: 9,5 µm (n33), 1,4 mm (n65),
5,9 mm (n129), hanya diagnostik.

## 7. Trigger

| Trigger | Terpicu? |
|---|---|
| rung B melanggar keamanan atau konsistensi > 2× | tidak (0 update noise-only; galat satu langkah B = A) |
| C atau D mengubah alasan berhenti > 5% | tidak (0%) |
| μ mencapai batas | tidak (μ maksimum 10 di n33/n65, 100 di n129) |
| PCG growth ≥ 3× | tidak (2,3× dan 2,5×) |

## 8. Regresi

E1 tension-n33, E2 uniaxial-n33, E3 drape-n33, E4-final rung D n33 dan E5 n33 dijalankan ulang setelah semua perubahan: seluruh keluaran biner **identik** (kecuali
waktu). Rung A di n33 identik bit-per-bit dengan E5, dibandingkan pada 6 file, dan pada 9 kolom pertama `gpu-robust.bin`.

## 9. Ringkasan

| Pertanyaan | Jawaban |
|---|---|
| Apakah aturan pengukuran baru menutup kegagalan E5? | sebagian: 5 gate berubah menjadi lulus, 1 berubah menjadi gagal (n33 FD), 2 tetap gagal |
| Apakah line search sadar noise memperbaiki lantai f32? | **tidak**: 0 update diterima karenanya; kegagalan 600–1.200× di atas batas noise |
| Apakah reduksi GPU mempercepat? | tidak terbukti: byte baca turun 3–4×, waktu berubah −3% sampai +6%, di dalam noise |
| Apa yang benar-benar menolong? | menghapus evaluasi jarak f64 dari loop (rung D): fase "lain" −139 ms di n129 |
| Apa penghambat sebenarnya? | PCG (kernel E1) 60–66% waktu; sisanya terhitung kecil |
| Rung final? | tidak ada (gate yang sama gagal di semua rung) |
