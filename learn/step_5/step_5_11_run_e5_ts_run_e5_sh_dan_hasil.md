# Step 5.11: `run-e5.ts`, `run-e5.sh`, dan Hasil Experiment 5

> Dokumen penutup Step 5: runner, skrip, lalu semua angka. Isinya Part 0 (menutup E4), Part 1 (kontak), regresi, trigger, dan apa artinya dengan jujur.

---

## 1. `gpu/src/node/run-e5.ts`

Runner E5 adalah kembaran `run-e4.ts` dengan konfigurasi tetap **rung D** (E4-final):

```ts
const SETTINGS: RobustSettings = { floorStop: true, forcing: true, regularize: true };
```

Urutan kerjanya:

1. Tolak problem tanpa collider (`problem.bin has no colliders`). Analogi: validasi input di awal handler.
2. Penjaga adapter: hanya RTX 3050 yang diterima, tanpa fallback diam-diam.
3. Cek kernel E3 + inersia (collider dimatikan), lalu cek kernel kontak.
4. Probe, lalu lintasan penuh yang diukur waktunya per fase.
5. Tulis ke `drape5/drape-nN/gpu/`: file yang sama dengan E4, ditambah `gpu-contact-kernel.bin` dan `gpu-probe-directions.bin`. `robust_fields` berisi
   9 kolom.

## 2. `run-e5.sh`

```bash
# 1) generate semua soal dulu (oracle f64, CPU berat)
for n in $SIZES $GPU_ONLY; do
  ... e4-generate --final true ...       # Part 0
  ... e5-generate ...                    # Part 1
done
# 2) baru GPU: E4-final rung D, skor, ringkasan
# 3) GPU: E5, skor, ringkasan (patokan dari experiment4-final.json)
```

Diagnosis langkah 89 dijalankan kalau `drape4/drape-n65/reference.bin` ada.

**Waktu oracle.** E5 n65 butuh sekitar 3 jam CPU satu thread. Sebanyak 47 langkah macet di lantai f64 dan masing-masing menghabiskan seluruh 200 iterasi
(sekitar 9.200 iterasi dari 11.807 total, ~0,9 detik per iterasi untuk faktorisasi band-Cholesky 12.675×12.675). Oracle sekarang mencetak progres per
langkah ke stderr, supaya run panjang bisa dipantau.

## 3. Hardware

RTX 3050 Laptop GPU, driver 32.0.16.1714, D3D12 lewat vgpu, Windows 11 10.0.26300. Semua oracle selesai sebelum GPU pertama diukur.

## 4. Part 0: E4-final

### Hasil

| Kasus | Newton/langkah | PCG/solve | ms/langkah | Berhenti (residual / update / batas) |
|---|---|---|---|---|
| n33 | 4,34 | 46,6 | 64,5 | 0 / 240 / 0 |
| n65 | 4,71 | 92,9 | **106,4** | 0 / 240 / 0 |
| n129 | 4,56 | 149,7 | 371,5 | 0 / 239 / 1 |

Probe 1 sebelumnya berhenti karena batas iterasi. Sekarang:

| | E4 | E4-final |
|---|---|---|
| n33 probe 1 | batas iterasi | **aturan update**, lantai f32 0,93× lantai representasi |
| n65 probe 1 | batas iterasi | **aturan update**, 0,97× |

Oracle f64:

| | n33 | n65 |
|---|---|---|
| konvergen | 240/240 | **236/240** |
| lewat cabang lantai | 0 | 0 |
| langkah 89 | – | konvergen di iterasi 34 |
| langkah 111 | – | konvergen |
| tidak konvergen | – | langkah 1–4 (1,0–4,1×10⁻¹⁰) |

### Gate

Semua gate akurasi E4 lulus di n33 dan n65, kecuali `f64_converged_every_step` di n65. Gate performa lulus: 106,4 ≤ 174 ms dan Newton 4,71 ≤ 8,54 + 2.
Menurut aturan, konfigurasi final = **tidak ada**, dan sekarang hanya karena satu alasan: rumus lantai f64 dari prompt (ε64·‖Σ|g_c|‖) 10⁵–10⁷ kali lebih
kecil dari lantai yang benar-benar teramati di langkah 1–4 (dokumen 5.3). Konstantanya tidak disetel.

Langkah 89 konvergen, jadi syarat STOP dari prompt tidak berlaku. Rung D tetap dipakai sebagai patokan Part 1 dengan catatan ini.

## 5. Part 1: kontak

### Kejadian dan probe

| | n33 | n65 |
|---|---|---|
| kontak bola pertama | langkah 11 | langkah 11 |
| kontak lantai pertama | langkah 54 | langkah 53 |
| kontak terbanyak | langkah 60 (393 vertex) | langkah 60 (1.593 vertex, f64) |
| probe | 1, 11, 54, 60, 240 | 1, 11, 53, 60, 240 |

Jumlah vertex yang aktif per langkah (GPU dan f64 hampir sama): puncaknya saat sudut kain menghantam lantai (langkah 60), lalu kain meluncur dan diam
dengan 202 (n33), 737 (n65), dan 2.973 (n129) vertex menyentuh.

### Performa

| Kasus | Newton/langkah | PCG/solve | ms/langkah | perakitan | ccd | PCG | line search | I/O | lain | vs E4-final |
|---|---|---|---|---|---|---|---|---|---|---|
| n33 | 7,21 | 50,1 | 127,9 | 25,0 | 2,9 | 69,3 | 4,0 | 14,7 | 12,0 | 2,0× |
| n65 | 10,34 | 115,7 | **358,4** | 46,5 | 4,4 | 215,8 | 5,8 | 27,6 | 58,4 | 3,4× |
| n129 | 10,46 | 291,5 | 1421,7 | 181,6 | 13,7 | 913,0 | 7,3 | 51,2 | 255,0 | 3,8× |

- Kontak membuat Newton 1,7–2,3× lebih banyak iterasi dan PCG 1,1–1,9× lebih banyak iterasi per solve. Barrier kaku memperburuk kondisi matriks.
- CCD sendiri murah (≤ 1% waktu). Fase "lain" naik di n129 karena cek jarak f64 per iterasi di host (16.641 vertex × 2 collider).
- Pertumbuhan PCG 2,31× (n33→n65) dan 2,52× (n65→n129), di bawah batas 3×. Multigrid belum dipaksa.

### Hasil per cek

| Cek | n33 | n65 | n129 |
|---|---|---|---|
| kernel WGSL + inersia | ✓ | ✓ | ✓ |
| kernel kontak (entri butuh cond / gagal / terburuk÷tol) | 444 / 0 / 0,16 | 2.134 / 0 / 0,22 | 7.958 / 0 / 0,22 |
| FD Φ total (gradien / Hessian) | 1,5e-8 / 6,2e-9 | 3,4e-8 / 5,7e-10 | **3,1e-6** / 2,3e-10 |
| FD barrier (gradien / Hessian) | 3,0e-7 / 4,4e-7 | **2,4e-6** / 4,7e-7 | 5,4e-7 / 4,3e-7 |
| identitas energi barrier (≤ 10⁻¹²) | 1,8e-13 | 4,5e-13 | **1,8e-12** |
| bebas penetrasi (min d/d̂: iterasi / keadaan) | ✓ 1,5e-3 / 7,2e-3 | ✓ 1,0e-3 / 5,2e-3 | ✓ 6,0e-4 / 2,0e-3 |
| oracle f64 konvergen | ✓ 240/240 | **✗ 192/240** | – |
| konsistensi satu langkah (≤ 10⁻³) | ✓ ≤ 2,3e-4 | ✓ ≤ 1,5e-4 | – |
| berhenti benar di probe | **✗** (probe 60 batas, 240 line search) | ✓ | – |
| energi turun (f64, f32) | ✓ | ✓ | ✓ (f32) |
| replay α_max (≤ 10⁻⁵) | **✗ 7,5e-5** | **✗ 3,9e-4** | – |
| replay forcing, berhenti, resolusi | ✓ | ✓ | ✓ |

Probe (galat satu langkah / lantai f32 ÷ lantai representasi / alasan berhenti):

| Probe | n33 | n65 |
|---|---|---|
| 1 | 1,0e-4 / – / residual (jatuh bebas, residual 0) | 1,0e-4 / – / residual |
| 11 | 9,0e-6 / 0,99× / update | 9,2e-6 / 0,99× / update |
| 53–54 | 6,1e-6 / 1,02× / update | 1,3e-5 / 1,03× / update |
| 60 | 1,4e-4 / 1,20× / **batas iterasi** | 1,5e-4 / 1,05× / update |
| 240 | 2,3e-4 / 1,20× / **line search gagal** | 1,5e-4 / 0,98× / update |

### Membaca kegagalan

1. **Replay α_max** gagal karena f32, bukan karena bug. Jarak ke bola |x − c| − r adalah selisih dua angka ≈ 0,125 m, galatnya ≈ 7×10⁻⁹ m. Relatif
   terhadap d ≈ 10⁻⁴ m, itu ~10⁻⁴. Semua selisih yang melewati batas berasal dari bola. Trigger "rumus jarak stabil" di prompt hanya berlaku kalau gate
   kernel kontak gagal, dan gate itu lulus, jadi trigger tidak menyala.
2. **FD n65/n129 dan identitas n129** gagal karena pembulatan f64 pengukuran, bukan karena turunan salah. Keadaan uji menaruh **setiap** vertex di dalam
   barrier. Turunan berarah acak tumbuh ~√N, sedangkan jumlah energi yang dibulatkan tumbuh ~N, sehingga galat relatif FD dan selisih penjumlahan naik
   seiring mesh. Di n33 dan di tes unit n9 semua lulus. Langkah FD barrier sudah diubah sekali (dokumen 5.6) dan tidak diubah lagi.
3. **Oracle n65:** 47 langkah macet di 1,0–2,4×10⁻¹⁰, kebanyakan langkah 195–240 saat kain diam (skala residual kecil). Ini lantai f64 yang sama dengan
   E4-final, dan rumus lantai prompt tidak menangkapnya. Satu langkah (127) benar-benar macet di 1,75×10⁻⁷ setelah 200 iterasi.
4. **Probe n33:** di langkah 60 (sudut menghantam lantai), Newton f32 kehabisan 30 iterasi dengan lantai 1,2× representasi. Di langkah 240, line search
   gagal di lantai 1,2×. Uji coba n17 sudah menunjukkan penyebabnya: di dekat lantai, ΔΦ(α) yang dihitung naik **linear** terhadap α, padahal gradien
   menjanjikan turun. Gerak vertex ~10⁻⁶ m hanya ~30 ulp f32, dan gaya barrier yang menyeimbangkan gaya elastis harus tepat sampai 0,2%. Di n33, 21
   langkah lintasan berakhir dengan line search gagal; di n65 hanya 1, di n129 0.
5. **Performa:** n65 358 ms > 2 × 106 = 213 ms.

### Diagnostik

| | n33 | n65 | n129 |
|---|---|---|---|
| drift pusat massa maks | 9,5 µm | 1,4 mm | 5,9 mm |
| beda bentuk akhir GPU vs f64 (÷ L) | 5,7% | 18,8% | – |

Kesetimbangan simetris di atas bola licin tidak stabil. Lintasan f32 dan f64 makin menjauh seiring waktu (terutama n65), walaupun setiap **satu langkah**
konsisten sampai ≤ 2,3×10⁻⁴. Itu sebabnya cek ini satu-langkah dan drift tidak di-gate.

## 6. Trigger berbasis bukti

| Trigger | Terpicu? | Bukti |
|---|---|---|
| μ mencapai batas → buang b′∇²d | tidak | μ maks 10 (n33, n65), 100 (n129); tidak ada langkah di batas, di GPU maupun oracle |
| kernel kontak gagal di bola → rumus jarak stabil | tidak | 0 entri gagal di semua ukuran |
| konsistensi satu langkah gagal dengan min d/d̂ < 0,12 → d̂ = 10⁻²·L | tidak | konsistensi lulus di semua probe |
| pertumbuhan PCG ≥ 3× → multigrid | tidak | 2,31× dan 2,52× |

## 7. Regresi

| Regresi | Hasil |
|---|---|
| E1 tension-n33 | `gpu-x.bin` identik, 151 iterasi; hanya waktu berubah |
| E2 uniaxial-n33 | iterates dan iterations identik |
| E3 drape-n33 | lintasan, iterations, dan probe identik |
| E4-final rung D n33 | lintasan, iterations, robust identik dengan run Part 0 |

## 8. Ringkasan

| Pertanyaan | Jawaban |
|---|---|
| Probe 1 E4 tertutup? | ya, berhenti lewat aturan update di 0,93–0,97× lantai representasi |
| Langkah 89 E4? | bukan bug: daerah tangen indefinit, butuh 34 iterasi; konvergen dengan cap 200 |
| Lantai f64 E4? | rumus prompt terlalu kecil 10⁵–10⁷×; 4 langkah n65 tetap gagal (dilaporkan, tidak disetel) |
| Kontak berjalan di GPU f32? | ya: tidak pernah menembus (min d/d̂ 6×10⁻⁴), energi selalu turun, konsistensi satu langkah ≤ 2,3×10⁻⁴ |
| Semua gate akurasi lulus? | **tidak**: replay α_max (f32), FD/identitas di mesh besar (pembulatan f64 pengukuran), oracle n65 (lantai f64), probe n33 (lantai f32) |
| Performa? | n65 358 ms/langkah (3,4× E4-final), gate ≤ 213 ms gagal |
