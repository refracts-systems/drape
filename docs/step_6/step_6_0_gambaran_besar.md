# Step 6.0: Gambaran Besar Experiment 6, Newton Kontak yang Jujur tentang Lantai f32

> Step 5 membuat kain menyentuh bola dan lantai tanpa menembus, tetapi meninggalkan enam gate gagal dan kecepatan 3,4× lebih lambat dari E4-final.
> Step 6 bertanya dua hal: **apakah gate-gate itu gagal karena fisikanya atau karena cara kita mengukurnya**, dan **apakah Newton kontak bisa dibuat
> lebih hemat per iterasi**. Jawabannya campuran, dan sebagian negatif.

---

## 1. Dua bagian

| Bagian | Pertanyaan | Cara |
|---|---|---|
| Part 0 | Apakah kegagalan E5 adalah galat pengukuran? | empat aturan baru, diterapkan ke data tersimpan (tanpa GPU, tanpa membuat ulang oracle 3 jam) |
| Part 1 | Bisakah Newton kontak lebih tangguh dan hemat? | tangga ablation A–D di scene E5 |

## 2. Part 0: empat aturan pengukuran (dokumen 6.1)

| Aturan | Isinya | Hasil |
|---|---|---|
| 0.1 | toleransi replay α_max sadar kondisi: max(10⁻⁵, 8·ε32·\|q\|/d0) | replay lulus di n33 dan n65 |
| 0.2 | FD pada patch 16 vertex dengan jumlah lokal | lulus di n65, **gagal di n33 dan n129** karena langkah FD 10⁻⁷ m terlalu besar untuk vertex yang sangat dekat |
| 0.3 | lantai f64 ε64·(‖Σ\|g_c\|‖ + ‖\|K\|·\|x\|‖): memasukkan pembulatan posisi | E4-final n65 konvergen; E5 n65 tinggal langkah 127 |
| 0.4 | skor ulang semua | 5 gate lulus, 1 menjadi gagal, 2 tetap |

## 3. Part 1: tangga ablation (dokumen 6.2–6.6)

| Rung | Perubahan | Hipotesis |
|---|---|---|
| A | E5 apa adanya | patokan |
| B | + line search sadar noise: terima jika Σδ ≤ c·α·gᵀd + 4·ε32·Σ\|δ\| | line search gagal karena noise penjumlahan |
| C | + reduksi GPU (norma, kemiringan, jumlah energi) | membaca seluruh buffer ke host itu mahal |
| D | + minimum α_max dan jarak di GPU | idem untuk batas dan jarak |

## 4. Hasil singkat (detail di dokumen 6.7)

- **Rung B tidak mengubah apa pun.** Tidak ada satu pun langkah yang diterima berkat suku noise. Percobaan yang paling dekat diterima masih 612× (n33) dan 1.225×
  (n65) di atas batas noise. Penyebab kegagalan di lantai f32 adalah kuantisasi posisi, bukan noise penjumlahan.
- **Rung C dan D menurunkan byte baca 3–5×** (n65: 270,8 → 81,6 → 63,6 KiB per iterasi Newton) tetapi **tidak terbukti mempercepat**: selisih waktu −3% sampai
  +6% di dalam noise run-ke-run (sampai 28%).
- **Satu efek nyata:** rung D menghapus evaluasi jarak f64 dari loop, dan fase "lain" di n129 turun 139 ms.
- Trajektori identik antar rung; tidak ada trigger yang menyala; regresi E1–E5 bit-identik.
- Tidak ada rung final: gate yang sama gagal di setiap rung (FD n33, probe n33, oracle n65 langkah 127).

## 5. Pelajaran pengukuran

GPU laptop melambat karena panas setelah run panjang: rung yang sama bisa 30% lebih lambat setelah n129. Karena itu `run-e6.sh` menjalankan tangga berselang-seling
beberapa putaran dan mengambil median (dokumen 6.5). Perbedaan antar rung di bawah sekitar 15% tidak boleh dibaca sebagai efek nyata.

## 6. Daftar dokumen

| No | File | Isi |
|---|---|---|
| 6.1 | oracle: drape.rs, contact.rs, e5_checks.rs, e5_compare.rs, e6_rescore.rs | aturan pengukuran Part 0 |
| 6.2 | sums.wgsl, argmin.wgsl, reductions.ts | reduksi di GPU |
| 6.3 | ccd.wgsl, timed-backend.ts | jarak per baris, fase reduce, penghitung byte |
| 6.4 | newton-ops.ts | tiga pengukuran Newton, line search sadar noise |
| 6.5 | robust-newton.ts, drape-e4.ts, contact-host.ts, run-e6.ts, merge-runs.ts | merangkai rung, median run |
| 6.6 | e6_compare.rs, e6_summarize.rs | penilaian, trigger, rung final |
| 6.7 | run-e6.sh dan hasil | semua angka |
