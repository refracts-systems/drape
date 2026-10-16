# Step 5.0: Gambaran Besar Experiment 5, Menutup E4 lalu Kontak Tanpa Gesekan

> Step 4 membuat Newton GPU 15× lebih cepat, tetapi tidak menghasilkan "konfigurasi final" karena dua gate gagal. Step 5 punya dua bagian:
> **Part 0** menutup kedua kegagalan itu dengan aturan yang ditetapkan sebelum run, dan **Part 1** menambah fisika baru pertama sejak Step 3:
> kain yang jatuh ke bola di atas lantai, tanpa gesekan, tanpa boleh menembus.

---

## 1. Part 0: dua kegagalan E4

| Kegagalan E4 | Perbaikan (dari prompt) | Hasil |
|---|---|---|
| probe 1 berhenti karena batas iterasi: ambang aturan update (1,7×10⁻⁸ m) di bawah resolusi f32 posisi (~3×10⁻⁸ m) | ambang = max(10⁻⁴·langkah, 2·res(x)) | **tertutup**: probe 1 berhenti lewat aturan update (dokumen 5.2) |
| oracle n65: 5 langkah macet di 1–4×10⁻¹⁰ | konvergen juga kalau ≤ 2·ε64·‖Σ\|g_c\|‖/skala | **tidak menolong**: rumusnya 10⁵–10⁷× terlalu kecil (dokumen 5.3) |
| oracle n65: langkah 89 macet di 8×10⁻⁴ | diagnosis, lalu cap oracle 30 → 200 | **tertutup**: konvergen di iterasi 34 (dokumen 5.3) |

Rung D dibekukan sebagai **E4-final** (n65: 106 ms/langkah) dan menjadi patokan Part 1. Gate oracle n65 tetap gagal di langkah 1–4, dan itu dilaporkan.

## 2. Part 1: kontak dengan collider analitik

```
        sheet 0,5 m × 0,5 m, dilepas datar di z = 0,26 m
                 ↓ gravitasi
             ___________
            (    bola    )   r = 0,125 m, pusat (0,25; 0,25; 0,125)
=============================  lantai z = 0
```

Model (dokumen 5.4–5.5):

- **Barrier IPC** b(d) = −κ(d − d̂)² ln(d/d̂) untuk jarak d < d̂ = 0,5 mm: energi yang naik tanpa batas saat vertex mendekati permukaan. Penetrasi tidak
  pernah menjadi pilihan minimisasi.
- **κ tetap**: vertex terberat yang diam di atas collider seimbang di d̂/2.
- **α_max**: line search tidak boleh membiarkan vertex kehilangan lebih dari 90% jaraknya dalam satu percobaan, supaya tidak "melompati" permukaan.
- **Titik awal feasible**: Newton mulai dari x_n + α_max(x̂ − x_n), karena x̂ bisa berada di dalam bola.
- Hanya vertex vs collider: Hessian kontak mengisi blok diagonal saja, sehingga pola matriks dan preconditioner tidak berubah.

Di GPU (dokumen 5.8–5.10): suku kontak masuk ke kernel inersia yang sudah ada, kernel baru `ccd.wgsl` menghitung α_max, dan Newton tetap di host. Dengan
nol collider, E3/E4 bit-identik.

## 3. Hasil singkat (detail di dokumen 5.11)

| | n33 | n65 | n129 |
|---|---|---|---|
| ms/langkah (E4-final) | 127,9 (64,5) | 358,4 (106,4) | 1421,7 (371,5) |
| Newton/langkah | 7,2 | 10,3 | 10,5 |
| min d/d̂ (tidak pernah menembus) | 1,5×10⁻³ | 1,0×10⁻³ | 6,0×10⁻⁴ |
| konsistensi satu langkah | ≤ 2,3×10⁻⁴ | ≤ 1,5×10⁻⁴ | – |

Yang berhasil:

- kain jatuh, menyentuh bola di langkah 11 dan lantai di langkah 53–54;
- tidak ada vertex yang pernah menembus;
- energi selalu turun;
- setiap kernel kontak lulus toleransinya;
- tidak ada trigger berbasis bukti yang menyala.

Yang gagal, dan kenapa:

| Gate | Penyebab terukur |
|---|---|
| replay α_max ≤ 10⁻⁵ | jarak ke bola \|x − c\| − r di f32: galat ~10⁻⁴ relatif di d ~10⁻⁴ m |
| FD kontak (n65, n129), identitas barrier (n129) | pembulatan f64 pengukuran saat semua vertex aktif (tumbuh dengan √N), bukan turunan salah |
| oracle n65 | 47 langkah di lantai f64 (rumus lantai prompt terlalu kecil) + langkah 127 macet di 1,75×10⁻⁷ |
| berhenti benar di probe n33 | lantai f32: ΔΦ dari posisi f32 tidak bisa memastikan penurunan (line search gagal, batas iterasi) |
| ms/langkah n65 ≤ 2× E4-final | 358 > 213 ms |

## 4. Cara menjalankan

```bash
./run-e5.sh                               # Part 0 + Part 1, n33 + n65 (oracle) + n129
SIZES="33" GPU_ONLY="" ./run-e5.sh        # satu ukuran
```

Oracle E5 n65 butuh ±3 jam CPU. Progres per langkah dicetak ke stderr.

## 5. Daftar dokumen

| No | File | Isi |
|---|---|---|
| 5.1 | newton.rs | trace per iterasi, konvergensi di lantai f64, `max_step` |
| 5.2 | e4_rules.rs | aturan update dengan resolusi f32 |
| 5.3 | e4_diagnose.rs | langkah 89 dan lantai f64 |
| 5.4 | contact.rs | collider, barrier, α_max |
| 5.5 | drape.rs | scene, barrier di Φ, titik awal feasible |
| 5.6 | e3_generate.rs, e5_checks.rs | skema, probe kejadian, cek turunan kontak |
| 5.7 | e4_compare.rs, e5_compare.rs, e5_summarize.rs | penilaian |
| 5.8 | contact.wgsl, inertia.wgsl, inertia_energy_change.wgsl | barrier di GPU, ln yang teliti |
| 5.9 | ccd.wgsl | α_max di GPU |
| 5.10 | robust-newton.ts, contact-host.ts, drape-e4.ts | Newton host dengan kontak |
| 5.11 | run-e5.ts, run-e5.sh | runner, skrip, semua hasil |
