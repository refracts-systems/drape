# Step 1.0: Gambaran Besar Experiment 1

> Tujuan dokumen ini: setelah membaca, kamu bisa menjelaskan ke rekan kerja **kenapa** Experiment 1 ada, **apa** yang diukur,
> dan **bagaimana** semua file di Step 1 saling terhubung. Belum ada kode GPU di sini; itu mulai dari dokumen 1.6.

---

## 1. Proyek ini mau membuat apa?

Ujung akhirnya adalah **simulator kain jersey** yang berjalan di browser: kain ditarik, dipakaikan ke badan, melar, dan kusut,
dihitung secara fisik (bukan animasi palsu). Metodenya disebut **implicit thin-shell FEM** (Finite Element Method untuk
cangkang tipis, diselesaikan secara implisit).

Kamu tidak perlu paham FEM dulu. Yang penting adalah satu fakta ini:

> Setiap langkah simulasi berakhir pada satu tugas yang sama: **menyelesaikan sistem persamaan linear raksasa** `A x = b`.

- `A` adalah matriks kekakuan: "seberapa keras setiap titik kain menolak digeser, dan bagaimana titik-titik saling memengaruhi".
- `b` adalah gaya yang bekerja (gravitasi, tarikan).
- `x` adalah yang dicari: berapa jauh setiap titik bergeser.

Untuk kain jersey beresolusi 2 mm, ada sekitar **65.000 titik**. Setiap titik punya 3 koordinat (x, y, z), jadi ada
**195.075 bilangan yang tidak diketahui**. Sistem persamaan dengan 195 ribu variabel harus diselesaikan **berkali-kali per detik**.

Kalau ini tidak bisa cepat dan akurat, seluruh program jersey tidak bisa jalan. Karena itu README proyek menyebut:
*"Everything else in the jersey programme waits on this."*

## 2. Analogi backend: kenapa ini sulit?

Bayangkan tabel `transfers` dengan 195 ribu akun. Saldo akhir tiap akun bergantung pada saldo akun tetangganya
(misalnya rata-rata tertimbang tetangga plus setoran). Kamu harus mencari saldo semua akun sekaligus yang konsisten dengan
semua aturan.

- **Cara pasti** (seperti `band.rs`): eliminasi Gauss/Cholesky. Hasilnya tepat, tetapi butuh memori dan waktu besar.
  Untuk 195 ribu variabel, oracle butuh sekitar 2,5 GB RAM dan 3–6 menit di CPU.
- **Cara iteratif** (PCG): tebak, hitung "selisih" (residual), perbaiki tebakan, ulangi. Setiap putaran murah:
  pada dasarnya satu perkalian matriks-vektor dan beberapa penjumlahan. Kalau soalnya "jinak", ratusan sampai ribuan
  putaran sudah cukup.

PCG cocok untuk GPU karena setiap putaran terdiri dari operasi yang sama untuk 65 ribu baris secara independen.
Ini persis bentuk kerja yang disukai GPU: **data-parallel**.

## 3. GPU dalam 5 menit untuk backend developer

| Konsep | CPU (yang kamu kenal) | GPU |
|---|---|---|
| Jumlah inti | 6–16 inti besar | ribuan inti kecil (RTX 3050 Laptop: 2048 "CUDA core") |
| Gaya kerja | tiap thread mengerjakan hal berbeda | semua thread menjalankan **fungsi yang sama** pada data berbeda |
| Memori | RAM bersama, cache besar | VRAM sendiri (4 GB di laptop ini), bandwidth tinggi (176 GB/s) |
| Komunikasi | panggil fungsi | upload buffer, kirim perintah, tunggu, download hasil |
| Yang mahal | komputasi berat per item | **memindahkan data** dan **sinkronisasi** |

Analogi yang paling dekat: GPU adalah **worker pool dengan 10.000 worker** yang semuanya hanya bisa menjalankan satu handler
yang sama per batch. Kamu tidak bisa bilang "worker 5 kerjakan A, worker 6 kerjakan B" dengan bebas; yang bisa kamu lakukan
adalah memberi tahu "handler ini dijalankan 65.025 kali, dan setiap pemanggilan diberi nomor urut". Handler memakai nomor itu
untuk tahu baris mana yang dikerjakan.

Bahasa untuk menulis handler itu adalah **WGSL** (WebGPU Shading Language). Cara mengirim handler dan data ke GPU dari
JavaScript/TypeScript adalah **WebGPU API**.

## 4. f32 vs f64: inti pertanyaan Experiment 1

GPU konsumen (dan WebGPU secara umum) hanya mendukung **f32** (angka pecahan 32-bit, kira-kira 7 digit desimal).
CPU dengan mudah memakai **f64** (64-bit, kira-kira 16 digit).

Analogi uang: f64 seperti menghitung dalam sen dengan 16 digit, f32 seperti hanya punya 7 digit. Kalau saldo kamu
Rp 12.345.678, dengan 7 digit kamu hanya bisa menulis Rp 12.345.680; Rp 2 hilang karena pembulatan.
Satu pembulatan tidak masalah, tetapi PCG melakukan **jutaan** operasi, dan galat pembulatan bisa menumpuk.

Pertanyaan Experiment 1 jadi:

> Bisakah solver PCG **f32** di GPU (lewat WebGPU) menyelesaikan sistem linear FEM kain
> **cukup akurat**, **deterministik**, dan **cukup cepat**?

"Cukup akurat", "deterministik", dan "cukup cepat" didefinisikan sebagai **gate**: angka batas yang ditulis di kode
**sebelum** mengukur, supaya tidak ada godaan menggeser batas setelah melihat hasil.

## 5. Siapa yang dipercaya? Konsep "oracle"

Kita butuh kunci jawaban. Itulah **oracle**: program Rust di folder `oracle/` yang:

1. Membangun soal (matriks `A`, vektor `b`) dalam f64.
2. Menyelesaikannya dengan **Cholesky pita f64**, metode langsung tanpa iterasi. Jawabannya `x_exact`.
3. Menjalankan PCG yang sama **dua kali di CPU**: sekali f64, sekali f32 dengan **urutan operasi yang persis sama
   dengan kode WGSL**. Versi f32 ini disebut **emulasi**: "beginilah seharusnya GPU berperilaku kalau ia mengikuti aturan
   pembulatan f32 yang standar".
4. Setelah GPU selesai, oracle membaca jawaban GPU dan menilainya: seberapa jauh dari `x_exact`, seberapa besar residual
   sebenarnya kalau dihitung ulang dalam f64, apakah cocok dengan emulasi f32, dan seterusnya.

Analogi: oracle seperti **golden-file test**. Program yang diuji (GPU) menghasilkan output, lalu pembanding yang dipercaya
(oracle) memutuskan apakah output itu benar.

## 6. Alur data Experiment 1

```
┌─────────────────────────────── oracle (Rust, f64) ───────────────────────────────┐
│ cases.rs: bangun mesh kain + matriks A + vektor b                                │
│ band.rs : selesaikan A x = b secara pasti  → x_exact                             │
│ pcg.rs  : PCG f64 dan PCG f32 (emulasi GPU) → jumlah iterasi, waktu CPU           │
│ generate.rs: tulis problem.{json,bin} dan reference.{json,bin}                   │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                        │ file
┌───────────────────────────────────────▼─────────── gpu (TypeScript + WGSL) ──────┐
│ files.ts / problem.ts: baca problem.json + problem.bin                           │
│ kernels.ts: baca 7 file .wgsl                                                     │
│ webgpu-backend.ts atau vgpu-backend.ts: buka GPU, buat buffer, kompilasi kernel   │
│ solver.ts: urutan dispatch PCG, cek konvergensi tiap 32 iterasi                   │
│ experiment.ts: ulangi 3x (cold/warm), cek determinisme, ukur waktu                │
│ hardware.ts: pastikan GPU = RTX 3050, bukan iGPU AMD atau software                │
│ tulis gpu-x.bin (jawaban) + gpu-result.json (laporan)                            │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                        │ file
┌───────────────────────────────────────▼─────────────── oracle lagi ──────────────┐
│ compare.rs: residual sejati f64, galat vs f64, cocok dengan emulasi?             │
│ criteria.rs: gate per kasus → result.json                                        │
│ summarize.rs: semua kasus → results/experiment1-*.json + pertumbuhan iterasi     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

Semua ini dijalankan oleh satu skrip: `run-e1.sh`.

## 7. Tiga soal uji

| Kasus | Fisika | Sifat matriks | Peran |
|---|---|---|---|
| `tension` | kain ditarik dua arah (prategang) | "jinak" (well-conditioned) | kasus produksi utama |
| `dynamic` | kain kendur + inersia langkah waktu 1/240 s | jinak berkat massa | kasus produksi kedua |
| `slack` | kain kendur tanpa inersia | sangat buruk (κ sampai 8×10⁸) | **diagnostik yang memang diharapkan gagal** |

Setiap kasus dijalankan di 4 resolusi: grid 33, 65, 129, dan 257 kolom. n257 setara jersey dengan elemen 2 mm.

## 8. Gate Experiment 1

Ditulis di `oracle/src/criteria.rs` dan `summarize.rs`:

| Gate | Arti untuk orang awam | Batas |
|---|---|---|
| `converged` | iterasi berhenti karena berhasil, bukan karena kehabisan putaran | status = converged |
| `forcing_term_ok` | residual sejati (dihitung ulang f64) cukup kecil untuk dipakai Newton | ≤ 1×10⁻² |
| `f32_iterations_ok` | f32 tidak butuh terlalu banyak putaran tambahan dibanding f64 | ≤ 1,5× iterasi f64 |
| `faithful_to_cpu_f32_emulation` | GPU berperilaku seperti emulasi f32 di CPU | iterasi dan solusi mirip |
| `deterministic` | 3 kali menjalankan → bit yang persis sama | identik |
| `speedup_ok` | GPU ≥ 10× lebih cepat daripada 1 inti CPU f64 | ≥ 10× |
| pertumbuhan iterasi | saat mesh diperhalus 2×, iterasi tidak meledak | < 3× per penghalusan |

Ditambah gate dari permintaan pengguna:

- overhead vgpu ≤ 1,15× WebGPU mentah,
- *storage error* f32 di n257 ≤ 1×10⁻³,
- bandwidth efektif ≥ 25% dari spesifikasi (kalau tidak, ganti ke format "sliced BSR").

Kenapa 1×10⁻² untuk residual? Solver linear ini akan dipakai di dalam metode **Newton** (Step 2). Newton tidak butuh
solusi linear yang sempurna; cukup residual turun 100×, maka setiap iterasi Newton tetap memberi sekitar 2 digit
akurasi tambahan. Istilahnya **inexact Newton** dengan *forcing term* 10⁻².

## 9. Hasil singkat (detail di dokumen 1.26)

- Semua gate akurasi **lulus** di n257, tetapi hanya setelah **residual replacement** diaktifkan (dokumen 1.4 dan 1.13).
- **Speedup gagal**: tension 6,57×, dynamic 8,62× (butuh 10×). GPU-nya stabil; patokan CPU-nya justru tidak stabil
  (8,06 s sampai 12,2 s untuk soal yang sama).
- Keputusan: **E1 = GAGAL hanya di `speedup_ok`**, dan pertanyaan speedup dibiarkan terbuka untuk hardware yang lebih kuat.

## 10. Peta dokumen Step 1

Urutan baca berikutnya:

1. Data: `bsr.rs` (1.1) → `cases.rs` (1.2) → `band.rs` (1.3) → `pcg.rs` (1.4) → `generate.rs` (1.5)
2. GPU: `state.wgsl` (1.6, dasar WGSL) → tujuh kernel (1.7–1.13)
3. Host: `kernels.ts` → `backend.ts` → `solver.ts` → dua backend → runner
4. Penilaian: `compare.rs`, `criteria.rs`, `summarize.rs`, verdict

## 11. Cara menjalankan

```bash
# Semua kasus, WebGPU mentah di Chromium, GPU hardware
CASES="tension dynamic slack" SIZES="33 65 129 257" REPLACE_EVERY=50 REPLACE_ABOVE=1e-2 ./run-e1.sh webgpu hardware

# Jalur vgpu (Node)
CASES="tension dynamic slack" SIZES="33 65 129 257" REPLACE_EVERY=50 REPLACE_ABOVE=1e-2 ./run-e1.sh vgpu hardware
```

Di Windows, jalankan lewat Git Bash. Waktu terlama adalah oracle di n257 (Cholesky beberapa menit per kasus).
