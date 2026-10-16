# Seri Belajar: Jersey Engine, dari Nol sampai Step 4

Seri ini menjelaskan **setiap file** yang dikerjakan di Step 1 (Experiment 1), Step 2 (Experiment 2), Step 3 (Experiment 3), dan Step 4 (Experiment 4) proyek `jersey-engine`.
Penulisnya mengasumsikan kamu adalah **backend developer**: paham bahasa pemrograman, API, thread, antrean job, database, dan JSON, tetapi
**belum pernah menulis kode GPU (WGSL / WebGPU)** dan tidak berlatar belakang komputasi numerik atau mekanika.

Setiap dokumen ditulis supaya bisa dibaca berdampingan dengan file aslinya. Potongan kode dikutip lalu dibedah baris per baris,
dan setiap istilah baru dijelaskan dengan analogi dunia backend.

## Cara membaca

1. Mulai dari `step_1/step_1_0_gambaran_besar.md`: masalah apa yang ingin dipecahkan, dan kenapa ada GPU, f32, f64, dan "oracle".
2. Baca berurutan. Urutannya mengikuti **alur data**, bukan urutan abjad file:
   masalah dibuat (Rust) → dikirim ke GPU (TypeScript) → dihitung (WGSL) → dibandingkan kembali (Rust) → diputuskan lulus/gagal.
3. Step 2 dibangun di atas Step 1, dan Step 3 di atas keduanya. Jangan lompat ke Step 2 sebelum paham `solver.ts` dan kernel PCG di Step 1,
   dan jangan ke Step 3 sebelum paham Newton dan perakitan gather di Step 2.

## Peta besar proyek

```
jersey-engine/
├── oracle/        Rust, f64 di mana-mana: "kunci jawaban" yang dipercaya
├── gpu/           TypeScript + WGSL: solver f32 di GPU, ini yang diuji
│   ├── kernels/   program GPU (WGSL), satu file = satu "fungsi" GPU
│   └── src/       kode host: menyiapkan buffer, mengirim kerja ke GPU, membaca hasil
├── cases/         data uji Step 1 (matriks, solusi referensi, hasil GPU)
├── coupon/        data uji Step 2 (kupon kain, langkah beban, hasil GPU)
├── drape/         data uji Step 3 (kain jatuh, lintasan, probe, hasil GPU)
├── drape4/        data uji Step 4 (referensi teregularisasi, hasil per rung ablation)
├── results/       ringkasan hasil per eksperimen + keputusan (verdict)
├── run-e1.sh      menjalankan Step 1 dari ujung ke ujung
├── run-e2.sh      menjalankan Step 2 dari ujung ke ujung
├── run-e3.sh      menjalankan Step 3 dari ujung ke ujung
└── run-e4.sh      menjalankan Step 4 (ablation A–D) dari ujung ke ujung
```

## Daftar isi

### Step 1: Solver linear f32 di GPU (Experiment 1)

| No | File | Isi |
|---|---|---|
| 1.0 | [gambaran besar](step_1/step_1_0_gambaran_besar.md) | Masalahnya, GPU vs CPU, f32 vs f64, oracle, gate lulus/gagal |
| 1.1 | [bsr.rs](step_1/step_1_1_bsr_rs.md) | Format matriks jarang (sparse) 3×3 blok: bahasa bersama CPU dan GPU |
| 1.2 | [cases.rs](step_1/step_1_2_cases_rs.md) | Tiga soal uji: tension, slack, dynamic, dan dari mana matriksnya datang |
| 1.3 | [band.rs](step_1/step_1_3_band_rs.md) | Cholesky pita: penyelesaian "pasti benar" untuk referensi |
| 1.4 | [pcg.rs](step_1/step_1_4_pcg_rs.md) | Algoritma PCG, emulasi f32 bit-per-bit, residual replacement |
| 1.5 | [generate.rs](step_1/step_1_5_generate_rs.md) | Membuat file soal + kunci jawaban, format biner, Lanczos |
| 1.6 | [state.wgsl](step_1/step_1_6_state_wgsl.md) | **Dasar WGSL bagian 1**: struct, buffer, tipe data, layout memori |
| 1.7 | [jacobi_invert.wgsl](step_1/step_1_7_jacobi_invert_wgsl.md) | **Dasar WGSL bagian 2**: thread, dispatch, satu thread per baris |
| 1.8 | [cg_init.wgsl](step_1/step_1_8_cg_init_wgsl.md) | **Dasar WGSL bagian 3**: memori workgroup, barrier, reduksi pohon |
| 1.9 | [spmv_dot.wgsl](step_1/step_1_9_spmv_dot_wgsl.md) | Perkalian matriks-vektor jarang + dot product |
| 1.10 | [reduce.wgsl](step_1/step_1_10_reduce_wgsl.md) | Reduksi akhir, `override`, mesin status CG di GPU |
| 1.11 | [update_dot.wgsl](step_1/step_1_11_update_dot_wgsl.md) | Update x, r, z dalam satu lintasan |
| 1.12 | [direction.wgsl](step_1/step_1_12_direction_wgsl.md) | Arah pencarian baru |
| 1.13 | [residual.wgsl](step_1/step_1_13_residual_wgsl.md) | Residual sejati, residual replacement, batas 10 storage buffer |
| 1.14 | [kernels.ts](step_1/step_1_14_kernels_ts.md) | Memuat file WGSL dan menempelkan header |
| 1.15 | [backend.ts](step_1/step_1_15_backend_ts.md) | Antarmuka 6 metode: abstraksi di atas WebGPU dan vgpu |
| 1.16 | [solver.ts](step_1/step_1_16_solver_ts.md) | Orkestrasi: buffer, urutan dispatch, cek konvergensi per chunk |
| 1.17 | [webgpu-backend.ts](step_1/step_1_17_webgpu_backend_ts.md) | WebGPU mentah: adapter, device, pipeline, bind group, timestamp |
| 1.18 | [vgpu-backend.ts](step_1/step_1_18_vgpu_backend_ts.md) | Jalur vgpu (Dawn di Node) |
| 1.19 | [problem.ts, cli.ts, files.ts](step_1/step_1_19_problem_ts_cli_ts_files_ts.md) | Parsing soal, argumen CLI, I/O file |
| 1.20 | [experiment.ts](step_1/step_1_20_experiment_ts.md) | Cold/warm run, determinisme, laporan waktu |
| 1.21 | [hardware.ts](step_1/step_1_21_hardware_ts.md) | Penjaga adapter: tolak iGPU dan software renderer |
| 1.22 | [run-browser.ts](step_1/step_1_22_run_browser_ts.md) | Chromium headless via Playwright, kisah DXC vs FXC |
| 1.23 | [run-vgpu.ts](step_1/step_1_23_run_vgpu_ts.md) | Runner vgpu |
| 1.24 | [compare.rs, criteria.rs](step_1/step_1_24_compare_rs_criteria_rs.md) | Menilai hasil GPU terhadap f64, aturan lulus/gagal |
| 1.25 | [summarize.rs, run-e1.sh](step_1/step_1_25_summarize_rs_run_e1_sh.md) | Ringkasan, pertumbuhan iterasi, skrip ujung ke ujung |
| 1.26 | [E1-VERDICT.md](step_1/step_1_26_E1_VERDICT_md.md) | Membaca hasil: kenapa E1 GAGAL hanya di speedup |

### Step 2: Mekanika nonlinear kain di GPU (Experiment 2)

| No | File | Isi |
|---|---|---|
| 2.0 | [gambaran besar](step_2/step_2_0_gambaran_besar.md) | Dari soal linear ke nonlinear: energi, gaya, Newton |
| 2.1 | [material.rs](step_2/step_2_1_material_rs.md) | Hukum material ortotropik St. Venant–Kirchhoff |
| 2.2 | [membrane.rs](step_2/step_2_2_membrane_rs.md) | Elemen segitiga: F, regangan Green, gaya, tangen, perubahan energi |
| 2.3 | [coupon.rs](step_2/step_2_3_coupon_rs.md) | Kupon uji, penjepit, prediktor, skala residual |
| 2.4 | [newton.rs](step_2/step_2_4_newton_rs.md) | Newton f64 + line search Armijo |
| 2.5 | [e2_checks.rs](step_2/step_2_5_e2_checks_rs.md) | Patch test, cek beda hingga (finite difference), jalur bolak-balik |
| 2.6 | [e2_generate.rs](step_2/step_2_6_e2_generate_rs.md) | File soal Step 2, peta "gather" untuk perakitan tanpa atomik |
| 2.7 | [membrane.wgsl](step_2/step_2_7_membrane_wgsl.md) | Header WGSL bersama: kolom F, regangan, tegangan |
| 2.8 | [membrane_element.wgsl](step_2/step_2_8_membrane_element_wgsl.md) | Kernel per elemen: gaya dan tangen |
| 2.9 | [membrane_energy_change.wgsl](step_2/step_2_9_membrane_energy_change_wgsl.md) | Perubahan energi tanpa pembatalan angka |
| 2.10 | [assemble_matrix.wgsl](step_2/step_2_10_assemble_matrix_wgsl.md) | Merakit matriks Newton ke format BSR |
| 2.11 | [assemble_force.wgsl](step_2/step_2_11_assemble_force_wgsl.md) | Merakit vektor gaya |
| 2.12 | [coupon.ts](step_2/step_2_12_coupon_ts.md) | Parsing soal kupon di TypeScript |
| 2.13 | [coupon-gpu.ts](step_2/step_2_13_coupon_gpu_ts.md) | Menyambungkan kernel membran ke solver E1 |
| 2.14 | [newton.ts](step_2/step_2_14_newton_ts.md) | Loop Newton f32 di host |
| 2.15 | [run-e2.ts](step_2/step_2_15_run_e2_ts.md) | Runner Step 2 dan format output biner |
| 2.16 | [e2_compare.rs, e2_summarize.rs](step_2/step_2_16_e2_compare_rs_e2_summarize_rs.md) | Menilai Newton f32 terhadap f64 |
| 2.17 | [run-e2.sh dan hasil](step_2/step_2_17_run_e2_sh_dan_hasil.md) | Hasil, lantai f32, dan artinya |

### Step 3: Kain dinamis 3D dengan tekukan (Experiment 3)

| No | File | Isi |
|---|---|---|
| 3.0 | [gambaran besar](step_3/step_3_0_gambaran_besar.md) | Backward Euler, hinge, probe satu langkah, trigger |
| 3.1 | [dual.rs](step_3/step_3_1_dual_rs.md) | Bilangan dual: turunan eksak otomatis |
| 3.2 | [hinge.rs](step_3/step_3_2_hinge_rs.md) | Sudut lipatan, Hessian eksak, perubahan energi hinge |
| 3.3 | [newton.rs](step_3/step_3_3_newton_rs.md) | Newton generik lewat trait `NewtonSystem` |
| 3.4 | [drape.rs](step_3/step_3_4_drape_rs.md) | Kain jatuh, potensial inkremental Φ, trigger tangen |
| 3.5 | [e3_checks.rs](step_3/step_3_5_e3_checks_rs.md) | Cek FD untuk Φ, riwayat energi |
| 3.6 | [e3_generate.rs](step_3/step_3_6_e3_generate_rs.md) | File soal, lintasan referensi, peta gather gabungan |
| 3.7 | [hinge.wgsl](step_3/step_3_7_hinge_wgsl.md) | Dual di WGSL, bug akurasi `atan2` |
| 3.8 | [hinge_element.wgsl](step_3/step_3_8_hinge_element_wgsl.md) | Gradien + Hessian hinge per thread |
| 3.9 | [hinge_energy_change.wgsl](step_3/step_3_9_hinge_energy_change_wgsl.md) | Perubahan energi hinge |
| 3.10 | [inertia.wgsl, inertia_energy_change.wgsl](step_3/step_3_10_inertia_wgsl_inertia_energy_change_wgsl.md) | Suku inersia backward Euler |
| 3.11 | [membrane_element.wgsl](step_3/step_3_11_membrane_element_wgsl.md) | Override `PROJECTED` (tegangan terjepit) |
| 3.12 | [kernels.ts, coupon.ts](step_3/step_3_12_kernels_ts_coupon_ts.md) | Perubahan kecil untuk reuse |
| 3.13 | [newton.ts](step_3/step_3_13_newton_ts.md) | Inti Newton f32 generik |
| 3.14 | [drape.ts](step_3/step_3_14_drape_ts.md) | Parsing soal drape |
| 3.15 | [drape-gpu.ts](step_3/step_3_15_drape_gpu_ts.md) | Buffer bersama, 12 pipeline |
| 3.16 | [drape-step.ts](step_3/step_3_16_drape_step_ts.md) | Langkah waktu f32, lintasan, probe |
| 3.17 | [run-e3.ts](step_3/step_3_17_run_e3_ts.md) | Runner Step 3 |
| 3.18 | [e3_compare.rs, e3_summarize.rs](step_3/step_3_18_e3_compare_rs_e3_summarize_rs.md) | Penilaian, probe, pertumbuhan PCG |
| 3.19 | [run-e3.sh dan hasil](step_3/step_3_19_run_e3_sh_dan_hasil.md) | Hasil per gate dan artinya |

### Step 4: Newton yang robust dan hemat (Experiment 4)

| No | File | Isi |
|---|---|---|
| 4.0 | [gambaran besar](step_4/step_4_0_gambaran_besar.md) | Ablation A–D, tiga perbaikan, hasil singkat |
| 4.1 | [newton.rs](step_4/step_4_1_newton_rs.md) | Regularisasi Levenberg–Marquardt K + μM/h² |
| 4.2 | [e4_rules.rs](step_4/step_4_2_e4_rules_rs.md) | Forcing Eisenstat–Walker, aturan berhenti, replay |
| 4.3 | [drape.rs, e3_generate.rs, e3_compare.rs](step_4/step_4_3_drape_rs_e3_generate_rs_e3_compare_rs.md) | Skema generator, folder per rung |
| 4.4 | [inertia.wgsl](step_4/step_4_4_inertia_wgsl.md) | μ di blok diagonal lewat buffer |
| 4.5 | [drape-gpu.ts, drape-step.ts, newton.ts](step_4/step_4_5_drape_gpu_ts_drape_step_ts_newton_ts.md) | Perubahan kecil untuk reuse |
| 4.6 | [timed-backend.ts](step_4/step_4_6_timed_backend_ts.md) | Decorator pengukur waktu per fase |
| 4.7 | [robust-newton.ts](step_4/step_4_7_robust_newton_ts.md) | Newton f32 dengan tiga saklar ablation |
| 4.8 | [drape-e4.ts, run-e4.ts](step_4/step_4_8_drape_e4_ts_run_e4_ts.md) | Langkah waktu robust, cek inersia, runner rung |
| 4.9 | [e4_compare.rs, e4_summarize.rs](step_4/step_4_9_e4_compare_rs_e4_summarize_rs.md) | Replay aturan, gate, rung final |
| 4.10 | [run-e4.sh dan hasil](step_4/step_4_10_run_e4_sh_dan_hasil.md) | Tabel ablation, gate per rung, kesimpulan |

## Glosarium cepat

| Istilah | Arti singkat | Analogi backend |
|---|---|---|
| GPU | Prosesor dengan ribuan inti kecil | Worker pool raksasa yang semua menjalankan fungsi yang sama |
| Kernel / shader | Fungsi yang dijalankan GPU | Handler yang dipanggil ribuan kali paralel, sekali per item |
| WGSL | Bahasa untuk menulis kernel WebGPU | "Bahasa query" untuk GPU, mirip Rust/C versi kecil |
| WebGPU | API standar untuk memakai GPU dari browser/Node | Driver database: cara membuka koneksi dan mengirim query |
| Dawn | Implementasi WebGPU buatan Google (dipakai Chrome) | Library klien database tertentu |
| vgpu | Library di atas WebGPU untuk Node | ORM tipis di atas driver |
| Buffer | Blok memori di GPU | Tabel/array di server lain yang harus di-upload dan di-download |
| Dispatch | Perintah "jalankan kernel ini N kali" | Submit batch job |
| Workgroup | Kelompok 256 thread yang berbagi memori cepat | Satu tim kecil dengan papan tulis bersama |
| f32 / f64 | Angka pecahan 32-bit / 64-bit | `float` vs `double` |
| Oracle | Program f64 di CPU yang jawabannya dipercaya | Test fixture / golden file |
| Residual | Seberapa jauh jawaban dari memenuhi persamaan | "Selisih saldo" setelah rekonsiliasi |
| Konvergen | Iterasi berhenti karena residual sudah cukup kecil | Retry loop yang berhasil |
| Gate | Syarat lulus/gagal yang ditetapkan sebelum mengukur | SLO / acceptance criteria |
