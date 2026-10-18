# Step 7.4: `run-e7.ts`, `outputs.ts`, `state-files.ts`, Runner dan Berkas Keluaran

> Runner Node yang menjalankan satu kasus Experiment 7 di GPU nyata dan menulis semua yang dibutuhkan penilai.

---

## 1. `run-e7.ts`: satu kasus, satu mode

```
node src/node/run-e7.ts --case ../drape7/drape-n33/base --mode cold [--start <awalan state>] [--checks]
```

| Mode | Titik awal |
|---|---|
| `cold` | posisi diam hasil `problem.bin` ("placed": simpul tengah 2·d̂ di atas bola), h = h0 |
| `warm` | keadaan setimbang GPU dari kasus `base` (`--start`), h = h akhir solve itu |
| `prolong` | hasil prolongasi n kasar → n halus (`--start`), h = h akhir solve kasar |

Alur: muat soal, buat backend `hardware` (`assertTargetAdapter` menolak iGPU dan adapter perangkat lunak, tanpa fallback diam-diam), siapkan GPU, lalu
bila `--checks`: tulis kernel dan probe. Terakhir `solveEquilibrium` dan tulis hasil. Variabel lingkungan opsional: `E7_PROGRESS=1` (satu baris per
langkah) dan `E7_BUDGET_S=<detik>` (batas dinding, alasan stop `time_budget`).

Pengaturan Newton dalam adalah rung D E6 (`floorStop`, `forcing`, `regularize`, `noiseAware`, `gpuReductions`, `fewReadbacks`).

## 2. Apa yang ditulis ke `gpu-<mode>/`

| File | Isi |
|---|---|
| `gpu-result.json` | skema `e7-gpu/1`: stop, langkah diterima, percobaan ditolak, iterasi Newton dan PCG, h akhir, r/L akhir, waktu total, waktu per langkah, waktu per fase, byte dibaca, info host |
| `gpu-ptc.bin` | 7 kolom per langkah diterima: step, h, residual, inner, rejected, pcgIterations, minDistance |
| `gpu-attempts.bin` | 6 kolom per percobaan: step, attempt, h, reason, recordStart, recordEnd |
| `gpu-iterations.bin`, `gpu-robust.bin` | catatan setiap iterasi Newton (format E6) |
| `gpu-states.bin` | semua keadaan diterima (hanya untuk mesh ≤ n65) |
| `gpu-equilibrium.{json,bin}` | keadaan akhir f32 + h akhir, dipakai sebagai awal `warm` dan prolongasi |
| `gpu-kernel.bin`, `gpu-contact-kernel.bin`, `gpu-inertia-h.bin`, `gpu-probe-*.bin` | hanya dengan `--checks` (putaran pertama di n17/n33) |

Berkas biner tidak masuk git (`.gitignore`), hanya JSON dan log.

## 3. `outputs.ts` dan `state-files.ts`

`outputs.ts` memuat pembantu yang dulu ada di `run-e6.ts` dan kini dipakai bersama E6 dan E7: nama kolom (`RECORD_FIELDS`, `EXTRA_FIELDS`), `table`
(baris → `Float64Array`), `bytesOf`, `join32`, `writeRun`, `newtonIterations`. Pemindahan ini tidak mengubah perilaku E6; regresi memastikan keluaran
E6 identik byte demi byte.

`state-files.ts` membaca dan menulis berkas keadaan f32: `<awalan>.json` (`vertices`, `final_h`, `dtype`) dan `<awalan>.bin` (x y z per simpul). Oracle
menulis format yang sama di `e7_state.rs`, jadi prolongasi oracle bisa dibaca GPU dan sebaliknya. `readState` menolak berkas yang bukan f32.

## 4. Pembacaan probe

`loadProbes` membaca `probes.json/.bin` dari oracle: langkah probe, h tiap langkah, dan posisi f32. Probe ada pada langkah diterima pertama, kontak bola
pertama, kontak lantai pertama, r/L ≤ 1e-1 pertama, dan r/L ≤ 1e-2 pertama. Dari tiap probe GPU menjalankan **satu** percobaan dan oracle menilai
selisihnya.
