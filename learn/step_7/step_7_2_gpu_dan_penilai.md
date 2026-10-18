# Step 7.2: Sisi GPU, Penilai, dan Cara Menjalankan Experiment 7

> Dokumen ini menjelaskan bagian Experiment 7 yang berjalan di GPU dan cara hasilnya dinilai terhadap oracle f64.

---

## 1. Satu langkah PTC di GPU (`gpu/src/drape-e7.ts`)

Untuk setiap percobaan dengan langkah h:

1. `prepare` menulis ulang `row_mass = M/h²` (dari massa satuan f32), `x̂ = x_k + h²g` (f32), dan μ = 0.
2. Titik awal Newton dibuat layak: `x_k + α_max (x̂ − x_k)`.
3. Newton dalam E4-final (LM, forcing EW, henti sadar-lantai, α ≤ α_max) dengan toleransi 0,1 dan maksimum 10 pembaruan; reduksi dan arg-min di GPU (rung D E6).
4. `classify` menolak percobaan bila Newton berakhir sebelum pembaruan pertama karena batas regularisasi atau gagal line-search, atau keadaan tidak hingga. Ulangi dengan h/2 (minimum h0/10), maksimal 12 kali, lalu "stalled".
5. **Pemicu μ-cap**: bila percobaan yang diterima berakhir di batas μ setelah ada kemajuan, ulangi tanpa suku b′∇²d (flag `colliders[3] = 1`). Percobaan yang digantikan dicatat dengan alasan 4 (`curvature_retry`).
6. Pengontrol: h naik ×2 bila <10 pembaruan dalam, turun ×0,5 bila tidak, dibatasi [h0/10, 1 s].

Aturan berhenti f32 (`ptc.ts`): (a) r/L ≤ 1e-3, (b) stagnasi (minimum 10 terakhir ≥ 0,95× minimum 10 sebelumnya **dan** r/L ≤ 0,25), (c) batas 1000 langkah
di n17/n33 dan 400 di n65/n129, (d) stalled. Satu tambahan di luar prompt: `time_budget`, batas dinding 20 menit yang hanya dipasang untuk n129 lewat
`E7_BUDGET_S`, karena solve n129 sampai 8,9 jam tanpa hasil; alasan stop itu dilaporkan apa adanya.

## 2. Penilai (`e7_compare.rs`, `e7_summarize.rs`)

`e7-compare --dir --mode` menulis `gpu-<mode>/result.json` dengan bendera: kernel WGSL, kernel kontak, kernel inersia pada h ∈ {h0, 0,1, 1,0}, bebas-tembus,
replay aturan (forcing, terminasi, α_max, penerimaan, pengontrol sampai 1e-12, penolakan), E_s tidak naik, oracle konvergen tanpa langkah μ-cap, residual
akhir f64 ≤ max(1e-3, 2× lantai representasi), satu-langkah ≤1e-3 di probe. `e7-summarize` menyusun tabel Part 1–3, gerbang, dan temuan, ke
`results/experiment7-vgpu-hardware.json`.

## 3. Cara menjalankan

```
./run-e7.sh                     # tahap oracle, gpu, score (STAGES="oracle gpu score")
STAGES=gpu FIRST_ROUND=2 SIZES="17 33" GPU_ONLY="65" ./run-e7.sh
STAGES=gpu PREVIOUS=65 SIZES= GPU_ONLY=129 REPEATS=1 ./run-e7.sh
```

Putaran GPU diselang-seling (laptop GPU melambat saat panas) dan median ms/langkah diambil oleh `merge-runs.ts`. Jangan menjalankan kerja CPU berat
(oracle atau scoring) bersamaan dengan run GPU yang diukur waktunya. `E7_PROGRESS=1` mencetak satu baris per langkah.

## 4. Jebakan yang ditemui

- `${SIZES:-17 33}` menganggap string kosong sebagai tidak diset; pakai `${SIZES-17 33}` agar `SIZES=` benar-benar kosong.
- Path `--start` relatif terhadap `gpu/`, bukan akar repo.
- Run n129 yang terlihat macet ternyata hanya sangat lambat; log progres per langkah membedakan keduanya.
