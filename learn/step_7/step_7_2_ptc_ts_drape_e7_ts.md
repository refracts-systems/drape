# Step 7.2: `ptc.ts` dan `drape-e7.ts`, Satu Langkah PTC di GPU

> Dua file ini adalah inti sisi GPU Experiment 7. `ptc.ts` berisi **aturan murni** (tanpa GPU): pengontrol langkah dan kriteria berhenti.
> `drape-e7.ts` menjalankan satu percobaan langkah di GPU dan merangkai loop solve. Newton dalam sendiri tidak ditulis ulang: ia memakai
> E4-final rung D dari Experiment 6.

---

## 1. `ptc.ts`: aturan yang bisa diuji tanpa GPU

```ts
export function nextStepSize(rules, h, inner) {
  const factor = inner < rules.inner_max_iterations ? rules.growth : rules.shrink;
  return Math.min(rules.h_max, Math.max(rules.h0 * rules.h_min_factor, h * factor));
}
```

Pengontrol langkah: kalau Newton dalam selesai dalam kurang dari 10 pembaruan, h naik ×2 (sistem mudah), kalau tidak turun ×0,5. Batasnya
[h0/10, 1 s]. Analogi backend: **TCP slow-start / AIMD**. Window membesar selama tidak ada kemacetan, dan menyusut begitu ada.

`parsePtcRules` membaca bagian `ptc` dari `problem.json` yang ditulis oracle, jadi oracle dan GPU memakai angka yang **sama persis** (h0 = 1/240,
h_max = 1, toleransi dalam 0,1, dan seterusnya). Menyalin konstanta di dua tempat adalah sumber bug klasik; di sini sumbernya satu.

### Kriteria berhenti f32 (`stopRule`)

| Urutan | Aturan | Arti |
|---|---|---|
| 1 | `tolerance` | r/L ≤ 1e-3 |
| 2 | `stagnation` | minimum 10 residual terakhir ≥ 0,95× minimum 10 sebelumnya **dan** r/L ≤ 0,25 |
| 3 | `cap` | 1000 langkah diterima di n17/n33, 400 di n65/n129 (`capFor`) |
| – | `stalled` | dipasang `drape-e7.ts` bila satu langkah gagal 12 kali |
| – | `time_budget` | batas dinding, opsional, hanya dipakai untuk n129 |

Kenapa stagnasi ada: di f32, residual statik tidak bisa turun di bawah **lantai representasi** (r/L ≈ 1e-2 di n17, 4e-2 di n33, 0,4 di n65), jadi
`tolerance` 1e-3 hampir tidak pernah tercapai. Stagnasi adalah cara jujur mengatakan "tidak ada kemajuan lagi". Syarat r/L ≤ 0,25 mencegah
kemacetan di tengah jatuh bebas dianggap selesai.

## 2. `drape-e7.ts`: satu percobaan, satu langkah, satu solve

### `prepare`
Menulis ulang `row_mass = M/h²` (dari massa satuan f32, `problem.unitMass`, dibagi h² di f32), prediksi `x̂ = x_k + h²g` dan μ = 0. Ini yang mengubah
solver dinamis E3–E6 menjadi solver PTC **tanpa mengubah kernel**: PTC adalah langkah dinamis dengan kecepatan nol.

### `attempt` dan `classify`
Satu percobaan: titik awal layak (`newtonStart`), lalu `solveStepRobust` dengan toleransi 0,1 dan paling banyak 10 pembaruan. `classify` memutuskan hasilnya:

| Kode | Arti |
|---|---|
| 0 accepted | diterima |
| 1 regularization_cap | Newton berhenti di batas μ **sebelum** pembaruan pertama |
| 2 line_search_failed | gagal line-search sebelum pembaruan pertama |
| 3 non_finite | keadaan tidak hingga |
| 4 curvature_retry | percobaan yang digantikan oleh pemicu μ-cap (lihat 7.3) |

Penting: gagal **setelah** ada kemajuan tetap diterima. Menolak hanya bila tidak ada satu pun pembaruan, karena itulah yang membuat keadaan lama (x_k)
masih utuh untuk diulang. Percobaan pilot sebelumnya mengembalikan keadaan awal yang terpotong dan meledak.

### `takeStep`
Mencoba h, h/2, h/4, … (minimum h0/10, paling banyak 12 penolakan). Semua percobaan, ditolak maupun diterima, dicatat di `AttemptLog` beserta rentang catatan
iterasinya, sehingga penilai bisa memutar ulang setiap keputusan. Pekerjaan yang ditolak ikut dihitung dalam waktu dan iterasi.

### `solveEquilibrium`
Loop utama:

1. `prepare` lalu hitung r_k di GPU (lewat reduksi GPU rung D).
2. `stopRule` (dan batas waktu bila ada). Bila berhenti, selesai.
3. `takeStep`; bila semua percobaan gagal: `stalled`.
4. h berikutnya dari `nextStepSize(h yang diterima, jumlah pembaruan dalam)`.

Keluarannya `EquilibriumRun`: semua keadaan diterima (hanya awal dan akhir untuk mesh >n65), log langkah, log percobaan, catatan iterasi, waktu per fase,
dan byte yang dibaca balik.

### `runPtcProbes` dan `evaluateInertiaAtSteps`
Dua alat penilaian. `runPtcProbes` menjalankan **satu** percobaan dari tiap keadaan probe oracle (dengan h yang diterima oracle) dan menyimpan semua iterasi,
agar kesalahan satu-langkah terhadap oracle bisa diukur. `evaluateInertiaAtSteps` mengevaluasi kernel inersia pada h ∈ {h0, 0,1, 1,0}; ia memeriksa bahwa
penulisan ulang `row_mass` benar di semua skala langkah.

## 3. Catatan desain

- **Tidak ada kernel baru untuk PTC.** Semua perilaku baru ada di tingkat host: bobot massa, x̂, dan keputusan menerima/menolak.
- **Loop Newton tetap di host** (aturan proyek): GPU hanya mengerjakan assembly, PCG, reduksi, dan arg-min.
- **Kesetaraan oracle-GPU** dijaga lewat `problem.json`: pengontrol, toleransi, dan batas berasal dari satu sumber.
