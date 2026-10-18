# Step 7.0: Gambaran Besar Experiment 7, Drape Kuasi-Statik

> Step 7 mengganti 240 langkah dinamis dengan pencarian **kesetimbangan akhir** kain di atas badan, karena itulah yang dibutuhkan editor. Percobaan pertama
> **berhenti** di aturan STOP Part 0.6; dua perbaikan yang kamu setujui (pengontrol baru dan awal "placed") membuatnya lanjut sampai selesai. Hasil akhirnya:
> **akurasi lolos di sebagian besar kasus, tetapi performa belum mencapai target editor**: n65 butuh 45 s (target 10 s), n129 tidak konvergen dalam 20 menit
> (target 30 s), dan PCG block-Jacobi tidak skala ke n129.

---

## 1. Idenya: pseudo-transient continuation (PTC)

Menurunkan energi statik E_s langsung bermasalah: tanpa inersia, Hessian-nya tunggal pada gerakan horizontal (bola tanpa gesekan). PTC menambahkan suku
proksimal dengan langkah waktu h:

```
G_h(x) = E_s(x) + (1/(2h²)) (x − x_k)ᵀ M (x − x_k),   x̂ = x_k + h²g,  v = 0
```

Ini persis energi langkah waktu E3–E6 dengan kecepatan nol. Akibatnya gradien G_h di awal langkah **adalah** residual statik r_k, dan E_s tidak pernah naik.
Analogi backend: **pemanasan koneksi bertahap**. Mulai dari langkah kecil yang aman, lalu perpanjang selama sistem terus membaik.

## 2. Percobaan pertama berhenti, lalu dua perbaikan

Pengontrol di prompt pertama (SER: h_{k+1} = h_k · min(2, r_k/r_{k+1})) gagal di jatuh bebas: saat kain jatuh, residual sama dengan berat sehingga r_k/r_{k+1}
tidak informatif, dan oracle f64 tidak konvergen dalam 400 langkah di n33. Aturan STOP bekerja sebagaimana dirancang: eksperimen berhenti dan dilaporkan
(`results/experiment7-part0-stop.json`). Dua perbaikan yang disetujui:

1. **Pengontrol mengikuti Newton dalam**: h naik ×2 bila Newton dalam selesai dalam <10 pembaruan, turun ×0,5 bila tidak.
2. **Awal "placed"**: simpul tengah diletakkan 2·d̂ di atas puncak bola, bukan dijatuhkan dari ketinggian.

Pilot menemukan dua cacat lain, juga diperbaiki: solve dalam yang gagal mengembalikan keadaan awal yang terpotong sehingga keadaan meledak (kini percobaan
**ditolak dan diulang** dari x_k dengan h/2, maksimal 12 kali), dan kriteria "floor" menerima keadaan yang menyimpang (kini dijaga r/L ≤ 1e-6).

## 3. Apa yang dibangun

| Bagian | File | Isi |
|---|---|---|
| Oracle f64 | `e7_ptc.rs`, `drape.rs`, `e7_run.rs`, `e7_prolong.rs`, `e7_state.rs` | PTC, solve cold/warm/prolong, probe, prolongasi |
| Penilai | `e7_compare.rs`, `e7_summarize.rs` | gerbang akurasi, replay aturan, tabel Part 1–3 |
| GPU | `gpu/src/ptc.ts`, `drape-e7.ts`, `node/run-e7.ts`, `outputs.ts`, `state-files.ts` | aturan stop, penolakan percobaan, pemicu μ-cap, runner |
| Skrip | `oracle-e7.sh`, `run-e7.sh` | referensi oracle, putaran GPU interleaved dengan median, scoring |

Pemicu μ-cap (kurvatur barrier b′∇²d dijatuhkan untuk satu langkah) ada di oracle dan GPU (`colliders[3]` di `inertia.wgsl`); nilainya 0 di semua kasus E5/E6.

## 4. Hasil ringkas (RTX 3050 Laptop, D3D12 via vgpu)

| n | cold (s) | berhenti karena | setelah edit (terburuk, s) | coarse-to-fine (s) |
|---|---|---|---|---|
| 17 | 19,4 | stagnasi | 8,5 | – |
| 33 | 53,1 | stagnasi | 21,5 | 27,1 (kasar 17 + prolong) |
| 65 | 45,3 | stagnasi | 32,7 | 86,6 |
| 129 | ≥1211 | batas waktu 20 menit | ≥1200 | ≥1246 |

- **Warm vs cold (Newton)**: n33 w1 0,38, w2 0,23, w3 0,28. Gerbang 0,25 hanya lolos di satu dari tiga. Di n129 warm justru lebih mahal (rasio 2,7 sampai 4,4).
- **Dua tingkat**: n33 total 0,51× cold (gerbang 0,5, melewati tipis); n65 1,91×; n129 ≈1,03× (kedua-duanya mencapai batas).
- **Pertumbuhan PCG per Newton**: 17→33 ×2,06, 33→65 ×0,80, 65→129 **×10,8** (di atas batas ×3): pemicu **multigrid berikutnya** menyala.
- **Lantai representasi f32** besar (r/L 1,1e-2 di n17, 4,3e-2 di n33, ≈0,41 di n65), sehingga solve f32 berakhir lewat stagnasi dan bukan toleransi.

## 5. Yang tidak lolos, dan kenapa

- n17 base cold: r/L akhir 3,3e-2 melebihi max(1e-3, 2× lantai) dan satu probe satu-langkah >1e-3.
- n33 base cold: satu probe satu-langkah (langkah 42: 0,018) >1e-3. Jalur PTC kacau di f32, jadi langkah tunggal dari keadaan oracle bisa menyimpang.
- n33 prolong: r/L akhir 0,148 melebihi 2× lantai (0,085).
- n17 w1 warm dan n33 w2 cold: oracle punya satu langkah μ-cap yang tidak tersembuhkan oleh pemicu, sehingga cek "nol langkah μ-cap" gagal.
- Gerbang performa: warm ≤0,25 (n33), dua tingkat ≤0,5 (n33), cold n65 ≤10 s, cold n129 ≤30 s.

Tidak ada toleransi yang dilonggarkan untuk mendapatkan lolos.

## 6. Pelajaran untuk editor

PTC dengan PCG block-Jacobi cukup untuk mesh sampai n65 (puluhan detik), tetapi tidak untuk n129 (PCG sampai 200.000 iterasi per langkah). Untuk editor
interaktif, langkah berikutnya adalah pengkondisi yang lebih baik (multigrid), bukan penyetelan lagi. Warm start membantu di n17–n65 (sekitar 1,4–5× lebih cepat dalam waktu dinding),
tetapi belum cukup untuk target interaktif.

Detail oracle ada di `step_7_1_oracle_ptc.md`; detail GPU, penilai, dan hasil ada di `step_7_2` sampai `step_7_7`.
