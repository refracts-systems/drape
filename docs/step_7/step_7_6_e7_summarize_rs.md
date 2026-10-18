# Step 7.6: `e7_summarize.rs`, Tabel Part 1–3, Gerbang, dan Temuan

> `oracle e7-summarize --dir drape7 --out results/experiment7-vgpu-hardware.json` membaca semua `result.json`, mencetak tabel, menghitung gerbang
> performa, dan menandai pemicu. Analogi backend: **laporan CI** yang merangkum ratusan job menjadi satu halaman.

---

## 1. Apa yang dikumpulkan

Untuk setiap `drape7/drape-n<N>/<edit>/gpu-<mode>/result.json` (edit ∈ base, w1–w3; mode ∈ cold, warm, prolong) satu baris dicetak: langkah, percobaan
ditolak, iterasi Newton dan PCG, detik per fase, MiB dibaca, alasan stop, r/L akhir, lantai representasi, h akhir, jarak minimum/d̂, jumlah langkah
yang kurvaturnya dijatuhkan, dan satu karakter per cek (`Y` lolos, `n` gagal, `-` tidak berlaku).

## 2. Gerbang dan batas

| Konstanta | Nilai | Dipakai untuk |
|---|---|---|
| `GATES` (9 bendera) | lihat 7.5 | gerbang akurasi di n17 dan n33 |
| `WARM_LIMIT` | 0,25 | Newton warm / cold per edit (gerbang di n33, temuan di n65) |
| `WARM_REPORT_LIMIT` | 0,5 | rasio warm yang dilaporkan |
| `TWO_TIER_LIMIT` | 0,5 | total ms (kasar cold + prolong) / halus cold (gerbang n33, temuan n65/n129) |
| `COLD_SECONDS` | n65: 10 s, n129: 30 s | waktu cold terhadap target interaktif |
| `PCG_GROWTH_LIMIT` | 3 | pertumbuhan PCG per Newton per penggandaan mesh |
| `FLOOR_MULTIPLE` | 10 | solve f32 cold yang berakhir jauh di atas lantai representasi (n17/n33), atau r/L > 1 (n65/n129) |

`contact_kernel_ok` dan `start_feasible` ditampilkan tetapi tidak menjadi gerbang akurasi (yang pertama hanya dilaporkan menurut prompt, yang kedua
adalah temuan).

## 3. Bagian ringkasan

- **warm**: per ukuran dan edit, rasio Newton, ms, PCG warm/cold, alasan stop keduanya, apakah cold oracle konvergen, dan residual awal.
- **two_tier**: ms dan Newton kasar cold + prolong terhadap halus cold, residual awal hasil prolongasi (≈ 5,5e2, jauh dari setimbang), kelayakan awal.
- **pcg_growth**: PCG per iterasi Newton pada base cold antar ukuran, dan apakah <3×.
- **floor_findings**: solve cold f32 yang berhenti bukan karena `tolerance` namun jauh di atas lantai. Tidak ada yang terpicu di run ini.
- **editor**: tabel detik per ukuran: cold, setelah edit (terburuk dari tiga), coarse-to-fine, dan alasan stop di baliknya.
- **cold_seconds**: cold n65 dan n129 terhadap 10 s dan 30 s.
- **accuracy**: `all_pass` dan daftar kasus yang gagal, hanya untuk n ≤ 33.

## 4. Catatan penafsiran

- "Dua tingkat" dihitung satu tingkat ke atas (kasar cold + prolong), bukan rantai n17 → n33 → n65 → n129.
- Waktu adalah median antar putaran (`merge-runs.ts`); satu putaran saja untuk n129.
- Untuk n129 semua solve berhenti di `time_budget`/`cap`, jadi angkanya batas bawah, bukan waktu sampai setimbang.
