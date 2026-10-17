# Step 7.0: Gambaran Besar Experiment 7, Drape Kuasi-Statik, dan Kenapa Ia Berhenti di Part 0.6

> Step 7 mengganti 240 langkah dinamis dengan pencarian **kesetimbangan akhir** kain di atas badan, karena itulah yang dibutuhkan editor. Eksperimen ini
> **berhenti** di aturan STOP Part 0.6: oracle f64 tidak mencapai kesetimbangan dalam 400 langkah di n33. Penyebabnya ada pada rancangan pengontrol langkah waktu
> di prompt, bukan pada solver, dan dokumen ini mencatat diagnosisnya.

---

## 1. Idenya: pseudo-transient continuation (PTC)

Menurunkan energi statik E_s (membran + hinge + gravitasi + barrier) langsung bermasalah: tanpa inersia, Hessian-nya tunggal pada gerakan horizontal (bola
tanpa gesekan). PTC menambahkan **suku proksimal** dengan langkah waktu h:

```
G_h(x) = E_s(x) + (1/(2h²)) (x − x_k)ᵀ M (x − x_k)
```

Ini persis energi langkah waktu E3–E6 dengan kecepatan nol dan x̂ = x_k + h²g. Dua akibatnya, yang kini punya tes unit:

- gradien G_h di awal langkah **adalah** residual statik r_k = ‖∇E_s(x_k)‖ (`the_start_of_a_step_residual_is_the_static_residual`);
- E_s tidak pernah naik antar langkah (`a_ptc_step_never_raises_the_static_energy`).

Analogi backend: **pemanasan koneksi bertahap**, mulai dari langkah kecil yang aman lalu memanjangkannya selama sistem terus membaik (SER:
h_{k+1} = clamp(h_k · min(2, r_k/r_{k+1}), h_0/10, 1 s)).

## 2. Apa yang dibangun (oracle saja)

| File | Isi |
|---|---|
| `drape.rs` | `SceneParams` (faktor material, radius, ukuran), `time_step` per langkah (`Cell`), material dan mesh yang diskalakan |
| `e7_ptc.rs` | pengontrol langkah, residual dan energi statik, lantai f64 statik, `ptc_solve` |
| `e7_run.rs` | `e7-generate` (soal dengan edit W1–W3), `e7-solve` (referensi cold/warm/prolong, probe) |
| `e7_prolong.rs` | prolongasi kasar → halus (rata-rata empat sudut) dan proyeksi keluar ke d = d̂ |
| `e7_state.rs` | berkas keadaan (posisi + h akhir) f32/f64 |

Sisi GPU dan penilai belum dibangun, karena aturan STOP menghentikan eksperimen sebelum Part 1.

## 3. Hasil: aturan STOP menyala

Oracle f64, cold start, 400 langkah:

| Ukuran | Langkah | r/L akhir | h akhir | μ maks | langkah cap-μ |
|---|---|---|---|---|---|
| n17 (uji awal) | 400 (cap) | 0,725 | 5,7×10⁻³ | 0 | 0 |
| **n33 (resmi)** | **400 (cap)** | **0,733** | **5,7×10⁻³** | **0** | **0** |

Per langkah (n33), ringkas:

| Keadaan | z pusat | z minimum | r/L | h | iterasi Newton |
|---|---|---|---|---|---|
| 0 | 0,26000 | 0,26000 | 1,000 | 4,17×10⁻³ | 0 |
| 50 | 0,25148 | 0,25148 | 1,000 | 4,17×10⁻³ | 0 |
| 75 | 0,25006 | 0,24716 | 0,989 | 4,22×10⁻³ | 4 |
| 100 | 0,25003 | 0,24260 | 0,951 | 4,38×10⁻³ | 5 |
| 200 | 0,25004 | 0,21449 | 0,890 | 4,68×10⁻³ | 5 |
| 300 | 0,25003 | 0,18810 | 0,824 | 5,06×10⁻³ | 5 |
| 399 | 0,25002 | 0,15915 | 0,733 | 5,68×10⁻³ | – |

## 4. Diagnosis

1. **Newton tidak gagal.** μ tetap 0, tidak ada batas μ, dan 57 langkah pertama bahkan selesai dengan **0 iterasi** (jatuh bebas: x̂ sudah solusinya).
2. **Residual statik selama jatuh bebas sama dengan berat kain** (r/L = 1,000). Maka r_k/r_{k+1} = 1 dan SER **tidak pernah memanjangkan h**. Setiap
   langkah menggeser kain hanya h₀²g = 0,17 mm (25 langkah → 4,3 mm, cocok dengan data).
3. Kain terbaring 1 cm di atas bola (z pusat 0,25), dan tepinya harus jatuh ke lantai sejauh sekitar 0,25 m. Pada 0,17 mm per langkah itu sekitar
   **1.500 langkah**, jauh di atas batas 400. Setelah kontak h memang tumbuh, tetapi hanya 4,17 → 5,7 ms (residual turun lambat, 1 → 0,73 dalam 340 langkah).
4. Kesimpulan: pengontrol "residual turun → h naik" tidak cocok untuk fase gerak yang residualnya konstan. Yang berguna sebagai sinyal bahwa langkah
   terlalu pendek adalah **kemudahan solusi**: langkah dengan 0–1 iterasi Newton jelas boleh diperpanjang.

Catatan rancangan: aturan itu ditetapkan di prompt sebelum run, dan prompt melarang "menemukan perbaikan lain" di dalam eksperimen yang sama. Karena itu
eksperimen berhenti dan hasilnya dicatat di `results/experiment7-part0-stop.json`. Perbaikan menjadi bahan revisi prompt Step 7, bukan keputusan diam-diam.

## 5. Menjalankan ulang diagnosis

```bash
oracle e7-generate --n 33 --edit base --out drape7/drape-n33/base
oracle e7-solve --dir drape7/drape-n33/base --mode cold --probes true     # berakhir dengan "STOP: the f64 oracle did not converge (cap)"
```
