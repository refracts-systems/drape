# Step 7.5: `e7_compare.rs`, Penilai Satu Run GPU

> `oracle e7-compare --dir <kasus> --mode <cold|warm|prolong>` membaca keluaran GPU, mengevaluasi ulang segalanya dalam f64, dan menulis
> `gpu-<mode>/result.json` berisi satu bendera lulus/gagal per cek. Analogi backend: **suite tes kontrak** yang dijalankan terhadap log produksi.

---

## 1. Cek yang dijalankan

| Bendera di `verdict` | Cek |
|---|---|
| `wgsl_kernels_ok` | gradien dan blok kernel GPU vs oracle ≤ 1e-5 |
| `inertia_step_sizes_ok` | kernel inersia pada h ∈ {h0, 0,1, 1,0} ≤ 1e-5 |
| `contact_kernel_ok` | kernel kontak + beda hingga seperti E6 (dilaporkan, tidak dijadikan gerbang baru) |
| `intersection_free` | jarak minimum ke bola dan lantai > 0 di semua keadaan diterima |
| `rules_replayed_ok` | forcing, terminasi, α_max, penerimaan, **pengontrol** (sampai 1e-12), dan penolakan diputar ulang dari log GPU |
| `static_energy_ok` | E_s tidak naik di antara keadaan diterima (toleransi 1e-6 relatif) |
| `oracle_converged` | oracle konvergen **tanpa** langkah μ-cap yang diterima |
| `final_state_residual_ok` | r/L f64 keadaan akhir ≤ max(1e-3, 2× lantai representasi) |
| `one_step_ok` | selisih satu-langkah di probe ≤ 1e-3 |
| `f32_stops_ok_at_probes` | di tiap probe GPU berhenti karena residual atau karena pembaruan di lantai f32 |
| `start_feasible` | (warm/prolong) awalnya bebas-tembus; temuan, bukan gerbang akurasi |

Lantai representasi: r/L dari keadaan setimbang oracle yang dibulatkan ke f32. Itulah batas terbaik yang bisa dicapai f32 sama sekali.

## 2. Replay pengontrol dan penolakan

`controller_replay` menghitung ulang h_{k+1} dari h_k dan jumlah pembaruan dalam, dibagi 2 sekali per penolakan, lalu membandingkan dengan log GPU:
selisih harus ≤ 1e-12. `rejection_replay` memeriksa bahwa setiap percobaan ditolak memang memenuhi aturan: alasan 1/2 hanya sah bila nol pembaruan dan
status terakhir cocok, alasan 3 bila ada nilai tak hingga, alasan 4 untuk pemicu μ-cap.

## 3. Bagian lain `result.json`

- `gpu`: ringkasan angka run (langkah, iterasi, waktu, fase, h min/maks, jarak minimum/d̂).
- `oracle` dan `oracle_cold`: konvergensi, jumlah langkah dan Newton, lantai representasi, langkah μ-cap dari referensi.
- `probes`: per probe, kesalahan satu-langkah, lantai f32, lantai representasi, status stop.
- `energy_gap_over_cold` dan `max_vertex_gap_over_side`: seberapa jauh keadaan akhir dari kesetimbangan oracle cold.
- `start_residual_over_load` dan `start_min_distance_over_dhat`: kualitas titik awal warm/prolong.
- `derivatives`: beda hingga kontak seperti E5/E6.

## 4. Untuk n65 dan n129

Tanpa referensi f64, cek yang butuh oracle (`oracle_converged`, `final_state_residual_ok`, probe) tidak dihitung, dan penilai hanya mengeluarkan cek
yang murni dari log GPU (kernel, bebas-tembus, replay, energi). Karena itu kolom bendera di tabel ringkasan berisi `-` untuk n65/n129.
