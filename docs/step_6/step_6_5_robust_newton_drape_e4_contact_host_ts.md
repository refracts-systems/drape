# Step 6.5: `robust-newton.ts`, `drape-e4.ts`, `contact-host.ts`, `drape-gpu.ts`, Merangkai Rung A–D

> Loop Newton host dari Step 4–5 tetap utuh. Yang berubah: ia memanggil `NewtonOps` (dokumen 6.4) dan mencatat lebih banyak per iterasi, dan setiap rung
> memilih implementasi yang berbeda lewat saklar di `RobustSettings`.

---

## 1. Saklar rung

```ts
export interface RobustSettings {
  readonly floorStop: boolean;
  readonly forcing: boolean;
  readonly regularize: boolean;
  readonly noiseAware?: boolean;
  readonly gpuReductions?: boolean;
  readonly fewReadbacks?: boolean;
}
```

Tiga saklar baru bersifat **opsional**. `undefined` dianggap `false`, sehingga runner E4 dan E5 tidak perlu disentuh dan `gpu-result.json`-nya tetap
berisi `settings` yang sama.

| Rung | noiseAware | gpuReductions | fewReadbacks |
|---|---|---|---|
| A (= E5) | – | – | – |
| B | ✓ | – | – |
| C | ✓ | ✓ | – |
| D | ✓ | ✓ | ✓ |

Hanya `noiseAware` dipakai di `robust-newton.ts`. Dua lainnya dipakai `drape-e4.ts` saat membangun konteks.

## 2. `drape-e4.ts`: memilih implementasi

```ts
function context<A, K>(gpu, problem, rules, settings, clock, keep): RobustContext<A, K> {
  const reducers = settings.gpuReductions === true ? gpu.reducers : null;
  if (settings.gpuReductions === true && reducers === null) throw new Error("gpuReductions need prepareDrapeGpu(..., withReducers)");
  const ops = reducers === null ? hostOps(gpu, clock) : gpuOps(gpu, reducers, clock, keep);
  const saveNoise = settings.noiseAware === true && problem.manifest.grid_columns <= MAX_NOISE_STATE_COLUMNS;
  return { ..., ops, contact: contactHooks(gpu, problem, settings.fewReadbacks === true),
    directions: keep ? [] : null, trials: keep ? [] : null, noiseStates: saveNoise ? [] : null };
}
```

- Dengan `keep` (probe): `gpuOps` dibuat dengan `verify`, dan `trials` dan `directions` dikumpulkan.
- `noiseStates` menyimpan keadaan sebelum dan sesudah setiap update yang diterima hanya berkat noise (untuk n ≤ 65, karena n129 tidak punya oracle).
- Konfigurasi yang salah (rung C tanpa kernel reduksi) **gagal keras**, bukan jatuh diam-diam ke versi host. Aturan "never fall back silently" juga berlaku di
  sini.
- `prepareDrapeGpu(..., withReducers)`: kernel reduksi dikompilasi hanya kalau diminta (E3, E4, dan E5 tidak memuatnya).

`resetClock` juga mereset `readBytes` dan fase `reduce`. Hasil lintasan menyertakan `readBytes` total.

## 3. `robust-newton.ts`: satu iterasi lebih mudah dibaca

`update` sekarang:

1. `solveDirection`: PCG, lalu `ops.slope(solusi)`. Gagal kalau PCG breakdown atau bukan arah turun; dengan regularisasi μ naik dan perakitan ulang
   lewat `ops.residual`.
2. `startStep`: tulis arah ke GPU dan ambil α_max (fase `ccd`).
3. `searchStep` di fase `line_search`.
4. Maju, lalu `recordOf` menyusun catatan robust:

| Kolom baru | Isi |
|---|---|
| `deltaSum`, `deltaAbs` | Σδ dan Σ\|δ\| langkah yang diterima |
| `noiseOnly` | 1 kalau diterima hanya berkat suku noise |
| `slope`, `slopeExact` | gᵀd dari reduksi/host, dan jumlah f64 dari nilai yang sama (probe, rung C dan D) |
| `normSquared`, `normSquaredExact` | idem untuk ‖rhs‖² |

Fungsi `update` dipecah (`searchOptions`, `recordOf`) supaya tiap fungsi < 20 baris. Percobaan α tiap line search (probe) dikumpulkan lewat `onTrial` di
`searchOptions`.

## 4. `contact-host.ts`: dua versi pengambilan α_max dan jarak

| | rung A–C | rung D |
|---|---|---|
| α_max | jalankan CCD, **baca seluruh `bounds`** (n float), cari minimum di host | CCD + `argmin` di GPU, baca 32 byte |
| jarak min per iterate | f64 di host dari posisi f32 (semua vertex × collider) | f32 di GPU (`distances` + argmin), baca 32 byte |

`ContactHooks.minDistance` sekarang `async` (versi GPU butuh `await`). Di rung D, jarak diukur **setelah** update dengan menulis posisi baru ke GPU dan
menjalankan CCD lagi, jadi yang dicatat adalah jarak pada iterate yang diterima. Evaluasi f64 dipindahkan keluar dari loop: ia dipakai hanya untuk 241
keadaan lintasan setelah run selesai (di penilai) dan iterate probe.

## 5. `run-e6.ts` dan `merge-runs.ts`

`run-e6.ts` = `run-e5.ts` + tabel `RUNGS` + keluaran tambahan:

| File | Isi |
|---|---|
| `gpu-probe-search.bin` | setiap percobaan α di probe: langkah, iterasi, halving, α, Σδ, Σ\|δ\|, gᵀd, diterima |
| `gpu-noise-states.bin` | keadaan sebelum/sesudah tiap update noise-only (probe dulu, lalu lintasan) |
| `gpu-result.json` | ditambah `read_bytes_total`, `newton_iterations_total`, `read_bytes_per_newton_iteration` |

Kolom robust bertambah menjadi 16 (`EXTRA_FIELDS`). Penilai membacanya **berdasarkan nama** dari `gpu-result.json`, sehingga file lama (5, 6, atau 9 kolom)
tetap terbaca.

### Median dari beberapa run

Satu pengukuran waktu di GPU laptop tidak bisa dipercaya: run pertama saya menunjukkan rung B di n33 berjalan 159 ms setelah run n129 yang panjang, padahal
sebelumnya 121 ms. Fase PCG sendiri, yang kodenya tidak berubah antar rung, naik 30%. GPU melambat karena panas. Karena itu:

- `run-e6.sh` menjalankan beberapa **putaran** (3 untuk n33 dan n65, 2 untuk n129), dan di dalam setiap putaran seluruh tangga A–D **berselang-seling**, supaya
  GPU yang sedang hangat mengenai semua rung dengan sama.
- Setiap run menyimpan `gpu-result-run<k>.json`. `merge-runs.ts` memilih run dengan **median** ms/langkah, menuliskan angkanya (termasuk rincian fase) ke
  `gpu-result.json`, dan menyimpan semua nilai di `ms_per_step_runs`.
- Keluaran numerik (lintasan, iterasi) identik antar putaran, karena GPU deterministik. Hanya waktunya yang berbeda.

Analogi backend: benchmark yang dijalankan beberapa kali dengan urutan diacak, lalu dilaporkan median-nya, bukan satu angka.
