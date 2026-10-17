# Step 7.1: Kode Oracle E7 (`e7_ptc.rs`, `e7_run.rs`, `e7_prolong.rs`, `e7_state.rs`, `drape.rs`)

> Kode oracle untuk kuasi-statik. Semuanya baru atau aditif: E1–E6 tidak berubah (45 tes lulus, termasuk 3 tes PTC dan 1 tes prolongasi).

---

## 1. `drape.rs`: parameter scene dan langkah waktu yang bisa diubah

```rust
pub struct SceneParams { pub material: f64, pub radius: f64, pub rest: f64 }
impl SceneParams { pub const BASE: SceneParams = SceneParams { material: 1.0, radius: 1.0, rest: 1.0 }; }
```

| Faktor | Efek |
|---|---|
| `material` | semua kekakuan membran dan lentur dikali faktor (kepadatan tetap) |
| `radius` | jari-jari bola dikali faktor; pusat tetap di tengah, tinggi = jari-jari baru; tinggi lepas dihitung ulang |
| `rest` | semua panjang diam dikali faktor terhadap pusat kain (topologi dan nomor vertex tetap) |

`Drape::for_scene` memanggil `for_scene_with(..., SceneParams::BASE)`, jadi semua pemanggil lama mendapat scene yang sama persis. `from_problem` membaca
`scene_params` dari `problem.json` (1 bila tidak ada).

`time_step: Cell<f64>`: `Cell` adalah wadah Rust yang boleh diubah lewat referensi bersama (single-thread). `mass_over_h2` membaca h dari sini, nilai awalnya
`TIME_STEP`, sehingga E3–E6 tidak berubah. PTC mengaturnya per langkah. Analogi: variabel konfigurasi yang bisa diganti saat runtime tanpa membuat ulang service.

## 2. `e7_ptc.rs`

| Fungsi | Isi |
|---|---|
| `next_step_size(h, r_prev, r)` | h_{k+1} = clamp(h · min(2, r_prev/r), h₀/10, 1) |
| `gravity_load` | L = ‖m·g‖ |
| `static_gradient`, `static_energy` | ∇E_s dan E_s (gravitasi sebagai potensial m·g·z) |
| `ptc_prediction` | x̂ = x + h²g di vertex bebas |
| `static_floor` | ε64(‖Σ\|g_c\|‖ + ‖\|K\|·\|x\|‖) dengan kontribusi dan tangen **statik** (tanpa inersia) |
| `ptc_solve` | loop: residual statik, kriteria, h dari pengontrol, langkah Newton inexact |

Langkah Newton: `solve_step_regularized` dengan toleransi 0,1 relatif terhadap residual awal langkah dan paling banyak 5 iterasi, mulai dari `feasible_start` (x̂ dipotong
oleh α_max, dokumen 5.5). Kriteria berhenti oracle: r/L ≤ 10⁻⁸ atau r ≤ 2·lantai; cap 400 langkah. Progres per langkah dicetak ke stderr.

## 3. `e7_run.rs`

- `generate`: menulis `problem.{json,bin}` (scene E5 dengan edit) ditambah array `row_unit_mass` (massa f32 per baris, untuk menulis ulang M/h² tiap langkah di GPU) dan
  bagian JSON `ptc` (aturan) dan `scene_params`.
- `solve`: menjalankan PTC lalu menulis `reference-<mode>.{json,bin}` (semua keadaan, h, residual, iterasi), `equilibrium-<mode>` (keadaan akhir + h akhir), dan untuk cold
  juga `probes.{json,bin}`. Cold yang tidak konvergen berakhir dengan error `STOP`, sesuai aturan.

Probe dipilih dari lintasan oracle: langkah 1, kontak bola pertama, kontak lantai pertama, dan langkah pertama dengan r/L ≤ 0,1 dan ≤ 0,01.

## 4. `e7_prolong.rs` dan `e7_state.rs`

Prolongasi n→2n−1: vertex kasar disalin, titik tengah sisi dirata-rata dari dua tetangga kasar, pusat kotak dari empat sudut. Satu rumus untuk semuanya:

```rust
fn interpolate(coarse, coarse_columns, i, j) -> Vec3   // rata-rata empat sudut kasar (i/2, ceil(i/2)) × (j/2, ceil(j/2))
```

Setiap vertex yang lebih dekat dari d̂ ke collider digeser sepanjang normalnya ke d = d̂ (satu lintasan, tanpa iterasi). Vertex yang tetap di dalam collider menghentikan proses.
`e7_state.rs` menyimpan keadaan ekuilibrium (posisi f32 atau f64 dan h akhir) sebagai pasangan `.json` + `.bin` yang bisa dibaca oracle dan GPU.
