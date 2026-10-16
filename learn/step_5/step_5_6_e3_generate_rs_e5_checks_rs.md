# Step 5.6: `e3_generate.rs` dan `e5_checks.rs`, Skema Baru, Probe Berbasis Kejadian, dan Cek Turunan dengan Kontak

> Generator Step 3 dipakai lagi untuk E4-final dan E5. Yang baru: skema membawa batas iterasi, faktor resolusi, dan scene; probe E5 dipilih dari
> **kejadian** di lintasan oracle; dan `problem.bin` E5 membawa data collider serta keadaan uji kontak.

---

## 1. Skema

```rust
pub struct Scheme {
    pub name: &'static str,
    pub tangent: TangentKind,
    pub solver: Solver,
    pub max_iterations: usize,
    pub resolution_factor: f64,
    pub scene: Scene,
    pub note: &'static str,
}
```

| Skema | Tangen | Regularisasi | Lantai f64 | Cap oracle | Resolusi | Scene |
|---|---|---|---|---|---|---|
| `E3_SCHEME` | proyeksi | – | – | 30 | 0 | sudut |
| `E4_SCHEME` | eksak | ✓ | – | 30 | 0 | sudut |
| `E4_FINAL_SCHEME` | eksak | ✓ | 2× | 200 | 2 | sudut |
| `E5_SCHEME` | eksak | ✓ | 2× | 200 | 2 | bola + lantai |

Ini pola **konfigurasi sebagai data**: satu fungsi `generate` melayani empat eksperimen. Analogi: satu job runner dengan profil konfigurasi berbeda per
environment.

Perintahnya:

```bash
oracle e4-generate --final true --n 65 --out drape4-final/drape-n65     # E4-final
oracle e5-generate --n 65 --out drape5/drape-n65                        # E5
```

## 2. Probe berbasis kejadian

E3/E4 memeriksa langkah tetap 1, 10, 60, 120, 240. Kontak terjadi pada waktu yang tidak diketahui sebelum simulasi, jadi E5 memilih probe dari lintasan
oracle:

```rust
/// Probe steps: 1, the first sphere and floor contacts, the most active step and the last step (sorted, without repeats).
pub fn event_probes(drape: &Drape, positions: &[Vec<Vec3>]) -> Vec<usize> {
    let events = contact_events(drape, positions);
    let mut steps: Vec<usize> = [Some(1), events.first_sphere, events.first_floor, Some(events.most_active), Some(positions.len() - 1)].into_iter().flatten().collect();
    steps.sort_unstable();
    steps.dedup();
    steps
}
```

- Kontak "aktif" berarti vertex berada dalam d̂ dari collider di akhir langkah.
- `Option` + `flatten()`: kejadian yang tidak pernah terjadi (misalnya kain tidak pernah menyentuh lantai) otomatis hilang dari daftar. Tidak ada angka
  ajaib seperti −1.
- Langkah-langkah kejadian ditulis juga ke `problem.json` (`first_sphere_contact_step`, dst.), supaya laporan bisa menyebutnya.

## 3. Data kontak di `problem.bin`

```rust
/// "colliders": [count, d̂, κ, 0, then 8 floats per collider]; "contact_state": the kernel-check state.
pub fn push_contact_arrays(arrays: &mut BinaryArrays, drape: &Drape) { ... }
```

Format `colliders`: header 4 float (jumlah, d̂, κ, padding), lalu 8 float per collider `[jenis, a, b, c, w, 0, 0, 0]`. Jenis 1 = lantai (a, b, c = normal;
w = offset), jenis 2 = bola (a, b, c = pusat; w = jari-jari). Panjang tetap 8 per collider (dengan padding) membuat indeks di WGSL cukup `4 + 8·k`.
Analogi: record berukuran tetap di file biner, seperti row di format kolumnar.

Problem E3/E4 tidak punya array ini. GPU lalu memakai `[0, 0, 0, 0]` (nol collider), sehingga kernelnya tetap bisa di-bind dan hasilnya bit-identik.

## 4. Keadaan uji kontak

```rust
/// A state with every free vertex inside the barrier: the centre patch radially above the sphere, the rest above the floor,
/// at distances log-uniform in [0.05·d̂, d̂].
pub fn contact_state(drape: &Drape) -> Vec<Vec3>
```

Untuk menguji kernel kontak dan turunan barrier, kita butuh keadaan di mana **setiap** vertex aktif, dengan jarak yang tersebar dari 0,05·d̂ sampai d̂:

- vertex dekat pusat (dalam 0,6·r secara horizontal) diletakkan radial di atas bola: c + (r + d)·n;
- sisanya di atas lantai: (x, y, d);
- d log-uniform: d̂·0,05^u dengan u ∈ [0, 1], supaya jarak kecil dan besar sama-sama terwakili.

Elemen yang menjembatani bola dan lantai sangat teregang. Itu tidak masalah, karena keadaan ini hanya untuk menguji turunan, bukan simulasi.

## 5. Cek turunan dengan barrier aktif

`contact_derivative_check` menghasilkan lima angka:

| Angka | Arti | Batas (sama dengan E3) |
|---|---|---|
| `gradient_error` | FD vs gradien Φ total (h = 10⁻⁷ m) | 10⁻⁶ |
| `hessian_error` | FD vs tangen Φ total | 10⁻⁵ |
| `barrier_gradient_error` | FD vs gradien barrier saja | 10⁻⁶ |
| `barrier_hessian_error` | FD vs Hessian barrier saja | 10⁻⁵ |
| `barrier_change_error` | Σ per-vertex b(baru) − b(lama) vs selisih total | 10⁻¹² |

Kenapa ada versi "barrier saja"? Di Φ total, energi membran yang teregang jauh lebih besar dari barrier, sehingga kesalahan di turunan barrier bisa
tersembunyi di bawah ukuran relatif total. Versi terpisah mengisolasinya.

### Langkah FD barrier

Uji coba pertama (n17) memakai h = 10⁻⁷ m juga untuk barrier dan mendapat galat 6,4×10⁻⁶ (> 10⁻⁶). Penyebabnya bukan turunan yang salah, melainkan
**galat truncation** beda hingga pusat, yang sebanding dengan (h/d)². Dengan d terkecil 2,5×10⁻⁵ m, (10⁻⁷/2,5×10⁻⁵)² ≈ 1,6×10⁻⁵. Pengukurannya sendiri
tidak cukup teliti.

```rust
/// The barrier-only FD step is this fraction of the smallest distance: central-difference truncation grows as (h/d)², and
/// the E3 step 1e-7 m gave 6.4e-6 at d = 2.5e-5 m in the first n17 run (a measurement artefact, not a derivative error).
const BARRIER_FD_FRACTION: f64 = 1e-3;
```

Langkah FD barrier sekarang 10⁻³·d_min (≈ 2,5×10⁻⁸ m), dan batasnya **tidak diubah**. Di n17 hasilnya 4×10⁻⁷ (gradien) dan 5×10⁻⁷ (Hessian), di n33
3×10⁻⁷ dan 4×10⁻⁷. Di n65 gradien barrier tetap gagal (2,4×10⁻⁶): di mesh besar pembulatan f64 dari jumlah energi semua vertex yang aktif mulai
mendominasi (dokumen 5.11). Langkahnya tidak diubah lagi. Perubahan ini
diambil setelah melihat hasil uji coba, jadi dilaporkan terbuka beserta kedua angkanya. FD untuk Φ total tetap memakai h E3.

## 6. Tes

- `contact_state_puts_every_vertex_inside_the_barrier_on_both_colliders`
- `active_barrier_derivatives_match_finite_differences` (mesh n9)
