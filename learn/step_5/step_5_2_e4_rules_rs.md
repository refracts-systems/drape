# Step 5.2: `oracle/src/e4_rules.rs`, Aturan Berhenti yang Sadar Resolusi f32

> Satu perubahan aturan di Part 0: aturan "berhenti kalau koreksi sudah kecil" dari E4 diberi **lantai** setinggi jarak antar-angka f32 di sekitar
> posisi kain. Tujuannya menutup kegagalan probe 1 di E4.

---

## 1. Masalah di E4

Aturan E4: berhenti kalau ‖αd‖∞ ≤ 10⁻⁴·‖x − x_n‖∞.

Di langkah pertama, kain baru turun h²g ≈ 1,7×10⁻⁴ m, sehingga ambangnya 1,7×10⁻⁸ m. Padahal koordinat kain sekitar 0,5 m, dan di sana f32 hanya bisa
membedakan angka yang berjarak ~3×10⁻⁸ m (lihat bagian 2). Koreksi sekecil ambang itu **tidak bisa direpresentasikan**, jadi aturan tidak pernah terpenuhi
dan Newton terus jalan sampai batas 30 iterasi.

Analogi backend: menunggu timestamp berubah kurang dari 1 nanodetik, padahal jam sistem hanya punya resolusi 1 mikrodetik.

## 2. Jarak antar-angka f32

Bilangan f32 = tanda × 1,mmm…(23 bit) × 2^e. Di antara 2^e dan 2^(e+1), jarak antara dua angka f32 berurutan selalu 2^(e−23).

```rust
/// Spacing of f32 numbers at `max_abs`: 2^(floor(log2(max_abs)) − 23), read from the f32 exponent bits.
pub fn f32_spacing(max_abs: f64) -> f64 {
    let exponent = ((max_abs as f32).to_bits() >> 23) & 0xff;
    2f64.powi(exponent as i32 - 127 - 23)
}
```

- `to_bits()` membaca 32 bit mentah f32. Bit 23–30 adalah eksponen dengan bias 127.
- Membaca eksponen langsung dari bit lebih aman daripada `log2`, yang bisa membulatkan 0,49999997 menjadi −1, padahal eksponennya −2.
- Contoh: 0,5 = 2⁻¹, jadi jaraknya 2⁻²⁴ ≈ 6×10⁻⁸. Tepat di bawahnya, 0,49999997 ada di binade 2⁻², jadi jaraknya 2⁻²⁵.

Tes mengunci ketiga kasus ini (`f32_spacing_reads_the_binade`).

## 3. Aturan baru

```rust
/// ‖αd‖∞ ≤ max(1e-4·‖x − x_n‖∞, 2·res); `resolution` is the recorded f32 spacing (0 for runs without the E4-final rule).
pub fn update_is_small(update_inf: f64, step_inf: f64, resolution: f64) -> bool {
    update_inf <= (UPDATE_STOP * step_inf).max(RESOLUTION_FACTOR * resolution)
}
```

- `res(x)` = jarak f32 di max |x_i| atas semua unknown.
- `RESOLUTION_FACTOR = 2`: koreksi ≤ 2 ulp berarti setiap koordinat bergeser paling banyak dua angka f32, yaitu derau pembulatan, bukan kemajuan.
- GPU menghitung `res` sendiri dan **mencatatnya** di setiap iterasi. Oracle memutar ulang keputusan memakai angka yang dicatat (dokumen 5.7), lalu
  memeriksa angka itu sendiri dari posisi probe (`resolution_replay`).

## 4. Satu sumber kebenaran per skema

```rust
pub fn settings_json(resolution_factor: f64) -> Json { ... .number("resolution_factor", resolution_factor) ... }
```

Faktornya ditulis ke `problem.json`, per skema:

| Skema | `resolution_factor` | Perilaku |
|---|---|---|
| E3, E4 | 0 | max(…, 0) = aturan E4 persis |
| E4-final, E5 | 2 | aturan baru |

Di GPU, `parseRobustRules` membaca faktor ini dan memakai 0 kalau tidak ada (file `problem.json` E4 lama), sehingga run E4 lama tetap bisa diulang
dengan hasil yang sama.

## 5. Gate probe tidak dilonggarkan

Gate tetap: probe lulus kalau berhenti lewat aturan residual, **atau** lewat aturan update dengan lantai f32 ≤ 2× lantai representasi. Yang berubah
hanya aturannya, dan aturan itu ditetapkan di prompt sebelum run.

Hasil (dokumen 5.11): probe 1 di n33 dan n65 sekarang berhenti lewat aturan update, dengan lantai f32 0,93× dan 0,97× lantai representasi.
