# Step 3.11: `gpu/kernels/membrane_element.wgsl` (Perubahan Step 3), Override `PROJECTED`

> Kernel membran E2 mendapat satu override baru: tegangan negatif dipotong di suku geometrik, sama dengan `MembraneElement::hessian`.
> Ini kembaran GPU dari trigger berbasis bukti. Defaultnya `false`, sehingga E2 tidak berubah.

---

## 1. Perubahan

```wgsl
// PROJECTED = false: exact tangent. PROJECTED = true: negative principal stress clamped in the geometric term
// (oracle MembraneElement::hessian), the E3 evidence-triggered tangent.
override PROJECTED: bool = false;
...
  let stress = stress_of(green_strain(f));
  let geometric_stress = select(stress, clamp_negative_stress(stress), PROJECTED);
  ...
    write_gradient(e, k, f, stress, area);                     // gaya: selalu tegangan asli
    ...
      write_block(e, k, l, f, geometric_stress, area);         // tangen: tegangan untuk suku geometrik
```

- **Gaya** selalu memakai tegangan asli. Gaya harus tetap turunan energi yang sebenarnya, kalau tidak Newton menyelesaikan soal yang salah.
- **Tangen** memakai tegangan yang (opsional) dipotong. Di `write_block`, tegangan hanya muncul di suku geometrik `gkᵀ S gl`, dan suku material tidak berubah.

`select(stress, clamp(stress), PROJECTED)` = `PROJECTED ? clamp : stress`. Kedua argumen `select` selalu dievaluasi, tetapi karena tanpa efek samping,
nilai yang terpilih tetap sama. Untuk E2 (`PROJECTED = false`) hasilnya identik bit-per-bit, dan regresi `uniaxial-n33` membuktikannya.

## 2. Memotong tegangan negatif

```wgsl
fn clamp_negative_stress(s: vec3f) -> vec3f {
  let mean = 0.5 * (s.x + s.y);
  let radius = sqrt(0.25 * (s.x - s.y) * (s.x - s.y) + s.z * s.z);
  let high = mean + radius;
  if (mean - radius >= 0.0) { return s; }
  if (high <= 0.0) { return vec3f(0.0); }
  let v = principal_direction(s, high);
  return vec3f(high * v.x * v.x, high * v.y * v.y, high * v.x * v.y);
}
```

Tegangan 2D punya dua **tegangan utama** (eigenvalue matriks 2×2): `mean ± radius` (lingkaran Mohr).

- Keduanya ≥ 0 (tarik di segala arah): biarkan.
- Keduanya ≤ 0 (tekan di segala arah): nol.
- Campuran: pertahankan hanya bagian tarik `high · v vᵀ`, dengan v arah utamanya.

Kenapa ini membuat tangen positif semidefinit? Kekakuan geometrik negatif hanya datang dari tegangan tekan. Membuangnya membuat setiap blok elemen ≥ 0.
Harganya: tangen bukan lagi turunan eksak, sehingga Newton konvergen **linear**, bukan kuadratik. Inilah yang terukur di E3.

Port satu-satu dari `clamp_negative_stress` dan `principal_direction` di `membrane.rs` (`hypot` di Rust = `length` di WGSL).

## 3. Kenapa fungsi ini di file kernel, bukan di header `membrane.wgsl`?

Header ditempel juga ke `membrane_energy_change.wgsl`, yang tidak butuh tangen. Fungsi yang hanya dipakai satu kernel diletakkan di kernel itu.

## 4. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| `PROJECTED` | override; default false = E2 identik |
| Gaya vs tangen | gaya selalu eksak; hanya tangen yang dimodifikasi |
| Tegangan utama | mean ± radius; buang bagian tekan |
| Efek | tangen PSD, Newton linear |
