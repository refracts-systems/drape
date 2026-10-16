# Step 5.8: `contact.wgsl`, `inertia.wgsl`, dan `inertia_energy_change.wgsl`, Barrier di GPU

> Kontak masuk ke GPU lewat **kernel yang sudah ada**: kernel inersia per baris. Hessian kontak hanya mengisi blok diagonal per vertex, dan kernel
> inersia sudah menulis blok diagonal per vertex, jadi tidak perlu kernel atau pola matriks baru.

---

## 1. Header bersama: `contact.wgsl`

`kernels.ts` bisa menempelkan satu file header di depan kernel (dokumen 1.14). Tiga kernel (inersia, perubahan energi inersia, CCD) memakai header yang
sama:

```ts
inertia: { file: "inertia.wgsl", header: "contact.wgsl" },
inertia_energy_change: { file: "inertia_energy_change.wgsl", header: "contact.wgsl" },
ccd: { file: "ccd.wgsl", header: "contact.wgsl" },
```

Di WGSL, deklarasi tingkat modul tidak bergantung urutan. Fungsi di header boleh memakai variabel `colliders` yang baru dideklarasikan (dengan nomor
binding masing-masing) di badan kernel. Analogi: header C yang memakai simbol `extern` yang didefinisikan di file lain.

### Membaca collider

```wgsl
struct Proximity {
  distance: f32,
  normal: vec3f,
  curvature: f32,
}

fn proximity(k: u32, x: vec3f) -> Proximity {
  let a = collider_point(k);
  if (is_plane(k)) {
    return Proximity(dot(a, x) - collider_size(k), a, 0.0);
  }
  let q = x - a;
  let len = length(q);
  return Proximity(len - collider_size(k), q / len, 1.0 / len);
}
```

Satu fungsi mengembalikan semua yang dibutuhkan: jarak, normal ∇d, dan kelengkungan (1/|x − c| untuk bola, 0 untuk lantai). `struct` WGSL dipakai seperti
value object kecil.

### ln(d/d̂) yang teliti

```wgsl
fn log_ratio(d: f32) -> f32 {
  let dhat = activation();
  if (d < 0.5 * dhat) {
    return log(d / dhat);
  }
  let u = (d - dhat) / dhat;
  let s = u / (2.0 + u);
  let s2 = s * s;
  var series = 1.0 / 17.0;
  for (var k = 15.0; k >= 1.0; k = k - 2.0) {
    series = series * s2 + 1.0 / k;
  }
  return 2.0 * s * series;
}
```

Spesifikasi WGSL menjamin `log` hanya teliti **2⁻²¹ absolut** pada argumen di [0,5; 2]. Untuk d mendekati d̂, ln(d/d̂) mendekati 0, sehingga galat
absolut 4,8×10⁻⁷ menjadi galat **relatif** besar. Contohnya di d = 0,99·d̂: ln ≈ −0,01, galat relatif ~5×10⁻⁵, melebihi batas kernel 10⁻⁵.

Pelajaran yang sama dengan `atan2` di Step 3 (4096 ulp, diganti Cephes `atanf`): **jangan percaya fungsi built-in GPU tanpa membaca batas akurasinya**.

Solusinya identitas ln(1 + u) = 2·atanh(s) dengan s = u/(2 + u), memakai deret 2(s + s³/3 + s⁵/5 + … + s¹⁷/17):

- di d ∈ [d̂/2, d̂), |s| ≤ 1/3, sehingga suku s¹⁹ sudah < 10⁻⁹ relatif;
- d − d̂ dihitung **tepat** di f32, karena dua angka f32 yang berselisih kurang dari 2× bisa dikurangkan tanpa galat (lemma Sterbenz);
- deret dievaluasi dengan skema Horner (dari suku tertinggi ke bawah), cara paling hemat dan stabil.

Di bawah d̂/2, `log` built-in teliti 3 ulp relatif, jadi dipakai langsung.

### Barrier dan turunannya

```wgsl
fn barrier_first(d: f32) -> f32 {
  if (d >= activation()) {
    return 0.0;
  }
  let t = d - activation();
  return -colliders[2] * (2.0 * t * log_ratio(d) + t * t / d);
}
```

Kembaran langsung dari `contact.rs` (dokumen 5.4), dengan κ = `colliders[2]`.

## 2. `inertia.wgsl`: gradien dan blok diagonal kontak

```wgsl
fn add_contact(row: u32, x: vec3f, k: u32) {
  let near = proximity(k, x);
  if (near.distance >= activation()) {
    return;
  }
  let first = barrier_first(near.distance);
  let second = barrier_second(near.distance);
  let n = near.normal;
  for (var r = 0u; r < 3u; r = r + 1u) {
    elem_grad[GRAD_OFFSET + 3u * row + r] = elem_grad[GRAD_OFFSET + 3u * row + r] + first * n[r];
    for (var c = 0u; c < 3u; c = c + 1u) {
      let identity = select(0.0, 1.0, r == c);
      let entry = second * n[r] * n[c] + first * near.curvature * (identity - n[r] * n[c]);
      elem_hess[HESS_OFFSET + 9u * row + 3u * r + c] = elem_hess[HESS_OFFSET + 9u * row + 3u * r + c] + entry;
    }
  }
}
```

- Kernel inersia lama tetap menulis (M/h²)(x − x̂) dan (1 + μ)(M/h²)·I seperti dulu. Setelah itu, **untuk setiap collider**, kontribusi kontak
  ditambahkan ke slot yang sama.
- Satu thread menangani satu baris (satu vertex). Tidak ada dua thread yang menulis slot yang sama, jadi baca-tambah-tulis aman tanpa atomik.
- Kontribusi ini lalu dikumpulkan ke matriks oleh `assemble_matrix` lewat peta kontribusi yang sama (blok inersia sudah terdaftar di diagonal sejak
  Step 3). Tidak ada perubahan di perakitan.

```wgsl
for (var k = 0u; k < collider_count(); k = k + 1u) {
  add_contact(row, x, k);
}
```

Dengan `collider_count() = 0` (E3/E4), loop tidak pernah berjalan, sehingga hasilnya **bit-identik**. Ini dicek dengan regresi E3 dan E4-final
(dokumen 5.11).

Binding baru: `@binding(7) colliders`. Kernel ini sekarang memakai 8 storage buffer, masih di bawah batas 10 per tahap (dokumen 1.13).

## 3. `inertia_energy_change.wgsl`: perubahan energi barrier

```wgsl
let inertia = row_mass[row] * (dot(offset, moved) + 0.5 * dot(moved, moved));
if (collider_count() == 0u) {
  elem_delta[DELTA_OFFSET + row] = inertia;
  return;
}
elem_delta[DELTA_OFFSET + row] = inertia + (contact_energy(x + moved) - contact_energy(x));
```

Line search Armijo butuh ΔΦ(α). Bagian inersia tetap rumus E3 yang eksak. Bagian barrier ditulis b(baru) − b(lama) per vertex, sesuai ketetapan prompt.
Cabang `collider_count() == 0u` membuat ekspresi E3/E4 tidak tersentuh.

Pengurangan b(baru) − b(lama) di f32 membawa derau ~ε32·|b| per vertex aktif. Di dekat lantai f32 Newton, derau ini bisa lebih besar dari penurunan
energi yang diminta Armijo, sehingga line search gagal. Dampaknya diukur di dokumen 5.11.
