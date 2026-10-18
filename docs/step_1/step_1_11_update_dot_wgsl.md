# Step 1.11: `gpu/kernels/update_dot.wgsl`, Update x, r, z dalam Satu Lintasan

> Contoh paling jelas dari **fusi kernel**: tiga update vektor, satu preconditioner, dan dua dot product dalam satu kernel.

---

## 1. Tugas

```
x += α p
r -= α q
z  = D⁻¹ r
partial rᵀz dan rᵀr
```

## 2. Isi utama

```wgsl
@group(0) @binding(0) var<storage, read_write> p: array<f32>;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read_write> dinv: array<f32>;
@group(0) @binding(3) var<storage, read_write> state: State;
@group(0) @binding(4) var<storage, read_write> x: array<f32>;
@group(0) @binding(5) var<storage, read_write> r: array<f32>;
@group(0) @binding(6) var<storage, read_write> z: array<f32>;
@group(0) @binding(7) var<storage, read_write> partials: array<f32>;
...
  if (row < arrayLength(&dinv) / 9u) {
    let alpha = state.alpha;
    let i = 3u * row;
    let d = 9u * row;
    x[i] = x[i] + alpha * p[i];  ... (3 komponen)
    let r0 = r[i] - alpha * q[i]; ...
    r[i] = r0; ...
    let z0 = dinv[d] * r0 + dinv[d + 1u] * r1 + dinv[d + 2u] * r2; ...
    z[i] = z0; ...
    rz = r0 * z0 + r1 * z1 + r2 * z2;
    rr = r0 * r0 + r1 * r1 + r2 * r2;
  }
```

## 3. Hal-hal yang patut diperhatikan

### α dibaca dari `State`

`state.alpha` ditulis oleh `reduce` MODE 1 pada dispatch sebelumnya. Batas dispatch menjamin nilai itu sudah terlihat.
Setiap thread membaca alamat yang sama; GPU melayaninya lewat cache, jadi murah.

### `r0` disimpan di register

```wgsl
let r0 = r[i] - alpha * q[i];
r[i] = r0;
let z0 = dinv[d] * r0 + ...
```

Nilai `r` baru langsung dipakai untuk `z` dan kedua dot product **tanpa membaca ulang dari memori**. Inilah inti fusi.

### Hitungan lalu lintas memori

Versi terpisah (satu kernel per operasi) akan membaca/menulis `x, p, r, q, dinv, z` berkali-kali. Versi fusi membaca
`x, p, r, q, dinv` sekali dan menulis `x, r, z` sekali. Untuk solver memory-bound, ini menghemat beberapa lintasan penuh 65 ribu baris
per iterasi.

### Kenapa semua binding `read_write`, bahkan `p` dan `q`?

Kernel ini tidak menulis `p` dan `q`, jadi `read` sudah cukup. Tetapi `p` dan `q` dibuat sebagai buffer `read-write` di host karena
kernel lain menulisnya. Di raw WebGPU dengan `layout: "auto"`, deklarasi di shader yang menentukan layout; `read_write` di sini aman
dan tidak mengubah hasil.

### Stop seragam

Sama dengan `spmv_dot`: thread 0 membaca `state.status`, lalu `workgroupUniformLoad`, dan seluruh workgroup keluar bersama kalau solve selesai.

## 4. Urutan operasi = emulasi

Bandingkan dengan `State::step` di `pcg.rs`:

```rust
self.x[i] = self.x[i] + alpha * self.p[i];
self.r[i] = self.r[i] - alpha * self.q[i];
...
self.z = (0..rows).flat_map(|row| system.precondition_row(row, &self.r)).collect();
let rz_new = reduce_rows(rows, |row| row_dot(&self.r, &self.z, row));
self.rr = reduce_rows(rows, |row| row_dot(&self.r, &self.r, row));
```

Setiap ekspresi punya bentuk yang sama: `x + α·p`, `r − α·q`, `d0*r0 + d1*r1 + d2*r2`. Di sinilah FMA bisa membuat GPU berbeda bit:
kompiler D3D boleh menghitung `x + α·p` sebagai satu instruksi *fused multiply-add* (satu pembulatan, bukan dua). Hasilnya
**lebih akurat**, tetapi tidak identik bit dengan emulasi CPU yang membulatkan dua kali. Karena itu kecocokan bit GPU-vs-emulasi
hanya 0,4–10%, walau jumlah iterasinya sama.

## 5. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Fusi | 3 update + preconditioner + 2 dot dalam satu lintasan |
| Register | `r0..r2` dipakai ulang tanpa baca memori lagi |
| FMA | kompiler GPU boleh menggabung `a*b+c` → beda bit dari emulasi |
| Partial | `(rᵀz, rᵀr)` per workgroup → `reduce` MODE 2 |
