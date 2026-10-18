# Step 3.10: `inertia.wgsl` dan `inertia_energy_change.wgsl`, Suku Inersia Backward Euler di GPU

> Dua kernel kecil, satu thread per **baris** (titik bebas). Keduanya mewujudkan suku `1/(2h²)(x − x̂)ᵀM(x − x̂)` dari potensial inkremental Φ.

---

## 1. `inertia.wgsl`: gradien dan blok diagonal

```wgsl
override GRAD_OFFSET: u32 = 0u;
override HESS_OFFSET: u32 = 0u;

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> x_hat: array<f32>;
@group(0) @binding(2) var<storage, read> row_vertex: array<u32>;
@group(0) @binding(3) var<storage, read> row_mass: array<f32>;
@group(0) @binding(4) var<storage, read_write> elem_grad: array<f32>;
@group(0) @binding(5) var<storage, read_write> elem_hess: array<f32>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  if (row >= arrayLength(&row_mass)) { return; }
  let v = row_vertex[row];
  let mass = row_mass[row];
  for (var d = 0u; d < 3u; d = d + 1u) {
    elem_grad[GRAD_OFFSET + 3u * row + d] = mass * (positions[3u * v + d] - x_hat[3u * v + d]);
  }
  for (var i = 0u; i < 9u; i = i + 1u) {
    elem_hess[HESS_OFFSET + 9u * row + i] = select(0.0, mass, i % 4u == 0u);
  }
}
```

- `row_mass` = m/h² per baris, dihitung oracle dalam f64 lalu dibulatkan.
- Gradien inersia: `(m/h²)(x − x̂)`.
- Hessian inersia: `(m/h²)·I`. Indeks 0, 4, 8 dalam blok row-major 3×3 adalah diagonal, sehingga `i % 4 == 0`.
- Ditulis ke bagian inersia dari buffer bersama (`9T + 12H` untuk gradien, `81T + 144H` untuk Hessian). Kernel perakitan E2 mengambilnya lewat peta gather.

**Kenapa blok diagonal ditulis ulang setiap iterasi**, padahal nilainya konstan? Antarmuka `Backend.write` hanya bisa menulis buffer dari offset 0. Menulis
sebagian buffer dari host berarti memperluas antarmuka E1. Kernel sekecil ini jauh lebih murah daripada itu, jadi diizinkan menulis ulang.

## 2. `inertia_energy_change.wgsl`

```wgsl
override DELTA_OFFSET: u32 = 0u;
...
  let offset = vec3f(positions[...]) - vec3f(x_hat[...]);
  let moved = vec3f(direction[...]) * step[0];
  elem_delta[DELTA_OFFSET + row] = row_mass[row] * (dot(offset, moved) + 0.5 * dot(moved, moved));
```

`½m|x + αd − x̂|² − ½m|x − x̂|² = m((x − x̂)·αd + ½|αd|²)`: eksak, tanpa pengurangan dua angka besar. Ini kernel E3 dengan binding terbanyak (7), masih di
bawah batas 10 yang diminta solver E1.

Saat pertama ditulis, kernel ini memuat fungsi `load` sisa yang tidak terpakai, dan fungsi itu dihapus sebelum run.

## 3. Kenapa per baris, bukan per titik?

Titik yang dipaku tidak punya baris. Dengan iterasi per baris (`row_vertex`), kernel otomatis hanya mengerjakan titik bebas, dan indeks bagian inersia
(`3T + 4H + row`) cocok dengan peta gather dari oracle.

## 4. Ringkasan

| Kernel | Menulis | Rumus |
|---|---|---|
| `inertia` | gradien + blok diagonal | (m/h²)(x − x̂), (m/h²)I |
| `inertia_energy_change` | ΔΦ inersia | (m/h²)((x−x̂)·αd + ½|αd|²) |
