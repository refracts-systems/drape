# Step 3.9: `gpu/kernels/hinge_energy_change.wgsl`, Perubahan Energi Hinge untuk Line Search

> Kembaran GPU dari `HingeElement::energy_change`. Kecil, tetapi menunjukkan pola "satu buffer `elem_delta` diisi tiga kernel".

---

## 1. Kode

```wgsl
override DELTA_OFFSET: u32 = 0u;
...
fn moved(v: u32) -> vec3f {
  return position(v) + vec3f(direction[3u * v], direction[3u * v + 1u], direction[3u * v + 2u]) * step[0];
}

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let h = gid.x;
  if (h >= arrayLength(&hinge_stiffness)) { return; }
  let v1 = hinge_vertices[4u * h]; ...
  let before = hinge_angle(position(v1), position(v2), position(v3), position(v4));
  let after = hinge_angle(moved(v1), moved(v2), moved(v3), moved(v4));
  elem_delta[DELTA_OFFSET + h] = hinge_stiffness[h] * (after - before) * (after + before);
}
```

- `moved(v)` = `x + d·α`, sama dengan Rust `add(x, scale(d, α))`.
- `k(θ' − θ)(θ' + θ)` per hinge.
- Ditulis di `elem_delta[T + h]`: setelah T entri membran (`DELTA_OFFSET = T`).

## 2. `elem_delta` bersama

```
elem_delta: [ membran: T | hinge: H | inersia: R ]
```

Tiga kernel menulis tiga bagian. Host menjumlahkan **seluruh** buffer dengan `Math.fround`, dan fungsi `energyChange` di `newton.ts` tidak perlu tahu ada
berapa jenis kontribusi. Itulah kenapa inti Newton E2 dipakai ulang tanpa perubahan logika.

## 3. Ketergantungan pada `atan2_accurate`

`before` dan `after` dihitung dengan fungsi sudut yang sama, sehingga sebagian galat sistematik `atan2` bawaan akan saling meniadakan di `after − before`. Tetapi
tidak seluruhnya, karena galat 4096 ULP tidak harus sama untuk dua sudut yang berbeda. Dengan `atan2_accurate`, masalah itu hilang di sumbernya.

## 4. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Rumus | k(θ' − θ)(θ' + θ) per hinge |
| Offset | bagian hinge dari `elem_delta` |
| Buffer bersama | host menjumlahkan semua kontribusi tanpa tahu asalnya |
