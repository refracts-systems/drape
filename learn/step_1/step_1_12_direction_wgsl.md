# Step 1.12: `gpu/kernels/direction.wgsl`, Arah Pencarian Baru

> Kernel terpendek di proyek. Ia berguna untuk melihat **kapan `return` dini boleh dipakai** (berbeda dengan kernel yang punya barrier).

---

## 1. Isi file

```wgsl
// p = z + βp, skipped once the solve has stopped.
@group(0) @binding(0) var<storage, read_write> z: array<f32>;
@group(0) @binding(1) var<storage, read_write> state: State;
@group(0) @binding(2) var<storage, read_write> p: array<f32>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  if (state.status != 0u || row >= arrayLength(&z) / 3u) {
    return;
  }
  let beta = state.beta;
  let i = 3u * row;
  p[i] = z[i] + beta * p[i];
  p[i + 1u] = z[i + 1u] + beta * p[i + 1u];
  p[i + 2u] = z[i + 2u] + beta * p[i + 2u];
}
```

## 2. Kenapa di sini boleh `if (state.status != 0u) return;` langsung?

Karena kernel ini **tidak punya `workgroupBarrier()`**. Tidak ada titik sinkronisasi yang bisa "ditinggalkan" oleh thread yang keluar
duluan. Setiap thread benar-benar independen, sehingga keluar kapan pun aman dan kompiler tidak mengeluh soal uniformity.

Bandingkan:

| Kernel | Ada barrier? | Cara berhenti |
|---|---|---|
| `jacobi_invert`, `direction` | tidak | `return` langsung |
| `spmv_dot`, `update_dot`, `reduce`, `residual` | ya | thread 0 baca → `workgroupUniformLoad` → semua keluar bersama |

## 3. Urutan di iterasi

```
spmv_dot → reduce(α) → update_dot → reduce(β, konvergensi) → direction
```

`direction` paling akhir karena ia butuh β dari `reduce` MODE 2. Kalau MODE 2 baru saja menyatakan konvergen (`status = 1`),
`direction` dilewati. Ini sesuai dengan Rust:

```rust
converged = state.rr <= tolerance2 * state.rr0;
if !converged {
    state.update_direction(beta);
}
```

## 4. Hubungan dengan residual replacement

Residual replacement dijalankan **setelah** `direction` di akhir setiap chunk 50 iterasi. Jadi `p` sudah diperbarui dengan β dari
residual rekursif, lalu `r`, `z`, `rz` diganti. Urutan yang sama ditiru di `pcg.rs`: `update_direction` dulu, baru `replace_residual`.
Kalau urutannya berbeda di salah satu sisi, emulasi tidak akan setia dan gate `faithful` gagal.

## 5. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| `p = z + βp` | arah konjugat baru |
| `return` dini | aman kalau kernel tidak punya barrier |
| Urutan | terakhir dalam iterasi; dilewati kalau sudah konvergen |
