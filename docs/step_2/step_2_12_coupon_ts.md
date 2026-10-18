# Step 2.12: `gpu/src/coupon.ts`, Parsing Soal Kupon di TypeScript

> Kembaran `problem.ts` untuk Step 2. Ia memakai ulang helper validasi yang di-`export` dari `problem.ts`, dan menunjukkan
> cara mengetik banyak array biner secara aman dengan tipe turunan dari daftar nama.

---

## 1. Tipe manifest

```ts
export interface CouponManifest {
  readonly case: string;
  readonly path: string;
  readonly grid_columns: number;
  readonly vertices: number;
  readonly triangles: number;
  readonly rows: number;
  readonly unknowns: number;
  readonly blocks: number;
  readonly fixed: number;
  readonly steps: number;
  readonly material: Material;
  readonly newton: NewtonSettings;
  readonly pcg: PcgSettings;
  readonly arrays: readonly ArrayEntry[];
}
```

Objek bersarang (`material`, `newton`, `pcg`) mencerminkan `problem.json` dari `e2_generate.rs`. Nama field `snake_case` = kunci JSON.

## 2. Daftar array sebagai sumber tipe

```ts
const F32_ARRAYS = ["rest_positions", "initial_positions", "step_gradients", "elem_geom", "fixed_positions", "kernel_state"] as const;
const U32_ARRAYS = ["elem_vertices", "row_vertex", "dof_mask", "row_ptr", "col_idx", "diag_idx", "block_contrib_ptr",
  "block_contrib_idx", "force_contrib_ptr", "force_contrib_idx", "fixed_vertex"] as const;

export type F32ArrayName = (typeof F32_ARRAYS)[number];
export type U32ArrayName = (typeof U32_ARRAYS)[number];

export interface CouponProblem {
  readonly manifest: CouponManifest;
  readonly f32: Readonly<Record<F32ArrayName, Float32Array<ArrayBuffer>>>;
  readonly u32: Readonly<Record<U32ArrayName, Uint32Array<ArrayBuffer>>>;
}
```

`(typeof X)[number]` mengubah tuple konstan menjadi union string: `"rest_positions" | "initial_positions" | …`. Lalu
`Record<F32ArrayName, Float32Array>` adalah objek dengan **tepat** kunci-kunci itu.

Hasilnya, di kode lain `problem.f32.initial_positions` dan `problem.u32.row_vertex` diperiksa compiler, sedangkan `problem.f32.row_vertex` (salah dtype)
adalah error kompilasi. Satu daftar, dua kegunaan: runtime (loop parsing) dan compile time (tipe).

## 3. Validasi angka bersarang

```ts
function numbers<K extends string>(source: unknown, keys: readonly K[], label: string): Record<K, number> {
  if (!isObject(source)) throw new Error(`problem.json: '${label}' is not an object`);
  return Object.fromEntries(keys.map((key) => [key, field(source, key, isNumber)])) as Record<K, number>;
}
```

Satu helper generik memvalidasi "objek berisi angka-angka dengan kunci ini". Dipakai untuk ukuran root, material, newton, dan pcg. `field` dan `isNumber`
diimpor dari `problem.ts` (fungsi-fungsi itu di-`export` di Step 2).

```ts
export function parseCouponManifest(text: string): CouponManifest {
  const value: unknown = JSON.parse(text);
  if (!isObject(value)) throw new Error("problem.json is not an object");
  return {
    ...sizes(value),
    case: field(value, "case", isString),
    path: field(value, "path", isString),
    material: numbers(Reflect.get(value, "material"), ["wale_N_per_m", "course_N_per_m", "coupling_N_per_m", "shear_N_per_m"] as const, "material"),
    newton: numbers(Reflect.get(value, "newton"), ["tolerance", "max_iterations", "armijo", "max_halvings"] as const, "newton"),
    pcg: numbers(Reflect.get(value, "pcg"), ["tolerance", "max_iterations", "replace_every", "replace_above"] as const, "pcg"),
    arrays: field(value, "arrays", isArray).map(parseEntry),
  };
}
```

## 4. Memotong biner

```ts
export function parseCouponProblem(manifest: CouponManifest, bytes: ArrayBuffer): CouponProblem {
  const f32 = Object.fromEntries(F32_ARRAYS.map((name) => [name, new Float32Array(slice(manifest, bytes, name, "f32"))]));
  const u32 = Object.fromEntries(U32_ARRAYS.map((name) => [name, new Uint32Array(slice(manifest, bytes, name, "u32"))]));
  return { manifest, f32: f32 as Record<F32ArrayName, Float32Array<ArrayBuffer>>, u32: u32 as Record<U32ArrayName, Uint32Array<ArrayBuffer>> };
}
```

`slice` dari `problem.ts` (sudah dibuat struktural supaya menerima manifest apa pun yang punya `arrays`). Cast `as Record<...>` di sini aman karena
`Object.fromEntries` atas daftar nama yang **sama** menghasilkan tepat kunci-kunci itu; TypeScript saja yang tidak bisa membuktikannya sendiri.
Ini bukan `any`.

## 5. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Reuse helper | `field`, `isNumber`, `slice` dari problem.ts |
| Daftar → tipe | `(typeof X)[number]` + `Record` |
| Validasi bersarang | satu helper generik `numbers` |
| Pemisahan dtype | `f32` dan `u32` di objek berbeda; salah dtype = error kompilasi |
