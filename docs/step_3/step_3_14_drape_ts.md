# Step 3.14: `gpu/src/drape.ts`, Parsing Soal Drape

> Pola yang sama dengan `coupon.ts` (dokumen 2.12), untuk manifest dan array Step 3.

---

## 1. Manifest

```ts
export interface DrapeManifest {
  readonly case: string;
  readonly grid_columns: number;
  readonly vertices: number;
  readonly triangles: number;
  readonly hinges: number;
  readonly rows: number;
  readonly unknowns: number;
  readonly blocks: number;
  readonly steps: number;
  readonly probes: number;
  readonly time_step_s: number;
  readonly gravity_m_per_s2: number;
  readonly tangent: string;
  readonly material: Material;
  readonly newton: NewtonSettings;
  readonly pcg: PcgSettings;
  readonly arrays: readonly ArrayEntry[];
}
```

Field baru dibanding kupon:

- `hinges`: jumlah hinge (ukuran bagian hinge di buffer lokal).
- `probes`: 5 kalau ada referensi f64, 0 untuk n129.
- `time_step_s`, `gravity_m_per_s2`: untuk menghitung x̂ di host. Angkanya datang dari oracle, bukan dari konstanta TypeScript.
- `tangent`: `"projected"` (efek trigger). Host memilih varian pipeline dari sini.

## 2. Array

```ts
const F32_ARRAYS = ["rest_positions", "elem_geom", "hinge_stiffness", "row_mass", "kernel_state", "probe_positions", "probe_velocities"] as const;
const U32_ARRAYS = ["elem_vertices", "hinge_vertices", "row_vertex", "dof_mask", "row_ptr", "col_idx", "diag_idx", "block_contrib_ptr",
  "block_contrib_idx", "force_contrib_ptr", "force_contrib_idx", "fixed_vertex", "probe_steps"] as const;
```

Sama seperti `coupon.ts`: daftar konstan sekaligus sumber tipe (`DrapeF32Name`, `DrapeU32Name`). Array probe boleh kosong (panjang 0) untuk n129;
`slice` menangani panjang 0 dengan benar.

## 3. Validasi

```ts
const SIZES = ["grid_columns", "vertices", "triangles", "hinges", "rows", "unknowns", "blocks", "steps", "probes", "time_step_s", "gravity_m_per_s2"] as const;

export function parseDrapeManifest(text: string): DrapeManifest {
  ...
  return {
    ...numbers(value, SIZES, "root"),
    case: field(value, "case", isString),
    tangent: field(value, "tangent", isString),
    material: numbers(Reflect.get(value, "material"), [...] as const, "material"),
    ...
  };
}
```

`numbers` dan `field` dipakai ulang dari Step 2. Spread `...numbers(...)` memasukkan semua angka ke objek hasil sekaligus.

## 4. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Pola kupon | manifest bertipe + array per dtype |
| Field baru | hinge, probe, h, g, jenis tangen |
| Probe kosong | n129 tanpa referensi |
