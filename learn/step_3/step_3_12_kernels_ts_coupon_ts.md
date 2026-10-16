# Step 3.12: `kernels.ts` dan `coupon.ts` (Perubahan Kecil Step 3)

> Dua perubahan kecil yang memungkinkan E3 memakai ulang infrastruktur E1/E2.

---

## 1. `kernels.ts`: empat kernel baru, header ketiga

```ts
  hinge_element: { file: "hinge_element.wgsl", header: "hinge.wgsl" },
  hinge_energy_change: { file: "hinge_energy_change.wgsl", header: "hinge.wgsl" },
  inertia: { file: "inertia.wgsl", header: null },
  inertia_energy_change: { file: "inertia_energy_change.wgsl", header: null },
```

Generalisasi `header` yang dibuat di Step 2 (dari `usesState: boolean` ke nama file) langsung terbayar: header ketiga (`hinge.wgsl`) cukup ditulis di registry,
tanpa mengubah `assembleKernels`.

Karena `KernelSources` diturunkan dari registry, `sources.hinge_element` langsung diperiksa compiler di `drape-gpu.ts`.

## 2. `coupon.ts`: `numbers` di-export

```ts
export function numbers<K extends string>(source: unknown, keys: readonly K[], label: string): Record<K, number> { ... }
```

Helper validasi "objek berisi angka dengan kunci ini" dari Step 2 dipakai ulang oleh `drape.ts`. Tipe `NewtonSettings`, `PcgSettings`, dan `Material` juga diimpor dari
`coupon.ts`, sehingga ketiga eksperimen berbagi satu definisi pengaturan.

## 3. Ringkasan

| File | Perubahan | Alasan |
|---|---|---|
| `kernels.ts` | +4 kernel, header `hinge.wgsl` | kernel E3 |
| `coupon.ts` | `export numbers` | parser drape memakai ulang validasi |
