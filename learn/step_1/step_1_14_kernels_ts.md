# Step 1.14: `gpu/src/kernels.ts`, Memuat File WGSL dan Menempelkan Header

> File kecil yang menjawab: "WGSL tidak punya `#include`, lalu bagaimana banyak kernel berbagi kode?".
> Versi sekarang sudah digeneralisasi di Step 2 supaya mendukung lebih dari satu jenis header.

---

## 1. Isi file (versi sekarang)

```ts
export const KERNEL_FILES = {
  jacobi_invert: { file: "jacobi_invert.wgsl", header: null },
  cg_init: { file: "cg_init.wgsl", header: null },
  spmv_dot: { file: "spmv_dot.wgsl", header: "state.wgsl" },
  reduce: { file: "reduce.wgsl", header: "state.wgsl" },
  update_dot: { file: "update_dot.wgsl", header: "state.wgsl" },
  direction: { file: "direction.wgsl", header: "state.wgsl" },
  residual: { file: "residual.wgsl", header: "state.wgsl" },
  membrane_element: { file: "membrane_element.wgsl", header: "membrane.wgsl" },
  membrane_energy_change: { file: "membrane_energy_change.wgsl", header: "membrane.wgsl" },
  assemble_matrix: { file: "assemble_matrix.wgsl", header: null },
  assemble_force: { file: "assemble_force.wgsl", header: null },
} as const;

export type KernelName = keyof typeof KERNEL_FILES;
export type KernelSources = Readonly<Record<KernelName, string>>;

const KERNEL_NAMES = Object.keys(KERNEL_FILES) as readonly KernelName[];

/** Reads every kernel through `readText` and prepends its shared header (State struct or membrane law). */
export async function assembleKernels(readText: (file: string) => Promise<string>): Promise<KernelSources> {
  const entries = await Promise.all(KERNEL_NAMES.map(async (name) => {
    const { file, header } = KERNEL_FILES[name];
    const body = await readText(file);
    return [name, header === null ? body : `${await readText(header)}\n${body}`] as const;
  }));
  return Object.fromEntries(entries) as KernelSources;
}
```

## 2. Registry sebagai tipe

`KERNEL_FILES` adalah objek konstan (`as const`). Dari objek itu TypeScript menurunkan:

- `KernelName` = `"jacobi_invert" | "cg_init" | ... | "assemble_force"` (gabungan semua kunci).
- `KernelSources` = objek dengan **tepat** kunci-kunci itu, masing-masing string.

Keuntungannya: di `solver.ts`, menulis `sources.spmv_dot` diperiksa compiler. Salah ketik `sources.spmv_dott` adalah error kompilasi,
bukan `undefined` saat runtime. Ini pola "konfigurasi sebagai sumber tipe" yang juga umum di backend TypeScript
(misalnya daftar route atau daftar event).

## 3. Header = include manual

`state.wgsl` ditempel di depan setiap kernel yang memakai `State`. Untuk Step 2, `membrane.wgsl` ditempel di depan dua kernel membran.
Kernel yang tidak butuh header diberi `null`.

**Perubahan di Step 2**: dulu bentuknya `usesState: boolean`. Begitu ada header kedua (`membrane.wgsl`), boolean tidak cukup lagi,
sehingga digeneralisasi menjadi nama file header. Perubahan ini tidak mengubah teks kernel E1 sedikit pun; tujuh kernel E1 tetap
mendapat teks yang identik.

## 4. Dependency injection untuk I/O

```ts
export async function assembleKernels(readText: (file: string) => Promise<string>)
```

Fungsi ini tidak tahu cara membaca file. Ia menerima **fungsi pembaca** dari luar:

- Di Node (`files.ts`): `(file) => readFile(join(KERNEL_DIR, file), "utf8")`
- Di browser (`browser-main.ts`): `(file) => fetch("/kernels/" + file).then(r => r.text())`

Satu logika, dua lingkungan. Pola yang sama dengan menyuntikkan *repository* atau *HTTP client* di service backend supaya bisa diuji
dan dipakai ulang.

## 5. `Promise.all`

Semua file dibaca **paralel**. Ada 11 kernel; header dibaca ulang per kernel (sederhana, dan file kecil). Hasilnya diubah kembali ke objek
dengan `Object.fromEntries`.

## 6. Jebakan yang perlu diingat

- Header ditempel sebagai **teks**. Nama di header (`State`, `WALE`, `columns_of`) masuk ke ruang nama kernel. Kalau kernel
  mendefinisikan nama yang sama, kompilasi gagal karena redefinisi.
- Nomor baris di pesan error kompilasi WGSL dihitung dari teks **gabungan**. Error di baris 12 kernel `spmv_dot` dilaporkan sebagai
  baris (12 + panjang header).

## 7. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| `as const` registry | daftar kernel sekaligus sumber tipe `KernelName` |
| Header | include manual dengan menggabung string |
| `readText` disuntik | logika sama di Node dan browser |
| Generalisasi Step 2 | `usesState: boolean` → `header: file \| null` |
