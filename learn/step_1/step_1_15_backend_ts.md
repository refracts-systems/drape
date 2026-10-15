# Step 1.15: `gpu/src/backend.ts`, Antarmuka 6 Metode di Atas WebGPU dan vgpu

> File ini hanya berisi tipe. Tetapi ia keputusan arsitektur terpenting di sisi host: **satu solver, dua backend GPU**,
> dan ia yang memungkinkan pengukuran "overhead vgpu vs WebGPU mentah".

---

## 1. Isi file

```ts
/** residual.wgsl in REPLACE mode binds 10 storage buffers; the WebGPU default per-stage limit is 8. */
export const REQUIRED_STORAGE_BUFFERS = 10;

export type ArrayAccess = "read" | "read-write";
export type HostArray = Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>;

export interface KernelSpec<GpuArray> {
  readonly label: string;
  readonly source: string;
  readonly constants: Readonly<Record<string, number>>;
  readonly bindings: Readonly<Record<string, GpuArray>>;
}

export interface Dispatch<GpuKernel> {
  readonly kernel: GpuKernel;
  readonly workgroups: number;
}

/** The only GPU surface Experiment 1 uses: storage arrays, compute kernels, one-pass submits, readback. */
export interface Backend<GpuArray, GpuKernel> {
  readonly name: string;
  readonly adapter: string;
  readonly gpuTiming: boolean;
  array(label: string, data: HostArray, access: ArrayAccess): GpuArray;
  write(target: GpuArray, data: HostArray): void;
  kernel(spec: KernelSpec<GpuArray>): Promise<GpuKernel>;
  run(dispatches: readonly Dispatch<GpuKernel>[]): Promise<number | null>;
  read(source: GpuArray): Promise<ArrayBuffer>;
  dispose(): void;
}
```

## 2. Pola: port & adapter (hexagonal)

`Backend` adalah **port**. Implementasinya, `webgpu-backend.ts` dan `vgpu-backend.ts`, adalah **adapter**. Solver (`solver.ts`)
hanya bergantung pada port.

Analogi backend: `interface Repository { find(), save() }` dengan implementasi Postgres dan in-memory. Logika bisnis tidak peduli
yang mana.

## 3. Generik `GpuArray` dan `GpuKernel`

Setiap backend punya tipe buffer dan kernelnya sendiri:

| Backend | `GpuArray` | `GpuKernel` |
|---|---|---|
| WebGPU mentah | `GPUBuffer` | `{ pipeline, bindGroup }` |
| vgpu | `StorageBuffer` | `Compute` |

Daripada `any` atau `unknown` + cast, interface dibuat generik. `solver.ts` ditulis sebagai `prepareSolver<A, K>(backend: Backend<A, K>, ...)`
dan tidak pernah tahu tipe konkretnya. Ini memenuhi aturan proyek "tanpa `any`" sambil tetap fleksibel.

## 4. Enam metode

| Metode | Tugas | Analogi |
|---|---|---|
| `array(label, data, access)` | buat buffer GPU dan upload data awal | `CREATE TABLE` + `INSERT` |
| `write(target, data)` | timpa isi buffer | `UPDATE` |
| `kernel(spec)` | kompilasi WGSL + ikat buffer → objek siap dispatch | *prepared statement* |
| `run(dispatches)` | jalankan urutan dispatch dalam **satu compute pass dan satu submit**, tunggu selesai | eksekusi batch dalam satu transaksi |
| `read(source)` | salin buffer GPU ke CPU | `SELECT` |
| `dispose()` | lepaskan device | tutup koneksi |

Kontrak `run` penting untuk performa: 32 iterasi × 5 dispatch = 160 dispatch dikirim **sekali**. Satu submit punya overhead tetap
(puluhan mikrodetik sampai milidetik di browser). Satu submit per dispatch akan jauh lebih lambat.

`run` mengembalikan `number | null`: milidetik waktu GPU kalau backend mendukung *timestamp query* (WebGPU mentah), atau `null` (vgpu).

## 5. `KernelSpec`: deskripsi kernel tanpa detail backend

```ts
{ label: "reduce_mode_1", source: "<teks WGSL>", constants: { MODE: 1 }, bindings: { partials: buf1, state: buf2 } }
```

- `constants`: nilai `override` (dokumen 1.10).
- `bindings`: **nama** variabel WGSL → buffer. Backend menerjemahkan nama ke `@binding(n)`.

## 6. `REQUIRED_STORAGE_BUFFERS`

Ditambahkan di sesi kerja setelah error batas 8 storage buffer (dokumen 1.13). Diletakkan di file **bersama** supaya kedua backend
meminta limit yang sama dari satu konstanta. Kalau kelak ada kernel dengan 11 binding, cukup ubah satu angka, dan perangkat yang
tidak sanggup menolak dengan pesan jelas saat membuat device.

## 7. Kenapa abstraksi ini penting untuk eksperimen?

Gate "overhead vgpu ≤ 1,15× WebGPU mentah" hanya adil kalau keduanya menjalankan **kernel yang sama, buffer yang sama, dan urutan dispatch yang sama**.
Dengan satu `solver.ts` di atas dua adapter, satu-satunya perbedaan adalah lapisan API itu sendiri. Hasil: 1,11× (tension) dan 1,14× (dynamic),
lulus.

## 8. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Port/adapter | solver bergantung pada interface, bukan WebGPU/vgpu |
| Generik | tipe buffer/kernel per backend tanpa `any` |
| `run` = 1 pass + 1 submit | batch besar untuk menekan overhead |
| `KernelSpec.bindings` | nama → buffer; backend memetakan ke nomor |
