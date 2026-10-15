# Step 1.19: `problem.ts`, `node/cli.ts`, `node/files.ts`, Parsing Soal, Argumen CLI, dan I/O

> Tiga file "lem" yang kecil. Di sini terlihat bagaimana TypeScript membaca format biner oracle dengan aman tanpa `any`,
> dan bagaimana CLI divalidasi.

---

## 1. `gpu/src/problem.ts`: membaca `problem.json` + `problem.bin`

### Tipe

```ts
export interface ProblemManifest {
  readonly case: string;
  readonly grid_columns: number;
  readonly rows: number;
  readonly unknowns: number;
  readonly blocks: number;
  readonly replace_every: number;
  readonly replace_above: number;
  readonly arrays: readonly ArrayEntry[];
}

export interface Problem {
  readonly manifest: ProblemManifest;
  readonly rowPtr: Uint32Array<ArrayBuffer>;
  readonly colIdx: Uint32Array<ArrayBuffer>;
  readonly diagIdx: Uint32Array<ArrayBuffer>;
  readonly vals: Float32Array<ArrayBuffer>;
  readonly rhs: Float32Array<ArrayBuffer>;
}
```

Nama field memakai `snake_case` karena sama persis dengan kunci JSON dari Rust. Ini kontrak lintas bahasa.

### Validasi JSON tanpa `any`

```ts
export function field<T>(source: object, key: string, check: (value: unknown) => value is T): T {
  const value: unknown = Reflect.get(source, key);
  if (!check(value)) {
    throw new Error(`problem.json: field '${key}' is missing or has the wrong type`);
  }
  return value;
}

export const isNumber = (value: unknown): value is number => typeof value === "number";
```

`JSON.parse` mengembalikan `any`. Di sini hasilnya langsung dianggap `unknown`, lalu setiap field diambil lewat `field(obj, "rows", isNumber)`:

- `Reflect.get` membaca properti tanpa membuat TypeScript protes soal tipe.
- `check` adalah **type guard** (`value is number`): kalau lolos, TypeScript tahu nilainya `number`.
- Kalau field hilang atau salah tipe, error menyebut nama field-nya.

Ini versi manual dari validator skema (zod, joi) tanpa dependensi. Di Step 2 fungsi-fungsi ini di-`export` supaya dipakai ulang oleh
`coupon.ts`.

Dua field baru di sesi ini bersifat **opsional dengan default 0**, supaya file soal lama (tanpa replacement) tetap terbaca:

```ts
replace_every: Reflect.has(value, "replace_every") ? field(value, "replace_every", isNumber) : 0,
replace_above: Reflect.has(value, "replace_above") ? field(value, "replace_above", isNumber) : 0,
```

### Memotong biner

```ts
export function slice(manifest: { readonly arrays: readonly ArrayEntry[] }, bytes: ArrayBuffer, name: string, dtype: string): ArrayBuffer {
  const entry = manifest.arrays.find((candidate) => candidate.name === name);
  if (entry === undefined || entry.dtype !== dtype) {
    throw new Error(`problem.bin: array '${name}' (${dtype}) not found`);
  }
  return bytes.slice(entry.offset, entry.offset + 4 * entry.length);
}
```

Manifest memberi offset dan panjang, lalu `bytes.slice` memotong bagiannya, dan `new Float32Array(...)` menafsirkannya sebagai float.
Tipe parameter `manifest` sengaja dibuat struktural (`{ arrays }`) di Step 2 supaya manifest kupon juga bisa memakainya.

**Catatan endianness**: Rust menulis little-endian; JavaScript TypedArray memakai endianness mesin, yang di x86/ARM modern juga
little-endian. Jadi tidak ada konversi.

### Pemeriksaan konsistensi

```ts
function checkSizes(problem: Problem): Problem {
  const { rows, unknowns, blocks } = problem.manifest;
  const consistent = problem.rowPtr.length === rows + 1 && problem.colIdx.length === blocks
    && problem.diagIdx.length === rows && problem.vals.length === 9 * blocks && problem.rhs.length === unknowns;
  ...
}
```

Fail fast: kalau panjang array tidak cocok dengan metadata, lebih baik error di sini daripada kernel GPU membaca di luar batas.

## 2. `gpu/src/node/cli.ts`: argumen

```ts
/** --case <dir> [--tol 1e-7] [--max-iter 20000] [--chunk 32] [--repeats 3] [--adapter <name>] */
export function parseCommon<Adapter extends string>(argv: readonly string[], adapters: readonly Adapter[], fallback: Adapter): CommonOptions<Adapter> {
  const { values } = parseArgs({ args: [...argv], strict: true, allowPositionals: false, options: { ... } });
  if (values.case === undefined || !isOneOf(values.adapter, adapters)) {
    throw new Error(`usage: ...`);
  }
  ...
}
```

- `parseArgs` bawaan Node (`node:util`), tanpa library CLI.
- `strict: true`: argumen tak dikenal adalah error. Salah ketik `--adaptor` tidak diam-diam diabaikan.
- Generik `Adapter extends string` membuat runner WebGPU menerima `hardware | software`, sedangkan runner vgpu `auto | hardware | software`,
  dengan satu fungsi.
- `positive(text, flag)`: angka harus positif dan finite.

Perhatikan: tidak ada opsi `--replace-every` di sini. Pengaturan replacement datang dari `problem.json` (dokumen 1.5), sehingga GPU
tidak bisa tidak sinkron dengan emulasi.

## 3. `gpu/src/node/files.ts`: membaca dan menulis file

```ts
export const KERNEL_DIR = fileURLToPath(new URL("../../kernels/", import.meta.url));

export async function loadProblem(caseDir: string): Promise<Problem> { ... }

export const loadKernels = (): Promise<KernelSources> =>
  assembleKernels((file) => readFile(join(KERNEL_DIR, file), "utf8"));

/** gpu-x.bin (little-endian f32, consumed by `oracle compare`) and gpu-result.json. */
export async function writeOutputs(caseDir: string, report: GpuReport, solution: Float32Array<ArrayBuffer>): Promise<void> {
  await writeFile(join(caseDir, "gpu-x.bin"), new Uint8Array(solution.buffer, solution.byteOffset, solution.byteLength));
  await writeFile(join(caseDir, "gpu-result.json"), `${JSON.stringify({ ...report, ...hostInfo() }, null, 2)}\n`);
}
```

- `import.meta.url` + `new URL("../../kernels/", ...)`: path relatif terhadap **file kode ini**, bukan terhadap direktori kerja.
  Runner bisa dipanggil dari mana pun.
- `gpu-x.bin`: jawaban GPU sebagai f32 mentah, dibaca `oracle compare`.
- `gpu-result.json`: laporan, ditambah `host_os` dan `host_gpus` dari `hardware.ts` (perubahan sesi ini). `compare.rs` lalu menyalinnya
  ke `result.json`, sehingga **setiap hasil mencatat GPU dan driver yang terdeteksi**, sesuai permintaan.

`printSummary` mencetak satu baris per kasus:

```
tension-n257 N=195075 converged it=1545 rec=9.76e-8 true32=7.82e-3 warm=1225.9ms gpu=1091.1ms total=1452.5ms det=true [nvidia / ampere / ...]
```

## 4. Ringkasan

| File | Peran | Teknik |
|---|---|---|
| `problem.ts` | JSON + biner → `Problem` | type guard, `Reflect.get`, slice per manifest |
| `cli.ts` | argumen → pengaturan | `parseArgs` strict, generik adapter |
| `files.ts` | I/O file | path relatif modul, output biner + JSON + info host |
