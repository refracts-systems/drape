# Step 1.17: `gpu/src/webgpu-backend.ts`, WebGPU Mentah dari Adapter sampai Timestamp

> Ini satu-satunya file yang berbicara langsung dengan API WebGPU. Kalau kamu ingin paham "apa yang terjadi saat kita
> menjalankan kernel", baca dokumen ini pelan-pelan. Setiap objek WebGPU dijelaskan dengan padanannya di dunia backend.

---

## 1. Hierarki objek WebGPU

```
navigator.gpu
 └─ GPUAdapter          (satu GPU fisik: RTX 3050 atau iGPU AMD)
     └─ GPUDevice       (sesi/koneksi logis ke GPU itu)
         ├─ GPUBuffer           (memori)
         ├─ GPUShaderModule     (WGSL yang sudah diparse)
         ├─ GPUComputePipeline  (shader + entry point + konstanta, siap jalan)
         ├─ GPUBindGroup        (ikatan buffer ke slot @binding)
         ├─ GPUCommandEncoder   (perekam perintah)
         │    └─ GPUComputePassEncoder (satu "pass" berisi dispatch)
         ├─ GPUQueue            (antrean eksekusi)
         └─ GPUQuerySet         (timestamp)
```

| WebGPU | Analogi backend |
|---|---|
| Adapter | server database fisik |
| Device | koneksi/sesi ke server itu |
| Buffer | tabel/blob di server |
| ShaderModule | kode stored procedure yang sudah diparse |
| Pipeline | prepared statement |
| BindGroup | parameter yang sudah diikat ke statement |
| CommandEncoder | builder transaksi |
| Queue.submit | COMMIT |

## 2. Membuat backend

```ts
export async function createWebGpuBackend(gpu: GPU): Promise<Backend<GPUBuffer, WebGpuKernel>> {
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) {
    throw new Error("navigator.gpu.requestAdapter() returned null: no WebGPU adapter");
  }
  const timestamps = adapter.features.has("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: timestamps ? ["timestamp-query"] : [],
    requiredLimits: { maxStorageBuffersPerShaderStage: REQUIRED_STORAGE_BUFFERS },
  });
  const context: Context = { device, errors: watchErrors(device), timer: timestamps ? createTimer(device) : null };
  return backendFor(context, describeAdapter(adapter.info));
}
```

- `powerPreference: "high-performance"` **meminta** GPU diskrit. Di Windows Chrome mengabaikannya (log: *"The powerPreference option is
  currently ignored when calling requestAdapter() on Windows"*). Pemilihan GPU sebenarnya ditentukan pengaturan Windows "Graphics
  preference: High performance" untuk `chrome.exe`, yang sudah dipasang pengguna untuk Chromium milik Playwright.
- **Feature** (kemampuan opsional seperti timestamp) dan **limit** (angka maksimum seperti jumlah storage buffer) harus diminta
  **saat membuat device**. Setelah itu tidak bisa diubah.

## 3. Menangkap error

```ts
function watchErrors(device: GPUDevice): string[] {
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (event) => errors.push(event.error.message));
  void device.lost.then((info) => errors.push(`device lost (${info.reason}): ${info.message}`));
  return errors;
}
```

WebGPU **asinkron**: sebagian besar error validasi tidak dilempar sebagai exception, melainkan dilaporkan belakangan lewat event.
Backend mengumpulkannya di array, lalu `throwIfErrors` dipanggil setelah operasi penting. Tanpa ini, kesalahan seperti "binding salah tipe"
hanya menghasilkan angka nol diam-diam.

`device.lost`: GPU bisa "hilang" (driver crash, laptop beralih GPU). Ini juga dicatat.

## 4. Buffer

```ts
const USAGE = { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, STORAGE: 0x80, QUERY_RESOLVE: 0x200 } as const;

function createArray(device: GPUDevice, label: string, data: HostArray, access: ArrayAccess): GPUBuffer {
  const buffer = device.createBuffer({ label: `${label} (${access})`, size: data.byteLength, usage: USAGE.STORAGE | USAGE.COPY_DST | USAGE.COPY_SRC });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}
```

Setiap buffer dideklarasikan **untuk apa** ia dipakai (bit flag `usage`):

- `STORAGE`: bisa diikat sebagai `var<storage>` di shader.
- `COPY_DST`: bisa ditulis host (`writeBuffer`).
- `COPY_SRC`: bisa disalin ke buffer lain (untuk dibaca balik).

Konstanta ditulis manual (bukan `GPUBufferUsage.STORAGE`) karena kode ini juga di-*strip* tipenya dan dijalankan di halaman browser;
angkanya adalah nilai standar dari spesifikasi.

`queue.writeBuffer` menjadwalkan upload. Upload terjadi berurutan dengan pekerjaan lain di antrean.

## 5. Kernel = pipeline + bind group

```ts
async function createKernel(context: Context, spec: KernelSpec<GPUBuffer>): Promise<WebGpuKernel> {
  const { device } = context;
  const module = device.createShaderModule({ label: spec.label, code: spec.source });
  await assertCompiled(module, spec.label);
  const pipeline = await device.createComputePipelineAsync({
    label: spec.label, layout: "auto", compute: { module, entryPoint: "cs_main", constants: spec.constants },
  });
  const bindGroup = device.createBindGroup({ label: spec.label, layout: pipeline.getBindGroupLayout(0), entries: bindGroupEntries(spec) });
  throwIfErrors(context.errors);
  return { pipeline, bindGroup };
}
```

1. **ShaderModule**: parse WGSL. `assertCompiled` membaca `getCompilationInfo()` dan melempar error lengkap dengan nomor baris kalau ada
   kesalahan sintaks atau tipe.
2. **Pipeline**: gabungan modul + entry point `cs_main` + nilai `override` (`constants: { MODE: 1 }`). Di sinilah kode mesin GPU
   dihasilkan. `createComputePipelineAsync` dipakai agar tidak memblokir.
3. `layout: "auto"`: WebGPU menyimpulkan layout binding dari deklarasi di shader (jumlah, tipe, read/read_write).
4. **BindGroup**: buffer konkret yang diikat ke slot-slot itu.

### Memetakan nama ke nomor binding

```ts
function bindingIndices(source: string): Map<string, number> {
  const declarations = source.matchAll(/@binding\((\d+)\)\s*var<[^>]*>\s*(\w+)/g);
  return new Map(Array.from(declarations, (match) => [match[2], Number(match[1])] as const));
}

function bindGroupEntries(spec: KernelSpec<GPUBuffer>): GPUBindGroupEntry[] {
  const indices = bindingIndices(spec.source);
  const unknown = Object.keys(spec.bindings).filter((name) => !indices.has(name));
  if (unknown.length > 0 || indices.size !== Object.keys(spec.bindings).length) {
    throw new Error(`${spec.label}: bindings [...] do not match shader [...]`);
  }
  ...
}
```

Regex mencari pola `@binding(N) var<...> nama` di teks WGSL. Lalu dicek dua arah: setiap nama yang diberikan host harus ada di shader,
dan jumlahnya harus sama. Salah ketik nama binding langsung menghasilkan error yang jelas. Ini validasi kontrak antara host dan
shader, seperti memvalidasi parameter query terhadap skema.

## 6. Menjalankan: satu pass, satu submit

```ts
async function runPass(context: Context, dispatches: readonly Dispatch<WebGpuKernel>[]): Promise<number | null> {
  const { device, timer } = context;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass(timer === null ? {} : { timestampWrites: timer.writes });
  for (const dispatch of dispatches) {
    pass.setPipeline(dispatch.kernel.pipeline);
    pass.setBindGroup(0, dispatch.kernel.bindGroup);
    pass.dispatchWorkgroups(dispatch.workgroups);
  }
  pass.end();
  timer?.resolve(encoder);
  device.queue.submit([encoder.finish()]);
  const milliseconds = timer === null ? await device.queue.onSubmittedWorkDone().then(() => null) : await timer.read();
  throwIfErrors(context.errors);
  return milliseconds;
}
```

- **CommandEncoder** merekam perintah, tetapi **tidak** menjalankannya.
- **ComputePass**: di dalamnya, untuk setiap dispatch: pilih pipeline, ikat bind group, `dispatchWorkgroups(255)`.
- `encoder.finish()` menghasilkan *command buffer*, lalu `queue.submit` mengirimnya ke GPU.
- WebGPU menjamin dispatch dalam satu pass dieksekusi berurutan **dengan visibilitas memori** antar dispatch (dispatch kedua melihat
  tulisan dispatch pertama). Inilah "batas dispatch = sinkronisasi global" dari dokumen 1.8.
- Host menunggu GPU selesai lewat `onSubmittedWorkDone()` atau dengan membaca hasil timestamp.

Analogi: merekam semua `INSERT/UPDATE` dalam satu transaksi, lalu `COMMIT` sekali.

## 7. Timestamp GPU

```ts
function createTimer(device: GPUDevice): PassTimer {
  const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolved = device.createBuffer({ size: 16, usage: USAGE.QUERY_RESOLVE | USAGE.COPY_SRC });
  const readback = device.createBuffer({ size: 16, usage: USAGE.MAP_READ | USAGE.COPY_DST });
  return {
    writes: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
    resolve: (encoder) => {
      encoder.resolveQuerySet(querySet, 0, 2, resolved, 0);
      encoder.copyBufferToBuffer(resolved, 0, readback, 0, 16);
    },
    read: async () => {
      await readback.mapAsync(MAP_MODE_READ);
      const ticks = new BigUint64Array(readback.getMappedRange().slice(0));
      readback.unmap();
      return Number(ticks[1] - ticks[0]) / 1e6;
    },
  };
}
```

GPU mencatat jam internalnya (nanodetik, 64-bit) di **awal** dan **akhir** pass. Dua angka itu di-*resolve* ke buffer, disalin ke buffer
yang bisa dipetakan (map) ke CPU, lalu dibaca sebagai `BigUint64Array`. Selisihnya ÷ 10⁶ = milidetik.

Ini waktu **murni GPU**, tanpa overhead JavaScript, IPC browser, atau scheduling. Inilah dasar angka bandwidth 58 GB/s (33%).

## 8. Membaca balik buffer

```ts
async function readBuffer(context: Context, source: GPUBuffer): Promise<ArrayBuffer> {
  const staging = device.createBuffer({ size: source.size, usage: USAGE.MAP_READ | USAGE.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, source.size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(MAP_MODE_READ);
  const copy = staging.getMappedRange().slice(0);
  staging.destroy();
  ...
}
```

Buffer `STORAGE` tidak bisa langsung dibaca CPU. Polanya:

1. Buat buffer **staging** dengan `MAP_READ`.
2. Salin storage → staging di GPU.
3. `mapAsync`: tunggu sampai GPU selesai dan memori terlihat CPU.
4. `slice(0)` menyalin isinya ke `ArrayBuffer` biasa (mapped range akan tidak valid setelah buffer dihancurkan).

Setiap `read` berarti satu round-trip. Itulah kenapa solver membaca `State` hanya per chunk.

## 9. Deskripsi adapter

```ts
/** backend and driver exist only with --enable-webgpu-developer-features. */
const developerField = (info: GPUAdapterInfo, key: string): string => {
  const value: unknown = Reflect.get(info, key);
  return typeof value === "string" ? value : "";
};

const describeAdapter = (info: GPUAdapterInfo): string =>
  [info.vendor, info.architecture, info.device, info.description, developerField(info, "backend"), developerField(info, "driver")]
    .filter((part) => part.length > 0).join(" / ") || "unknown adapter";
```

Hasil nyata: `nvidia / ampere / 0x25a2 / NVIDIA GeForce RTX 3050 Laptop GPU / D3D12 / D3D12 driver version 32.0.16.1714`.

`backend` dan `driver` **bukan** bagian tipe standar `GPUAdapterInfo`; Chrome hanya mengisinya dengan flag developer. Karena itu dibaca
lewat `Reflect.get` + pengecekan `typeof`, tanpa `any`. Field inilah yang membuktikan run memakai **D3D12** dan **driver RTX 3050**.

## 10. Ringkasan

| Langkah | Objek WebGPU |
|---|---|
| Pilih GPU | `requestAdapter` |
| Buka sesi + minta fitur/limit | `requestDevice` |
| Memori | `createBuffer` + `queue.writeBuffer` |
| Kompilasi | `createShaderModule` → `createComputePipelineAsync` |
| Ikat buffer | `createBindGroup` |
| Rekam + kirim | `CommandEncoder` → `ComputePass` → `queue.submit` |
| Ukur | `QuerySet` timestamp |
| Baca balik | copy ke staging → `mapAsync` |
