# Step 1.18: `gpu/src/node/vgpu-backend.ts`, Jalur vgpu (Dawn di Node)

> Adapter kedua untuk interface `Backend`. vgpu adalah library di atas WebGPU untuk Node.js. File ini menunjukkan berapa banyak
> boilerplate WebGPU yang disembunyikan library, dan apa yang hilang karenanya.

---

## 1. Isi lengkap

```ts
import { prepareShader } from "@vgpu/wgsl/prepare";
import { compute, frame, init, storage, type Compute, type Gpu, type StorageBuffer } from "vgpu/node";
import { REQUIRED_STORAGE_BUFFERS, type ArrayAccess, type Backend, type Dispatch, type HostArray } from "../backend.ts";

export type VgpuAdapter = "auto" | "hardware" | "software";

function createArray(gpu: Gpu, data: HostArray, access: ArrayAccess): StorageBuffer {
  const buffer = storage(gpu, data.byteLength, { access });
  buffer.write(data);
  return buffer;
}

/** One frame computePass per run: one encoder, one submit, dispatches in order. */
async function runFrame(gpu: Gpu, dispatches: readonly Dispatch<Compute>[]): Promise<null> {
  const submitted = frame(gpu, (current) => current.computePass({ label: "e1-pcg" }, (pass) => {
    dispatches.forEach((dispatch) => pass.dispatch(dispatch.kernel, dispatch.workgroups));
  }));
  await submitted.done;
  return null;
}

export async function createVgpuBackend(adapter: VgpuAdapter): Promise<Backend<StorageBuffer, Compute>> {
  const gpu = await init({ adapter, requiredLimits: { maxStorageBuffersPerShaderStage: REQUIRED_STORAGE_BUFFERS } });
  return {
    name: "vgpu-node-dawn",
    adapter: `${gpu.adapter.name} (${gpu.adapter.type})`,
    gpuTiming: false,
    array: (_label, data, access) => createArray(gpu, data, access),
    write: (target, data) => target.write(data),
    kernel: (spec) => compute(gpu, prepareShader(spec.source), { label: spec.label, set: spec.bindings, constants: spec.constants }).compile(),
    run: (dispatches) => runFrame(gpu, dispatches),
    read: (source) => source.read(),
    dispose: () => gpu.dispose(),
  };
}
```

## 2. Perbandingan dengan WebGPU mentah

| Tugas | WebGPU mentah (dok. 1.17) | vgpu |
|---|---|---|
| Buka GPU | `requestAdapter` + `requestDevice` + event error | `init({ adapter, requiredLimits })` |
| Buffer | `createBuffer` + usage flags + `writeBuffer` | `storage(gpu, size, { access })` + `.write` |
| Kernel | module + compile info + pipeline + regex binding + bind group | `compute(gpu, shader, { set, constants }).compile()` |
| Jalankan | encoder + pass + loop + submit + tunggu | `frame(...computePass(...))` + `.done` |
| Baca | staging + copy + map + slice | `.read()` |
| Timestamp | query set manual | belum ada → `gpuTiming: false` |

vgpu mengikat binding **berdasarkan nama** sendiri (`set: spec.bindings`), jadi tidak perlu regex. `prepareShader` dari `@vgpu/wgsl`
menyiapkan sumber WGSL untuk vgpu.

## 3. `adapter: "hardware"` dan string adapter

```ts
adapter: `${gpu.adapter.name} (${gpu.adapter.type})`,
```

Hasil nyata di laptop ini: `D3D12 driver version 32.0.16.1714 (gpu)`.

vgpu **tidak** memberi nama GPU, hanya versi driver. Karena itu identifikasi GPU dilakukan lewat versi driver:
`32.0.16.1714` = RTX 3050, `31.0.21925.1001` = iGPU AMD. `hardware.ts` memakai fakta ini (dokumen 1.21).

## 4. Kenapa tidak ada `gpu_ms`?

vgpu 0.6 belum mengekspos timestamp query. Akibatnya untuk vgpu, waktu diukur dari **jam dinding** host (`performance.now()`),
yang mencakup overhead JavaScript dan submit. Bandwidth vgpu dihitung dari waktu ini: 46,8 GB/s (26,5% rated), sedikit lebih rendah
daripada 58 GB/s versi timestamp.

## 5. Akses buffer: aturan vgpu

Sesi kerja menguji dua kombinasi dengan probe kecil:

| Buffer dibuat | Diikat sebagai | Hasil |
|---|---|---|
| `access: "read"` | `var<storage, read_write>` | **ditolak** ("pipeline compilation failed") |
| `access: "read-write"` | `var<storage, read>` | diterima |

Karena vgpu membangun layout dari `access` buffer, menulis ke buffer yang dibuat `read` itu dilarang. Ini alasan `vals` dan `rhs` di
`solver.ts` diubah ke `read-write` di Step 2.

## 6. Insiden Application Control

Di tengah run kedua, Windows sempat menolak memuat binary Dawn:

```
VGPUError: Stock webgpu failed to load on win32/x64 ...
stock: Error: An Application Control policy has blocked this file.
```

Binary yang sama sudah dimuat berkali-kali sebelumnya. Satu kali retry berhasil, dan tidak ada pengaturan keamanan yang diubah.
Pelajarannya: native module (`.node`) di Windows bisa dicegat kebijakan keamanan OS; error ini bukan bug kode.

## 7. Hasil pembanding

| n257 warm solve | WebGPU (Chromium) | vgpu (Node) | rasio |
|---|---|---|---|
| tension | 1226 ms | 1360 ms | 1,11× |
| dynamic | 1158 ms | 1325 ms | 1,14× |

Gate ≤ 1,15× lulus. Menariknya, di ukuran kecil (n33–n129) vgpu justru **lebih cepat**, karena WebGPU di Chromium membayar biaya IPC
antara proses halaman dan proses GPU browser untuk setiap submit.

## 8. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| vgpu | WebGPU dengan boilerplate disembunyikan, berjalan di Node |
| Binding by name | tidak perlu regex `@binding` |
| Tanpa timestamp | waktu dari jam dinding host |
| Adapter string | hanya versi driver → identifikasi GPU lewat driver |
| Aturan akses | buffer `read` tidak boleh ditulis shader |
