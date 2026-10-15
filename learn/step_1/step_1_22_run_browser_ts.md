# Step 1.22: `gpu/src/node/run-browser.ts`, Chromium Headless via Playwright, dan Kisah DXC vs FXC

> Jalur "WebGPU mentah" berjalan di dalam browser sungguhan, karena itulah lingkungan produksi simulator jersey. File ini
> menyalakan Chromium tanpa layar, menyajikan kode dan data lewat server HTTP lokal, lalu mengambil hasilnya.
> Di sinilah masalah hardware paling menarik di sesi kerja ditemukan.

---

## 1. Arsitektur

```
Node (run-browser.ts)
 ├─ server HTTP 127.0.0.1:<port acak>
 │    /                → halaman HTML kecil
 │    /src/*.ts        → kode TS yang di-strip tipenya → JavaScript
 │    /kernels/*.wgsl  → kernel
 │    /case/*          → problem.json, problem.bin
 └─ Playwright → Chromium headless
          halaman memuat browser-main.ts → window.runE1(settings)
          ↓ hasil (laporan + solusi) dikembalikan lewat page.evaluate
 ← Node menulis gpu-x.bin + gpu-result.json
```

## 2. Menyajikan TypeScript tanpa bundler

```ts
if (folder === "src") return { body: stripTypeScriptTypes(await readFile(join(SOURCE_DIR, name), "utf8")), type: "text/javascript" };
```

`stripTypeScriptTypes` (bawaan Node 22+, masih eksperimental) menghapus anotasi tipe sehingga TS menjadi JS valid. Kode TS proyek ditulis
dengan aturan `erasableSyntaxOnly`: tidak memakai fitur TS yang butuh transformasi (enum, namespace). Hasilnya, file `.ts` yang sama
dipakai di Node dan di browser tanpa build step.

Keamanan server: hanya nama file yang cocok `^[\w-]+\.(ts|wgsl|json|bin)$` yang dilayani, dan hanya dari tiga folder yang diketahui.
Server hanya listen di `127.0.0.1`. Ini mencegah path traversal (`/src/../../secret`).

## 3. Flag Chromium

```ts
const BASE_FLAGS = ["--enable-unsafe-webgpu", "--enable-webgpu-developer-features"];
const LINUX_VULKAN = ["--enable-features=Vulkan", "--use-angle=vulkan", "--disable-vulkan-surface"];
const SWIFTSHADER = ["--use-vulkan=swiftshader", "--use-webgpu-adapter=swiftshader", "--no-sandbox"];
/** Playwright's Chromium ships D3DCompiler_47 (FXC) but not dxil.dll/dxcompiler.dll, so D3D12 device creation fails with DXC. */
const WINDOWS_FXC = ["--disable-dawn-features=use_dxc"];

function chromiumFlags(adapter: (typeof ADAPTERS)[number]): string[] {
  const vulkan = process.platform === "linux" || adapter === "software" ? LINUX_VULKAN : [];
  const shaderCompiler = process.platform === "win32" && adapter === "hardware" ? WINDOWS_FXC : [];
  return [...BASE_FLAGS, ...vulkan, ...shaderCompiler, ...(adapter === "software" ? SWIFTSHADER : [])];
}
```

- `--enable-unsafe-webgpu`: aktifkan WebGPU di mode headless/otomasi.
- `--enable-webgpu-developer-features`: memunculkan `info.backend` dan `info.driver` (bukti D3D12 + versi driver di laporan) dan
  timestamp query presisi penuh.

## 4. Kisah DXC vs FXC: debugging hardware nyata

**Gejala**: run pertama di laptop gagal:

```
OperationError: Failed to execute 'requestDevice' on 'GPUAdapter': DynamicLib.Open: dxil.dll Windows Error: 87
```

**Latar belakang**: di Windows, Dawn menerjemahkan WGSL ke HLSL, lalu mengompilasinya untuk D3D12 dengan salah satu dari dua kompiler
Microsoft:

| Kompiler | DLL | Output | Status |
|---|---|---|---|
| **DXC** (baru) | `dxcompiler.dll` + `dxil.dll` | DXIL (Shader Model 6) | default Chrome modern |
| **FXC** (lama) | `D3DCompiler_47.dll` | DXBC (Shader Model 5.1) | fallback |

**Investigasi**:

1. Cari DLL di folder Chromium Playwright: hanya ada `D3DCompiler_47.dll`. **DXC tidak dibundel.**
2. Probe kecil dengan tiga konfigurasi:

```
playwright chromium, default : fail nvidia|ampere|RTX 3050: DynamicLib.Open: dxil.dll ...
playwright chromium, no DXC  : ok nvidia|ampere|0x25a2|NVIDIA GeForce RTX 3050 Laptop GPU
system chrome 155            : ok amd|gcn-5|0x1638|AMD Radeon(TM) Graphics
```

**Kesimpulan**:

- Chromium Playwright memilih GPU yang benar (RTX 3050), tetapi butuh flag `--disable-dawn-features=use_dxc` agar memakai FXC.
- Chrome sistem punya DXC, tetapi memilih **iGPU AMD**, sehingga tidak bisa dipakai (pengaturan preferensi GPU Windows hanya dipasang
  untuk `chrome.exe` Playwright).

Satu-satunya jalur yang memenuhi syarat adalah Chromium Playwright + FXC. Flag hanya dipasang untuk `win32` + `hardware`.

**Dampak**: FXC dan DXC bisa menghasilkan kode mesin yang sedikit berbeda (misalnya keputusan FMA). Karena itu laporan menyebut eksplisit
"Dawn, FXC shader compiler". Hasil akurasinya sama dengan jalur vgpu (iterasi identik), jadi FXC tidak mengubah kesimpulan.

## 5. Menjalankan di halaman

```ts
async function solveInPage(browser: Browser, port: number, settings: ExperimentSettings): Promise<BrowserOutcome> {
  const page = await browser.newPage();
  page.on("console", (message) => console.error(`[page] ${message.text()}`));
  page.on("pageerror", (error) => console.error(`[page error] ${error.message}`));
  await page.goto(`http://127.0.0.1:${port}/`);
  await page.waitForFunction(() => window.runE1 !== undefined, undefined, { timeout: 30_000 });
  return page.evaluate((input) => {
    const run = window.runE1;
    if (run === undefined) throw new Error("runE1 missing");
    return run(input);
  }, settings);
}
```

- Console halaman diteruskan ke stderr Node, sehingga log browser (seperti peringatan `powerPreference`) terlihat.
- `waitForFunction`: tunggu modul selesai dimuat dan `window.runE1` tersedia.
- `page.evaluate`: jalankan fungsi di halaman; argumen dan hasil **diserialisasi** (structured clone). Karena itu solusi dikirim sebagai
  `number[]`, lalu diubah kembali ke `Float32Array` di Node.

Halaman harus dimuat dari `http://127.0.0.1`, bukan `about:blank`. WebGPU hanya tersedia di **secure context**; probe pertama di
`about:blank` gagal dengan `navigator.gpu` undefined.

## 6. Penjaga dan output

```ts
const outcome = await solveInPage(browser, portOf(server), options.settings);
assertTargetAdapter(options.adapter, outcome.report.adapter);
await writeOutputs(options.caseDir, outcome.report, Float32Array.from(outcome.solution));
```

Adapter dicek **sebelum** menulis apa pun (dokumen 1.21). `finally` menutup browser dan server walau terjadi error.

## 7. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Playwright | mengendalikan Chromium headless dari Node |
| Server lokal | menyajikan TS (di-strip), WGSL, dan data dengan whitelist nama file |
| Secure context | WebGPU butuh http(s) localhost, bukan about:blank |
| DXC vs FXC | Chromium Playwright tanpa dxil.dll → paksa FXC |
| Chrome sistem | memilih iGPU AMD → tidak dipakai |
| `page.evaluate` | hasil diserialisasi; Float32Array dikirim sebagai array angka |
