# Step 1.21: `gpu/src/node/hardware.ts`, Penjaga Adapter dan Info Host

> File baru di sesi kerja. Tugasnya sederhana tetapi kritis untuk integritas eksperimen: **jangan pernah mencatat hasil dari GPU yang salah**.
> Laptop ini punya dua GPU, dan angka dari GPU yang salah tidak boleh tercatat sebagai bukti hardware.

---

## 1. Masalahnya

| GPU | Driver | Status |
|---|---|---|
| AMD Radeon(TM) Graphics (iGPU) | 31.0.21925.1001 | **tidak boleh** dipakai |
| NVIDIA GeForce RTX 3050 Laptop GPU | 32.0.16.1714 | target |
| SwiftShader / WARP / llvmpipe (software) | – | **tidak boleh** (CPU yang berpura-pura GPU) |

Browser dan library memilih GPU berdasarkan pengaturan OS dan heuristik sendiri. Di sesi kerja terbukti:

- Chromium milik Playwright memilih RTX 3050 (pengaturan Windows sudah "High performance").
- **Chrome 155 terpasang di sistem memilih iGPU AMD.**

Kalau runner diam-diam memakai iGPU, semua angka benchmark jadi tidak berarti. Pengguna juga meminta: *"Do not silently fall back to
SwiftShader/software adapters or to the AMD iGPU."*

## 2. Isi file

```ts
export interface HostInfo {
  readonly host_os: string;
  readonly host_gpus: string;
}

const SOFTWARE_ADAPTER = /swiftshader|llvmpipe|lavapipe|software|microsoft basic|warp/i;
/** The AMD iGPU on the measurement laptop; vgpu reports only the D3D12 driver version, so match that too. */
const INTEGRATED_ADAPTER = /\bamd\b|radeon|31\.0\.21925\.1001/i;
const VIDEO_CONTROLLERS = "Get-CimInstance Win32_VideoController | ForEach-Object { \"$($_.Name) $($_.DriverVersion)\" }";

function videoControllers(): string {
  if (process.platform !== "win32") return "unavailable (not Windows)";
  const output = execFileSync("powershell.exe", ["-NoProfile", "-Command", VIDEO_CONTROLLERS], { encoding: "utf8" });
  return output.split(/\r?\n/).filter((line) => line.trim().length > 0).join("; ");
}

export const hostInfo = (): HostInfo => ({ host_os: `${version()} ${release()}`, host_gpus: videoControllers() });

/** Timings from a CPU rasterizer or the integrated GPU must never be read as discrete-GPU results. */
export function assertTargetAdapter(requested: string, adapter: string): void {
  if (requested === "software") return;
  if (SOFTWARE_ADAPTER.test(adapter) || INTEGRATED_ADAPTER.test(adapter)) {
    throw new Error(`asked for the hardware adapter but got '${adapter}' (software or integrated GPU); refusing to record it`);
  }
}
```

## 3. Daftar tolak (deny-list)

Dua regex:

- **Software**: nama renderer CPU yang dikenal (SwiftShader milik Chrome, WARP milik Microsoft, llvmpipe/lavapipe di Linux).
- **iGPU**: kata `amd`, `radeon`, **atau versi driver AMD**. Pola versi driver penting karena vgpu hanya melaporkan
  `D3D12 driver version 31.0.21925.1001`, tanpa nama GPU. Tanpa pola ini, iGPU akan lolos di jalur vgpu.

`\b` di `\bamd\b` = batas kata, supaya kata lain yang mengandung "amd" tidak ikut cocok.

`requested === "software"`: kalau pengguna **sengaja** meminta software adapter (misalnya di mesin tanpa GPU), penjaga tidak ikut campur.
Hasil seperti itu diberi label jelas (`experiment1-webgpu-software.json`), dan README mencatat bahwa timing-nya tidak valid.

## 4. Kapan dipanggil

- `run-browser.ts`: setelah solve di halaman selesai, sebelum hasil ditulis.
- `run-vgpu.ts` dan `run-e2.ts`: langsung setelah `init`, sebelum solve.

Kalau gagal, runner melempar error dan tidak ada file hasil yang ditulis. `run-e1.sh` memakai `set -e`, jadi seluruh skrip berhenti.

## 5. Info host di setiap hasil

`hostInfo()` menjalankan perintah PowerShell yang diminta pengguna:

```powershell
Get-CimInstance Win32_VideoController | Select Name, DriverVersion
```

Hasilnya digabung menjadi satu string dan dicatat di setiap `gpu-result.json`, lalu disalin ke `result.json`:

```json
"host_os": "Windows 11 Home Single Language 10.0.26300",
"host_gpus": "AMD Radeon(TM) Graphics 31.0.21925.1001; NVIDIA GeForce RTX 3050 Laptop GPU 32.0.16.1714"
```

Dengan string adapter, setiap hasil bisa diaudit belakangan: GPU apa yang ada di mesin, dan GPU mana yang dipakai.

`execFileSync` (bukan `exec`) dipakai dengan argumen array, jadi tidak ada shell injection; perintahnya juga konstanta.

## 6. Prinsip

> Lebih baik eksperimen **berhenti dengan error** daripada **berhasil dengan data yang salah**.

Ini sama dengan prinsip di backend: validasi konfigurasi saat startup dan gagal keras, daripada melayani request dengan konfigurasi salah.

## 7. Ringkasan

| Konsep | Satu kalimat |
|---|---|
| Deny-list | software renderer + iGPU AMD (nama atau versi driver) |
| Versi driver | satu-satunya identitas GPU yang diberikan vgpu |
| Opt-in software | `--adapter software` dilewati penjaga, diberi label |
| Audit | setiap hasil mencatat OS, semua GPU + driver, dan adapter yang dipakai |
