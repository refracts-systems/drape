import { execFileSync } from "node:child_process";
import { release, version } from "node:os";

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
