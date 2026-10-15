import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { GpuReport } from "../experiment.ts";
import { assembleKernels, type KernelSources } from "../kernels.ts";
import { parseManifest, parseProblem, type Problem } from "../problem.ts";

export const KERNEL_DIR = fileURLToPath(new URL("../../kernels/", import.meta.url));
export const SOURCE_DIR = fileURLToPath(new URL("../", import.meta.url));

export async function loadProblem(caseDir: string): Promise<Problem> {
  const manifest = parseManifest(await readFile(join(caseDir, "problem.json"), "utf8"));
  const bytes = await readFile(join(caseDir, "problem.bin"));
  return parseProblem(manifest, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

export const loadKernels = (): Promise<KernelSources> =>
  assembleKernels((file) => readFile(join(KERNEL_DIR, file), "utf8"));

/** gpu-x.bin (little-endian f32, consumed by `oracle compare`) and gpu-result.json. */
export async function writeOutputs(caseDir: string, report: GpuReport, solution: Float32Array<ArrayBuffer>): Promise<void> {
  await writeFile(join(caseDir, "gpu-x.bin"), new Uint8Array(solution.buffer, solution.byteOffset, solution.byteLength));
  await writeFile(join(caseDir, "gpu-result.json"), `${JSON.stringify(report, null, 2)}\n`);
}

export function printSummary(report: GpuReport): void {
  const { timing } = report;
  const gpu = timing.gpu_ms === null ? "n/a" : timing.gpu_ms.toFixed(1);
  console.log(`${report.case}-n${report.grid_columns} N=${report.unknowns} ${report.status} it=${report.iterations} `
    + `rec=${report.recursive_relative_residual.toExponential(2)} true32=${report.gpu_f32_true_relative_residual.toExponential(2)} `
    + `warm=${timing.warm_solve_ms.toFixed(1)}ms gpu=${gpu}ms total=${timing.total_ms.toFixed(1)}ms det=${report.deterministic_repeat} [${report.adapter}]`);
}
