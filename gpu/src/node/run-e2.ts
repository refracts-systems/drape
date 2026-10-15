// Experiment 2 through vgpu/node: f32 WGSL membrane kernels + the E1 PCG inside a host Newton loop.
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { parseCouponManifest, parseCouponProblem, type CouponProblem } from "../coupon.ts";
import { prepareCouponGpu } from "../coupon-gpu.ts";
import { evaluateKernelState, solvePath, type IterationRecord, type PathResult } from "../newton.ts";
import { loadKernels } from "./files.ts";
import { assertTargetAdapter, hostInfo } from "./hardware.ts";
import { createVgpuBackend, type VgpuAdapter } from "./vgpu-backend.ts";

const ADAPTERS: readonly VgpuAdapter[] = ["auto", "hardware", "software"];
const RECORD_FIELDS = ["step", "iteration", "relativeResidual", "pcgIterations", "pcgStatus", "linearResidual", "stepLength", "halvings", "status"] as const;

function options(): { readonly caseDir: string; readonly adapter: VgpuAdapter } {
  const { values } = parseArgs({ options: { case: { type: "string" }, adapter: { type: "string", default: "hardware" } } });
  const adapter = ADAPTERS.find((name) => name === values.adapter);
  if (values.case === undefined || adapter === undefined) throw new Error(`usage: --case <coupon dir> [--adapter ${ADAPTERS.join("|")}]`);
  return { caseDir: values.case, adapter };
}

async function loadCoupon(caseDir: string): Promise<CouponProblem> {
  const manifest = parseCouponManifest(await readFile(join(caseDir, "problem.json"), "utf8"));
  const bytes = await readFile(join(caseDir, "problem.bin"));
  return parseCouponProblem(manifest, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

const bytesOf = (values: Float32Array<ArrayBuffer> | Float64Array<ArrayBuffer>): Uint8Array => new Uint8Array(values.buffer, values.byteOffset, values.byteLength);

/** gpu-iterations.bin: one row of little-endian f64 per Newton iterate, columns RECORD_FIELDS. */
const recordTable = (records: readonly IterationRecord[]): Float64Array<ArrayBuffer> =>
  Float64Array.from(records.flatMap((entry) => RECORD_FIELDS.map((key) => entry[key])));

function joinIterates(iterates: readonly Float32Array<ArrayBuffer>[]): Float32Array<ArrayBuffer> {
  const joined = new Float32Array(iterates.reduce((length, x) => length + x.length, 0));
  iterates.reduce((offset, x) => (joined.set(x, offset), offset + x.length), 0);
  return joined;
}

async function writeOutputs(caseDir: string, adapter: string, path: PathResult, kernel: Float32Array<ArrayBuffer>): Promise<void> {
  await writeFile(join(caseDir, "gpu-kernel.bin"), bytesOf(kernel));
  await writeFile(join(caseDir, "gpu-iterations.bin"), bytesOf(recordTable(path.records)));
  await writeFile(join(caseDir, "gpu-iterates.bin"), bytesOf(joinIterates(path.iterates)));
  const report = { schema: "e2-gpu/1", backend: "vgpu-node-dawn", adapter, record_fields: RECORD_FIELDS, records: path.records.length, solve_ms: path.solveMs, ...hostInfo() };
  await writeFile(join(caseDir, "gpu-result.json"), `${JSON.stringify(report, null, 2)}\n`);
}

function printSummary(problem: CouponProblem, path: PathResult): void {
  const finals = path.records.filter((entry) => entry.status !== 0);
  const iterations = finals.map((entry) => entry.iteration).join(" ");
  const worst = Math.max(...finals.map((entry) => entry.relativeResidual));
  const failed = finals.filter((entry) => entry.status !== 1).length;
  console.log(`${problem.manifest.case} f32 Newton iterations per step [${iterations}] worst final ${worst.toExponential(2)} unconverged=${failed} ${path.solveMs.toFixed(0)}ms`);
}

async function main(): Promise<void> {
  const { caseDir, adapter } = options();
  const [problem, sources] = await Promise.all([loadCoupon(caseDir), loadKernels()]);
  const backend = await createVgpuBackend(adapter);
  try {
    assertTargetAdapter(adapter, backend.adapter);
    const gpu = await prepareCouponGpu(backend, problem, sources);
    const kernel = await evaluateKernelState(gpu, problem);
    const path = await solvePath(gpu, problem);
    await writeOutputs(caseDir, backend.adapter, path, kernel);
    printSummary(problem, path);
  } finally {
    backend.dispose();
  }
}

await main();
