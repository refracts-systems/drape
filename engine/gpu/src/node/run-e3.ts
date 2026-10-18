// Experiment 3 through vgpu/node: dynamic thin-shell drape, f32 WGSL membrane + hinge + inertia with the E1 PCG.
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { parseDrapeManifest, parseDrapeProblem, type DrapeProblem } from "../drape.ts";
import { prepareDrapeGpu } from "../drape-gpu.ts";
import { evaluateDrapeKernels, runProbes, simulateDrape, type ProbeResult, type TrajectoryResult } from "../drape-step.ts";
import type { IterationRecord } from "../newton.ts";
import { loadKernels } from "./files.ts";
import { assertTargetAdapter, hostInfo } from "./hardware.ts";
import { createVgpuBackend, type VgpuAdapter } from "./vgpu-backend.ts";

const ADAPTERS: readonly VgpuAdapter[] = ["auto", "hardware", "software"];
const RECORD_FIELDS = ["step", "iteration", "relativeResidual", "pcgIterations", "pcgStatus", "linearResidual", "stepLength", "halvings", "status"] as const;

function options(): { readonly caseDir: string; readonly adapter: VgpuAdapter } {
  const { values } = parseArgs({ options: { case: { type: "string" }, adapter: { type: "string", default: "hardware" } } });
  const adapter = ADAPTERS.find((name) => name === values.adapter);
  if (values.case === undefined || adapter === undefined) throw new Error(`usage: --case <drape dir> [--adapter ${ADAPTERS.join("|")}]`);
  return { caseDir: values.case, adapter };
}

async function loadDrape(caseDir: string): Promise<DrapeProblem> {
  const manifest = parseDrapeManifest(await readFile(join(caseDir, "problem.json"), "utf8"));
  const bytes = await readFile(join(caseDir, "problem.bin"));
  return parseDrapeProblem(manifest, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

const bytesOf = (values: Float32Array<ArrayBuffer> | Float64Array<ArrayBuffer>): Uint8Array => new Uint8Array(values.buffer, values.byteOffset, values.byteLength);

const recordTable = (records: readonly IterationRecord[]): Float64Array<ArrayBuffer> =>
  Float64Array.from(records.flatMap((entry) => RECORD_FIELDS.map((key) => entry[key])));

function join32(parts: readonly Float32Array<ArrayBuffer>[]): Float32Array<ArrayBuffer> {
  const joined = new Float32Array(parts.reduce((length, x) => length + x.length, 0));
  parts.reduce((offset, x) => (joined.set(x, offset), offset + x.length), 0);
  return joined;
}

interface Outputs {
  readonly adapter: string;
  readonly kernel: Float32Array<ArrayBuffer>;
  readonly trajectory: TrajectoryResult;
  readonly probes: ProbeResult;
}

async function writeOutputs(caseDir: string, problem: DrapeProblem, out: Outputs): Promise<void> {
  const write = (name: string, data: Uint8Array): Promise<void> => writeFile(join(caseDir, name), data);
  await write("gpu-kernel.bin", bytesOf(out.kernel));
  await write("gpu-trajectory.bin", bytesOf(join32(out.trajectory.positions)));
  await write("gpu-iterations.bin", bytesOf(recordTable(out.trajectory.records)));
  await write("gpu-probe-iterations.bin", bytesOf(recordTable(out.probes.records)));
  await write("gpu-probe-iterates.bin", bytesOf(join32(out.probes.iterates)));
  const report = { schema: "e3-gpu/1", backend: "vgpu-node-dawn", adapter: out.adapter, tangent: problem.manifest.tangent, record_fields: RECORD_FIELDS,
    steps: problem.manifest.steps, trajectory_ms: out.trajectory.ms, ms_per_step: out.trajectory.ms / problem.manifest.steps, ...hostInfo() };
  await writeFile(join(caseDir, "gpu-result.json"), `${JSON.stringify(report, null, 2)}\n`);
}

function printSummary(problem: DrapeProblem, out: Outputs): void {
  const finals = out.trajectory.records.filter((entry) => entry.status !== 0);
  const unconverged = finals.filter((entry) => entry.status !== 1).length;
  const newton = finals.reduce((sum, entry) => sum + entry.iteration, 0);
  const worst = Math.max(...finals.map((entry) => entry.relativeResidual));
  console.log(`${problem.manifest.case} f32 ${problem.manifest.tangent}: ${newton} Newton iterations over ${finals.length} steps, ${unconverged} not at 1e-6, `
    + `worst final ${worst.toExponential(2)}, ${(out.trajectory.ms / problem.manifest.steps).toFixed(1)} ms/step`);
}

async function main(): Promise<void> {
  const { caseDir, adapter } = options();
  const [problem, sources] = await Promise.all([loadDrape(caseDir), loadKernels()]);
  const backend = await createVgpuBackend(adapter);
  try {
    assertTargetAdapter(adapter, backend.adapter);
    const gpu = await prepareDrapeGpu(backend, problem, sources);
    const kernel = await evaluateDrapeKernels(gpu, problem);
    const probes = await runProbes(gpu, problem);
    const trajectory = await simulateDrape(gpu, problem);
    const out: Outputs = { adapter: backend.adapter, kernel, trajectory, probes };
    await writeOutputs(caseDir, problem, out);
    printSummary(problem, out);
  } finally {
    backend.dispose();
  }
}

await main();
