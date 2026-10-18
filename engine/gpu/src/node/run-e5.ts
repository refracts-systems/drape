// Experiment 5 through vgpu/node: the free sheet falling onto a sphere resting on a floor, E4-final robust Newton (rung D)
// with the IPC barrier of the analytic colliders.
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { evaluateContactKernel } from "../contact-host.ts";
import { parseDrapeManifest, parseDrapeProblem, type DrapeProblem } from "../drape.ts";
import { prepareDrapeGpu } from "../drape-gpu.ts";
import { evaluateKernelsWithInertia, runRobustProbes, simulateRobust, type RobustRun, type RobustTrajectory } from "../drape-e4.ts";
import type { IterationRecord } from "../newton.ts";
import { parseRobustRules, type RobustRecord, type RobustRules, type RobustSettings } from "../robust-newton.ts";
import { createPhaseClock, timeBackend } from "../timed-backend.ts";
import { loadKernels } from "./files.ts";
import { assertTargetAdapter, hostInfo } from "./hardware.ts";
import { createVgpuBackend } from "./vgpu-backend.ts";

const RECORD_FIELDS = ["step", "iteration", "relativeResidual", "pcgIterations", "pcgStatus", "linearResidual", "stepLength", "halvings", "status"] as const;
const EXTRA_FIELDS = ["eta", "mu", "updateInf", "stepInf", "retries", "resolution", "alphaMax", "limitingVertex", "minDistance"] as const;
/** E4-final (rung D): floor-aware stop with the f32-resolution floor, Eisenstat–Walker forcing, regularised exact tangent. */
const SETTINGS: RobustSettings = { floorStop: true, forcing: true, regularize: true };

function caseDir(): string {
  const { values } = parseArgs({ options: { case: { type: "string" } } });
  if (values.case === undefined) throw new Error("usage: --case <drape5 case dir>");
  return values.case;
}

async function loadDrape(dir: string): Promise<{ readonly problem: DrapeProblem; readonly rules: RobustRules }> {
  const text = await readFile(join(dir, "problem.json"), "utf8");
  const bytes = await readFile(join(dir, "problem.bin"));
  const problem = parseDrapeProblem(parseDrapeManifest(text), bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  if (problem.contact.count === 0) throw new Error(`${dir}: problem.bin has no colliders (not an Experiment 5 problem)`);
  return { problem, rules: parseRobustRules(text) };
}

const bytesOf = (values: Float32Array<ArrayBuffer> | Float64Array<ArrayBuffer>): Uint8Array => new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
const table = <T>(rows: readonly T[], fields: readonly (keyof T)[]): Float64Array<ArrayBuffer> =>
  Float64Array.from(rows.flatMap((row) => fields.map((field) => Number(row[field]))));

function join32(parts: readonly Float32Array<ArrayBuffer>[]): Float32Array<ArrayBuffer> {
  const joined = new Float32Array(parts.reduce((length, x) => length + x.length, 0));
  parts.reduce((offset, x) => (joined.set(x, offset), offset + x.length), 0);
  return joined;
}

interface Outputs {
  readonly adapter: string;
  readonly kernel: Float32Array<ArrayBuffer>;
  readonly contactKernel: Float32Array<ArrayBuffer>;
  readonly probes: RobustRun;
  readonly trajectory: RobustTrajectory;
}

async function writeRun(dir: string, prefix: string, run: RobustRun): Promise<void> {
  await writeFile(join(dir, `gpu-${prefix}iterations.bin`), bytesOf(table<IterationRecord>(run.records, RECORD_FIELDS)));
  await writeFile(join(dir, `gpu-${prefix}robust.bin`), bytesOf(table<RobustRecord>(run.extras, EXTRA_FIELDS)));
}

async function writeOutputs(dir: string, problem: DrapeProblem, out: Outputs): Promise<void> {
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, "gpu-kernel.bin"), bytesOf(out.kernel));
  await writeFile(join(dir, "gpu-contact-kernel.bin"), bytesOf(out.contactKernel));
  await writeFile(join(dir, "gpu-trajectory.bin"), bytesOf(join32(out.trajectory.positions)));
  await writeFile(join(dir, "gpu-probe-iterates.bin"), bytesOf(join32(out.probes.iterates)));
  await writeFile(join(dir, "gpu-probe-directions.bin"), bytesOf(join32(out.probes.directions)));
  await writeRun(dir, "", out.trajectory);
  await writeRun(dir, "probe-", out.probes);
  const report = { schema: "e5-gpu/1", backend: "vgpu-node-dawn", adapter: out.adapter, rung: "D", settings: SETTINGS, record_fields: RECORD_FIELDS, robust_fields: EXTRA_FIELDS,
    steps: problem.manifest.steps, trajectory_ms: out.trajectory.ms, ms_per_step: out.trajectory.ms / problem.manifest.steps, phase_ms_per_step: out.trajectory.phases, ...hostInfo() };
  await writeFile(join(dir, "gpu-result.json"), `${JSON.stringify(report, null, 2)}\n`);
}

function printSummary(problem: DrapeProblem, out: Outputs): void {
  const records = out.trajectory.records;
  const solves = records.filter((entry) => Number.isFinite(entry.pcgIterations));
  const pcg = solves.reduce((sum, entry) => sum + entry.pcgIterations, 0) / Math.max(1, solves.length);
  const finals = records.filter((entry) => entry.status !== 0 && Number.isNaN(entry.pcgIterations));
  const nearest = out.trajectory.extras.reduce((m, extra) => (Number.isFinite(extra.minDistance) ? Math.min(m, extra.minDistance) : m), Number.POSITIVE_INFINITY);
  const phases = Object.entries(out.trajectory.phases).map(([name, ms]) => `${name} ${ms.toFixed(1)}`).join(", ");
  console.log(`${problem.manifest.case}: ${(solves.length / problem.manifest.steps).toFixed(2)} Newton/step, ${pcg.toFixed(1)} PCG/solve, `
    + `${finals.filter((f) => f.status === 1).length} residual / ${finals.filter((f) => f.status === 4).length} update / ${finals.filter((f) => f.status === 2).length} limit stops, `
    + `min d ${nearest.toExponential(3)} m, ${(out.trajectory.ms / problem.manifest.steps).toFixed(1)} ms/step [${phases}]`);
}

async function main(): Promise<void> {
  const dir = caseDir();
  const [{ problem, rules }, sources] = await Promise.all([loadDrape(dir), loadKernels()]);
  const clock = createPhaseClock();
  const inner = await createVgpuBackend("hardware");
  try {
    assertTargetAdapter("hardware", inner.adapter);
    const gpu = await prepareDrapeGpu(timeBackend(inner, clock), problem, sources, false);
    const kernel = await evaluateKernelsWithInertia(gpu, problem, rules);
    const { triangles: t, hinges: h } = problem.manifest;
    const contactKernel = await evaluateContactKernel(gpu, problem, { grad: 9 * t + 12 * h, hess: 81 * t + 144 * h });
    const probes = await runRobustProbes(gpu, problem, rules, SETTINGS, clock);
    const trajectory = await simulateRobust(gpu, problem, rules, SETTINGS, clock);
    const out: Outputs = { adapter: inner.adapter, kernel, contactKernel, probes, trajectory };
    await writeOutputs(join(dir, "gpu"), problem, out);
    printSummary(problem, out);
  } finally {
    inner.dispose();
  }
}

await main();
