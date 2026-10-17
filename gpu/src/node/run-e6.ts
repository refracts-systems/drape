// Experiment 6 through vgpu/node: the E5 scene with one rung of the ablation ladder (A: E5, B: noise-aware Armijo,
// C: GPU reductions, D: GPU-side minima). Outputs go to <case>/e6-rung-<X>/.
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { evaluateContactKernel } from "../contact-host.ts";
import { parseDrapeManifest, parseDrapeProblem, type DrapeProblem } from "../drape.ts";
import { prepareDrapeGpu } from "../drape-gpu.ts";
import { evaluateKernelsWithInertia, runRobustProbes, simulateRobust, type RobustRun, type RobustTrajectory } from "../drape-e4.ts";
import type { IterationRecord } from "../newton.ts";
import { parseRobustRules, type RobustRecord, type RobustRules, type RobustSettings, type TrialRecord } from "../robust-newton.ts";
import { createPhaseClock, timeBackend } from "../timed-backend.ts";
import { loadKernels } from "./files.ts";
import { assertTargetAdapter, hostInfo } from "./hardware.ts";
import { createVgpuBackend } from "./vgpu-backend.ts";

const RECORD_FIELDS = ["step", "iteration", "relativeResidual", "pcgIterations", "pcgStatus", "linearResidual", "stepLength", "halvings", "status"] as const;
const EXTRA_FIELDS = ["eta", "mu", "updateInf", "stepInf", "retries", "resolution", "alphaMax", "limitingVertex", "minDistance",
  "deltaSum", "deltaAbs", "noiseOnly", "slope", "slopeExact", "normSquared", "normSquaredExact"] as const;
const TRIAL_FIELDS = ["step", "iteration", "halving", "alpha", "sum", "abs", "slope", "accepted"] as const;

/** E5 rung D (floor stop, forcing, regularisation) plus the Experiment 6 switches. */
const BASE: RobustSettings = { floorStop: true, forcing: true, regularize: true };
const RUNGS = {
  A: { ...BASE, noiseAware: false, gpuReductions: false, fewReadbacks: false },
  B: { ...BASE, noiseAware: true, gpuReductions: false, fewReadbacks: false },
  C: { ...BASE, noiseAware: true, gpuReductions: true, fewReadbacks: false },
  D: { ...BASE, noiseAware: true, gpuReductions: true, fewReadbacks: true },
} as const;
type Rung = keyof typeof RUNGS;

function options(): { readonly caseDir: string; readonly rung: Rung } {
  const { values } = parseArgs({ options: { case: { type: "string" }, rung: { type: "string", default: "A" } } });
  const rung = (Object.keys(RUNGS) as Rung[]).find((name) => name === values.rung);
  if (values.case === undefined || rung === undefined) throw new Error("usage: --case <drape5 case dir> [--rung A|B|C|D]");
  return { caseDir: values.case, rung };
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

const newtonIterations = (run: RobustRun): number => run.records.filter((entry) => Number.isFinite(entry.pcgIterations)).length;

async function writeOutputs(dir: string, rung: Rung, problem: DrapeProblem, out: Outputs): Promise<void> {
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, "gpu-kernel.bin"), bytesOf(out.kernel));
  await writeFile(join(dir, "gpu-contact-kernel.bin"), bytesOf(out.contactKernel));
  await writeFile(join(dir, "gpu-trajectory.bin"), bytesOf(join32(out.trajectory.positions)));
  await writeFile(join(dir, "gpu-probe-iterates.bin"), bytesOf(join32(out.probes.iterates)));
  await writeFile(join(dir, "gpu-probe-directions.bin"), bytesOf(join32(out.probes.directions)));
  await writeFile(join(dir, "gpu-probe-search.bin"), bytesOf(table<TrialRecord & { readonly accepted: boolean }>(out.probes.trials, TRIAL_FIELDS)));
  await writeFile(join(dir, "gpu-noise-states.bin"), bytesOf(join32([...out.probes.noiseStates, ...out.trajectory.noiseStates])));
  await writeRun(dir, "", out.trajectory);
  await writeRun(dir, "probe-", out.probes);
  const newton = newtonIterations(out.trajectory);
  const report = { schema: "e6-gpu/1", backend: "vgpu-node-dawn", adapter: out.adapter, rung, settings: RUNGS[rung], record_fields: RECORD_FIELDS, robust_fields: EXTRA_FIELDS,
    steps: problem.manifest.steps, trajectory_ms: out.trajectory.ms, ms_per_step: out.trajectory.ms / problem.manifest.steps, phase_ms_per_step: out.trajectory.phases,
    read_bytes_total: out.trajectory.readBytes, newton_iterations_total: newton, read_bytes_per_newton_iteration: out.trajectory.readBytes / newton, ...hostInfo() };
  await writeFile(join(dir, "gpu-result.json"), `${JSON.stringify(report, null, 2)}\n`);
}

function printSummary(problem: DrapeProblem, rung: Rung, out: Outputs): void {
  const records = out.trajectory.records;
  const solves = records.filter((entry) => Number.isFinite(entry.pcgIterations));
  const pcg = solves.reduce((sum, entry) => sum + entry.pcgIterations, 0) / Math.max(1, solves.length);
  const finals = records.filter((entry) => entry.status !== 0 && (Number.isNaN(entry.pcgIterations) || entry.status === 3));
  const count = (status: number): number => finals.filter((f) => f.status === status).length;
  const noise = out.trajectory.extras.filter((extra) => extra.noiseOnly === 1).length;
  const phases = Object.entries(out.trajectory.phases).map(([name, ms]) => `${name} ${ms.toFixed(1)}`).join(", ");
  console.log(`${problem.manifest.case} rung ${rung}: ${(solves.length / problem.manifest.steps).toFixed(2)} Newton/step, ${pcg.toFixed(1)} PCG/solve, `
    + `${count(1)} residual / ${count(4)} update / ${count(2)} limit / ${count(3)} line-search stops, ${noise} noise-accepted, `
    + `${(out.trajectory.ms / problem.manifest.steps).toFixed(1)} ms/step [${phases}], ${(out.trajectory.readBytes / solves.length / 1024).toFixed(1)} KiB read/iteration`);
}

async function main(): Promise<void> {
  const { caseDir, rung } = options();
  const [{ problem, rules }, sources] = await Promise.all([loadDrape(caseDir), loadKernels()]);
  const clock = createPhaseClock();
  const inner = await createVgpuBackend("hardware");
  try {
    assertTargetAdapter("hardware", inner.adapter);
    const gpu = await prepareDrapeGpu(timeBackend(inner, clock), problem, sources, false, true);
    const kernel = await evaluateKernelsWithInertia(gpu, problem, rules);
    const { triangles: t, hinges: h } = problem.manifest;
    const contactKernel = await evaluateContactKernel(gpu, problem, { grad: 9 * t + 12 * h, hess: 81 * t + 144 * h });
    const probes = await runRobustProbes(gpu, problem, rules, RUNGS[rung], clock);
    const trajectory = await simulateRobust(gpu, problem, rules, RUNGS[rung], clock);
    const out: Outputs = { adapter: inner.adapter, kernel, contactKernel, probes, trajectory };
    await writeOutputs(join(caseDir, `e6-rung-${rung}`), rung, problem, out);
    printSummary(problem, rung, out);
  } finally {
    inner.dispose();
  }
}

await main();
