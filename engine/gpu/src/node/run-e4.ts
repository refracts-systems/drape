// Experiment 4 through vgpu/node: the E3 drape with the robust Newton, one ablation rung per run (A–D).
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { parseDrapeManifest, parseDrapeProblem, type DrapeProblem } from "../drape.ts";
import { prepareDrapeGpu } from "../drape-gpu.ts";
import { evaluateKernelsWithInertia, runRobustProbes, simulateRobust, type RobustRun, type RobustTrajectory } from "../drape-e4.ts";
import type { IterationRecord } from "../newton.ts";
import { parseRobustRules, type RobustRecord, type RobustRules, type RobustSettings } from "../robust-newton.ts";
import { createPhaseClock, timeBackend } from "../timed-backend.ts";
import { loadKernels } from "./files.ts";
import { assertTargetAdapter, hostInfo } from "./hardware.ts";
import { createVgpuBackend, type VgpuAdapter } from "./vgpu-backend.ts";

const RECORD_FIELDS = ["step", "iteration", "relativeResidual", "pcgIterations", "pcgStatus", "linearResidual", "stepLength", "halvings", "status"] as const;
const EXTRA_FIELDS = ["eta", "mu", "updateInf", "stepInf", "retries", "resolution"] as const;

/** A: E3 configuration. B: + floor-aware stop. C: + Eisenstat–Walker forcing. D: + regularised exact tangent. */
const RUNGS = {
  A: { projected: true, settings: { floorStop: false, forcing: false, regularize: false } },
  B: { projected: true, settings: { floorStop: true, forcing: false, regularize: false } },
  C: { projected: true, settings: { floorStop: true, forcing: true, regularize: false } },
  D: { projected: false, settings: { floorStop: true, forcing: true, regularize: true } },
} as const;
type Rung = keyof typeof RUNGS;

function options(): { readonly caseDir: string; readonly rung: Rung } {
  const { values } = parseArgs({ options: { case: { type: "string" }, rung: { type: "string", default: "D" } } });
  const rung = (Object.keys(RUNGS) as Rung[]).find((name) => name === values.rung);
  if (values.case === undefined || rung === undefined) throw new Error("usage: --case <drape dir> [--rung A|B|C|D]");
  return { caseDir: values.case, rung };
}

async function loadDrape(caseDir: string): Promise<{ readonly problem: DrapeProblem; readonly rules: RobustRules }> {
  const text = await readFile(join(caseDir, "problem.json"), "utf8");
  const bytes = await readFile(join(caseDir, "problem.bin"));
  const problem = parseDrapeProblem(parseDrapeManifest(text), bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
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
  readonly probes: RobustRun;
  readonly trajectory: RobustTrajectory;
}

async function writeRun(dir: string, prefix: string, run: RobustRun): Promise<void> {
  await writeFile(join(dir, `gpu-${prefix}iterations.bin`), bytesOf(table<IterationRecord>(run.records, RECORD_FIELDS)));
  await writeFile(join(dir, `gpu-${prefix}robust.bin`), bytesOf(table<RobustRecord>(run.extras, EXTRA_FIELDS)));
}

async function writeOutputs(dir: string, rung: Rung, problem: DrapeProblem, out: Outputs): Promise<void> {
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, "gpu-kernel.bin"), bytesOf(out.kernel));
  await writeFile(join(dir, "gpu-trajectory.bin"), bytesOf(join32(out.trajectory.positions)));
  await writeFile(join(dir, "gpu-probe-iterates.bin"), bytesOf(join32(out.probes.iterates)));
  await writeRun(dir, "", out.trajectory);
  await writeRun(dir, "probe-", out.probes);
  const report = { schema: "e4-gpu/1", backend: "vgpu-node-dawn", adapter: out.adapter, rung, settings: RUNGS[rung], record_fields: RECORD_FIELDS, robust_fields: EXTRA_FIELDS,
    steps: problem.manifest.steps, trajectory_ms: out.trajectory.ms, ms_per_step: out.trajectory.ms / problem.manifest.steps, phase_ms_per_step: out.trajectory.phases, ...hostInfo() };
  await writeFile(join(dir, "gpu-result.json"), `${JSON.stringify(report, null, 2)}\n`);
}

function printSummary(problem: DrapeProblem, rung: Rung, out: Outputs): void {
  const finals = out.trajectory.records.filter((entry) => entry.status !== 0 && Number.isNaN(entry.pcgIterations));
  const solves = out.trajectory.records.filter((entry) => Number.isFinite(entry.pcgIterations));
  const pcg = solves.reduce((sum, entry) => sum + entry.pcgIterations, 0) / Math.max(1, solves.length);
  const phases = Object.entries(out.trajectory.phases).map(([name, ms]) => `${name} ${ms.toFixed(1)}`).join(", ");
  console.log(`${problem.manifest.case} rung ${rung}: ${(solves.length / problem.manifest.steps).toFixed(2)} Newton/step, ${pcg.toFixed(1)} PCG/solve, `
    + `${finals.filter((f) => f.status === 1).length} residual / ${finals.filter((f) => f.status === 4).length} update stops, ${(out.trajectory.ms / problem.manifest.steps).toFixed(1)} ms/step [${phases}]`);
}

async function main(): Promise<void> {
  const { caseDir, rung } = options();
  const [{ problem, rules }, sources] = await Promise.all([loadDrape(caseDir), loadKernels()]);
  const clock = createPhaseClock();
  const inner = await createVgpuBackend("hardware");
  try {
    assertTargetAdapter("hardware", inner.adapter);
    const gpu = await prepareDrapeGpu(timeBackend(inner, clock), problem, sources, RUNGS[rung].projected);
    const settings: RobustSettings = RUNGS[rung].settings;
    const kernel = await evaluateKernelsWithInertia(gpu, problem, rules);
    const probes = await runRobustProbes(gpu, problem, rules, settings, clock);
    const trajectory = await simulateRobust(gpu, problem, rules, settings, clock);
    const out: Outputs = { adapter: inner.adapter, kernel, probes, trajectory };
    await writeOutputs(join(caseDir, `rung-${rung}`), rung, problem, out);
    printSummary(problem, rung, out);
  } finally {
    inner.dispose();
  }
}

await main();
