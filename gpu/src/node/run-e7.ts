// Experiment 7 through vgpu/node: quasi-static drape by pseudo-transient continuation, cold / warm / prolonged start. Outputs go to
// <case>/gpu-<mode>/. Rung D of Experiment 6 (GPU reductions and GPU minima) is the baseline solver.
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { evaluateContactKernel } from "../contact-host.ts";
import { parseDrapeManifest, parseDrapeProblem, type DrapeProblem } from "../drape.ts";
import { evaluateKernelsWithInertia } from "../drape-e4.ts";
import { REJECTION, evaluateInertiaAtSteps, runPtcProbes, solveEquilibrium, type EquilibriumRun, type PtcProbes, type PtcStepLog, type PtcStart } from "../drape-e7.ts";
import { prepareDrapeGpu } from "../drape-gpu.ts";
import { capFor, parsePtcRules, type PtcRules } from "../ptc.ts";
import { slice } from "../problem.ts";
import { parseRobustRules, type RobustRules, type RobustSettings } from "../robust-newton.ts";
import { createPhaseClock, timeBackend } from "../timed-backend.ts";
import { loadKernels } from "./files.ts";
import { assertTargetAdapter, hostInfo } from "./hardware.ts";
import { bytesOf, EXTRA_FIELDS, join32, RECORD_FIELDS, table, trialTable, writeRun } from "./outputs.ts";
import { readState, writeState } from "./state-files.ts";
import { createVgpuBackend } from "./vgpu-backend.ts";

/** E6 rung D: floor stop, forcing, regularisation, noise-aware search (inert), GPU reductions and GPU minima. */
const SETTINGS: RobustSettings = { floorStop: true, forcing: true, regularize: true, noiseAware: true, gpuReductions: true, fewReadbacks: true };
const MODES = ["cold", "warm", "prolong"] as const;
type Mode = (typeof MODES)[number];
/** Step sizes of the inertia-kernel check. */
const CHECK_STEPS = [1 / 240, 0.1, 1.0] as const;
/** All accepted states are kept up to this mesh size; larger meshes keep the first and the last. */
const KEEP_STATES_COLUMNS = 65;

interface Options {
  readonly caseDir: string;
  readonly mode: Mode;
  readonly start: string | null;
  readonly checks: boolean;
}

function options(): Options {
  const { values } = parseArgs({ options: { case: { type: "string" }, mode: { type: "string", default: "cold" }, start: { type: "string" }, checks: { type: "boolean", default: false } } });
  const mode = MODES.find((name) => name === values.mode);
  if (values.case === undefined || mode === undefined) throw new Error("usage: --case <case dir> [--mode cold|warm|prolong] [--start <state prefix>] [--checks]");
  if (mode !== "cold" && values.start === undefined) throw new Error(`mode ${mode} needs --start <state prefix>`);
  return { caseDir: values.case, mode, start: values.start ?? null, checks: values.checks };
}

interface Loaded {
  readonly problem: DrapeProblem;
  readonly rules: RobustRules;
  readonly ptc: PtcRules;
}

async function load(dir: string): Promise<Loaded> {
  const text = await readFile(join(dir, "problem.json"), "utf8");
  const bytes = await readFile(join(dir, "problem.bin"));
  const problem = parseDrapeProblem(parseDrapeManifest(text), bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  if (problem.contact.count === 0 || problem.unitMass === null) throw new Error(`${dir}: not an Experiment 7 problem (colliders and row_unit_mass are required)`);
  return { problem, rules: parseRobustRules(text), ptc: parsePtcRules(text) };
}

async function startOf(opts: Options, loaded: Loaded): Promise<PtcStart> {
  if (opts.start === null) return { positions: Float32Array.from(loaded.problem.f32.rest_positions), h: loaded.ptc.h0 };
  const state = await readState(opts.start);
  return { positions: state.positions, h: state.finalH };
}

/** probes.json / probes.bin of the oracle (cold base solves of n17 and n33): the f32 start state and h of each probed step. */
async function loadProbes(dir: string, vertices: number): Promise<PtcProbes | null> {
  try {
    const manifest: unknown = JSON.parse(await readFile(join(dir, "probes.json"), "utf8"));
    const bytes = await readFile(join(dir, "probes.bin"));
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const arrays = (manifest as { readonly arrays: Parameters<typeof slice>[0]["arrays"] }).arrays;
    const steps = Array.from(new Uint32Array(slice({ arrays }, buffer, "probe_steps", "u32")));
    const h = Array.from(new Float32Array(slice({ arrays }, buffer, "probe_h", "f32")));
    const all = new Float32Array(slice({ arrays }, buffer, "probe_positions", "f32"));
    return { steps, h, positions: steps.map((_, i) => Float32Array.from(all.subarray(3 * vertices * i, 3 * vertices * (i + 1)))) };
  } catch {
    return null;
  }
}

interface Checks {
  readonly kernel: Float32Array<ArrayBuffer>;
  readonly contactKernel: Float32Array<ArrayBuffer>;
  readonly inertia: Float32Array<ArrayBuffer>;
}

async function writeChecks(dir: string, checks: Checks): Promise<void> {
  await writeFile(join(dir, "gpu-kernel.bin"), bytesOf(checks.kernel));
  await writeFile(join(dir, "gpu-contact-kernel.bin"), bytesOf(checks.contactKernel));
  await writeFile(join(dir, "gpu-inertia-h.bin"), bytesOf(checks.inertia));
}

async function writeSolve(dir: string, mode: Mode, loaded: Loaded, adapter: string, result: EquilibriumRun): Promise<void> {
  const { steps, attempts, run } = result;
  await writeRun(dir, "", run);
  await writeFile(join(dir, "gpu-ptc.bin"), bytesOf(table(steps, ["step", "h", "residual", "inner", "rejected", "pcgIterations", "minDistance"])));
  await writeFile(join(dir, "gpu-attempts.bin"), bytesOf(table(attempts, ["step", "attempt", "h", "reason", "recordStart", "recordEnd"])));
  await writeFile(join(dir, "gpu-states.bin"), bytesOf(join32(result.states)));
  await writeState(join(dir, "gpu-equilibrium"), result.states[result.states.length - 1], result.finalH);
  const updates = run.records.filter((entry) => Number.isFinite(entry.pcgIterations));
  const report = { schema: "e7-gpu/1", backend: "vgpu-node-dawn", adapter, mode, settings: SETTINGS, record_fields: RECORD_FIELDS, robust_fields: EXTRA_FIELDS,
    grid_columns: loaded.problem.manifest.grid_columns, stop: result.stop, cap: capFor(loaded.problem.manifest.grid_columns), accepted_steps: steps.length,
    rejected_attempts: attempts.filter((a) => a.reason !== 0 && a.reason !== REJECTION.curvature_retry).length, curvature_retry_attempts: attempts.filter((a) => a.reason === REJECTION.curvature_retry).length, curvature_dropped_steps: steps.filter((entry) => entry.curvatureDropped).length, newton_iterations_total: updates.length, pcg_iterations_total: updates.reduce((sum, e) => sum + e.pcgIterations, 0),
    final_h: result.finalH, final_residual: result.finalResidual, final_relative_residual: result.finalResidual / loaded.ptc.gravity_load,
    trajectory_ms: result.ms, ms_per_step: result.ms / Math.max(1, steps.length), phase_ms_per_step: result.phases, read_bytes_total: result.readBytes, ...hostInfo() };
  await writeFile(join(dir, "gpu-result.json"), `${JSON.stringify(report, null, 2)}\n`);
}

function printSummary(loaded: Loaded, mode: Mode, result: EquilibriumRun): void {
  const phases = Object.entries(result.phases).map(([name, ms]) => `${name} ${(ms * Math.max(1, result.steps.length) / 1000).toFixed(2)}s`).join(", ");
  const rejected = result.attempts.filter((a) => a.reason !== 0 && a.reason !== REJECTION.curvature_retry).length;
  console.log(`${loaded.problem.manifest.case} ${mode}: ${result.steps.length} steps (+${rejected} rejected), stop ${result.stop}, r/L ${(result.finalResidual / loaded.ptc.gravity_load).toExponential(2)}, `
    + `${(result.ms / 1000).toFixed(2)} s [${phases}], h next ${result.finalH.toExponential(2)}`);
}

async function main(): Promise<void> {
  const opts = options();
  const [loaded, sources] = await Promise.all([load(opts.caseDir), loadKernels()]);
  const clock = createPhaseClock();
  const inner = await createVgpuBackend("hardware");
  try {
    assertTargetAdapter("hardware", inner.adapter);
    const gpu = await prepareDrapeGpu(timeBackend(inner, clock), loaded.problem, sources, false, true);
    const dir = join(opts.caseDir, `gpu-${opts.mode}`);
    await mkdir(dir, { recursive: true });
    const { problem, rules, ptc } = loaded;
    if (opts.checks) await writeChecks(dir, await runChecks(gpu, loaded));
    const probes = opts.checks ? await loadProbes(opts.caseDir, problem.manifest.vertices) : null;
    if (probes !== null) await writeProbes(dir, await runPtcProbes(gpu, problem, rules, SETTINGS, ptc, clock, probes));
    const progress = process.env.E7_PROGRESS === undefined ? null : (log: PtcStepLog, ms: number): void => console.error(`step ${log.step} h ${log.h.toExponential(2)} r ${log.residual.toExponential(2)} inner ${log.inner} rej ${log.rejected} pcg ${log.pcgIterations} ${(ms / 1000).toFixed(1)} s`);
    const budget = process.env.E7_BUDGET_S === undefined ? null : 1000 * Number(process.env.E7_BUDGET_S);
    const result = await solveEquilibrium(gpu, problem, rules, SETTINGS, ptc, clock, await startOf(opts, loaded), capFor(problem.manifest.grid_columns), problem.manifest.grid_columns <= KEEP_STATES_COLUMNS, { onStep: progress, budgetMs: budget });
    await writeSolve(dir, opts.mode, loaded, inner.adapter, result);
    printSummary(loaded, opts.mode, result);
  } finally {
    inner.dispose();
  }
}

async function runChecks(gpu: Parameters<typeof evaluateKernelsWithInertia>[0], loaded: Loaded): Promise<Checks> {
  const { triangles: t, hinges: h } = loaded.problem.manifest;
  const kernel = await evaluateKernelsWithInertia(gpu, loaded.problem, loaded.rules);
  const contactKernel = await evaluateContactKernel(gpu, loaded.problem, { grad: 9 * t + 12 * h, hess: 81 * t + 144 * h });
  return { kernel, contactKernel, inertia: await evaluateInertiaAtSteps(gpu, loaded.problem, CHECK_STEPS) };
}

async function writeProbes(dir: string, run: Awaited<ReturnType<typeof runPtcProbes>>): Promise<void> {
  await writeRun(dir, "probe-", run);
  await writeFile(join(dir, "gpu-probe-iterates.bin"), bytesOf(join32(run.iterates)));
  await writeFile(join(dir, "gpu-probe-directions.bin"), bytesOf(join32(run.directions)));
  await writeFile(join(dir, "gpu-probe-search.bin"), bytesOf(trialTable(run.trials)));
}

await main();
