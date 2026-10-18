import type { DrapeProblem } from "./drape.ts";
import { context, newtonStart, phaseMeans, resetClock, type RobustRun } from "./drape-e4.ts";
import type { DrapeGpu } from "./drape-gpu.ts";
import { STEP_STATUS, type IterationRecord } from "./newton.ts";
import { nextStepSize, stopRule, type PtcRules, type StopRule } from "./ptc.ts";
import { solveStepRobust, type RobustContext, type RobustRules, type RobustSettings } from "./robust-newton.ts";
import { setBarrierCurvature, withoutColliders } from "./contact-host.ts";
import type { Phase, PhaseClock } from "./timed-backend.ts";

/** Why an attempt was rejected (0 = accepted): the two failures of rule 0.5, a non-finite state, and the superseded run of the μ-cap curvature trigger. */
export const REJECTION = { accepted: 0, regularization_cap: 1, line_search_failed: 2, non_finite: 3, curvature_retry: 4 } as const;

/** One accepted PTC step. `residual` is r_k (f32, absolute) at the start of the step. */
export interface PtcStepLog {
  readonly step: number;
  readonly h: number;
  readonly residual: number;
  readonly inner: number;
  readonly rejected: number;
  readonly pcgIterations: number;
  readonly minDistance: number;
  readonly curvatureDropped: boolean;
}

/** One attempt (accepted or rejected) and the range of iteration records it produced. */
export interface AttemptLog {
  readonly step: number;
  readonly attempt: number;
  readonly h: number;
  readonly reason: number;
  readonly recordStart: number;
  readonly recordEnd: number;
}

export interface EquilibriumRun {
  readonly states: readonly Float32Array<ArrayBuffer>[];
  readonly steps: readonly PtcStepLog[];
  readonly attempts: readonly AttemptLog[];
  readonly run: Pick<RobustRun, "records" | "extras">;
  readonly stop: StopRule;
  readonly finalH: number;
  readonly finalResidual: number;
  readonly residuals: readonly number[];
  readonly ms: number;
  readonly readBytes: number;
  readonly phases: Readonly<Record<Phase | "io", number>>;
}

/** The start of a solve: positions and the first step size. */
export interface PtcStart {
  readonly positions: Float32Array<ArrayBuffer>;
  readonly h: number;
}

/** x̂ = x + h²g on the unknowns in f32, velocity discarded (the oracle's ptc_prediction). */
export function ptcPrediction(problem: DrapeProblem, positions: Float32Array<ArrayBuffer>, h: number): Float32Array<ArrayBuffer> {
  const fall = Math.fround(Math.fround(h * h) * Math.fround(-problem.manifest.gravity_m_per_s2));
  const predicted = Float32Array.from(positions);
  problem.u32.row_vertex.forEach((vertex) => { predicted[3 * vertex + 2] = positions[3 * vertex + 2] + fall; });
  return predicted;
}

/** Rewrites row_mass = M/h² (f32 unit mass / h², rounded) and x̂ = x + h²g for the coming attempt. */
function prepare<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem, positions: Float32Array<ArrayBuffer>, h: number): Float32Array<ArrayBuffer> {
  if (problem.unitMass === null) throw new Error("problem.bin has no row_unit_mass (not an Experiment 7 problem)");
  gpu.backend.write(gpu.rowMass, Float32Array.from(problem.unitMass, (mass) => mass / (h * h)));
  const xHat = ptcPrediction(problem, positions, h);
  gpu.backend.write(gpu.xHat, xHat);
  gpu.backend.write(gpu.regularization, Float32Array.of(0));
  return xHat;
}

/** The Newton settings of a PTC step: tolerance 0.1 and at most 10 inner updates. */
function withInnerNewton<A, K>(ctx: RobustContext<A, K>, ptc: PtcRules): RobustContext<A, K> {
  return { ...ctx, model: { ...ctx.model, newton: { ...ctx.model.newton, tolerance: ptc.inner_tolerance, max_iterations: ptc.inner_max_iterations } } };
}

/** Accepted updates of an attempt and why it failed, from its iteration records and its final state. */
export function classify(records: readonly IterationRecord[], next: Float32Array): { readonly updates: number; readonly reason: number } {
  const updates = records.filter((entry) => entry.status === STEP_STATUS.running && Number.isFinite(entry.pcgIterations)).length;
  const last = records[records.length - 1]?.status;
  if (next.some((value) => !Number.isFinite(value))) return { updates, reason: REJECTION.non_finite };
  if (updates === 0 && last === STEP_STATUS.regularization_cap) return { updates, reason: REJECTION.regularization_cap };
  if (updates === 0 && last === STEP_STATUS.line_search_failed) return { updates, reason: REJECTION.line_search_failed };
  return { updates, reason: REJECTION.accepted };
}

interface Attempt {
  readonly next: Float32Array<ArrayBuffer>;
  readonly updates: number;
  readonly reason: number;
  readonly recordStart: number;
  readonly recordEnd: number;
}

/** One attempt at step size h from `positions`; `ready` says the residual of this state at this h is already on the GPU. */
async function attempt<A, K>(ctx: RobustContext<A, K>, gpu: DrapeGpu<A, K>, problem: DrapeProblem, positions: Float32Array<ArrayBuffer>, h: number, scale: number, step: number, ready: boolean): Promise<Attempt> {
  const xHat = prepare(gpu, problem, positions, h);
  if (!ready) await ctx.ops.residual(positions);
  const next = await newtonStart(ctx, gpu, positions, xHat);
  const recordStart = ctx.records.length;
  await solveStepRobust(ctx, next, scale, step, positions);
  const recordEnd = ctx.records.length;
  return { next, recordStart, recordEnd, ...classify(ctx.records.slice(recordStart, recordEnd), next) };
}

/** An accepted attempt that ended on the μ cap after making progress: the evidence for dropping b′∇²d (Experiment 7 trigger). */
function hitCap(result: Attempt, records: readonly IterationRecord[]): boolean {
  return result.reason === REJECTION.accepted && result.updates > 0 && records.slice(result.recordStart, result.recordEnd).some((entry) => entry.status === STEP_STATUS.regularization_cap);
}

/** `attempt`, redone without the b′∇²d term when the first run hit the μ cap; the superseded run is logged with reason `curvature_retry`. */
async function attemptWithTrigger<A, K>(ctx: RobustContext<A, K>, gpu: DrapeGpu<A, K>, problem: DrapeProblem, positions: Float32Array<ArrayBuffer>, h: number, scale: number, where: { readonly step: number; readonly index: number; readonly ready: boolean }, attempts: AttemptLog[]): Promise<Attempt & { readonly dropped: boolean }> {
  const first = await attempt(ctx, gpu, problem, positions, h, scale, where.step, where.ready);
  if (!hitCap(first, ctx.records)) return { ...first, dropped: false };
  attempts.push({ step: where.step, attempt: where.index, h, reason: REJECTION.curvature_retry, recordStart: first.recordStart, recordEnd: first.recordEnd });
  setBarrierCurvature(gpu, problem, true);
  try {
    return { ...(await attempt(ctx, gpu, problem, positions, h, scale, where.step, false)), dropped: true };
  } finally {
    setBarrierCurvature(gpu, problem, false);
  }
}

interface Accepted {
  readonly next: Float32Array<ArrayBuffer>;
  readonly log: PtcStepLog;
}

const pcgOf = (records: readonly IterationRecord[]): number => records.reduce((sum, entry) => sum + (Number.isFinite(entry.pcgIterations) ? entry.pcgIterations : 0), 0);

/** A step from `positions`: attempts at h, h/2, … until one is accepted (rule 0.5); null when all fail. */
async function takeStep<A, K>(ctx: RobustContext<A, K>, gpu: DrapeGpu<A, K>, problem: DrapeProblem, ptc: PtcRules, state: { readonly positions: Float32Array<ArrayBuffer>; readonly h: number; readonly scale: number; readonly step: number }, attempts: AttemptLog[]): Promise<Accepted | null> {
  const floor = ptc.h0 * ptc.h_min_factor;
  for (let rejected = 0; rejected <= ptc.max_rejections; rejected += 1) {
    const h = Math.max(floor, state.h * ptc.shrink ** rejected);
    const result = await attemptWithTrigger(ctx, gpu, problem, state.positions, h, state.scale, { step: state.step, index: rejected, ready: rejected === 0 }, attempts);
    attempts.push({ step: state.step, attempt: rejected, h, reason: result.reason, recordStart: result.recordStart, recordEnd: result.recordEnd });
    if (result.reason === REJECTION.accepted) {
      const records = ctx.records.slice(result.recordStart, result.recordEnd);
      const extras = ctx.extras.slice(result.recordStart, result.recordEnd);
      const minDistance = extras.map((extra) => extra.minDistance).filter(Number.isFinite).pop() ?? Number.NaN;
      return { next: result.next, log: { step: state.step, h, residual: state.scale, inner: result.updates, rejected, pcgIterations: pcgOf(records), minDistance, curvatureDropped: result.dropped } };
    }
  }
  return null;
}

/** Optional progress callback and wall-clock budget of a solve. */
export interface Watch {
  readonly onStep: ((log: PtcStepLog, elapsedMs: number) => void) | null;
  readonly budgetMs: number | null;
}

/** The solve loop: measure r_k, apply the stop rules, take a step, adapt h from the inner count. */
export async function solveEquilibrium<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem, rules: RobustRules, settings: RobustSettings, ptc: PtcRules, clock: PhaseClock, start: PtcStart, cap: number, keepStates: boolean, watch: Watch = { onStep: null, budgetMs: null }): Promise<EquilibriumRun> {
  const ctx = withInnerNewton(context(gpu, problem, rules, settings, clock, false), ptc);
  const state = { positions: Float32Array.from(start.positions), h: start.h };
  const out = { states: [Float32Array.from(start.positions)], steps: [] as PtcStepLog[], attempts: [] as AttemptLog[], residuals: [] as number[] };
  resetClock(clock);
  const started = performance.now();
  let stop: StopRule | null = null;
  while (stop === null) {
    prepare(gpu, problem, state.positions, state.h);
    const scale = (await ctx.ops.residual(state.positions)).norm;
    out.residuals.push(scale);
    stop = stopRule(ptc, out.residuals, out.steps.length, cap) ?? (watch.budgetMs !== null && performance.now() - started > watch.budgetMs ? "time_budget" : null);
    if (stop !== null) break;
    const taken = await takeStep(ctx, gpu, problem, ptc, { positions: state.positions, h: state.h, scale, step: out.steps.length + 1 }, out.attempts);
    if (taken === null) { stop = "stalled"; break; }
    out.steps.push(taken.log);
    watch.onStep?.(taken.log, performance.now() - started);
    state.positions = taken.next;
    state.h = nextStepSize(ptc, taken.log.h, taken.log.inner);
    if (keepStates) out.states.push(Float32Array.from(taken.next));
  }
  const ms = performance.now() - started;
  const states = keepStates ? out.states : [out.states[0], Float32Array.from(state.positions)];
  return { ...out, states, run: { records: ctx.records, extras: ctx.extras }, stop, finalH: state.h, finalResidual: out.residuals[out.residuals.length - 1], ms, readBytes: clock.readBytes, phases: phaseMeans(clock, ms, Math.max(1, out.steps.length)) };
}

/** A probe: the f32 start state and step size of one oracle step. */
export interface PtcProbes {
  readonly steps: readonly number[];
  readonly h: readonly number[];
  readonly positions: readonly Float32Array<ArrayBuffer>[];
}

/** One PTC attempt from each oracle probe state at the oracle's accepted h, keeping every iterate for the oracle. */
export async function runPtcProbes<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem, rules: RobustRules, settings: RobustSettings, ptc: PtcRules, clock: PhaseClock, probes: PtcProbes): Promise<RobustRun> {
  const ctx = withInnerNewton(context(gpu, problem, rules, settings, clock, true), ptc);
  for (let index = 0; index < probes.steps.length; index += 1) {
    const positions = probes.positions[index];
    const xHat = prepare(gpu, problem, positions, probes.h[index]);
    const scale = (await ctx.ops.residual(positions)).norm;
    const next = await newtonStart(ctx, gpu, positions, xHat);
    await solveStepRobust(ctx, next, scale, probes.steps[index], positions);
  }
  return { records: ctx.records, extras: ctx.extras, iterates: ctx.iterates ?? [], directions: ctx.directions ?? [], trials: ctx.trials ?? [], noiseStates: ctx.noiseStates ?? [] };
}

/** The inertia kernel (colliders off) at x = kernel state, x̂ = rest, μ = 0 for each step size: [gradient 3R | Hessian 9R] per h. */
export async function evaluateInertiaAtSteps<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem, steps: readonly number[]): Promise<Float32Array<ArrayBuffer>> {
  const { triangles: t, hinges: h, rows } = problem.manifest;
  const parts: Float32Array<ArrayBuffer>[] = [];
  for (const size of steps) {
    gpu.backend.write(gpu.positions, problem.f32.kernel_state);
    gpu.backend.write(gpu.xHat, problem.f32.rest_positions);
    gpu.backend.write(gpu.regularization, Float32Array.of(0));
    gpu.backend.write(gpu.rowMass, Float32Array.from(problem.unitMass ?? [], (mass) => mass / (size * size)));
    await withoutColliders(gpu, problem, () => gpu.backend.run(gpu.inertia));
    const [gradient, hessian] = await Promise.all([gpu.backend.read(gpu.elemGrad), gpu.backend.read(gpu.elemHess)]);
    parts.push(new Float32Array(new Float32Array(gradient, 4 * (9 * t + 12 * h), 3 * rows)), new Float32Array(new Float32Array(hessian, 4 * (81 * t + 144 * h), 9 * rows)));
  }
  const joined = new Float32Array(parts.reduce((length, part) => length + part.length, 0));
  parts.reduce((offset, part) => (joined.set(part, offset), offset + part.length), 0);
  return joined;
}
