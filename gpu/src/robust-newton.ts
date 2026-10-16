import { numbers } from "./coupon.ts";
import { isObject } from "./problem.ts";
import {
  advance, assemble, dot32, lineSearch, linearStatus, norm32, record, scatter, STEP_STATUS, terminal,
  type IterationRecord, type NewtonContext, type NewtonGpu, type StepStatus,
} from "./newton.ts";
import { solveOnce, type SolveRun } from "./solver.ts";
import { inPhase, type PhaseClock } from "./timed-backend.ts";

const CHUNK_ITERATIONS = 32;
const RULE_KEYS = ["eta_start", "eta_min", "eta_max", "eta_gamma", "eta_safeguard", "update_stop", "mu_start", "mu_cap", "kernel_mu"] as const;
export type RobustRules = Readonly<Record<(typeof RULE_KEYS)[number], number>>;

/** Experiment 4 rule constants from problem.json (written by the oracle's e4_rules). */
export function parseRobustRules(text: string): RobustRules {
  const value: unknown = JSON.parse(text);
  if (!isObject(value)) throw new Error("problem.json is not an object");
  return numbers(Reflect.get(value, "robust"), RULE_KEYS, "robust");
}

/** Ablation switches: floor-aware stop, Eisenstat–Walker forcing, K + μ·M/h² regularisation. */
export interface RobustSettings {
  readonly floorStop: boolean;
  readonly forcing: boolean;
  readonly regularize: boolean;
}

/** What each iterate's decision rested on, recorded next to its IterationRecord for the oracle to replay. */
export interface RobustRecord {
  readonly eta: number;
  readonly mu: number;
  readonly updateInf: number;
  readonly stepInf: number;
  readonly retries: number;
}

export interface RobustGpu<A, K> extends NewtonGpu<A, K> {
  readonly regularization: A;
}

export interface RobustContext<A, K> extends NewtonContext<A, K> {
  readonly gpu: RobustGpu<A, K>;
  readonly rules: RobustRules;
  readonly settings: RobustSettings;
  readonly extras: RobustRecord[];
  readonly clock: PhaseClock;
}

/** Newton state carried across the iterations of one time step. */
interface StepState {
  mu: number;
  eta: number;
  previousNorm: number;
}

const f32 = Math.fround;

/** Eisenstat–Walker choice 2 in f32: η = γ·ratio², safeguarded by γ·η_prev² above the threshold, clamped. */
export function nextForcing(rules: RobustRules, previous: number, ratio: number): number {
  const eta = f32(rules.eta_gamma * f32(ratio * ratio));
  const safeguard = f32(rules.eta_gamma * f32(previous * previous));
  const guarded = safeguard > rules.eta_safeguard ? Math.max(eta, safeguard) : eta;
  return Math.min(rules.eta_max, Math.max(rules.eta_min, guarded));
}

const raiseMu = (rules: RobustRules, mu: number): number => Math.min(rules.mu_cap, Math.max(rules.mu_start, 10 * mu));
const relaxMu = (rules: RobustRules, mu: number): number => (mu / 10 < rules.mu_start ? 0 : mu / 10);

function pushRecord<A, K>(context: RobustContext<A, K>, entry: IterationRecord, extra: RobustRecord, positions: Float32Array<ArrayBuffer>): void {
  record(context, entry, positions);
  context.extras.push(extra);
}

function writeMu<A, K>(context: RobustContext<A, K>, mu: number): void {
  if (context.settings.regularize) context.gpu.backend.write(context.gpu.regularization, Float32Array.of(mu));
}

/** PCG direction for the current μ; under regularisation μ rises (with reassembly) on breakdown or non-descent, null at the cap. */
async function solveDirection<A, K>(context: RobustContext<A, K>, positions: Float32Array<ArrayBuffer>, rhs: Float32Array, state: StepState): Promise<{ readonly run: SolveRun; readonly retries: number } | null> {
  const settings = { tolerance: state.eta, maxIterations: context.model.pcg.max_iterations, chunkIterations: CHUNK_ITERATIONS };
  for (let retries = 0; ; retries += 1) {
    const run = await inPhase(context.clock, "pcg", () => solveOnce(context.gpu.solver, settings));
    const failed = run.status === "breakdown" || !(-dot32(rhs, run.solution) < 0);
    if (!failed || !context.settings.regularize) return { run, retries };
    if (state.mu >= context.rules.mu_cap) return null;
    state.mu = raiseMu(context.rules, state.mu);
    writeMu(context, state.mu);
    await inPhase(context.clock, "assembly", () => assemble(context.gpu, positions));
  }
}

function infNorm(values: Float32Array): number {
  return values.reduce((largest, v) => Math.max(largest, Math.abs(v)), 0);
}

/** ‖x − x_n‖∞ over the unknowns. */
function stepInf(context: { readonly model: { readonly rowVertex: Uint32Array } }, positions: Float32Array, origin: Float32Array): number {
  return Array.from(context.model.rowVertex).reduce((largest, v) =>
    Math.max(largest, ...[0, 1, 2].map((d) => Math.abs(positions[3 * v + d] - origin[3 * v + d]))), 0);
}

function forcingFor<A, K>(context: RobustContext<A, K>, state: StepState, norm: number, iteration: number): number {
  if (!context.settings.forcing) return context.model.pcg.tolerance;
  return iteration === 0 ? context.rules.eta_start : nextForcing(context.rules, state.eta, f32(norm / state.previousNorm));
}

const emptyExtra = (state: StepState): RobustRecord => ({ eta: Number.NaN, mu: state.mu, updateInf: Number.NaN, stepInf: Number.NaN, retries: 0 });

/** Terminal record at `positions`: residual rule, iteration limit, regularisation cap or update rule. */
async function finish<A, K>(context: RobustContext<A, K>, where: { readonly step: number; readonly iteration: number }, relative: number, status: StepStatus, state: StepState, positions: Float32Array<ArrayBuffer>): Promise<void> {
  pushRecord(context, terminal(where.step, where.iteration, relative, status), emptyExtra(state), positions);
}

/** One update: direction, line search, records; returns the accepted update's ‖αd‖∞, or null when the step ends. */
async function update<A, K>(context: RobustContext<A, K>, positions: Float32Array<ArrayBuffer>, rhs: Float32Array, state: StepState, where: { readonly step: number; readonly iteration: number; readonly relative: number; readonly origin: Float32Array }): Promise<number | null> {
  const solved = await solveDirection(context, positions, rhs, state);
  if (solved === null) {
    await finish(context, where, where.relative, STEP_STATUS.regularization_cap, state, positions);
    return null;
  }
  const direction = solved.run.solution;
  context.gpu.backend.write(context.gpu.direction, scatter(context.model, direction));
  const search = await inPhase(context.clock, "line_search", () => lineSearch(context, -dot32(rhs, direction)));
  const updateInf = search === null ? Number.NaN : search.stepLength * infNorm(direction);
  const entry: IterationRecord = { step: where.step, iteration: where.iteration, relativeResidual: where.relative, pcgIterations: solved.run.iterations, pcgStatus: linearStatus(solved.run),
    linearResidual: solved.run.gpuTrueRelativeResidual, stepLength: search?.stepLength ?? Number.NaN, halvings: search?.halvings ?? Number.NaN, status: search === null ? STEP_STATUS.line_search_failed : STEP_STATUS.running };
  if (search !== null) advance(context.model, positions, direction, search.stepLength);
  pushRecord(context, entry, { eta: state.eta, mu: state.mu, updateInf, stepInf: stepInf(context, positions, where.origin), retries: solved.retries }, positions);
  if (search !== null && context.settings.regularize) state.mu = search.halvings === 0 ? relaxMu(context.rules, state.mu) : state.mu;
  return search === null ? null : updateInf;
}

/** Newton from `positions` (updated in place); `origin` is x_n, the state the time step started from. */
export async function solveStepRobust<A, K>(context: RobustContext<A, K>, positions: Float32Array<ArrayBuffer>, scale: number, step: number, origin: Float32Array): Promise<void> {
  const state: StepState = { mu: 0, eta: context.rules.eta_start, previousNorm: Number.NaN };
  for (let iteration = 0; ; iteration += 1) {
    writeMu(context, state.mu);
    const rhs = await inPhase(context.clock, "assembly", () => assemble(context.gpu, positions));
    const norm = norm32(rhs);
    const relative = f32(norm / scale);
    const where = { step, iteration, relative, origin };
    if (relative <= context.model.newton.tolerance || iteration === context.model.newton.max_iterations) {
      return finish(context, where, relative, relative <= context.model.newton.tolerance ? STEP_STATUS.converged : STEP_STATUS.iteration_limit, state, positions);
    }
    state.eta = forcingFor(context, state, norm, iteration);
    state.previousNorm = norm;
    const updateInf = await update(context, positions, rhs, state, where);
    if (updateInf === null) return;
    if (context.settings.floorStop && updateInf <= context.rules.update_stop * stepInf(context, positions, origin)) {
      return finishByUpdate(context, { step, iteration: iteration + 1 }, scale, state, positions);
    }
  }
}

async function finishByUpdate<A, K>(context: RobustContext<A, K>, where: { readonly step: number; readonly iteration: number }, scale: number, state: StepState, positions: Float32Array<ArrayBuffer>): Promise<void> {
  writeMu(context, state.mu);
  const rhs = await inPhase(context.clock, "assembly", () => assemble(context.gpu, positions));
  return finish(context, where, f32(norm32(rhs) / scale), STEP_STATUS.update_rule, state, positions);
}
