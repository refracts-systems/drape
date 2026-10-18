import { numbers } from "./coupon.ts";
import { isObject } from "./problem.ts";
import {
  advance, linearStatus, record, scatter, STEP_STATUS, terminal,
  type IterationRecord, type NewtonContext, type NewtonGpu, type StepStatus,
} from "./newton.ts";
import { searchStep, type NewtonOps, type NoisySearch, type Residual, type Slope, type Trial } from "./newton-ops.ts";
import { solveOnce, type SolveRun } from "./solver.ts";
import { inPhase, type PhaseClock } from "./timed-backend.ts";

const CHUNK_ITERATIONS = 32;
const RULE_KEYS = ["eta_start", "eta_min", "eta_max", "eta_gamma", "eta_safeguard", "update_stop", "mu_start", "mu_cap", "kernel_mu"] as const;
/** `resolution_factor` is absent from E4 problems: 0 keeps the E4 update rule. */
export type RobustRules = Readonly<Record<(typeof RULE_KEYS)[number] | "resolution_factor", number>>;

/** Experiment 4 rule constants from problem.json (written by the oracle's e4_rules). */
export function parseRobustRules(text: string): RobustRules {
  const value: unknown = JSON.parse(text);
  if (!isObject(value)) throw new Error("problem.json is not an object");
  const robust: unknown = Reflect.get(value, "robust");
  const factor: unknown = isObject(robust) ? Reflect.get(robust, "resolution_factor") : undefined;
  return { ...numbers(robust, RULE_KEYS, "robust"), resolution_factor: typeof factor === "number" ? factor : 0 };
}

/** Ablation switches: floor-aware stop, Eisenstat–Walker forcing, K + μ·M/h² regularisation, and (Experiment 6) the noise-aware
 * line search, GPU reductions and GPU-side bound/distance minima. The last two choose the `ops` and contact hooks at setup. */
export interface RobustSettings {
  readonly floorStop: boolean;
  readonly forcing: boolean;
  readonly regularize: boolean;
  readonly noiseAware?: boolean;
  readonly gpuReductions?: boolean;
  readonly fewReadbacks?: boolean;
}

/** What each iterate's decision rested on, recorded next to its IterationRecord for the oracle to replay. */
export interface RobustRecord {
  readonly eta: number;
  readonly mu: number;
  readonly updateInf: number;
  readonly stepInf: number;
  readonly retries: number;
  readonly resolution: number;
  readonly alphaMax: number;
  readonly limitingVertex: number;
  readonly minDistance: number;
  readonly deltaSum: number;
  readonly deltaAbs: number;
  readonly noiseOnly: number;
  readonly slope: number;
  readonly slopeExact: number;
  readonly normSquared: number;
  readonly normSquaredExact: number;
}

export interface RobustGpu<A, K> extends NewtonGpu<A, K> {
  readonly regularization: A;
}

/** Line-search start of the colliders and the host-side f64 distance check (Experiment 5). */
export interface ContactHooks {
  /** α_max and its limiting vertex for the positions and direction currently on the GPU. */
  readonly maxStep: () => Promise<{ readonly alpha: number; readonly vertex: number }>;
  /** Smallest collider distance over the unknowns, in f64 from the f32 positions. */
  readonly minDistance: (positions: Float32Array<ArrayBuffer>) => Promise<number>;
}

export interface RobustContext<A, K> extends NewtonContext<A, K> {
  readonly gpu: RobustGpu<A, K>;
  readonly rules: RobustRules;
  readonly settings: RobustSettings;
  readonly extras: RobustRecord[];
  readonly clock: PhaseClock;
  readonly contact: ContactHooks | null;
  /** Probes only: x_k and the scattered direction p_k of every update, for the oracle's α_max replay. */
  readonly directions: Float32Array<ArrayBuffer>[] | null;
  readonly ops: NewtonOps;
  /** Probes only: every tried step length of every line search, for the oracle's acceptance replay. */
  readonly trials: TrialRecord[] | null;
  /** x before and after every noise-only accepted update (n ≤ 65): the oracle evaluates f64 Φ across them. */
  readonly noiseStates: Float32Array<ArrayBuffer>[] | null;
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

/** A tried step length with where it was tried. */
export interface TrialRecord extends Trial {
  readonly step: number;
  readonly iteration: number;
  readonly halving: number;
}

/** PCG direction for the current μ; under regularisation μ rises (with reassembly) on breakdown or non-descent, null at the cap. */
async function solveDirection<A, K>(context: RobustContext<A, K>, positions: Float32Array<ArrayBuffer>, state: StepState): Promise<{ readonly run: SolveRun; readonly retries: number; readonly slope: Slope } | null> {
  const settings = { tolerance: state.eta, maxIterations: context.model.pcg.max_iterations, chunkIterations: CHUNK_ITERATIONS };
  for (let retries = 0; ; retries += 1) {
    const run = await inPhase(context.clock, "pcg", () => solveOnce(context.gpu.solver, settings));
    const slope = await context.ops.slope(run.solution);
    const failed = run.status === "breakdown" || !(slope.value < 0);
    if (!failed || !context.settings.regularize) return { run, retries, slope };
    if (state.mu >= context.rules.mu_cap) return null;
    state.mu = raiseMu(context.rules, state.mu);
    writeMu(context, state.mu);
    await context.ops.residual(positions);
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

/** Spacing of f32 numbers at max |x_i| over the unknowns: 2^(exponent − 23), from the f32 exponent bits. */
function resolutionOf(context: { readonly model: { readonly rowVertex: Uint32Array } }, positions: Float32Array): number {
  const largest = Array.from(context.model.rowVertex).reduce((m, v) => Math.max(m, ...[0, 1, 2].map((d) => Math.abs(positions[3 * v + d]))), 0);
  const bits = new Uint32Array(Float32Array.of(largest).buffer)[0];
  return 2 ** (((bits >>> 23) & 0xff) - 150);
}

/** The update rule: ‖αd‖∞ ≤ max(update_stop·‖x − x_n‖∞, resolution_factor·res(x)). */
const updateIsSmall = (rules: RobustRules, extra: RobustRecord): boolean =>
  extra.updateInf <= Math.max(rules.update_stop * extra.stepInf, rules.resolution_factor * extra.resolution);

function forcingFor<A, K>(context: RobustContext<A, K>, state: StepState, norm: number, iteration: number): number {
  if (!context.settings.forcing) return context.model.pcg.tolerance;
  return iteration === 0 ? context.rules.eta_start : nextForcing(context.rules, state.eta, f32(norm / state.previousNorm));
}

const NAN = Number.NaN;

const emptyExtra = (state: StepState): RobustRecord => ({ eta: NAN, mu: state.mu, updateInf: NAN, stepInf: NAN, retries: 0, resolution: NAN, alphaMax: NAN, limitingVertex: NAN, minDistance: NAN,
  deltaSum: NAN, deltaAbs: NAN, noiseOnly: NAN, slope: NAN, slopeExact: NAN, normSquared: NAN, normSquaredExact: NAN });

const NO_LIMIT = { alpha: 1, vertex: -1 } as const;

/** Writes p_k and returns the line-search start: α_max of the colliders (timed as "ccd"), or 1 without colliders. */
async function startStep<A, K>(context: RobustContext<A, K>, positions: Float32Array<ArrayBuffer>, direction: Float32Array): Promise<{ readonly alpha: number; readonly vertex: number }> {
  const scattered = scatter(context.model, direction);
  context.gpu.backend.write(context.gpu.direction, scattered);
  context.directions?.push(Float32Array.from(positions), scattered);
  const contact = context.contact;
  return contact === null ? NO_LIMIT : inPhase(context.clock, "ccd", () => contact.maxStep());
}

/** Terminal record at `positions`: residual rule, iteration limit, regularisation cap or update rule. */
async function finish<A, K>(context: RobustContext<A, K>, where: { readonly step: number; readonly iteration: number }, relative: number, status: StepStatus, state: StepState, positions: Float32Array<ArrayBuffer>): Promise<void> {
  pushRecord(context, terminal(where.step, where.iteration, relative, status), emptyExtra(state), positions);
}

/** Where an update happens: step, iteration, relative residual, the residual measurement and x_n. */
interface Where {
  readonly step: number;
  readonly iteration: number;
  readonly relative: number;
  readonly residual: Residual;
  readonly origin: Float32Array;
}

function searchOptions<A, K>(context: RobustContext<A, K>, where: Where): { readonly noiseAware: boolean; readonly onTrial: ((trial: Trial) => void) | null } {
  const trials = context.trials;
  const counter = { halving: 0 };
  return { noiseAware: context.settings.noiseAware === true,
    onTrial: trials === null ? null : (trial) => { trials.push({ ...trial, step: where.step, iteration: where.iteration, halving: counter.halving }); counter.halving += 1; } };
}

/** The robust record of one accepted (or failed) update, with everything the oracle replays. */
async function recordOf<A, K>(context: RobustContext<A, K>, found: { readonly search: NoisySearch | null; readonly updateInf: number; readonly limit: { readonly alpha: number; readonly vertex: number } },
  solved: { readonly retries: number; readonly slope: Slope }, state: StepState, where: Where, positions: Float32Array<ArrayBuffer>): Promise<RobustRecord> {
  const { search, updateInf, limit } = found;
  return { eta: state.eta, mu: state.mu, updateInf, stepInf: stepInf(context, positions, where.origin), retries: solved.retries, resolution: resolutionOf(context, positions),
    alphaMax: limit.alpha, limitingVertex: limit.vertex, minDistance: (await context.contact?.minDistance(positions)) ?? NAN,
    deltaSum: search?.sum ?? NAN, deltaAbs: search?.abs ?? NAN, noiseOnly: search === null ? NAN : Number(search.noiseOnly),
    slope: solved.slope.value, slopeExact: solved.slope.exact, normSquared: where.residual.normSquared, normSquaredExact: where.residual.normSquaredExact };
}

/** One update: direction, line search, records; returns the accepted update's robust record, or null when the step ends. */
async function update<A, K>(context: RobustContext<A, K>, positions: Float32Array<ArrayBuffer>, state: StepState, where: Where): Promise<RobustRecord | null> {
  const solved = await solveDirection(context, positions, state);
  if (solved === null) {
    await finish(context, where, where.relative, STEP_STATUS.regularization_cap, state, positions);
    return null;
  }
  const direction = solved.run.solution;
  const limit = await startStep(context, positions, direction);
  const search = await inPhase(context.clock, "line_search", () => searchStep(context.ops, context.model, solved.slope.value, limit.alpha, searchOptions(context, where)));
  const updateInf = search === null ? Number.NaN : search.stepLength * infNorm(direction);
  const entry: IterationRecord = { step: where.step, iteration: where.iteration, relativeResidual: where.relative, pcgIterations: solved.run.iterations, pcgStatus: linearStatus(solved.run),
    linearResidual: solved.run.gpuTrueRelativeResidual, stepLength: search?.stepLength ?? Number.NaN, halvings: search?.halvings ?? Number.NaN, status: search === null ? STEP_STATUS.line_search_failed : STEP_STATUS.running };
  const before = search?.noiseOnly === true && context.noiseStates !== null ? Float32Array.from(positions) : null;
  if (search !== null) advance(context.model, positions, direction, search.stepLength);
  if (before !== null) context.noiseStates?.push(before, Float32Array.from(positions));
  const extra = await recordOf(context, { search, updateInf, limit }, solved, state, where, positions);
  pushRecord(context, entry, extra, positions);
  if (search !== null && context.settings.regularize) state.mu = search.halvings === 0 ? relaxMu(context.rules, state.mu) : state.mu;
  return search === null ? null : extra;
}

/** Newton from `positions` (updated in place); `origin` is x_n, the state the time step started from. */
export async function solveStepRobust<A, K>(context: RobustContext<A, K>, positions: Float32Array<ArrayBuffer>, scale: number, step: number, origin: Float32Array): Promise<void> {
  const state: StepState = { mu: 0, eta: context.rules.eta_start, previousNorm: Number.NaN };
  for (let iteration = 0; ; iteration += 1) {
    writeMu(context, state.mu);
    const residual = await context.ops.residual(positions);
    const norm = residual.norm;
    const relative = f32(norm / scale);
    const where = { step, iteration, relative, residual, origin };
    if (relative <= context.model.newton.tolerance || iteration === context.model.newton.max_iterations) {
      return finish(context, where, relative, relative <= context.model.newton.tolerance ? STEP_STATUS.converged : STEP_STATUS.iteration_limit, state, positions);
    }
    state.eta = forcingFor(context, state, norm, iteration);
    state.previousNorm = norm;
    const extra = await update(context, positions, state, where);
    if (extra === null) return;
    if (context.settings.floorStop && updateIsSmall(context.rules, extra)) {
      return finishByUpdate(context, { step, iteration: iteration + 1 }, scale, state, positions);
    }
  }
}

async function finishByUpdate<A, K>(context: RobustContext<A, K>, where: { readonly step: number; readonly iteration: number }, scale: number, state: StepState, positions: Float32Array<ArrayBuffer>): Promise<void> {
  writeMu(context, state.mu);
  const residual = await context.ops.residual(positions);
  return finish(context, where, f32(residual.norm / scale), STEP_STATUS.update_rule, state, positions);
}
