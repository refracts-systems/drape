import type { Backend, Dispatch } from "./backend.ts";
import type { CouponProblem, NewtonSettings, PcgSettings } from "./coupon.ts";
import type { CouponGpu } from "./coupon-gpu.ts";
import { solveOnce, type PreparedSolver, type SolveRun, type SolveSettings } from "./solver.ts";

const CHUNK_ITERATIONS = 32;
const IDENTITY = [1, 0, 0, 1] as const;
export const STEP_STATUS = { running: 0, converged: 1, iteration_limit: 2, line_search_failed: 3, update_rule: 4, regularization_cap: 5 } as const;
export type StepStatus = (typeof STEP_STATUS)[keyof typeof STEP_STATUS];

/** One Newton iterate: f32 residual ‖g‖/scale, the PCG solve and line search taken from it (NaN on the last iterate). */
export interface IterationRecord {
  readonly step: number;
  readonly iteration: number;
  readonly relativeResidual: number;
  readonly pcgIterations: number;
  readonly pcgStatus: number;
  readonly linearResidual: number;
  readonly stepLength: number;
  readonly halvings: number;
  readonly status: StepStatus;
}

export interface PathResult {
  readonly records: readonly IterationRecord[];
  readonly iterates: readonly Float32Array<ArrayBuffer>[];
  readonly solveMs: number;
}

/** The GPU surface Newton needs: an evaluate pass filling the solver's vals/rhs, and an energy-change pass. */
export interface NewtonGpu<A, K> {
  readonly backend: Backend<A, K>;
  readonly solver: PreparedSolver<A, K>;
  readonly positions: A;
  readonly direction: A;
  readonly step: A;
  readonly elemDelta: A;
  readonly evaluate: readonly Dispatch<K>[];
  readonly energyChange: readonly Dispatch<K>[];
}

/** Settings and the row → vertex map of one discretised energy. */
export interface NewtonModel {
  readonly newton: NewtonSettings;
  readonly pcg: PcgSettings;
  readonly rowVertex: Uint32Array<ArrayBuffer>;
  readonly vertices: number;
}

/** Iterates are kept only when an oracle will evaluate every one of them. */
export interface NewtonContext<A, K> {
  readonly gpu: NewtonGpu<A, K>;
  readonly model: NewtonModel;
  readonly records: IterationRecord[];
  readonly iterates: Float32Array<ArrayBuffer>[] | null;
}

export interface LineSearch {
  readonly stepLength: number;
  readonly halvings: number;
}

export const norm32 = (values: Float32Array): number => Math.fround(Math.sqrt(values.reduce((sum, v) => Math.fround(sum + Math.fround(v * v)), 0)));
export const dot32 = (a: Float32Array, b: Float32Array): number => a.reduce((sum, v, i) => Math.fround(sum + Math.fround(v * b[i])), 0);

/** Assembles tangent and −g at `positions` into the solver's vals/rhs; returns −g. */
export async function assemble<A, K>(gpu: NewtonGpu<A, K>, positions: Float32Array<ArrayBuffer>): Promise<Float32Array<ArrayBuffer>> {
  gpu.backend.write(gpu.positions, positions);
  await gpu.backend.run(gpu.evaluate);
  return new Float32Array(await gpu.backend.read(gpu.solver.arrays.rhs));
}

export async function energyChange<A, K>(gpu: NewtonGpu<A, K>, stepLength: number): Promise<number> {
  gpu.backend.write(gpu.step, Float32Array.of(stepLength));
  await gpu.backend.run(gpu.energyChange);
  const deltas = new Float32Array(await gpu.backend.read(gpu.elemDelta));
  return deltas.reduce((sum, v) => Math.fround(sum + v), 0);
}

/** Backtracking from α = 1 until ΔΠ ≤ c·α·gᵀd; null when d is not a descent direction or halving runs out. */
export async function lineSearch<A, K>(context: NewtonContext<A, K>, slope: number): Promise<LineSearch | null> {
  const { armijo, max_halvings } = context.model.newton;
  if (!(slope < 0)) return null;
  for (let halvings = 0; halvings <= max_halvings; halvings += 1) {
    const stepLength = 0.5 ** halvings;
    if (await energyChange(context.gpu, stepLength) <= Math.fround(armijo * stepLength * slope)) return { stepLength, halvings };
  }
  return null;
}

export function scatter(model: NewtonModel, rows: Float32Array): Float32Array<ArrayBuffer> {
  const vertices = new Float32Array(3 * model.vertices);
  model.rowVertex.forEach((vertex, row) => vertices.set(rows.subarray(3 * row, 3 * row + 3), 3 * vertex));
  return vertices;
}

export function advance(model: NewtonModel, positions: Float32Array, direction: Float32Array, stepLength: number): void {
  model.rowVertex.forEach((vertex, row) => {
    [0, 1, 2].forEach((d) => { positions[3 * vertex + d] += stepLength * direction[3 * row + d]; });
  });
}

function pcgSettings(model: NewtonModel): SolveSettings {
  const { tolerance, max_iterations } = model.pcg;
  return { tolerance, maxIterations: max_iterations, chunkIterations: CHUNK_ITERATIONS };
}

export function record<A, K>(context: NewtonContext<A, K>, entry: IterationRecord, positions: Float32Array<ArrayBuffer>): void {
  context.records.push(entry);
  context.iterates?.push(Float32Array.from(positions));
}

export const terminal = (step: number, iteration: number, relativeResidual: number, status: StepStatus): IterationRecord =>
  ({ step, iteration, relativeResidual, pcgIterations: Number.NaN, pcgStatus: Number.NaN, linearResidual: Number.NaN, stepLength: Number.NaN, halvings: Number.NaN, status });

export const linearStatus = (run: SolveRun): number => ["running", "converged", "iteration_limit", "breakdown"].indexOf(run.status);

/** One Newton update from `positions`: E1 PCG on the assembled system, then the Armijo line search. */
async function update<A, K>(context: NewtonContext<A, K>, positions: Float32Array<ArrayBuffer>, rhs: Float32Array, where: { readonly step: number; readonly iteration: number; readonly relative: number }): Promise<boolean> {
  const run = await solveOnce(context.gpu.solver, pcgSettings(context.model));
  const direction = run.solution;
  context.gpu.backend.write(context.gpu.direction, scatter(context.model, direction));
  const search = await lineSearch(context, -dot32(rhs, direction));
  const status = search === null ? STEP_STATUS.line_search_failed : STEP_STATUS.running;
  record(context, { step: where.step, iteration: where.iteration, relativeResidual: where.relative, pcgIterations: run.iterations, pcgStatus: linearStatus(run),
    linearResidual: run.gpuTrueRelativeResidual, stepLength: search?.stepLength ?? Number.NaN, halvings: search?.halvings ?? Number.NaN, status }, positions);
  if (search !== null) advance(context.model, positions, direction, search.stepLength);
  return search !== null;
}

/** Newton from `positions` (updated in place) until ‖g‖/scale ≤ tolerance, the iteration limit, or a failed line search. */
export async function solveStep<A, K>(context: NewtonContext<A, K>, positions: Float32Array<ArrayBuffer>, scale: number, step: number): Promise<void> {
  const { tolerance, max_iterations } = context.model.newton;
  for (let iteration = 0; ; iteration += 1) {
    const rhs = await assemble(context.gpu, positions);
    const relative = Math.fround(norm32(rhs) / scale);
    if (relative <= tolerance || iteration === max_iterations) {
      record(context, terminal(step, iteration, relative, relative <= tolerance ? STEP_STATUS.converged : STEP_STATUS.iteration_limit), positions);
      return;
    }
    if (!(await update(context, positions, rhs, { step, iteration, relative }))) return;
  }
}

const couponModel = (problem: CouponProblem): NewtonModel =>
  ({ newton: problem.manifest.newton, pcg: problem.manifest.pcg, rowVertex: problem.u32.row_vertex, vertices: problem.manifest.vertices });

function stepGradient(problem: CouponProblem, step: number): readonly number[] {
  return step < 0 ? IDENTITY : Array.from(problem.f32.step_gradients.subarray(4 * step, 4 * step + 4));
}

/** Load-step predictor: adds (F(to) − F(from))·X to every vertex, as in the oracle. */
function predict(problem: CouponProblem, positions: Float32Array, step: number): void {
  const [from, to] = [stepGradient(problem, step - 1), stepGradient(problem, step)];
  const rest = problem.f32.rest_positions;
  for (let v = 0; v < problem.manifest.vertices; v += 1) {
    [0, 1].forEach((r) => {
      positions[3 * v + r] += (to[2 * r] - from[2 * r]) * rest[3 * v] + (to[2 * r + 1] - from[2 * r + 1]) * rest[3 * v + 1];
    });
  }
}

function applyBoundary(problem: CouponProblem, positions: Float32Array, step: number): void {
  const { fixed_vertex: fixed } = problem.u32;
  const offset = 3 * step * fixed.length;
  fixed.forEach((vertex, i) => positions.set(problem.f32.fixed_positions.subarray(offset + 3 * i, offset + 3 * i + 3), 3 * vertex));
}

/** ‖g‖ at the previous equilibrium with only the grips moved: the scale of this step's residual. */
async function stepScale<A, K>(gpu: NewtonGpu<A, K>, problem: CouponProblem, positions: Float32Array<ArrayBuffer>, step: number): Promise<number> {
  const gripped = Float32Array.from(positions);
  applyBoundary(problem, gripped, step);
  return norm32(await assemble(gpu, gripped));
}

/** Every load step from the previous f32 equilibrium plus the affine predictor; all iterates are kept for the oracle. */
export async function solvePath<A, K>(gpu: CouponGpu<A, K>, problem: CouponProblem): Promise<PathResult> {
  const context: NewtonContext<A, K> = { gpu, model: couponModel(problem), records: [], iterates: [] };
  const positions = Float32Array.from(problem.f32.initial_positions);
  const started = performance.now();
  for (let step = 0; step < problem.manifest.steps; step += 1) {
    const scale = await stepScale(gpu, problem, positions, step);
    predict(problem, positions, step);
    applyBoundary(problem, positions, step);
    await solveStep(context, positions, scale, step);
  }
  return { records: context.records, iterates: context.iterates ?? [], solveMs: performance.now() - started };
}

/** Element gradients and tangents at the oracle's kernel-check state, for the WGSL-vs-f64 element comparison. */
export async function evaluateKernelState<A, K>(gpu: CouponGpu<A, K>, problem: CouponProblem): Promise<Float32Array<ArrayBuffer>> {
  gpu.backend.write(gpu.positions, problem.f32.kernel_state);
  await gpu.backend.run(gpu.element);
  const [gradient, tangent] = await Promise.all([gpu.backend.read(gpu.elemGrad), gpu.backend.read(gpu.elemHess)]);
  const joined = new Float32Array(gradient.byteLength / 4 + tangent.byteLength / 4);
  joined.set(new Float32Array(gradient));
  joined.set(new Float32Array(tangent), gradient.byteLength / 4);
  return joined;
}
