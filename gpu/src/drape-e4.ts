import type { DrapeProblem } from "./drape.ts";
import type { DrapeGpu } from "./drape-gpu.ts";
import { drapeModel, evaluateDrapeKernels, prediction, velocityOf, type State } from "./drape-step.ts";
import { assemble, norm32, type IterationRecord } from "./newton.ts";
import { solveStepRobust, type RobustContext, type RobustRecord, type RobustRules, type RobustSettings } from "./robust-newton.ts";
import { inPhase, type Phase, type PhaseClock } from "./timed-backend.ts";
import { contactHooks, withoutColliders } from "./contact-host.ts";

export interface RobustRun {
  readonly records: readonly IterationRecord[];
  readonly extras: readonly RobustRecord[];
  readonly iterates: readonly Float32Array<ArrayBuffer>[];
  /** Probes only: x_k and p_k of every update, alternating. */
  readonly directions: readonly Float32Array<ArrayBuffer>[];
}

export interface RobustTrajectory extends RobustRun {
  readonly positions: readonly Float32Array<ArrayBuffer>[];
  readonly ms: number;
  readonly phases: Readonly<Record<Phase | "io", number>>;
}

function context<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem, rules: RobustRules, settings: RobustSettings, clock: PhaseClock, keep: boolean): RobustContext<A, K> {
  return { gpu, model: drapeModel(problem), records: [], iterates: keep ? [] : null, rules, settings, extras: [], clock, contact: contactHooks(gpu, problem), directions: keep ? [] : null };
}

/** One time step with the robust Newton: write x̂, scale by ‖∇Φ(x_n)‖, Newton from x̂ (cut back to x_n + α_max·(x̂ − x_n)
 * with colliders) with x_n as origin. */
async function timeStep<A, K>(ctx: RobustContext<A, K>, gpu: DrapeGpu<A, K>, problem: DrapeProblem, state: State, step: number): Promise<State> {
  const xHat = prediction(problem, state);
  gpu.backend.write(gpu.xHat, xHat);
  gpu.backend.write(gpu.regularization, Float32Array.of(0));
  const scale = norm32(await inPhase(ctx.clock, "assembly", () => assemble(gpu, state.positions)));
  const next = await newtonStart(ctx, gpu, state.positions, xHat);
  await solveStepRobust(ctx, next, scale, step, state.positions);
  return { positions: next, velocity: velocityOf(next, state.positions, Math.fround(problem.manifest.time_step_s)) };
}

/** x̂ without colliders; with colliders x_n + α_max·(x̂ − x_n), α_max from the CCD kernel at x_n (already on the GPU). */
async function newtonStart<A, K>(ctx: RobustContext<A, K>, gpu: DrapeGpu<A, K>, origin: Float32Array, xHat: Float32Array<ArrayBuffer>): Promise<Float32Array<ArrayBuffer>> {
  const contact = ctx.contact;
  if (contact === null) return Float32Array.from(xHat);
  const towards = Float32Array.from(xHat, (value, i) => value - origin[i]);
  gpu.backend.write(gpu.direction, towards);
  const { alpha } = await inPhase(ctx.clock, "ccd", () => contact.maxStep());
  return Float32Array.from(origin, (value, i) => value + alpha * towards[i]);
}

/** Mean per-step wall clock: GPU submits per phase, host I/O, and the remainder ("other"). */
function phaseMeans(clock: PhaseClock, totalMs: number, steps: number): Readonly<Record<Phase | "io", number>> {
  const { assembly, pcg, line_search, ccd } = clock.runMs;
  const other = totalMs - assembly - pcg - line_search - ccd - clock.ioMs;
  return { assembly: assembly / steps, pcg: pcg / steps, line_search: line_search / steps, ccd: ccd / steps, io: clock.ioMs / steps, other: other / steps };
}

function resetClock(clock: PhaseClock): void {
  (["assembly", "pcg", "line_search", "ccd", "other"] as const).forEach((phase) => { clock.runMs[phase] = 0; });
  clock.ioMs = 0;
}

/** The full f32 trajectory from rest with the robust Newton, timed per phase. */
export async function simulateRobust<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem, rules: RobustRules, settings: RobustSettings, clock: PhaseClock): Promise<RobustTrajectory> {
  const ctx = context(gpu, problem, rules, settings, clock, false);
  const rest = Float32Array.from(problem.f32.rest_positions);
  const positions: Float32Array<ArrayBuffer>[] = [rest];
  resetClock(clock);
  const started = performance.now();
  for (let step = 0, state: State = { positions: rest, velocity: new Float32Array(rest.length) }; step < problem.manifest.steps; step += 1) {
    state = await timeStep(ctx, gpu, problem, state, step + 1);
    positions.push(state.positions);
  }
  const ms = performance.now() - started;
  return { records: ctx.records, extras: ctx.extras, iterates: [], directions: [], positions, ms, phases: phaseMeans(clock, ms, problem.manifest.steps) };
}

function probeState(problem: DrapeProblem, probe: number): State {
  const size = 3 * problem.manifest.vertices;
  const slice = (data: Float32Array<ArrayBuffer>): Float32Array<ArrayBuffer> => Float32Array.from(data.subarray(probe * size, (probe + 1) * size));
  return { positions: slice(problem.f32.probe_positions), velocity: slice(problem.f32.probe_velocities) };
}

/** One robust GPU step from each f64 probe state, keeping every iterate for the oracle. */
export async function runRobustProbes<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem, rules: RobustRules, settings: RobustSettings, clock: PhaseClock): Promise<RobustRun> {
  const ctx = context(gpu, problem, rules, settings, clock, true);
  for (let probe = 0; probe < problem.manifest.probes; probe += 1) {
    await timeStep(ctx, gpu, problem, probeState(problem, probe), problem.u32.probe_steps[probe]);
  }
  return { records: ctx.records, extras: ctx.extras, iterates: ctx.iterates ?? [], directions: ctx.directions ?? [] };
}

/** The E3 element-kernel check plus the inertia kernel (colliders off) at x = kernel state, x̂ = rest, μ = kernel_mu: [gradient 3R | Hessian 9R]. */
export async function evaluateKernelsWithInertia<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem, rules: RobustRules): Promise<Float32Array<ArrayBuffer>> {
  const elements = await evaluateDrapeKernels(gpu, problem);
  const { triangles: t, hinges: h, rows } = problem.manifest;
  gpu.backend.write(gpu.positions, problem.f32.kernel_state);
  gpu.backend.write(gpu.xHat, problem.f32.rest_positions);
  gpu.backend.write(gpu.regularization, Float32Array.of(rules.kernel_mu));
  await withoutColliders(gpu, problem, () => gpu.backend.run(gpu.inertia));
  const [gradient, hessian] = await Promise.all([gpu.backend.read(gpu.elemGrad), gpu.backend.read(gpu.elemHess)]);
  const parts = [elements, new Float32Array(gradient, 4 * (9 * t + 12 * h), 3 * rows), new Float32Array(hessian, 4 * (81 * t + 144 * h), 9 * rows)];
  const joined = new Float32Array(parts.reduce((length, part) => length + part.length, 0));
  parts.reduce((offset, part) => (joined.set(part, offset), offset + part.length), 0);
  return joined;
}
