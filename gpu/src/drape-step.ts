import type { DrapeProblem } from "./drape.ts";
import type { DrapeGpu } from "./drape-gpu.ts";
import { assemble, norm32, solveStep, type IterationRecord, type NewtonContext, type NewtonModel } from "./newton.ts";

/** Positions and velocities of one f32 state. */
interface State {
  readonly positions: Float32Array<ArrayBuffer>;
  readonly velocity: Float32Array<ArrayBuffer>;
}

export interface TrajectoryResult {
  readonly records: readonly IterationRecord[];
  readonly positions: readonly Float32Array<ArrayBuffer>[];
  readonly ms: number;
}

export interface ProbeResult {
  readonly records: readonly IterationRecord[];
  readonly iterates: readonly Float32Array<ArrayBuffer>[];
}

const drapeModel = (problem: DrapeProblem): NewtonModel =>
  ({ newton: problem.manifest.newton, pcg: problem.manifest.pcg, rowVertex: problem.u32.row_vertex, vertices: problem.manifest.vertices });

/** x̂ = x + h·v + h²·g on free vertices in f32 arithmetic (same order as the oracle); pinned vertices stay. */
function prediction(problem: DrapeProblem, state: State): Float32Array<ArrayBuffer> {
  const h = Math.fround(problem.manifest.time_step_s);
  const fall = Math.fround(Math.fround(h * h) * Math.fround(-problem.manifest.gravity_m_per_s2));
  const predicted = Float32Array.from(state.positions);
  problem.u32.row_vertex.forEach((vertex) => {
    [0, 1, 2].forEach((d) => {
      const i = 3 * vertex + d;
      const drift = Math.fround(state.positions[i] + Math.fround(h * state.velocity[i]));
      predicted[i] = d === 2 ? drift + fall : drift;
    });
  });
  return predicted;
}

/** Backward-Euler velocity (x_{n+1} − x_n)/h, rounded to f32. */
function velocityOf(next: Float32Array, previous: Float32Array, timeStep: number): Float32Array<ArrayBuffer> {
  return Float32Array.from(next, (value, i) => (value - previous[i]) / timeStep);
}

/** One time step: write x̂, scale by ‖∇Φ(x_n)‖, Newton on Φ from x̂. */
async function timeStep<A, K>(context: NewtonContext<A, K>, gpu: DrapeGpu<A, K>, problem: DrapeProblem, state: State, step: number): Promise<State> {
  const xHat = prediction(problem, state);
  gpu.backend.write(gpu.xHat, xHat);
  const scale = norm32(await assemble(gpu, state.positions));
  const next = Float32Array.from(xHat);
  await solveStep(context, next, scale, step);
  return { positions: next, velocity: velocityOf(next, state.positions, Math.fround(problem.manifest.time_step_s)) };
}

/** The full f32 trajectory from rest: every step's Newton records and every state's positions. */
export async function simulateDrape<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem): Promise<TrajectoryResult> {
  const context: NewtonContext<A, K> = { gpu, model: drapeModel(problem), records: [], iterates: null };
  const rest = Float32Array.from(problem.f32.rest_positions);
  const positions: Float32Array<ArrayBuffer>[] = [rest];
  const started = performance.now();
  for (let step = 0, state: State = { positions: rest, velocity: new Float32Array(rest.length) }; step < problem.manifest.steps; step += 1) {
    state = await timeStep(context, gpu, problem, state, step + 1);
    positions.push(state.positions);
  }
  return { records: context.records, positions, ms: performance.now() - started };
}

function probeState(problem: DrapeProblem, probe: number): State {
  const size = 3 * problem.manifest.vertices;
  const slice = (data: Float32Array<ArrayBuffer>): Float32Array<ArrayBuffer> => Float32Array.from(data.subarray(probe * size, (probe + 1) * size));
  return { positions: slice(problem.f32.probe_positions), velocity: slice(problem.f32.probe_velocities) };
}

/** One GPU step from each f64 probe state (state k−1 of the oracle trajectory), keeping every iterate. */
export async function runProbes<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem): Promise<ProbeResult> {
  const context: NewtonContext<A, K> = { gpu, model: drapeModel(problem), records: [], iterates: [] };
  for (let probe = 0; probe < problem.manifest.probes; probe += 1) {
    await timeStep(context, gpu, problem, probeState(problem, probe), problem.u32.probe_steps[probe]);
  }
  return { records: context.records, iterates: context.iterates ?? [] };
}

async function elementArrays<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem, exact: boolean): Promise<readonly Float32Array<ArrayBuffer>[]> {
  const { triangles: t, hinges: h } = problem.manifest;
  gpu.backend.write(gpu.positions, problem.f32.kernel_state);
  await gpu.backend.run(exact ? gpu.exactElements : gpu.projectedElements);
  const [gradient, hessian] = await Promise.all([gpu.backend.read(gpu.elemGrad), gpu.backend.read(gpu.elemHess)]);
  return [new Float32Array(gradient, 0, 9 * t + 12 * h), new Float32Array(hessian, 0, 81 * t + 144 * h)];
}

/** Membrane and hinge gradients and Hessians at the oracle's kernel state: exact kernels, then projected ones. */
export async function evaluateDrapeKernels<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem): Promise<Float32Array<ArrayBuffer>> {
  const parts = [...await elementArrays(gpu, problem, true), ...await elementArrays(gpu, problem, false)];
  const joined = new Float32Array(parts.reduce((length, part) => length + part.length, 0));
  parts.reduce((offset, part) => (joined.set(part, offset), offset + part.length), 0);
  return joined;
}
