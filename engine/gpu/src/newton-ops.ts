import type { Backend } from "./backend.ts";
import { assemble, dot32, norm32, type LineSearch, type NewtonGpu, type NewtonModel } from "./newton.ts";
import { decodeReduced, SLOT, type Reducers } from "./reductions.ts";
import { inPhase, type PhaseClock } from "./timed-backend.ts";

const f32 = Math.fround;
/** Noise-aware Armijo (Experiment 6): accept when Σδ ≤ c·α·gᵀd + NOISE_FACTOR·ε32·Σ|δ|, ε32 = 2^-23. */
export const NOISE_FACTOR = 4;
export const EPSILON32 = 2 ** -23;

/** Residual of the assembled system: its f32 norm, and (GPU reductions, probes) the reduction and the f64 sum of the same rhs. */
export interface Residual {
  readonly norm: number;
  readonly normSquared: number;
  readonly normSquaredExact: number;
}

/** gᵀd from the assembled rhs and the PCG solution; `exact` is the f64 sum of the same f32 values (probes with GPU reductions). */
export interface Slope {
  readonly value: number;
  readonly exact: number;
}

/** Σδ and Σ|δ| of the energy-change terms of one trial step. */
export interface EnergySums {
  readonly sum: number;
  readonly abs: number;
}

/** What the Newton loop asks of the GPU between decisions: residual norm, slope and energy change. The host variant reads the
 * full rhs and energy-change terms; the GPU variant reduces them on the device and reads a few floats. */
export interface NewtonOps {
  readonly residual: (positions: Float32Array<ArrayBuffer>) => Promise<Residual>;
  readonly slope: (solution: Float32Array<ArrayBuffer>) => Promise<Slope>;
  readonly energy: (stepLength: number) => Promise<EnergySums>;
}

const NOT_MEASURED = Number.NaN;

/** Experiments 3–5: the host holds the rhs, sums sequentially in f32, and reads every energy-change term. */
export function hostOps<A, K>(gpu: NewtonGpu<A, K>, clock: PhaseClock): NewtonOps {
  let rhs = new Float32Array(0);
  return {
    residual: async (positions) => {
      rhs = await inPhase(clock, "assembly", () => assemble(gpu, positions));
      return { norm: norm32(rhs), normSquared: NOT_MEASURED, normSquaredExact: NOT_MEASURED };
    },
    slope: async (solution) => ({ value: -dot32(rhs, solution), exact: NOT_MEASURED }),
    energy: async (stepLength) => hostEnergy(gpu, stepLength),
  };
}

async function hostEnergy<A, K>(gpu: NewtonGpu<A, K>, stepLength: number): Promise<EnergySums> {
  gpu.backend.write(gpu.step, Float32Array.of(stepLength));
  await gpu.backend.run(gpu.energyChange);
  const deltas = new Float32Array(await gpu.backend.read(gpu.elemDelta));
  return { sum: deltas.reduce((total, v) => f32(total + v), 0), abs: deltas.reduce((total, v) => f32(total + Math.abs(v)), 0) };
}

const sumOf = (values: Float32Array, other: Float32Array | null): number =>
  values.reduce((total, v, i) => total + v * (other === null ? v : other[i]), 0);

async function readReduced<A, K>(backend: Backend<A, K>, reduced: A): Promise<Float32Array<ArrayBuffer>> {
  return decodeReduced(await backend.read(reduced)).floats;
}

/** What the GPU variant needs from the device: the backend, the reducers, the phase clock and whether to read the f64 sums too. */
interface Device<A, K> {
  readonly gpu: NewtonGpu<A, K>;
  readonly reducers: Reducers<A, K>;
  readonly clock: PhaseClock;
  readonly verify: boolean;
}

const rhsOf = async <A, K>(d: Device<A, K>): Promise<Float32Array<ArrayBuffer>> => new Float32Array(await d.gpu.backend.read(d.gpu.solver.arrays.rhs));

async function gpuResidual<A, K>(d: Device<A, K>, positions: Float32Array<ArrayBuffer>): Promise<Residual> {
  const { backend } = d.gpu;
  backend.write(d.gpu.positions, positions);
  await inPhase(d.clock, "assembly", () => backend.run([...d.gpu.evaluate, ...d.reducers.rhsStats]));
  const normSquared = (await readReduced(backend, d.reducers.reduced))[SLOT.rhsNormSquared];
  const exact = d.verify ? sumOf(await rhsOf(d), null) : NOT_MEASURED;
  return { norm: f32(Math.sqrt(normSquared)), normSquared, normSquaredExact: exact };
}

async function gpuSlope<A, K>(d: Device<A, K>, solution: Float32Array<ArrayBuffer>): Promise<Slope> {
  await inPhase(d.clock, "reduce", () => d.gpu.backend.run(d.reducers.rhsStats));
  const value = -(await readReduced(d.gpu.backend, d.reducers.reduced))[SLOT.rhsDot];
  return { value, exact: d.verify ? -sumOf(await rhsOf(d), solution) : NOT_MEASURED };
}

async function gpuEnergy<A, K>(d: Device<A, K>, stepLength: number): Promise<EnergySums> {
  d.gpu.backend.write(d.gpu.step, Float32Array.of(stepLength));
  await d.gpu.backend.run([...d.gpu.energyChange, ...d.reducers.energySums]);
  const sums = await readReduced(d.gpu.backend, d.reducers.reduced);
  return { sum: sums[SLOT.deltaSum], abs: sums[SLOT.deltaAbs] };
}

/** Experiment 6, rungs C and D: reductions on the GPU, each submitted together with the kernels it follows (one submit per
 * measurement; the f64 sums of the same rhs are read as well when `verify`). */
export function gpuOps<A, K>(gpu: NewtonGpu<A, K>, reducers: Reducers<A, K>, clock: PhaseClock, verify: boolean): NewtonOps {
  const device = { gpu, reducers, clock, verify };
  return { residual: (positions) => gpuResidual(device, positions), slope: (solution) => gpuSlope(device, solution), energy: (stepLength) => gpuEnergy(device, stepLength) };
}

/** One tried step length with the numbers its decision rested on, for the oracle's acceptance replay. */
export interface Trial {
  readonly alpha: number;
  readonly sum: number;
  readonly abs: number;
  readonly slope: number;
  readonly accepted: boolean;
}

/** An accepted step length, and whether the noise term alone made it acceptable. */
export interface NoisySearch extends LineSearch {
  readonly noiseOnly: boolean;
  readonly sum: number;
  readonly abs: number;
}

/** Plain Armijo threshold c·α·gᵀd in f32. */
const armijoThreshold = (armijo: number, alpha: number, slope: number): number => f32(armijo * alpha * slope);

/** Backtracking from `initial`: accept when Σδ ≤ c·α·gᵀd, or with `noiseAware` also when Σδ ≤ c·α·gᵀd + 4·ε32·Σ|δ|;
 * null when d is not a descent direction or halving runs out. With noiseAware off this is exactly the E3–E5 search. */
export async function searchStep(ops: NewtonOps, model: NewtonModel, slope: number, initial: number, options: { readonly noiseAware: boolean; readonly onTrial: ((trial: Trial) => void) | null }): Promise<NoisySearch | null> {
  const { armijo, max_halvings } = model.newton;
  if (!(slope < 0)) return null;
  for (let halvings = 0; halvings <= max_halvings; halvings += 1) {
    const alpha = initial * 0.5 ** halvings;
    const { sum, abs } = await ops.energy(alpha);
    const plain = sum <= armijoThreshold(armijo, alpha, slope);
    const accepted = plain || (options.noiseAware && sum <= armijoThreshold(armijo, alpha, slope) + NOISE_FACTOR * EPSILON32 * abs);
    options.onTrial?.({ alpha, sum, abs, slope, accepted });
    if (accepted) return { stepLength: alpha, halvings, noiseOnly: !plain, sum, abs };
  }
  return null;
}
