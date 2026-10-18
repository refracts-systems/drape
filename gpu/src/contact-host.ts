import type { DrapeProblem } from "./drape.ts";
import type { DrapeGpu } from "./drape-gpu.ts";
import { decodeReduced, SLOT, type Reducers } from "./reductions.ts";
import type { ContactHooks } from "./robust-newton.ts";

const HEADER = 4;
const STRIDE = 8;
const PLANE = 1;

/** Distance of `x` to collider k of the packed array, in f64: plane n·x − w, sphere |x − c| − w. */
function distance(colliders: Float32Array, k: number, x: readonly number[]): number {
  const base = HEADER + STRIDE * k;
  const [a, b, c, w] = [colliders[base + 1], colliders[base + 2], colliders[base + 3], colliders[base + 4]];
  if (colliders[base] === PLANE) return a * x[0] + b * x[1] + c * x[2] - w;
  return Math.hypot(x[0] - a, x[1] - b, x[2] - c) - w;
}

/** Smallest collider distance over the unknowns, evaluated in f64 from f32 positions. */
export function minDistance(problem: DrapeProblem, positions: Float32Array): number {
  const { colliders, count } = problem.contact;
  return Array.from(problem.u32.row_vertex).reduce((smallest, v) => {
    const x = [positions[3 * v], positions[3 * v + 1], positions[3 * v + 2]];
    return Array.from({ length: count }, (_, k) => distance(colliders, k, x)).reduce((m, d) => Math.min(m, d), smallest);
  }, Number.POSITIVE_INFINITY);
}

/** The minimum per-row bound of the CCD kernel and the vertex of its row (reads every per-row bound). */
async function maxStep<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem): Promise<{ readonly alpha: number; readonly vertex: number }> {
  await gpu.backend.run(gpu.ccd);
  const bounds = new Float32Array(await gpu.backend.read(gpu.bounds));
  const row = bounds.reduce((best, value, i) => (value < bounds[best] ? i : best), 0);
  return { alpha: bounds[row], vertex: bounds[row] < 1 ? problem.u32.row_vertex[row] : -1 };
}

/** The same minimum reduced on the GPU: (α_max, row) read back as 32 bytes. */
async function maxStepOnGpu<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem, reducers: Reducers<A, K>): Promise<{ readonly alpha: number; readonly vertex: number }> {
  await gpu.backend.run([...gpu.ccd, ...reducers.boundMin]);
  const { floats, boundRow } = decodeReduced(await gpu.backend.read(reducers.reduced));
  const alpha = floats[SLOT.bound];
  return { alpha, vertex: alpha < 1 ? problem.u32.row_vertex[boundRow] : -1 };
}

/** Smallest vertex-collider distance of `positions` in f32, reduced on the GPU. */
async function minDistanceOnGpu<A, K>(gpu: DrapeGpu<A, K>, reducers: Reducers<A, K>, positions: Float32Array<ArrayBuffer>): Promise<number> {
  gpu.backend.write(gpu.positions, positions);
  await gpu.backend.run([...gpu.ccd, ...reducers.distanceMin]);
  return decodeReduced(await gpu.backend.read(reducers.reduced)).floats[SLOT.distance];
}

/** Collider hooks for the robust Newton, or null when the problem has no colliders (Experiments 3 and 4). With `gpuMinima`
 * the bound minimum and the per-iterate minimum distance are reduced on the GPU (Experiment 6, rung D); otherwise the host
 * reads every bound and evaluates the distance in f64. */
export function contactHooks<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem, gpuMinima = false): ContactHooks | null {
  if (problem.contact.count === 0) return null;
  const reducers = gpu.reducers;
  if (gpuMinima && reducers === null) throw new Error("GPU minima need the reduction kernels (prepareDrapeGpu withReducers)");
  if (gpuMinima && reducers !== null) {
    return { maxStep: () => maxStepOnGpu(gpu, problem, reducers), minDistance: (positions) => minDistanceOnGpu(gpu, reducers, positions) };
  }
  return { maxStep: () => maxStep(gpu, problem), minDistance: async (positions) => minDistance(problem, positions) };
}

/** Runs `work` with the collider count set to 0 (kernel checks at states that are not collision-free), then restores it. */
export async function withoutColliders<A, K, T>(gpu: DrapeGpu<A, K>, problem: DrapeProblem, work: () => Promise<T>): Promise<T> {
  const disabled = Float32Array.from(problem.contact.colliders);
  disabled[0] = 0;
  gpu.backend.write(gpu.colliders, disabled);
  try {
    return await work();
  } finally {
    gpu.backend.write(gpu.colliders, Float32Array.from(problem.contact.colliders));
  }
}

/** Barrier gradient and diagonal blocks at the contact state: x̂ = x, μ = 0 and row masses 0 isolate the contact terms. */
export async function evaluateContactKernel<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem, offsets: { readonly grad: number; readonly hess: number }): Promise<Float32Array<ArrayBuffer>> {
  const state = problem.contact.state;
  if (state === null) throw new Error("problem.bin: colliders without contact_state");
  const rows = problem.manifest.rows;
  [gpu.positions, gpu.xHat].forEach((target) => gpu.backend.write(target, state));
  gpu.backend.write(gpu.regularization, Float32Array.of(0));
  gpu.backend.write(gpu.rowMass, new Float32Array(rows));
  await gpu.backend.run(gpu.inertia);
  gpu.backend.write(gpu.rowMass, problem.f32.row_mass);
  const [gradient, hessian] = await Promise.all([gpu.backend.read(gpu.elemGrad), gpu.backend.read(gpu.elemHess)]);
  const joined = new Float32Array(12 * rows);
  joined.set(new Float32Array(gradient, 4 * offsets.grad, 3 * rows));
  joined.set(new Float32Array(hessian, 4 * offsets.hess, 9 * rows), 3 * rows);
  return joined;
}

/** Experiment 7: header[3] of the collider buffer switches the b′∇²d term of the barrier Hessian off (drop) or on. */
export function setBarrierCurvature<A, K>(gpu: DrapeGpu<A, K>, problem: DrapeProblem, drop: boolean): void {
  const header = Float32Array.from(problem.contact.colliders);
  header[3] = drop ? 1 : 0;
  gpu.backend.write(gpu.colliders, header);
}
