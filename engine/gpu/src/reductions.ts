import type { Backend, Dispatch, KernelSpec } from "./backend.ts";
import type { KernelSources } from "./kernels.ts";

const WORKGROUP = 256;
/** Pass-0 workgroups of a reduction; the final pass adds this many partial pairs in one workgroup. */
const MAX_GROUPS = 128;

/** Floats of the `reduced` buffer: [Σrhs², Σrhs·x, Σδ, Σ|δ|, min bound, its row (u32 bits), min distance, its row (u32 bits)]. */
const RESULT_FLOATS = 8;

/** Where each result sits in the `reduced` buffer. */
export const SLOT = { rhsNormSquared: 0, rhsDot: 1, deltaSum: 2, deltaAbs: 3, bound: 4, boundRow: 5, distance: 6, distanceRow: 7 } as const;

/** GPU reductions for the Newton host loop: sums over the residual, the PCG solution and the energy-change terms, and arg-min
 * over the CCD bounds and distances. Each result is a few floats in one buffer, read back in 32 bytes. */
export interface Reducers<A, K> {
  readonly reduced: A;
  /** (Σ rhs², Σ rhs·x) from the assembled rhs and the PCG solution. */
  readonly rhsStats: readonly Dispatch<K>[];
  /** (Σ δ, Σ |δ|) over the energy-change terms. */
  readonly energySums: readonly Dispatch<K>[];
  /** Smallest CCD bound and its row. */
  readonly boundMin: readonly Dispatch<K>[];
  /** Smallest vertex-collider distance and its row. */
  readonly distanceMin: readonly Dispatch<K>[];
}

/** The arrays the reductions read: rhs and solution of the solver, the energy-change terms, and the CCD outputs. */
export interface ReductionInputs<A> {
  readonly rhs: A;
  readonly solution: A;
  readonly deltas: A;
  readonly bounds: A;
  readonly distances: A;
  readonly deltaLength: number;
  readonly vectorLength: number;
  readonly rows: number;
}

const groupsFor = (length: number): number => Math.max(1, Math.min(MAX_GROUPS, Math.ceil(length / WORKGROUP)));

function sumSpecs<A>(sources: KernelSources, inputs: ReductionInputs<A>, buffers: { readonly partials: A; readonly reduced: A }): readonly KernelSpec<A>[] {
  const spec = (label: string, a: A, mode: number, pass: number, offset: number, groups: number): KernelSpec<A> =>
    ({ label, source: sources.sums, constants: { MODE: mode, PASS: pass, OUT_OFFSET: offset, GROUPS: groups },
      bindings: { a, b: mode === 0 ? inputs.solution : a, partials: buffers.partials, out: buffers.reduced } });
  const rhsGroups = groupsFor(inputs.vectorLength);
  const deltaGroups = groupsFor(inputs.deltaLength);
  return [
    spec("sums_rhs_partial", inputs.rhs, 0, 0, SLOT.rhsNormSquared, rhsGroups), spec("sums_rhs_final", inputs.rhs, 0, 1, SLOT.rhsNormSquared, rhsGroups),
    spec("sums_delta_partial", inputs.deltas, 1, 0, SLOT.deltaSum, deltaGroups), spec("sums_delta_final", inputs.deltas, 1, 1, SLOT.deltaSum, deltaGroups),
  ];
}

function argminSpecs<A>(sources: KernelSources, inputs: ReductionInputs<A>, buffers: { readonly partials: A; readonly reduced: A }): readonly KernelSpec<A>[] {
  const groups = groupsFor(inputs.rows);
  const spec = (label: string, values: A, pass: number, offset: number): KernelSpec<A> =>
    ({ label, source: sources.argmin, constants: { PASS: pass, OUT_OFFSET: offset, GROUPS: groups },
      bindings: { values, partials: buffers.partials, out: buffers.reduced } });
  return [
    spec("argmin_bound_partial", inputs.bounds, 0, SLOT.bound), spec("argmin_bound_final", inputs.bounds, 1, SLOT.bound),
    spec("argmin_distance_partial", inputs.distances, 0, SLOT.distance), spec("argmin_distance_final", inputs.distances, 1, SLOT.distance),
  ];
}

/** Compiles the reduction kernels and fixes their dispatch lists. */
export async function prepareReducers<A, K>(backend: Backend<A, K>, sources: KernelSources, inputs: ReductionInputs<A>): Promise<Reducers<A, K>> {
  const reduced = backend.array("reduced", new Float32Array(RESULT_FLOATS), "read-write");
  const partials = backend.array("reduction_partials", new Float32Array(2 * MAX_GROUPS), "read-write");
  const buffers = { partials, reduced };
  const kernels = await Promise.all([...sumSpecs(sources, inputs, buffers), ...argminSpecs(sources, inputs, buffers)].map((spec) => backend.kernel(spec)));
  const [rhsPartial, rhsFinal, deltaPartial, deltaFinal, boundPartial, boundFinal, distancePartial, distanceFinal] = kernels;
  const pass = (partial: K, final: K, groups: number): readonly Dispatch<K>[] => [{ kernel: partial, workgroups: groups }, { kernel: final, workgroups: 1 }];
  return {
    reduced,
    rhsStats: pass(rhsPartial, rhsFinal, groupsFor(inputs.vectorLength)),
    energySums: pass(deltaPartial, deltaFinal, groupsFor(inputs.deltaLength)),
    boundMin: pass(boundPartial, boundFinal, groupsFor(inputs.rows)),
    distanceMin: pass(distancePartial, distanceFinal, groupsFor(inputs.rows)),
  };
}

/** The reduced buffer as floats, with its two index slots decoded. */
export interface ReducedValues {
  readonly floats: Float32Array<ArrayBuffer>;
  readonly boundRow: number;
  readonly distanceRow: number;
}

export function decodeReduced(bytes: ArrayBuffer): ReducedValues {
  const floats = new Float32Array(bytes);
  const words = new Uint32Array(bytes);
  return { floats, boundRow: words[SLOT.boundRow], distanceRow: words[SLOT.distanceRow] };
}
