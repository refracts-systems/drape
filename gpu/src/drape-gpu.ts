import type { Backend, Dispatch, KernelSpec } from "./backend.ts";
import type { DrapeProblem } from "./drape.ts";
import type { KernelSources } from "./kernels.ts";
import type { NewtonGpu } from "./newton.ts";
import type { Problem } from "./problem.ts";
import { prepareSolver, type PreparedSolver } from "./solver.ts";

const WORKGROUP = 256;

/** The E2 membrane and assembly kernels plus hinge and inertia kernels, all writing one shared set of local arrays. */
export interface DrapeGpu<A, K> extends NewtonGpu<A, K> {
  readonly xHat: A;
  readonly regularization: A;
  readonly colliders: A;
  readonly bounds: A;
  readonly rowMass: A;
  readonly ccd: readonly Dispatch<K>[];
  readonly elemGrad: A;
  readonly elemHess: A;
  readonly inertia: readonly Dispatch<K>[];
  readonly exactElements: readonly Dispatch<K>[];
  readonly projectedElements: readonly Dispatch<K>[];
}

interface DrapeArrays<A> {
  readonly positions: A; readonly xHat: A; readonly direction: A; readonly step: A;
  readonly elemVertices: A; readonly elemGeom: A; readonly hingeVertices: A; readonly hingeStiffness: A;
  readonly rowVertex: A; readonly rowMass: A; readonly dofMask: A;
  readonly blockPtr: A; readonly blockIdx: A; readonly forcePtr: A; readonly forceIdx: A;
  readonly elemGrad: A; readonly elemHess: A; readonly elemDelta: A; readonly regularization: A;
  readonly colliders: A; readonly bounds: A;
}

/** Offsets (in floats) of the hinge and inertia sections behind the membrane entries of the shared local arrays. */
interface Layout {
  readonly hingeGrad: number; readonly inertiaGrad: number;
  readonly hingeHess: number; readonly inertiaHess: number;
  readonly hingeDelta: number; readonly inertiaDelta: number;
}

function layout(problem: DrapeProblem): Layout {
  const { triangles: t, hinges: h } = problem.manifest;
  return { hingeGrad: 9 * t, inertiaGrad: 9 * t + 12 * h, hingeHess: 81 * t, inertiaHess: 81 * t + 144 * h, hingeDelta: t, inertiaDelta: t + h };
}

function linearProblem(problem: DrapeProblem): Problem {
  const m = problem.manifest;
  return {
    manifest: { case: m.case, grid_columns: m.grid_columns, rows: m.rows, unknowns: m.unknowns, blocks: m.blocks,
      replace_every: m.pcg.replace_every, replace_above: m.pcg.replace_above, arrays: [] },
    rowPtr: problem.u32.row_ptr, colIdx: problem.u32.col_idx, diagIdx: problem.u32.diag_idx,
    vals: new Float32Array(9 * m.blocks), rhs: new Float32Array(m.unknowns),
  };
}

function drapeArrays<A, K>(backend: Backend<A, K>, problem: DrapeProblem): DrapeArrays<A> {
  const { f32, u32, manifest: m } = problem;
  const scratch = (label: string, length: number): A => backend.array(label, new Float32Array(length), "read-write");
  const fixed = <T extends Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>>(label: string, data: T): A => backend.array(label, data, "read");
  return {
    positions: backend.array("positions", Float32Array.from(f32.rest_positions), "read-write"), xHat: scratch("x_hat", 3 * m.vertices),
    direction: scratch("direction", 3 * m.vertices), step: scratch("step", 1),
    elemVertices: fixed("elem_vertices", u32.elem_vertices), elemGeom: fixed("elem_geom", f32.elem_geom),
    hingeVertices: fixed("hinge_vertices", u32.hinge_vertices), hingeStiffness: fixed("hinge_stiffness", f32.hinge_stiffness),
    rowVertex: fixed("row_vertex", u32.row_vertex), rowMass: fixed("row_mass", f32.row_mass), dofMask: fixed("dof_mask", u32.dof_mask),
    blockPtr: fixed("block_contrib_ptr", u32.block_contrib_ptr), blockIdx: fixed("block_contrib_idx", u32.block_contrib_idx),
    forcePtr: fixed("force_contrib_ptr", u32.force_contrib_ptr), forceIdx: fixed("force_contrib_idx", u32.force_contrib_idx),
    elemGrad: scratch("elem_grad", 9 * m.triangles + 12 * m.hinges + 3 * m.rows),
    elemHess: scratch("elem_hess", 81 * m.triangles + 144 * m.hinges + 9 * m.rows),
    elemDelta: scratch("elem_delta", m.triangles + m.hinges + m.rows),
    regularization: backend.array("regularization", new Float32Array(1), "read"),
    colliders: backend.array("colliders", Float32Array.from(problem.contact.colliders), "read"), bounds: scratch("ccd_bounds", m.rows),
  };
}

function materialConstants(problem: DrapeProblem): Record<string, number> {
  const m = problem.manifest.material;
  return { WALE: m.wale_N_per_m, COURSE: m.course_N_per_m, COUPLING: m.coupling_N_per_m, SHEAR: m.shear_N_per_m };
}

function membraneSpec<A>(sources: KernelSources, problem: DrapeProblem, a: DrapeArrays<A>, projected: boolean): KernelSpec<A> {
  return { label: projected ? "membrane_element_projected" : "membrane_element_exact", source: sources.membrane_element,
    constants: { ...materialConstants(problem), PROJECTED: Number(projected) },
    bindings: { positions: a.positions, elem_vertices: a.elemVertices, elem_geom: a.elemGeom, elem_grad: a.elemGrad, elem_hess: a.elemHess } };
}

function hingeSpec<A>(sources: KernelSources, problem: DrapeProblem, a: DrapeArrays<A>, exact: boolean): KernelSpec<A> {
  const offsets = layout(problem);
  return { label: exact ? "hinge_element_exact" : "hinge_element_gauss_newton", source: sources.hinge_element,
    constants: { GRAD_OFFSET: offsets.hingeGrad, HESS_OFFSET: offsets.hingeHess, EXACT: Number(exact) },
    bindings: { positions: a.positions, hinge_vertices: a.hingeVertices, hinge_stiffness: a.hingeStiffness, elem_grad: a.elemGrad, elem_hess: a.elemHess } };
}

function inertiaSpec<A>(sources: KernelSources, problem: DrapeProblem, a: DrapeArrays<A>): KernelSpec<A> {
  const offsets = layout(problem);
  return { label: "inertia", source: sources.inertia, constants: { GRAD_OFFSET: offsets.inertiaGrad, HESS_OFFSET: offsets.inertiaHess },
    bindings: { positions: a.positions, x_hat: a.xHat, row_vertex: a.rowVertex, row_mass: a.rowMass, elem_grad: a.elemGrad, elem_hess: a.elemHess, regularization: a.regularization, colliders: a.colliders } };
}

function assemblySpecs<A, K>(sources: KernelSources, a: DrapeArrays<A>, solver: PreparedSolver<A, K>): readonly KernelSpec<A>[] {
  const s = solver.arrays;
  return [
    { label: "assemble_matrix", source: sources.assemble_matrix, constants: {},
      bindings: { row_ptr: s.rowPtr, col_idx: s.colIdx, dof_mask: a.dofMask, block_contrib_ptr: a.blockPtr, block_contrib_idx: a.blockIdx, elem_hess: a.elemHess, vals: s.vals } },
    { label: "assemble_force", source: sources.assemble_force, constants: {},
      bindings: { dof_mask: a.dofMask, force_contrib_ptr: a.forcePtr, force_contrib_idx: a.forceIdx, elem_grad: a.elemGrad, rhs: s.rhs } },
  ];
}

function energySpecs<A>(sources: KernelSources, problem: DrapeProblem, a: DrapeArrays<A>): readonly KernelSpec<A>[] {
  const offsets = layout(problem);
  return [
    { label: "membrane_energy_change", source: sources.membrane_energy_change, constants: materialConstants(problem),
      bindings: { positions: a.positions, direction: a.direction, elem_vertices: a.elemVertices, elem_geom: a.elemGeom, step: a.step, elem_delta: a.elemDelta } },
    { label: "hinge_energy_change", source: sources.hinge_energy_change, constants: { DELTA_OFFSET: offsets.hingeDelta },
      bindings: { positions: a.positions, direction: a.direction, hinge_vertices: a.hingeVertices, hinge_stiffness: a.hingeStiffness, step: a.step, elem_delta: a.elemDelta } },
    { label: "inertia_energy_change", source: sources.inertia_energy_change, constants: { DELTA_OFFSET: offsets.inertiaDelta },
      bindings: { positions: a.positions, x_hat: a.xHat, direction: a.direction, row_vertex: a.rowVertex, row_mass: a.rowMass, step: a.step, elem_delta: a.elemDelta, colliders: a.colliders } },
  ];
}

/** The per-row line-search bound α ≤ α_max of the colliders; KEEP comes from problem.json when there are colliders. */
function ccdSpec<A>(sources: KernelSources, problem: DrapeProblem, a: DrapeArrays<A>): KernelSpec<A> {
  const keep = problem.manifest.keep_fraction;
  if (problem.contact.count > 0 && !Number.isFinite(keep)) throw new Error("problem.json: colliders without contact.keep_fraction");
  return { label: "ccd", source: sources.ccd, constants: problem.contact.count > 0 ? { KEEP: keep } : {},
    bindings: { positions: a.positions, direction: a.direction, row_vertex: a.rowVertex, colliders: a.colliders, bounds: a.bounds } };
}

/** Elements, hinges, rows: the workgroup counts of the three kinds of kernel. */
function workgroups(problem: DrapeProblem): { readonly elements: number; readonly hinges: number; readonly rows: number } {
  const count = (n: number): number => Math.ceil(n / WORKGROUP);
  return { elements: count(problem.manifest.triangles), hinges: count(problem.manifest.hinges), rows: count(problem.manifest.rows) };
}

/** `projected` picks the simulation tangent (clamped membrane + Gauss–Newton hinge); it defaults to the problem's own kind. */
export async function prepareDrapeGpu<A, K>(backend: Backend<A, K>, problem: DrapeProblem, sources: KernelSources, projected = problem.manifest.tangent === "projected"): Promise<DrapeGpu<A, K>> {
  const solver = await prepareSolver(backend, linearProblem(problem), sources);
  const a = drapeArrays(backend, problem);
  const specs = [membraneSpec(sources, problem, a, projected), hingeSpec(sources, problem, a, !projected), inertiaSpec(sources, problem, a),
    ...assemblySpecs(sources, a, solver), ...energySpecs(sources, problem, a),
    membraneSpec(sources, problem, a, false), hingeSpec(sources, problem, a, true), membraneSpec(sources, problem, a, true), hingeSpec(sources, problem, a, false),
    ccdSpec(sources, problem, a)];
  const [membrane, hinge, inertia, matrix, force, membraneDelta, hingeDelta, inertiaDelta, membraneExact, hingeExact, membraneProjected, hingeGaussNewton, ccd] =
    await Promise.all(specs.map((spec) => backend.kernel(spec)));
  const n = workgroups(problem);
  return {
    backend, solver, positions: a.positions, xHat: a.xHat, regularization: a.regularization, colliders: a.colliders, bounds: a.bounds, rowMass: a.rowMass, direction: a.direction, step: a.step, elemGrad: a.elemGrad, elemHess: a.elemHess, elemDelta: a.elemDelta,
    evaluate: [{ kernel: membrane, workgroups: n.elements }, { kernel: hinge, workgroups: n.hinges }, { kernel: inertia, workgroups: n.rows },
      { kernel: matrix, workgroups: n.rows }, { kernel: force, workgroups: n.rows }],
    energyChange: [{ kernel: membraneDelta, workgroups: n.elements }, { kernel: hingeDelta, workgroups: n.hinges }, { kernel: inertiaDelta, workgroups: n.rows }],
    inertia: [{ kernel: inertia, workgroups: n.rows }],
    ccd: [{ kernel: ccd, workgroups: n.rows }],
    exactElements: [{ kernel: membraneExact, workgroups: n.elements }, { kernel: hingeExact, workgroups: n.hinges }],
    projectedElements: [{ kernel: membraneProjected, workgroups: n.elements }, { kernel: hingeGaussNewton, workgroups: n.hinges }],
  };
}
