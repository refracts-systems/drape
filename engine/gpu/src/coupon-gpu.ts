import type { Backend, Dispatch, KernelSpec } from "./backend.ts";
import type { CouponProblem } from "./coupon.ts";
import type { KernelSources } from "./kernels.ts";
import type { Problem } from "./problem.ts";
import { prepareSolver, type PreparedSolver } from "./solver.ts";

const WORKGROUP = 256;

/** Membrane kernels plus the unchanged E1 PCG, whose vals/rhs the assembly kernels write in place. */
export interface CouponGpu<A, K> {
  readonly backend: Backend<A, K>;
  readonly solver: PreparedSolver<A, K>;
  readonly positions: A;
  readonly direction: A;
  readonly step: A;
  readonly elemGrad: A;
  readonly elemHess: A;
  readonly elemDelta: A;
  readonly element: readonly Dispatch<K>[];
  readonly evaluate: readonly Dispatch<K>[];
  readonly energyChange: readonly Dispatch<K>[];
}

interface MeshArrays<A> {
  readonly positions: A; readonly direction: A; readonly step: A; readonly elemVertices: A; readonly elemGeom: A;
  readonly elemGrad: A; readonly elemHess: A; readonly elemDelta: A; readonly dofMask: A;
  readonly blockPtr: A; readonly blockIdx: A; readonly forcePtr: A; readonly forceIdx: A;
}

/** The E1 solver's view of the Newton system: the coupon's BSR pattern with zero values until assembled. */
function linearProblem(problem: CouponProblem): Problem {
  const m = problem.manifest;
  return {
    manifest: { case: m.case, grid_columns: m.grid_columns, rows: m.rows, unknowns: m.unknowns, blocks: m.blocks,
      replace_every: m.pcg.replace_every, replace_above: m.pcg.replace_above, arrays: [] },
    rowPtr: problem.u32.row_ptr, colIdx: problem.u32.col_idx, diagIdx: problem.u32.diag_idx,
    vals: new Float32Array(9 * m.blocks), rhs: new Float32Array(m.unknowns),
  };
}

function meshArrays<A, K>(backend: Backend<A, K>, problem: CouponProblem): MeshArrays<A> {
  const { f32, u32, manifest } = problem;
  const scratch = (label: string, length: number): A => backend.array(label, new Float32Array(length), "read-write");
  return {
    positions: backend.array("positions", Float32Array.from(f32.initial_positions), "read-write"),
    direction: scratch("direction", 3 * manifest.vertices), step: scratch("step", 1),
    elemVertices: backend.array("elem_vertices", u32.elem_vertices, "read"), elemGeom: backend.array("elem_geom", f32.elem_geom, "read"),
    elemGrad: scratch("elem_grad", 9 * manifest.triangles), elemHess: scratch("elem_hess", 81 * manifest.triangles),
    elemDelta: scratch("elem_delta", manifest.triangles), dofMask: backend.array("dof_mask", u32.dof_mask, "read"),
    blockPtr: backend.array("block_contrib_ptr", u32.block_contrib_ptr, "read"), blockIdx: backend.array("block_contrib_idx", u32.block_contrib_idx, "read"),
    forcePtr: backend.array("force_contrib_ptr", u32.force_contrib_ptr, "read"), forceIdx: backend.array("force_contrib_idx", u32.force_contrib_idx, "read"),
  };
}

function materialConstants(problem: CouponProblem): Record<string, number> {
  const m = problem.manifest.material;
  return { WALE: m.wale_N_per_m, COURSE: m.course_N_per_m, COUPLING: m.coupling_N_per_m, SHEAR: m.shear_N_per_m };
}

type CouponStage = "element" | "energy" | "matrix" | "force";

function stageSpecs<A, K>(sources: KernelSources, problem: CouponProblem, a: MeshArrays<A>, solver: PreparedSolver<A, K>): Record<CouponStage, KernelSpec<A>> {
  const material = materialConstants(problem);
  const s = solver.arrays;
  return {
    element: { label: "membrane_element", source: sources.membrane_element, constants: material,
      bindings: { positions: a.positions, elem_vertices: a.elemVertices, elem_geom: a.elemGeom, elem_grad: a.elemGrad, elem_hess: a.elemHess } },
    energy: { label: "membrane_energy_change", source: sources.membrane_energy_change, constants: material,
      bindings: { positions: a.positions, direction: a.direction, elem_vertices: a.elemVertices, elem_geom: a.elemGeom, step: a.step, elem_delta: a.elemDelta } },
    matrix: { label: "assemble_matrix", source: sources.assemble_matrix, constants: {},
      bindings: { row_ptr: s.rowPtr, col_idx: s.colIdx, dof_mask: a.dofMask, block_contrib_ptr: a.blockPtr, block_contrib_idx: a.blockIdx, elem_hess: a.elemHess, vals: s.vals } },
    force: { label: "assemble_force", source: sources.assemble_force, constants: {},
      bindings: { dof_mask: a.dofMask, force_contrib_ptr: a.forcePtr, force_contrib_idx: a.forceIdx, elem_grad: a.elemGrad, rhs: s.rhs } },
  };
}

export async function prepareCouponGpu<A, K>(backend: Backend<A, K>, problem: CouponProblem, sources: KernelSources): Promise<CouponGpu<A, K>> {
  const solver = await prepareSolver(backend, linearProblem(problem), sources);
  const arrays = meshArrays(backend, problem);
  const specs = stageSpecs(sources, problem, arrays, solver);
  const [element, energy, matrix, force] = await Promise.all([specs.element, specs.energy, specs.matrix, specs.force].map((spec) => backend.kernel(spec)));
  const elements = Math.ceil(problem.manifest.triangles / WORKGROUP);
  const rows = Math.ceil(problem.manifest.rows / WORKGROUP);
  return {
    backend, solver, positions: arrays.positions, direction: arrays.direction, step: arrays.step,
    elemGrad: arrays.elemGrad, elemHess: arrays.elemHess, elemDelta: arrays.elemDelta,
    element: [{ kernel: element, workgroups: elements }],
    evaluate: [{ kernel: element, workgroups: elements }, { kernel: matrix, workgroups: rows }, { kernel: force, workgroups: rows }],
    energyChange: [{ kernel: energy, workgroups: elements }],
  };
}
