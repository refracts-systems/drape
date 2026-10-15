import type { Backend, Dispatch, KernelSpec } from "./backend.ts";
import type { KernelSources } from "./kernels.ts";
import type { Problem } from "./problem.ts";

const WORKGROUP = 256;
const STATUS = ["running", "converged", "iteration_limit", "breakdown"] as const;
export type SolveStatus = (typeof STATUS)[number];

export interface SolveSettings {
  readonly tolerance: number;
  readonly maxIterations: number;
  readonly chunkIterations: number;
}

interface Arrays<A> {
  readonly rowPtr: A; readonly colIdx: A; readonly vals: A; readonly diagIdx: A; readonly rhs: A;
  readonly dinv: A; readonly x: A; readonly r: A; readonly z: A; readonly p: A; readonly q: A;
  readonly partials: A; readonly state: A;
}

export interface PreparedSolver<A, K> {
  readonly backend: Backend<A, K>;
  readonly arrays: Arrays<A>;
  readonly setup: readonly Dispatch<K>[];
  readonly iteration: readonly Dispatch<K>[];
  readonly finish: readonly Dispatch<K>[];
  /** Residual replacement after every `replaceEvery` iterations (0 = never), as in the oracle's f32 emulation. */
  readonly replacement: readonly Dispatch<K>[];
  readonly replaceEvery: number;
  readonly replaceAbove: number;
}

export interface SolveRun {
  readonly status: SolveStatus;
  readonly iterations: number;
  readonly recursiveRelativeResidual: number;
  readonly gpuTrueRelativeResidual: number;
  readonly solution: Float32Array<ArrayBuffer>;
  readonly gpuMs: number | null;
  readonly solveMs: number;
  readonly readbackMs: number;
  readonly chunks: number;
}

interface StateSnapshot {
  readonly rr: number;
  readonly rr0: number;
  readonly trueRr: number;
  readonly iteration: number;
  readonly status: number;
}

const workgroupsFor = (rows: number): number => Math.ceil(rows / WORKGROUP);

function createArrays<A, K>(backend: Backend<A, K>, problem: Problem): Arrays<A> {
  const rows = problem.manifest.rows;
  const vector = (label: string): A => backend.array(label, new Float32Array(3 * rows), "read-write");
  return {
    rowPtr: backend.array("row_ptr", problem.rowPtr, "read"),
    colIdx: backend.array("col_idx", problem.colIdx, "read"),
    vals: backend.array("vals", problem.vals, "read"),
    diagIdx: backend.array("diag_idx", problem.diagIdx, "read"),
    rhs: backend.array("rhs", problem.rhs, "read"),
    dinv: backend.array("dinv", new Float32Array(9 * rows), "read-write"),
    x: vector("x"), r: vector("r"), z: vector("z"), p: vector("p"), q: vector("q"),
    partials: backend.array("partials", new Float32Array(2 * workgroupsFor(rows)), "read-write"),
    state: backend.array("state", new Uint32Array(16), "read-write"),
  };
}

type Stage = "invert" | "init" | "start" | "spmv" | "alpha" | "update" | "beta" | "direction" | "residual" | "measure" | "replace" | "replaced";

function reduceSpec<A>(sources: KernelSources, arrays: Arrays<A>, mode: number): KernelSpec<A> {
  const label = `reduce_mode_${mode}`;
  return { label, source: sources.reduce, constants: { MODE: mode }, bindings: { partials: arrays.partials, state: arrays.state } };
}

function residualSpec<A>(sources: KernelSources, a: Arrays<A>, replace: boolean): KernelSpec<A> {
  const bindings = { row_ptr: a.rowPtr, col_idx: a.colIdx, vals: a.vals, rhs: a.rhs, x: a.x, partials: a.partials, dinv: a.dinv, r: a.r, z: a.z, state: a.state };
  return { label: replace ? "residual_replace" : "residual", source: sources.residual, constants: { REPLACE: replace ? 1 : 0 }, bindings };
}

function stageSpecs<A>(sources: KernelSources, a: Arrays<A>): Record<Stage, KernelSpec<A>> {
  const spec = (label: keyof KernelSources, bindings: Record<string, A>): KernelSpec<A> =>
    ({ label, source: sources[label], constants: {}, bindings });
  return {
    invert: spec("jacobi_invert", { vals: a.vals, diag_idx: a.diagIdx, dinv: a.dinv }),
    init: spec("cg_init", { rhs: a.rhs, dinv: a.dinv, x: a.x, r: a.r, z: a.z, p: a.p, partials: a.partials }),
    start: reduceSpec(sources, a, 0),
    spmv: spec("spmv_dot", { row_ptr: a.rowPtr, col_idx: a.colIdx, vals: a.vals, p: a.p, q: a.q, partials: a.partials, state: a.state }),
    alpha: reduceSpec(sources, a, 1),
    update: spec("update_dot", { p: a.p, q: a.q, dinv: a.dinv, state: a.state, x: a.x, r: a.r, z: a.z, partials: a.partials }),
    beta: reduceSpec(sources, a, 2),
    direction: spec("direction", { z: a.z, state: a.state, p: a.p }),
    residual: residualSpec(sources, a, false),
    measure: reduceSpec(sources, a, 3),
    replace: residualSpec(sources, a, true),
    replaced: reduceSpec(sources, a, 4),
  };
}

async function createStages<A, K>(backend: Backend<A, K>, specs: Record<Stage, KernelSpec<A>>): Promise<Record<Stage, K>> {
  const stages = Object.keys(specs) as Stage[];
  const kernels = await Promise.all(stages.map((stage) => backend.kernel(specs[stage])));
  return Object.fromEntries(stages.map((stage, index) => [stage, kernels[index]])) as Record<Stage, K>;
}

/** Uploads the matrix, compiles all kernels and fixes the dispatch order of setup, one CG iteration, and the true-residual check. */
export async function prepareSolver<A, K>(backend: Backend<A, K>, problem: Problem, sources: KernelSources): Promise<PreparedSolver<A, K>> {
  const arrays = createArrays(backend, problem);
  const k = await createStages(backend, stageSpecs(sources, arrays));
  const rows = workgroupsFor(problem.manifest.rows);
  const at = (kernel: K, workgroups: number): Dispatch<K> => ({ kernel, workgroups });
  return {
    backend,
    arrays,
    setup: [at(k.invert, rows), at(k.init, rows), at(k.start, 1)],
    iteration: [at(k.spmv, rows), at(k.alpha, 1), at(k.update, rows), at(k.beta, 1), at(k.direction, rows)],
    finish: [at(k.residual, rows), at(k.measure, 1)],
    replacement: [at(k.replace, rows), at(k.replaced, 1)],
    replaceEvery: problem.manifest.replace_every,
    replaceAbove: problem.manifest.replace_above,
  };
}

function initialState(settings: SolveSettings, replaceAbove: number): Uint32Array<ArrayBuffer> {
  const words = new Uint32Array(16);
  const floats = new Float32Array(words.buffer);
  floats[6] = settings.tolerance * settings.tolerance;
  floats[11] = replaceAbove * replaceAbove;
  words[10] = settings.maxIterations;
  return words;
}

function decodeState(bytes: ArrayBuffer): StateSnapshot {
  const floats = new Float32Array(bytes);
  const words = new Uint32Array(bytes);
  return { rr: floats[4], rr0: floats[5], trueRr: floats[7], iteration: words[8], status: words[9] };
}

const sumTimings = (values: readonly (number | null)[]): number | null =>
  values.some((value) => value === null) ? null : values.reduce<number>((sum, value) => sum + (value ?? 0), 0);

/** With replacement on, a chunk is exactly one replacement period, so replacements land after iterations K, 2K, … */
function chunkDispatches<A, K>(solver: PreparedSolver<A, K>, settings: SolveSettings): { readonly chunk: readonly Dispatch<K>[]; readonly iterations: number } {
  const iterations = solver.replaceEvery > 0 ? solver.replaceEvery : settings.chunkIterations;
  const tail = solver.replaceEvery > 0 ? solver.replacement : [];
  return { chunk: [...Array.from({ length: iterations }, () => solver.iteration).flat(), ...tail], iterations };
}

async function iterate<A, K>(solver: PreparedSolver<A, K>, settings: SolveSettings, timings: (number | null)[]): Promise<number> {
  const { chunk, iterations } = chunkDispatches(solver, settings);
  const chunkLimit = Math.ceil(settings.maxIterations / iterations) + 1;
  for (let chunks = 1; chunks <= chunkLimit; chunks += 1) {
    timings.push(await solver.backend.run(chunk));
    const state = decodeState(await solver.backend.read(solver.arrays.state));
    if (state.status !== 0) {
      return chunks;
    }
  }
  throw new Error("GPU never raised a stop status: the iteration counter or reduction is broken");
}

function summarize(state: StateSnapshot, solution: Float32Array<ArrayBuffer>, timings: readonly (number | null)[], clock: { readonly started: number; readonly solved: number; readonly chunks: number }): SolveRun {
  return {
    status: STATUS[state.status] ?? "breakdown",
    iterations: state.iteration,
    recursiveRelativeResidual: Math.sqrt(state.rr / state.rr0),
    gpuTrueRelativeResidual: Math.sqrt(state.trueRr / state.rr0),
    solution,
    gpuMs: sumTimings(timings),
    solveMs: clock.solved - clock.started,
    readbackMs: performance.now() - clock.solved,
    chunks: clock.chunks,
  };
}

/** One complete PCG solve from x = 0, with convergence read back once per chunk of iterations. */
export async function solveOnce<A, K>(solver: PreparedSolver<A, K>, settings: SolveSettings): Promise<SolveRun> {
  const { backend, arrays } = solver;
  const started = performance.now();
  const timings: (number | null)[] = [];
  backend.write(arrays.state, initialState(settings, solver.replaceAbove));
  timings.push(await backend.run(solver.setup));
  const chunks = await iterate(solver, settings, timings);
  timings.push(await backend.run(solver.finish));
  const state = decodeState(await backend.read(arrays.state));
  const solved = performance.now();
  const solution = new Float32Array(await backend.read(arrays.x));
  return summarize(state, solution, timings, { started, solved, chunks });
}
