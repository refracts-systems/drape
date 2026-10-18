import type { Backend } from "./backend.ts";
import type { KernelSources } from "./kernels.ts";
import type { Problem } from "./problem.ts";
import { prepareSolver, solveOnce, type SolveRun, type SolveSettings, type SolveStatus } from "./solver.ts";

export interface ExperimentSettings extends SolveSettings {
  readonly repeats: number;
}

export interface GpuTiming {
  readonly prepare_ms: number;
  readonly first_solve_ms: number;
  readonly warm_solve_ms: number;
  readonly warm_ms_per_iteration: number;
  readonly gpu_ms: number | null;
  readonly gpu_ms_source: "timestamp-query" | "unavailable";
  readonly readback_ms: number;
  readonly total_ms: number;
}

export interface GpuReport {
  readonly schema: "e1-gpu/1";
  readonly case: string;
  readonly grid_columns: number;
  readonly backend: string;
  readonly adapter: string;
  readonly rows: number;
  readonly unknowns: number;
  readonly blocks: number;
  readonly tolerance: number;
  readonly max_iterations: number;
  readonly chunk_iterations: number;
  readonly replace_every: number;
  readonly replace_above: number;
  readonly status: SolveStatus;
  readonly iterations: number;
  readonly recursive_relative_residual: number;
  readonly gpu_f32_true_relative_residual: number;
  readonly repeats: number;
  readonly deterministic_repeat: boolean;
  readonly timing: GpuTiming;
}

export interface ExperimentResult {
  readonly report: GpuReport;
  readonly solution: Float32Array<ArrayBuffer>;
}

const bitwiseEqual = (a: Float32Array<ArrayBuffer>, b: Float32Array<ArrayBuffer>): boolean => {
  const left = new Uint32Array(a.buffer);
  const right = new Uint32Array(b.buffer);
  return left.length === right.length && left.every((word, index) => word === right[index]);
};

const deterministic = (runs: readonly SolveRun[]): boolean =>
  runs.every((run) => run.iterations === runs[0].iterations && bitwiseEqual(run.solution, runs[0].solution));

/** Fastest repeat after the first; the first run carries lazy pipeline and allocation costs. */
const warmest = (runs: readonly SolveRun[]): SolveRun =>
  runs.slice(1).reduce((best, run) => (run.solveMs < best.solveMs ? run : best), runs[runs.length > 1 ? 1 : 0]);

function timing(prepareMs: number, runs: readonly SolveRun[], gpuTiming: boolean): GpuTiming {
  const [first, warm] = [runs[0], warmest(runs)];
  return {
    prepare_ms: prepareMs,
    first_solve_ms: first.solveMs,
    warm_solve_ms: warm.solveMs,
    warm_ms_per_iteration: warm.solveMs / Math.max(1, warm.iterations),
    gpu_ms: warm.gpuMs,
    gpu_ms_source: gpuTiming ? "timestamp-query" : "unavailable",
    readback_ms: first.readbackMs,
    total_ms: prepareMs + first.solveMs + first.readbackMs,
  };
}

function report<A, K>(backend: Backend<A, K>, problem: Problem, settings: ExperimentSettings, runs: readonly SolveRun[], prepareMs: number): GpuReport {
  const { manifest } = problem;
  const first = runs[0];
  return {
    schema: "e1-gpu/1", case: manifest.case, grid_columns: manifest.grid_columns, backend: backend.name, adapter: backend.adapter,
    rows: manifest.rows, unknowns: manifest.unknowns, blocks: manifest.blocks,
    tolerance: settings.tolerance, max_iterations: settings.maxIterations, chunk_iterations: settings.chunkIterations,
    replace_every: manifest.replace_every,
    replace_above: manifest.replace_above,
    status: first.status, iterations: first.iterations,
    recursive_relative_residual: first.recursiveRelativeResidual,
    gpu_f32_true_relative_residual: first.gpuTrueRelativeResidual,
    repeats: runs.length, deterministic_repeat: deterministic(runs),
    timing: timing(prepareMs, runs, backend.gpuTiming),
  };
}

/** Uploads once, then repeats the full solve to separate cold cost, warm cost and run-to-run determinism. */
export async function runExperiment<A, K>(backend: Backend<A, K>, problem: Problem, sources: KernelSources, settings: ExperimentSettings): Promise<ExperimentResult> {
  const started = performance.now();
  const solver = await prepareSolver(backend, problem, sources);
  const prepareMs = performance.now() - started;
  const runs: SolveRun[] = [];
  for (let repeat = 0; repeat < Math.max(1, settings.repeats); repeat += 1) {
    runs.push(await solveOnce(solver, settings));
  }
  return { report: report(backend, problem, settings, runs, prepareMs), solution: runs[0].solution };
}
