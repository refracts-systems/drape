import { runExperiment, type ExperimentSettings, type GpuReport } from "./experiment.ts";
import { assembleKernels } from "./kernels.ts";
import { parseManifest, parseProblem } from "./problem.ts";
import { createWebGpuBackend } from "./webgpu-backend.ts";

export interface BrowserOutcome {
  readonly report: GpuReport;
  readonly solution: readonly number[];
}

declare global {
  interface Window {
    runE1?: (settings: ExperimentSettings) => Promise<BrowserOutcome>;
  }
}

async function fetchOk(path: string): Promise<Response> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`GET ${path}: ${response.status}`);
  }
  return response;
}

async function runInPage(settings: ExperimentSettings): Promise<BrowserOutcome> {
  const manifest = parseManifest(await (await fetchOk("/case/problem.json")).text());
  const problem = parseProblem(manifest, await (await fetchOk("/case/problem.bin")).arrayBuffer());
  const sources = await assembleKernels(async (file) => (await fetchOk(`/kernels/${file}`)).text());
  const backend = await createWebGpuBackend(navigator.gpu);
  try {
    const result = await runExperiment(backend, problem, sources, settings);
    return { report: result.report, solution: Array.from(result.solution) };
  } finally {
    backend.dispose();
  }
}

window.runE1 = runInPage;
