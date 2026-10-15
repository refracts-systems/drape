// Experiment 1 through vgpu/node: identical WGSL, problem files and dispatch order as run-browser.ts.
import { runExperiment } from "../experiment.ts";
import { parseCommon } from "./cli.ts";
import { loadKernels, loadProblem, printSummary, writeOutputs } from "./files.ts";
import { assertTargetAdapter } from "./hardware.ts";
import { createVgpuBackend } from "./vgpu-backend.ts";

const ADAPTERS = ["auto", "hardware", "software"] as const;

async function main(): Promise<void> {
  const options = parseCommon(process.argv.slice(2), ADAPTERS, "hardware");
  const [problem, sources] = await Promise.all([loadProblem(options.caseDir), loadKernels()]);
  const backend = await createVgpuBackend(options.adapter);
  try {
    assertTargetAdapter(options.adapter, backend.adapter);
    const result = await runExperiment(backend, problem, sources, options.settings);
    await writeOutputs(options.caseDir, result.report, result.solution);
    printSummary(result.report);
  } finally {
    backend.dispose();
  }
}

await main();
