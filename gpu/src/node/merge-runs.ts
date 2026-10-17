// Experiment 6: several timed runs of one rung leave gpu-result-run<k>.json next to the (identical) outputs; this writes the
// timing of the run with the median ms/step into gpu-result.json and keeps every run's ms/step in `ms_per_step_runs`.
import { readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { parseArgs } from "node:util";

interface Timing {
  readonly ms_per_step: number;
  readonly trajectory_ms: number;
  readonly phase_ms_per_step: Readonly<Record<string, number>>;
  readonly read_bytes_total: number;
}

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null;

function timing(value: unknown): Timing {
  if (!isRecord(value) || typeof value.ms_per_step !== "number" || typeof value.trajectory_ms !== "number" || !isRecord(value.phase_ms_per_step) || typeof value.read_bytes_total !== "number") {
    throw new Error("gpu-result-run: unexpected shape");
  }
  const phases = Object.fromEntries(Object.entries(value.phase_ms_per_step).filter((entry): entry is [string, number] => typeof entry[1] === "number"));
  return { ms_per_step: value.ms_per_step, trajectory_ms: value.trajectory_ms, phase_ms_per_step: phases, read_bytes_total: value.read_bytes_total };
}

async function main(): Promise<void> {
  const { values } = parseArgs({ options: { dir: { type: "string" } } });
  const dir = values.dir;
  if (dir === undefined) throw new Error("usage: --dir <e6-rung dir>");
  const names = (await readdir(dir)).filter((name) => /^gpu-result-run\d+\.json$/.test(name)).sort();
  if (names.length === 0) throw new Error(`${dir}: no gpu-result-run<k>.json`);
  const runs = await Promise.all(names.map(async (name) => JSON.parse(await readFile(join(dir, name), "utf8")) as unknown));
  const ordered = runs.map(timing).sort((a, b) => a.ms_per_step - b.ms_per_step);
  const median = ordered[Math.floor(ordered.length / 2)];
  const base: unknown = JSON.parse(await readFile(join(dir, "gpu-result.json"), "utf8"));
  if (!isRecord(base)) throw new Error("gpu-result.json is not an object");
  const merged = { ...base, ...median, ms_per_step_runs: ordered.map((run) => run.ms_per_step), runs: ordered.length };
  await writeFile(join(dir, "gpu-result.json"), `${JSON.stringify(merged, null, 2)}\n`);
  console.log(`${dir}: median ${median.ms_per_step.toFixed(1)} ms/step of ${ordered.map((run) => run.ms_per_step.toFixed(1)).join(", ")}`);
}

await main();
