// Helpers shared by the Experiment 6 and 7 runners: record tables as f64 files and the f32 state files.
import { writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { IterationRecord } from "../newton.ts";
import type { RobustRecord, TrialRecord } from "../robust-newton.ts";
import type { RobustRun } from "../drape-e4.ts";

export const RECORD_FIELDS = ["step", "iteration", "relativeResidual", "pcgIterations", "pcgStatus", "linearResidual", "stepLength", "halvings", "status"] as const;
export const EXTRA_FIELDS = ["eta", "mu", "updateInf", "stepInf", "retries", "resolution", "alphaMax", "limitingVertex", "minDistance",
  "deltaSum", "deltaAbs", "noiseOnly", "slope", "slopeExact", "normSquared", "normSquaredExact"] as const;
export const TRIAL_FIELDS = ["step", "iteration", "halving", "alpha", "sum", "abs", "slope", "accepted"] as const;

export const bytesOf = (values: Float32Array<ArrayBuffer> | Float64Array<ArrayBuffer>): Uint8Array => new Uint8Array(values.buffer, values.byteOffset, values.byteLength);

export const table = <T>(rows: readonly T[], fields: readonly (keyof T)[]): Float64Array<ArrayBuffer> =>
  Float64Array.from(rows.flatMap((row) => fields.map((field) => Number(row[field]))));

export function join32(parts: readonly Float32Array<ArrayBuffer>[]): Float32Array<ArrayBuffer> {
  const joined = new Float32Array(parts.reduce((length, x) => length + x.length, 0));
  parts.reduce((offset, x) => (joined.set(x, offset), offset + x.length), 0);
  return joined;
}

/** gpu-<prefix>iterations.bin and gpu-<prefix>robust.bin of one run. */
export async function writeRun(dir: string, prefix: string, run: Pick<RobustRun, "records" | "extras">): Promise<void> {
  await writeFile(join(dir, `gpu-${prefix}iterations.bin`), bytesOf(table<IterationRecord>(run.records, RECORD_FIELDS)));
  await writeFile(join(dir, `gpu-${prefix}robust.bin`), bytesOf(table<RobustRecord>(run.extras, EXTRA_FIELDS)));
}

export const trialTable = (trials: readonly TrialRecord[]): Float64Array<ArrayBuffer> => table<TrialRecord & { readonly accepted: boolean }>(trials, TRIAL_FIELDS);

/** Accepted updates of a record list: the updates whose line search succeeded. */
export const newtonIterations = (run: Pick<RobustRun, "records">): number => run.records.filter((entry) => Number.isFinite(entry.pcgIterations)).length;
