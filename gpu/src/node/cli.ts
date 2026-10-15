import { parseArgs } from "node:util";
import type { ExperimentSettings } from "../experiment.ts";

export interface CommonOptions<Adapter extends string> {
  readonly caseDir: string;
  readonly adapter: Adapter;
  readonly settings: ExperimentSettings;
}

const isOneOf = <T extends string>(value: string, options: readonly T[]): value is T =>
  options.some((option) => option === value);

function positive(text: string, flag: string): number {
  const value = Number(text);
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`--${flag} must be a positive number, got '${text}'`);
  }
  return value;
}

/** --case <dir> [--tol 1e-7] [--max-iter 20000] [--chunk 32] [--repeats 3] [--adapter <name>] */
export function parseCommon<Adapter extends string>(argv: readonly string[], adapters: readonly Adapter[], fallback: Adapter): CommonOptions<Adapter> {
  const fallbackName: string = fallback;
  const { values } = parseArgs({
    args: [...argv],
    strict: true,
    allowPositionals: false,
    options: {
      case: { type: "string" }, tol: { type: "string", default: "1e-7" }, "max-iter": { type: "string", default: "20000" },
      chunk: { type: "string", default: "32" }, repeats: { type: "string", default: "3" }, adapter: { type: "string", default: fallbackName },
    },
  });
  if (values.case === undefined || !isOneOf(values.adapter, adapters)) {
    throw new Error(`usage: --case <case dir> [--adapter ${adapters.join("|")}] [--tol 1e-7] [--max-iter 20000] [--chunk 32] [--repeats 3]`);
  }
  const settings: ExperimentSettings = {
    tolerance: positive(values.tol, "tol"), maxIterations: positive(values["max-iter"], "max-iter"),
    chunkIterations: positive(values.chunk, "chunk"), repeats: positive(values.repeats, "repeats"),
  };
  return { caseDir: values.case, adapter: values.adapter, settings };
}
