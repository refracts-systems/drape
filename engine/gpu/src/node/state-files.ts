// Experiment 7 equilibrium state files: <prefix>.json {"vertices", "final_h", "dtype"} + <prefix>.bin (little endian x y z per vertex).
import { readFile, writeFile } from "node:fs/promises";
import { bytesOf } from "./outputs.ts";

export interface StateFile {
  readonly positions: Float32Array<ArrayBuffer>;
  readonly finalH: number;
}

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null;

export async function writeState(prefix: string, positions: Float32Array<ArrayBuffer>, finalH: number): Promise<void> {
  await writeFile(`${prefix}.bin`, bytesOf(positions));
  await writeFile(`${prefix}.json`, `${JSON.stringify({ vertices: positions.length / 3, final_h: finalH, dtype: "f32" })}\n`);
}

/** Reads a state written by the GPU runner or by the oracle's e7-prolong (f32 files only: the GPU starts from f32 states). */
export async function readState(prefix: string): Promise<StateFile> {
  const meta: unknown = JSON.parse(await readFile(`${prefix}.json`, "utf8"));
  if (!isRecord(meta) || typeof meta.final_h !== "number" || meta.dtype !== "f32") throw new Error(`${prefix}.json: not an f32 state file`);
  const bytes = await readFile(`${prefix}.bin`);
  return { positions: new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)), finalH: meta.final_h };
}
