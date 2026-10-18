export interface ArrayEntry {
  readonly name: string;
  readonly dtype: string;
  readonly length: number;
  readonly offset: number;
}

export interface ProblemManifest {
  readonly case: string;
  readonly grid_columns: number;
  readonly rows: number;
  readonly unknowns: number;
  readonly blocks: number;
  readonly replace_every: number;
  readonly replace_above: number;
  readonly arrays: readonly ArrayEntry[];
}

export interface Problem {
  readonly manifest: ProblemManifest;
  readonly rowPtr: Uint32Array<ArrayBuffer>;
  readonly colIdx: Uint32Array<ArrayBuffer>;
  readonly diagIdx: Uint32Array<ArrayBuffer>;
  readonly vals: Float32Array<ArrayBuffer>;
  readonly rhs: Float32Array<ArrayBuffer>;
}

export function field<T>(source: object, key: string, check: (value: unknown) => value is T): T {
  const value: unknown = Reflect.get(source, key);
  if (!check(value)) {
    throw new Error(`problem.json: field '${key}' is missing or has the wrong type`);
  }
  return value;
}

export const isNumber = (value: unknown): value is number => typeof value === "number";
export const isString = (value: unknown): value is string => typeof value === "string";
export const isObject = (value: unknown): value is object => typeof value === "object" && value !== null;
export const isArray = (value: unknown): value is readonly unknown[] => Array.isArray(value);

export function parseEntry(value: unknown): ArrayEntry {
  if (!isObject(value)) {
    throw new Error("problem.json: array entry is not an object");
  }
  return {
    name: field(value, "name", isString),
    dtype: field(value, "dtype", isString),
    length: field(value, "length", isNumber),
    offset: field(value, "offset", isNumber),
  };
}

export function parseManifest(text: string): ProblemManifest {
  const value: unknown = JSON.parse(text);
  if (!isObject(value)) {
    throw new Error("problem.json is not an object");
  }
  return {
    case: field(value, "case", isString),
    grid_columns: field(value, "grid_columns", isNumber),
    rows: field(value, "rows", isNumber),
    unknowns: field(value, "unknowns", isNumber),
    blocks: field(value, "blocks", isNumber),
    replace_every: Reflect.has(value, "replace_every") ? field(value, "replace_every", isNumber) : 0,
    replace_above: Reflect.has(value, "replace_above") ? field(value, "replace_above", isNumber) : 0,
    arrays: field(value, "arrays", isArray).map(parseEntry),
  };
}

export function slice(manifest: { readonly arrays: readonly ArrayEntry[] }, bytes: ArrayBuffer, name: string, dtype: string): ArrayBuffer {
  const entry = manifest.arrays.find((candidate) => candidate.name === name);
  if (entry === undefined || entry.dtype !== dtype) {
    throw new Error(`problem.bin: array '${name}' (${dtype}) not found`);
  }
  return bytes.slice(entry.offset, entry.offset + 4 * entry.length);
}

function checkSizes(problem: Problem): Problem {
  const { rows, unknowns, blocks } = problem.manifest;
  const consistent = problem.rowPtr.length === rows + 1 && problem.colIdx.length === blocks
    && problem.diagIdx.length === rows && problem.vals.length === 9 * blocks && problem.rhs.length === unknowns;
  if (!consistent || unknowns !== 3 * rows) {
    throw new Error("problem.bin: array lengths disagree with problem.json");
  }
  return problem;
}

export function parseProblem(manifest: ProblemManifest, bytes: ArrayBuffer): Problem {
  return checkSizes({
    manifest,
    rowPtr: new Uint32Array(slice(manifest, bytes, "row_ptr", "u32")),
    colIdx: new Uint32Array(slice(manifest, bytes, "col_idx", "u32")),
    diagIdx: new Uint32Array(slice(manifest, bytes, "diag_idx", "u32")),
    vals: new Float32Array(slice(manifest, bytes, "vals", "f32")),
    rhs: new Float32Array(slice(manifest, bytes, "rhs", "f32")),
  });
}
