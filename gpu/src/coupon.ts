import { field, isArray, isNumber, isObject, isString, parseEntry, slice, type ArrayEntry } from "./problem.ts";

export interface NewtonSettings {
  readonly tolerance: number;
  readonly max_iterations: number;
  readonly armijo: number;
  readonly max_halvings: number;
}

export interface PcgSettings {
  readonly tolerance: number;
  readonly max_iterations: number;
  readonly replace_every: number;
  readonly replace_above: number;
}

export interface Material {
  readonly wale_N_per_m: number;
  readonly course_N_per_m: number;
  readonly coupling_N_per_m: number;
  readonly shear_N_per_m: number;
}

export interface CouponManifest {
  readonly case: string;
  readonly path: string;
  readonly grid_columns: number;
  readonly vertices: number;
  readonly triangles: number;
  readonly rows: number;
  readonly unknowns: number;
  readonly blocks: number;
  readonly fixed: number;
  readonly steps: number;
  readonly material: Material;
  readonly newton: NewtonSettings;
  readonly pcg: PcgSettings;
  readonly arrays: readonly ArrayEntry[];
}

const F32_ARRAYS = ["rest_positions", "initial_positions", "step_gradients", "elem_geom", "fixed_positions", "kernel_state"] as const;
const U32_ARRAYS = ["elem_vertices", "row_vertex", "dof_mask", "row_ptr", "col_idx", "diag_idx", "block_contrib_ptr",
  "block_contrib_idx", "force_contrib_ptr", "force_contrib_idx", "fixed_vertex"] as const;

export type F32ArrayName = (typeof F32_ARRAYS)[number];
export type U32ArrayName = (typeof U32_ARRAYS)[number];

export interface CouponProblem {
  readonly manifest: CouponManifest;
  readonly f32: Readonly<Record<F32ArrayName, Float32Array<ArrayBuffer>>>;
  readonly u32: Readonly<Record<U32ArrayName, Uint32Array<ArrayBuffer>>>;
}

function numbers<K extends string>(source: unknown, keys: readonly K[], label: string): Record<K, number> {
  if (!isObject(source)) throw new Error(`problem.json: '${label}' is not an object`);
  return Object.fromEntries(keys.map((key) => [key, field(source, key, isNumber)])) as Record<K, number>;
}

function sizes(value: object): Omit<CouponManifest, "case" | "path" | "material" | "newton" | "pcg" | "arrays"> {
  return numbers(value, ["grid_columns", "vertices", "triangles", "rows", "unknowns", "blocks", "fixed", "steps"] as const, "root");
}

export function parseCouponManifest(text: string): CouponManifest {
  const value: unknown = JSON.parse(text);
  if (!isObject(value)) throw new Error("problem.json is not an object");
  return {
    ...sizes(value),
    case: field(value, "case", isString),
    path: field(value, "path", isString),
    material: numbers(Reflect.get(value, "material"), ["wale_N_per_m", "course_N_per_m", "coupling_N_per_m", "shear_N_per_m"] as const, "material"),
    newton: numbers(Reflect.get(value, "newton"), ["tolerance", "max_iterations", "armijo", "max_halvings"] as const, "newton"),
    pcg: numbers(Reflect.get(value, "pcg"), ["tolerance", "max_iterations", "replace_every", "replace_above"] as const, "pcg"),
    arrays: field(value, "arrays", isArray).map(parseEntry),
  };
}

export function parseCouponProblem(manifest: CouponManifest, bytes: ArrayBuffer): CouponProblem {
  const f32 = Object.fromEntries(F32_ARRAYS.map((name) => [name, new Float32Array(slice(manifest, bytes, name, "f32"))]));
  const u32 = Object.fromEntries(U32_ARRAYS.map((name) => [name, new Uint32Array(slice(manifest, bytes, name, "u32"))]));
  return {
    manifest,
    f32: f32 as Record<F32ArrayName, Float32Array<ArrayBuffer>>,
    u32: u32 as Record<U32ArrayName, Uint32Array<ArrayBuffer>>,
  };
}
