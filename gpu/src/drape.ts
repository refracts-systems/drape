import { numbers, type Material, type NewtonSettings, type PcgSettings } from "./coupon.ts";
import { field, isArray, isObject, isString, parseEntry, slice, type ArrayEntry } from "./problem.ts";

export interface DrapeManifest {
  readonly case: string;
  readonly grid_columns: number;
  readonly vertices: number;
  readonly triangles: number;
  readonly hinges: number;
  readonly rows: number;
  readonly unknowns: number;
  readonly blocks: number;
  readonly steps: number;
  readonly probes: number;
  readonly time_step_s: number;
  readonly gravity_m_per_s2: number;
  readonly tangent: string;
  readonly material: Material;
  readonly newton: NewtonSettings;
  readonly pcg: PcgSettings;
  /** Line-search keep fraction of the colliders (NaN without a "contact" section). */
  readonly keep_fraction: number;
  readonly arrays: readonly ArrayEntry[];
}

const F32_ARRAYS = ["rest_positions", "elem_geom", "hinge_stiffness", "row_mass", "kernel_state", "probe_positions", "probe_velocities"] as const;
const U32_ARRAYS = ["elem_vertices", "hinge_vertices", "row_vertex", "dof_mask", "row_ptr", "col_idx", "diag_idx", "block_contrib_ptr",
  "block_contrib_idx", "force_contrib_ptr", "force_contrib_idx", "fixed_vertex", "probe_steps"] as const;

export type DrapeF32Name = (typeof F32_ARRAYS)[number];
export type DrapeU32Name = (typeof U32_ARRAYS)[number];

export interface DrapeProblem {
  readonly manifest: DrapeManifest;
  readonly f32: Readonly<Record<DrapeF32Name, Float32Array<ArrayBuffer>>>;
  readonly u32: Readonly<Record<DrapeU32Name, Uint32Array<ArrayBuffer>>>;
  readonly contact: DrapeContact;
  /** Unit masses (f32, one per unknown row) of Experiment 7 problems; null before. */
  readonly unitMass: Float32Array<ArrayBuffer> | null;
}

const SIZES = ["grid_columns", "vertices", "triangles", "hinges", "rows", "unknowns", "blocks", "steps", "probes", "time_step_s", "gravity_m_per_s2"] as const;

export function parseDrapeManifest(text: string): DrapeManifest {
  const value: unknown = JSON.parse(text);
  if (!isObject(value)) throw new Error("problem.json is not an object");
  return {
    ...numbers(value, SIZES, "root"),
    case: field(value, "case", isString),
    tangent: field(value, "tangent", isString),
    material: numbers(Reflect.get(value, "material"), ["wale_N_per_m", "course_N_per_m", "coupling_N_per_m", "shear_N_per_m"] as const, "material"),
    newton: numbers(Reflect.get(value, "newton"), ["tolerance", "max_iterations", "armijo", "max_halvings"] as const, "newton"),
    pcg: numbers(Reflect.get(value, "pcg"), ["tolerance", "max_iterations", "replace_every", "replace_above"] as const, "pcg"),
    keep_fraction: keepFraction(Reflect.get(value, "contact")),
    arrays: field(value, "arrays", isArray).map(parseEntry),
  };
}

export function parseDrapeProblem(manifest: DrapeManifest, bytes: ArrayBuffer): DrapeProblem {
  const f32 = Object.fromEntries(F32_ARRAYS.map((name) => [name, new Float32Array(slice(manifest, bytes, name, "f32"))]));
  const u32 = Object.fromEntries(U32_ARRAYS.map((name) => [name, new Uint32Array(slice(manifest, bytes, name, "u32"))]));
  return { manifest, f32: f32 as Record<DrapeF32Name, Float32Array<ArrayBuffer>>, u32: u32 as Record<DrapeU32Name, Uint32Array<ArrayBuffer>>, contact: parseContact(manifest, bytes), unitMass: has(manifest, "row_unit_mass") ? new Float32Array(slice(manifest, bytes, "row_unit_mass", "f32")) : null };
}

/** Experiment 5 colliders ([count, d̂, κ, 0, 8 floats per collider]) and the contact kernel-check state; no colliders before. */
export interface DrapeContact {
  readonly colliders: Float32Array<ArrayBuffer>;
  readonly state: Float32Array<ArrayBuffer> | null;
  readonly count: number;
}

const has = (manifest: DrapeManifest, name: string): boolean => manifest.arrays.some((entry) => entry.name === name);

function parseContact(manifest: DrapeManifest, bytes: ArrayBuffer): DrapeContact {
  if (!has(manifest, "colliders")) return { colliders: new Float32Array(4), state: null, count: 0 };
  const colliders = new Float32Array(slice(manifest, bytes, "colliders", "f32"));
  return { colliders, state: new Float32Array(slice(manifest, bytes, "contact_state", "f32")), count: colliders[0] };
}

function keepFraction(contact: unknown): number {
  const value: unknown = isObject(contact) ? Reflect.get(contact, "keep_fraction") : undefined;
  return typeof value === "number" ? value : Number.NaN;
}
