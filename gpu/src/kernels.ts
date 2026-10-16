export const KERNEL_FILES = {
  jacobi_invert: { file: "jacobi_invert.wgsl", header: null },
  cg_init: { file: "cg_init.wgsl", header: null },
  spmv_dot: { file: "spmv_dot.wgsl", header: "state.wgsl" },
  reduce: { file: "reduce.wgsl", header: "state.wgsl" },
  update_dot: { file: "update_dot.wgsl", header: "state.wgsl" },
  direction: { file: "direction.wgsl", header: "state.wgsl" },
  residual: { file: "residual.wgsl", header: "state.wgsl" },
  membrane_element: { file: "membrane_element.wgsl", header: "membrane.wgsl" },
  membrane_energy_change: { file: "membrane_energy_change.wgsl", header: "membrane.wgsl" },
  assemble_matrix: { file: "assemble_matrix.wgsl", header: null },
  assemble_force: { file: "assemble_force.wgsl", header: null },
  hinge_element: { file: "hinge_element.wgsl", header: "hinge.wgsl" },
  hinge_energy_change: { file: "hinge_energy_change.wgsl", header: "hinge.wgsl" },
  inertia: { file: "inertia.wgsl", header: "contact.wgsl" },
  inertia_energy_change: { file: "inertia_energy_change.wgsl", header: "contact.wgsl" },
  ccd: { file: "ccd.wgsl", header: "contact.wgsl" },
} as const;

export type KernelName = keyof typeof KERNEL_FILES;
export type KernelSources = Readonly<Record<KernelName, string>>;

const KERNEL_NAMES = Object.keys(KERNEL_FILES) as readonly KernelName[];

/** Reads every kernel through `readText` and prepends its shared header (State struct, membrane law or hinge angle). */
export async function assembleKernels(readText: (file: string) => Promise<string>): Promise<KernelSources> {
  const entries = await Promise.all(KERNEL_NAMES.map(async (name) => {
    const { file, header } = KERNEL_FILES[name];
    const body = await readText(file);
    return [name, header === null ? body : `${await readText(header)}\n${body}`] as const;
  }));
  return Object.fromEntries(entries) as KernelSources;
}
