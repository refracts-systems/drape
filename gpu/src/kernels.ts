export const KERNEL_FILES = {
  jacobi_invert: { file: "jacobi_invert.wgsl", usesState: false },
  cg_init: { file: "cg_init.wgsl", usesState: false },
  spmv_dot: { file: "spmv_dot.wgsl", usesState: true },
  reduce: { file: "reduce.wgsl", usesState: true },
  update_dot: { file: "update_dot.wgsl", usesState: true },
  direction: { file: "direction.wgsl", usesState: true },
  residual: { file: "residual.wgsl", usesState: false },
} as const;

export type KernelName = keyof typeof KERNEL_FILES;
export type KernelSources = Readonly<Record<KernelName, string>>;

const KERNEL_NAMES = Object.keys(KERNEL_FILES) as readonly KernelName[];

/** Reads every kernel through `readText` and prepends the shared State struct where it is used. */
export async function assembleKernels(readText: (file: string) => Promise<string>): Promise<KernelSources> {
  const header = await readText("state.wgsl");
  const entries = await Promise.all(KERNEL_NAMES.map(async (name) => {
    const { file, usesState } = KERNEL_FILES[name];
    const body = await readText(file);
    return [name, usesState ? `${header}\n${body}` : body] as const;
  }));
  return Object.fromEntries(entries) as KernelSources;
}
