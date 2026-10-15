export type ArrayAccess = "read" | "read-write";
export type HostArray = Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>;

export interface KernelSpec<GpuArray> {
  readonly label: string;
  readonly source: string;
  readonly constants: Readonly<Record<string, number>>;
  readonly bindings: Readonly<Record<string, GpuArray>>;
}

export interface Dispatch<GpuKernel> {
  readonly kernel: GpuKernel;
  readonly workgroups: number;
}

/** The only GPU surface Experiment 1 uses: storage arrays, compute kernels, one-pass submits, readback. */
export interface Backend<GpuArray, GpuKernel> {
  readonly name: string;
  readonly adapter: string;
  readonly gpuTiming: boolean;
  array(label: string, data: HostArray, access: ArrayAccess): GpuArray;
  write(target: GpuArray, data: HostArray): void;
  kernel(spec: KernelSpec<GpuArray>): Promise<GpuKernel>;
  /** All dispatches go into one compute pass and one submit; resolves when the GPU is done, with GPU ms when timestamps exist. */
  run(dispatches: readonly Dispatch<GpuKernel>[]): Promise<number | null>;
  read(source: GpuArray): Promise<ArrayBuffer>;
  dispose(): void;
}
