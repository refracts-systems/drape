// The subset of vgpu 0.6 that Experiment 1 relies on, transcribed from vgpu.sh/docs (reference/vgpu/compute, frame, init).
// Used only by tsconfig.no-vgpu.json (machines where vgpu cannot be installed). tsconfig.json excludes it,
// so after `npm install vgpu @vgpu/wgsl` the real types are checked and any mismatch fails `npm run typecheck`.
declare module "vgpu/node" {
  export type StorageAccess = "read" | "read-write";
  export interface StorageOptions {
    readonly access?: StorageAccess;
    readonly indirect?: boolean;
  }
  export interface StorageBuffer {
    readonly size: number;
    readonly access: StorageAccess;
    read(): Promise<ArrayBuffer>;
    write(data: BufferSource): void;
  }
  export interface ShaderSource {
    readonly version: 2;
  }
  export interface ComputeOptions {
    readonly label?: string;
    readonly set?: Record<string, unknown>;
    readonly constants?: Readonly<Record<string, number | boolean>>;
    readonly entry?: string;
  }
  export interface Compute {
    compile(): Promise<Compute>;
    dispatch(x: number, y?: number, z?: number): void;
    dispose(): void;
  }
  export interface FrameComputePass {
    dispatch(compute: Compute, x: number, y?: number, z?: number): void;
  }
  export interface Frame {
    computePass(options: { readonly label?: string }, encode: (pass: FrameComputePass) => void): void;
  }
  export interface SubmittedFrame {
    readonly done: Promise<void>;
  }
  export interface Gpu {
    readonly adapter: { readonly name: string; readonly type: string };
    dispose(): void;
  }
  export function init(options?: { readonly adapter?: "auto" | "hardware" | "software" }): Promise<Gpu>;
  export function storage(gpu: Gpu, bytes: number, options?: StorageAccess | StorageOptions): StorageBuffer;
  export function compute(gpu: Gpu, source: ShaderSource, options?: ComputeOptions): Compute;
  export function frame(gpu: Gpu, encode: (frame: Frame) => void): SubmittedFrame;
}

declare module "@vgpu/wgsl/prepare" {
  import type { ShaderSource } from "vgpu/node";
  export function prepareShader(source: string): ShaderSource;
}
