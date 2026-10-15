import { prepareShader } from "@vgpu/wgsl/prepare";
import { compute, frame, init, storage, type Compute, type Gpu, type StorageBuffer } from "vgpu/node";
import type { ArrayAccess, Backend, Dispatch, HostArray } from "../backend.ts";

export type VgpuAdapter = "auto" | "hardware" | "software";

function createArray(gpu: Gpu, data: HostArray, access: ArrayAccess): StorageBuffer {
  const buffer = storage(gpu, data.byteLength, { access });
  buffer.write(data);
  return buffer;
}

/** One frame computePass per run: one encoder, one submit, dispatches in order. */
async function runFrame(gpu: Gpu, dispatches: readonly Dispatch<Compute>[]): Promise<null> {
  const submitted = frame(gpu, (current) => current.computePass({ label: "e1-pcg" }, (pass) => {
    dispatches.forEach((dispatch) => pass.dispatch(dispatch.kernel, dispatch.workgroups));
  }));
  await submitted.done;
  return null;
}

/** vgpu/node (Dawn): same WGSL, same dispatch list as the raw-WebGPU path. No GPU timer yet, so gpu_ms is null. */
export async function createVgpuBackend(adapter: VgpuAdapter): Promise<Backend<StorageBuffer, Compute>> {
  const gpu = await init({ adapter });
  return {
    name: "vgpu-node-dawn",
    adapter: `${gpu.adapter.name} (${gpu.adapter.type})`,
    gpuTiming: false,
    array: (_label, data, access) => createArray(gpu, data, access),
    write: (target, data) => target.write(data),
    kernel: (spec) => compute(gpu, prepareShader(spec.source), { label: spec.label, set: spec.bindings, constants: spec.constants }).compile(),
    run: (dispatches) => runFrame(gpu, dispatches),
    read: (source) => source.read(),
    dispose: () => gpu.dispose(),
  };
}
