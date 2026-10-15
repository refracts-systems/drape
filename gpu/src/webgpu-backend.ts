import { REQUIRED_STORAGE_BUFFERS, type ArrayAccess, type Backend, type Dispatch, type HostArray, type KernelSpec } from "./backend.ts";

const USAGE = { MAP_READ: 0x1, COPY_SRC: 0x4, COPY_DST: 0x8, STORAGE: 0x80, QUERY_RESOLVE: 0x200 } as const;
const MAP_MODE_READ = 0x1;

export interface WebGpuKernel {
  readonly pipeline: GPUComputePipeline;
  readonly bindGroup: GPUBindGroup;
}

interface PassTimer {
  readonly writes: GPUComputePassTimestampWrites;
  resolve(encoder: GPUCommandEncoder): void;
  read(): Promise<number>;
}

interface Context {
  readonly device: GPUDevice;
  readonly errors: string[];
  readonly timer: PassTimer | null;
}

function watchErrors(device: GPUDevice): string[] {
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (event) => errors.push(event.error.message));
  void device.lost.then((info) => errors.push(`device lost (${info.reason}): ${info.message}`));
  return errors;
}

function throwIfErrors(errors: readonly string[]): void {
  if (errors.length > 0) {
    throw new Error(`WebGPU error: ${errors.join(" | ")}`);
  }
}

function createTimer(device: GPUDevice): PassTimer {
  const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolved = device.createBuffer({ size: 16, usage: USAGE.QUERY_RESOLVE | USAGE.COPY_SRC });
  const readback = device.createBuffer({ size: 16, usage: USAGE.MAP_READ | USAGE.COPY_DST });
  return {
    writes: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
    resolve: (encoder) => {
      encoder.resolveQuerySet(querySet, 0, 2, resolved, 0);
      encoder.copyBufferToBuffer(resolved, 0, readback, 0, 16);
    },
    read: async () => {
      await readback.mapAsync(MAP_MODE_READ);
      const ticks = new BigUint64Array(readback.getMappedRange().slice(0));
      readback.unmap();
      return Number(ticks[1] - ticks[0]) / 1e6;
    },
  };
}

function bindingIndices(source: string): Map<string, number> {
  const declarations = source.matchAll(/@binding\((\d+)\)\s*var<[^>]*>\s*(\w+)/g);
  return new Map(Array.from(declarations, (match) => [match[2], Number(match[1])] as const));
}

function bindGroupEntries(spec: KernelSpec<GPUBuffer>): GPUBindGroupEntry[] {
  const indices = bindingIndices(spec.source);
  const unknown = Object.keys(spec.bindings).filter((name) => !indices.has(name));
  if (unknown.length > 0 || indices.size !== Object.keys(spec.bindings).length) {
    throw new Error(`${spec.label}: bindings [${Object.keys(spec.bindings)}] do not match shader [${[...indices.keys()]}]`);
  }
  return Object.entries(spec.bindings).map(([name, buffer]) => ({ binding: indices.get(name) ?? -1, resource: { buffer } }));
}

async function assertCompiled(module: GPUShaderModule, label: string): Promise<void> {
  const info = await module.getCompilationInfo();
  const problems = info.messages.filter((message) => message.type === "error");
  if (problems.length > 0) {
    throw new Error(`${label}: ${problems.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("; ")}`);
  }
}

async function createKernel(context: Context, spec: KernelSpec<GPUBuffer>): Promise<WebGpuKernel> {
  const { device } = context;
  const module = device.createShaderModule({ label: spec.label, code: spec.source });
  await assertCompiled(module, spec.label);
  const pipeline = await device.createComputePipelineAsync({
    label: spec.label, layout: "auto", compute: { module, entryPoint: "cs_main", constants: spec.constants },
  });
  const bindGroup = device.createBindGroup({ label: spec.label, layout: pipeline.getBindGroupLayout(0), entries: bindGroupEntries(spec) });
  throwIfErrors(context.errors);
  return { pipeline, bindGroup };
}

async function runPass(context: Context, dispatches: readonly Dispatch<WebGpuKernel>[]): Promise<number | null> {
  const { device, timer } = context;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass(timer === null ? {} : { timestampWrites: timer.writes });
  for (const dispatch of dispatches) {
    pass.setPipeline(dispatch.kernel.pipeline);
    pass.setBindGroup(0, dispatch.kernel.bindGroup);
    pass.dispatchWorkgroups(dispatch.workgroups);
  }
  pass.end();
  timer?.resolve(encoder);
  device.queue.submit([encoder.finish()]);
  const milliseconds = timer === null ? await device.queue.onSubmittedWorkDone().then(() => null) : await timer.read();
  throwIfErrors(context.errors);
  return milliseconds;
}

async function readBuffer(context: Context, source: GPUBuffer): Promise<ArrayBuffer> {
  const { device } = context;
  const staging = device.createBuffer({ size: source.size, usage: USAGE.MAP_READ | USAGE.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, source.size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(MAP_MODE_READ);
  const copy = staging.getMappedRange().slice(0);
  staging.destroy();
  throwIfErrors(context.errors);
  return copy;
}

function createArray(device: GPUDevice, label: string, data: HostArray, access: ArrayAccess): GPUBuffer {
  const buffer = device.createBuffer({ label: `${label} (${access})`, size: data.byteLength, usage: USAGE.STORAGE | USAGE.COPY_DST | USAGE.COPY_SRC });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

/** backend and driver exist only with --enable-webgpu-developer-features. */
const developerField = (info: GPUAdapterInfo, key: string): string => {
  const value: unknown = Reflect.get(info, key);
  return typeof value === "string" ? value : "";
};

const describeAdapter = (info: GPUAdapterInfo): string =>
  [info.vendor, info.architecture, info.device, info.description, developerField(info, "backend"), developerField(info, "driver")]
    .filter((part) => part.length > 0).join(" / ") || "unknown adapter";

function backendFor(context: Context, adapter: string): Backend<GPUBuffer, WebGpuKernel> {
  const { device } = context;
  return {
    name: "webgpu-chromium-dawn",
    adapter,
    gpuTiming: context.timer !== null,
    array: (label, data, access) => createArray(device, label, data, access),
    write: (target, data) => device.queue.writeBuffer(target, 0, data),
    kernel: (spec) => createKernel(context, spec),
    run: (dispatches) => runPass(context, dispatches),
    read: (source) => readBuffer(context, source),
    dispose: () => device.destroy(),
  };
}

/** Raw WebGPU: the reference path that vgpu is measured against, on the same Dawn stack. */
export async function createWebGpuBackend(gpu: GPU): Promise<Backend<GPUBuffer, WebGpuKernel>> {
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) {
    throw new Error("navigator.gpu.requestAdapter() returned null: no WebGPU adapter");
  }
  const timestamps = adapter.features.has("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: timestamps ? ["timestamp-query"] : [],
    requiredLimits: { maxStorageBuffersPerShaderStage: REQUIRED_STORAGE_BUFFERS },
  });
  const context: Context = { device, errors: watchErrors(device), timer: timestamps ? createTimer(device) : null };
  return backendFor(context, describeAdapter(adapter.info));
}
