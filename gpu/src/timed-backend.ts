import type { Backend } from "./backend.ts";

export type Phase = "assembly" | "pcg" | "line_search" | "ccd" | "other";

/** Wall-clock time of GPU submits per Newton phase, and of host readbacks/writes (vgpu has no GPU timestamps). */
export interface PhaseClock {
  current: Phase;
  readonly runMs: Record<Phase, number>;
  ioMs: number;
}

export const createPhaseClock = (): PhaseClock => ({ current: "other", runMs: { assembly: 0, pcg: 0, line_search: 0, ccd: 0, other: 0 }, ioMs: 0 });

/** The same backend, with every run() charged to the clock's current phase and every read()/write() to host I/O. */
export function timeBackend<A, K>(inner: Backend<A, K>, clock: PhaseClock): Backend<A, K> {
  return {
    name: inner.name,
    adapter: inner.adapter,
    gpuTiming: inner.gpuTiming,
    array: (label, data, access) => inner.array(label, data, access),
    write: (target, data) => {
      const started = performance.now();
      inner.write(target, data);
      clock.ioMs += performance.now() - started;
    },
    kernel: (spec) => inner.kernel(spec),
    run: async (dispatches) => {
      const started = performance.now();
      const gpuMs = await inner.run(dispatches);
      clock.runMs[clock.current] += performance.now() - started;
      return gpuMs;
    },
    read: async (source) => {
      const started = performance.now();
      const bytes = await inner.read(source);
      clock.ioMs += performance.now() - started;
      return bytes;
    },
    dispose: () => inner.dispose(),
  };
}

/** Runs `work` with the clock's phase set, restoring "other" afterwards. */
export async function inPhase<T>(clock: PhaseClock, phase: Phase, work: () => Promise<T>): Promise<T> {
  clock.current = phase;
  try {
    return await work();
  } finally {
    clock.current = "other";
  }
}
