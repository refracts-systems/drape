import { numbers } from "./coupon.ts";
import { isObject } from "./problem.ts";

const PTC_KEYS = ["h0", "h_max", "h_min_factor", "growth", "shrink", "inner_tolerance", "inner_max_iterations", "max_rejections", "equilibrium_f32",
  "stagnation_window", "stagnation_factor", "gravity_load"] as const;

/** Pseudo-transient continuation rules of Experiment 7, read from problem.json (written by the oracle's e7_run). */
export type PtcRules = Readonly<Record<(typeof PTC_KEYS)[number], number>>;

export function parsePtcRules(text: string): PtcRules {
  const value: unknown = JSON.parse(text);
  if (!isObject(value)) throw new Error("problem.json is not an object");
  return numbers(Reflect.get(value, "ptc"), PTC_KEYS, "ptc");
}

/** The f32 stop of a solve that has not reached the tolerance: it must stay below this r/L to count as stagnation. */
export const STAGNATION_LOAD_LIMIT = 0.25;

/** Accepted-step cap of the f32 solve: 1000 up to n33, 400 above (time budget; reported as not converged). */
export const capFor = (columns: number): number => (columns <= 33 ? 1000 : 400);

/** h_{k+1} = clamp(h_k · growth, h_0/10, h_max) when the accepted step used fewer than the cap of inner updates, else h_k · shrink. */
export function nextStepSize(rules: PtcRules, h: number, inner: number): number {
  const factor = inner < rules.inner_max_iterations ? rules.growth : rules.shrink;
  return Math.min(rules.h_max, Math.max(rules.h0 * rules.h_min_factor, h * factor));
}

/** The smallest residual of a window. */
const smallest = (window: readonly number[]): number => window.reduce((m, r) => Math.min(m, r), Number.POSITIVE_INFINITY);

/** Stagnation: the minimum over the last `window` residuals is at least `factor` times the minimum over the `window` before, and the
 * latest r/L is at most 0.25. `history` holds r_0 … r_k (absolute). */
export function stagnated(rules: PtcRules, history: readonly number[]): boolean {
  const window = rules.stagnation_window;
  if (history.length < 2 * window) return false;
  const last = history.slice(-window);
  const before = history.slice(-2 * window, -window);
  return smallest(last) >= rules.stagnation_factor * smallest(before) && last[window - 1] / rules.gravity_load <= STAGNATION_LOAD_LIMIT;
}

/** Why a solve ended. "time_budget" is not one of the prompt rules: a wall-clock limit the runner may set for the largest mesh, reported as such. */
export type StopRule = "tolerance" | "stagnation" | "cap" | "stalled" | "time_budget";

/** The rule that ends the solve at residual history r_0 … r_k after `accepted` steps, or null to continue. */
export function stopRule(rules: PtcRules, history: readonly number[], accepted: number, cap: number): StopRule | null {
  const latest = history[history.length - 1];
  if (latest / rules.gravity_load <= rules.equilibrium_f32) return "tolerance";
  if (stagnated(rules, history)) return "stagnation";
  return accepted >= cap ? "cap" : null;
}
