use crate::compare::{speedup, CaseFiles, Scores};
use crate::io::Json;
use crate::jsonscan::{flag, number, string};

/// Inexact-Newton forcing term: a converged f32 solve must cut the true (f64-evaluated) residual at least 100×,
/// so every Newton iteration still gains two digits. Final accuracy belongs to the nonlinear residual (Experiment 2).
pub const FORCING_LIMIT: f64 = 1e-2;
/// f32 rounding may cost CG some extra iterations; beyond this ratio orthogonality is lost and f32 CG is unreliable.
pub const F32_ITERATION_RATIO_LIMIT: f64 = 1.5;
/// Iteration growth per mesh halving: about 2 for second-order (membrane) conditioning, about 4 for fourth-order (bending).
pub const ITERATION_GROWTH_LIMIT: f64 = 3.0;
/// The GPU must beat one CPU core running f64 PCG by this factor, roughly break-even with a well-threaded f64 CPU solver.
pub const SPEEDUP_LIMIT: f64 = 10.0;
const EMULATION_ITERATION_SLACK: f64 = 0.05;

pub struct Verdict {
    pub converged: bool,
    pub forcing_ok: bool,
    pub f32_iterations_ok: bool,
    pub faithful: bool,
    pub deterministic: bool,
    pub speedup_ok: bool,
}

/// Per-case checks; iteration growth needs several mesh sizes and is judged by `summarize`.
pub fn verdict(case: &CaseFiles, scores: &Scores) -> Verdict {
    let (gpu, reference) = (case.gpu.as_str(), case.reference.as_str());
    let gpu_iterations = number(gpu, "", "iterations");
    let emulated = number(reference, "pcg_f32_cpu", "iterations");
    let same_iterations = (gpu_iterations - emulated).abs() <= 2.0 + EMULATION_ITERATION_SLACK * emulated;
    let m = &scores.metrics;
    Verdict {
        converged: string(gpu, "", "status") == "converged",
        forcing_ok: m.true_relative_residual <= FORCING_LIMIT,
        f32_iterations_ok: gpu_iterations <= F32_ITERATION_RATIO_LIMIT * number(reference, "pcg_f64_cpu", "iterations"),
        faithful: same_iterations && scores.difference_vs_cpu_f32 <= (2.0 * m.relative_error).max(1e-6),
        deterministic: flag(gpu, "", "deterministic_repeat"),
        speedup_ok: speedup(case) >= SPEEDUP_LIMIT,
    }
}

impl Verdict {
    pub fn json(&self) -> Json {
        Json::new()
            .flag("converged", self.converged)
            .flag("forcing_term_ok", self.forcing_ok)
            .flag("f32_iterations_ok", self.f32_iterations_ok)
            .flag("faithful_to_cpu_f32_emulation", self.faithful)
            .flag("deterministic", self.deterministic)
            .flag("speedup_ok", self.speedup_ok)
            .number("forcing_limit", FORCING_LIMIT)
            .number("f32_iteration_ratio_limit", F32_ITERATION_RATIO_LIMIT)
            .number("speedup_limit", SPEEDUP_LIMIT)
    }
}
