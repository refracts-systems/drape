use crate::cases::GRAVITY;
use crate::drape::{feasible_start, Drape, DrapeStep};
use crate::linalg::{identity_scaled, vector_norm, Vec3};
use crate::newton::{solve_step_regularized, NewtonSettings, NewtonSystem, StepStatus};

/// Pseudo-transient continuation (Experiment 7): at step k minimise E_s(x) + (1/(2h²))(x − x_k)ᵀM(x − x_k), i.e. the dynamic
/// time-step energy with v = 0 and x̂ = x_k + h²g, with an inexact inner Newton, a step size that follows the inner Newton
/// (double when it converged within its iteration cap, halve when it hit the cap) and rejection of steps that make no progress.
pub struct Ptc {
    pub h0: f64,
    pub h_max: f64,
    pub h_min_factor: f64,
    pub growth: f64,
    pub shrink: f64,
    pub inner_tolerance: f64,
    pub inner_max: usize,
    pub max_rejections: usize,
    pub max_steps: usize,
    pub equilibrium: f64,
    pub floor_factor: f64,
    /// The floor criterion is only believed below this r/L (a diverged state has a huge floor too).
    pub floor_guard: f64,
}

pub const PTC: Ptc = Ptc {
    h0: 1.0 / 240.0, h_max: 1.0, h_min_factor: 0.1, growth: 2.0, shrink: 0.5, inner_tolerance: 0.1, inner_max: 10, max_rejections: 12,
    max_steps: 1000, equilibrium: 1e-8, floor_factor: 2.0, floor_guard: 1e-6,
};

/// h_{k+1} = clamp(h_k · growth, h_0/10, h_max) when the inner Newton of the accepted step converged within its cap (`inner` < cap),
/// h_k · shrink when it used the whole cap.
pub fn next_step_size(h: f64, inner: usize) -> f64 {
    let factor = if inner < PTC.inner_max { PTC.growth } else { PTC.shrink };
    (h * factor).clamp(PTC.h0 * PTC.h_min_factor, PTC.h_max)
}

/// L = ‖m·g‖₂ over the free unknowns: the load that normalises every static residual.
pub fn gravity_load(drape: &Drape) -> f64 {
    vector_norm(&drape.free.iter().map(|&i| drape.mass[i] * GRAVITY).collect::<Vec<f64>>())
}

/// ∇E_s on the unknowns: membrane + hinge + barrier gradient plus the gravity potential gradient (0, 0, m·g).
pub fn static_gradient(drape: &Drape, x: &[Vec3]) -> Vec<f64> {
    let gradient = drape.potential_gradient(x);
    drape.free.iter().flat_map(|&i| [gradient[i][0], gradient[i][1], gradient[i][2] + drape.mass[i] * GRAVITY]).collect()
}

/// E_s = Π + Σ m g z.
pub fn static_energy(drape: &Drape, x: &[Vec3]) -> f64 {
    drape.potential(x) + (0..x.len()).map(|i| drape.mass[i] * GRAVITY * x[i][2]).sum::<f64>()
}

/// x̂ = x + h²g on free vertices (velocity discarded).
pub fn ptc_prediction(drape: &Drape, x: &[Vec3], h: f64) -> Vec<Vec3> {
    let mut predicted = x.to_vec();
    drape.free.iter().for_each(|&i| predicted[i][2] = x[i][2] - h * h * GRAVITY);
    predicted
}

/// ε64 (‖Σ_c |g_c|‖₂ + ‖|K|·|x|‖₂) with the static contributions (gravity, membrane, hinge, barrier) and the static tangent
/// (no inertia): the rounding floor of ‖∇E_s‖.
pub fn static_floor(drape: &Drape, x: &[Vec3]) -> f64 {
    let step = DrapeStep { drape, x_hat: x.to_vec() };
    let mut contributions = step.absolute_contributions(x);
    drape.free.iter().enumerate().for_each(|(row, &i)| contributions[3 * row + 2] += drape.mass[i] * GRAVITY - (drape.mass_over_h2(i) * (x[i][2] - step.x_hat[i][2])).abs());
    let mut tangent = step.tangent(x);
    drape.free.iter().enumerate().for_each(|(row, &i)| tangent.add(row, row, &identity_scaled(-drape.mass_over_h2(i))));
    tangent.vals.iter_mut().for_each(|v| *v = v.abs());
    let stiffness = tangent.multiply(&step.gather(x).iter().map(|v| v.abs()).collect::<Vec<f64>>());
    f64::EPSILON * (vector_norm(&contributions) + vector_norm(&stiffness))
}

/// One accepted PTC step: the step size, the residual it started from, its inner Newton iterations, and what the rejected
/// attempts before it cost.
pub struct PtcStep {
    pub h: f64,
    pub residual: f64,
    pub inner_iterations: usize,
    pub rejected_attempts: usize,
    pub max_mu: f64,
    pub mu_cap: bool,
    /// The step hit the μ cap and was solved again without the b′∇²d term.
    pub curvature_dropped: bool,
}

/// All states x_0..x_n, the accepted steps between them, and why the run ended.
pub struct PtcRun {
    pub states: Vec<Vec<Vec3>>,
    pub steps: Vec<PtcStep>,
    pub final_residual: f64,
    pub final_h: f64,
    pub reason: &'static str,
}

impl PtcRun {
    pub fn converged(&self) -> bool {
        self.reason == "tolerance" || self.reason == "floor"
    }

    pub fn last(&self) -> &[Vec3] {
        self.states.last().expect("a run always has its start state")
    }
}

fn inner_settings() -> NewtonSettings {
    NewtonSettings { tolerance: PTC.inner_tolerance, max_iterations: PTC.inner_max, armijo: 1e-4, max_halvings: 30 }
}

/// Why a run at residual `residual` (absolute) may stop: ≤ 1e-8·L, or ≤ 2·floor while still ≤ 1e-6·L.
fn equilibrium_reason(drape: &Drape, x: &[Vec3], residual: f64) -> Option<&'static str> {
    let load = gravity_load(drape);
    if residual / load <= PTC.equilibrium {
        Some("tolerance")
    } else if residual / load <= PTC.floor_guard && residual <= PTC.floor_factor * static_floor(drape, x) {
        Some("floor")
    } else {
        None
    }
}

/// An attempt that made no usable progress: the inner Newton failed before its first update (regularisation cap or failed line
/// search), or produced a non-finite state.
fn failed(status: StepStatus, iterations: usize, positions: &[Vec3]) -> bool {
    let stuck = iterations == 0 && matches!(status, StepStatus::RegularizationCap | StepStatus::LineSearchFailed);
    stuck || positions.iter().flatten().any(|v| !v.is_finite())
}

/// The inner Newton of one attempt at step size `h` from `x`.
fn inner_solve(drape: &Drape, x: &[Vec3], h: f64, residual: f64) -> Result<crate::newton::StepSolution, String> {
    drape.time_step.set(h);
    let system = DrapeStep { drape, x_hat: ptc_prediction(drape, x, h) };
    solve_step_regularized(&system, feasible_start(&system, x), residual, &inner_settings(), None)
}

/// One attempt at step size `h` from `x`: the new state and step, or None when it failed. A step that made progress but hit the μ cap
/// is solved again from `x` without the b′∇²d term of the barrier Hessian (the evidence-triggered change of Experiment 5).
fn attempt(drape: &Drape, x: &[Vec3], h: f64, residual: f64) -> Result<Option<(Vec<Vec3>, PtcStep)>, String> {
    let mut solution = inner_solve(drape, x, h, residual)?;
    let capped = solution.regularization.cap_hit && !solution.searches.is_empty();
    if capped {
        drape.barrier_curvature.set(false);
        let retry = inner_solve(drape, x, h, residual);
        drape.barrier_curvature.set(true);
        solution = retry?;
    }
    let iterations = solution.searches.len();
    if failed(solution.status, iterations, &solution.positions) {
        return Ok(None);
    }
    let regularization = solution.regularization;
    let step = PtcStep { h, residual, inner_iterations: iterations, rejected_attempts: 0, max_mu: regularization.max_mu, mu_cap: regularization.cap_hit, curvature_dropped: capped };
    Ok(Some((solution.positions, step)))
}

/// A step from `x`: attempts at h, h/2, … until one makes progress (at most `max_rejections` rejections).
fn take_step(drape: &Drape, x: &[Vec3], h: f64, residual: f64) -> Result<Option<(Vec<Vec3>, PtcStep)>, String> {
    let mut size = h;
    for rejected in 0..=PTC.max_rejections {
        if let Some((next, mut step)) = attempt(drape, x, size, residual)? {
            step.rejected_attempts = rejected;
            return Ok(Some((next, step)));
        }
        size = (size * PTC.shrink).max(PTC.h0 * PTC.h_min_factor);
    }
    Ok(None)
}

/// PTC from `start` with the first step size `h_start` until the criterion holds, the step cap is reached, or a step cannot be
/// taken even at the smallest step size ("stalled"). Progress goes to stderr.
pub fn ptc_solve(drape: &Drape, start: Vec<Vec3>, h_start: f64) -> Result<PtcRun, String> {
    let load = gravity_load(drape);
    let mut run = PtcRun { states: vec![start], steps: Vec::new(), final_residual: f64::NAN, final_h: h_start, reason: "cap" };
    loop {
        let x = run.last().to_vec();
        run.final_residual = vector_norm(&static_gradient(drape, &x));
        if let Some(reason) = equilibrium_reason(drape, &x, run.final_residual) {
            run.reason = reason;
            break;
        }
        if run.steps.len() == PTC.max_steps { break; }
        let h = run.steps.last().map_or(h_start, |s| next_step_size(s.h, s.inner_iterations));
        let Some((next, step)) = take_step(drape, &x, h, run.final_residual)? else { run.reason = "stalled"; break; };
        eprintln!("ptc step {} h={:.3e} r/L={:.3e} inner={} rejected={} mu={}", run.steps.len() + 1, step.h, run.final_residual / load, step.inner_iterations, step.rejected_attempts, step.max_mu);
        run.steps.push(step);
        run.states.push(next);
    }
    run.final_h = run.steps.last().map_or(h_start, |s| next_step_size(s.h, s.inner_iterations));
    Ok(run)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drape::{Scene, TangentKind};

    fn drape() -> Drape {
        Drape::for_scene(9, TangentKind::Exact, Scene::SphereOnFloor)
    }

    #[test]
    fn step_size_doubles_below_the_inner_cap_halves_at_it_and_is_clamped() {
        assert_eq!(next_step_size(0.01, 3), 0.02);
        assert_eq!(next_step_size(0.01, PTC.inner_max), 0.005);
        assert_eq!(next_step_size(0.9, 1), PTC.h_max);
        assert_eq!(next_step_size(PTC.h0 * PTC.h_min_factor, PTC.inner_max), PTC.h0 * PTC.h_min_factor);
    }

    #[test]
    fn the_start_of_a_step_residual_is_the_static_residual() {
        let d = drape();
        let x: Vec<Vec3> = d.rest_positions().iter().map(|p| [p[0], p[1], p[2] + 0.002 * (p[0] * 31.0).sin()]).collect();
        d.time_step.set(0.05);
        let system = DrapeStep { drape: &d, x_hat: ptc_prediction(&d, &x, 0.05) };
        let dynamic = system.residual(&x);
        let statics = static_gradient(&d, &x);
        assert!(dynamic.iter().zip(&statics).all(|(a, b)| (a - b).abs() <= 1e-12 * (1.0 + b.abs())));
    }

    #[test]
    fn a_ptc_step_never_raises_the_static_energy() {
        let d = drape();
        let mut states = vec![d.rest_positions()];
        let mut h = PTC.h0;
        for _ in 0..6 {
            let x = states.last().unwrap().clone();
            let residual = vector_norm(&static_gradient(&d, &x));
            let (next, step) = take_step(&d, &x, h, residual).unwrap().expect("a step can be taken");
            h = next_step_size(step.h, step.inner_iterations);
            states.push(next);
        }
        let energies: Vec<f64> = states.iter().map(|x| static_energy(&d, x)).collect();
        assert!(energies.windows(2).all(|w| w[1] <= w[0] + 1e-15), "{energies:?}");
    }

    #[test]
    fn a_failed_attempt_is_a_regularisation_failure_before_any_update_or_a_non_finite_state() {
        let finite = vec![[0.0; 3]];
        assert!(failed(StepStatus::RegularizationCap, 0, &finite));
        assert!(!failed(StepStatus::RegularizationCap, 2, &finite));
        assert!(!failed(StepStatus::Converged, 0, &finite));
        assert!(failed(StepStatus::Converged, 3, &[[f64::NAN, 0.0, 0.0]]));
    }
}
