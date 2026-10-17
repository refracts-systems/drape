use crate::cases::GRAVITY;
use crate::drape::{feasible_start, Drape, DrapeStep};
use crate::linalg::{identity_scaled, vector_norm, Vec3};
use crate::newton::{solve_step_regularized, NewtonSettings, NewtonSystem, StepStatus};

/// Pseudo-transient continuation (Experiment 7): at step k minimise E_s(x) + (1/(2h²))(x − x_k)ᵀM(x − x_k), i.e. the dynamic
/// time-step energy with v = 0 and x̂ = x_k + h²g, with a step size controller (SER) and an inexact inner Newton.
pub struct Ptc {
    pub h0: f64,
    pub h_max: f64,
    pub h_min_factor: f64,
    pub growth_cap: f64,
    pub inner_tolerance: f64,
    pub inner_max: usize,
    pub max_steps: usize,
    pub equilibrium: f64,
    pub floor_factor: f64,
}

pub const PTC: Ptc = Ptc { h0: 1.0 / 240.0, h_max: 1.0, h_min_factor: 0.1, growth_cap: 2.0, inner_tolerance: 0.1, inner_max: 5, max_steps: 400, equilibrium: 1e-8, floor_factor: 2.0 };

/// h_{k+1} = clamp(h_k · min(growth cap, r_k / r_{k+1}), h_0/10, h_max).
pub fn next_step_size(h: f64, previous_residual: f64, residual: f64) -> f64 {
    (h * (previous_residual / residual).min(PTC.growth_cap)).clamp(PTC.h0 * PTC.h_min_factor, PTC.h_max)
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

/// One PTC step as the oracle took it.
pub struct PtcStep {
    pub h: f64,
    pub residual: f64,
    pub inner_iterations: usize,
    pub max_mu: f64,
    pub mu_cap: bool,
}

/// All states x_0..x_n, the steps between them, and why the run ended.
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

/// Why a run at residual `residual` (absolute) may stop: ≤ 1e-8·L, or ≤ 2·floor.
fn equilibrium_reason(drape: &Drape, x: &[Vec3], residual: f64) -> Option<&'static str> {
    let load = gravity_load(drape);
    if residual / load <= PTC.equilibrium {
        Some("tolerance")
    } else if residual <= PTC.floor_factor * static_floor(drape, x) {
        Some("floor")
    } else {
        None
    }
}

fn take_step(drape: &Drape, x: &[Vec3], h: f64, residual: f64) -> Result<(Vec<Vec3>, PtcStep), String> {
    drape.time_step.set(h);
    let system = DrapeStep { drape, x_hat: ptc_prediction(drape, x, h) };
    let solution = solve_step_regularized(&system, feasible_start(&system, x), residual, &inner_settings(), None)?;
    let regularization = solution.regularization;
    let step = PtcStep { h, residual, inner_iterations: solution.searches.len(), max_mu: regularization.max_mu, mu_cap: solution.status == StepStatus::RegularizationCap || regularization.cap_hit };
    Ok((solution.positions, step))
}

/// PTC from `start` with the first step size `h_start` until the criterion holds, the step cap is reached, or (STOP rule) the
/// run cannot continue. Progress goes to stderr.
pub fn ptc_solve(drape: &Drape, start: Vec<Vec3>, h_start: f64) -> Result<PtcRun, String> {
    let load = gravity_load(drape);
    let mut run = PtcRun { states: vec![start], steps: Vec::new(), final_residual: f64::NAN, final_h: h_start, reason: "cap" };
    loop {
        let x = run.last().to_vec();
        let residual = vector_norm(&static_gradient(drape, &x));
        run.final_residual = residual;
        if let Some(reason) = equilibrium_reason(drape, &x, residual) {
            run.reason = reason;
            break;
        }
        if run.steps.len() == PTC.max_steps {
            break;
        }
        let h = run.steps.last().map_or(h_start, |s| next_step_size(s.h, s.residual, residual));
        let (next, step) = take_step(drape, &x, h, residual)?;
        eprintln!("ptc step {} h={:.3e} r/L={:.3e} inner={} mu={}", run.steps.len() + 1, h, residual / load, step.inner_iterations, step.max_mu);
        run.final_h = h;
        run.steps.push(step);
        run.states.push(next);
    }
    run.final_h = run.steps.last().map_or(h_start, |s| next_step_size(s.h, s.residual, run.final_residual));
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
    fn step_size_grows_with_the_residual_drop_and_is_clamped() {
        assert_eq!(next_step_size(0.01, 4.0, 1.0), 0.02);
        assert_eq!(next_step_size(0.01, 1.0, 4.0), 0.0025);
        assert_eq!(next_step_size(0.9, 4.0, 1.0), PTC.h_max);
        assert_eq!(next_step_size(PTC.h0, 1.0, 1000.0), PTC.h0 * PTC.h_min_factor);
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
        let start = d.rest_positions();
        let run = ptc_solve_steps(&d, start, 6);
        let energies: Vec<f64> = run.iter().map(|x| static_energy(&d, x)).collect();
        assert!(energies.windows(2).all(|w| w[1] <= w[0] + 1e-15), "{energies:?}");
    }

    fn ptc_solve_steps(d: &Drape, start: Vec<Vec3>, steps: usize) -> Vec<Vec<Vec3>> {
        let mut states = vec![start];
        let mut h = PTC.h0;
        for _ in 0..steps {
            let x = states.last().unwrap().clone();
            let residual = vector_norm(&static_gradient(d, &x));
            let (next, step) = take_step(d, &x, h, residual).unwrap();
            h = next_step_size(step.h, residual, vector_norm(&static_gradient(d, &next)));
            states.push(next);
        }
        states
    }
}
