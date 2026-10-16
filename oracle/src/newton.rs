use crate::band::BandCholesky;
use crate::bsr::Bsr;
use crate::coupon::{free_dot, Coupon};
use crate::linalg::{vector_norm, Vec3};

/// What Newton needs from a discretised energy: residual and tangent on the unknowns, a cancellation-free
/// energy change, and the map between unknown rows and vertices.
pub trait NewtonSystem {
    fn residual(&self, x: &[Vec3]) -> Vec<f64>;
    fn tangent(&self, x: &[Vec3]) -> Bsr;
    fn energy_change(&self, x: &[Vec3], d: &[Vec3], alpha: f64) -> f64;
    fn scatter(&self, rows: &[f64]) -> Vec<Vec3>;
    fn gather(&self, vertices: &[Vec3]) -> Vec<f64>;
}

impl NewtonSystem for Coupon {
    fn residual(&self, x: &[Vec3]) -> Vec<f64> {
        self.free_gradient(x)
    }
    fn tangent(&self, x: &[Vec3]) -> Bsr {
        Coupon::tangent(self, x)
    }
    fn energy_change(&self, x: &[Vec3], d: &[Vec3], alpha: f64) -> f64 {
        Coupon::energy_change(self, x, d, alpha)
    }
    fn scatter(&self, rows: &[f64]) -> Vec<Vec3> {
        Coupon::scatter(self, rows)
    }
    fn gather(&self, vertices: &[Vec3]) -> Vec<f64> {
        self.free.iter().flat_map(|&v| vertices[v]).collect()
    }
}

pub struct NewtonSettings {
    pub tolerance: f64,
    pub max_iterations: usize,
    pub armijo: f64,
    pub max_halvings: usize,
}

/// One accepted Newton update: Armijo step length and the number of halvings it took.
pub struct LineSearch {
    pub step_length: f64,
    pub halvings: usize,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum StepStatus {
    Converged,
    IterationLimit,
    LineSearchFailed,
}

impl StepStatus {
    pub fn name(&self) -> &'static str {
        match self {
            StepStatus::Converged => "converged",
            StepStatus::IterationLimit => "iteration_limit",
            StepStatus::LineSearchFailed => "line_search_failed",
        }
    }
}

/// One load or time step: ‖g(x_k)‖ / step scale for every iterate, and the line search behind each update.
pub struct StepSolution {
    pub positions: Vec<Vec3>,
    pub residuals: Vec<f64>,
    pub searches: Vec<LineSearch>,
    pub status: StepStatus,
}

/// Newton with an exact f64 band-Cholesky step and Armijo backtracking on the energy.
pub fn solve_step<S: NewtonSystem>(system: &S, start: Vec<Vec3>, scale: f64, settings: &NewtonSettings) -> Result<StepSolution, String> {
    let mut step = StepSolution { positions: start, residuals: Vec::new(), searches: Vec::new(), status: StepStatus::IterationLimit };
    loop {
        let gradient = system.residual(&step.positions);
        let relative = vector_norm(&gradient) / scale;
        step.residuals.push(relative);
        if relative <= settings.tolerance {
            step.status = StepStatus::Converged;
            return Ok(step);
        }
        if step.searches.len() == settings.max_iterations {
            return Ok(step);
        }
        let direction = newton_direction(system, &step.positions, &gradient)?;
        let Some(search) = line_search(system, &step.positions, &gradient, &direction, settings) else {
            step.status = StepStatus::LineSearchFailed;
            return Ok(step);
        };
        advance(&mut step.positions, &direction, search.step_length);
        step.searches.push(search);
    }
}

fn newton_direction<S: NewtonSystem>(system: &S, x: &[Vec3], gradient: &[f64]) -> Result<Vec<Vec3>, String> {
    let negative: Vec<f64> = gradient.iter().map(|g| -g).collect();
    let rows = BandCholesky::factor(&system.tangent(x))?.solve(&negative);
    Ok(system.scatter(&rows))
}

/// Backtracking from α = 1: accept when ΔΠ ≤ c·α·gᵀd. None when d is not a descent direction or halving runs out.
pub fn line_search<S: NewtonSystem>(system: &S, x: &[Vec3], gradient: &[f64], direction: &[Vec3], settings: &NewtonSettings) -> Option<LineSearch> {
    let slope = free_dot(gradient, &system.gather(direction));
    if slope >= 0.0 {
        return None;
    }
    (0..=settings.max_halvings)
        .map(|halvings| LineSearch { step_length: 0.5f64.powi(halvings as i32), halvings })
        .find(|s| system.energy_change(x, direction, s.step_length) <= settings.armijo * s.step_length * slope)
}

fn advance(x: &mut [Vec3], direction: &[Vec3], step_length: f64) {
    for (p, d) in x.iter_mut().zip(direction) {
        (0..3).for_each(|k| p[k] += step_length * d[k]);
    }
}

/// Runs every load step of the coupon's path from the previous equilibrium plus the affine predictor.
pub fn solve_path(coupon: &Coupon, settings: &NewtonSettings) -> Result<Vec<StepSolution>, String> {
    let mut current = coupon.initial_positions();
    let (mut steps, mut previous_factor) = (Vec::new(), 0.0);
    for t in coupon.path.load_factors() {
        let scale = coupon.step_scale(&current, t);
        coupon.predict(&mut current, previous_factor, t);
        let step = solve_step(coupon, current.clone(), scale, settings)?;
        previous_factor = t;
        current = step.positions.clone();
        steps.push(step);
    }
    Ok(steps)
}

/// Index of the first iterate at or below `tolerance`, if any.
pub fn iterations_to(residuals: &[f64], tolerance: f64) -> Option<usize> {
    residuals.iter().position(|&r| r <= tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coupon::LoadPath;
    use crate::e2_checks::{derivative_check, patch_check, path_independence};

    fn settings() -> NewtonSettings {
        NewtonSettings { tolerance: 1e-12, max_iterations: 30, armijo: 1e-4, max_halvings: 30 }
    }

    #[test]
    fn armijo_backtracks_an_overlong_newton_step() {
        let coupon = Coupon::build(LoadPath::Uniaxial, 9);
        let mut x = coupon.initial_positions();
        coupon.predict(&mut x, 0.0, 0.5);
        let gradient = coupon.free_gradient(&x);
        let direction: Vec<Vec3> = newton_direction(&coupon, &x, &gradient).unwrap().iter().map(|d| d.map(|v| 4.0 * v)).collect();
        let search = line_search(&coupon, &x, &gradient, &direction, &settings()).unwrap();
        assert!(search.halvings >= 1, "{}", search.halvings);
        let rows: Vec<f64> = coupon.free.iter().flat_map(|&v| direction[v]).collect();
        let slope = free_dot(&gradient, &rows);
        assert!(coupon.energy_change(&x, &direction, search.step_length) <= 1e-4 * search.step_length * slope);
    }

    #[test]
    fn patch_test_reproduces_the_affine_map() {
        let coupon = Coupon::build(LoadPath::Patch, 9);
        let steps = solve_path(&coupon, &settings()).unwrap();
        let check = patch_check(&coupon, &steps[0].positions);
        assert!(check.displacement_error < 1e-12 && check.strain_error < 1e-10, "{} {}", check.displacement_error, check.strain_error);
    }

    #[test]
    fn assembled_derivatives_match_finite_differences() {
        let check = derivative_check(&Coupon::build(LoadPath::Biaxial, 9));
        assert!(check.gradient_error < 1e-6 && check.tangent_error < 1e-5 && check.energy_change_error < 1e-12);
    }

    #[test]
    fn unloading_retraces_loading() {
        let coupon = Coupon::build(LoadPath::Uniaxial, 9);
        let positions: Vec<Vec<Vec3>> = solve_path(&coupon, &settings()).unwrap().into_iter().map(|s| s.positions).collect();
        let check = path_independence(&coupon, &positions);
        assert!(check.stress_gap < 1e-9 && check.final_displacement < 1e-12, "{} {}", check.stress_gap, check.final_displacement);
    }
}
