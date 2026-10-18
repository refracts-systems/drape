use crate::drape::{Drape, DrapeStep, TangentKind};
use crate::io::Json;
use crate::linalg::{difference_norm, vector_norm, Vec3};
use crate::newton::NewtonSystem;
use crate::rng::Lcg;

const FD_STEP: f64 = 1e-7;
const FD_DIRECTIONS: usize = 6;
const LINE_STEP: f64 = 1e-3;

/// FD checks of Φ with the exact tangent (hinges included) at a random bent, moving state.
pub struct DrapeDerivativeCheck {
    pub gradient_error: f64,
    pub hessian_error: f64,
    pub hinge_energy_change_error: f64,
    pub energy_change_error: f64,
}

pub fn drape_derivative_check(columns: usize) -> DrapeDerivativeCheck {
    let drape = Drape::build(columns, TangentKind::Exact);
    let mut rng = Lcg::new(53);
    let x = bent_state(&drape, &mut rng);
    let x_hat: Vec<Vec3> = x.iter().map(|p| p.map(|v| v + 0.02 * spacing(&drape) * rng.uniform())).collect();
    let step = DrapeStep { drape: &drape, x_hat };
    let directions: Vec<Vec<Vec3>> = (0..FD_DIRECTIONS).map(|_| direction(&drape, &mut rng)).collect();
    let worst = |f: &dyn Fn(&[Vec3]) -> f64| directions.iter().map(|v| f(v)).fold(0.0, f64::max);
    DrapeDerivativeCheck {
        gradient_error: worst(&|v| gradient_error(&step, &x, v)),
        hessian_error: worst(&|v| hessian_error(&step, &x, v)),
        hinge_energy_change_error: worst(&|v| hinge_change_error(&drape, &x, v)),
        energy_change_error: worst(&|v| total_change_error(&step, &x, v)),
    }
}

fn spacing(drape: &Drape) -> f64 {
    drape.mesh.width / (drape.mesh.columns - 1) as f64
}

/// A smooth fold plus noise: hinges are bent (θ ≠ 0) and membranes stretched and compressed.
pub fn bent_state(drape: &Drape, rng: &mut Lcg) -> Vec<Vec3> {
    let (w, h, s) = (drape.mesh.width, drape.mesh.height, spacing(drape));
    let fold = |p: [f64; 2]| 0.05 * (std::f64::consts::PI * p[0] / w).sin() * (2.0 * std::f64::consts::PI * p[1] / h).cos();
    drape.mesh.rest.iter().map(|p| [p[0] + 0.1 * s * rng.uniform(), p[1] + 0.1 * s * rng.uniform(), fold(*p) + 0.1 * s * rng.uniform()]).collect()
}

fn direction(drape: &Drape, rng: &mut Lcg) -> Vec<Vec3> {
    let mut v = vec![[0.0; 3]; drape.mesh.rest.len()];
    drape.free.iter().for_each(|&i| v[i] = [rng.uniform(), rng.uniform(), rng.uniform()]);
    v
}

fn shifted(x: &[Vec3], v: &[Vec3], h: f64) -> Vec<Vec3> {
    x.iter().zip(v).map(|(p, d)| std::array::from_fn(|k| p[k] + h * d[k])).collect()
}

fn gradient_error(step: &DrapeStep, x: &[Vec3], v: &[Vec3]) -> f64 {
    let fd = (step.value(&shifted(x, v, FD_STEP)) - step.value(&shifted(x, v, -FD_STEP))) / (2.0 * FD_STEP);
    let exact: f64 = step.residual(x).iter().zip(step.gather(v)).map(|(g, d)| g * d).sum();
    (fd - exact).abs() / exact.abs()
}

fn hessian_error(step: &DrapeStep, x: &[Vec3], v: &[Vec3]) -> f64 {
    let (plus, minus) = (step.residual(&shifted(x, v, FD_STEP)), step.residual(&shifted(x, v, -FD_STEP)));
    let fd: Vec<f64> = plus.iter().zip(&minus).map(|(p, m)| (p - m) / (2.0 * FD_STEP)).collect();
    let exact = step.tangent(x).multiply(&step.gather(v));
    difference_norm(&fd, &exact) / vector_norm(&exact)
}

fn hinge_change_error(drape: &Drape, x: &[Vec3], v: &[Vec3]) -> f64 {
    let moved = shifted(x, v, LINE_STEP);
    let direct: f64 = drape.hinges.iter().map(|h| h.energy(&moved) - h.energy(x)).sum();
    let change: f64 = drape.hinges.iter().map(|h| h.energy_change(x, v, LINE_STEP)).sum();
    (change - direct).abs() / direct.abs()
}

fn total_change_error(step: &DrapeStep, x: &[Vec3], v: &[Vec3]) -> f64 {
    let direct = step.value(&shifted(x, v, LINE_STEP)) - step.value(x);
    (step.energy_change(x, v, LINE_STEP) - direct).abs() / direct.abs()
}

impl DrapeDerivativeCheck {
    pub fn json(&self) -> Json {
        Json::new()
            .number("fd_step_m", FD_STEP)
            .integer("directions", FD_DIRECTIONS)
            .number("gradient_relative_error", self.gradient_error)
            .number("hessian_relative_error", self.hessian_error)
            .number("hinge_energy_change_relative_error", self.hinge_energy_change_error)
            .number("energy_change_relative_error", self.energy_change_error)
    }
}

/// Total energy per state and the largest per-step increase relative to max |E| (E₀ = 0 for the flat sheet at rest).
pub struct EnergyHistory {
    pub energies: Vec<f64>,
    pub scale: f64,
    pub worst_increase: f64,
}

pub fn energy_history(drape: &Drape, positions: &[Vec<Vec3>], velocities: &[Vec<Vec3>]) -> EnergyHistory {
    let energies: Vec<f64> = positions.iter().zip(velocities).map(|(x, v)| drape.total_energy(x, v)).collect();
    let scale = energies.iter().fold(0.0f64, |m, e| m.max(e.abs()));
    let worst_increase = energies.windows(2).map(|w| (w[1] - w[0]) / scale).fold(f64::NEG_INFINITY, f64::max);
    EnergyHistory { energies, scale, worst_increase }
}

/// Backward-Euler velocities (x_{n+1} − x_n)/h of a stored trajectory, v₀ = 0.
pub fn velocities_of(positions: &[Vec<Vec3>], time_step: f64) -> Vec<Vec<Vec3>> {
    let zero = vec![[0.0; 3]; positions[0].len()];
    std::iter::once(zero)
        .chain(positions.windows(2).map(|w| w[1].iter().zip(&w[0]).map(|(p, q)| std::array::from_fn(|d| (p[d] - q[d]) / time_step)).collect()))
        .collect()
}
