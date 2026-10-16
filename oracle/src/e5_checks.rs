use crate::contact::{Collider, KEEP_FRACTION};
use crate::drape::{Drape, DrapeStep, Trajectory, ACTIVATION_DISTANCE};
use crate::io::{BinaryArrays, Json};
use crate::linalg::{difference_norm, vector_norm, Vec3};
use crate::newton::NewtonSystem;
use crate::rng::Lcg;

const FD_STEP: f64 = 1e-7;
/// The barrier-only FD step is this fraction of the smallest distance: central-difference truncation grows as (h/d)², and
/// the E3 step 1e-7 m gave 6.4e-6 at d = 2.5e-5 m in the first n17 run (a measurement artefact, not a derivative error).
const BARRIER_FD_FRACTION: f64 = 1e-3;
const FD_DIRECTIONS: usize = 6;
/// Contact-state distances span [0.05·d̂, d̂], log-uniform.
const NEAREST: f64 = 0.05;
/// Sheet vertices within this fraction of the sphere radius (in xy) of its centre lie on the sphere, the rest on the floor.
const SPHERE_PATCH: f64 = 0.6;

/// Which colliders a vertex is within d̂ of.
fn near(drape: &Drape, x: Vec3, sphere: bool) -> bool {
    drape.contact.colliders.iter().filter(|c| matches!(c, Collider::Sphere { .. }) == sphere).any(|c| c.distance(x) < drape.contact.barrier.dhat)
}

/// Per state: free vertices within d̂ of the sphere and of the floor.
pub fn active_counts(drape: &Drape, positions: &[Vec<Vec3>]) -> Vec<(usize, usize)> {
    positions.iter().map(|x| {
        let count = |sphere: bool| drape.free.iter().filter(|&&i| near(drape, x[i], sphere)).count();
        (count(true), count(false))
    }).collect()
}

/// First state with an active sphere contact, first with an active floor contact, and the state with most active contacts.
pub struct ContactEvents {
    pub first_sphere: Option<usize>,
    pub first_floor: Option<usize>,
    pub most_active: usize,
}

pub fn contact_events(drape: &Drape, positions: &[Vec<Vec3>]) -> ContactEvents {
    let counts = active_counts(drape, positions);
    let total = |k: usize| counts[k].0 + counts[k].1;
    let most_active = (1..counts.len()).fold(1, |best, k| if total(k) > total(best) { k } else { best });
    ContactEvents {
        first_sphere: (1..counts.len()).find(|&k| counts[k].0 > 0),
        first_floor: (1..counts.len()).find(|&k| counts[k].1 > 0),
        most_active,
    }
}

/// Probe steps: 1, the first sphere and floor contacts, the most active step and the last step (sorted, without repeats).
pub fn event_probes(drape: &Drape, positions: &[Vec<Vec3>]) -> Vec<usize> {
    let events = contact_events(drape, positions);
    let mut steps: Vec<usize> = [Some(1), events.first_sphere, events.first_floor, Some(events.most_active), Some(positions.len() - 1)].into_iter().flatten().collect();
    steps.sort_unstable();
    steps.dedup();
    steps
}

/// A state with every free vertex inside the barrier: the centre patch radially above the sphere, the rest above the floor,
/// at distances log-uniform in [0.05·d̂, d̂].
pub fn contact_state(drape: &Drape) -> Vec<Vec3> {
    let mut rng = Lcg::new(131);
    let dhat = drape.contact.barrier.dhat;
    drape.mesh.rest.iter().map(|p| {
        let d = dhat * NEAREST.powf(0.5 * (1.0 + rng.uniform()));
        drape.contact.colliders.iter().find_map(|c| on_sphere(c, *p, d)).unwrap_or([p[0], p[1], d])
    }).collect()
}

fn on_sphere(collider: &Collider, p: [f64; 2], d: f64) -> Option<Vec3> {
    let Collider::Sphere { centre, radius } = *collider else { return None };
    let (dx, dy) = (p[0] - centre[0], p[1] - centre[1]);
    if (dx * dx + dy * dy).sqrt() >= SPHERE_PATCH * radius {
        return None;
    }
    let n = [dx / radius, dy / radius, (1.0 - (dx * dx + dy * dy) / (radius * radius)).sqrt()];
    Some(std::array::from_fn(|k| centre[k] + (radius + d) * n[k]))
}

/// "colliders": [count, d̂, κ, 0, then 8 floats per collider]; "contact_state": the kernel-check state.
pub fn push_contact_arrays(arrays: &mut BinaryArrays, drape: &Drape) {
    let barrier = drape.contact.barrier;
    let header = [drape.contact.colliders.len() as f64, barrier.dhat, barrier.kappa, 0.0];
    let packed = drape.contact.colliders.iter().flat_map(|c| c.packed());
    arrays.push_f32("colliders", header.into_iter().chain(packed).map(|v| v as f32));
    arrays.push_f32("contact_state", contact_state(drape).into_iter().flatten().map(|v| v as f32));
}

fn optional(value: Option<usize>) -> f64 {
    value.map_or(f64::NAN, |v| v as f64)
}

pub fn contact_json(drape: &Drape, trajectory: Option<&Trajectory>) -> Json {
    let barrier = drape.contact.barrier;
    let json = Json::new().integer("colliders", drape.contact.colliders.len()).number("dhat_m", barrier.dhat).number("kappa", barrier.kappa)
        .number("keep_fraction", KEEP_FRACTION).number("release_height_m", drape.lift);
    let Some(t) = trajectory.filter(|_| drape.has_contact()) else { return json };
    let events = contact_events(drape, &t.positions);
    json.number("first_sphere_contact_step", optional(events.first_sphere)).number("first_floor_contact_step", optional(events.first_floor))
        .integer("most_active_step", events.most_active)
}

/// FD checks with every barrier active: Φ (inertia + elastic + barrier) and the barrier alone, plus the barrier
/// energy-change identity (per-vertex change vs difference of totals).
pub struct ContactDerivativeCheck {
    pub gradient_error: f64,
    pub hessian_error: f64,
    pub barrier_gradient_error: f64,
    pub barrier_hessian_error: f64,
    pub barrier_change_error: f64,
    pub barrier_step: f64,
}

fn direction(drape: &Drape, rng: &mut Lcg) -> Vec<Vec3> {
    let mut v = vec![[0.0; 3]; drape.mesh.rest.len()];
    drape.free.iter().for_each(|&i| v[i] = [rng.uniform(), rng.uniform(), rng.uniform()]);
    v
}

fn shifted(x: &[Vec3], v: &[Vec3], h: f64) -> Vec<Vec3> {
    x.iter().zip(v).map(|(p, d)| std::array::from_fn(|k| p[k] + h * d[k])).collect()
}

pub fn contact_derivative_check(drape: &Drape) -> ContactDerivativeCheck {
    let mut rng = Lcg::new(59);
    let x = contact_state(drape);
    let x_hat: Vec<Vec3> = x.iter().map(|p| p.map(|v| v + 1e-4 * rng.uniform())).collect();
    let step = DrapeStep { drape, x_hat };
    let directions: Vec<Vec<Vec3>> = (0..FD_DIRECTIONS).map(|_| direction(drape, &mut rng)).collect();
    let worst = |f: &dyn Fn(&[Vec3]) -> f64| directions.iter().map(|v| f(v)).fold(0.0, f64::max);
    let barrier_step = BARRIER_FD_FRACTION * drape.free.iter().map(|&i| drape.contact.min_distance(x[i])).fold(f64::INFINITY, f64::min);
    ContactDerivativeCheck {
        gradient_error: worst(&|v| directional_error(&|z| step.value(z), &step.residual(&x), &step.gather(v), &x, v, FD_STEP)),
        hessian_error: worst(&|v| hessian_error(&|z| step.residual(z), &step.tangent(&x).multiply(&step.gather(v)), &x, v, FD_STEP)),
        barrier_gradient_error: worst(&|v| directional_error(&|z| drape.barrier_energy(z), &barrier_rows(drape, &x), &step.gather(v), &x, v, barrier_step)),
        barrier_hessian_error: worst(&|v| hessian_error(&|z| barrier_rows(drape, z), &barrier_hessian_times(drape, &x, v), &x, v, barrier_step)),
        barrier_change_error: worst(&|v| barrier_change_error(&step, &x, v)),
        barrier_step,
    }
}

fn barrier_rows(drape: &Drape, x: &[Vec3]) -> Vec<f64> {
    drape.free.iter().flat_map(|&i| drape.contact.gradient(x[i])).collect()
}

fn barrier_hessian_times(drape: &Drape, x: &[Vec3], v: &[Vec3]) -> Vec<f64> {
    drape.free.iter().flat_map(|&i| {
        let h = drape.contact.hessian(x[i]);
        std::array::from_fn::<f64, 3, _>(|r| (0..3).map(|c| h[r][c] * v[i][c]).sum())
    }).collect()
}

fn directional_error(value: &dyn Fn(&[Vec3]) -> f64, gradient: &[f64], rows: &[f64], x: &[Vec3], v: &[Vec3], h: f64) -> f64 {
    let fd = (value(&shifted(x, v, h)) - value(&shifted(x, v, -h))) / (2.0 * h);
    let exact: f64 = gradient.iter().zip(rows).map(|(g, d)| g * d).sum();
    (fd - exact).abs() / exact.abs()
}

fn hessian_error(gradient: &dyn Fn(&[Vec3]) -> Vec<f64>, exact: &[f64], x: &[Vec3], v: &[Vec3], h: f64) -> f64 {
    let (plus, minus) = (gradient(&shifted(x, v, h)), gradient(&shifted(x, v, -h)));
    let fd: Vec<f64> = plus.iter().zip(&minus).map(|(p, m)| (p - m) / (2.0 * h)).collect();
    difference_norm(&fd, exact) / vector_norm(exact)
}

/// Along half the largest admissible step, so every vertex stays strictly outside its collider.
fn barrier_change_error(step: &DrapeStep, x: &[Vec3], v: &[Vec3]) -> f64 {
    let alpha = 0.5 * step.max_step(x, v).min(ACTIVATION_DISTANCE);
    let direct = step.drape.barrier_energy(&shifted(x, v, alpha)) - step.drape.barrier_energy(x);
    (step.barrier_change(x, v, alpha) - direct).abs() / direct.abs()
}

impl ContactDerivativeCheck {
    pub fn json(&self) -> Json {
        Json::new()
            .number("fd_step_m", FD_STEP)
            .number("barrier_fd_step_m", self.barrier_step)
            .integer("directions", FD_DIRECTIONS)
            .number("gradient_relative_error", self.gradient_error)
            .number("hessian_relative_error", self.hessian_error)
            .number("barrier_gradient_relative_error", self.barrier_gradient_error)
            .number("barrier_hessian_relative_error", self.barrier_hessian_error)
            .number("barrier_energy_change_relative_error", self.barrier_change_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drape::{Scene, TangentKind};

    fn scene() -> Drape {
        Drape::for_scene(9, TangentKind::Exact, Scene::SphereOnFloor)
    }

    #[test]
    fn contact_state_puts_every_vertex_inside_the_barrier_on_both_colliders() {
        let drape = scene();
        let x = contact_state(&drape);
        let dhat = drape.contact.barrier.dhat;
        assert!(drape.free.iter().all(|&i| (NEAREST * dhat * 0.999..dhat).contains(&drape.contact.min_distance(x[i]))));
        let counts = active_counts(&drape, &[x]);
        assert!(counts[0].0 > 0 && counts[0].1 > 0, "{:?}", counts);
    }

    #[test]
    fn active_barrier_derivatives_match_finite_differences() {
        let check = contact_derivative_check(&scene());
        assert!(check.gradient_error <= 1e-6 && check.barrier_gradient_error <= 1e-6, "{} {}", check.gradient_error, check.barrier_gradient_error);
        assert!(check.hessian_error <= 1e-5 && check.barrier_hessian_error <= 1e-5, "{} {}", check.hessian_error, check.barrier_hessian_error);
        assert!(check.barrier_change_error <= 1e-12, "{}", check.barrier_change_error);
    }
}
