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

/// A patch of consecutive free vertices with a random direction on it, for a derivative check whose rounding does not
/// grow with the mesh.
struct Patch {
    vertices: Vec<usize>,
    direction: Vec<Vec3>,
}

/// Vertices per patch.
const PATCH_SIZE: usize = 16;

fn patches(drape: &Drape, rng: &mut Lcg) -> Vec<Patch> {
    let span = drape.free.len() - PATCH_SIZE;
    (0..FD_DIRECTIONS).map(|_| {
        let start = (((rng.uniform() + 1.0) / 2.0) * span as f64) as usize;
        let vertices: Vec<usize> = drape.free[start..start + PATCH_SIZE].to_vec();
        let mut direction = vec![[0.0; 3]; drape.mesh.rest.len()];
        vertices.iter().for_each(|&i| direction[i] = [rng.uniform(), rng.uniform(), rng.uniform()]);
        Patch { vertices, direction }
    }).collect()
}

fn global_directions(drape: &Drape, rng: &mut Lcg) -> Vec<Patch> {
    (0..FD_DIRECTIONS).map(|_| {
        let mut direction = vec![[0.0; 3]; drape.mesh.rest.len()];
        drape.free.iter().for_each(|&i| direction[i] = [rng.uniform(), rng.uniform(), rng.uniform()]);
        Patch { vertices: drape.free.clone(), direction }
    }).collect()
}

fn shifted(x: &[Vec3], v: &[Vec3], h: f64) -> Vec<Vec3> {
    x.iter().zip(v).map(|(p, d)| std::array::from_fn(|k| p[k] + h * d[k])).collect()
}

/// The five errors of one direction: Φ gradient, Φ Hessian, barrier gradient, barrier Hessian, barrier energy change.
/// Differences of Φ and of the barrier use sums over the patch only (all vertices for a global direction).
fn errors(step: &DrapeStep, x: &[Vec3], patch: &Patch, scaled_phi_step: bool) -> [f64; 5] {
    let drape = step.drape;
    let (v, rows) = (&patch.direction, step.gather(&patch.direction));
    let barrier_step = BARRIER_FD_FRACTION * patch.vertices.iter().map(|&i| drape.contact.min_distance(x[i])).fold(f64::INFINITY, f64::min);
    let local_barrier = |z: &[Vec3]| patch.vertices.iter().map(|&i| drape.contact.energy(z[i])).sum::<f64>();
    let phi_step = if scaled_phi_step { barrier_step } else { FD_STEP };
    let alpha = 0.5 * step.max_step(x, v).min(ACTIVATION_DISTANCE);
    let identity = {
        let direct = local_barrier(&shifted(x, v, alpha)) - local_barrier(x);
        (step.barrier_change(x, v, alpha) - direct).abs() / direct.abs()
    };
    [
        directional_error(&|z| step.local_value(z, &patch.vertices), &step.residual(x), &rows, x, v, phi_step),
        hessian_error(&|z| step.residual(z), &step.tangent(x).multiply(&rows), x, v, phi_step),
        directional_error(&local_barrier, &barrier_rows(drape, x), &rows, x, v, barrier_step),
        hessian_error(&|z| barrier_rows(drape, z), &barrier_hessian_times(drape, x, v), x, v, barrier_step),
        identity,
    ]
}

fn worst(step: &DrapeStep, x: &[Vec3], patches: &[Patch], scaled_phi_step: bool) -> [f64; 5] {
    patches.iter().map(|p| errors(step, x, p, scaled_phi_step)).fold([0.0; 5], |w, e| std::array::from_fn(|k| w[k].max(e[k])))
}

/// FD checks with every barrier active: Φ (inertia + elastic + barrier), the barrier alone and the barrier energy-change
/// identity. The gated numbers use patch directions; the global-direction numbers are reported without a gate.
pub struct ContactDerivativeCheck {
    pub patch: [f64; 5],
    pub global: [f64; 5],
    pub patch_scaled_phi_step: [f64; 5],
}

pub fn contact_derivative_check(drape: &Drape) -> ContactDerivativeCheck {
    let mut rng = Lcg::new(59);
    let x = contact_state(drape);
    let x_hat: Vec<Vec3> = x.iter().map(|p| p.map(|v| v + 1e-4 * rng.uniform())).collect();
    let step = DrapeStep { drape, x_hat };
    let drawn = patches(drape, &mut rng);
    let (patch, scaled) = (worst(&step, &x, &drawn, false), worst(&step, &x, &drawn, true));
    ContactDerivativeCheck { patch, global: worst(&step, &x, &global_directions(drape, &mut rng), false), patch_scaled_phi_step: scaled }
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

impl ContactDerivativeCheck {
    pub fn json(&self) -> Json {
        let set = |e: &[f64; 5]| Json::new().number("gradient_relative_error", e[0]).number("hessian_relative_error", e[1])
            .number("barrier_gradient_relative_error", e[2]).number("barrier_hessian_relative_error", e[3])
            .number("barrier_energy_change_relative_error", e[4]);
        Json::new()
            .number("fd_step_m", FD_STEP)
            .integer("directions", FD_DIRECTIONS)
            .integer("patch_vertices", PATCH_SIZE)
            .raw("patch", set(&self.patch).render())
            .raw("global_no_gate", set(&self.global).render())
            .raw("patch_phi_step_scaled_no_gate", set(&self.patch_scaled_phi_step).render())
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

    /// The derivative code is checked with the Φ finite-difference step scaled to the smallest distance of the patch: the fixed
    /// step of the gate (1e-7 m) has a truncation error (h/d)² of its own once a vertex is within ~1e-5 m of a collider.
    #[test]
    fn active_barrier_derivatives_match_finite_differences() {
        let check = contact_derivative_check(&scene());
        let e = check.patch_scaled_phi_step;
        assert!(e[0] <= 1e-6 && e[2] <= 1e-6, "{} {}", e[0], e[2]);
        assert!(e[1] <= 1e-5 && e[3] <= 1e-5, "{} {}", e[1], e[3]);
        assert!(e[4] <= 1e-12, "{}", e[4]);
        assert!(check.patch[2] <= 1e-6 && check.patch[3] <= 1e-5, "{:?}", check.patch);
    }
}
