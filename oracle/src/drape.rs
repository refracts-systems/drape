use crate::bsr::Bsr;
use crate::contact::{Barrier, Collider, Contact};
use crate::cases::{hinge_elements, GRAVITY, SHEET_SIZE, TIME_STEP};
use crate::hinge::HingeElement;
use crate::linalg::{dot, identity_scaled, vector_norm, Mat3, Vec3};
use crate::material::Material;
use crate::membrane::MembraneElement;
use crate::mesh::Mesh;
use crate::newton::{solve_step, solve_step_regularized, NewtonSettings, NewtonSystem, RegularizedSystem, StepSolution};

/// 1.0 s of backward Euler at h = 1/240 s.
pub const STEPS: usize = 240;

/// Newton tangent. `Projected` (clamped membrane stress + Gauss–Newton hinge) replaced `Exact` after the band Cholesky
/// reported a non-positive-definite exact tangent at step 87 of the n33 drape: the evidence-triggered change.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TangentKind {
    Exact,
    Projected,
}

impl TangentKind {
    pub fn name(&self) -> &'static str {
        match self {
            TangentKind::Exact => "exact",
            TangentKind::Projected => "projected",
        }
    }
}

/// What the sheet does: E3/E4 hang from the two top corners; E5 falls free onto a sphere resting on a floor.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Scene {
    PinnedCorners,
    SphereOnFloor,
}

impl Scene {
    pub fn name(&self) -> &'static str {
        match self {
            Scene::PinnedCorners => "pinned_corners",
            Scene::SphereOnFloor => "sphere_on_floor",
        }
    }

    /// problem.json files written before Experiment 5 have no scene: they are the pinned drape.
    pub fn parse(text: &str) -> Scene {
        if text == "sphere_on_floor" { Scene::SphereOnFloor } else { Scene::PinnedCorners }
    }
}

/// Experiment 5 scene, in units of the sheet side L: sphere radius 0.25·L under the sheet centre on the floor z = 0,
/// barrier activation d̂ = 1e-3·L, sheet released flat at 2r + 0.02·L.
const SPHERE_RADIUS: f64 = 0.25 * SHEET_SIZE;
pub const ACTIVATION_DISTANCE: f64 = 1e-3 * SHEET_SIZE;
const RELEASE_HEIGHT: f64 = 2.0 * SPHERE_RADIUS + 0.02 * SHEET_SIZE;

fn scene_contact(scene: Scene, mass: &[f64]) -> (Contact, f64) {
    if scene == Scene::PinnedCorners {
        return (Contact::none(), 0.0);
    }
    let floor = Collider::Plane { normal: [0.0, 0.0, 1.0], offset: 0.0 };
    let ball = Collider::Sphere { centre: [0.5 * SHEET_SIZE, 0.5 * SHEET_SIZE, SPHERE_RADIUS], radius: SPHERE_RADIUS };
    let heaviest = mass.iter().copied().fold(0.0, f64::max);
    let barrier = Barrier { dhat: ACTIVATION_DISTANCE, kappa: Barrier::resting_kappa(ACTIVATION_DISTANCE, heaviest, GRAVITY) };
    (Contact { colliders: vec![floor, ball], barrier }, RELEASE_HEIGHT)
}

/// A 0.5 m knit sheet, flat in xy at height `lift`, falling under gravity (−z): pinned at its two top corners (E3/E4),
/// or free above analytic colliders (E5).
pub struct Drape {
    pub mesh: Mesh,
    pub material: Material,
    pub membranes: Vec<MembraneElement>,
    pub hinges: Vec<HingeElement>,
    pub fixed: Vec<usize>,
    pub row_of: Vec<Option<usize>>,
    pub free: Vec<usize>,
    pub pattern: Bsr,
    pub mass: Vec<f64>,
    pub tangent_kind: TangentKind,
    pub scene: Scene,
    pub contact: Contact,
    pub lift: f64,
}

/// Free vertices, their unknown rows and the BSR pattern of membranes and hinges over them.
fn unknown_layout(mesh: &Mesh, fixed: &[usize]) -> (Vec<usize>, Vec<Option<usize>>, Bsr) {
    let free: Vec<usize> = (0..mesh.rest.len()).filter(|v| !fixed.contains(v)).collect();
    let mut row_of = vec![None; mesh.rest.len()];
    free.iter().enumerate().for_each(|(row, &v)| row_of[v] = Some(row));
    let to_rows = |vs: &[usize]| vs.iter().filter_map(|&v| row_of[v]).collect::<Vec<usize>>();
    let groups: Vec<Vec<usize>> = mesh.triangles.iter().map(|t| to_rows(t)).chain(mesh.hinges.iter().map(|h| to_rows(h))).collect();
    let pattern = Bsr::with_pattern(free.len(), &groups);
    (free, row_of, pattern)
}

impl Drape {
    pub fn build(columns: usize, tangent_kind: TangentKind) -> Drape {
        Drape::for_scene(columns, tangent_kind, Scene::PinnedCorners)
    }

    pub fn for_scene(columns: usize, tangent_kind: TangentKind, scene: Scene) -> Drape {
        let mesh = Mesh::grid(columns, columns, SHEET_SIZE, SHEET_SIZE);
        let material = Material::knit_baseline();
        let fixed = if scene == Scene::PinnedCorners { vec![(mesh.rows - 1) * mesh.columns, mesh.rows * mesh.columns - 1] } else { Vec::new() };
        let (free, row_of, pattern) = unknown_layout(&mesh, &fixed);
        let membranes: Vec<MembraneElement> = mesh.triangles.iter().map(|&t| MembraneElement::new(t, &mesh.rest)).collect();
        let hinges = hinge_elements(&mesh, &material);
        let mass: Vec<f64> = mesh.vertex_areas().iter().map(|a| material.density * a).collect();
        let (contact, lift) = scene_contact(scene, &mass);
        Drape { mesh, material, membranes, hinges, fixed, row_of, free, pattern, mass, tangent_kind, scene, contact, lift }
    }

    /// The drape a problem.json describes (its "scene", pinned when absent).
    pub fn from_problem(problem: &str, tangent_kind: TangentKind) -> Drape {
        let columns = crate::jsonscan::number(problem, "", "grid_columns") as usize;
        Drape::for_scene(columns, tangent_kind, Scene::parse(&crate::jsonscan::string(problem, "", "scene")))
    }

    pub fn rows(&self) -> usize {
        self.free.len()
    }

    pub fn rest_positions(&self) -> Vec<Vec3> {
        self.mesh.rest.iter().map(|p| [p[0], p[1], self.lift]).collect()
    }

    pub fn has_contact(&self) -> bool {
        !self.contact.colliders.is_empty()
    }

    /// x̂ = x + h·v + h²·g on free vertices; pinned vertices stay where they are.
    pub fn prediction(&self, x: &[Vec3], v: &[Vec3]) -> Vec<Vec3> {
        let mut predicted = x.to_vec();
        for &i in &self.free {
            let gravity = [0.0, 0.0, -GRAVITY];
            (0..3).for_each(|d| predicted[i][d] = x[i][d] + TIME_STEP * v[i][d] + TIME_STEP * TIME_STEP * gravity[d]);
        }
        predicted
    }

    pub fn potential(&self, x: &[Vec3]) -> f64 {
        let membrane: f64 = self.membranes.iter().map(|e| e.energy(x, &self.material)).sum();
        let elastic = membrane + self.hinges.iter().map(|h| h.energy(x)).sum::<f64>();
        if self.has_contact() { elastic + self.barrier_energy(x) } else { elastic }
    }

    /// Σ over free vertices of the collider barriers.
    pub fn barrier_energy(&self, x: &[Vec3]) -> f64 {
        self.free.iter().map(|&i| self.contact.energy(x[i])).sum()
    }

    /// ∂(Π_membrane + Π_hinge + Π_barrier)/∂x for every vertex.
    pub fn potential_gradient(&self, x: &[Vec3]) -> Vec<Vec3> {
        let mut gradient = vec![[0.0; 3]; x.len()];
        for element in &self.membranes {
            scatter_add(&mut gradient, &element.vertices, &element.gradient(x, &self.material));
        }
        if self.has_contact() {
            self.free.iter().for_each(|&i| scatter_add(&mut gradient, &[i], &[self.contact.gradient(x[i])]));
        }
        for hinge in &self.hinges {
            scatter_add(&mut gradient, &hinge.vertices, &hinge.gradient(x));
        }
        gradient
    }

    /// Kinetic + elastic + gravitational energy (z up, pinned vertices at rest).
    pub fn total_energy(&self, x: &[Vec3], v: &[Vec3]) -> f64 {
        let kinetic: f64 = (0..x.len()).map(|i| 0.5 * self.mass[i] * dot(v[i], v[i])).sum();
        let gravitational: f64 = (0..x.len()).map(|i| self.mass[i] * GRAVITY * x[i][2]).sum();
        kinetic + gravitational + self.potential(x)
    }

    pub fn mass_over_h2(&self, vertex: usize) -> f64 {
        self.mass[vertex] / (TIME_STEP * TIME_STEP)
    }

    pub fn membrane_hessian(&self, element: &MembraneElement, x: &[Vec3]) -> [[Mat3; 3]; 3] {
        match self.tangent_kind {
            TangentKind::Exact => element.tangent(x, &self.material),
            TangentKind::Projected => element.hessian(x, &self.material),
        }
    }

    pub fn hinge_hessian(&self, hinge: &HingeElement, x: &[Vec3]) -> [[Mat3; 4]; 4] {
        match self.tangent_kind {
            TangentKind::Exact => hinge.tangent(x),
            TangentKind::Projected => hinge.hessian(x),
        }
    }

    fn add_blocks<const N: usize>(&self, matrix: &mut Bsr, vertices: &[usize; N], blocks: &[[Mat3; N]; N]) {
        for (k, &vk) in vertices.iter().enumerate() {
            for (l, &vl) in vertices.iter().enumerate() {
                if let (Some(r), Some(c)) = (self.row_of[vk], self.row_of[vl]) {
                    matrix.add(r, c, &blocks[k][l]);
                }
            }
        }
    }
}

fn scatter_add<const N: usize>(target: &mut [Vec3], vertices: &[usize; N], values: &[Vec3; N]) {
    for (k, &v) in vertices.iter().enumerate() {
        (0..3).for_each(|d| target[v][d] += values[k][d]);
    }
}

/// One backward-Euler step as an energy: Φ(x) = 1/(2h²)(x − x̂)ᵀM(x − x̂) + Π(x).
pub struct DrapeStep<'a> {
    pub drape: &'a Drape,
    pub x_hat: Vec<Vec3>,
}

impl DrapeStep<'_> {
    /// Φ itself, for finite-difference checks only (Newton never subtracts two values of it).
    pub fn value(&self, x: &[Vec3]) -> f64 {
        let inertia: f64 = self.drape.free.iter().map(|&i| {
            let offset: Vec3 = std::array::from_fn(|k| x[i][k] - self.x_hat[i][k]);
            0.5 * self.drape.mass_over_h2(i) * dot(offset, offset)
        }).sum();
        inertia + self.drape.potential(x)
    }

    /// Φ restricted to what a perturbation of the vertices in `patch` can change: their inertia and barrier terms and the
    /// membranes and hinges that touch them. Differences of it equal differences of Φ without the rounding of the global sum.
    pub fn local_value(&self, x: &[Vec3], patch: &[usize]) -> f64 {
        let mut touched = vec![false; self.drape.mesh.rest.len()];
        patch.iter().for_each(|&i| touched[i] = true);
        let drape = self.drape;
        let inertia: f64 = patch.iter().map(|&i| {
            let offset: Vec3 = std::array::from_fn(|k| x[i][k] - self.x_hat[i][k]);
            0.5 * drape.mass_over_h2(i) * dot(offset, offset) + drape.contact.energy(x[i])
        }).sum();
        let membrane: f64 = drape.membranes.iter().filter(|e| e.vertices.iter().any(|&v| touched[v])).map(|e| e.energy(x, &drape.material)).sum();
        inertia + membrane + drape.hinges.iter().filter(|h| h.vertices.iter().any(|&v| touched[v])).map(|h| h.energy(x)).sum::<f64>()
    }

    /// α_max (the minimum over free vertices) with its limiting vertex and collider, for the replay tolerance.
    pub fn max_step_detail(&self, x: &[Vec3], d: &[Vec3]) -> (f64, Option<(usize, usize)>) {
        self.drape.free.iter().fold((1.0, None), |(alpha, at), &i| {
            let (step, collider) = self.drape.contact.max_step_limited(x[i], d[i]);
            match collider {
                Some(k) if step < alpha => (step, Some((i, k))),
                _ => (alpha, at),
            }
        })
    }

    /// Σ over free vertices of b(d(x + αd)) − b(d(x)), per vertex as the WGSL kernel forms it.
    pub fn barrier_change(&self, x: &[Vec3], d: &[Vec3], alpha: f64) -> f64 {
        let contact = &self.drape.contact;
        self.drape.free.iter().map(|&i| contact.energy(std::array::from_fn(|k| x[i][k] + alpha * d[i][k])) - contact.energy(x[i])).sum()
    }
}

impl NewtonSystem for DrapeStep<'_> {
    fn residual(&self, x: &[Vec3]) -> Vec<f64> {
        let gradient = self.drape.potential_gradient(x);
        self.drape.free.iter()
            .flat_map(|&i| std::array::from_fn::<f64, 3, _>(|d| self.drape.mass_over_h2(i) * (x[i][d] - self.x_hat[i][d]) + gradient[i][d]))
            .collect()
    }

    /// M/h² + membrane and hinge Hessians of the drape's tangent kind.
    fn tangent(&self, x: &[Vec3]) -> Bsr {
        let drape = self.drape;
        let mut matrix = Bsr { row_ptr: drape.pattern.row_ptr.clone(), col_idx: drape.pattern.col_idx.clone(), vals: drape.pattern.vals.clone() };
        drape.free.iter().enumerate().for_each(|(row, &i)| matrix.add(row, row, &identity_scaled(drape.mass_over_h2(i))));
        drape.membranes.iter().for_each(|e| drape.add_blocks(&mut matrix, &e.vertices, &drape.membrane_hessian(e, x)));
        drape.hinges.iter().for_each(|h| drape.add_blocks(&mut matrix, &h.vertices, &drape.hinge_hessian(h, x)));
        if drape.has_contact() {
            drape.free.iter().enumerate().for_each(|(row, &i)| matrix.add(row, row, &drape.contact.hessian(x[i])));
        }
        matrix
    }

    /// Inertia (x − x̂)ᵀM(αd) + ½α²dᵀMd, plus per-element membrane and hinge changes.
    fn energy_change(&self, x: &[Vec3], d: &[Vec3], alpha: f64) -> f64 {
        let drape = self.drape;
        let inertia: f64 = drape.free.iter().map(|&i| {
            let step: Vec3 = std::array::from_fn(|k| alpha * d[i][k]);
            let offset: Vec3 = std::array::from_fn(|k| x[i][k] - self.x_hat[i][k]);
            drape.mass_over_h2(i) * (dot(offset, step) + 0.5 * dot(step, step))
        }).sum();
        let membrane: f64 = drape.membranes.iter().map(|e| e.energy_change(x, d, alpha, &drape.material)).sum();
        let elastic = inertia + membrane + drape.hinges.iter().map(|h| h.energy_change(x, d, alpha)).sum::<f64>();
        if drape.has_contact() { elastic + self.barrier_change(x, d, alpha) } else { elastic }
    }

    fn scatter(&self, rows: &[f64]) -> Vec<Vec3> {
        let mut vertices = vec![[0.0; 3]; self.drape.mesh.rest.len()];
        self.drape.free.iter().enumerate().for_each(|(row, &v)| vertices[v] = [rows[3 * row], rows[3 * row + 1], rows[3 * row + 2]]);
        vertices
    }

    /// The largest α ≤ 1 keeping every free vertex at ≥ 0.1 of its distance to every collider.
    fn max_step(&self, x: &[Vec3], d: &[Vec3]) -> f64 {
        self.drape.free.iter().map(|&i| self.drape.contact.max_step(x[i], d[i])).fold(1.0, f64::min)
    }

    fn gather(&self, vertices: &[Vec3]) -> Vec<f64> {
        self.drape.free.iter().flat_map(|&v| vertices[v]).collect()
    }
}

impl RegularizedSystem for DrapeStep<'_> {
    /// K + μ·M/h²: the tangent of the drape's kind with μ times the inertia block added on the diagonal.
    fn regularized_tangent(&self, x: &[Vec3], mu: f64) -> Bsr {
        let mut matrix = self.tangent(x);
        self.drape.free.iter().enumerate().for_each(|(row, &i)| matrix.add(row, row, &identity_scaled(mu * self.drape.mass_over_h2(i))));
        matrix
    }

    fn residual_floor(&self, x: &[Vec3]) -> f64 {
        f64::EPSILON * (vector_norm(&self.absolute_contributions(x)) + vector_norm(&self.stiffness_times_positions(x)))
    }
}

impl DrapeStep<'_> {
    /// |K|·|x|: the entrywise absolute tangent (μ = 0) times the absolute unknown positions, i.e. the force error that the
    /// rounding of the positions themselves produces.
    pub fn stiffness_times_positions(&self, x: &[Vec3]) -> Vec<f64> {
        let mut tangent = self.tangent(x);
        tangent.vals.iter_mut().for_each(|v| *v = v.abs());
        tangent.multiply(&self.gather(x).iter().map(|v| v.abs()).collect::<Vec<f64>>())
    }

    /// Per unknown: |inertia| + Σ|membrane| + Σ|hinge| + |barrier| contributions to ∇Φ.
    pub fn absolute_contributions(&self, x: &[Vec3]) -> Vec<f64> {
        let drape = self.drape;
        let mut sums = vec![[0.0; 3]; x.len()];
        drape.membranes.iter().for_each(|e| scatter_add(&mut sums, &e.vertices, &e.gradient(x, &drape.material).map(|g| g.map(f64::abs))));
        drape.hinges.iter().for_each(|h| scatter_add(&mut sums, &h.vertices, &h.gradient(x).map(|g| g.map(f64::abs))));
        drape.free.iter().for_each(|&i| scatter_add(&mut sums, &[i], &[drape.contact.gradient(x[i]).map(f64::abs)]));
        drape.free.iter()
            .flat_map(|&i| std::array::from_fn::<f64, 3, _>(|d| (drape.mass_over_h2(i) * (x[i][d] - self.x_hat[i][d])).abs() + sums[i][d]))
            .collect()
    }
}

/// How each time step is solved: plain or regularised Newton, and the optional f64-floor convergence factor.
#[derive(Clone, Copy)]
pub struct Solver {
    pub regularized: bool,
    pub floor_factor: Option<f64>,
}

/// With colliders, Newton starts from x_n + α_max·(x̂ − x_n): the prediction cut back by the line-search rule so every
/// vertex keeps at least 0.1 of its distance (x̂ itself may lie inside a collider).
pub fn feasible_start(step: &DrapeStep, x: &[Vec3]) -> Vec<Vec3> {
    let towards: Vec<Vec3> = step.x_hat.iter().zip(x).map(|(p, q)| std::array::from_fn(|k| p[k] - q[k])).collect();
    let alpha = step.max_step(x, &towards);
    x.iter().zip(&towards).map(|(q, d)| std::array::from_fn(|k| q[k] + alpha * d[k])).collect()
}

/// One time step from (x, v): Newton on Φ starting at x̂ (or its feasible cut with colliders), residual scaled by ‖∇Φ(x)‖.
pub struct TimeStep {
    pub solution: StepSolution,
    pub scale: f64,
    pub velocity: Vec<Vec3>,
}

pub fn time_step(drape: &Drape, x: &[Vec3], v: &[Vec3], settings: &NewtonSettings, solver: Solver) -> Result<TimeStep, String> {
    let step = DrapeStep { drape, x_hat: drape.prediction(x, v) };
    let scale = vector_norm(&step.residual(x));
    let start = if drape.has_contact() { feasible_start(&step, x) } else { step.x_hat.clone() };
    let solution = if solver.regularized { solve_step_regularized(&step, start, scale, settings, solver.floor_factor)? } else { solve_step(&step, start, scale, settings)? };
    let velocity = solution.positions.iter().zip(x).map(|(p, q)| std::array::from_fn(|d| (p[d] - q[d]) / TIME_STEP)).collect();
    Ok(TimeStep { solution, scale, velocity })
}

/// States 0..=steps (positions and velocities) and every step's Newton record.
pub struct Trajectory {
    pub positions: Vec<Vec<Vec3>>,
    pub velocities: Vec<Vec<Vec3>>,
    pub steps: Vec<TimeStep>,
}

pub fn simulate(drape: &Drape, steps: usize, settings: &NewtonSettings, solver: Solver) -> Result<Trajectory, String> {
    let rest = drape.rest_positions();
    let mut trajectory = Trajectory { velocities: vec![vec![[0.0; 3]; rest.len()]], positions: vec![rest], steps: Vec::new() };
    for _ in 0..steps {
        let (x, v) = (trajectory.positions.last().unwrap(), trajectory.velocities.last().unwrap());
        let step = time_step(drape, x, v, settings, solver)?;
        let s = &step.solution;
        eprintln!("step {} {} iterations={} final={:.3e} max_mu={}", trajectory.steps.len() + 1, s.status.name(), s.searches.len(), s.residuals.last().unwrap(), s.regularization.max_mu);
        trajectory.positions.push(step.solution.positions.clone());
        trajectory.velocities.push(step.velocity.clone());
        trajectory.steps.push(step);
    }
    Ok(trajectory)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::e2_generate::reference_settings;

    #[test]
    fn residual_is_the_gradient_of_the_incremental_potential() {
        let drape = Drape::build(7, TangentKind::Exact);
        let x = drape.rest_positions();
        let v: Vec<Vec3> = (0..x.len()).map(|i| [0.1 * (i % 3) as f64, -0.05, 0.2]).collect();
        let step = DrapeStep { drape: &drape, x_hat: drape.prediction(&x, &v) };
        let mut y = x.clone();
        drape.free.iter().enumerate().for_each(|(k, &i)| y[i][2] += 1e-3 * ((k % 5) as f64 - 2.0));
        let phi = |z: &[Vec3]| {
            let inertia: f64 = drape.free.iter().map(|&i| 0.5 * drape.mass_over_h2(i) * (0..3).map(|d| (z[i][d] - step.x_hat[i][d]).powi(2)).sum::<f64>()).sum();
            inertia + drape.potential(z)
        };
        let residual = step.residual(&y);
        for (row, &i) in drape.free.iter().enumerate().step_by(4) {
            for d in 0..3 {
                let fd = crate::testing::central(&y, i, d, 1e-7, &phi);
                let exact = residual[3 * row + d];
                assert!((fd - exact).abs() <= 1e-6 * (1.0 + exact.abs()), "row {row} d{d}: {fd} vs {exact}");
            }
        }
    }

    #[test]
    fn feasible_start_keeps_a_tenth_of_every_collider_distance() {
        let drape = Drape::for_scene(9, TangentKind::Exact, Scene::SphereOnFloor);
        let x: Vec<Vec3> = drape.rest_positions().iter().map(|p| [p[0], p[1], 0.2501]).collect();
        let v = vec![[0.0, 0.0, -2.0]; x.len()];
        let step = DrapeStep { drape: &drape, x_hat: drape.prediction(&x, &v) };
        let start = feasible_start(&step, &x);
        let kept = drape.free.iter().map(|&i| drape.contact.min_distance(start[i]) / drape.contact.min_distance(x[i])).fold(f64::INFINITY, f64::min);
        assert!((kept - 0.1).abs() < 1e-9, "{kept}");
    }

    #[test]
    fn first_step_falls_by_h_squared_g_and_energy_decreases() {
        let drape = Drape::build(9, TangentKind::Projected);
        let trajectory = simulate(&drape, 3, &reference_settings(), Solver { regularized: false, floor_factor: None }).unwrap();
        let centre = drape.mesh.rest.len() / 2;
        let drop = trajectory.positions[1][centre][2];
        assert!((drop + TIME_STEP * TIME_STEP * GRAVITY).abs() < 1e-6, "{drop}");
        let energies: Vec<f64> = (0..4).map(|n| drape.total_energy(&trajectory.positions[n], &trajectory.velocities[n])).collect();
        assert!(energies.windows(2).all(|w| w[1] <= w[0]), "{energies:?}");
    }
}

