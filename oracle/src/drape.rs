use crate::bsr::Bsr;
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

/// A 0.5 m knit sheet, flat in xy, pinned at its two top corners, falling under gravity (−z).
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
}

impl Drape {
    pub fn build(columns: usize, tangent_kind: TangentKind) -> Drape {
        let mesh = Mesh::grid(columns, columns, SHEET_SIZE, SHEET_SIZE);
        let material = Material::knit_baseline();
        let fixed = vec![(mesh.rows - 1) * mesh.columns, mesh.rows * mesh.columns - 1];
        let free: Vec<usize> = (0..mesh.rest.len()).filter(|v| !fixed.contains(v)).collect();
        let mut row_of = vec![None; mesh.rest.len()];
        free.iter().enumerate().for_each(|(row, &v)| row_of[v] = Some(row));
        let membranes: Vec<MembraneElement> = mesh.triangles.iter().map(|&t| MembraneElement::new(t, &mesh.rest)).collect();
        let hinges = hinge_elements(&mesh, &material);
        let to_rows = |vs: &[usize]| vs.iter().filter_map(|&v| row_of[v]).collect::<Vec<usize>>();
        let groups: Vec<Vec<usize>> = mesh.triangles.iter().map(|t| to_rows(t)).chain(mesh.hinges.iter().map(|h| to_rows(h))).collect();
        let pattern = Bsr::with_pattern(free.len(), &groups);
        let mass = mesh.vertex_areas().iter().map(|a| material.density * a).collect();
        Drape { mesh, material, membranes, hinges, fixed, row_of, free, pattern, mass, tangent_kind }
    }

    pub fn rows(&self) -> usize {
        self.free.len()
    }

    pub fn rest_positions(&self) -> Vec<Vec3> {
        self.mesh.rest.iter().map(|p| [p[0], p[1], 0.0]).collect()
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
        membrane + self.hinges.iter().map(|h| h.energy(x)).sum::<f64>()
    }

    /// ∂(Π_membrane + Π_hinge)/∂x for every vertex.
    pub fn potential_gradient(&self, x: &[Vec3]) -> Vec<Vec3> {
        let mut gradient = vec![[0.0; 3]; x.len()];
        for element in &self.membranes {
            scatter_add(&mut gradient, &element.vertices, &element.gradient(x, &self.material));
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
        inertia + membrane + drape.hinges.iter().map(|h| h.energy_change(x, d, alpha)).sum::<f64>()
    }

    fn scatter(&self, rows: &[f64]) -> Vec<Vec3> {
        let mut vertices = vec![[0.0; 3]; self.drape.mesh.rest.len()];
        self.drape.free.iter().enumerate().for_each(|(row, &v)| vertices[v] = [rows[3 * row], rows[3 * row + 1], rows[3 * row + 2]]);
        vertices
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
}

/// One time step from (x, v): Newton on Φ starting at x̂, residual scaled by ‖∇Φ(x)‖.
pub struct TimeStep {
    pub solution: StepSolution,
    pub scale: f64,
    pub velocity: Vec<Vec3>,
}

pub fn time_step(drape: &Drape, x: &[Vec3], v: &[Vec3], settings: &NewtonSettings, regularized: bool) -> Result<TimeStep, String> {
    let step = DrapeStep { drape, x_hat: drape.prediction(x, v) };
    let scale = vector_norm(&step.residual(x));
    let solution = if regularized { solve_step_regularized(&step, step.x_hat.clone(), scale, settings)? } else { solve_step(&step, step.x_hat.clone(), scale, settings)? };
    let velocity = solution.positions.iter().zip(x).map(|(p, q)| std::array::from_fn(|d| (p[d] - q[d]) / TIME_STEP)).collect();
    Ok(TimeStep { solution, scale, velocity })
}

/// States 0..=steps (positions and velocities) and every step's Newton record.
pub struct Trajectory {
    pub positions: Vec<Vec<Vec3>>,
    pub velocities: Vec<Vec<Vec3>>,
    pub steps: Vec<TimeStep>,
}

pub fn simulate(drape: &Drape, steps: usize, settings: &NewtonSettings, regularized: bool) -> Result<Trajectory, String> {
    let rest = drape.rest_positions();
    let mut trajectory = Trajectory { velocities: vec![vec![[0.0; 3]; rest.len()]], positions: vec![rest], steps: Vec::new() };
    for _ in 0..steps {
        let (x, v) = (trajectory.positions.last().unwrap(), trajectory.velocities.last().unwrap());
        let step = time_step(drape, x, v, settings, regularized)?;
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
    fn first_step_falls_by_h_squared_g_and_energy_decreases() {
        let drape = Drape::build(9, TangentKind::Projected);
        let trajectory = simulate(&drape, 3, &reference_settings(), false).unwrap();
        let centre = drape.mesh.rest.len() / 2;
        let drop = trajectory.positions[1][centre][2];
        assert!((drop + TIME_STEP * TIME_STEP * GRAVITY).abs() < 1e-6, "{drop}");
        let energies: Vec<f64> = (0..4).map(|n| drape.total_energy(&trajectory.positions[n], &trajectory.velocities[n])).collect();
        assert!(energies.windows(2).all(|w| w[1] <= w[0]), "{energies:?}");
    }
}

