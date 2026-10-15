use crate::bsr::Bsr;
use crate::hinge::{hinge_stiffness, HingeElement};
use crate::linalg::{identity_scaled, Vec3};
use crate::material::Material;
use crate::membrane::MembraneElement;
use crate::mesh::Mesh;
use std::f64::consts::PI;

pub const SHEET_SIZE: f64 = 0.5;
pub const TIME_STEP: f64 = 1.0 / 240.0;
pub const GRAVITY: f64 = 9.81;
pub const IN_PLANE_LOAD: f64 = 2.0;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CaseKind {
    Tension,
    Slack,
    Dynamic,
}

impl CaseKind {
    pub fn parse(name: &str) -> Result<CaseKind, String> {
        match name {
            "tension" => Ok(CaseKind::Tension),
            "slack" => Ok(CaseKind::Slack),
            "dynamic" => Ok(CaseKind::Dynamic),
            other => Err(format!("unknown case '{other}' (tension | slack | dynamic)")),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            CaseKind::Tension => "tension",
            CaseKind::Slack => "slack",
            CaseKind::Dynamic => "dynamic",
        }
    }

    /// (wale stretch, course stretch) of the uniform state the Hessian is taken at.
    pub fn stretch(&self) -> (f64, f64) {
        match self {
            CaseKind::Tension => (1.03, 1.08),
            _ => (1.0, 1.0),
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            CaseKind::Tension => "quasi-static Newton matrix of a biaxially prestretched clamped sheet (membrane + geometric + hinge)",
            CaseKind::Slack => "quasi-static Newton matrix of an unstressed clamped sheet: out-of-plane stiffness from bending only",
            CaseKind::Dynamic => "slack sheet plus backward-Euler inertia M/h^2 with h = 1/240 s",
        }
    }
}

pub struct Problem {
    pub kind: CaseKind,
    pub mesh: Mesh,
    pub matrix: Bsr,
    pub rhs: Vec<f64>,
    pub material: Material,
}

pub fn build(kind: CaseKind, columns: usize) -> Problem {
    let mesh = Mesh::grid(columns, columns, SHEET_SIZE, SHEET_SIZE);
    let material = Material::knit_baseline();
    let (row_of, free_vertices) = free_rows(&mesh);
    let positions = deformed_positions(&mesh, kind.stretch());
    let membranes: Vec<MembraneElement> = mesh.triangles.iter().map(|&t| MembraneElement::new(t, &mesh.rest)).collect();
    let hinges = hinge_elements(&mesh, &material);
    let mut matrix = Bsr::with_pattern(free_vertices.len(), &coupling_groups(&mesh, &row_of));
    assemble_membranes(&mut matrix, &membranes, &positions, &material, &row_of);
    assemble_hinges(&mut matrix, &hinges, &positions, &row_of);
    if kind == CaseKind::Dynamic {
        add_inertia(&mut matrix, &mesh, &material, &free_vertices);
    }
    let rhs = load_vector(&mesh, &material, &free_vertices);
    Problem { kind, mesh, matrix, rhs, material }
}

fn free_rows(mesh: &Mesh) -> (Vec<Option<usize>>, Vec<usize>) {
    let free: Vec<usize> = (0..mesh.rest.len()).filter(|&v| !mesh.is_boundary(v)).collect();
    let mut row_of = vec![None; mesh.rest.len()];
    for (row, &v) in free.iter().enumerate() {
        row_of[v] = Some(row);
    }
    (row_of, free)
}

fn deformed_positions(mesh: &Mesh, (wale, course): (f64, f64)) -> Vec<Vec3> {
    mesh.rest.iter().map(|p| [wale * p[0], course * p[1], 0.0]).collect()
}

fn hinge_elements(mesh: &Mesh, material: &Material) -> Vec<HingeElement> {
    mesh.hinges
        .iter()
        .map(|&vertices| {
            let (a, b) = (mesh.rest[vertices[2]], mesh.rest[vertices[3]]);
            let length = (b[0] - a[0]).hypot(b[1] - a[1]);
            let direction = [(b[0] - a[0]) / length, (b[1] - a[1]) / length];
            let stiffness = hinge_stiffness(&mesh.rest, vertices, material.hinge_bending(direction));
            HingeElement { vertices, stiffness }
        })
        .collect()
}

fn coupling_groups(mesh: &Mesh, row_of: &[Option<usize>]) -> Vec<Vec<usize>> {
    let to_rows = |vs: &[usize]| vs.iter().filter_map(|&v| row_of[v]).collect::<Vec<usize>>();
    let triangles = mesh.triangles.iter().map(|t| to_rows(t));
    let hinges = mesh.hinges.iter().map(|h| to_rows(h));
    triangles.chain(hinges).collect()
}

fn assemble_membranes(m: &mut Bsr, elements: &[MembraneElement], x: &[Vec3], material: &Material, row_of: &[Option<usize>]) {
    for element in elements {
        let hessian = element.hessian(x, material);
        for (k, &vk) in element.vertices.iter().enumerate() {
            for (l, &vl) in element.vertices.iter().enumerate() {
                if let (Some(r), Some(c)) = (row_of[vk], row_of[vl]) {
                    m.add(r, c, &hessian[k][l]);
                }
            }
        }
    }
}

fn assemble_hinges(m: &mut Bsr, hinges: &[HingeElement], x: &[Vec3], row_of: &[Option<usize>]) {
    for hinge in hinges {
        let hessian = hinge.hessian(x);
        for (k, &vk) in hinge.vertices.iter().enumerate() {
            for (l, &vl) in hinge.vertices.iter().enumerate() {
                if let (Some(r), Some(c)) = (row_of[vk], row_of[vl]) {
                    m.add(r, c, &hessian[k][l]);
                }
            }
        }
    }
}

fn add_inertia(m: &mut Bsr, mesh: &Mesh, material: &Material, free: &[usize]) {
    let areas = mesh.vertex_areas();
    for (row, &v) in free.iter().enumerate() {
        let mass = material.density * areas[v];
        m.add(row, row, &identity_scaled(mass / (TIME_STEP * TIME_STEP)));
    }
}

/// Newton right-hand side at the uniform state: gravity out of plane plus a smooth in-plane load.
fn load_vector(mesh: &Mesh, material: &Material, free: &[usize]) -> Vec<f64> {
    let areas = mesh.vertex_areas();
    let (w, h) = (mesh.width, mesh.height);
    free.iter()
        .flat_map(|&v| {
            let [px, py] = mesh.rest[v];
            let shape = (PI * px / w).sin() * (PI * py / h).sin();
            let skew = (2.0 * PI * px / w).sin() * (PI * py / h).sin();
            let area = areas[v];
            [area * IN_PLANE_LOAD * shape, 0.5 * area * IN_PLANE_LOAD * skew, -material.density * area * GRAVITY]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembled_matrices_are_symmetric() {
        for kind in [CaseKind::Tension, CaseKind::Slack, CaseKind::Dynamic] {
            let problem = build(kind, 9);
            assert!(problem.matrix.asymmetry() < 1e-12, "{kind:?}");
        }
    }

    #[test]
    fn uniform_prestretch_has_no_interior_force() {
        let mesh = Mesh::grid(7, 7, SHEET_SIZE, SHEET_SIZE);
        let material = Material::knit_baseline();
        let x = deformed_positions(&mesh, CaseKind::Tension.stretch());
        let mut force = vec![[0.0f64; 3]; mesh.rest.len()];
        for &t in &mesh.triangles {
            let g = MembraneElement::new(t, &mesh.rest).gradient(&x, &material);
            for k in 0..3 {
                for d in 0..3 {
                    force[t[k]][d] += g[k][d];
                }
            }
        }
        let interior = (0..mesh.rest.len()).filter(|&v| !mesh.is_boundary(v));
        assert!(interior.flat_map(|v| force[v]).all(|f| f.abs() < 1e-12));
    }

    #[test]
    fn material_is_convex() {
        assert!(Material::knit_baseline().is_convex());
    }
}
