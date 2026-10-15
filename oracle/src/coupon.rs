use crate::bsr::Bsr;
use crate::linalg::{dot, Mat3, Vec3};
use crate::material::Material;
use crate::membrane::MembraneElement;
use crate::mesh::Mesh;
use crate::rng::Lcg;

pub const COUPON_SIZE: f64 = 0.1;
/// In-plane coupon: every vertex keeps z = 0, so only the first two components are unknowns.
pub const FREE_COMPONENTS: [bool; 3] = [true, true, false];
const LOAD_STEPS: usize = 10;
const PATCH_JITTER: f64 = 0.2;
const PATCH_START_PERTURBATION: f64 = 0.1;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum LoadPath {
    Uniaxial,
    Biaxial,
    Patch,
}

impl LoadPath {
    pub fn parse(name: &str) -> Result<LoadPath, String> {
        match name {
            "uniaxial" => Ok(LoadPath::Uniaxial),
            "biaxial" => Ok(LoadPath::Biaxial),
            "patch" => Ok(LoadPath::Patch),
            other => Err(format!("unknown load path '{other}' (uniaxial | biaxial | patch)")),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            LoadPath::Uniaxial => "uniaxial",
            LoadPath::Biaxial => "biaxial",
            LoadPath::Patch => "patch",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            LoadPath::Uniaxial => "wale-direction coupon gripped on the left and right edges, stretched to 1.20 and unloaded",
            LoadPath::Biaxial => "jaws on the middle half of every edge, stretched to 1.15 (wale) x 1.075 (course) and unloaded",
            LoadPath::Patch => "jittered mesh, affine boundary with shear, perturbed start: the exact solution is the affine map",
        }
    }

    /// Boundary deformation gradient (2×2, rest → deformed) at load factor t ∈ [0, 1].
    pub fn boundary_gradient(&self, t: f64) -> [[f64; 2]; 2] {
        match self {
            LoadPath::Uniaxial => [[1.0 + 0.2 * t, 0.0], [0.0, 1.0]],
            LoadPath::Biaxial => [[1.0 + 0.15 * t, 0.0], [0.0, 1.0 + 0.075 * t]],
            LoadPath::Patch => [[1.0 + 0.1 * t, 0.05 * t], [0.02 * t, 1.0 - 0.05 * t]],
        }
    }

    /// Ten load steps up and ten back down to the rest state; the patch test is a single step.
    pub fn load_factors(&self) -> Vec<f64> {
        match self {
            LoadPath::Patch => vec![1.0],
            _ => (1..=LOAD_STEPS).chain((0..LOAD_STEPS).rev()).map(|k| k as f64 / LOAD_STEPS as f64).collect(),
        }
    }
}

pub struct Coupon {
    pub path: LoadPath,
    pub mesh: Mesh,
    pub material: Material,
    pub elements: Vec<MembraneElement>,
    pub fixed: Vec<usize>,
    pub row_of: Vec<Option<usize>>,
    pub free: Vec<usize>,
    pub pattern: Bsr,
}

impl Coupon {
    pub fn build(path: LoadPath, columns: usize) -> Coupon {
        let mut mesh = Mesh::grid(columns, columns, COUPON_SIZE, COUPON_SIZE);
        if path == LoadPath::Patch {
            jitter_interior(&mut mesh);
        }
        let fixed: Vec<usize> = (0..mesh.rest.len()).filter(|&v| is_fixed(&mesh, path, v)).collect();
        let free: Vec<usize> = (0..mesh.rest.len()).filter(|v| !fixed.contains(v)).collect();
        let mut row_of = vec![None; mesh.rest.len()];
        free.iter().enumerate().for_each(|(row, &v)| row_of[v] = Some(row));
        let elements: Vec<MembraneElement> = mesh.triangles.iter().map(|&t| MembraneElement::new(t, &mesh.rest)).collect();
        let groups: Vec<Vec<usize>> = mesh.triangles.iter().map(|t| t.iter().filter_map(|&v| row_of[v]).collect()).collect();
        let pattern = Bsr::with_pattern(free.len(), &groups);
        Coupon { path, mesh, material: Material::knit_baseline(), elements, fixed, row_of, free, pattern }
    }

    pub fn rows(&self) -> usize {
        self.free.len()
    }

    pub fn rest_positions(&self) -> Vec<Vec3> {
        self.mesh.rest.iter().map(|p| [p[0], p[1], 0.0]).collect()
    }

    /// The boundary map applied to every vertex: the exact solution of the patch test.
    pub fn affine(&self, t: f64) -> Vec<Vec3> {
        let f = self.path.boundary_gradient(t);
        self.mesh.rest.iter().map(|p| [f[0][0] * p[0] + f[0][1] * p[1], f[1][0] * p[0] + f[1][1] * p[1], 0.0]).collect()
    }

    /// State before the first load step: the rest state, with the interior perturbed for the patch test.
    pub fn initial_positions(&self) -> Vec<Vec3> {
        let mut x = self.rest_positions();
        if self.path == LoadPath::Patch {
            let spacing = COUPON_SIZE / (self.mesh.columns - 1) as f64;
            let mut rng = Lcg::new(99);
            self.free.iter().for_each(|&v| (0..2).for_each(|d| x[v][d] += PATCH_START_PERTURBATION * spacing * rng.uniform()));
        }
        x
    }

    /// Load-step predictor: adds the affine increment (F(to) − F(from))·X to every vertex, grips exact.
    pub fn predict(&self, x: &mut [Vec3], from: f64, to: f64) {
        let (a, b) = (self.path.boundary_gradient(from), self.path.boundary_gradient(to));
        for (p, rest) in x.iter_mut().zip(&self.mesh.rest) {
            (0..2).for_each(|r| p[r] += (b[r][0] - a[r][0]) * rest[0] + (b[r][1] - a[r][1]) * rest[1]);
        }
        self.apply_boundary(x, to);
    }

    /// Out-of-balance force created by moving only the grips: the scale of a load step's nonlinear residual.
    pub fn step_scale(&self, previous: &[Vec3], t: f64) -> f64 {
        let mut gripped = previous.to_vec();
        self.apply_boundary(&mut gripped, t);
        crate::linalg::vector_norm(&self.free_gradient(&gripped))
    }

    pub fn apply_boundary(&self, x: &mut [Vec3], t: f64) {
        let target = self.affine(t);
        self.fixed.iter().for_each(|&v| x[v] = target[v]);
    }

    pub fn energy(&self, x: &[Vec3]) -> f64 {
        self.elements.iter().map(|e| e.energy(x, &self.material)).sum()
    }

    /// Change of the total energy for x → x + αd, summed from the cancellation-free element formula.
    pub fn energy_change(&self, x: &[Vec3], d: &[Vec3], alpha: f64) -> f64 {
        self.elements.iter().map(|e| e.energy_change(x, d, alpha, &self.material)).sum()
    }

    /// ∂Π/∂x for every vertex, fixed ones included (their entries are the grip reactions).
    pub fn vertex_gradient(&self, x: &[Vec3]) -> Vec<Vec3> {
        let mut gradient = vec![[0.0; 3]; x.len()];
        for element in &self.elements {
            let g = element.gradient(x, &self.material);
            for (k, &v) in element.vertices.iter().enumerate() {
                (0..3).for_each(|d| gradient[v][d] += g[k][d]);
            }
        }
        gradient
    }

    /// Nonlinear residual on the unknowns: ∂Π/∂x of free vertices with constrained components zeroed.
    pub fn free_gradient(&self, x: &[Vec3]) -> Vec<f64> {
        let gradient = self.vertex_gradient(x);
        self.free.iter().flat_map(|&v| mask_vector(gradient[v])).collect()
    }

    /// Exact tangent on the unknowns; constrained components become identity rows and columns.
    pub fn tangent(&self, x: &[Vec3]) -> Bsr {
        let mut matrix = Bsr { row_ptr: self.pattern.row_ptr.clone(), col_idx: self.pattern.col_idx.clone(), vals: self.pattern.vals.clone() };
        for element in &self.elements {
            self.add_element(&mut matrix, element, &element.tangent(x, &self.material));
        }
        mask_matrix(&mut matrix);
        matrix
    }

    fn add_element(&self, matrix: &mut Bsr, element: &MembraneElement, hessian: &[[Mat3; 3]; 3]) {
        for (k, &vk) in element.vertices.iter().enumerate() {
            for (l, &vl) in element.vertices.iter().enumerate() {
                if let (Some(r), Some(c)) = (self.row_of[vk], self.row_of[vl]) {
                    matrix.add(r, c, &hessian[k][l]);
                }
            }
        }
    }

    /// Spreads an unknown vector (3 per row) onto all vertices; fixed vertices get zero.
    pub fn scatter(&self, rows: &[f64]) -> Vec<Vec3> {
        let mut vertices = vec![[0.0; 3]; self.mesh.rest.len()];
        self.free.iter().enumerate().for_each(|(row, &v)| vertices[v] = [rows[3 * row], rows[3 * row + 1], rows[3 * row + 2]]);
        vertices
    }

    /// Nominal (first Piola) grip stresses in N/m: right-edge x reaction per rest height, top-edge y reaction per rest width.
    pub fn grip_stress(&self, x: &[Vec3]) -> [f64; 2] {
        let gradient = self.vertex_gradient(x);
        let columns = self.mesh.columns;
        let right: f64 = (0..x.len()).filter(|v| v % columns == columns - 1).map(|v| gradient[v][0]).sum();
        let top: f64 = (0..x.len()).filter(|v| v / columns == self.mesh.rows - 1).map(|v| gradient[v][1]).sum();
        [right / self.mesh.height, top / self.mesh.width]
    }
}

fn is_fixed(mesh: &Mesh, path: LoadPath, vertex: usize) -> bool {
    let (column, row) = (vertex % mesh.columns, vertex / mesh.columns);
    match path {
        LoadPath::Uniaxial => column == 0 || column == mesh.columns - 1,
        LoadPath::Biaxial => on_side_jaw(mesh, column, row),
        LoadPath::Patch => mesh.is_boundary(vertex),
    }
}

/// A vertex on an edge lies under a jaw when its position along that edge is in the middle half.
fn on_side_jaw(mesh: &Mesh, column: usize, row: usize) -> bool {
    let middle = |i: usize, n: usize| 4 * i >= n - 1 && 4 * i <= 3 * (n - 1);
    let vertical_edge = column == 0 || column == mesh.columns - 1;
    let horizontal_edge = row == 0 || row == mesh.rows - 1;
    (vertical_edge && middle(row, mesh.rows)) || (horizontal_edge && middle(column, mesh.columns))
}

fn jitter_interior(mesh: &mut Mesh) {
    let spacing = mesh.width / (mesh.columns - 1) as f64;
    let mut rng = Lcg::new(2024);
    for v in 0..mesh.rest.len() {
        if !mesh.is_boundary(v) {
            mesh.rest[v][0] += PATCH_JITTER * spacing * rng.uniform();
            mesh.rest[v][1] += PATCH_JITTER * spacing * rng.uniform();
        }
    }
}

pub fn mask_vector(v: Vec3) -> [f64; 3] {
    std::array::from_fn(|d| if FREE_COMPONENTS[d] { v[d] } else { 0.0 })
}

fn mask_matrix(matrix: &mut Bsr) {
    for row in 0..matrix.rows() {
        for k in matrix.row_ptr[row]..matrix.row_ptr[row + 1] {
            let diagonal = matrix.col_idx[k] == row;
            for (i, j) in (0..3).flat_map(|i| (0..3).map(move |j| (i, j))) {
                if !(FREE_COMPONENTS[i] && FREE_COMPONENTS[j]) {
                    matrix.vals[9 * k + 3 * i + j] = if diagonal && i == j { 1.0 } else { 0.0 };
                }
            }
        }
    }
}

pub fn free_dot(a: &[f64], b: &[f64]) -> f64 {
    a.chunks(3).zip(b.chunks(3)).map(|(p, q)| dot([p[0], p[1], p[2]], [q[0], q[1], q[2]])).sum()
}
