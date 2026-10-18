use crate::coupon::{Coupon, LoadPath, COUPON_SIZE, FREE_COMPONENTS};
use crate::e2_checks::{derivative_check, patch_check, path_independence};
use crate::io::{render_number, write_text, BinaryArrays, Json};
use crate::newton::{iterations_to, solve_path, NewtonSettings, StepSolution};
use crate::linalg::Vec3;
use crate::rng::Lcg;
use std::fs;
use std::path::Path;

/// GPU f32 Newton: stop at ‖g‖/‖g₀‖ ≤ 1e-6 per load step.
pub const NEWTON_TOLERANCE: f64 = 1e-6;
/// The f64 oracle runs to its own floor so its solution is the reference for the f32 one.
const REFERENCE_TOLERANCE: f64 = 1e-12;
const MAX_NEWTON_ITERATIONS: usize = 30;
const ARMIJO: f64 = 1e-4;
const MAX_HALVINGS: usize = 30;
/// The E1 solver unchanged: PCG to 1e-7 with gated residual replacement.
const PCG_TOLERANCE: f64 = 1e-7;
const PCG_MAX_ITERATIONS: usize = 20000;
const REPLACE_EVERY: usize = 50;
const REPLACE_ABOVE: f64 = 1e-2;

pub fn reference_settings() -> NewtonSettings {
    NewtonSettings { tolerance: REFERENCE_TOLERANCE, max_iterations: MAX_NEWTON_ITERATIONS, armijo: ARMIJO, max_halvings: MAX_HALVINGS }
}

pub fn run(path: LoadPath, columns: usize, dir: &Path) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let coupon = Coupon::build(path, columns);
    let steps = solve_path(&coupon, &reference_settings())?;
    let problem = problem_arrays(&coupon);
    problem.write(&dir.join("problem.bin"))?;
    write_text(&dir.join("problem.json"), &problem_json(&coupon, &problem.manifest()))?;
    let mut reference = BinaryArrays::new();
    let positions: Vec<f64> = steps.iter().flat_map(|s| s.positions.iter().flatten().copied()).collect();
    reference.push_f64("step_positions", &positions);
    reference.write(&dir.join("reference.bin"))?;
    write_text(&dir.join("reference.json"), &reference_json(&coupon, &steps, &reference.manifest()))?;
    println!("{}", summary(&coupon, &steps));
    Ok(())
}

fn problem_arrays(coupon: &Coupon) -> BinaryArrays {
    let mut arrays = BinaryArrays::new();
    arrays.push_f32("rest_positions", coupon.rest_positions().iter().flatten().map(|&v| v as f32));
    arrays.push_f32("initial_positions", coupon.initial_positions().iter().flatten().map(|&v| v as f32));
    arrays.push_f32("step_gradients", step_gradients(coupon).into_iter().map(|v| v as f32));
    arrays.push_u32("elem_vertices", coupon.elements.iter().flat_map(|e| e.vertices).map(|v| v as u32));
    arrays.push_f32("elem_geom", coupon.elements.iter().flat_map(|e| geometry(e)).map(|v| v as f32));
    arrays.push_u32("row_vertex", coupon.free.iter().map(|&v| v as u32));
    arrays.push_u32("dof_mask", coupon.free.iter().map(|_| dof_mask()));
    push_pattern(&mut arrays, coupon);
    push_contributions(&mut arrays, "block", &block_contributions(coupon));
    push_contributions(&mut arrays, "force", &force_contributions(coupon));
    arrays.push_u32("fixed_vertex", coupon.fixed.iter().map(|&v| v as u32));
    arrays.push_f32("fixed_positions", fixed_positions(coupon).into_iter().map(|v| v as f32));
    arrays.push_f32("kernel_state", kernel_state(coupon).into_iter().map(|v| v as f32));
    arrays
}

fn geometry(element: &crate::membrane::MembraneElement) -> [f64; 7] {
    let g = element.grad_n;
    [g[0][0], g[0][1], g[1][0], g[1][1], g[2][0], g[2][1], element.area]
}

fn dof_mask() -> u32 {
    (0..3).filter(|&d| FREE_COMPONENTS[d]).map(|d| 1u32 << d).sum()
}

fn push_pattern(arrays: &mut BinaryArrays, coupon: &Coupon) {
    let m = &coupon.pattern;
    arrays.push_u32("row_ptr", m.row_ptr.iter().map(|&v| v as u32));
    arrays.push_u32("col_idx", m.col_idx.iter().map(|&v| v as u32));
    arrays.push_u32("diag_idx", m.diagonal_positions().into_iter().map(|v| v as u32));
}

/// For each BSR block, the element Hessian blocks (index 9e + 3k + l) that sum into it, in element order.
fn block_contributions(coupon: &Coupon) -> Vec<Vec<u32>> {
    let mut lists = vec![Vec::new(); coupon.pattern.blocks()];
    for (e, element) in coupon.elements.iter().enumerate() {
        for (k, l) in (0..3).flat_map(|k| (0..3).map(move |l| (k, l))) {
            if let (Some(r), Some(c)) = (coupon.row_of[element.vertices[k]], coupon.row_of[element.vertices[l]]) {
                lists[coupon.pattern.position(r, c)].push((9 * e + 3 * k + l) as u32);
            }
        }
    }
    lists
}

/// For each unknown row, the element gradient entries (index 3e + k) that sum into it, in element order.
fn force_contributions(coupon: &Coupon) -> Vec<Vec<u32>> {
    let mut lists = vec![Vec::new(); coupon.rows()];
    for (e, element) in coupon.elements.iter().enumerate() {
        for (k, &v) in element.vertices.iter().enumerate() {
            if let Some(r) = coupon.row_of[v] {
                lists[r].push((3 * e + k) as u32);
            }
        }
    }
    lists
}

fn push_contributions(arrays: &mut BinaryArrays, name: &str, lists: &[Vec<u32>]) {
    let offsets = std::iter::once(0).chain(lists.iter().scan(0u32, |sum, l| {
        *sum += l.len() as u32;
        Some(*sum)
    }));
    arrays.push_u32(&format!("{name}_contrib_ptr"), offsets);
    arrays.push_u32(&format!("{name}_contrib_idx"), lists.iter().flatten().copied());
}

/// Boundary deformation gradient of every load step, row-major 2×2; the predictor starts from F(0) = I.
fn step_gradients(coupon: &Coupon) -> Vec<f64> {
    coupon.path.load_factors().iter().flat_map(|&t| coupon.path.boundary_gradient(t).into_iter().flatten()).collect()
}

fn fixed_positions(coupon: &Coupon) -> Vec<f64> {
    coupon.path.load_factors().iter().flat_map(|&t| {
        let target = coupon.affine(t);
        coupon.fixed.iter().flat_map(move |&v| target[v]).collect::<Vec<f64>>()
    }).collect()
}

/// A stretched, perturbed state for comparing the WGSL element kernel against the f64 element routines.
pub fn kernel_state(coupon: &Coupon) -> Vec<f64> {
    let spacing = COUPON_SIZE / (coupon.mesh.columns - 1) as f64;
    let mut rng = Lcg::new(31);
    let state = coupon.affine(0.7);
    state.iter().flat_map(|p| [p[0] + 0.1 * spacing * rng.uniform(), p[1] + 0.1 * spacing * rng.uniform(), 0.0]).collect()
}

fn problem_json(coupon: &Coupon, manifest: &str) -> String {
    let m = &coupon.material;
    let material = Json::new().number("wale_N_per_m", m.wale).number("course_N_per_m", m.course)
        .number("coupling_N_per_m", m.coupling).number("shear_N_per_m", m.shear);
    let newton = Json::new().number("tolerance", NEWTON_TOLERANCE).integer("max_iterations", MAX_NEWTON_ITERATIONS)
        .number("armijo", ARMIJO).integer("max_halvings", MAX_HALVINGS);
    let pcg = Json::new().number("tolerance", PCG_TOLERANCE).integer("max_iterations", PCG_MAX_ITERATIONS)
        .integer("replace_every", REPLACE_EVERY).number("replace_above", REPLACE_ABOVE);
    Json::new()
        .text("schema", "e2-problem/1")
        .text("case", &format!("{}-n{}", coupon.path.name(), coupon.mesh.columns))
        .text("path", coupon.path.name())
        .text("description", coupon.path.description())
        .integer("grid_columns", coupon.mesh.columns)
        .number("coupon_size_m", COUPON_SIZE)
        .integer("vertices", coupon.mesh.rest.len())
        .integer("triangles", coupon.elements.len())
        .integer("rows", coupon.rows())
        .integer("unknowns", 3 * coupon.rows())
        .integer("blocks", coupon.pattern.blocks())
        .integer("fixed", coupon.fixed.len())
        .integer("steps", coupon.path.load_factors().len())
        .raw("load_factors", render_list(&coupon.path.load_factors()))
        .raw("material", material.render())
        .raw("newton", newton.render())
        .raw("pcg", pcg.render())
        .raw("arrays", manifest.to_string())
        .render()
}

pub fn render_list(values: &[f64]) -> String {
    format!("[{}]", values.iter().map(|&v| render_number(v)).collect::<Vec<String>>().join(","))
}

fn step_json(coupon: &Coupon, t: f64, step: &StepSolution) -> String {
    let f = coupon.path.boundary_gradient(t);
    let [pxx, pyy] = coupon.grip_stress(&step.positions);
    Json::new()
        .number("load_factor", t)
        .number("strain_x", f[0][0] - 1.0)
        .number("strain_y", f[1][1] - 1.0)
        .number("stress_xx_N_per_m", pxx)
        .number("stress_yy_N_per_m", pyy)
        .text("status", step.status.name())
        .integer("newton_iterations", step.searches.len())
        .number("iterations_to_newton_tolerance", iterations_to(&step.residuals, NEWTON_TOLERANCE).map_or(f64::NAN, |i| i as f64))
        .raw("relative_residuals", render_list(&step.residuals))
        .raw("step_lengths", render_list(&step.searches.iter().map(|s| s.step_length).collect::<Vec<f64>>()))
        .raw("halvings", render_list(&step.searches.iter().map(|s| s.halvings as f64).collect::<Vec<f64>>()))
        .render()
}

fn reference_json(coupon: &Coupon, steps: &[StepSolution], manifest: &str) -> String {
    let factors = coupon.path.load_factors();
    let rendered: Vec<String> = factors.iter().zip(steps).map(|(&t, s)| step_json(coupon, t, s)).collect();
    let checks = match coupon.path {
        LoadPath::Patch => patch_check(coupon, &steps[0].positions).json(),
        _ => path_independence(coupon, &steps.iter().map(|s| s.positions.clone()).collect::<Vec<Vec<Vec3>>>()).json(),
    };
    Json::new()
        .text("schema", "e2-reference/1")
        .text("solver", "f64 Newton, exact band-Cholesky step, Armijo backtracking on the cancellation-free energy change")
        .number("reference_tolerance", REFERENCE_TOLERANCE)
        .raw("derivative_check", derivative_check(coupon).json().render())
        .raw("checks", checks.render())
        .raw("steps", format!("[{}]", rendered.join(",")))
        .raw("arrays", manifest.to_string())
        .render()
}

fn summary(coupon: &Coupon, steps: &[StepSolution]) -> String {
    let newton: Vec<String> = steps.iter().map(|s| format!("{}", s.searches.len())).collect();
    let floor = steps.iter().map(|s| *s.residuals.last().unwrap_or(&f64::NAN)).fold(0.0, f64::max);
    format!("{:<8} n={} rows={} | f64 Newton iterations per step [{}] | worst final residual {:.2e}",
        coupon.path.name(), coupon.mesh.columns, coupon.rows(), newton.join(" "), floor)
}
