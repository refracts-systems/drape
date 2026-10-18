use crate::cases::{GRAVITY, TIME_STEP};
use crate::drape::{simulate, Drape, Scene, Solver, TangentKind, Trajectory, STEPS};
use crate::e2_generate::render_list;
use crate::e3_checks::{bent_state, drape_derivative_check, energy_history};
use crate::e4_rules::RESOLUTION_FACTOR;
use crate::io::{write_text, BinaryArrays, Json};
use crate::newton::{iterations_to, NewtonSettings};
use crate::rng::Lcg;
use std::fs;
use std::path::Path;

/// GPU f32 Newton per time step: ‖∇Φ‖ / ‖∇Φ(x_n)‖ ≤ 1e-6.
pub const NEWTON_TOLERANCE: f64 = 1e-6;
const REFERENCE_TOLERANCE: f64 = 1e-10;
const MAX_NEWTON_ITERATIONS: usize = 30;
/// Oracle-only Newton cap of the E4-final and E5 references (the GPU cap stays 30).
const FINAL_NEWTON_ITERATIONS: usize = 200;
/// An f64 step also converges within this factor of its measured rounding floor (E4-final and E5).
pub const F64_FLOOR_FACTOR: f64 = 2.0;
const ARMIJO: f64 = 1e-4;
const MAX_HALVINGS: usize = 30;
const PCG_TOLERANCE: f64 = 1e-7;
const PCG_MAX_ITERATIONS: usize = 20000;
const REPLACE_EVERY: usize = 50;
const REPLACE_ABOVE: f64 = 1e-2;
/// Probe k is the step from state k−1 to state k of the f64 trajectory.
pub const PROBE_STEPS: [usize; 5] = [1, 10, 60, 120, 240];
/// Evidence-triggered: the exact tangent lost positive definiteness at step 87 of the n33 drape.
pub const TANGENT: TangentKind = TangentKind::Projected;

/// How a drape experiment builds its reference: tangent kind, plain or regularised Newton, and the note recorded with it.
pub struct Scheme {
    pub name: &'static str,
    pub tangent: TangentKind,
    pub solver: Solver,
    pub max_iterations: usize,
    pub resolution_factor: f64,
    pub scene: Scene,
    pub note: &'static str,
}

pub const E3_SCHEME: Scheme = Scheme { name: "e3", tangent: TANGENT, solver: Solver { regularized: false, floor_factor: None }, max_iterations: MAX_NEWTON_ITERATIONS, resolution_factor: 0.0, scene: Scene::PinnedCorners,
    note: "band Cholesky: exact tangent not positive definite at step 87 of drape-n33 (pivot -1.21e-1)" };

/// Experiment 4: exact tangent with adaptive mass regularisation K + μ·M/h² (Levenberg–Marquardt).
pub const E4_SCHEME: Scheme = Scheme { name: "e4", tangent: TangentKind::Exact, solver: Solver { regularized: true, floor_factor: None }, max_iterations: MAX_NEWTON_ITERATIONS, resolution_factor: 0.0, scene: Scene::PinnedCorners,
    note: "exact tangent + adaptive K + mu*M/h^2 (mu 0 per step, x10 from 1e-3 on non-PD or non-descent, /10 after a full step, cap 1e3)" };

/// E4-final: the E4 scheme converging at 1e-10 or within 2x of the measured f64 floor, with an oracle cap of 200 iterations.
pub const E4_FINAL_SCHEME: Scheme = Scheme { name: "e4-final", tangent: TangentKind::Exact, solver: Solver { regularized: true, floor_factor: Some(F64_FLOOR_FACTOR) },
    max_iterations: FINAL_NEWTON_ITERATIONS, resolution_factor: RESOLUTION_FACTOR, scene: Scene::PinnedCorners, note: "E4 scheme; converged at 1e-10 or <= 2 x eps64*||sum|g_c|||/scale; oracle cap 200 iterations" };

pub fn reference_settings() -> NewtonSettings {
    NewtonSettings { tolerance: REFERENCE_TOLERANCE, max_iterations: MAX_NEWTON_ITERATIONS, armijo: ARMIJO, max_halvings: MAX_HALVINGS }
}

pub fn scheme_settings(scheme: &Scheme) -> NewtonSettings {
    NewtonSettings { max_iterations: scheme.max_iterations, ..reference_settings() }
}

pub fn run(columns: usize, with_reference: bool, dir: &Path) -> Result<(), String> {
    generate(columns, with_reference, dir, &E3_SCHEME)
}

pub fn generate(columns: usize, with_reference: bool, dir: &Path, scheme: &Scheme) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let drape = Drape::for_scene(columns, scheme.tangent, scheme.scene);
    let trajectory = if with_reference { Some(simulate(&drape, STEPS, &scheme_settings(scheme), scheme.solver)?) } else { None };
    let probes = trajectory.as_ref().map_or(Vec::new(), |t| probe_steps(&drape, t));
    let problem = problem_arrays(&drape, trajectory.as_ref(), &probes);
    problem.write(&dir.join("problem.bin"))?;
    write_text(&dir.join("problem.json"), &problem_json(&drape, trajectory.as_ref(), scheme, &problem.manifest(), &probes, Json::new()))?;
    if let Some(t) = &trajectory {
        let mut reference = BinaryArrays::new();
        reference.push_f64("positions", &t.positions.iter().flatten().flatten().copied().collect::<Vec<f64>>());
        push_step_records(&mut reference, t);
        reference.write(&dir.join("reference.bin"))?;
        write_text(&dir.join("reference.json"), &reference_json(&drape, t, scheme, &reference.manifest()))?;
        println!("{}", summary(&drape, t));
    }
    Ok(())
}

pub(crate) fn problem_arrays(drape: &Drape, trajectory: Option<&Trajectory>, probes: &[usize]) -> BinaryArrays {
    let mut arrays = BinaryArrays::new();
    arrays.push_f32("rest_positions", drape.rest_positions().iter().flatten().map(|&v| v as f32));
    arrays.push_u32("elem_vertices", drape.membranes.iter().flat_map(|e| e.vertices).map(|v| v as u32));
    arrays.push_f32("elem_geom", drape.membranes.iter().flat_map(|e| geometry(e)).map(|v| v as f32));
    arrays.push_u32("hinge_vertices", drape.hinges.iter().flat_map(|h| h.vertices).map(|v| v as u32));
    arrays.push_f32("hinge_stiffness", drape.hinges.iter().map(|h| h.stiffness as f32));
    arrays.push_u32("row_vertex", drape.free.iter().map(|&v| v as u32));
    arrays.push_u32("dof_mask", drape.free.iter().map(|_| 0b111));
    arrays.push_f32("row_mass", drape.free.iter().map(|&v| drape.mass_over_h2(v) as f32));
    push_pattern(&mut arrays, drape);
    push_contributions(&mut arrays, "block", &block_contributions(drape));
    push_contributions(&mut arrays, "force", &force_contributions(drape));
    arrays.push_u32("fixed_vertex", drape.fixed.iter().map(|&v| v as u32));
    arrays.push_f32("kernel_state", kernel_state(drape).into_iter().map(|v| v as f32));
    push_probes(&mut arrays, trajectory, probes);
    if drape.has_contact() {
        crate::e5_checks::push_contact_arrays(&mut arrays, drape);
    }
    arrays
}

fn geometry(element: &crate::membrane::MembraneElement) -> [f64; 7] {
    let g = element.grad_n;
    [g[0][0], g[0][1], g[1][0], g[1][1], g[2][0], g[2][1], element.area]
}

fn push_pattern(arrays: &mut BinaryArrays, drape: &Drape) {
    let m = &drape.pattern;
    arrays.push_u32("row_ptr", m.row_ptr.iter().map(|&v| v as u32));
    arrays.push_u32("col_idx", m.col_idx.iter().map(|&v| v as u32));
    arrays.push_u32("diag_idx", m.diagonal_positions().into_iter().map(|v| v as u32));
}

/// Local 3×3 blocks in units of 9 floats: mass 9T + 16H + row, then membrane 9e + 3k + l, then hinge 9T + 16h + 4k + l.
fn block_contributions(drape: &Drape) -> Vec<Vec<u32>> {
    let (t, h) = (drape.membranes.len(), drape.hinges.len());
    let mut lists = vec![Vec::new(); drape.pattern.blocks()];
    (0..drape.rows()).for_each(|row| lists[drape.pattern.position(row, row)].push((9 * t + 16 * h + row) as u32));
    for (e, element) in drape.membranes.iter().enumerate() {
        add_pairs(drape, &mut lists, &element.vertices, |k, l| 9 * e + 3 * k + l);
    }
    for (i, hinge) in drape.hinges.iter().enumerate() {
        add_pairs(drape, &mut lists, &hinge.vertices, |k, l| 9 * t + 16 * i + 4 * k + l);
    }
    lists
}

fn add_pairs(drape: &Drape, lists: &mut [Vec<u32>], vertices: &[usize], index: impl Fn(usize, usize) -> usize) {
    for (k, &vk) in vertices.iter().enumerate() {
        for (l, &vl) in vertices.iter().enumerate() {
            if let (Some(r), Some(c)) = (drape.row_of[vk], drape.row_of[vl]) {
                lists[drape.pattern.position(r, c)].push(index(k, l) as u32);
            }
        }
    }
}

/// Local gradients in units of 3 floats: inertia 3T + 4H + row, then membrane 3e + k, then hinge 3T + 4h + k.
fn force_contributions(drape: &Drape) -> Vec<Vec<u32>> {
    let (t, h) = (drape.membranes.len(), drape.hinges.len());
    let mut lists: Vec<Vec<u32>> = (0..drape.rows()).map(|row| vec![(3 * t + 4 * h + row) as u32]).collect();
    let mut add = |vertices: &[usize], index: &dyn Fn(usize) -> usize| {
        vertices.iter().enumerate().filter_map(|(k, &v)| drape.row_of[v].map(|r| (r, k))).for_each(|(r, k)| lists[r].push(index(k) as u32));
    };
    drape.membranes.iter().enumerate().for_each(|(e, element)| add(&element.vertices, &|k| 3 * e + k));
    drape.hinges.iter().enumerate().for_each(|(i, hinge)| add(&hinge.vertices, &|k| 3 * t + 4 * i + k));
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

/// The bent, noisy state of the FD check, for the WGSL-vs-f64 kernel comparison.
pub fn kernel_state(drape: &Drape) -> Vec<f64> {
    bent_state(drape, &mut Lcg::new(97)).into_iter().flatten().collect()
}

/// State k−1 (positions, velocities) of every probe step, for one GPU step from the f64 trajectory.
fn push_probes(arrays: &mut BinaryArrays, trajectory: Option<&Trajectory>, probes: &[usize]) {
    let starts: Vec<usize> = probes.iter().map(|k| k - 1).collect();
    let gather = |pick: &dyn Fn(&Trajectory, usize) -> Vec<f32>| trajectory.map_or(Vec::new(), |t| starts.iter().flat_map(|&s| pick(t, s)).collect::<Vec<f32>>());
    arrays.push_u32("probe_steps", starts.iter().map(|s| (s + 1) as u32));
    arrays.push_f32("probe_positions", gather(&|t, s| t.positions[s].iter().flatten().map(|&v| v as f32).collect()).into_iter());
    arrays.push_f32("probe_velocities", gather(&|t, s| t.velocities[s].iter().flatten().map(|&v| v as f32).collect()).into_iter());
}

/// Per-step f64 Newton records for e3-compare: iterations, final residual, convergence flag, halvings, scale, and every
/// iterate's residual (flattened with offsets).
fn push_step_records(arrays: &mut BinaryArrays, t: &Trajectory) {
    let per_step = |f: &dyn Fn(&crate::drape::TimeStep) -> f64| t.steps.iter().map(f).collect::<Vec<f64>>();
    arrays.push_f64("newton_iterations", &per_step(&|s| s.solution.searches.len() as f64));
    arrays.push_f64("final_residual", &per_step(&|s| *s.solution.residuals.last().unwrap()));
    arrays.push_f64("converged", &per_step(&|s| f64::from(u8::from(s.solution.status.converged()))));
    arrays.push_f64("converged_at_floor", &per_step(&|s| f64::from(u8::from(s.solution.status.name() == "converged_at_floor"))));
    arrays.push_f64("halvings", &per_step(&|s| s.solution.searches.iter().map(|x| x.halvings).sum::<usize>() as f64));
    arrays.push_f64("scale", &per_step(&|s| s.scale));
    let offsets: Vec<f64> = std::iter::once(0.0).chain(t.steps.iter().scan(0.0, |sum, s| { *sum += s.solution.residuals.len() as f64; Some(*sum) })).collect();
    arrays.push_f64("residual_offsets", &offsets);
    arrays.push_f64("residuals", &t.steps.iter().flat_map(|s| s.solution.residuals.clone()).collect::<Vec<f64>>());
    arrays.push_f64("max_mu", &per_step(&|s| s.solution.regularization.max_mu));
    arrays.push_f64("regularization_retries", &per_step(&|s| s.solution.regularization.retries as f64));
    arrays.push_f64("mu_cap_hit", &per_step(&|s| f64::from(u8::from(s.solution.regularization.cap_hit))));
}

pub(crate) fn problem_json(drape: &Drape, trajectory: Option<&Trajectory>, scheme: &Scheme, manifest: &str, probes: &[usize], extra: Json) -> String {
    let m = &drape.material;
    let material = Json::new().number("wale_N_per_m", m.wale).number("course_N_per_m", m.course)
        .number("coupling_N_per_m", m.coupling).number("shear_N_per_m", m.shear);
    let newton = Json::new().number("tolerance", NEWTON_TOLERANCE).integer("max_iterations", MAX_NEWTON_ITERATIONS)
        .number("armijo", ARMIJO).integer("max_halvings", MAX_HALVINGS);
    let pcg = Json::new().number("tolerance", PCG_TOLERANCE).integer("max_iterations", PCG_MAX_ITERATIONS)
        .integer("replace_every", REPLACE_EVERY).number("replace_above", REPLACE_ABOVE);
    Json::new()
        .text("schema", &format!("{}-problem/1", scheme.name))
        .text("case", &format!("drape-n{}", drape.mesh.columns))
        .integer("grid_columns", drape.mesh.columns)
        .integer("vertices", drape.mesh.rest.len())
        .integer("triangles", drape.membranes.len())
        .integer("hinges", drape.hinges.len())
        .integer("rows", drape.rows())
        .integer("unknowns", 3 * drape.rows())
        .integer("blocks", drape.pattern.blocks())
        .integer("steps", STEPS)
        .number("time_step_s", TIME_STEP)
        .number("gravity_m_per_s2", GRAVITY)
        .text("tangent", scheme.tangent.name())
        .flag("regularized", scheme.solver.regularized)
        .text("scene", drape.scene.name())
        .flag("reference", trajectory.is_some())
        .integer("probes", probes.len())
        .raw("material", material.render())
        .raw("newton", newton.render())
        .raw("pcg", pcg.render())
        .raw("robust", crate::e4_rules::settings_json(scheme.resolution_factor).render())
        .raw("contact", crate::e5_checks::contact_json(drape, trajectory).render())
        .extend(extra)
        .raw("arrays", manifest.to_string())
        .render()
}

fn step_json(n: usize, t: &Trajectory, energy: f64) -> String {
    let step = &t.steps[n];
    let s = &step.solution;
    Json::new()
        .integer("step", n + 1)
        .text("status", s.status.name())
        .integer("newton_iterations", s.searches.len())
        .number("iterations_to_newton_tolerance", iterations_to(&s.residuals, NEWTON_TOLERANCE).map_or(f64::NAN, |i| i as f64))
        .number("final_residual", *s.residuals.last().unwrap_or(&f64::NAN))
        .integer("line_search_halvings", s.searches.iter().map(|x| x.halvings).sum())
        .number("scale", step.scale)
        .number("total_energy_J", energy)
        .raw("relative_residuals", render_list(&s.residuals))
        .render()
}

fn reference_json(drape: &Drape, t: &Trajectory, scheme: &Scheme, manifest: &str) -> String {
    let energy = energy_history(drape, &t.positions, &t.velocities);
    let steps: Vec<String> = (0..t.steps.len()).map(|n| step_json(n, t, energy.energies[n + 1])).collect();
    Json::new()
        .text("schema", &format!("{}-reference/1", scheme.name))
        .text("solver", "f64 Newton on the backward-Euler incremental potential, exact band-Cholesky step, Armijo")
        .text("tangent", scheme.tangent.name())
        .flag("regularized", scheme.solver.regularized)
        .text("note", scheme.note)
        .number("reference_tolerance", REFERENCE_TOLERANCE)
        .raw("derivative_check", drape_derivative_check(drape.mesh.columns).json().render())
        .number("energy_scale_J", energy.scale)
        .number("worst_energy_increase_relative", energy.worst_increase)
        .raw("steps", format!("[{}]", steps.join(",")))
        .raw("arrays", manifest.to_string())
        .render()
}

fn summary(drape: &Drape, t: &Trajectory) -> String {
    let unconverged = t.steps.iter().filter(|s| !s.solution.status.converged()).count();
    let iterations: usize = t.steps.iter().map(|s| s.solution.searches.len()).sum();
    let worst = t.steps.iter().map(|s| *s.solution.residuals.last().unwrap()).fold(0.0, f64::max);
    format!("drape n={} rows={} | f64 Newton {} iterations over {} steps, {unconverged} unconverged, worst final {worst:.2e}",
        drape.mesh.columns, drape.rows(), iterations, t.steps.len())
}

/// E3/E4 probe at fixed steps; with contact, at the generator's contact events (e5_checks::event_probes).
fn probe_steps(drape: &Drape, trajectory: &Trajectory) -> Vec<usize> {
    if drape.has_contact() { crate::e5_checks::event_probes(drape, &trajectory.positions) } else { PROBE_STEPS.to_vec() }
}

/// Experiment 5: the E4-final scheme on the free sheet falling onto a sphere resting on a floor (IPC barrier, frictionless).
pub const E5_SCHEME: Scheme = Scheme { name: "e5", tangent: TangentKind::Exact, solver: Solver { regularized: true, floor_factor: Some(F64_FLOOR_FACTOR) },
    max_iterations: FINAL_NEWTON_ITERATIONS, resolution_factor: RESOLUTION_FACTOR, scene: Scene::SphereOnFloor,
    note: "E4-final scheme + vertex-vs-analytic-collider IPC barrier (floor + sphere), alpha <= alpha_max keeping 0.1 of every distance" };

/// Experiment 7: the E5 scene solved to equilibrium by pseudo-transient continuation (the scheme only labels the problem files;
/// the solver is e7_ptc).
pub const E7_SCHEME: Scheme = Scheme { name: "e7", tangent: TangentKind::Exact, solver: Solver { regularized: true, floor_factor: Some(F64_FLOOR_FACTOR) },
    max_iterations: 5, resolution_factor: RESOLUTION_FACTOR, scene: Scene::SphereOnFloor,
    note: "quasi-static drape: pseudo-transient continuation, SER step size, inexact inner Newton (tol 0.1, 5 iterations)" };
