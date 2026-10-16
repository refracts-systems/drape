use crate::cases::TIME_STEP;
use crate::drape::{Drape, DrapeStep};
use crate::e2_generate::render_list;
use crate::e3_checks::{drape_derivative_check, energy_history, velocities_of, DrapeDerivativeCheck};
use crate::e3_generate::{kernel_state, NEWTON_TOLERANCE, TANGENT};
use crate::io::{locate, read_f32_file, read_f64, write_text, Json};
use crate::jsonscan::{flag, number, string};
use crate::linalg::{difference_norm, vector_norm, Vec3};
use crate::newton::NewtonSystem;
use std::fs;
use std::path::Path;

/// WGSL kernels vs f64 at the same f32-rounded state.
const KERNEL_LIMIT: f64 = 1e-5;
const FD_GRADIENT_LIMIT: f64 = 1e-6;
const FD_HESSIAN_LIMIT: f64 = 1e-5;
const HINGE_IDENTITY_LIMIT: f64 = 1e-12;
/// One GPU step from an f64 state vs the f64 next state, relative to ‖x_{n+1} − x_n‖.
const ONE_STEP_LIMIT: f64 = 1e-3;
/// f32 Newton may stop above 1e-6 only at its floor: within 2x of the representation floor.
const FLOOR_FACTOR: f64 = 2.0;
/// Energy increase per step relative to max |E|: f64 rounding allowance, and the f32 bar.
const F64_ENERGY_LIMIT: f64 = 1e-12;
const F32_ENERGY_LIMIT: f64 = 1e-6;
/// An f64 step "stops at its own floor" when it ends within 10x of the 1e-10 tolerance without decreasing over its last 10 iterates.
const F64_FLOOR_BOUND: f64 = 1e-9;
const FLOOR_WINDOW: usize = 10;
const RECORD_FIELDS: usize = 9;

struct Record {
    step: usize,
    residual: f64,
    pcg_iterations: f64,
    linear_residual: f64,
    halvings: f64,
    status: f64,
}

fn read_records(path: &Path) -> Result<Vec<Record>, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let values: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    Ok(values.chunks_exact(RECORD_FIELDS).map(|r| Record {
        step: r[0] as usize, residual: r[2], pcg_iterations: r[3], linear_residual: r[5], halvings: r[7], status: r[8],
    }).collect())
}

fn to_states(values: &[f32], vertices: usize) -> Vec<Vec<Vec3>> {
    values.chunks_exact(3 * vertices).map(|x| x.chunks_exact(3).map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect()).collect()
}

fn round_f32(x: &[Vec3]) -> Vec<Vec3> {
    x.iter().map(|p| p.map(|v| v as f32 as f64)).collect()
}

/// The f64 trajectory written by e3-generate.
struct Reference {
    positions: Vec<Vec<Vec3>>,
    iterations: Vec<f64>,
    final_residual: Vec<f64>,
    converged: Vec<f64>,
    residual_offsets: Vec<f64>,
    residuals: Vec<f64>,
}

impl Reference {
    fn load(dir: &Path, vertices: usize) -> Result<Reference, String> {
        let manifest = fs::read_to_string(dir.join("reference.json")).map_err(|e| format!("reference.json: {e}"))?;
        let bytes = fs::read(dir.join("reference.bin")).map_err(|e| e.to_string())?;
        let array = |name: &str| locate(&manifest, name).map(|(offset, length)| read_f64(&bytes, offset, length));
        let flat = array("positions")?;
        Ok(Reference {
            positions: flat.chunks_exact(3 * vertices).map(|x| x.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect()).collect(),
            iterations: array("newton_iterations")?, final_residual: array("final_residual")?, converged: array("converged")?,
            residual_offsets: array("residual_offsets")?, residuals: array("residuals")?,
        })
    }

    fn residual_history(&self, step: usize) -> &[f64] {
        &self.residuals[self.residual_offsets[step - 1] as usize..self.residual_offsets[step] as usize]
    }

    /// Converged, or stagnated within 10x of the tolerance over its last iterates.
    fn at_floor(&self, step: usize) -> bool {
        let history = self.residual_history(step);
        let tail = &history[history.len().saturating_sub(FLOOR_WINDOW)..];
        let stagnated = tail.iter().copied().fold(f64::INFINITY, f64::min) >= 0.1 * tail[0];
        self.converged[step - 1] == 1.0 || (self.final_residual[step - 1] <= F64_FLOOR_BOUND && stagnated)
    }
}

/// Membrane and hinge gradients and Hessians from WGSL vs f64, exact kernels then projected ones.
fn kernel_errors(drape: &Drape, dir: &Path) -> Result<[f64; 6], String> {
    let gpu = read_f32_file(&dir.join("gpu-kernel.bin"))?;
    let state: Vec<Vec3> = round_f32(&kernel_state(drape).chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect::<Vec<Vec3>>());
    let membrane_grad: Vec<f64> = drape.membranes.iter().flat_map(|e| e.gradient(&state, &drape.material).into_iter().flatten()).collect();
    let hinge_grad: Vec<f64> = drape.hinges.iter().flat_map(|h| h.gradient(&state).into_iter().flatten()).collect();
    let flat3 = |m: [[[[f64; 3]; 3]; 3]; 3]| m.into_iter().flatten().flatten().flatten().collect::<Vec<f64>>();
    let flat4 = |m: [[[[f64; 3]; 3]; 4]; 4]| m.into_iter().flatten().flatten().flatten().collect::<Vec<f64>>();
    let membrane_exact: Vec<f64> = drape.membranes.iter().flat_map(|e| flat3(e.tangent(&state, &drape.material))).collect();
    let membrane_projected: Vec<f64> = drape.membranes.iter().flat_map(|e| flat3(e.hessian(&state, &drape.material))).collect();
    let hinge_exact: Vec<f64> = drape.hinges.iter().flat_map(|h| flat4(h.tangent(&state))).collect();
    let hinge_projected: Vec<f64> = drape.hinges.iter().flat_map(|h| flat4(h.hessian(&state))).collect();
    let sections = [&membrane_grad, &hinge_grad, &membrane_exact, &hinge_exact, &membrane_grad, &hinge_grad, &membrane_projected, &hinge_projected];
    let mut offset = 0;
    let errors: Vec<f64> = sections.iter().map(|expected| {
        let got: Vec<f64> = gpu[offset..offset + expected.len()].iter().map(|&v| v as f64).collect();
        offset += expected.len();
        difference_norm(&got, expected) / vector_norm(expected)
    }).collect();
    Ok([errors[0].max(errors[4]), errors[1].max(errors[5]), errors[2], errors[3], errors[6], errors[7]])
}

/// One probe: GPU step from f64 state k−1 vs f64 state k, its iterates evaluated in f64, and the representation floor.
struct Probe {
    step: usize,
    error: f64,
    f32_residuals: Vec<f64>,
    f64_residuals_of_f32_iterates: Vec<f64>,
    reference_residuals: Vec<f64>,
    representation_floor: f64,
}

fn probe(drape: &Drape, reference: &Reference, step: usize, records: &[&Record], iterates: &[Vec<Vec3>]) -> Probe {
    let (previous, next) = (&reference.positions[step - 1], &reference.positions[step]);
    let start_velocity = if step == 1 { vec![[0.0; 3]; previous.len()] } else { velocity(&reference.positions[step - 2], previous) };
    let gpu_step = DrapeStep { drape, x_hat: drape.prediction(&round_f32(previous), &round_f32(&start_velocity)) };
    let exact_step = DrapeStep { drape, x_hat: drape.prediction(previous, &start_velocity) };
    let gpu_scale = vector_norm(&gpu_step.residual(&round_f32(previous)));
    let exact_scale = vector_norm(&exact_step.residual(previous));
    let last = iterates.last().unwrap();
    Probe {
        step,
        error: difference_norm_states(last, next) / difference_norm_states(next, previous),
        f32_residuals: records.iter().map(|r| r.residual).collect(),
        f64_residuals_of_f32_iterates: iterates.iter().map(|x| vector_norm(&gpu_step.residual(x)) / gpu_scale).collect(),
        reference_residuals: reference.residual_history(step).to_vec(),
        representation_floor: vector_norm(&exact_step.residual(&round_f32(next))) / exact_scale,
    }
}

fn velocity(previous: &[Vec3], current: &[Vec3]) -> Vec<Vec3> {
    current.iter().zip(previous).map(|(p, q)| std::array::from_fn(|d| (p[d] - q[d]) / TIME_STEP)).collect()
}

fn difference_norm_states(a: &[Vec3], b: &[Vec3]) -> f64 {
    a.iter().zip(b).flat_map(|(p, q)| (0..3).map(move |d| (p[d] - q[d]).powi(2))).sum::<f64>().sqrt()
}

impl Probe {
    fn f32_floor(&self) -> f64 {
        self.f32_residuals.iter().copied().fold(f64::INFINITY, f64::min)
    }

    fn newton_ok(&self) -> bool {
        self.f32_floor() <= NEWTON_TOLERANCE || self.f32_floor() <= FLOOR_FACTOR * self.representation_floor
    }

    fn json(&self) -> String {
        Json::new()
            .integer("step", self.step)
            .number("one_step_displacement_error", self.error)
            .number("f32_floor", self.f32_floor())
            .number("representation_floor", self.representation_floor)
            .flag("f32_newton_ok", self.newton_ok())
            .raw("f32_residuals", render_list(&self.f32_residuals))
            .raw("f64_residuals_of_f32_iterates", render_list(&self.f64_residuals_of_f32_iterates))
            .raw("f64_reference_residuals", render_list(&self.reference_residuals))
            .render()
    }
}

fn probes(drape: &Drape, reference: &Reference, dir: &Path) -> Result<Vec<Probe>, String> {
    let records = read_records(&dir.join("gpu-probe-iterations.bin"))?;
    let iterates = to_states(&read_f32_file(&dir.join("gpu-probe-iterates.bin"))?, drape.mesh.rest.len());
    let mut steps: Vec<usize> = records.iter().map(|r| r.step).collect();
    steps.dedup();
    Ok(steps.into_iter().map(|step| {
        let picked: Vec<usize> = (0..records.len()).filter(|&i| records[i].step == step).collect();
        let chosen: Vec<&Record> = picked.iter().map(|&i| &records[i]).collect();
        let states: Vec<Vec<Vec3>> = picked.iter().map(|&i| iterates[i].clone()).collect();
        probe(drape, reference, step, &chosen, &states)
    }).collect())
}

/// Per-step f32 Newton results of the GPU trajectory.
struct StepRecords {
    iterations: Vec<f64>,
    final_residual: Vec<f64>,
    converged: Vec<bool>,
    pcg_per_solve: f64,
    worst_linear_residual: f64,
    halvings: f64,
}

fn step_records(records: &[Record], steps: usize) -> StepRecords {
    let finals: Vec<&Record> = records.iter().filter(|r| r.status != 0.0 && r.pcg_iterations.is_nan() || r.status == 3.0).collect();
    let solves: Vec<&Record> = records.iter().filter(|r| r.pcg_iterations.is_finite()).collect();
    let per_step = |step: usize| records.iter().filter(|r| r.step == step).count() as f64 - 1.0;
    StepRecords {
        iterations: (1..=steps).map(per_step).collect(),
        final_residual: finals.iter().map(|r| r.residual).collect(),
        converged: finals.iter().map(|r| r.status == 1.0).collect(),
        pcg_per_solve: solves.iter().map(|r| r.pcg_iterations).sum::<f64>() / solves.len().max(1) as f64,
        worst_linear_residual: solves.iter().map(|r| r.linear_residual).fold(0.0, f64::max),
        halvings: solves.iter().map(|r| if r.halvings.is_finite() { r.halvings } else { 0.0 }).sum(),
    }
}

struct Run {
    drape: Drape,
    gpu: String,
    trajectory: Vec<Vec<Vec3>>,
    records: StepRecords,
    kernels: [f64; 6],
    derivatives: DrapeDerivativeCheck,
}

fn load_run(problem_dir: &Path, gpu_dir: &Path) -> Result<Run, String> {
    let problem = fs::read_to_string(problem_dir.join("problem.json")).map_err(|e| format!("problem.json: {e}"))?;
    let drape = Drape::build(number(&problem, "", "grid_columns") as usize, TANGENT);
    let gpu = fs::read_to_string(gpu_dir.join("gpu-result.json")).map_err(|e| format!("gpu-result.json: {e}"))?;
    let trajectory = to_states(&read_f32_file(&gpu_dir.join("gpu-trajectory.bin"))?, drape.mesh.rest.len());
    let records = step_records(&read_records(&gpu_dir.join("gpu-iterations.bin"))?, trajectory.len() - 1);
    let kernels = kernel_errors(&drape, gpu_dir)?;
    let derivatives = drape_derivative_check(drape.mesh.columns);
    Ok(Run { drape, gpu, trajectory, records, kernels, derivatives })
}

fn shape_difference(gpu: &[Vec3], reference: &[Vec3]) -> (f64, f64) {
    let squares: Vec<f64> = gpu.iter().zip(reference).map(|(p, q)| (0..3).map(|d| (p[d] - q[d]).powi(2)).sum::<f64>()).collect();
    let max = squares.iter().copied().fold(0.0, f64::max).sqrt();
    let rms = (squares.iter().sum::<f64>() / squares.len() as f64).sqrt();
    let size = crate::cases::SHEET_SIZE;
    (max / size, rms / size)
}

fn reference_json(run: &Run, reference: &Reference, probes: &[Probe]) -> Json {
    let drape = &run.drape;
    let velocities = velocities_of(&reference.positions, TIME_STEP);
    let energy = energy_history(drape, &reference.positions, &velocities);
    let steps = reference.iterations.len();
    let floor_steps = (1..=steps).filter(|&s| reference.at_floor(s)).count();
    let (max_shape, rms_shape) = shape_difference(run.trajectory.last().unwrap(), reference.positions.last().unwrap());
    let rendered: Vec<String> = probes.iter().map(Probe::json).collect();
    Json::new()
        .integer("f64_steps_converged", reference.converged.iter().filter(|&&c| c == 1.0).count())
        .integer("f64_steps_converged_or_at_floor", floor_steps)
        .number("f64_worst_final_residual", reference.final_residual.iter().copied().fold(0.0, f64::max))
        .number("f64_mean_newton_iterations", reference.iterations.iter().sum::<f64>() / steps as f64)
        .number("f64_worst_energy_increase_relative", energy.worst_increase)
        .raw("f64_newton_iterations", render_list(&reference.iterations))
        .raw("f64_final_residuals", render_list(&reference.final_residual))
        .number("final_shape_max_difference_per_sheet_size", max_shape)
        .number("final_shape_rms_difference_per_sheet_size", rms_shape)
        .raw("probes", format!("[{}]", rendered.join(",")))
}

fn verdict(run: &Run, reference: Option<(&Reference, &[Probe])>, f32_energy: f64) -> Json {
    let d = &run.derivatives;
    let base = Json::new()
        .flag("fd_gradient_ok", d.gradient_error <= FD_GRADIENT_LIMIT)
        .flag("fd_hessian_ok", d.hessian_error <= FD_HESSIAN_LIMIT)
        .flag("hinge_energy_change_identity_ok", d.hinge_energy_change_error <= HINGE_IDENTITY_LIMIT)
        .flag("wgsl_kernels_ok", run.kernels.iter().all(|&e| e <= KERNEL_LIMIT))
        .flag("energy_f32_ok", f32_energy <= F32_ENERGY_LIMIT);
    let Some((reference, probes)) = reference else { return base };
    let velocities = velocities_of(&reference.positions, TIME_STEP);
    let f64_energy = energy_history(&run.drape, &reference.positions, &velocities).worst_increase;
    base.flag("f64_converged_or_at_floor", (1..=reference.iterations.len()).all(|s| reference.at_floor(s)))
        .flag("energy_f64_ok", f64_energy <= F64_ENERGY_LIMIT)
        .flag("one_step_ok", probes.iter().all(|p| p.error <= ONE_STEP_LIMIT))
        .flag("f32_newton_ok_at_probes", probes.iter().all(Probe::newton_ok))
}

fn trajectory_json(run: &Run, f32_energy: f64) -> Json {
    let r = &run.records;
    Json::new()
        .integer("f32_steps_converged", r.converged.iter().filter(|&&c| c).count())
        .number("f32_worst_final_residual", r.final_residual.iter().copied().fold(0.0, f64::max))
        .number("f32_mean_newton_iterations", r.iterations.iter().sum::<f64>() / r.iterations.len() as f64)
        .number("pcg_iterations_per_newton_solve", r.pcg_per_solve)
        .number("worst_pcg_true_relative_residual", r.worst_linear_residual)
        .number("line_search_halvings", r.halvings)
        .number("f32_worst_energy_increase_relative", f32_energy)
        .raw("f32_newton_iterations", render_list(&r.iterations))
        .raw("f32_final_residuals", render_list(&r.final_residual))
}

fn result_json(run: &Run, problem_dir: &Path, gpu_dir: &Path) -> Result<String, String> {
    let velocities = velocities_of(&run.trajectory, TIME_STEP);
    let f32_energy = energy_history(&run.drape, &run.trajectory, &velocities).worst_increase;
    let problem = fs::read_to_string(problem_dir.join("problem.json")).map_err(|e| e.to_string())?;
    let reference = if flag(&problem, "", "reference") { Some(Reference::load(problem_dir, run.drape.mesh.rest.len())?) } else { None };
    let probes = match &reference { Some(r) => probes(&run.drape, r, gpu_dir)?, None => Vec::new() };
    let k = run.kernels;
    let json = Json::new()
        .text("schema", "e3-result/1")
        .text("case", &format!("drape-n{}", run.drape.mesh.columns))
        .integer("unknowns", 3 * run.drape.rows())
        .text("tangent", TANGENT.name())
        .text("adapter", &string(&run.gpu, "", "adapter"))
        .text("host_gpus", &string(&run.gpu, "", "host_gpus"))
        .text("host_os", &string(&run.gpu, "", "host_os"))
        .number("ms_per_step", number(&run.gpu, "", "ms_per_step"))
        .raw("kernel_errors", Json::new().number("membrane_gradient", k[0]).number("hinge_gradient", k[1]).number("membrane_exact_tangent", k[2])
            .number("hinge_exact_hessian", k[3]).number("membrane_projected_hessian", k[4]).number("hinge_gauss_newton_hessian", k[5]).render())
        .raw("derivative_check", run.derivatives.json().render())
        .raw("trajectory", trajectory_json(run, f32_energy).render());
    let json = match &reference { Some(r) => json.raw("reference_comparison", reference_json(run, r, &probes).render()), None => json };
    Ok(json.raw("verdict", verdict(run, reference.as_ref().map(|r| (r, probes.as_slice())), f32_energy).render()).render())
}

/// Scores a GPU drape run against the f64 oracle and writes result.json (schema e3-result/1).
pub fn run(dir: &Path) -> Result<(), String> {
    let json = score(dir, dir)?;
    write_text(&dir.join("result.json"), &json)?;
    println!("{json}");
    Ok(())
}


/// The e3-result/1 JSON for a GPU run whose outputs live in `gpu_dir` and whose problem and reference live in `problem_dir`.
pub fn score(problem_dir: &Path, gpu_dir: &Path) -> Result<String, String> {
    let run = load_run(problem_dir, gpu_dir)?;
    result_json(&run, problem_dir, gpu_dir)
}

/// What Experiment 4 needs from each probe: its step, one-step error, f32 floor and representation floor.
pub struct ProbeSummary {
    pub step: usize,
    pub error: f64,
    pub f32_floor: f64,
    pub representation_floor: f64,
}

pub fn probe_summaries(problem_dir: &Path, gpu_dir: &Path) -> Result<Vec<ProbeSummary>, String> {
    let problem = fs::read_to_string(problem_dir.join("problem.json")).map_err(|e| e.to_string())?;
    let drape = Drape::build(number(&problem, "", "grid_columns") as usize, TANGENT);
    let reference = Reference::load(problem_dir, drape.mesh.rest.len())?;
    Ok(probes(&drape, &reference, gpu_dir)?.into_iter()
        .map(|p| ProbeSummary { step: p.step, error: p.error, f32_floor: p.f32_floor(), representation_floor: p.representation_floor })
        .collect())
}
