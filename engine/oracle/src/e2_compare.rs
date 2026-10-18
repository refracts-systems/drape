use crate::coupon::{Coupon, LoadPath};
use crate::e2_checks::{derivative_check, patch_check, path_independence, DerivativeCheck};
use crate::e2_generate::{kernel_state, reference_settings, render_list, NEWTON_TOLERANCE};
use crate::io::{read_f32_file, write_text, Json};
use crate::jsonscan::{number, string};
use crate::linalg::{difference_norm, vector_norm, Vec3};
use crate::newton::{iterations_to, solve_path, StepSolution};
use std::fs;
use std::path::Path;

/// WGSL element gradient/tangent vs the f64 element routines at the same f32 input (≈ 100 ε_f32).
pub const KERNEL_LIMIT: f64 = 1e-5;
/// f32 solution vs f64 solution, relative to the peak displacement / peak grip stress: E1's storage-error bar.
pub const SOLUTION_LIMIT: f64 = 1e-3;
/// While both are above the f32 floor, f32 Newton residuals must stay within this factor of f64 Newton's.
pub const TRACKING_FACTOR: f64 = 2.0;
const FD_GRADIENT_LIMIT: f64 = 1e-6;
const FD_TANGENT_LIMIT: f64 = 1e-5;
const F64_EXACT_LIMIT: f64 = 1e-12;
/// Element strains amplify the leftover position error of a 1e-12 Newton stop by 1/(element size · strain).
const F64_PATCH_STRAIN_LIMIT: f64 = 1e-10;
const F64_PATH_LIMIT: f64 = 1e-9;
const RECORD_FIELDS: usize = 9;

/// One GPU Newton iterate as written by run-e2.ts (gpu-iterations.bin row).
struct Record {
    step: usize,
    residual: f64,
    pcg_iterations: f64,
    linear_residual: f64,
    step_length: f64,
    halvings: f64,
    status: f64,
}

fn read_records(dir: &Path) -> Result<Vec<Record>, String> {
    let bytes = fs::read(dir.join("gpu-iterations.bin")).map_err(|e| format!("gpu-iterations.bin: {e}"))?;
    let values: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    Ok(values.chunks_exact(RECORD_FIELDS).map(|r| Record {
        step: r[0] as usize, residual: r[2], pcg_iterations: r[3], linear_residual: r[5], step_length: r[6], halvings: r[7], status: r[8],
    }).collect())
}

fn read_iterates(dir: &Path, vertices: usize) -> Result<Vec<Vec<Vec3>>, String> {
    let values = read_f32_file(&dir.join("gpu-iterates.bin"))?;
    Ok(values.chunks_exact(3 * vertices).map(|x| x.chunks_exact(3).map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect()).collect())
}

fn round_f32(x: &[Vec3]) -> Vec<Vec3> {
    x.iter().map(|p| p.map(|v| v as f32 as f64)).collect()
}

/// Everything the oracle knows about one coupon run: f64 reference path and the GPU's iterates.
struct Run {
    coupon: Coupon,
    reference: Vec<StepSolution>,
    records: Vec<Record>,
    iterates: Vec<Vec<Vec3>>,
}

impl Run {
    fn load(dir: &Path) -> Result<Run, String> {
        let problem = fs::read_to_string(dir.join("problem.json")).map_err(|e| format!("problem.json: {e}"))?;
        let coupon = Coupon::build(LoadPath::parse(&string(&problem, "", "path"))?, number(&problem, "", "grid_columns") as usize);
        let reference = solve_path(&coupon, &reference_settings())?;
        let iterates = read_iterates(dir, coupon.mesh.rest.len())?;
        Ok(Run { coupon, reference, records: read_records(dir)?, iterates })
    }

    fn step_indices(&self, step: usize) -> Vec<usize> {
        (0..self.records.len()).filter(|&i| self.records[i].step == step).collect()
    }

    /// Final GPU positions of every step.
    fn finals(&self) -> Vec<Vec<Vec3>> {
        (0..self.reference.len()).map(|s| self.iterates[*self.step_indices(s).last().unwrap()].clone()).collect()
    }
}

/// Per-step f64 view of the f32 path: residuals of the GPU iterates in f64, the f32 representation floor, errors.
struct StepScore {
    f32_residuals: Vec<f64>,
    f64_residuals_of_f32_iterates: Vec<f64>,
    reference_residuals: Vec<f64>,
    representation_floor: f64,
    displacement_error: f64,
    stress_error: f64,
    tracking: f64,
    status: f64,
    pcg_iterations: Vec<f64>,
    linear_residuals: Vec<f64>,
    step_lengths: Vec<f64>,
    halvings: f64,
}

fn score_step(run: &Run, step: usize, previous_gpu: &[Vec3], previous_reference: &[Vec3], peaks: (f64, f64)) -> StepScore {
    let (coupon, t) = (&run.coupon, run.coupon.path.load_factors()[step]);
    let indices = run.step_indices(step);
    let records: Vec<&Record> = indices.iter().map(|&i| &run.records[i]).collect();
    let (gpu_scale, reference_scale) = (coupon.step_scale(previous_gpu, t), coupon.step_scale(previous_reference, t));
    let relative = |x: &[Vec3], scale: f64| vector_norm(&coupon.free_gradient(x)) / scale;
    let reference = &run.reference[step];
    let last = &run.iterates[*indices.last().unwrap()];
    let [p, q] = [coupon.grip_stress(last), coupon.grip_stress(&reference.positions)];
    let taken: Vec<&&Record> = records.iter().filter(|r| r.step_length.is_finite()).collect();
    StepScore {
        f32_residuals: records.iter().map(|r| r.residual).collect(),
        f64_residuals_of_f32_iterates: indices.iter().map(|&i| relative(&run.iterates[i], gpu_scale)).collect(),
        reference_residuals: reference.residuals.clone(),
        representation_floor: relative(&round_f32(&reference.positions), reference_scale),
        displacement_error: max_abs_difference(last, &reference.positions) / peaks.0,
        stress_error: (p[0] - q[0]).abs().max((p[1] - q[1]).abs()) / peaks.1,
        tracking: tracking(&records.iter().map(|r| r.residual).collect::<Vec<f64>>(), &reference.residuals),
        status: records.last().map_or(f64::NAN, |r| r.status),
        pcg_iterations: taken.iter().map(|r| r.pcg_iterations).collect(),
        linear_residuals: taken.iter().map(|r| r.linear_residual).collect(),
        step_lengths: taken.iter().map(|r| r.step_length).collect(),
        halvings: taken.iter().map(|r| r.halvings).sum(),
    }
}

fn max_abs_difference(a: &[Vec3], b: &[Vec3]) -> f64 {
    a.iter().zip(b).flat_map(|(p, q)| (0..3).map(move |d| (p[d] - q[d]).abs())).fold(0.0, f64::max)
}

/// Largest ratio between f32 and f64 Newton residuals at equal iteration count, while f64 is above the f32 floor
/// (taken as the smallest f32 residual of the step times 10).
fn tracking(f32_residuals: &[f64], f64_residuals: &[f64]) -> f64 {
    let floor = 10.0 * f32_residuals.iter().copied().fold(f64::INFINITY, f64::min);
    f32_residuals.iter().zip(f64_residuals)
        .filter(|(_, &r64)| r64 > floor)
        .map(|(&r32, &r64)| (r32 / r64).max(r64 / r32))
        .fold(1.0, f64::max)
}

fn peaks(run: &Run) -> (f64, f64) {
    let rest = run.coupon.rest_positions();
    let displacement = run.reference.iter().map(|s| max_abs_difference(&s.positions, &rest)).fold(0.0, f64::max);
    let stress = run.reference.iter().flat_map(|s| run.coupon.grip_stress(&s.positions).map(f64::abs)).fold(0.0, f64::max);
    (displacement, stress)
}

fn score_steps(run: &Run) -> Vec<StepScore> {
    let (finals, peak) = (run.finals(), peaks(run));
    let initial = run.coupon.initial_positions();
    (0..run.reference.len()).map(|s| {
        let previous_gpu = if s == 0 { round_f32(&initial) } else { finals[s - 1].clone() };
        let previous_reference = if s == 0 { initial.clone() } else { run.reference[s - 1].positions.clone() };
        score_step(run, s, &previous_gpu, &previous_reference, peak)
    }).collect()
}

/// WGSL element gradient and tangent at the kernel-check state vs f64 at the same f32-rounded positions.
fn kernel_errors(run: &Run, dir: &Path) -> Result<(f64, f64), String> {
    let coupon = &run.coupon;
    let gpu = read_f32_file(&dir.join("gpu-kernel.bin"))?;
    let state: Vec<Vec3> = kernel_state(coupon).chunks_exact(3).map(|p| [p[0] as f32 as f64, p[1] as f32 as f64, p[2] as f32 as f64]).collect();
    let gradient: Vec<f64> = coupon.elements.iter().flat_map(|e| e.gradient(&state, &coupon.material).into_iter().flatten()).collect();
    let tangent: Vec<f64> = coupon.elements.iter().flat_map(|e| e.tangent(&state, &coupon.material).into_iter().flatten().flatten().flatten()).collect();
    let (g32, h32): (Vec<f64>, Vec<f64>) = (gpu[..gradient.len()].iter().map(|&v| v as f64).collect(), gpu[gradient.len()..].iter().map(|&v| v as f64).collect());
    Ok((difference_norm(&g32, &gradient) / vector_norm(&gradient), difference_norm(&h32, &tangent) / vector_norm(&tangent)))
}

impl StepScore {
    fn json(&self, t: f64) -> String {
        Json::new()
            .number("load_factor", t)
            .number("status", self.status)
            .number("f32_iterations_to_newton_tolerance", iterations_to(&self.f32_residuals, NEWTON_TOLERANCE).map_or(f64::NAN, |i| i as f64))
            .number("f64_iterations_to_newton_tolerance", iterations_to(&self.reference_residuals, NEWTON_TOLERANCE).map_or(f64::NAN, |i| i as f64))
            .number("f32_floor", self.f32_residuals.iter().copied().fold(f64::INFINITY, f64::min))
            .number("f64_floor_of_f32_iterates", self.f64_residuals_of_f32_iterates.iter().copied().fold(f64::INFINITY, f64::min))
            .number("representation_floor", self.representation_floor)
            .number("displacement_error", self.displacement_error)
            .number("stress_error", self.stress_error)
            .number("tracking_ratio", self.tracking)
            .number("line_search_halvings", self.halvings)
            .raw("f32_residuals", render_list(&self.f32_residuals))
            .raw("f64_residuals_of_f32_iterates", render_list(&self.f64_residuals_of_f32_iterates))
            .raw("f64_reference_residuals", render_list(&self.reference_residuals))
            .raw("pcg_iterations", render_list(&self.pcg_iterations))
            .raw("pcg_true_relative_residuals", render_list(&self.linear_residuals))
            .raw("step_lengths", render_list(&self.step_lengths))
            .render()
    }
}

struct Verdict {
    derivatives: DerivativeCheck,
    f64_exact: bool,
    kernel: (f64, f64),
}

fn worst(scores: &[StepScore], field: fn(&StepScore) -> f64) -> f64 {
    scores.iter().map(field).fold(0.0, f64::max)
}

fn f64_exact_check(run: &Run) -> (bool, Json) {
    match run.coupon.path {
        LoadPath::Patch => {
            let check = patch_check(&run.coupon, &run.reference[0].positions);
            (check.displacement_error <= F64_EXACT_LIMIT && check.strain_error <= F64_PATCH_STRAIN_LIMIT, check.json())
        }
        _ => {
            let check = path_independence(&run.coupon, &run.reference.iter().map(|s| s.positions.clone()).collect::<Vec<Vec<Vec3>>>());
            (check.stress_gap.max(check.final_displacement) <= F64_PATH_LIMIT, check.json())
        }
    }
}

fn f32_physics_check(run: &Run) -> Json {
    let finals = run.finals();
    match run.coupon.path {
        LoadPath::Patch => patch_check(&run.coupon, &finals[0]).json(),
        _ => path_independence(&run.coupon, &finals).json(),
    }
}

fn verdict_json(verdict: &Verdict, scores: &[StepScore]) -> Json {
    let d = &verdict.derivatives;
    let reached = scores.iter().all(|s| s.status == 1.0);
    Json::new()
        .flag("fd_gradient_ok", d.gradient_error <= FD_GRADIENT_LIMIT)
        .flag("fd_tangent_ok", d.tangent_error <= FD_TANGENT_LIMIT)
        .flag("energy_change_identity_ok", d.energy_change_error <= F64_EXACT_LIMIT)
        .flag("f64_exactness_ok", verdict.f64_exact)
        .flag("wgsl_kernel_ok", verdict.kernel.0.max(verdict.kernel.1) <= KERNEL_LIMIT)
        .flag("f32_reaches_newton_tolerance", reached)
        .flag("f32_tracks_f64_newton", worst(scores, |s| s.tracking) <= TRACKING_FACTOR)
        .flag("f32_displacement_ok", worst(scores, |s| s.displacement_error) <= SOLUTION_LIMIT)
        .flag("f32_stress_ok", worst(scores, |s| s.stress_error) <= SOLUTION_LIMIT)
}

fn result_json(run: &Run, dir: &Path, scores: &[StepScore]) -> Result<String, String> {
    let gpu = fs::read_to_string(dir.join("gpu-result.json")).map_err(|e| format!("gpu-result.json: {e}"))?;
    let kernel = kernel_errors(run, dir)?;
    let (f64_exact, exact_json) = f64_exact_check(run);
    let verdict = Verdict { derivatives: derivative_check(&run.coupon), f64_exact, kernel };
    let factors = run.coupon.path.load_factors();
    let steps: Vec<String> = scores.iter().zip(&factors).map(|(s, &t)| s.json(t)).collect();
    Ok(Json::new()
        .text("schema", "e2-result/1")
        .text("case", &format!("{}-n{}", run.coupon.path.name(), run.coupon.mesh.columns))
        .integer("unknowns", 3 * run.coupon.rows())
        .text("backend", &string(&gpu, "", "backend"))
        .text("adapter", &string(&gpu, "", "adapter"))
        .text("host_gpus", &string(&gpu, "", "host_gpus"))
        .text("host_os", &string(&gpu, "", "host_os"))
        .number("gpu_path_ms", number(&gpu, "", "solve_ms"))
        .number("kernel_gradient_relative_error", kernel.0)
        .number("kernel_tangent_relative_error", kernel.1)
        .number("worst_f32_floor", worst(scores, |s| s.f32_residuals.iter().copied().fold(f64::INFINITY, f64::min)))
        .number("worst_f64_floor_of_f32_iterates", worst(scores, |s| s.f64_residuals_of_f32_iterates.iter().copied().fold(f64::INFINITY, f64::min)))
        .number("worst_representation_floor", worst(scores, |s| s.representation_floor))
        .number("worst_displacement_error", worst(scores, |s| s.displacement_error))
        .number("worst_stress_error", worst(scores, |s| s.stress_error))
        .number("worst_tracking_ratio", worst(scores, |s| s.tracking))
        .raw("derivative_check", verdict.derivatives.json().render())
        .raw("f64_exactness", exact_json.render())
        .raw("f32_physics", f32_physics_check(run).render())
        .raw("verdict", verdict_json(&verdict, scores).render())
        .raw("steps", format!("[{}]", steps.join(",")))
        .render())
}

/// Scores a GPU coupon run against the f64 oracle and writes result.json (schema e2-result/1).
pub fn run(dir: &Path) -> Result<(), String> {
    let run = Run::load(dir)?;
    let scores = score_steps(&run);
    let json = result_json(&run, dir, &scores)?;
    write_text(&dir.join("result.json"), &json)?;
    println!("{json}");
    Ok(())
}
