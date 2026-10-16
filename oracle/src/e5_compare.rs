use crate::cases::SHEET_SIZE;
use crate::contact::Collider;
use crate::drape::{Drape, DrapeStep, TangentKind};
use crate::e2_generate::render_list;
use crate::e3_compare::{probe_summaries, score, Reference};
use crate::e4_compare::{
    by_step, forcing_replay, inertia_error, oracle_stats, probe_stops_ok, probes_json, read_iterates, robust_json, termination_replay,
    trajectory_stats, Iterate, FORCING_REPLAY_LIMIT, KERNEL_LIMIT, NEWTON_EXCESS_LIMIT,
};
use crate::e4_rules::f32_spacing;
use crate::e5_checks::{active_counts, contact_derivative_check, contact_state};
use crate::io::{read_f32_file, write_text, Json};
use crate::jsonscan::{flag, string};
use crate::linalg::Vec3;
use crate::newton::NewtonSystem;
use std::fs;
use std::path::Path;

const FD_GRADIENT_LIMIT: f64 = 1e-6;
const FD_HESSIAN_LIMIT: f64 = 1e-5;
const BARRIER_IDENTITY_LIMIT: f64 = 1e-12;
/// Contact kernel: per-entry tolerance max(1e-5, 8·ε32·cond), fixed before any run.
const CONTACT_KERNEL_RELATIVE: f64 = 1e-5;
const CONTACT_KERNEL_COND_FACTOR: f64 = 8.0;
/// α_max recomputed in f64 from the recorded probe iterates and directions must match the GPU to 1e-5 relative.
const ALPHA_REPLAY_LIMIT: f64 = 1e-5;

fn to_states(values: &[f32], vertices: usize) -> Vec<Vec<Vec3>> {
    values.chunks_exact(3 * vertices).map(|x| x.chunks_exact(3).map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect()).collect()
}

/// cond = max over active colliders of (|x − c|/d)·max(1, |b″·d/b′|); the plane's |x − c|/d is 1.
fn condition(drape: &Drape, x: Vec3) -> f64 {
    let barrier = drape.contact.barrier;
    drape.contact.colliders.iter().filter(|c| c.distance(x) < barrier.dhat).map(|c| {
        let d = c.distance(x);
        let lever = match c { Collider::Plane { .. } => 1.0, Collider::Sphere { .. } => 1.0 / (c.curvature(x) * d) };
        lever * (barrier.second(d) * d / barrier.first(d)).abs().max(1.0)
    }).fold(1.0, f64::max)
}

/// Entries of one row group (3 gradient or 9 Hessian values) against f64, relative to the group's largest expected entry:
/// (entries above 1e-5, failures above the tolerance, worst error/tolerance).
fn group_errors(got: &[f32], expected: &[f64], tolerance: f64) -> (usize, usize, f64) {
    let scale = expected.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    got.iter().zip(expected).fold((0, 0, 0.0), |(cond, fail, worst), (&g, &e)| {
        let error = (g as f64 - e).abs() / scale;
        (cond + usize::from(error > CONTACT_KERNEL_RELATIVE), fail + usize::from(error > tolerance), worst.max(error / tolerance))
    })
}

/// WGSL contact kernel vs f64 at the f32-rounded contact state: [entries needing the cond branch, failures, worst ratio].
fn contact_kernel(drape: &Drape, gpu_dir: &Path) -> Result<[f64; 3], String> {
    let got = read_f32_file(&gpu_dir.join("gpu-contact-kernel.bin"))?;
    let rows = drape.rows();
    let state: Vec<Vec3> = contact_state(drape).iter().map(|p| p.map(|v| v as f32 as f64)).collect();
    let totals = drape.free.iter().enumerate().fold([0.0, 0.0, 0.0f64], |sum, (row, &i)| {
        let tolerance = CONTACT_KERNEL_RELATIVE.max(CONTACT_KERNEL_COND_FACTOR * f32::EPSILON as f64 * condition(drape, state[i]));
        let hessian: Vec<f64> = drape.contact.hessian(state[i]).into_iter().flatten().collect();
        let g = group_errors(&got[3 * row..3 * row + 3], &drape.contact.gradient(state[i]), tolerance);
        let h = group_errors(&got[3 * rows + 9 * row..3 * rows + 9 * row + 9], &hessian, tolerance);
        [sum[0] + (g.0 + h.0) as f64, sum[1] + (g.1 + h.1) as f64, sum[2].max(g.2).max(h.2)]
    });
    Ok(totals)
}

fn min_distance(drape: &Drape, states: &[Vec<Vec3>]) -> f64 {
    states.iter().flat_map(|x| drape.free.iter().map(move |&i| drape.contact.min_distance(x[i]))).fold(f64::INFINITY, f64::min)
}

/// Largest relative gap between the GPU's α_max and the f64 rule at the recorded probe iterate x_k and direction p_k.
fn alpha_replay(drape: &Drape, gpu_dir: &Path, iterates: &[Iterate]) -> Result<f64, String> {
    let pairs = to_states(&read_f32_file(&gpu_dir.join("gpu-probe-directions.bin"))?, drape.mesh.rest.len());
    let updates: Vec<&Iterate> = iterates.iter().filter(|it| it.pcg_iterations.is_finite()).collect();
    if pairs.len() != 2 * updates.len() {
        return Err(format!("{} direction states for {} probe updates", pairs.len(), updates.len()));
    }
    let step = DrapeStep { drape, x_hat: Vec::new() };
    Ok(updates.iter().zip(pairs.chunks_exact(2)).map(|(it, pair)| ((step.max_step(&pair[0], &pair[1]) - it.alpha_max) / it.alpha_max).abs()).fold(0.0, f64::max))
}

/// The f32 spacing the GPU recorded for each probe update equals f32_spacing(max |x_i|) of the iterate it was recorded at.
fn resolution_replay(drape: &Drape, gpu_dir: &Path, iterates: &[Iterate]) -> Result<bool, String> {
    let states = to_states(&read_f32_file(&gpu_dir.join("gpu-probe-iterates.bin"))?, drape.mesh.rest.len());
    Ok(iterates.iter().zip(&states).filter(|(it, _)| it.pcg_iterations.is_finite()).all(|(it, x)| {
        let largest = drape.free.iter().flat_map(|&i| x[i]).fold(0.0f64, |m, v| m.max(v.abs()));
        f32_spacing(largest) == it.resolution
    }))
}

/// Mass-weighted horizontal centre of the sheet minus the sphere axis, per state.
fn com_drift(drape: &Drape, states: &[Vec<Vec3>]) -> Vec<f64> {
    let total: f64 = drape.mass.iter().sum();
    states.iter().map(|x| {
        let c: [f64; 2] = std::array::from_fn(|k| x.iter().zip(&drape.mass).map(|(p, m)| m * p[k]).sum::<f64>() / total - 0.5 * SHEET_SIZE);
        c[0].hypot(c[1])
    }).collect()
}

fn contact_json(drape: &Drape, gpu_states: &[Vec<Vec3>], reference: Option<&Reference>) -> Json {
    let counts = |states: &[Vec<Vec3>]| active_counts(drape, states).iter().map(|(s, f)| (s + f) as f64).collect::<Vec<f64>>();
    let drift = com_drift(drape, gpu_states);
    let json = Json::new().raw("gpu_active_contacts", render_list(&counts(gpu_states)))
        .number("gpu_com_drift_max_m", drift.iter().copied().fold(0.0, f64::max)).number("gpu_com_drift_final_m", *drift.last().unwrap());
    match reference {
        Some(r) => json.raw("f64_active_contacts", render_list(&counts(&r.positions))).number("f64_min_distance_over_dhat", min_distance(drape, &r.positions) / drape.contact.barrier.dhat),
        None => json,
    }
}

/// Intersection checks: f32 trajectory states in f64, the GPU's f64 record of every accepted iterate, probe iterates in f64.
fn distances(drape: &Drape, gpu_states: &[Vec<Vec3>], gpu_dir: &Path, iterates: &[&[Iterate]]) -> Result<[f64; 3], String> {
    let recorded = iterates.iter().flat_map(|its| its.iter()).map(|it| it.min_distance).filter(|d| d.is_finite()).fold(f64::INFINITY, f64::min);
    let probe_states = to_states(&read_f32_file(&gpu_dir.join("gpu-probe-iterates.bin"))?, drape.mesh.rest.len());
    Ok([min_distance(drape, gpu_states), recorded, min_distance(drape, &probe_states)])
}

struct Scored {
    json: Json,
    verdict: Json,
}

fn contact_checks(drape: &Drape, gpu_dir: &Path, gpu_states: &[Vec<Vec3>], iterates: &[&[Iterate]]) -> Result<Scored, String> {
    let kernel = contact_kernel(drape, gpu_dir)?;
    let fd = contact_derivative_check(drape);
    let nearest = distances(drape, gpu_states, gpu_dir, iterates)?;
    let dhat = drape.contact.barrier.dhat;
    let json = Json::new().number("contact_kernel_entries_needing_cond", kernel[0]).number("contact_kernel_failures", kernel[1])
        .number("contact_kernel_worst_error_over_tolerance", kernel[2]).raw("contact_derivative_check", fd.json().render())
        .number("min_distance_over_dhat_f32_states", nearest[0] / dhat).number("min_distance_over_dhat_accepted_iterates", nearest[1] / dhat)
        .number("min_distance_over_dhat_probe_iterates", nearest[2] / dhat);
    let verdict = Json::new().flag("contact_kernel_ok", kernel[1] == 0.0)
        .flag("contact_fd_ok", fd.gradient_error.max(fd.barrier_gradient_error) <= FD_GRADIENT_LIMIT && fd.hessian_error.max(fd.barrier_hessian_error) <= FD_HESSIAN_LIMIT)
        .flag("barrier_energy_change_identity_ok", fd.barrier_change_error <= BARRIER_IDENTITY_LIMIT)
        .flag("intersection_free", nearest.iter().all(|&d| d > 0.0));
    Ok(Scored { json, verdict })
}

fn base_verdict(base: &str, inertia: f64, rules_ok: bool) -> Json {
    let keep = |json: Json, key: &str| json.flag(key, flag(base, "verdict", key));
    let json = Json::new().flag("wgsl_kernels_ok", flag(base, "verdict", "wgsl_kernels_ok") && inertia <= KERNEL_LIMIT);
    let json = ["fd_gradient_ok", "fd_hessian_ok", "hinge_energy_change_identity_ok", "energy_f32_ok"].iter().fold(json, |j, k| keep(j, k));
    json.flag("rules_replayed_ok", rules_ok)
}

fn reference_checks(dir: &Path, gpu_dir: &Path, base: &str, newton_per_step: f64) -> Result<Scored, String> {
    let probes = probe_summaries(dir, gpu_dir)?;
    let probe_iterates = read_iterates(gpu_dir, "probe-")?;
    let probe_groups = by_step(&probe_iterates);
    let oracle = oracle_stats(dir)?;
    let verdict = Json::new()
        .flag("f64_converged_every_step", oracle.unconverged == 0.0 && oracle.caps == 0.0)
        .flag("energy_f64_ok", flag(base, "verdict", "energy_f64_ok"))
        .flag("one_step_ok", flag(base, "verdict", "one_step_ok"))
        .flag("f32_stops_ok_at_probes", probe_stops_ok(&probes, &probe_groups))
        .flag("newton_iterations_ok", newton_per_step <= oracle.mean + NEWTON_EXCESS_LIMIT);
    let json = Json::new().raw("oracle", Json::new().number("mean_newton_iterations", oracle.mean).number("unconverged_steps", oracle.unconverged)
        .number("converged_at_floor_steps", oracle.at_floor).number("mu_cap_steps", oracle.caps).render()).raw("probes", probes_json(&probes, &probe_groups));
    Ok(Scored { json, verdict })
}


/// Forcing, termination (with resolution) and, at probes, α_max and resolution replays: (all pass, α_max gap, resolution ok).
fn rules(drape: &Drape, gpu_dir: &Path, groups: &[Vec<&Iterate>], probe_iterates: Option<&[Iterate]>) -> Result<(bool, f64, bool), String> {
    let alpha = probe_iterates.map_or(Ok(0.0), |its| alpha_replay(drape, gpu_dir, its))?;
    let resolution = probe_iterates.map_or(Ok(true), |its| resolution_replay(drape, gpu_dir, its))?;
    let forcing_ok = forcing_replay(groups, true, f64::NAN) <= FORCING_REPLAY_LIMIT && termination_replay(groups, true);
    Ok((forcing_ok && alpha <= ALPHA_REPLAY_LIMIT && resolution, alpha, resolution))
}

/// Scores one drape5 case: every E3/E4 check plus the contact checks; writes <dir>/gpu/result.json.
pub fn run(dir: &Path) -> Result<(), String> {
    let (gpu_dir, problem) = (dir.join("gpu"), fs::read_to_string(dir.join("problem.json")).map_err(|e| e.to_string())?);
    let gpu = fs::read_to_string(gpu_dir.join("gpu-result.json")).map_err(|e| e.to_string())?;
    let drape = Drape::from_problem(&problem, TangentKind::Exact);
    let base = score(dir, &gpu_dir)?;
    let iterates = read_iterates(&gpu_dir, "")?;
    let stats = trajectory_stats(&by_step(&iterates));
    let gpu_states = to_states(&read_f32_file(&gpu_dir.join("gpu-trajectory.bin"))?, drape.mesh.rest.len());
    let reference = if flag(&problem, "", "reference") { Some(Reference::load(dir, drape.mesh.rest.len())?) } else { None };
    let probe_iterates = if reference.is_some() { read_iterates(&gpu_dir, "probe-")? } else { Vec::new() };
    let (rules_ok, alpha, resolution_ok) = rules(&drape, &gpu_dir, &by_step(&iterates), reference.as_ref().map(|_| probe_iterates.as_slice()))?;
    let contact = contact_checks(&drape, &gpu_dir, &gpu_states, &[&iterates, &probe_iterates])?;
    let verdict = base_verdict(&base, inertia_error(&drape, &gpu_dir)?, rules_ok).extend(contact.verdict);
    let json = Json::new().text("schema", "e5-result/1").text("case", &string(&problem, "", "case")).integer("unknowns", 3 * drape.rows())
        .raw("robust", robust_json(&stats, &gpu).number("alpha_replay_error", alpha).flag("resolution_replay_ok", resolution_ok).render())
        .raw("contact", contact.json.extend(contact_json(&drape, &gpu_states, reference.as_ref())).render());
    let checked = if reference.is_some() { Some(reference_checks(dir, &gpu_dir, &base, stats.newton_per_step)?) } else { None };
    let (verdict, json) = match checked { Some(c) => (verdict.extend(c.verdict), json.extend(c.json)), None => (verdict, json) };
    write_result(&gpu_dir, verdict, json, base)
}

/// The E5 verdict first (jsonscan reads the first "verdict"), then the E5 details, then the full E3 result.
fn write_result(gpu_dir: &Path, verdict: Json, json: Json, base: String) -> Result<(), String> {
    let rendered = Json::new().raw("verdict", verdict.render()).extend(json).raw("e3_checks", base).render();
    write_text(&gpu_dir.join("result.json"), &rendered)?;
    println!("{rendered}");
    Ok(())
}
