use crate::cases::TIME_STEP;
use crate::drape::{Drape, DrapeStep};
use crate::e2_generate::render_list;
use crate::e3_compare::{probe_summaries, Reference};
use crate::e4_compare::{by_step, Iterate};
use crate::e5_compare::{score_case, write_result, CaseScore};
use crate::io::{read_f32_file, Json};
use crate::jsonscan::{flag, number};
use crate::linalg::Vec3;
use std::fs;
use std::path::Path;

/// Armijo constant and noise term of the GPU search (Experiment 6): accept when Σδ ≤ c·α·gᵀd + 4·ε32·Σ|δ|, ε32 = 2^-23.
const ARMIJO: f64 = 1e-4;
const NOISE_FACTOR: f64 = 4.0;
const EPSILON32: f64 = 1.0 / 8388608.0;
/// A noise-only update that raises f64 Φ by more than this multiple of its noise bound drops rung B.
const SAFETY_FACTOR: f64 = 2.0;
/// GPU reductions of ‖rhs‖² and gᵀd against the f64 sums of the same f32 values.
const REDUCTION_LIMIT: f64 = 1e-5;
const TRIAL_FIELDS: usize = 8;

fn read_f64_rows(path: &Path, width: usize) -> Result<Vec<Vec<f64>>, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let values: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    Ok(values.chunks_exact(width).map(<[f64]>::to_vec).collect())
}

/// One tried step length of a probe line search as the GPU recorded it.
struct Trial {
    alpha: f64,
    sum: f64,
    abs: f64,
    slope: f64,
    accepted: bool,
}

fn trials(gpu_dir: &Path) -> Result<Vec<Trial>, String> {
    Ok(read_f64_rows(&gpu_dir.join("gpu-probe-search.bin"), TRIAL_FIELDS)?.iter()
        .map(|r| Trial { alpha: r[3], sum: r[4], abs: r[5], slope: r[6], accepted: r[7] == 1.0 }).collect())
}

/// The accept decision of the search, in f64 from the recorded numbers: plain Armijo, or with the noise term.
fn replayed_decision(trial: &Trial, noise_aware: bool) -> bool {
    let plain = (ARMIJO * trial.alpha * trial.slope) as f32 as f64;
    trial.sum <= plain || (noise_aware && trial.sum <= plain + NOISE_FACTOR * EPSILON32 * trial.abs)
}

/// How a probe line search that ended without an accepted step missed: the smallest (Σδ − c·α·gᵀd) / (4·ε32·Σ|δ|) over its
/// rejected trials (values above 1 mean the noise term was far too small to explain the failure).
fn rejected_excess(trials: &[Trial]) -> f64 {
    trials.iter().filter(|t| !t.accepted).map(|t| (t.sum - (ARMIJO * t.alpha * t.slope) as f32 as f64) / (NOISE_FACTOR * EPSILON32 * t.abs)).fold(f64::INFINITY, f64::min)
}

/// Acceptance replay: (every decision agrees, trials, rejected trials, smallest excess over the noise bound).
fn acceptance(gpu_dir: &Path, noise_aware: bool) -> Result<(bool, usize, usize, f64), String> {
    let all = trials(gpu_dir)?;
    let agrees = all.iter().all(|t| replayed_decision(t, noise_aware) == t.accepted);
    let rejected = all.iter().filter(|t| !t.accepted).count();
    Ok((agrees, all.len(), rejected, rejected_excess(&all)))
}

/// Worst relative error of the GPU reductions of ‖rhs‖² and gᵀd against the f64 sums of the same values, at probe iterates.
fn reduction_errors(iterates: &[Iterate]) -> (f64, f64, usize) {
    let rows: Vec<&Iterate> = iterates.iter().filter(|it| it.norm_squared_exact.is_finite()).collect();
    let norm = rows.iter().map(|it| ((it.norm_squared - it.norm_squared_exact) / it.norm_squared_exact).abs()).fold(0.0, f64::max);
    let slope = iterates.iter().filter(|it| it.slope_exact.is_finite()).map(|it| ((it.slope - it.slope_exact) / it.slope_exact).abs()).fold(0.0, f64::max);
    (norm, slope, rows.len())
}

fn round_f32(x: &[Vec3]) -> Vec<Vec3> {
    x.iter().map(|p| p.map(|v| v as f32 as f64)).collect()
}

fn to_states(values: &[f32], vertices: usize) -> Vec<Vec<Vec3>> {
    values.chunks_exact(3 * vertices).map(|x| x.chunks_exact(3).map(|p| [p[0] as f64, p[1] as f64, p[2] as f64]).collect()).collect()
}

fn velocity(previous: &[Vec3], current: &[Vec3]) -> Vec<Vec3> {
    current.iter().zip(previous).map(|(p, q)| std::array::from_fn(|d| ((p[d] - q[d]) / TIME_STEP) as f32 as f64)).collect()
}

/// x̂ of time step `step` from the states x_{step−1} and x_{step−2}, both rounded to f32 as the GPU holds them.
fn prediction_for(drape: &Drape, states: &dyn Fn(usize) -> Vec<Vec3>, step: usize) -> Vec<Vec3> {
    let previous = round_f32(&states(step - 1));
    let start_velocity = if step == 1 { vec![[0.0; 3]; previous.len()] } else { velocity(&round_f32(&states(step - 2)), &previous) };
    drape.prediction(&previous, &start_velocity)
}

/// Noise-only updates: (count, violations, worst ratio of the f64 Φ increase to the noise bound), over the probe updates and the
/// trajectory updates the GPU saved (before/after states, probes first). Each update is judged in f64 at its own x̂.
struct NoiseSafety {
    updates: usize,
    violations: usize,
    worst_ratio: f64,
}

fn noise_safety(score: &CaseScore, gpu_dir: &Path, reference: &Reference) -> Result<NoiseSafety, String> {
    let states = to_states(&read_f32_file(&gpu_dir.join("gpu-noise-states.bin"))?, score.drape.mesh.rest.len());
    let probe = noise_rows(&score.probe_iterates);
    let trajectory = noise_rows(&score.iterates);
    if states.len() != 2 * (probe.len() + trajectory.len()) {
        return Err(format!("{} noise states for {} noise-only updates", states.len(), probe.len() + trajectory.len()));
    }
    let probe_x_hat = |step: usize| prediction_for(&score.drape, &|k| reference.positions[k].clone(), step);
    let trajectory_x_hat = |step: usize| prediction_for(&score.drape, &|k| score.gpu_states[k].clone(), step);
    let judged = probe.iter().map(|&(step, abs)| (probe_x_hat(step), abs)).chain(trajectory.iter().map(|&(step, abs)| (trajectory_x_hat(step), abs)));
    let ratios: Vec<f64> = judged.zip(states.chunks_exact(2)).map(|((x_hat, abs), pair)| {
        let step = DrapeStep { drape: &score.drape, x_hat };
        (step.value(&pair[1]) - step.value(&pair[0])) / (NOISE_FACTOR * EPSILON32 * abs)
    }).collect();
    Ok(NoiseSafety { updates: ratios.len(), violations: ratios.iter().filter(|&&r| r > SAFETY_FACTOR).count(), worst_ratio: ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max) })
}

/// (step, Σ|δ|) of every update accepted only thanks to the noise term, in order.
fn noise_rows(iterates: &[Iterate]) -> Vec<(usize, f64)> {
    iterates.iter().filter(|it| it.noise_only == 1.0).map(|it| (it.step, it.delta_abs)).collect()
}

/// Per-step stop status of the trajectory, for the stop-change trigger between rungs.
pub fn step_statuses(iterates: &[Iterate]) -> Vec<f64> {
    by_step(iterates).iter().filter_map(|g| g.last().map(|it| it.status)).collect()
}

fn rung_flag(gpu: &str, key: &str) -> bool {
    flag(gpu, "settings", key)
}

fn e6_json(gpu: &str, accept: (bool, usize, usize, f64), reduction: (f64, f64, usize), safety: Option<&NoiseSafety>, noise_trajectory: usize, probe_errors: &[f64]) -> Json {
    let json = Json::new().text("schema", "e6-checks/1").integer("probe_trials", accept.1).integer("probe_rejected_trials", accept.2)
        .number("smallest_rejected_excess_over_noise_bound", accept.3).integer("noise_accepted_trajectory_updates", noise_trajectory)
        .number("read_bytes_per_newton_iteration", number(gpu, "", "read_bytes_per_newton_iteration")).raw("probe_one_step_errors", render_list(probe_errors))
        .number("reduction_norm_squared_worst_error", reduction.0).number("reduction_slope_worst_error", reduction.1).integer("reduction_checked_iterates", reduction.2);
    match safety {
        Some(s) => json.integer("noise_only_updates_checked", s.updates).integer("noise_safety_violations", s.violations).number("noise_safety_worst_ratio", s.worst_ratio),
        None => json,
    }
}

/// The Experiment 6 gates of one rung that exist for it: the acceptance replay always, the reduction accuracy with GPU reductions,
/// the noise safety with the noise-aware search.
fn rung_verdict(gpu: &str, accept: bool, reduction: (f64, f64, usize), safety: Option<&NoiseSafety>) -> Json {
    let verdict = Json::new().flag("acceptance_replay_ok", accept);
    let verdict = if rung_flag(gpu, "gpuReductions") {
        verdict.flag("reduction_accuracy_ok", reduction.0 <= REDUCTION_LIMIT && reduction.1 <= REDUCTION_LIMIT && reduction.2 > 0)
    } else { verdict };
    match safety {
        Some(s) => verdict.flag("noise_safety_ok", s.violations == 0),
        None => verdict,
    }
}

/// Scores one rung of an E6 case: every E5 check (with the Part 0 rules) plus the acceptance replay, the reduction accuracy and the
/// noise-safety evidence; writes <dir>/e6-rung-X/result.json.
pub fn run(dir: &Path, rung: &str) -> Result<(), String> {
    let gpu_dir = dir.join(format!("e6-rung-{rung}"));
    let score = score_case(dir, &gpu_dir)?;
    let gpu = fs::read_to_string(gpu_dir.join("gpu-result.json")).map_err(|e| e.to_string())?;
    let noise_aware = rung_flag(&gpu, "noiseAware");
    let accept = if score.reference.is_some() { acceptance(&gpu_dir, noise_aware)? } else { (true, 0, 0, f64::NAN) };
    let reduction = reduction_errors(&score.probe_iterates);
    let safety = match (&score.reference, noise_aware) {
        (Some(reference), true) => Some(noise_safety(&score, &gpu_dir, reference)?),
        _ => None,
    };
    let verdict = if score.reference.is_some() { rung_verdict(&gpu, accept.0, reduction, safety.as_ref()) } else { Json::new() }.extend(score.verdict);
    let probe_errors: Vec<f64> = if score.reference.is_some() { probe_summaries(dir, &gpu_dir)?.iter().map(|p| p.error).collect() } else { Vec::new() };
    let details = e6_json(&gpu, accept, reduction, safety.as_ref(), noise_rows(&score.iterates).len(), &probe_errors);
    let json = Json::new().text("rung", rung).raw("e6", details.render()).extend(score.json);
    write_result(&gpu_dir, verdict, json, score.base)
}
