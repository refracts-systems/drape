use crate::drape::{Drape, TangentKind};
use crate::e3_compare::{probe_summaries, score, ProbeSummary};
use crate::e3_generate::kernel_state;
use crate::e4_rules::{forcing_sequence, update_is_small, KERNEL_MU};
use crate::io::{locate, read_f32_file, read_f64, write_text, Json};
use crate::jsonscan::{flag, number, string};
use crate::linalg::{difference_norm, vector_norm};
use crate::newton::MU_CAP;
use std::fs;
use std::path::Path;

const KERNEL_LIMIT: f64 = 1e-5;
/// f32 replay of the GPU forcing terms must match the f64 rule to f32 precision.
const FORCING_REPLAY_LIMIT: f64 = 1e-5;
const FLOOR_FACTOR: f64 = 2.0;
/// Mean GPU Newton iterations per step may exceed the f64 oracle mean by at most this many.
const NEWTON_EXCESS_LIMIT: f64 = 2.0;
/// E3 n65 ms per step divided by 10.
const N65_MS_LIMIT: f64 = 174.0;
const RECORD_FIELDS: usize = 9;
const EXTRA_FIELDS: usize = 5;
const STATUS_RESIDUAL: f64 = 1.0;
const STATUS_UPDATE: f64 = 4.0;

/// One GPU iterate: its IterationRecord fields and the robust-Newton inputs recorded next to it.
struct Iterate {
    step: usize,
    relative: f64,
    pcg_iterations: f64,
    status: f64,
    eta: f64,
    mu: f64,
    update_inf: f64,
    step_inf: f64,
    retries: f64,
}

fn read_f64_rows(path: &Path, width: usize) -> Result<Vec<Vec<f64>>, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let values: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    Ok(values.chunks_exact(width).map(<[f64]>::to_vec).collect())
}

fn read_iterates(dir: &Path, prefix: &str) -> Result<Vec<Iterate>, String> {
    let records = read_f64_rows(&dir.join(format!("gpu-{prefix}iterations.bin")), RECORD_FIELDS)?;
    let extras = read_f64_rows(&dir.join(format!("gpu-{prefix}robust.bin")), EXTRA_FIELDS)?;
    Ok(records.iter().zip(&extras).map(|(r, e)| Iterate {
        step: r[0] as usize, relative: r[2], pcg_iterations: r[3], status: r[8], eta: e[0], mu: e[1], update_inf: e[2], step_inf: e[3], retries: e[4],
    }).collect())
}

fn by_step(iterates: &[Iterate]) -> Vec<Vec<&Iterate>> {
    let mut groups: Vec<Vec<&Iterate>> = Vec::new();
    for it in iterates {
        match groups.last_mut() {
            Some(group) if group[0].step == it.step => group.push(it),
            _ => groups.push(vec![it]),
        }
    }
    groups
}

/// Largest relative gap between the GPU's recorded η and the f64 rule replayed on its recorded residuals.
fn forcing_replay(groups: &[Vec<&Iterate>], forcing: bool, fixed: f64) -> f64 {
    groups.iter().flat_map(|group| {
        let updates: Vec<&&Iterate> = group.iter().filter(|it| it.pcg_iterations.is_finite()).collect();
        let norms: Vec<f64> = updates.iter().map(|it| it.relative).collect();
        let expected = if forcing { forcing_sequence(&norms) } else { vec![fixed; norms.len()] };
        updates.iter().zip(expected).map(|(it, eta)| ((it.eta - eta) / eta).abs()).collect::<Vec<f64>>()
    }).fold(0.0, f64::max)
}

/// Every update-rule stop is justified by its last update, and with the floor stop on, no update the rule would stop on was followed.
fn termination_replay(groups: &[Vec<&Iterate>], floor_stop: bool) -> bool {
    groups.iter().all(|group| {
        let updates: Vec<&&Iterate> = group.iter().filter(|it| it.pcg_iterations.is_finite() && it.update_inf.is_finite()).collect();
        let stopped = group.last().is_some_and(|it| it.status == STATUS_UPDATE);
        let last_ok = !stopped || updates.last().is_some_and(|it| update_is_small(it.update_inf, it.step_inf));
        let early = updates.iter().rev().skip(usize::from(stopped)).any(|it| update_is_small(it.update_inf, it.step_inf));
        last_ok && !(floor_stop && early)
    })
}

/// WGSL inertia kernel at x = kernel state, x̂ = rest, μ = kernel_mu vs f64: gradient (M/h²)(x − x̂), block (1 + μ)(M/h²)I.
fn inertia_error(drape: &Drape, dir: &Path) -> Result<f64, String> {
    let gpu = read_f32_file(&dir.join("gpu-kernel.bin"))?;
    let rows = drape.rows();
    let start = gpu.len() - 12 * rows;
    let state: Vec<f64> = kernel_state(drape).iter().map(|&v| v as f32 as f64).collect();
    let rest = drape.rest_positions();
    let mu = KERNEL_MU as f32 as f64;
    let gradient: Vec<f64> = drape.free.iter().flat_map(|&i| (0..3).map(move |d| (i, d))).map(|(i, d)| drape.mass_over_h2(i) as f32 as f64 * (state[3 * i + d] - rest[i][d])).collect();
    let hessian: Vec<f64> = drape.free.iter().flat_map(|&i| (0..9).map(move |k| if k % 4 == 0 { (1.0 + mu) * (drape.mass_over_h2(i) as f32 as f64) } else { 0.0 })).collect();
    let expected: Vec<f64> = gradient.into_iter().chain(hessian).collect();
    let got: Vec<f64> = gpu[start..].iter().map(|&v| v as f64).collect();
    Ok(difference_norm(&got, &expected) / vector_norm(&expected))
}

/// Mean f64 Newton iterations per step and steps that hit the μ cap, from reference.bin.
fn oracle_stats(dir: &Path) -> Result<(f64, f64, f64), String> {
    let manifest = fs::read_to_string(dir.join("reference.json")).map_err(|e| e.to_string())?;
    let bytes = fs::read(dir.join("reference.bin")).map_err(|e| e.to_string())?;
    let array = |name: &str| locate(&manifest, name).map(|(offset, length)| read_f64(&bytes, offset, length));
    let (iterations, converged, caps) = (array("newton_iterations")?, array("converged")?, array("mu_cap_hit")?);
    let mean = iterations.iter().sum::<f64>() / iterations.len() as f64;
    Ok((mean, iterations.len() as f64 - converged.iter().sum::<f64>(), caps.iter().sum()))
}

/// Per-step GPU statistics of one rung's trajectory.
struct Trajectory {
    newton_per_step: f64,
    pcg_per_solve: f64,
    stops: [usize; 6],
    regularized_steps: usize,
    retries: f64,
    max_mu: f64,
}

fn trajectory_stats(groups: &[Vec<&Iterate>]) -> Trajectory {
    let solves: Vec<&&Iterate> = groups.iter().flatten().filter(|it| it.pcg_iterations.is_finite()).collect();
    let mut stops = [0usize; 6];
    groups.iter().filter_map(|g| g.last()).for_each(|it| stops[(it.status as usize).min(5)] += 1);
    Trajectory {
        newton_per_step: solves.len() as f64 / groups.len() as f64,
        pcg_per_solve: solves.iter().map(|it| it.pcg_iterations).sum::<f64>() / solves.len().max(1) as f64,
        stops,
        regularized_steps: groups.iter().filter(|g| g.iter().any(|it| it.mu > 0.0)).count(),
        retries: solves.iter().map(|it| it.retries).sum(),
        max_mu: groups.iter().flatten().map(|it| it.mu).fold(0.0, f64::max),
    }
}

/// A probe stops properly when it reaches the residual rule, or the update rule within 2x of the representation floor.
fn probe_stops_ok(probes: &[ProbeSummary], groups: &[Vec<&Iterate>]) -> bool {
    probes.iter().all(|p| {
        let status = groups.iter().find(|g| g[0].step == p.step).and_then(|g| g.last()).map_or(f64::NAN, |it| it.status);
        status == STATUS_RESIDUAL || (status == STATUS_UPDATE && p.f32_floor <= FLOOR_FACTOR * p.representation_floor)
    })
}

fn probes_json(probes: &[ProbeSummary], groups: &[Vec<&Iterate>]) -> String {
    let rows: Vec<String> = probes.iter().map(|p| {
        let status = groups.iter().find(|g| g[0].step == p.step).and_then(|g| g.last()).map_or(f64::NAN, |it| it.status);
        Json::new().integer("step", p.step).number("one_step_displacement_error", p.error).number("f32_floor", p.f32_floor)
            .number("representation_floor", p.representation_floor).number("stop_status", status).render()
    }).collect();
    format!("[{}]", rows.join(","))
}

fn robust_json(t: &Trajectory, gpu: &str) -> Json {
    let names = ["running", "residual_rule", "iteration_limit", "line_search_failed", "update_rule", "regularization_cap"];
    let stops = names.iter().zip(t.stops).fold(Json::new(), |json, (name, count)| json.integer(name, count));
    Json::new()
        .number("newton_iterations_per_step", t.newton_per_step)
        .number("pcg_iterations_per_solve", t.pcg_per_solve)
        .raw("stops", stops.render())
        .integer("regularized_steps", t.regularized_steps)
        .number("regularization_retries", t.retries)
        .number("max_mu", t.max_mu)
        .number("ms_per_step", number(gpu, "", "ms_per_step"))
        .raw("phase_ms_per_step", Json::new().number("assembly", number(gpu, "phase_ms_per_step", "assembly")).number("pcg", number(gpu, "phase_ms_per_step", "pcg"))
            .number("line_search", number(gpu, "phase_ms_per_step", "line_search")).number("io", number(gpu, "phase_ms_per_step", "io"))
            .number("other", number(gpu, "phase_ms_per_step", "other")).render())
}

/// Scores one ablation rung of a drape4 case: the full E3 checks plus the Experiment 4 gates; writes <dir>/rung-X/result.json.
pub fn run(dir: &Path, rung: &str) -> Result<(), String> {
    let gpu_dir = dir.join(format!("rung-{rung}"));
    let problem = fs::read_to_string(dir.join("problem.json")).map_err(|e| e.to_string())?;
    let gpu = fs::read_to_string(gpu_dir.join("gpu-result.json")).map_err(|e| e.to_string())?;
    let drape = Drape::build(number(&problem, "", "grid_columns") as usize, TangentKind::Exact);
    let base = score(dir, &gpu_dir)?;
    let groups_owned = read_iterates(&gpu_dir, "")?;
    let groups = by_step(&groups_owned);
    let stats = trajectory_stats(&groups);
    let inertia = inertia_error(&drape, &gpu_dir)?;
    let (forcing, floor_stop) = (flag(&gpu, "settings", "forcing"), flag(&gpu, "settings", "floorStop"));
    let replay = forcing_replay(&groups, forcing, number(&problem, "pcg", "tolerance"));
    let terminations = termination_replay(&groups, floor_stop);
    let reference = flag(&problem, "", "reference");
    let e4 = robust_json(&stats, &gpu).number("inertia_kernel_error", inertia).number("forcing_replay_error", replay).flag("termination_replay_ok", terminations);
    let mut verdict = Json::new()
        .flag("wgsl_kernels_ok", flag(&base, "verdict", "wgsl_kernels_ok") && inertia <= KERNEL_LIMIT)
        .flag("fd_gradient_ok", flag(&base, "verdict", "fd_gradient_ok"))
        .flag("fd_hessian_ok", flag(&base, "verdict", "fd_hessian_ok"))
        .flag("hinge_energy_change_identity_ok", flag(&base, "verdict", "hinge_energy_change_identity_ok"))
        .flag("energy_f32_ok", flag(&base, "verdict", "energy_f32_ok"))
        .flag("rules_replayed_ok", replay <= FORCING_REPLAY_LIMIT && terminations);
    let columns = number(&problem, "", "grid_columns") as usize;
    if columns == 65 {
        verdict = verdict.flag("ms_per_step_ok", number(&gpu, "", "ms_per_step") <= N65_MS_LIMIT);
    }
    let mut json = Json::new().text("schema", "e4-result/1").text("case", &string(&problem, "", "case")).text("rung", rung).integer("unknowns", 3 * drape.rows());
    if reference {
        let probes = probe_summaries(dir, &gpu_dir)?;
        let probe_groups_owned = read_iterates(&gpu_dir, "probe-")?;
        let probe_groups = by_step(&probe_groups_owned);
        let (oracle_mean, oracle_unconverged, oracle_caps) = oracle_stats(dir)?;
        verdict = verdict
            .flag("f64_converged_every_step", oracle_unconverged == 0.0 && oracle_caps == 0.0)
            .flag("energy_f64_ok", flag(&base, "verdict", "energy_f64_ok"))
            .flag("one_step_ok", flag(&base, "verdict", "one_step_ok"))
            .flag("f32_stops_ok_at_probes", probe_stops_ok(&probes, &probe_groups))
            .flag("newton_iterations_ok", stats.newton_per_step <= oracle_mean + NEWTON_EXCESS_LIMIT);
        json = json.raw("oracle", Json::new().number("mean_newton_iterations", oracle_mean).number("unconverged_steps", oracle_unconverged)
            .number("mu_cap_steps", oracle_caps).number("mu_cap", MU_CAP).render()).raw("probes", probes_json(&probes, &probe_groups));
    }
    let rendered = json.raw("robust", e4.render()).raw("verdict", verdict.render()).raw("e3_checks", base).render();
    write_text(&gpu_dir.join("result.json"), &rendered)?;
    println!("{rendered}");
    Ok(())
}
