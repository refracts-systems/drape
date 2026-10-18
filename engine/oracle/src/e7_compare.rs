use crate::drape::{Drape, DrapeStep, TangentKind};
use crate::e3_compare::kernel_errors;
use crate::e4_compare::{by_step, forcing_replay, inertia_error, read_f64_rows, read_iterates, termination_replay, Iterate};
use crate::e5_checks::contact_derivative_check;
use crate::e5_compare::{alpha_replay, contact_kernel, resolution_replay, to_states};
use crate::e6_compare::acceptance;
use crate::e7_ptc::{ptc_prediction, static_energy, static_gradient};
use crate::io::{locate, read_f32_file, read_f64, write_text, Json};
use crate::jsonscan::{flag, number, string};
use crate::linalg::{difference_norm, vector_norm, Vec3};
use crate::newton::NewtonSystem;
use std::fs;
use std::path::{Path, PathBuf};

const KERNEL_LIMIT: f64 = 1e-5;
const ONE_STEP_LIMIT: f64 = 1e-3;
const FLOOR_FACTOR: f64 = 2.0;
/// f32 equilibrium tolerance of the final-state gate, before the representation-floor allowance.
const RESIDUAL_FLOOR_ALLOWANCE: f64 = 1e-3;
/// Static energy may rise by at most this fraction of max |E_s| along the accepted f32 states.
const ENERGY_LIMIT: f64 = 1e-6;
const CONTROLLER_TOLERANCE: f64 = 1e-12;
const STATUS_RESIDUAL: f64 = 1.0;
const STATUS_UPDATE: f64 = 4.0;

/// One scored GPU run: the case directory (problem and oracle references), the run directory and the parsed problem.
struct Run {
    dir: PathBuf,
    gpu_dir: PathBuf,
    problem: String,
    gpu: String,
    load: f64,
}

impl Run {
    fn open(dir: &Path, mode: &str) -> Result<Run, String> {
        let gpu_dir = dir.join(format!("gpu-{mode}"));
        let problem = fs::read_to_string(dir.join("problem.json")).map_err(|e| e.to_string())?;
        let gpu = fs::read_to_string(gpu_dir.join("gpu-result.json")).map_err(|e| format!("{}: {e}", gpu_dir.display()))?;
        let load = number(&problem, "ptc", "gravity_load");
        Ok(Run { dir: dir.to_path_buf(), gpu_dir, problem, gpu, load })
    }

    fn drape(&self) -> Drape {
        Drape::from_problem(&self.problem, TangentKind::Exact)
    }

    fn columns(&self) -> usize {
        number(&self.problem, "", "grid_columns") as usize
    }

    fn rows(&self, file: &str, width: usize) -> Result<Vec<Vec<f64>>, String> {
        read_f64_rows(&self.gpu_dir.join(file), width)
    }

    fn states(&self, drape: &Drape) -> Result<Vec<Vec<Vec3>>, String> {
        Ok(to_states(&read_f32_file(&self.gpu_dir.join("gpu-states.bin"))?, drape.mesh.rest.len()))
    }
}

/// What the f64 oracle recorded for a case and mode (reference-<mode>.json), if it was run.
struct Reference {
    converged: bool,
    steps: f64,
    newton: f64,
    rejected: f64,
    floor: f64,
    mu_cap_steps: f64,
}

fn reference(dir: &Path, mode: &str) -> Option<Reference> {
    let text = fs::read_to_string(dir.join(format!("reference-{mode}.json"))).ok()?;
    Some(Reference { converged: flag(&text, "", "converged"), steps: number(&text, "", "ptc_steps"), newton: number(&text, "", "newton_iterations"),
        rejected: number(&text, "", "rejected_attempts"), floor: number(&text, "", "representation_floor"), mu_cap_steps: number(&text, "", "mu_cap_steps") })
}

fn reference_positions(dir: &Path, mode: &str, vertices: usize) -> Result<Vec<Vec<Vec3>>, String> {
    let manifest = fs::read_to_string(dir.join(format!("reference-{mode}.json"))).map_err(|e| e.to_string())?;
    let bytes = fs::read(dir.join(format!("reference-{mode}.bin"))).map_err(|e| e.to_string())?;
    let (offset, length) = locate(&manifest, "positions")?;
    Ok(read_f64(&bytes, offset, length).chunks_exact(3 * vertices).map(|x| x.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect()).collect())
}

fn reference_array(dir: &Path, mode: &str, name: &str) -> Result<Vec<f64>, String> {
    let manifest = fs::read_to_string(dir.join(format!("reference-{mode}.json"))).map_err(|e| e.to_string())?;
    let bytes = fs::read(dir.join(format!("reference-{mode}.bin"))).map_err(|e| e.to_string())?;
    locate(&manifest, name).map(|(offset, length)| read_f64(&bytes, offset, length))
}

fn round_f32(x: &[Vec3]) -> Vec<Vec3> {
    x.iter().map(|p| p.map(|v| v as f32 as f64)).collect()
}

/// Smallest vertex-collider distance over the unknowns of one state, divided by d̂.
fn min_distance_ratio(drape: &Drape, state: &[Vec3]) -> f64 {
    drape.free.iter().map(|&i| drape.contact.min_distance(state[i])).fold(f64::INFINITY, f64::min) / drape.contact.barrier.dhat
}

/// Largest rise of E_s between consecutive states relative to max |E_s| (negative: it only falls).
fn energy_rise(drape: &Drape, states: &[Vec<Vec3>]) -> f64 {
    let energies: Vec<f64> = states.iter().map(|x| static_energy(drape, x)).collect();
    let scale = energies.iter().fold(0.0f64, |m, e| m.max(e.abs()));
    energies.windows(2).map(|w| (w[1] - w[0]) / scale).fold(f64::NEG_INFINITY, f64::max)
}

/// r/L of one state in f64.
fn residual_over_load(drape: &Drape, state: &[Vec3], load: f64) -> f64 {
    vector_norm(&static_gradient(drape, state)) / load
}

/// h_{k+1} must be h_k·growth (inner < cap) or h_k·shrink, clamped, divided by 2 once per rejected attempt of step k+1.
fn controller_replay(problem: &str, rows: &[Vec<f64>]) -> bool {
    let p = |key: &str| number(problem, "ptc", key);
    let (growth, shrink, cap, floor, top) = (p("growth"), p("shrink"), p("inner_max_iterations"), p("h0") * p("h_min_factor"), p("h_max"));
    rows.windows(2).all(|pair| {
        let factor = if pair[0][3] < cap { growth } else { shrink };
        let base = (pair[0][1] * factor).clamp(floor, top);
        let expected = (base * shrink.powi(pair[1][4] as i32)).max(floor);
        ((pair[1][1] - expected) / expected).abs() <= CONTROLLER_TOLERANCE
    })
}

/// The records of every attempt (accepted or rejected) as one group each, from gpu-attempts.bin.
fn attempt_groups<'a>(iterates: &'a [Iterate], attempts: &[Vec<f64>]) -> Vec<Vec<&'a Iterate>> {
    attempts.iter().map(|a| iterates[a[4] as usize..a[5] as usize].iter().collect()).collect()
}

/// A rejection is a regularisation-cap or line-search failure before any update (reasons 1 and 2) or a non-finite state (3); 4 is a run superseded by the curvature trigger.
fn rejection_replay(iterates: &[Iterate], attempts: &[Vec<f64>]) -> bool {
    attempts.iter().filter(|a| a[3] != 0.0).all(|a| {
        let records = &iterates[a[4] as usize..a[5] as usize];
        let updates = records.iter().filter(|r| r.status == 0.0 && r.pcg_iterations.is_finite()).count();
        let last = records.last().map_or(f64::NAN, |r| r.status);
        a[3] == 3.0 || a[3] == 4.0 || (updates == 0 && ((a[3] == 1.0 && last == 5.0) || (a[3] == 2.0 && last == 3.0)))
    })
}

/// The start state of a warm or prolonged run: its r/L (f64) and whether it is feasible, from <prefix>.json/.bin (f32).
fn start_info(run: &Run, drape: &Drape, mode: &str) -> Option<(f64, f64)> {
    let prefix = match mode {
        "warm" => run.dir.parent()?.join("base").join("gpu-cold").join("gpu-equilibrium"),
        "prolong" => run.dir.join("start-prolong-gpu"),
        _ => return None,
    };
    let bytes = fs::read(format!("{}.bin", prefix.display())).ok()?;
    let floats: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let state = to_states(&floats, drape.mesh.rest.len()).pop()?;
    Some((residual_over_load(drape, &state, run.load), min_distance_ratio(drape, &state)))
}

/// A probe: the step, the one-step error against the oracle, the f32 floor, the representation floor and the stop status.
struct Probe {
    step: usize,
    error: f64,
    f32_floor: f64,
    representation_floor: f64,
    status: f64,
}

impl Probe {
    fn stops_ok(&self) -> bool {
        self.status == STATUS_RESIDUAL || (self.status == STATUS_UPDATE && self.f32_floor <= FLOOR_FACTOR * self.representation_floor)
    }
}

fn state_gap(a: &[Vec3], b: &[Vec3]) -> f64 {
    a.iter().zip(b).flat_map(|(p, q)| (0..3).map(move |d| (p[d] - q[d]).powi(2))).sum::<f64>().sqrt()
}

/// The probes of a cold base run: the GPU's last iterate of each probed step against the oracle's next state.
fn probes(run: &Run, mode: &str) -> Result<Vec<Probe>, String> {
    let drape = run.drape();
    let iterates = read_iterates(&run.gpu_dir, "probe-")?;
    let states = to_states(&read_f32_file(&run.gpu_dir.join("gpu-probe-iterates.bin"))?, drape.mesh.rest.len());
    let oracle = reference_positions(&run.dir, mode, drape.mesh.rest.len())?;
    let steps = reference_array(&run.dir, mode, "h")?;
    Ok(by_step(&iterates).iter().map(|group| probe(&drape, group, &iterates, &states, &oracle, &steps)).collect())
}

fn probe(drape: &Drape, group: &[&Iterate], iterates: &[Iterate], states: &[Vec<Vec3>], oracle: &[Vec<Vec3>], h: &[f64]) -> Probe {
    let step = group[0].step;
    let last = iterates.iter().rposition(|it| it.step == step).expect("a probe step has records");
    let (previous, next) = (&oracle[step - 1], &oracle[step]);
    drape.time_step.set(h[step - 1] as f32 as f64);
    let system = DrapeStep { drape, x_hat: ptc_prediction(drape, previous, h[step - 1] as f32 as f64) };
    let scale = vector_norm(&system.residual(previous));
    let representation = vector_norm(&system.residual(&round_f32(next))) / scale;
    Probe { step, error: state_gap(&states[last], next) / state_gap(next, previous), f32_floor: group.iter().map(|it| it.relative).fold(f64::INFINITY, f64::min),
        representation_floor: representation, status: group.last().map_or(f64::NAN, |it| it.status) }
}

fn probe_json(p: &Probe) -> String {
    Json::new().integer("step", p.step).number("one_step_displacement_error", p.error).number("f32_floor", p.f32_floor)
        .number("representation_floor", p.representation_floor).number("stop_status", p.status).flag("stop_ok", p.stops_ok()).render()
}

/// Inertia kernel at the three check step sizes against f64: the worst relative error of [gradient | block diagonal] over them.
fn inertia_h_error(run: &Run, drape: &Drape) -> Result<f64, String> {
    let got = read_f32_file(&run.gpu_dir.join("gpu-inertia-h.bin"))?;
    let rows = drape.rows();
    let state: Vec<f64> = crate::e3_generate::kernel_state(drape).iter().map(|&v| v as f32 as f64).collect();
    let rest = round_f32(&drape.rest_positions());
    let sizes = [1.0 / 240.0, 0.1, 1.0];
    Ok(sizes.iter().enumerate().map(|(k, &h)| {
        let expected: Vec<f64> = inertia_expected(drape, &state, &rest, h);
        let seen: Vec<f64> = got[12 * rows * k..12 * rows * (k + 1)].iter().map(|&v| v as f64).collect();
        difference_norm(&seen, &expected) / vector_norm(&expected)
    }).fold(0.0, f64::max))
}

/// [m32(x − x̂) per component | m32 on the block diagonal] with m32 = f32(unit mass / h²).
fn inertia_expected(drape: &Drape, state: &[f64], rest: &[Vec3], h: f64) -> Vec<f64> {
    let weight = |i: usize| (drape.mass[i] as f32 as f64 / (h * h)) as f32 as f64;
    let gradient = drape.free.iter().flat_map(|&i| (0..3).map(move |d| (i, d))).map(|(i, d)| weight(i) * (state[3 * i + d] - rest[i][d])).collect::<Vec<f64>>();
    let blocks = drape.free.iter().flat_map(|&i| (0..9).map(move |k| if k % 4 == 0 { weight(i) } else { 0.0 })).collect::<Vec<f64>>();
    gradient.into_iter().chain(blocks).collect()
}

/// The kernel gates of a run that carries kernel outputs (cold base runs started with --checks).
fn kernel_gates(run: &Run, drape: &Drape) -> Result<(Json, Json), String> {
    if !run.gpu_dir.join("gpu-kernel.bin").exists() {
        return Ok((Json::new(), Json::new()));
    }
    let elements = kernel_errors(drape, &run.gpu_dir)?.iter().all(|&e| e <= KERNEL_LIMIT);
    let inertia = inertia_error(drape, &run.gpu_dir)?;
    let contact = contact_kernel(drape, &run.gpu_dir)?;
    let inertia_h = inertia_h_error(run, drape)?;
    let verdict = Json::new().flag("wgsl_kernels_ok", elements && inertia <= KERNEL_LIMIT).flag("contact_kernel_ok", contact[1] == 0.0).flag("inertia_step_sizes_ok", inertia_h <= KERNEL_LIMIT);
    Ok((verdict, Json::new()
        .number("inertia_step_sizes_error", inertia_h).number("contact_kernel_entries_needing_cond", contact[0]).number("contact_kernel_worst_error_over_tolerance", contact[2])))
}

/// The f64 oracle's verdict on its own solve: converged, no μ-cap accepted step.
fn oracle_json(r: &Reference) -> Json {
    Json::new().flag("converged", r.converged).number("ptc_steps", r.steps).number("newton_iterations", r.newton).number("rejected_attempts", r.rejected)
        .number("representation_floor", r.floor).number("mu_cap_accepted_steps", r.mu_cap_steps)
}

/// The replays of one run: forcing and termination per attempt, the controller, the rejections, and on probes alpha, resolution and acceptance.
fn replays(run: &Run, drape: &Drape, steps: &[Vec<f64>], attempts: &[Vec<f64>]) -> Result<(bool, Json), String> {
    let iterates = read_iterates(&run.gpu_dir, "")?;
    let groups = attempt_groups(&iterates, attempts);
    let forcing = forcing_replay(&groups, true, f64::NAN);
    let (termination, controller, rejection) = (termination_replay(&groups, true), controller_replay(&run.problem, steps), rejection_replay(&iterates, attempts));
    let probe_ok = probe_replays(run, drape)?;
    let ok = forcing <= 1e-5 && termination && controller && rejection && probe_ok;
    Ok((ok, Json::new().number("forcing_replay_error", forcing).flag("termination_replay_ok", termination).flag("controller_replay_ok", controller)
        .flag("rejection_replay_ok", rejection).flag("probe_replays_ok", probe_ok)))
}

/// Alpha, resolution and acceptance replays on the probe iterates (true when the run has none).
fn probe_replays(run: &Run, drape: &Drape) -> Result<bool, String> {
    if !run.gpu_dir.join("gpu-probe-iterates.bin").exists() {
        return Ok(true);
    }
    let iterates = read_iterates(&run.gpu_dir, "probe-")?;
    let alpha = alpha_replay(drape, &run.gpu_dir, &iterates)?;
    let resolution = resolution_replay(drape, &run.gpu_dir, &iterates)?;
    let accepted = acceptance(&run.gpu_dir, true)?;
    Ok(alpha.worst_ratio <= 1.0 && resolution && accepted.0)
}

/// Intersection-free: every kept f32 state and every probe iterate in f64, and the GPU's f32 minimum distance of every accepted step.
fn intersection(run: &Run, drape: &Drape, states: &[Vec<Vec3>], steps: &[Vec<f64>]) -> Result<(bool, f64), String> {
    let mut ratios: Vec<f64> = states.iter().map(|x| min_distance_ratio(drape, x)).collect();
    ratios.extend(steps.iter().map(|s| s[6] / drape.contact.barrier.dhat).filter(|r| r.is_finite()));
    if run.gpu_dir.join("gpu-probe-iterates.bin").exists() {
        let probe = to_states(&read_f32_file(&run.gpu_dir.join("gpu-probe-iterates.bin"))?, drape.mesh.rest.len());
        ratios.extend(probe.iter().map(|x| min_distance_ratio(drape, x)));
    }
    let smallest = ratios.iter().copied().fold(f64::INFINITY, f64::min);
    Ok((smallest > 0.0, smallest))
}

/// The gates and numbers that need the oracle reference of this run's mode or of the cold solve of the same case.
fn oracle_gates(run: &Run, mode: &str, drape: &Drape, final_state: &[Vec3]) -> (Json, Json) {
    let own = reference(&run.dir, mode);
    let cold = reference(&run.dir, "cold");
    let residual = residual_over_load(drape, final_state, run.load);
    let mut verdict = Json::new();
    if let Some(r) = &own {
        verdict = verdict.flag("oracle_converged", r.converged && r.mu_cap_steps == 0.0);
    }
    if let Some(c) = &cold {
        verdict = verdict.flag("final_state_residual_ok", residual <= RESIDUAL_FLOOR_ALLOWANCE.max(FLOOR_FACTOR * c.floor));
    }
    let details = Json::new().raw("oracle", own.as_ref().map_or_else(|| "null".to_string(), |r| oracle_json(r).render()))
        .raw("oracle_cold", cold.as_ref().map_or_else(|| "null".to_string(), |r| oracle_json(r).render())).number("final_residual_over_load_f64", residual);
    (verdict, details)
}

/// Diagnostics against the oracle's cold equilibrium of the same case: static energy gap and largest vertex gap per sheet side.
fn cold_gaps(run: &Run, drape: &Drape, final_state: &[Vec3]) -> Json {
    let Ok(states) = reference_positions(&run.dir, "cold", drape.mesh.rest.len()) else { return Json::new() };
    let Some(cold) = states.last() else { return Json::new() };
    let gap = final_state.iter().zip(cold).map(|(p, q)| (0..3).map(|d| (p[d] - q[d]).powi(2)).sum::<f64>().sqrt()).fold(0.0, f64::max);
    Json::new().number("energy_gap_over_cold", (static_energy(drape, final_state) - static_energy(drape, cold)) / static_energy(drape, cold).abs())
        .number("max_vertex_gap_over_side", gap / crate::cases::SHEET_SIZE)
}

fn gpu_json(run: &Run, steps: &[Vec<f64>], min_ratio: f64) -> Json {
    let g = |key: &str| number(&run.gpu, "", key);
    let phases = ["assembly", "pcg", "line_search", "ccd", "reduce", "io", "other"].iter().fold(Json::new(), |json, name| json.number(name, number(&run.gpu, "phase_ms_per_step", name) * g("accepted_steps")));
    Json::new().text("stop", &string(&run.gpu, "", "stop")).number("accepted_steps", g("accepted_steps")).number("rejected_attempts", g("rejected_attempts")).number("curvature_dropped_steps", g("curvature_dropped_steps"))
        .number("newton_iterations", g("newton_iterations_total")).number("pcg_iterations", g("pcg_iterations_total")).number("ms", g("trajectory_ms"))
        .raw("phase_ms_total", phases.render()).number("read_bytes", g("read_bytes_total")).number("final_h", g("final_h"))
        .number("final_relative_residual_gpu", g("final_relative_residual")).number("min_distance_over_dhat", min_ratio)
        .number("h_min", steps.iter().map(|s| s[1]).fold(f64::INFINITY, f64::min)).number("h_max", steps.iter().map(|s| s[1]).fold(0.0, f64::max))
        .raw("ms_runs", gpu_runs(&run.gpu)).integer("grid_columns", run.columns())
}

/// ms_per_step_runs of the merged gpu-result.json as a list (empty for a single run).
fn gpu_runs(gpu: &str) -> String {
    let Some(at) = gpu.find("\"ms_per_step_runs\"") else { return "[]".to_string() };
    let open = at + gpu[at..].find('[').unwrap_or(0);
    let close = open + gpu[open..].find(']').unwrap_or(0);
    gpu[open..=close].split_whitespace().collect::<Vec<&str>>().join("")
}

/// The probe results as a JSON list, and the probe gates added to the verdict (cold runs with probe outputs only).
fn probe_json_list(run: &Run, mode: &str, verdict: Json) -> Result<(Json, String), String> {
    if !run.gpu_dir.join("gpu-probe-iterates.bin").exists() || reference(&run.dir, mode).is_none() {
        return Ok((verdict, "[]".to_string()));
    }
    let found = probes(run, mode)?;
    let verdict = verdict.flag("one_step_ok", found.iter().all(|p| p.error <= ONE_STEP_LIMIT)).flag("f32_stops_ok_at_probes", found.iter().all(Probe::stops_ok));
    Ok((verdict, format!("[{}]", found.iter().map(probe_json).collect::<Vec<String>>().join(","))))
}

/// The gates that need only the GPU outputs of the run: kernels, intersection, replays, energy.
fn run_gates(run: &Run, drape: &Drape, states: &[Vec<Vec3>], steps: &[Vec<f64>], attempts: &[Vec<f64>]) -> Result<(Json, Json, f64), String> {
    let (replay_ok, replay_json) = replays(run, drape, steps, attempts)?;
    let (intersection_ok, min_ratio) = intersection(run, drape, states, steps)?;
    let (kernel_verdict, kernel_details) = kernel_gates(run, drape)?;
    let mut verdict = kernel_verdict.flag("intersection_free", intersection_ok).flag("rules_replayed_ok", replay_ok);
    if states.len() > 2 {
        verdict = verdict.flag("static_energy_ok", energy_rise(drape, states) <= ENERGY_LIMIT);
    }
    Ok((verdict, replay_json.extend(kernel_details), min_ratio))
}

/// Scores one GPU run (<dir>/gpu-<mode>) of an Experiment 7 case and writes <dir>/gpu-<mode>/result.json.
pub fn run(dir: &Path, mode: &str) -> Result<(), String> {
    let run = Run::open(dir, mode)?;
    let drape = run.drape();
    let (steps, attempts) = (run.rows("gpu-ptc.bin", 7)?, run.rows("gpu-attempts.bin", 6)?);
    let states = run.states(&drape)?;
    let final_state = states.last().ok_or("no states")?;
    let (gates, replay_json, min_ratio) = run_gates(&run, &drape, &states, &steps, &attempts)?;
    let (oracle_verdict, oracle_details) = oracle_gates(&run, mode, &drape, final_state);
    let (verdict, probe_list) = probe_json_list(&run, mode, gates.extend(oracle_verdict))?;
    let start = start_info(&run, &drape, mode);
    let verdict = match start { Some((_, d)) => verdict.flag("start_feasible", d > 0.0), None => verdict };
    let start_json = start.map_or_else(Json::new, |(r, d)| Json::new().number("start_residual_over_load", r).number("start_min_distance_over_dhat", d));
    let rise = if states.len() > 2 { energy_rise(&drape, &states) } else { f64::NAN };
    let json = Json::new().text("schema", "e7-result/1").text("case", &string(&run.problem, "", "case")).text("edit", &string(&run.problem, "", "edit")).text("mode", mode)
        .raw("gpu", gpu_json(&run, &steps, min_ratio).render()).raw("replays", replay_json.render()).raw("probes", probe_list).extend(oracle_details)
        .extend(start_json).extend(cold_gaps(&run, &drape, final_state)).number("energy_rise_relative", rise).raw("derivatives", contact_derivative_check(&drape).json().render());
    let rendered = Json::new().raw("verdict", verdict.render()).extend(json).render();
    write_text(&run.gpu_dir.join("result.json"), &rendered)?;
    println!("{rendered}");
    Ok(())
}
