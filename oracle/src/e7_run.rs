use crate::drape::{Drape, Scene, SceneParams, TangentKind};
use crate::e2_generate::render_list;
use crate::e3_generate::{problem_arrays, problem_json, E7_SCHEME};
use crate::e5_checks::contact_events;
use crate::e7_ptc::{gravity_load, ptc_solve, static_energy, static_gradient, PtcRun, PTC};
use crate::e7_state::{read_state, write_state, State};
use crate::io::{write_text, BinaryArrays, Json};
use crate::linalg::{vector_norm, Vec3};
use std::fs;
use std::path::Path;

/// GPU stop rule constants (Experiment 7): equilibrium at r/L ≤ 1e-3, or stagnation over two windows of 10 steps.
const GPU_EQUILIBRIUM: f64 = 1e-3;
const STAGNATION_WINDOW: usize = 10;
const STAGNATION_FACTOR: f64 = 0.95;

/// The three edits of Part 2 (stiffness, body, size) and the base scene.
pub fn edit_params(edit: &str) -> Result<SceneParams, String> {
    match edit {
        "base" => Ok(SceneParams::BASE),
        "w1" => Ok(SceneParams { material: 1.25, ..SceneParams::BASE }),
        "w2" => Ok(SceneParams { radius: 0.96, ..SceneParams::BASE }),
        "w3" => Ok(SceneParams { rest: 1.03, ..SceneParams::BASE }),
        other => Err(format!("unknown edit '{other}' (base|w1|w2|w3)")),
    }
}

fn ptc_json(drape: &Drape) -> Json {
    Json::new().number("h0", PTC.h0).number("h_max", PTC.h_max).number("h_min_factor", PTC.h_min_factor).number("growth_cap", PTC.growth_cap)
        .number("inner_tolerance", PTC.inner_tolerance).integer("inner_max_iterations", PTC.inner_max).integer("max_steps", PTC.max_steps)
        .number("equilibrium_f64", PTC.equilibrium).number("equilibrium_f32", GPU_EQUILIBRIUM).integer("stagnation_window", STAGNATION_WINDOW)
        .number("stagnation_factor", STAGNATION_FACTOR).number("gravity_load", gravity_load(drape))
}

fn params_json(params: SceneParams) -> Json {
    Json::new().number("material_factor", params.material).number("radius_factor", params.radius).number("rest_factor", params.rest)
}

/// Writes problem.json / problem.bin of the E5 scene with the edit applied, plus the PTC rules and the unit masses the GPU needs
/// to rewrite M/h² for every step.
pub fn generate(columns: usize, edit: &str, dir: &Path) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let params = edit_params(edit)?;
    let drape = Drape::for_scene_with(columns, TangentKind::Exact, Scene::SphereOnFloor, params);
    let mut arrays = problem_arrays(&drape, None, &[]);
    arrays.push_f32("row_unit_mass", drape.free.iter().map(|&v| drape.mass[v] as f32));
    arrays.write(&dir.join("problem.bin"))?;
    let extra = Json::new().text("edit", edit).raw("scene_params", params_json(params).render()).raw("ptc", ptc_json(&drape).render());
    write_text(&dir.join("problem.json"), &problem_json(&drape, None, &E7_SCHEME, &arrays.manifest(), &[], extra))
}

fn flat(run: &PtcRun) -> Vec<f64> {
    run.states.iter().flatten().flatten().copied().collect()
}

fn run_json(drape: &Drape, run: &PtcRun, mode: &str) -> Json {
    let load = gravity_load(drape);
    let rounded: Vec<Vec3> = run.last().iter().map(|p| p.map(|v| v as f32 as f64)).collect();
    let floor = vector_norm(&static_gradient(drape, &rounded)) / load;
    let newton: usize = run.steps.iter().map(|s| s.inner_iterations).sum();
    Json::new().text("schema", "e7-reference/1").text("mode", mode).flag("converged", run.converged()).text("reason", run.reason)
        .integer("ptc_steps", run.steps.len()).integer("newton_iterations", newton).number("final_relative_residual", run.final_residual / load)
        .number("representation_floor", floor).number("final_h", run.final_h).number("gravity_load", load)
        .number("static_energy", static_energy(drape, run.last())).integer("mu_cap_steps", run.steps.iter().filter(|s| s.mu_cap).count())
        .number("max_mu", run.steps.iter().map(|s| s.max_mu).fold(0.0, f64::max))
}

fn write_reference(dir: &Path, mode: &str, drape: &Drape, run: &PtcRun) -> Result<(), String> {
    let mut arrays = BinaryArrays::new();
    arrays.push_f64("positions", &flat(run));
    arrays.push_f64("h", &run.steps.iter().map(|s| s.h).collect::<Vec<f64>>());
    arrays.push_f64("residual", &run.steps.iter().map(|s| s.residual).chain([run.final_residual]).collect::<Vec<f64>>());
    arrays.push_f64("inner_iterations", &run.steps.iter().map(|s| s.inner_iterations as f64).collect::<Vec<f64>>());
    arrays.write(&dir.join(format!("reference-{mode}.bin")))?;
    let json = run_json(drape, run, mode).raw("arrays", arrays.manifest());
    write_text(&dir.join(format!("reference-{mode}.json")), &json.render())
}

/// Probe steps (1-based; step s goes from state s−1 to s) chosen from the oracle trajectory: step 1, the first steps whose end
/// state touches the sphere / the floor, and the first steps whose end state has r/L ≤ 1e-1 and ≤ 1e-2.
fn probe_steps(drape: &Drape, run: &PtcRun) -> Vec<usize> {
    let load = gravity_load(drape);
    let events = contact_events(drape, &run.states);
    let first = |threshold: f64| (1..run.states.len()).find(|&s| vector_norm(&static_gradient(drape, &run.states[s])) / load <= threshold);
    let mut steps: Vec<usize> = [Some(1), events.first_sphere, events.first_floor, first(1e-1), first(1e-2)].into_iter().flatten().collect();
    steps.sort_unstable();
    steps.dedup();
    steps
}

/// probes.bin / probes.json: the f32 start state and the f32 step size of every probe step (what the GPU repeats once).
fn write_probes(dir: &Path, drape: &Drape, run: &PtcRun) -> Result<(), String> {
    let steps = probe_steps(drape, run);
    let mut arrays = BinaryArrays::new();
    arrays.push_u32("probe_steps", steps.iter().map(|&s| s as u32));
    arrays.push_f32("probe_h", steps.iter().map(|&s| run.steps[s - 1].h as f32));
    arrays.push_f32("probe_positions", steps.iter().flat_map(|&s| run.states[s - 1].iter().flatten().map(|&v| v as f32).collect::<Vec<f32>>()));
    arrays.write(&dir.join("probes.bin"))?;
    let listed = render_list(&steps.iter().map(|&s| s as f64).collect::<Vec<f64>>());
    write_text(&dir.join("probes.json"), &Json::new().integer("probes", steps.len()).raw("steps", listed).raw("arrays", arrays.manifest()).render())
}

/// Solves the case in `dir` by PTC from the flat release (cold) or from the state at `start` (warm, prolong) and writes
/// reference-<mode>.{json,bin}, equilibrium-<mode>.{json,bin} and, for cold solves with `with_probes`, probes.{json,bin}.
pub fn solve(dir: &Path, mode: &str, start: Option<&Path>, with_probes: bool) -> Result<(), String> {
    let problem = fs::read_to_string(dir.join("problem.json")).map_err(|e| e.to_string())?;
    let drape = Drape::from_problem(&problem, TangentKind::Exact);
    let (positions, h) = match start {
        Some(prefix) => { let s = read_state(prefix)?; (s.positions, s.final_h) }
        None => (drape.rest_positions(), PTC.h0),
    };
    let run = ptc_solve(&drape, positions, h)?;
    write_reference(dir, mode, &drape, &run)?;
    write_state(&dir.join(format!("equilibrium-{mode}")), &State { positions: run.last().to_vec(), final_h: run.final_h, single: false })?;
    if with_probes {
        write_probes(dir, &drape, &run)?;
    }
    println!("{}", run_json(&drape, &run, mode).render());
    if mode == "cold" && !run.converged() {
        return Err(format!("STOP: the f64 oracle did not converge ({})", run.reason));
    }
    Ok(())
}
