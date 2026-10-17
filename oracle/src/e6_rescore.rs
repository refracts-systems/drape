use crate::drape::{Drape, TangentKind};
use crate::e3_compare::Reference;
use crate::e4_diagnose::step_from;
use crate::e4_compare;
use crate::e5_compare;
use crate::io::{locate, read_f64, write_text, Json};
use crate::jsonscan::{flag, number};
use crate::newton::RegularizedSystem;
use std::fs;
use std::path::{Path, PathBuf};

/// An f64 step is converged at 1e-10, or within this factor of its measured rounding floor.
pub const FLOOR_FACTOR: f64 = 2.0;

/// One step that did not reach 1e-10: its final relative residual, the measured floor and their ratio.
pub struct FloorRow {
    pub step: usize,
    pub residual: f64,
    pub floor: f64,
}

/// How the stored f64 steps of a reference converge: at the tolerance, within 2x the measured floor, or not at all.
pub struct Convergence {
    pub by_tolerance: usize,
    pub by_floor: usize,
    pub failing: Vec<usize>,
    pub rows: Vec<FloorRow>,
}

fn scale_of(dir: &Path) -> Result<Vec<f64>, String> {
    let manifest = fs::read_to_string(dir.join("reference.json")).map_err(|e| e.to_string())?;
    let bytes = fs::read(dir.join("reference.bin")).map_err(|e| e.to_string())?;
    locate(&manifest, "scale").map(|(offset, length)| read_f64(&bytes, offset, length))
}

/// Classifies every stored step of the case in `dir` with
/// floor_k = ε64·(‖Σ|g_c|‖₂ + ‖|K|·|x|‖₂)/scale_k at the final stored iterate; converged ⇔ residual ≤ 1e-10 or ≤ 2·floor_k.
pub fn classify(dir: &Path) -> Result<Convergence, String> {
    let problem = fs::read_to_string(dir.join("problem.json")).map_err(|e| e.to_string())?;
    let drape = Drape::from_problem(&problem, TangentKind::Exact);
    let reference = Reference::load(dir, drape.mesh.rest.len())?;
    let scale = scale_of(dir)?;
    let mut result = Convergence { by_tolerance: 0, by_floor: 0, failing: Vec::new(), rows: Vec::new() };
    for step in 1..=reference.converged.len() {
        if reference.converged[step - 1] == 1.0 {
            result.by_tolerance += 1;
            continue;
        }
        let floor = step_from(&drape, &reference, step).residual_floor(&reference.positions[step]) / scale[step - 1];
        let residual = reference.final_residual[step - 1];
        result.rows.push(FloorRow { step, residual, floor });
        if residual <= FLOOR_FACTOR * floor { result.by_floor += 1 } else { result.failing.push(step) }
    }
    Ok(result)
}

fn rows_json(c: &Convergence) -> String {
    let rows: Vec<String> = c.rows.iter().map(|r| Json::new().integer("step", r.step).number("final_relative_residual", r.residual)
        .number("floor", r.floor).number("residual_over_floor", r.residual / r.floor).render()).collect();
    format!("[{}]", rows.join(","))
}

fn convergence_json(dir: &Path, c: &Convergence) -> Json {
    Json::new().text("case", &dir.display().to_string().replace('\\', "/")).integer("converged_by_tolerance", c.by_tolerance).integer("converged_by_floor", c.by_floor)
        .integer("failing", c.failing.len()).raw("failing_steps", format!("[{}]", c.failing.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",")))
        .raw("unconverged_rows", rows_json(c))
}

/// Gate flags of a stored result.json, in the order of `keys` ("-" when a key is absent).
fn marks(text: &str, keys: &[&str]) -> Vec<&'static str> {
    keys.iter().map(|k| if !text.contains(&format!("\"{k}\"")) { "-" } else if flag(text, "verdict", k) { "Y" } else { "n" }).collect()
}

const GATES: [&str; 14] = [
    "wgsl_kernels_ok", "contact_kernel_ok", "fd_gradient_ok", "fd_hessian_ok", "contact_fd_ok", "hinge_energy_change_identity_ok",
    "barrier_energy_change_identity_ok", "intersection_free", "f64_converged_every_step", "one_step_ok", "f32_stops_ok_at_probes",
    "energy_f64_ok", "energy_f32_ok", "rules_replayed_ok",
];

fn gate_changes(label: &str, before: &str, after: &str) -> Vec<String> {
    let (b, a) = (marks(before, &GATES), marks(after, &GATES));
    GATES.iter().zip(b.iter().zip(&a)).filter(|(_, (x, y))| x != y).map(|(k, (x, y))| format!("{label}: {k} {x} -> {y}")).collect()
}

fn rescore_one(label: &str, dir: &Path, result: &Path, rescore: &dyn Fn() -> Result<(), String>) -> Result<(Vec<String>, Json), String> {
    let before = fs::read_to_string(result).unwrap_or_default();
    rescore()?;
    let after = fs::read_to_string(result).map_err(|e| e.to_string())?;
    let convergence = if flag(&after, "", "reference") || number(&after, "oracle", "unconverged_steps").is_finite() { Some(classify(dir)?) } else { None };
    let json = convergence.as_ref().map_or_else(Json::new, |c| convergence_json(dir, c));
    Ok((gate_changes(label, &before, &after), json.text("label", label)))
}

/// Re-scores the stored case in `dir` (`result` is its result.json) and records its label, gate changes and convergence.
fn rescore_case(label: String, dir: PathBuf, result: PathBuf, rescore: &dyn Fn(&Path) -> Result<(), String>) -> Result<(Vec<String>, String), String> {
    let (changes, json) = rescore_one(&label, &dir, &result, &|| rescore(&dir))?;
    Ok((changes, json.render()))
}

/// Part 0.4: re-scores the stored E4-final rung D and E5 GPU outputs with the Experiment 6 measurement rules (no GPU runs)
/// and reports which gates change status, the convergence branch counts and the steps that still fail.
pub fn run(root: &Path, out: &Path) -> Result<(), String> {
    let (mut changes, mut cases) = (Vec::new(), Vec::new());
    for n in [33, 65, 129] {
        let dir = root.join(format!("drape4-final/drape-n{n}"));
        let (c, j) = rescore_case(format!("e4-final n{n}"), dir.clone(), dir.join("rung-D").join("result.json"), &|d| e4_compare::run(d, "D"))?;
        (changes.extend(c), cases.push(j));
    }
    for n in [33, 65, 129] {
        let dir = root.join(format!("drape5/drape-n{n}"));
        let (c, j) = rescore_case(format!("e5 n{n}"), dir.clone(), dir.join("gpu").join("result.json"), &e5_compare::run)?;
        (changes.extend(c), cases.push(j));
    }
    changes.iter().for_each(|c| println!("{c}"));
    let summary = Json::new().text("schema", "e6-rescore/1").raw("cases", format!("[{}]", cases.join(",")))
        .raw("gate_changes", format!("[{}]", changes.iter().map(|c| format!("\"{c}\"")).collect::<Vec<_>>().join(",")));
    write_text(out, &summary.render())
}
