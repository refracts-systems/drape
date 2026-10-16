use crate::criteria::ITERATION_GROWTH_LIMIT;
use crate::io::{write_text, Json};
use crate::jsonscan::{flag, number, string};
use std::fs;
use std::path::Path;

/// Accuracy gates decide the final rung; ms_per_step_ok and newton_iterations_ok are the performance gates.
const ACCURACY: [&str; 10] = [
    "wgsl_kernels_ok", "fd_gradient_ok", "fd_hessian_ok", "hinge_energy_change_identity_ok", "energy_f32_ok", "rules_replayed_ok",
    "f64_converged_every_step", "energy_f64_ok", "one_step_ok", "f32_stops_ok_at_probes",
];
const PERFORMANCE: [&str; 2] = ["newton_iterations_ok", "ms_per_step_ok"];
const RUNGS: [&str; 4] = ["A", "B", "C", "D"];

fn mark(text: &str, key: &str) -> &'static str {
    if !text.contains(&format!("\"{key}\"")) { "-" } else if flag(text, "verdict", key) { "Y" } else { "n" }
}

fn accuracy_ok(text: &str) -> bool {
    ACCURACY.iter().all(|k| mark(text, k) != "n")
}

fn print_row(text: &str) {
    let marks: Vec<&str> = ACCURACY.iter().chain(PERFORMANCE.iter()).map(|k| mark(text, k)).collect();
    println!(
        "{:<11} rung {} newton/step={:<6.2} pcg/solve={:<7.1} ms/step={:<8.1} [asm {:.1} pcg {:.1} ls {:.1} io {:.1} other {:.1}] stops res/upd/lim={}/{}/{} reg-steps={} accuracy+perf={}",
        string(text, "", "case"), string(text, "", "rung"), number(text, "robust", "newton_iterations_per_step"), number(text, "robust", "pcg_iterations_per_solve"),
        number(text, "robust", "ms_per_step"), number(text, "phase_ms_per_step", "assembly"), number(text, "phase_ms_per_step", "pcg"),
        number(text, "phase_ms_per_step", "line_search"), number(text, "phase_ms_per_step", "io"), number(text, "phase_ms_per_step", "other"),
        number(text, "stops", "residual_rule"), number(text, "stops", "update_rule"), number(text, "stops", "iteration_limit"),
        number(text, "robust", "regularized_steps"), marks.join(""),
    );
}

/// PCG iterations per solve between consecutive meshes of one rung (the E1 iteration-growth gate).
fn growth(results: &[String], rung: &str) -> Vec<Json> {
    let rows: Vec<&String> = results.iter().filter(|r| string(r, "", "rung") == rung).collect();
    rows.windows(2).map(|pair| {
        let ratio = number(pair[1], "robust", "pcg_iterations_per_solve") / number(pair[0], "robust", "pcg_iterations_per_solve");
        Json::new().text("rung", rung).text("from", &string(pair[0], "", "case")).text("to", &string(pair[1], "", "case"))
            .number("pcg_iteration_growth", ratio).flag("within_limit", ratio < ITERATION_GROWTH_LIMIT)
    }).collect()
}

fn load(dir: &Path) -> Result<Vec<String>, String> {
    let mut results = Vec::new();
    for case in fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?.filter_map(Result::ok) {
        results.extend(RUNGS.iter().filter_map(|r| fs::read_to_string(case.path().join(format!("rung-{r}")).join("result.json")).ok()));
    }
    results.sort_by(|a, b| string(a, "", "rung").cmp(&string(b, "", "rung")).then(number(a, "", "unknowns").total_cmp(&number(b, "", "unknowns"))));
    Ok(results)
}

/// Collects every drape4/*/rung-*/result.json, prints the ablation table and picks the final rung: the last one whose
/// accuracy gates pass in every case.
pub fn run(dir: &Path, out: &Path) -> Result<(), String> {
    let results = load(dir)?;
    println!("gates: {} then {}", ACCURACY.join("/"), PERFORMANCE.join("/"));
    results.iter().for_each(|text| print_row(text));
    let scaling: Vec<Json> = RUNGS.iter().flat_map(|r| growth(&results, r)).collect();
    scaling.iter().for_each(|g| println!("scaling {}", g.render()));
    let passing: Vec<&str> = RUNGS.iter().copied().filter(|r| results.iter().filter(|t| string(t, "", "rung") == *r).all(|t| accuracy_ok(t))).collect();
    let final_rung = passing.last().copied().unwrap_or("none");
    println!("final configuration: rung {final_rung}");
    let rendered: Vec<String> = scaling.iter().map(Json::render).collect();
    let summary = Json::new().text("schema", "e4-summary/1").text("final_rung", final_rung)
        .raw("pcg_growth", format!("[{}]", rendered.join(","))).raw("cases", format!("[{}]", results.join(",")));
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    write_text(out, &summary.render())
}
