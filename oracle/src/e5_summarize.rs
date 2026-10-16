use crate::criteria::ITERATION_GROWTH_LIMIT;
use crate::io::{write_text, Json};
use crate::jsonscan::{flag, number, string};
use std::fs;
use std::path::Path;

/// Every Part 1 check is an accuracy gate; Newton iterations, n65 ms/step against E4-final and PCG growth are performance gates.
const ACCURACY: [&str; 14] = [
    "wgsl_kernels_ok", "contact_kernel_ok", "fd_gradient_ok", "fd_hessian_ok", "contact_fd_ok", "hinge_energy_change_identity_ok",
    "barrier_energy_change_identity_ok", "intersection_free", "f64_converged_every_step", "one_step_ok", "f32_stops_ok_at_probes",
    "energy_f64_ok", "energy_f32_ok", "rules_replayed_ok",
];
/// n65 ms per step may be at most this multiple of E4-final's.
const MS_FACTOR: f64 = 2.0;

fn mark(text: &str, key: &str) -> &'static str {
    if !text.contains(&format!("\"{key}\"")) { "-" } else if flag(text, "verdict", key) { "Y" } else { "n" }
}

fn print_row(text: &str) {
    let marks: Vec<&str> = ACCURACY.iter().map(|k| mark(text, k)).collect();
    let phase = |name: &str| number(text, "phase_ms_per_step", name);
    println!(
        "{:<11} newton/step={:<6.2} pcg/solve={:<7.1} ms/step={:<8.1} [asm {:.1} ccd {:.1} pcg {:.1} ls {:.1} io {:.1} other {:.1}] stops res/upd/lim/ls={}/{}/{}/{} min d/dhat={:.3e} accuracy={} newton_ok={}",
        string(text, "", "case"), number(text, "robust", "newton_iterations_per_step"), number(text, "robust", "pcg_iterations_per_solve"), number(text, "robust", "ms_per_step"),
        phase("assembly"), phase("ccd"), phase("pcg"), phase("line_search"), phase("io"), phase("other"),
        number(text, "stops", "residual_rule"), number(text, "stops", "update_rule"), number(text, "stops", "iteration_limit"), number(text, "stops", "line_search_failed"),
        number(text, "contact", "min_distance_over_dhat_accepted_iterates"), marks.join(""), mark(text, "newton_iterations_ok"),
    );
}

fn growth(results: &[String]) -> Vec<Json> {
    results.windows(2).map(|pair| {
        let ratio = number(&pair[1], "robust", "pcg_iterations_per_solve") / number(&pair[0], "robust", "pcg_iterations_per_solve");
        Json::new().text("from", &string(&pair[0], "", "case")).text("to", &string(&pair[1], "", "case"))
            .number("pcg_iteration_growth", ratio).flag("within_limit", ratio < ITERATION_GROWTH_LIMIT)
    }).collect()
}

/// E4-final n65 ms/step from its summary (the `robust` block that follows the drape-n65 case).
fn baseline_ms(baseline: &Path) -> Result<f64, String> {
    let text = fs::read_to_string(baseline).map_err(|e| format!("{}: {e}", baseline.display()))?;
    let at = text.find("\"case\":\"drape-n65\"").ok_or("E4-final summary has no drape-n65 case")?;
    Ok(number(&text[at..], "robust", "ms_per_step"))
}

fn load(dir: &Path) -> Result<Vec<String>, String> {
    let mut results: Vec<String> = fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?.filter_map(Result::ok)
        .filter_map(|case| fs::read_to_string(case.path().join("gpu").join("result.json")).ok()).collect();
    results.sort_by(|a, b| number(a, "", "unknowns").total_cmp(&number(b, "", "unknowns")));
    Ok(results)
}

/// Collects drape5/*/gpu/result.json, prints the per-case table and gates, and writes the Experiment 5 summary.
pub fn run(dir: &Path, baseline: &Path, out: &Path) -> Result<(), String> {
    let results = load(dir)?;
    println!("accuracy gates: {}", ACCURACY.join("/"));
    results.iter().for_each(|text| print_row(text));
    let scaling = growth(&results);
    scaling.iter().for_each(|g| println!("scaling {}", g.render()));
    let accuracy = results.iter().all(|t| ACCURACY.iter().all(|k| mark(t, k) != "n"));
    let reference_ms = baseline_ms(baseline)?;
    let n65_ms = results.iter().find(|t| string(t, "", "case") == "drape-n65").map_or(f64::NAN, |t| number(t, "robust", "ms_per_step"));
    let performance = Json::new().number("e4_final_n65_ms_per_step", reference_ms).number("n65_ms_per_step", n65_ms)
        .flag("ms_per_step_ok", n65_ms <= MS_FACTOR * reference_ms).flag("newton_iterations_ok", results.iter().all(|t| mark(t, "newton_iterations_ok") != "n"))
        .flag("pcg_growth_ok", scaling.iter().all(|g| g.render().contains("\"within_limit\":true")));
    println!("accuracy gates pass in every case: {accuracy}\nperformance: {}", performance.render());
    let rendered: Vec<String> = scaling.iter().map(Json::render).collect();
    let summary = Json::new().text("schema", "e5-summary/1").flag("accuracy_ok", accuracy).raw("performance", performance.render())
        .raw("pcg_growth", format!("[{}]", rendered.join(","))).raw("cases", format!("[{}]", results.join(",")));
    write_text(out, &summary.render())
}
