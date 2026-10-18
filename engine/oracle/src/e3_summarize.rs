use crate::criteria::ITERATION_GROWTH_LIMIT;
use crate::io::{write_text, Json};
use crate::jsonscan::{flag, number, string};
use std::fs;
use std::path::Path;

const CHECKS: [&str; 9] = [
    "fd_gradient_ok", "fd_hessian_ok", "hinge_energy_change_identity_ok", "wgsl_kernels_ok", "energy_f32_ok",
    "f64_converged_or_at_floor", "energy_f64_ok", "one_step_ok", "f32_newton_ok_at_probes",
];

fn short(value: f64) -> String {
    if value.is_finite() { format!("{value:.2e}") } else { "-".to_string() }
}

fn print_row(text: &str) {
    let pass = CHECKS.map(|k| if !text.contains(&format!("\"{k}\"")) { "-" } else if flag(text, "verdict", k) { "Y" } else { "n" });
    println!(
        "{:<11} N={:<6} ms/step={:<8} pcg/solve={:<8} f32conv={:<4} f32worst={:<8} f64conv={:<4} f64worst={:<8} shape={:<8} {}={}",
        string(text, "", "case"), number(text, "", "unknowns"), short(number(text, "", "ms_per_step")),
        short(number(text, "trajectory", "pcg_iterations_per_newton_solve")), number(text, "trajectory", "f32_steps_converged"),
        short(number(text, "trajectory", "f32_worst_final_residual")), number(text, "reference_comparison", "f64_steps_converged"),
        short(number(text, "reference_comparison", "f64_worst_final_residual")),
        short(number(text, "reference_comparison", "final_shape_max_difference_per_sheet_size")), CHECKS.join("/"), pass.join("/"),
    );
}

/// PCG iterations per Newton solve between consecutive mesh sizes (the E1 iteration-growth gate).
fn growth(results: &[String]) -> Vec<Json> {
    results.windows(2).map(|pair| {
        let ratio = number(&pair[1], "trajectory", "pcg_iterations_per_newton_solve") / number(&pair[0], "trajectory", "pcg_iterations_per_newton_solve");
        Json::new().text("from", &string(&pair[0], "", "case")).text("to", &string(&pair[1], "", "case"))
            .number("pcg_iteration_growth", ratio).flag("within_limit", ratio < ITERATION_GROWTH_LIMIT)
    }).collect()
}

/// Collects every <dir>/*/result.json (schema e3-result/1), smallest mesh first, and prints the check table.
pub fn run(dir: &Path, out: &Path) -> Result<(), String> {
    let mut results: Vec<String> = fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|entry| fs::read_to_string(entry.ok()?.path().join("result.json")).ok())
        .collect();
    results.sort_by(|a, b| number(a, "", "unknowns").total_cmp(&number(b, "", "unknowns")));
    results.iter().for_each(|text| print_row(text));
    let scaling = growth(&results);
    scaling.iter().for_each(|g| println!("scaling {}", g.render()));
    let rendered: Vec<String> = scaling.iter().map(Json::render).collect();
    let summary = Json::new().text("schema", "e3-summary/1").raw("pcg_growth", format!("[{}]", rendered.join(",")))
        .raw("cases", format!("[{}]", results.join(",")));
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    write_text(out, &summary.render())
}
