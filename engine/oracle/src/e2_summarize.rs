use crate::io::{write_text, Json};
use crate::jsonscan::{flag, number, string};
use std::fs;
use std::path::Path;

const CHECKS: [&str; 9] = [
    "fd_gradient_ok", "fd_tangent_ok", "energy_change_identity_ok", "f64_exactness_ok", "wgsl_kernel_ok",
    "f32_reaches_newton_tolerance", "f32_tracks_f64_newton", "f32_displacement_ok", "f32_stress_ok",
];

fn short(value: f64) -> String {
    if value.is_finite() { format!("{value:.2e}") } else { "null".to_string() }
}

fn print_row(text: &str) {
    let pass = CHECKS.map(|k| if flag(text, "verdict", k) { "Y" } else { "n" });
    println!(
        "{:<12} N={:<5} f32floor={:<8} f64(f32)={:<8} repfloor={:<8} disp={:<8} stress={:<8} track={:<8} kernel={:<8}/{:<8} {}={}",
        string(text, "", "case"), number(text, "", "unknowns"), short(number(text, "", "worst_f32_floor")),
        short(number(text, "", "worst_f64_floor_of_f32_iterates")), short(number(text, "", "worst_representation_floor")),
        short(number(text, "", "worst_displacement_error")), short(number(text, "", "worst_stress_error")),
        short(number(text, "", "worst_tracking_ratio")), short(number(text, "", "kernel_gradient_relative_error")),
        short(number(text, "", "kernel_tangent_relative_error")), CHECKS.join("/"), pass.join("/"),
    );
}

/// Collects every <dir>/*/result.json (schema e2-result/1) into one file and prints the check table.
pub fn run(dir: &Path, out: &Path) -> Result<(), String> {
    let mut results: Vec<String> = fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|entry| fs::read_to_string(entry.ok()?.path().join("result.json")).ok())
        .collect();
    results.sort();
    results.iter().for_each(|text| print_row(text));
    let summary = Json::new().text("schema", "e2-summary/1").raw("cases", format!("[{}]", results.join(",")));
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    write_text(out, &summary.render())
}
