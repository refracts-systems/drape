use crate::criteria::ITERATION_GROWTH_LIMIT;
use crate::io::{write_text, Json};
use crate::jsonscan::{flag, number, string};
use std::fs;
use std::path::Path;

struct Row {
    case: String,
    columns: f64,
    text: String,
}

fn load_rows(dir: &Path) -> Result<Vec<Row>, String> {
    let mut rows: Vec<Row> = fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|entry| fs::read_to_string(entry.ok()?.path().join("result.json")).ok())
        .map(|text| Row { case: string(&text, "", "case"), columns: number(&text, "", "grid_columns"), text })
        .collect();
    rows.sort_by(|a, b| a.case.cmp(&b.case).then(a.columns.total_cmp(&b.columns)));
    Ok(rows)
}

/// Iteration growth between consecutive refinements of the same case.
fn scaling(rows: &[Row]) -> Vec<Json> {
    rows.windows(2)
        .filter(|pair| pair[0].case == pair[1].case)
        .map(|pair| {
            let growth = number(&pair[1].text, "", "iterations") / number(&pair[0].text, "", "iterations");
            Json::new()
                .text("case", &pair[0].case)
                .number("from_grid_columns", pair[0].columns)
                .number("to_grid_columns", pair[1].columns)
                .number("iteration_growth", growth)
                .flag("within_limit", growth <= ITERATION_GROWTH_LIMIT)
        })
        .collect()
}

fn short(value: f64) -> String {
    if value.is_finite() { format!("{value:.2e}") } else { "null".to_string() }
}

fn print_row(row: &Row) {
    let t = &row.text;
    let checks = ["converged", "forcing_term_ok", "f32_iterations_ok", "faithful_to_cpu_f32_emulation", "deterministic", "speedup_ok"];
    let pass = checks.map(|k| if flag(t, "verdict", k) { "Y" } else { "n" });
    println!(
        "{:<8} n={:<4} N={:<7} it={:<6} res={:<8} err64={:<8} kappa={:<8} ms/it={:<8} speedup={:<8} conv/forcing/f32it/faithful/det/speed={}",
        row.case, row.columns, number(t, "", "unknowns"), number(t, "", "iterations"),
        short(number(t, "", "relative_residual")), short(number(t, "", "relative_error_vs_f64")),
        short(number(t, "", "condition_jacobi_preconditioned")), short(number(t, "", "ms_per_iteration")),
        short(number(t, "", "speedup_vs_cpu_f64_single_thread")), pass.join("/"),
    );
}

/// Collects every <dir>/*/result.json into one file and prints the table the kill criteria are read from.
pub fn run(dir: &Path, out: &Path) -> Result<(), String> {
    let rows = load_rows(dir)?;
    rows.iter().for_each(print_row);
    let growth = scaling(&rows);
    growth.iter().for_each(|g| println!("scaling {}", g.render()));
    let cases: Vec<String> = rows.iter().map(|row| row.text.clone()).collect();
    let rendered_growth: Vec<String> = growth.iter().map(Json::render).collect();
    let summary = Json::new()
        .text("schema", "e1-summary/1")
        .number("iteration_growth_limit", ITERATION_GROWTH_LIMIT)
        .raw("scaling", format!("[{}]", rendered_growth.join(",")))
        .raw("cases", format!("[{}]", cases.join(",")));
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    write_text(out, &summary.render())
}
