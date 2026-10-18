use crate::io::{write_text, Json};
use crate::jsonscan::{flag, number, string};
use std::fs;
use std::path::Path;

/// Verdict flags that gate accuracy at n17 and n33. `contact_kernel_ok` (reported, not re-gated) and `start_feasible` (a finding)
/// are listed in the table but do not gate.
const GATES: [&str; 9] = [
    "wgsl_kernels_ok", "inertia_step_sizes_ok", "intersection_free", "rules_replayed_ok", "static_energy_ok", "oracle_converged",
    "final_state_residual_ok", "one_step_ok", "f32_stops_ok_at_probes",
];
const REPORTED: [&str; 2] = ["contact_kernel_ok", "start_feasible"];
const EDITS: [&str; 3] = ["w1", "w2", "w3"];
/// Warm / cold Newton iterations per edit; two-tier total ms over fine cold ms; cold seconds at n65 and n129; PCG growth per doubling.
const WARM_LIMIT: f64 = 0.25;
const WARM_REPORT_LIMIT: f64 = 0.5;
const TWO_TIER_LIMIT: f64 = 0.5;
const COLD_SECONDS: [(usize, f64); 2] = [(65, 10.0), (129, 30.0)];
const PCG_GROWTH_LIMIT: f64 = 3.0;
/// A cold f32 solve that stops with r/L above this multiple of the representation floor (n17, n33) or above 1 (n65, n129) is a finding.
const FLOOR_MULTIPLE: f64 = 10.0;

/// One scored GPU run of a case: size, edit, mode and the text of its result.json.
struct Run {
    text: String,
    n: usize,
    edit: String,
    mode: String,
}

impl Run {
    fn gpu(&self, key: &str) -> f64 {
        number(&self.text, "gpu", key)
    }

    fn seconds(&self) -> f64 {
        self.gpu("ms") / 1000.0
    }

    fn rel(&self) -> f64 {
        self.gpu("final_relative_residual_gpu")
    }

    fn floor(&self) -> f64 {
        number(&self.text, "oracle_cold", "representation_floor")
    }

    fn mark(&self, key: &str) -> &'static str {
        if !self.text.contains(&format!("\"{key}\"")) { "-" } else if flag(&self.text, "verdict", key) { "Y" } else { "n" }
    }

    fn gates_ok(&self) -> bool {
        GATES.iter().all(|k| self.mark(k) != "n")
    }

    fn marks(&self) -> String {
        GATES.iter().chain(REPORTED.iter()).map(|k| self.mark(k)).collect()
    }
}

fn load(root: &Path) -> Result<Vec<Run>, String> {
    let mut runs = Vec::new();
    for case in fs::read_dir(root).map_err(|e| format!("{}: {e}", root.display()))?.filter_map(Result::ok) {
        for edit in ["base", "w1", "w2", "w3"] {
            for mode in ["cold", "warm", "prolong"] {
                let Ok(text) = fs::read_to_string(case.path().join(edit).join(format!("gpu-{mode}")).join("result.json")) else { continue };
                runs.push(Run { n: number(&text, "gpu", "grid_columns") as usize, edit: edit.to_string(), mode: mode.to_string(), text });
            }
        }
    }
    runs.sort_by(|a, b| a.n.cmp(&b.n).then(a.edit.cmp(&b.edit)).then(a.mode.cmp(&b.mode)));
    Ok(runs)
}

fn find<'a>(runs: &'a [Run], n: usize, edit: &str, mode: &str) -> Option<&'a Run> {
    runs.iter().find(|r| r.n == n && r.edit == edit && r.mode == mode)
}

fn sizes(runs: &[Run]) -> Vec<usize> {
    let mut all: Vec<usize> = runs.iter().map(|r| r.n).collect();
    all.dedup();
    all
}

fn print_row(r: &Run) {
    let phase = |name: &str| number(&r.text, "phase_ms_total", name) / 1000.0;
    println!(
        "n{:<4} {:<5} {:<8} steps={:<4} rej={:<3} newton={:<5} pcg={:<7} {:>7.2} s [asm {:.1} pcg {:.1} ls {:.1} ccd {:.1} reduce {:.1} io {:.1} other {:.1}] read={:.1} MiB stop={:<11} r/L={:.2e} floor={:.2e} h={:.3e} min d/dhat={:.3e} drops={} gates={}",
        r.n, r.edit, r.mode, r.gpu("accepted_steps"), r.gpu("rejected_attempts"), r.gpu("newton_iterations"), r.gpu("pcg_iterations"), r.seconds(),
        phase("assembly"), phase("pcg"), phase("line_search"), phase("ccd"), phase("reduce"), phase("io"), phase("other"), r.gpu("read_bytes") / 1048576.0,
        string(&r.text, "gpu", "stop"), r.rel(), r.floor(), r.gpu("final_h"), r.gpu("min_distance_over_dhat"), r.gpu("curvature_dropped_steps"), r.marks(),
    );
}

/// Warm over cold for one edit at one size: Newton iterations, PCG iterations and milliseconds.
fn warm_json(runs: &[Run], n: usize, edit: &str) -> Option<Json> {
    let (cold, warm) = (find(runs, n, edit, "cold")?, find(runs, n, edit, "warm")?);
    let ratio = warm.gpu("newton_iterations") / cold.gpu("newton_iterations");
    Some(Json::new().integer("n", n).text("edit", edit).number("cold_newton", cold.gpu("newton_iterations")).number("warm_newton", warm.gpu("newton_iterations"))
        .number("newton_ratio", ratio).number("ms_ratio", warm.gpu("ms") / cold.gpu("ms")).number("pcg_ratio", warm.gpu("pcg_iterations") / cold.gpu("pcg_iterations"))
        .text("cold_stop", &string(&cold.text, "gpu", "stop")).text("warm_stop", &string(&warm.text, "gpu", "stop"))
        .flag("oracle_cold_converged", flag(&cold.text, "oracle", "converged")).number("warm_start_residual_over_load", number(&warm.text, "", "start_residual_over_load"))
        .flag("within_gate", ratio <= WARM_LIMIT).flag("within_report_limit", ratio <= WARM_REPORT_LIMIT))
}

/// Two-tier at size n: coarse cold + prolonged solve against the fine cold solve of the same case.
fn tier_json(runs: &[Run], coarse: usize, n: usize) -> Option<Json> {
    let (c, p, f) = (find(runs, coarse, "base", "cold")?, find(runs, n, "base", "prolong")?, find(runs, n, "base", "cold")?);
    let total_ms = c.gpu("ms") + p.gpu("ms");
    let ratio = total_ms / f.gpu("ms");
    Some(Json::new().integer("coarse_n", coarse).integer("n", n).number("fine_cold_ms", f.gpu("ms")).number("coarse_cold_ms", c.gpu("ms")).number("prolong_ms", p.gpu("ms"))
        .number("total_ms_ratio", ratio).number("total_newton", c.gpu("newton_iterations") + p.gpu("newton_iterations")).number("fine_cold_newton", f.gpu("newton_iterations"))
        .number("prolong_newton_ratio", p.gpu("newton_iterations") / f.gpu("newton_iterations")).number("start_residual_over_load", number(&p.text, "", "start_residual_over_load"))
        .number("start_min_distance_over_dhat", number(&p.text, "", "start_min_distance_over_dhat")).flag("start_feasible", number(&p.text, "", "start_min_distance_over_dhat") > 0.0)
        .text("prolong_stop", &string(&p.text, "gpu", "stop")).number("prolong_final_relative_residual", p.rel()).flag("within_gate", ratio <= TWO_TIER_LIMIT))
}

/// PCG iterations per Newton iteration of the base cold solves: growth per doubling of the mesh.
fn pcg_growth(runs: &[Run]) -> Vec<Json> {
    let per = |r: &Run| r.gpu("pcg_iterations") / r.gpu("newton_iterations");
    let colds: Vec<&Run> = runs.iter().filter(|r| r.edit == "base" && r.mode == "cold").collect();
    colds.windows(2).map(|w| Json::new().integer("from", w[0].n).integer("to", w[1].n).number("pcg_per_newton_from", per(w[0])).number("pcg_per_newton_to", per(w[1]))
        .number("growth", per(w[1]) / per(w[0])).flag("within_limit", per(w[1]) / per(w[0]) < PCG_GROWTH_LIMIT)).collect()
}

/// Cold f32 solves that ended by cap, stalled or stagnation far above the representation floor (n17, n33) or above r/L = 1 (n65, n129).
fn floor_findings(runs: &[Run]) -> Vec<Json> {
    runs.iter().filter(|r| r.edit == "base" && r.mode == "cold" && string(&r.text, "gpu", "stop") != "tolerance").filter(|r| {
        if r.n <= 33 { r.rel() > FLOOR_MULTIPLE * r.floor() } else { r.rel() > 1.0 }
    }).map(|r| Json::new().integer("n", r.n).text("stop", &string(&r.text, "gpu", "stop")).number("final_relative_residual", r.rel()).number("representation_floor", r.floor())).collect()
}

fn seconds_json(runs: &[Run], n: usize) -> Json {
    let sec = |edit: &str, mode: &str| find(runs, n, edit, mode).map_or(f64::NAN, Run::seconds);
    let stop = |edit: &str, mode: &str| find(runs, n, edit, mode).map_or(String::from("-"), |r| string(&r.text, "gpu", "stop"));
    let warm: Vec<f64> = EDITS.iter().map(|e| sec(e, "warm")).filter(|s| s.is_finite()).collect();
    let worst = warm.iter().copied().fold(f64::NAN, f64::max);
    let coarse = sizes(runs).into_iter().filter(|m| *m < n).last().and_then(|m| find(runs, m, "base", "cold")).map_or(f64::NAN, Run::seconds);
    Json::new().integer("n", n).number("cold_s", sec("base", "cold")).text("cold_stop", &stop("base", "cold")).number("after_edit_worst_s", worst)
        .text("after_edit_stops", &EDITS.iter().map(|e| stop(e, "warm")).collect::<Vec<_>>().join("/")).number("coarse_to_fine_s", coarse + sec("base", "prolong"))
        .number("prolong_only_s", sec("base", "prolong")).text("prolong_stop", &stop("base", "prolong"))
}

fn render(values: &[Json]) -> String {
    format!("[{}]", values.iter().map(Json::render).collect::<Vec<_>>().join(","))
}

fn accuracy(runs: &[Run]) -> Json {
    let sized: Vec<&Run> = runs.iter().filter(|r| r.n <= 33).collect();
    let failing: Vec<String> = sized.iter().filter(|r| !r.gates_ok()).map(|r| format!("n{} {} {}", r.n, r.edit, r.mode)).collect();
    Json::new().flag("all_pass", failing.is_empty()).text("failing", &failing.join("; "))
}

/// Collects drape7/*/<edit>/gpu-<mode>/result.json, prints the Part 1-3 tables, the gates and the triggers, writes the summary.
pub fn run(root: &Path, out: &Path) -> Result<(), String> {
    let runs = load(root)?;
    println!("gates: {} | reported: {}", GATES.join("/"), REPORTED.join("/"));
    runs.iter().for_each(print_row);
    let ns = sizes(&runs);
    let warm: Vec<Json> = ns.iter().flat_map(|n| EDITS.iter().filter_map(|e| warm_json(&runs, *n, e)).collect::<Vec<_>>()).collect();
    let tiers: Vec<Json> = ns.windows(2).filter_map(|w| tier_json(&runs, w[0], w[1])).collect();
    let growth = pcg_growth(&runs);
    let findings = floor_findings(&runs);
    let editor: Vec<Json> = ns.iter().map(|n| seconds_json(&runs, *n)).collect();
    let cold_seconds: Vec<Json> = COLD_SECONDS.iter().filter_map(|(n, limit)| find(&runs, *n, "base", "cold").map(|r| Json::new().integer("n", *n).number("seconds", r.seconds()).number("target_seconds", *limit).flag("within_target", r.seconds() <= *limit))).collect();
    let acc = accuracy(&runs);
    for (title, items) in [("warm", &warm), ("two-tier", &tiers), ("pcg growth", &growth), ("floor findings", &findings), ("editor", &editor), ("cold vs target", &cold_seconds)] {
        items.iter().for_each(|j| println!("{title}: {}", j.render()));
    }
    println!("accuracy at n17 and n33: {}", acc.render());
    let summary = Json::new().text("schema", "e7-summary/1").raw("accuracy", acc.render()).raw("warm", render(&warm)).raw("two_tier", render(&tiers)).raw("pcg_growth", render(&growth))
        .raw("floor_findings", render(&findings)).raw("editor", render(&editor)).raw("cold_seconds", render(&cold_seconds))
        .raw("cases", format!("[{}]", runs.iter().map(|r| r.text.clone()).collect::<Vec<_>>().join(",")));
    write_text(out, &summary.render())
}
