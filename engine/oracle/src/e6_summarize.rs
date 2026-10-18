use crate::criteria::ITERATION_GROWTH_LIMIT;
use crate::e4_compare::read_iterates;
use crate::e6_compare::step_statuses;
use crate::io::{write_text, Json};
use crate::jsonscan::{flag, number, string};
use std::fs;
use std::path::Path;

/// Every E5 accuracy gate (with the Part 0 rules) plus the three Experiment 6 checks; "-" marks a gate that does not apply.
const ACCURACY: [&str; 17] = [
    "wgsl_kernels_ok", "contact_kernel_ok", "fd_gradient_ok", "fd_hessian_ok", "contact_fd_ok", "hinge_energy_change_identity_ok",
    "barrier_energy_change_identity_ok", "intersection_free", "f64_converged_every_step", "one_step_ok", "f32_stops_ok_at_probes",
    "energy_f64_ok", "energy_f32_ok", "rules_replayed_ok", "acceptance_replay_ok", "reduction_accuracy_ok", "noise_safety_ok",
];
const RUNGS: [&str; 4] = ["A", "B", "C", "D"];
/// n65 ms per step of the final rung vs rung A, and n65 bytes read back per Newton iteration vs rung A.
const MS_FACTOR: f64 = 0.8;
const BYTES_FACTOR: f64 = 0.1;
/// Rung B is dropped when a probe's one-step error is worse than rung A's by more than this factor.
const CONSISTENCY_FACTOR: f64 = 2.0;
/// C or D change the stop reason of more than this fraction of the n33 steps relative to the preceding rung: reported.
const STOP_CHANGE_LIMIT: f64 = 0.05;

/// One scored run: a rung of one case.
struct Run {
    text: String,
    dir: String,
}

fn mark(text: &str, key: &str) -> &'static str {
    if !text.contains(&format!("\"{key}\"")) { "-" } else if flag(text, "verdict", key) { "Y" } else { "n" }
}

fn accuracy_ok(text: &str) -> bool {
    ACCURACY.iter().all(|k| mark(text, k) != "n")
}

fn print_row(run: &Run) {
    let t = &run.text;
    let marks: Vec<&str> = ACCURACY.iter().map(|k| mark(t, k)).collect();
    let phase = |name: &str| number(t, "phase_ms_per_step", name);
    println!(
        "{:<10} rung {} newton/step={:<6.2} pcg/solve={:<7.1} ms/step={:<8.1} [asm {:.1} ccd {:.1} reduce {:.1} pcg {:.1} ls {:.1} io {:.1} other {:.1}] KiB/iter={:.1} stops res/upd/lim/ls={}/{}/{}/{} noise-accepted={} accuracy={} newton_ok={}",
        string(t, "", "case"), string(t, "", "rung"), number(t, "robust", "newton_iterations_per_step"), number(t, "robust", "pcg_iterations_per_solve"), number(t, "robust", "ms_per_step"),
        phase("assembly"), phase("ccd"), phase("reduce"), phase("pcg"), phase("line_search"), phase("io"), phase("other"),
        number(t, "e6", "read_bytes_per_newton_iteration") / 1024.0,
        number(t, "stops", "residual_rule"), number(t, "stops", "update_rule"), number(t, "stops", "iteration_limit"), number(t, "stops", "line_search_failed"),
        number(t, "e6", "noise_accepted_trajectory_updates"), marks.join(""), mark(t, "newton_iterations_ok"),
    );
}

fn load(root: &Path) -> Result<Vec<Run>, String> {
    let mut runs = Vec::new();
    for case in fs::read_dir(root).map_err(|e| format!("{}: {e}", root.display()))?.filter_map(Result::ok) {
        for rung in RUNGS {
            let dir = case.path().join(format!("e6-rung-{rung}"));
            if let Ok(text) = fs::read_to_string(dir.join("result.json")) {
                runs.push(Run { text, dir: dir.display().to_string() });
            }
        }
    }
    runs.sort_by(|a, b| string(&a.text, "", "rung").cmp(&string(&b.text, "", "rung")).then(number(&a.text, "", "unknowns").total_cmp(&number(&b.text, "", "unknowns"))));
    Ok(runs)
}

fn of_rung<'a>(runs: &'a [Run], rung: &str) -> Vec<&'a Run> {
    runs.iter().filter(|r| string(&r.text, "", "rung") == rung).collect()
}

fn case_of<'a>(runs: &'a [Run], rung: &str, case: &str) -> Option<&'a Run> {
    runs.iter().find(|r| string(&r.text, "", "rung") == rung && string(&r.text, "", "case") == case)
}

/// The numbers of a JSON list that follows `"key":[` (null reads as NaN).
fn list(text: &str, key: &str) -> Vec<f64> {
    let Some(at) = text.find(&format!("\"{key}\":[")) else { return Vec::new() };
    let start = text[at..].find('[').map_or(0, |i| at + i + 1);
    let end = text[start..].find(']').map_or(text.len(), |i| start + i);
    text[start..end].split(',').filter(|s| !s.is_empty()).map(|s| s.trim().parse().unwrap_or(f64::NAN)).collect()
}

/// Rung B is dropped when it accepts a noise-only update that raises f64 Φ by more than twice its noise bound (its noise_safety
/// gate fails), or a probe's one-step consistency is worse than rung A's by more than 2x.
fn rung_b_dropped(runs: &[Run]) -> (bool, String) {
    let unsafe_update = of_rung(runs, "B").iter().any(|r| mark(&r.text, "noise_safety_ok") == "n");
    let worse = of_rung(runs, "B").iter().filter_map(|b| case_of(runs, "A", &string(&b.text, "", "case")).map(|a| (a, *b))).any(|(a, b)| {
        let (ea, eb) = (list(&a.text, "probe_one_step_errors"), list(&b.text, "probe_one_step_errors"));
        ea.iter().zip(&eb).any(|(x, y)| *y > CONSISTENCY_FACTOR * *x)
    });
    (unsafe_update || worse, format!("unsafe noise-only update: {unsafe_update}; consistency worse than A by more than 2x: {worse}"))
}

fn statuses(run: &Run) -> Result<Vec<f64>, String> {
    Ok(step_statuses(&read_iterates(Path::new(&run.dir), "")?))
}

/// Fraction of the n33 trajectory steps whose stop reason differs between two rungs.
fn stop_change(runs: &[Run], from: &str, to: &str) -> Result<Option<f64>, String> {
    let (Some(a), Some(b)) = (case_of(runs, from, "drape-n33"), case_of(runs, to, "drape-n33")) else { return Ok(None) };
    let (sa, sb) = (statuses(a)?, statuses(b)?);
    Ok(Some(sa.iter().zip(&sb).filter(|(x, y)| x != y).count() as f64 / sa.len().max(1) as f64))
}

fn growth(runs: &[Run], rung: &str) -> Vec<Json> {
    let rows = of_rung(runs, rung);
    rows.windows(2).map(|pair| {
        let ratio = number(&pair[1].text, "robust", "pcg_iterations_per_solve") / number(&pair[0].text, "robust", "pcg_iterations_per_solve");
        Json::new().text("rung", rung).text("from", &string(&pair[0].text, "", "case")).text("to", &string(&pair[1].text, "", "case"))
            .number("pcg_iteration_growth", ratio).flag("within_limit", ratio < ITERATION_GROWTH_LIMIT)
    }).collect()
}

/// Performance of one rung against rung A: n65 ms/step ratio, n65 bytes-per-iteration ratio, Newton gate, PCG growth.
fn performance(runs: &[Run], rung: &str) -> Json {
    let value = |r: &str, section: &str, key: &str| case_of(runs, r, "drape-n65").map_or(f64::NAN, |x| number(&x.text, section, key));
    let ms = value(rung, "robust", "ms_per_step") / value("A", "robust", "ms_per_step");
    let bytes = value(rung, "e6", "read_bytes_per_newton_iteration") / value("A", "e6", "read_bytes_per_newton_iteration");
    let newton = of_rung(runs, rung).iter().all(|r| mark(&r.text, "newton_iterations_ok") != "n");
    let pcg = growth(runs, rung).iter().all(|g| g.render().contains("\"within_limit\":true"));
    Json::new().text("rung", rung).number("n65_ms_ratio_vs_A", ms).flag("ms_ok", ms <= MS_FACTOR).number("n65_bytes_ratio_vs_A", bytes)
        .flag("bytes_ok", bytes <= BYTES_FACTOR).flag("newton_iterations_ok", newton).flag("pcg_growth_ok", pcg)
}

/// The rung chosen as final: the last whose accuracy gates pass in every case, among the rungs rung B's trigger left in play.
fn final_rung<'a>(runs: &[Run], dropped: bool) -> &'a str {
    let usable = RUNGS.iter().copied().filter(|r| !dropped || *r == "A");
    usable.filter(|r| !of_rung(runs, r).is_empty() && of_rung(runs, r).iter().all(|x| accuracy_ok(&x.text))).last().unwrap_or("none")
}

/// How the stop reason of the n33 steps changes along the ladder A→B→C→D.
fn stop_changes(runs: &[Run]) -> Vec<Json> {
    [("A", "B"), ("B", "C"), ("C", "D")].iter().filter_map(|(a, b)| stop_change(runs, a, b).ok().flatten().map(|f| (a, b, f)))
        .map(|(a, b, f)| Json::new().text("from", a).text("to", b).number("n33_stop_change_fraction", f).flag("over_limit", f > STOP_CHANGE_LIMIT)).collect()
}

fn render(values: &[Json]) -> String {
    format!("[{}]", values.iter().map(Json::render).collect::<Vec<_>>().join(","))
}

/// Collects drape5/*/e6-rung-*/result.json, prints the ablation table, the accuracy gates, the triggers and the performance
/// gates, and writes the Experiment 6 summary.
pub fn run(root: &Path, out: &Path) -> Result<(), String> {
    let runs = load(root)?;
    println!("accuracy gates: {}", ACCURACY.join("/"));
    runs.iter().for_each(print_row);
    let (dropped, why) = rung_b_dropped(&runs);
    let (final_rung, changes) = (final_rung(&runs, dropped), stop_changes(&runs));
    let perf: Vec<Json> = RUNGS.iter().filter(|r| !of_rung(&runs, r).is_empty()).map(|r| performance(&runs, r)).collect();
    println!("rung B dropped: {dropped} ({why})\nfinal rung: {final_rung}");
    changes.iter().for_each(|c| println!("stop change {}", c.render()));
    perf.iter().for_each(|p| println!("performance {}", p.render()));
    let growths: Vec<Json> = RUNGS.iter().flat_map(|r| growth(&runs, r)).collect();
    let summary = Json::new().text("schema", "e6-summary/1").text("final_rung", final_rung).flag("rung_b_dropped", dropped).text("rung_b_dropped_reason", &why)
        .raw("stop_changes", render(&changes)).raw("performance", render(&perf)).raw("pcg_growth", render(&growths))
        .raw("cases", format!("[{}]", runs.iter().map(|r| r.text.clone()).collect::<Vec<_>>().join(",")));
    write_text(out, &summary.render())
}
