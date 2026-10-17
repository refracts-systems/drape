use crate::cases::TIME_STEP;
use crate::drape::{Drape, DrapeStep, TangentKind};
use crate::e2_generate::render_list;
use crate::e3_compare::Reference;
use crate::e3_generate::{scheme_settings, E4_FINAL_SCHEME};
use crate::io::{write_text, Json};
use crate::jsonscan::number;
use crate::linalg::{vector_norm, Vec3};
use crate::newton::{solve_step_regularized, IterateTrace, NewtonSystem, RegularizedSystem, StepSolution};
use std::fs;
use std::path::Path;

/// The time step `step` of a stored f64 trajectory: x_{k−1}, v_{k−1} and the step's energy.
pub(crate) fn step_from<'a>(drape: &'a Drape, reference: &Reference, step: usize) -> DrapeStep<'a> {
    let x = &reference.positions[step - 1];
    let v: Vec<Vec3> = if step == 1 { vec![[0.0; 3]; x.len()] } else {
        x.iter().zip(&reference.positions[step - 2]).map(|(p, q)| std::array::from_fn(|d| (p[d] - q[d]) / TIME_STEP)).collect()
    };
    DrapeStep { drape, x_hat: drape.prediction(x, &v) }
}

fn trace_json(index: usize, trace: &IterateTrace, relative: f64) -> String {
    Json::new().integer("iteration", index).number("relative_residual", relative).number("mu", trace.mu).number("alpha", trace.step_length)
        .number("slope_gTd", trace.slope).number("delta_phi", trace.energy_change).number("update_inf", trace.update_inf)
        .integer("cholesky_retries", trace.retries).number("line_search_start", trace.max_step).render()
}

fn rerun_json(drape: &Drape, reference: &Reference, step: usize) -> Result<String, String> {
    let system = step_from(drape, reference, step);
    let scale = vector_norm(&system.residual(&reference.positions[step - 1]));
    let solution: StepSolution = solve_step_regularized(&system, system.x_hat.clone(), scale, &scheme_settings(&E4_FINAL_SCHEME), E4_FINAL_SCHEME.solver.floor_factor)?;
    let original = reference.residual_history(step);
    let reproduced = original.iter().zip(&solution.residuals).map(|(a, b)| (a - b).abs() / a).fold(0.0, f64::max);
    let rows: Vec<String> = solution.trace.iter().enumerate().map(|(k, t)| trace_json(k, t, solution.residuals[k])).collect();
    Ok(Json::new().integer("step", step).text("status", solution.status.name()).integer("newton_iterations", solution.searches.len())
        .number("final_relative_residual", *solution.residuals.last().unwrap())
        .number("final_floor", system.residual_floor(&solution.positions) / scale)
        .number("first_30_reproduce_e4_relative", reproduced)
        .raw("iterations", format!("[{}]", rows.join(","))).render())
}

/// Final residual and measured f64 floor (both relative) at the stored final state of every unconverged step.
fn floors_json(drape: &Drape, reference: &Reference) -> String {
    let rows: Vec<String> = (1..reference.converged.len() + 1).filter(|&s| reference.converged[s - 1] != 1.0).map(|s| {
        let system = step_from(drape, reference, s);
        let scale = vector_norm(&system.residual(&reference.positions[s - 1]));
        let floor = system.residual_floor(&reference.positions[s]) / scale;
        Json::new().integer("step", s).number("final_relative_residual", reference.final_residual[s - 1]).number("floor", floor)
            .number("residual_over_floor", reference.final_residual[s - 1] / floor).render()
    }).collect();
    format!("[{}]", rows.join(","))
}

/// Re-solves `step` of an E4 reference with the E4-final scheme, tracing every update, and evaluates the f64 floor at the
/// stored final state of every unconverged step.
pub fn run(dir: &Path, step: usize, out: &Path) -> Result<(), String> {
    let problem = fs::read_to_string(dir.join("problem.json")).map_err(|e| e.to_string())?;
    let drape = Drape::build(number(&problem, "", "grid_columns") as usize, TangentKind::Exact);
    let reference = Reference::load(dir, drape.mesh.rest.len())?;
    let json = Json::new().text("schema", "e4-diagnosis/1").text("case", &dir.display().to_string())
        .raw("original_residuals", render_list(reference.residual_history(step)))
        .raw("rerun", rerun_json(&drape, &reference, step)?).raw("unconverged_floors", floors_json(&drape, &reference)).render();
    write_text(out, &json)?;
    println!("{json}");
    Ok(())
}
