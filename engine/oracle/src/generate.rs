use crate::band::BandCholesky;
use crate::cases::{build, CaseKind, Problem, SHEET_SIZE, TIME_STEP};
use crate::io::{render_history, render_number, write_text, BinaryArrays, Json};
use crate::lanczos;
use crate::metrics::{evaluate, relative_difference};
use crate::pcg::{solve, PcgReport, PcgSettings};
use std::fs;
use std::path::Path;
use std::time::Instant;

pub struct GenerateOptions {
    pub kind: CaseKind,
    pub columns: usize,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub replace_every: usize,
    pub replace_above: f64,
}

pub fn run(options: &GenerateOptions, dir: &Path) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let problem = build(options.kind, options.columns);
    check_symmetry(&problem)?;
    let problem_arrays = problem_arrays(&problem);
    problem_arrays.write(&dir.join("problem.bin"))?;
    write_text(&dir.join("problem.json"), &problem_json(&problem, options, &problem_arrays.manifest()))?;
    let reference = Reference::compute(&problem, options)?;
    let reference_arrays = reference.arrays(&problem);
    reference_arrays.write(&dir.join("reference.bin"))?;
    write_text(&dir.join("reference.json"), &reference.json(&problem, &reference_arrays.manifest()))?;
    println!("{}", reference.summary(&problem));
    Ok(())
}

fn check_symmetry(problem: &Problem) -> Result<(), String> {
    let asymmetry = problem.matrix.asymmetry();
    if asymmetry > 1e-12 {
        return Err(format!("assembled matrix is not symmetric (relative asymmetry {asymmetry:e})"));
    }
    Ok(())
}

fn problem_arrays(problem: &Problem) -> BinaryArrays {
    let m = &problem.matrix;
    let mut arrays = BinaryArrays::new();
    arrays.push_u32("row_ptr", m.row_ptr.iter().map(|&v| v as u32));
    arrays.push_u32("col_idx", m.col_idx.iter().map(|&v| v as u32));
    arrays.push_u32("diag_idx", m.diagonal_positions().into_iter().map(|v| v as u32));
    arrays.push_f32("vals", m.vals.iter().map(|&v| v as f32));
    arrays.push_f32("rhs", problem.rhs.iter().map(|&v| v as f32));
    arrays
}

fn problem_json(problem: &Problem, options: &GenerateOptions, manifest: &str) -> String {
    let (wale, course) = problem.kind.stretch();
    let m = &problem.material;
    let material = Json::new()
        .number("wale_N_per_m", m.wale)
        .number("course_N_per_m", m.course)
        .number("coupling_N_per_m", m.coupling)
        .number("shear_N_per_m", m.shear)
        .number("bend_wale_N_m", m.bend_wale)
        .number("bend_course_N_m", m.bend_course)
        .number("density_kg_per_m2", m.density);
    Json::new()
        .text("schema", "e1-problem/1")
        .text("case", problem.kind.name())
        .text("description", problem.kind.description())
        .integer("grid_columns", options.columns)
        .number("sheet_size_m", SHEET_SIZE)
        .number("element_size_m", SHEET_SIZE / (options.columns - 1) as f64)
        .integer("rows", problem.matrix.rows())
        .integer("unknowns", 3 * problem.matrix.rows())
        .integer("blocks", problem.matrix.blocks())
        .integer("triangles", problem.mesh.triangles.len())
        .integer("hinges", problem.mesh.hinges.len())
        .number("stretch_wale", wale)
        .number("stretch_course", course)
        .number("time_step_s", if problem.kind == CaseKind::Dynamic { TIME_STEP } else { 0.0 })
        .integer("replace_every", options.replace_every)
        .number("replace_above", options.replace_above)
        .raw("material", material.render())
        .raw("arrays", manifest.to_string())
        .render()
}

struct Reference {
    exact: Vec<f64>,
    exact_of_rounded_system: Vec<f64>,
    cholesky_seconds: f64,
    bandwidth: usize,
    pcg64: PcgReport,
    pcg32: PcgReport,
}

impl Reference {
    fn compute(problem: &Problem, options: &GenerateOptions) -> Result<Reference, String> {
        let start = Instant::now();
        let factor = BandCholesky::factor(&problem.matrix)?;
        let exact = factor.solve(&problem.rhs);
        let cholesky_seconds = start.elapsed().as_secs_f64();
        let rounded_rhs: Vec<f64> = problem.rhs.iter().map(|&v| v as f32 as f64).collect();
        let exact_of_rounded_system = BandCholesky::factor(&problem.matrix.rounded_to_f32())?.solve(&rounded_rhs);
        let plain = PcgSettings { tolerance: options.tolerance, max_iterations: options.max_iterations, history_stride: 16, replace_every: 0, replace_above: 0.0 };
        let pcg64 = solve::<f64>(&problem.matrix, &problem.rhs, &plain);
        let pcg32 = solve::<f32>(&problem.matrix, &problem.rhs, &PcgSettings { replace_every: options.replace_every, replace_above: options.replace_above, ..plain });
        Ok(Reference { exact, exact_of_rounded_system, cholesky_seconds, bandwidth: factor.bandwidth(), pcg64, pcg32 })
    }

    fn arrays(&self, problem: &Problem) -> BinaryArrays {
        let m = &problem.matrix;
        let mut arrays = BinaryArrays::new();
        arrays.push_u32("row_ptr", m.row_ptr.iter().map(|&v| v as u32));
        arrays.push_u32("col_idx", m.col_idx.iter().map(|&v| v as u32));
        arrays.push_f64("vals", &m.vals);
        arrays.push_f64("rhs", &problem.rhs);
        arrays.push_f64("x_exact", &self.exact);
        arrays.push_f64("x_exact_rounded_system", &self.exact_of_rounded_system);
        arrays.push_f64("x_pcg_f32_cpu", &self.pcg32.solution);
        arrays
    }

    fn json(&self, problem: &Problem, manifest: &str) -> String {
        let cholesky_residual = evaluate(&problem.matrix, &problem.rhs, &self.exact, &self.exact).true_relative_residual;
        let spectrum = lanczos::estimate(&self.pcg64.alphas, &self.pcg64.betas);
        Json::new()
            .text("schema", "e1-reference/1")
            .text("case", problem.kind.name())
            .raw("cholesky_f64", self.cholesky_json(cholesky_residual))
            .number("storage_error_f32_system", relative_difference(&self.exact_of_rounded_system, &self.exact))
            .number("residual_floor_f32_rounded_exact", self.rounded_exact_residual(problem))
            .raw("pcg_f64_cpu", pcg_json(&self.pcg64, problem, &self.exact))
            .raw("pcg_f32_cpu", pcg_json(&self.pcg32, problem, &self.exact))
            .raw("spectrum_jacobi_preconditioned", spectrum_json(spectrum.as_ref(), self.pcg64.iterations))
            .raw("arrays", manifest.to_string())
            .render()
    }

    /// True residual of the exact solution merely rounded to f32: no f32 solver can beat this.
    fn rounded_exact_residual(&self, problem: &Problem) -> f64 {
        let rounded: Vec<f64> = self.exact.iter().map(|&v| v as f32 as f64).collect();
        evaluate(&problem.matrix, &problem.rhs, &self.exact, &rounded).true_relative_residual
    }

    fn cholesky_json(&self, residual: f64) -> String {
        Json::new()
            .number("seconds", self.cholesky_seconds)
            .integer("bandwidth", self.bandwidth)
            .number("true_relative_residual", residual)
            .render()
    }

    fn summary(&self, problem: &Problem) -> String {
        let m64 = evaluate(&problem.matrix, &problem.rhs, &self.exact, &self.pcg64.solution);
        let m32 = evaluate(&problem.matrix, &problem.rhs, &self.exact, &self.pcg32.solution);
        format!(
            "{:<8} unknowns {:>7} | cholesky {:.2}s | pcg f64 it {:>6} res {:.2e} | pcg f32 it {:>6} res {:.2e} err {:.2e}",
            problem.kind.name(),
            3 * problem.matrix.rows(),
            self.cholesky_seconds,
            self.pcg64.iterations,
            m64.true_relative_residual,
            self.pcg32.iterations,
            m32.true_relative_residual,
            m32.relative_error
        )
    }
}

fn pcg_json(report: &PcgReport, problem: &Problem, exact: &[f64]) -> String {
    let metrics = evaluate(&problem.matrix, &problem.rhs, exact, &report.solution);
    metrics
        .json()
        .integer("iterations", report.iterations)
        .flag("converged", report.converged)
        .number("recursive_relative_residual", report.recursive_relative_residual)
        .number("seconds", report.seconds)
        .raw("history", render_history(&report.history))
        .render()
}

fn spectrum_json(spectrum: Option<&lanczos::Spectrum>, iterations: usize) -> String {
    match spectrum {
        Some(s) => Json::new()
            .number("smallest", s.smallest)
            .number("largest", s.largest)
            .number("condition", s.condition())
            .integer("lanczos_steps", iterations)
            .render(),
        None => format!("{{\"condition\":{}}}", render_number(f64::NAN)),
    }
}
