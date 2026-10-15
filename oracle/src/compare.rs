use crate::bsr::Bsr;
use crate::criteria::{verdict, Verdict};
use crate::io::{locate, read_f32_file, read_f64, read_u32, write_text, Json};
use crate::jsonscan::{flag, number, string};
use crate::metrics::{bitwise_equal_fraction, evaluate, relative_difference, SolutionMetrics};
use std::fs;
use std::path::Path;

/// Compulsory memory traffic of one PCG iteration: BSR values + column indices, plus every vector touched.
const BYTES_PER_BLOCK: f64 = 40.0;
const BYTES_PER_ROW: f64 = 196.0;

pub struct CaseFiles {
    problem: String,
    pub reference: String,
    pub gpu: String,
    matrix: Bsr,
    rhs: Vec<f64>,
    exact: Vec<f64>,
    exact_rounded_system: Vec<f64>,
    cpu_f32: Vec<f32>,
}

impl CaseFiles {
    fn load(dir: &Path) -> Result<CaseFiles, String> {
        let text = |name: &str| fs::read_to_string(dir.join(name)).map_err(|e| format!("{name}: {e}"));
        let reference = text("reference.json")?;
        let bytes = fs::read(dir.join("reference.bin")).map_err(|e| e.to_string())?;
        let f64s = |name: &str| locate(&reference, name).map(|(offset, length)| read_f64(&bytes, offset, length));
        let u32s = |name: &str| locate(&reference, name).map(|(offset, length)| read_u32(&bytes, offset, length));
        let indices = |name: &str| u32s(name).map(|v| v.into_iter().map(|i| i as usize).collect());
        Ok(CaseFiles {
            matrix: Bsr { row_ptr: indices("row_ptr")?, col_idx: indices("col_idx")?, vals: f64s("vals")? },
            rhs: f64s("rhs")?,
            exact: f64s("x_exact")?,
            exact_rounded_system: f64s("x_exact_rounded_system")?,
            cpu_f32: f64s("x_pcg_f32_cpu")?.iter().map(|&v| v as f32).collect(),
            problem: text("problem.json")?,
            gpu: text("gpu-result.json")?,
            reference,
        })
    }
}

pub struct Scores {
    pub metrics: SolutionMetrics,
    pub error_vs_f32_system: f64,
    pub difference_vs_cpu_f32: f64,
    pub bitwise_vs_cpu_f32: f64,
}

fn score(case: &CaseFiles, gpu32: &[f32]) -> Scores {
    let gpu: Vec<f64> = gpu32.iter().map(|&v| v as f64).collect();
    let cpu: Vec<f64> = case.cpu_f32.iter().map(|&v| v as f64).collect();
    Scores {
        metrics: evaluate(&case.matrix, &case.rhs, &case.exact, &gpu),
        error_vs_f32_system: relative_difference(&gpu, &case.exact_rounded_system),
        difference_vs_cpu_f32: relative_difference(&gpu, &cpu),
        bitwise_vs_cpu_f32: bitwise_equal_fraction(gpu32, &case.cpu_f32),
    }
}

fn effective_bandwidth_gb_s(case: &CaseFiles) -> f64 {
    let bytes = BYTES_PER_BLOCK * number(&case.problem, "", "blocks") + BYTES_PER_ROW * number(&case.problem, "", "rows");
    let gpu_ms = number(&case.gpu, "timing", "gpu_ms");
    let ms = if gpu_ms.is_finite() { gpu_ms } else { number(&case.gpu, "timing", "warm_solve_ms") };
    bytes * number(&case.gpu, "", "iterations") / (ms * 1e6)
}

pub fn speedup(case: &CaseFiles) -> f64 {
    1e3 * number(&case.reference, "pcg_f64_cpu", "seconds") / number(&case.gpu, "timing", "warm_solve_ms")
}

fn identity(case: &CaseFiles) -> Json {
    let (p, g) = (&case.problem, &case.gpu);
    Json::new()
        .text("schema", "e1-result/1")
        .text("case", &string(p, "", "case"))
        .integer("grid_columns", number(p, "", "grid_columns") as usize)
        .number("element_size_m", number(p, "", "element_size_m"))
        .integer("unknowns", number(p, "", "unknowns") as usize)
        .text("backend", &string(g, "", "backend"))
        .text("adapter", &string(g, "", "adapter"))
        .text("status", &string(g, "", "status"))
        .integer("iterations", number(g, "", "iterations") as usize)
        .integer("iterations_cpu_f32", number(&case.reference, "pcg_f32_cpu", "iterations") as usize)
        .integer("iterations_cpu_f64", number(&case.reference, "pcg_f64_cpu", "iterations") as usize)
}

fn accuracy(json: Json, case: &CaseFiles, scores: &Scores) -> Json {
    let (r, m) = (&case.reference, &scores.metrics);
    json.number("relative_residual", m.true_relative_residual)
        .number("recursive_relative_residual", number(&case.gpu, "", "recursive_relative_residual"))
        .number("gpu_f32_true_relative_residual", number(&case.gpu, "", "gpu_f32_true_relative_residual"))
        .number("residual_floor_f32_rounded_exact", number(r, "", "residual_floor_f32_rounded_exact"))
        .number("relative_error_vs_f64", m.relative_error)
        .number("relative_error_in_plane", m.relative_error_in_plane)
        .number("relative_error_out_of_plane", m.relative_error_out_of_plane)
        .number("relative_error_vs_f32_system", scores.error_vs_f32_system)
        .number("storage_error_f32_system", number(r, "", "storage_error_f32_system"))
        .number("condition_jacobi_preconditioned", number(r, "spectrum_jacobi_preconditioned", "condition"))
        .number("relative_difference_vs_cpu_f32_pcg", scores.difference_vs_cpu_f32)
        .number("bitwise_equal_fraction_vs_cpu_f32_pcg", scores.bitwise_vs_cpu_f32)
        .flag("deterministic_repeat", flag(&case.gpu, "", "deterministic_repeat"))
}

fn timing(json: Json, case: &CaseFiles) -> Json {
    let g = &case.gpu;
    json.number("gpu_time_ms", number(g, "timing", "gpu_ms"))
        .text("gpu_time_source", &string(g, "timing", "gpu_ms_source"))
        .number("solve_time_ms", number(g, "timing", "warm_solve_ms"))
        .number("total_time_ms", number(g, "timing", "total_ms"))
        .number("ms_per_iteration", number(g, "timing", "warm_ms_per_iteration"))
        .number("effective_bandwidth_gb_s", effective_bandwidth_gb_s(case))
        .number("cpu_f64_pcg_single_thread_ms", 1e3 * number(&case.reference, "pcg_f64_cpu", "seconds"))
        .number("cpu_f64_band_cholesky_ms", 1e3 * number(&case.reference, "cholesky_f64", "seconds"))
        .number("speedup_vs_cpu_f64_single_thread", speedup(case))
}

/// Scores gpu-x.bin against the f64 reference and writes result.json (schema e1-result/1).
pub fn run(dir: &Path) -> Result<(), String> {
    let case = CaseFiles::load(dir)?;
    let gpu32 = read_f32_file(&dir.join("gpu-x.bin"))?;
    if gpu32.len() != case.rhs.len() {
        return Err(format!("gpu-x.bin has {} values, expected {}", gpu32.len(), case.rhs.len()));
    }
    let scores = score(&case, &gpu32);
    let checks: Verdict = verdict(&case, &scores);
    let json = timing(accuracy(identity(&case), &case, &scores), &case).raw("verdict", checks.json().render());
    write_text(&dir.join("result.json"), &json.render())?;
    println!("{}", json.render());
    Ok(())
}
