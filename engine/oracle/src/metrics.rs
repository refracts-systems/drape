use crate::bsr::Bsr;
use crate::io::Json;
use crate::linalg::{difference_norm, vector_norm};

pub struct SolutionMetrics {
    pub true_relative_residual: f64,
    pub relative_error: f64,
    pub relative_error_in_plane: f64,
    pub relative_error_out_of_plane: f64,
}

pub fn evaluate(matrix: &Bsr, rhs: &[f64], exact: &[f64], x: &[f64]) -> SolutionMetrics {
    let residual = difference_norm(&matrix.multiply(x), rhs) / vector_norm(rhs);
    SolutionMetrics {
        true_relative_residual: residual,
        relative_error: difference_norm(x, exact) / vector_norm(exact),
        relative_error_in_plane: component_error(exact, x, |d| d < 2),
        relative_error_out_of_plane: component_error(exact, x, |d| d == 2),
    }
}

fn component_error(exact: &[f64], x: &[f64], keep: impl Fn(usize) -> bool) -> f64 {
    let pick = |v: &[f64]| v.iter().enumerate().filter(|(i, _)| keep(i % 3)).map(|(_, &x)| x).collect::<Vec<f64>>();
    let (e, y) = (pick(exact), pick(x));
    let scale = vector_norm(&e);
    if scale == 0.0 { difference_norm(&y, &e) } else { difference_norm(&y, &e) / scale }
}

impl SolutionMetrics {
    pub fn json(&self) -> Json {
        Json::new()
            .number("true_relative_residual", self.true_relative_residual)
            .number("relative_error_vs_f64", self.relative_error)
            .number("relative_error_in_plane", self.relative_error_in_plane)
            .number("relative_error_out_of_plane", self.relative_error_out_of_plane)
    }
}

pub fn relative_difference(a: &[f64], b: &[f64]) -> f64 {
    let scale = vector_norm(b);
    if scale == 0.0 { difference_norm(a, b) } else { difference_norm(a, b) / scale }
}

pub fn bitwise_equal_fraction(a: &[f32], b: &[f32]) -> f64 {
    let equal = a.iter().zip(b).filter(|(x, y)| x.to_bits() == y.to_bits()).count();
    equal as f64 / a.len().max(1) as f64
}
