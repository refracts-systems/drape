/// Extreme eigenvalues of the preconditioned operator D⁻¹K estimated from PCG coefficients
/// (the CG–Lanczos tridiagonal), by Sturm-sequence bisection.
pub struct Spectrum {
    pub smallest: f64,
    pub largest: f64,
}

impl Spectrum {
    pub fn condition(&self) -> f64 {
        self.largest / self.smallest
    }
}

pub fn estimate(alphas: &[f64], betas: &[f64]) -> Option<Spectrum> {
    if alphas.len() < 2 {
        return None;
    }
    let (diagonal, offdiagonal) = tridiagonal(alphas, betas);
    let (low, high) = gershgorin(&diagonal, &offdiagonal);
    let smallest = bisect(&diagonal, &offdiagonal, low, high, 1);
    let largest = bisect(&diagonal, &offdiagonal, low, high, diagonal.len());
    Some(Spectrum { smallest, largest })
}

fn tridiagonal(alphas: &[f64], betas: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = alphas.len();
    let diagonal = (0..n)
        .map(|j| if j == 0 { 1.0 / alphas[0] } else { 1.0 / alphas[j] + betas[j - 1] / alphas[j - 1] })
        .collect();
    let offdiagonal = (0..n - 1).map(|j| betas[j].sqrt() / alphas[j]).collect();
    (diagonal, offdiagonal)
}

fn gershgorin(d: &[f64], e: &[f64]) -> (f64, f64) {
    let radius = |i: usize| {
        let left = if i > 0 { e[i - 1].abs() } else { 0.0 };
        let right = if i < e.len() { e[i].abs() } else { 0.0 };
        left + right
    };
    let low = (0..d.len()).map(|i| d[i] - radius(i)).fold(f64::INFINITY, f64::min);
    let high = (0..d.len()).map(|i| d[i] + radius(i)).fold(f64::NEG_INFINITY, f64::max);
    (low.min(0.0), high)
}

/// Number of eigenvalues strictly below lambda.
fn count_below(d: &[f64], e: &[f64], lambda: f64) -> usize {
    let mut count = 0;
    let mut q = d[0] - lambda;
    for i in 0..d.len() {
        if i > 0 {
            let guarded = if q == 0.0 { f64::MIN_POSITIVE } else { q };
            q = d[i] - lambda - e[i - 1] * e[i - 1] / guarded;
        }
        if q < 0.0 {
            count += 1;
        }
    }
    count
}

/// The k-th smallest eigenvalue (1-based).
fn bisect(d: &[f64], e: &[f64], mut low: f64, mut high: f64, k: usize) -> f64 {
    for _ in 0..200 {
        let middle = 0.5 * (low + high);
        if middle == low || middle == high {
            break;
        }
        if count_below(d, e, middle) >= k {
            high = middle;
        } else {
            low = middle;
        }
    }
    0.5 * (low + high)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases::{build, CaseKind};
    use crate::pcg::{solve, PcgSettings};

    #[test]
    fn spectrum_estimate_is_positive_and_ordered() {
        let problem = build(CaseKind::Tension, 11);
        let settings = PcgSettings { tolerance: 1e-12, max_iterations: 2000, history_stride: 100, replace_every: 0, replace_above: 0.0 };
        let report = solve::<f64>(&problem.matrix, &problem.rhs, &settings);
        let spectrum = estimate(&report.alphas, &report.betas).unwrap();
        assert!(spectrum.smallest > 0.0 && spectrum.largest > spectrum.smallest);
    }
}
