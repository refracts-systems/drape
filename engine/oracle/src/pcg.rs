use crate::bsr::Bsr;
use std::ops::{Add, Div, Mul, Sub};
use std::time::Instant;

pub const WORKGROUP: usize = 256;

pub trait Real: Copy + PartialOrd + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> {
    fn zero() -> Self;
    fn of(v: f64) -> Self;
    fn wide(self) -> f64;
}

impl Real for f32 {
    fn zero() -> f32 {
        0.0
    }
    fn of(v: f64) -> f32 {
        v as f32
    }
    fn wide(self) -> f64 {
        self as f64
    }
}

impl Real for f64 {
    fn zero() -> f64 {
        0.0
    }
    fn of(v: f64) -> f64 {
        v
    }
    fn wide(self) -> f64 {
        self
    }
}

pub struct PcgSettings {
    pub tolerance: f64,
    pub max_iterations: usize,
    pub history_stride: usize,
    /// Residual replacement r = b − Ax every this many iterations (0 = never), as residual.wgsl in REPLACE mode.
    pub replace_every: usize,
    /// Replacement only while the recursive relative residual is above this (keeps the f32 rounding floor from resetting r forever).
    pub replace_above: f64,
}

pub struct PcgReport {
    pub iterations: usize,
    pub converged: bool,
    pub recursive_relative_residual: f64,
    pub solution: Vec<f64>,
    pub alphas: Vec<f64>,
    pub betas: Vec<f64>,
    pub history: Vec<(usize, f64)>,
    pub seconds: f64,
}

/// Block-Jacobi PCG that reproduces the WGSL kernels' arithmetic order:
/// per-row 3-term dots, 256-wide workgroup trees, and a strided single-workgroup final reduction.
pub fn solve<T: Real>(matrix: &Bsr, rhs: &[f64], settings: &PcgSettings) -> PcgReport {
    let start = Instant::now();
    let system = System::<T>::new(matrix, rhs);
    let mut state = State::<T>::initialize(&system);
    let mut report = PcgReport::empty();
    let tolerance2 = T::of(settings.tolerance * settings.tolerance);
    let mut converged = state.rr <= T::zero();
    while !converged && report.iterations < settings.max_iterations {
        let (alpha, beta) = state.step(&system);
        report.record(alpha, beta, state.relative(), settings.history_stride);
        converged = state.rr <= tolerance2 * state.rr0;
        if !converged {
            state.update_direction(beta);
        }
        if !converged && replaces(settings, report.iterations, &state) {
            state.replace_residual(&system);
        }
    }
    report.finish(converged, state.relative(), &state.x, start.elapsed().as_secs_f64());
    report
}

fn replaces<T: Real>(settings: &PcgSettings, iterations: usize, state: &State<T>) -> bool {
    let due = settings.replace_every > 0 && iterations % settings.replace_every == 0;
    due && state.rr > T::of(settings.replace_above * settings.replace_above) * state.rr0
}

impl PcgReport {
    fn empty() -> PcgReport {
        PcgReport {
            iterations: 0,
            converged: false,
            recursive_relative_residual: 1.0,
            solution: Vec::new(),
            alphas: Vec::new(),
            betas: Vec::new(),
            history: Vec::new(),
            seconds: 0.0,
        }
    }

    fn record(&mut self, alpha: f64, beta: f64, relative: f64, stride: usize) {
        self.iterations += 1;
        self.alphas.push(alpha);
        self.betas.push(beta);
        if self.iterations % stride == 0 {
            self.history.push((self.iterations, relative));
        }
    }

    fn finish<T: Real>(&mut self, converged: bool, relative: f64, x: &[T], seconds: f64) {
        self.converged = converged;
        self.recursive_relative_residual = relative;
        self.solution = x.iter().map(|v| v.wide()).collect();
        self.seconds = seconds;
        self.history.push((self.iterations, relative));
    }
}

struct System<T: Real> {
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    vals: Vec<T>,
    inverse_diagonal: Vec<T>,
    rhs: Vec<T>,
}

impl<T: Real> System<T> {
    fn new(matrix: &Bsr, rhs: &[f64]) -> System<T> {
        let vals: Vec<T> = matrix.vals.iter().map(|&v| T::of(v)).collect();
        let inverse_diagonal = matrix
            .diagonal_positions()
            .iter()
            .flat_map(|&k| invert3(&vals[9 * k..9 * k + 9]))
            .collect();
        System {
            row_ptr: matrix.row_ptr.clone(),
            col_idx: matrix.col_idx.clone(),
            vals,
            inverse_diagonal,
            rhs: rhs.iter().map(|&v| T::of(v)).collect(),
        }
    }

    fn rows(&self) -> usize {
        self.row_ptr.len() - 1
    }

    fn multiply_row(&self, row: usize, p: &[T]) -> [T; 3] {
        let mut acc = [T::zero(); 3];
        for k in self.row_ptr[row]..self.row_ptr[row + 1] {
            let c = 3 * self.col_idx[k];
            let v = &self.vals[9 * k..9 * k + 9];
            for r in 0..3 {
                acc[r] = acc[r] + (v[3 * r] * p[c] + v[3 * r + 1] * p[c + 1] + v[3 * r + 2] * p[c + 2]);
            }
        }
        acc
    }

    fn precondition_row(&self, row: usize, r: &[T]) -> [T; 3] {
        let d = &self.inverse_diagonal[9 * row..9 * row + 9];
        let (a, b, c) = (r[3 * row], r[3 * row + 1], r[3 * row + 2]);
        [d[0] * a + d[1] * b + d[2] * c, d[3] * a + d[4] * b + d[5] * c, d[6] * a + d[7] * b + d[8] * c]
    }
}

struct State<T: Real> {
    x: Vec<T>,
    r: Vec<T>,
    z: Vec<T>,
    p: Vec<T>,
    q: Vec<T>,
    rz: T,
    rr: T,
    rr0: T,
}

impl<T: Real> State<T> {
    fn initialize(system: &System<T>) -> State<T> {
        let n = system.rhs.len();
        let r = system.rhs.clone();
        let z: Vec<T> = (0..system.rows()).flat_map(|row| system.precondition_row(row, &r)).collect();
        let rz = reduce_rows(system.rows(), |row| row_dot(&r, &z, row));
        let rr = reduce_rows(system.rows(), |row| row_dot(&r, &r, row));
        let p = z.clone();
        State { x: vec![T::zero(); n], r, z, p, q: vec![T::zero(); n], rz, rr, rr0: rr }
    }

    fn step(&mut self, system: &System<T>) -> (f64, f64) {
        self.q = (0..system.rows()).flat_map(|row| system.multiply_row(row, &self.p)).collect();
        let pq = reduce_rows(system.rows(), |row| row_dot(&self.p, &self.q, row));
        let alpha = self.rz / pq;
        for i in 0..self.x.len() {
            self.x[i] = self.x[i] + alpha * self.p[i];
            self.r[i] = self.r[i] - alpha * self.q[i];
        }
        self.z = (0..system.rows()).flat_map(|row| system.precondition_row(row, &self.r)).collect();
        let rz_new = reduce_rows(system.rows(), |row| row_dot(&self.r, &self.z, row));
        self.rr = reduce_rows(system.rows(), |row| row_dot(&self.r, &self.r, row));
        let beta = rz_new / self.rz;
        self.rz = rz_new;
        (alpha.wide(), beta.wide())
    }

    fn update_direction(&mut self, beta: f64) {
        let beta = T::of(beta);
        for i in 0..self.p.len() {
            self.p[i] = self.z[i] + beta * self.p[i];
        }
    }

    /// r = b − Ax, z = D⁻¹r and fresh rᵀz, rᵀr; p is kept (van der Vorst–Ye residual replacement).
    fn replace_residual(&mut self, system: &System<T>) {
        self.r = (0..system.rows())
            .flat_map(|row| {
                let ax = system.multiply_row(row, &self.x);
                [0, 1, 2].map(|d| system.rhs[3 * row + d] - ax[d])
            })
            .collect();
        self.z = (0..system.rows()).flat_map(|row| system.precondition_row(row, &self.r)).collect();
        self.rz = reduce_rows(system.rows(), |row| row_dot(&self.r, &self.z, row));
        self.rr = reduce_rows(system.rows(), |row| row_dot(&self.r, &self.r, row));
    }

    fn relative(&self) -> f64 {
        (self.rr.wide() / self.rr0.wide()).sqrt()
    }
}

fn row_dot<T: Real>(a: &[T], b: &[T], row: usize) -> T {
    let i = 3 * row;
    a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2]
}

/// Workgroup trees of 256 rows, then one 256-thread strided pass and a final tree.
pub fn reduce_rows<T: Real>(rows: usize, value: impl Fn(usize) -> T) -> T {
    let groups = rows.div_ceil(WORKGROUP);
    let partials: Vec<T> = (0..groups)
        .map(|g| {
            let mut lane: Vec<T> = (0..WORKGROUP).map(|t| row_or_zero(g * WORKGROUP + t, rows, &value)).collect();
            tree(&mut lane)
        })
        .collect();
    let mut lane: Vec<T> = (0..WORKGROUP).map(|t| strided_sum(&partials, t)).collect();
    tree(&mut lane)
}

fn row_or_zero<T: Real>(row: usize, rows: usize, value: &impl Fn(usize) -> T) -> T {
    if row < rows { value(row) } else { T::zero() }
}

fn strided_sum<T: Real>(partials: &[T], start: usize) -> T {
    partials.iter().skip(start).step_by(WORKGROUP).fold(T::zero(), |acc, &v| acc + v)
}

fn tree<T: Real>(lane: &mut [T]) -> T {
    let mut stride = WORKGROUP / 2;
    while stride > 0 {
        for t in 0..stride {
            lane[t] = lane[t] + lane[t + stride];
        }
        stride /= 2;
    }
    lane[0]
}

/// Adjugate inverse in the same operation order as bsr_jacobi_invert.wgsl.
pub fn invert3<T: Real>(m: &[T]) -> [T; 9] {
    let [a, b, c, d, e, f, g, h, i] = [m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]];
    let c00 = e * i - f * h;
    let c01 = c * h - b * i;
    let c02 = b * f - c * e;
    let c10 = f * g - d * i;
    let c11 = a * i - c * g;
    let c12 = c * d - a * f;
    let c20 = d * h - e * g;
    let c21 = b * g - a * h;
    let c22 = a * e - b * d;
    let det = a * c00 + b * c10 + c * c20;
    [c00, c01, c02, c10, c11, c12, c20, c21, c22].map(|v| v / det)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::band::BandCholesky;
    use crate::cases::{build, CaseKind};
    use crate::linalg::{difference_norm, vector_norm};

    #[test]
    fn f64_pcg_matches_cholesky_on_tension_case() {
        let problem = build(CaseKind::Tension, 13);
        let exact = BandCholesky::factor(&problem.matrix).unwrap().solve(&problem.rhs);
        let settings = PcgSettings { tolerance: 1e-12, max_iterations: 5000, history_stride: 10, replace_every: 0, replace_above: 0.0 };
        let report = solve::<f64>(&problem.matrix, &problem.rhs, &settings);
        assert!(report.converged);
        let error = difference_norm(&report.solution, &exact) / vector_norm(&exact);
        assert!(error < 1e-9, "{error}");
    }

    #[test]
    fn residual_replacement_keeps_f32_pcg_on_the_true_residual() {
        let problem = build(CaseKind::Tension, 33);
        let settings = |replace_every| PcgSettings { tolerance: 1e-7, max_iterations: 5000, history_stride: 10, replace_every, replace_above: 1e-2 };
        let true_residual = |report: &PcgReport| difference_norm(&problem.matrix.multiply(&report.solution), &problem.rhs) / vector_norm(&problem.rhs);
        let plain = solve::<f32>(&problem.matrix, &problem.rhs, &settings(0));
        let replaced = solve::<f32>(&problem.matrix, &problem.rhs, &settings(10));
        assert!(replaced.converged);
        assert!(true_residual(&replaced) < true_residual(&plain), "{} vs {}", true_residual(&replaced), true_residual(&plain));
    }

    #[test]
    fn inverse_times_matrix_is_identity() {
        let m = [4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let inv = invert3::<f64>(&m);
        for r in 0..3 {
            for c in 0..3 {
                let v: f64 = (0..3).map(|k| m[3 * r + k] * inv[3 * k + c]).sum();
                assert!((v - if r == c { 1.0 } else { 0.0 }).abs() < 1e-14);
            }
        }
    }
}
