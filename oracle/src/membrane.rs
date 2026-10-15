use crate::linalg::{add, dot, identity_scaled, mat_add_scaled, outer, scale, sub, Mat3, Vec3};
use crate::material::Material;

/// Constant-strain triangle with rest coordinates already in material axes.
/// grad_n[k] = (dN_k/dX_wale, dN_k/dX_course).
pub struct MembraneElement {
    pub vertices: [usize; 3],
    pub grad_n: [[f64; 2]; 3],
    pub area: f64,
}

pub type ElementHessian = [[Mat3; 3]; 3];

impl MembraneElement {
    pub fn new(vertices: [usize; 3], rest: &[[f64; 2]]) -> MembraneElement {
        let [p0, p1, p2] = vertices.map(|v| rest[v]);
        let dm = [[p1[0] - p0[0], p2[0] - p0[0]], [p1[1] - p0[1], p2[1] - p0[1]]];
        let det = dm[0][0] * dm[1][1] - dm[0][1] * dm[1][0];
        let g1 = [dm[1][1] / det, -dm[0][1] / det];
        let g2 = [-dm[1][0] / det, dm[0][0] / det];
        let g0 = [-g1[0] - g2[0], -g1[1] - g2[1]];
        MembraneElement { vertices, grad_n: [g0, g1, g2], area: 0.5 * det.abs() }
    }

    /// Deformation gradient columns (F_wale, F_course) from edge differences.
    pub fn columns(&self, x: &[Vec3]) -> (Vec3, Vec3) {
        let [v0, v1, v2] = self.vertices;
        let (e1, e2) = (sub(x[v1], x[v0]), sub(x[v2], x[v0]));
        let [g1, g2] = [self.grad_n[1], self.grad_n[2]];
        let wale = add(scale(e1, g1[0]), scale(e2, g2[0]));
        let course = add(scale(e1, g1[1]), scale(e2, g2[1]));
        (wale, course)
    }

    pub fn strain(&self, x: &[Vec3]) -> [f64; 3] {
        let (fw, fc) = self.columns(x);
        [0.5 * (dot(fw, fw) - 1.0), 0.5 * (dot(fc, fc) - 1.0), 0.5 * dot(fw, fc)]
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn energy(&self, x: &[Vec3], material: &Material) -> f64 {
        self.area * material.energy_density(self.strain(x))
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn gradient(&self, x: &[Vec3], material: &Material) -> [Vec3; 3] {
        let (fw, fc) = self.columns(x);
        let [sww, scc, swc] = material.stress(self.strain(x));
        let pw = add(scale(fw, sww), scale(fc, swc));
        let pc = add(scale(fw, swc), scale(fc, scc));
        self.grad_n.map(|g| scale(add(scale(pw, g[0]), scale(pc, g[1])), self.area))
    }

    /// PSD-projected Hessian (negative stress clamped): the E1 linear-system matrices.
    pub fn hessian(&self, x: &[Vec3], material: &Material) -> ElementHessian {
        self.hessian_with_stress(x, material, clamp_negative_stress(material.stress(self.strain(x))))
    }

    /// Exact tangent ∂²W/∂x², the matrix Newton needs for quadratic convergence.
    pub fn tangent(&self, x: &[Vec3], material: &Material) -> ElementHessian {
        self.hessian_with_stress(x, material, material.stress(self.strain(x)))
    }

    /// W(x + αd) − W(x), exact for St. Venant–Kirchhoff (W is quadratic in E) and free of the
    /// cancellation in subtracting two large energies: S(E):ΔE + W(ΔE).
    pub fn energy_change(&self, x: &[Vec3], d: &[Vec3], alpha: f64, material: &Material) -> f64 {
        let (fw, fc) = self.columns(x);
        let (dw, dc) = self.columns(d);
        let (dw, dc) = (scale(dw, alpha), scale(dc, alpha));
        let change = [
            dot(fw, dw) + 0.5 * dot(dw, dw),
            dot(fc, dc) + 0.5 * dot(dc, dc),
            0.5 * (dot(fw, dc) + dot(dw, fc) + dot(dw, dc)),
        ];
        let [sww, scc, swc] = material.stress(self.strain(x));
        let first = sww * change[0] + scc * change[1] + 2.0 * swc * change[2];
        self.area * (first + material.energy_density(change))
    }

    fn hessian_with_stress(&self, x: &[Vec3], material: &Material, stress: [f64; 3]) -> ElementHessian {
        let (fw, fc) = self.columns(x);
        let strain_rows = self.grad_n.map(|g| strain_vectors(g, fw, fc));
        let mut hessian = [[[[0.0; 3]; 3]; 3]; 3];
        for k in 0..3 {
            for l in 0..3 {
                hessian[k][l] = self.block(k, l, &strain_rows, stress, material);
            }
        }
        hessian
    }

    fn block(&self, k: usize, l: usize, rows: &[[Vec3; 3]; 3], stress: [f64; 3], m: &Material) -> Mat3 {
        let ([wk, ck, sk], [wl, cl, sl]) = (rows[k], rows[l]);
        let mut block = identity_scaled(geometric_factor(self.grad_n[k], self.grad_n[l], stress));
        mat_add_scaled(&mut block, &outer(wk, wl), m.wale);
        mat_add_scaled(&mut block, &outer(ck, cl), m.course);
        mat_add_scaled(&mut block, &outer(wk, cl), m.coupling);
        mat_add_scaled(&mut block, &outer(ck, wl), m.coupling);
        mat_add_scaled(&mut block, &outer(sk, sl), 4.0 * m.shear);
        block.map(|row| row.map(|v| v * self.area))
    }
}

/// b vectors: dE_ww, dE_cc, dE_wc with respect to the vertex position.
fn strain_vectors(g: [f64; 2], fw: Vec3, fc: Vec3) -> [Vec3; 3] {
    let shear = scale(add(scale(fw, g[1]), scale(fc, g[0])), 0.5);
    [scale(fw, g[0]), scale(fc, g[1]), shear]
}

fn geometric_factor(gk: [f64; 2], gl: [f64; 2], s: [f64; 3]) -> f64 {
    let [sww, scc, swc] = s;
    gk[0] * (sww * gl[0] + swc * gl[1]) + gk[1] * (swc * gl[0] + scc * gl[1])
}

/// Projects the 2×2 stress onto its positive part so the geometric block stays PSD.
pub fn clamp_negative_stress(s: [f64; 3]) -> [f64; 3] {
    let [a, c, b] = s;
    let mean = 0.5 * (a + c);
    let radius = (0.25 * (a - c) * (a - c) + b * b).sqrt();
    let (high, low) = (mean + radius, mean - radius);
    if low >= 0.0 {
        return s;
    }
    if high <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let v = principal_direction(a, b, c, high);
    [high * v[0] * v[0], high * v[1] * v[1], high * v[0] * v[1]]
}

fn principal_direction(a: f64, b: f64, c: f64, eigenvalue: f64) -> [f64; 2] {
    let first = [b, eigenvalue - a];
    let second = [eigenvalue - c, b];
    let pick = if first[0].hypot(first[1]) >= second[0].hypot(second[1]) { first } else { second };
    let length = pick[0].hypot(pick[1]);
    if length == 0.0 {
        return if a >= c { [1.0, 0.0] } else { [0.0, 1.0] };
    }
    [pick[0] / length, pick[1] / length]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{random_positions, Lcg};

    fn sample_element() -> (MembraneElement, Vec<[f64; 2]>) {
        let rest = vec![[0.0, 0.0], [0.004, 0.0005], [0.001, 0.0045]];
        (MembraneElement::new([0, 1, 2], &rest), rest)
    }

    #[test]
    fn rest_state_is_stress_free() {
        let (element, rest) = sample_element();
        let x: Vec<Vec3> = rest.iter().map(|p| [p[0], p[1], 0.0]).collect();
        let g = element.gradient(&x, &Material::knit_baseline());
        assert!(g.iter().flatten().all(|v| v.abs() < 1e-14));
    }

    #[test]
    fn gradient_matches_energy_finite_difference() {
        let (element, rest) = sample_element();
        let material = Material::knit_baseline();
        let x = random_positions(&rest, &mut Lcg::new(7), 1.15, 0.0008);
        let g = element.gradient(&x, &material);
        let h = 1e-8;
        for v in 0..3 {
            for d in 0..3 {
                let fd = crate::testing::central(&x, v, d, h, |y| element.energy(y, &material));
                assert!((fd - g[v][d]).abs() <= 1e-6 * (1.0 + g[v][d].abs()), "v{v} d{d}: {fd} vs {}", g[v][d]);
            }
        }
    }

    #[test]
    fn hessian_matches_gradient_finite_difference_in_tension() {
        let (element, rest) = sample_element();
        let material = Material::knit_baseline();
        let x = crate::testing::stretched(&rest, 1.04, 1.07, &mut Lcg::new(3), 1e-5);
        let hessian = element.hessian(&x, &material);
        let h = 1e-8;
        for l in 0..3 {
            for d in 0..3 {
                let fd = crate::testing::gradient_derivative(&x, l, d, h, |y| element.gradient(y, &material));
                for k in 0..3 {
                    for r in 0..3 {
                        let exact = hessian[k][l][r][d];
                        assert!((fd[k][r] - exact).abs() <= 1e-5 * (1.0 + exact.abs()), "k{k} l{l} r{r} d{d}");
                    }
                }
            }
        }
    }

    #[test]
    fn exact_tangent_matches_gradient_finite_difference_in_compression() {
        let (element, rest) = sample_element();
        let material = Material::knit_baseline();
        let x = crate::testing::stretched(&rest, 0.93, 1.05, &mut Lcg::new(11), 1e-5);
        let tangent = element.tangent(&x, &material);
        for l in 0..3 {
            for d in 0..3 {
                let fd = crate::testing::gradient_derivative(&x, l, d, 1e-8, |y| element.gradient(y, &material));
                for k in 0..3 {
                    for r in 0..3 {
                        let exact = tangent[k][l][r][d];
                        assert!((fd[k][r] - exact).abs() <= 1e-5 * (1.0 + exact.abs()), "k{k} l{l} r{r} d{d}");
                    }
                }
            }
        }
    }

    #[test]
    fn energy_change_equals_energy_difference() {
        let (element, rest) = sample_element();
        let material = Material::knit_baseline();
        let x = random_positions(&rest, &mut Lcg::new(5), 1.1, 0.0005);
        let d = random_positions(&rest, &mut Lcg::new(6), 0.0, 0.0003);
        let moved: Vec<Vec3> = x.iter().zip(&d).map(|(p, q)| add(*p, scale(*q, 0.7))).collect();
        let direct = element.energy(&moved, &material) - element.energy(&x, &material);
        let change = element.energy_change(&x, &d, 0.7, &material);
        assert!((direct - change).abs() <= 1e-12 * direct.abs().max(element.energy(&x, &material)), "{direct} vs {change}");
    }

    #[test]
    fn rigid_rotation_is_stress_free() {
        let (element, rest) = sample_element();
        let (c, s) = (0.6f64, 0.8f64);
        let x: Vec<Vec3> = rest.iter().map(|p| [c * p[0] - s * p[1], s * p[0] + c * p[1], 0.3]).collect();
        let strain = element.strain(&x);
        assert!(strain.iter().all(|e| e.abs() < 1e-13));
    }
}
