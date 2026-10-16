use crate::dual::{add3, cross3, dot3, scale3, sub3, Dual, Scalar, V3};
use crate::linalg::{add, cross, dot, norm, outer, scale, sub, Mat3, Vec3};

/// Discrete hinge with vertices [left opposite x1, right opposite x2, edge start x3, edge end x4].
/// Energy = stiffness * theta^2, rest angle zero (panels are cut flat).
pub struct HingeElement {
    pub vertices: [usize; 4],
    pub stiffness: f64,
}

impl HingeElement {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn angle(&self, x: &[Vec3]) -> f64 {
        let [x1, x2, x3, x4] = self.vertices.map(|v| x[v]);
        let edge = sub(x4, x3);
        let n1 = cross(sub(x1, x3), sub(x1, x4));
        let n2 = cross(sub(x2, x4), sub(x2, x3));
        let (u1, u2) = (scale(n1, 1.0 / norm(n1)), scale(n2, 1.0 / norm(n2)));
        let sine = dot(cross(u1, u2), scale(edge, 1.0 / norm(edge)));
        sine.atan2(dot(u1, u2))
    }

    /// d theta / d x_k (Bridson et al. 2003 bending modes; sign fixed by finite-difference test).
    pub fn angle_gradient(&self, x: &[Vec3]) -> [Vec3; 4] {
        let [x1, x2, x3, x4] = self.vertices.map(|v| x[v]);
        let edge = sub(x4, x3);
        let length = norm(edge);
        let n1 = cross(sub(x1, x3), sub(x1, x4));
        let n2 = cross(sub(x2, x4), sub(x2, x3));
        let (a1, a2) = (scale(n1, 1.0 / dot(n1, n1)), scale(n2, 1.0 / dot(n2, n2)));
        let u1 = scale(a1, length);
        let u2 = scale(a2, length);
        let u3 = add_pair(a1, dot(sub(x1, x4), edge) / length, a2, dot(sub(x2, x4), edge) / length);
        let u4 = add_pair(a1, -dot(sub(x1, x3), edge) / length, a2, -dot(sub(x2, x3), edge) / length);
        [u1, u2, u3, u4].map(|u| scale(u, -1.0))
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn energy(&self, x: &[Vec3]) -> f64 {
        let theta = self.angle(x);
        self.stiffness * theta * theta
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn gradient(&self, x: &[Vec3]) -> [Vec3; 4] {
        let factor = 2.0 * self.stiffness * self.angle(x);
        self.angle_gradient(x).map(|g| scale(g, factor))
    }

    /// Gauss–Newton Hessian 2k ∇θ∇θᵀ: PSD, and exact when θ = 0.
    pub fn hessian(&self, x: &[Vec3]) -> [[Mat3; 4]; 4] {
        let g = self.angle_gradient(x);
        let factor = 2.0 * self.stiffness;
        let mut blocks = [[[[0.0; 3]; 3]; 4]; 4];
        for k in 0..4 {
            for l in 0..4 {
                blocks[k][l] = outer(g[k], g[l]).map(|row| row.map(|v| v * factor));
            }
        }
        blocks
    }
}

impl HingeElement {
    /// Exact Hessian of θ: each column is the forward-mode derivative of `angle_gradient_of` along one coordinate.
    pub fn angle_hessian(&self, x: &[Vec3]) -> [[Mat3; 4]; 4] {
        let points = self.vertices.map(|v| x[v]);
        let mut blocks = [[[[0.0; 3]; 3]; 4]; 4];
        for (l, c) in (0..4).flat_map(|l| (0..3).map(move |c| (l, c))) {
            let seeded: [V3<Dual>; 4] = std::array::from_fn(|k| std::array::from_fn(|d| Dual { v: points[k][d], t: f64::from(u8::from(k == l && d == c)) }));
            let column = angle_gradient_of(seeded);
            (0..4).for_each(|k| (0..3).for_each(|r| blocks[k][l][r][c] = column[k][r].t));
        }
        blocks
    }

    /// Exact energy Hessian 2k(∇θ∇θᵀ + θ∇²θ), the tangent Newton needs away from the flat state.
    pub fn tangent(&self, x: &[Vec3]) -> [[Mat3; 4]; 4] {
        let (theta, g, h) = (self.angle(x), self.angle_gradient(x), self.angle_hessian(x));
        let factor = 2.0 * self.stiffness;
        std::array::from_fn(|k| std::array::from_fn(|l| std::array::from_fn(|r| std::array::from_fn(|c| factor * (g[k][r] * g[l][c] + theta * h[k][l][r][c])))))
    }

    /// k(θ' − θ)(θ' + θ) for x → x + αd: one hinge's energy change, never a difference of global energies.
    pub fn energy_change(&self, x: &[Vec3], d: &[Vec3], alpha: f64) -> f64 {
        let moved: Vec<Vec3> = self.vertices.iter().map(|&v| add(x[v], scale(d[v], alpha))).collect();
        let local = HingeElement { vertices: [0, 1, 2, 3], stiffness: self.stiffness };
        let (before, after) = (self.angle(x), local.angle(&moved));
        self.stiffness * (after - before) * (after + before)
    }
}

/// `angle_gradient` written once for any scalar (f64 or Dual), same operation order.
pub fn angle_gradient_of<T: Scalar>(p: [V3<T>; 4]) -> [V3<T>; 4] {
    let [x1, x2, x3, x4] = p;
    let edge = sub3(x4, x3);
    let length = dot3(edge, edge).sqrt();
    let n1 = cross3(sub3(x1, x3), sub3(x1, x4));
    let n2 = cross3(sub3(x2, x4), sub3(x2, x3));
    let (a1, a2) = (scale3(n1, T::constant(1.0) / dot3(n1, n1)), scale3(n2, T::constant(1.0) / dot3(n2, n2)));
    let u1 = scale3(a1, length);
    let u2 = scale3(a2, length);
    let u3 = add3(scale3(a1, dot3(sub3(x1, x4), edge) / length), scale3(a2, dot3(sub3(x2, x4), edge) / length));
    let u4 = add3(scale3(a1, -dot3(sub3(x1, x3), edge) / length), scale3(a2, -dot3(sub3(x2, x3), edge) / length));
    [u1, u2, u3, u4].map(|u| scale3(u, T::constant(-1.0)))
}

fn add_pair(a: Vec3, sa: f64, b: Vec3, sb: f64) -> Vec3 {
    let (p, q) = (scale(a, sa), scale(b, sb));
    [p[0] + q[0], p[1] + q[1], p[2] + q[2]]
}

/// c_h = k_b(e) * 3 |e|^2 / (A1 + A2) from rest geometry.
pub fn hinge_stiffness(rest: &[[f64; 2]], vertices: [usize; 4], bending: f64) -> f64 {
    let [p1, p2, p3, p4] = vertices.map(|v| rest[v]);
    let edge = [p4[0] - p3[0], p4[1] - p3[1]];
    let length2 = edge[0] * edge[0] + edge[1] * edge[1];
    let area = |p: [f64; 2]| 0.5 * ((p3[0] - p[0]) * (p4[1] - p[1]) - (p3[1] - p[1]) * (p4[0] - p[0])).abs();
    bending * 3.0 * length2 / (area(p1) + area(p2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{central, gradient_derivative4};

    fn bent_hinge(height: f64) -> (HingeElement, Vec<Vec3>) {
        let x = vec![[0.5, 0.9, height], [0.45, -1.1, -0.2 * height], [0.0, 0.0, 0.0], [1.0, 0.05, 0.0]];
        (HingeElement { vertices: [0, 1, 2, 3], stiffness: 1.3 }, x)
    }

    #[test]
    fn flat_hinge_has_zero_angle() {
        let (hinge, x) = bent_hinge(0.0);
        assert!(hinge.angle(&x).abs() < 1e-14);
    }

    #[test]
    fn angle_gradient_matches_finite_difference() {
        let (hinge, x) = bent_hinge(0.4);
        let g = hinge.angle_gradient(&x);
        for v in 0..4 {
            for d in 0..3 {
                let fd = central(&x, v, d, 1e-7, |y| hinge.angle(y));
                assert!((fd - g[v][d]).abs() < 1e-6, "v{v} d{d}: fd {fd} analytic {}", g[v][d]);
            }
        }
    }

    #[test]
    fn generic_angle_gradient_matches_the_f64_one() {
        let (hinge, x) = bent_hinge(0.4);
        let generic = angle_gradient_of(hinge.vertices.map(|v| x[v]));
        assert_eq!(generic, hinge.angle_gradient(&x));
    }

    #[test]
    fn exact_tangent_matches_gradient_finite_difference_when_bent() {
        let (hinge, x) = bent_hinge(0.4);
        let tangent = hinge.tangent(&x);
        for l in 0..4 {
            for d in 0..3 {
                let fd = gradient_derivative4(&x, l, d, 1e-7, |y| hinge.gradient(y));
                for k in 0..4 {
                    for r in 0..3 {
                        let exact = tangent[k][l][r][d];
                        assert!((fd[k][r] - exact).abs() < 1e-6 * (1.0 + exact.abs()), "k{k} l{l} r{r} d{d}: fd {} exact {exact}", fd[k][r]);
                    }
                }
            }
        }
    }

    #[test]
    fn angle_hessian_is_symmetric() {
        let (hinge, x) = bent_hinge(0.3);
        let h = hinge.angle_hessian(&x);
        for (k, l, r, c) in (0..4).flat_map(|k| (0..4).flat_map(move |l| (0..3).flat_map(move |r| (0..3).map(move |c| (k, l, r, c))))) {
            assert!((h[k][l][r][c] - h[l][k][c][r]).abs() < 1e-10, "k{k} l{l} r{r} c{c}");
        }
    }

    #[test]
    fn energy_change_equals_energy_difference() {
        let (hinge, x) = bent_hinge(0.4);
        let d = vec![[0.01, -0.02, 0.03], [0.0, 0.01, -0.02], [0.005, 0.0, 0.01], [-0.01, 0.02, 0.0]];
        let moved: Vec<Vec3> = x.iter().zip(&d).map(|(p, q)| add(*p, scale(*q, 0.6))).collect();
        let direct = hinge.energy(&moved) - hinge.energy(&x);
        assert!((hinge.energy_change(&x, &d, 0.6) - direct).abs() < 1e-14, "{direct}");
    }

    #[test]
    fn gauss_newton_hessian_is_exact_at_flat_state() {
        let (hinge, x) = bent_hinge(0.0);
        let hessian = hinge.hessian(&x);
        for l in 0..4 {
            for d in 0..3 {
                let fd = gradient_derivative4(&x, l, d, 1e-7, |y| hinge.gradient(y));
                for k in 0..4 {
                    for r in 0..3 {
                        assert!((fd[k][r] - hessian[k][l][r][d]).abs() < 1e-5, "k{k} l{l} r{r} d{d}");
                    }
                }
            }
        }
    }
}
