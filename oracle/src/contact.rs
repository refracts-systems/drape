use crate::linalg::{dot, outer, sub, Mat3, Vec3};

/// Analytic rigid collider: a plane n·x = offset (unit n) or a sphere |x − centre| = radius.
#[derive(Clone, Copy, Debug)]
pub enum Collider {
    Plane { normal: Vec3, offset: f64 },
    Sphere { centre: Vec3, radius: f64 },
}

/// A line-search start keeps every vertex at least this fraction of its current distance from every collider.
pub const KEEP_FRACTION: f64 = 0.1;

impl Collider {
    pub fn distance(&self, x: Vec3) -> f64 {
        match *self {
            Collider::Plane { normal, offset } => dot(normal, x) - offset,
            Collider::Sphere { centre, radius } => length(sub(x, centre)) - radius,
        }
    }

    /// ∇d: the plane normal, or the unit vector from the sphere centre.
    pub fn normal(&self, x: Vec3) -> Vec3 {
        match *self {
            Collider::Plane { normal, .. } => normal,
            Collider::Sphere { centre, .. } => {
                let q = sub(x, centre);
                q.map(|v| v / length(q))
            }
        }
    }

    /// 1/|x − c| for the sphere (∇²d = (I − n nᵀ)/|x − c|), 0 for the plane.
    pub fn curvature(&self, x: Vec3) -> f64 {
        match *self {
            Collider::Plane { .. } => 0.0,
            Collider::Sphere { centre, .. } => 1.0 / length(sub(x, centre)),
        }
    }

    /// Largest α ∈ (0, 1] with d(x + αp) ≥ 0.1·d(x).
    pub fn max_step(&self, x: Vec3, p: Vec3) -> f64 {
        let d0 = self.distance(x);
        match *self {
            Collider::Plane { normal, .. } => plane_step(d0, dot(normal, p)),
            Collider::Sphere { centre, radius } => sphere_step(d0, sub(x, centre), p, radius),
        }
    }

    /// [kind, a, b, c, w, 0, 0, 0]: kind 1 plane (n, offset), kind 2 sphere (centre, radius).
    pub fn packed(&self) -> [f64; 8] {
        match *self {
            Collider::Plane { normal: n, offset } => [1.0, n[0], n[1], n[2], offset, 0.0, 0.0, 0.0],
            Collider::Sphere { centre: c, radius } => [2.0, c[0], c[1], c[2], radius, 0.0, 0.0, 0.0],
        }
    }
}

fn length(v: Vec3) -> f64 {
    dot(v, v).sqrt()
}

/// d(α) = d0 + α·rate: the root of d(α) = 0.1·d0 when approaching.
fn plane_step(d0: f64, rate: f64) -> f64 {
    if rate >= 0.0 { 1.0 } else { (1.0 - KEEP_FRACTION) * d0 / -rate }.min(1.0)
}

/// |q + αp| = R' with R' = r + 0.1·d0: smallest positive root of a·α² + b·α + c, c = 0.9·d0·(|q| + R') > 0, stable form.
fn sphere_step(d0: f64, q: Vec3, p: Vec3, radius: f64) -> f64 {
    let (a, b) = (dot(p, p), 2.0 * dot(q, p));
    let target = radius + KEEP_FRACTION * d0;
    let c = (1.0 - KEEP_FRACTION) * d0 * (length(q) + target);
    let discriminant = b * b - 4.0 * a * c;
    if b >= 0.0 || discriminant < 0.0 {
        return 1.0;
    }
    (2.0 * c / (-b + discriminant.sqrt())).min(1.0)
}

/// IPC barrier b(d) = −κ (d − d̂)² ln(d/d̂) on 0 < d < d̂.
#[derive(Clone, Copy, Debug)]
pub struct Barrier {
    pub dhat: f64,
    pub kappa: f64,
}

impl Barrier {
    pub fn value(&self, d: f64) -> f64 {
        if d >= self.dhat { 0.0 } else if d <= 0.0 { f64::INFINITY } else { -self.kappa * (d - self.dhat).powi(2) * (d / self.dhat).ln() }
    }

    pub fn first(&self, d: f64) -> f64 {
        if d >= self.dhat { return 0.0 }
        let t = d - self.dhat;
        -self.kappa * (2.0 * t * (d / self.dhat).ln() + t * t / d)
    }

    pub fn second(&self, d: f64) -> f64 {
        if d >= self.dhat { return 0.0 }
        let t = d - self.dhat;
        -self.kappa * (2.0 * (d / self.dhat).ln() + 4.0 * t / d - t * t / (d * d))
    }

    /// κ for which a vertex of mass `mass` under gravity `gravity` rests at d = d̂/2: κ = m·g / |b′(d̂/2)/κ|.
    pub fn resting_kappa(dhat: f64, mass: f64, gravity: f64) -> f64 {
        let unit = Barrier { dhat, kappa: 1.0 };
        mass * gravity / unit.first(0.5 * dhat).abs()
    }
}

/// Every collider acting on every vertex through one barrier.
#[derive(Clone, Debug)]
pub struct Contact {
    pub colliders: Vec<Collider>,
    pub barrier: Barrier,
}

impl Contact {
    pub fn none() -> Contact {
        Contact { colliders: Vec::new(), barrier: Barrier { dhat: 0.0, kappa: 0.0 } }
    }

    pub fn energy(&self, x: Vec3) -> f64 {
        self.colliders.iter().map(|c| self.barrier.value(c.distance(x))).sum()
    }

    pub fn gradient(&self, x: Vec3) -> Vec3 {
        self.colliders.iter().fold([0.0; 3], |sum, c| {
            let slope = self.barrier.first(c.distance(x));
            let n = c.normal(x);
            std::array::from_fn(|k| sum[k] + slope * n[k])
        })
    }

    /// Σ b″ n nᵀ + b′ (I − n nᵀ)/|x − c| (the second term only on the sphere).
    pub fn hessian(&self, x: Vec3) -> Mat3 {
        self.hessian_with(x, true)
    }

    /// The same Hessian; with `curvature` false the b′∇²d term is dropped (the evidence-triggered change of Experiments 5 and 7).
    pub fn hessian_with(&self, x: Vec3, curvature_term: bool) -> Mat3 {
        self.colliders.iter().fold([[0.0; 3]; 3], |sum, c| {
            let d = c.distance(x);
            let (first, second, curvature) = (self.barrier.first(d), self.barrier.second(d), if curvature_term { c.curvature(x) } else { 0.0 });
            let nn = outer(c.normal(x), c.normal(x));
            std::array::from_fn(|i| std::array::from_fn(|j| {
                let identity = if i == j { 1.0 } else { 0.0 };
                sum[i][j] + second * nn[i][j] + first * curvature * (identity - nn[i][j])
            }))
        })
    }

    pub fn max_step(&self, x: Vec3, p: Vec3) -> f64 {
        self.colliders.iter().map(|c| c.max_step(x, p)).fold(1.0, f64::min)
    }

    /// α_max and the collider that sets it (None when no collider limits the step).
    pub fn max_step_limited(&self, x: Vec3, p: Vec3) -> (f64, Option<usize>) {
        self.colliders.iter().enumerate().fold((1.0, None), |(alpha, at), (k, c)| {
            let step = c.max_step(x, p);
            if step < alpha { (step, Some(k)) } else { (alpha, at) }
        })
    }

    /// |x − c|/d for the sphere, 1 for the plane: how much the f32 error of |x − c| is magnified in d.
    pub fn distance_condition(&self, k: usize, x: Vec3) -> f64 {
        match self.colliders[k] {
            Collider::Plane { .. } => 1.0,
            Collider::Sphere { centre, .. } => length(sub(x, centre)) / self.colliders[k].distance(x),
        }
    }

    /// Smallest distance to any collider (∞ without colliders).
    pub fn min_distance(&self, x: Vec3) -> f64 {
        self.colliders.iter().map(|c| c.distance(x)).fold(f64::INFINITY, f64::min)
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    fn contact() -> Contact {
        let floor = Collider::Plane { normal: [0.0, 0.0, 1.0], offset: 0.0 };
        let ball = Collider::Sphere { centre: [0.25, 0.25, 0.125], radius: 0.125 };
        Contact { colliders: vec![floor, ball], barrier: Barrier { dhat: 5e-4, kappa: 2.0 } }
    }

    #[test]
    fn barrier_derivatives_match_finite_differences() {
        let b = Barrier { dhat: 5e-4, kappa: 3.0 };
        for d in [3e-5, 1e-4, 2.5e-4, 4.9e-4] {
            let h = 1e-9;
            let first = (b.value(d + h) - b.value(d - h)) / (2.0 * h);
            let second = (b.first(d + h) - b.first(d - h)) / (2.0 * h);
            assert!((first - b.first(d)).abs() <= 1e-6 * b.first(d).abs(), "{d}: {first} vs {}", b.first(d));
            assert!((second - b.second(d)).abs() <= 1e-6 * b.second(d).abs(), "{d}: {second} vs {}", b.second(d));
        }
    }

    #[test]
    fn resting_kappa_balances_gravity_at_half_dhat() {
        let kappa = Barrier::resting_kappa(5e-4, 2e-5, 9.81);
        assert!((Barrier { dhat: 5e-4, kappa }.first(2.5e-4) + 2e-5 * 9.81).abs() < 1e-18);
    }

    #[test]
    fn contact_gradient_and_hessian_match_finite_differences() {
        let c = contact();
        let x = [0.27, 0.24, 0.2502];
        let h = 1e-9;
        for k in 0..3 {
            let (mut plus, mut minus) = (x, x);
            plus[k] += h;
            minus[k] -= h;
            let fd = (c.energy(plus) - c.energy(minus)) / (2.0 * h);
            assert!((fd - c.gradient(x)[k]).abs() <= 1e-6 * c.gradient(x)[k].abs().max(1e-9), "{k}");
            let column: Vec3 = std::array::from_fn(|i| (c.gradient(plus)[i] - c.gradient(minus)[i]) / (2.0 * h));
            (0..3).for_each(|i| assert!((column[i] - c.hessian(x)[i][k]).abs() <= 1e-5 * c.hessian(x)[k][k].abs(), "{i}{k}"));
        }
    }

    #[test]
    fn max_step_keeps_a_tenth_of_the_distance() {
        let c = contact();
        let x = [0.4, 0.4, 2e-4];
        let alpha = c.max_step(x, [0.0, 0.0, -1e-3]);
        assert!((x[2] - alpha * 1e-3 - 2e-5).abs() < 1e-15, "{alpha}");
        let top = [0.25, 0.25, 0.25 + 1e-4];
        let beta = c.max_step(top, [0.0, 0.0, -1e-3]);
        assert!((c.colliders[1].distance([0.25, 0.25, top[2] - beta * 1e-3]) - 1e-5).abs() < 1e-14, "{beta}");
        assert_eq!(c.max_step(top, [0.0, 0.0, 1e-3]), 1.0);
    }
}
