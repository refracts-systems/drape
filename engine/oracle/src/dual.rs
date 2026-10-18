use std::ops::{Add, Div, Mul, Neg, Sub};

/// Forward-mode dual number: a value and its derivative along one direction. Differentiating an
/// analytic gradient with it gives the exact Hessian column (to rounding), with no new formulas.
#[derive(Clone, Copy, Debug)]
pub struct Dual {
    pub v: f64,
    pub t: f64,
}

pub trait Scalar: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> {
    fn constant(value: f64) -> Self;
    fn sqrt(self) -> Self;
}

impl Scalar for f64 {
    fn constant(value: f64) -> f64 {
        value
    }
    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }
}

impl Scalar for Dual {
    fn constant(value: f64) -> Dual {
        Dual { v: value, t: 0.0 }
    }
    fn sqrt(self) -> Dual {
        let s = self.v.sqrt();
        Dual { v: s, t: self.t / (2.0 * s) }
    }
}

impl Add for Dual {
    type Output = Dual;
    fn add(self, o: Dual) -> Dual {
        Dual { v: self.v + o.v, t: self.t + o.t }
    }
}

impl Sub for Dual {
    type Output = Dual;
    fn sub(self, o: Dual) -> Dual {
        Dual { v: self.v - o.v, t: self.t - o.t }
    }
}

impl Mul for Dual {
    type Output = Dual;
    fn mul(self, o: Dual) -> Dual {
        Dual { v: self.v * o.v, t: self.t * o.v + self.v * o.t }
    }
}

impl Div for Dual {
    type Output = Dual;
    fn div(self, o: Dual) -> Dual {
        Dual { v: self.v / o.v, t: (self.t * o.v - self.v * o.t) / (o.v * o.v) }
    }
}

impl Neg for Dual {
    type Output = Dual;
    fn neg(self) -> Dual {
        Dual { v: -self.v, t: -self.t }
    }
}

pub type V3<T> = [T; 3];

pub fn sub3<T: Scalar>(a: V3<T>, b: V3<T>) -> V3<T> {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn add3<T: Scalar>(a: V3<T>, b: V3<T>) -> V3<T> {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn scale3<T: Scalar>(a: V3<T>, s: T) -> V3<T> {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn dot3<T: Scalar>(a: V3<T>, b: V3<T>) -> T {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross3<T: Scalar>(a: V3<T>, b: V3<T>) -> V3<T> {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dual_derivatives_match_calculus() {
        let x = Dual { v: 2.0, t: 1.0 };
        let y = (x * x + Dual::constant(3.0)) / x.sqrt();
        let exact = (2.0 * 2.0 * 2.0f64.sqrt() - (4.0 + 3.0) / (2.0 * 2.0f64.sqrt())) / 2.0;
        assert!((y.t - exact).abs() < 1e-14, "{} vs {exact}", y.t);
    }
}
