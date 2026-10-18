use crate::linalg::Vec3;
pub use crate::rng::Lcg;

pub fn random_positions(rest: &[[f64; 2]], rng: &mut Lcg, stretch: f64, noise: f64) -> Vec<Vec3> {
    rest.iter()
        .map(|p| [stretch * p[0] + noise * rng.uniform(), p[1] + noise * rng.uniform(), noise * rng.uniform()])
        .collect()
}

pub fn stretched(rest: &[[f64; 2]], wale: f64, course: f64, rng: &mut Lcg, noise: f64) -> Vec<Vec3> {
    rest.iter()
        .map(|p| [wale * p[0] + noise * rng.uniform(), course * p[1] + noise * rng.uniform(), noise * rng.uniform()])
        .collect()
}

pub fn central(x: &[Vec3], vertex: usize, axis: usize, h: f64, f: impl Fn(&[Vec3]) -> f64) -> f64 {
    let (mut plus, mut minus) = (x.to_vec(), x.to_vec());
    plus[vertex][axis] += h;
    minus[vertex][axis] -= h;
    (f(&plus) - f(&minus)) / (2.0 * h)
}

pub fn gradient_derivative(x: &[Vec3], vertex: usize, axis: usize, h: f64, g: impl Fn(&[Vec3]) -> [Vec3; 3]) -> [Vec3; 3] {
    let (mut plus, mut minus) = (x.to_vec(), x.to_vec());
    plus[vertex][axis] += h;
    minus[vertex][axis] -= h;
    let (gp, gm) = (g(&plus), g(&minus));
    std::array::from_fn(|k| std::array::from_fn(|r| (gp[k][r] - gm[k][r]) / (2.0 * h)))
}

pub fn gradient_derivative4(x: &[Vec3], vertex: usize, axis: usize, h: f64, g: impl Fn(&[Vec3]) -> [Vec3; 4]) -> [Vec3; 4] {
    let (mut plus, mut minus) = (x.to_vec(), x.to_vec());
    plus[vertex][axis] += h;
    minus[vertex][axis] -= h;
    let (gp, gm) = (g(&plus), g(&minus));
    std::array::from_fn(|k| std::array::from_fn(|r| (gp[k][r] - gm[k][r]) / (2.0 * h)))
}
