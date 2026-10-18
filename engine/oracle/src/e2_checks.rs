use crate::coupon::{mask_vector, Coupon, COUPON_SIZE};
use crate::io::Json;
use crate::linalg::{difference_norm, vector_norm, Vec3};
use crate::rng::Lcg;

const FD_STEP: f64 = 1e-7;
const FD_DIRECTIONS: usize = 6;

/// Patch test: the computed field must be the affine boundary map, with one constant F and E in every element.
pub struct PatchCheck {
    pub displacement_error: f64,
    pub deformation_gradient_error: f64,
    pub strain_error: f64,
}

pub fn patch_check(coupon: &Coupon, x: &[Vec3]) -> PatchCheck {
    let f = coupon.path.boundary_gradient(1.0);
    let affine = coupon.affine(1.0);
    let displacement = x.iter().zip(&affine).flat_map(|(p, q)| (0..3).map(move |d| (p[d] - q[d]).abs())).fold(0.0, f64::max);
    let exact_columns = ([f[0][0], f[1][0], 0.0], [f[0][1], f[1][1], 0.0]);
    let exact_strain = green_strain(exact_columns);
    let scale = exact_strain.iter().fold(0.0f64, |m, e| m.max(e.abs()));
    let (mut gradient_error, mut strain_error) = (0.0f64, 0.0f64);
    for element in &coupon.elements {
        let columns = element.columns(x);
        gradient_error = gradient_error.max(max_difference(&[columns.0, columns.1], &[exact_columns.0, exact_columns.1]));
        strain_error = strain_error.max(max_difference(&[element.strain(x)], &[exact_strain]) / scale);
    }
    PatchCheck { displacement_error: displacement / COUPON_SIZE, deformation_gradient_error: gradient_error, strain_error }
}

fn green_strain((fw, fc): (Vec3, Vec3)) -> [f64; 3] {
    let dot = |a: Vec3, b: Vec3| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    [0.5 * (dot(fw, fw) - 1.0), 0.5 * (dot(fc, fc) - 1.0), 0.5 * dot(fw, fc)]
}

fn max_difference(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter().zip(b).flat_map(|(p, q)| (0..3).map(move |d| (p[d] - q[d]).abs())).fold(0.0, f64::max)
}

impl PatchCheck {
    pub fn json(&self) -> Json {
        Json::new()
            .number("displacement_error_per_coupon_size", self.displacement_error)
            .number("deformation_gradient_error", self.deformation_gradient_error)
            .number("relative_strain_error", self.strain_error)
    }
}

/// Finite-difference checks of the assembled energy → gradient → tangent chain at a random stretched state.
pub struct DerivativeCheck {
    pub gradient_error: f64,
    pub tangent_error: f64,
    pub energy_change_error: f64,
}

pub fn derivative_check(coupon: &Coupon) -> DerivativeCheck {
    let mut rng = Lcg::new(77);
    let x = random_state(coupon, &mut rng);
    let directions: Vec<Vec<Vec3>> = (0..FD_DIRECTIONS).map(|_| random_direction(coupon, &mut rng)).collect();
    let fold = |f: &dyn Fn(&[Vec3]) -> f64| directions.iter().map(|v| f(v)).fold(0.0, f64::max);
    DerivativeCheck {
        gradient_error: fold(&|v| gradient_error(coupon, &x, v)),
        tangent_error: fold(&|v| tangent_error(coupon, &x, v)),
        energy_change_error: fold(&|v| energy_change_error(coupon, &x, v)),
    }
}

fn random_state(coupon: &Coupon, rng: &mut Lcg) -> Vec<Vec3> {
    let spacing = COUPON_SIZE / (coupon.mesh.columns - 1) as f64;
    let mut x = coupon.affine(0.5);
    coupon.free.iter().for_each(|&v| x[v] = mask_vector([x[v][0] + 0.05 * spacing * rng.uniform(), x[v][1] + 0.05 * spacing * rng.uniform(), 0.0]));
    x
}

/// Unit max-norm direction on the free components of free vertices.
fn random_direction(coupon: &Coupon, rng: &mut Lcg) -> Vec<Vec3> {
    let mut v = vec![[0.0; 3]; coupon.mesh.rest.len()];
    coupon.free.iter().for_each(|&i| v[i] = mask_vector([rng.uniform(), rng.uniform(), 0.0]));
    v
}

fn shifted(x: &[Vec3], v: &[Vec3], h: f64) -> Vec<Vec3> {
    x.iter().zip(v).map(|(p, d)| [p[0] + h * d[0], p[1] + h * d[1], p[2] + h * d[2]]).collect()
}

fn rows_of(coupon: &Coupon, v: &[Vec3]) -> Vec<f64> {
    coupon.free.iter().flat_map(|&i| v[i]).collect()
}

fn gradient_error(coupon: &Coupon, x: &[Vec3], v: &[Vec3]) -> f64 {
    let fd = (coupon.energy(&shifted(x, v, FD_STEP)) - coupon.energy(&shifted(x, v, -FD_STEP))) / (2.0 * FD_STEP);
    let exact = crate::coupon::free_dot(&coupon.free_gradient(x), &rows_of(coupon, v));
    (fd - exact).abs() / exact.abs()
}

fn tangent_error(coupon: &Coupon, x: &[Vec3], v: &[Vec3]) -> f64 {
    let (plus, minus) = (coupon.free_gradient(&shifted(x, v, FD_STEP)), coupon.free_gradient(&shifted(x, v, -FD_STEP)));
    let fd: Vec<f64> = plus.iter().zip(&minus).map(|(p, m)| (p - m) / (2.0 * FD_STEP)).collect();
    let exact = coupon.tangent(x).multiply(&rows_of(coupon, v));
    difference_norm(&fd, &exact) / vector_norm(&exact)
}

fn energy_change_error(coupon: &Coupon, x: &[Vec3], v: &[Vec3]) -> f64 {
    let alpha = 1e-3;
    let direct = coupon.energy(&shifted(x, v, alpha)) - coupon.energy(x);
    (coupon.energy_change(x, v, alpha) - direct).abs() / direct.abs()
}

impl DerivativeCheck {
    pub fn json(&self) -> Json {
        Json::new()
            .number("fd_step_m", FD_STEP)
            .integer("directions", FD_DIRECTIONS)
            .number("gradient_relative_error", self.gradient_error)
            .number("tangent_relative_error", self.tangent_error)
            .number("energy_change_relative_error", self.energy_change_error)
    }
}

/// Hyperelastic loading/unloading: unloading must retrace the loading curve and end at the rest state.
pub fn path_independence(coupon: &Coupon, positions: &[Vec<Vec3>]) -> PathCheck {
    let factors = coupon.path.load_factors();
    let stresses: Vec<[f64; 2]> = positions.iter().map(|x| coupon.grip_stress(x)).collect();
    let peak = stresses.iter().flat_map(|s| s.iter().map(|v| v.abs())).fold(0.0, f64::max);
    let (half, rest) = (factors.len() / 2, coupon.rest_positions());
    let gap = (half..factors.len()).map(|u| hysteresis(&factors, &stresses, u)).fold(0.0, f64::max);
    let final_displacement = max_difference(&positions[factors.len() - 1], &rest) / COUPON_SIZE;
    PathCheck { peak_stress: peak, stress_gap: gap / peak, final_displacement }
}

pub struct PathCheck {
    pub peak_stress: f64,
    pub stress_gap: f64,
    pub final_displacement: f64,
}

impl PathCheck {
    pub fn json(&self) -> Json {
        Json::new()
            .number("peak_nominal_stress_N_per_m", self.peak_stress)
            .number("unload_vs_load_stress_gap_relative", self.stress_gap)
            .number("final_displacement_per_coupon_size", self.final_displacement)
    }
}

fn hysteresis(factors: &[f64], stresses: &[[f64; 2]], unload: usize) -> f64 {
    let matching = (0..unload).find(|&l| (factors[l] - factors[unload]).abs() < 1e-12);
    let load = matching.map(|l| stresses[l]).unwrap_or([0.0, 0.0]);
    (0..2).map(|d| (stresses[unload][d] - load[d]).abs()).fold(0.0, f64::max)
}
