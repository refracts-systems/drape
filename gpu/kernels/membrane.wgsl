// Constant-strain triangle and orthotropic St. Venant–Kirchhoff law in material axes (wale = x, course = y),
// prepended by the host to the membrane kernels. Same formulas and order as oracle membrane.rs / material.rs.
override WALE: f32 = 0.0;
override COURSE: f32 = 0.0;
override COUPLING: f32 = 0.0;
override SHEAR: f32 = 0.0;

// Columns of the deformation gradient F = [∂x/∂X_wale, ∂x/∂X_course] from edge vectors and shape gradients.
struct Columns {
  wale: vec3f,
  course: vec3f,
}

fn columns_of(e1: vec3f, e2: vec3f, g1: vec2f, g2: vec2f) -> Columns {
  return Columns(e1 * g1.x + e2 * g2.x, e1 * g1.y + e2 * g2.y);
}

// Green strain (E_ww, E_cc, E_wc).
fn green_strain(f: Columns) -> vec3f {
  return vec3f(0.5 * (dot(f.wale, f.wale) - 1.0), 0.5 * (dot(f.course, f.course) - 1.0), 0.5 * dot(f.wale, f.course));
}

// Second Piola–Kirchhoff stress (S_ww, S_cc, S_wc).
fn stress_of(e: vec3f) -> vec3f {
  return vec3f(WALE * e.x + COUPLING * e.y, COURSE * e.y + COUPLING * e.x, 2.0 * SHEAR * e.z);
}

fn energy_density(e: vec3f) -> f32 {
  return 0.5 * WALE * e.x * e.x + 0.5 * COURSE * e.y * e.y + COUPLING * e.x * e.y + 2.0 * SHEAR * e.z * e.z;
}
