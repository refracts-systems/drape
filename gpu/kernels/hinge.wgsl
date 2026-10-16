// Discrete hinge (dihedral angle θ between two triangles sharing edge x3–x4), prepended by the host to the hinge kernels.
// Same formulas and order as oracle hinge.rs. D1/D3 are forward-mode dual numbers: value v and derivative t along one
// coordinate, so differentiating the analytic ∇θ gives exact columns of ∇²θ.
struct D1 {
  v: f32,
  t: f32,
}

struct D3 {
  v: vec3f,
  t: vec3f,
}

fn d1(value: f32) -> D1 {
  return D1(value, 0.0);
}

fn d1_div(a: D1, b: D1) -> D1 {
  return D1(a.v / b.v, (a.t * b.v - a.v * b.t) / (b.v * b.v));
}

fn d1_neg(a: D1) -> D1 {
  return D1(-a.v, -a.t);
}

fn d1_sqrt(a: D1) -> D1 {
  let s = sqrt(a.v);
  return D1(s, a.t / (2.0 * s));
}

fn d3_sub(a: D3, b: D3) -> D3 {
  return D3(a.v - b.v, a.t - b.t);
}

fn d3_add(a: D3, b: D3) -> D3 {
  return D3(a.v + b.v, a.t + b.t);
}

fn d3_scale(a: D3, s: D1) -> D3 {
  return D3(a.v * s.v, a.t * s.v + a.v * s.t);
}

fn d3_dot(a: D3, b: D3) -> D1 {
  return D1(dot(a.v, b.v), dot(a.t, b.v) + dot(a.v, b.t));
}

fn d3_cross(a: D3, b: D3) -> D3 {
  return D3(cross(a.v, b.v), cross(a.t, b.v) + cross(a.v, b.t));
}

const PI: f32 = 3.14159265358979;

// Cephes atanf (about 2 ulp). The WGSL builtin atan/atan2 is only required to be within 4096 ulp (≈ 2.4e-4 relative),
// which put every hinge angle, force and Hessian 1e-4 off the f64 values: too coarse for the 1e-5 kernel gate.
fn atan_accurate(value: f32) -> f32 {
  let x = abs(value);
  var offset = 0.0;
  var t = x;
  if (x > 2.414213562373095) {
    offset = 0.5 * PI;
    t = -1.0 / x;
  } else if (x > 0.4142135623730950) {
    offset = 0.25 * PI;
    t = (x - 1.0) / (x + 1.0);
  }
  let z = t * t;
  let y = offset + ((((0.0805374449538 * z - 0.138776856032) * z + 0.199777106478) * z - 0.333329491539) * z * t + t);
  return select(y, -y, value < 0.0);
}

fn atan2_accurate(y: f32, x: f32) -> f32 {
  if (x > 0.0) {
    return atan_accurate(y / x);
  }
  if (x < 0.0) {
    return atan_accurate(y / x) + select(-PI, PI, y >= 0.0);
  }
  return select(-0.5 * PI, 0.5 * PI, y >= 0.0) * f32(y != 0.0);
}

// θ = atan2((û1 × û2)·ê, û1·û2) with the face normals of (x1, x3, x4) and (x2, x4, x3).
fn hinge_angle(x1: vec3f, x2: vec3f, x3: vec3f, x4: vec3f) -> f32 {
  let edge = x4 - x3;
  let n1 = cross(x1 - x3, x1 - x4);
  let n2 = cross(x2 - x4, x2 - x3);
  let u1 = n1 * (1.0 / length(n1));
  let u2 = n2 * (1.0 / length(n2));
  let sine = dot(cross(u1, u2), edge * (1.0 / length(edge)));
  return atan2_accurate(sine, dot(u1, u2));
}

// ∂θ/∂x_k for k = 1..4 (Bridson et al. bending modes), carried through dual numbers.
fn angle_gradient_dual(x1: D3, x2: D3, x3: D3, x4: D3) -> array<D3, 4> {
  let edge = d3_sub(x4, x3);
  let edge_length = d1_sqrt(d3_dot(edge, edge));
  let n1 = d3_cross(d3_sub(x1, x3), d3_sub(x1, x4));
  let n2 = d3_cross(d3_sub(x2, x4), d3_sub(x2, x3));
  let a1 = d3_scale(n1, d1_div(d1(1.0), d3_dot(n1, n1)));
  let a2 = d3_scale(n2, d1_div(d1(1.0), d3_dot(n2, n2)));
  let u1 = d3_scale(a1, edge_length);
  let u2 = d3_scale(a2, edge_length);
  let u3 = d3_add(d3_scale(a1, d1_div(d3_dot(d3_sub(x1, x4), edge), edge_length)), d3_scale(a2, d1_div(d3_dot(d3_sub(x2, x4), edge), edge_length)));
  let u4 = d3_add(d3_scale(a1, d1_div(d1_neg(d3_dot(d3_sub(x1, x3), edge)), edge_length)), d3_scale(a2, d1_div(d1_neg(d3_dot(d3_sub(x2, x3), edge)), edge_length)));
  let flip = d1(-1.0);
  return array<D3, 4>(d3_scale(u1, flip), d3_scale(u2, flip), d3_scale(u3, flip), d3_scale(u4, flip));
}
