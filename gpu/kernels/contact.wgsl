// Vertex vs analytic-collider IPC barrier (Experiment 5), shared by the inertia, inertia-energy-change and CCD kernels.
// colliders: [count, d̂, κ, 0, then 8 floats per collider: kind, a, b, c, w, 0, 0, 0];
// kind 1: plane d = n·x − w with n = (a, b, c); kind 2: sphere d = |x − c| − w with c = (a, b, c).
// count 0 (Experiments 3 and 4) skips every contact term.

struct Proximity {
  distance: f32,
  normal: vec3f,
  curvature: f32,
}

fn collider_count() -> u32 {
  return u32(colliders[0]);
}

fn activation() -> f32 {
  return colliders[1];
}

fn collider_point(k: u32) -> vec3f {
  let base = 4u + 8u * k;
  return vec3f(colliders[base + 1u], colliders[base + 2u], colliders[base + 3u]);
}

fn is_plane(k: u32) -> bool {
  return colliders[4u + 8u * k] == 1.0;
}

fn collider_size(k: u32) -> f32 {
  return colliders[4u + 8u * k + 4u];
}

// Distance, unit normal ∇d and curvature 1/|x − c| (sphere; 0 for the plane, where ∇²d = 0).
fn proximity(k: u32, x: vec3f) -> Proximity {
  let a = collider_point(k);
  if (is_plane(k)) {
    return Proximity(dot(a, x) - collider_size(k), a, 0.0);
  }
  let q = x - a;
  let len = length(q);
  return Proximity(len - collider_size(k), q / len, 1.0 / len);
}

// ln(d/d̂). On [d̂/2, d̂) WGSL log is only accurate to 2^-21 absolute, so there it is 2·atanh(s), s = u/(2 + u),
// u = (d − d̂)/d̂ (|s| ≤ 1/3, series to s^17); below d̂/2 the built-in log is accurate to 3 ULP.
fn log_ratio(d: f32) -> f32 {
  let dhat = activation();
  if (d < 0.5 * dhat) {
    return log(d / dhat);
  }
  let u = (d - dhat) / dhat;
  let s = u / (2.0 + u);
  let s2 = s * s;
  var series = 1.0 / 17.0;
  for (var k = 15.0; k >= 1.0; k = k - 2.0) {
    series = series * s2 + 1.0 / k;
  }
  return 2.0 * s * series;
}

// b(d) = −κ (d − d̂)² ln(d/d̂) on d < d̂, and its first two derivatives.
fn barrier(d: f32) -> f32 {
  if (d >= activation()) {
    return 0.0;
  }
  let t = d - activation();
  return -colliders[2] * t * t * log_ratio(d);
}

fn barrier_first(d: f32) -> f32 {
  if (d >= activation()) {
    return 0.0;
  }
  let t = d - activation();
  return -colliders[2] * (2.0 * t * log_ratio(d) + t * t / d);
}

fn barrier_second(d: f32) -> f32 {
  if (d >= activation()) {
    return 0.0;
  }
  let t = d - activation();
  return -colliders[2] * (2.0 * log_ratio(d) + 4.0 * t / d - t * t / (d * d));
}

fn contact_energy(x: vec3f) -> f32 {
  var energy = 0.0;
  for (var k = 0u; k < collider_count(); k = k + 1u) {
    energy = energy + barrier(proximity(k, x).distance);
  }
  return energy;
}
