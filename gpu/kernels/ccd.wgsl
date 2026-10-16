// Per unknown row: the largest α ∈ (0, 1] with d(x + αp) ≥ KEEP·d(x) for every collider (contact.wgsl); the host takes the
// minimum over rows as the line-search start α_max. Plane: linear root. Sphere: |q + αp| = w + KEEP·d0, smallest positive root
// of a·α² + b·α + c with c = (1 − KEEP)·d0·(|q| + w + KEEP·d0), in the cancellation-free form 2c / (−b + √disc).
override KEEP: f32 = 0.1;

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> direction: array<f32>;
@group(0) @binding(2) var<storage, read> row_vertex: array<u32>;
@group(0) @binding(3) var<storage, read> colliders: array<f32>;
@group(0) @binding(4) var<storage, read_write> bounds: array<f32>;

fn sphere_step(k: u32, x: vec3f, p: vec3f, d0: f32) -> f32 {
  let q = x - collider_point(k);
  let target_radius = collider_size(k) + KEEP * d0;
  let a = dot(p, p);
  let b = 2.0 * dot(q, p);
  let c = (1.0 - KEEP) * d0 * (length(q) + target_radius);
  let discriminant = b * b - 4.0 * a * c;
  if (b >= 0.0 || discriminant < 0.0) {
    return 1.0;
  }
  return min(1.0, 2.0 * c / (-b + sqrt(discriminant)));
}

fn collider_step(k: u32, x: vec3f, p: vec3f) -> f32 {
  let d0 = proximity(k, x).distance;
  if (!is_plane(k)) {
    return sphere_step(k, x, p, d0);
  }
  let rate = dot(collider_point(k), p);
  if (rate >= 0.0) {
    return 1.0;
  }
  return min(1.0, (1.0 - KEEP) * d0 / -rate);
}

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  if (row >= arrayLength(&row_vertex)) {
    return;
  }
  let v = row_vertex[row];
  let x = vec3f(positions[3u * v], positions[3u * v + 1u], positions[3u * v + 2u]);
  let p = vec3f(direction[3u * v], direction[3u * v + 1u], direction[3u * v + 2u]);
  var alpha = 1.0;
  for (var k = 0u; k < collider_count(); k = k + 1u) {
    alpha = min(alpha, collider_step(k, x, p));
  }
  bounds[row] = alpha;
}
