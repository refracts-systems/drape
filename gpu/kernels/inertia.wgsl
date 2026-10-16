// Per unknown row: inertia gradient (M/h²)(x − x̂) and diagonal block (1 + μ)(M/h²)·I into the shared local arrays,
// plus every active collider barrier: gradient b′n and block b″nnᵀ + b′(I − nnᵀ)/|x − c| (contact.wgsl).
// regularization[0] = μ of the K + μ·M/h² shift (Experiment 4); always 0 in Experiment 3. No colliders before Experiment 5.
override GRAD_OFFSET: u32 = 0u;
override HESS_OFFSET: u32 = 0u;

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> x_hat: array<f32>;
@group(0) @binding(2) var<storage, read> row_vertex: array<u32>;
@group(0) @binding(3) var<storage, read> row_mass: array<f32>;
@group(0) @binding(4) var<storage, read_write> elem_grad: array<f32>;
@group(0) @binding(5) var<storage, read_write> elem_hess: array<f32>;
@group(0) @binding(6) var<storage, read> regularization: array<f32>;
@group(0) @binding(7) var<storage, read> colliders: array<f32>;

fn add_contact(row: u32, x: vec3f, k: u32) {
  let near = proximity(k, x);
  if (near.distance >= activation()) {
    return;
  }
  let first = barrier_first(near.distance);
  let second = barrier_second(near.distance);
  let n = near.normal;
  for (var r = 0u; r < 3u; r = r + 1u) {
    elem_grad[GRAD_OFFSET + 3u * row + r] = elem_grad[GRAD_OFFSET + 3u * row + r] + first * n[r];
    for (var c = 0u; c < 3u; c = c + 1u) {
      let identity = select(0.0, 1.0, r == c);
      let entry = second * n[r] * n[c] + first * near.curvature * (identity - n[r] * n[c]);
      elem_hess[HESS_OFFSET + 9u * row + 3u * r + c] = elem_hess[HESS_OFFSET + 9u * row + 3u * r + c] + entry;
    }
  }
}

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  if (row >= arrayLength(&row_mass)) {
    return;
  }
  let v = row_vertex[row];
  let mass = row_mass[row];
  for (var d = 0u; d < 3u; d = d + 1u) {
    elem_grad[GRAD_OFFSET + 3u * row + d] = mass * (positions[3u * v + d] - x_hat[3u * v + d]);
  }
  for (var i = 0u; i < 9u; i = i + 1u) {
    elem_hess[HESS_OFFSET + 9u * row + i] = select(0.0, mass * (1.0 + regularization[0]), i % 4u == 0u);
  }
  let x = vec3f(positions[3u * v], positions[3u * v + 1u], positions[3u * v + 2u]);
  for (var k = 0u; k < collider_count(); k = k + 1u) {
    add_contact(row, x, k);
  }
}
