// Per unknown row: (M/h²)((x − x̂)·αd + ½|αd|²), the exact inertia change, written after the hinge entries of elem_delta.
override DELTA_OFFSET: u32 = 0u;

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> x_hat: array<f32>;
@group(0) @binding(2) var<storage, read> direction: array<f32>;
@group(0) @binding(3) var<storage, read> row_vertex: array<u32>;
@group(0) @binding(4) var<storage, read> row_mass: array<f32>;
@group(0) @binding(5) var<storage, read> step: array<f32>;
@group(0) @binding(6) var<storage, read_write> elem_delta: array<f32>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  if (row >= arrayLength(&row_mass)) {
    return;
  }
  let v = row_vertex[row];
  let offset = vec3f(positions[3u * v], positions[3u * v + 1u], positions[3u * v + 2u]) - vec3f(x_hat[3u * v], x_hat[3u * v + 1u], x_hat[3u * v + 2u]);
  let moved = vec3f(direction[3u * v], direction[3u * v + 1u], direction[3u * v + 2u]) * step[0];
  elem_delta[DELTA_OFFSET + row] = row_mass[row] * (dot(offset, moved) + 0.5 * dot(moved, moved));
}
