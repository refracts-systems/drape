// p = z + βp, skipped once the solve has stopped.
@group(0) @binding(0) var<storage, read_write> z: array<f32>;
@group(0) @binding(1) var<storage, read_write> state: State;
@group(0) @binding(2) var<storage, read_write> p: array<f32>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  if (state.status != 0u || row >= arrayLength(&z) / 3u) {
    return;
  }
  let beta = state.beta;
  let i = 3u * row;
  p[i] = z[i] + beta * p[i];
  p[i + 1u] = z[i + 1u] + beta * p[i + 1u];
  p[i + 2u] = z[i + 2u] + beta * p[i + 2u];
}
