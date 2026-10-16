// Per hinge: k(θ' − θ)(θ' + θ) for x → x + αd, written after the membrane entries of elem_delta.
override DELTA_OFFSET: u32 = 0u;

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> direction: array<f32>;
@group(0) @binding(2) var<storage, read> hinge_vertices: array<u32>;
@group(0) @binding(3) var<storage, read> hinge_stiffness: array<f32>;
@group(0) @binding(4) var<storage, read> step: array<f32>;
@group(0) @binding(5) var<storage, read_write> elem_delta: array<f32>;

fn position(v: u32) -> vec3f {
  return vec3f(positions[3u * v], positions[3u * v + 1u], positions[3u * v + 2u]);
}

fn moved(v: u32) -> vec3f {
  return position(v) + vec3f(direction[3u * v], direction[3u * v + 1u], direction[3u * v + 2u]) * step[0];
}

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let h = gid.x;
  if (h >= arrayLength(&hinge_stiffness)) {
    return;
  }
  let v1 = hinge_vertices[4u * h];
  let v2 = hinge_vertices[4u * h + 1u];
  let v3 = hinge_vertices[4u * h + 2u];
  let v4 = hinge_vertices[4u * h + 3u];
  let before = hinge_angle(position(v1), position(v2), position(v3), position(v4));
  let after = hinge_angle(moved(v1), moved(v2), moved(v3), moved(v4));
  elem_delta[DELTA_OFFSET + h] = hinge_stiffness[h] * (after - before) * (after + before);
}
