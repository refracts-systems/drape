// Per element: W(x + αd) − W(x) = S(E):ΔE + W(ΔE), exact for St. Venant–Kirchhoff and free of the
// cancellation in subtracting two energies. step[0] = α; direction holds d per vertex.
@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> direction: array<f32>;
@group(0) @binding(2) var<storage, read> elem_vertices: array<u32>;
@group(0) @binding(3) var<storage, read> elem_geom: array<f32>;
@group(0) @binding(4) var<storage, read> step: array<f32>;
@group(0) @binding(5) var<storage, read_write> elem_delta: array<f32>;

fn position(v: u32) -> vec3f {
  return vec3f(positions[3u * v], positions[3u * v + 1u], positions[3u * v + 2u]);
}

fn displacement(v: u32) -> vec3f {
  return vec3f(direction[3u * v], direction[3u * v + 1u], direction[3u * v + 2u]);
}

fn shape_gradient(e: u32, k: u32) -> vec2f {
  return vec2f(elem_geom[7u * e + 2u * k], elem_geom[7u * e + 2u * k + 1u]);
}

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let e = gid.x;
  if (e >= arrayLength(&elem_vertices) / 3u) {
    return;
  }
  let v0 = elem_vertices[3u * e];
  let v1 = elem_vertices[3u * e + 1u];
  let v2 = elem_vertices[3u * e + 2u];
  let g1 = shape_gradient(e, 1u);
  let g2 = shape_gradient(e, 2u);
  let f = columns_of(position(v1) - position(v0), position(v2) - position(v0), g1, g2);
  let d = columns_of(displacement(v1) - displacement(v0), displacement(v2) - displacement(v0), g1, g2);
  let dw = d.wale * step[0];
  let dc = d.course * step[0];
  let change = vec3f(
    dot(f.wale, dw) + 0.5 * dot(dw, dw),
    dot(f.course, dc) + 0.5 * dot(dc, dc),
    0.5 * (dot(f.wale, dc) + dot(dw, f.course) + dot(dw, dc)),
  );
  let s = stress_of(green_strain(f));
  let first = s.x * change.x + s.y * change.y + 2.0 * s.z * change.z;
  elem_delta[e] = elem_geom[7u * e + 6u] * (first + energy_density(change));
}
