// Per element: F, Green strain, stress, nodal gradient ∂W/∂x (3 × vec3) and exact tangent (3 × 3 blocks of 3 × 3).
@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> elem_vertices: array<u32>;
@group(0) @binding(2) var<storage, read> elem_geom: array<f32>;
@group(0) @binding(3) var<storage, read_write> elem_grad: array<f32>;
@group(0) @binding(4) var<storage, read_write> elem_hess: array<f32>;

fn position(v: u32) -> vec3f {
  return vec3f(positions[3u * v], positions[3u * v + 1u], positions[3u * v + 2u]);
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
  let x0 = position(elem_vertices[3u * e]);
  let f = columns_of(position(elem_vertices[3u * e + 1u]) - x0, position(elem_vertices[3u * e + 2u]) - x0, shape_gradient(e, 1u), shape_gradient(e, 2u));
  let stress = stress_of(green_strain(f));
  let area = elem_geom[7u * e + 6u];
  for (var k = 0u; k < 3u; k = k + 1u) {
    write_gradient(e, k, f, stress, area);
    for (var l = 0u; l < 3u; l = l + 1u) {
      write_block(e, k, l, f, stress, area);
    }
  }
}

fn write_gradient(e: u32, k: u32, f: Columns, s: vec3f, area: f32) {
  let g = shape_gradient(e, k);
  let pw = f.wale * s.x + f.course * s.z;
  let pc = f.wale * s.z + f.course * s.y;
  let value = (pw * g.x + pc * g.y) * area;
  let o = 9u * e + 3u * k;
  elem_grad[o] = value.x; elem_grad[o + 1u] = value.y; elem_grad[o + 2u] = value.z;
}

// Derivatives of (E_ww, E_cc, E_wc) with respect to vertex k's position.
struct StrainRows {
  wale: vec3f,
  course: vec3f,
  shear: vec3f,
}

fn strain_rows(g: vec2f, f: Columns) -> StrainRows {
  return StrainRows(f.wale * g.x, f.course * g.y, (f.wale * g.y + f.course * g.x) * 0.5);
}

fn write_block(e: u32, k: u32, l: u32, f: Columns, s: vec3f, area: f32) {
  let gk = shape_gradient(e, k);
  let gl = shape_gradient(e, l);
  let a = strain_rows(gk, f);
  let b = strain_rows(gl, f);
  let geometric = gk.x * (s.x * gl.x + s.z * gl.y) + gk.y * (s.z * gl.x + s.y * gl.y);
  let o = 81u * e + 9u * (3u * k + l);
  for (var r = 0u; r < 3u; r = r + 1u) {
    for (var c = 0u; c < 3u; c = c + 1u) {
      var value = select(0.0, geometric, r == c);
      value = value + a.wale[r] * b.wale[c] * WALE;
      value = value + a.course[r] * b.course[c] * COURSE;
      value = value + a.wale[r] * b.course[c] * COUPLING;
      value = value + a.course[r] * b.wale[c] * COUPLING;
      value = value + a.shear[r] * b.shear[c] * (4.0 * SHEAR);
      elem_hess[o + 3u * r + c] = value * area;
    }
  }
}
