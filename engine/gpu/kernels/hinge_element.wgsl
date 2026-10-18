// Per hinge: gradient 2kθ∇θ (4 × vec3) and Hessian (4 × 4 blocks of 3 × 3) into the shared local arrays.
// EXACT = true: 2k(∇θ∇θᵀ + θ∇²θ), ∇²θ by 12 forward-mode passes. EXACT = false: Gauss–Newton 2k∇θ∇θᵀ.
override GRAD_OFFSET: u32 = 0u;
override HESS_OFFSET: u32 = 0u;
override EXACT: bool = true;

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> hinge_vertices: array<u32>;
@group(0) @binding(2) var<storage, read> hinge_stiffness: array<f32>;
@group(0) @binding(3) var<storage, read_write> elem_grad: array<f32>;
@group(0) @binding(4) var<storage, read_write> elem_hess: array<f32>;

fn position(v: u32) -> vec3f {
  return vec3f(positions[3u * v], positions[3u * v + 1u], positions[3u * v + 2u]);
}

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let h = gid.x;
  if (h >= arrayLength(&hinge_stiffness)) {
    return;
  }
  var x: array<vec3f, 4>;
  for (var k = 0u; k < 4u; k = k + 1u) {
    x[k] = position(hinge_vertices[4u * h + k]);
  }
  let theta = hinge_angle(x[0], x[1], x[2], x[3]);
  let factor = 2.0 * hinge_stiffness[h];
  var g = angle_gradient_dual(D3(x[0], vec3f(0.0)), D3(x[1], vec3f(0.0)), D3(x[2], vec3f(0.0)), D3(x[3], vec3f(0.0)));
  for (var k = 0u; k < 4u; k = k + 1u) {
    let value = g[k].v * (factor * theta);
    let o = GRAD_OFFSET + 12u * h + 3u * k;
    elem_grad[o] = value.x; elem_grad[o + 1u] = value.y; elem_grad[o + 2u] = value.z;
  }
  for (var column = 0u; column < 12u; column = column + 1u) {
    write_column(h, column, x, g, theta, factor);
  }
}

// One column (vertex l, component c) of the hinge Hessian.
fn write_column(h: u32, column: u32, x: array<vec3f, 4>, g: array<D3, 4>, theta: f32, factor: f32) {
  let l = column / 3u;
  let c = column % 3u;
  var curvature: array<D3, 4>;
  if (EXACT) {
    curvature = seeded_gradient(x, l, c);
  }
  var gradient = g;
  for (var k = 0u; k < 4u; k = k + 1u) {
    for (var r = 0u; r < 3u; r = r + 1u) {
      let second = select(0.0, theta * curvature[k].t[r], EXACT);
      elem_hess[HESS_OFFSET + 144u * h + 9u * (4u * k + l) + 3u * r + c] = factor * (gradient[k].v[r] * gradient[l].v[c] + second);
    }
  }
}

// ∇θ with vertex l, component c seeded: its .t parts are column (l, c) of ∇²θ.
fn seeded_gradient(x: array<vec3f, 4>, l: u32, c: u32) -> array<D3, 4> {
  var source = x;
  var points: array<D3, 4>;
  for (var k = 0u; k < 4u; k = k + 1u) {
    points[k] = D3(source[k], select(vec3f(0.0), vec3f(f32(c == 0u), f32(c == 1u), f32(c == 2u)), k == l));
  }
  return angle_gradient_dual(points[0], points[1], points[2], points[3]);
}
