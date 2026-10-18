// True residual s = b − A x in f32.
// REPLACE = false: per-workgroup partial sum of sᵀs (the GPU's own drift check).
// REPLACE = true: residual replacement r = s, z = D⁻¹s, partial sums of rᵀz and rᵀr; skipped once the solve has stopped
// or the recursive residual is below the replacement gate.
override REPLACE: bool = false;

@group(0) @binding(0) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read> col_idx: array<u32>;
@group(0) @binding(2) var<storage, read> vals: array<f32>;
@group(0) @binding(3) var<storage, read> rhs: array<f32>;
@group(0) @binding(4) var<storage, read_write> x: array<f32>;
@group(0) @binding(5) var<storage, read_write> partials: array<f32>;
@group(0) @binding(6) var<storage, read_write> dinv: array<f32>;
@group(0) @binding(7) var<storage, read_write> r: array<f32>;
@group(0) @binding(8) var<storage, read_write> z: array<f32>;
@group(0) @binding(9) var<storage, read_write> state: State;

var<workgroup> lane_a: array<f32, 256>;
var<workgroup> lane_b: array<f32, 256>;
var<workgroup> stop: u32;

@compute @workgroup_size(256)
fn cs_main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u,
) {
  if (lid.x == 0u) {
    stop = select(0u, state.status | u32(state.rr <= state.replace_above2 * state.rr0), REPLACE);
  }
  if (workgroupUniformLoad(&stop) != 0u) {
    return;
  }
  let row = gid.x;
  var sums = vec2f(0.0, 0.0);
  if (row < arrayLength(&row_ptr) - 1u) {
    sums = residual_row(row);
  }
  lane_a[lid.x] = sums.x;
  lane_b[lid.x] = sums.y;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (lid.x < stride) {
      lane_a[lid.x] = lane_a[lid.x] + lane_a[lid.x + stride];
      lane_b[lid.x] = lane_b[lid.x] + lane_b[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    partials[2u * wid.x] = lane_a[0];
    partials[2u * wid.x + 1u] = lane_b[0];
  }
}

// (sᵀs, 0) when measuring; (rᵀz, rᵀr) after writing r and z when replacing.
fn residual_row(row: u32) -> vec2f {
  var a0 = 0.0;
  var a1 = 0.0;
  var a2 = 0.0;
  for (var k = row_ptr[row]; k < row_ptr[row + 1u]; k = k + 1u) {
    let c = 3u * col_idx[k];
    let o = 9u * k;
    let x0 = x[c]; let x1 = x[c + 1u]; let x2 = x[c + 2u];
    a0 = a0 + (vals[o] * x0 + vals[o + 1u] * x1 + vals[o + 2u] * x2);
    a1 = a1 + (vals[o + 3u] * x0 + vals[o + 4u] * x1 + vals[o + 5u] * x2);
    a2 = a2 + (vals[o + 6u] * x0 + vals[o + 7u] * x1 + vals[o + 8u] * x2);
  }
  let i = 3u * row;
  let s0 = rhs[i] - a0;
  let s1 = rhs[i + 1u] - a1;
  let s2 = rhs[i + 2u] - a2;
  let ss = s0 * s0 + s1 * s1 + s2 * s2;
  if (!REPLACE) {
    return vec2f(ss, 0.0);
  }
  let d = 9u * row;
  let z0 = dinv[d] * s0 + dinv[d + 1u] * s1 + dinv[d + 2u] * s2;
  let z1 = dinv[d + 3u] * s0 + dinv[d + 4u] * s1 + dinv[d + 5u] * s2;
  let z2 = dinv[d + 6u] * s0 + dinv[d + 7u] * s1 + dinv[d + 8u] * s2;
  r[i] = s0; r[i + 1u] = s1; r[i + 2u] = s2;
  z[i] = z0; z[i + 1u] = z1; z[i + 2u] = z2;
  return vec2f(s0 * z0 + s1 * z1 + s2 * z2, ss);
}
