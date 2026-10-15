// D⁻¹: inverse of each 3×3 diagonal block (adjugate / determinant), same order as oracle pcg::invert3.
@group(0) @binding(0) var<storage, read> vals: array<f32>;
@group(0) @binding(1) var<storage, read> diag_idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> dinv: array<f32>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  if (row >= arrayLength(&diag_idx)) {
    return;
  }
  let o = 9u * diag_idx[row];
  let a = vals[o]; let b = vals[o + 1u]; let c = vals[o + 2u];
  let d = vals[o + 3u]; let e = vals[o + 4u]; let f = vals[o + 5u];
  let g = vals[o + 6u]; let h = vals[o + 7u]; let i = vals[o + 8u];
  let c00 = e * i - f * h;
  let c01 = c * h - b * i;
  let c02 = b * f - c * e;
  let c10 = f * g - d * i;
  let c11 = a * i - c * g;
  let c12 = c * d - a * f;
  let c20 = d * h - e * g;
  let c21 = b * g - a * h;
  let c22 = a * e - b * d;
  let det = a * c00 + b * c10 + c * c20;
  let w = 9u * row;
  dinv[w] = c00 / det; dinv[w + 1u] = c01 / det; dinv[w + 2u] = c02 / det;
  dinv[w + 3u] = c10 / det; dinv[w + 4u] = c11 / det; dinv[w + 5u] = c12 / det;
  dinv[w + 6u] = c20 / det; dinv[w + 7u] = c21 / det; dinv[w + 8u] = c22 / det;
}
