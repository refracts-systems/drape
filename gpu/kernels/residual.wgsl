// True residual s = b − A x in f32; per-workgroup partial sum of sᵀs (the GPU's own drift check).
@group(0) @binding(0) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read> col_idx: array<u32>;
@group(0) @binding(2) var<storage, read> vals: array<f32>;
@group(0) @binding(3) var<storage, read> rhs: array<f32>;
@group(0) @binding(4) var<storage, read_write> x: array<f32>;
@group(0) @binding(5) var<storage, read_write> partials: array<f32>;

var<workgroup> lane: array<f32, 256>;

@compute @workgroup_size(256)
fn cs_main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u,
) {
  let row = gid.x;
  var ss = 0.0;
  if (row < arrayLength(&row_ptr) - 1u) {
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
    ss = s0 * s0 + s1 * s1 + s2 * s2;
  }
  lane[lid.x] = ss;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (lid.x < stride) {
      lane[lid.x] = lane[lid.x] + lane[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    partials[2u * wid.x] = lane[0];
    partials[2u * wid.x + 1u] = 0.0;
  }
}
