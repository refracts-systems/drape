// q = A p (BSR, 3×3 row-major blocks); per-workgroup partial sum of pᵀq.
@group(0) @binding(0) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read> col_idx: array<u32>;
@group(0) @binding(2) var<storage, read> vals: array<f32>;
@group(0) @binding(3) var<storage, read_write> p: array<f32>;
@group(0) @binding(4) var<storage, read_write> q: array<f32>;
@group(0) @binding(5) var<storage, read_write> partials: array<f32>;
@group(0) @binding(6) var<storage, read_write> state: State;

var<workgroup> lane: array<f32, 256>;
var<workgroup> stop: u32;

@compute @workgroup_size(256)
fn cs_main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u,
) {
  if (lid.x == 0u) {
    stop = state.status;
  }
  if (workgroupUniformLoad(&stop) != 0u) {
    return;
  }
  let row = gid.x;
  var pq = 0.0;
  if (row < arrayLength(&row_ptr) - 1u) {
    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    for (var k = row_ptr[row]; k < row_ptr[row + 1u]; k = k + 1u) {
      let c = 3u * col_idx[k];
      let o = 9u * k;
      let p0 = p[c]; let p1 = p[c + 1u]; let p2 = p[c + 2u];
      a0 = a0 + (vals[o] * p0 + vals[o + 1u] * p1 + vals[o + 2u] * p2);
      a1 = a1 + (vals[o + 3u] * p0 + vals[o + 4u] * p1 + vals[o + 5u] * p2);
      a2 = a2 + (vals[o + 6u] * p0 + vals[o + 7u] * p1 + vals[o + 8u] * p2);
    }
    let i = 3u * row;
    q[i] = a0; q[i + 1u] = a1; q[i + 2u] = a2;
    pq = p[i] * a0 + p[i + 1u] * a1 + p[i + 2u] * a2;
  }
  lane[lid.x] = pq;
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
