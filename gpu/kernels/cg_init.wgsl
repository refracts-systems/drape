// x = 0, r = b, z = D⁻¹r, p = z; per-workgroup partial sums of rᵀz and rᵀr.
@group(0) @binding(0) var<storage, read> rhs: array<f32>;
@group(0) @binding(1) var<storage, read_write> dinv: array<f32>;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> r: array<f32>;
@group(0) @binding(4) var<storage, read_write> z: array<f32>;
@group(0) @binding(5) var<storage, read_write> p: array<f32>;
@group(0) @binding(6) var<storage, read_write> partials: array<f32>;

var<workgroup> lane_a: array<f32, 256>;
var<workgroup> lane_b: array<f32, 256>;

@compute @workgroup_size(256)
fn cs_main(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
  @builtin(workgroup_id) wid: vec3u,
) {
  let row = gid.x;
  var rz = 0.0;
  var rr = 0.0;
  if (row < arrayLength(&dinv) / 9u) {
    let i = 3u * row;
    let d = 9u * row;
    let r0 = rhs[i]; let r1 = rhs[i + 1u]; let r2 = rhs[i + 2u];
    let z0 = dinv[d] * r0 + dinv[d + 1u] * r1 + dinv[d + 2u] * r2;
    let z1 = dinv[d + 3u] * r0 + dinv[d + 4u] * r1 + dinv[d + 5u] * r2;
    let z2 = dinv[d + 6u] * r0 + dinv[d + 7u] * r1 + dinv[d + 8u] * r2;
    x[i] = 0.0; x[i + 1u] = 0.0; x[i + 2u] = 0.0;
    r[i] = r0; r[i + 1u] = r1; r[i + 2u] = r2;
    z[i] = z0; z[i + 1u] = z1; z[i + 2u] = z2;
    p[i] = z0; p[i + 1u] = z1; p[i + 2u] = z2;
    rz = r0 * z0 + r1 * z1 + r2 * z2;
    rr = r0 * r0 + r1 * r1 + r2 * r2;
  }
  lane_a[lid.x] = rz;
  lane_b[lid.x] = rr;
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
