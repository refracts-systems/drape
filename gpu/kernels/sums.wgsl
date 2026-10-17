// Two-pass sum reduction for the Newton host loop (Experiment 6; the E1 reduce kernels are not touched).
// PASS 0: every workgroup strides over `a` (and `b`) and writes two partial sums.
//   MODE 0: (Σ a², Σ a·b)      MODE 1: (Σ a, Σ |a|)
// PASS 1: one workgroup adds the GROUPS partial pairs and writes the pair to out[OUT_OFFSET], out[OUT_OFFSET + 1].
override MODE: u32 = 0u;
override PASS: u32 = 0u;
override OUT_OFFSET: u32 = 0u;
override GROUPS: u32 = 1u;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

var<workgroup> first: array<f32, 256>;
var<workgroup> second: array<f32, 256>;

fn strided_sums(lane: u32, group: u32, groups: u32) -> vec2f {
  var sums = vec2f(0.0, 0.0);
  let stride = groups * 256u;
  for (var i = group * 256u + lane; i < arrayLength(&a); i = i + stride) {
    let v = a[i];
    let w = b[i];
    sums = sums + select(vec2f(v, abs(v)), vec2f(v * v, v * w), MODE == 0u);
  }
  return sums;
}

@compute @workgroup_size(256)
fn cs_main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) groups: vec3u) {
  var mine = vec2f(0.0, 0.0);
  if (PASS == 0u) {
    mine = strided_sums(lid.x, wid.x, groups.x);
  } else if (lid.x < GROUPS) {
    mine = vec2f(partials[2u * lid.x], partials[2u * lid.x + 1u]);
  }
  first[lid.x] = mine.x;
  second[lid.x] = mine.y;
  workgroupBarrier();
  for (var span = 128u; span > 0u; span = span >> 1u) {
    if (lid.x < span) {
      first[lid.x] = first[lid.x] + first[lid.x + span];
      second[lid.x] = second[lid.x] + second[lid.x + span];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    if (PASS == 0u) {
      partials[2u * wid.x] = first[0];
      partials[2u * wid.x + 1u] = second[0];
    } else {
      out[OUT_OFFSET] = first[0];
      out[OUT_OFFSET + 1u] = second[0];
    }
  }
}
