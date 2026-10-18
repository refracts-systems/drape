// Two-pass arg-min for the Newton host loop (Experiment 6): the smallest value of `values` and the index where it occurs
// (ties go to the lower index; NaN values are ignored).
// PASS 0: every workgroup strides over `values` and writes (min, index bits) to partials[2·group].
// PASS 1: one workgroup reduces the GROUPS partial pairs to out[OUT_OFFSET] (value) and out[OUT_OFFSET + 1] (index bits).
override PASS: u32 = 0u;
override OUT_OFFSET: u32 = 0u;
override GROUPS: u32 = 1u;

@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

const NONE: u32 = 0xffffffffu;
var<workgroup> best: array<f32, 256>;
var<workgroup> where_: array<u32, 256>;

fn strided_min(lane: u32, group: u32, groups: u32) -> vec2u {
  var value = 3.0e38;
  var index = NONE;
  let stride = groups * 256u;
  for (var i = group * 256u + lane; i < arrayLength(&values); i = i + stride) {
    if (values[i] < value) {
      value = values[i];
      index = i;
    }
  }
  return vec2u(bitcast<u32>(value), index);
}

fn better(value: f32, index: u32, other_value: f32, other_index: u32) -> bool {
  return other_value < value || (other_value == value && other_index < index);
}

@compute @workgroup_size(256)
fn cs_main(@builtin(local_invocation_id) lid: vec3u, @builtin(workgroup_id) wid: vec3u, @builtin(num_workgroups) groups: vec3u) {
  var mine = vec2u(bitcast<u32>(3.0e38), NONE);
  if (PASS == 0u) {
    mine = strided_min(lid.x, wid.x, groups.x);
  } else if (lid.x < GROUPS) {
    mine = vec2u(bitcast<u32>(partials[2u * lid.x]), bitcast<u32>(partials[2u * lid.x + 1u]));
  }
  best[lid.x] = bitcast<f32>(mine.x);
  where_[lid.x] = mine.y;
  workgroupBarrier();
  for (var span = 128u; span > 0u; span = span >> 1u) {
    if (lid.x < span && better(best[lid.x], where_[lid.x], best[lid.x + span], where_[lid.x + span])) {
      best[lid.x] = best[lid.x + span];
      where_[lid.x] = where_[lid.x + span];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    if (PASS == 0u) {
      partials[2u * wid.x] = best[0];
      partials[2u * wid.x + 1u] = bitcast<f32>(where_[0]);
    } else {
      out[OUT_OFFSET] = best[0];
      out[OUT_OFFSET + 1u] = bitcast<f32>(where_[0]);
    }
  }
}
