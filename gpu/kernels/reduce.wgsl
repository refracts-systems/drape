// One workgroup: strided sum of per-workgroup partials, a fixed tree, then the CG scalar update.
// MODE 0 = INIT (rᵀz, rᵀr), 1 = ALPHA (pᵀq), 2 = BETA (rᵀz, rᵀr, convergence), 3 = TRUE (‖b − Ax‖²).
override MODE: u32 = 0u;

@group(0) @binding(0) var<storage, read_write> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> state: State;

var<workgroup> lane_a: array<f32, 256>;
var<workgroup> lane_b: array<f32, 256>;
var<workgroup> stop: u32;

@compute @workgroup_size(256)
fn cs_main(@builtin(local_invocation_id) lid: vec3u) {
  if (lid.x == 0u) {
    stop = select(state.status, 0u, MODE == 0u || MODE == 3u);
  }
  if (workgroupUniformLoad(&stop) != 0u) {
    return;
  }
  let count = arrayLength(&partials) / 2u;
  var a = 0.0;
  var b = 0.0;
  for (var j = lid.x; j < count; j = j + 256u) {
    a = a + partials[2u * j];
    b = b + partials[2u * j + 1u];
  }
  lane_a[lid.x] = a;
  lane_b[lid.x] = b;
  workgroupBarrier();
  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    if (lid.x < stride) {
      lane_a[lid.x] = lane_a[lid.x] + lane_a[lid.x + stride];
      lane_b[lid.x] = lane_b[lid.x] + lane_b[lid.x + stride];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    finish(lane_a[0], lane_b[0]);
  }
}

fn finish(sum_a: f32, sum_b: f32) {
  switch MODE {
    case 0u: { start(sum_a, sum_b); }
    case 1u: { step_length(sum_a); }
    case 2u: { advance(sum_a, sum_b); }
    default: { state.true_rr = sum_a; }
  }
}

fn start(rz: f32, rr: f32) {
  state.rz = rz;
  state.rr = rr;
  state.rr0 = rr;
  state.iteration = 0u;
  state.status = select(0u, 1u, rr <= 0.0);
}

// Bit test, because WGSL lets compilers assume NaN and Inf never occur.
fn is_finite(value: f32) -> bool {
  return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn step_length(pq: f32) {
  let alpha = state.rz / pq;
  state.pq = pq;
  state.alpha = alpha;
  state.status = select(3u, 0u, pq > 0.0 && is_finite(alpha));
}

fn advance(rz: f32, rr: f32) {
  state.beta = rz / state.rz;
  state.rz = rz;
  state.rr = rr;
  state.iteration = state.iteration + 1u;
  if (rr <= state.tol2 * state.rr0) {
    state.status = 1u;
  } else if (state.iteration >= state.max_iterations) {
    state.status = 2u;
  }
}
