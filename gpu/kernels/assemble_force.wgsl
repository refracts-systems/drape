// Newton right-hand side −∂Π/∂x on the unknowns, one row per thread: element gradients summed in element
// order; constrained components are zero.
@group(0) @binding(0) var<storage, read> dof_mask: array<u32>;
@group(0) @binding(1) var<storage, read> force_contrib_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> force_contrib_idx: array<u32>;
@group(0) @binding(3) var<storage, read> elem_grad: array<f32>;
@group(0) @binding(4) var<storage, read_write> rhs: array<f32>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  if (row >= arrayLength(&dof_mask)) {
    return;
  }
  for (var d = 0u; d < 3u; d = d + 1u) {
    var sum = 0.0;
    for (var j = force_contrib_ptr[row]; j < force_contrib_ptr[row + 1u]; j = j + 1u) {
      sum = sum + elem_grad[3u * force_contrib_idx[j] + d];
    }
    rhs[3u * row + d] = select(0.0, -sum, ((dof_mask[row] >> d) & 1u) == 1u);
  }
}
