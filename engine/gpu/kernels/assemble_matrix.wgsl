// Newton matrix in the E1 BSR layout, one row per thread: each block is the sum of its element tangent
// blocks in element order (no atomics, deterministic). Constrained components become identity rows/columns.
@group(0) @binding(0) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(1) var<storage, read> col_idx: array<u32>;
@group(0) @binding(2) var<storage, read> dof_mask: array<u32>;
@group(0) @binding(3) var<storage, read> block_contrib_ptr: array<u32>;
@group(0) @binding(4) var<storage, read> block_contrib_idx: array<u32>;
@group(0) @binding(5) var<storage, read> elem_hess: array<f32>;
@group(0) @binding(6) var<storage, read_write> vals: array<f32>;

fn is_free(mask: u32, component: u32) -> bool {
  return ((mask >> component) & 1u) == 1u;
}

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
  let row = gid.x;
  if (row >= arrayLength(&row_ptr) - 1u) {
    return;
  }
  for (var k = row_ptr[row]; k < row_ptr[row + 1u]; k = k + 1u) {
    let column = col_idx[k];
    for (var i = 0u; i < 9u; i = i + 1u) {
      var sum = 0.0;
      for (var j = block_contrib_ptr[k]; j < block_contrib_ptr[k + 1u]; j = j + 1u) {
        sum = sum + elem_hess[9u * block_contrib_idx[j] + i];
      }
      let r = i / 3u;
      let c = i % 3u;
      let free = is_free(dof_mask[row], r) && is_free(dof_mask[column], c);
      vals[9u * k + i] = select(select(0.0, 1.0, column == row && r == c), sum, free);
    }
  }
}
