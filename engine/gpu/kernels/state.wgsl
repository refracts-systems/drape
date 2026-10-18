// Shared CG scalars, prepended by the host to every kernel that reads or writes them.
// replace_above2: residual replacement runs only while rr > replace_above2 · rr0.
// status: 0 running, 1 converged, 2 iteration limit, 3 breakdown (pᵀq ≤ 0 or not finite).
struct State {
  rz: f32, pq: f32, alpha: f32, beta: f32,
  rr: f32, rr0: f32, tol2: f32, true_rr: f32,
  iteration: u32, status: u32, max_iterations: u32, replace_above2: f32,
  pad1: u32, pad2: u32, pad3: u32, pad4: u32,
}
