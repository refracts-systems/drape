use crate::drape::{Drape, TangentKind};
use crate::e7_state::{read_state, write_state, State};
use crate::linalg::Vec3;
use std::fs;
use std::path::Path;

/// Fine vertex (i, j) of the (2n − 1)² grid from the coarse n² grid (vertex = j·columns + i): coarse vertices are copied, edge
/// midpoints average their two coarse neighbours, quad centres average the four coarse corners.
fn interpolate(coarse: &[Vec3], coarse_columns: usize, i: usize, j: usize) -> Vec3 {
    let (i0, i1) = (i / 2, i.div_ceil(2));
    let (j0, j1) = (j / 2, j.div_ceil(2));
    let corners = [(i0, j0), (i1, j0), (i0, j1), (i1, j1)];
    let sum = corners.iter().fold([0.0; 3], |acc, &(a, b)| std::array::from_fn(|k| acc[k] + coarse[b * coarse_columns + a][k]));
    sum.map(|v| v / 4.0)
}

/// The prolongation: the average of the four (possibly repeated) coarse corners of every fine vertex. For a vertex at a coarse
/// location all four corners coincide; on an edge midpoint two pairs coincide; at a quad centre all four differ.
pub fn prolong(coarse: &[Vec3], coarse_columns: usize) -> Vec<Vec3> {
    let fine_columns = 2 * (coarse_columns - 1) + 1;
    (0..fine_columns * fine_columns).map(|v| interpolate(coarse, coarse_columns, v % fine_columns, v / fine_columns)).collect()
}

/// Every vertex closer than d̂ to a collider moves along that collider's normal to d = d̂ (one pass over the colliders, in order).
pub fn project_out(drape: &Drape, positions: &mut [Vec3]) -> usize {
    let dhat = drape.contact.barrier.dhat;
    let mut moved = 0;
    for x in positions.iter_mut() {
        let before = *x;
        for collider in &drape.contact.colliders {
            let d = collider.distance(*x);
            if d < dhat {
                let n = collider.normal(*x);
                *x = std::array::from_fn(|k| x[k] + (dhat - d) * n[k]);
            }
        }
        moved += usize::from(*x != before);
    }
    moved
}

/// Vertices at or inside a collider: the start would be infeasible for the barrier.
fn infeasible(drape: &Drape, positions: &[Vec3]) -> Vec<usize> {
    (0..positions.len()).filter(|&v| drape.contact.min_distance(positions[v]) <= 0.0).collect()
}

/// Reads the coarse state at `coarse` (a state prefix, f32 or f64), prolongs it to the mesh of `fine_dir` and writes the start
/// state (same dtype) at `out`, with the coarse solve's final step size.
pub fn run(coarse: &Path, coarse_columns: usize, fine_dir: &Path, out: &Path) -> Result<(), String> {
    let problem = fs::read_to_string(fine_dir.join("problem.json")).map_err(|e| e.to_string())?;
    let drape = Drape::from_problem(&problem, TangentKind::Exact);
    let state = read_state(coarse)?;
    let mut positions = prolong(&state.positions, coarse_columns);
    if positions.len() != drape.mesh.rest.len() {
        return Err(format!("prolonged {} vertices, the fine mesh has {}", positions.len(), drape.mesh.rest.len()));
    }
    let moved = project_out(&drape, &mut positions);
    let bad = infeasible(&drape, &positions);
    println!("prolongation: {moved} vertices projected to d = d̂, {} infeasible", bad.len());
    if !bad.is_empty() {
        return Err(format!("infeasible start after projection: vertices {:?}", &bad[..bad.len().min(20)]));
    }
    write_state(out, &State { positions, final_h: state.final_h, single: state.single })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prolongation_copies_coarse_vertices_and_averages_the_rest() {
        let coarse: Vec<Vec3> = (0..9).map(|v| [(v % 3) as f64, (v / 3) as f64, ((v % 3) * (v / 3)) as f64]).collect();
        let fine = prolong(&coarse, 3);
        assert_eq!(fine.len(), 25);
        assert_eq!(fine[0], coarse[0]);
        assert_eq!(fine[2], coarse[1]);
        assert_eq!(fine[1], [0.5, 0.0, 0.0]);
        assert_eq!(fine[5 + 1], [0.5, 0.5, 0.25]);
        assert_eq!(fine[3 * 5 + 3], [1.5, 1.5, 2.25]);
    }
}
