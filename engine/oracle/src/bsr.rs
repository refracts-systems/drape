use crate::linalg::Mat3;

/// Block sparse row matrix with 3×3 blocks stored row-major (9 values per block).
pub struct Bsr {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub vals: Vec<f64>,
}

impl Bsr {
    /// Pattern from coupling groups: every pair of rows inside one group shares a block.
    pub fn with_pattern(rows: usize, groups: &[Vec<usize>]) -> Bsr {
        let mut columns: Vec<Vec<usize>> = (0..rows).map(|r| vec![r]).collect();
        for group in groups {
            for &r in group {
                columns[r].extend(group.iter().copied());
            }
        }
        let mut row_ptr = vec![0];
        let mut col_idx = Vec::new();
        for mut row in columns {
            row.sort_unstable();
            row.dedup();
            col_idx.extend(row);
            row_ptr.push(col_idx.len());
        }
        let vals = vec![0.0; 9 * col_idx.len()];
        Bsr { row_ptr, col_idx, vals }
    }

    pub fn rows(&self) -> usize {
        self.row_ptr.len() - 1
    }

    pub fn blocks(&self) -> usize {
        self.col_idx.len()
    }

    pub fn position(&self, row: usize, column: usize) -> usize {
        let (start, end) = (self.row_ptr[row], self.row_ptr[row + 1]);
        let offset = self.col_idx[start..end].binary_search(&column).expect("block outside sparsity pattern");
        start + offset
    }

    pub fn add(&mut self, row: usize, column: usize, block: &Mat3) {
        let base = 9 * self.position(row, column);
        for r in 0..3 {
            for c in 0..3 {
                self.vals[base + 3 * r + c] += block[r][c];
            }
        }
    }

    pub fn multiply(&self, x: &[f64]) -> Vec<f64> {
        let mut y = vec![0.0; x.len()];
        for row in 0..self.rows() {
            for k in self.row_ptr[row]..self.row_ptr[row + 1] {
                let column = self.col_idx[k];
                for r in 0..3 {
                    let v = &self.vals[9 * k + 3 * r..9 * k + 3 * r + 3];
                    y[3 * row + r] += v[0] * x[3 * column] + v[1] * x[3 * column + 1] + v[2] * x[3 * column + 2];
                }
            }
        }
        y
    }

    pub fn diagonal_positions(&self) -> Vec<usize> {
        (0..self.rows()).map(|r| self.position(r, r)).collect()
    }

    /// Largest |A_ij - A_ji| relative to the largest entry.
    pub fn asymmetry(&self) -> f64 {
        let largest = self.vals.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut worst = 0.0f64;
        for row in 0..self.rows() {
            for k in self.row_ptr[row]..self.row_ptr[row + 1] {
                let mirror = self.position(self.col_idx[k], row);
                for r in 0..3 {
                    for c in 0..3 {
                        worst = worst.max((self.vals[9 * k + 3 * r + c] - self.vals[9 * mirror + 3 * c + r]).abs());
                    }
                }
            }
        }
        worst / largest
    }

    pub fn rounded_to_f32(&self) -> Bsr {
        let vals = self.vals.iter().map(|&v| v as f32 as f64).collect();
        Bsr { row_ptr: self.row_ptr.clone(), col_idx: self.col_idx.clone(), vals }
    }
}
