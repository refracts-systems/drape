use crate::bsr::Bsr;

/// Dense-band Cholesky in f64: the oracle's direct solver. Independent of any Krylov code path.
pub struct BandCholesky {
    size: usize,
    bandwidth: usize,
    lower: Vec<f64>,
}

impl BandCholesky {
    pub fn factor(matrix: &Bsr) -> Result<BandCholesky, String> {
        let mut band = BandCholesky::from_matrix(matrix);
        for i in 0..band.size {
            band.factor_row(i)?;
        }
        Ok(band)
    }

    pub fn bandwidth(&self) -> usize {
        self.bandwidth
    }

    fn from_matrix(matrix: &Bsr) -> BandCholesky {
        let size = 3 * matrix.rows();
        let bandwidth = band_of(matrix);
        let mut band = BandCholesky { size, bandwidth, lower: vec![0.0; size * (bandwidth + 1)] };
        for row in 0..matrix.rows() {
            for k in matrix.row_ptr[row]..matrix.row_ptr[row + 1] {
                band.copy_block(row, matrix.col_idx[k], &matrix.vals[9 * k..9 * k + 9]);
            }
        }
        band
    }

    fn copy_block(&mut self, row: usize, column: usize, block: &[f64]) {
        for r in 0..3 {
            for c in 0..3 {
                let (i, j) = (3 * row + r, 3 * column + c);
                if j <= i {
                    let at = self.index(i, j);
                    self.lower[at] = block[3 * r + c];
                }
            }
        }
    }

    fn index(&self, i: usize, j: usize) -> usize {
        i * (self.bandwidth + 1) + (j + self.bandwidth - i)
    }

    fn first(&self, i: usize) -> usize {
        i.saturating_sub(self.bandwidth)
    }

    fn factor_row(&mut self, i: usize) -> Result<(), String> {
        for j in self.first(i)..=i {
            let start = self.first(i).max(self.first(j));
            let (ri, rj) = (self.index(i, start), self.index(j, start));
            let overlap = j - start;
            let product: f64 = (0..overlap).map(|k| self.lower[ri + k] * self.lower[rj + k]).sum();
            let value = self.lower[self.index(i, j)] - product;
            let at = self.index(i, j);
            self.lower[at] = if i == j { positive_sqrt(value, i)? } else { value / self.lower[self.index(j, j)] };
        }
        Ok(())
    }

    pub fn solve(&self, rhs: &[f64]) -> Vec<f64> {
        let mut y = rhs.to_vec();
        for i in 0..self.size {
            let start = self.first(i);
            let s: f64 = (start..i).map(|k| self.lower[self.index(i, k)] * y[k]).sum();
            y[i] = (y[i] - s) / self.lower[self.index(i, i)];
        }
        for i in (0..self.size).rev() {
            let end = (i + self.bandwidth + 1).min(self.size);
            let s: f64 = (i + 1..end).map(|k| self.lower[self.index(k, i)] * y[k]).sum();
            y[i] = (y[i] - s) / self.lower[self.index(i, i)];
        }
        y
    }
}

fn band_of(matrix: &Bsr) -> usize {
    let mut widest = 0;
    for row in 0..matrix.rows() {
        for k in matrix.row_ptr[row]..matrix.row_ptr[row + 1] {
            let column = matrix.col_idx[k];
            if column <= row {
                widest = widest.max(3 * (row - column) + 2);
            }
        }
    }
    widest
}

fn positive_sqrt(value: f64, i: usize) -> Result<f64, String> {
    if value > 0.0 {
        Ok(value.sqrt())
    } else {
        Err(format!("matrix not positive definite at row {i} (pivot {value:e})"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases::{build, CaseKind};
    use crate::linalg::{difference_norm, vector_norm};

    #[test]
    fn cholesky_solves_tension_case() {
        let problem = build(CaseKind::Tension, 11);
        let x = BandCholesky::factor(&problem.matrix).unwrap().solve(&problem.rhs);
        let residual = difference_norm(&problem.matrix.multiply(&x), &problem.rhs) / vector_norm(&problem.rhs);
        assert!(residual < 1e-12, "{residual}");
    }
}
