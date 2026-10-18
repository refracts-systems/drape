use std::collections::HashMap;

pub struct Mesh {
    pub columns: usize,
    pub rows: usize,
    pub width: f64,
    pub height: f64,
    pub rest: Vec<[f64; 2]>,
    pub triangles: Vec<[usize; 3]>,
    pub hinges: Vec<[usize; 4]>,
}

impl Mesh {
    pub fn grid(columns: usize, rows: usize, width: f64, height: f64) -> Mesh {
        let rest = grid_vertices(columns, rows, width, height);
        let triangles = grid_triangles(columns, rows);
        let hinges = interior_hinges(&rest, &triangles);
        Mesh { columns, rows, width, height, rest, triangles, hinges }
    }

    pub fn is_boundary(&self, vertex: usize) -> bool {
        let i = vertex % self.columns;
        let j = vertex / self.columns;
        i == 0 || j == 0 || i == self.columns - 1 || j == self.rows - 1
    }

    pub fn vertex_areas(&self) -> Vec<f64> {
        let mut areas = vec![0.0; self.rest.len()];
        for triangle in &self.triangles {
            let area = rest_area(&self.rest, triangle);
            for &v in triangle {
                areas[v] += area / 3.0;
            }
        }
        areas
    }
}

fn grid_vertices(columns: usize, rows: usize, width: f64, height: f64) -> Vec<[f64; 2]> {
    let dx = width / (columns - 1) as f64;
    let dy = height / (rows - 1) as f64;
    (0..rows)
        .flat_map(|j| (0..columns).map(move |i| [i as f64 * dx, j as f64 * dy]))
        .collect()
}

fn grid_triangles(columns: usize, rows: usize) -> Vec<[usize; 3]> {
    let mut triangles = Vec::with_capacity(2 * (columns - 1) * (rows - 1));
    for j in 0..rows - 1 {
        for i in 0..columns - 1 {
            triangles.extend(cell_triangles(columns, i, j));
        }
    }
    triangles
}

fn cell_triangles(columns: usize, i: usize, j: usize) -> [[usize; 3]; 2] {
    let v00 = j * columns + i;
    let v10 = v00 + 1;
    let v01 = v00 + columns;
    let v11 = v01 + 1;
    if (i + j) % 2 == 0 {
        [[v00, v10, v11], [v00, v11, v01]]
    } else {
        [[v00, v10, v01], [v10, v11, v01]]
    }
}

pub fn rest_area(rest: &[[f64; 2]], triangle: &[usize; 3]) -> f64 {
    let [a, b, c] = triangle.map(|v| rest[v]);
    0.5 * cross2(sub2(b, a), sub2(c, a)).abs()
}

fn sub2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn cross2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

/// Hinge vertex order: [left opposite, right opposite, edge start, edge end].
fn interior_hinges(rest: &[[f64; 2]], triangles: &[[usize; 3]]) -> Vec<[usize; 4]> {
    let mut edges: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for triangle in triangles {
        for k in 0..3 {
            let (a, b, opposite) = (triangle[k], triangle[(k + 1) % 3], triangle[(k + 2) % 3]);
            edges.entry((a.min(b), a.max(b))).or_default().push(opposite);
        }
    }
    let mut hinges: Vec<[usize; 4]> = edges
        .into_iter()
        .filter(|(_, opposites)| opposites.len() == 2)
        .map(|((a, b), opposites)| orient_hinge(rest, a, b, opposites[0], opposites[1]))
        .collect();
    hinges.sort();
    hinges
}

fn orient_hinge(rest: &[[f64; 2]], a: usize, b: usize, first: usize, second: usize) -> [usize; 4] {
    let is_left = cross2(sub2(rest[b], rest[a]), sub2(rest[first], rest[a])) > 0.0;
    if is_left {
        [first, second, a, b]
    } else {
        [second, first, a, b]
    }
}
