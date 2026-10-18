pub type Vec3 = [f64; 3];
pub type Mat3 = [[f64; 3]; 3];

pub fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn scale(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn norm(a: Vec3) -> f64 {
    dot(a, a).sqrt()
}

pub fn outer(a: Vec3, b: Vec3) -> Mat3 {
    let mut m = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            m[r][c] = a[r] * b[c];
        }
    }
    m
}

pub fn mat_add_scaled(target: &mut Mat3, source: &Mat3, factor: f64) {
    for r in 0..3 {
        for c in 0..3 {
            target[r][c] += source[r][c] * factor;
        }
    }
}

pub fn identity_scaled(s: f64) -> Mat3 {
    [[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]]
}

pub fn vector_norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

pub fn difference_norm(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}
