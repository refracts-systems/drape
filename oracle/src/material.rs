/// Orthotropic St. Venant–Kirchhoff membrane in material axes (wale = x, course = y),
/// plus directional hinge bending. Units: N/m for membrane, N·m for bending, kg/m² density.
#[derive(Clone, Copy)]
pub struct Material {
    pub wale: f64,
    pub course: f64,
    pub coupling: f64,
    pub shear: f64,
    pub bend_wale: f64,
    pub bend_course: f64,
    pub density: f64,
}

impl Material {
    pub fn knit_baseline() -> Material {
        Material {
            wale: 1000.0,
            course: 300.0,
            coupling: 100.0,
            shear: 50.0,
            bend_wale: 2.0e-6,
            bend_course: 1.0e-6,
            density: 0.13,
        }
    }

    /// Second Piola–Kirchhoff stress (S_ww, S_cc, S_wc) from Green strain (E_ww, E_cc, E_wc).
    pub fn stress(&self, strain: [f64; 3]) -> [f64; 3] {
        let [eww, ecc, ewc] = strain;
        [
            self.wale * eww + self.coupling * ecc,
            self.course * ecc + self.coupling * eww,
            2.0 * self.shear * ewc,
        ]
    }

    /// Energy density W(E) per unit rest area.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn energy_density(&self, strain: [f64; 3]) -> f64 {
        let [eww, ecc, ewc] = strain;
        0.5 * self.wale * eww * eww
            + 0.5 * self.course * ecc * ecc
            + self.coupling * eww * ecc
            + 2.0 * self.shear * ewc * ewc
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn is_convex(&self) -> bool {
        self.coupling * self.coupling < self.wale * self.course && self.shear > 0.0
    }

    /// Bending stiffness of a hinge whose rest edge has unit direction (dx, dy).
    pub fn hinge_bending(&self, direction: [f64; 2]) -> f64 {
        self.bend_wale * direction[1] * direction[1] + self.bend_course * direction[0] * direction[0]
    }
}
