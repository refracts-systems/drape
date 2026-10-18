/// Experiment 4 rules shared by the GPU host loop and the oracle checks: Eisenstat–Walker forcing (choice 2) and the
/// floor-aware Newton stop. The GPU implements them in f32; e4-compare replays its recorded inputs through these.
use crate::io::Json;
use crate::newton::{MU_CAP, MU_START};

pub const ETA_START: f64 = 1e-2;
pub const ETA_MIN: f64 = 1e-7;
pub const ETA_MAX: f64 = 1e-2;
const GAMMA: f64 = 0.9;
const SAFEGUARD_THRESHOLD: f64 = 0.1;
/// Stop once the accepted update is this small relative to the step taken so far (‖αd‖∞ ≤ 1e-4·‖x − x_n‖∞).
pub const UPDATE_STOP: f64 = 1e-4;

/// η_k = 0.9·(‖g_k‖/‖g_{k−1}‖)², safeguarded by 0.9·η_{k−1}² when that exceeds 0.1, clamped to [1e-7, 1e-2].
pub fn next_forcing(previous: f64, residual_ratio: f64) -> f64 {
    let eta = GAMMA * residual_ratio * residual_ratio;
    let safeguard = GAMMA * previous * previous;
    let guarded = if safeguard > SAFEGUARD_THRESHOLD { eta.max(safeguard) } else { eta };
    guarded.clamp(ETA_MIN, ETA_MAX)
}

/// The forcing term of every Newton iteration from the residual norms ‖g_0‖, ‖g_1‖, …: η_0 = 1e-2, then `next_forcing`.
pub fn forcing_sequence(norms: &[f64]) -> Vec<f64> {
    norms.iter().enumerate().scan(ETA_START, |eta, (k, &norm)| {
        if k > 0 {
            *eta = next_forcing(*eta, norm / norms[k - 1]);
        }
        Some(*eta)
    }).collect()
}

/// Factor on the f32 position spacing below which an update cannot move any coordinate meaningfully (E4-final and E5).
pub const RESOLUTION_FACTOR: f64 = 2.0;

/// Spacing of f32 numbers at `max_abs`: 2^(floor(log2(max_abs)) − 23), read from the f32 exponent bits.
pub fn f32_spacing(max_abs: f64) -> f64 {
    let exponent = ((max_abs as f32).to_bits() >> 23) & 0xff;
    2f64.powi(exponent as i32 - 127 - 23)
}

/// ‖αd‖∞ ≤ max(1e-4·‖x − x_n‖∞, 2·res); `resolution` is the recorded f32 spacing (0 for runs without the E4-final rule).
pub fn update_is_small(update_inf: f64, step_inf: f64, resolution: f64) -> bool {
    update_inf <= (UPDATE_STOP * step_inf).max(RESOLUTION_FACTOR * resolution)
}

/// μ used by the WGSL inertia-kernel check: the regularised diagonal block must equal (1 + μ)·m/h².
pub const KERNEL_MU: f64 = 0.37;

/// The rule constants the GPU host loop reads from problem.json (one source of truth with the oracle).
pub fn settings_json(resolution_factor: f64) -> Json {
    Json::new()
        .number("eta_start", ETA_START).number("eta_min", ETA_MIN).number("eta_max", ETA_MAX).number("eta_gamma", GAMMA)
        .number("eta_safeguard", SAFEGUARD_THRESHOLD).number("update_stop", UPDATE_STOP)
        .number("resolution_factor", resolution_factor).number("mu_start", MU_START).number("mu_cap", MU_CAP).number("kernel_mu", KERNEL_MU)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forcing_starts_at_its_maximum_and_follows_the_squared_residual_ratio() {
        let etas = forcing_sequence(&[1.0, 1e-1, 1e-3, 1e-6]);
        assert_eq!(etas[0], ETA_START);
        assert!((etas[1] - 9e-3).abs() < 1e-15, "{}", etas[1]);
        assert!((etas[2] - 9e-5).abs() < 1e-17, "{}", etas[2]);
        assert!((etas[3] - 9e-7).abs() < 1e-19, "{}", etas[3]);
    }

    #[test]
    fn forcing_is_clamped_to_its_interval() {
        assert_eq!(next_forcing(1e-2, 0.9), ETA_MAX);
        assert_eq!(next_forcing(1e-2, 1e-6), ETA_MIN);
    }

    #[test]
    fn update_rule_compares_with_the_step_taken() {
        assert!(update_is_small(1e-9, 1e-4, 0.0));
        assert!(!update_is_small(2e-8, 1e-4, 0.0));
        assert!(update_is_small(1e-7, 1e-4, f32_spacing(0.5)));
    }

    #[test]
    fn f32_spacing_reads_the_binade() {
        assert_eq!(f32_spacing(0.5), 2f64.powi(-24));
        assert_eq!(f32_spacing(0.49999997), 2f64.powi(-25));
        assert_eq!(f32_spacing(1.0), 2f64.powi(-23));
    }
}
