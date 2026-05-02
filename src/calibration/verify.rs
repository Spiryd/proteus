/// Calibration verification: compare synthetic summaries to empirical targets.
use serde::{Deserialize, Serialize};

use super::summary::EmpiricalSummary;

/// Field-wise relative/absolute error summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationDiff {
    pub mean_abs_err: f64,
    pub variance_rel_err: f64,
    pub q05_abs_err: f64,
    pub q95_abs_err: f64,
    pub abs_acf1_abs_err: f64,
    pub sign_change_abs_err: f64,
}

/// Human-readable verification outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationVerification {
    pub synthetic_n: usize,
    pub empirical_n: usize,
    pub diff: CalibrationDiff,
    pub within_tolerance: bool,
    pub notes: Vec<String>,
}

/// Tolerance policy for verification checks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationTolerance {
    pub mean_abs_max: f64,
    pub variance_rel_max: f64,
    pub quantile_abs_max: f64,
    pub abs_acf1_abs_max: f64,
    pub sign_change_abs_max: f64,
}

impl Default for VerificationTolerance {
    fn default() -> Self {
        Self {
            mean_abs_max: 0.25,
            variance_rel_max: 0.50,
            quantile_abs_max: 0.30,
            abs_acf1_abs_max: 0.20,
            sign_change_abs_max: 0.20,
        }
    }
}

pub fn verify_calibration(
    empirical: &EmpiricalSummary,
    synthetic: &EmpiricalSummary,
    tol: &VerificationTolerance,
) -> CalibrationVerification {
    let diff = CalibrationDiff {
        mean_abs_err: (synthetic.mean - empirical.mean).abs(),
        variance_rel_err: rel_err(synthetic.variance, empirical.variance),
        q05_abs_err: (synthetic.q05 - empirical.q05).abs(),
        q95_abs_err: (synthetic.q95 - empirical.q95).abs(),
        abs_acf1_abs_err: (synthetic.abs_acf1 - empirical.abs_acf1).abs(),
        sign_change_abs_err: (synthetic.sign_change_rate - empirical.sign_change_rate).abs(),
    };

    let within = diff.mean_abs_err <= tol.mean_abs_max
        && diff.variance_rel_err <= tol.variance_rel_max
        && diff.q05_abs_err <= tol.quantile_abs_max
        && diff.q95_abs_err <= tol.quantile_abs_max
        && diff.abs_acf1_abs_err <= tol.abs_acf1_abs_max
        && diff.sign_change_abs_err <= tol.sign_change_abs_max;

    let mut notes = Vec::new();
if within {
            notes.push("synthetic summaries are within configured tolerance".to_string());
        } else {
            notes.push("synthetic summaries exceed at least one tolerance bound".to_string());
    }

    CalibrationVerification {
        synthetic_n: synthetic.n,
        empirical_n: empirical.n,
        diff,
        within_tolerance: within,
        notes,
    }
}

fn rel_err(x: f64, target: f64) -> f64 {
    if target.abs() < 1e-12 {
        (x - target).abs()
    } else {
        ((x - target) / target).abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(mean: f64, var: f64) -> EmpiricalSummary {
        EmpiricalSummary {
            n: 1000,
            mean,
            variance: var,
            std_dev: var.sqrt(),
            q01: -2.0,
            q05: -1.0,
            q50: 0.0,
            q95: 1.0,
            q99: 2.0,
            tail_freq_q95: 0.05,
            abs_exceed_2std: 0.05,
            acf1: 0.1,
            abs_acf1: 0.2,
            sign_change_rate: 0.5,
            high_episode_mean_duration: 4.0,
            low_episode_mean_duration: 12.0,
        }
    }

    #[test]
    fn verify_passes_when_close() {
        let e = s(0.0, 1.0);
        let y = s(0.01, 1.1);
        let r = verify_calibration(&e, &y, &VerificationTolerance::default());
        assert!(r.within_tolerance);
    }

    #[test]
    fn verify_fails_when_far() {
        let e = s(0.0, 1.0);
        let mut y = s(2.0, 4.0);
        y.q05 = -5.0;
        let r = verify_calibration(&e, &y, &VerificationTolerance::default());
        assert!(!r.within_tolerance);
    }
}
