/// Calibration verification: compare synthetic summaries to empirical targets.
use serde::{Deserialize, Serialize};

use super::mapping::{CalibrationMappingConfig, CalibrationStrategy, MeanPolicy};
use super::summary::EmpiricalSummary;

/// Outcome of comparing the dispersion (std-dev) of the synthetic stream
/// against the empirical stream.
///
/// Sim-to-real training requires the synthetic and real feature streams to
/// live on the **same numerical scale** so the EM fitted on synthetic data
/// transfers cleanly to the real test stream.  This struct records the result
/// of that check; it is treated as advisory unless `enforce_strict` is set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScaleConsistencyCheck {
    pub empirical_std: f64,
    pub synthetic_std: f64,
    /// `|syn - emp| / max(|emp|, eps)`
    pub relative_error: f64,
    pub tolerance: f64,
    pub within_tolerance: bool,
}

/// Default relative tolerance for the scale-consistency check (10%).
pub const DEFAULT_SCALE_TOLERANCE: f64 = 0.10;

/// Compare the synthetic and empirical standard deviations.
///
/// Returns `within_tolerance == true` iff `|syn - emp| / max(|emp|, 1e-12) <= tol`.
///
/// **Sim-to-real contract (B′1, see notes/sim_to_real_plan.md).**
/// When the synthetic stream is generated for the purpose of training a model
/// that will then run on the real test partition, the two streams must agree
/// in scale — otherwise the trained Gaussians will misclassify the test
/// stream.  This function is the verifier for that contract.
pub fn scale_consistency_check(
    empirical: &EmpiricalSummary,
    synthetic: &EmpiricalSummary,
    tol: f64,
) -> ScaleConsistencyCheck {
    let denom = empirical.std_dev.abs().max(1e-12);
    let rel = (synthetic.std_dev - empirical.std_dev).abs() / denom;
    ScaleConsistencyCheck {
        empirical_std: empirical.std_dev,
        synthetic_std: synthetic.std_dev,
        relative_error: rel,
        tolerance: tol,
        within_tolerance: rel <= tol,
    }
}

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

/// C′4.1 — Policy-aware verifier mask.  Each boolean toggles whether the
/// corresponding field is **checked** against tolerance bounds.  Masked-off
/// fields are still recorded in the per-field results but do not contribute
/// to the global `within_tolerance` verdict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationTargetMask {
    pub mean: bool,
    pub variance: bool,
    pub q05: bool,
    pub q95: bool,
    pub abs_acf1: bool,
    pub sign_change_rate: bool,
}

impl Default for VerificationTargetMask {
    fn default() -> Self {
        Self {
            mean: true,
            variance: true,
            q05: true,
            q95: true,
            abs_acf1: true,
            sign_change_rate: true,
        }
    }
}

impl VerificationTargetMask {
    /// C′4.2 — Derive a mask from the calibration mapping configuration.
    ///
    /// * `MeanPolicy::ZeroCentered` ⇒ the synthetic mean is fixed at 0 by
    ///   construction; checking it against the empirical mean is a tautology
    ///   that misfires when the real data has non-zero drift, so we disable
    ///   the mean target.
    /// * `CalibrationStrategy::QuickEm { .. }` ⇒ the EM is fitted directly on
    ///   the partition observations, which means dependence summaries
    ///   (`abs_acf1`, `sign_change_rate`) and quantile spread emerge as
    ///   side-effects rather than calibration targets.  We keep those
    ///   targets disabled, leaving the mean / variance / q05 / q95 checks
    ///   as the load-bearing comparisons.
    pub fn for_policy(config: &CalibrationMappingConfig) -> Self {
        let mut mask = Self::default();
        if matches!(config.mean_policy, MeanPolicy::ZeroCentered) {
            mask.mean = false;
        }
        if matches!(config.strategy, CalibrationStrategy::QuickEm { .. }) {
            mask.abs_acf1 = false;
            mask.sign_change_rate = false;
        }
        mask
    }
}

/// Single-field verification record used by `CalibrationVerification::field_results`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldVerification {
    pub checked: bool,
    pub passed: bool,
    pub diff: f64,
    pub tolerance: f64,
}

/// Human-readable verification outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationVerification {
    pub synthetic_n: usize,
    pub empirical_n: usize,
    pub diff: CalibrationDiff,
    pub within_tolerance: bool,
    pub notes: Vec<String>,
    /// C′4.3 — per-field pass/fail records honouring the verifier mask.
    #[serde(default)]
    pub field_results: FieldResults,
    /// Mask in effect when the verifier ran (so consumers can re-derive the
    /// global verdict).
    #[serde(default)]
    pub mask: VerificationTargetMask,
}

/// Per-field verification record bundle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct FieldResults {
    pub mean: Option<FieldVerification>,
    pub variance: Option<FieldVerification>,
    pub q05: Option<FieldVerification>,
    pub q95: Option<FieldVerification>,
    pub abs_acf1: Option<FieldVerification>,
    pub sign_change_rate: Option<FieldVerification>,
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

pub fn verify_calibration_masked(
    empirical: &EmpiricalSummary,
    synthetic: &EmpiricalSummary,
    tol: &VerificationTolerance,
    mask: &VerificationTargetMask,
) -> CalibrationVerification {
    let diff = CalibrationDiff {
        mean_abs_err: (synthetic.mean - empirical.mean).abs(),
        variance_rel_err: rel_err(synthetic.variance, empirical.variance),
        q05_abs_err: (synthetic.q05 - empirical.q05).abs(),
        q95_abs_err: (synthetic.q95 - empirical.q95).abs(),
        abs_acf1_abs_err: (synthetic.abs_acf1 - empirical.abs_acf1).abs(),
        sign_change_abs_err: (synthetic.sign_change_rate - empirical.sign_change_rate).abs(),
    };

    let field = |checked: bool, err: f64, t: f64| FieldVerification {
        checked,
        passed: err <= t,
        diff: err,
        tolerance: t,
    };

    let field_results = FieldResults {
        mean: Some(field(mask.mean, diff.mean_abs_err, tol.mean_abs_max)),
        variance: Some(field(
            mask.variance,
            diff.variance_rel_err,
            tol.variance_rel_max,
        )),
        q05: Some(field(mask.q05, diff.q05_abs_err, tol.quantile_abs_max)),
        q95: Some(field(mask.q95, diff.q95_abs_err, tol.quantile_abs_max)),
        abs_acf1: Some(field(
            mask.abs_acf1,
            diff.abs_acf1_abs_err,
            tol.abs_acf1_abs_max,
        )),
        sign_change_rate: Some(field(
            mask.sign_change_rate,
            diff.sign_change_abs_err,
            tol.sign_change_abs_max,
        )),
    };

    // Global verdict — only checked fields participate.
    let checked_fields = [
        field_results.mean.as_ref(),
        field_results.variance.as_ref(),
        field_results.q05.as_ref(),
        field_results.q95.as_ref(),
        field_results.abs_acf1.as_ref(),
        field_results.sign_change_rate.as_ref(),
    ];
    let within = checked_fields
        .iter()
        .flatten()
        .filter(|f| f.checked)
        .all(|f| f.passed);

    let mut notes = Vec::new();
    if within {
        notes.push("synthetic summaries are within configured tolerance (mask-aware)".to_string());
    } else {
        notes.push(
            "synthetic summaries exceed at least one tolerance bound (mask-aware)".to_string(),
        );
    }
    let masked_off: Vec<&str> = [
        ("mean", mask.mean),
        ("variance", mask.variance),
        ("q05", mask.q05),
        ("q95", mask.q95),
        ("abs_acf1", mask.abs_acf1),
        ("sign_change_rate", mask.sign_change_rate),
    ]
    .into_iter()
    .filter_map(|(name, on)| if on { None } else { Some(name) })
    .collect();
    if !masked_off.is_empty() {
        notes.push(format!("mask disabled fields: {}", masked_off.join(", ")));
    }

    CalibrationVerification {
        synthetic_n: synthetic.n,
        empirical_n: empirical.n,
        diff,
        within_tolerance: within,
        notes,
        field_results,
        mask: mask.clone(),
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
        let r = verify_calibration_masked(
            &e,
            &y,
            &VerificationTolerance::default(),
            &VerificationTargetMask::default(),
        );
        assert!(r.within_tolerance);
    }

    #[test]
    fn verify_fails_when_far() {
        let e = s(0.0, 1.0);
        let mut y = s(2.0, 4.0);
        y.q05 = -5.0;
        let r = verify_calibration_masked(
            &e,
            &y,
            &VerificationTolerance::default(),
            &VerificationTargetMask::default(),
        );
        assert!(!r.within_tolerance);
    }

    #[test]
    fn scale_check_passes_when_close() {
        let e = s(0.0, 1.0);
        let y = s(0.0, 1.05_f64 * 1.05); // std ≈ 1.05 → rel err 0.05
        let c = scale_consistency_check(&e, &y, DEFAULT_SCALE_TOLERANCE);
        assert!(c.within_tolerance, "rel_err={}", c.relative_error);
        assert!((c.relative_error - 0.05).abs() < 1e-9);
    }

    #[test]
    fn scale_check_fails_on_raw_vs_zscored() {
        // Raw log-return (σ ≈ 0.01) vs z-scored (σ = 1.0): the failure mode B′1
        // is designed to catch.
        let emp = s(0.0, 1.0);
        let mut syn_raw = s(0.0, 1.0);
        syn_raw.std_dev = 0.01;
        let c = scale_consistency_check(&emp, &syn_raw, DEFAULT_SCALE_TOLERANCE);
        assert!(!c.within_tolerance);
        assert!(c.relative_error > 0.9);
    }

    #[test]
    fn mask_for_policy_disables_mean_when_zero_centered() {
        let mut cfg = CalibrationMappingConfig::default();
        cfg.mean_policy = MeanPolicy::ZeroCentered;
        let mask = VerificationTargetMask::for_policy(&cfg);
        assert!(!mask.mean);
        assert!(mask.variance);
    }

    #[test]
    fn mask_for_policy_disables_dependence_under_quick_em() {
        let mut cfg = CalibrationMappingConfig::default();
        cfg.strategy = CalibrationStrategy::QuickEm {
            max_iter: 10,
            tol: 1e-5,
        };
        let mask = VerificationTargetMask::for_policy(&cfg);
        assert!(mask.mean);
        assert!(!mask.abs_acf1);
        assert!(!mask.sign_change_rate);
    }

    #[test]
    fn mask_excludes_disabled_field_from_global_verdict() {
        let e = s(0.0, 1.0);
        // mean diverges far but everything else matches.
        let mut y = s(0.0, 1.0);
        y.mean = 5.0;
        let tol = VerificationTolerance::default();
        let mut mask = VerificationTargetMask::default();
        mask.mean = false;
        let r = verify_calibration_masked(&e, &y, &tol, &mask);
        assert!(
            r.within_tolerance,
            "global verdict should ignore masked mean"
        );
        let mean_field = r.field_results.mean.as_ref().unwrap();
        assert!(!mean_field.checked);
        assert!(!mean_field.passed);
    }

    #[test]
    fn field_results_populated_for_all_fields() {
        let e = s(0.0, 1.0);
        let y = s(0.01, 1.05);
        let r = verify_calibration_masked(
            &e,
            &y,
            &VerificationTolerance::default(),
            &VerificationTargetMask::default(),
        );
        assert!(r.field_results.mean.is_some());
        assert!(r.field_results.variance.is_some());
        assert!(r.field_results.q05.is_some());
        assert!(r.field_results.q95.is_some());
        assert!(r.field_results.abs_acf1.is_some());
        assert!(r.field_results.sign_change_rate.is_some());
    }
}
