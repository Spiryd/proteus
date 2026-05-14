/// End-to-end Phase 17 calibration workflow report.
use rand::SeedableRng;
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};

use crate::model::simulate;
use crate::model::simulate::JumpParams;

use super::mapping::{CalibratedSyntheticParams, CalibrationMappingConfig, calibrate_to_synthetic};
use super::summary::{EmpiricalCalibrationProfile, EmpiricalSummary, summarize_observation_values};
use super::verify::{
    CalibrationVerification, DEFAULT_SCALE_TOLERANCE, ScaleConsistencyCheck,
    VerificationTargetMask, VerificationTolerance, scale_consistency_check,
    verify_calibration_masked,
};

/// Full calibration artifact: mapping inputs, calibrated parameters, synthetic check.
#[derive(Debug, Clone)]
pub struct CalibrationReport {
    pub empirical_profile: EmpiricalCalibrationProfile,
    pub calibrated: CalibratedSyntheticParams,
    pub synthetic_summary: EmpiricalSummary,
    pub verification: CalibrationVerification,
    /// B′1: dispersion agreement between synthetic and empirical streams.
    pub scale_check: ScaleConsistencyCheck,
    pub rng_seed: u64,
}

/// Serializable lightweight report for export (without full model params object).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationReportView {
    pub asset: String,
    pub feature_label: String,
    /// E3 — bar-aware frequency tag (e.g. `"daily"` or `"intraday_15min"`).
    #[serde(default)]
    pub frequency: String,
    pub horizon: usize,
    pub expected_durations: Vec<f64>,
    pub empirical_n: usize,
    pub synthetic_n: usize,
    pub verification_passed: bool,
    pub verification_notes: Vec<String>,
    /// E4 — calibration mapping notes (mean/variance policy decisions, C′2
    /// ratio-guard actions, EM strategy info, etc.).
    #[serde(default)]
    pub mapping_notes: Vec<String>,
    /// B′1: dispersion agreement between synthetic and empirical streams.
    pub scale_check: ScaleConsistencyCheck,
    /// C′4.3 — per-field verification record bundle.
    #[serde(default)]
    pub field_results: super::verify::FieldResults,
    /// C′4.1 — verifier mask in effect when the report was produced.
    #[serde(default)]
    pub verification_mask: super::verify::VerificationTargetMask,
    /// Seed used to drive the synthetic simulation underlying this report
    /// (recorded for reproducibility).
    #[serde(default)]
    pub rng_seed: u64,
}

impl CalibrationReport {
    pub fn view(&self) -> CalibrationReportView {
        CalibrationReportView {
            asset: self.empirical_profile.tag.asset.clone(),
            feature_label: self.empirical_profile.tag.feature_label.clone(),
            frequency: self.empirical_profile.tag.frequency.clone(),
            horizon: self.calibrated.horizon,
            expected_durations: self.calibrated.expected_durations.clone(),
            empirical_n: self.empirical_profile.summary.n,
            synthetic_n: self.synthetic_summary.n,
            verification_passed: self.verification.within_tolerance,
            verification_notes: self.verification.notes.clone(),
            mapping_notes: self.calibrated.mapping_notes.clone(),
            scale_check: self.scale_check.clone(),
            field_results: self.verification.field_results.clone(),
            verification_mask: self.verification.mask.clone(),
            rng_seed: self.rng_seed,
        }
    }
}

/// Execute calibration workflow:
/// 1) empirical summary profile is given,
/// 2) map -> calibrated synthetic params,
/// 3) simulate synthetic sequence,
/// 4) summarize synthetic y_t,
/// 5) compare against empirical targets.
pub fn run_calibration_workflow(
    empirical_profile: EmpiricalCalibrationProfile,
    mapping_config: CalibrationMappingConfig,
    tol: VerificationTolerance,
    rng_seed: u64,
) -> anyhow::Result<CalibrationReport> {
    let calibrated = calibrate_to_synthetic(&empirical_profile, &mapping_config)?;

    let mut rng = SmallRng::seed_from_u64(rng_seed);

    // Convert CalibrationMappingConfig.jump → JumpParams so the simulation
    // layer (model::simulate) remains independent of the calibration module.
    let jump_params: Option<JumpParams> = calibrated.jump.as_ref().map(|j| JumpParams {
        prob: j.jump_prob,
        scale_mult: j.jump_scale_mult,
    });
    let sim = simulate::simulate_with_jump(
        calibrated.model_params.clone(),
        calibrated.horizon,
        &mut rng,
        jump_params.as_ref(),
    )?;

    let synthetic_summary = summarize_observation_values(&sim.observations);
    let mask = VerificationTargetMask::for_policy(&mapping_config);
    let verification = verify_calibration_masked(
        &empirical_profile.summary,
        &synthetic_summary,
        &tol,
        &mask,
    );
    let scale_check = scale_consistency_check(
        &empirical_profile.summary,
        &synthetic_summary,
        DEFAULT_SCALE_TOLERANCE,
    );

    Ok(CalibrationReport {
        empirical_profile,
        calibrated,
        synthetic_summary,
        verification,
        scale_check,
        rng_seed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::summary::{
        CalibrationDatasetTag, CalibrationPartition, EmpiricalSummary, SummaryTargetSet,
    };
    use crate::features::FeatureFamily;

    fn profile() -> EmpiricalCalibrationProfile {
        EmpiricalCalibrationProfile {
            tag: CalibrationDatasetTag {
                asset: "SPY".to_string(),
                frequency: "daily".to_string(),
                feature_label: "log_return".to_string(),
                partition: CalibrationPartition::TrainOnly,
            },
            feature_family: FeatureFamily::LogReturn,
            targets: SummaryTargetSet::Full,
            summary: EmpiricalSummary {
                n: 1200,
                mean: 0.0,
                variance: 1.0,
                std_dev: 1.0,
                q01: -2.3,
                q05: -1.64,
                q50: 0.0,
                q95: 1.64,
                q99: 2.3,
                tail_freq_q95: 0.05,
                abs_exceed_2std: 0.05,
                acf1: 0.0,
                abs_acf1: 0.15,
                sign_change_rate: 0.5,
                high_episode_mean_duration: 8.0,
                low_episode_mean_duration: 20.0,
            },
            observations: Vec::new(),
        }
    }

    #[test]
    fn workflow_returns_report() {
        let report = run_calibration_workflow(
            profile(),
            CalibrationMappingConfig::default(),
            VerificationTolerance::default(),
            42,
        )
        .unwrap();
        assert!(report.synthetic_summary.n > 0);
        assert_eq!(
            report.calibrated.horizon,
            CalibrationMappingConfig::default().horizon
        );
    }

    #[test]
    fn report_view_serializable_shape() {
        let report = run_calibration_workflow(
            profile(),
            CalibrationMappingConfig::default(),
            VerificationTolerance::default(),
            42,
        )
        .unwrap();
        let v = report.view();
        assert_eq!(v.asset, "SPY");
        assert_eq!(v.feature_label, "log_return");
    }

    #[test]
    fn report_view_surfaces_mapping_notes_and_frequency() {
        let report = run_calibration_workflow(
            profile(),
            CalibrationMappingConfig::default(),
            VerificationTolerance::default(),
            42,
        )
        .unwrap();
        let v = report.view();
        assert_eq!(v.frequency, "daily");
        assert!(!v.mapping_notes.is_empty(), "mapping_notes should be populated");
        // JSON round-trip preserves new fields.
        let json = serde_json::to_string(&v).unwrap();
        assert!(json.contains("mapping_notes"));
        assert!(json.contains("frequency"));
        assert!(json.contains("field_results"));
    }
}
