#![allow(dead_code)]
/// Scenario packaging for calibrated synthetic experiments.
use serde::{Deserialize, Serialize};

use super::mapping::{
    CalibratedSyntheticParams, CalibrationMappingConfig, JumpContamination, MeanPolicy,
    VariancePolicy,
};
use super::summary::EmpiricalCalibrationProfile;

/// Named scenario presets for Phase 17.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibratedScenarioKind {
    CalmVsTurbulent,
    PersistentStates,
    ShockContaminated,
    AssetSpecific,
}

/// Experiment-ready scenario descriptor.
#[derive(Debug, Clone)]
pub struct CalibratedScenario {
    pub kind: CalibratedScenarioKind,
    pub label: String,
    pub mapping_config: CalibrationMappingConfig,
    pub params: CalibratedSyntheticParams,
}

pub fn preset_mapping(kind: CalibratedScenarioKind) -> CalibrationMappingConfig {
    match kind {
        CalibratedScenarioKind::CalmVsTurbulent => CalibrationMappingConfig {
            k: 2,
            horizon: 2_000,
            mean_policy: MeanPolicy::EmpiricalBaseline,
            variance_policy: VariancePolicy::QuantileAnchored,
            target_durations: vec![25.0, 8.0],
            symmetric_offdiag: true,
            jump: None,
        },
        CalibratedScenarioKind::PersistentStates => CalibrationMappingConfig {
            k: 2,
            horizon: 2_000,
            mean_policy: MeanPolicy::EmpiricalBaseline,
            variance_policy: VariancePolicy::RatioAroundEmpirical {
                low_mult: 0.6,
                high_mult: 1.8,
            },
            target_durations: vec![40.0, 15.0],
            symmetric_offdiag: true,
            jump: None,
        },
        CalibratedScenarioKind::ShockContaminated => CalibrationMappingConfig {
            k: 2,
            horizon: 2_000,
            mean_policy: MeanPolicy::EmpiricalBaseline,
            variance_policy: VariancePolicy::QuantileAnchored,
            target_durations: vec![25.0, 8.0],
            symmetric_offdiag: true,
            jump: Some(JumpContamination {
                jump_prob: 0.01,
                jump_scale_mult: 4.0,
            }),
        },
        CalibratedScenarioKind::AssetSpecific => CalibrationMappingConfig {
            k: 3,
            horizon: 2_500,
            mean_policy: MeanPolicy::SymmetricAroundEmpirical,
            variance_policy: VariancePolicy::QuantileAnchored,
            target_durations: vec![35.0, 14.0, 7.0],
            symmetric_offdiag: true,
            jump: None,
        },
    }
}

pub fn scenario_label(
    kind: &CalibratedScenarioKind,
    profile: &EmpiricalCalibrationProfile,
) -> String {
    let base = match kind {
        CalibratedScenarioKind::CalmVsTurbulent => "calm_turbulent",
        CalibratedScenarioKind::PersistentStates => "persistent_states",
        CalibratedScenarioKind::ShockContaminated => "shock_contaminated",
        CalibratedScenarioKind::AssetSpecific => "asset_specific",
    };
    format!(
        "{}_{}_{}",
        profile.tag.asset, profile.tag.feature_label, base
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_have_valid_k() {
        for k in [
            CalibratedScenarioKind::CalmVsTurbulent,
            CalibratedScenarioKind::PersistentStates,
            CalibratedScenarioKind::ShockContaminated,
            CalibratedScenarioKind::AssetSpecific,
        ] {
            let cfg = preset_mapping(k);
            assert!(cfg.k >= 2);
            assert!(cfg.horizon >= 2);
        }
    }
}
