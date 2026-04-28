#![allow(dead_code)]
/// Calibration mapping: empirical summary functionals -> synthetic generator parameters.
///
/// This module implements the operator
///
///   K: (T_1^real, ..., T_m^real) -> theta
///
/// where `theta` contains the synthetic Markov-switching generator settings.
use serde::{Deserialize, Serialize};

use crate::model::ModelParams;

use super::summary::{EmpiricalCalibrationProfile, SummaryTargetSet};

/// How regime means are assigned from empirical summaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MeanPolicy {
    ZeroCentered,
    SymmetricAroundEmpirical,
    EmpiricalBaseline,
}

/// How regime variances are anchored to empirical summaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VariancePolicy {
    /// Low/high variances from empirical low/high quantile scales.
    QuantileAnchored,
    /// Fixed ratio around empirical variance.
    RatioAroundEmpirical { low_mult: f64, high_mult: f64 },
}

/// Optional jump contamination settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JumpContamination {
    /// With probability `jump_prob`, add an outlier shock.
    pub jump_prob: f64,
    /// Shock std multiplier relative to sqrt(regime_variance).
    pub jump_scale_mult: f64,
}

/// Synthetic generator parameters produced by calibration.
#[derive(Debug, Clone)]
pub struct CalibratedSyntheticParams {
    pub model_params: ModelParams,
    pub horizon: usize,
    pub expected_durations: Vec<f64>,
    pub jump: Option<JumpContamination>,
    pub mapping_notes: Vec<String>,
}

/// Mapping configuration for phase-17 calibration.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct CalibrationMappingConfig {
    pub k: usize,
    pub horizon: usize,
    pub mean_policy: MeanPolicy,
    pub variance_policy: VariancePolicy,

    /// Target expected durations d_j used in p_jj = 1 - 1/d_j.
    /// If empty, defaults are inferred from empirical episode durations.
    pub target_durations: Vec<f64>,

    /// Off-diagonal row allocation style.
    pub symmetric_offdiag: bool,

    /// Optional jump contamination settings.
    pub jump: Option<JumpContamination>,
}

impl Default for CalibrationMappingConfig {
    fn default() -> Self {
        Self {
            k: 2,
            horizon: 2_000,
            mean_policy: MeanPolicy::EmpiricalBaseline,
            variance_policy: VariancePolicy::QuantileAnchored,
            target_durations: vec![],
            symmetric_offdiag: true,
            jump: None,
        }
    }
}

/// Build calibrated synthetic params from empirical calibration profile.
pub fn calibrate_to_synthetic(
    profile: &EmpiricalCalibrationProfile,
    config: &CalibrationMappingConfig,
) -> anyhow::Result<CalibratedSyntheticParams> {
    if config.k < 2 {
        anyhow::bail!("calibration mapping requires k>=2, got {}", config.k);
    }
    if config.horizon < 2 {
        anyhow::bail!("calibration horizon must be >=2, got {}", config.horizon);
    }

    let mut notes = Vec::new();

    let means = map_means(profile, config);
    notes.push(format!("means mapped with policy {:?}", config.mean_policy));

    let variances = map_variances(profile, config)?;
    notes.push(format!(
        "variances mapped with policy {:?}",
        config.variance_policy
    ));

    let durations = if config.target_durations.is_empty() {
        infer_durations(profile, config.k)
    } else {
        config.target_durations.clone()
    };
    if durations.len() != config.k {
        anyhow::bail!(
            "target duration length {} does not match k={}.",
            durations.len(),
            config.k
        );
    }

    let transition_rows = durations_to_transition_rows(&durations, config.symmetric_offdiag)?;
    notes.push("transition rows mapped via p_jj = 1 - 1/d_j".to_string());

    let pi = vec![1.0 / config.k as f64; config.k];
    let model_params = ModelParams::new(pi, transition_rows, means, variances);
    model_params.validate()?;

    if matches!(profile.targets, SummaryTargetSet::MarginalOnly) {
        notes.push("targets=MarginalOnly: persistence mapping treated as weak prior".to_string());
    }

    Ok(CalibratedSyntheticParams {
        model_params,
        horizon: config.horizon,
        expected_durations: durations,
        jump: config.jump.clone(),
        mapping_notes: notes,
    })
}

fn map_means(profile: &EmpiricalCalibrationProfile, config: &CalibrationMappingConfig) -> Vec<f64> {
    let m = profile.summary.mean;
    match config.mean_policy {
        MeanPolicy::ZeroCentered => {
            // Symmetric small separation around 0.
            if config.k == 2 {
                vec![
                    -0.25 * profile.summary.std_dev,
                    0.25 * profile.summary.std_dev,
                ]
            } else {
                centered_grid(config.k, 0.5 * profile.summary.std_dev)
            }
        }
        MeanPolicy::SymmetricAroundEmpirical => {
            if config.k == 2 {
                vec![
                    m - 0.25 * profile.summary.std_dev,
                    m + 0.25 * profile.summary.std_dev,
                ]
            } else {
                centered_grid(config.k, 0.5 * profile.summary.std_dev)
                    .into_iter()
                    .map(|x| x + m)
                    .collect()
            }
        }
        MeanPolicy::EmpiricalBaseline => vec![m; config.k],
    }
}

fn map_variances(
    profile: &EmpiricalCalibrationProfile,
    config: &CalibrationMappingConfig,
) -> anyhow::Result<Vec<f64>> {
    let var = profile.summary.variance.max(1e-12);
    let std = var.sqrt();
    let vals = match &config.variance_policy {
        VariancePolicy::RatioAroundEmpirical {
            low_mult,
            high_mult,
        } => {
            if *low_mult <= 0.0 || *high_mult <= 0.0 {
                anyhow::bail!("variance multipliers must be >0");
            }
            if config.k == 2 {
                vec![var * low_mult, var * high_mult]
            } else {
                (0..config.k)
                    .map(|i| {
                        let frac = i as f64 / (config.k - 1) as f64;
                        let mult = low_mult + frac * (high_mult - low_mult);
                        var * mult
                    })
                    .collect()
            }
        }
        VariancePolicy::QuantileAnchored => {
            // Convert quantile spread into low/high sigma anchors.
            let low_sigma = ((profile.summary.q50 - profile.summary.q05).abs() / 1.64485).max(1e-6);
            let high_sigma =
                ((profile.summary.q95 - profile.summary.q50).abs() / 1.64485).max(1e-6);
            if config.k == 2 {
                vec![low_sigma * low_sigma, high_sigma * high_sigma]
            } else {
                (0..config.k)
                    .map(|i| {
                        let frac = i as f64 / (config.k - 1) as f64;
                        let s = low_sigma + frac * (high_sigma - low_sigma);
                        s * s
                    })
                    .collect()
            }
        }
    };

    if vals.iter().any(|v| *v <= 0.0 || !v.is_finite()) {
        anyhow::bail!("mapped variances must be finite and positive");
    }
    // Keep regime ordering from calm->turbulent.
    let mut sorted = vals;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // If nearly flat, spread slightly to avoid duplicate regimes.
    if sorted.first() == sorted.last() {
        let base = std.max(1e-6);
        sorted = (0..config.k)
            .map(|i| {
                let m = 1.0 + 0.1 * i as f64;
                (base * m).powi(2)
            })
            .collect();
    }
    Ok(sorted)
}

fn infer_durations(profile: &EmpiricalCalibrationProfile, k: usize) -> Vec<f64> {
    let low = profile.summary.low_episode_mean_duration.max(2.0);
    let high = profile.summary.high_episode_mean_duration.max(2.0);
    if k == 2 {
        vec![low, high]
    } else {
        (0..k)
            .map(|i| {
                let frac = i as f64 / (k - 1) as f64;
                low + frac * (high - low)
            })
            .collect()
    }
}

/// Map target expected durations d_j to transition probabilities using
/// p_jj = 1 - 1/d_j and symmetric off-diagonal allocation.
fn durations_to_transition_rows(
    durations: &[f64],
    symmetric_offdiag: bool,
) -> anyhow::Result<Vec<Vec<f64>>> {
    let k = durations.len();
    if k < 2 {
        anyhow::bail!("durations must have len>=2");
    }

    let mut rows = vec![vec![0.0; k]; k];
    for i in 0..k {
        let d = durations[i];
        if d <= 1.0 || !d.is_finite() {
            anyhow::bail!("duration d[{i}] must be >1 and finite, got {d}");
        }
        let p_ii = (1.0 - 1.0 / d).clamp(1e-6, 1.0 - 1e-6);
        rows[i][i] = p_ii;
        let rem = 1.0 - p_ii;
        if symmetric_offdiag {
            let each = rem / (k - 1) as f64;
            for (j, cell) in rows[i].iter_mut().enumerate() {
                if j != i {
                    *cell = each;
                }
            }
        } else {
            // Simple deterministic next-regime bias.
            let next = (i + 1) % k;
            rows[i][next] = rem;
        }
    }
    Ok(rows)
}

fn centered_grid(k: usize, scale: f64) -> Vec<f64> {
    if k == 1 {
        return vec![0.0];
    }
    let mid = (k - 1) as f64 / 2.0;
    (0..k).map(|i| (i as f64 - mid) * scale).collect()
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
                n: 1000,
                mean: 0.001,
                variance: 0.0004,
                std_dev: 0.02,
                q01: -0.05,
                q05: -0.03,
                q50: 0.0,
                q95: 0.03,
                q99: 0.05,
                tail_freq_q95: 0.05,
                abs_exceed_2std: 0.04,
                acf1: 0.1,
                abs_acf1: 0.2,
                sign_change_rate: 0.45,
                high_episode_mean_duration: 6.0,
                low_episode_mean_duration: 20.0,
            },
        }
    }

    #[test]
    fn duration_mapping_formula_matches() {
        let rows = durations_to_transition_rows(&[10.0, 5.0], true).unwrap();
        assert!((rows[0][0] - 0.9).abs() < 1e-12);
        assert!((rows[1][1] - 0.8).abs() < 1e-12);
        assert!((rows[0][1] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn calibrate_outputs_valid_model_params() {
        let cfg = CalibrationMappingConfig::default();
        let out = calibrate_to_synthetic(&profile(), &cfg).unwrap();
        out.model_params.validate().unwrap();
        assert_eq!(out.model_params.k, 2);
        assert_eq!(out.horizon, 2000);
    }

    #[test]
    fn calibrate_respects_target_duration() {
        let cfg = CalibrationMappingConfig {
            target_durations: vec![30.0, 5.0],
            ..CalibrationMappingConfig::default()
        };
        let out = calibrate_to_synthetic(&profile(), &cfg).unwrap();
        assert_eq!(out.expected_durations, vec![30.0, 5.0]);
        let p00 = out.model_params.transition_row(0)[0];
        assert!((p00 - (1.0 - 1.0 / 30.0)).abs() < 1e-10);
    }

    #[test]
    fn quantile_variance_mapping_positive() {
        let vars = map_variances(&profile(), &CalibrationMappingConfig::default()).unwrap();
        assert!(vars.iter().all(|v| *v > 0.0));
    }
}
