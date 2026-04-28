use serde::{Deserialize, Serialize};

/// Top-level experiment definition.
///
/// One run is fully determined by this object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub meta: RunMetaConfig,
    pub mode: ExperimentMode,
    pub data: DataConfig,
    pub features: FeatureConfig,
    pub model: ModelConfig,
    pub detector: DetectorConfig,
    pub evaluation: EvaluationConfig,
    pub output: OutputConfig,
    pub reproducibility: ReproducibilityConfig,
}

impl ExperimentConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.meta.run_label.trim().is_empty() {
            anyhow::bail!("run_label must be non-empty");
        }
        self.data.validate()?;
        self.features.validate()?;
        self.model.validate()?;
        self.detector.validate()?;
        self.evaluation.validate_for_mode(&self.mode)?;
        self.output.validate()?;
        self.reproducibility.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunMetaConfig {
    pub run_label: String,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentMode {
    Synthetic,
    Real,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataConfig {
    Synthetic {
        scenario_id: String,
        horizon: usize,
        dataset_id: Option<String>,
    },
    Real {
        asset: String,
        frequency: RealFrequency,
        dataset_id: String,
        date_start: Option<String>,
        date_end: Option<String>,
    },
}

impl DataConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        match self {
            Self::Synthetic {
                scenario_id,
                horizon,
                ..
            } => {
                if scenario_id.trim().is_empty() {
                    anyhow::bail!("synthetic scenario_id must be non-empty");
                }
                if *horizon < 2 {
                    anyhow::bail!("synthetic horizon must be >= 2");
                }
            }
            Self::Real {
                asset, dataset_id, ..
            } => {
                if asset.trim().is_empty() {
                    anyhow::bail!("real asset must be non-empty");
                }
                if dataset_id.trim().is_empty() {
                    anyhow::bail!("real dataset_id must be non-empty");
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RealFrequency {
    Daily,
    Intraday5m,
    Intraday15m,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureConfig {
    pub family: FeatureFamilyConfig,
    pub scaling: ScalingPolicyConfig,
    pub session_aware: bool,
}

impl FeatureConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        match self.family {
            FeatureFamilyConfig::RollingVol { window, .. }
            | FeatureFamilyConfig::StandardizedReturn { window, .. } => {
                if window == 0 {
                    anyhow::bail!("rolling window must be > 0");
                }
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FeatureFamilyConfig {
    LogReturn,
    AbsReturn,
    SquaredReturn,
    RollingVol {
        window: usize,
        session_reset: bool,
    },
    StandardizedReturn {
        window: usize,
        epsilon: f64,
        session_reset: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingPolicyConfig {
    None,
    ZScore,
    RobustZScore,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelConfig {
    pub k_regimes: usize,
    pub training: TrainingMode,
    pub em_max_iter: usize,
    pub em_tol: f64,
}

impl ModelConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.k_regimes < 2 {
            anyhow::bail!("k_regimes must be >= 2");
        }
        if self.em_max_iter == 0 {
            anyhow::bail!("em_max_iter must be > 0");
        }
        if !(self.em_tol.is_finite() && self.em_tol > 0.0) {
            anyhow::bail!("em_tol must be finite and > 0");
        }
        match &self.training {
            TrainingMode::LoadFrozen { artifact_id } => {
                if artifact_id.trim().is_empty() {
                    anyhow::bail!("artifact_id must be non-empty in LoadFrozen mode");
                }
            }
            TrainingMode::FitOffline => {}
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TrainingMode {
    FitOffline,
    LoadFrozen { artifact_id: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectorConfig {
    pub detector_type: DetectorType,
    pub threshold: f64,
    pub persistence_required: usize,
    pub cooldown: usize,
}

impl DetectorConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if !self.threshold.is_finite() {
            anyhow::bail!("detector threshold must be finite");
        }
        if self.persistence_required == 0 {
            anyhow::bail!("persistence_required must be >= 1");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectorType {
    HardSwitch,
    PosteriorTransition,
    Surprise,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvaluationConfig {
    Synthetic {
        matching_window: usize,
    },
    Real {
        /// Path to a JSON file containing `Vec<ProxyEvent>`.
        /// Empty string (default) means no proxy events; Route A will report zero coverage.
        #[serde(default)]
        proxy_events_path: String,
        route_a_point_pre_bars: usize,
        route_a_point_post_bars: usize,
        route_a_causal_only: bool,
        route_b_min_segment_len: usize,
    },
}

impl EvaluationConfig {
    pub fn validate_for_mode(&self, mode: &ExperimentMode) -> anyhow::Result<()> {
        match (mode, self) {
            (ExperimentMode::Synthetic, EvaluationConfig::Synthetic { .. }) => Ok(()),
            (ExperimentMode::Real, EvaluationConfig::Real { .. }) => Ok(()),
            (ExperimentMode::Synthetic, EvaluationConfig::Real { .. }) => {
                anyhow::bail!("synthetic mode requires synthetic evaluation config")
            }
            (ExperimentMode::Real, EvaluationConfig::Synthetic { .. }) => {
                anyhow::bail!("real mode requires real evaluation config")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputConfig {
    pub root_dir: String,
    pub write_json: bool,
    pub write_csv: bool,
    pub save_traces: bool,
}

impl OutputConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.root_dir.trim().is_empty() {
            anyhow::bail!("output root_dir must be non-empty");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReproducibilityConfig {
    pub seed: Option<u64>,
    pub deterministic_run_id: bool,
    pub save_config_snapshot: bool,
    pub record_git_info: bool,
}

impl ReproducibilityConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.deterministic_run_id && self.seed.is_none() {
            anyhow::bail!("deterministic_run_id requires a seed");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_base() -> ExperimentConfig {
        ExperimentConfig {
            meta: RunMetaConfig {
                run_label: "run_a".to_string(),
                notes: None,
            },
            mode: ExperimentMode::Synthetic,
            data: DataConfig::Synthetic {
                scenario_id: "scn_a".to_string(),
                horizon: 100,
                dataset_id: None,
            },
            features: FeatureConfig {
                family: FeatureFamilyConfig::LogReturn,
                scaling: ScalingPolicyConfig::None,
                session_aware: false,
            },
            model: ModelConfig {
                k_regimes: 2,
                training: TrainingMode::FitOffline,
                em_max_iter: 50,
                em_tol: 1e-6,
            },
            detector: DetectorConfig {
                detector_type: DetectorType::Surprise,
                threshold: 2.0,
                persistence_required: 1,
                cooldown: 0,
            },
            evaluation: EvaluationConfig::Synthetic { matching_window: 5 },
            output: OutputConfig {
                root_dir: "out".to_string(),
                write_json: true,
                write_csv: false,
                save_traces: false,
            },
            reproducibility: ReproducibilityConfig {
                seed: Some(7),
                deterministic_run_id: true,
                save_config_snapshot: true,
                record_git_info: false,
            },
        }
    }

    #[test]
    fn valid_config_passes() {
        valid_base().validate().unwrap();
    }

    #[test]
    fn mismatched_mode_eval_fails() {
        let mut c = valid_base();
        c.mode = ExperimentMode::Real;
        assert!(c.validate().is_err());
    }

    #[test]
    fn deterministic_run_id_without_seed_fails() {
        let mut c = valid_base();
        c.reproducibility.seed = None;
        assert!(c.validate().is_err());
    }
}
