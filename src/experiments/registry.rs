/// Compile-time experiment registry.
///
/// Define all experiments here as typed Rust constructors instead of JSON
/// files. This gives full type-safety and IDE navigation.
use super::config::{
    DataConfig, DetectorConfig, DetectorType, EvaluationConfig, ExperimentConfig, ExperimentMode,
    FeatureConfig, FeatureFamilyConfig, ModelConfig, OutputConfig, ReproducibilityConfig,
    RunMetaConfig, ScalingPolicyConfig, TrainingMode,
};

// ---------------------------------------------------------------------------
// Registry entry
// ---------------------------------------------------------------------------

/// A registered experiment: a static id, human description, and a builder fn.
pub struct RegisteredExperiment {
    pub id: &'static str,
    pub description: &'static str,
    /// Returns a fresh, owned [`ExperimentConfig`].
    pub build: fn() -> ExperimentConfig,
}

/// The canonical experiment registry.
///
/// Add new experiments here — they are immediately available in the E2E run,
/// parameter search, and all CLI menus.
pub fn registry() -> Vec<RegisteredExperiment> {
    vec![
        RegisteredExperiment {
            id: "hard_switch",
            description: "HardSwitch        — 2-regime synthetic, LogReturn/ZScore, horizon 2000",
            build: hard_switch,
        },
        RegisteredExperiment {
            id: "posterior_transition",
            description:
                "PosteriorTransition — 2-regime synthetic, LogReturn/ZScore, horizon 2000",
            build: posterior_transition,
        },
        RegisteredExperiment {
            id: "surprise",
            description: "Surprise          — 2-regime synthetic, LogReturn/ZScore, horizon 2000",
            build: surprise,
        },
    ]
}

// ---------------------------------------------------------------------------
// Shared base — override per-experiment fields after calling this
// ---------------------------------------------------------------------------

fn base(run_label: &str, notes: &str) -> ExperimentConfig {
    ExperimentConfig {
        meta: RunMetaConfig {
            run_label: run_label.to_string(),
            notes: Some(notes.to_string()),
        },
        mode: ExperimentMode::Synthetic,
        data: DataConfig::Synthetic {
            scenario_id: "scenario_calibrated".to_string(),
            horizon: 2000,
            dataset_id: None,
        },
        features: FeatureConfig {
            family: FeatureFamilyConfig::LogReturn,
            scaling: ScalingPolicyConfig::ZScore,
            session_aware: false,
        },
        model: ModelConfig {
            k_regimes: 2,
            training: TrainingMode::FitOffline,
            em_max_iter: 200,
            em_tol: 1e-6,
        },
        // Filled in by each constructor.
        detector: DetectorConfig {
            detector_type: DetectorType::Surprise,
            threshold: 2.5,
            persistence_required: 2,
            cooldown: 5,
        },
        evaluation: EvaluationConfig::Synthetic { matching_window: 20 },
        output: OutputConfig {
            root_dir: "./runs".to_string(),
            write_json: true,
            write_csv: true,
            save_traces: true,
        },
        reproducibility: ReproducibilityConfig {
            seed: Some(42),
            deterministic_run_id: true,
            save_config_snapshot: true,
            record_git_info: false,
        },
    }
}

// ---------------------------------------------------------------------------
// Experiment constructors
// ---------------------------------------------------------------------------

pub fn hard_switch() -> ExperimentConfig {
    let mut cfg = base("hard_switch", "Hard Switch detector — full synthetic run");
    cfg.detector = DetectorConfig {
        detector_type: DetectorType::HardSwitch,
        threshold: 0.5,
        persistence_required: 2,
        cooldown: 5,
    };
    cfg
}

pub fn posterior_transition() -> ExperimentConfig {
    let mut cfg = base(
        "posterior_transition",
        "Posterior Transition detector — full synthetic run",
    );
    cfg.detector = DetectorConfig {
        detector_type: DetectorType::PosteriorTransition,
        threshold: 0.3,
        persistence_required: 2,
        cooldown: 5,
    };
    cfg
}

pub fn surprise() -> ExperimentConfig {
    let mut cfg = base("surprise", "Surprise detector — full synthetic run");
    cfg.detector = DetectorConfig {
        detector_type: DetectorType::Surprise,
        threshold: 2.5,
        persistence_required: 2,
        cooldown: 5,
    };
    cfg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_all_valid() {
        for entry in registry() {
            let cfg = (entry.build)();
            cfg.validate()
                .unwrap_or_else(|e| panic!("registry entry '{}' failed validation: {e}", entry.id));
        }
    }

    #[test]
    fn registry_ids_unique() {
        let reg = registry();
        let mut ids: Vec<&str> = reg.iter().map(|e| e.id).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(before, ids.len(), "duplicate registry ids detected");
    }
}
