/// Compile-time experiment registry.
///
/// Define all experiments here as typed Rust constructors instead of JSON
/// files. This gives full type-safety and IDE navigation.
use super::config::{
    DataConfig, DetectorConfig, DetectorType, EvaluationConfig, ExperimentConfig, ExperimentMode,
    FeatureConfig, FeatureFamilyConfig, ModelConfig, OutputConfig, RealFrequency,
    ReproducibilityConfig, RunMetaConfig, ScalingPolicyConfig, TrainingMode,
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
        RegisteredExperiment {
            id: "real_spy_daily_hard_switch",
            description: "SPY daily  — HardSwitch on adj-close log-returns (real data)",
            build: real_spy_daily_hard_switch,
        },
        RegisteredExperiment {
            id: "real_wti_daily_surprise",
            description: "WTI daily  — Surprise detector on spot-price log-returns (real data)",
            build: real_wti_daily_surprise,
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

// ---------------------------------------------------------------------------
// Real-data base (shared fields for both SPY and WTI real experiments)
// ---------------------------------------------------------------------------

fn base_real(run_label: &str, notes: &str, asset: &str, frequency: RealFrequency) -> ExperimentConfig {
    ExperimentConfig {
        meta: RunMetaConfig {
            run_label: run_label.to_string(),
            notes: Some(notes.to_string()),
        },
        mode: ExperimentMode::Real,
        data: DataConfig::Real {
            asset: asset.to_string(),
            frequency,
            dataset_id: format!("{}_daily", asset.to_lowercase()),
            date_start: Some("2018-01-01".to_string()),
            date_end: None,
        },
        features: FeatureConfig {
            family: FeatureFamilyConfig::LogReturn,
            scaling: ScalingPolicyConfig::ZScore,
            session_aware: false,
        },
        model: ModelConfig {
            k_regimes: 2,
            training: TrainingMode::FitOffline,
            em_max_iter: 300,
            em_tol: 1e-7,
        },
        detector: DetectorConfig {
            detector_type: DetectorType::HardSwitch,
            threshold: 0.55,
            persistence_required: 2,
            cooldown: 5,
        },
        evaluation: EvaluationConfig::Real {
            proxy_events_path: format!(
                "data/proxy_events/{}.json",
                asset.to_lowercase()
            ),
            route_a_point_pre_bars: 5,
            route_a_point_post_bars: 10,
            route_a_causal_only: false,
            route_b_min_segment_len: 10,
        },
        output: OutputConfig {
            root_dir: "./runs".to_string(),
            write_json: true,
            write_csv: true,
            save_traces: true,
        },
        reproducibility: ReproducibilityConfig {
            seed: None, // real data; no simulation seed needed
            deterministic_run_id: false, // real experiments carry no RNG seed
            save_config_snapshot: true,
            record_git_info: false,
        },
    }
}

// ---------------------------------------------------------------------------
// Real-data experiment constructors
// ---------------------------------------------------------------------------

/// SPY daily adjusted-close log-returns — HardSwitch detector.
///
/// Proxy events in `data/proxy_events/spy.json` cover major macro events
/// (COVID crash, Fed rate cycles, banking crises) against which alarm
/// alignment is benchmarked via Route A.
pub fn real_spy_daily_hard_switch() -> ExperimentConfig {
    let mut cfg = base_real(
        "real_spy_daily_hard_switch",
        "SPY daily | HardSwitch | adj-close log-returns | Route A+B evaluation",
        "SPY",
        RealFrequency::Daily,
    );
    cfg.detector = DetectorConfig {
        detector_type: DetectorType::HardSwitch,
        threshold: 0.55,
        persistence_required: 2,
        cooldown: 5,
    };
    cfg
}

/// WTI crude-oil daily spot-price log-returns — Surprise detector.
///
/// Supply shocks (OPEC cuts, COVID demand collapse) are expected to register
/// as high-surprise events.  Proxy events in `data/proxy_events/wti.json`.
pub fn real_wti_daily_surprise() -> ExperimentConfig {
    let mut cfg = base_real(
        "real_wti_daily_surprise",
        "WTI daily | Surprise | spot-price log-returns | Route A+B evaluation",
        "WTI",
        RealFrequency::Daily,
    );
    cfg.data = DataConfig::Real {
        asset: "WTI".to_string(),
        frequency: RealFrequency::Daily,
        dataset_id: "wti_daily".to_string(),
        date_start: Some("2018-01-01".to_string()),
        date_end: None,
    };
    cfg.detector = DetectorConfig {
        detector_type: DetectorType::Surprise,
        threshold: 3.0,
        persistence_required: 2,
        cooldown: 10,
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
