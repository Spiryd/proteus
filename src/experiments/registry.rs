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
            description: "PosteriorTransition — 2-regime synthetic, LogReturn/ZScore, horizon 2000",
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
        RegisteredExperiment {
            id: "real_spy_intraday_hard_switch",
            description: "SPY 15min  — HardSwitch on intraday adj-close log-returns (2022-2025)",
            build: real_spy_intraday_hard_switch,
        },
        RegisteredExperiment {
            id: "simreal_spy_daily_hard_switch",
            description: "SPY daily  — Sim-to-real: EM trained on synthetic stream calibrated to SPY via Quick-EM",
            build: simreal_spy_daily_hard_switch,
        },
        RegisteredExperiment {
            id: "posterior_transition_tv",
            description: "PosteriorTransitionTV — 2-regime synthetic, LogReturn/ZScore, TotalVariation score",
            build: posterior_transition_tv,
        },
        RegisteredExperiment {
            id: "hard_switch_shock",
            description: "HardSwitch (shock_contaminated) — 2-regime synthetic with jump contamination",
            build: hard_switch_shock,
        },
        RegisteredExperiment {
            id: "hard_switch_frozen",
            description: "HardSwitch (LoadFrozen) — loads pre-fitted model from data/frozen_models/hard_switch_frozen",
            build: hard_switch_frozen,
        },
        RegisteredExperiment {
            id: "hard_switch_multi_start",
            description: "HardSwitch (multi-start EM, n_starts=3) — exercises multi_start_summary.json artifact",
            build: hard_switch_multi_start,
        },
        RegisteredExperiment {
            id: "surprise_ema",
            description: "Surprise EMA — 2-regime synthetic, Surprise detector with ema_alpha=0.3 slow baseline",
            build: surprise_ema,
        },
        RegisteredExperiment {
            id: "squared_return_surprise",
            description: "SquaredReturn — 2-regime synthetic, Surprise detector on squared-return feature",
            build: squared_return_surprise,
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
            em_n_starts: 1,
        },
        // Filled in by each constructor.
        detector: DetectorConfig {
            detector_type: DetectorType::Surprise,
            threshold: 2.5,
            persistence_required: 2,
            cooldown: 5,
            ema_alpha: None,
        },
        evaluation: EvaluationConfig::Synthetic {
            matching_window: 20,
        },
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
        ema_alpha: None,
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
        ema_alpha: None,
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
        ema_alpha: None,
    };
    cfg
}

pub fn posterior_transition_tv() -> ExperimentConfig {
    let mut cfg = base(
        "posterior_transition_tv",
        "Posterior Transition TV detector — full synthetic run (TotalVariation score)",
    );
    cfg.detector = DetectorConfig {
        detector_type: DetectorType::PosteriorTransitionTV,
        threshold: 0.3,
        persistence_required: 2,
        cooldown: 5,
        ema_alpha: None,
    };
    cfg
}

pub fn hard_switch_shock() -> ExperimentConfig {
    let mut cfg = base(
        "hard_switch_shock",
        "HardSwitch detector — shock-contaminated synthetic run (jump contamination path)",
    );
    cfg.data = DataConfig::Synthetic {
        scenario_id: "shock_contaminated".to_string(),
        horizon: 2000,
        dataset_id: None,
    };
    cfg.detector = DetectorConfig {
        detector_type: DetectorType::HardSwitch,
        threshold: 0.5,
        persistence_required: 2,
        cooldown: 5,
        ema_alpha: None,
    };
    cfg
}

pub fn hard_switch_frozen() -> ExperimentConfig {
    let mut cfg = base(
        "hard_switch_frozen",
        "HardSwitch detector — loads pre-fitted model from data/frozen_models/hard_switch_frozen",
    );
    cfg.model.training = TrainingMode::LoadFrozen {
        artifact_id: "data/frozen_models/hard_switch_frozen".to_string(),
    };
    cfg.detector = DetectorConfig {
        detector_type: DetectorType::HardSwitch,
        threshold: 0.5,
        persistence_required: 2,
        cooldown: 5,
        ema_alpha: None,
    };
    cfg
}

pub fn hard_switch_multi_start() -> ExperimentConfig {
    let mut cfg = base(
        "hard_switch_multi_start",
        "HardSwitch detector — multi-start EM (3 starts), exercises multi_start_summary.json",
    );
    cfg.model.em_n_starts = 3;
    cfg.detector = DetectorConfig {
        detector_type: DetectorType::HardSwitch,
        threshold: 0.5,
        persistence_required: 2,
        cooldown: 5,
        ema_alpha: None,
    };
    cfg
}

// ---------------------------------------------------------------------------
// Real-data base (shared fields for both SPY and WTI real experiments)
// ---------------------------------------------------------------------------

fn base_real(
    run_label: &str,
    notes: &str,
    asset: &str,
    frequency: RealFrequency,
) -> ExperimentConfig {
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
            em_n_starts: 1,
        },
        detector: DetectorConfig {
            detector_type: DetectorType::HardSwitch,
            threshold: 0.55,
            persistence_required: 2,
            cooldown: 5,
            ema_alpha: None,
        },
        evaluation: EvaluationConfig::Real {
            proxy_events_path: format!("data/proxy_events/{}.json", asset.to_lowercase()),
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
            seed: None,                  // real data; no simulation seed needed
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
        ema_alpha: None,
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
        ema_alpha: None,
    };
    cfg
}

/// SPY 15-minute intraday — HardSwitch on log-returns.
///
/// Uses 2022-01-01 to 2025-12-31 to keep the EM training set tractable
/// (~30,000 bars).  Proxy events from `data/proxy_events/spy.json`.
pub fn real_spy_intraday_hard_switch() -> ExperimentConfig {
    ExperimentConfig {
        meta: RunMetaConfig {
            run_label: "real_spy_intraday_hard_switch".to_string(),
            notes: Some(
                "SPY 15min intraday | HardSwitch | adj-close log-returns | Route A+B evaluation"
                    .to_string(),
            ),
        },
        mode: ExperimentMode::Real,
        data: DataConfig::Real {
            asset: "SPY".to_string(),
            frequency: RealFrequency::Intraday15m,
            dataset_id: "spy_intraday_15min".to_string(),
            date_start: Some("2022-01-01".to_string()),
            date_end: Some("2025-12-31".to_string()),
        },
        features: FeatureConfig {
            family: FeatureFamilyConfig::LogReturn,
            scaling: ScalingPolicyConfig::ZScore,
            session_aware: true,
        },
        model: ModelConfig {
            k_regimes: 2,
            training: TrainingMode::FitOffline,
            em_max_iter: 200,
            em_tol: 1e-6,
            em_n_starts: 1,
        },
        detector: DetectorConfig {
            detector_type: DetectorType::HardSwitch,
            threshold: 0.55,
            persistence_required: 3,
            cooldown: 10,
            ema_alpha: None,
        },
        evaluation: EvaluationConfig::Real {
            proxy_events_path: "data/proxy_events/spy.json".to_string(),
            route_a_point_pre_bars: 20,
            route_a_point_post_bars: 40,
            route_a_causal_only: false,
            route_b_min_segment_len: 20,
        },
        output: OutputConfig {
            root_dir: "./runs".to_string(),
            write_json: true,
            write_csv: true,
            save_traces: true,
        },
        reproducibility: ReproducibilityConfig {
            seed: None,
            deterministic_run_id: false,
            save_config_snapshot: true,
            record_git_info: false,
        },
    }
}

/// **Sim-to-real:** SPY daily HardSwitch trained on synthetic stream
/// calibrated from real SPY via Quick-EM.
///
/// Pipeline:
/// 1. Load SPY daily adjusted-close prices from the DuckDB cache.
/// 2. Quick-EM calibration: fit a 2-regime Gaussian MSM on the real training
///    partition log-returns; use the result as the synthetic generator.
/// 3. Simulate 2 000 synthetic observations from the calibrated `ModelParams`.
/// 4. Apply a single z-score scaler (fitted on real-train) to both real and
///    synthetic streams (B\u20321 contract).
/// 5. Train Baum-Welch EM on the scaled synthetic stream only.
/// 6. Run the synthetic-trained `FrozenModel` online over the real series.
/// 7. Evaluate against SPY proxy events via Route A + Route B.
pub fn simreal_spy_daily_hard_switch() -> ExperimentConfig {
    use crate::calibration::{CalibrationMappingConfig, CalibrationStrategy};

    ExperimentConfig {
        meta: RunMetaConfig {
            run_label: "simreal_spy_daily_hard_switch".to_string(),
            notes: Some(
                "Sim-to-real | SPY daily | Quick-EM calibration | synthetic-trained EM | real-tested HardSwitch"
                    .to_string(),
            ),
        },
        mode: ExperimentMode::SimToReal,
        data: DataConfig::CalibratedSynthetic {
            real_asset: "SPY".to_string(),
            real_frequency: RealFrequency::Daily,
            real_dataset_id: "spy_daily".to_string(),
            real_date_start: Some("2018-01-01".to_string()),
            real_date_end: None,
            horizon: 2000,
            mapping: CalibrationMappingConfig {
                k: 2,
                horizon: 2000,
                strategy: CalibrationStrategy::QuickEm { max_iter: 100, tol: 1e-6 },
                ..CalibrationMappingConfig::default()
            },
            dataset_id: Some("simreal_spy_daily".to_string()),
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
            em_n_starts: 1,
        },
        detector: DetectorConfig {
            detector_type: DetectorType::HardSwitch,
            threshold: 0.55,
            persistence_required: 2,
            cooldown: 5,
            ema_alpha: None,
        },
        evaluation: EvaluationConfig::Real {
            proxy_events_path: "data/proxy_events/spy.json".to_string(),
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
            seed: Some(42),
            deterministic_run_id: true,
            save_config_snapshot: true,
            record_git_info: false,
        },
    }
}

// ---------------------------------------------------------------------------

/// `SurpriseDetector` with EMA baseline smoothing (`ema_alpha = 0.95`).
///
/// This exercises the exponential-moving-average score variant of the Surprise
/// detector (see `docs/changepoint_detectors.md`).  All other settings match
/// the `surprise` baseline experiment so results are directly comparable.
pub fn surprise_ema() -> ExperimentConfig {
    let mut cfg = base(
        "surprise_ema",
        "Surprise EMA — Surprise detector with ema_alpha=0.3 slow-moving baseline",
    );
    cfg.detector = DetectorConfig {
        detector_type: DetectorType::Surprise,
        threshold: 2.5,
        persistence_required: 2,
        cooldown: 5,
        // ema_alpha drives a *deviation-from-baseline* score: raw - EMA(raw).
        // A small alpha (slow baseline) keeps the baseline near the long-run
        // mean so that sudden spikes produce large positive deviations.
        // alpha=0.95 would make the baseline track the current value almost
        // instantly, flattening all deviations to ~0.  Use 0.3 instead.
        ema_alpha: Some(0.3),
    };
    cfg
}

/// Synthetic experiment using `SquaredReturn` as the observed feature.
///
/// `SquaredReturn` is the squared log-return $r_t^2$, a classical second-moment
/// volatility proxy.  This experiment exercises the `SquaredReturn` code path
/// end-to-end and produces results comparable with the `LogReturn`-based
/// `hard_switch` and `surprise` baselines.
pub fn squared_return_surprise() -> ExperimentConfig {
    let mut cfg = base(
        "squared_return_surprise",
        "SquaredReturn feature — Surprise detector on squared log-returns",
    );
    cfg.features = FeatureConfig {
        family: FeatureFamilyConfig::SquaredReturn,
        scaling: ScalingPolicyConfig::ZScore,
        session_aware: false,
    };
    cfg.detector = DetectorConfig {
        detector_type: DetectorType::Surprise,
        threshold: 2.5,
        persistence_required: 2,
        cooldown: 5,
        ema_alpha: None,
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
