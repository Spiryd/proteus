#![allow(dead_code)]
/// Grid search over detector and model parameters.
///
/// [`ParamGrid`] sweeps `threshold`, `persistence_required`, and `cooldown`.
/// [`ModelGrid`] sweeps `k_regimes` and `features.family`.
/// [`optimize_full`] searches the joint (model × detector) grid.
/// Results are sorted by score descending (best first).
use super::config::{ExperimentConfig, FeatureFamilyConfig};
use super::result::{EvaluationSummary, ExperimentResult, RunStatus};
use super::runner::{ExperimentBackend, ExperimentRunner};

// ---------------------------------------------------------------------------
// Grid definition
// ---------------------------------------------------------------------------

/// The set of parameter values to sweep.
#[derive(Debug, Clone)]
pub struct ParamGrid {
    pub thresholds: Vec<f64>,
    pub persistence_values: Vec<usize>,
    pub cooldown_values: Vec<usize>,
}

impl Default for ParamGrid {
    fn default() -> Self {
        Self {
            thresholds: vec![0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            persistence_values: vec![1, 2, 3],
            cooldown_values: vec![0, 2, 5],
        }
    }
}

impl ParamGrid {
    /// Grid tuned for `HardSwitch` real-data experiments.
    /// Threshold sweeps [0.30, 0.80] — the posterior-majority range.
    pub fn for_real_hard_switch() -> Self {
        Self {
            thresholds: vec![0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80],
            persistence_values: vec![1, 2, 3, 5],
            cooldown_values: vec![0, 3, 5, 10],
        }
    }

    /// Grid tuned for `Surprise` real-data experiments.
    /// Threshold sweeps [1.0, 6.0] — surprise score (z-score-like) range.
    pub fn for_real_surprise() -> Self {
        Self {
            thresholds: vec![1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
            persistence_values: vec![1, 2, 3, 5],
            cooldown_values: vec![0, 5, 10, 20],
        }
    }

    /// Grid tuned for `PosteriorTransition` real-data experiments.
    pub fn for_real_posterior_transition() -> Self {
        Self {
            thresholds: vec![0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
            persistence_values: vec![1, 2, 3],
            cooldown_values: vec![0, 3, 5, 10],
        }
    }
}

impl ParamGrid {
    pub fn n_points(&self) -> usize {
        self.thresholds.len() * self.persistence_values.len() * self.cooldown_values.len()
    }

    /// Choose the default real grid based on detector type name.
    pub fn for_real_detector(detector_type: &super::config::DetectorType) -> Self {
        match detector_type {
            super::config::DetectorType::HardSwitch => Self::for_real_hard_switch(),
            super::config::DetectorType::Surprise => Self::for_real_surprise(),
            super::config::DetectorType::PosteriorTransition => {
                Self::for_real_posterior_transition()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Model grid
// ---------------------------------------------------------------------------

/// The set of model-level values to sweep (k_regimes and feature family).
#[derive(Debug, Clone)]
pub struct ModelGrid {
    pub k_regimes_values: Vec<usize>,
    pub feature_families: Vec<FeatureFamilyConfig>,
}

impl Default for ModelGrid {
    /// Default grid: k_regimes ∈ {2, 3}, five feature families (daily-safe).
    fn default() -> Self {
        Self {
            k_regimes_values: vec![2, 3],
            feature_families: vec![
                FeatureFamilyConfig::LogReturn,
                FeatureFamilyConfig::AbsReturn,
                FeatureFamilyConfig::SquaredReturn,
                FeatureFamilyConfig::RollingVol { window: 5, session_reset: false },
                FeatureFamilyConfig::RollingVol { window: 20, session_reset: false },
            ],
        }
    }
}

impl ModelGrid {
    /// Variant with `session_reset: true` rolling-vol families — suitable for
    /// intraday data where rolling windows should reset at session boundaries.
    pub fn for_intraday() -> Self {
        Self {
            k_regimes_values: vec![2, 3],
            feature_families: vec![
                FeatureFamilyConfig::LogReturn,
                FeatureFamilyConfig::AbsReturn,
                FeatureFamilyConfig::SquaredReturn,
                FeatureFamilyConfig::RollingVol { window: 5, session_reset: true },
                FeatureFamilyConfig::RollingVol { window: 20, session_reset: true },
            ],
        }
    }

    pub fn n_points(&self) -> usize {
        self.k_regimes_values.len() * self.feature_families.len()
    }
}

/// Return a short display name for a [`FeatureFamilyConfig`] variant.
pub fn feature_family_name(f: &FeatureFamilyConfig) -> String {
    match f {
        FeatureFamilyConfig::LogReturn => "log_return".to_string(),
        FeatureFamilyConfig::AbsReturn => "abs_return".to_string(),
        FeatureFamilyConfig::SquaredReturn => "sq_return".to_string(),
        FeatureFamilyConfig::RollingVol { window, .. } => format!("roll_vol_{window}"),
        FeatureFamilyConfig::StandardizedReturn { window, .. } => format!("std_ret_{window}"),
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Outcome for a single grid point.
#[derive(Debug, Clone)]
pub struct SearchPoint {
    // --- model params ---
    pub k_regimes: usize,
    pub feature_family: FeatureFamilyConfig,
    // --- detector params ---
    pub threshold: f64,
    pub persistence_required: usize,
    pub cooldown: usize,
    /// Combined score used for ranking (higher = better).
    pub score: f64,
    pub n_alarms: usize,
    pub coverage: f64,
    pub precision_like: f64,
    pub status: RunStatus,
}

/// Full result from an optimization run: all scored grid points + best config.
#[derive(Debug, Clone)]
pub struct OptimizeResult {
    /// All search points, sorted by score descending.
    pub points: Vec<SearchPoint>,
    /// The best-scoring `ExperimentConfig` (ready to hand to a runner).
    pub best_config: ExperimentConfig,
    /// Number of grid points evaluated.
    pub n_evaluated: usize,
}

/// Apply the best `SearchPoint`'s params onto `base` and return the patched config.
/// Patches both model fields (`k_regimes`, `features.family`) and detector
/// fields (`threshold`, `persistence_required`, `cooldown`).
pub fn apply_best(base: &ExperimentConfig, best: &SearchPoint) -> ExperimentConfig {
    let mut cfg = base.clone();
    cfg.model.k_regimes = best.k_regimes;
    cfg.features.family = best.feature_family.clone();
    cfg.detector.threshold = best.threshold;
    cfg.detector.persistence_required = best.persistence_required;
    cfg.detector.cooldown = best.cooldown;
    cfg.reproducibility.deterministic_run_id = false;
    cfg
}


/// Run a full grid search over detector parameters.
///
/// For each grid point, clones `base`, overrides `detector.threshold /
/// persistence_required / cooldown`, and calls `runner.run`. Results are
/// returned sorted by `score` descending.
pub fn grid_search<B: ExperimentBackend>(
    runner: &ExperimentRunner<B>,
    base: &ExperimentConfig,
    grid: &ParamGrid,
) -> Vec<SearchPoint> {
    let mut points = Vec::with_capacity(grid.n_points());

    for &threshold in &grid.thresholds {
        for &persistence_required in &grid.persistence_values {
            for &cooldown in &grid.cooldown_values {
                let mut cfg = base.clone();
                cfg.detector.threshold = threshold;
                cfg.detector.persistence_required = persistence_required;
                cfg.detector.cooldown = cooldown;
                // Each grid point gets a unique run — don't reuse the
                // deterministic id from the base config.
                cfg.reproducibility.deterministic_run_id = false;

                let result = runner.run(cfg);
                let (coverage, precision_like, n_alarms) = extract_metrics(&result);
                let score = combined_score(coverage, precision_like);

                points.push(SearchPoint {
                    k_regimes: base.model.k_regimes,
                    feature_family: base.features.family.clone(),
                    threshold,
                    persistence_required,
                    cooldown,
                    score,
                    n_alarms,
                    coverage,
                    precision_like,
                    status: result.status,
                });
            }
        }
    }

    // Best first.
    points.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    points
}

/// High-level optimization driver.
///
/// Runs a full grid search using the detector-appropriate grid, then returns
/// an [`OptimizeResult`] with all scored points and the best-patched config.
///
/// `progress_cb` is called before each grid point with `(point_index, total)`.
pub fn optimize<B: ExperimentBackend, F>(
    runner: &ExperimentRunner<B>,
    base: &ExperimentConfig,
    grid: &ParamGrid,
    mut progress_cb: F,
) -> OptimizeResult
where
    F: FnMut(usize, usize),
{
    let total = grid.n_points();
    let mut idx = 0usize;
    let mut points = Vec::with_capacity(total);

    for &threshold in &grid.thresholds {
        for &persistence_required in &grid.persistence_values {
            for &cooldown in &grid.cooldown_values {
                progress_cb(idx, total);
                idx += 1;

                let mut cfg = base.clone();
                cfg.detector.threshold = threshold;
                cfg.detector.persistence_required = persistence_required;
                cfg.detector.cooldown = cooldown;
                cfg.reproducibility.deterministic_run_id = false;
                // During grid search disable artifact writes for speed.
                cfg.output.write_json = false;
                cfg.output.write_csv = false;
                cfg.output.save_traces = false;

                let result = runner.run(cfg);
                let (coverage, precision_like, n_alarms) = extract_metrics(&result);
                let score = combined_score(coverage, precision_like);

                points.push(SearchPoint {
                    k_regimes: base.model.k_regimes,
                    feature_family: base.features.family.clone(),
                    threshold,
                    persistence_required,
                    cooldown,
                    score,
                    n_alarms,
                    coverage,
                    precision_like,
                    status: result.status,
                });
            }
        }
    }

    points.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let best_config = points
        .first()
        .map(|p| apply_best(base, p))
        .unwrap_or_else(|| base.clone());

    OptimizeResult {
        n_evaluated: points.len(),
        points,
        best_config,
    }
}

/// High-level joint optimization driver.
///
/// Sweeps the Cartesian product of `model_grid` (k_regimes × feature_family)
/// and `detector_grid` (threshold × persistence × cooldown). Results are
/// returned sorted by score descending.
///
/// `progress_cb` is called before each grid point with `(point_index, total)`.
pub fn optimize_full<B: ExperimentBackend, F>(
    runner: &ExperimentRunner<B>,
    base: &ExperimentConfig,
    model_grid: &ModelGrid,
    detector_grid: &ParamGrid,
    mut progress_cb: F,
) -> OptimizeResult
where
    F: FnMut(usize, usize),
{
    let total = model_grid.n_points() * detector_grid.n_points();
    let mut idx = 0usize;
    let mut points = Vec::with_capacity(total);

    for &k_regimes in &model_grid.k_regimes_values {
        for family in &model_grid.feature_families {
            for &threshold in &detector_grid.thresholds {
                for &persistence_required in &detector_grid.persistence_values {
                    for &cooldown in &detector_grid.cooldown_values {
                        progress_cb(idx, total);
                        idx += 1;

                        let mut cfg = base.clone();
                        cfg.model.k_regimes = k_regimes;
                        cfg.features.family = family.clone();
                        cfg.detector.threshold = threshold;
                        cfg.detector.persistence_required = persistence_required;
                        cfg.detector.cooldown = cooldown;
                        cfg.reproducibility.deterministic_run_id = false;
                        cfg.output.write_json = false;
                        cfg.output.write_csv = false;
                        cfg.output.save_traces = false;

                        let result = runner.run(cfg);
                        let (coverage, precision_like, n_alarms) = extract_metrics(&result);
                        let score = combined_score(coverage, precision_like);

                        points.push(SearchPoint {
                            k_regimes,
                            feature_family: family.clone(),
                            threshold,
                            persistence_required,
                            cooldown,
                            score,
                            n_alarms,
                            coverage,
                            precision_like,
                            status: result.status,
                        });
                    }
                }
            }
        }
    }

    points.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    let best_config = points
        .first()
        .map(|p| apply_best(base, p))
        .unwrap_or_else(|| base.clone());

    OptimizeResult {
        n_evaluated: points.len(),
        points,
        best_config,
    }
}


fn extract_metrics(result: &ExperimentResult) -> (f64, f64, usize) {
    let n_alarms = result
        .detector_summary
        .as_ref()
        .map(|d| d.n_alarms)
        .unwrap_or(0);

    match result.evaluation_summary.as_ref() {
        Some(EvaluationSummary::Synthetic {
            coverage,
            precision_like,
            ..
        }) => (*coverage, *precision_like, n_alarms),
        Some(EvaluationSummary::Real {
            event_coverage,
            alarm_relevance,
            segmentation_coherence,
        }) => {
            let combined = (event_coverage + alarm_relevance + segmentation_coherence) / 3.0;
            (combined, combined, n_alarms)
        }
        None => (0.0, 0.0, n_alarms),
    }
}

/// Combined score: equal-weight average of coverage and precision-like.
fn combined_score(coverage: f64, precision_like: f64) -> f64 {
    0.5 * coverage + 0.5 * precision_like
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::config::{
        DataConfig, DetectorConfig, DetectorType, EvaluationConfig, ExperimentConfig,
        ExperimentMode, FeatureConfig, FeatureFamilyConfig, ModelConfig, OutputConfig,
        ReproducibilityConfig, RunMetaConfig, ScalingPolicyConfig, TrainingMode,
    };
    use crate::experiments::runner::{DryRunBackend, ExperimentRunner};

    fn base_cfg() -> ExperimentConfig {
        ExperimentConfig {
            meta: RunMetaConfig {
                run_label: "search_test".to_string(),
                notes: None,
            },
            mode: ExperimentMode::Synthetic,
            data: DataConfig::Synthetic {
                scenario_id: "s".to_string(),
                horizon: 200,
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
                em_max_iter: 50,
                em_tol: 1e-6,
            },
            detector: DetectorConfig {
                detector_type: DetectorType::Surprise,
                threshold: 2.0,
                persistence_required: 1,
                cooldown: 0,
            },
            evaluation: EvaluationConfig::Synthetic { matching_window: 10 },
            output: OutputConfig {
                root_dir: "./runs_test".to_string(),
                write_json: false,
                write_csv: false,
                save_traces: false,
            },
            reproducibility: ReproducibilityConfig {
                seed: Some(1),
                deterministic_run_id: false,
                save_config_snapshot: false,
                record_git_info: false,
            },
        }
    }

    #[test]
    fn grid_search_returns_sorted_results() {
        let runner = ExperimentRunner::new(DryRunBackend);
        let grid = ParamGrid {
            thresholds: vec![0.5, 2.0],
            persistence_values: vec![1, 2],
            cooldown_values: vec![0],
        };
        let results = grid_search(&runner, &base_cfg(), &grid);
        assert_eq!(results.len(), 4);
        // Sorted descending by score.
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score);
        }
    }

    #[test]
    fn param_grid_n_points() {
        let grid = ParamGrid::default();
        assert_eq!(grid.n_points(), 8 * 3 * 3);
    }
}
