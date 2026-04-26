/// Grid search over detector parameters.
///
/// Sweeps a `ParamGrid` over `threshold`, `persistence_required`, and
/// `cooldown` while holding everything else fixed in a base
/// [`ExperimentConfig`]. Results are sorted by score descending (best first).
use super::config::{ExperimentConfig, EvaluationConfig};
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
    pub fn n_points(&self) -> usize {
        self.thresholds.len() * self.persistence_values.len() * self.cooldown_values.len()
    }
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Outcome for a single grid point.
#[derive(Debug, Clone)]
pub struct SearchPoint {
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

// ---------------------------------------------------------------------------
// Search driver
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
