#![allow(dead_code)]
/// Real synthetic experiment backend.
///
/// Wires the full math stack for synthetic (simulated) experiments:
/// 1. `resolve_data`      — simulate from a calibrated Markov-Switching model
/// 2. `build_features`    — passthrough (simulated data is already in feature space)
/// 3. `train_or_load_model` — run full Baum-Welch EM on the training partition
/// 4. `run_online`        — stream all observations through the Hamilton filter + detector
/// 5. `evaluate_synthetic` — event-matching against ground-truth changepoints
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::benchmark::matching::{EventMatcher, MatchConfig};
use crate::benchmark::metrics::MetricSuite;
use crate::benchmark::truth::ChangePointTruth;
use crate::model::params::ModelParams;
use crate::model::simulate::simulate;

use super::config::{DataConfig, EvaluationConfig, ExperimentConfig};
use super::runner::{
    DataBundle, ExperimentBackend, FeatureBundle, ModelArtifact, OnlineRunArtifact, RealEvalArtifact,
    SyntheticEvalArtifact,
};
use super::shared::{run_online_shared, train_or_load_model_shared};

// =========================================================================
// Backend
// =========================================================================

/// Backend that performs real simulation, EM training, and online detection.
///
/// Only `ExperimentMode::Synthetic` is fully supported; `Real` mode falls
/// back to a stub (the data infrastructure for live DuckDB ingestion is
/// handled separately in Phase 19/22).
#[derive(Debug, Clone, Default)]
pub struct SyntheticBackend;

impl SyntheticBackend {
    pub fn new() -> Self {
        Self
    }
}

impl ExperimentBackend for SyntheticBackend {
    // ------------------------------------------------------------------
    // Stage 1 — Resolve data (simulate from calibrated MSM)
    // ------------------------------------------------------------------
    fn resolve_data(&self, cfg: &ExperimentConfig) -> anyhow::Result<DataBundle> {
        match &cfg.data {
            DataConfig::Synthetic {
                scenario_id,
                horizon,
                dataset_id,
            } => {
                let horizon = *horizon;
                let seed = cfg.reproducibility.seed.unwrap_or(42);
                let mut rng = StdRng::seed_from_u64(seed);

                let params = build_synthetic_params(scenario_id, cfg.model.k_regimes)?;
                let sim = simulate(params, horizon, &mut rng)?;

                // Derive changepoints: positions (1-based) where regime changed.
                let changepoints: Vec<usize> = (1..sim.states.len())
                    .filter(|&i| sim.states[i] != sim.states[i - 1])
                    .map(|i| i + 1) // 1-based
                    .collect();

                let train_n = (horizon as f64 * 0.70) as usize;

                Ok(DataBundle {
                    dataset_key: format!("synthetic:{}:{}", scenario_id, dataset_id.as_deref().unwrap_or("")),
                    n_observations: sim.observations.len(),
                    observations: sim.observations,
                    changepoint_truth: Some(changepoints),
                    train_n,
                    timestamps: vec![],
                })
            }
            DataConfig::Real { dataset_id, .. } => {
                // Real data backend not yet implemented — return a stub.
                Ok(DataBundle {
                    dataset_key: format!("real:{dataset_id}"),
                    n_observations: 0,
                    observations: vec![],
                    changepoint_truth: None,
                    train_n: 0,
                    timestamps: vec![],
                })
            }
        }
    }

    // ------------------------------------------------------------------
    // Stage 2 — Build features
    //
    // For synthetic experiments the simulated values are already in the
    // correct observation space (Gaussian emissions).  We apply only
    // Z-score normalisation on the training partition when requested;
    // the feature family is ignored for simulated data.
    // ------------------------------------------------------------------
    fn build_features(
        &self,
        cfg: &ExperimentConfig,
        data: &DataBundle,
    ) -> anyhow::Result<FeatureBundle> {
        use crate::experiments::config::ScalingPolicyConfig;

        if data.observations.is_empty() {
            return Ok(FeatureBundle {
                feature_label: format!("{:?}", cfg.features.family),
                n_observations: 0,
                observations: vec![],
                train_n: 0,
                timestamps: vec![],
            });
        }

        let obs = &data.observations;
        let train_n = data.train_n.min(obs.len());

        let scaled: Vec<f64> = match &cfg.features.scaling {
            ScalingPolicyConfig::None => obs.clone(),
            ScalingPolicyConfig::ZScore | ScalingPolicyConfig::RobustZScore => {
                let train_slice = &obs[..train_n];
                let mean = train_slice.iter().sum::<f64>() / train_n as f64;
                let var = train_slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / train_n as f64;
                let std = var.sqrt().max(1e-10);
                obs.iter().map(|x| (x - mean) / std).collect()
            }
        };

        Ok(FeatureBundle {
            feature_label: format!("{:?}", cfg.features.family),
            n_observations: scaled.len(),
            observations: scaled,
            train_n,
            timestamps: vec![],
        })
    }

    // ------------------------------------------------------------------
    // Stage 3 — Train model (Baum-Welch EM on training partition)
    // ------------------------------------------------------------------
    fn train_or_load_model(
        &self,
        cfg: &ExperimentConfig,
        features: &FeatureBundle,
    ) -> anyhow::Result<ModelArtifact> {
        train_or_load_model_shared(cfg, features)
    }

    // ------------------------------------------------------------------
    // Stage 4 — Online detection
    // ------------------------------------------------------------------
    fn run_online(
        &self,
        cfg: &ExperimentConfig,
        model: &ModelArtifact,
        features: &FeatureBundle,
    ) -> anyhow::Result<OnlineRunArtifact> {
        run_online_shared(cfg, model, features)
    }

    // ------------------------------------------------------------------
    // Stage 5 — Evaluate (synthetic: event matching vs ground truth)
    // ------------------------------------------------------------------
    fn evaluate_synthetic(
        &self,
        cfg: &ExperimentConfig,
        online: &OnlineRunArtifact,
    ) -> anyhow::Result<SyntheticEvalArtifact> {
        // We need the changepoint truth from the DataBundle, but
        // ExperimentBackend::evaluate_synthetic only receives the OnlineRunArtifact.
        // The workaround: we need the changepoints stored somewhere accessible.
        // We'll reconstruct them from the config + seed by re-simulating.
        let (changepoints, stream_len) = regenerate_changepoints(cfg)?;

        let window = match cfg.evaluation {
            EvaluationConfig::Synthetic { matching_window } => matching_window,
            _ => 20,
        };

        let truth = ChangePointTruth::new(changepoints.clone(), stream_len)?;
        let n_events = truth.times.len();

        // Build AlarmEvent list from alarm_indices
        let alarm_events: Vec<crate::detector::AlarmEvent> = online
            .alarm_indices
            .iter()
            .map(|&t| crate::detector::AlarmEvent {
                t,
                score: 0.0, // score not stored here, just t matters for matching
                detector_kind: map_detector_kind(&cfg.detector.detector_type),
                dominant_regime_before: None,
                dominant_regime_after: 0,
            })
            .collect();

        let matcher = EventMatcher::new(MatchConfig { window });
        let match_result = matcher.match_events(&truth, &alarm_events);
        let metrics = MetricSuite::from_match(&match_result);

        // Legacy fields for compatibility
        let coverage = metrics.recall;
        let precision_like = metrics.precision;

        Ok(SyntheticEvalArtifact {
            n_events,
            coverage: if coverage.is_nan() { 0.0 } else { coverage },
            precision_like: if precision_like.is_nan() { 0.0 } else { precision_like },
            precision: Some(if metrics.precision.is_nan() { 0.0 } else { metrics.precision }),
            recall: Some(if metrics.recall.is_nan() { 0.0 } else { metrics.recall }),
            miss_rate: Some(if metrics.miss_rate.is_nan() { 0.0 } else { metrics.miss_rate }),
            false_alarm_rate: Some(metrics.false_alarm_rate),
            delay_mean: if metrics.delay.n > 0 && !metrics.delay.mean.is_nan() {
                Some(metrics.delay.mean)
            } else {
                None
            },
            delay_median: if metrics.delay.n > 0 && !metrics.delay.median.is_nan() {
                Some(metrics.delay.median)
            } else {
                None
            },
            n_true_positive: Some(metrics.n_true_positive),
            n_false_positive: Some(metrics.n_false_positive),
            n_missed: Some(metrics.n_missed),
        })
    }

    fn evaluate_real(
        &self,
        _cfg: &ExperimentConfig,
        _online: &OnlineRunArtifact,
    ) -> anyhow::Result<RealEvalArtifact> {
        // Real data mode not yet implemented in this backend.
        anyhow::bail!("SyntheticBackend does not support real-data evaluation");
    }
}

// =========================================================================
// Helper: build initial ModelParams for EM from quantile splitting
// =========================================================================

fn init_params_from_obs(obs: &[f64], k: usize) -> anyhow::Result<ModelParams> {
    if obs.is_empty() || k < 2 {
        anyhow::bail!("init_params_from_obs: need at least 1 observation and k >= 2");
    }

    let mut sorted = obs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();

    // Split into k quantile groups and compute per-group mean/var.
    let chunk = (n / k).max(1);
    let mut means = Vec::with_capacity(k);
    let mut variances = Vec::with_capacity(k);

    for j in 0..k {
        let start = j * chunk;
        let end = if j == k - 1 { n } else { (j + 1) * chunk };
        let slice = &sorted[start..end];
        let m = slice.iter().sum::<f64>() / slice.len() as f64;
        let v = slice.iter().map(|x| (x - m).powi(2)).sum::<f64>() / slice.len() as f64;
        means.push(m);
        variances.push(v.max(1e-6));
    }

    // Uniform initial distribution.
    let pi = vec![1.0 / k as f64; k];

    // High self-transition (0.9 on diagonal), uniform off-diagonal.
    let off_diag = 0.1 / (k - 1) as f64;
    let transition_rows: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            (0..k)
                .map(|j| if i == j { 0.9 } else { off_diag })
                .collect()
        })
        .collect();

    Ok(ModelParams::new(pi, transition_rows, means, variances))
}

// =========================================================================
// Helper: build ground-truth ModelParams for a named scenario
// =========================================================================

fn build_synthetic_params(scenario_id: &str, k: usize) -> anyhow::Result<ModelParams> {
    match scenario_id {
        "scenario_calibrated" | "calm_turbulent" | "" => {
            // K=2: low-vol calm vs high-vol turbulent (log-return scale)
            Ok(ModelParams::new(
                vec![0.5, 0.5],
                vec![vec![0.96, 0.04], vec![0.12, 0.88]],
                vec![0.0002, -0.0001],
                vec![0.0001, 0.0004],
            ))
        }
        "persistent_states" => {
            Ok(ModelParams::new(
                vec![0.5, 0.5],
                vec![vec![0.975, 0.025], vec![0.067, 0.933]], // durations ~40 and ~15
                vec![0.0001, -0.0001],
                vec![0.00008, 0.00032],
            ))
        }
        "shock_contaminated" => {
            Ok(ModelParams::new(
                vec![0.5, 0.5],
                vec![vec![0.96, 0.04], vec![0.12, 0.88]],
                vec![0.0002, -0.0005],
                vec![0.0001, 0.0016], // high-vol with occasional shocks
            ))
        }
        _ if k == 3 => {
            // K=3: calm / moderate / volatile
            Ok(ModelParams::new(
                vec![0.34, 0.33, 0.33],
                vec![
                    vec![0.971, 0.020, 0.009],
                    vec![0.013, 0.930, 0.057],
                    vec![0.007, 0.136, 0.857],
                ],
                vec![0.0003, 0.0, -0.0002],
                vec![0.00009, 0.00018, 0.0004],
            ))
        }
        _ => {
            // Fall back to a 2-regime default regardless of name.
            build_synthetic_params("scenario_calibrated", k)
        }
    }
}

// =========================================================================
// Helper: re-simulate changepoints for evaluation stage
// =========================================================================

fn regenerate_changepoints(cfg: &ExperimentConfig) -> anyhow::Result<(Vec<usize>, usize)> {
    match &cfg.data {
        DataConfig::Synthetic {
            scenario_id,
            horizon,
            ..
        } => {
            let horizon = *horizon;
            let seed = cfg.reproducibility.seed.unwrap_or(42);
            let mut rng = StdRng::seed_from_u64(seed);
            let params = build_synthetic_params(scenario_id, cfg.model.k_regimes)?;
            let sim = simulate(params, horizon, &mut rng)?;
            let changepoints: Vec<usize> = (1..sim.states.len())
                .filter(|&i| sim.states[i] != sim.states[i - 1])
                .map(|i| i + 1)
                .collect();
            Ok((changepoints, horizon))
        }
        DataConfig::Real { .. } => Ok((vec![], 0)),
    }
}

// =========================================================================
// Helper: map config DetectorType to detector::DetectorKind
// =========================================================================

fn map_detector_kind(
    dt: &crate::experiments::config::DetectorType,
) -> crate::detector::DetectorKind {
    use crate::detector::DetectorKind;
    use crate::experiments::config::DetectorType;
    match dt {
        DetectorType::HardSwitch => DetectorKind::HardSwitch,
        DetectorType::PosteriorTransition => DetectorKind::PosteriorTransition,
        DetectorType::Surprise => DetectorKind::Surprise,
    }
}
