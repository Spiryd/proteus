/// Shared pipeline stages for experiment backends.
///
/// Both `SyntheticBackend` and `RealBackend` perform identical Baum-Welch EM
/// training and identical online Hamilton-filter detection.  These functions
/// encapsulate those stages once so both backends can call them without
/// duplicating code.
///
/// # Theoretical justification
///
/// The Markov-Switching Model (MSM) and its associated learning and inference
/// algorithms are data-agnostic: they operate on a sequence of scalar
/// observations $y_1, \ldots, y_T$ and are indifferent to whether those
/// observations were simulated or sourced from a real exchange.  The only
/// difference between the two experiment modes is how $y_t$ is produced and
/// how results are evaluated.
use crate::detector::{
    HardSwitchConfig, HardSwitchDetector, PersistencePolicy, PosteriorTransitionConfig,
    PosteriorTransitionDetector, SurpriseConfig, SurpriseDetector,
};
use crate::detector::frozen::{FrozenModel, StreamingSession};
use crate::model::em::{EmConfig, fit_em};
use crate::model::params::ModelParams;
use crate::online::OnlineFilterState;

use super::config::{DetectorType, ExperimentConfig, TrainingMode};
use super::result::FittedParamsSummary;
use super::runner::{FeatureBundle, ModelArtifact, OnlineRunArtifact};

// =========================================================================
// Stage 3 — Train or load model
// =========================================================================

/// Run Baum-Welch EM on the training partition, or load a frozen model from
/// disk.
///
/// Shared between `SyntheticBackend` and `RealBackend`.
pub fn train_or_load_model_shared(
    cfg: &ExperimentConfig,
    features: &FeatureBundle,
) -> anyhow::Result<ModelArtifact> {
    if features.observations.is_empty() {
        return Ok(ModelArtifact {
            source: "stub:empty".to_string(),
            k_regimes: cfg.model.k_regimes,
            diagnostics_ok: false,
            params: None,
            ll_history: vec![],
            n_iter: 0,
            converged: false,
            diagnostics: None,
            multi_start_summary: None,
        });
    }

    match &cfg.model.training {
        TrainingMode::FitOffline => {
            let train_obs = &features.observations[..features.train_n.min(features.observations.len())];
            let k = cfg.model.k_regimes;
            let n_starts = cfg.model.em_n_starts.max(1);
            let em_cfg = EmConfig {
                tol: cfg.model.em_tol,
                max_iter: cfg.model.em_max_iter,
                var_floor: 1e-6,
            };

            // Run EM n_starts times; keep the result with the highest final
            // log-likelihood (multi-start robustness).
            let mut all_results: Vec<crate::model::em::EmResult> =
                Vec::with_capacity(n_starts);
            for _ in 0..n_starts {
                let init_params = init_params_from_obs(train_obs, k)?;
                if let Ok(r) = fit_em(train_obs, init_params, &em_cfg) {
                    all_results.push(r);
                }
            }
            if all_results.is_empty() {
                anyhow::bail!("all {n_starts} EM start(s) failed");
            }

            // Run cross-run diagnostics when we have more than one result.
            let multi_start_summary = if all_results.len() > 1 {
                crate::model::compare_runs(&all_results, train_obs).ok()
            } else {
                None
            };

            // Pick the run with the highest final log-likelihood.
            let best_idx = all_results
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    let ll_a = a.ll_history.last().copied().unwrap_or(f64::NEG_INFINITY);
                    let ll_b = b.ll_history.last().copied().unwrap_or(f64::NEG_INFINITY);
                    ll_a.partial_cmp(&ll_b).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            let em_result = all_results.swap_remove(best_idx);
            let converged = em_result.converged;
            let n_iter = em_result.n_iter;
            let ll_history = em_result.ll_history.clone();
            let params = em_result.params.clone();

            // Run the post-fit trust diagnostic on the best result.
            let diagnostics = crate::model::diagnose(&em_result, train_obs).ok();
            let diagnostics_ok = diagnostics
                .as_ref()
                .map(|d| d.is_trustworthy)
                .unwrap_or(converged);

            Ok(ModelArtifact {
                source: "fitted".to_string(),
                k_regimes: k,
                diagnostics_ok,
                params: Some(params),
                ll_history,
                n_iter,
                converged,
                diagnostics,
                multi_start_summary,
            })
        }
        TrainingMode::LoadFrozen { artifact_id } => {
            let path = std::path::PathBuf::from(artifact_id).join("model_params.json");
            if path.exists() {
                let json = std::fs::read_to_string(&path)?;
                let fps: FittedParamsSummary = serde_json::from_str(&json)?;
                let params = ModelParams::new(fps.pi, fps.transition, fps.means, fps.variances);
                let k = params.k;
                Ok(ModelArtifact {
                    source: format!("loaded:{artifact_id}"),
                    k_regimes: k,
                    diagnostics_ok: true,
                    params: Some(params),
                    ll_history: fps.ll_history,
                    n_iter: fps.n_iter,
                    converged: fps.converged,
                    diagnostics: None,
                    multi_start_summary: None,
                })
            } else {
                anyhow::bail!(
                    "LoadFrozen: model file not found at {}",
                    path.display()
                )
            }
        }
    }
}

// =========================================================================
// Stage 4 — Online detection
// =========================================================================

/// Stream all feature observations through the Hamilton filter and detector,
/// collecting alarms, scores, and posteriors.
///
/// Shared between `SyntheticBackend` and `RealBackend`.
pub fn run_online_shared(
    cfg: &ExperimentConfig,
    model: &ModelArtifact,
    features: &FeatureBundle,
) -> anyhow::Result<OnlineRunArtifact> {
    let params = model.params.as_ref().ok_or_else(|| {
        anyhow::anyhow!("run_online: no fitted model params available")
    })?;

    let frozen = FrozenModel::new(params.clone())?;
    let filter_state = OnlineFilterState::new(frozen.params());
    let save = cfg.output.save_traces;
    let obs = &features.observations;

    macro_rules! run_session {
        ($detector:expr) => {{
            let mut session = StreamingSession::new(frozen, filter_state, $detector);
            let outputs = session.step_batch(obs)?;

            let mut alarm_indices = Vec::new();
            let mut score_trace = if save { Vec::with_capacity(obs.len()) } else { vec![] };
            let mut regime_posteriors = if save { Vec::with_capacity(obs.len()) } else { vec![] };

            for out in &outputs {
                if save {
                    score_trace.push(out.detector.score);
                    regime_posteriors.push(out.filter.filtered.clone());
                }
                if out.detector.alarm {
                    alarm_indices.push(out.filter.t);
                }
            }

            Ok(OnlineRunArtifact {
                n_steps: obs.len(),
                n_alarms: alarm_indices.len(),
                alarm_indices,
                score_trace,
                regime_posteriors,
            })
        }};
    }

    let det = &cfg.detector;
    let persistence = PersistencePolicy::new(det.persistence_required, det.cooldown);

    match &det.detector_type {
        DetectorType::HardSwitch => {
            let detector = HardSwitchDetector::new(HardSwitchConfig {
                confidence_threshold: det.threshold.max(0.0),
                persistence,
            });
            run_session!(detector)
        }
        DetectorType::PosteriorTransition => {
            let detector = PosteriorTransitionDetector::new(PosteriorTransitionConfig {
                score_kind: crate::detector::PosteriorTransitionScoreKind::LeavePrevious,
                threshold: det.threshold,
                persistence,
            });
            run_session!(detector)
        }
        DetectorType::PosteriorTransitionTV => {
            let detector = PosteriorTransitionDetector::new(PosteriorTransitionConfig {
                score_kind: crate::detector::PosteriorTransitionScoreKind::TotalVariation,
                threshold: det.threshold,
                persistence,
            });
            run_session!(detector)
        }
        DetectorType::Surprise => {
            let detector = SurpriseDetector::new(SurpriseConfig {
                threshold: det.threshold,
                ema_alpha: det.ema_alpha,
                persistence,
            });
            run_session!(detector)
        }
    }
}

// =========================================================================
// Helper: build initial ModelParams for EM from quantile splitting
// =========================================================================

/// Initialise EM parameters by splitting observations into `k` quantile bins
/// and computing per-bin mean and variance.
///
/// **Why quantile initialisation?**  K-means or random restarts are more
/// robust but require more computation.  Quantile splitting provides a
/// sensible warm start that avoids symmetry breaking issues common with
/// uniform initialisations: the k groups already differ in mean, so the EM
/// will not degenerate to k copies of the same Gaussian.
pub fn init_params_from_obs(obs: &[f64], k: usize) -> anyhow::Result<ModelParams> {
    if obs.is_empty() || k < 2 {
        anyhow::bail!("init_params_from_obs: need at least 1 observation and k >= 2");
    }

    let mut sorted = obs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();

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

    let pi = vec![1.0 / k as f64; k];
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
