#![allow(dead_code)]
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use super::artifact::{prepare_run_dir, snapshot_config, snapshot_result, write_json_file};
use super::config::{DataConfig, EvaluationConfig, ExperimentConfig, ExperimentMode, TrainingMode};
use super::result::{
    ArtifactRef, DetectorSummary, EvaluationSummary, ExperimentResult, FittedParamsSummary,
    ModelSummary, RunMetadata, RunStage, RunStatus, StageTiming,
};

#[derive(Debug, Clone)]
pub struct DataBundle {
    pub dataset_key: String,
    pub n_observations: usize,
    /// Actual observation sequence (empty in DryRunBackend).
    pub observations: Vec<f64>,
    /// Ground-truth changepoint times (1-based); None if real data or dry run.
    pub changepoint_truth: Option<Vec<usize>>,
    /// Number of observations belonging to the training partition.
    pub train_n: usize,
    /// Bar timestamps parallel to `observations`.  Empty for synthetic / dry-run.
    pub timestamps: Vec<chrono::NaiveDateTime>,
    /// Split summary JSON (populated by RealBackend; None for synthetic/dry-run).
    pub split_summary_json: Option<String>,
    /// Data-quality validation report JSON (populated by RealBackend; None for synthetic/dry-run).
    pub validation_report_json: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FeatureBundle {
    pub feature_label: String,
    pub n_observations: usize,
    /// Transformed observation sequence (empty in DryRunBackend).
    pub observations: Vec<f64>,
    /// Training partition size (mirrors DataBundle::train_n after warmup trim).
    pub train_n: usize,
    /// Bar timestamps parallel to `observations`.  Empty for synthetic / dry-run.
    pub timestamps: Vec<chrono::NaiveDateTime>,
}

#[derive(Debug, Clone)]
pub struct ModelArtifact {
    pub source: String,
    pub k_regimes: usize,
    pub diagnostics_ok: bool,
    /// Fitted model parameters; None in DryRunBackend.
    pub params: Option<crate::model::ModelParams>,
    /// EM log-likelihood history (empty in DryRunBackend).
    pub ll_history: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    /// Full diagnostics bundle from the post-fit trust layer; None in
    /// DryRunBackend and LoadFrozen paths.
    pub diagnostics: Option<crate::model::FittedModelDiagnostics>,
    /// Cross-run comparison summary; populated only when `em_n_starts > 1`.
    pub multi_start_summary: Option<crate::model::MultiStartSummary>,
}

#[derive(Debug, Clone)]
pub struct OnlineRunArtifact {
    pub n_steps: usize,
    pub n_alarms: usize,
    /// 1-based step indices at which alarms fired (empty in DryRunBackend).
    pub alarm_indices: Vec<usize>,
    /// Detector score per step (empty unless save_traces = true).
    pub score_trace: Vec<f64>,
    /// Filtered regime posteriors per step (empty unless save_traces = true).
    pub regime_posteriors: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct SyntheticEvalArtifact {
    pub n_events: usize,
    pub coverage: f64,
    pub precision_like: f64,
    // Full metric suite; None in DryRunBackend.
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub miss_rate: Option<f64>,
    pub false_alarm_rate: Option<f64>,
    pub delay_mean: Option<f64>,
    pub delay_median: Option<f64>,
    pub n_true_positive: Option<usize>,
    pub n_false_positive: Option<usize>,
    pub n_missed: Option<usize>,
    /// Per-event detection delays in bars; empty in DryRunBackend.
    pub per_event_delays: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct RealEvalArtifact {
    pub event_coverage: f64,
    pub alarm_relevance: f64,
    pub segmentation_coherence: f64,
    /// Full Route A result JSON (populated by RealBackend; None for dry-run).
    pub route_a_result_json: Option<String>,
    /// Full Route B result JSON (populated by RealBackend; None for dry-run).
    pub route_b_result_json: Option<String>,
}

pub trait ExperimentBackend {
    fn resolve_data(&self, cfg: &ExperimentConfig) -> anyhow::Result<DataBundle>;
    fn build_features(
        &self,
        cfg: &ExperimentConfig,
        data: &DataBundle,
    ) -> anyhow::Result<FeatureBundle>;
    fn train_or_load_model(
        &self,
        cfg: &ExperimentConfig,
        features: &FeatureBundle,
    ) -> anyhow::Result<ModelArtifact>;
    fn run_online(
        &self,
        cfg: &ExperimentConfig,
        model: &ModelArtifact,
        features: &FeatureBundle,
    ) -> anyhow::Result<OnlineRunArtifact>;
    fn evaluate_synthetic(
        &self,
        cfg: &ExperimentConfig,
        online: &OnlineRunArtifact,
    ) -> anyhow::Result<SyntheticEvalArtifact>;
    fn evaluate_real(
        &self,
        cfg: &ExperimentConfig,
        online: &OnlineRunArtifact,
    ) -> anyhow::Result<RealEvalArtifact>;
}

/// Default backend used for orchestration tests and dry-runs.
#[derive(Debug, Clone, Default)]
pub struct DryRunBackend;

impl ExperimentBackend for DryRunBackend {
    fn resolve_data(&self, cfg: &ExperimentConfig) -> anyhow::Result<DataBundle> {
        let n = match &cfg.data {
            DataConfig::Synthetic { horizon, .. } => *horizon,
            DataConfig::Real { .. } => 500,
        };
        Ok(DataBundle {
            dataset_key: match &cfg.data {
                DataConfig::Synthetic { scenario_id, .. } => format!("synthetic:{scenario_id}"),
                DataConfig::Real { dataset_id, .. } => format!("real:{dataset_id}"),
            },
            n_observations: n,
            observations: vec![],
            changepoint_truth: None,
            train_n: (n as f64 * 0.7) as usize,
            timestamps: vec![],
            split_summary_json: None,
            validation_report_json: None,
        })
    }

    fn build_features(
        &self,
        cfg: &ExperimentConfig,
        data: &DataBundle,
    ) -> anyhow::Result<FeatureBundle> {
        let n = data.n_observations.saturating_sub(1);
        Ok(FeatureBundle {
            feature_label: format!("{:?}", cfg.features.family),
            n_observations: n,
            observations: vec![],
            train_n: (n as f64 * 0.7) as usize,
            timestamps: vec![],
        })
    }

    fn train_or_load_model(
        &self,
        cfg: &ExperimentConfig,
        _features: &FeatureBundle,
    ) -> anyhow::Result<ModelArtifact> {
        let source = match &cfg.model.training {
            TrainingMode::FitOffline => "fitted".to_string(),
            TrainingMode::LoadFrozen { artifact_id } => format!("loaded:{artifact_id}"),
        };
        Ok(ModelArtifact {
            source,
            k_regimes: cfg.model.k_regimes,
            diagnostics_ok: true,
            params: None,
            ll_history: vec![],
            n_iter: 0,
            converged: false,
            diagnostics: None,
            multi_start_summary: None,
        })
    }

    fn run_online(
        &self,
        cfg: &ExperimentConfig,
        _model: &ModelArtifact,
        features: &FeatureBundle,
    ) -> anyhow::Result<OnlineRunArtifact> {
        // Deterministic pseudo alarm count for reproducibility tests.
        let base = features.n_observations as f64;
        let alarms =
            ((base / 100.0) * (1.0 + cfg.detector.threshold.abs() / 10.0)).round() as usize;
        Ok(OnlineRunArtifact {
            n_steps: features.n_observations,
            n_alarms: alarms,
            alarm_indices: vec![],
            score_trace: vec![],
            regime_posteriors: vec![],
        })
    }

    fn evaluate_synthetic(
        &self,
        cfg: &ExperimentConfig,
        online: &OnlineRunArtifact,
    ) -> anyhow::Result<SyntheticEvalArtifact> {
        let w = match cfg.evaluation {
            EvaluationConfig::Synthetic { matching_window } => matching_window,
            _ => 1,
        };
        let coverage =
            (online.n_alarms as f64 / (w.max(1) as f64 + online.n_alarms as f64)).min(1.0);
        Ok(SyntheticEvalArtifact {
            n_events: w.max(1),
            coverage,
            precision_like: 1.0 - (1.0 / (1.0 + online.n_alarms as f64)),
            precision: None,
            recall: None,
            miss_rate: None,
            false_alarm_rate: None,
            delay_mean: None,
            delay_median: None,
            n_true_positive: None,
            n_false_positive: None,
            n_missed: None,
            per_event_delays: vec![],
        })
    }

    fn evaluate_real(
        &self,
        cfg: &ExperimentConfig,
        online: &OnlineRunArtifact,
    ) -> anyhow::Result<RealEvalArtifact> {
        let (post, min_seg) = match cfg.evaluation {
            EvaluationConfig::Real {
                route_a_point_post_bars,
                route_b_min_segment_len,
                ..
            } => (route_a_point_post_bars, route_b_min_segment_len),
            _ => (1, 5),
        };
        Ok(RealEvalArtifact {
            event_coverage: (online.n_alarms as f64
                / (post.max(1) as f64 + online.n_alarms as f64))
                .min(1.0),
            alarm_relevance: if online.n_alarms == 0 { 0.0 } else { 0.5 },
            segmentation_coherence: 1.0 / (min_seg.max(1) as f64),
            route_a_result_json: None,
            route_b_result_json: None,
        })
    }
}

pub struct ExperimentRunner<B: ExperimentBackend> {
    backend: B,
}

impl<B: ExperimentBackend> ExperimentRunner<B> {
    pub fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn run(&self, cfg: ExperimentConfig) -> ExperimentResult {
        let started_at = now_epoch_ms();
        let config_hash = hash_config(&cfg);
        let run_id = generate_run_id(&cfg, config_hash);

        let mode_str = match cfg.mode {
            ExperimentMode::Synthetic => "synthetic",
            ExperimentMode::Real => "real",
        }
        .to_string();

        let mut artifacts = Vec::<ArtifactRef>::new();
        let mut timings = Vec::<StageTiming>::new();
        let mut warnings = Vec::<String>::new();
        #[allow(unused_assignments)]
        let mut model_summary = None;
        #[allow(unused_assignments)]
        let mut detector_summary = None;
        #[allow(unused_assignments)]
        let mut evaluation_summary = None;

        let mut status = RunStatus::Success;

        let run_dir = match prepare_run_dir(
            cfg.output.root_dir.as_str(),
            mode_str.as_str(),
            cfg.meta.run_label.as_str(),
            run_id.as_str(),
        ) {
            Ok(p) => p,
            Err(e) => {
                return self.failed_result(
                    &cfg,
                    run_id,
                    config_hash,
                    started_at,
                    RunStage::Export,
                    format!("failed to prepare output directory: {e}"),
                );
            }
        };

        if cfg.reproducibility.save_config_snapshot
            && let Ok(a) = snapshot_config(&run_dir, &cfg)
        {
            artifacts.push(a);
        }

        if let Err(e) = cfg.validate() {
            return self.failed_with_artifacts(
                &cfg,
                run_id,
                config_hash,
                started_at,
                RunStage::ResolveData,
                format!("invalid config: {e}"),
                artifacts,
                run_dir,
            );
        }

        let data = match timed_stage(RunStage::ResolveData, &mut timings, || {
            self.backend.resolve_data(&cfg)
        }) {
            Ok(v) => v,
            Err(e) => {
                return self.failed_with_artifacts(
                    &cfg,
                    run_id,
                    config_hash,
                    started_at,
                    RunStage::ResolveData,
                    format!("data resolution failed: {e}"),
                    artifacts,
                    run_dir,
                );
            }
        };

        let features = match timed_stage(RunStage::BuildFeatures, &mut timings, || {
            self.backend.build_features(&cfg, &data)
        }) {
            Ok(v) => v,
            Err(e) => {
                return self.failed_with_artifacts(
                    &cfg,
                    run_id,
                    config_hash,
                    started_at,
                    RunStage::BuildFeatures,
                    format!("feature build failed: {e}"),
                    artifacts,
                    run_dir,
                );
            }
        };

        let model = match timed_stage(RunStage::TrainOrLoadModel, &mut timings, || {
            self.backend.train_or_load_model(&cfg, &features)
        }) {
            Ok(v) => {
                let fitted_params = v.params.as_ref().map(|p| {
                    let k = p.k;
                    let transition_rows: Vec<Vec<f64>> = (0..k)
                        .map(|i| p.transition[i * k..(i + 1) * k].to_vec())
                        .collect();
                    FittedParamsSummary {
                        pi: p.pi.clone(),
                        transition: transition_rows,
                        means: p.means.clone(),
                        variances: p.variances.clone(),
                        log_likelihood: v.ll_history.last().copied().unwrap_or(f64::NAN),
                        ll_history: v.ll_history.clone(),
                        n_iter: v.n_iter,
                        converged: v.converged,
                    }
                });
                model_summary = Some(ModelSummary {
                    k_regimes: v.k_regimes,
                    training_mode: v.source.clone(),
                    diagnostics_ok: v.diagnostics_ok,
                    fitted_params,
                });
                v
            }
            Err(e) => {
                return self.failed_with_artifacts(
                    &cfg,
                    run_id,
                    config_hash,
                    started_at,
                    RunStage::TrainOrLoadModel,
                    format!("training/model load failed: {e}"),
                    artifacts,
                    run_dir,
                );
            }
        };

        let online = match timed_stage(RunStage::RunOnline, &mut timings, || {
            self.backend.run_online(&cfg, &model, &features)
        }) {
            Ok(v) => {
                detector_summary = Some(DetectorSummary {
                    detector_type: format!("{:?}", cfg.detector.detector_type),
                    threshold: cfg.detector.threshold,
                    persistence_required: cfg.detector.persistence_required,
                    cooldown: cfg.detector.cooldown,
                    n_alarms: v.n_alarms,
                    alarm_indices: v.alarm_indices.clone(),
                });
                v
            }
            Err(e) => {
                return self.failed_with_artifacts(
                    &cfg,
                    run_id,
                    config_hash,
                    started_at,
                    RunStage::RunOnline,
                    format!("online execution failed: {e}"),
                    artifacts,
                    run_dir,
                );
            }
        };

        // Evaluate — inline timing so we can capture the full artifact before
        // it is consumed by the EvaluationSummary conversion.
        #[allow(unused_assignments)]
        let mut real_eval_artifact: Option<RealEvalArtifact> = None;
        #[allow(unused_assignments)]
        let mut syn_eval_artifact: Option<SyntheticEvalArtifact> = None;
        let eval_t0 = Instant::now();
        let eval_out: anyhow::Result<EvaluationSummary> = match cfg.mode {
            ExperimentMode::Synthetic => {
                self.backend.evaluate_synthetic(&cfg, &online).map(|s| {
                    let summary = EvaluationSummary::Synthetic {
                        n_events: s.n_events,
                        coverage: s.coverage,
                        precision_like: s.precision_like,
                        precision: s.precision,
                        recall: s.recall,
                        miss_rate: s.miss_rate,
                        false_alarm_rate: s.false_alarm_rate,
                        delay_mean: s.delay_mean,
                        delay_median: s.delay_median,
                        n_true_positive: s.n_true_positive,
                        n_false_positive: s.n_false_positive,
                        n_missed: s.n_missed,
                    };
                    syn_eval_artifact = Some(s);
                    summary
                })
            }
            ExperimentMode::Real => {
                self.backend.evaluate_real(&cfg, &online).map(|artifact| {
                    let summary = EvaluationSummary::Real {
                        event_coverage: artifact.event_coverage,
                        alarm_relevance: artifact.alarm_relevance,
                        segmentation_coherence: artifact.segmentation_coherence,
                    };
                    real_eval_artifact = Some(artifact);
                    summary
                })
            }
        };
        timings.push(StageTiming {
            stage: RunStage::Evaluate,
            duration_ms: eval_t0.elapsed().as_millis(),
        });
        let eval_result = match eval_out {
            Ok(v) => v,
            Err(e) => {
                return self.failed_with_artifacts(
                    &cfg,
                    run_id,
                    config_hash,
                    started_at,
                    RunStage::Evaluate,
                    format!("evaluation failed: {e}"),
                    artifacts,
                    run_dir,
                );
            }
        };

        evaluation_summary = Some(eval_result);

        let score_trace = if cfg.output.save_traces {
            online.score_trace.clone()
        } else {
            vec![]
        };
        let regime_posteriors = if cfg.output.save_traces {
            online.regime_posteriors.clone()
        } else {
            vec![]
        };

        let mut result = ExperimentResult {
            metadata: RunMetadata {
                run_id: run_id.clone(),
                run_label: cfg.meta.run_label.clone(),
                mode: mode_str,
                started_at_epoch_ms: started_at,
                finished_at_epoch_ms: now_epoch_ms(),
                seed: cfg.reproducibility.seed,
                config_hash,
            },
            status: status.clone(),
            timings,
            model_summary,
            detector_summary,
            evaluation_summary,
            artifacts,
            warnings: vec![],
            score_trace,
            regime_posteriors,
        };

        let export_t0 = Instant::now();
        let export_result = if cfg.output.write_json {
            match snapshot_result(&run_dir, &result) {
                Ok(a) => {
                    result.artifacts.push(a);
                    let summary_path = run_dir.join("summary.json");
                    match write_json_file(&summary_path, &result.evaluation_summary) {
                        Ok(()) => {
                            result.artifacts.push(ArtifactRef {
                                name: "summary".to_string(),
                                path: summary_path.to_string_lossy().to_string(),
                                kind: "json".to_string(),
                            });
                            Ok(())
                        }
                        Err(e) => Err(e),
                    }
                }
                Err(e) => Err(e),
            }
        } else {
            Ok(())
        };

        // Save fitted model params as JSON for LoadFrozen reuse
        if cfg.output.write_json
            && let Some(ms) = &result.model_summary
            && let Some(fp) = &ms.fitted_params
        {
            let params_path = run_dir.join("model_params.json");
            if let Ok(json) = serde_json::to_string_pretty(fp)
                && let Ok(()) = std::fs::write(&params_path, &json)
            {
                result.artifacts.push(ArtifactRef {
                    name: "model_params".to_string(),
                    path: params_path.to_string_lossy().to_string(),
                    kind: "json".to_string(),
                });
            }

            // Save human-readable fit_summary.json (subset of model_params).
            let fit_summary = serde_json::json!({
                "k_regimes": cfg.model.k_regimes,
                "n_iter": fp.n_iter,
                "converged": fp.converged,
                "log_likelihood_final": fp.log_likelihood,
                "log_likelihood_initial": fp.ll_history.first().copied().unwrap_or(f64::NAN),
                "pi": fp.pi,
                "transition": fp.transition,
                "means": fp.means,
                "variances": fp.variances,
                "convergence_reason": if fp.converged { "tolerance_met" } else { "max_iter_reached" },
            });
            let fit_path = run_dir.join("fit_summary.json");
            if let Ok(json) = serde_json::to_string_pretty(&fit_summary)
                && let Ok(()) = std::fs::write(&fit_path, json)
            {
                result.artifacts.push(ArtifactRef {
                    name: "fit_summary".to_string(),
                    path: fit_path.to_string_lossy().to_string(),
                    kind: "json".to_string(),
                });
            }
        }

        // Save diagnostics.json — post-fit trust layer output.
        if cfg.output.write_json
            && let Some(diag) = &model.diagnostics
        {
            let diag_path = run_dir.join("diagnostics.json");
            if let Ok(json) = serde_json::to_string_pretty(&diagnostics_to_json(diag))
                && let Ok(()) = std::fs::write(&diag_path, json)
            {
                result.artifacts.push(ArtifactRef {
                    name: "diagnostics".to_string(),
                    path: diag_path.to_string_lossy().to_string(),
                    kind: "json".to_string(),
                });
            }
            // Propagate diagnostic warnings into run warnings.
            for w in &diag.warnings {
                warnings.push(format!("diagnostic: {}", w.description()));
            }
        }

        // Save multi_start_summary.json — populated when em_n_starts > 1.
        if cfg.output.write_json
            && let Some(ms) = &model.multi_start_summary
        {
            let mss_path = run_dir.join("multi_start_summary.json");
            if let Ok(json) = serde_json::to_string_pretty(&multi_start_summary_to_json(ms))
                && let Ok(()) = std::fs::write(&mss_path, json)
            {
                result.artifacts.push(ArtifactRef {
                    name: "multi_start_summary".to_string(),
                    path: mss_path.to_string_lossy().to_string(),
                    kind: "json".to_string(),
                });
            }
            // Propagate instability warning from multi-start comparison.
            for w in &ms.warnings {
                warnings.push(format!("multi_start: {}", w.description()));
            }
        }

        // Save loglikelihood_history.csv — one row per EM iteration.
        if cfg.output.write_json
            && let Some(ms) = &result.model_summary
            && let Some(fp) = &ms.fitted_params
            && cfg.output.write_csv
            && !fp.ll_history.is_empty()
        {
            let ll_path = run_dir.join("loglikelihood_history.csv");
            let rows: Vec<String> = fp.ll_history.iter().enumerate()
                .map(|(i, &ll)| format!("{i},{ll:.8}"))
                .collect();
            let content = format!("iteration,log_likelihood\n{}", rows.join("\n"));
            if let Ok(()) = std::fs::write(&ll_path, content) {
                result.artifacts.push(ArtifactRef {
                    name: "loglikelihood_history".to_string(),
                    path: ll_path.to_string_lossy().to_string(),
                    kind: "csv".to_string(),
                });
            }
        }

        // Save feature_summary.json — feature pipeline metadata.
        if cfg.output.write_json {
            let obs_slice = &features.observations;
            let (fmean, fvar) = if obs_slice.is_empty() {
                (0.0_f64, 0.0_f64)
            } else {
                let m = obs_slice.iter().sum::<f64>() / obs_slice.len() as f64;
                let v = obs_slice.iter().map(|x| (x - m).powi(2)).sum::<f64>()
                    / obs_slice.len() as f64;
                (m, v)
            };
            let fmin = obs_slice.iter().copied().fold(f64::INFINITY, f64::min);
            let fmax = obs_slice.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let warmup_trimmed = data.n_observations.saturating_sub(features.n_observations);
            let feature_summary = serde_json::json!({
                "feature_label": features.feature_label,
                "n_raw_observations": data.n_observations,
                "n_feature_observations": features.n_observations,
                "warmup_trimmed": warmup_trimmed,
                "train_n": features.train_n,
                "val_n": features.n_observations.saturating_sub(features.train_n).saturating_sub(
                    features.n_observations.saturating_sub(features.train_n) / 2
                ),
                "scaling": format!("{:?}", cfg.features.scaling),
                "session_aware": cfg.features.session_aware,
                "obs_mean": fmean,
                "obs_variance": fvar,
                "obs_std": fvar.sqrt(),
                "obs_min": if fmin.is_finite() { fmin } else { 0.0 },
                "obs_max": if fmax.is_finite() { fmax } else { 0.0 },
            });
            let fs_path = run_dir.join("feature_summary.json");
            if let Ok(json) = serde_json::to_string_pretty(&feature_summary)
                && let Ok(()) = std::fs::write(&fs_path, json)
            {
                result.artifacts.push(ArtifactRef {
                    name: "feature_summary".to_string(),
                    path: fs_path.to_string_lossy().to_string(),
                    kind: "json".to_string(),
                });
            }
        }

        // Export score trace and alarm list as CSV if save_traces = true
        if cfg.output.save_traces && cfg.output.write_csv {            if !result.score_trace.is_empty() {
                let trace_path = run_dir.join("score_trace.csv");
                let rows: Vec<String> = result
                    .score_trace
                    .iter()
                    .enumerate()
                    .map(|(i, s)| format!("{},{s:.8}", i + 1))
                    .collect();
                let content = format!("t,score\n{}", rows.join("\n"));
                if let Ok(()) = std::fs::write(&trace_path, content) {
                    result.artifacts.push(ArtifactRef {
                        name: "score_trace".to_string(),
                        path: trace_path.to_string_lossy().to_string(),
                        kind: "csv".to_string(),
                    });
                }
            }
            if let Some(ds) = &result.detector_summary
                && !ds.alarm_indices.is_empty()
            {
                    let alarm_path = run_dir.join("alarms.csv");
                    let rows: Vec<String> = ds
                        .alarm_indices
                        .iter()
                        .enumerate()
                        .map(|(i, t)| format!("{},{}", i + 1, t))
                        .collect();
                    let content = format!("alarm_n,t\n{}", rows.join("\n"));
                    if let Ok(()) = std::fs::write(&alarm_path, content) {
                        result.artifacts.push(ArtifactRef {
                            name: "alarms".to_string(),
                            path: alarm_path.to_string_lossy().to_string(),
                            kind: "csv".to_string(),
                        });
                    }
            }
        }

        // Export real-evaluation summary CSVs (Route A alignment + Route B segmentation)
        if cfg.output.write_csv
            && let Some(EvaluationSummary::Real {
                event_coverage,
                alarm_relevance,
                segmentation_coherence,
            }) = &result.evaluation_summary
        {
            let real_eval_path = run_dir.join("real_eval_summary.csv");
            let content = format!(
                "metric,value\nevent_coverage,{event_coverage:.6}\nalarm_relevance,{alarm_relevance:.6}\nsegmentation_coherence,{segmentation_coherence:.6}\n"
            );
            if let Ok(()) = std::fs::write(&real_eval_path, content) {
                result.artifacts.push(ArtifactRef {
                    name: "real_eval_summary".to_string(),
                    path: real_eval_path.to_string_lossy().to_string(),
                    kind: "csv".to_string(),
                });
            }
        }

        // Save true changepoints (synthetic runs only).
        if cfg.output.write_csv
            && let Some(ref cps) = data.changepoint_truth
            && !cps.is_empty()
        {
            let cp_path = run_dir.join("changepoints.csv");
            let rows: Vec<String> = cps
                .iter()
                .enumerate()
                .map(|(i, &t)| format!("{},{}", i + 1, t))
                .collect();
            let content = format!("cp_n,t\n{}", rows.join("\n"));
            if let Ok(()) = std::fs::write(&cp_path, content) {
                result.artifacts.push(ArtifactRef {
                    name: "changepoints".to_string(),
                    path: cp_path.to_string_lossy().to_string(),
                    kind: "csv".to_string(),
                });
            }
        }

        // Save regime posteriors as CSV (when save_traces = true).
        if cfg.output.save_traces && cfg.output.write_csv && !result.regime_posteriors.is_empty() {
            let post_path = run_dir.join("regime_posteriors.csv");
            let k = cfg.model.k_regimes;
            let header = (0..k)
                .map(|j| format!("p{j}"))
                .collect::<Vec<_>>()
                .join(",");
            let rows: Vec<String> = result
                .regime_posteriors
                .iter()
                .enumerate()
                .map(|(i, post)| {
                    let vals: Vec<String> =
                        post.iter().map(|v| format!("{v:.8}")).collect();
                    format!("{},{}", i + 1, vals.join(","))
                })
                .collect();
            let content = format!("t,{header}\n{}", rows.join("\n"));
            if let Ok(()) = std::fs::write(&post_path, content) {
                result.artifacts.push(ArtifactRef {
                    name: "regime_posteriors".to_string(),
                    path: post_path.to_string_lossy().to_string(),
                    kind: "csv".to_string(),
                });
            }
        }

        // ---- Auto-generate plots ------------------------------------------------
        // Plots are generated whenever observations exist (any run with real data
        // or save_traces=true for synthetic).  For synthetic runs without real
        // timestamps, sequential day-indexed timestamps are synthesised.
        // Skipped in test builds (font renderer unavailable in headless CI).

        // Save detector_config.json — always written when write_json is true.
        if cfg.output.write_json
            && let Some(ds) = &result.detector_summary
        {
            let detector_cfg_json = serde_json::json!({
                "detector_type": ds.detector_type,
                "threshold": ds.threshold,
                "persistence_required": ds.persistence_required,
                "cooldown": ds.cooldown,
            });
            let dc_path = run_dir.join("detector_config.json");
            if let Ok(json) = serde_json::to_string_pretty(&detector_cfg_json)
                && let Ok(()) = std::fs::write(&dc_path, json)
            {
                result.artifacts.push(ArtifactRef {
                    name: "detector_config".to_string(),
                    path: dc_path.to_string_lossy().to_string(),
                    kind: "json".to_string(),
                });
            }
        }

        // Save split_summary.json and data_quality.json (populated by RealBackend).
        if cfg.output.write_json {
            if let Some(ref json) = data.split_summary_json {
                let path = run_dir.join("split_summary.json");
                if let Ok(()) = std::fs::write(&path, json) {
                    result.artifacts.push(ArtifactRef {
                        name: "split_summary".to_string(),
                        path: path.to_string_lossy().to_string(),
                        kind: "json".to_string(),
                    });
                }
            }
            if let Some(ref json) = data.validation_report_json {
                let path = run_dir.join("data_quality.json");
                if let Ok(()) = std::fs::write(&path, json) {
                    result.artifacts.push(ArtifactRef {
                        name: "data_quality".to_string(),
                        path: path.to_string_lossy().to_string(),
                        kind: "json".to_string(),
                    });
                }
            }
        }

        // Save Route A + B detail JSONs (populated by RealBackend).
        if cfg.output.write_json
            && let Some(ref rfa) = real_eval_artifact
        {
            if let Some(ref json) = rfa.route_a_result_json {
                let path = run_dir.join("route_a_result.json");
                if let Ok(()) = std::fs::write(&path, json) {
                    result.artifacts.push(ArtifactRef {
                        name: "route_a_result".to_string(),
                        path: path.to_string_lossy().to_string(),
                        kind: "json".to_string(),
                    });
                }
            }
            if let Some(ref json) = rfa.route_b_result_json {
                let path = run_dir.join("route_b_result.json");
                if let Ok(()) = std::fs::write(&path, json) {
                    result.artifacts.push(ArtifactRef {
                        name: "route_b_result".to_string(),
                        path: path.to_string_lossy().to_string(),
                        kind: "json".to_string(),
                    });
                }
            }
        }

        #[cfg(not(test))]
        {
            try_register_plot_font();
            let plot_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                generate_plots(
                    &cfg,
                    &data,
                    &features,
                    &mut result,
                    &mut warnings,
                    &run_dir,
                    syn_eval_artifact.as_ref(),
                    real_eval_artifact.as_ref(),
                );
            }));
            if plot_result.is_err() {
                warnings.push("plot generation skipped (font backend unavailable)".to_string());
            }
        }

        result.timings.push(StageTiming {
            stage: RunStage::Export,
            duration_ms: export_t0.elapsed().as_millis(),
        });

        if let Err(e) = export_result {
            warnings.push(format!("export stage failed: {e}"));
            status = RunStatus::PartialSuccess;
        }

        result.status = status;
        result.metadata.finished_at_epoch_ms = now_epoch_ms();
        result.warnings = warnings;
        result
    }

    fn failed_result(
        &self,
        cfg: &ExperimentConfig,
        run_id: String,
        config_hash: u64,
        started_at: u128,
        stage: RunStage,
        message: String,
    ) -> ExperimentResult {
        ExperimentResult {
            metadata: RunMetadata {
                run_id,
                run_label: cfg.meta.run_label.clone(),
                mode: format!("{:?}", cfg.mode).to_lowercase(),
                started_at_epoch_ms: started_at,
                finished_at_epoch_ms: now_epoch_ms(),
                seed: cfg.reproducibility.seed,
                config_hash,
            },
            status: RunStatus::Failed { stage, message },
            timings: vec![],
            model_summary: None,
            detector_summary: None,
            evaluation_summary: None,
            artifacts: vec![],
            warnings: vec![],
            score_trace: vec![],
            regime_posteriors: vec![],
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn failed_with_artifacts(
        &self,
        cfg: &ExperimentConfig,
        run_id: String,
        config_hash: u64,
        started_at: u128,
        stage: RunStage,
        message: String,
        artifacts: Vec<ArtifactRef>,
        run_dir: PathBuf,
    ) -> ExperimentResult {
        let mut r = ExperimentResult {
            metadata: RunMetadata {
                run_id,
                run_label: cfg.meta.run_label.clone(),
                mode: format!("{:?}", cfg.mode).to_lowercase(),
                started_at_epoch_ms: started_at,
                finished_at_epoch_ms: now_epoch_ms(),
                seed: cfg.reproducibility.seed,
                config_hash,
            },
            status: RunStatus::Failed {
                stage,
                message: message.clone(),
            },
            timings: vec![],
            model_summary: None,
            detector_summary: None,
            evaluation_summary: None,
            artifacts,
            warnings: vec![message],
            score_trace: vec![],
            regime_posteriors: vec![],
        };

        if cfg.output.write_json
            && let Ok(a) = snapshot_result(&run_dir, &r)
        {
            r.artifacts.push(a);
        }
        r
    }
}

fn timed_stage<T>(
    stage: RunStage,
    timings: &mut Vec<StageTiming>,
    f: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<T> {
    let t0 = Instant::now();
    let out = f();
    timings.push(StageTiming {
        stage,
        duration_ms: t0.elapsed().as_millis(),
    });
    out
}

fn now_epoch_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn hash_config(cfg: &ExperimentConfig) -> u64 {
    let json = serde_json::to_string(cfg).unwrap_or_default();
    let mut h = DefaultHasher::new();
    json.hash(&mut h);
    h.finish()
}

fn generate_run_id(cfg: &ExperimentConfig, config_hash: u64) -> String {
    if cfg.reproducibility.deterministic_run_id {
        let seed = cfg.reproducibility.seed.unwrap_or(0);
        format!("run_{config_hash:016x}_{seed}")
    } else {
        format!("run_{:016x}_{:x}", config_hash, now_epoch_ms())
    }
}

/// Try to register a Windows system font as "sans-serif" so plotters ab_glyph
/// can render text.  Called once before each plot batch.  Silently does nothing
/// if no suitable font file is found.
#[cfg(not(test))]
fn try_register_plot_font() {
    use plotters::style::FontStyle;
    let candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\verdana.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
    ];
    for path in &candidates {
        if let Ok(bytes) = std::fs::read(path) {
            let leaked: &'static [u8] = Box::leak(bytes.into_boxed_slice());
            let _ = plotters::style::register_font("sans-serif", FontStyle::Normal, leaked);
            let _ = plotters::style::register_font("sans-serif", FontStyle::Bold, leaked);
            return;
        }
    }
}

/// Generate signal, score, posterior, delay-distribution, and segmentation
/// plots and append artifacts to `result`.
/// Only called in non-test builds to avoid font-renderer panics in headless CI.
#[cfg(not(test))]
#[allow(clippy::too_many_arguments)]
fn generate_plots(
    cfg: &ExperimentConfig,
    data: &DataBundle,
    features: &FeatureBundle,
    result: &mut ExperimentResult,
    warnings: &mut Vec<String>,
    run_dir: &std::path::Path,
    syn_eval: Option<&SyntheticEvalArtifact>,
    real_eval: Option<&RealEvalArtifact>,
) {
    use crate::reporting::plot::{
        render_detector_scores, render_regime_posteriors, render_signal_with_alarms,
        DetectorScoresPlotInput, RegimePosteriorPlotInput, SignalWithAlarmsPlotInput,
    };

    let n_obs = features.observations.len();
    if n_obs == 0 {
        return;
    }

    let ts_utc = ensure_plot_timestamps(&features.timestamps, n_obs);

    let alarm_set: std::collections::HashSet<usize> = result
        .detector_summary
        .as_ref()
        .map(|ds| ds.alarm_indices.iter().copied().collect())
        .unwrap_or_default();
    let alarm_flags: Vec<(chrono::DateTime<chrono::Utc>, bool)> = ts_utc
        .iter()
        .enumerate()
        .map(|(i, ts)| (*ts, alarm_set.contains(&(i + 1))))
        .collect();

    let true_cps: Option<Vec<chrono::DateTime<chrono::Utc>>> =
        data.changepoint_truth.as_ref().map(|cps| {
            cps.iter()
                .filter_map(|&i| ts_utc.get(i.saturating_sub(1)).copied())
                .collect()
        });

    let feature_label = format!("{:?}", cfg.features.family);

    // Plot 1: Signal with alarms.
    let signal_input = SignalWithAlarmsPlotInput {
        timestamps: ts_utc.clone(),
        observations: features.observations.clone(),
        alarms: alarm_flags.clone(),
        true_changepoints: true_cps,
        title: format!("{} — Signal & Alarms", cfg.meta.run_label),
        y_label: feature_label,
    };
    let signal_path = run_dir.join("signal_alarms.png");
    match render_signal_with_alarms(&signal_input, &signal_path) {
        Ok(()) => result.artifacts.push(ArtifactRef {
            name: "plot_signal_alarms".to_string(),
            path: signal_path.to_string_lossy().to_string(),
            kind: "png".to_string(),
        }),
        Err(e) => warnings.push(format!("signal_alarms plot failed: {e}")),
    }

    // Plot 2: Detector scores.
    if !result.score_trace.is_empty() {
        let score_input = DetectorScoresPlotInput {
            timestamps: ts_utc.clone(),
            scores: result.score_trace.clone(),
            threshold: cfg.detector.threshold,
            alarms: alarm_flags.clone(),
            title: format!(
                "{} — Detector Scores ({:?})",
                cfg.meta.run_label, cfg.detector.detector_type
            ),
        };
        let score_path = run_dir.join("detector_scores.png");
        match render_detector_scores(&score_input, &score_path) {
            Ok(()) => result.artifacts.push(ArtifactRef {
                name: "plot_detector_scores".to_string(),
                path: score_path.to_string_lossy().to_string(),
                kind: "png".to_string(),
            }),
            Err(e) => warnings.push(format!("detector_scores plot failed: {e}")),
        }
    }

    // Plot 3: Regime posteriors.
    if !result.regime_posteriors.is_empty() {
        let post_input = RegimePosteriorPlotInput {
            timestamps: ts_utc.clone(),
            posteriors: result.regime_posteriors.clone(),
            title: format!(
                "{} — Regime Posteriors (K={})",
                cfg.meta.run_label, cfg.model.k_regimes
            ),
        };
        let post_path = run_dir.join("regime_posteriors.png");
        match render_regime_posteriors(&post_input, &post_path) {
            Ok(()) => result.artifacts.push(ArtifactRef {
                name: "plot_regime_posteriors".to_string(),
                path: post_path.to_string_lossy().to_string(),
                kind: "png".to_string(),
            }),
            Err(e) => warnings.push(format!("regime_posteriors plot failed: {e}")),
        }
    }

    // Plot 4: Detection delay distribution (synthetic runs with matched events).
    if let Some(se) = syn_eval
        && !se.per_event_delays.is_empty()
    {
        use crate::reporting::plot::{render_delay_distribution, DelayDistributionPlotInput};
        let delay_input = DelayDistributionPlotInput {
            delays: se.per_event_delays.clone(),
            title: format!("{} — Detection Delay Distribution", cfg.meta.run_label),
        };
        let delay_path = run_dir.join("delay_distribution.png");
        match render_delay_distribution(&delay_input, &delay_path) {
            Ok(()) => result.artifacts.push(ArtifactRef {
                name: "plot_delay_distribution".to_string(),
                path: delay_path.to_string_lossy().to_string(),
                kind: "png".to_string(),
            }),
            Err(e) => warnings.push(format!("delay_distribution plot failed: {e}")),
        }
    }

    // Plot 5: Route B segmentation (real runs).
    if let Some(re) = real_eval
        && let Some(ref route_b_json) = re.route_b_result_json
    {
        use crate::real_eval::route_b::SegmentationEvaluationResult;
        use crate::reporting::plot::{render_segmentation, SegmentationPlotInput};
        if let Ok(route_b) = serde_json::from_str::<SegmentationEvaluationResult>(route_b_json) {
            let seg_ts = ensure_plot_timestamps(&features.timestamps, features.observations.len());
            let segs: Vec<_> = route_b
                .segments
                .iter()
                .map(|s| (s.start_ts.and_utc(), s.end_ts.and_utc(), true))
                .collect();
            if !segs.is_empty() {
                let seg_input = SegmentationPlotInput {
                    timestamps: seg_ts,
                    observations: features.observations.clone(),
                    segments: segs,
                    title: format!("{} — Route B Segmentation", cfg.meta.run_label),
                };
                let seg_path = run_dir.join("segmentation.png");
                match render_segmentation(&seg_input, &seg_path) {
                    Ok(()) => result.artifacts.push(ArtifactRef {
                        name: "plot_segmentation".to_string(),
                        path: seg_path.to_string_lossy().to_string(),
                        kind: "png".to_string(),
                    }),
                    Err(e) => warnings.push(format!("segmentation plot failed: {e}")),
                }
            }
        }
    }
}

/// Serialises a [`MultiStartSummary`] to a JSON value for artifact export.
fn multi_start_summary_to_json(ms: &crate::model::MultiStartSummary) -> serde_json::Value {
    let runs: Vec<serde_json::Value> = ms
        .runs
        .iter()
        .map(|r| {
            let exp_durations: Vec<serde_json::Value> = r
                .expected_durations
                .iter()
                .map(|&d| {
                    if d.is_finite() {
                        serde_json::json!(d)
                    } else {
                        serde_json::Value::Null
                    }
                })
                .collect();
            serde_json::json!({
                "log_likelihood": r.log_likelihood,
                "n_iter": r.n_iter,
                "converged": r.converged,
                "ordered_means": r.ordered_means,
                "ordered_variances": r.ordered_variances,
                "expected_durations": exp_durations,
                "occupancy_shares": r.occupancy_shares,
            })
        })
        .collect();
    serde_json::json!({
        "n_starts": ms.runs.len(),
        "best_ll": ms.best_ll,
        "runner_up_ll": ms.runner_up_ll,
        "ll_spread": ms.ll_spread,
        "top2_gap": ms.top2_gap,
        "n_converged": ms.n_converged,
        "warnings": ms.warnings.iter().map(|w| w.description()).collect::<Vec<_>>(),
        "runs": runs,
    })
}

/// Serialises a [`FittedModelDiagnostics`] to a JSON value for artifact export.
///
/// `expected_durations` entries that are `f64::INFINITY` are represented as
/// `null` so the output is valid JSON.
fn diagnostics_to_json(diag: &crate::model::FittedModelDiagnostics) -> serde_json::Value {
    let exp_durations: Vec<serde_json::Value> = diag
        .regimes
        .expected_durations
        .iter()
        .map(|&d| {
            if d.is_finite() {
                serde_json::json!(d)
            } else {
                serde_json::Value::Null
            }
        })
        .collect();
    serde_json::json!({
        "is_trustworthy": diag.is_trustworthy,
        "param_validity": {
            "valid": diag.param_validity.valid,
            "max_pi_dev": diag.param_validity.max_pi_dev,
            "max_row_dev": diag.param_validity.max_row_dev,
            "min_variance": diag.param_validity.min_variance,
            "all_params_finite": diag.param_validity.all_params_finite,
        },
        "posterior_validity": {
            "max_filtered_dev": diag.posterior_validity.max_filtered_dev,
            "max_smoothed_dev": diag.posterior_validity.max_smoothed_dev,
            "max_pairwise_dev": diag.posterior_validity.max_pairwise_dev,
            "max_marginal_consistency_err": diag.posterior_validity.max_marginal_consistency_err,
        },
        "convergence": {
            "stop_reason": format!("{:?}", diag.convergence.stop_reason),
            "n_iter": diag.convergence.n_iter,
            "initial_ll": diag.convergence.initial_ll,
            "final_ll": diag.convergence.final_ll,
            "ll_gain": diag.convergence.ll_gain,
            "min_delta": diag.convergence.min_delta,
            "is_monotone": diag.convergence.is_monotone,
            "largest_negative_delta": diag.convergence.largest_negative_delta,
        },
        "regimes": {
            "means": diag.regimes.means,
            "variances": diag.regimes.variances,
            "occupancy_weights": diag.regimes.occupancy_weights,
            "occupancy_shares": diag.regimes.occupancy_shares,
            "self_transition_probs": diag.regimes.self_transition_probs,
            "expected_durations": exp_durations,
            "hard_counts": diag.regimes.hard_counts,
        },
        "warnings": diag.warnings.iter().map(|w| w.description()).collect::<Vec<_>>(),
    })
}

/// Returns `stored` as UTC if non-empty, otherwise synthesises sequential
/// day-indexed timestamps from 2000-01-01 (used for synthetic experiments
/// that have no real timestamps).
fn ensure_plot_timestamps(
    stored: &[chrono::NaiveDateTime],
    n: usize,
) -> Vec<chrono::DateTime<chrono::Utc>> {
    if stored.is_empty() {
        let base = chrono::NaiveDate::from_ymd_opt(2000, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc();
        (0..n)
            .map(|i| base + chrono::Duration::days(i as i64))
            .collect()
    } else {
        stored.iter().map(chrono::NaiveDateTime::and_utc).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::config::{
        DataConfig, EvaluationConfig, ExperimentConfig, ExperimentMode,
    };
    use crate::experiments::config::{
        DetectorConfig, DetectorType, FeatureConfig, FeatureFamilyConfig, ModelConfig,
        OutputConfig, ReproducibilityConfig, RunMetaConfig, ScalingPolicyConfig, TrainingMode,
    };

    #[derive(Default)]
    struct FailingBackend;

    impl ExperimentBackend for FailingBackend {
        fn resolve_data(&self, _cfg: &ExperimentConfig) -> anyhow::Result<DataBundle> {
            Ok(DataBundle {
                dataset_key: "x".to_string(),
                n_observations: 100,
                observations: vec![],
                changepoint_truth: None,
                train_n: 70,
                timestamps: vec![],
                split_summary_json: None,
                validation_report_json: None,
            })
        }
        fn build_features(
            &self,
            _cfg: &ExperimentConfig,
            _data: &DataBundle,
        ) -> anyhow::Result<FeatureBundle> {
            anyhow::bail!("feature stage crash")
        }
        fn train_or_load_model(
            &self,
            _cfg: &ExperimentConfig,
            _features: &FeatureBundle,
        ) -> anyhow::Result<ModelArtifact> {
            unreachable!()
        }
        fn run_online(
            &self,
            _cfg: &ExperimentConfig,
            _model: &ModelArtifact,
            _features: &FeatureBundle,
        ) -> anyhow::Result<OnlineRunArtifact> {
            unreachable!()
        }
        fn evaluate_synthetic(
            &self,
            _cfg: &ExperimentConfig,
            _online: &OnlineRunArtifact,
        ) -> anyhow::Result<SyntheticEvalArtifact> {
            unreachable!()
        }
        fn evaluate_real(
            &self,
            _cfg: &ExperimentConfig,
            _online: &OnlineRunArtifact,
        ) -> anyhow::Result<RealEvalArtifact> {
            unreachable!()
        }
    }

    fn base_cfg(mode: ExperimentMode) -> ExperimentConfig {
        ExperimentConfig {
            meta: RunMetaConfig {
                run_label: "runner_test".to_string(),
                notes: None,
            },
            mode: mode.clone(),
            data: match mode {
                ExperimentMode::Synthetic => DataConfig::Synthetic {
                    scenario_id: "scn".to_string(),
                    horizon: 200,
                    dataset_id: None,
                },
                ExperimentMode::Real => DataConfig::Real {
                    asset: "SPY".to_string(),
                    frequency: crate::experiments::config::RealFrequency::Daily,
                    dataset_id: "spy_daily".to_string(),
                    date_start: None,
                    date_end: None,
                },
            },
            features: FeatureConfig {
                family: FeatureFamilyConfig::LogReturn,
                scaling: ScalingPolicyConfig::None,
                session_aware: false,
            },
            model: ModelConfig {
                k_regimes: 2,
                training: TrainingMode::FitOffline,
                em_max_iter: 10,
                em_tol: 1e-5,
                em_n_starts: 1,
            },
            detector: DetectorConfig {
                detector_type: DetectorType::Surprise,
                threshold: 2.0,
                persistence_required: 1,
                cooldown: 0,
                ema_alpha: None,
            },
            evaluation: match mode {
                ExperimentMode::Synthetic => EvaluationConfig::Synthetic { matching_window: 5 },
                ExperimentMode::Real => EvaluationConfig::Real {
                    proxy_events_path: String::new(),
                    route_a_point_pre_bars: 2,
                    route_a_point_post_bars: 2,
                    route_a_causal_only: true,
                    route_b_min_segment_len: 5,
                },
            },
            output: OutputConfig {
                root_dir: std::env::temp_dir().to_string_lossy().to_string(),
                write_json: true,
                write_csv: false,
                save_traces: false,
            },
            reproducibility: ReproducibilityConfig {
                seed: Some(11),
                deterministic_run_id: true,
                save_config_snapshot: true,
                record_git_info: false,
            },
        }
    }

    #[test]
    fn routes_synthetic_mode() {
        let runner = ExperimentRunner::new(DryRunBackend);
        let r = runner.run(base_cfg(ExperimentMode::Synthetic));
        assert!(r.is_success());
        match r.evaluation_summary {
            Some(EvaluationSummary::Synthetic { .. }) => {}
            _ => panic!("expected synthetic evaluation summary"),
        }
    }

    #[test]
    fn routes_real_mode() {
        let runner = ExperimentRunner::new(DryRunBackend);
        let r = runner.run(base_cfg(ExperimentMode::Real));
        assert!(r.is_success());
        match r.evaluation_summary {
            Some(EvaluationSummary::Real { .. }) => {}
            _ => panic!("expected real evaluation summary"),
        }
    }

    #[test]
    fn deterministic_run_id_is_reproducible() {
        let runner = ExperimentRunner::new(DryRunBackend);
        let c = base_cfg(ExperimentMode::Synthetic);
        let r1 = runner.run(c.clone());
        let r2 = runner.run(c);
        assert_eq!(r1.metadata.run_id, r2.metadata.run_id);
        assert_eq!(r1.metadata.config_hash, r2.metadata.config_hash);
    }

    #[test]
    fn failure_stage_is_recorded() {
        let runner = ExperimentRunner::new(FailingBackend);
        let r = runner.run(base_cfg(ExperimentMode::Synthetic));
        match r.status {
            RunStatus::Failed {
                stage: RunStage::BuildFeatures,
                ..
            } => {}
            _ => panic!("expected failed status at BuildFeatures"),
        }
    }
}
