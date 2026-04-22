use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use super::artifact::{prepare_run_dir, snapshot_config, snapshot_result, write_json_file};
use super::config::{DataConfig, EvaluationConfig, ExperimentConfig, ExperimentMode, TrainingMode};
use super::result::{
    ArtifactRef, DetectorSummary, EvaluationSummary, ExperimentResult, ModelSummary, RunMetadata,
    RunStage, RunStatus, StageTiming,
};

#[derive(Debug, Clone)]
pub struct DataBundle {
    pub dataset_key: String,
    pub n_observations: usize,
}

#[derive(Debug, Clone)]
pub struct FeatureBundle {
    pub feature_label: String,
    pub n_observations: usize,
}

#[derive(Debug, Clone)]
pub struct ModelArtifact {
    pub source: String,
    pub k_regimes: usize,
    pub diagnostics_ok: bool,
}

#[derive(Debug, Clone)]
pub struct OnlineRunArtifact {
    pub n_steps: usize,
    pub n_alarms: usize,
}

#[derive(Debug, Clone)]
pub struct SyntheticEvalArtifact {
    pub n_events: usize,
    pub coverage: f64,
    pub precision_like: f64,
}

#[derive(Debug, Clone)]
pub struct RealEvalArtifact {
    pub event_coverage: f64,
    pub alarm_relevance: f64,
    pub segmentation_coherence: f64,
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
        })
    }

    fn build_features(
        &self,
        cfg: &ExperimentConfig,
        data: &DataBundle,
    ) -> anyhow::Result<FeatureBundle> {
        Ok(FeatureBundle {
            feature_label: format!("{:?}", cfg.features.family),
            n_observations: data.n_observations.saturating_sub(1),
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
        let mut model_summary = None;
        let mut detector_summary = None;
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
                model_summary = Some(ModelSummary {
                    k_regimes: v.k_regimes,
                    training_mode: v.source.clone(),
                    diagnostics_ok: v.diagnostics_ok,
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

        let eval_result = match timed_stage(RunStage::Evaluate, &mut timings, || match cfg.mode {
            ExperimentMode::Synthetic => self.backend.evaluate_synthetic(&cfg, &online).map(|s| {
                EvaluationSummary::Synthetic {
                    n_events: s.n_events,
                    coverage: s.coverage,
                    precision_like: s.precision_like,
                }
            }),
            ExperimentMode::Real => {
                self.backend
                    .evaluate_real(&cfg, &online)
                    .map(|r| EvaluationSummary::Real {
                        event_coverage: r.event_coverage,
                        alarm_relevance: r.alarm_relevance,
                        segmentation_coherence: r.segmentation_coherence,
                    })
            }
        }) {
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
        }
    }

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
        format!("run_{:016x}_{seed}", config_hash)
    } else {
        format!("run_{:016x}_{:x}", config_hash, now_epoch_ms())
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
            },
            detector: DetectorConfig {
                detector_type: DetectorType::Surprise,
                threshold: 2.0,
                persistence_required: 1,
                cooldown: 0,
            },
            evaluation: match mode {
                ExperimentMode::Synthetic => EvaluationConfig::Synthetic { matching_window: 5 },
                ExperimentMode::Real => EvaluationConfig::Real {
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
