use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStage {
    ResolveData,
    BuildFeatures,
    TrainOrLoadModel,
    RunOnline,
    Evaluate,
    Export,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    Success,
    PartialSuccess,
    Failed { stage: RunStage, message: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunMetadata {
    pub run_id: String,
    pub run_label: String,
    pub mode: String,
    pub started_at_epoch_ms: u128,
    pub finished_at_epoch_ms: u128,
    pub seed: Option<u64>,
    pub config_hash: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageTiming {
    pub stage: RunStage,
    pub duration_ms: u128,
}

/// Fitted model parameters saved into the result for thesis inspection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FittedParamsSummary {
    /// Initial regime distribution π (length K).
    pub pi: Vec<f64>,
    /// Transition matrix P as row vectors (K × K).
    pub transition: Vec<Vec<f64>>,
    /// Emission means μ_j (length K).
    pub means: Vec<f64>,
    /// Emission variances σ²_j (length K).
    pub variances: Vec<f64>,
    /// Final observed-data log-likelihood.
    pub log_likelihood: f64,
    /// Log-likelihood at each EM iteration (includes pre-loop baseline).
    pub ll_history: Vec<f64>,
    /// Number of EM iterations completed.
    pub n_iter: usize,
    /// Whether the EM tolerance criterion was met.
    pub converged: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSummary {
    pub k_regimes: usize,
    pub training_mode: String,
    pub diagnostics_ok: bool,
    /// Fitted model parameters; populated only when training actually ran.
    pub fitted_params: Option<FittedParamsSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectorSummary {
    pub detector_type: String,
    pub threshold: f64,
    pub persistence_required: usize,
    pub cooldown: usize,
    pub n_alarms: usize,
    /// 1-based step indices at which alarms fired.
    pub alarm_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvaluationSummary {
    Synthetic {
        n_events: usize,
        coverage: f64,
        precision_like: f64,
        // Full MetricSuite fields; None when using DryRunBackend.
        precision: Option<f64>,
        recall: Option<f64>,
        miss_rate: Option<f64>,
        false_alarm_rate: Option<f64>,
        delay_mean: Option<f64>,
        delay_median: Option<f64>,
        n_true_positive: Option<usize>,
        n_false_positive: Option<usize>,
        n_missed: Option<usize>,
    },
    Real {
        event_coverage: f64,
        alarm_relevance: f64,
        segmentation_coherence: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRef {
    pub name: String,
    pub path: String,
    pub kind: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentResult {
    pub metadata: RunMetadata,
    pub status: RunStatus,
    pub timings: Vec<StageTiming>,
    pub model_summary: Option<ModelSummary>,
    pub detector_summary: Option<DetectorSummary>,
    pub evaluation_summary: Option<EvaluationSummary>,
    pub artifacts: Vec<ArtifactRef>,
    pub warnings: Vec<String>,
    /// Detector score at each step; populated only when save_traces = true.
    pub score_trace: Vec<f64>,
    /// Filtered regime posteriors at each step (T × K); populated only when save_traces = true.
    pub regime_posteriors: Vec<Vec<f64>>,
    /// Number of feature observations produced by the feature pipeline
    /// (after warmup trimming).  Populated by all real and synthetic backends.
    pub n_feature_obs: Option<usize>,
}

impl ExperimentResult {
    pub fn is_success(&self) -> bool {
        matches!(self.status, RunStatus::Success | RunStatus::PartialSuccess)
    }
}
