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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSummary {
    pub k_regimes: usize,
    pub training_mode: String,
    pub diagnostics_ok: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectorSummary {
    pub detector_type: String,
    pub threshold: f64,
    pub persistence_required: usize,
    pub cooldown: usize,
    pub n_alarms: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvaluationSummary {
    Synthetic {
        n_events: usize,
        coverage: f64,
        precision_like: f64,
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
}

impl ExperimentResult {
    pub fn is_success(&self) -> bool {
        matches!(self.status, RunStatus::Success | RunStatus::PartialSuccess)
    }
}
