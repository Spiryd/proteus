#![allow(unused_imports)]
pub mod artifact;
pub mod batch;
pub mod config;
pub mod real_backend;
pub mod registry;
pub mod result;
pub mod runner;
pub mod search;
pub mod shared;
pub mod sim_to_real_backend;
pub mod synthetic_backend;

pub use batch::{BatchConfig, BatchResult, run_batch};
pub use config::{
    DataConfig, DetectorConfig, DetectorType, EvaluationConfig, ExperimentConfig, ExperimentMode,
    FeatureConfig, FeatureFamilyConfig, ModelConfig, OutputConfig, RealFrequency,
    ReproducibilityConfig, RunMetaConfig, ScalingPolicyConfig, TrainingMode,
};
pub use real_backend::RealBackend;
pub use result::{
    ArtifactRef, DetectorSummary, EvaluationSummary, ExperimentResult, FittedParamsSummary,
    ModelSummary, RunMetadata, RunStage, RunStatus, StageTiming,
};
pub use runner::{
    DataBundle, DryRunBackend, ExperimentBackend, ExperimentRunner, FeatureBundle, ModelArtifact,
    OnlineRunArtifact, RealEvalArtifact, SyntheticEvalArtifact,
};
pub use synthetic_backend::SyntheticBackend;
