#![allow(unused_imports)]
/// Phase 17: Synthetic-to-real calibration layer.
///
/// This module provides a principled bridge between:
/// - empirical feature observations (`FeatureStream`), and
/// - calibrated synthetic Markov-switching scenarios.
///
/// Architecture:
/// 1) `summary`  — extract empirical targets T_1..T_m
/// 2) `mapping`  — map targets to synthetic generator params theta
/// 3) `verify`   — synthetic-vs-empirical discrepancy checks
/// 4) `report`   — end-to-end calibration workflow artifact
pub mod mapping;
pub mod report;
pub mod summary;
pub mod verify;

pub use mapping::{
    CalibratedSyntheticParams, CalibrationMappingConfig, JumpContamination, MeanPolicy,
    VariancePolicy, calibrate_to_synthetic,
};
pub use report::{CalibrationReport, CalibrationReportView, run_calibration_workflow};
pub use summary::{
    CalibrationDatasetTag, CalibrationPartition, EmpiricalCalibrationProfile, EmpiricalSummary,
    SummaryTargetSet, summarize_feature_stream, summarize_observation_values,
};
pub use verify::{
    CalibrationDiff, CalibrationVerification, VerificationTolerance, verify_calibration,
};
