use std::path::Path;
use serde::Serialize;
use crate::experiments::{ExperimentConfig, ExperimentResult};

pub fn write_json_file<T: Serialize>(path: &Path, value: &T) -> anyhow::Result<()> {
    let data = serde_json::to_vec_pretty(value)?;
    std::fs::write(path, data)?;
    Ok(())
}

pub fn export_config(path: &Path, config: &ExperimentConfig) -> anyhow::Result<()> {
    write_json_file(path, config)
}

pub fn export_result(path: &Path, result: &ExperimentResult) -> anyhow::Result<()> {
    write_json_file(path, result)
}

pub fn export_evaluation_summary(
    path: &Path,
    summary: &crate::experiments::EvaluationSummary,
) -> anyhow::Result<()> {
    write_json_file(path, summary)
}
