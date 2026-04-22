use std::path::PathBuf;
use crate::experiments::{ExperimentResult, ExperimentConfig};
use super::artifact::RunArtifactLayout;
use super::export::json;
use super::table::MetricsTableBuilder;

/// Orchestrates full reporting for a single run
pub struct RunReporter {
    artifact_layout: RunArtifactLayout,
}

impl RunReporter {
    pub fn new(artifact_layout: RunArtifactLayout) -> Self {
        Self { artifact_layout }
    }

    /// Export all run artifacts
    pub fn export_run(
        &self,
        config: &ExperimentConfig,
        result: &ExperimentResult,
    ) -> anyhow::Result<()> {
        // Ensure directories exist
        self.artifact_layout.ensure_directories()?;

        // Export config and metadata
        json::export_config(&self.artifact_layout.config_snapshot_path(), config)?;
        json::export_result(&self.artifact_layout.metadata_path(), result)?;

        // Export evaluation summary
        if let Some(eval_summary) = &result.evaluation_summary {
            json::export_evaluation_summary(
                &self.artifact_layout.evaluation_summary_path(),
                eval_summary,
            )?;
        }

        Ok(())
    }

    /// Generate all plots for the run
    pub fn generate_plots(&self) -> anyhow::Result<()> {
        // Load traces from CSV files
        // For each plot type, construct input and render
        // This is data-driven from the exported traces
        Ok(())
    }

    /// Generate all tables for the run
    pub fn generate_tables(&self, _result: &ExperimentResult) -> anyhow::Result<()> {
        // Build metrics table from result
        // Write to markdown, CSV, and LaTeX
        Ok(())
    }
}

/// Aggregates results across multiple runs
pub struct AggregateReporter {
    pub runs: Vec<RunArtifactLayout>,
}

impl AggregateReporter {
    pub fn new(runs: Vec<RunArtifactLayout>) -> Self {
        Self { runs }
    }

    /// Generate aggregate comparison table
    pub fn generate_comparison_table(&self) -> anyhow::Result<String> {
        let mut builder = MetricsTableBuilder::new();

        for _layout in &self.runs {
            // Load each run's result and build a row
            // Add to builder
        }

        Ok(builder.to_csv())
    }

    /// Generate aggregate plots comparing detectors
    pub fn generate_aggregate_plots(&self) -> anyhow::Result<()> {
        // Collect metrics from all runs
        // Generate comparison plots (e.g., boxplot of delays)
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reporting::artifact::ArtifactRootConfig;
    use std::path::PathBuf;

    #[test]
    fn test_run_reporter_creation() {
        let root_cfg = ArtifactRootConfig {
            root_dir: PathBuf::from("./runs"),
            mode: "synthetic".to_string(),
            dataset_or_scenario: "scenario_calibrated".to_string(),
        };

        let layout = RunArtifactLayout::new(&root_cfg, "run_test_001".to_string());
        let _reporter = RunReporter::new(layout);
    }

    #[test]
    fn test_aggregate_reporter_creation() {
        let root_cfg = ArtifactRootConfig {
            root_dir: PathBuf::from("./runs"),
            mode: "synthetic".to_string(),
            dataset_or_scenario: "scenario_calibrated".to_string(),
        };

        let layout1 = RunArtifactLayout::new(&root_cfg, "run_001".to_string());
        let layout2 = RunArtifactLayout::new(&root_cfg, "run_002".to_string());

        let reporter = AggregateReporter::new(vec![layout1, layout2]);
        assert_eq!(reporter.runs.len(), 2);
    }
}
