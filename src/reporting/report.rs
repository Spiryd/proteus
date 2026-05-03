#![allow(dead_code)]
use super::artifact::RunArtifactLayout;
use super::export::json;
use super::table::{MetricsTableBuilder, MetricsTableRow};
use crate::experiments::{EvaluationSummary, ExperimentConfig, ExperimentResult};

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

    /// Generate all tables for the run (metrics.md / metrics.csv / metrics.tex)
    pub fn generate_tables(&self, result: &ExperimentResult) -> anyhow::Result<()> {
        let run_dir = &self.artifact_layout.root;

        let mut builder = MetricsTableBuilder::new();
        let n_alarms = result.detector_summary.as_ref().map_or(0, |d| d.n_alarms);
        let (detector_type, threshold) = result
            .detector_summary
            .as_ref()
            .map_or_else(|| ("unknown".to_string(), 0.0), |d| (d.detector_type.clone(), d.threshold));

        let (coverage, precision, delay_mean, delay_median) =
            match &result.evaluation_summary {
                Some(EvaluationSummary::Synthetic {
                    coverage,
                    precision_like,
                    precision,
                    recall,
                    delay_mean,
                    delay_median,
                    ..
                }) => (
                    Some(recall.unwrap_or(*coverage)),
                    Some(precision.unwrap_or(*precision_like)),
                    *delay_mean,
                    *delay_median,
                ),
                Some(EvaluationSummary::Real {
                    event_coverage,
                    alarm_relevance,
                    ..
                }) => (Some(*event_coverage), Some(*alarm_relevance), None, None),
                None => (None, None, None, None),
            };

        builder.add_row(MetricsTableRow {
            run_id: result.metadata.run_id.clone(),
            scenario_or_asset: result.metadata.run_label.clone(),
            detector_type,
            threshold,
            n_alarms,
            coverage,
            precision,
            delay_mean,
            delay_median,
        });

        let md = builder.to_markdown();
        let csv = builder.to_csv();
        let tex = builder.to_latex();

        std::fs::write(run_dir.join("metrics.md"), &md)?;
        std::fs::write(run_dir.join("metrics.csv"), &csv)?;
        std::fs::write(run_dir.join("metrics.tex"), &tex)?;

        Ok(())
    }
}

/// Aggregates results across multiple runs.
///
/// Loads each run's `evaluation_summary.json` and builds a combined metrics
/// table.
pub struct AggregateReporter {
    pub runs: Vec<RunArtifactLayout>,
}

impl AggregateReporter {
    pub fn new(runs: Vec<RunArtifactLayout>) -> Self {
        Self { runs }
    }

    /// Build a combined metrics table (Markdown) from all run evaluation
    /// summaries.  Runs whose summary file cannot be read are silently
    /// skipped.
    ///
    /// Tries `root/results/evaluation_summary.json` first (legacy
    /// `RunArtifactLayout` path), then falls back to `root/summary.json`
    /// (the flat layout written by the experiment runner).
    pub fn generate_comparison_table(&self) -> anyhow::Result<String> {
        use crate::experiments::EvaluationSummary;

        let mut builder = MetricsTableBuilder::new();

        for layout in &self.runs {
            // Try the nested layout path first, then fall back to flat root layout.
            let candidate_paths = [
                layout.evaluation_summary_path(),
                layout.root.join("summary.json"),
            ];
            let json = candidate_paths
                .iter()
                .find_map(|p| std::fs::read_to_string(p).ok());
            let Some(json) = json else { continue };
            let Ok(eval) = serde_json::from_str::<EvaluationSummary>(&json) else {
                continue;
            };
            let (coverage, precision) = match &eval {
                EvaluationSummary::Synthetic {
                    coverage,
                    precision_like,
                    precision,
                    recall,
                    ..
                } => (
                    Some(recall.unwrap_or(*coverage)),
                    Some(precision.unwrap_or(*precision_like)),
                ),
                EvaluationSummary::Real {
                    event_coverage,
                    alarm_relevance,
                    ..
                } => (Some(*event_coverage), Some(*alarm_relevance)),
            };

            // Try to read result.json for detector metadata.
            let (detector_type, threshold, n_alarms) = std::fs::read_to_string(
                layout.root.join("result.json"),
            )
            .ok()
            .and_then(|s| serde_json::from_str::<ExperimentResult>(&s).ok())
            .and_then(|r| r.detector_summary)
            .map(|ds| (ds.detector_type, ds.threshold, ds.n_alarms))
            .unwrap_or_else(|| (String::new(), 0.0, 0));

            builder.add_row(MetricsTableRow {
                run_id: layout.run_id.clone(),
                scenario_or_asset: layout.run_id.clone(),
                detector_type,
                threshold,
                n_alarms,
                coverage,
                precision,
                delay_mean: None,
                delay_median: None,
            });
        }

        Ok(builder.to_markdown())
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
