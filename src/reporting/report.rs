use super::artifact::RunArtifactLayout;
use super::table::{MetricsTableBuilder, MetricsTableRow};
use crate::experiments::{EvaluationSummary, ExperimentResult};

/// Orchestrates full reporting for a single run
pub struct RunReporter {
    artifact_layout: RunArtifactLayout,
}

impl RunReporter {
    pub fn new(artifact_layout: RunArtifactLayout) -> Self {
        Self { artifact_layout }
    }

    /// Generate all tables for the run (metrics.md / metrics.csv / metrics.tex)
    pub fn generate_tables(&self, result: &ExperimentResult) -> anyhow::Result<()> {
        let run_dir = &self.artifact_layout.root;

        let mut builder = MetricsTableBuilder::new();
        let n_alarms = result.detector_summary.as_ref().map_or(0, |d| d.n_alarms);
        let (detector_type, threshold) = result.detector_summary.as_ref().map_or_else(
            || ("unknown".to_string(), 0.0),
            |d| (d.detector_type.clone(), d.threshold),
        );

        let (coverage, precision, delay_mean, delay_median) = match &result.evaluation_summary {
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
    pub fn generate_comparison_table(&self) -> anyhow::Result<String> {
        use crate::experiments::EvaluationSummary;

        let mut builder = MetricsTableBuilder::new();

        for layout in &self.runs {
            let Ok(json) = std::fs::read_to_string(layout.evaluation_summary_path()) else {
                continue;
            };
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
            let (detector_type, threshold, n_alarms) =
                std::fs::read_to_string(layout.root.join("result.json"))
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
    use std::path::PathBuf;

    fn make_layout(run_id: &str) -> RunArtifactLayout {
        RunArtifactLayout {
            run_id: run_id.to_string(),
            root: PathBuf::from("./runs/synthetic/scenario_calibrated").join(run_id),
        }
    }

    #[test]
    fn test_run_reporter_creation() {
        let _reporter = RunReporter::new(make_layout("run_test_001"));
    }

    #[test]
    fn test_aggregate_reporter_creation() {
        let reporter =
            AggregateReporter::new(vec![make_layout("run_001"), make_layout("run_002")]);
        assert_eq!(reporter.runs.len(), 2);
    }
}
