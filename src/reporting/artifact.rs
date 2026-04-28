#![allow(dead_code)]
use std::fs;
use std::path::PathBuf;

/// Root directory policy for run artifacts
pub struct ArtifactRootConfig {
    pub root_dir: PathBuf,
    pub mode: String,                // "synthetic" | "real"
    pub dataset_or_scenario: String, // e.g., "scenario_calibrated" or "spy_daily"
}

/// Full path resolver for a single run
pub struct RunArtifactLayout {
    pub run_id: String,
    pub root: PathBuf,
}

impl RunArtifactLayout {
    pub fn new(root_config: &ArtifactRootConfig, run_id: String) -> Self {
        let mut root = root_config.root_dir.clone();
        root.push(&root_config.mode);
        root.push(&root_config.dataset_or_scenario);
        root.push(&run_id);
        Self { run_id, root }
    }

    pub fn config_dir(&self) -> PathBuf {
        self.root.join("config")
    }
    pub fn metadata_dir(&self) -> PathBuf {
        self.root.join("metadata")
    }
    pub fn results_dir(&self) -> PathBuf {
        self.root.join("results")
    }
    pub fn traces_dir(&self) -> PathBuf {
        self.root.join("traces")
    }
    pub fn plots_dir(&self) -> PathBuf {
        self.root.join("plots")
    }
    pub fn tables_dir(&self) -> PathBuf {
        self.root.join("tables")
    }

    pub fn config_snapshot_path(&self) -> PathBuf {
        self.config_dir().join("experiment_config.json")
    }
    pub fn metadata_path(&self) -> PathBuf {
        self.metadata_dir().join("run_metadata.json")
    }
    pub fn evaluation_summary_path(&self) -> PathBuf {
        self.results_dir().join("evaluation_summary.json")
    }
    pub fn alarms_csv_path(&self) -> PathBuf {
        self.traces_dir().join("alarms.csv")
    }
    pub fn feature_trace_csv_path(&self) -> PathBuf {
        self.traces_dir().join("feature_trace.csv")
    }
    pub fn score_trace_csv_path(&self) -> PathBuf {
        self.traces_dir().join("score_trace.csv")
    }
    pub fn regime_posterior_csv_path(&self) -> PathBuf {
        self.traces_dir().join("regime_posterior.csv")
    }
    pub fn matched_events_csv_path(&self) -> PathBuf {
        self.traces_dir().join("matched_events.csv")
    }
    pub fn segments_csv_path(&self) -> PathBuf {
        self.traces_dir().join("segments.csv")
    }

    pub fn signal_alarms_plot_path(&self) -> PathBuf {
        self.plots_dir().join("signal_with_alarms.png")
    }
    pub fn detector_scores_plot_path(&self) -> PathBuf {
        self.plots_dir().join("detector_scores.png")
    }
    pub fn regime_posteriors_plot_path(&self) -> PathBuf {
        self.plots_dir().join("regime_posteriors.png")
    }
    pub fn segmentation_plot_path(&self) -> PathBuf {
        self.plots_dir().join("segmentation.png")
    }
    pub fn delay_distribution_plot_path(&self) -> PathBuf {
        self.plots_dir().join("delay_distribution.png")
    }

    pub fn metrics_table_md_path(&self) -> PathBuf {
        self.tables_dir().join("metrics_table.md")
    }
    pub fn metrics_table_tex_path(&self) -> PathBuf {
        self.tables_dir().join("metrics_table.tex")
    }
    pub fn segment_summary_table_path(&self) -> PathBuf {
        self.tables_dir().join("segment_summary.csv")
    }

    pub fn ensure_directories(&self) -> anyhow::Result<()> {
        for dir in [
            self.config_dir(),
            self.metadata_dir(),
            self.results_dir(),
            self.traces_dir(),
            self.plots_dir(),
            self.tables_dir(),
        ] {
            fs::create_dir_all(&dir)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_artifact_paths() {
        let root_cfg = ArtifactRootConfig {
            root_dir: PathBuf::from("./runs"),
            mode: "synthetic".to_string(),
            dataset_or_scenario: "scenario_calibrated".to_string(),
        };

        let layout = RunArtifactLayout::new(&root_cfg, "run_abc123_42".to_string());

        assert_eq!(
            layout.root,
            PathBuf::from("./runs/synthetic/scenario_calibrated/run_abc123_42")
        );
        assert_eq!(
            layout.config_dir(),
            PathBuf::from("./runs/synthetic/scenario_calibrated/run_abc123_42/config")
        );
        assert!(
            layout
                .config_snapshot_path()
                .ends_with("experiment_config.json")
        );
        assert!(layout.alarms_csv_path().ends_with("alarms.csv"));
    }
}
