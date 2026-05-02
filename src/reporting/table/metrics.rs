use std::fmt::Write as _;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsTableRow {
    pub run_id: String,
    pub scenario_or_asset: String,
    pub detector_type: String,
    pub threshold: f64,
    pub n_alarms: usize,
    pub coverage: Option<f64>,
    pub precision: Option<f64>,
    pub delay_mean: Option<f64>,
    pub delay_median: Option<f64>,
}

pub struct MetricsTableBuilder {
    rows: Vec<MetricsTableRow>,
}

impl MetricsTableBuilder {
    pub fn new() -> Self {
        Self { rows: Vec::new() }
    }

    pub fn add_row(&mut self, row: MetricsTableRow) {
        self.rows.push(row);
    }

    pub fn to_markdown(&self) -> String {
        if self.rows.is_empty() {
            return String::new();
        }

        let mut out = String::new();
        out.push_str("| Run ID | Scenario/Asset | Detector | Threshold | Alarms | Coverage | Precision | Delay Mean | Delay Median |\n");
        out.push_str("|--------|---|---|---|---|---|---|---|---|\n");

        for row in &self.rows {
            let coverage_str = row
                .coverage
                .map_or_else(String::new, |v| format!("{v:.3}"));
            let precision_str = row
                .precision
                .map_or_else(String::new, |v| format!("{v:.3}"));
            let delay_mean_str = row
                .delay_mean
                .map_or_else(String::new, |v| format!("{v:.2}"));
            let delay_median_str = row
                .delay_median
                .map_or_else(String::new, |v| format!("{v:.2}"));

            let _ = writeln!(
                out,
                "| {} | {} | {} | {:.2} | {} | {} | {} | {} | {} |",
                row.run_id,
                row.scenario_or_asset,
                row.detector_type,
                row.threshold,
                row.n_alarms,
                coverage_str,
                precision_str,
                delay_mean_str,
                delay_median_str
            );
        }

        out
    }

    pub fn to_csv(&self) -> String {
        if self.rows.is_empty() {
            return String::new();
        }

        let mut out = String::new();
        out.push_str("run_id,scenario_or_asset,detector_type,threshold,n_alarms,coverage,precision,delay_mean,delay_median\n");

        for row in &self.rows {
            let coverage_str = row.coverage.map(|v| v.to_string()).unwrap_or_default();
            let precision_str = row.precision.map(|v| v.to_string()).unwrap_or_default();
            let delay_mean_str = row.delay_mean.map(|v| v.to_string()).unwrap_or_default();
            let delay_median_str = row.delay_median.map(|v| v.to_string()).unwrap_or_default();

            let _ = writeln!(
                out,
                "{},{},{},{},{},{},{},{},{}",
                row.run_id,
                row.scenario_or_asset,
                row.detector_type,
                row.threshold,
                row.n_alarms,
                coverage_str,
                precision_str,
                delay_mean_str,
                delay_median_str
            );
        }

        out
    }

    pub fn to_latex(&self) -> String {
        if self.rows.is_empty() {
            return String::new();
        }

        let mut out = String::new();
        out.push_str("\\begin{tabular}{|c|c|c|r|r|c|c|c|c|}\n");
        out.push_str("\\hline\n");
        out.push_str("Run ID & Scenario & Detector & Threshold & Alarms & Coverage & Precision & Delay Mean & Delay Median \\\\\n");
        out.push_str("\\hline\n");

        for row in &self.rows {
            let coverage_str = row
                .coverage
                .map_or_else(|| "--".to_string(), |v| format!("{v:.3}"));
            let precision_str = row
                .precision
                .map_or_else(|| "--".to_string(), |v| format!("{v:.3}"));
            let delay_mean_str = row
                .delay_mean
                .map_or_else(|| "--".to_string(), |v| format!("{v:.2}"));
            let delay_median_str = row
                .delay_median
                .map_or_else(|| "--".to_string(), |v| format!("{v:.2}"));

            let _ = writeln!(
                out,
                "{} & {} & {} & {:.2} & {} & {} & {} & {} & {} \\\\",
                row.run_id,
                row.scenario_or_asset,
                row.detector_type,
                row.threshold,
                row.n_alarms,
                coverage_str,
                precision_str,
                delay_mean_str,
                delay_median_str
            );
        }

        out.push_str("\\hline\n");
        out.push_str("\\end{tabular}\n");

        out
    }
}

impl Default for MetricsTableBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_table_markdown() {
        let mut builder = MetricsTableBuilder::new();
        builder.add_row(MetricsTableRow {
            run_id: "run_001".to_string(),
            scenario_or_asset: "synthetic_calibrated".to_string(),
            detector_type: "mse".to_string(),
            threshold: 0.5,
            n_alarms: 10,
            coverage: Some(0.95),
            precision: Some(0.88),
            delay_mean: Some(2.5),
            delay_median: Some(2.0),
        });

        let markdown = builder.to_markdown();
        assert!(markdown.contains("run_001"));
        assert!(markdown.contains("0.950"));
        assert!(markdown.contains("0.88"));
        assert!(markdown.contains("|"));
    }

    #[test]
    fn test_metrics_table_csv() {
        let mut builder = MetricsTableBuilder::new();
        builder.add_row(MetricsTableRow {
            run_id: "run_002".to_string(),
            scenario_or_asset: "spy_daily".to_string(),
            detector_type: "cusum".to_string(),
            threshold: 1.0,
            n_alarms: 5,
            coverage: None,
            precision: Some(1.0),
            delay_mean: None,
            delay_median: Some(1.0),
        });

        let csv = builder.to_csv();
        assert!(csv.contains("run_id,"));
        assert!(csv.contains("run_002"));
        assert!(csv.contains("spy_daily"));
    }

    #[test]
    fn test_metrics_table_latex() {
        let mut builder = MetricsTableBuilder::new();
        builder.add_row(MetricsTableRow {
            run_id: "run_003".to_string(),
            scenario_or_asset: "test".to_string(),
            detector_type: "detector".to_string(),
            threshold: 0.5,
            n_alarms: 3,
            coverage: Some(0.8),
            precision: Some(0.9),
            delay_mean: Some(1.0),
            delay_median: Some(1.0),
        });

        let latex = builder.to_latex();
        assert!(latex.contains("\\begin{tabular}"));
        assert!(latex.contains("\\hline"));
        assert!(latex.contains("\\end{tabular}"));
    }

    #[test]
    fn test_empty_table() {
        let builder = MetricsTableBuilder::new();
        assert_eq!(builder.to_markdown(), "");
        assert_eq!(builder.to_csv(), "");
        assert_eq!(builder.to_latex(), "");
    }
}
