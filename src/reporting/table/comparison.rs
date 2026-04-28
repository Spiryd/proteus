#![allow(dead_code)]
/// Aggregate comparison table builder for comparing detectors across runs
pub struct ComparisonTableBuilder;

impl ComparisonTableBuilder {
    pub fn new() -> Self {
        Self
    }

    pub fn to_csv(&self) -> String {
        "detector,n_runs,mean_coverage,mean_precision,mean_delay\n".to_string()
    }
}

impl Default for ComparisonTableBuilder {
    fn default() -> Self {
        Self::new()
    }
}
