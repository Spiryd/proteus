use std::path::PathBuf;

/// Resolved on-disk layout for a single experiment run.
///
/// The runner writes all artifacts directly into [`root`], using a flat
/// layout (no nested `config/`/`results/` subdirectories).  Aggregate
/// tooling (`compare-runs`) only needs `root` and `run_id` for lookups.
pub struct RunArtifactLayout {
    pub run_id: String,
    pub root: PathBuf,
}

impl RunArtifactLayout {
    /// Path used by [`AggregateReporter`] to load evaluation summaries.
    ///
    /// Tries the flat `<root>/summary.json` path first; callers that read
    /// from nested layouts can fall back as needed.
    pub fn evaluation_summary_path(&self) -> PathBuf {
        self.root.join("summary.json")
    }
}
