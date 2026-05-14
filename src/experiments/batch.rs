use super::config::ExperimentConfig;
use super::result::ExperimentResult;
use super::runner::{ExperimentBackend, ExperimentRunner};

#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub runs: Vec<ExperimentConfig>,
    pub stop_on_error: bool,
}

#[derive(Debug, Clone)]
pub struct BatchResult {
    pub run_results: Vec<ExperimentResult>,
    pub n_success: usize,
    pub n_failed: usize,
}

pub fn run_batch<B: ExperimentBackend>(
    runner: &ExperimentRunner<B>,
    cfg: BatchConfig,
) -> BatchResult {
    let mut out = Vec::with_capacity(cfg.runs.len());
    let mut n_success = 0usize;
    let mut n_failed = 0usize;

    for run in cfg.runs {
        let r = runner.run(run);
        if r.is_success() {
            n_success += 1;
        } else {
            n_failed += 1;
            if cfg.stop_on_error {
                out.push(r);
                break;
            }
        }
        out.push(r);
    }

    BatchResult {
        run_results: out,
        n_success,
        n_failed,
    }
}
