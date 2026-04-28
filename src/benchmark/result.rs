#![allow(dead_code)]
/// Per-stream and aggregate benchmark result objects, timing, and labels.
use std::time::Duration;

use super::matching::MatchResult;
use super::metrics::{MetricSuite, Summary};
use super::truth::StreamMeta;

// =========================================================================
// Benchmark label
// =========================================================================

/// Identifies a detector/scenario combination in benchmark tables.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct BenchmarkLabel {
    /// Detector variant name, e.g. `"hard_switch"`.
    pub detector_id: String,
    /// Scenario class name, e.g. `"mean_shift_k2"`.
    pub scenario_id: String,
}

// =========================================================================
// Timing summary
// =========================================================================

/// Per-step and total runtime summaries for one stream.
#[derive(Debug, Clone)]
pub struct TimingSummary {
    /// Total elapsed time for the entire streaming run.
    pub total: Duration,
    /// Number of observations processed.
    pub n_steps: usize,
    /// Mean time per observation = `total / n_steps`.
    pub mean_step: Duration,
}

impl TimingSummary {
    pub fn new(total: Duration, n_steps: usize) -> Self {
        let mean_step = if n_steps > 0 {
            total / n_steps as u32
        } else {
            Duration::ZERO
        };
        Self {
            total,
            n_steps,
            mean_step,
        }
    }
}

// =========================================================================
// Per-stream result
// =========================================================================

/// Complete evaluation result for one benchmark stream.
///
/// Holds:
/// - the raw [`MatchResult`] (event-level labels),
/// - the computed [`MetricSuite`] (aggregate statistics),
/// - optional [`TimingSummary`] (if the session was timed),
/// - optional [`StreamMeta`] (scenario/detector labels for grouping).
#[derive(Debug, Clone)]
pub struct RunResult {
    pub match_result: MatchResult,
    pub metrics: MetricSuite,
    pub timing: Option<TimingSummary>,
    pub meta: StreamMeta,
}

impl RunResult {
    pub fn new(match_result: MatchResult, timing: Option<TimingSummary>, meta: StreamMeta) -> Self {
        let metrics = MetricSuite::from_match(&match_result);
        Self {
            match_result,
            metrics,
            timing,
            meta,
        }
    }

    /// Convenience: label derived from `meta`.
    pub fn label(&self) -> BenchmarkLabel {
        BenchmarkLabel {
            detector_id: self.meta.detector_id.clone(),
            scenario_id: self.meta.scenario_id.clone(),
        }
    }
}

// =========================================================================
// Aggregate result
// =========================================================================

/// Aggregate benchmark results across N repeated runs for one label.
///
/// All per-run metric values are summarized into `Summary` statistics.
///
/// `NaN` values (e.g. precision when a run had zero alarms) are excluded
/// from summaries — `Summary::n` reflects how many non-NaN values
/// contributed.
#[derive(Debug, Clone)]
pub struct AggregateResult {
    /// Which detector/scenario combination this summarizes.
    pub label: BenchmarkLabel,
    /// Number of runs aggregated.
    pub n_runs: usize,
    /// Summary of per-run recall values (NaN-excluded).
    pub recall: Summary,
    /// Summary of per-run precision values (NaN-excluded).
    pub precision: Summary,
    /// Summary of per-run miss-rate values (NaN-excluded).
    pub miss_rate: Summary,
    /// Summary of per-run false-alarm-rate values.
    pub false_alarm_rate: Summary,
    /// Summary of per-run mean detection delay (NaN-excluded).
    pub mean_delay: Summary,
    /// Summary of per-run median detection delay (NaN-excluded).
    pub median_delay: Summary,
    /// Summary of first-false-alarm time (only runs that had a false alarm).
    pub first_false_alarm: Summary,
    /// Summary of mean-step timing in nanoseconds (only runs with timing).
    pub mean_step_ns: Summary,
}

impl AggregateResult {
    /// Build an aggregate result from a slice of per-stream [`RunResult`]s.
    ///
    /// All runs in `results` must share the same `BenchmarkLabel`; the first
    /// run's label is used.
    pub fn from_runs(results: &[RunResult]) -> Self {
        assert!(
            !results.is_empty(),
            "AggregateResult requires at least one run"
        );

        let label = results[0].label();
        let n_runs = results.len();

        let mut recalls = collect_finite(results.iter().map(|r| r.metrics.recall));
        let mut precisions = collect_finite(results.iter().map(|r| r.metrics.precision));
        let mut miss_rates = collect_finite(results.iter().map(|r| r.metrics.miss_rate));
        let mut fars = collect_finite(results.iter().map(|r| r.metrics.false_alarm_rate));
        let mut mean_delays = collect_finite(results.iter().map(|r| r.metrics.delay.mean));
        let mut median_delays = collect_finite(results.iter().map(|r| r.metrics.delay.median));
        let mut ffa_times = collect_finite(
            results
                .iter()
                .filter_map(|r| r.metrics.first_false_alarm_t.map(|t| t as f64)),
        );
        let mut step_ns = collect_finite(
            results
                .iter()
                .filter_map(|r| r.timing.as_ref().map(|t| t.mean_step.as_nanos() as f64)),
        );

        Self {
            label,
            n_runs,
            recall: Summary::from_slice(&mut recalls),
            precision: Summary::from_slice(&mut precisions),
            miss_rate: Summary::from_slice(&mut miss_rates),
            false_alarm_rate: Summary::from_slice(&mut fars),
            mean_delay: Summary::from_slice(&mut mean_delays),
            median_delay: Summary::from_slice(&mut median_delays),
            first_false_alarm: Summary::from_slice(&mut ffa_times),
            mean_step_ns: Summary::from_slice(&mut step_ns),
        }
    }
}

fn collect_finite(iter: impl Iterator<Item = f64>) -> Vec<f64> {
    iter.filter(|v| v.is_finite()).collect()
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::matching::{EventMatcher, MatchConfig};
    use crate::benchmark::truth::ChangePointTruth;
    use crate::detector::{AlarmEvent, DetectorKind};

    fn alarm(t: usize) -> AlarmEvent {
        AlarmEvent {
            t,
            score: 1.0,
            detector_kind: DetectorKind::HardSwitch,
            dominant_regime_before: Some(0),
            dominant_regime_after: 1,
        }
    }

    fn make_run(tau: Vec<usize>, alarm_ts: Vec<usize>, w: usize) -> RunResult {
        let truth = ChangePointTruth::new(tau, 200).unwrap();
        let alarms: Vec<AlarmEvent> = alarm_ts.iter().map(|&t| alarm(t)).collect();
        let mr = EventMatcher::new(MatchConfig { window: w }).match_events(&truth, &alarms);
        RunResult::new(mr, None, StreamMeta::default())
    }

    #[test]
    fn run_result_metrics_populated() {
        let r = make_run(vec![50], vec![52], 10);
        assert_eq!(r.metrics.n_detected, 1);
        assert!((r.metrics.recall - 1.0).abs() < 1e-12);
    }

    #[test]
    fn aggregate_from_two_perfect_runs() {
        let r1 = make_run(vec![50, 100], vec![50, 100], 10);
        let r2 = make_run(vec![50, 100], vec![51, 102], 10);
        let agg = AggregateResult::from_runs(&[r1, r2]);
        assert_eq!(agg.n_runs, 2);
        assert!((agg.recall.mean - 1.0).abs() < 1e-12);
        assert!((agg.precision.mean - 1.0).abs() < 1e-12);
    }

    #[test]
    fn aggregate_nan_excluded_from_summaries() {
        // Run 1: 1 changepoint detected → recall = 1.0
        // Run 2: no changepoints (no-change stream) → recall = NaN; excluded.
        let r1 = make_run(vec![50], vec![52], 10);
        let r2 = make_run(vec![], vec![], 10);
        let agg = AggregateResult::from_runs(&[r1, r2]);
        // recall summary only has 1 value (the NaN from r2 is excluded)
        assert_eq!(agg.recall.n, 1);
        assert!((agg.recall.mean - 1.0).abs() < 1e-12);
    }

    #[test]
    fn timing_summary_mean_step() {
        let ts = TimingSummary::new(Duration::from_millis(100), 200);
        assert_eq!(ts.mean_step, Duration::from_micros(500));
    }
}
