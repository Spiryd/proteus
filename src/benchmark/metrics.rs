/// Metric computation layer for the online benchmarking protocol.
///
/// All metrics in this module are derived from a [`MatchResult`] and, where
/// needed, the original stream length.  No detector or filter code is
/// called here.
///
/// # Metric taxonomy
///
/// | Symbol | Name | Definition |
/// |---|---|---|
/// | M | True changepoints | `truth.m()` |
/// | N | Total alarms | `n_alarms()` |
/// | TP | True-positive alarms | first match in window |
/// | FP | False-positive alarms | `N − TP` |
/// | M_det | Detected changepoints | `n_detected()` |
/// | M_miss | Missed changepoints | `M − M_det` |
///
/// Precision and recall use the event-level definitions (see below).
use super::matching::MatchResult;

// =========================================================================
// Summary statistics helper
// =========================================================================

/// Basic five-number summary over a `Vec<f64>`.
#[derive(Debug, Clone)]
pub struct Summary {
    pub n: usize,
    pub mean: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
}

impl Summary {
    /// Compute from a mutable slice (will be sorted in place for median).
    pub fn from_slice(values: &mut [f64]) -> Self {
        let n = values.len();
        if n == 0 {
            return Self {
                n: 0,
                mean: f64::NAN,
                median: f64::NAN,
                min: f64::NAN,
                max: f64::NAN,
            };
        }
        let mean = values.iter().sum::<f64>() / n as f64;
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if n % 2 == 1 {
            values[n / 2]
        } else {
            (values[n / 2 - 1] + values[n / 2]) / 2.0
        };
        let min = values[0];
        let max = values[n - 1];
        Self {
            n,
            mean,
            median,
            min,
            max,
        }
    }
}

// =========================================================================
// MetricSuite
// =========================================================================

/// Full metric suite derived from one [`MatchResult`].
///
/// All quantities are computed at construction time from the match result.
/// The metrics are kept together in one struct for easy export and aggregation.
#[derive(Debug, Clone)]
pub struct MetricSuite {
    // --- Event counts -------------------------------------------------------
    /// M — total true changepoints.
    pub n_changepoints: usize,
    /// N — total detector alarms.
    pub n_alarms: usize,
    /// TP — true-positive alarms.
    pub n_true_positive: usize,
    /// FP — false-positive alarms.
    pub n_false_positive: usize,
    /// M_det — matched (detected) changepoints.
    pub n_detected: usize,
    /// M_miss — missed changepoints.
    pub n_missed: usize,

    // --- Event-level rates --------------------------------------------------
    /// Precision = TP / N.  `NaN` if N = 0.
    ///
    /// Fraction of alarms that are true positives.
    pub precision: f64,
    /// Recall = M_det / M.  `NaN` if M = 0 (no changepoints to detect).
    ///
    /// Fraction of true changepoints that were detected within the window.
    pub recall: f64,
    /// Miss rate = M_miss / M = 1 − Recall.  `NaN` if M = 0.
    pub miss_rate: f64,

    // --- False alarm rate ---------------------------------------------------
    /// FAR = FP / T where T = stream_len.
    ///
    /// False positives per observation.  Multiply by 1000 for "per 1k obs."
    pub false_alarm_rate: f64,
    /// Time of the first false-positive alarm.  `None` if no false alarms.
    pub first_false_alarm_t: Option<usize>,

    // --- Detection delay ----------------------------------------------------
    /// Summary statistics (mean, median, min, max) over detection delays of
    /// all matched changepoints.  Fields are `NaN` if no changepoints were
    /// detected.
    pub delay: Summary,
}

impl MetricSuite {
    /// Compute the full metric suite from a [`MatchResult`].
    pub fn from_match(m: &MatchResult) -> Self {
        let n_changepoints = m.n_changepoints();
        let n_alarms = m.n_alarms();
        let n_true_positive = m.n_true_positive();
        let n_false_positive = m.n_false_positive();
        let n_detected = m.n_detected();
        let n_missed = m.n_missed();

        let precision = if n_alarms == 0 {
            f64::NAN
        } else {
            n_true_positive as f64 / n_alarms as f64
        };
        let recall = if n_changepoints == 0 {
            f64::NAN
        } else {
            n_detected as f64 / n_changepoints as f64
        };
        let miss_rate = if n_changepoints == 0 {
            f64::NAN
        } else {
            n_missed as f64 / n_changepoints as f64
        };
        let false_alarm_rate = if m.stream_len == 0 {
            f64::NAN
        } else {
            n_false_positive as f64 / m.stream_len as f64
        };
        let first_false_alarm_t = m.first_false_alarm_t();

        let mut delay_values: Vec<f64> = m.delays().into_iter().map(|d| d as f64).collect();
        let delay = Summary::from_slice(&mut delay_values);

        Self {
            n_changepoints,
            n_alarms,
            n_true_positive,
            n_false_positive,
            n_detected,
            n_missed,
            precision,
            recall,
            miss_rate,
            false_alarm_rate,
            first_false_alarm_t,
            delay,
        }
    }
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

    fn run_match(tau: Vec<usize>, alarm_ts: Vec<usize>, w: usize) -> MatchResult {
        let truth = ChangePointTruth::new(tau, 200).unwrap();
        let alarms: Vec<AlarmEvent> = alarm_ts.iter().map(|&t| alarm(t)).collect();
        EventMatcher::new(MatchConfig { window: w }).match_events(&truth, &alarms)
    }

    // --- precision / recall -------------------------------------------------

    #[test]
    fn perfect_detection_precision_recall_one() {
        let r = run_match(vec![50, 100], vec![50, 100], 10);
        let m = MetricSuite::from_match(&r);
        assert!((m.precision - 1.0).abs() < 1e-12);
        assert!((m.recall - 1.0).abs() < 1e-12);
        assert!((m.miss_rate).abs() < 1e-12);
    }

    #[test]
    fn no_alarms_recall_zero_and_miss_rate_one() {
        let r = run_match(vec![50], vec![], 10);
        let m = MetricSuite::from_match(&r);
        assert!(m.precision.is_nan()); // 0 alarms → NaN precision
        assert!((m.recall).abs() < 1e-12);
        assert!((m.miss_rate - 1.0).abs() < 1e-12);
    }

    #[test]
    fn only_false_alarms_precision_zero() {
        let r = run_match(vec![50], vec![10, 20], 10);
        let m = MetricSuite::from_match(&r);
        assert!((m.precision).abs() < 1e-12);
        assert!((m.recall).abs() < 1e-12);
    }

    // --- false alarm rate ---------------------------------------------------

    #[test]
    fn false_alarm_rate_correct() {
        // 2 FP alarms in stream of length 200 → rate = 0.01
        let r = run_match(vec![], vec![20, 60], 10);
        let m = MetricSuite::from_match(&r);
        assert!((m.false_alarm_rate - 2.0 / 200.0).abs() < 1e-12);
    }

    #[test]
    fn first_false_alarm_none_when_no_fp() {
        let r = run_match(vec![50], vec![52], 10);
        let m = MetricSuite::from_match(&r);
        assert!(m.first_false_alarm_t.is_none());
    }

    #[test]
    fn first_false_alarm_correct_time() {
        // alarm at 10 is FP (before τ=50), alarm at 52 is TP
        let r = run_match(vec![50], vec![10, 52], 10);
        let m = MetricSuite::from_match(&r);
        assert_eq!(m.first_false_alarm_t, Some(10));
    }

    // --- delay summary ------------------------------------------------------

    #[test]
    fn delay_summary_nan_when_nothing_detected() {
        let r = run_match(vec![50], vec![], 10);
        let m = MetricSuite::from_match(&r);
        assert!(m.delay.mean.is_nan());
    }

    #[test]
    fn delay_summary_correct_values() {
        // τ=50 matched at 52 (delay 2), τ=100 matched at 103 (delay 3)
        let r = run_match(vec![50, 100], vec![52, 103], 10);
        let m = MetricSuite::from_match(&r);
        assert!((m.delay.mean - 2.5).abs() < 1e-12);
        assert!((m.delay.min - 2.0).abs() < 1e-12);
        assert!((m.delay.max - 3.0).abs() < 1e-12);
    }

    // --- no-change stream ---------------------------------------------------

    #[test]
    fn no_change_stream_nan_recall_and_miss_rate() {
        // M = 0 → recall and miss_rate are NaN by convention
        let r = run_match(vec![], vec![20], 10);
        let m = MetricSuite::from_match(&r);
        assert!(m.recall.is_nan());
        assert!(m.miss_rate.is_nan());
        assert!((m.precision).abs() < 1e-12); // 0 TP / 1 alarm = 0
    }
}
