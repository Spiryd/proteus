/// Causal event matching between true changepoints and detector alarms.
///
/// # Matching protocol
///
/// Given:
/// - true changepoint set `C* = {τ₁, …, τₘ}` with `τ₁ < … < τₘ`,
/// - detector alarm set   `A  = {a₁, …, aₙ}` with `a₁ < … < aₙ`,
/// - detection window width `w ≥ 0`.
///
/// The matcher performs **greedy chronological matching**:
///
/// 1. Process changepoints in ascending order (τ₁ first).
/// 2. For changepoint τₘ, search for the earliest unused alarm `aₙ` such that
///    `τₘ ≤ aₙ < τₘ + w`.
/// 3. If found: the pair (τₘ, aₙ) is a **match**.  Mark aₙ as used.
/// 4. If not found: τₘ is a **miss**.
/// 5. Any alarm not matched to any changepoint is a **false positive**.
///
/// # Causal constraint
///
/// The window is `[τₘ, τₘ + w)` — only alarms at or **after** the changepoint
/// are eligible.  An alarm before τₘ cannot be credited as detecting τₘ.
/// This is the fundamental causality requirement of online evaluation.
///
/// # Policies
///
/// - **One alarm per changepoint**: only the first matching alarm counts; later
///   alarms in the same window are left as potential false positives for other
///   changepoints.
/// - **One changepoint per alarm**: each alarm is matched at most once.
/// - **First-match chronological**: changepoints are processed in τ₁, τ₂, … order.
use super::truth::ChangePointTruth;
use crate::detector::AlarmEvent;

// =========================================================================
// Configuration
// =========================================================================

/// Parameters for the event-matching protocol.
#[derive(Debug, Clone)]
pub struct MatchConfig {
    /// Detection window half-width `w`.
    ///
    /// Alarm `aₙ` matches changepoint `τₘ` iff `τₘ ≤ aₙ < τₘ + w`.
    /// `w = 0` would allow only same-step matches; typical values are 5–50
    /// depending on the expected reaction time of the detector.
    pub window: usize,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self { window: 10 }
    }
}

// =========================================================================
// Match outcome for a single alarm
// =========================================================================

/// Outcome assigned to one alarm after matching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlarmOutcome {
    /// This alarm is the first match for a true changepoint at `tau`.
    TruePositive { tau: usize },
    /// This alarm does not match any true changepoint.
    FalsePositive,
}

// =========================================================================
// Match outcome for a single changepoint
// =========================================================================

/// Outcome assigned to one true changepoint after matching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangePointOutcome {
    /// Matched to alarm at time `alarm_t`.  Detection delay = `alarm_t - tau`.
    Detected { alarm_t: usize },
    /// No alarm was found in the window `[tau, tau + w)`.
    Missed,
}

// =========================================================================
// Match result
// =========================================================================

/// Full event-matching result for one stream.
///
/// Produced by [`EventMatcher::match_events`].
#[derive(Debug, Clone)]
pub struct MatchResult {
    /// Per-changepoint outcomes, ordered as `τ₁, τ₂, …, τₘ`.
    pub changepoint_outcomes: Vec<(usize, ChangePointOutcome)>,
    /// Per-alarm outcomes, in the order the alarms were supplied.
    pub alarm_outcomes: Vec<(usize, AlarmOutcome)>,
    /// Total stream length T.
    pub stream_len: usize,
    /// Detection window width used.
    pub window: usize,
}

impl MatchResult {
    /// Number of matched (detected) changepoints.
    pub fn n_detected(&self) -> usize {
        self.changepoint_outcomes
            .iter()
            .filter(|(_, o)| matches!(o, ChangePointOutcome::Detected { .. }))
            .count()
    }

    /// Number of missed changepoints.
    pub fn n_missed(&self) -> usize {
        self.changepoint_outcomes
            .iter()
            .filter(|(_, o)| matches!(o, ChangePointOutcome::Missed))
            .count()
    }

    /// Number of true-positive alarms.
    pub fn n_true_positive(&self) -> usize {
        self.alarm_outcomes
            .iter()
            .filter(|(_, o)| matches!(o, AlarmOutcome::TruePositive { .. }))
            .count()
    }

    /// Number of false-positive alarms.
    pub fn n_false_positive(&self) -> usize {
        self.alarm_outcomes
            .iter()
            .filter(|(_, o)| matches!(o, AlarmOutcome::FalsePositive))
            .count()
    }

    /// Total number of alarms.
    pub fn n_alarms(&self) -> usize {
        self.alarm_outcomes.len()
    }

    /// Total number of true changepoints.
    pub fn n_changepoints(&self) -> usize {
        self.changepoint_outcomes.len()
    }

    /// Detection delays for all matched changepoints.
    pub fn delays(&self) -> Vec<usize> {
        self.changepoint_outcomes
            .iter()
            .filter_map(|(tau, o)| {
                if let ChangePointOutcome::Detected { alarm_t } = o {
                    Some(alarm_t - tau)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Time index of the first false-positive alarm, if any.
    pub fn first_false_alarm_t(&self) -> Option<usize> {
        self.alarm_outcomes
            .iter()
            .find(|(_, o)| matches!(o, AlarmOutcome::FalsePositive))
            .map(|(t, _)| *t)
    }
}

// =========================================================================
// Event matcher
// =========================================================================

/// Performs causal event matching between a [`ChangePointTruth`] and a
/// sequence of [`AlarmEvent`]s.
pub struct EventMatcher {
    pub config: MatchConfig,
}

impl EventMatcher {
    pub fn new(config: MatchConfig) -> Self {
        Self { config }
    }

    /// Match alarms against the ground-truth changepoints.
    ///
    /// `alarms` must already be sorted ascending by `t`; they are produced
    /// that way by any [`crate::detector::Detector`] implementation.
    pub fn match_events(&self, truth: &ChangePointTruth, alarms: &[AlarmEvent]) -> MatchResult {
        let w = self.config.window;

        // We mark which alarms have been consumed so each can match at most once.
        let mut alarm_used = vec![false; alarms.len()];

        // --- process changepoints in chronological order ---
        let mut changepoint_outcomes = Vec::with_capacity(truth.times.len());
        for &tau in &truth.times {
            let window_end = tau + w; // exclusive upper bound
            // Find the earliest unused alarm in [tau, window_end).
            let found = alarms
                .iter()
                .enumerate()
                .find(|(i, a)| !alarm_used[*i] && a.t >= tau && a.t < window_end);
            if let Some((idx, a)) = found {
                alarm_used[idx] = true;
                changepoint_outcomes.push((tau, ChangePointOutcome::Detected { alarm_t: a.t }));
            } else {
                changepoint_outcomes.push((tau, ChangePointOutcome::Missed));
            }
        }

        // --- label all alarms ---
        let alarm_outcomes: Vec<(usize, AlarmOutcome)> = alarms
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if alarm_used[i] {
                    // Find which tau it matched.
                    let tau = changepoint_outcomes
                        .iter()
                        .find_map(|(tau, o)| {
                            if let ChangePointOutcome::Detected { alarm_t } = o {
                                if *alarm_t == a.t { Some(*tau) } else { None }
                            } else {
                                None
                            }
                        })
                        .unwrap_or(0); // guaranteed to find since alarm_used[i]
                    (a.t, AlarmOutcome::TruePositive { tau })
                } else {
                    (a.t, AlarmOutcome::FalsePositive)
                }
            })
            .collect();

        MatchResult {
            changepoint_outcomes,
            alarm_outcomes,
            stream_len: truth.stream_len,
            window: w,
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
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

    fn truth(times: Vec<usize>, len: usize) -> ChangePointTruth {
        ChangePointTruth::new(times, len).unwrap()
    }

    fn matcher(w: usize) -> EventMatcher {
        EventMatcher::new(MatchConfig { window: w })
    }

    #[test]
    fn exact_match_single_changepoint() {
        let m = matcher(10);
        let r = m.match_events(&truth(vec![50], 100), &[alarm(50)]);
        assert_eq!(r.n_detected(), 1);
        assert_eq!(r.n_missed(), 0);
        assert_eq!(r.n_true_positive(), 1);
        assert_eq!(r.n_false_positive(), 0);
        assert_eq!(r.delays(), vec![0]);
    }

    #[test]
    fn match_within_window() {
        let m = matcher(10);
        let r = m.match_events(&truth(vec![50], 100), &[alarm(55)]);
        assert_eq!(r.n_detected(), 1);
        assert_eq!(r.delays(), vec![5]);
    }

    #[test]
    fn alarm_outside_window_is_false_positive() {
        let m = matcher(10);
        // alarm at 65 is outside [50, 60) → FP; changepoint is missed
        let r = m.match_events(&truth(vec![50], 100), &[alarm(65)]);
        assert_eq!(r.n_detected(), 0);
        assert_eq!(r.n_missed(), 1);
        assert_eq!(r.n_false_positive(), 1);
    }

    #[test]
    fn alarm_before_changepoint_is_false_positive() {
        // Causal constraint: alarm at 45 cannot match τ=50.
        let m = matcher(10);
        let r = m.match_events(&truth(vec![50], 100), &[alarm(45)]);
        assert_eq!(r.n_detected(), 0);
        assert_eq!(r.n_false_positive(), 1);
    }

    #[test]
    fn multiple_alarms_one_changepoint_only_first_counts() {
        // window = 10, τ = 50; alarms at 50, 52, 58 — only 50 is TP.
        let m = matcher(10);
        let r = m.match_events(&truth(vec![50], 100), &[alarm(50), alarm(52), alarm(58)]);
        assert_eq!(r.n_true_positive(), 1);
        assert_eq!(r.n_false_positive(), 2);
    }

    #[test]
    fn multiple_changepoints_matched_in_order() {
        let m = matcher(10);
        let r = m.match_events(&truth(vec![30, 70], 100), &[alarm(32), alarm(75)]);
        assert_eq!(r.n_detected(), 2);
        assert_eq!(r.n_false_positive(), 0);
        let delays = r.delays();
        assert_eq!(delays, vec![2, 5]);
    }

    #[test]
    fn no_change_stream_all_alarms_false_positive() {
        let m = matcher(10);
        let r = m.match_events(&truth(vec![], 100), &[alarm(20), alarm(60)]);
        assert_eq!(r.n_detected(), 0);
        assert_eq!(r.n_false_positive(), 2);
    }

    #[test]
    fn no_alarms_all_changepoints_missed() {
        let m = matcher(10);
        let r = m.match_events(&truth(vec![30, 70], 100), &[]);
        assert_eq!(r.n_missed(), 2);
        assert_eq!(r.n_alarms(), 0);
    }

    #[test]
    fn first_false_alarm_time_correct() {
        let m = matcher(10);
        // τ=50, alarms at 20 (FP), 52 (TP)
        let r = m.match_events(&truth(vec![50], 100), &[alarm(20), alarm(52)]);
        assert_eq!(r.first_false_alarm_t(), Some(20));
    }

    #[test]
    fn window_boundary_inclusive_start_exclusive_end() {
        let m = matcher(10);
        // window [50, 60): alarm at 59 matches, alarm at 60 does not.
        let r_in = m.match_events(&truth(vec![50], 100), &[alarm(59)]);
        assert_eq!(r_in.n_detected(), 1);
        let r_out = m.match_events(&truth(vec![50], 100), &[alarm(60)]);
        assert_eq!(r_out.n_detected(), 0);
    }
}
