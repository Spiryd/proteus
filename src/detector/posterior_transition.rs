#![allow(dead_code)]
use super::{
    AlarmEvent, Detector, DetectorInput, DetectorKind, DetectorOutput, PersistencePolicy,
    dominant_regime,
};

// =========================================================================
// Score variant
// =========================================================================

/// Which posterior-transition score function to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PosteriorTransitionScoreKind {
    /// Leave-Previous-Regime score.
    ///
    /// Let `r_{t-1} = argmax_j α_{t-1|t-1}(j)`.
    ///
    /// ```text
    /// s_t^leave = 1 − α_{t|t}(r_{t-1})
    /// ```
    ///
    /// Lies in [0, 1].  Large when the current posterior has little mass on
    /// the regime that was dominant at the previous step.
    LeavePrevious,

    /// Total-Variation (posterior-shift magnitude) score.
    ///
    /// ```text
    /// s_t^TV = (1/2) Σ_j | α_{t|t}(j) − α_{t-1|t-1}(j) |
    /// ```
    ///
    /// Lies in [0, 1].  Large when the full posterior distribution has moved
    /// substantially between consecutive steps.  Equal to zero iff the two
    /// consecutive posteriors are identical.
    TotalVariation,
}

// =========================================================================
// Configuration
// =========================================================================

/// Configuration for the [`PosteriorTransitionDetector`].
#[derive(Debug, Clone)]
pub struct PosteriorTransitionConfig {
    /// Which score function to use.
    pub score_kind: PosteriorTransitionScoreKind,
    /// Alarm threshold: fire when `score >= threshold`.
    pub threshold: f64,
    /// Alarm-stabilization policy.
    pub persistence: PersistencePolicy,
}

impl Default for PosteriorTransitionConfig {
    /// `LeavePrevious` score, threshold `0.5`, immediate alarm.
    fn default() -> Self {
        Self {
            score_kind: PosteriorTransitionScoreKind::LeavePrevious,
            threshold: 0.5,
            persistence: PersistencePolicy::default(),
        }
    }
}

// =========================================================================
// Detector
// =========================================================================

/// Posterior Transition Detector.
///
/// # Scores
///
/// Two score variants are available via [`PosteriorTransitionScoreKind`]:
///
/// **LeavePrevious** (`s_t^leave`): measures how much posterior mass has left
/// the previously dominant regime.
///
/// ```text
/// r_{t-1} = argmax_j α_{t-1|t-1}(j)
/// s_t^leave = 1 − α_{t|t}(r_{t-1})   ∈ [0, 1]
/// ```
///
/// **TotalVariation** (`s_t^TV`): measures the total-variation distance
/// between consecutive filtered posteriors.
///
/// ```text
/// s_t^TV = (1/2) Σ_j | α_{t|t}(j) − α_{t-1|t-1}(j) |   ∈ [0, 1]
/// ```
///
/// # Semantics
///
/// Unlike the hard-switch detector, both scores are **continuous**: they
/// capture the magnitude of posterior migration rather than only the discrete
/// event of a label change.  This makes the detector less sensitive to small
/// transient fluctuations while remaining responsive to genuine structural
/// shifts in regime beliefs.
///
/// # Warmup
///
/// The first step stores the initial filtered posterior and produces
/// `ready = false`.  Scores are available from the second step onward.
#[derive(Debug, Clone)]
pub struct PosteriorTransitionDetector {
    pub config: PosteriorTransitionConfig,
    prev_filtered: Option<Vec<f64>>,
}

impl PosteriorTransitionDetector {
    /// Construct with the given configuration.
    pub fn new(config: PosteriorTransitionConfig) -> Self {
        Self {
            config,
            prev_filtered: None,
        }
    }
}

impl Default for PosteriorTransitionDetector {
    fn default() -> Self {
        Self::new(PosteriorTransitionConfig::default())
    }
}

impl Detector for PosteriorTransitionDetector {
    fn update(&mut self, input: &DetectorInput) -> DetectorOutput {
        let t = input.t;

        // Warmup: store first posterior, no score available yet.
        let Some(ref prev) = self.prev_filtered else {
            self.prev_filtered = Some(input.filtered.clone());
            return DetectorOutput {
                score: 0.0,
                alarm: false,
                alarm_event: None,
                t,
                ready: false,
            };
        };

        let score = match self.config.score_kind {
            PosteriorTransitionScoreKind::LeavePrevious => {
                let r_prev = dominant_regime(prev);
                1.0 - input.filtered[r_prev]
            }
            PosteriorTransitionScoreKind::TotalVariation => {
                0.5 * prev
                    .iter()
                    .zip(input.filtered.iter())
                    .map(|(p, q)| (q - p).abs())
                    .sum::<f64>()
            }
        };

        let threshold_crossed = score >= self.config.threshold;
        let alarm = self.config.persistence.check(threshold_crossed);

        let prev_dominant = dominant_regime(prev);
        let current_dominant = dominant_regime(&input.filtered);

        let alarm_event = alarm.then_some(AlarmEvent {
            t,
            score,
            detector_kind: DetectorKind::PosteriorTransition,
            dominant_regime_before: Some(prev_dominant),
            dominant_regime_after: current_dominant,
        });

        self.prev_filtered = Some(input.filtered.clone());

        DetectorOutput {
            score,
            alarm,
            alarm_event,
            t,
            ready: true,
        }
    }

    fn reset(&mut self) {
        self.prev_filtered = None;
        self.config.persistence.reset();
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::DetectorInput;

    fn make_input(filtered: Vec<f64>, t: usize) -> DetectorInput {
        let k = filtered.len();
        DetectorInput {
            predicted_next: vec![1.0 / k as f64; k],
            predictive_density: 1.0,
            log_predictive: 0.0,
            filtered,
            t,
        }
    }

    #[test]
    fn posterior_transition_not_ready_on_first_step() {
        let mut det = PosteriorTransitionDetector::default();
        let out = det.update(&make_input(vec![0.6, 0.4], 1));
        assert!(!out.ready);
        assert!(!out.alarm);
        assert_eq!(out.score, 0.0);
    }

    #[test]
    fn posterior_transition_leave_score_correct_value() {
        // prev dominant = 0 (α_{t-1|t-1}(0) = 0.9)
        // score = 1 − α_{t|t}(0) = 1 − 0.7 = 0.3
        let mut det = PosteriorTransitionDetector::default();
        det.update(&make_input(vec![0.9, 0.1], 1));
        let out = det.update(&make_input(vec![0.7, 0.3], 2));
        assert!(out.ready);
        assert!((out.score - 0.3).abs() < 1e-12, "score={}", out.score);
    }

    #[test]
    fn posterior_transition_leave_score_in_unit_interval() {
        let mut det = PosteriorTransitionDetector::default();
        det.update(&make_input(vec![0.9, 0.1], 1));
        let out = det.update(&make_input(vec![0.2, 0.8], 2));
        assert!(out.score >= 0.0 && out.score <= 1.0, "score={}", out.score);
    }

    #[test]
    fn posterior_transition_tv_score_extremes() {
        // Maximum TV: posteriors are antipodal → TV = 1.
        let config = PosteriorTransitionConfig {
            score_kind: PosteriorTransitionScoreKind::TotalVariation,
            threshold: 0.5,
            persistence: PersistencePolicy::default(),
        };
        let mut det = PosteriorTransitionDetector::new(config);
        det.update(&make_input(vec![1.0, 0.0], 1));
        let out = det.update(&make_input(vec![0.0, 1.0], 2));
        assert!((out.score - 1.0).abs() < 1e-12, "score={}", out.score);
    }

    #[test]
    fn posterior_transition_tv_score_zero_when_identical() {
        let config = PosteriorTransitionConfig {
            score_kind: PosteriorTransitionScoreKind::TotalVariation,
            threshold: 0.5,
            persistence: PersistencePolicy::default(),
        };
        let mut det = PosteriorTransitionDetector::new(config);
        det.update(&make_input(vec![0.7, 0.3], 1));
        let out = det.update(&make_input(vec![0.7, 0.3], 2));
        assert!(out.score.abs() < 1e-12, "score={}", out.score);
        assert!(!out.alarm);
    }

    #[test]
    fn posterior_transition_alarm_when_score_exceeds_threshold() {
        // score = 1 − α_{t|t}(0) = 1 − 0.1 = 0.9 > threshold 0.5
        let config = PosteriorTransitionConfig {
            score_kind: PosteriorTransitionScoreKind::LeavePrevious,
            threshold: 0.5,
            persistence: PersistencePolicy::default(),
        };
        let mut det = PosteriorTransitionDetector::new(config);
        det.update(&make_input(vec![0.9, 0.1], 1));
        let out = det.update(&make_input(vec![0.1, 0.9], 2));
        assert!(out.alarm, "score {} should exceed threshold 0.5", out.score);
        assert!((out.score - 0.9).abs() < 1e-12);
    }
}
