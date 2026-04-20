use super::{
    AlarmEvent, Detector, DetectorInput, DetectorKind, DetectorOutput, PersistencePolicy,
    dominant_regime,
};

// =========================================================================
// Configuration
// =========================================================================

/// Configuration for the [`SurpriseDetector`].
#[derive(Debug, Clone)]
pub struct SurpriseConfig {
    /// Alarm threshold: fire when the chosen score `>= threshold`.
    pub threshold: f64,
    /// EMA smoothing coefficient `α ∈ (0, 1]` for the baseline tracker.
    ///
    /// - `None` — use the raw surprise score `s_t = −log c_t`.
    /// - `Some(α)` — use the baseline-adjusted score `s_t − b_{t−1}`, where
    ///   `b_t = α·s_t + (1−α)·b_{t−1}` is an exponentially weighted moving
    ///   average of recent surprise.  The first step initializes the baseline
    ///   and produces `ready = false`.
    pub ema_alpha: Option<f64>,
    /// Alarm-stabilization policy.
    pub persistence: PersistencePolicy,
}

impl Default for SurpriseConfig {
    /// Raw surprise score, threshold `3.0`, immediate alarm, no EMA baseline.
    fn default() -> Self {
        Self {
            threshold: 3.0,
            ema_alpha: None,
            persistence: PersistencePolicy::default(),
        }
    }
}

// =========================================================================
// Detector
// =========================================================================

/// Surprise (Predictive Instability) Detector.
///
/// # Score
///
/// The raw surprise score is the negative log predictive density:
///
/// ```text
/// s_t^surp = −log c_t
/// ```
///
/// where `c_t = f(yₜ | y_{1:t−1})` is the one-step-ahead predictive density
/// produced by the online filter.  Large values indicate that the new
/// observation was highly unexpected under the model's current regime mixture
/// prediction.
///
/// # Baseline-adjusted variant
///
/// When `ema_alpha = Some(α)`, an exponentially weighted moving average
/// baseline `b_t` is maintained and the **adjusted score** is used:
///
/// ```text
/// s_t^adj = s_t^surp − b_{t−1}
/// b_t     = α · s_t^surp + (1−α) · b_{t−1}
/// ```
///
/// The previous baseline is used in the score (causal: `b_{t−1}` is known
/// before `yₜ` arrives).  The baseline is updated after the score is
/// computed.  This normalizes the surprise relative to the model's recent
/// predictive track record, distinguishing globally noisy regimes from
/// genuinely abnormal prediction failures.
///
/// # Semantics
///
/// The surprise detector captures a **different notion of change** from the
/// regime-switch detectors: it fires when the model's predictive structure
/// becomes incompatible with the incoming observation, even before the
/// filtered posterior has fully migrated to a new regime.
///
/// # Warmup
///
/// - When `ema_alpha = None`: ready from the first step.
/// - When `ema_alpha = Some(_)`: the first step initializes the baseline and
///   produces `ready = false`.  Scores are available from the second step.
#[derive(Debug, Clone)]
pub struct SurpriseDetector {
    pub config: SurpriseConfig,
    /// Current EMA baseline.  `None` until the first observation.
    ema_baseline: Option<f64>,
}

impl SurpriseDetector {
    /// Construct with the given configuration.
    pub fn new(config: SurpriseConfig) -> Self {
        Self {
            config,
            ema_baseline: None,
        }
    }
}

impl Default for SurpriseDetector {
    fn default() -> Self {
        Self::new(SurpriseConfig::default())
    }
}

impl Detector for SurpriseDetector {
    fn update(&mut self, input: &DetectorInput) -> DetectorOutput {
        let t = input.t;
        let raw_surprise = -input.log_predictive;
        let current_dominant = dominant_regime(&input.filtered);

        let (score, ready) = match self.config.ema_alpha {
            None => (raw_surprise, true),
            Some(alpha) => match self.ema_baseline {
                None => {
                    // First step: initialize baseline; not ready.
                    self.ema_baseline = Some(raw_surprise);
                    (0.0, false)
                }
                Some(baseline) => {
                    // Adjusted score uses the *previous* baseline (causal).
                    let adjusted = raw_surprise - baseline;
                    // Update baseline after computing score.
                    self.ema_baseline = Some(alpha * raw_surprise + (1.0 - alpha) * baseline);
                    (adjusted, true)
                }
            },
        };

        if !ready {
            return DetectorOutput {
                score: 0.0,
                alarm: false,
                alarm_event: None,
                t,
                ready: false,
            };
        }

        let threshold_crossed = score >= self.config.threshold;
        let alarm = self.config.persistence.check(threshold_crossed);

        let alarm_event = alarm.then_some(AlarmEvent {
            t,
            score,
            detector_kind: DetectorKind::Surprise,
            dominant_regime_before: None,
            dominant_regime_after: current_dominant,
        });

        DetectorOutput {
            score,
            alarm,
            alarm_event,
            t,
            ready: true,
        }
    }

    fn reset(&mut self) {
        self.ema_baseline = None;
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

    fn make_input(log_predictive: f64, t: usize) -> DetectorInput {
        DetectorInput {
            filtered: vec![0.5, 0.5],
            predicted_next: vec![0.5, 0.5],
            predictive_density: log_predictive.exp(),
            log_predictive,
            t,
        }
    }

    #[test]
    fn surprise_score_equals_neg_log_predictive() {
        let mut det = SurpriseDetector::default(); // no EMA baseline
        let out = det.update(&make_input(-2.5, 1));
        assert!(out.ready);
        assert!((out.score - 2.5).abs() < 1e-12, "score={}", out.score);
    }

    #[test]
    fn surprise_no_alarm_below_threshold() {
        let mut det = SurpriseDetector::default(); // threshold = 3.0
        let out = det.update(&make_input(-1.0, 1)); // surprise = 1.0 < 3.0
        assert!(!out.alarm);
    }

    #[test]
    fn surprise_alarm_above_threshold() {
        let config = SurpriseConfig {
            threshold: 2.0,
            ema_alpha: None,
            persistence: PersistencePolicy::default(),
        };
        let mut det = SurpriseDetector::new(config);
        let out = det.update(&make_input(-5.0, 1)); // surprise = 5.0 > 2.0
        assert!(out.alarm, "surprise 5.0 should exceed threshold 2.0");
    }

    #[test]
    fn surprise_ema_not_ready_on_first_step() {
        let config = SurpriseConfig {
            threshold: 1.0,
            ema_alpha: Some(0.3),
            persistence: PersistencePolicy::default(),
        };
        let mut det = SurpriseDetector::new(config);
        let out = det.update(&make_input(-10.0, 1)); // huge surprise but first step
        assert!(!out.ready);
        assert!(!out.alarm);
    }

    #[test]
    fn surprise_ema_adjusted_score_and_alarm() {
        // Step 1: baseline initialized at raw_surprise = 2.0; ready=false.
        // Step 2: raw_surprise=5.0, baseline=2.0, adjusted=3.0 > threshold=2.0 → alarm.
        //         new baseline = 0.5*5 + 0.5*2 = 3.5
        let config = SurpriseConfig {
            threshold: 2.0,
            ema_alpha: Some(0.5),
            persistence: PersistencePolicy::default(),
        };
        let mut det = SurpriseDetector::new(config);
        det.update(&make_input(-2.0, 1)); // not ready; baseline = 2.0
        let out = det.update(&make_input(-5.0, 2)); // adjusted = 5.0 - 2.0 = 3.0
        assert!(out.ready);
        assert!(
            (out.score - 3.0).abs() < 1e-12,
            "adjusted score={}",
            out.score
        );
        assert!(out.alarm);
    }
}
