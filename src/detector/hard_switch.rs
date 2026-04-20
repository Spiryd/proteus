use super::{
    AlarmEvent, Detector, DetectorInput, DetectorKind, DetectorOutput, PersistencePolicy,
    dominant_regime,
};

// =========================================================================
// Configuration
// =========================================================================

/// Configuration for the [`HardSwitchDetector`].
#[derive(Debug, Clone)]
pub struct HardSwitchConfig {
    /// Minimum value of `max_j α_{t|t}(j)` required to allow an alarm.
    ///
    /// Set to `0.0` (the default) to disable confidence gating and alarm on
    /// every dominant-regime switch regardless of posterior certainty.  Set
    /// to, e.g., `0.7` to suppress alarms when the posterior is too diffuse.
    pub confidence_threshold: f64,
    /// Alarm-stabilization policy.
    pub persistence: PersistencePolicy,
}

impl Default for HardSwitchConfig {
    /// No confidence gating; alarm immediately on every regime switch.
    fn default() -> Self {
        Self {
            confidence_threshold: 0.0,
            persistence: PersistencePolicy::default(),
        }
    }
}

// =========================================================================
// Detector
// =========================================================================

/// Hard Switch Detector.
///
/// # Score
///
/// Let `Ŝ_t = argmax_j α_{t|t}(j)` be the dominant regime at time `t`.
///
/// ```text
/// s_t^hard = 1  if  Ŝ_t ≠ Ŝ_{t-1}  and  max_j α_{t|t}(j) ≥ confidence_threshold
///          = 0  otherwise
/// ```
///
/// An alarm is emitted when `s_t^hard = 1` and the [`PersistencePolicy`]
/// approves (i.e. the required number of consecutive threshold crossings have
/// been reached).
///
/// # Semantics
///
/// This detector defines "change" as a discrete switch in the most likely
/// latent regime.  It is the simplest possible regime-change rule and serves
/// as a natural baseline detector for comparison with the more continuous
/// [`super::PosteriorTransitionDetector`] and [`super::SurpriseDetector`].
///
/// # Warmup
///
/// No alarm is produced on the first step because there is no previous
/// dominant regime to compare against.  The output has `ready = false` at
/// `t = 1`.
///
/// # Persistence semantics
///
/// With `required_consecutive = N`, the detector alarms after N consecutive
/// steps each of which produced a regime switch.  This fires when the system
/// rapidly alternates between two or more regimes — a distinct instability
/// signal that complements the soft posterior-shift detectors.
#[derive(Debug, Clone)]
pub struct HardSwitchDetector {
    pub config: HardSwitchConfig,
    prev_dominant: Option<usize>,
}

impl HardSwitchDetector {
    /// Construct with the given configuration.
    pub fn new(config: HardSwitchConfig) -> Self {
        Self {
            config,
            prev_dominant: None,
        }
    }
}

impl Default for HardSwitchDetector {
    fn default() -> Self {
        Self::new(HardSwitchConfig::default())
    }
}

impl Detector for HardSwitchDetector {
    fn update(&mut self, input: &DetectorInput) -> DetectorOutput {
        let t = input.t;
        let current_dominant = dominant_regime(&input.filtered);
        let confidence = input.filtered[current_dominant];

        // Warmup: no previous dominant regime available yet.
        let Some(prev) = self.prev_dominant else {
            self.prev_dominant = Some(current_dominant);
            return DetectorOutput {
                score: 0.0,
                alarm: false,
                alarm_event: None,
                t,
                ready: false,
            };
        };

        let regime_changed = current_dominant != prev;
        let confidence_ok = confidence >= self.config.confidence_threshold;
        let threshold_crossed = regime_changed && confidence_ok;

        let score = if threshold_crossed { 1.0 } else { 0.0 };
        let alarm = self.config.persistence.check(threshold_crossed);

        let alarm_event = alarm.then_some(AlarmEvent {
            t,
            score,
            detector_kind: DetectorKind::HardSwitch,
            dominant_regime_before: Some(prev),
            dominant_regime_after: current_dominant,
        });

        self.prev_dominant = Some(current_dominant);

        DetectorOutput {
            score,
            alarm,
            alarm_event,
            t,
            ready: true,
        }
    }

    fn reset(&mut self) {
        self.prev_dominant = None;
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
    fn hard_switch_not_ready_on_first_step() {
        let mut det = HardSwitchDetector::default();
        let out = det.update(&make_input(vec![0.9, 0.1], 1));
        assert!(!out.ready);
        assert!(!out.alarm);
        assert_eq!(out.score, 0.0);
    }

    #[test]
    fn hard_switch_alarm_when_dominant_changes() {
        let mut det = HardSwitchDetector::default();
        det.update(&make_input(vec![0.9, 0.1], 1)); // warmup: dominant = 0
        let out = det.update(&make_input(vec![0.1, 0.9], 2)); // dominant = 1
        assert!(out.ready);
        assert!(out.alarm);
        let ev = out.alarm_event.unwrap();
        assert_eq!(ev.dominant_regime_before, Some(0));
        assert_eq!(ev.dominant_regime_after, 1);
    }

    #[test]
    fn hard_switch_no_alarm_when_regime_stays_same() {
        let mut det = HardSwitchDetector::default();
        det.update(&make_input(vec![0.9, 0.1], 1));
        let out = det.update(&make_input(vec![0.8, 0.2], 2)); // dominant still 0
        assert!(!out.alarm);
        assert_eq!(out.score, 0.0);
    }

    #[test]
    fn hard_switch_confidence_gate_suppresses_uncertain_switch() {
        let config = HardSwitchConfig {
            confidence_threshold: 0.8,
            persistence: PersistencePolicy::default(),
        };
        let mut det = HardSwitchDetector::new(config);
        det.update(&make_input(vec![0.9, 0.1], 1)); // dominant = 0
        // Dominant switches to 1 but with only 0.6 confidence — below gate.
        let out = det.update(&make_input(vec![0.4, 0.6], 2));
        assert!(!out.alarm, "confidence 0.6 < 0.8 should suppress alarm");
        assert_eq!(out.score, 0.0);
    }

    #[test]
    fn hard_switch_persistence_requires_consecutive_switches() {
        // With required_consecutive=2: alarm fires after 2 consecutive
        // dominant-regime switches, detecting rapid regime alternation.
        let config = HardSwitchConfig {
            confidence_threshold: 0.0,
            persistence: PersistencePolicy::new(2, 0),
        };
        let mut det = HardSwitchDetector::new(config);
        det.update(&make_input(vec![0.9, 0.1], 1)); // warmup: prev=0
        let out1 = det.update(&make_input(vec![0.1, 0.9], 2)); // switch 0→1; 1 crossing
        assert!(!out1.alarm, "1 crossing not enough");
        let out2 = det.update(&make_input(vec![0.9, 0.1], 3)); // switch 1→0; 2nd crossing
        assert!(out2.alarm, "2nd consecutive switch should alarm");
    }
}
