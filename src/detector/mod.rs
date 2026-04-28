#![allow(unused_imports, dead_code)]
/// Changepoint detection layer for the Gaussian Markov Switching Model.
///
/// # Architecture
///
/// The detector layer sits strictly above the online inference layer.  It
/// consumes causal outputs of [`crate::online::OnlineFilterState::step`] and
/// applies a score function plus an alarm policy to decide whether a
/// changepoint has occurred.
///
/// ```text
/// Online Filter  →  DetectorInput  →  Detector::update  →  DetectorOutput
///                                                              └ AlarmEvent
/// ```
///
/// The filter and detector are intentionally separated:
/// - the filter maintains probabilistic regime beliefs,
/// - the detector applies a score and alarm policy on top.
///
/// # Three detector variants
///
/// 1. [`HardSwitchDetector`] — alarms when the dominant regime label changes:
///    `Ŝ_t ≠ Ŝ_{t-1}` where `Ŝ_t = argmax_j α_{t|t}(j)`.
///
/// 2. [`PosteriorTransitionDetector`] — alarms when posterior mass has migrated
///    away from the previously dominant regime; two score variants available.
///
/// 3. [`SurpriseDetector`] — alarms when the predictive surprise
///    `s_t = −log c_t` (or a baseline-adjusted version) exceeds a threshold.
///
/// All three share the [`Detector`] trait and the [`PersistencePolicy`]
/// alarm stabilizer.
///
/// # Causal guarantee
///
/// Every detector produces decisions using only `y_{1:t}`.  Smoothed
/// quantities, EM updates, and future observations are never accessed here.
pub mod frozen;
pub mod hard_switch;
pub mod posterior_transition;
pub mod surprise;

pub use frozen::{FrozenModel, SessionStepOutput, StreamingSession};
pub use hard_switch::{HardSwitchConfig, HardSwitchDetector};
pub use posterior_transition::{
    PosteriorTransitionConfig, PosteriorTransitionDetector, PosteriorTransitionScoreKind,
};
pub use surprise::{SurpriseConfig, SurpriseDetector};

use crate::online::OnlineStepResult;

// =========================================================================
// Detector identifier
// =========================================================================

/// Tag identifying which detector variant produced a [`DetectorOutput`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorKind {
    HardSwitch,
    PosteriorTransition,
    Surprise,
}

// =========================================================================
// Detector input
// =========================================================================

/// Causal quantities from one online filter step, consumed by detector variants.
///
/// Contains only the filtered posterior `α_{t|t}`, the one-step-ahead
/// predicted posterior `α_{t+1|t}`, and the predictive density `c_t`.  These
/// are exactly the causal quantities available at time `t`.
///
/// Constructed from [`OnlineStepResult`] via `DetectorInput::from`.  Can also
/// be constructed directly in tests.
#[derive(Debug, Clone)]
pub struct DetectorInput {
    /// Filtered posterior `α_{t|t}(j) = Pr(Sₜ=j | y_{1:t})`.  Length K.
    pub filtered: Vec<f64>,
    /// One-step-ahead predicted posterior `α_{t+1|t}(j)`.  Length K.
    pub predicted_next: Vec<f64>,
    /// Predictive density `c_t = f(yₜ | y_{1:t-1})`.
    pub predictive_density: f64,
    /// Log predictive density `log c_t`.
    pub log_predictive: f64,
    /// 1-based observation index (matches `OnlineStepResult::t`).
    pub t: usize,
}

impl From<&OnlineStepResult> for DetectorInput {
    fn from(r: &OnlineStepResult) -> Self {
        Self {
            filtered: r.filtered.clone(),
            predicted_next: r.predicted_next.clone(),
            predictive_density: r.predictive_density,
            log_predictive: r.log_predictive,
            t: r.t,
        }
    }
}

impl From<OnlineStepResult> for DetectorInput {
    fn from(r: OnlineStepResult) -> Self {
        Self {
            filtered: r.filtered,
            predicted_next: r.predicted_next,
            predictive_density: r.predictive_density,
            log_predictive: r.log_predictive,
            t: r.t,
        }
    }
}

// =========================================================================
// Detector output and alarm event
// =========================================================================

/// Output produced by one detector update step.
#[derive(Debug, Clone)]
pub struct DetectorOutput {
    /// Computed change score for this step.
    pub score: f64,
    /// Whether an alarm is raised at this step.
    pub alarm: bool,
    /// Alarm event; `Some` iff `alarm == true`.
    pub alarm_event: Option<AlarmEvent>,
    /// Time index from the `DetectorInput`.
    pub t: usize,
    /// Whether the detector has seen enough history to produce scores.
    /// `false` during warmup (e.g. the first step for detectors that need
    /// the previous filtered posterior).
    pub ready: bool,
}

/// A changepoint alarm emitted by a detector at time `t`.
#[derive(Debug, Clone)]
pub struct AlarmEvent {
    /// Time index at which the alarm fires.
    pub t: usize,
    /// Change score that triggered the alarm.
    pub score: f64,
    /// Which detector variant produced this alarm.
    pub detector_kind: DetectorKind,
    /// Dominant regime before the alarm (`argmax α_{t-1|t-1}`), if available.
    /// `None` for detectors that do not track the previous dominant regime.
    pub dominant_regime_before: Option<usize>,
    /// Dominant regime at the alarm step (`argmax α_{t|t}`).
    pub dominant_regime_after: usize,
}

// =========================================================================
// Detector trait
// =========================================================================

/// Common interface for all changepoint detector variants.
///
/// A detector:
/// - consumes one [`DetectorInput`] per step,
/// - maintains its own internal state (previous posteriors, counters, etc.),
/// - computes a causal change score,
/// - optionally raises an alarm via a [`PersistencePolicy`],
/// - never accesses future observations.
///
/// Implementations are expected to be object-safe.
pub trait Detector {
    /// Process one online filter update and return the detector decision.
    fn update(&mut self, input: &DetectorInput) -> DetectorOutput;

    /// Reset the detector to its initial (pre-observation) state.
    fn reset(&mut self);
}

// =========================================================================
// Persistence / cooldown policy
// =========================================================================

/// Shared alarm-stabilization policy, composable with any detector variant.
///
/// # Purpose
///
/// Online detectors can produce transient score spikes caused by noisy
/// observations or short-lived posterior oscillations.  The persistence
/// policy suppresses isolated spikes by requiring that the threshold be
/// crossed for `required_consecutive` consecutive steps before emitting
/// an alarm.  A `cooldown` parameter prevents immediate re-triggering.
///
/// # Behavior
///
/// 1. An alarm fires only after `required_consecutive` consecutive threshold
///    crossings.
/// 2. After each alarm, a cooldown of `cooldown` steps suppresses further
///    alarms.  The consecutive counter is reset during cooldown.
/// 3. Any threshold miss (score below threshold) resets the consecutive counter.
///
/// # Defaults
///
/// `PersistencePolicy::default()` produces `required_consecutive = 1` and
/// `cooldown = 0`, which corresponds to immediate alarming with no
/// refractory period.
#[derive(Debug, Clone)]
pub struct PersistencePolicy {
    /// Minimum consecutive threshold crossings required to emit an alarm.
    /// Clamped to at least 1 internally.
    pub required_consecutive: usize,
    /// Steps of suppression after each alarm.  0 = no cooldown.
    pub cooldown: usize,
    /// Internal: current run of consecutive threshold crossings.
    pub consecutive_count: usize,
    /// Internal: remaining cooldown steps.
    pub cooldown_remaining: usize,
}

impl Default for PersistencePolicy {
    /// Alarm immediately (1 crossing), no cooldown.
    fn default() -> Self {
        Self {
            required_consecutive: 1,
            cooldown: 0,
            consecutive_count: 0,
            cooldown_remaining: 0,
        }
    }
}

impl PersistencePolicy {
    /// Construct a policy with the given persistence and cooldown parameters.
    pub fn new(required_consecutive: usize, cooldown: usize) -> Self {
        Self {
            required_consecutive: required_consecutive.max(1),
            cooldown,
            consecutive_count: 0,
            cooldown_remaining: 0,
        }
    }

    /// Record whether the detector score crossed the threshold this step.
    ///
    /// Returns `true` iff an alarm should fire at this step.  Updates the
    /// consecutive counter and cooldown counter in-place.
    pub fn check(&mut self, threshold_crossed: bool) -> bool {
        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            self.consecutive_count = 0;
            return false;
        }
        if threshold_crossed {
            self.consecutive_count += 1;
            if self.consecutive_count >= self.required_consecutive {
                self.consecutive_count = 0;
                self.cooldown_remaining = self.cooldown;
                return true;
            }
        } else {
            self.consecutive_count = 0;
        }
        false
    }

    /// Reset all internal state to the initial condition.
    pub fn reset(&mut self) {
        self.consecutive_count = 0;
        self.cooldown_remaining = 0;
    }
}

// =========================================================================
// Shared helper: dominant regime
// =========================================================================

/// Return the index of the regime with the highest filtered probability.
/// Ties broken by the lower index.
pub(crate) fn dominant_regime(filtered: &[f64]) -> usize {
    filtered
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// =========================================================================
// Tests — shared infrastructure
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // PersistencePolicy tests
    // -----------------------------------------------------------------------

    #[test]
    fn persistence_default_fires_immediately() {
        let mut p = PersistencePolicy::default();
        assert!(p.check(true));
        assert!(!p.check(false));
        assert!(p.check(true));
    }

    #[test]
    fn persistence_requires_consecutive_crossings() {
        let mut p = PersistencePolicy::new(3, 0);
        assert!(!p.check(true)); // 1
        assert!(!p.check(true)); // 2
        assert!(p.check(true)); // 3 → alarm
        assert!(!p.check(true)); // 1 again (reset after alarm)
        assert!(!p.check(true)); // 2
        assert!(p.check(true)); // 3 → alarm again
    }

    #[test]
    fn persistence_miss_resets_counter() {
        let mut p = PersistencePolicy::new(2, 0);
        assert!(!p.check(true)); // 1
        assert!(!p.check(false)); // miss → reset to 0
        assert!(!p.check(true)); // 1 again
        assert!(p.check(true)); // 2 → alarm
    }

    #[test]
    fn persistence_cooldown_suppresses_refire() {
        let mut p = PersistencePolicy::new(1, 2); // 2-step cooldown
        assert!(p.check(true)); // alarm; cooldown_remaining = 2
        assert!(!p.check(true)); // cooldown 2 → 1; no alarm
        assert!(!p.check(true)); // cooldown 1 → 0; no alarm
        assert!(p.check(true)); // cooldown expired; alarm
    }

    #[test]
    fn persistence_reset_clears_state() {
        let mut p = PersistencePolicy::new(2, 5);
        p.check(true);
        p.check(true); // fires; cooldown_remaining = 5
        p.reset();
        assert_eq!(p.consecutive_count, 0);
        assert_eq!(p.cooldown_remaining, 0);
        // Should be able to fire again immediately after reset.
        assert!(!p.check(true)); // 1
        assert!(p.check(true)); // 2 → alarm
    }

    // -----------------------------------------------------------------------
    // DetectorInput conversion
    // -----------------------------------------------------------------------

    #[test]
    fn detector_input_from_online_step_result() {
        let r = crate::online::OnlineStepResult {
            filtered: vec![0.3, 0.7],
            predicted_next: vec![0.35, 0.65],
            predictive_density: 0.25,
            log_predictive: 0.25_f64.ln(),
            t: 5,
        };
        let input = DetectorInput::from(&r);
        assert_eq!(input.t, 5);
        assert_eq!(input.filtered, vec![0.3, 0.7]);
        assert_eq!(input.predicted_next, vec![0.35, 0.65]);
        assert!((input.log_predictive - 0.25_f64.ln()).abs() < 1e-15);
    }
}
