/// Fixed-parameter policy for the offline-trained, online-filtered detector.
///
/// # Design decision
///
/// The streaming detector in this project uses a **two-stage** architecture:
///
/// ## Stage 1 — Offline training
///
/// A `ModelParams` object is estimated from historical data using the offline
/// EM pipeline and frozen into a [`FrozenModel`]:
///
/// ```text
/// historical data  →  fit_em()  →  EmResult  →  FrozenModel
/// ```
///
/// ## Stage 2 — Online detection
///
/// A [`StreamingSession`] combines the frozen model with mutable online state
/// and runs causally:
///
/// ```text
/// FrozenModel  +  OnlineFilterState  +  Detector
///     ↓
/// for each yₜ:
///     OnlineFilterState::step(yₜ, &frozen.params)  →  OnlineStepResult
///     Detector::update(&DetectorInput::from(step))   →  DetectorOutput
/// ```
///
/// The key invariant:
///
/// ```text
/// FrozenModel.params  remains byte-for-byte unchanged throughout the stream.
/// ```
///
/// # Separation of concerns
///
/// | Object | Mutable during stream? |
/// |---|---|
/// | `FrozenModel.params` | No |
/// | `OnlineFilterState` | Yes (filtered posterior, t) |
/// | Detector state | Yes (prev. posterior, EWM baseline, counters) |
///
/// # What is intentionally excluded
///
/// The `FrozenModel` / `StreamingSession` pair does **not** support:
/// - online EM re-estimation,
/// - sliding-window refits,
/// - forgetting factors,
/// - adaptive means, variances, or transition rows.
///
/// Those belong to later extensions.  This module defines the **fixed-parameter
/// baseline** — the first benchmarkable detector mode.
use anyhow::Result;

use crate::model::em::EmResult;
use crate::model::params::ModelParams;
use crate::online::OnlineFilterState;

use super::{Detector, DetectorInput, DetectorOutput};

// =========================================================================
// FrozenModel
// =========================================================================

/// A fitted Markov Switching model held immutable for streaming detection.
///
/// Wraps a [`ModelParams`] object estimated offline and explicitly documents
/// the fixed-parameter policy: the contained parameters are never modified
/// after construction.
///
/// # Constructors
///
/// ```rust,ignore
/// // From a completed EM run:
/// let frozen = FrozenModel::from_em_result(&em_result);
///
/// // From a manually constructed / pre-validated ModelParams:
/// let frozen = FrozenModel::new(params)?;
/// ```
///
/// # Invariant
///
/// All methods on `FrozenModel` take `&self` (shared, immutable reference).
/// The `params` field is not `pub`; callers access it through [`params()`].
/// This prevents accidental mutation at call sites.
#[derive(Debug, Clone)]
pub struct FrozenModel {
    params: ModelParams,
}

impl FrozenModel {
    /// Construct from a validated `ModelParams`.
    ///
    /// Validates the parameter object before freezing.
    pub fn new(params: ModelParams) -> Result<Self> {
        params.validate()?;
        Ok(Self { params })
    }

    /// Convert the final fitted parameters from an EM run into a frozen model.
    ///
    /// Carries over only `EmResult::params`; all EM history, diagnostics,
    /// and offline-only quantities are discarded.  The conversion validates
    /// the parameter object.
    pub fn from_em_result(em: &EmResult) -> Result<Self> {
        Self::new(em.params.clone())
    }

    /// Read-only access to the frozen parameters.
    pub fn params(&self) -> &ModelParams {
        &self.params
    }

    /// Number of regimes K.
    pub fn k(&self) -> usize {
        self.params.k
    }
}

// =========================================================================
// StreamingSession
// =========================================================================

/// An online detection session pairing a [`FrozenModel`] with mutable
/// streaming state and a concrete detector variant.
///
/// The session enforces the offline-trained / online-filtered design:
/// - `model` is immutable throughout the run,
/// - `filter_state` and `detector` are the only mutable objects,
/// - each call to [`step`] processes exactly one observation causally.
///
/// # Type parameter
///
/// `D: Detector` — the detector variant to use.  Any of
/// [`HardSwitchDetector`], [`PosteriorTransitionDetector`], or
/// [`SurpriseDetector`] (and any future variants) can be plugged in.
///
/// # Example
///
/// ```rust,ignore
/// let frozen  = FrozenModel::from_em_result(&em_result)?;
/// let filter  = OnlineFilterState::new(frozen.params());
/// let det     = HardSwitchDetector::default();
/// let mut session = StreamingSession::new(frozen, filter, det);
///
/// for y in live_stream {
///     let out = session.step(y)?;
///     if out.detector.alarm {
///         let ev = out.detector.alarm_event.unwrap();
///         // handle alarm
///     }
/// }
/// ```
///
/// [`HardSwitchDetector`]: crate::detector::HardSwitchDetector
/// [`PosteriorTransitionDetector`]: crate::detector::PosteriorTransitionDetector
/// [`SurpriseDetector`]: crate::detector::SurpriseDetector
pub struct StreamingSession<D: Detector> {
    model: FrozenModel,
    filter_state: OnlineFilterState,
    detector: D,
}

impl<D: Detector> StreamingSession<D> {
    /// Construct a new session.
    ///
    /// The `filter_state` should be freshly initialised from `frozen.params()`
    /// via `OnlineFilterState::new(frozen.params())`.
    pub fn new(model: FrozenModel, filter_state: OnlineFilterState, detector: D) -> Self {
        Self {
            model,
            filter_state,
            detector,
        }
    }

    /// Process one observation and return the combined filter + detector output.
    ///
    /// Internally:
    /// 1. Calls `OnlineFilterState::step(y, &self.model.params)` — params passed
    ///    by immutable reference; the filter state is the only thing mutated.
    /// 2. Builds a `DetectorInput` from the step result.
    /// 3. Calls `Detector::update(&input)` — detector state is mutated.
    /// 4. Returns both results bundled in a [`SessionStepOutput`].
    pub fn step(&mut self, y: f64) -> Result<SessionStepOutput> {
        let filter_out = self.filter_state.step(y, self.model.params())?;
        let det_input = DetectorInput::from(&filter_out);
        let detector_out = self.detector.update(&det_input);
        Ok(SessionStepOutput {
            filter: filter_out,
            detector: detector_out,
        })
    }

    /// Process a batch of observations sequentially and return all outputs.
    pub fn step_batch(&mut self, obs: &[f64]) -> Result<Vec<SessionStepOutput>> {
        obs.iter().map(|&y| self.step(y)).collect()
    }

    /// Read-only access to the frozen model.
    pub fn model(&self) -> &FrozenModel {
        &self.model
    }

    /// Read-only access to the current filter state.
    pub fn filter_state(&self) -> &OnlineFilterState {
        &self.filter_state
    }

    /// Read-only access to the detector.
    pub fn detector(&self) -> &D {
        &self.detector
    }

    /// Reset the filter state and detector to their initial (pre-observation)
    /// condition.  The frozen model is unaffected.
    pub fn reset(&mut self) {
        self.filter_state = OnlineFilterState::new(self.model.params());
        self.detector.reset();
    }
}

// =========================================================================
// Session step output
// =========================================================================

/// Combined output of one [`StreamingSession::step`] call.
pub struct SessionStepOutput {
    /// Raw online filter step result (filtered posterior, predictive density, …).
    pub filter: crate::online::OnlineStepResult,
    /// Detector decision for this step (score, alarm flag, optional alarm event).
    pub detector: DetectorOutput,
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::{
        HardSwitchDetector, PersistencePolicy, SurpriseConfig, SurpriseDetector,
    };
    use crate::model::params::ModelParams;

    // Build a minimal 2-regime model with clean, well-separated parameters.
    fn make_params() -> ModelParams {
        ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-3.0, 3.0],
            vec![1.0, 1.0],
        )
    }

    // -----------------------------------------------------------------------
    // FrozenModel construction
    // -----------------------------------------------------------------------

    #[test]
    fn frozen_model_new_accepts_valid_params() {
        let params = make_params();
        let frozen = FrozenModel::new(params).unwrap();
        assert_eq!(frozen.k(), 2);
    }

    #[test]
    fn frozen_model_from_em_result_discards_history() {
        let params = make_params();
        let em = EmResult {
            params,
            log_likelihood: -42.0,
            ll_history: vec![-100.0, -42.0],
            n_iter: 1,
            converged: true,
        };
        let frozen = FrozenModel::from_em_result(&em).unwrap();
        assert_eq!(frozen.k(), 2);
        // No ll_history or n_iter stored in FrozenModel.
    }

    #[test]
    fn frozen_model_params_is_immutable_reference() {
        // The public API only exposes &ModelParams, never &mut ModelParams.
        // This test asserts the method signature by calling it.
        let frozen = FrozenModel::new(make_params()).unwrap();
        let p: &ModelParams = frozen.params();
        assert_eq!(p.k, 2);
    }

    // -----------------------------------------------------------------------
    // StreamingSession — basic step
    // -----------------------------------------------------------------------

    #[test]
    fn streaming_session_step_produces_output() {
        let frozen = FrozenModel::new(make_params()).unwrap();
        let filter = OnlineFilterState::new(frozen.params());
        let det = HardSwitchDetector::default();
        let mut session = StreamingSession::new(frozen, filter, det);

        let out = session.step(2.8).unwrap();
        assert_eq!(out.filter.t, 1);
        assert!(out.filter.filtered.iter().sum::<f64>() - 1.0 < 1e-9);
    }

    #[test]
    fn streaming_session_step_batch_length_matches() {
        let frozen = FrozenModel::new(make_params()).unwrap();
        let filter = OnlineFilterState::new(frozen.params());
        let det = HardSwitchDetector::default();
        let mut session = StreamingSession::new(frozen, filter, det);

        let obs = vec![3.0, -3.0, 3.0, -3.0, 3.0];
        let outputs = session.step_batch(&obs).unwrap();
        assert_eq!(outputs.len(), 5);
        for (i, o) in outputs.iter().enumerate() {
            assert_eq!(o.filter.t, i + 1);
        }
    }

    #[test]
    fn streaming_session_params_unchanged_after_steps() {
        // Snapshot the transition matrix before and after a streaming run
        // and assert it is byte-identical.
        let params = make_params();
        let transition_before = params.transition.clone();
        let frozen = FrozenModel::new(params).unwrap();
        let filter = OnlineFilterState::new(frozen.params());
        let det = HardSwitchDetector::default();
        let mut session = StreamingSession::new(frozen, filter, det);

        let obs: Vec<f64> = (0..50)
            .map(|i| if i % 2 == 0 { 3.0 } else { -3.0 })
            .collect();
        session.step_batch(&obs).unwrap();

        assert_eq!(
            session.model().params().transition,
            transition_before,
            "transition matrix must not change during streaming"
        );
    }

    #[test]
    fn streaming_session_reset_restores_initial_state() {
        let frozen = FrozenModel::new(make_params()).unwrap();
        let filter = OnlineFilterState::new(frozen.params());
        let det = HardSwitchDetector::default();
        let mut session = StreamingSession::new(frozen, filter, det);

        session.step_batch(&[3.0, -3.0, 3.0]).unwrap();
        assert_eq!(session.filter_state().t, 3);
        session.reset();
        assert_eq!(session.filter_state().t, 0);
    }

    #[test]
    fn streaming_session_surprise_detector_composes() {
        let config = SurpriseConfig {
            threshold: 5.0,
            ema_alpha: Some(0.2),
            persistence: PersistencePolicy::default(),
        };
        let frozen = FrozenModel::new(make_params()).unwrap();
        let filter = OnlineFilterState::new(frozen.params());
        let det = SurpriseDetector::new(config);
        let mut session = StreamingSession::new(frozen, filter, det);

        // Mild outlier — still unusual under both regimes but densities remain
        // strictly positive (z-score ~6 from the nearer regime mean).
        let out = session.step(9.0).unwrap();
        assert_eq!(out.filter.t, 1);
    }
}
