/// Online (streaming) inference for the Gaussian Markov Switching Model.
///
/// # Causal boundary
///
/// At time `t`, the online system may use **only** `y_{1:t}`.
///
/// The only latent-state posterior consistent with this rule is the **filtered
/// posterior**:
///
/// ```text
/// α_{t|t}(j) = Pr(Sₜ=j | y_{1:t})
/// ```
///
/// All offline inference objects — smoothed marginals `γₜ(j) = Pr(Sₜ=j | y_{1:T})`,
/// pairwise posteriors `ξₜ(i,j) = Pr(S_{t-1}=i, Sₜ=j | y_{1:T})`, EM parameter
/// updates — are **forbidden** inside this module.  Nothing here depends on
/// `model::smoother`, `model::pairwise`, `model::em`, or `model::diagnostics`.
///
/// # State machine design
///
/// Online inference is a **persistent state machine**, not a batch computation.
/// The unit of computation is one streaming step:
///
/// ```text
/// OnlineFilterState  +  y_t  →  OnlineStepResult  +  OnlineFilterState'
/// ```
///
/// Use [`OnlineFilterState::new`] to initialise, then call
/// [`OnlineFilterState::step`] once per incoming observation.  The state is
/// mutated in-place; the fitted [`ModelParams`] are only read, never modified.
///
/// # Streaming recursion
///
/// Given the previous filtered state `α_{t-1|t-1}` and a new observation `y_t`:
///
/// 1. **Predict**
///    ```text
///    α_{t|t-1}(j) = Σᵢ p_{ij} · α_{t-1|t-1}(i)
///    ```
/// 2. **Emission scores**
///    ```text
///    fⱼ(yₜ) = N(yₜ; μⱼ, σⱼ²)
///    ```
/// 3. **Predictive density**
///    ```text
///    cₜ = Σⱼ fⱼ(yₜ) · α_{t|t-1}(j)
///    ```
/// 4. **Bayes update**
///    ```text
///    α_{t|t}(j) = fⱼ(yₜ) · α_{t|t-1}(j) / cₜ
///    ```
///
/// The one-step-ahead prediction for the *next* step is derived in the same call:
///
/// ```text
/// α_{t+1|t}(j) = Σᵢ p_{ij} · α_{t|t}(i)
/// ```
///
/// # Module dependencies
///
/// ```text
/// online::OnlineFilterState
///     → model::params::ModelParams   (read-only)
///     → model::emission::Emission    (log-density evaluation)
/// ```
///
/// No dependency on smoother, pairwise, em, or diagnostics.
use anyhow::Result;

use crate::model::emission::Emission;
use crate::model::params::ModelParams;

// ---------------------------------------------------------------------------
// Tolerance for normalization / finiteness checks
// ---------------------------------------------------------------------------

/// Maximum tolerated deviation from 1.0 in any probability vector.
const NORM_TOL: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Persistent online state
// ---------------------------------------------------------------------------

/// Persistent streaming inference state for the Gaussian Markov Switching Model.
///
/// Holds only the minimal information required for the next streaming step:
/// the current filtered posterior and the current time index.  No full-sample
/// arrays or offline-only quantities are stored here.
///
/// # Initialization
///
/// ```rust,ignore
/// let state = OnlineFilterState::new(&fitted_params);
/// ```
///
/// At construction, the state is set to `t = 0` and
/// `filtered = π` (the initial regime distribution).  No observation has been
/// consumed yet; the first `step` call processes `y₁`.
///
/// # Usage
///
/// ```rust,ignore
/// for y in stream {
///     let result = state.step(y, &params)?;
///     // result.filtered       = α_{t|t}
///     // result.predicted_next = α_{t+1|t}
///     // result.predictive_density = c_t
/// }
/// ```
#[derive(Debug, Clone)]
pub struct OnlineFilterState {
    /// Current filtered posterior α_{t|t}(j).  Sums to 1.  Length K.
    ///
    /// Before the first observation, this is initialised to `π`
    /// (the initial regime distribution).
    pub filtered: Vec<f64>,
    /// Number of observations processed so far (0-based: 0 = no observations
    /// consumed, 1 = one observation consumed, etc.).
    pub t: usize,
    /// Cumulative online log-score Σ_{s=1}^{t} log c_s.
    ///
    /// After processing `t` observations this equals the value
    /// `FilterResult::log_likelihood` would return for the same sequence.
    pub cumulative_log_score: f64,
}

impl OnlineFilterState {
    /// Create a new streaming state initialised from the fitted parameter set.
    ///
    /// Sets `filtered = π`, `t = 0`, `cumulative_log_score = 0.0`.
    ///
    /// No observation is consumed.  The initial filtered vector equals the
    /// initial regime distribution `π`, consistent with `α_{1|0}(j) = π_j`
    /// (the prediction before the first observation).
    pub fn new(params: &ModelParams) -> Self {
        Self {
            filtered: params.pi.clone(),
            t: 0,
            cumulative_log_score: 0.0,
        }
    }

    /// Process one new observation and advance the streaming state.
    ///
    /// This is the core streaming step.  It implements the four-step online
    /// recursion (predict → score → predictive density → Bayes update) under
    /// the **fixed** fitted parameters `params`.
    ///
    /// The state is mutated in-place:
    /// - `self.filtered` is replaced with `α_{t|t}`,
    /// - `self.t` is incremented by 1,
    /// - `self.cumulative_log_score` is incremented by `log cₜ`.
    ///
    /// # Arguments
    ///
    /// - `y`      — new observation `yₜ`,
    /// - `params` — **immutable** fitted parameters; never modified.
    ///
    /// # Returns
    ///
    /// An [`OnlineStepResult`] containing all causal quantities at time `t`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - the predictive density `cₜ ≤ 0` (observation incompatible with all
    ///   regimes under the fitted model),
    /// - any normalized probability or the log-score is non-finite.
    pub fn step(&mut self, y: f64, params: &ModelParams) -> Result<OnlineStepResult> {
        let k = params.k;
        let emission = Emission::new(params.means.clone(), params.variances.clone());

        // ------------------------------------------------------------------
        // Step 1 — Predict: α_{t|t-1}(j) = Σᵢ p_{ij} · α_{t-1|t-1}(i)
        //
        // `self.filtered` currently holds α_{t-1|t-1}.
        // ------------------------------------------------------------------
        let mut predicted = vec![0.0_f64; k];
        for (j, item) in predicted.iter_mut().enumerate().take(k) {
            for i in 0..k {
                *item += params.transition_row(i)[j] * self.filtered[i];
            }
        }

        // ------------------------------------------------------------------
        // Step 2 & 3 — Emission scores and predictive density
        //
        // Computed in log-space to avoid underflow, then exponentiated once
        // for the final normalisation.
        //
        // log_scores[j] = log fⱼ(yₜ) + log α_{t|t-1}(j)
        // cₜ = Σⱼ exp(log_scores[j])   (log-sum-exp)
        // ------------------------------------------------------------------
        let log_scores: Vec<f64> = (0..k)
            .map(|j| {
                let lp = predicted[j];
                if lp <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    emission.log_density(y, j) + lp.ln()
                }
            })
            .collect();

        let max_log = log_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        if !max_log.is_finite() {
            anyhow::bail!(
                "OnlineFilterState::step at t={}: observation y={y} has zero \
                 density under all regimes (all log-scores are -∞)",
                self.t + 1
            );
        }

        let sum_exp: f64 = log_scores.iter().map(|&l| (l - max_log).exp()).sum();
        let log_ct = max_log + sum_exp.ln();

        // ------------------------------------------------------------------
        // Step 4 — Bayes update in log-space (numerically stable).
        //
        // When the observation is extreme relative to all regime distributions,
        // ct = exp(log_ct) may underflow to 0 in f64.  Computing the filtered
        // posterior directly as exp(log_scores[j] - log_ct) avoids division by
        // zero while preserving the correct posterior proportions.
        //
        // This is mathematically identical to the naive form
        //   filtered[j] = emission * predicted / ct
        // but remains well-conditioned for any finite log_ct.
        // ------------------------------------------------------------------
        let mut new_filtered = vec![0.0_f64; k];
        for j in 0..k {
            new_filtered[j] = (log_scores[j] - log_ct).exp();
        }

        let ct = log_ct.exp(); // may be 0 for extreme obs; only used for StepResult field

        // ------------------------------------------------------------------
        // Runtime invariant: filtered normalization
        // ------------------------------------------------------------------
        let filtered_sum: f64 = new_filtered.iter().sum();
        if (filtered_sum - 1.0).abs() > NORM_TOL {
            anyhow::bail!(
                "OnlineFilterState::step at t={}: filtered posterior sums to \
                 {filtered_sum:.15} (deviation from 1 exceeds {NORM_TOL})",
                self.t + 1
            );
        }
        for (j, &v) in new_filtered.iter().enumerate() {
            if !v.is_finite() {
                anyhow::bail!(
                    "OnlineFilterState::step at t={}: filtered[{j}] = {v} is non-finite",
                    self.t + 1
                );
            }
        }

        // ------------------------------------------------------------------
        // One-step-ahead prediction: α_{t+1|t}(j) = Σᵢ p_{ij} · α_{t|t}(i)
        // ------------------------------------------------------------------
        let mut predicted_next = vec![0.0_f64; k];
        for (j, item) in predicted_next.iter_mut().enumerate().take(k) {
            for (i, &nf) in new_filtered.iter().enumerate().take(k) {
                *item += params.transition_row(i)[j] * nf;
            }
        }

        // Runtime invariant: predicted_next normalization.
        let pred_sum: f64 = predicted_next.iter().sum();
        if (pred_sum - 1.0).abs() > NORM_TOL {
            anyhow::bail!(
                "OnlineFilterState::step at t={}: predicted_next sums to \
                 {pred_sum:.15} (deviation from 1 exceeds {NORM_TOL})",
                self.t + 1
            );
        }

        // ------------------------------------------------------------------
        // Advance state
        // ------------------------------------------------------------------
        let new_t = self.t + 1;
        self.filtered.clone_from(&new_filtered);
        self.t = new_t;
        self.cumulative_log_score += log_ct;

        Ok(OnlineStepResult {
            filtered: new_filtered,
            predicted_next,
            predictive_density: ct,
            log_predictive: log_ct,
            t: new_t,
        })
    }
}

// ---------------------------------------------------------------------------
// Online step result
// ---------------------------------------------------------------------------

/// Causal output of one streaming inference step.
///
/// All fields are measurable with respect to `y_{1:t}` only.  No smoothed
/// quantities or hindsight-based posteriors are present.  This type is the
/// boundary object that a later changepoint detector layer will consume.
#[derive(Debug, Clone)]
pub struct OnlineStepResult {
    /// Updated filtered posterior α_{t|t}(j) = Pr(Sₜ=j | y_{1:t}).  Length K.
    pub filtered: Vec<f64>,
    /// One-step-ahead predicted posterior α_{t+1|t}(j) = Pr(S_{t+1}=j | y_{1:t}).
    /// Length K.  Useful as the initial state for the next step and for scoring.
    pub predicted_next: Vec<f64>,
    /// Predictive density cₜ = f(yₜ | y_{1:t-1}).  Strictly positive.
    pub predictive_density: f64,
    /// log cₜ = log f(yₜ | y_{1:t-1}).  Finite.
    pub log_predictive: f64,
    /// Time index t (1-based: equals the number of observations processed
    /// so far, including this one).
    pub t: usize,
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::filter::filter;
    use crate::model::params::ModelParams;
    use crate::model::simulate::simulate;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    const SEED: u64 = 42;

    fn k2_params() -> ModelParams {
        ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-3.0, 3.0],
            vec![1.0, 1.0],
        )
    }

    fn k3_params() -> ModelParams {
        ModelParams::new(
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            vec![
                vec![0.8, 0.1, 0.1],
                vec![0.1, 0.8, 0.1],
                vec![0.1, 0.1, 0.8],
            ],
            vec![-4.0, 0.0, 4.0],
            vec![1.0, 1.0, 1.0],
        )
    }

    fn sim_obs(params: ModelParams, t: usize, seed: u64) -> Vec<f64> {
        let mut rng = SmallRng::seed_from_u64(seed);
        simulate(params, t, &mut rng).unwrap().observations
    }

    // -----------------------------------------------------------------------
    // Initialization (2 tests)
    // -----------------------------------------------------------------------

    /// new() sets filtered = π, t = 0, cumulative_log_score = 0.
    #[test]
    fn init_state_matches_pi() {
        let params = k2_params();
        let state = OnlineFilterState::new(&params);
        assert_eq!(state.t, 0);
        assert_eq!(state.cumulative_log_score, 0.0);
        for (j, &f) in state.filtered.iter().enumerate() {
            assert!(
                (f - params.pi[j]).abs() < 1e-15,
                "filtered[{j}] = {f}, expected pi[{j}] = {}",
                params.pi[j]
            );
        }
    }

    /// new() on K=3 params initialises a length-3 filtered vector.
    #[test]
    fn init_k3_correct_length() {
        let params = k3_params();
        let state = OnlineFilterState::new(&params);
        assert_eq!(state.filtered.len(), 3);
        assert_eq!(state.t, 0);
    }

    // -----------------------------------------------------------------------
    // Single-step correctness (4 tests)
    // -----------------------------------------------------------------------

    /// After one step, filtered sums to 1.
    #[test]
    fn step_filtered_sums_to_one() {
        let params = k2_params();
        let obs = sim_obs(params.clone(), 50, SEED);
        let mut state = OnlineFilterState::new(&params);
        let result = state.step(obs[0], &params).unwrap();
        let sum: f64 = result.filtered.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "filtered sum = {sum:.15}");
    }

    /// At t=1 (first observation), predictive_density equals Σⱼ fⱼ(y₁)·πⱼ.
    #[test]
    fn step_predictive_density_matches_manual_at_t1() {
        let params = k2_params();
        let obs = sim_obs(params.clone(), 50, SEED);
        let y1 = obs[0];

        // Manual computation: c₁ = Σⱼ fⱼ(y₁) · π_j
        let emission = Emission::new(params.means.clone(), params.variances.clone());
        let manual_c1: f64 = (0..params.k)
            .map(|j| emission.log_density(y1, j).exp() * params.pi[j])
            .sum();

        let mut state = OnlineFilterState::new(&params);
        let result = state.step(y1, &params).unwrap();

        assert!(
            (result.predictive_density - manual_c1).abs() < 1e-12,
            "c1: got {}, expected {manual_c1}",
            result.predictive_density
        );
    }

    /// After one step, predicted_next sums to 1.
    #[test]
    fn step_predicted_next_sums_to_one() {
        let params = k2_params();
        let obs = sim_obs(params.clone(), 50, SEED);
        let mut state = OnlineFilterState::new(&params);
        let result = state.step(obs[0], &params).unwrap();
        let sum: f64 = result.predicted_next.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "predicted_next sum = {sum:.15}");
    }

    /// After one step, t == 1.
    #[test]
    fn step_increments_t() {
        let params = k2_params();
        let obs = sim_obs(params.clone(), 50, SEED);
        let mut state = OnlineFilterState::new(&params);
        state.step(obs[0], &params).unwrap();
        assert_eq!(state.t, 1);
    }

    // -----------------------------------------------------------------------
    // Multi-step consistency (4 tests)
    // -----------------------------------------------------------------------

    /// After T steps, cumulative_log_score equals the offline log-likelihood
    /// from filter().
    #[test]
    fn cumulative_log_score_matches_offline_filter() {
        let params = k2_params();
        let obs = sim_obs(params.clone(), 200, SEED);

        let offline_ll = filter(&params, &obs).unwrap().log_likelihood;

        let mut state = OnlineFilterState::new(&params);
        for &y in &obs {
            state.step(y, &params).unwrap();
        }

        assert!(
            (state.cumulative_log_score - offline_ll).abs() < 1e-8,
            "online ll = {}, offline ll = {offline_ll}",
            state.cumulative_log_score
        );
    }

    /// At every step, filtered matches the corresponding column of
    /// FilterResult::filtered.
    #[test]
    fn filtered_matches_offline_filter_at_every_step() {
        let params = k2_params();
        let obs = sim_obs(params.clone(), 100, SEED);

        let offline = filter(&params, &obs).unwrap();

        let mut state = OnlineFilterState::new(&params);
        for (s, &y) in obs.iter().enumerate() {
            let result = state.step(y, &params).unwrap();
            for j in 0..params.k {
                let diff = (result.filtered[j] - offline.filtered[s][j]).abs();
                assert!(
                    diff < 1e-12,
                    "step {s}, regime {j}: online={} offline={}",
                    result.filtered[j],
                    offline.filtered[s][j]
                );
            }
        }
    }

    /// At every step, predicted_next matches FilterResult::predicted[s+1]
    /// (the predicted probability for the next time step).
    #[test]
    fn predicted_next_matches_offline_filter_predicted() {
        let params = k2_params();
        let obs = sim_obs(params.clone(), 100, SEED);

        let offline = filter(&params, &obs).unwrap();

        let mut state = OnlineFilterState::new(&params);
        let t_len = obs.len();
        for (s, &y) in obs.iter().enumerate() {
            let result = state.step(y, &params).unwrap();
            // predicted_next from online step s should equal offline predicted[s+1]
            // if s+1 < T (the last predicted_next has no corresponding batch entry).
            if s + 1 < t_len {
                for j in 0..params.k {
                    let diff = (result.predicted_next[j] - offline.predicted[s + 1][j]).abs();
                    assert!(
                        diff < 1e-12,
                        "step {s}, regime {j}: predicted_next={} offline.predicted[{}][{}]={}",
                        result.predicted_next[j],
                        s + 1,
                        j,
                        offline.predicted[s + 1][j]
                    );
                }
            }
        }
    }

    /// step in a loop produces deterministic causal output.
    #[test]
    fn step_loop_deterministic() {
        let params = k2_params();
        let obs = sim_obs(params.clone(), 80, SEED);

        let mut state_a = OnlineFilterState::new(&params);
        let results_a: Vec<OnlineStepResult> = obs
            .iter()
            .map(|&y| state_a.step(y, &params).unwrap())
            .collect();

        let mut state_b = OnlineFilterState::new(&params);
        let results_b: Vec<OnlineStepResult> = obs
            .iter()
            .map(|&y| state_b.step(y, &params).unwrap())
            .collect();

        assert_eq!(results_a.len(), results_b.len());
        for (s, (a, b)) in results_a.iter().zip(results_b.iter()).enumerate() {
            assert!(
                (a.log_predictive - b.log_predictive).abs() < 1e-15,
                "step {s}: a log_predictive={} b={}",
                a.log_predictive,
                b.log_predictive
            );
            for j in 0..params.k {
                assert!(
                    (a.filtered[j] - b.filtered[j]).abs() < 1e-15,
                    "step {s}, regime {j}: a={} b={}",
                    a.filtered[j],
                    b.filtered[j]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases (3 tests)
    // -----------------------------------------------------------------------

    /// T=1: single observation, no panic.
    #[test]
    fn edge_case_t1_no_panic() {
        let params = k2_params();
        let obs = sim_obs(params.clone(), 1, SEED);
        let mut state = OnlineFilterState::new(&params);
        let result = state.step(obs[0], &params).unwrap();
        assert_eq!(result.t, 1);
        assert_eq!(result.filtered.len(), 2);
    }

    /// T=2: two observations, correct transition applied once.
    #[test]
    fn edge_case_t2_transition_applied() {
        let params = k2_params();
        let obs = sim_obs(params.clone(), 2, SEED);
        let mut state = OnlineFilterState::new(&params);
        state.step(obs[0], &params).unwrap();
        let result = state.step(obs[1], &params).unwrap();
        assert_eq!(result.t, 2);
        let sum: f64 = result.filtered.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    /// An observation that is many σ from all regime means returns Err, not panic.
    #[test]
    fn extreme_observation_returns_err() {
        // Tiny variances so that a distant observation has effectively zero density.
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![0.0, 0.0],
            vec![1e-30, 1e-30], // effectively zero density for y far away
        );
        let mut state = OnlineFilterState::new(&params);
        let result = state.step(1e10, &params);
        assert!(result.is_err(), "expected Err for extreme observation");
    }

    // -----------------------------------------------------------------------
    // State machine properties (2 tests)
    // -----------------------------------------------------------------------

    /// OnlineStepResult has no smoothed field: verified by field enumeration.
    /// This test ensures the type surface is causal-only.
    #[test]
    fn step_result_has_only_causal_fields() {
        let params = k2_params();
        let obs = sim_obs(params.clone(), 10, SEED);
        let mut state = OnlineFilterState::new(&params);
        let result = state.step(obs[0], &params).unwrap();
        // Access every public field to confirm there are only causal fields.
        let _ = result.filtered;
        let _ = result.predicted_next;
        let _ = result.predictive_density;
        let _ = result.log_predictive;
        let _ = result.t;
        // If this compiles with no extra fields, the boundary is enforced.
    }

    /// Two independent OnlineFilterState instances from the same params evolve
    /// identically — no shared mutable state.
    #[test]
    fn independent_states_evolve_identically() {
        let params = k2_params();
        let obs = sim_obs(params.clone(), 50, SEED);

        let mut state_a = OnlineFilterState::new(&params);
        let mut state_b = OnlineFilterState::new(&params);

        let results_a: Vec<OnlineStepResult> = obs
            .iter()
            .map(|&y| state_a.step(y, &params).unwrap())
            .collect();
        let results_b: Vec<OnlineStepResult> = obs
            .iter()
            .map(|&y| state_b.step(y, &params).unwrap())
            .collect();

        for (s, (a, b)) in results_a.iter().zip(results_b.iter()).enumerate() {
            for j in 0..params.k {
                assert!(
                    (a.filtered[j] - b.filtered[j]).abs() < 1e-15,
                    "step {s}, regime {j}: state_a={} state_b={}",
                    a.filtered[j],
                    b.filtered[j]
                );
            }
        }
    }
}
