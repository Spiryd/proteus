/// EM estimator for the Gaussian Markov Switching Model.
///
/// # Architecture
///
/// This module is the only estimation layer in the project.  It builds on top
/// of the inference components (filter, smoother, pairwise) without modifying
/// them.  The data flow is strictly one-way:
///
/// ```text
/// ModelParams  →  filter()   →  FilterResult
///              →  smooth()   →  SmootherResult
///              →  pairwise() →  PairwiseResult
///              ↓
///         EStepResult  (bundle for the M-step)
///              ↓
///           m_step()   →  ModelParams (updated)
/// ```
///
/// # EM algorithm
///
/// At iteration m, given Θ^(m):
///
/// ## E-step
/// ```text
/// γₜ^(m)(j) = Pr(Sₜ=j | y_{1:T}; Θ^(m))          (smoothed marginals)
/// ξₜ^(m)(i,j) = Pr(S_{t-1}=i, Sₜ=j | y_{1:T}; Θ^(m))  (pairwise posteriors)
/// ```
///
/// ## M-step
/// ```text
/// π_j^(m+1) = γ₁^(m)(j)
///
/// p_ij^(m+1) = N_ij^(m) / M_i^(m)
///   where N_ij^(m) = Σ_t ξₜ^(m)(i,j)
///         M_i^(m)  = Σ_{t=1}^{T-1} γₜ^(m)(i)
///
/// μ_j^(m+1) = (Σ_t γₜ^(m)(j) yₜ) / W_j^(m)
///
/// (σ_j²)^(m+1) = (Σ_t γₜ^(m)(j) (yₜ - μ_j^(m+1))²) / W_j^(m)
///   where W_j^(m) = Σ_t γₜ^(m)(j)
/// ```
///
/// # Degenerate-case guards
///
/// - Transition row update skipped when M_i < WEIGHT_FLOOR (frozen at Θ^(m)).
/// - Mean / variance update skipped when W_j < WEIGHT_FLOOR.
/// - Variance floor VAR_FLOOR applied after each variance update.
use anyhow::Result;

use super::filter::filter;
use super::pairwise::pairwise;
use super::params::ModelParams;
use super::smoother::smooth;

/// Posterior weight below which a regime is considered degenerate for an
/// M-step update.  Updates that would divide by a smaller value are frozen.
const WEIGHT_FLOOR: f64 = 1e-10;

/// Minimum variance enforced after each variance M-step update to prevent
/// collapse to zero (variance degeneracy).
const VAR_FLOOR: f64 = 1e-6;

/// Tolerance for the monotonicity check: log-likelihood may decrease by at
/// most this amount across iterations before a warning is emitted.
const MONOTONE_TOL: f64 = 1e-8;

// ---------------------------------------------------------------------------
// E-step result
// ---------------------------------------------------------------------------

/// Bundle of all posterior quantities produced by one E-step pass.
///
/// This is the sole input to `m_step` (together with the observation sequence).
/// It carries no parameter-update logic.
#[derive(Debug, Clone)]
pub struct EStepResult {
    /// `smoothed[t][j]` = γ_{t+1}(j) = Pr(S_{t+1}=j | y_{1:T}).
    /// Shape: T × K.
    pub smoothed: Vec<Vec<f64>>,
    /// `expected_transitions[i][j]` = N_ij^exp = Σ_t ξₜ(i,j).
    /// Shape: K × K.
    pub expected_transitions: Vec<Vec<f64>>,
    /// Observed-data log-likelihood log L(Θ) = Σ_t log cₜ.
    pub log_likelihood: f64,
}

// ---------------------------------------------------------------------------
// EM configuration
// ---------------------------------------------------------------------------

/// Convergence configuration for the EM controller.
#[derive(Debug, Clone)]
pub struct EmConfig {
    /// Stop when |log L^(m+1) − log L^(m)| < tol.  Default: 1e-6.
    pub tol: f64,
    /// Hard iteration limit.  Default: 1000.
    pub max_iter: usize,
    /// Minimum allowed variance.  Applied after every variance update.
    /// Default: 1e-6.
    pub var_floor: f64,
}

impl Default for EmConfig {
    fn default() -> Self {
        Self {
            tol: 1e-6,
            max_iter: 1000,
            var_floor: VAR_FLOOR,
        }
    }
}

// ---------------------------------------------------------------------------
// Fitted-model result
// ---------------------------------------------------------------------------

/// The output of a completed EM estimation run.
#[derive(Debug, Clone)]
pub struct EmResult {
    /// Fitted model parameters Θ̂.
    pub params: ModelParams,
    /// Observed-data log-likelihood at Θ̂.
    pub log_likelihood: f64,
    /// Log-likelihood at each EM iteration (length = n_iter + 1, including
    /// the pre-loop baseline at iteration 0).
    pub ll_history: Vec<f64>,
    /// Number of EM iterations performed (E-step + M-step pairs).
    pub n_iter: usize,
    /// Whether the log-likelihood tolerance was met before hitting max_iter.
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// E-step
// ---------------------------------------------------------------------------

/// Run one E-step: filter → smooth → pairwise → bundle.
///
/// Returns an [`EStepResult`] containing the posterior quantities needed by
/// [`m_step`].
fn e_step(params: &ModelParams, obs: &[f64]) -> Result<EStepResult> {
    let fr = filter(params, obs)?;
    let sr = smooth(params, &fr)?;
    let pr = pairwise(params, &fr, &sr)?;

    Ok(EStepResult {
        smoothed: sr.smoothed,
        expected_transitions: pr.expected_transitions,
        log_likelihood: fr.log_likelihood,
    })
}

// ---------------------------------------------------------------------------
// M-step
// ---------------------------------------------------------------------------

/// Run one M-step: compute updated parameters from the E-step bundle.
///
/// # Parameters
/// - `e`        — E-step result from the previous E-step call.
/// - `obs`      — Original observation sequence (needed for mean/variance updates).
/// - `current`  — Current parameters (used to freeze degenerate rows/regimes).
/// - `var_floor` — Minimum allowed variance after the update.
///
/// # Returns
///
/// A new [`ModelParams`] with updated π, P, means, and variances.
fn m_step(
    e: &EStepResult,
    obs: &[f64],
    current: &ModelParams,
    var_floor: f64,
) -> Result<ModelParams> {
    let t_len = obs.len();
    let k = current.k;

    // ------------------------------------------------------------------
    // 1. Initial distribution: π_j^(m+1) = γ₁(j) = smoothed[0][j]
    // ------------------------------------------------------------------
    let new_pi = e.smoothed[0].clone();

    // ------------------------------------------------------------------
    // 2. Transition matrix
    //
    //   p_ij^(m+1) = N_ij / M_i
    //   M_i = Σ_{t=0}^{T-2} smoothed[t][i]   (0-based, dropping last step)
    // ------------------------------------------------------------------
    let mut new_transition = current.transition.clone();
    for i in 0..k {
        let m_i: f64 = (0..t_len - 1).map(|t| e.smoothed[t][i]).sum();
        if m_i < WEIGHT_FLOOR {
            // Regime i attracted negligible posterior weight as a "from" state.
            // Freeze this row to avoid division by near-zero.
            continue;
        }
        for j in 0..k {
            new_transition[i * k + j] = e.expected_transitions[i][j] / m_i;
        }
    }

    // ------------------------------------------------------------------
    // 3. Regime means
    //
    //   μ_j^(m+1) = (Σ_t γₜ(j) yₜ) / W_j
    //   W_j = Σ_t γₜ(j)
    // ------------------------------------------------------------------
    let mut new_means = current.means.clone();
    let mut weights = vec![0.0_f64; k]; // W_j, reused for variance update
    for j in 0..k {
        let w: f64 = e.smoothed.iter().map(|row| row[j]).sum();
        weights[j] = w;
        if w < WEIGHT_FLOOR {
            // Regime j attracted negligible weight; freeze its parameters.
            continue;
        }
        let weighted_sum: f64 = obs
            .iter()
            .zip(e.smoothed.iter())
            .map(|(&y, row)| row[j] * y)
            .sum();
        new_means[j] = weighted_sum / w;
    }

    // ------------------------------------------------------------------
    // 4. Regime variances
    //
    //   (σ_j²)^(m+1) = (Σ_t γₜ(j) (yₜ - μ_j^(m+1))²) / W_j
    //
    // Note: μ_j^(m+1) (already updated above) is used here, not μ_j^(m).
    // A variance floor is applied to prevent collapse.
    // ------------------------------------------------------------------
    let mut new_variances = current.variances.clone();
    for j in 0..k {
        let w = weights[j];
        if w < WEIGHT_FLOOR {
            continue;
        }
        let mu = new_means[j];
        let weighted_ss: f64 = obs
            .iter()
            .zip(e.smoothed.iter())
            .map(|(&y, row)| {
                let diff = y - mu;
                row[j] * diff * diff
            })
            .sum();
        new_variances[j] = (weighted_ss / w).max(var_floor);
    }

    let updated = ModelParams::new(
        new_pi,
        (0..k)
            .map(|i| new_transition[i * k..(i + 1) * k].to_vec())
            .collect(),
        new_means,
        new_variances,
    );
    updated.validate()?;
    Ok(updated)
}

// ---------------------------------------------------------------------------
// EM controller
// ---------------------------------------------------------------------------

/// Run EM estimation for the Gaussian Markov Switching Model.
///
/// Starting from `init_params`, alternates E-step and M-step until the
/// log-likelihood change is below `config.tol` or `config.max_iter` is
/// reached.
///
/// # Arguments
/// - `obs`         — Observation sequence y_{1:T}.
/// - `init_params` — Initial parameter guess Θ^(0).
/// - `config`      — Convergence configuration.
///
/// # Returns
///
/// An [`EmResult`] containing the fitted parameters, log-likelihood, full
/// iteration history, and convergence status.
///
/// # Errors
///
/// Returns an error if the filter, smoother, or pairwise pass fails at any
/// iteration, or if the M-step produces parameters that fail validation.
pub fn fit_em(obs: &[f64], init_params: ModelParams, config: &EmConfig) -> Result<EmResult> {
    if obs.is_empty() {
        anyhow::bail!("fit_em: observation sequence is empty");
    }
    init_params.validate()?;

    let mut params = init_params;
    let mut ll_history: Vec<f64> = Vec::with_capacity(config.max_iter + 1);
    let mut converged = false;

    // ------------------------------------------------------------------
    // Baseline E-step at m=0 to record log L(Θ^(0)) and verify the
    // filter runs cleanly on the initial parameters.
    // ------------------------------------------------------------------
    let mut e = e_step(&params, obs)?;
    ll_history.push(e.log_likelihood);

    for _iter in 0..config.max_iter {
        // M-step: update parameters from the current E-step bundle.
        params = m_step(&e, obs, &params, config.var_floor)?;

        // E-step under the new parameters.
        e = e_step(&params, obs)?;
        let ll_new = e.log_likelihood;

        // Monotonicity check.
        let ll_prev = *ll_history.last().unwrap();
        if ll_new < ll_prev - MONOTONE_TOL {
            // EM guarantees nondecreasing likelihood; a decrease beyond the
            // tolerance indicates a numerical issue.
            eprintln!(
                "fit_em: log-likelihood decreased from {ll_prev:.8} to {ll_new:.8} \
                 (drop = {:.2e}); possible numerical issue",
                ll_prev - ll_new
            );
        }

        ll_history.push(ll_new);

        // Convergence check.
        if (ll_new - ll_prev).abs() < config.tol {
            converged = true;
            break;
        }
    }

    let n_iter = ll_history.len() - 1; // excludes the baseline
    let log_likelihood = *ll_history.last().unwrap();

    Ok(EmResult {
        params,
        log_likelihood,
        ll_history,
        n_iter,
        converged,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::{ModelParams, simulate};
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    const SEED: u64 = 42;

    fn default_k2_params() -> ModelParams {
        ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-3.0, 3.0],
            vec![1.0, 1.0],
        )
    }

    fn tight_config() -> EmConfig {
        EmConfig {
            tol: 1e-8,
            max_iter: 500,
            ..Default::default()
        }
    }

    // -----------------------------------------------------------------------
    // Structural / M-step correctness tests
    // -----------------------------------------------------------------------

    /// After one EM iteration from known parameters, π^(1) must equal
    /// smoothed[0] from the baseline E-step exactly.
    #[test]
    fn pi_update_equals_smoothed_at_t0() {
        let params = default_k2_params();
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 200, &mut rng).unwrap();

        let config = EmConfig {
            max_iter: 1,
            ..Default::default()
        };
        let result = fit_em(&sim.observations, params.clone(), &config).unwrap();

        // Baseline E-step smoothed[0] should match the π used for the update.
        let e0 = e_step(&params, &sim.observations).unwrap();
        for j in 0..2 {
            let diff = (result.params.pi[j] - e0.smoothed[0][j]).abs();
            assert!(
                diff < 1e-12,
                "π[{j}]: got {:.14}, expected {:.14}",
                result.params.pi[j],
                e0.smoothed[0][j]
            );
        }
    }

    /// After any number of M-step updates, each row of P must sum to 1.
    #[test]
    fn transition_rows_sum_to_one() {
        let params = default_k2_params();
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 500, &mut rng).unwrap();

        let result = fit_em(&sim.observations, params, &tight_config()).unwrap();

        for i in 0..2 {
            let row_sum: f64 = result.params.transition_row(i).iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-12,
                "P row {i} sums to {row_sum:.15}"
            );
        }
    }

    /// All variances must remain strictly positive after EM.
    #[test]
    fn variances_remain_positive() {
        let params = default_k2_params();
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 500, &mut rng).unwrap();

        let result = fit_em(&sim.observations, params, &tight_config()).unwrap();

        for j in 0..2 {
            assert!(
                result.params.variances[j] > 0.0,
                "variance[{j}] = {} is not positive",
                result.params.variances[j]
            );
        }
    }

    /// EmResult fields are internally consistent.
    #[test]
    fn result_fields_consistent() {
        let params = default_k2_params();
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 300, &mut rng).unwrap();

        let result = fit_em(&sim.observations, params, &tight_config()).unwrap();

        // ll_history has n_iter+1 entries (baseline + one per iteration).
        assert_eq!(result.ll_history.len(), result.n_iter + 1);
        // Last entry in ll_history matches reported log_likelihood.
        assert_eq!(*result.ll_history.last().unwrap(), result.log_likelihood);
        // log_likelihood must be finite.
        assert!(
            result.log_likelihood.is_finite(),
            "log_likelihood is not finite: {}",
            result.log_likelihood
        );
    }

    // -----------------------------------------------------------------------
    // EM ascent test (most important correctness property)
    // -----------------------------------------------------------------------

    /// Log-likelihood must be nondecreasing across all EM iterations.
    ///
    /// This is the fundamental EM guarantee.  A violation indicates a bug in
    /// the E-step, M-step, or their interface.
    #[test]
    fn log_likelihood_nondecreasing() {
        let params = default_k2_params();
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 1_000, &mut rng).unwrap();

        let result = fit_em(&sim.observations, params, &tight_config()).unwrap();

        for m in 1..result.ll_history.len() {
            assert!(
                result.ll_history[m] >= result.ll_history[m - 1] - MONOTONE_TOL,
                "EM ascent violated at step {m}: \
                 ll[{m}]={:.10} < ll[{}]={:.10}",
                result.ll_history[m],
                m - 1,
                result.ll_history[m - 1]
            );
        }
    }

    /// Ascent holds for K=3.
    #[test]
    fn log_likelihood_nondecreasing_k3() {
        let params = ModelParams::new(
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            vec![
                vec![0.8, 0.1, 0.1],
                vec![0.1, 0.8, 0.1],
                vec![0.1, 0.1, 0.8],
            ],
            vec![-5.0, 0.0, 5.0],
            vec![1.0, 1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 1_500, &mut rng).unwrap();

        let result = fit_em(&sim.observations, params, &tight_config()).unwrap();

        for m in 1..result.ll_history.len() {
            assert!(
                result.ll_history[m] >= result.ll_history[m - 1] - MONOTONE_TOL,
                "K=3 EM ascent violated at step {m}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Behavioral / recovery tests
    // -----------------------------------------------------------------------

    /// On strongly separated simulated data, EM should converge (hit tol
    /// before max_iter) and the fitted log-likelihood must exceed the
    /// log-likelihood at the initial parameters.
    #[test]
    fn converges_and_improves_likelihood() {
        // True params: well-separated means, high persistence.
        let true_params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.95, 0.05], vec![0.05, 0.95]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(true_params.clone(), 1_000, &mut rng).unwrap();

        // Init: same structure but slightly perturbed means and variances.
        let init = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.8, 0.2], vec![0.2, 0.8]],
            vec![-3.0, 3.0],
            vec![2.0, 2.0],
        );

        let result = fit_em(&sim.observations, init.clone(), &tight_config()).unwrap();

        assert!(result.converged, "EM did not converge");
        let ll_init = e_step(&init, &sim.observations).unwrap().log_likelihood;
        assert!(
            result.log_likelihood > ll_init,
            "fitted ll={:.4} should exceed initial ll={:.4}",
            result.log_likelihood,
            ll_init
        );
    }

    /// Fitted means should be closer to the true means than the initial means
    /// on well-separated simulated data (up to label-switching: compare
    /// sorted means).
    #[test]
    fn mean_recovery_on_separated_data() {
        let true_params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.95, 0.05], vec![0.05, 0.95]],
            vec![-6.0, 6.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(true_params.clone(), 2_000, &mut rng).unwrap();

        let init = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.8, 0.2], vec![0.2, 0.8]],
            vec![-3.0, 3.0],
            vec![2.0, 2.0],
        );
        let result = fit_em(&sim.observations, init, &tight_config()).unwrap();

        // Sort fitted and true means; check proximity (tolerance: 1.0).
        let mut fitted_means = result.params.means.clone();
        let mut true_means = true_params.means.clone();
        fitted_means.sort_by(|a, b| a.partial_cmp(b).unwrap());
        true_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for j in 0..2 {
            let err = (fitted_means[j] - true_means[j]).abs();
            assert!(
                err < 1.0,
                "regime {j}: fitted mean {:.3} is far from true mean {:.3} (err={err:.3})",
                fitted_means[j],
                true_means[j]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Edge-case / robustness tests
    // -----------------------------------------------------------------------

    /// T=2 (minimal non-trivial sample): EM must not panic.
    #[test]
    fn minimal_sample_no_panic() {
        let params = default_k2_params();
        let config = EmConfig {
            max_iter: 5,
            ..Default::default()
        };
        let result = fit_em(&[-3.0, 3.0], params, &config);
        assert!(result.is_ok(), "fit_em panicked or errored on T=2");
    }

    /// Empty observation sequence returns an error, not a panic.
    #[test]
    fn empty_obs_returns_error() {
        let params = default_k2_params();
        let result = fit_em(&[], params, &EmConfig::default());
        assert!(result.is_err(), "expected error for empty obs");
    }

    /// max_iter=0: returns the baseline E-step result, n_iter=0, converged=false.
    #[test]
    fn zero_iterations_returns_baseline() {
        let params = default_k2_params();
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 100, &mut rng).unwrap();

        let config = EmConfig {
            max_iter: 0,
            ..Default::default()
        };
        let result = fit_em(&sim.observations, params.clone(), &config).unwrap();

        assert_eq!(result.n_iter, 0);
        assert!(!result.converged);
        assert_eq!(result.ll_history.len(), 1);
        // Parameters must be unchanged (no M-step was run).
        assert_eq!(result.params.means, params.means);
    }

    /// Multiple seeds: EM ascent invariant holds across diverse random samples.
    #[test]
    fn multi_seed_ascent() {
        let params = default_k2_params();
        let config = EmConfig {
            tol: 1e-6,
            max_iter: 200,
            ..Default::default()
        };
        for seed in 0_u64..8 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let sim = simulate(params.clone(), 500, &mut rng).unwrap();
            let result = fit_em(&sim.observations, params.clone(), &config).unwrap();
            for m in 1..result.ll_history.len() {
                assert!(
                    result.ll_history[m] >= result.ll_history[m - 1] - MONOTONE_TOL,
                    "seed {seed}: EM ascent violated at step {m}"
                );
            }
        }
    }
}
