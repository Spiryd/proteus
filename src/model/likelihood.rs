#![allow(dead_code)]
/// Observed-data log-likelihood for the Gaussian Markov Switching Model.
///
/// # Mathematical specification
///
/// The **observed-data likelihood** integrates out the hidden regime path:
///
/// ```text
/// L(Θ) = f(y₁,…,y_T; Θ)
///       = ∏ₜ cₜ
/// ```
///
/// where the **one-step predictive density** at each time t is
///
/// ```text
/// cₜ = f(yₜ | y_{1:t-1}; Θ)
///    = Σⱼ f_j(yₜ) · α_{t|t-1}(j)
/// ```
///
/// and α_{t|t-1}(j) = Pr(Sₜ=j | y_{1:t-1}) are the predicted regime
/// probabilities computed by the Hamilton forward filter.
///
/// The **log-likelihood** is
///
/// ```text
/// log L(Θ) = Σₜ log cₜ
/// ```
///
/// # Dual role of cₜ
///
/// The predictive density cₜ is simultaneously:
/// 1. **Likelihood contribution** — one factor of the observed-data likelihood.
/// 2. **Normalization constant** — the denominator in the filtered posterior
///    α_{t|t}(j) = f_j(yₜ) · α_{t|t-1}(j) / cₜ.
///
/// These two roles cannot be separated: filtering and likelihood evaluation
/// are the same computation.
///
/// # Architecture
///
/// This module is a **thin interface layer** over `filter`.  It holds no
/// statistical logic of its own.  Its purpose is to expose the scalar
/// `log L(Θ)` as a first-class function so that optimization code can call
/// a single, clear entry point without instantiating a `FilterResult`.
///
/// Callers that need the full time history of predicted/filtered probabilities
/// should call `filter()` directly.  Callers that only need to evaluate or
/// optimize the likelihood should call `log_likelihood()`.
use anyhow::Result;

use super::filter::filter;
use super::params::ModelParams;

/// Compute the observed-data log-likelihood log L(Θ) = Σₜ log cₜ.
///
/// This is the primary objective for maximum likelihood parameter estimation.
/// It calls the forward filter once and returns only the scalar total.
///
/// # Errors
///
/// Returns an error if `filter()` fails (invalid parameters, empty observations,
/// or zero predictive density at any time step).
pub fn log_likelihood(params: &ModelParams, obs: &[f64]) -> Result<f64> {
    Ok(filter(params, obs)?.log_likelihood)
}

/// Compute the per-time log-predictive contributions log cₜ for t = 1,…,T.
///
/// Returns a `Vec<f64>` of length T where `contributions[t]` = log cₜ₊₁.
///
/// This is the decomposition of the total log-likelihood:
///
/// ```text
/// log L(Θ) = Σₜ contributions[t]
/// ```
///
/// Useful for:
/// - per-time fit diagnostics,
/// - identifying time periods where the model is most surprised,
/// - verifying the additive decomposition identity.
///
/// # Errors
///
/// Returns an error under the same conditions as `log_likelihood`.
pub fn log_likelihood_contributions(params: &ModelParams, obs: &[f64]) -> Result<Vec<f64>> {
    Ok(filter(params, obs)?.log_predictive)
}

// ---------------------------------------------------------------------------
// Tests — statistical properties of the likelihood layer
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::simulate::simulate;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    const SEED: u64 = 42;

    fn separated_params() -> ModelParams {
        // Well-separated regimes: μ₀=−10, μ₁=10, σ²=1, persistent transitions.
        ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.99, 0.01], vec![0.01, 0.99]],
            vec![-10.0, 10.0],
            vec![1.0, 1.0],
        )
    }

    fn misfit_params() -> ModelParams {
        // Means near zero: assigns low density to observations at ±10.
        // This is a genuinely worse model, not just a label permutation.
        ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.99, 0.01], vec![0.01, 0.99]],
            vec![-1.0, 1.0], // wrong means
            vec![1.0, 1.0],
        )
    }

    // -----------------------------------------------------------------------
    // Test 1 — Decomposition identity: LL = sum of per-time contributions
    // -----------------------------------------------------------------------
    #[test]
    fn test_decomposition_identity() {
        let params = separated_params();
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 500, &mut rng).unwrap();

        let total = log_likelihood(&params, &sim.observations).unwrap();
        let contribs = log_likelihood_contributions(&params, &sim.observations).unwrap();

        let summed: f64 = contribs.iter().sum();
        assert!(
            (total - summed).abs() < 1e-10,
            "log_likelihood={total:.12} ≠ sum(contributions)={summed:.12}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 2 — Contributions vector has length T
    // -----------------------------------------------------------------------
    #[test]
    fn test_contributions_length() {
        let params = separated_params();
        let obs = vec![-10.0, 10.0, -9.5, 9.8, -10.2];
        let contribs = log_likelihood_contributions(&params, &obs).unwrap();
        assert_eq!(contribs.len(), obs.len());
    }

    // -----------------------------------------------------------------------
    // Test 3 — All log-predictive contributions are finite
    // -----------------------------------------------------------------------
    #[test]
    fn test_contributions_finite() {
        let params = separated_params();
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 1_000, &mut rng).unwrap();
        let contribs = log_likelihood_contributions(&params, &sim.observations).unwrap();

        for (t, &lc) in contribs.iter().enumerate() {
            assert!(lc.is_finite(), "contribution[{t}] = {lc} is not finite");
        }
    }

    // -----------------------------------------------------------------------
    // Test 4 — T=1 exact likelihood
    //
    // π=(0.5,0.5), μ=(0,0), σ²=(1,1), y=0:
    //   c₁ = 0.5·N(0;0,1) + 0.5·N(0;0,1) = N(0;0,1) = 1/√(2π)
    //   log L = log(1/√(2π)) = -½ ln(2π)
    // -----------------------------------------------------------------------
    #[test]
    fn test_t1_exact_likelihood() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![0.0, 0.0],
            vec![1.0, 1.0],
        );
        let expected = -(2.0 * std::f64::consts::PI).ln() / 2.0;
        let got = log_likelihood(&params, &[0.0]).unwrap();

        assert!(
            (got - expected).abs() < 1e-12,
            "log_likelihood={got:.12}, expected={expected:.12}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 5 — True parameters score higher than misfit parameters on own data
    //
    // Data generated under `separated_params`.
    // Evaluating under `misfit_params` (means swapped) should give a lower LL
    // on average because the misfit model assigns low density to each observation.
    // -----------------------------------------------------------------------
    #[test]
    fn test_true_params_score_higher_than_misfit() {
        let true_params = separated_params();
        let bad_params = misfit_params();

        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(true_params.clone(), 2_000, &mut rng).unwrap();

        let ll_true = log_likelihood(&true_params, &sim.observations).unwrap();
        let ll_bad = log_likelihood(&bad_params, &sim.observations).unwrap();

        assert!(
            ll_true > ll_bad,
            "true LL={ll_true:.4} should exceed misfit LL={ll_bad:.4}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6 — LL improves when emission means are moved toward observed data
    //
    // For a single observation y=5.0, moving μⱼ from 0.0 toward 5.0 should
    // increase the likelihood.
    // -----------------------------------------------------------------------
    #[test]
    fn test_ll_improves_as_mean_approaches_observation() {
        let means = [0.0_f64, 2.5, 4.9];
        let obs = [5.0_f64];
        let mut prev_ll = f64::NEG_INFINITY;

        for &mu in &means {
            let params = ModelParams::new(
                vec![1.0, 0.0], // all weight on regime 0
                vec![vec![1.0, 0.0], vec![0.0, 1.0]],
                vec![mu, 0.0],
                vec![1.0, 1.0],
            );
            let ll = log_likelihood(&params, &obs).unwrap();
            assert!(
                ll > prev_ll,
                "LL should increase as μ→y: μ={mu}, ll={ll:.6}, prev={prev_ll:.6}"
            );
            prev_ll = ll;
        }
    }

    // -----------------------------------------------------------------------
    // Test 7 — LL is invariant to regime label permutation when π and P are
    //           symmetric and observations are unlabeled
    //
    // Swapping all regime indices (π, P rows/cols, μ, σ²) on a symmetric model
    // with independent observations must give the same LL.
    // -----------------------------------------------------------------------
    #[test]
    fn test_ll_invariant_to_symmetric_permutation() {
        // Symmetric model: π=(0.5,0.5), P diagonal=0.9
        let params_a = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-3.0, 3.0],
            vec![1.0, 2.0],
        );
        // Same model, regime labels swapped
        let params_b = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![3.0, -3.0], // μ swapped
            vec![2.0, 1.0],  // σ² swapped
        );

        let obs = vec![2.8, -2.9, 3.1, -3.0, 2.7, -2.8];
        let ll_a = log_likelihood(&params_a, &obs).unwrap();
        let ll_b = log_likelihood(&params_b, &obs).unwrap();

        assert!(
            (ll_a - ll_b).abs() < 1e-10,
            "symmetric permutation changed LL: {ll_a:.10} vs {ll_b:.10}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 8 — LL depends on transition matrix (not only emission parameters)
    //
    // Same emission params, different transition matrices: the LL should differ
    // when the observation sequence has persistent structure.
    // -----------------------------------------------------------------------
    #[test]
    fn test_ll_sensitive_to_transition_matrix() {
        // Persistent transitions — should fit persistent data better
        let persistent = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.99, 0.01], vec![0.01, 0.99]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        // Rapidly switching transitions — should fit persistent data worse
        let switching = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.3, 0.7], vec![0.7, 0.3]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );

        // Generate 1000 observations from the persistent model
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(persistent.clone(), 1_000, &mut rng).unwrap();

        let ll_persistent = log_likelihood(&persistent, &sim.observations).unwrap();
        let ll_switching = log_likelihood(&switching, &sim.observations).unwrap();

        assert!(
            ll_persistent > ll_switching,
            "persistent LL={ll_persistent:.4} should exceed switching LL={ll_switching:.4} \
             on data generated from the persistent model"
        );
    }

    // -----------------------------------------------------------------------
    // Test 9 — LL depends on initial distribution
    //
    // With T=1, the initial distribution is the only source of regime weights,
    // so changing π must change the LL when emission params differ across regimes.
    // -----------------------------------------------------------------------
    #[test]
    fn test_ll_sensitive_to_initial_distribution() {
        // π strongly favors regime 0 (mean=0); y=0 fits regime 0 well
        let params_favor_0 = ModelParams::new(
            vec![0.99, 0.01],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![0.0, 100.0],
            vec![1.0, 1.0],
        );
        // π strongly favors regime 1 (mean=100); y=0 fits regime 1 poorly
        let params_favor_1 = ModelParams::new(
            vec![0.01, 0.99],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![0.0, 100.0],
            vec![1.0, 1.0],
        );

        let obs = [0.0_f64];
        let ll_0 = log_likelihood(&params_favor_0, &obs).unwrap();
        let ll_1 = log_likelihood(&params_favor_1, &obs).unwrap();

        assert!(
            ll_0 > ll_1,
            "π favoring the well-fitting regime should give higher LL: {ll_0:.6} vs {ll_1:.6}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 10 — Error propagation: empty obs and invalid params return Err
    // -----------------------------------------------------------------------
    #[test]
    fn test_errors_propagated_from_filter() {
        let params = separated_params();
        assert!(
            log_likelihood(&params, &[]).is_err(),
            "empty obs should error"
        );

        let bad_params = ModelParams::new(
            vec![0.4, 0.4], // pi sums to 0.8
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        );
        assert!(
            log_likelihood(&bad_params, &[0.0]).is_err(),
            "invalid params should error"
        );
    }
}
