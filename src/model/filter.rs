/// Forward filter (Hamilton filter) for the Gaussian Markov Switching Model.
///
/// # Mathematical specification
///
/// The filter runs a sequential Bayesian recursion over T observations.
///
/// ## Notation
/// - `α_{t|t-1}(j)` = `Pr(Sₜ=j | y_{1:t-1})` — predicted probability for regime j at time t
/// - `α_{t|t}(j)`   = `Pr(Sₜ=j | y_{1:t})`   — filtered probability for regime j at time t
/// - `f_j(yₜ)`      = regime-conditional emission density (from the `Emission` model)
/// - `cₜ`           = `f(yₜ | y_{1:t-1})`      — predictive (marginal) density of yₜ
///
/// ## Recursion
///
/// For t = 1:
/// ```text
/// α_{1|0}(j) = π_j
/// c₁         = Σⱼ f_j(y₁) · α_{1|0}(j)
/// α_{1|1}(j) = f_j(y₁) · π_j / c₁
/// ```
///
/// For t = 2, …, T:
/// ```text
/// α_{t|t-1}(j) = Σᵢ p_{ij} · α_{t-1|t-1}(i)      (prediction)
/// cₜ           = Σⱼ f_j(yₜ) · α_{t|t-1}(j)         (predictive density)
/// α_{t|t}(j)   = f_j(yₜ) · α_{t|t-1}(j) / cₜ      (Bayes update)
/// ```
///
/// Log-likelihood:
/// ```text
/// log L(Θ) = Σₜ log cₜ
/// ```
///
/// # Numerical approach
///
/// Emission densities can be extremely small when observations are many standard
/// deviations from a regime mean.  To prevent underflow, all per-step products
/// `f_j(yₜ) · α_{t|t-1}(j)` are computed in log-space and combined via the
/// numerically stable log-sum-exp identity before exponentiating only once to
/// form the normalized filtered probabilities.
///
/// # Architecture
///
/// `filter` is intentionally separated from:
/// - `Emission`    — provides  `log_density(y, j)` without knowing about transitions,
/// - `ModelParams` — provides  `transition_row(i)` and `pi` without knowing about observations.
///
/// The filter mediates between these two layers.  No emission formula appears
/// here; all calls go through `Emission::log_density`.
use anyhow::Result;

use super::emission::Emission;
use super::params::ModelParams;

/// The complete output of one forward-filter pass.
///
/// All time-indexed fields use 0-based indexing:
/// - index `0` corresponds to t = 1 (first observation),
/// - index `T-1` corresponds to t = T (last observation).
///
/// Both `predicted` and `filtered` are retained because the backward
/// smoother (Phase 5) requires access to `predicted[t]` at every t.
/// Do not drop either array before that phase.
#[derive(Debug, Clone)]
pub struct FilterResult {
    /// T — number of observations processed.
    pub t: usize,
    /// K — number of regimes.
    pub k: usize,
    /// `predicted[t][j]` = α_{t+1|t}(j) = Pr(S_{t+1}=j | y_{1:t+1}).
    ///
    /// At index 0 this equals `π` (no observations observed yet).
    /// Shape: T × K.
    pub predicted: Vec<Vec<f64>>,
    /// `filtered[t][j]` = α_{t+1|t+1}(j) = Pr(S_{t+1}=j | y_{1:t+1}).
    ///
    /// This is the primary inference output of the filter.
    /// Shape: T × K.
    pub filtered: Vec<Vec<f64>>,
    /// `log_predictive[t]` = log cₜ₊₁ = log f(y_{t+1} | y_{1:t}).
    ///
    /// Shape: T.
    pub log_predictive: Vec<f64>,
    /// Total sample log-likelihood: log L(Θ) = Σₜ log cₜ.
    ///
    /// This is the quantity that EM and model selection will optimize.
    pub log_likelihood: f64,
}

/// Run the Hamilton forward filter on observation sequence `obs`.
///
/// Validates `params`, constructs the emission layer, then runs the
/// predict–update recursion for each observation in order.
///
/// # Errors
///
/// Returns an error if:
/// - `obs` is empty,
/// - `params.validate()` fails,
/// - any predictive density cₜ is zero (the observation is incompatible with
///   every regime — indicates a model specification problem).
pub fn filter(params: &ModelParams, obs: &[f64]) -> Result<FilterResult> {
    if obs.is_empty() {
        anyhow::bail!("filter: observation sequence must not be empty");
    }
    params.validate()?;

    let k = params.k;
    let t_len = obs.len();

    // Build the emission layer from the already-validated params.
    // Emission is constructed once and reused across all time steps.
    let emission = Emission::new(params.means.clone(), params.variances.clone());

    let mut predicted = Vec::with_capacity(t_len);
    let mut filtered: Vec<Vec<f64>> = Vec::with_capacity(t_len);
    let mut log_predictive = Vec::with_capacity(t_len);
    let mut log_likelihood = 0.0_f64;

    // ------------------------------------------------------------------
    // t = 1  (index 0): predicted = π  (no prior observations)
    // ------------------------------------------------------------------
    let pred_0 = params.pi.clone();
    let (filt_0, log_c_0) = bayes_update(&emission, obs[0], &pred_0, k)?;
    predicted.push(pred_0);
    filtered.push(filt_0);
    log_predictive.push(log_c_0);
    log_likelihood += log_c_0;

    // ------------------------------------------------------------------
    // t = 2, …, T  (indices 1..t_len)
    // ------------------------------------------------------------------
    for idx in 1..t_len {
        let filt_prev = &filtered[idx - 1];

        // Prediction step: α_{t|t-1}(j) = Σᵢ p_{ij} · α_{t-1|t-1}(i)
        let pred_t = predict(params, filt_prev, k);

        // Bayes update: normalize emission-weighted predicted probs
        let (filt_t, log_c_t) = bayes_update(&emission, obs[idx], &pred_t, k)?;

        predicted.push(pred_t);
        filtered.push(filt_t);
        log_predictive.push(log_c_t);
        log_likelihood += log_c_t;
    }

    Ok(FilterResult {
        t: t_len,
        k,
        predicted,
        filtered,
        log_predictive,
        log_likelihood,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Prediction step: α_{t|t-1}(j) = Σᵢ p_{ij} · α_{t-1|t-1}(i).
///
/// `P` is stored row-major: `transition_row(i)[j]` = p_{ij}
/// (probability of transitioning FROM regime i TO regime j).
///
/// Equivalently this is the matrix product  α_{prev} · P  treating α as a row vector.
#[inline]
fn predict(params: &ModelParams, filtered_prev: &[f64], k: usize) -> Vec<f64> {
    let mut pred = vec![0.0_f64; k];
    for (j, p) in pred.iter_mut().enumerate() {
        *p = (0..k)
            .map(|i| params.transition_row(i)[j] * filtered_prev[i])
            .sum();
    }
    pred
}

/// Bayes update step: combine predicted probs with emission densities.
///
/// Returns `(filtered, log_c)` where:
/// - `filtered[j]` = α_{t|t}(j) = f_j(y) · α_{t|t-1}(j) / cₜ
/// - `log_c`       = log cₜ = log Σⱼ f_j(y) · α_{t|t-1}(j)
///
/// Uses log-sum-exp to avoid floating-point underflow when emission densities
/// are very small (e.g. observation far in the tail of a narrow Gaussian).
#[inline]
fn bayes_update(
    emission: &Emission,
    y: f64,
    predicted: &[f64],
    k: usize,
) -> Result<(Vec<f64>, f64)> {
    // Unnormalized log-weights: log(f_j(y)) + log(α_{t|t-1}(j))
    // If predicted[j] == 0, the regime has zero prior weight → −∞ in log-space.
    let log_unnorm: Vec<f64> = (0..k)
        .map(|j| {
            if predicted[j] > 0.0 {
                emission.log_density(y, j) + predicted[j].ln()
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect();

    // log cₜ = log Σⱼ exp(log_unnorm[j])
    let log_c = log_sum_exp(&log_unnorm);

    if !log_c.is_finite() {
        anyhow::bail!(
            "filter: predictive density is zero — observation y={y:.6} \
             is incompatible with all {k} regimes. \
             Check that emission means and variances cover the observed range."
        );
    }

    // Normalized filtered probabilities: exp(log_unnorm[j] - log_c)
    let filtered: Vec<f64> = log_unnorm.iter().map(|&lu| (lu - log_c).exp()).collect();

    Ok((filtered, log_c))
}

/// Numerically stable log-sum-exp.
///
/// Computes `log Σᵢ exp(xᵢ)` as `max(x) + log Σᵢ exp(xᵢ − max(x))`
/// to avoid overflow/underflow.
fn log_sum_exp(log_vals: &[f64]) -> f64 {
    let max = log_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return f64::NEG_INFINITY;
    }
    max + log_vals.iter().map(|&v| (v - max).exp()).sum::<f64>().ln()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::simulate::simulate;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    const TOL: f64 = 1e-10;
    const SEED: u64 = 42;

    /// Two-regime model with strongly separated means and persistent transitions.
    fn separated_params() -> ModelParams {
        ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.99, 0.01], vec![0.01, 0.99]],
            vec![-10.0, 10.0],
            vec![1.0, 1.0],
        )
    }

    /// Two-regime model with equal means and moderate switching (sanity base case).
    fn symmetric_params() -> ModelParams {
        ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![0.0, 0.0],
            vec![1.0, 1.0],
        )
    }

    // -----------------------------------------------------------------------
    // Test 1 — Output length invariants
    // -----------------------------------------------------------------------
    #[test]
    fn test_output_lengths() {
        let params = separated_params();
        let obs = vec![1.0, 2.0, 3.0, -1.0, -2.0];
        let result = filter(&params, &obs).unwrap();

        assert_eq!(result.t, 5);
        assert_eq!(result.k, 2);
        assert_eq!(result.predicted.len(), 5);
        assert_eq!(result.filtered.len(), 5);
        assert_eq!(result.log_predictive.len(), 5);
        for t in 0..5 {
            assert_eq!(result.predicted[t].len(), 2, "predicted[{t}] wrong length");
            assert_eq!(result.filtered[t].len(), 2, "filtered[{t}] wrong length");
        }
    }

    // -----------------------------------------------------------------------
    // Test 2 — At t=1, predicted[0] equals π exactly
    // -----------------------------------------------------------------------
    #[test]
    fn test_predicted_t1_equals_pi() {
        let params = separated_params();
        let result = filter(&params, &[0.0]).unwrap();

        for j in 0..2 {
            assert!(
                (result.predicted[0][j] - params.pi[j]).abs() < TOL,
                "predicted[0][{j}] = {}, expected π[{j}] = {}",
                result.predicted[0][j],
                params.pi[j]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 3 — Predicted probabilities sum to 1 at every t
    // -----------------------------------------------------------------------
    #[test]
    fn test_predicted_sum_to_1() {
        let params = separated_params();
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 200, &mut rng).unwrap();
        let result = filter(&params, &sim.observations).unwrap();

        for t in 0..result.t {
            let s: f64 = result.predicted[t].iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-12,
                "predicted[{t}] sums to {s:.15}, expected 1"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 4 — Filtered probabilities sum to 1 at every t
    // -----------------------------------------------------------------------
    #[test]
    fn test_filtered_sum_to_1() {
        let params = separated_params();
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 200, &mut rng).unwrap();
        let result = filter(&params, &sim.observations).unwrap();

        for t in 0..result.t {
            let s: f64 = result.filtered[t].iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-12,
                "filtered[{t}] sums to {s:.15}, expected 1"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 5 — All probabilities lie in [0, 1]
    // -----------------------------------------------------------------------
    #[test]
    fn test_probabilities_in_unit_interval() {
        let params = separated_params();
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 200, &mut rng).unwrap();
        let result = filter(&params, &sim.observations).unwrap();

        for t in 0..result.t {
            for j in 0..result.k {
                let p = result.predicted[t][j];
                let f = result.filtered[t][j];
                assert!(
                    p >= 0.0 && p <= 1.0 + 1e-12,
                    "predicted[{t}][{j}]={p} out of [0,1]"
                );
                assert!(
                    f >= 0.0 && f <= 1.0 + 1e-12,
                    "filtered[{t}][{j}]={f} out of [0,1]"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 6 — log_likelihood equals sum of log_predictive
    // -----------------------------------------------------------------------
    #[test]
    fn test_log_likelihood_equals_sum_of_log_predictive() {
        let params = separated_params();
        let obs = vec![-9.8, 10.2, 9.7, -10.1, -9.5];
        let result = filter(&params, &obs).unwrap();

        let expected: f64 = result.log_predictive.iter().sum();
        assert!(
            (result.log_likelihood - expected).abs() < TOL,
            "log_likelihood={:.12}, sum(log_predictive)={:.12}",
            result.log_likelihood,
            expected
        );
    }

    // -----------------------------------------------------------------------
    // Test 7 — All log_predictive values are finite
    // -----------------------------------------------------------------------
    #[test]
    fn test_log_predictive_is_finite() {
        let params = separated_params();
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 500, &mut rng).unwrap();
        let result = filter(&params, &sim.observations).unwrap();

        for (t, &lp) in result.log_predictive.iter().enumerate() {
            assert!(lp.is_finite(), "log_predictive[{t}] = {lp} is not finite");
        }
    }

    // -----------------------------------------------------------------------
    // Test 8 — T=1 exact Bayes posterior
    //
    // π = (0.3, 0.7),  μ = (0.0, 1.0),  σ² = (1.0, 1.0),  y = 0.0
    //
    // f_0(0) = N(0;0,1),  f_1(0) = N(0;1,1) = exp(-0.5) · f_0(0)
    // unnorm_0 = 0.3 · f_0(0),  unnorm_1 = 0.7 · exp(-0.5) · f_0(0)
    // filtered[0] = (0.3, 0.7·exp(-0.5)) / (0.3 + 0.7·exp(-0.5))
    // -----------------------------------------------------------------------
    #[test]
    fn test_t1_exact_posterior() {
        let params = ModelParams::new(
            vec![0.3, 0.7],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        );
        let result = filter(&params, &[0.0]).unwrap();

        let unnorm_0 = 0.3_f64;
        let unnorm_1 = 0.7_f64 * (-0.5_f64).exp();
        let total = unnorm_0 + unnorm_1;
        let expected_f0 = unnorm_0 / total;
        let expected_f1 = unnorm_1 / total;

        assert!(
            (result.filtered[0][0] - expected_f0).abs() < 1e-12,
            "filtered[0][0]={:.10}, expected={:.10}",
            result.filtered[0][0],
            expected_f0
        );
        assert!(
            (result.filtered[0][1] - expected_f1).abs() < 1e-12,
            "filtered[0][1]={:.10}, expected={:.10}",
            result.filtered[0][1],
            expected_f1
        );
    }

    // -----------------------------------------------------------------------
    // Test 9 — T=1 exact log-likelihood
    //
    // With π=(0.5, 0.5), identical regimes μ=(0,0), σ²=(1,1), y=0:
    //   c₁ = f(0;0,1)·0.5 + f(0;0,1)·0.5 = f(0;0,1) = 1/√(2π)
    //   log c₁ = -½·ln(2π)
    // -----------------------------------------------------------------------
    #[test]
    fn test_t1_exact_log_likelihood() {
        let params = symmetric_params();
        let result = filter(&params, &[0.0]).unwrap();

        // log N(0;0,1) = -½ ln(2π) - ½ ln(1) - 0 = -½ ln(2π)
        let expected_log_c = -(2.0 * std::f64::consts::PI).ln() / 2.0;
        assert!(
            (result.log_likelihood - expected_log_c).abs() < 1e-12,
            "log_likelihood={:.10}, expected={:.10}",
            result.log_likelihood,
            expected_log_c
        );
    }

    // -----------------------------------------------------------------------
    // Test 10 — Extreme observation locks onto the matching regime
    //
    // With μ=(-10, 10), σ²=(1,1) and y=10.0:
    // f_0(10) = N(10; -10, 1) ≈ 0,  f_1(10) = N(10; 10, 1) = peak
    // → filtered[0][1] should be extremely close to 1.
    // -----------------------------------------------------------------------
    #[test]
    fn test_extreme_observation_locks_regime() {
        let params = separated_params();
        let result = filter(&params, &[10.0]).unwrap();

        assert!(
            result.filtered[0][1] > 0.9999,
            "filtered prob of high-mean regime when y=10: {:.6}, expected >0.9999",
            result.filtered[0][1]
        );
        assert!(
            result.filtered[0][0] < 1e-4,
            "filtered prob of low-mean regime when y=10: {:.6}, expected <1e-4",
            result.filtered[0][0]
        );
    }

    // -----------------------------------------------------------------------
    // Test 11 — Prediction step is consistent with manual matrix multiply
    //
    // After filtering y₁, predicted[1] must equal  filtered[0] · P  (row-vector · P).
    // -----------------------------------------------------------------------
    #[test]
    fn test_prediction_step_matches_matrix_multiply() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.8, 0.2], vec![0.3, 0.7]],
            vec![0.0, 5.0],
            vec![1.0, 1.0],
        );
        let result = filter(&params, &[0.0, 2.5]).unwrap();

        let f0 = &result.filtered[0];
        // Manual: predicted[1][j] = Σᵢ P[i][j] · filtered[0][i]
        let expected_pred1_0 = f0[0] * 0.8 + f0[1] * 0.3;
        let expected_pred1_1 = f0[0] * 0.2 + f0[1] * 0.7;

        assert!(
            (result.predicted[1][0] - expected_pred1_0).abs() < 1e-12,
            "predicted[1][0]={:.10}, expected={:.10}",
            result.predicted[1][0],
            expected_pred1_0
        );
        assert!(
            (result.predicted[1][1] - expected_pred1_1).abs() < 1e-12,
            "predicted[1][1]={:.10}, expected={:.10}",
            result.predicted[1][1],
            expected_pred1_1
        );
    }

    // -----------------------------------------------------------------------
    // Test 12 — Filter tracks the true hidden state on a long simulation
    //
    // With strongly separated means (μ₀=-10, μ₁=10 and σ²=1) and persistent
    // transitions (p_ii=0.99), the argmax of filtered[t] should match the
    // true simulated state at least 95% of the time.
    // -----------------------------------------------------------------------
    #[test]
    fn test_filter_tracks_true_state() {
        let params = separated_params();
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 2_000, &mut rng).unwrap();
        let result = filter(&params, &sim.observations).unwrap();

        let correct: usize = (0..sim.t)
            .filter(|&t| {
                let pred_regime = if result.filtered[t][0] >= result.filtered[t][1] {
                    0
                } else {
                    1
                };
                pred_regime == sim.states[t]
            })
            .count();
        let accuracy = correct as f64 / sim.t as f64;

        assert!(
            accuracy > 0.95,
            "filter state-tracking accuracy={:.3}, expected >0.95",
            accuracy
        );
    }

    // -----------------------------------------------------------------------
    // Test 13 — Empty observations return an error
    // -----------------------------------------------------------------------
    #[test]
    fn test_empty_obs_returns_error() {
        let params = separated_params();
        assert!(filter(&params, &[]).is_err());
    }

    // -----------------------------------------------------------------------
    // Test 14 — Invalid params return an error
    // -----------------------------------------------------------------------
    #[test]
    fn test_invalid_params_returns_error() {
        // π sums to 0.8 — invalid
        let bad = ModelParams::new(
            vec![0.4, 0.4],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        );
        assert!(filter(&bad, &[0.0]).is_err());
    }
}
