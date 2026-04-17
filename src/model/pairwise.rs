/// Pairwise posterior transition probabilities for the Gaussian Markov Switching Model.
///
/// # Mathematical specification
///
/// ## The new inference object
///
/// After the forward filter (Phase 4) and backward smoother (Phase 7), every
/// marginal posterior `Pr(Sₜ=j | y_{1:T})` is already available.  Phase 8
/// extends this to the **joint** posterior over two adjacent time steps:
///
/// ```text
/// ξₜ(i,j) = Pr(S_{t-1}=i, Sₜ=j | y_{1:T}),   t = 2, …, T.
/// ```
///
/// ## Central formula
///
/// ```text
/// ξₜ(i,j) = α_{t-1|t-1}(i) · p_ij · γₜ(j) / α_{t|t-1}(j)
/// ```
///
/// where:
/// - `α_{t-1|t-1}(i)` = filtered probability of regime i at time t-1,
/// - `p_ij`            = transition probability from regime i to regime j,
/// - `γₜ(j)`           = smoothed probability of regime j at time t,
/// - `α_{t|t-1}(j)`    = predicted probability of regime j at time t.
///
/// ## Derivation
///
/// Starting from the joint probability and using the filter update identity
/// `α_{t|t}(j) = f_j(yₜ) · α_{t|t-1}(j) / cₜ`, the emission density `f_j(yₜ)`
/// and predictive normalizer `cₜ` cancel, yielding Form B above.  No re-evaluation
/// of emissions is needed.
///
/// ## Structural invariants
///
/// For every transition step s (0-based, covering math t = s+2):
///
/// ### A — Nonnegativity
/// ```text
/// ξₜ(i,j) ≥ 0   for all i, j.
/// ```
///
/// ### B — Unit sum
/// ```text
/// Σᵢ Σⱼ ξₜ(i,j) = 1.
/// ```
/// Proof: substitute the prediction identity Σᵢ α_{t-1|t-1}(i) · p_ij = α_{t|t-1}(j)
/// and the unit-sum of γₜ.
///
/// ### C — Column marginal (next-time smoothed)
/// ```text
/// Σᵢ ξₜ(i,j) = γₜ(j).
/// ```
/// Proof: factor out j-terms; inner sum equals α_{t|t-1}(j) by Chapman-Kolmogorov,
/// which cancels the denominator.
///
/// ### D — Row marginal (previous-time smoothed)
/// ```text
/// Σⱼ ξₜ(i,j) = γ_{t-1}(i).
/// ```
/// Proof: the inner sum equals the backward multiplier used by the smoother at step
/// t-1, whose product with filtered[t-1][i] produces smoothed[t-1][i] = γ_{t-1}(i).
/// Holds to floating-point precision; the smoother's per-step renormalization
/// introduces an error of O(machine epsilon).
///
/// ## 0-based indexing
///
/// Arrays are indexed `s = 0, …, T-2`, corresponding to math t = 2, …, T.
///
/// | Array access         | Math quantity           |
/// |----------------------|-------------------------|
/// | `filtered[s][i]`     | α_{s+1\|s+1}(i)         |
/// | `predicted[s+1][j]`  | α_{s+2\|s+1}(j)         |
/// | `smoothed[s+1][j]`   | γ_{s+2}(j)              |
/// | `xi[s][i][j]`        | ξ_{s+2}(i,j)            |
///
/// ## Expected transition counts
///
/// ```text
/// N_ij^exp = Σₜ ξₜ(i,j) = Σₛ xi[s][i][j]
/// ```
///
/// These are the key deliverable for the EM M-step.
use anyhow::Result;

use super::filter::FilterResult;
use super::params::ModelParams;
use super::smoother::SmootherResult;

/// Minimum predicted probability below which the denominator is treated as zero.
/// Identical to the guard in `smoother.rs`.
const DENOM_FLOOR: f64 = 1e-300;

/// The complete output of one pairwise-posterior pass.
///
/// Contains the (T-1) × K × K tensor of joint posterior transition probabilities
/// and the K × K matrix of posterior expected transition counts.
#[derive(Debug, Clone)]
pub struct PairwiseResult {
    /// T — number of observations (same as the FilterResult it was derived from).
    pub t: usize,
    /// K — number of regimes.
    pub k: usize,
    /// `xi[s][i][j]` = ξ_{s+2}(i,j) = Pr(S_{s+1}=i, S_{s+2}=j | y_{1:T}).
    ///
    /// 0-based index `s` ranges over `0..T-1`, corresponding to math steps t = 2..=T.
    /// Empty (`len() == 0`) when T = 1.
    /// Shape: (T-1) × K × K.
    pub xi: Vec<Vec<Vec<f64>>>,
    /// `expected_transitions[i][j]` = N_ij^exp = Σₜ ξₜ(i,j) = Σₛ xi[s][i][j].
    ///
    /// Posterior expected number of transitions from regime i to regime j.
    /// Shape: K × K.
    pub expected_transitions: Vec<Vec<f64>>,
}

/// Compute pairwise posterior transition probabilities from a completed filter
/// and smoother pass.
///
/// This function never reads the raw observation sequence.  All information
/// required was already encoded in `filter_result` and `smoother_result`.
///
/// # Errors
///
/// Returns an error if:
/// - `filter_result.k` does not match `params.k`,
/// - `filter_result.k` does not match `smoother_result.k`,
/// - `filter_result.t` does not match `smoother_result.t`,
/// - `params.validate()` fails.
pub fn pairwise(
    params: &ModelParams,
    filter_result: &FilterResult,
    smoother_result: &SmootherResult,
) -> Result<PairwiseResult> {
    if filter_result.k != params.k {
        anyhow::bail!(
            "pairwise: filter_result.k={} != params.k={}",
            filter_result.k,
            params.k
        );
    }
    if filter_result.k != smoother_result.k {
        anyhow::bail!(
            "pairwise: filter_result.k={} != smoother_result.k={}",
            filter_result.k,
            smoother_result.k
        );
    }
    if filter_result.t != smoother_result.t {
        anyhow::bail!(
            "pairwise: filter_result.t={} != smoother_result.t={}",
            filter_result.t,
            smoother_result.t
        );
    }
    params.validate()?;

    let t_len = filter_result.t;
    let k = filter_result.k;

    // T=1: no transitions exist; return empty tensor.
    if t_len <= 1 {
        return Ok(PairwiseResult {
            t: t_len,
            k,
            xi: vec![],
            expected_transitions: vec![vec![0.0; k]; k],
        });
    }

    // ------------------------------------------------------------------
    // Allocate output.
    // xi[s] is a K×K matrix stored as Vec<Vec<f64>>.
    // s ranges over 0..t_len-1, covering math t = 2..=T.
    // ------------------------------------------------------------------
    let mut xi: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0_f64; k]; k]; t_len - 1];
    let mut expected_transitions: Vec<Vec<f64>> = vec![vec![0.0_f64; k]; k];

    for s in 0..t_len - 1 {
        // 0-based mapping:
        //   filtered[s][i]    = α_{s+1|s+1}(i)   (math: α_{t-1|t-1}(i), t = s+2)
        //   predicted[s+1][j] = α_{s+2|s+1}(j)   (math: α_{t|t-1}(j))
        //   smoothed[s+1][j]  = γ_{s+2}(j)        (math: γₜ(j))
        let filt_prev = &filter_result.filtered[s];
        let pred_next = &filter_result.predicted[s + 1];
        let smooth_next = &smoother_result.smoothed[s + 1];

        for i in 0..k {
            let row_p = params.transition_row(i);
            for j in 0..k {
                let denom = pred_next[j];
                if denom < DENOM_FLOOR {
                    // Predicted weight is negligibly small; contribution skipped.
                    xi[s][i][j] = 0.0;
                } else {
                    xi[s][i][j] = filt_prev[i] * row_p[j] * smooth_next[j] / denom;
                }

                expected_transitions[i][j] += xi[s][i][j];
            }
        }
    }

    Ok(PairwiseResult {
        t: t_len,
        k,
        xi,
        expected_transitions,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::smoother::smooth;
    use super::super::{ModelParams, filter, simulate};
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    const SEED: u64 = 42;
    const TOL: f64 = 1e-10;

    // -----------------------------------------------------------------------
    // Shared helpers
    // -----------------------------------------------------------------------

    /// Run filter → smooth → pairwise and assert all structural invariants.
    fn run_and_check(params: &ModelParams, obs: &[f64]) -> PairwiseResult {
        let fr = filter(params, obs).expect("filter must not fail");
        let sr = smooth(params, &fr).expect("smooth must not fail");
        let pr = pairwise(params, &fr, &sr).expect("pairwise must not fail");

        let t_len = fr.t;
        let k = fr.k;

        assert_eq!(pr.t, t_len);
        assert_eq!(pr.k, k);
        assert_eq!(pr.xi.len(), if t_len > 0 { t_len - 1 } else { 0 });
        assert_eq!(pr.expected_transitions.len(), k);

        for s in 0..pr.xi.len() {
            // Invariant A: nonnegativity.
            for i in 0..k {
                for j in 0..k {
                    assert!(
                        pr.xi[s][i][j] >= -1e-15,
                        "xi[{s}][{i}][{j}] = {} is negative",
                        pr.xi[s][i][j]
                    );
                }
            }

            // Invariant B: unit sum over all (i,j) pairs.
            let total: f64 = pr.xi[s].iter().flatten().sum();
            assert!(
                (total - 1.0).abs() < TOL,
                "xi[{s}] sums to {total:.15}, expected 1"
            );

            // Invariant C: column marginal Σᵢ xi[s][i][j] == smoothed[s+1][j].
            for j in 0..k {
                let col_sum: f64 = (0..k).map(|i| pr.xi[s][i][j]).sum();
                let expected = sr.smoothed[s + 1][j];
                assert!(
                    (col_sum - expected).abs() < TOL,
                    "Invariant C: Σᵢ xi[{s}][i][{j}] = {col_sum:.15} != smoothed[{}][{j}] = {expected:.15}",
                    s + 1
                );
            }

            // Invariant D: row marginal Σⱼ xi[s][i][j] == smoothed[s][i].
            for i in 0..k {
                let row_sum: f64 = pr.xi[s][i].iter().sum();
                let expected = sr.smoothed[s][i];
                assert!(
                    (row_sum - expected).abs() < TOL,
                    "Invariant D: Σⱼ xi[{s}][{i}][j] = {row_sum:.15} != smoothed[{s}][{i}] = {expected:.15}"
                );
            }
        }

        // Invariant E: expected transition counts are finite and nonneg.
        for i in 0..k {
            for j in 0..k {
                let n = pr.expected_transitions[i][j];
                assert!(
                    n >= -1e-12 && n.is_finite(),
                    "expected_transitions[{i}][{j}] = {n} is not finite/nonneg"
                );
            }
        }

        pr
    }

    // -----------------------------------------------------------------------
    // Structural tests
    // -----------------------------------------------------------------------

    /// T=1: the pairwise tensor is empty; expected_transitions is all zeros.
    #[test]
    fn empty_when_single_observation() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let fr = filter(&params, &[2.5]).unwrap();
        let sr = smooth(&params, &fr).unwrap();
        let pr = pairwise(&params, &fr, &sr).unwrap();

        assert_eq!(pr.xi.len(), 0);
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(pr.expected_transitions[i][j], 0.0);
            }
        }
    }

    /// All five invariants hold for K=2 with persistent regimes.
    #[test]
    fn structural_invariants_k2_persistent() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.99, 0.01], vec![0.01, 0.99]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 1_000, &mut rng).unwrap();
        run_and_check(&params, &sim.observations);
    }

    /// All five invariants hold for K=2 with frequent switching.
    #[test]
    fn structural_invariants_k2_mixing() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.4, 0.6], vec![0.6, 0.4]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 1_000, &mut rng).unwrap();
        run_and_check(&params, &sim.observations);
    }

    /// All five invariants hold for K=3.
    #[test]
    fn structural_invariants_k3() {
        let params = ModelParams::new(
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            vec![
                vec![0.9, 0.05, 0.05],
                vec![0.05, 0.9, 0.05],
                vec![0.05, 0.05, 0.9],
            ],
            vec![-8.0, 0.0, 8.0],
            vec![1.0, 1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 2_000, &mut rng).unwrap();
        run_and_check(&params, &sim.observations);
    }

    /// T=2: one transition step; xi has length 1.
    #[test]
    fn two_observations_one_transition() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.8, 0.2], vec![0.2, 0.8]],
            vec![-3.0, 3.0],
            vec![1.0, 1.0],
        );
        let pr = run_and_check(&params, &[-3.0, 3.0]);
        assert_eq!(pr.xi.len(), 1);
    }

    /// Long sample (T=5000): numerical stability — no NaN, no Inf.
    #[test]
    fn long_sample_numerical_stability() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.99, 0.01], vec![0.01, 0.99]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 5_000, &mut rng).unwrap();
        let pr = run_and_check(&params, &sim.observations);
        for i in 0..2 {
            for j in 0..2 {
                assert!(pr.expected_transitions[i][j].is_finite());
            }
        }
    }

    // -----------------------------------------------------------------------
    // Behavioral tests
    // -----------------------------------------------------------------------

    /// With strongly persistent regimes (p_ii = 0.95), the diagonal expected
    /// transition counts N_ii^exp should exceed all off-diagonal counts N_ij^exp.
    #[test]
    fn dominant_diagonal_for_persistent_regimes() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.95, 0.05], vec![0.05, 0.95]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 2_000, &mut rng).unwrap();
        let pr = run_and_check(&params, &sim.observations);

        let n = &pr.expected_transitions;
        // Each diagonal entry should dominate its row.
        for i in 0..2 {
            for j in 0..2 {
                if i != j {
                    assert!(
                        n[i][i] > n[i][j],
                        "N[{i}][{i}]={:.3} should exceed N[{i}][{j}]={:.3}",
                        n[i][i],
                        n[i][j]
                    );
                }
            }
        }
    }

    /// At a definitive regime switch, the off-diagonal xi(0→1) should dominate
    /// at the switching step.
    ///
    /// Block design: 50 obs at -5 (regime 0), 50 obs at +5 (regime 1).
    /// At the switch boundary (s = 49, covering the transition from t=50 to t=51),
    /// ξ(0,1) should be the dominant entry.
    #[test]
    fn offdiag_dominant_at_regime_switch() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let obs: Vec<f64> = vec![-5.0_f64; 50]
            .into_iter()
            .chain(vec![5.0_f64; 50])
            .collect();
        let pr = run_and_check(&params, &obs);

        // s = 49: transition from t=50 (still in regime-0 block) to t=51 (regime-1 block).
        let xi_switch = &pr.xi[49];
        let xi_01 = xi_switch[0][1];
        let xi_00 = xi_switch[0][0];
        let xi_11 = xi_switch[1][1];
        let xi_10 = xi_switch[1][0];

        assert!(
            xi_01 > xi_00 && xi_01 > xi_11 && xi_01 > xi_10,
            "at regime switch ξ(0→1)={xi_01:.4} should dominate: \
             ξ(0,0)={xi_00:.4}  ξ(1,1)={xi_11:.4}  ξ(1,0)={xi_10:.4}"
        );
    }

    /// The expected transition counts summed over rows should equal the
    /// posterior expected regime occupancy from the smoothed marginals:
    ///
    ///   Σⱼ N_ij^exp = Σₛ smoothed[s][i]  for s = 0..T-2
    ///
    /// This is the consistency identity from Section 13 of phase8.md.
    #[test]
    fn expected_counts_consistent_with_smoothed_occupancy() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 500, &mut rng).unwrap();
        let fr = filter(&params, &sim.observations).unwrap();
        let sr = smooth(&params, &fr).unwrap();
        let pr = pairwise(&params, &fr, &sr).unwrap();

        let t_len = fr.t;
        for i in 0..2 {
            // Σⱼ N_ij^exp = row sum of expected_transitions
            let row_sum: f64 = pr.expected_transitions[i].iter().sum();

            // M_i^exp = Σₛ smoothed[s][i] for s = 0..T-2 (math t = 1..T-1)
            let occupancy: f64 = (0..t_len - 1).map(|s| sr.smoothed[s][i]).sum();

            assert!(
                (row_sum - occupancy).abs() < TOL,
                "regime {i}: Σⱼ N_ij^exp = {row_sum:.10} != M_i^exp = {occupancy:.10}"
            );
        }
    }

    /// Result is deterministic — calling pairwise twice on identical inputs
    /// yields byte-for-byte identical output.
    #[test]
    fn pairwise_is_deterministic() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 200, &mut rng).unwrap();
        let fr = filter(&params, &sim.observations).unwrap();
        let sr = smooth(&params, &fr).unwrap();

        let pr1 = pairwise(&params, &fr, &sr).unwrap();
        let pr2 = pairwise(&params, &fr, &sr).unwrap();

        for s in 0..pr1.xi.len() {
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(
                        pr1.xi[s][i][j], pr2.xi[s][i][j],
                        "xi[{s}][{i}][{j}] differs between calls"
                    );
                }
            }
        }
    }

    /// Multiple seeds: structural invariants hold across diverse random samples.
    #[test]
    fn multi_seed_invariants() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        for seed in 0_u64..10 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let sim = simulate(params.clone(), 300, &mut rng).unwrap();
            run_and_check(&params, &sim.observations);
        }
    }

    /// With weakly separated regimes, pairwise probabilities may be diffuse
    /// (no entry dominates).  This is the correct behavior and not a bug.
    #[test]
    fn weak_separation_pairwise_remains_diffuse() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.7, 0.3], vec![0.3, 0.7]],
            vec![-0.2, 0.2],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 500, &mut rng).unwrap();
        let pr = run_and_check(&params, &sim.observations);

        // No single xi[s] entry should routinely exceed 0.9;
        // sample a mid-run step to avoid boundary effects.
        let s = pr.xi.len() / 2;
        let max_val = pr.xi[s]
            .iter()
            .flatten()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_val < 0.9,
            "expected diffuse pairwise at step {s} for weak separation, but max={max_val:.4}"
        );
    }
}
