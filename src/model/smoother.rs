/// Backward smoother (Kim smoother / Hamilton-Kim) for the Gaussian Markov Switching Model.
///
/// # Mathematical specification
///
/// ## Notation (consistent with filter.rs)
/// - `alpha_{t|t}(j)`   = `Pr(St=j | y_{1:t})`   -- filtered probability (from Phase 4)
/// - `alpha_{t+1|t}(j)` = `Pr(S_{t+1}=j | y_{1:t})` -- predicted probability (from Phase 4)
/// - `gamma_t(j)`        = `Pr(St=j | y_{1:T})`   -- smoothed probability (this module)
///
/// ## Terminal condition (Section 8 of phase7.md)
///
/// At the final time T there is no future information, so:
///
/// ```text
/// gamma_T(j) = alpha_{T|T}(j)
/// ```
///
/// ## Backward recursion (Section 9 of phase7.md)
///
/// For t = T-1, T-2, ..., 1:
///
/// ```text
/// gamma_t(i) = alpha_{t|t}(i) * sum_j [ p_ij * gamma_{t+1}(j) / alpha_{t+1|t}(j) ]
/// ```
///
/// Here:
/// - `alpha_{t|t}(i)`   = filtered probability at time t for regime i,
/// - `p_ij`             = transition probability from regime i to regime j,
/// - `alpha_{t+1|t}(j)` = predicted probability at time t+1 for regime j,
/// - `gamma_{t+1}(j)`   = already-computed smoothed probability at time t+1.
///
/// ## Indexing convention (0-based, matching FilterResult)
///
/// `FilterResult` uses 0-based indexing where array index `s` corresponds to
/// 1-based time step `t = s+1`.
///
/// | 0-based index `s` | 1-based time `t` | `filtered[s]`  | `predicted[s]`     |
/// |-------------------|------------------|----------------|--------------------|
/// | 0                 | 1                | alpha_{1|1}    | alpha_{1|0} = pi   |
/// | s                 | s+1              | alpha_{s+1|s+1}| alpha_{s+1|s}      |
/// | T-1               | T                | alpha_{T|T}    | alpha_{T|T-1}      |
///
/// The backward recursion in 0-based indices, stepping from `s = T-2` down to `s = 0`:
///
/// ```text
/// smoothed[s][i] = filtered[s][i]
///                  * sum_j ( p_ij * smoothed[s+1][j] / predicted[s+1][j] )
/// ```
///
/// because:
/// - `alpha_{t|t}(i)`   = `filtered[s][i]`    (t = s+1, s = t-1)
/// - `alpha_{t+1|t}(j)` = `predicted[s+1][j]` (predicted[idx] = alpha_{idx+1|idx})
/// - `gamma_{t+1}(j)`   = `smoothed[s+1][j]`
///
/// ## Numerical guard
///
/// If `predicted[s+1][j]` falls below `DENOM_FLOOR` (e.g. 1e-300), that regime
/// contributes negligibly and the corresponding term is skipped.  This avoids a
/// division-by-zero without any log-space transformation.
///
/// ## Architecture
///
/// This module consumes only `FilterResult` and `ModelParams`.  It never reads
/// the raw observation sequence and never calls the emission model.  All the
/// information it needs was already encoded in the forward pass.
use anyhow::Result;

use super::filter::FilterResult;
use super::params::ModelParams;

/// Minimum denominator value below which a predicted probability is treated as
/// zero for the purpose of the backward recursion.
const DENOM_FLOOR: f64 = 1e-300;

/// The complete output of one backward smoothing pass.
///
/// All time-indexed fields use 0-based indexing matching `FilterResult`:
/// - index `0`   corresponds to t = 1 (first observation),
/// - index `T-1` corresponds to t = T (last observation).
#[derive(Debug, Clone)]
pub struct SmootherResult {
    /// T -- number of observations (same as the FilterResult it was derived from).
    pub t: usize,
    /// K -- number of regimes.
    pub k: usize,
    /// `smoothed[s][j]` = gamma_{s+1}(j) = Pr(S_{s+1}=j | y_{1:T}).
    ///
    /// At index `T-1` this equals `filtered[T-1]` (terminal condition).
    /// Shape: T x K.
    pub smoothed: Vec<Vec<f64>>,
}

/// Run the backward smoothing pass on the output of `filter`.
///
/// This function takes the already-completed `FilterResult` and traverses the
/// stored predicted and filtered probability arrays backward in time, applying
/// the recursion from Section 9 of phase7.md at each step.
///
/// # Errors
///
/// Returns an error if:
/// - `filter_result.t` is zero (empty filter result),
/// - `filter_result.k` does not match `params.k`,
/// - `params.validate()` fails.
pub fn smooth(params: &ModelParams, filter_result: &FilterResult) -> Result<SmootherResult> {
    if filter_result.t == 0 {
        anyhow::bail!("smooth: filter result has no time steps");
    }
    if filter_result.k != params.k {
        anyhow::bail!(
            "smooth: filter result has k={} but params has k={}",
            filter_result.k,
            params.k
        );
    }
    params.validate()?;

    let t_len = filter_result.t;
    let k = filter_result.k;

    // Allocate the output: T x K, initialised to 0.
    let mut smoothed: Vec<Vec<f64>> = vec![vec![0.0_f64; k]; t_len];

    // ------------------------------------------------------------------
    // Step 1 -- Terminal condition (Section 8 of phase7.md):
    //   gamma_T(j) = alpha_{T|T}(j)
    //   In 0-based: smoothed[T-1] = filtered[T-1]
    // ------------------------------------------------------------------
    smoothed[t_len - 1].copy_from_slice(&filter_result.filtered[t_len - 1]);

    // ------------------------------------------------------------------
    // Step 2 -- Backward recursion (Section 9 of phase7.md):
    //   For s = T-2 down to 0 (1-based t = T-1 down to 1):
    //
    //   smoothed[s][i] = filtered[s][i]
    //                    * sum_j( p_ij * smoothed[s+1][j] / predicted[s+1][j] )
    // ------------------------------------------------------------------
    for s in (0..t_len - 1).rev() {
        // smoothed[s+1] = gamma at 1-based time s+2 (already computed above).
        // predicted[s+1] = alpha_{s+2|s+1} = denominator for this step.
        let gamma_next = &smoothed[s + 1].clone();
        let pred_next = &filter_result.predicted[s + 1];
        let filt_curr = &filter_result.filtered[s];

        for i in 0..k {
            // Backward multiplier: sum_j [ p_ij * gamma_{t+1}(j) / alpha_{t+1|t}(j) ]
            let multiplier: f64 = (0..k)
                .map(|j| {
                    let denom = pred_next[j];
                    if denom < DENOM_FLOOR {
                        // regime j had negligible predicted weight; its contribution
                        // to the backward multiplier is also negligible.
                        0.0
                    } else {
                        params.transition_row(i)[j] * gamma_next[j] / denom
                    }
                })
                .sum();

            smoothed[s][i] = filt_curr[i] * multiplier;
        }

        // In exact arithmetic the backward recursion is self-normalizing
        // (Section 15 of phase7.md).  In floating-point, rounding errors in the
        // denominator term can push the sum slightly above or below 1.  A single
        // renormalization step per time step eliminates cumulative drift.
        let sum: f64 = smoothed[s].iter().sum();
        if sum > 0.0 {
            for j in 0..k {
                smoothed[s][j] /= sum;
            }
        }
    }

    Ok(SmootherResult {
        t: t_len,
        k,
        smoothed,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::{ModelParams, filter, simulate};
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    const SEED: u64 = 42;
    const TOL: f64 = 1e-11;

    // -----------------------------------------------------------------------
    // Shared helpers
    // -----------------------------------------------------------------------

    fn assert_prob_vec(v: &[f64], label: &str, s: usize) {
        let sum: f64 = v.iter().sum();
        assert!(
            (sum - 1.0).abs() < TOL,
            "{label}[{s}] sums to {sum:.15}, expected 1"
        );
        for (j, &p) in v.iter().enumerate() {
            assert!(
                p >= -1e-15 && p <= 1.0 + 1e-15,
                "{label}[{s}][{j}] = {p:.6} out of [0,1]"
            );
        }
    }

    /// Run filter + smooth and assert all structural invariants.
    fn run_and_check(params: &ModelParams, obs: &[f64]) -> (FilterResult, SmootherResult) {
        let fr = filter(params, obs).expect("filter must not fail");
        let sr = smooth(params, &fr).expect("smooth must not fail");

        // Structural invariants (Section 15 of phase7.md)
        assert_eq!(sr.t, fr.t);
        assert_eq!(sr.k, fr.k);

        for s in 0..sr.t {
            assert_prob_vec(&sr.smoothed[s], "smoothed", s);
        }

        // Terminal condition (Section 8): smoothed[T-1] == filtered[T-1]
        let last = sr.t - 1;
        for j in 0..sr.k {
            assert!(
                (sr.smoothed[last][j] - fr.filtered[last][j]).abs() < TOL,
                "terminal: smoothed[T-1][{j}]={:.12} != filtered[T-1][{j}]={:.12}",
                sr.smoothed[last][j],
                fr.filtered[last][j]
            );
        }

        (fr, sr)
    }

    fn argmax(v: &[f64]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    fn accuracy(probs: &[Vec<f64>], true_states: &[usize]) -> f64 {
        let correct = (0..probs.len())
            .filter(|&t| argmax(&probs[t]) == true_states[t])
            .count();
        correct as f64 / probs.len() as f64
    }

    // -----------------------------------------------------------------------
    // Structural tests
    // -----------------------------------------------------------------------

    /// Terminal condition: smoothed[T-1] == filtered[T-1] exactly.
    #[test]
    fn terminal_condition_exact() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 500, &mut rng).unwrap();
        run_and_check(&params, &sim.observations);
    }

    /// Invariants hold for a range of K=2 scenarios.
    #[test]
    fn structural_invariants_k2_scenarios() {
        let scenarios: Vec<(Vec<f64>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>)> = vec![
            // (pi, P, means, variances)
            (vec![0.5, 0.5], vec![vec![0.99, 0.01], vec![0.01, 0.99]], vec![-10.0, 10.0], vec![1.0, 1.0]),
            (vec![0.5, 0.5], vec![vec![0.9, 0.1], vec![0.1, 0.9]],   vec![-0.5, 0.5],  vec![1.0, 1.0]),
            (vec![0.5, 0.5], vec![vec![0.4, 0.6], vec![0.6, 0.4]],   vec![-5.0, 5.0],  vec![1.0, 1.0]),
            (vec![0.5, 0.5], vec![vec![0.95, 0.05], vec![0.05, 0.95]], vec![0.0, 0.0], vec![0.25, 9.0]),
        ];

        for (pi, p_mat, means, vars) in scenarios {
            let params = ModelParams::new(pi, p_mat, means, vars);
            let mut rng = SmallRng::seed_from_u64(SEED);
            let sim = simulate(params.clone(), 1_000, &mut rng).unwrap();
            run_and_check(&params, &sim.observations);
        }
    }

    /// Single observation: smoothed[0] == filtered[0] (T=1 edge case).
    #[test]
    fn single_observation_smoothed_equals_filtered() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let fr = filter(&params, &[2.5]).unwrap();
        let sr = smooth(&params, &fr).unwrap();
        for j in 0..sr.k {
            assert!(
                (sr.smoothed[0][j] - fr.filtered[0][j]).abs() < TOL,
                "T=1: smoothed[0][{j}]={:.12} != filtered[0][{j}]={:.12}",
                sr.smoothed[0][j],
                fr.filtered[0][j]
            );
        }
    }

    /// K=3 structural invariants hold.
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

    /// Long sample (T=20_000): numerical stability -- all invariants still hold.
    #[test]
    fn long_sample_numerical_stability() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.99, 0.01], vec![0.01, 0.99]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 20_000, &mut rng).unwrap();
        run_and_check(&params, &sim.observations);
    }

    // -----------------------------------------------------------------------
    // Behavioral tests
    // -----------------------------------------------------------------------

    /// On strongly separated, persistent data, smoothed classification accuracy
    /// must be at least as high as filtered classification accuracy.
    ///
    /// Phase 7 section 18: "smoothed probabilities align better with the true
    /// hidden regimes than filtered probabilities."
    #[test]
    fn smoothed_accuracy_at_least_as_good_as_filtered() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.95, 0.05], vec![0.05, 0.95]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 5_000, &mut rng).unwrap();
        let (fr, sr) = run_and_check(&params, &sim.observations);

        let acc_filt = accuracy(&fr.filtered, &sim.states);
        let acc_smooth = accuracy(&sr.smoothed, &sim.states);

        assert!(
            acc_smooth >= acc_filt - 1e-9,
            "smoothed accuracy={acc_smooth:.4} should be >= filtered accuracy={acc_filt:.4}"
        );
    }

    /// Near a hard regime-change boundary the smoother should be more decisive
    /// than the filter.
    ///
    /// Construct an observation sequence: 50 obs at -10 (regime 0 certain),
    /// a single ambiguous obs at 0 (equally likely under both regimes), then
    /// 50 obs at +10 (regime 1 certain).
    ///
    /// At the ambiguous observation (index 50):
    ///   - the filter only uses y_{1:50} and can be uncertain since it just
    ///     saw regime 0 for a long time,
    ///   - the smoother also uses y_{51:100} which strongly indicate regime 1,
    ///     so it should assign higher probability to regime 1 at that point.
    #[test]
    fn smoother_more_decisive_at_regime_boundary() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-10.0, 10.0],
            vec![1.0, 1.0],
        );

        let mut obs: Vec<f64> = vec![-10.0; 50];
        obs.push(0.0); // ambiguous observation at index 50
        obs.extend(vec![10.0; 50]);

        let (fr, sr) = run_and_check(&params, &obs);

        // At the ambiguous time step (index 50, 0-based), the smoother has
        // already seen 50 future observations firmly in regime 1, so it should
        // assign at least as much weight to regime 1 as the filter does.
        let filter_regime1_at_50 = fr.filtered[50][1];
        let smooth_regime1_at_50 = sr.smoothed[50][1];

        assert!(
            smooth_regime1_at_50 >= filter_regime1_at_50,
            "smoother[50][1]={smooth_regime1_at_50:.4} should be >= filter[50][1]={filter_regime1_at_50:.4} \
             at the ambiguous boundary step"
        );
    }

    /// Deep in the interior of a long stable-regime run, filtering and smoothing
    /// converge (Section 17, Case 1 of phase7.md).
    ///
    /// Near the middle of 1000 observations firmly in regime 0, both the
    /// filtered and smoothed probabilities for regime 0 should be > 0.99.
    #[test]
    fn interior_stable_run_filter_and_smooth_converge() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.99, 0.01], vec![0.01, 0.99]],
            vec![-10.0, 10.0],
            vec![1.0, 1.0],
        );

        // All observations at the regime-0 mean: maximum evidence for regime 0.
        let obs = vec![-10.0_f64; 1_000];
        let (fr, sr) = run_and_check(&params, &obs);

        let mid = 500;
        let filt_regime0 = fr.filtered[mid][0];
        let smooth_regime0 = sr.smoothed[mid][0];

        assert!(
            filt_regime0 > 0.99,
            "filter mid-interior regime-0 prob = {filt_regime0:.6}, expected > 0.99"
        );
        assert!(
            smooth_regime0 > 0.99,
            "smoother mid-interior regime-0 prob = {smooth_regime0:.6}, expected > 0.99"
        );
    }

    /// The smoother identifies regime transitions earlier than the filter.
    ///
    /// Block design: 50 obs near regime 0, 50 obs near regime 1 (at the exact
    /// regime means).  Measure the first step after the block switch (index 50)
    /// where each sequence exceeds 0.5 for regime 1.  The smoother has already
    /// seen the future block, so it should flip to regime 1 at the same step or
    /// earlier than the filter.
    ///
    /// This validates phase7.md Section 17 Case 2: "smoothing often differs most
    /// strongly from filtering around transition points."
    #[test]
    fn smoother_revises_beliefs_at_transition() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-3.0, 3.0],
            vec![1.0, 1.0],
        );

        // Observations exactly at regime means to maximise evidence.
        let obs: Vec<f64> = vec![-3.0; 50].into_iter().chain(vec![3.0; 50]).collect();
        let (fr, sr) = run_and_check(&params, &obs);

        // First step on or after index 50 where the distribution exceeds 0.5 for regime 1.
        let first_flip = |probs: &[Vec<f64>]| {
            (50..probs.len())
                .find(|&t| probs[t][1] > 0.5)
                .unwrap_or(probs.len())
        };

        let filter_flip = first_flip(&fr.filtered);
        let smooth_flip = first_flip(&sr.smoothed);

        assert!(
            smooth_flip <= filter_flip,
            "smoother should flip to regime 1 at the same step or earlier than the filter: \
             smoother flips at step {smooth_flip}, filter at {filter_flip}"
        );
    }

    /// Weakly separated regimes: even after smoothing posteriors may remain
    /// diffuse (Section 20 Mistake 4 of phase7.md).
    ///
    /// This is correct behavior; we simply verify invariants hold and that
    /// smoothed posteriors do not claim false certainty.
    #[test]
    fn weak_separation_smoothed_may_remain_diffuse() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-0.3, 0.3],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 1_000, &mut rng).unwrap();
        let (_fr, sr) = run_and_check(&params, &sim.observations);

        // Mean max smoothed probability: should not be near 1.0 for weak separation
        let mean_max: f64 = sr
            .smoothed
            .iter()
            .map(|s| s.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
            .sum::<f64>()
            / sr.t as f64;

        assert!(
            mean_max < 0.90,
            "weak separation: mean max smoothed prob={mean_max:.4}, \
             expected < 0.90 (diffuse posteriors are correct behavior here)"
        );
    }

    /// The backward pass does not re-read observations (design property):
    /// smoothing with a cloned FilterResult and the same params produces
    /// identical output.
    #[test]
    fn smooth_is_deterministic_given_filter_result() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.95, 0.05], vec![0.05, 0.95]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 500, &mut rng).unwrap();
        let fr = filter(&params, &sim.observations).unwrap();

        let sr1 = smooth(&params, &fr).unwrap();
        let sr2 = smooth(&params, &fr.clone()).unwrap();

        for t in 0..sr1.t {
            for j in 0..sr1.k {
                assert_eq!(
                    sr1.smoothed[t][j], sr2.smoothed[t][j],
                    "smooth is not deterministic at smoothed[{t}][{j}]"
                );
            }
        }
    }

    /// Multi-seed stress test: invariants hold across 50 independent short runs.
    #[test]
    fn multi_seed_short_runs() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        for seed in 0u64..50 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let sim = simulate(params.clone(), 10, &mut rng).unwrap();
            run_and_check(&params, &sim.observations);
        }
    }
}
