/// Phase 6 - forward filter validation on simulated data.
///
/// This module contains only test infrastructure.  See the module-level doc in
/// the `#[cfg(test)]` block for design rationale.
#[cfg(test)]
mod tests {
    //! Phase 6 validation tests for the forward filter.
    //!
    //! # Strategy
    //!
    //! For each scenario:
    //! 1. Define parameters T (controlled experiment design).
    //! 2. Simulate (y_{1:T}, S_{1:T}) under T with the Phase-2 simulator.
    //! 3. Run the forward filter with the same T.
    //! 4. Check *structural invariants* (normalization, bounds, finiteness).
    //! 5. Check *behavioral plausibility* (sharpness, responsiveness, stability).
    //! 6. Where relevant, compare argmax posteriors to the known true states.
    //!
    //! # Scenario families (phase6.md)
    //!
    //! A - Strongly separated regimes
    //! B - Weakly separated regimes
    //! C - Mean-switching only
    //! D - Variance-switching only
    //! E - Highly persistent transitions
    //! F - Frequent switching
    //! G - Short samples (T=10)
    //! H - Long samples (T=50_000, numerical stability)

    use super::super::{FilterResult, ModelParams, filter, log_likelihood, simulate};
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    const SEED: u64 = 42;

    // -----------------------------------------------------------------------
    // Shared helpers
    // -----------------------------------------------------------------------

    /// Assert every entry in `v` ? [0,1] and the vector sums to 1.
    fn assert_prob_vec(v: &[f64], label: &str, t: usize) {
        let sum: f64 = v.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-11,
            "{label}[{t}] sums to {sum:.15}, expected 1"
        );
        for (j, &p) in v.iter().enumerate() {
            assert!(
                p >= -1e-15 && p <= 1.0 + 1e-15,
                "{label}[{t}][{j}] = {p:.6} out of [0,1]"
            );
        }
    }

    /// Run `filter` and verify all structural invariants at every step.
    fn check_invariants(params: &ModelParams, obs: &[f64]) -> FilterResult {
        let result = filter(params, obs).expect("filter must not fail on valid params");

        for t in 0..result.t {
            assert_prob_vec(&result.predicted[t], "predicted", t);
            assert_prob_vec(&result.filtered[t], "filtered", t);
            let lp = result.log_predictive[t];
            assert!(lp.is_finite(), "log_predictive[{t}] = {lp} is not finite");
        }

        let sum_lp: f64 = result.log_predictive.iter().sum();
        assert!(
            (result.log_likelihood - sum_lp).abs() < 1e-10,
            "log_likelihood={:.12} != sum log_predictive={:.12}",
            result.log_likelihood,
            sum_lp
        );

        result
    }

    /// Argmax of a slice.
    fn argmax(v: &[f64]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Fraction of steps where argmax(filtered[t]) == true_states[t].
    fn accuracy(result: &FilterResult, true_states: &[usize]) -> f64 {
        let correct = (0..result.t)
            .filter(|&t| argmax(&result.filtered[t]) == true_states[t])
            .count();
        correct as f64 / result.t as f64
    }

    // -----------------------------------------------------------------------
    // Structural invariant tests - one per scenario family
    // -----------------------------------------------------------------------

    #[test]
    fn scenario_a_invariants() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.99, 0.01], vec![0.01, 0.99]],
            vec![-10.0, 10.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 1_000, &mut rng).unwrap();
        check_invariants(&params, &sim.observations);
    }

    #[test]
    fn scenario_b_invariants() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-0.5, 0.5],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 1_000, &mut rng).unwrap();
        check_invariants(&params, &sim.observations);
    }

    #[test]
    fn scenario_c_invariants() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.95, 0.05], vec![0.05, 0.95]],
            vec![-5.0, 5.0],
            vec![2.0, 2.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 1_000, &mut rng).unwrap();
        check_invariants(&params, &sim.observations);
    }

    #[test]
    fn scenario_d_invariants() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.95, 0.05], vec![0.05, 0.95]],
            vec![0.0, 0.0],
            vec![0.25, 9.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 1_000, &mut rng).unwrap();
        check_invariants(&params, &sim.observations);
    }

    #[test]
    fn scenario_e_invariants() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.999, 0.001], vec![0.001, 0.999]],
            vec![-3.0, 3.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 2_000, &mut rng).unwrap();
        check_invariants(&params, &sim.observations);
    }

    #[test]
    fn scenario_f_invariants() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.4, 0.6], vec![0.6, 0.4]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 1_000, &mut rng).unwrap();
        check_invariants(&params, &sim.observations);
    }

    #[test]
    fn scenario_g_invariants() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 10, &mut rng).unwrap();
        check_invariants(&params, &sim.observations);
    }

    #[test]
    fn scenario_h_invariants() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.99, 0.01], vec![0.01, 0.99]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 50_000, &mut rng).unwrap();
        check_invariants(&params, &sim.observations);
    }

    // -----------------------------------------------------------------------
    // Behavioral tests
    // -----------------------------------------------------------------------

    /// Scenario A - posterior beliefs should be very sharp when regimes are
    /// strongly separated (mu = +/-10, sigma^2 = 1).
    #[test]
    fn scenario_a_sharp_posteriors() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.99, 0.01], vec![0.01, 0.99]],
            vec![-10.0, 10.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 2_000, &mut rng).unwrap();
        let result = check_invariants(&params, &sim.observations);

        let mean_max: f64 = result
            .filtered
            .iter()
            .map(|f| f.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
            .sum::<f64>()
            / result.t as f64;

        assert!(
            mean_max > 0.97,
            "scenario A: mean max filtered prob = {mean_max:.4}, expected > 0.97"
        );
    }

    /// Scenario A - argmax filter should match the true hidden state >= 98% of the time.
    #[test]
    fn scenario_a_classification_accuracy() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.99, 0.01], vec![0.01, 0.99]],
            vec![-10.0, 10.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 5_000, &mut rng).unwrap();
        let result = check_invariants(&params, &sim.observations);

        let acc = accuracy(&result, &sim.states);
        assert!(
            acc > 0.98,
            "scenario A: accuracy = {acc:.4}, expected > 0.98"
        );
    }

    /// Scenario B - weakly separated params produce more diffuse posteriors
    /// than strongly separated params on comparable data.
    #[test]
    fn scenario_b_diffuse_vs_sharp_posteriors() {
        let params_strong = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-10.0, 10.0],
            vec![1.0, 1.0],
        );
        let params_weak = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-0.5, 0.5],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);

        let sim_s = simulate(params_strong.clone(), 2_000, &mut rng).unwrap();
        let res_s = check_invariants(&params_strong, &sim_s.observations);
        let mean_max_s: f64 = res_s
            .filtered
            .iter()
            .map(|f| f.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
            .sum::<f64>()
            / res_s.t as f64;

        let sim_w = simulate(params_weak.clone(), 2_000, &mut rng).unwrap();
        let res_w = check_invariants(&params_weak, &sim_w.observations);
        let mean_max_w: f64 = res_w
            .filtered
            .iter()
            .map(|f| f.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
            .sum::<f64>()
            / res_w.t as f64;

        assert!(
            mean_max_s > mean_max_w,
            "strong separation (mean_max={mean_max_s:.4}) should yield sharper \
             posteriors than weak separation (mean_max={mean_max_w:.4})"
        );
    }

    /// Scenario C - mean-switching only: argmax filter tracks location of observations.
    #[test]
    fn scenario_c_tracks_mean_location() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.95, 0.05], vec![0.05, 0.95]],
            vec![-5.0, 5.0],
            vec![2.0, 2.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 3_000, &mut rng).unwrap();
        let result = check_invariants(&params, &sim.observations);

        let acc = accuracy(&result, &sim.states);
        assert!(
            acc > 0.75,
            "scenario C: accuracy = {acc:.4}, expected > 0.75"
        );
    }

    /// Scenario D - a single large observation (y=3) should favor the high-variance
    /// regime when both means are zero.
    #[test]
    fn scenario_d_single_extreme_obs_favors_wide_regime() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.95, 0.05], vec![0.05, 0.95]],
            vec![0.0, 0.0],
            vec![0.25, 9.0],
        );
        let result = check_invariants(&params, &[3.0]);

        assert!(
            result.filtered[0][1] > result.filtered[0][0],
            "scenario D: wide-variance regime should dominate for y=3; \
             filtered[0][1]={:.4}, filtered[0][0]={:.4}",
            result.filtered[0][1],
            result.filtered[0][0]
        );
    }

    /// Scenario D (sequence) - log-likelihood must stay finite over a moderate run.
    #[test]
    fn scenario_d_variance_sequence_stable() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.95, 0.05], vec![0.05, 0.95]],
            vec![0.0, 0.0],
            vec![0.25, 9.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 2_000, &mut rng).unwrap();
        let result = check_invariants(&params, &sim.observations);
        assert!(
            result.log_likelihood.is_finite(),
            "scenario D: log_likelihood is not finite"
        );
    }

    /// Scenario E - very persistent transitions (p_ii=0.999) should produce
    /// small step-to-step changes in predicted probabilities.
    #[test]
    fn scenario_e_smooth_evolution() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.999, 0.001], vec![0.001, 0.999]],
            vec![-3.0, 3.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 5_000, &mut rng).unwrap();
        let result = check_invariants(&params, &sim.observations);

        let large_jumps = (1..result.t)
            .filter(|&t| (result.predicted[t][0] - result.filtered[t - 1][0]).abs() > 0.05)
            .count();
        let jump_frac = large_jumps as f64 / (result.t - 1) as f64;

        assert!(
            jump_frac < 0.005,
            "scenario E: {:.2}% of prediction steps jumped >0.05 - expected <=0.5%",
            jump_frac * 100.0
        );
    }

    /// Scenario E vs F - a switching filter should have larger step-to-step
    /// changes in filtered probabilities than a persistent filter on the same data.
    #[test]
    fn scenario_e_vs_f_persistence_controls_responsiveness() {
        let persistent = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.99, 0.01], vec![0.01, 0.99]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let switching = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.4, 0.6], vec![0.6, 0.4]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(persistent.clone(), 2_000, &mut rng).unwrap();

        let res_p = check_invariants(&persistent, &sim.observations);
        let res_s = check_invariants(&switching, &sim.observations);

        let mean_step_change = |res: &FilterResult| {
            (1..res.t)
                .map(|t| (res.filtered[t][0] - res.filtered[t - 1][0]).abs())
                .sum::<f64>()
                / (res.t - 1) as f64
        };

        let change_p = mean_step_change(&res_p);
        let change_s = mean_step_change(&res_s);

        assert!(
            change_s > change_p,
            "switching filter mean_change={change_s:.5} should exceed \
             persistent filter mean_change={change_p:.5}"
        );
    }

    /// Scenario F - a frequently-switching matrix (p_ij=0.4/0.6) should keep
    /// predicted probabilities close to the stationary distribution.
    #[test]
    fn scenario_f_predicted_close_to_stationary() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.4, 0.6], vec![0.6, 0.4]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 2_000, &mut rng).unwrap();
        let result = check_invariants(&params, &sim.observations);

        let mean_dist: f64 = result
            .predicted
            .iter()
            .map(|p| (p[0] - 0.5_f64).abs())
            .sum::<f64>()
            / result.t as f64;

        assert!(
            mean_dist < 0.20,
            "scenario F: mean |predicted[t][0] - 0.5| = {mean_dist:.4}, expected < 0.20"
        );
    }

    /// Scenario G - structural invariants hold across 50 independent short runs.
    #[test]
    fn scenario_g_multi_seed_short_runs() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        for seed in 0u64..50 {
            let mut rng = SmallRng::seed_from_u64(seed);
            let sim = simulate(params.clone(), 10, &mut rng).unwrap();
            check_invariants(&params, &sim.observations);
        }
    }

    /// Scenario H - last filtered distribution remains a valid probability
    /// vector after 50 000 steps.
    #[test]
    fn scenario_h_final_distribution_valid() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.99, 0.01], vec![0.01, 0.99]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 50_000, &mut rng).unwrap();
        let result = check_invariants(&params, &sim.observations);

        assert_prob_vec(
            result.filtered.last().unwrap(),
            "filtered[T-1]",
            result.t - 1,
        );
        assert!(
            result.log_likelihood.is_finite(),
            "log_likelihood not finite at T=50_000"
        );
    }

    /// Scenario H - per-step average log-likelihood should stay in a physically
    /// reasonable range over T=20_000.
    #[test]
    fn scenario_h_per_step_log_likelihood_stable() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.99, 0.01], vec![0.01, 0.99]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 20_000, &mut rng).unwrap();
        let ll = log_likelihood(&params, &sim.observations).unwrap();

        let per_step = ll / sim.t as f64;
        assert!(
            per_step > -10.0 && per_step < 0.0,
            "scenario H: per-step LL = {per_step:.4}, expected in (-10, 0)"
        );
    }

    // -----------------------------------------------------------------------
    // Regime-change boundary tests
    // -----------------------------------------------------------------------

    /// A hard observation block switch (100x-10 then 100x+10) should be
    /// clearly detected by the filter.
    #[test]
    fn posterior_shifts_at_hard_regime_change() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-10.0, 10.0],
            vec![1.0, 1.0],
        );
        let obs: Vec<f64> = vec![-10.0; 100]
            .into_iter()
            .chain(vec![10.0; 100])
            .collect();
        let result = check_invariants(&params, &obs);

        let early = result.filtered[90][0];
        let late = result.filtered[190][1];

        assert!(
            early > 0.95,
            "regime 0 filtered at t=90 = {early:.4}, expected > 0.95"
        );
        assert!(
            late > 0.95,
            "regime 1 filtered at t=190 = {late:.4}, expected > 0.95"
        );
    }

    /// Stronger emission evidence leads to faster posterior recovery after a switch.
    ///
    /// After 50 identical regime-0 observations the filter is near-certainty about
    /// regime 0 for both models.  Once the observations shift to the regime-1 mean,
    /// the model with large separation (mu=+-5, sigma=1 -> 10-sigma gap) should
    /// cross filtered[1] > 0.9 in fewer steps than the model with small separation
    /// (mu=+-0.5, sigma=1 -> 1-sigma gap) where each observation provides weak
    /// evidence and the filter updates slowly.
    #[test]
    fn stronger_evidence_speeds_posterior_recovery() {
        // Strong: mu = +-5, sigma^2 = 1  ->  y = +5 is 10 sigma from mu_0 = -5.
        // First obs after block switch drives filtered[1] to ~1 immediately.
        let params_strong = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        // Weak: mu = +-0.5, sigma^2 = 1  ->  y = +0.5 is only 1 sigma from mu_0 = -0.5.
        // Each post-switch obs provides a likelihood ratio of roughly e^0.5 ~ 1.6
        // so the filter accumulates evidence gradually over many steps.
        let params_weak = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![-0.5, 0.5],
            vec![1.0, 1.0],
        );

        // Observations placed exactly at the regime means to maximise the contrast.
        let obs_strong: Vec<f64> = vec![-5.0; 50].into_iter().chain(vec![5.0; 50]).collect();
        let obs_weak: Vec<f64> = vec![-0.5; 50].into_iter().chain(vec![0.5; 50]).collect();

        let res_s = check_invariants(&params_strong, &obs_strong);
        let res_w = check_invariants(&params_weak, &obs_weak);

        let steps_to_flip = |res: &FilterResult| {
            (50..res.t)
                .find(|&t| res.filtered[t][1] > 0.9)
                .map(|t| t - 50)
                .unwrap_or(res.t)
        };

        let steps_s = steps_to_flip(&res_s);
        let steps_w = steps_to_flip(&res_w);

        assert!(
            steps_s < steps_w,
            "strong evidence should flip faster: {steps_s} steps vs weak {steps_w} steps"
        );
    }

    /// Persistent transitions slow down posterior recovery at a regime change.
    #[test]
    fn persistent_transitions_slow_posterior_recovery() {
        let persistent = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.99, 0.01], vec![0.01, 0.99]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );
        let less_persistent = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.7, 0.3], vec![0.3, 0.7]],
            vec![-5.0, 5.0],
            vec![1.0, 1.0],
        );

        let obs: Vec<f64> = vec![-5.0; 30].into_iter().chain(vec![5.0; 70]).collect();

        let res_p = check_invariants(&persistent, &obs);
        let res_lp = check_invariants(&less_persistent, &obs);

        let first_flip = |res: &FilterResult| {
            (30..res.t)
                .find(|&t| res.filtered[t][1] > 0.8)
                .map(|t| t - 30)
                .unwrap_or(res.t)
        };

        let steps_p = first_flip(&res_p);
        let steps_lp = first_flip(&res_lp);

        assert!(
            steps_p >= steps_lp,
            "persistent ({steps_p} steps) should be at least as slow as \
             less-persistent ({steps_lp} steps) at recovering after a switch"
        );
    }

    // -----------------------------------------------------------------------
    // Three-regime (K=3) validation
    // -----------------------------------------------------------------------

    /// K=3 structural invariants hold over a moderate-length run.
    #[test]
    fn three_regime_invariants() {
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
        let sim = simulate(params.clone(), 3_000, &mut rng).unwrap();
        let result = check_invariants(&params, &sim.observations);

        for t in 0..result.t {
            assert_eq!(
                result.filtered[t].len(),
                3,
                "three-regime: filtered[{t}] wrong length"
            );
        }
    }

    /// K=3 argmax classification accuracy on strongly separated, persistent data.
    #[test]
    fn three_regime_classification_accuracy() {
        let params = ModelParams::new(
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            vec![
                vec![0.95, 0.025, 0.025],
                vec![0.025, 0.95, 0.025],
                vec![0.025, 0.025, 0.95],
            ],
            vec![-8.0, 0.0, 8.0],
            vec![1.0, 1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params.clone(), 5_000, &mut rng).unwrap();
        let result = check_invariants(&params, &sim.observations);

        let acc = accuracy(&result, &sim.states);
        assert!(
            acc > 0.90,
            "K=3 classification accuracy = {acc:.4}, expected > 0.90"
        );
    }
}
