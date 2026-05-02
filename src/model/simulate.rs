#![allow(dead_code)]
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform, weighted::WeightedIndex};

use super::params::ModelParams;

/// Optional jump-contamination parameters for `simulate_with_jump`.
///
/// After each base Gaussian draw, with probability `prob` an independent shock
/// N(0, (scale_mult × σⱼ)²) is added, where σⱼ is the std-dev of the active
/// regime.  This models outlier contamination without altering the Markov
/// transition structure.
#[derive(Debug, Clone, PartialEq)]
pub struct JumpParams {
    /// Probability that any single observation receives a jump shock.
    pub prob: f64,
    /// Shock std-dev expressed as a multiplier of the active-regime std-dev.
    pub scale_mult: f64,
}

/// The complete output of one simulation run.
///
/// Both the hidden path and the observations are retained so they can later
/// be used as ground truth when testing filtering, smoothing, and EM.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// T — number of time steps.
    pub t: usize,
    /// K — number of regimes (copied from params for convenience).
    pub k: usize,
    /// Hidden regime path S₁,…,S_T (0-based indices into 0..k).
    pub states: Vec<usize>,
    /// Observation sequence y₁,…,y_T.
    pub observations: Vec<f64>,
    /// The parameter set that generated this sample.
    pub params: ModelParams,
}

/// Simulate the full hidden regime path S₁,…,S_T.
///
/// - S₁ ~ π  (categorical draw from the initial distribution)
/// - Sₜ | S_{t-1}=i ~ Categorical(P[i,·])  for t = 2,…,T
fn simulate_hidden_path(params: &ModelParams, t: usize, rng: &mut impl Rng) -> Vec<usize> {
    let mut states = Vec::with_capacity(t);

    // Step 2 of the spec: draw S₁ ~ π.
    let init_dist = WeightedIndex::new(&params.pi).expect("pi was already validated");
    states.push(init_dist.sample(rng));

    // Step 3: evolve the chain for t=2,…,T.
    for _ in 1..t {
        let prev = *states.last().unwrap();
        let row = params.transition_row(prev);
        let row_dist = WeightedIndex::new(row).expect("transition row was already validated");
        states.push(row_dist.sample(rng));
    }

    states
}

/// Simulate the observation sequence y₁,…,y_T given the hidden path.
///
/// yₜ | Sₜ=j ~ N(μⱼ, σⱼ²)
///
/// Note: `rand_distr::Normal::new` takes (mean, std_dev), so we pass σⱼ = √σⱼ².
fn simulate_observations(params: &ModelParams, states: &[usize], rng: &mut impl Rng) -> Vec<f64> {
    states
        .iter()
        .map(|&j| {
            let mu = params.means[j];
            let sigma = params.variances[j].sqrt();
            let dist = Normal::new(mu, sigma).expect("variance was already validated");
            dist.sample(rng)
        })
        .collect()
}

/// Simulate a complete sample from the Gaussian Markov Switching Model.
///
/// Validates `params`, then runs:
/// 1. Layer A — hidden regime path via [`simulate_hidden_path`]
/// 2. Layer B — observations via [`simulate_observations`]
///
/// `rng` is caller-supplied so the caller controls seeding and reproducibility.
pub fn simulate(
    params: ModelParams,
    t: usize,
    rng: &mut impl Rng,
) -> anyhow::Result<SimulationResult> {
    if t < 1 {
        anyhow::bail!("simulate: T must be ≥ 1, got {t}");
    }
    params.validate()?;

    let states = simulate_hidden_path(&params, t, rng);
    let observations = simulate_observations(&params, &states, rng);
    let k = params.k;

    Ok(SimulationResult {
        t,
        k,
        states,
        observations,
        params,
    })
}

/// Simulate with optional Bernoulli jump contamination.
///
/// Equivalent to [`simulate`] when `jump` is `None`.  When `jump` is `Some`,
/// each observation has independent probability `jump.prob` of receiving an
/// additive shock drawn from N(0, (jump.scale_mult × σⱼ)²), where σⱼ is the
/// std-dev of the active regime at that time step.
pub fn simulate_with_jump(
    params: ModelParams,
    t: usize,
    rng: &mut impl Rng,
    jump: Option<&JumpParams>,
) -> anyhow::Result<SimulationResult> {
    if t < 1 {
        anyhow::bail!("simulate_with_jump: T must be ≥ 1, got {t}");
    }
    params.validate()?;

    let states = simulate_hidden_path(&params, t, rng);
    let mut observations = simulate_observations(&params, &states, rng);

    if let Some(j) = jump {
        let uniform = Uniform::new(0.0_f64, 1.0).expect("valid uniform range");
        for (obs, &state) in observations.iter_mut().zip(states.iter()) {
            if uniform.sample(rng) < j.prob {
                let regime_std = params.variances[state].sqrt();
                let shock_std = j.scale_mult * regime_std;
                let shock_dist = Normal::new(0.0_f64, shock_std).expect("shock_std > 0");
                *obs += shock_dist.sample(rng);
            }
        }
    }

    let k = params.k;
    Ok(SimulationResult {
        t,
        k,
        states,
        observations,
        params,
    })
}

// ---------------------------------------------------------------------------
// Tests — all six sanity checks from the spec
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    const SEED: u64 = 42;

    fn persistent_2state() -> ModelParams {
        ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.99, 0.01], vec![0.01, 0.99]],
            vec![0.0, 5.0],
            vec![1.0, 1.0],
        )
    }

    fn switching_2state() -> ModelParams {
        ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.5, 0.5], vec![0.5, 0.5]],
            vec![0.0, 5.0],
            vec![1.0, 1.0],
        )
    }

    // ------------------------------------------------------------------
    // Test A — Initial-state distribution
    // ------------------------------------------------------------------
    #[test]
    fn test_a_initial_state_distribution() {
        let pi = vec![0.3, 0.7];
        let params = ModelParams::new(
            pi.clone(),
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let n_runs = 20_000;
        let mut counts = vec![0usize; 2];
        for _ in 0..n_runs {
            let result = simulate(params.clone(), 1, &mut rng).unwrap();
            counts[result.states[0]] += 1;
        }
        for j in 0..2 {
            let empirical = counts[j] as f64 / n_runs as f64;
            // Allow ±3% absolute tolerance.
            assert!(
                (empirical - pi[j]).abs() < 0.03,
                "regime {j}: empirical π̂={empirical:.4}, expected π={:.4}",
                pi[j]
            );
        }
    }

    // ------------------------------------------------------------------
    // Test B — Empirical transition frequencies match P
    // ------------------------------------------------------------------
    #[test]
    fn test_b_transition_frequencies() {
        let params = persistent_2state();
        let mut rng = SmallRng::seed_from_u64(SEED);
        let result = simulate(params.clone(), 100_000, &mut rng).unwrap();

        // Count transitions i→j.
        let mut counts = vec![vec![0usize; 2]; 2];
        for t in 1..result.states.len() {
            counts[result.states[t - 1]][result.states[t]] += 1;
        }
        for i in 0..2 {
            let row_total: usize = counts[i].iter().sum();
            for j in 0..2 {
                let empirical = counts[i][j] as f64 / row_total as f64;
                let expected = params.transition_row(i)[j];
                assert!(
                    (empirical - expected).abs() < 0.01,
                    "P[{i},{j}]: empirical={empirical:.4}, expected={expected:.4}"
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // Test C — Persistence: long diagonal → long runs
    // ------------------------------------------------------------------
    #[test]
    fn test_c_persistence() {
        let params = persistent_2state(); // p_ii = 0.99 → E[run] ≈ 100
        let mut rng = SmallRng::seed_from_u64(SEED);
        let result = simulate(params, 200_000, &mut rng).unwrap();

        let mean_run = mean_run_length(&result.states);
        assert!(
            mean_run > 50.0,
            "expected mean run length > 50 for p_ii=0.99, got {mean_run:.1}"
        );
    }

    // ------------------------------------------------------------------
    // Test D — Switching: equal probabilities → short runs
    // ------------------------------------------------------------------
    #[test]
    fn test_d_switching() {
        let params = switching_2state(); // p_ij = 0.5 → E[run] ≈ 2
        let mut rng = SmallRng::seed_from_u64(SEED);
        let result = simulate(params, 100_000, &mut rng).unwrap();

        let mean_run = mean_run_length(&result.states);
        assert!(
            mean_run < 5.0,
            "expected mean run length < 5 for uniform switching, got {mean_run:.1}"
        );
    }

    // ------------------------------------------------------------------
    // Test E — Emission means: per-regime sample mean ≈ μⱼ
    // ------------------------------------------------------------------
    #[test]
    fn test_e_emission_means() {
        let means = vec![-10.0_f64, 10.0];
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            means.clone(),
            vec![1.0, 1.0],
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let result = simulate(params, 50_000, &mut rng).unwrap();

        for j in 0..2 {
            let obs_j: Vec<f64> = (0..result.t)
                .filter(|&i| result.states[i] == j)
                .map(|i| result.observations[i])
                .collect();
            let n = obs_j.len() as f64;
            let sample_mean = obs_j.iter().sum::<f64>() / n;
            assert!(
                (sample_mean - means[j]).abs() < 0.1,
                "regime {j}: sample mean={sample_mean:.4}, μ={}",
                means[j]
            );
        }
    }

    // ------------------------------------------------------------------
    // Test F — Emission variances: per-regime sample variance ≈ σⱼ²
    // ------------------------------------------------------------------
    #[test]
    fn test_f_emission_variances() {
        let variances = vec![0.1_f64, 10.0];
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![0.0, 0.0], // equal means to isolate variance
            variances.clone(),
        );
        let mut rng = SmallRng::seed_from_u64(SEED);
        let result = simulate(params, 50_000, &mut rng).unwrap();

        for j in 0..2 {
            let obs_j: Vec<f64> = (0..result.t)
                .filter(|&i| result.states[i] == j)
                .map(|i| result.observations[i])
                .collect();
            let n = obs_j.len() as f64;
            let mean = obs_j.iter().sum::<f64>() / n;
            let sample_var = obs_j.iter().map(|&y| (y - mean).powi(2)).sum::<f64>() / (n - 1.0);
            // Allow 10% relative error.
            let rel_err = (sample_var - variances[j]).abs() / variances[j];
            assert!(
                rel_err < 0.10,
                "regime {j}: sample var={sample_var:.4}, σ²={:.4}, rel_err={rel_err:.4}",
                variances[j]
            );
        }
    }

    // ------------------------------------------------------------------
    // Validation tests
    // ------------------------------------------------------------------
    #[test]
    fn test_validate_rejects_k1() {
        let p = ModelParams::new(vec![1.0], vec![vec![1.0]], vec![0.0], vec![1.0]);
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_bad_pi_sum() {
        let p = ModelParams::new(
            vec![0.4, 0.4], // sums to 0.8, not 1
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        );
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_bad_transition_row() {
        let p = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.8, 0.1], vec![0.1, 0.9]], // row 0 sums to 0.9
            vec![0.0, 1.0],
            vec![1.0, 1.0],
        );
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_nonpositive_variance() {
        let p = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![0.0, 1.0],
            vec![1.0, 0.0], // σ²=0 is invalid
        );
        assert!(p.validate().is_err());
    }

    // ------------------------------------------------------------------
    // Helper
    // ------------------------------------------------------------------
    fn mean_run_length(states: &[usize]) -> f64 {
        if states.is_empty() {
            return 0.0;
        }
        let mut runs = 0usize;
        for i in 1..states.len() {
            if states[i] != states[i - 1] {
                runs += 1;
            }
        }
        // Count the final run.
        runs += 1;
        states.len() as f64 / runs as f64
    }
}
