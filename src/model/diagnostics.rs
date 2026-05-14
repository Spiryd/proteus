/// Diagnostics and trust-check layer for the Gaussian Markov Switching Model.
///
/// # Purpose
///
/// Phase 10 turns a fitted model into a *trustworthy* one.  Fitting can
/// converge numerically yet still produce degenerate, uninterpretable, or
/// unstable solutions.  This module answers six questions:
///
/// 1. **Parameter validity** — is Θ̂ mathematically valid?
/// 2. **Posterior coherence** — are the filtered/smoothed/pairwise outputs
///    internally consistent?
/// 3. **EM convergence** — did the optimization path behave credibly?
/// 4. **Regime interpretation** — are the regimes meaningful and persistent?
/// 5. **Multi-start stability** — do different starts agree on the solution?
/// 6. **Trust report** — can the fitted model be inspected and justified?
///
/// # Architecture
///
/// This module is a pure *consumer*.  It calls `filter`, `smooth`, and
/// `pairwise` to run a final inference pass under the fitted parameters, but
/// nothing in the inference stack calls back into `diagnostics`.
///
/// # Main entry points
///
/// - [`diagnose`] — full diagnostics bundle for a single fitted model.
/// - [`compare_runs`] — cross-run comparison for multi-start robustness.
use anyhow::Result;

use super::em::EmResult;
use super::filter::filter;
use super::pairwise::pairwise;
use super::params::ModelParams;
use super::smoother::smooth;

// ===========================================================================
// Diagnostic thresholds (named constants for transparency)
// ===========================================================================

/// Variance below this threshold triggers [`DiagnosticWarning::NearZeroVariance`].
const NEAR_ZERO_VAR_THRESHOLD: f64 = 1e-4;

/// Posterior occupancy share below this triggers
/// [`DiagnosticWarning::NearlyUnusedRegime`].
const NEARLY_UNUSED_REGIME_SHARE: f64 = 0.01;

/// Self-transition probability above this triggers
/// [`DiagnosticWarning::SuspiciousPersistence`].
const SUSPICIOUS_PERSISTENCE_P: f64 = 1.0 - 1e-6;

/// Log-likelihood spread across multiple starts above which
/// [`DiagnosticWarning::UnstableAcrossStarts`] is raised.
const UNSTABLE_STARTS_LL_GAP: f64 = 1.0;

/// Maximum posterior deviation from exact normalization considered acceptable.
/// Used to set `FittedModelDiagnostics::is_trustworthy`.
const POSTERIOR_TOL: f64 = 1e-8;

/// Tolerance for the EM monotonicity check: a log-likelihood decrease larger
/// than this across one iteration is considered a violation.
const MONOTONE_TOL: f64 = 1e-8;

// ===========================================================================
// Stop reason
// ===========================================================================

/// The reason the EM iteration loop terminated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// Log-likelihood change fell below the configured tolerance.
    Converged,
    /// Maximum iteration count was reached before the tolerance was met.
    IterationCap,
}

// ===========================================================================
// Diagnostic warnings
// ===========================================================================

/// A warning condition detected during diagnostics.
///
/// Warnings are non-fatal: the fitted model may still be usable, but each
/// variant signals a specific concern about numerical or substantive quality.
#[derive(Debug, Clone)]
pub enum DiagnosticWarning {
    /// Regime `j` has a fitted variance below [`NEAR_ZERO_VAR_THRESHOLD`],
    /// suggesting numerical instability or overfitting.
    NearZeroVariance { regime: usize, value: f64 },

    /// Regime `j` contributes less than [`NEARLY_UNUSED_REGIME_SHARE`] of the
    /// total posterior mass, suggesting it is effectively unused or redundant.
    NearlyUnusedRegime { regime: usize, occupancy_share: f64 },

    /// The log-likelihood decreased by `drop` between EM iterations `iteration-1`
    /// and `iteration`, violating the EM ascent guarantee beyond
    /// [`MONOTONE_TOL`].
    EmNonMonotonicity { iteration: usize, drop: f64 },

    /// Regime `j` has a self-transition probability above
    /// [`SUSPICIOUS_PERSISTENCE_P`], implying a very large (possibly infinite)
    /// expected duration that may reflect degeneracy.
    SuspiciousPersistence {
        regime: usize,
        p_self: f64,
        expected_duration: f64,
    },

    /// Multiple EM runs converged to substantially different log-likelihood
    /// values, suggesting the solution is sensitive to initialization.
    UnstableAcrossStarts { ll_spread: f64 },
}

impl DiagnosticWarning {
    /// Returns a human-readable description of the warning including all
    /// payload fields, suitable for embedding in JSON artifacts and reports.
    pub fn description(&self) -> String {
        match self {
            Self::NearZeroVariance { regime, value } => {
                format!("Regime {regime} has near-zero variance {value:.2e}")
            }
            Self::NearlyUnusedRegime {
                regime,
                occupancy_share,
            } => {
                format!("Regime {regime} occupancy share {occupancy_share:.4} is nearly zero")
            }
            Self::EmNonMonotonicity { iteration, drop } => {
                format!("EM non-monotonicity at iteration {iteration}: drop = {drop:.2e}")
            }
            Self::SuspiciousPersistence {
                regime,
                p_self,
                expected_duration,
            } => {
                format!(
                    "Regime {regime} self-transition p={p_self:.6}, expected duration = {expected_duration:.1}"
                )
            }
            Self::UnstableAcrossStarts { ll_spread } => {
                format!("Log-likelihood spread across starts = {ll_spread:.4}")
            }
        }
    }
}

// ===========================================================================
// Diagnostic structs
// ===========================================================================

// ---------------------------------------------------------------------------
// Parameter validity layer
// ---------------------------------------------------------------------------

/// Results of the structural parameter validity check.
///
/// Covers π normalization, P row-stochasticity, variance positivity, and
/// parameter finiteness.
#[derive(Debug, Clone)]
pub struct ParamValidity {
    /// `true` iff all parameter constraints are satisfied.
    pub valid: bool,
    /// `|Σⱼ πⱼ − 1|`
    pub max_pi_dev: f64,
    /// `max_i |Σⱼ p_{ij} − 1|`
    pub max_row_dev: f64,
    /// Minimum variance across all regimes `min_j σⱼ²`.
    pub min_variance: f64,
    /// `true` iff every mean, variance, π entry, and transition entry is finite.
    pub all_params_finite: bool,
}

// ---------------------------------------------------------------------------
// Posterior validity layer
// ---------------------------------------------------------------------------

/// Results of the posterior probability consistency check.
///
/// All fields are maximum absolute deviations from the exact normalisation
/// or consistency condition.
#[derive(Debug, Clone)]
pub struct PosteriorValidity {
    /// `max_t |Σⱼ filtered[t][j] − 1|`
    pub max_filtered_dev: f64,
    /// `max_t |Σⱼ smoothed[t][j] − 1|`
    pub max_smoothed_dev: f64,
    /// `max_t |Σᵢⱼ ξₜ(i,j) − 1|`
    pub max_pairwise_dev: f64,
    /// `max_{t,j} |Σᵢ ξₜ(i,j) − γₜ(j)|`  (column marginal consistency).
    pub max_marginal_consistency_err: f64,
}

// ---------------------------------------------------------------------------
// EM convergence layer
// ---------------------------------------------------------------------------

/// Summary of the EM optimization path.
#[derive(Debug, Clone)]
pub struct ConvergenceSummary {
    /// Why the EM loop stopped.
    pub stop_reason: StopReason,
    /// Number of E-step / M-step pairs performed.
    pub n_iter: usize,
    /// Log-likelihood at iteration 0 (before any M-step).
    pub initial_ll: f64,
    /// Log-likelihood at the final iteration.
    pub final_ll: f64,
    /// `final_ll − initial_ll`
    pub ll_gain: f64,
    /// Smallest log-likelihood increment across consecutive iteration pairs.
    pub min_delta: f64,
    /// Magnitude of the largest *negative* log-likelihood drop.
    /// `0.0` when the history is fully monotone.
    pub largest_negative_delta: f64,
    /// `true` iff no drop exceeds [`MONOTONE_TOL`].
    pub is_monotone: bool,
}

// ---------------------------------------------------------------------------
// Regime interpretation layer
// ---------------------------------------------------------------------------

/// Interpretive summaries for each regime after fitting.
#[derive(Debug, Clone)]
pub struct RegimeSummary {
    /// Fitted regime means μⱼ (0-based).
    pub means: Vec<f64>,
    /// Fitted regime variances σⱼ² (0-based).
    pub variances: Vec<f64>,
    /// `Wⱼ = Σₜ γₜ(j)` — total posterior mass attributed to regime j.
    pub occupancy_weights: Vec<f64>,
    /// `Wⱼ / T` — expected fraction of time spent in regime j.
    pub occupancy_shares: Vec<f64>,
    /// Diagonal of P: `(p_{11}, …, p_{KK})`.
    pub self_transition_probs: Vec<f64>,
    /// `1 / (1 − p_{jj})` — expected duration in regime j.
    /// `f64::INFINITY` when `p_{jj} = 1.0`.
    pub expected_durations: Vec<f64>,
    /// Number of time steps at which regime j achieves the highest smoothed
    /// probability (hard-assignment count).
    pub hard_counts: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Top-level diagnostics bundle
// ---------------------------------------------------------------------------

/// Complete diagnostics bundle for a single fitted model.
///
/// Returned by [`diagnose`].
#[derive(Debug, Clone)]
pub struct FittedModelDiagnostics {
    /// Parameter validity check results.
    pub param_validity: ParamValidity,
    /// Posterior probability consistency check results.
    pub posterior_validity: PosteriorValidity,
    /// EM optimization path summary.
    pub convergence: ConvergenceSummary,
    /// Per-regime interpretation summaries.
    pub regimes: RegimeSummary,
    /// All warning conditions detected during diagnostics.
    pub warnings: Vec<DiagnosticWarning>,
    /// `true` iff parameter validity passes and all posterior deviations are
    /// below [`POSTERIOR_TOL`].
    pub is_trustworthy: bool,
}

// ---------------------------------------------------------------------------
// Multi-start comparison types
// ---------------------------------------------------------------------------

/// Diagnostics summary for a single EM run, suitable for cross-run comparison.
///
/// Regimes are reordered by increasing mean so runs are comparable across
/// different label permutations.
#[derive(Debug, Clone)]
pub struct RunSummary {
    /// Final observed-data log-likelihood.
    pub log_likelihood: f64,
    /// Number of EM iterations performed.
    pub n_iter: usize,
    /// Whether the run met the convergence tolerance.
    pub converged: bool,
    /// Fitted means reordered by increasing mean.
    pub ordered_means: Vec<f64>,
    /// Fitted variances reordered to match `ordered_means`.
    pub ordered_variances: Vec<f64>,
    /// Expected durations `1/(1−p_{jj})` reordered to match `ordered_means`.
    pub expected_durations: Vec<f64>,
    /// Posterior occupancy shares `Wⱼ/T` reordered to match `ordered_means`.
    pub occupancy_shares: Vec<f64>,
}

/// Comparison summary across multiple EM runs.
///
/// Returned by [`compare_runs`].
#[derive(Debug, Clone)]
pub struct MultiStartSummary {
    /// Individual run summaries sorted by log-likelihood descending (best first).
    pub runs: Vec<RunSummary>,
    /// Log-likelihood of the best run.
    pub best_ll: f64,
    /// Log-likelihood of the second-best run.
    /// `f64::NEG_INFINITY` when only one run is provided.
    pub runner_up_ll: f64,
    /// `best_ll − worst_ll`
    pub ll_spread: f64,
    /// `best_ll − runner_up_ll`
    pub top2_gap: f64,
    /// Number of runs that met the convergence tolerance.
    pub n_converged: usize,
    /// Warnings from the multi-run comparison (e.g. instability across starts).
    pub warnings: Vec<DiagnosticWarning>,
}

// ===========================================================================
// Public entry points
// ===========================================================================

/// Compute the full diagnostics bundle for a single fitted model.
///
/// Runs one inference pass (filter → smooth → pairwise) under the fitted
/// parameters to compute posterior consistency metrics.  Everything else is
/// derived from `result` alone.
///
/// # Errors
///
/// Returns an error if the inference pass fails (e.g. invalid parameters or
/// an observation incompatible with the fitted model).
pub fn diagnose(result: &EmResult, obs: &[f64]) -> Result<FittedModelDiagnostics> {
    let params = &result.params;

    // Parameter validity — no inference needed.
    let pv = check_param_validity(params);

    // Single inference pass under the fitted parameters.
    let fr = filter(params, obs)?;
    let sr = smooth(params, &fr)?;
    let pr = pairwise(params, &fr, &sr)?;

    // Posterior consistency — derived from the inference outputs.
    let post = compute_posterior_validity(&fr.filtered, &sr.smoothed, &pr.xi, params.k);

    // Convergence summary — derived from ll_history.
    let conv = summarize_convergence(result);

    // Regime interpretation — requires smoothed output.
    let regimes = summarize_regimes(params, &sr.smoothed);

    // Warnings — aggregate over all layers.
    let warnings = collect_warnings(&pv, &regimes, &conv, &result.ll_history);

    // Trust flag: validity + tight posterior normalization.
    let is_trustworthy = pv.valid
        && post.max_filtered_dev < POSTERIOR_TOL
        && post.max_smoothed_dev < POSTERIOR_TOL
        && post.max_pairwise_dev < POSTERIOR_TOL
        && post.max_marginal_consistency_err < POSTERIOR_TOL;

    Ok(FittedModelDiagnostics {
        param_validity: pv,
        posterior_validity: post,
        convergence: conv,
        regimes,
        warnings,
        is_trustworthy,
    })
}

/// Build a multi-start comparison summary from a slice of EM results.
///
/// Each run's regime parameters are reordered by increasing mean to make
/// runs comparable across label permutations.  Results are sorted by
/// log-likelihood descending (best first).
///
/// An inference pass is performed per run to compute posterior occupancy
/// shares; pass `obs` (the same sequence used for fitting).
///
/// # Errors
///
/// Returns an error if any inference pass fails.
pub fn compare_runs(results: &[EmResult], obs: &[f64]) -> Result<MultiStartSummary> {
    let mut runs: Vec<RunSummary> = results
        .iter()
        .map(|r| build_run_summary(r, obs))
        .collect::<Result<Vec<_>>>()?;

    // Sort best-first by log-likelihood.
    runs.sort_by(|a, b| {
        b.log_likelihood
            .partial_cmp(&a.log_likelihood)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let best_ll = runs.first().map_or(f64::NEG_INFINITY, |r| r.log_likelihood);
    let runner_up_ll = runs.get(1).map_or(f64::NEG_INFINITY, |r| r.log_likelihood);
    let worst_ll = runs.last().map_or(f64::NEG_INFINITY, |r| r.log_likelihood);

    let ll_spread = if runs.len() > 1 {
        best_ll - worst_ll
    } else {
        0.0
    };
    let top2_gap = if runs.len() > 1 {
        best_ll - runner_up_ll
    } else {
        0.0
    };

    let n_converged = results.iter().filter(|r| r.converged).count();

    let mut warnings = Vec::new();
    if runs.len() > 1 && ll_spread > UNSTABLE_STARTS_LL_GAP {
        warnings.push(DiagnosticWarning::UnstableAcrossStarts { ll_spread });
    }

    Ok(MultiStartSummary {
        runs,
        best_ll,
        runner_up_ll,
        ll_spread,
        top2_gap,
        n_converged,
        warnings,
    })
}

// ===========================================================================
// Private helpers
// ===========================================================================

fn check_param_validity(params: &ModelParams) -> ParamValidity {
    let k = params.k;

    let pi_sum: f64 = params.pi.iter().sum();
    let max_pi_dev = (pi_sum - 1.0).abs();

    let max_row_dev = (0..k)
        .map(|i| {
            let row_sum: f64 = params.transition_row(i).iter().sum();
            (row_sum - 1.0).abs()
        })
        .fold(0.0_f64, f64::max);

    let min_variance = params
        .variances
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    let all_params_finite = params.means.iter().all(|v| v.is_finite())
        && params.variances.iter().all(|v| v.is_finite())
        && params.pi.iter().all(|v| v.is_finite())
        && params.transition.iter().all(|v| v.is_finite());

    // A fit is valid when validate() passes (structural constraints) AND all
    // parameters are finite (catches NaN / ±∞ that validate() may miss).
    let valid = params.validate().is_ok() && all_params_finite;

    ParamValidity {
        valid,
        max_pi_dev,
        max_row_dev,
        min_variance,
        all_params_finite,
    }
}

/// Compute posterior validity metrics directly from inference outputs.
///
/// Arguments use references to the raw probability arrays rather than the
/// result structs so this function does not need to re-run inference.
fn compute_posterior_validity(
    filtered: &[Vec<f64>],
    smoothed: &[Vec<f64>],
    xi: &[Vec<Vec<f64>>],
    k: usize,
) -> PosteriorValidity {
    // Filtered normalization.
    let max_filtered_dev = filtered
        .iter()
        .map(|row| (row.iter().sum::<f64>() - 1.0).abs())
        .fold(0.0_f64, f64::max);

    // Smoothed normalization.
    let max_smoothed_dev = smoothed
        .iter()
        .map(|row| (row.iter().sum::<f64>() - 1.0).abs())
        .fold(0.0_f64, f64::max);

    // Pairwise normalization.
    let max_pairwise_dev = if xi.is_empty() {
        0.0
    } else {
        xi.iter()
            .map(|mat| {
                let total: f64 = mat.iter().flat_map(|row| row.iter()).sum();
                (total - 1.0).abs()
            })
            .fold(0.0_f64, f64::max)
    };

    // Column marginal consistency: Σᵢ ξₜ(i,j) = γₜ(j).
    //
    // xi[s] covers math t = s+2; smoothed[s+1] = γ_{s+2}(j).
    let max_marginal_consistency_err = if xi.is_empty() {
        0.0
    } else {
        let mut max_err = 0.0_f64;
        for (s, xi_step) in xi.iter().enumerate() {
            for j in 0..k {
                let col_sum: f64 = (0..k).map(|i| xi_step[i][j]).sum();
                let gamma_j = smoothed[s + 1][j];
                let err = (col_sum - gamma_j).abs();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        max_err
    };

    PosteriorValidity {
        max_filtered_dev,
        max_smoothed_dev,
        max_pairwise_dev,
        max_marginal_consistency_err,
    }
}

fn summarize_convergence(result: &EmResult) -> ConvergenceSummary {
    let hist = &result.ll_history;

    let initial_ll = *hist.first().unwrap_or(&f64::NEG_INFINITY);
    let final_ll = *hist.last().unwrap_or(&f64::NEG_INFINITY);
    let ll_gain = final_ll - initial_ll;

    let stop_reason = if result.converged {
        StopReason::Converged
    } else {
        StopReason::IterationCap
    };

    let mut min_delta = if hist.len() >= 2 { f64::INFINITY } else { 0.0 };
    let mut largest_negative_delta = 0.0_f64;
    let mut is_monotone = true;

    for window in hist.windows(2) {
        let delta = window[1] - window[0];
        if delta < min_delta {
            min_delta = delta;
        }
        if delta < -MONOTONE_TOL {
            is_monotone = false;
            let drop = -delta;
            if drop > largest_negative_delta {
                largest_negative_delta = drop;
            }
        }
    }

    ConvergenceSummary {
        stop_reason,
        n_iter: result.n_iter,
        initial_ll,
        final_ll,
        ll_gain,
        min_delta,
        largest_negative_delta,
        is_monotone,
    }
}

fn summarize_regimes(params: &ModelParams, smoothed: &[Vec<f64>]) -> RegimeSummary {
    let k = params.k;
    let t = smoothed.len();

    // Wⱼ = Σₜ γₜ(j)
    let occupancy_weights: Vec<f64> = (0..k)
        .map(|j| smoothed.iter().map(|row| row[j]).sum())
        .collect();

    let t_f64 = t as f64;
    let occupancy_shares: Vec<f64> = occupancy_weights.iter().map(|&w| w / t_f64).collect();

    let self_transition_probs: Vec<f64> = (0..k).map(|j| params.transition_row(j)[j]).collect();

    let expected_durations: Vec<f64> = self_transition_probs
        .iter()
        .map(|&p| compute_expected_duration(p))
        .collect();

    // Hard counts: argmax γₜ(j) at each time step.
    let mut hard_counts = vec![0usize; k];
    for row in smoothed {
        let best = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx);
        hard_counts[best] += 1;
    }

    RegimeSummary {
        means: params.means.clone(),
        variances: params.variances.clone(),
        occupancy_weights,
        occupancy_shares,
        self_transition_probs,
        expected_durations,
        hard_counts,
    }
}

fn collect_warnings(
    pv: &ParamValidity,
    regimes: &RegimeSummary,
    conv: &ConvergenceSummary,
    ll_history: &[f64],
) -> Vec<DiagnosticWarning> {
    let _ = (pv, conv); // fields used implicitly via regimes / ll_history
    let mut warnings = Vec::new();

    // Near-zero variance.
    for (j, &var) in regimes.variances.iter().enumerate() {
        if var < NEAR_ZERO_VAR_THRESHOLD {
            warnings.push(DiagnosticWarning::NearZeroVariance {
                regime: j,
                value: var,
            });
        }
    }

    // Nearly unused regimes.
    for (j, &share) in regimes.occupancy_shares.iter().enumerate() {
        if share < NEARLY_UNUSED_REGIME_SHARE {
            warnings.push(DiagnosticWarning::NearlyUnusedRegime {
                regime: j,
                occupancy_share: share,
            });
        }
    }

    // EM non-monotonicity: one warning per violating iteration.
    for (i, window) in ll_history.windows(2).enumerate() {
        let delta = window[1] - window[0];
        if delta < -MONOTONE_TOL {
            warnings.push(DiagnosticWarning::EmNonMonotonicity {
                iteration: i + 1,
                drop: -delta,
            });
        }
    }

    // Suspicious persistence.
    for (j, &p) in regimes.self_transition_probs.iter().enumerate() {
        if p > SUSPICIOUS_PERSISTENCE_P {
            warnings.push(DiagnosticWarning::SuspiciousPersistence {
                regime: j,
                p_self: p,
                expected_duration: regimes.expected_durations[j],
            });
        }
    }

    warnings
}

/// Compute the expected duration for a regime with self-transition probability
/// `p_self`.
///
/// Returns `f64::INFINITY` for an absorbing regime (`p_self == 1.0`).
fn compute_expected_duration(p_self: f64) -> f64 {
    let denom = 1.0 - p_self;
    if denom.abs() < f64::EPSILON {
        f64::INFINITY
    } else {
        1.0 / denom
    }
}

/// Return the canonical regime ordering: a permutation of `0..K` sorted by
/// increasing fitted mean.
///
/// Used to align regimes across different EM runs that may have permuted labels.
fn canonical_regime_order(params: &ModelParams) -> Vec<usize> {
    let mut order: Vec<usize> = (0..params.k).collect();
    order.sort_by(|&a, &b| {
        params.means[a]
            .partial_cmp(&params.means[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order
}

/// Build a [`RunSummary`] for one EM result, with regimes reordered by
/// increasing mean and occupancy computed from a final inference pass.
fn build_run_summary(result: &EmResult, obs: &[f64]) -> Result<RunSummary> {
    let params = &result.params;
    let order = canonical_regime_order(params);

    // Run inference to get smoothed posteriors for occupancy.
    let fr = filter(params, obs)?;
    let sr = smooth(params, &fr)?;
    let regimes = summarize_regimes(params, &sr.smoothed);

    let ordered_means: Vec<f64> = order.iter().map(|&i| params.means[i]).collect();
    let ordered_variances: Vec<f64> = order.iter().map(|&i| params.variances[i]).collect();
    let expected_durations: Vec<f64> = order
        .iter()
        .map(|&i| regimes.expected_durations[i])
        .collect();
    let occupancy_shares: Vec<f64> = order.iter().map(|&i| regimes.occupancy_shares[i]).collect();

    Ok(RunSummary {
        log_likelihood: result.log_likelihood,
        n_iter: result.n_iter,
        converged: result.converged,
        ordered_means,
        ordered_variances,
        expected_durations,
        occupancy_shares,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::em::{EmConfig, fit_em};
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

    fn separated_k2_params() -> ModelParams {
        ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.95, 0.05], vec![0.05, 0.95]],
            vec![-5.0, 5.0],
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

    fn fit(params: ModelParams, t: usize, seed: u64) -> (EmResult, Vec<f64>) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let sim = simulate(params.clone(), t, &mut rng).unwrap();
        let result = fit_em(&sim.observations, params, &tight_config()).unwrap();
        (result, sim.observations)
    }

    // -----------------------------------------------------------------------
    // Parameter validity (4 tests)
    // -----------------------------------------------------------------------

    /// A well-formed fitted model passes validity with near-zero deviations.
    #[test]
    fn param_validity_valid_params_passes() {
        let (result, obs) = fit(k2_params(), 500, SEED);
        let d = diagnose(&result, &obs).unwrap();
        assert!(d.param_validity.valid, "expected valid=true");
        assert!(
            d.param_validity.max_pi_dev < 1e-12,
            "max_pi_dev = {}",
            d.param_validity.max_pi_dev
        );
        assert!(
            d.param_validity.max_row_dev < 1e-12,
            "max_row_dev = {}",
            d.param_validity.max_row_dev
        );
        assert!(
            d.param_validity.min_variance > 0.0,
            "min_variance = {}",
            d.param_validity.min_variance
        );
    }

    /// A parameter set with π that doesn't sum to 1 reports valid=false
    /// with a nonzero max_pi_dev.
    #[test]
    fn param_validity_bad_pi_fails() {
        let mut params = k2_params();
        params.pi = vec![0.3, 0.3]; // sum = 0.6
        let pv = check_param_validity(&params);
        assert!(!pv.valid, "expected valid=false");
        assert!(
            (pv.max_pi_dev - 0.4).abs() < 1e-12,
            "max_pi_dev = {}",
            pv.max_pi_dev
        );
    }

    /// A near-zero variance triggers the NearZeroVariance warning but the
    /// fit itself may still be structurally valid.
    #[test]
    fn param_validity_near_zero_variance_warns() {
        let mut params = k2_params();
        params.variances[0] = 1e-5; // below threshold 1e-4
        let pv = check_param_validity(&params);
        // params.validate() accepts variances > 0, so structurally valid
        assert!(pv.valid, "expected valid=true for near-zero variance");
        // Build a synthetic EmResult so we can call diagnose.
        let _config = EmConfig {
            max_iter: 0,
            ..Default::default()
        };
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(k2_params(), 200, &mut rng).unwrap();
        // We check warnings via collect_warnings directly.
        let regimes = summarize_regimes(&params, &vec![vec![0.5, 0.5]; 10]);
        let conv = ConvergenceSummary {
            stop_reason: StopReason::Converged,
            n_iter: 1,
            initial_ll: -100.0,
            final_ll: -90.0,
            ll_gain: 10.0,
            min_delta: 10.0,
            largest_negative_delta: 0.0,
            is_monotone: true,
        };
        let warnings = collect_warnings(&pv, &regimes, &conv, &[-100.0, -90.0]);
        let has_near_zero = warnings.iter().any(|w| {
            matches!(w, DiagnosticWarning::NearZeroVariance { regime: 0, value: v } if *v < 1e-4)
        });
        assert!(has_near_zero, "expected NearZeroVariance warning");
        let _ = sim; // suppress unused warning
    }

    /// NaN in a mean field causes valid=false and all_params_finite=false.
    #[test]
    fn param_validity_nan_mean_fails() {
        let mut params = k2_params();
        params.means[0] = f64::NAN;
        let pv = check_param_validity(&params);
        assert!(!pv.valid, "expected valid=false for NaN mean");
        assert!(!pv.all_params_finite, "expected all_params_finite=false");
    }

    // -----------------------------------------------------------------------
    // Posterior validity (5 tests)
    // -----------------------------------------------------------------------

    /// Under well-separated fitted parameters, all posterior deviations are
    /// below the diagnostic tolerance.
    #[test]
    fn posterior_validity_clean_fit_within_tolerance() {
        let (result, obs) = fit(separated_k2_params(), 500, SEED);
        let d = diagnose(&result, &obs).unwrap();
        let pv = &d.posterior_validity;
        assert!(
            pv.max_filtered_dev < POSTERIOR_TOL,
            "filtered dev = {}",
            pv.max_filtered_dev
        );
        assert!(
            pv.max_smoothed_dev < POSTERIOR_TOL,
            "smoothed dev = {}",
            pv.max_smoothed_dev
        );
        assert!(
            pv.max_pairwise_dev < POSTERIOR_TOL,
            "pairwise dev = {}",
            pv.max_pairwise_dev
        );
        assert!(
            pv.max_marginal_consistency_err < POSTERIOR_TOL,
            "marginal consistency err = {}",
            pv.max_marginal_consistency_err
        );
    }

    /// The filtered normalization deviation should be negligibly small.
    #[test]
    fn posterior_validity_filtered_normalization_near_zero() {
        let (result, obs) = fit(k2_params(), 300, SEED);
        let d = diagnose(&result, &obs).unwrap();
        assert!(
            d.posterior_validity.max_filtered_dev < 1e-12,
            "max_filtered_dev = {}",
            d.posterior_validity.max_filtered_dev
        );
    }

    /// Column marginal consistency: Σᵢ ξₜ(i,j) agrees with γₜ(j).
    #[test]
    fn posterior_validity_marginal_consistency_holds() {
        let (result, obs) = fit(k2_params(), 300, SEED);
        let d = diagnose(&result, &obs).unwrap();
        assert!(
            d.posterior_validity.max_marginal_consistency_err < POSTERIOR_TOL,
            "marginal consistency err = {}",
            d.posterior_validity.max_marginal_consistency_err
        );
    }

    /// Pairwise normalization: Σᵢⱼ ξₜ(i,j) = 1 at every transition step.
    #[test]
    fn posterior_validity_pairwise_normalization_holds() {
        let (result, obs) = fit(k2_params(), 300, SEED);
        let d = diagnose(&result, &obs).unwrap();
        assert!(
            d.posterior_validity.max_pairwise_dev < POSTERIOR_TOL,
            "pairwise dev = {}",
            d.posterior_validity.max_pairwise_dev
        );
    }

    /// T=2 edge case: diagnose does not panic.
    #[test]
    fn posterior_validity_t2_no_panic() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let params = k2_params();
        let sim = simulate(params.clone(), 2, &mut rng).unwrap();
        let config = EmConfig {
            max_iter: 1,
            ..Default::default()
        };
        let result = fit_em(&sim.observations, params, &config).unwrap();
        diagnose(&result, &sim.observations).unwrap();
    }

    // -----------------------------------------------------------------------
    // EM convergence (4 tests)
    // -----------------------------------------------------------------------

    /// A normally converged run reports is_monotone=true and
    /// largest_negative_delta=0.0.
    #[test]
    fn convergence_monotone_run_is_monotone() {
        let (result, obs) = fit(k2_params(), 500, SEED);
        let d = diagnose(&result, &obs).unwrap();
        assert!(d.convergence.is_monotone, "expected monotone history");
        assert_eq!(
            d.convergence.largest_negative_delta, 0.0,
            "largest_negative_delta should be 0 for monotone run"
        );
    }

    /// A synthetic non-monotone ll_history triggers EmNonMonotonicity warnings.
    #[test]
    fn convergence_nonmonotone_history_triggers_warning() {
        // Build a synthetic EmResult with a deliberate drop.
        let params = k2_params();
        let ll_history = vec![-500.0, -490.0, -495.0, -488.0]; // drop at iter 2
        let result = EmResult {
            params: params.clone(),
            log_likelihood: -488.0,
            ll_history,
            n_iter: 3,
            converged: false,
        };
        let mut rng = SmallRng::seed_from_u64(SEED);
        let sim = simulate(params, 100, &mut rng).unwrap();
        let d = diagnose(&result, &sim.observations).unwrap();
        let has_warning = d
            .warnings
            .iter()
            .any(|w| matches!(w, DiagnosticWarning::EmNonMonotonicity { .. }));
        assert!(has_warning, "expected EmNonMonotonicity warning");
        assert!(!d.convergence.is_monotone);
    }

    /// A converged run reports StopReason::Converged.
    #[test]
    fn convergence_converged_run_reports_converged() {
        let (result, obs) = fit(separated_k2_params(), 500, SEED);
        let d = diagnose(&result, &obs).unwrap();
        assert_eq!(d.convergence.stop_reason, StopReason::Converged);
    }

    /// A run that hits max_iter reports StopReason::IterationCap.
    #[test]
    fn convergence_iter_cap_reports_iteration_cap() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let params = k2_params();
        let sim = simulate(params.clone(), 200, &mut rng).unwrap();
        let config = EmConfig {
            max_iter: 1,
            tol: 0.0, // never converge by tolerance
            ..Default::default()
        };
        let result = fit_em(&sim.observations, params, &config).unwrap();
        let d = diagnose(&result, &sim.observations).unwrap();
        assert_eq!(d.convergence.stop_reason, StopReason::IterationCap);
    }

    // -----------------------------------------------------------------------
    // Regime interpretation (4 tests)
    // -----------------------------------------------------------------------

    /// Well-separated K=2 data: finite durations and both regimes used.
    #[test]
    fn regimes_separated_k2_finite_durations_and_occupancy() {
        let (result, obs) = fit(separated_k2_params(), 1000, SEED);
        let d = diagnose(&result, &obs).unwrap();
        for (j, &d_j) in d.regimes.expected_durations.iter().enumerate() {
            assert!(
                d_j.is_finite(),
                "expected finite duration for regime {j}, got {d_j}"
            );
            assert!(d_j >= 1.0, "duration must be ≥ 1 for regime {j}");
        }
        for (j, &share) in d.regimes.occupancy_shares.iter().enumerate() {
            assert!(share > 0.0, "regime {j} has zero occupancy share");
        }
    }

    /// Expected duration formula: 1/(1 − p_jj), verified to float precision.
    #[test]
    fn regimes_expected_duration_formula_correct() {
        let p_self = 0.9_f64;
        let expected = 1.0 / (1.0 - p_self);
        let computed = compute_expected_duration(p_self);
        assert!(
            (computed - expected).abs() < 1e-12,
            "duration: got {computed}, expected {expected}"
        );
    }

    /// Hard counts must sum to T.
    #[test]
    fn regimes_hard_counts_sum_to_t() {
        let t = 300;
        let (result, obs) = fit(k2_params(), t, SEED);
        let d = diagnose(&result, &obs).unwrap();
        let total: usize = d.regimes.hard_counts.iter().sum();
        assert_eq!(total, t, "hard counts sum to {total}, expected {t}");
    }

    /// A regime with p_self > 1 − 1e-6 triggers SuspiciousPersistence.
    #[test]
    fn regimes_suspicious_persistence_warning_emitted() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![1.0 - 1e-10, 1e-10], vec![0.1, 0.9]], // regime 0 nearly absorbing
            vec![-3.0, 3.0],
            vec![1.0, 1.0],
        );
        let pv = check_param_validity(&params);
        let smoothed = vec![vec![0.5, 0.5]; 10];
        let regimes = summarize_regimes(&params, &smoothed);
        let conv = ConvergenceSummary {
            stop_reason: StopReason::Converged,
            n_iter: 1,
            initial_ll: -100.0,
            final_ll: -90.0,
            ll_gain: 10.0,
            min_delta: 10.0,
            largest_negative_delta: 0.0,
            is_monotone: true,
        };
        let warnings = collect_warnings(&pv, &regimes, &conv, &[-100.0, -90.0]);
        let has = warnings.iter().any(|w| {
            matches!(
                w,
                DiagnosticWarning::SuspiciousPersistence { regime: 0, .. }
            )
        });
        assert!(has, "expected SuspiciousPersistence for regime 0");
    }

    // -----------------------------------------------------------------------
    // Multi-start (3 tests)
    // -----------------------------------------------------------------------

    /// compare_runs sorts runs by log-likelihood descending.
    #[test]
    fn multistart_runs_sorted_descending() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let params = k2_params();
        let sim = simulate(params.clone(), 300, &mut rng).unwrap();

        let seeds = [SEED, SEED + 1, SEED + 2];
        let results: Vec<EmResult> = seeds
            .iter()
            .map(|&s| {
                let mut r2 = SmallRng::seed_from_u64(s);
                let _init = simulate(params.clone(), 300, &mut r2).unwrap();
                // Use a randomized init_params by perturbing means slightly.
                let mut p = params.clone();
                p.means[0] += (s as f64) * 0.1;
                fit_em(&sim.observations, p, &tight_config()).unwrap()
            })
            .collect();

        let summary = compare_runs(&results, &sim.observations).unwrap();
        for w in summary.runs.windows(2) {
            assert!(
                w[0].log_likelihood >= w[1].log_likelihood,
                "runs not sorted: {} < {}",
                w[0].log_likelihood,
                w[1].log_likelihood
            );
        }
    }

    /// canonical_regime_order returns indices sorted by increasing mean.
    #[test]
    fn multistart_canonical_order_sorts_by_mean() {
        let params = ModelParams::new(
            vec![0.5, 0.5],
            vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            vec![5.0, -3.0], // regime 0 has larger mean
            vec![1.0, 1.0],
        );
        let order = canonical_regime_order(&params);
        // regime 1 (mean=-3) should come first
        assert_eq!(order[0], 1);
        assert_eq!(order[1], 0);
    }

    /// top2_gap equals best_ll minus runner_up_ll.
    #[test]
    fn multistart_top2_gap_is_best_minus_runner_up() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        let params = k2_params();
        let sim = simulate(params.clone(), 300, &mut rng).unwrap();

        let mut p1 = params.clone();
        let mut p2 = params.clone();
        p1.means[0] = -3.0;
        p2.means[0] = -2.5;

        let r1 = fit_em(&sim.observations, p1, &tight_config()).unwrap();
        let r2 = fit_em(&sim.observations, p2, &tight_config()).unwrap();
        let summary = compare_runs(&[r1, r2], &sim.observations).unwrap();

        let expected_gap = (summary.best_ll - summary.runner_up_ll).abs();
        let reported_gap = summary.top2_gap.abs();
        assert!(
            (expected_gap - reported_gap).abs() < 1e-10,
            "top2_gap mismatch: reported {reported_gap}, computed {expected_gap}"
        );
    }
}
