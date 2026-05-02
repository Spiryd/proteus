# Diagnostics and Trust Checks — Phase 10

## Purpose

Phase 10 turns the fitted Gaussian Markov Switching Model from something that
*runs* into something that can be *trusted, inspected, and compared*.

A model can converge numerically, return valid-looking parameters, and produce
plausible smoothed regime probabilities — and still fail in subtle ways.  The
diagnostics layer makes those failures visible by answering six questions:

1. Is the fitted parameter set mathematically valid?
2. Are posterior probabilities normalized and internally consistent?
3. Did EM converge credibly?
4. Are the estimated regimes interpretable?
5. How persistent are the regimes?
6. Are results stable across multiple initializations?

---

## Module boundary

```
src/model/diagnostics.rs   ← only diagnostics file
```

`diagnostics.rs` is a pure consumer.  It calls `filter`, `smooth`, and
`pairwise`, reads `ModelParams` and `EmResult` fields, and has no reverse
dependencies inside the model stack.

`diagnose()` and `compare_runs()` are called automatically from
`experiments::shared::train_or_load_model_shared()` after every EM fit.
Results are stored in `ModelArtifact` and exported by the runner to:

- `diagnostics.json` — written for every run,
- `multi_start_summary.json` — written when `ModelConfig::em_n_starts > 1`.

Diagnostic warnings are propagated into `ExperimentResult::warnings` with the
prefix `"diagnostic: "`.  Multi-start warnings use the prefix `"multi_start: "`.
`ModelArtifact::diagnostics_ok` is set to `FittedModelDiagnostics::is_trustworthy`
rather than the raw EM `converged` flag.

---

## Entry points

```rust
/// Full diagnostics bundle for one fitted model.
pub fn diagnose(result: &EmResult, obs: &[f64]) -> Result<FittedModelDiagnostics>

/// Cross-run comparison for multiple EM starts.
pub fn compare_runs(results: &[EmResult], obs: &[f64]) -> Result<MultiStartSummary>
```

`diagnose` performs a single inference pass (filter → smooth → pairwise) under
the fitted parameters.  All posterior checks, regime summaries, and warnings
are derived from that one pass.

---

## 1. Parameter validity

### Math

The fitted parameter set $\Theta = (\pi, P, \mu_1,\dots,\mu_K, \sigma_1^2,\dots,\sigma_K^2)$ must satisfy:

$$\pi_j \ge 0, \qquad \sum_j \pi_j = 1,$$

$$p_{ij} \ge 0, \qquad \sum_j p_{ij} = 1 \quad \forall i,$$

$$\sigma_j^2 > 0 \quad \forall j, \qquad \mu_j \in \mathbb{R} \quad \forall j.$$

### Scalar summaries

| Field | Value |
|---|---|
| `max_pi_dev` | $\lvert \sum_j \pi_j - 1 \rvert$ |
| `max_row_dev` | $\max_i \lvert \sum_j p_{ij} - 1 \rvert$ |
| `min_variance` | $\min_j \sigma_j^2$ |
| `all_params_finite` | `true` iff no NaN or ±∞ |
| `valid` | `true` iff all structural constraints pass |

`valid` is the conjunction of `ModelParams::validate()` succeeding and
`all_params_finite` being true.

### API

```rust
pub struct ParamValidity {
    pub valid: bool,
    pub max_pi_dev: f64,
    pub max_row_dev: f64,
    pub min_variance: f64,
    pub all_params_finite: bool,
}
```

---

## 2. Posterior probability diagnostics

After fitting, a final inference pass is run under $\hat\Theta$.  Four
normalization and consistency conditions are checked.

### 2.1 Filtered normalization

$$\sum_{j=1}^K \Pr(S_t=j \mid y_{1:t}) = 1 \quad \forall t$$

Scalar: $\max_t \lvert \sum_j \text{filtered}[t][j] - 1 \rvert$.

### 2.2 Smoothed normalization

$$\sum_{j=1}^K \gamma_t(j) = 1 \quad \forall t$$

Scalar: $\max_t \lvert \sum_j \text{smoothed}[t][j] - 1 \rvert$.

### 2.3 Pairwise normalization

$$\sum_{i=1}^K \sum_{j=1}^K \xi_t(i,j) = 1 \quad \forall t = 2,\dots,T$$

Scalar: $\max_t \lvert \sum_{ij} \xi_t(i,j) - 1 \rvert$.

### 2.4 Column marginal consistency

$$\sum_{i=1}^K \xi_t(i,j) = \gamma_t(j) \quad \forall t, j$$

This is the strongest internal consistency check: it requires the pairwise
posteriors and the smoothed marginals to agree at every time step for every
regime.

Scalar: $\max_{t,j} \lvert \sum_i \xi_t(i,j) - \gamma_t(j) \rvert$.

### API

```rust
pub struct PosteriorValidity {
    pub max_filtered_dev: f64,
    pub max_smoothed_dev: f64,
    pub max_pairwise_dev: f64,
    pub max_marginal_consistency_err: f64,
}
```

---

## 3. EM convergence diagnostics

These are derived from `EmResult::ll_history` without additional inference.

### Stop reason

$$\text{stop\_reason} = \begin{cases}
\text{Converged} & \text{if } \lvert \ell^{(m+1)} - \ell^{(m)} \rvert < \text{tol}, \\
\text{IterationCap} & \text{if max\_iter was reached.}
\end{cases}$$

### Monotonicity

EM guarantees nondecreasing observed-data log-likelihood:

$$\log L(\Theta^{(m+1)}) \ge \log L(\Theta^{(m)}).$$

A decrease beyond $10^{-8}$ is a violation and triggers
`DiagnosticWarning::EmNonMonotonicity`.

### Scalar summaries

| Field | Description |
|---|---|
| `initial_ll` | $\log L(\Theta^{(0)})$ |
| `final_ll` | $\log L(\hat\Theta)$ |
| `ll_gain` | `final_ll − initial_ll` |
| `min_delta` | smallest log-likelihood increment across all pairs |
| `largest_negative_delta` | magnitude of the worst monotonicity violation; `0.0` if monotone |
| `is_monotone` | `true` iff no drop exceeds $10^{-8}$ |

### API

```rust
pub enum StopReason { Converged, IterationCap }

pub struct ConvergenceSummary {
    pub stop_reason: StopReason,
    pub n_iter: usize,
    pub initial_ll: f64,
    pub final_ll: f64,
    pub ll_gain: f64,
    pub min_delta: f64,
    pub largest_negative_delta: f64,
    pub is_monotone: bool,
}
```

---

## 4. Regime interpretation diagnostics

These require the smoothed posteriors from the final inference pass.

### 4.1 Posterior occupancy

$$W_j = \sum_{t=1}^T \gamma_t(j), \qquad \text{share}_j = \frac{W_j}{T}.$$

A regime with $\text{share}_j < 0.01$ triggers `DiagnosticWarning::NearlyUnusedRegime`.

### 4.2 Expected regime duration

For a regime $j$ with self-transition probability $p_{jj}$, the expected
duration of each visit follows a geometric structure:

$$\mathbb{E}[\text{duration in regime } j] = \frac{1}{1 - p_{jj}}.$$

This is one of the most important interpretive outputs of a Markov Switching
Model.

- $p_{jj} \to 1$ implies very long-lived regimes.
- $p_{jj}$ small implies short-lived, rapidly switching regimes.

A value $p_{jj} > 1 - 10^{-6}$ triggers
`DiagnosticWarning::SuspiciousPersistence` (possible absorbing state).

Returns `f64::INFINITY` when $p_{jj} = 1$.

### 4.3 Hard classification counts

$$\hat{S}_t = \arg\max_j \gamma_t(j).$$

Count of assignments per regime.  Total must equal $T$.

### API

```rust
pub struct RegimeSummary {
    pub means: Vec<f64>,
    pub variances: Vec<f64>,
    pub occupancy_weights: Vec<f64>,   // Wⱼ
    pub occupancy_shares: Vec<f64>,    // Wⱼ / T
    pub self_transition_probs: Vec<f64>,
    pub expected_durations: Vec<f64>,
    pub hard_counts: Vec<usize>,
}
```

---

## 5. Warning system

### Warning variants

```rust
pub enum DiagnosticWarning {
    NearZeroVariance { regime: usize, value: f64 },
    NearlyUnusedRegime { regime: usize, occupancy_share: f64 },
    EmNonMonotonicity { iteration: usize, drop: f64 },
    SuspiciousPersistence { regime: usize, p_self: f64, expected_duration: f64 },
    UnstableAcrossStarts { ll_spread: f64 },
}
```

### Thresholds

| Constant | Value | Triggers |
|---|---|---|
| `NEAR_ZERO_VAR_THRESHOLD` | $10^{-4}$ | `NearZeroVariance` |
| `NEARLY_UNUSED_REGIME_SHARE` | $0.01$ | `NearlyUnusedRegime` |
| `SUSPICIOUS_PERSISTENCE_P` | $1 - 10^{-6}$ | `SuspiciousPersistence` |
| `UNSTABLE_STARTS_LL_GAP` | $1.0$ | `UnstableAcrossStarts` |
| `MONOTONE_TOL` | $10^{-8}$ | `EmNonMonotonicity` |

Warnings are non-fatal: the model may still be valid and usable.

---

## 6. Trust flag

```rust
is_trustworthy = param_validity.valid
              && max_filtered_dev         < 1e-8
              && max_smoothed_dev         < 1e-8
              && max_pairwise_dev         < 1e-8
              && max_marginal_consistency_err < 1e-8
```

`is_trustworthy` does not require convergence.  A non-converged fit can still
be mathematically valid and interpretable; the `stop_reason` field explains why
the loop stopped.

### Top-level bundle

```rust
pub struct FittedModelDiagnostics {
    pub param_validity: ParamValidity,
    pub posterior_validity: PosteriorValidity,
    pub convergence: ConvergenceSummary,
    pub regimes: RegimeSummary,
    pub warnings: Vec<DiagnosticWarning>,
    pub is_trustworthy: bool,
}
```

---

## 7. Multi-start stability

### Motivation

EM is sensitive to initialization.  The likelihood surface is non-convex;
different starts can converge to different local optima or equivalent
label-switched solutions.  A single run is not sufficient evidence of a
globally good fit.

### Label switching

If two runs differ only by permuting regime labels, they are the same solution
under a different labeling.  To make runs comparable, regimes within each
`RunSummary` are reordered by increasing mean before comparison.

### Primary selection criterion

Rank runs by final log-likelihood.  Choose the converged run with the highest
$\log L(\hat\Theta)$.

### Stability check

If `best_ll − worst_ll > 1.0` (in log-likelihood units), an
`UnstableAcrossStarts` warning is emitted.

### API

```rust
pub struct RunSummary {
    pub log_likelihood: f64,
    pub n_iter: usize,
    pub converged: bool,
    pub ordered_means: Vec<f64>,
    pub ordered_variances: Vec<f64>,
    pub expected_durations: Vec<f64>,
    pub occupancy_shares: Vec<f64>,
}

pub struct MultiStartSummary {
    pub runs: Vec<RunSummary>,      // sorted best-first
    pub best_ll: f64,
    pub runner_up_ll: f64,
    pub ll_spread: f64,             // best − worst
    pub top2_gap: f64,              // best − runner_up
    pub n_converged: usize,
    pub warnings: Vec<DiagnosticWarning>,
}
```

---

## 8. Validated properties (20 tests)

### Parameter validity (4)

| Test | Property |
|---|---|
| `param_validity_valid_params_passes` | Well-formed fit: `valid=true`, deviations < 1e-12 |
| `param_validity_bad_pi_fails` | π summing to 0.6: `valid=false`, `max_pi_dev=0.4` |
| `param_validity_near_zero_variance_warns` | σ² = 1e-5: `NearZeroVariance` warning emitted |
| `param_validity_nan_mean_fails` | NaN mean: `valid=false`, `all_params_finite=false` |

### Posterior validity (5)

| Test | Property |
|---|---|
| `posterior_validity_clean_fit_within_tolerance` | All posterior deviations < `POSTERIOR_TOL` |
| `posterior_validity_filtered_normalization_near_zero` | `max_filtered_dev < 1e-12` |
| `posterior_validity_marginal_consistency_holds` | Column marginal error < `POSTERIOR_TOL` |
| `posterior_validity_pairwise_normalization_holds` | Pairwise dev < `POSTERIOR_TOL` |
| `posterior_validity_t2_no_panic` | T=2 edge case does not panic |

### EM convergence (4)

| Test | Property |
|---|---|
| `convergence_monotone_run_is_monotone` | `is_monotone=true`, `largest_negative_delta=0.0` |
| `convergence_nonmonotone_history_triggers_warning` | Synthetic drop → `EmNonMonotonicity` |
| `convergence_converged_run_reports_converged` | `StopReason::Converged` |
| `convergence_iter_cap_reports_iteration_cap` | `StopReason::IterationCap` |

### Regime interpretation (4)

| Test | Property |
|---|---|
| `regimes_separated_k2_finite_durations_and_occupancy` | Finite durations, both regimes used |
| `regimes_expected_duration_formula_correct` | $1/(1-0.9) = 10$ to float precision |
| `regimes_hard_counts_sum_to_t` | $\sum_j \text{hard\_counts}[j] = T$ |
| `regimes_suspicious_persistence_warning_emitted` | $p_{jj} \approx 1$ → `SuspiciousPersistence` |

### Multi-start (3)

| Test | Property |
|---|---|
| `multistart_runs_sorted_descending` | `compare_runs` returns runs best-first |
| `multistart_canonical_order_sorts_by_mean` | `ordered_means` always sorted ascending |
| `multistart_top2_gap_is_best_minus_runner_up` | `top2_gap = best_ll − runner_up_ll` |

---

## 9. Common diagnostics mistakes to avoid

| Mistake | Consequence |
|---|---|
| Treating convergence as correctness | A converged EM can still produce a degenerate solution |
| Reporting only fitted parameters | Hidden-state models need posterior and persistence summaries |
| Ignoring multiple starts | Non-convex likelihood makes single-start results unsafe |
| Treating expected durations as optional | They are the primary interpretive output of the Markov chain |
| Hiding diagnostics in debug output | Diagnostics must be first-class structured results |

---

## 10. References

- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). *Maximum likelihood
  from incomplete data via the EM algorithm.* JRSS-B 39(1), 1–38.
- Hamilton, J. D. (1989). *A new approach to the economic analysis of
  nonstationary time series.* Econometrica 57(2), 357–384.
- Kim, C.-J. (1994). *Dynamic linear models with Markov-switching.*
  Journal of Econometrics 60, 1–22.
