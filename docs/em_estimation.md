# EM Estimation — Phase 9

## Purpose

Phase 9 turns the inference pipeline into a full parameter estimator.  Up to
Phase 8, every component assumed $\Theta = (\pi, P, \mu_1,\dots,\mu_K,
\sigma_1^2,\dots,\sigma_K^2)$ was known.  Phase 9 inverts this: $\Theta$ is
the unknown and $y_{1:T}$ is the given.

The objective being maximized remains the observed-data log-likelihood already
computed by the filter:

$$\log L(\Theta) = \sum_{t=1}^T \log c_t, \qquad c_t = f(y_t \mid y_{1:t-1};\Theta).$$

EM is used as the primary estimator because it reuses the full inference stack
and yields simple closed-form parameter updates that automatically satisfy all
model constraints.

---

## EM Algorithm

### E-step

Given current parameters $\Theta^{(m)}$, run a complete inference pass:

1. **Forward filter** → `FilterResult` (filtered probs, predicted probs, $\log L(\Theta^{(m)})$)
2. **Backward smoother** → `SmootherResult` (smoothed marginals $\gamma_t^{(m)}(j)$)
3. **Pairwise pass** → `PairwiseResult` (pairwise posteriors $\xi_t^{(m)}(i,j)$, expected counts $N_{ij}^{(m)}$)

### M-step

Update all parameters from the E-step bundle.

**Initial distribution:**

$$\pi_j^{(m+1)} = \gamma_1^{(m)}(j)$$

Read from `smoothed[0][j]`. Automatically sums to 1.

**Transition matrix:** For each row $i$:

$$p_{ij}^{(m+1)} = \frac{N_{ij}^{(m)}}{M_i^{(m)}}, \qquad M_i^{(m)} = \sum_{t=1}^{T-1} \gamma_t^{(m)}(i)$$

Row sums to 1 by Invariant D of Phase 8. Row frozen when $M_i < 10^{-10}$.

**Regime means:**

$$\mu_j^{(m+1)} = \frac{\sum_{t=1}^T \gamma_t^{(m)}(j)\, y_t}{W_j^{(m)}}, \qquad W_j^{(m)} = \sum_{t=1}^T \gamma_t^{(m)}(j)$$

Update skipped when $W_j < 10^{-10}$ (regime degeneracy).

**Regime variances:**

$$(\sigma_j^2)^{(m+1)} = \frac{\sum_{t=1}^T \gamma_t^{(m)}(j)\,(y_t - \mu_j^{(m+1)})^2}{W_j^{(m)}}$$

Uses the *updated* mean $\mu_j^{(m+1)}$. A variance floor of $10^{-6}$ is applied after each update to prevent collapse.

---

## Iteration Structure

```
baseline E-step  →  record ll_history[0]

for m = 0..max_iter:
    M-step  →  Θ^(m+1)
    E-step  →  ll^(m+1)
    append ll_history

    if |ll^(m+1) − ll^(m)| < tol:
        converged = true
        break
```

A monotonicity check emits a warning (non-fatal) when the log-likelihood drops
by more than $10^{-8}$ — a violation of the EM ascent guarantee that signals a
numerical issue.

---

## API

```rust
/// Convergence configuration.
pub struct EmConfig {
    pub tol: f64,        // log-likelihood change tolerance (default 1e-6)
    pub max_iter: usize, // hard iteration cap (default 1000)
    pub var_floor: f64,  // minimum variance after each update (default 1e-6)
}

/// Bundle produced by one E-step pass.
pub struct EStepResult {
    pub smoothed: Vec<Vec<f64>>,              // γ_t(j), shape T×K
    pub expected_transitions: Vec<Vec<f64>>,  // N_ij^exp, shape K×K
    pub log_likelihood: f64,
}

/// Output of a completed EM run.
pub struct EmResult {
    pub params: ModelParams,    // Θ̂
    pub log_likelihood: f64,    // log L(Θ̂)
    pub ll_history: Vec<f64>,   // log L at each iteration (length n_iter+1)
    pub n_iter: usize,
    pub converged: bool,
}

/// Top-level estimation entry point.
pub fn fit_em(obs: &[f64], init_params: ModelParams, config: &EmConfig) -> Result<EmResult>
```

`fit_em` never reads emission densities directly — all emission logic is
encapsulated in the filter.

---

## Degenerate-Case Guards

| Condition | Effect |
|---|---|
| $M_i^{(m)} < 10^{-10}$ | Transition row $i$ frozen at $\Theta^{(m)}$ |
| $W_j^{(m)} < 10^{-10}$ | Mean and variance for regime $j$ frozen |
| $(\sigma_j^2)^{(m+1)} < 10^{-6}$ | Variance floor applied |
| $\log L$ decreases by $> 10^{-8}$ | Warning printed; iteration continues |

---

## Invariants After Every M-step

| Quantity | Invariant |
|---|---|
| $\pi$ | Sums to 1; all entries $\ge 0$ |
| Each row of $P$ | Sums to 1; all entries $\ge 0$ |
| $\sigma_j^2$ | Strictly positive (variance floor enforced) |
| $\log L$ | Finite |
| $\log L$ sequence | Nondecreasing up to $10^{-8}$ |

`ModelParams::validate()` is called after every M-step to enforce the first
three invariants programmatically.

---

## Module Structure

```
src/model/em.rs   ← only file with estimation logic
```

`em.rs` calls `filter`, `smoother`, `pairwise` — no reverse dependencies.
`mod.rs` re-exports `EmConfig`, `EmResult`, `EStepResult`, and `fit_em`.

---

## Validated Behavioral Properties (12 tests)

| Test | Property |
|---|---|
| `pi_update_equals_smoothed_at_t0` | $\pi^{(1)} = \gamma_1^{(0)}$ exactly |
| `transition_rows_sum_to_one` | All $P$ rows sum to 1 after convergence |
| `variances_remain_positive` | All $\sigma_j^2 > 0$ after convergence |
| `result_fields_consistent` | `ll_history.len() == n_iter+1`; last entry matches `log_likelihood` |
| `log_likelihood_nondecreasing` | EM ascent holds for K=2, T=1000 |
| `log_likelihood_nondecreasing_k3` | EM ascent holds for K=3 |
| `converges_and_improves_likelihood` | Converges before max_iter; fitted ll > initial ll |
| `mean_recovery_on_separated_data` | Sorted fitted means within 1.0 of true means |
| `minimal_sample_no_panic` | T=2 does not panic |
| `empty_obs_returns_error` | Returns `Err` for empty input |
| `zero_iterations_returns_baseline` | `n_iter=0`, params unchanged, `ll_history` length 1 |
| `multi_seed_ascent` | Ascent invariant holds across 8 random seeds |

---

## What EM Convergence Means

`converged = true` means the log-likelihood tolerance was met — not that the
global maximum was found. EM converges to a **local** maximum or saddle point.
Different starting points can yield different final likelihoods.  The
`ll_history` is the primary diagnostic for evaluating the quality of a run.

---

## References

- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). *Maximum likelihood from incomplete data via the EM algorithm*. JRSS-B 39(1), 1–38.
- Kim, C.-J. (1994). *Dynamic linear models with Markov-switching*. Journal of Econometrics 60, 1–22.
- Hamilton, J. D. (1994). *Time Series Analysis*. Chapter 22.
