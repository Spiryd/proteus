# Proteus — Observed-Data Log-Likelihood (Phase 5)

Phase 5 isolates the **observed-data log-likelihood** as a first-class statistical object and connects it precisely to the forward filter from Phase 4.

The central question this phase answers is:

> What quantity is actually being optimized when the model is fitted to data?

---

## Two likelihoods, one model

The Gaussian Markov Switching Model has two distinct likelihood objects. Understanding the difference is essential before any estimation phase.

### Complete-data likelihood

If the hidden regime path $S_{1:T}$ were observed, the likelihood would factor cleanly:

$$
\Pr(S_{1:T},\, y_{1:T};\, \Theta)
\;=\;
\pi_{S_1}\, f(y_1 \mid S_1)
\prod_{t=2}^T p_{S_{t-1},S_t}\, f(y_t \mid S_t)
$$

This is a product of transition probabilities and emission densities along a single, known path.  
It is **not** the estimation target because $S_{1:T}$ is latent.

### Observed-data likelihood

What is actually available is only the observations $y_{1:T}$.  
The observed-data likelihood integrates out all possible hidden paths:

$$
L(\Theta) = f(y_{1:T};\, \Theta)
$$

This is the correct estimation target.  
It is what the forward filter computes.

---

## Factorization over time

The observed-data likelihood decomposes sequentially by the chain rule:

$$
L(\Theta)
= f(y_1;\, \Theta) \prod_{t=2}^T f(y_t \mid y_{1:t-1};\, \Theta)
= \prod_{t=1}^T c_t
$$

where each $c_t$ is the **one-step predictive density** of $y_t$ given all past observations.

Taking logarithms:

$$
\log L(\Theta) = \sum_{t=1}^T \log c_t
$$

This is the single most important formula in the project. It is both the estimation objective and the output of the forward filter.

---

## The predictive density $c_t$

At each time $t$, the current regime $S_t$ is unknown. Summing over all possible regimes gives:

$$
c_t
= f(y_t \mid y_{1:t-1};\, \Theta)
= \sum_{j=1}^K f_j(y_t)\, \alpha_{t|t-1}(j)
$$

where:
- $f_j(y_t) = f(y_t \mid S_t=j;\,\theta_j)$ is the regime-conditional emission density (Phase 3)
- $\alpha_{t|t-1}(j) = \Pr(S_t=j \mid y_{1:t-1})$ is the predicted regime probability (Phase 4)

$c_t$ plays a **dual role** in the algorithm:

| Role | Where it appears |
|---|---|
| **Likelihood contribution** | $\log L(\Theta) = \sum_t \log c_t$ |
| **Bayes normalization constant** | $\alpha_{t\|t}(j) = f_j(y_t)\,\alpha_{t\|t-1}(j) \;/\; c_t$ |

These two roles cannot be separated. The filter computes both simultaneously. The same scalar that turns raw unnormalized posterior weights into probabilities is also the per-step contribution to the likelihood.

### Initialization at $t=1$

At $t=1$ there is no prior observation history. The predicted probabilities are the initial distribution:

$$
\alpha_{1|0}(j) = \pi_j
\qquad \Rightarrow \qquad
c_1 = \sum_{j=1}^K f_j(y_1)\,\pi_j
$$

The initial distribution $\pi$ is therefore part of the likelihood and influences estimation.

---

## Module layout

```
src/
  model/
    likelihood.rs  — log_likelihood(), log_likelihood_contributions()
    filter.rs      — filter(), FilterResult — computes log_likelihood as a field
    emission.rs    — Emission::log_density() — Phase 3
    params.rs      — ModelParams — Phase 2
    simulate.rs    — simulate() — Phase 2
```

`likelihood.rs` contains no statistical logic. It is a **thin interface layer** over `filter`. Its purpose is to expose the scalar $\log L(\Theta)$ as a clean first-class function.

---

## Public API

### `log_likelihood`

```rust
pub fn log_likelihood(params: &ModelParams, obs: &[f64]) -> anyhow::Result<f64>
```

Returns $\log L(\Theta) = \sum_t \log c_t$.

This is the entry point for any optimization or model-selection code that only needs the scalar total.

### `log_likelihood_contributions`

```rust
pub fn log_likelihood_contributions(params: &ModelParams, obs: &[f64]) -> anyhow::Result<Vec<f64>>
```

Returns a `Vec<f64>` of length $T$ where `contributions[t]` = $\log c_{t+1}$.

The decomposition identity holds exactly:

$$
\log L(\Theta) = \sum_{t=0}^{T-1} \texttt{contributions[t]}
$$

Useful for per-time diagnostics and identifying time periods where the model is most surprised.

### When to call each

| Caller | Use |
|---|---|
| Optimizer / EM M-step | `log_likelihood` — only the scalar is needed |
| Diagnostics / debugging | `log_likelihood_contributions` — inspect time-point fit |
| Smoother / EM E-step | `filter()` directly — needs `predicted`, `filtered`, and `log_predictive` |

---

## Example

```rust
use proteus::model::{ModelParams, log_likelihood, log_likelihood_contributions};

let params = ModelParams::new(
    vec![0.5, 0.5],
    vec![vec![0.99, 0.01], vec![0.01, 0.99]],
    vec![-5.0, 5.0],
    vec![1.0, 1.0],
);
let obs = vec![-4.8, -5.1, 4.9, 5.2, 5.0, -4.7];

// Total log-likelihood (for optimization)
let ll = log_likelihood(&params, &obs).unwrap();
println!("log L = {ll:.4}");

// Per-time contributions (for diagnostics)
let contribs = log_likelihood_contributions(&params, &obs).unwrap();
for (t, lc) in contribs.iter().enumerate() {
    println!("log c_{} = {lc:.4}", t + 1);
}
// Identity: ll == contribs.iter().sum()
```

---

## Invariants

These must hold for any valid parameter set and non-empty observation sequence:

| Invariant | Condition |
|---|---|
| Predictive density positive | $c_t > 0$ for all $t$ |
| Log-predictive finite | $\log c_t \in (-\infty, 0]$ typically, always finite |
| Decomposition identity | $\log L = \sum_t \log c_t$ exactly |
| Sensitivity to emission params | Changing $\mu_j$ or $\sigma_j^2$ changes $\log L$ |
| Sensitivity to transition params | Changing $P$ changes $\log L$ on structured data |
| Sensitivity to $\pi$ | Changing $\pi$ changes $\log L$ even for $T=1$ |

---

## What log $c_t$ tells you locally

Each term $\log c_t$ measures how well the model explained observation $y_t$ given the past.

| Scenario | Effect on $c_t$ |
|---|---|
| $y_t$ near the likely regime's mean | $c_t$ large — model unsurprised |
| $y_t$ far from all regime means | $c_t$ small — model surprised |
| One regime dominates and fits $y_t$ | $c_t \approx f_j(y_t)$ for the dominant $j$ |
| Regimes are evenly weighted | $c_t$ is a genuine mixture average |

A run of small $\log c_t$ values indicates a systematic misfit — a structural feature of the observations that the model cannot explain.

---

## Why the observed-data likelihood is not a sum of emission log-densities

A common mistake is to compute:

$$
\sum_t \log f_{\hat{j}_t}(y_t) \quad \text{for some chosen regime sequence } \hat{j}_{1:T}
$$

This is **not** the observed-data log-likelihood. That expression conditions on a single hard regime path, ignoring regime uncertainty. It equals the complete-data likelihood at one specific path, not the observed-data likelihood.

The correct object always involves the mixture:

$$
\log c_t = \log \sum_{j=1}^K f_j(y_t)\,\alpha_{t|t-1}(j)
$$

where the predicted probabilities $\alpha_{t|t-1}(j)$ properly account for the Markov dynamics and all uncertainty about the current regime.

---

## Quantities to retain for later phases

| Quantity | Stored in | Needed by |
|---|---|---|
| $\log L(\Theta)$ | `FilterResult::log_likelihood` | EM convergence check, AIC/BIC |
| $\log c_t$ per time | `FilterResult::log_predictive` | Diagnostics, per-step fit |
| $\alpha_{t\|t-1}(j)$ per time | `FilterResult::predicted` | Smoother (Phase 6) |
| $\alpha_{t\|t}(j)$ per time | `FilterResult::filtered` | Smoother (Phase 6), EM E-step |

---

## Test suite

All 10 tests are in `src/model/likelihood.rs` under `#[cfg(test)]`.  
Run with `cargo test model::likelihood`.

```
running 10 tests
test model::likelihood::tests::test_contributions_length                       ... ok
test model::likelihood::tests::test_errors_propagated_from_filter              ... ok
test model::likelihood::tests::test_ll_improves_as_mean_approaches_observation ... ok
test model::likelihood::tests::test_ll_invariant_to_symmetric_permutation      ... ok
test model::likelihood::tests::test_t1_exact_likelihood                        ... ok
test model::likelihood::tests::test_ll_sensitive_to_initial_distribution       ... ok
test model::likelihood::tests::test_decomposition_identity                     ... ok
test model::likelihood::tests::test_contributions_finite                       ... ok
test model::likelihood::tests::test_ll_sensitive_to_transition_matrix          ... ok
test model::likelihood::tests::test_true_params_score_higher_than_misfit       ... ok
test result: ok. 10 passed; 0 failed; 0 ignored; finished in 0.01s
```

### Test descriptions

| Test | What is verified |
|---|---|
| `test_decomposition_identity` | $\log L = \sum_t \log c_t$ to machine precision on 500-step simulation |
| `test_contributions_length` | `contributions` vector has length T |
| `test_contributions_finite` | No $\pm\infty$ or NaN in per-time contributions over 1000 steps |
| `test_t1_exact_likelihood` | Closed-form: $\pi=(0.5,0.5)$, identical regimes, $y=0$ gives $\log L = -\tfrac{1}{2}\ln(2\pi)$ |
| `test_true_params_score_higher_than_misfit` | True params (μ=±10) score higher than misfit params (μ=±1) on own data |
| `test_ll_improves_as_mean_approaches_observation` | LL increases monotonically as $\mu_j \to y$ |
| `test_ll_invariant_to_symmetric_permutation` | Swapping regime labels in a symmetric model leaves LL unchanged |
| `test_ll_sensitive_to_transition_matrix` | Persistent model scores higher than switching model on persistent data |
| `test_ll_sensitive_to_initial_distribution` | $\pi$ favoring the better-fitting regime gives higher LL at T=1 |
| `test_errors_propagated_from_filter` | Empty obs and invalid params both return `Err` |

---

## Phase connections

| Phase | Contribution to likelihood |
|---|---|
| Phase 1 | Defines $\Theta = (\pi, P, \mu, \sigma^2)$ |
| Phase 2 | `simulate` produces ground-truth data for statistical tests |
| Phase 3 | `Emission::log_density(y, j)` provides $\log f_j(y_t)$ |
| Phase 4 | `filter()` computes $\alpha_{t\|t-1}$, $c_t$, and $\log L$ in one pass |
| **Phase 5** | Formalizes $\log L$ as the estimation objective; exposes `log_likelihood()` |
| Phase 6 | EM will call `log_likelihood()` to monitor convergence after each M-step |

---

## What comes next (Phase 6)

Phase 6 will implement the **backward smoother**, which computes:

$$
\Pr(S_t = j \mid y_{1:T})
$$

— the posterior regime probability given the **full** observation sequence, not just the past.

The smoother requires the forward quantities stored in `FilterResult::predicted` and `FilterResult::filtered`. It will not modify the likelihood computation in any way; it only adds a reverse pass that uses what the filter already produced.
