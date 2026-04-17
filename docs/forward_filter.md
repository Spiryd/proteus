# Proteus — Forward Filter (Phase 4)

Phase 4 implements the first real inference mechanism of the Markov Switching Model: the **Hamilton forward filter**.

It answers the central inferential question of the model:

> Given the observed data so far, which regime is currently most plausible?

---

## The filter in one equation

At each time $t$, the filter combines two sources of information:

$$
\underbrace{\alpha_{t|t}(j)}_{\text{filtered}} \;\propto\;
\underbrace{f_j(y_t)}_{\text{emission}} \;\cdot\;
\underbrace{\alpha_{t|t-1}(j)}_{\text{predicted}}
$$

The predicted probability comes from the Markov chain.  
The emission density comes from the observation model.  
Their normalized product is the filtered (posterior) regime probability.

---

## Notation fixed for this phase

| Symbol | Meaning | In code |
|---|---|---|
| $\alpha_{t\|t-1}(j)$ | $\Pr(S_t=j \mid y_{1:t-1})$ | `predicted[t-1][j]` |
| $\alpha_{t\|t}(j)$ | $\Pr(S_t=j \mid y_{1:t})$ | `filtered[t-1][j]` |
| $f_j(y_t)$ | $f(y_t \mid S_t=j;\,\theta_j)$ | `Emission::log_density(y, j)` |
| $c_t$ | $f(y_t \mid y_{1:t-1})$ | `exp(log_predictive[t-1])` |
| $\log L(\Theta)$ | $\sum_t \log c_t$ | `log_likelihood` |

All time indices are stored 0-based: index `t` corresponds to observation $y_{t+1}$ in math notation.

---

## Full recursion

**Initialization** — $t = 1$ (index 0):

$$
\alpha_{1|0}(j) = \pi_j
$$

$$
c_1 = \sum_{j=1}^K f_j(y_1)\,\pi_j
$$

$$
\alpha_{1|1}(j) = \frac{f_j(y_1)\,\pi_j}{c_1}
$$

**Recursion** — $t = 2, \dots, T$ (indices $1, \dots, T-1$):

$$
\alpha_{t|t-1}(j) = \sum_{i=1}^K p_{ij}\,\alpha_{t-1|t-1}(i)
\tag{prediction}
$$

$$
c_t = \sum_{j=1}^K f_j(y_t)\,\alpha_{t|t-1}(j)
\tag{predictive density}
$$

$$
\alpha_{t|t}(j) = \frac{f_j(y_t)\,\alpha_{t|t-1}(j)}{c_t}
\tag{Bayes update}
$$

**Log-likelihood:**

$$
\log L(\Theta) = \sum_{t=1}^T \log c_t
$$

---

## Module layout

```
src/
  model/
    filter.rs    — FilterResult, filter(), predict(), bayes_update(), log_sum_exp()
    emission.rs  — Emission: log_density, log_density_vec (Phase 3)
    params.rs    — ModelParams: transition_row, validate (Phase 2)
    simulate.rs  — simulate: ground-truth path generator (Phase 2)
    mod.rs       — re-exports FilterResult, filter, Emission, ModelParams, …
```

`filter.rs` does not contain any emission formula.  
All emission calls go through `Emission::log_density(y, j)`.  
All transition calls go through `ModelParams::transition_row(i)`.

---

## `FilterResult`

```rust
pub struct FilterResult {
    pub t: usize,                    // T — number of observations
    pub k: usize,                    // K — number of regimes
    pub predicted: Vec<Vec<f64>>,    // T × K:  α_{t|t-1}(j)
    pub filtered:  Vec<Vec<f64>>,    // T × K:  α_{t|t}(j)
    pub log_predictive: Vec<f64>,    // T:       log cₜ
    pub log_likelihood: f64,         // Σₜ log cₜ
}
```

**Both arrays are retained.** The backward smoother (Phase 5) requires `predicted[t]` at every $t$. Dropping it before that phase would require re-running the filter.

---

## `filter`

```rust
pub fn filter(params: &ModelParams, obs: &[f64]) -> anyhow::Result<FilterResult>
```

Validates `params`, constructs an `Emission` from the validated parameters, then runs the full recursion.

### Example

```rust
use proteus::model::{ModelParams, filter};

let params = ModelParams::new(
    vec![0.5, 0.5],
    vec![vec![0.99, 0.01], vec![0.01, 0.99]],
    vec![-5.0, 5.0],
    vec![1.0, 1.0],
);
let obs = vec![-4.8, -5.1, 4.9, 5.2, 5.0, -4.7];
let result = filter(&params, &obs).unwrap();

println!("log L = {:.4}", result.log_likelihood);
for t in 0..result.t {
    let dominant = if result.filtered[t][0] > result.filtered[t][1] { 0 } else { 1 };
    println!("t={}: dominant regime = {}", t + 1, dominant);
}
```

### Errors

| Condition | Error message |
|---|---|
| `obs` is empty | `"filter: observation sequence must not be empty"` |
| `params.validate()` fails | forwarded from `ModelParams::validate` |
| $c_t = 0$ at any $t$ | `"filter: predictive density is zero …"` |

---

## Internal structure

### `predict` — prediction step

```rust
fn predict(params: &ModelParams, filtered_prev: &[f64], k: usize) -> Vec<f64>
```

Computes $\alpha_{t|t-1}(j) = \sum_i p_{ij}\,\alpha_{t-1|t-1}(i)$.

This is the matrix product $\alpha_{\text{prev}} \cdot P$ where $\alpha$ is treated as a row vector. The transition matrix `P` is row-major so `transition_row(i)[j]` = $p_{ij}$.

### `bayes_update` — update step

```rust
fn bayes_update(emission: &Emission, y: f64, predicted: &[f64], k: usize)
    -> Result<(Vec<f64>, f64)>
```

Returns `(filtered, log_c)`.

1. Computes unnormalized log-weights: $\log f_j(y) + \log \alpha_{t|t-1}(j)$ for each $j$.
2. Applies log-sum-exp to get $\log c_t$.
3. Exponentiates the difference to normalize.

Returns an error if $\log c_t = -\infty$ (all regimes assign zero density to $y$).

### `log_sum_exp` — numerical stability

```rust
fn log_sum_exp(log_vals: &[f64]) -> f64
```

Computes $\log \sum_i \exp(x_i)$ as $\max(x) + \log \sum_i \exp(x_i - \max(x))$.

This is the only place where exponentials appear. No raw density products are ever formed. All intermediate values are log-space until the final normalization.

---

## Numerical strategy

The emission density $f_j(y) = (2\pi\sigma_j^2)^{-\frac{1}{2}}\exp\!\bigl(-\frac{(y-\mu_j)^2}{2\sigma_j^2}\bigr)$ can be extremely small when $y$ is many standard deviations from $\mu_j$.

If computed in the raw domain, the product $f_j(y) \cdot \alpha_{t|t-1}(j)$ would underflow to 0 before normalization.

The filter avoids this by:

1. Calling `Emission::log_density(y, j)` — already log-space.
2. Adding $\log \alpha_{t|t-1}(j)$ — still log-space.
3. Applying log-sum-exp to compute $\log c_t$ exactly.
4. Exponentiating each $\log w_j - \log c_t$ to get $\alpha_{t|t}(j)$.

No product of densities is ever computed directly.

---

## Invariants at every time step

These must hold at every $t$ and are verified by the test suite:

| Invariant | Check |
|---|---|
| Predicted probs sum to 1 | $\sum_j \alpha_{t\|t-1}(j) = 1$ |
| Filtered probs sum to 1 | $\sum_j \alpha_{t\|t}(j) = 1$ |
| All probs in $[0, 1]$ | $\alpha \in [0,1]^K$ |
| Predictive density positive | $\log c_t$ is finite |
| Log-likelihood is a sum | $\log L = \sum_t \log c_t$ |

---

## Test suite

All 14 tests are in `src/model/filter.rs` under `#[cfg(test)]`.  
Run with `cargo test model::filter`.

```
running 14 tests
test model::filter::tests::test_empty_obs_returns_error                       ... ok
test model::filter::tests::test_extreme_observation_locks_regime              ... ok
test model::filter::tests::test_log_likelihood_equals_sum_of_log_predictive   ... ok
test model::filter::tests::test_output_lengths                                ... ok
test model::filter::tests::test_predicted_t1_equals_pi                        ... ok
test model::filter::tests::test_prediction_step_matches_matrix_multiply       ... ok
test model::filter::tests::test_filtered_sum_to_1                             ... ok
test model::filter::tests::test_invalid_params_returns_error                  ... ok
test model::filter::tests::test_t1_exact_log_likelihood                       ... ok
test model::filter::tests::test_predicted_sum_to_1                            ... ok
test model::filter::tests::test_t1_exact_posterior                            ... ok
test model::filter::tests::test_probabilities_in_unit_interval                ... ok
test model::filter::tests::test_log_predictive_is_finite                      ... ok
test model::filter::tests::test_filter_tracks_true_state                      ... ok
test result: ok. 14 passed; 0 failed; 0 ignored; finished in 0.00s
```

### Test descriptions

| Test | What is verified |
|---|---|
| `test_output_lengths` | `predicted`, `filtered`, `log_predictive` all length T; each inner vec length K |
| `test_predicted_t1_equals_pi` | `predicted[0]` equals $\pi$ exactly |
| `test_predicted_sum_to_1` | $\sum_j \alpha_{t\|t-1}(j) = 1$ at every $t$ on a 200-step simulation |
| `test_filtered_sum_to_1` | $\sum_j \alpha_{t\|t}(j) = 1$ at every $t$ on a 200-step simulation |
| `test_probabilities_in_unit_interval` | All predicted and filtered values lie in $[0, 1]$ |
| `test_log_likelihood_equals_sum_of_log_predictive` | $\log L = \sum_t \log c_t$ exactly |
| `test_log_predictive_is_finite` | No $\pm\infty$ or NaN in `log_predictive` over 500 steps |
| `test_t1_exact_posterior` | Closed-form Bayes update: $\pi=(0.3,0.7)$, $\mu=(0,1)$, $\sigma^2=(1,1)$, $y=0$ |
| `test_t1_exact_log_likelihood` | $\log c_1 = -\tfrac{1}{2}\ln(2\pi)$ when both regimes are identical |
| `test_extreme_observation_locks_regime` | $y=10$, $\mu=(-10,10)$: `filtered[0][1] > 0.9999` |
| `test_prediction_step_matches_matrix_multiply` | `predicted[1]` verified against manual $\alpha_0 \cdot P$ |
| `test_filter_tracks_true_state` | Argmax(filtered) matches true simulated state on ≥95% of 2000 steps |
| `test_empty_obs_returns_error` | Empty `obs` slice returns `Err` |
| `test_invalid_params_returns_error` | Bad $\pi$ sum returns `Err` |

---

## What is stored for future phases

| Field | Needed by |
|---|---|
| `predicted[t]` | Backward smoother (Phase 5): $\beta_t / \alpha_{t\|t-1}$ ratio |
| `filtered[t]` | EM E-step (Phase 6): starting point for pairwise posterior |
| `log_predictive[t]` | Likelihood diagnostics, model comparison |
| `log_likelihood` | EM convergence check, model selection (AIC/BIC) |

---

## Phase connections

| Phase | Role in filter |
|---|---|
| Phase 1 | Defined the model: $\pi$, $P$, $\theta_j$ |
| Phase 2 | `simulate` produces ground-truth $\{S_t, y_t\}$ for `test_filter_tracks_true_state` |
| Phase 3 | `Emission::log_density` is the only interface the filter uses for observations |
| **Phase 4** | Combines Phases 1–3 into the predict–update recursion |
| Phase 5 | Backward smoother will read `predicted` and `filtered` from `FilterResult` |
| Phase 6 | EM E-step will call `filter` repeatedly and use `log_likelihood` to check convergence |

---

## What comes next (Phase 5)

Phase 5 will implement the **backward smoother** (Kim smoother):

$$
\Pr(S_t = j \mid y_{1:T}) \;\prosto;\; \alpha_{t|t}(j) \;\cdot\; \beta_t(j)
$$

where $\beta_t(j)$ is the backward variable computed by a reverse pass over the observations.

The smoother requires access to every `predicted[t]` stored in `FilterResult` — which is why that field is retained rather than discarded after the filter pass.
