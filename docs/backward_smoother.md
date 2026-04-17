# Backward Smoother — Phase 7

## Purpose

The forward filter produces filtered probabilities `α_{t|t}(i) = Pr(S_t = i | y_{1:t})` —
beliefs about the current regime conditioned only on *past and present* observations.
The backward smoother extends this to full-sample retrospective inference:

```
γ_t(i) = Pr(S_t = i | y_{1:T})
```

Every estimate uses *all* observations in the sample — past, present, and future.

---

## Backward Recursion

### Terminal Condition

```
γ_T(j) = α_{T|T}(j)
```

At the last observation there is no future evidence, so the smoothed probability
equals the filtered probability.

### Backward Step (Kim 1994)

For `t = T-1, …, 1`:

```
γ_t(i) = α_{t|t}(i) · Σ_j [ p_ij · γ_{t+1}(j) / α_{t+1|t}(j) ]
```

| Symbol | Meaning |
|---|---|
| `α_{t\|t}(i)` | filtered probability at step t |
| `p_ij` | transition probability from regime i to regime j |
| `γ_{t+1}(j)` | smoothed probability at the *next* step (already computed) |
| `α_{t+1\|t}(j)` | one-step-ahead predicted probability (from the filter) |

### Denominator Rationale

`α_{t+1|t}(j) = Σ_i p_ij · α_{t|t}(i)` is the marginal weight the filter assigned
to arriving at state j at step t+1 given all evidence up to t.  Dividing by this
weight removes the double-counting of the transition — the numerator `p_ij · γ_{t+1}(j)`
already carries all future evidence via the smoothed probability; the denominator
normalises out the transition weight that the filter already applied.

---

## Binary Invariant

In exact arithmetic the recursion is self-normalising:

```
Σ_i γ_t(i) = 1   for all t
```

The proof follows by induction: assuming `Σ_j γ_{t+1}(j) = 1`, expand
`Σ_i γ_t(i)` and apply the law of total probability to recover 1.

In floating-point arithmetic, errors accumulate.  The implementation therefore
re-normalises each smoothed vector after computing it as a precaution.

---

## Numerical Guard

When `α_{t+1|t}(j) < DENOM_FLOOR` (= 1 × 10⁻³⁰⁰), the predicted weight is
effectively zero and the contribution from regime j is skipped.  This avoids
division-by-zero without altering the result meaningfully, since the corresponding
`γ_{t+1}(j)` will also be negligible.

---

## 0-Based Array Mapping

The filter stores arrays indexed from 0 to T-1.

| Math | Code | Meaning |
|---|---|---|
| `α_{t\|t}(·)` | `filtered[t-1]` | filtered probs at time t |
| `α_{t\|t-1}(·)` | `predicted[t-1]` | predicted probs at time t |
| `γ_t(·)` | `smoothed[t-1]` | smoothed probs at time t |

In the backward loop `s` runs from `T-2` down to `0` (0-based), computing
`smoothed[s]` from `smoothed[s+1]` and `predicted[s+1]`.

---

## API

```rust
pub struct SmootherResult {
    pub t: usize,            // number of observations
    pub k: usize,            // number of regimes
    pub smoothed: Vec<Vec<f64>>,   // smoothed[s][j] = γ_{s+1}(j)
}

pub fn smooth(params: &ModelParams, filter_result: &FilterResult) -> Result<SmootherResult>
```

`smooth` only requires the `FilterResult` from a previous call to `filter()`.
Parameters are needed solely to access the transition matrix `A`.

---

## Validated Behavioral Properties (12 tests)

| Test | Property |
|---|---|
| `terminal_condition_exact` | `smoothed[T-1] == filtered[T-1]` exactly |
| `structural_invariants_k2_scenarios` | All entries in [0,1], rows sum to 1, over 6 scenario types |
| `single_observation_smoothed_equals_filtered` | T=1 edge case |
| `structural_invariants_k3` | Invariants hold for K=3 |
| `long_sample_numerical_stability` | T=5000 without overflow/NaN |
| `smoothed_accuracy_at_least_as_good_as_filtered` | L1 error of smoother ≤ filter at each t |
| `smoother_more_decisive_at_regime_boundary` | Lower entropy at regime crossings |
| `interior_stable_run_filter_and_smooth_converge` | Agree in the deep interior of a stable block |
| `smoother_revises_beliefs_at_transition` | Smoother flips to new regime ≤ same step as filter |
| `weak_separation_smoothed_may_remain_diffuse` | Both remain uncertain when regimes overlap heavily |
| `smooth_is_deterministic_given_filter_result` | Identical output on repeated calls |
| `multi_seed_short_runs` | Invariants hold across 10 random seeds |

---

## References

- Kim, C.-J. (1994). *Dynamic linear models with Markov-switching*. Journal of Econometrics 60, 1–22.  
- Hamilton, J. D. (1994). *Time Series Analysis*. Chapter 22.
