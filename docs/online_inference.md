# Online (Streaming) Inference — Phase 11

## Purpose

Phase 11 recasts the Gaussian Markov Switching Model from an offline analysis
tool into a causally usable streaming inference backbone.

Up to Phase 10, all inference was retrospective: the strongest posterior was
the smoother output $\gamma_t(j) = \Pr(S_t=j \mid y_{1:T})$, which conditions
on the entire observed sample.  That object is correct for estimation,
diagnostics, and EM, but it violates the causal rule for streaming systems.

Phase 11 establishes a strict online mode governed by one rule:

$$\boxed{\text{At time } t,\ \text{the online system may use only } y_{1:t}.}$$

---

## Offline vs online — the fundamental split

| Property | Offline | Online |
|---|---|---|
| Primary posterior | $\gamma_t(j) = \Pr(S_t=j \mid y_{1:T})$ | $\alpha_{t\vert t}(j) = \Pr(S_t=j \mid y_{1:t})$ |
| Conditions on | Full sample $y_{1:T}$ | Observations up to $t$ only |
| Requires future data | Yes | No |
| Execution model | Batch (whole array) | State machine (one step) |
| Module | `src/model/` | `src/online/` |

### Offline components (training-time only)

- EM estimation (`model::em`)
- Backward smoothing (`model::smoother`)
- Pairwise full-sample posteriors (`model::pairwise`)
- Post-fit diagnostics (`model::diagnostics`)

### Online components (runtime only)

- One-step prediction: $\alpha_{t|t-1}$
- One-step Bayes update: $\alpha_{t|t}$
- Predictive density: $c_t$
- Cumulative log-score: $\sum_{s=1}^t \log c_s$

---

## Forbidden operations in streaming mode

| Operation | Why forbidden |
|---|---|
| $\gamma_t(j) = \Pr(S_t=j \mid y_{1:T})$ | Requires future observations |
| $\xi_t(i,j) = \Pr(S_{t-1}=i, S_t=j \mid y_{1:T})$ | Derived from smoothing |
| EM M-step inside the streaming loop | Mixes training with runtime |
| Any alarm based on smoothed quantities | Makes online benchmarks invalid |

The `src/online/` module enforces this at the module level: it has no
dependency on `model::smoother`, `model::pairwise`, `model::em`, or
`model::diagnostics`.

---

## The streaming recursion

The mathematical content is the same forward filter recursion from Phase 4.
What changes is the execution model: instead of a batch loop over a fixed
array, it is a one-step state machine transition.

Given the previous filtered state $\alpha_{t-1|t-1}$ and a new observation
$y_t$:

**Step 1 — Predict:**
$$\alpha_{t|t-1}(j) = \sum_{i=1}^K p_{ij}\,\alpha_{t-1|t-1}(i)$$

**Step 2 — Emission scores:**
$$f_j(y_t) = \mathcal{N}(y_t;\,\mu_j,\sigma_j^2)$$

**Step 3 — Predictive density:**
$$c_t = \sum_{j=1}^K f_j(y_t)\,\alpha_{t|t-1}(j)$$

**Step 4 — Bayes update:**
$$\alpha_{t|t}(j) = \frac{f_j(y_t)\,\alpha_{t|t-1}(j)}{c_t}$$

**Step 5 — One-step-ahead prediction (next step's input):**
$$\alpha_{t+1|t}(j) = \sum_{i=1}^K p_{ij}\,\alpha_{t|t}(i)$$

Steps 4 and 5 are computed in the same call.  No extra inference is needed.

### Numerics

Steps 2 and 3 are computed in log-space:

$$\log w_j = \log f_j(y_t) + \log \alpha_{t|t-1}(j)$$

$$\log c_t = \log\!\sum_j \exp\!\bigl(\log w_j\bigr)$$

using the log-sum-exp identity to avoid underflow.

**Step 4 is also computed in log-space:**

$$\alpha_{t|t}(j) = \exp\!\bigl(\log w_j - \log c_t\bigr)$$

This formulation is numerically robust even when $c_t = \exp(\log c_t)$ underflows to exactly 0 in `f64` — which can occur for extreme observations in high-frequency (e.g., 15-min intraday) data. Computing the ratio in log-space avoids the division-by-zero that would otherwise arise. The scalar $c_t$ is still computed from `exp(log_ct)` for the `OnlineStepResult` field (it may be 0 for extreme steps; this is recorded but does not cause an error).

---

## Initialization

Before the first observation, the state is set to:

$$\alpha_{1|0}(j) = \pi_j$$

This is the predicted distribution for time $t=1$ given no observations.  The
first `step` call consumes $y_1$ and produces $\alpha_{1|1}$ and $c_1$.

---

## Persistent state

`OnlineFilterState` stores only what is needed for the next step:

| Field | Value |
|---|---|
| `filtered` | $\alpha_{t\vert t}(j)$, length $K$ |
| `t` | observations processed so far |
| `cumulative_log_score` | $\sum_{s=1}^t \log c_s$ |

No full-sample arrays, no smoothed quantities, no EM history.

After processing $T$ observations, `cumulative_log_score` equals the
observed-data log-likelihood $\log L(\Theta) = \sum_{t=1}^T \log c_t$ that the
offline `filter()` would return for the same sequence.

---

## API

```rust
/// Persistent streaming inference state.
pub struct OnlineFilterState {
    pub filtered: Vec<f64>,          // α_{t|t}(j), length K
    pub t: usize,                    // observations processed
    pub cumulative_log_score: f64,   // Σ log c_s
}

impl OnlineFilterState {
    /// Initialize from fitted parameters.  Sets filtered = π, t = 0.
    pub fn new(params: &ModelParams) -> Self

    /// Process one new observation.  Mutates self in-place.
    /// params is read-only — never modified.
    pub fn step(&mut self, y: f64, params: &ModelParams) -> Result<OnlineStepResult>

    /// Process a slice of observations sequentially.
    pub fn step_batch(&mut self, obs: &[f64], params: &ModelParams) -> Result<Vec<OnlineStepResult>>
}

/// Causal output of one streaming step.
/// Contains only quantities measurable with respect to y_{1:t}.
pub struct OnlineStepResult {
    pub filtered: Vec<f64>,         // α_{t|t}(j)
    pub predicted_next: Vec<f64>,   // α_{t+1|t}(j)
    pub predictive_density: f64,    // c_t
    pub log_predictive: f64,        // log c_t
    pub t: usize,                   // 1-based observation count
}
```

`OnlineStepResult` has no smoothed fields.  A downstream changepoint detector
layer cannot accidentally access offline-only quantities.

---

## Runtime invariants

Checked on every `step` call; returns `Err` on violation rather than
propagating NaN:

| Invariant | Check |
|---|---|
| Filtered normalization | $\lvert\sum_j \alpha_{t\vert t}(j) - 1\rvert < 10^{-9}$ |
| Predicted normalization | $\lvert\sum_j \alpha_{t+1\vert t}(j) - 1\rvert < 10^{-9}$ |
| Positive predictive density | $c_t > 0$ |
| Finite log-predictive | $\log c_t \ne \pm\infty, \ne \text{NaN}$ |
| No NaN in filtered | all entries finite |

---

## Module structure

```
src/
  model/           ← offline stack (unchanged)
    filter.rs
    smoother.rs    ← offline-only; not reachable from online/
    em.rs          ← offline-only
    diagnostics.rs ← offline-only
    ...
  online/
    mod.rs         ← streaming inference only
```

Dependencies of `src/online/mod.rs`:

```
online::OnlineFilterState
    → model::params::ModelParams   (read-only reference)
    → model::emission::Emission    (log-density evaluation)
```

No dependency on `smoother`, `pairwise`, `em`, or `diagnostics`.

---

## Consistency with offline filter

`OnlineFilterState` and `FilterResult` from Phase 4 are parallel, not
interchangeable:

- `FilterResult` stores the *full* batch output — all filtered and predicted
  columns for all $T$ time steps.
- `OnlineFilterState` stores only the *current* filtered vector.

**Consistency invariant:** running `step_batch` on $y_{1:T}$ produces
`filtered[t]` that matches `FilterResult::filtered[t]` at every step, and
`cumulative_log_score` that matches `FilterResult::log_likelihood`, both to
floating-point precision ($< 10^{-8}$).  This is verified in the test suite.

---

## Validated properties (15 tests)

### Initialization (2)

| Test | Property |
|---|---|
| `init_state_matches_pi` | `filtered == π`, `t == 0`, `cumulative_log_score == 0.0` |
| `init_k3_correct_length` | K=3 params → length-3 filtered vector |

### Single-step correctness (4)

| Test | Property |
|---|---|
| `step_filtered_sums_to_one` | $\sum_j \alpha_{t\vert t}(j) = 1$ after one step |
| `step_predictive_density_matches_manual_at_t1` | $c_1 = \sum_j f_j(y_1)\pi_j$ exactly |
| `step_predicted_next_sums_to_one` | $\sum_j \alpha_{t+1\vert t}(j) = 1$ after one step |
| `step_increments_t` | `t == 1` after first step |

### Multi-step consistency (4)

| Test | Property |
|---|---|
| `cumulative_log_score_matches_offline_filter` | Online log-score equals offline log-likelihood to $10^{-8}$ |
| `filtered_matches_offline_filter_at_every_step` | `filtered[j]` matches `FilterResult::filtered[s][j]` to $10^{-12}$ at every step |
| `predicted_next_matches_offline_filter_predicted` | `predicted_next[j]` matches `FilterResult::predicted[s+1][j]` to $10^{-12}$ |
| `step_batch_equals_step_loop` | `step_batch` and loop over `step` produce identical results |

### Edge cases (3)

| Test | Property |
|---|---|
| `edge_case_t1_no_panic` | T=1 single observation: no panic, `t == 1` |
| `edge_case_t2_transition_applied` | T=2: filtered sums to 1 after second step |
| `extreme_observation_returns_err` | Observation with near-zero density returns `Err`, not panic |

### State machine properties (2)

| Test | Property |
|---|---|
| `step_result_has_only_causal_fields` | `OnlineStepResult` has no smoothed fields (compile-time boundary) |
| `independent_states_evolve_identically` | Two independent `OnlineFilterState` instances produce identical results |

---

## What Phase 11 does not yet define

- What counts as a changepoint
- How alarms are triggered
- How thresholds are calibrated
- How online performance is benchmarked
- Adaptive online parameter learning

These are later phases.  Phase 11 only ensures that the regime model runs
causally on a stream — forming a clean foundation for the changepoint detector
to be built on top.

---

## Implementation Note: Two Streaming APIs

Two code paths advance the filter step-by-step.

### `OnlineFilterState` — production path

```rust
// src/detector/frozen.rs  (also re-exported via experiments/shared.rs)
let mut state = OnlineFilterState::new(&params);
for obs in stream {
    let step = state.step(obs);   // returns FilterStep { predicted, filtered, log_p }
    // detector sees step.log_p
}
```

`OnlineFilterState` is the path called by `ExperimentRunner::run` via
`src/experiments/shared.rs::run_online_detection`.  It is the **only path
that produces thesis results**.

### `StreamingSession` — demo / manual testing wrapper

`StreamingSession` (also in `src/detector/frozen.rs`) is a higher-level
convenience struct that wraps `OnlineFilterState` together with a
`ChangePointDetector`.  It is **not** called from the experiment runner and
does not appear in any registered experiment.  It exists for:

- interactive demos and CLI probing,
- unit tests that want a one-shot session lifecycle,
- future work that adds a REPL-style streaming interface.

Both paths produce identical filter numerics — they share the same
`ModelParams` and call the same `filter::predict` / `bayes_update` internals.

---

## References

- Hamilton, J. D. (1989). *A new approach to the economic analysis of
  nonstationary time series.* Econometrica 57(2), 357–384.
- Kim, C.-J. (1994). *Dynamic linear models with Markov-switching.*
  Journal of Econometrics 60, 1–22.
