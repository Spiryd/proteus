# Phase 6 — Forward Filter Validation

## Purpose

Phase 6 validates the Hamilton forward filter (Phase 4) on simulated data where
both the true parameters **Θ** and the hidden state path **S₁:T** are known.
The goal is to confirm that the filter is:

1. **Structurally correct** — probability vectors stay normalized, finite, and
   non-negative at every time step.
2. **Behaviorally plausible** — sharpness, responsiveness, and stability vary
   with emission separation and transition persistence in the directions predicted
   by theory.

No parameter estimation is performed; that begins in Phase 8 (EM algorithm).

---

## Validation workflow

For every test scenario:

```
ModelParams (Θ)
      │
      ▼
  simulate()  ──► (y₁:T, S₁:T)     ← known truth
      │
      ▼
   filter()   ──► FilterResult
      │
      ├─► assert_prob_vec(predicted[t])  for all t    ← normalization
      ├─► assert_prob_vec(filtered[t])   for all t    ← normalization
      ├─► log_predictive[t] is finite    for all t
      └─► log_likelihood == Σ log_predictive[t]       ← additive consistency
```

---

## Scenario families

### A — Strongly separated regimes

| Parameter | Value |
|-----------|-------|
| μ         | (−10, +10) |
| σ²        | (1, 1)     |
| p_ii      | 0.99       |

**Expectations**: mean max filtered probability > 0.97; argmax classification
accuracy > 98%.

### B — Weakly separated regimes

| Parameter | Value |
|-----------|-------|
| μ         | (−0.5, +0.5) |
| σ²        | (1, 1)       |
| p_ii      | 0.90         |

**Expectation**: posteriors are more diffuse than in scenario A.  The test
confirms that mean max filtered probability for scenario B is strictly lower
than for scenario A on comparable data.

### C — Mean-switching only

| Parameter | Value |
|-----------|-------|
| μ         | (−5, +5) |
| σ²        | (2, 2)   |  ← equal variances
| p_ii      | 0.95     |

**Expectation**: the filter tracks the location of observations; argmax
classification accuracy > 75%.

### D — Variance-switching only

| Parameter | Value |
|-----------|-------|
| μ         | (0, 0) |   ← equal means
| σ²        | (0.25, 9) |
| p_ii      | 0.95       |

**Expectations**:
- A single extreme observation y = 3 should favor the high-variance regime
  (regime 1) since P(y=3 | N(0, 0.25)) << P(y=3 | N(0, 9)).
- Log-likelihood remains finite over a 2 000-step run.

### E — Highly persistent transitions

| Parameter | Value |
|-----------|-------|
| μ         | (−3, +3) |
| σ²        | (1, 1)   |
| p_ii      | 0.999    |

**Expectation**: The prediction step introduces only tiny mixing so
|predicted[t][0] − filtered[t-1][0]| > 0.05 on fewer than 0.5% of steps.

Comparison test (E vs F): a switching-matrix filter applied to the same data
should have larger mean step-to-step changes in filtered probabilities than a
persistent-matrix filter.

### F — Frequent switching

| Parameter | Value |
|-----------|-------|
| μ         | (−5, +5) |
| σ²        | (1, 1)   |
| p_ii      | 0.40     |

The stationary distribution of this matrix is (0.5, 0.5).  **Expectation**:
mean |predicted[t][0] − 0.5| < 0.20 because the prediction step contracts
toward the stationary distribution at every step.

### G — Short samples (T = 10)

**Expectation**: structural invariants hold for every seed in 50 independent
runs even with minimal data.

### H — Long samples (T = 50 000)

**Expectations**:
- Final filtered distribution is a valid probability vector.
- Log-likelihood is finite.
- Per-step average log-likelihood lies in (−10, 0) over T = 20 000 (not
  compounding underflow or overflow).

---

## Regime-change boundary tests

| Test | Design | Assertion |
|------|--------|-----------|
| `posterior_shifts_at_hard_regime_change` | 100 obs at −10 then 100 at +10 | filtered[0][90] > 0.95; filtered[1][190] > 0.95 |
| `stronger_evidence_speeds_posterior_recovery` | μ = ±5 (strong) vs μ = ±0.5 (weak); block switch at t = 50 | steps-to-flip(strong) < steps-to-flip(weak) |
| `persistent_transitions_slow_posterior_recovery` | Same block obs, p_ii = 0.99 vs 0.70 | steps-to-flip(persistent) ≥ steps-to-flip(less-persistent) |

For `stronger_evidence_speeds_posterior_recovery`, the key contrast is that
observations placed exactly at μ₁ = +0.5 when σ² = 1 provide only a
likelihood ratio of ≈ e^{0.5} ≈ 1.65 per step, so the filter accumulates
evidence slowly.  With μ₁ = +5 and σ² = 1, the likelihood ratio per
post-switch observation is ≈ e^{50}, driving an immediate flip.

---

## Three-regime (K = 3) tests

| Parameter | Value |
|-----------|-------|
| K         | 3 |
| π         | (⅓, ⅓, ⅓) |
| μ         | (−8, 0, +8) |
| σ²        | (1, 1, 1) |
| p_ii      | 0.90 / 0.95 |

**Tests**:
- Structural invariants hold over T = 3 000.
- Argmax classification accuracy > 90% over T = 5 000 with p_ii = 0.95.

---

## Numerical stability notes

All convex-combination operations in the filter are performed in log-space with
a single `log_sum_exp` exponentiation point per step, bounding numerical
error.  The scenario-H tests (T = 50 000) confirm that neither the predicted
nor filtered distributions drift outside [0, 1] after many predict-update
cycles.

---

## Test summary

| Scenario | Tests | All pass |
|----------|-------|----------|
| A — Strongly separated | Invariants, sharp posteriors, accuracy ≥ 98% | ✅ |
| B — Weakly separated | Invariants, diffuse posteriors < A | ✅ |
| C — Mean-switching | Invariants, accuracy ≥ 75% | ✅ |
| D — Variance-switching | Invariants, single-obs direction, finite LL | ✅ |
| E — Highly persistent | Invariants, smooth evolution (< 0.5% large jumps) | ✅ |
| F — Frequent switching | Invariants, predicted near stationary | ✅ |
| G — Short samples | Invariants ×50 seeds | ✅ |
| H — Long samples | Invariants, finite LL, stable per-step LL | ✅ |
| E vs F comparison | Switching > persistent step-change | ✅ |
| Regime-change boundary | Hard block switch; evidence speed; persistence slowdown | ✅ |
| K = 3 three-regime | Invariants, accuracy ≥ 90% | ✅ |
| **Total** | **25** | **25 / 25** |
