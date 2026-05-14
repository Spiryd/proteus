# Pairwise Posterior Transition Probabilities — Phase 8

## Purpose

After the forward filter and backward smoother, every **marginal** posterior
`Pr(Sₜ=j | y_{1:T})` is available.  Phase 8 extends this to the **joint**
posterior over two adjacent time steps:

$$\xi_t(i,j) = \Pr(S_{t-1}=i,\, S_t=j \mid y_{1:T}), \qquad t = 2,\dots,T.$$

These pairwise probabilities answer a question the smoother cannot:
> "How likely was the specific transition $i \to j$ between times $t-1$ and $t$?"

They are the structural prerequisite for EM-based transition-matrix estimation.

---

## Central Formula

$$\boxed{\xi_t(i,j) = \alpha_{t-1|t-1}(i)\; p_{ij}\; \frac{\gamma_t(j)}{\alpha_{t|t-1}(j)}}$$

This is Form B from the phase document.  It requires only quantities already
stored by the filter and smoother — no re-evaluation of emission densities.

### Derivation

Starting from the proportional form and applying the filter update identity

$$\alpha_{t|t}(j) = \frac{f_j(y_t)\,\alpha_{t|t-1}(j)}{c_t},$$

the emission density $f_j(y_t)$ and predictive normalizer $c_t$ cancel, collapsing
Form A into Form B.

### Three-part interpretation

| Factor | Symbol | Meaning |
|---|---|---|
| Past information | $\alpha_{t-1\|t-1}(i)$ | How likely was regime $i$ at $t-1$, given $y_{1:t-1}$? |
| Transition structure | $p_{ij}$ | How likely is a move $i \to j$ under the Markov chain? |
| Future correction | $\gamma_t(j)/\alpha_{t\|t-1}(j)$ | How much does the full sample revise the prediction for regime $j$ at $t$? |

### Posterior expected indicator

Introducing $I_t(i,j) = \mathbf{1}\{S_{t-1}=i, S_t=j\}$:

$$\xi_t(i,j) = \mathbb{E}[I_t(i,j) \mid y_{1:T}].$$

This is exact.  $\xi_t(i,j)$ is literally the posterior expected value of the
binary indicator that the transition $i \to j$ occurred between $t-1$ and $t$.

---

## Structural Invariants

These hold at every step $t = 2,\dots,T$ and are verified in the test suite.

### A — Nonnegativity
$$\xi_t(i,j) \ge 0 \quad \forall\, i,j.$$

Follows directly from all three factors in the formula being nonneg.

### B — Unit sum
$$\sum_{i=1}^K \sum_{j=1}^K \xi_t(i,j) = 1.$$

Proof: factor the sum over $j$ using the prediction identity
$\sum_i \alpha_{t-1|t-1}(i)\,p_{ij} = \alpha_{t|t-1}(j)$, which cancels the
denominator; the remaining sum over $\gamma_t(j)$ equals 1.

### C — Column marginal equals next-time smoothed
$$\sum_{i=1}^K \xi_t(i,j) = \gamma_t(j).$$

The column sum factors as $\frac{\gamma_t(j)}{\alpha_{t|t-1}(j)} \cdot \alpha_{t|t-1}(j) = \gamma_t(j)$.

### D — Row marginal equals previous-time smoothed
$$\sum_{j=1}^K \xi_t(i,j) = \gamma_{t-1}(i).$$

The inner sum equals the backward multiplier used by the smoother; its product
with $\alpha_{t-1|t-1}(i)$ recovers $\gamma_{t-1}(i)$.

### E — Expected transition counts are finite and nonneg
$$N_{ij}^{\text{exp}} = \sum_{t=2}^T \xi_t(i,j) \ge 0, \quad \text{finite}.$$

---

## 0-Based Index Mapping

The implementation uses 0-based indices throughout, matching `FilterResult` and
`SmootherResult`.

| Math quantity | Array access | Notes |
|---|---|---|
| $\alpha_{t-1\|t-1}(i)$ | `filtered[s][i]` | time $t-1$ is index $s = t-2$ |
| $p_{ij}$ | `params.transition_row(i)[j]` | |
| $\gamma_t(j)$ | `smoothed[s+1][j]` | time $t$ is index $s+1$ |
| $\alpha_{t\|t-1}(j)$ | `predicted[s+1][j]` | `predicted[s]` = $\alpha_{s+1\|s}$ |
| $\xi_t(i,j)$ | `xi[s][i][j]` | $s = 0,\dots,T-2$ covering $t = 2,\dots,T$ |

The loop runs `s` from `0` to `T-2`.  When T=1 there are no transitions; the
tensor is empty and `expected_transitions` is all zeros.

---

## Numerical Guard

When `predicted[s+1][j] < DENOM_FLOOR` (= 1×10⁻³⁰⁰), the predicted weight for
regime $j$ is negligibly small and the contribution to `xi[s][i][j]` is set to
zero — identical to the guard in the backward smoother.

---

## API

```rust
pub struct PairwiseResult {
    pub xi: Vec<Vec<Vec<f64>>>,              // xi[s][i][j], shape (T-1)×K×K
    pub expected_transitions: Vec<Vec<f64>>, // N_ij^exp, shape K×K
}

pub fn pairwise(
    params: &ModelParams,
    filter_result: &FilterResult,
    smoother_result: &SmootherResult,
) -> Result<PairwiseResult>
```

The function consumes only `FilterResult`, `SmootherResult`, and `ModelParams`.
It never reads the raw observation sequence.

---

## Expected Transition Counts

$$N_{ij}^{\text{exp}} = \sum_{t=2}^T \xi_t(i,j) = \sum_{s=0}^{T-2} \texttt{xi[s][i][j]}$$

Interpretation:
- $N_{ii}^{\text{exp}}$ large → regime $i$ is persistent in the posterior.
- $N_{ij}^{\text{exp}}$ large ($i \ne j$) → the posterior believes switching $i \to j$ was frequent.
- Off-diagonal counts all small → the model sees few switches (strong persistence).

The companion occupancy quantity is

$$M_i^{\text{exp}} = \sum_{t=1}^{T-1} \gamma_t(i) = \sum_{j=1}^K N_{ij}^{\text{exp}},$$

where the equality of the two expressions is Invariant D summed over time.
Computing both sides and comparing them is itself a strong validation.

---

## Connection to EM (future phase)

The M-step update for the transition matrix will use:

$$p_{ij}^{\text{new}} = \frac{N_{ij}^{\text{exp}}}{M_i^{\text{exp}}} = \frac{\sum_{t=2}^T \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}.$$

Phase 8 provides exactly $N_{ij}^{\text{exp}}$; the denominator $M_i^{\text{exp}}$
follows directly from the smoother output.

---

## Validated Behavioral Properties (12 tests)

| Test | Property |
|---|---|
| `empty_when_single_observation` | T=1: `xi` is empty, counts are zero |
| `structural_invariants_k2_persistent` | All five invariants, K=2, p_ii=0.99 |
| `structural_invariants_k2_mixing` | All five invariants, K=2, p_ii=0.4 |
| `structural_invariants_k3` | All five invariants, K=3 |
| `two_observations_one_transition` | T=2: `xi` has exactly one matrix |
| `long_sample_numerical_stability` | T=5000: no NaN/Inf, invariants hold |
| `dominant_diagonal_for_persistent_regimes` | Diagonal $N_{ii}^{\text{exp}}$ exceeds all off-diagonal entries |
| `offdiag_dominant_at_regime_switch` | $\xi(0 \to 1)$ dominates at a block switch boundary |
| `expected_counts_consistent_with_smoothed_occupancy` | $\sum_j N_{ij}^{\text{exp}} = M_i^{\text{exp}}$ (Section 13 identity) |
| `pairwise_is_deterministic` | Identical output on repeated calls |
| `multi_seed_invariants` | Invariants hold across 10 random seeds |
| `weak_separation_pairwise_remains_diffuse` | Diffuse posteriors under weak regime separation |

---

## References

- Kim, C.-J. (1994). *Dynamic linear models with Markov-switching*. Journal of Econometrics 60, 1–22.
- Hamilton, J. D. (1994). *Time Series Analysis*. Chapter 22.
- Shumway, R. H. & Stoffer, D. S. (2000). *Time Series Analysis and Its Applications*. Chapter 6.
