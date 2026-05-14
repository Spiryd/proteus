# Proteus — Model Chapter: Code-to-Theory Reference

**Scope:** Every `src/model/*.rs` file that is **live in the end-to-end experiment pipeline**.
`src/model/validation.rs` is excluded (scenario-only test fixtures, no live caller).

Status legend:

- **LIVE** — called from `ExperimentRunner` stages, reachable via every CLI experiment subcommand.
- **LIVE (OPTIMIZER)** — called from `direct_optimize` / `cmd_param_search`, not from the primary EM training loop.
- **EXCLUDED** — dead in the live pipeline (test fixtures only).

---

## E2E Pipeline Stage Map (model modules only)

```
CLI dispatch
    │
    ▼ Stage 2 ── resolve_data
    │   SyntheticBackend  →  simulate::simulate()         [src/model/simulate.rs]
    │                        ModelParams (init params)    [src/model/params.rs]
    │
    ▼ Stage 4 ── train_or_load_model
    │   fit_em(obs, init_params, config)                  [src/model/em.rs]
    │     E-step loop:
    │       filter(obs, params)        → FilterResult     [src/model/filter.rs]
    │         └── Emission::log_density(y, j, params)     [src/model/emission.rs]
    │       smooth(filter_result, params) → SmootherResult [src/model/smoother.rs]
    │       pairwise(filter, smoother, params) → PairwiseResult [src/model/pairwise.rs]
    │     M-step: updates ModelParams
    │   diagnose(em_result, obs)                          [src/model/diagnostics.rs]
    │     └── runs filter + smooth + pairwise once more
    │
    ▼ Stage 5 ── run_online
    │   OnlineFilterState::step(y, params)  [src/online/mod.rs]
    │     mirrors Hamilton recursion using same ModelParams + Emission
    │
    ▼ Optimizer path (direct_optimize / param_search only)
        log_likelihood(obs, params)                       [src/model/likelihood.rs]
        compare_runs(runs)                                [src/model/diagnostics.rs]
```

---

## 1. `params.rs` — Model parameter container θ = (π, A, μ, σ²)

### Theoretical object

The complete parametrisation of a K-regime Gaussian MSM:

$$\theta = (\pi,\, A,\, \mu_1,\dots,\mu_K,\, \sigma_1^2,\dots,\sigma_K^2)$$

where $\pi \in \Delta^{K-1}$, $A \in \mathbb{R}^{K \times K}$ row-stochastic, $\sigma_k^2 > 0$.

### Expected invariants

$$\sum_j \pi_j = 1, \quad \sum_j A_{ij} = 1 \; \forall i, \quad \sigma_k^2 > 0 \; \forall k$$

### Key types

| Type / function | Role |
|---|---|
| `ModelParams { k, pi, transition, means, variances }` | Canonical θ container; `transition` stored flat row-major, length `k²` |
| `ModelParams::new(pi, transition_rows, means, variances)` | Constructor; concatenates rows internally |
| `ModelParams::validate()` | Enforces K ≥ 2, simplex constraints, positive variances |
| `ModelParams::transition_row(i)` | Extracts row i of A for use in predict step |

### Stage usage

| Stage | How params enter |
|---|---|
| Stage 2 (synthetic) | Passed as ground-truth θ to `simulate` |
| Stage 4 | Initialised from config; mutated by M-step each EM iteration; final θ̂ serialised to `model_params.json` |
| Stage 5 | Passed frozen to `OnlineFilterState::step` per bar |

### Artifacts

`model_params.json`, `config.snapshot.json` (initial params echoed), `fit_summary.json` (subset).

### Tests

`validate()` rejects: k < 2, non-simplex π, non-stochastic rows, non-positive variances.

### Status

**LIVE** — referenced by every module and every CLI subcommand.

---

## 2. `emission.rs` — Gaussian log-density $\log f_j(y_t)$

### Theoretical object

$$y_t \mid S_t = j \;\sim\; \mathcal{N}(\mu_j,\, \sigma_j^2)$$

$$\log f_j(y_t) = -\tfrac{1}{2}\ln(2\pi) - \tfrac{1}{2}\ln\sigma_j^2 - \frac{(y_t - \mu_j)^2}{2\sigma_j^2}$$

### Key types

| Type / function | Role |
|---|---|
| `Emission { k, means, variances }` | Holds (μ, σ²) extracted from `ModelParams` |
| `Emission::log_density(y, j) -> f64` | Returns $\log f_j(y_t)$; called once per regime per time step inside `filter` |
| `const HALF_LN_2PI: f64` | Precomputed $\tfrac{1}{2}\ln(2\pi) \approx 0.9189$ — single definition site for the Gaussian constant |

### Stage usage

`Emission` is constructed inside `filter()` and `OnlineFilterState::step()`. It is never instantiated by application code directly; it is strictly an implementation detail of those two callers.

### Artifacts

None direct — densities feed the normalisation constants $c_t$ which appear in `loglikelihood_history.csv` and `score_trace.csv`.

### Tests

11 unit tests: boundary variance, large $|y - \mu|$, agreement with closed-form.

### Status

**LIVE** — called on every bar, every iteration, by both the batch filter and the online step.

---

## 3. `simulate.rs` — Gaussian MSM data-generating process

### Theoretical object

Draw one realisation $(S_{1:T},\, y_{1:T})$ from θ:

$$S_1 \sim \pi, \quad S_t \mid S_{t-1} = i \sim A_{i,\cdot}, \quad y_t \mid S_t = j \sim \mathcal{N}(\mu_j, \sigma_j^2)$$

### Key types

| Type / function | Role |
|---|---|
| `SimulationResult { states, observations }` | Full sample: hidden path + observations |
| `SimulationResult::changepoints() -> Vec<usize>` | 1-based indices where $S_t \neq S_{t-1}$; used as ground truth for benchmarking |
| `simulate(params, T, rng) -> SimulationResult` | Standard draw from the model |
| `simulate_with_jump(params, T, JumpParams, rng)` | Adds shock contamination $N(0, (\text{mult}\cdot\sigma_j)^2)$ with probability `prob` per bar |
| `JumpParams { prob, scale_mult }` | Jump-contamination hyper-parameters |

### Stage usage

Called exclusively in **Stage 2** by `SyntheticBackend::resolve_data`. The returned `SimulationResult::changepoints()` populates `DataBundle::changepoint_truth`, which is later written to `changepoints.csv` and consumed by `ChangePointTruth` in the benchmark evaluation.

### Data flow

```
SyntheticBackend::resolve_data
    → simulate(params, T, rng)
    → SimulationResult
        .observations   → DataBundle.observations (fed to Stage 3 features)
        .changepoints() → DataBundle.changepoint_truth (ground truth for Stage 6)
```

### Artifacts

`changepoints.csv` (truth), `result.json` (embeds simulation metadata).

### Tests

17 unit tests: state-distribution convergence, deterministic seeding, jump injection correctness.

### Status

**LIVE** — every synthetic experiment invokes `simulate`.

---

## 4. `filter.rs` — Hamilton forward filter $\hat\alpha_{t|t}$

### Theoretical object

Sequential Bayesian update producing filtered regime posteriors:

$$\hat\alpha_{t|t-1}(j) = \sum_i A_{ij}\,\hat\alpha_{t-1|t-1}(i) \qquad \text{(predict)}$$

$$c_t = \sum_j f_j(y_t)\,\hat\alpha_{t|t-1}(j) \qquad \text{(predictive density)}$$

$$\hat\alpha_{t|t}(j) = \frac{f_j(y_t)\,\hat\alpha_{t|t-1}(j)}{c_t} \qquad \text{(update)}$$

$$\log L(\theta) = \sum_t \log c_t$$

### Implementation notes

All per-step products $f_j(y_t)\,\hat\alpha_{t|t-1}(j)$ are computed in log-space and combined via numerically stable log-sum-exp before exponentiating once — preventing underflow when observations lie far from regime means.

### Key types

| Type / function | Role |
|---|---|
| `FilterResult { t, k, predicted, filtered, log_predictive, log_likelihood }` | Full output; both `predicted` and `filtered` retained (smoother needs `predicted`) |
| `filter(obs, params) -> Result<FilterResult>` | Runs the full T-step recursion |
| `log_predictive[t]` | $\log c_{t+1}$ per bar; feeds `SurpriseDetector` score |
| `log_likelihood` | $\sum_t \log c_t$; used by EM convergence check |

### Stage usage

| Caller | Purpose |
|---|---|
| `fit_em` E-step (Stage 4) | Produces `FilterResult` that feeds `smooth` and `pairwise` |
| `diagnose` (Stage 4, post-fit) | Final validation pass under fitted θ̂ |
| `log_likelihood` wrapper | Thin shell for optimizer path |

`filter` is the **computational core**: emission evaluation, transition prediction, and likelihood accumulation all happen here.

### Call chain within EM

```
fit_em  →  e_step  →  filter(obs, θ^(m))  →  FilterResult
                   →  smooth(FilterResult, θ^(m))
                   →  pairwise(FilterResult, SmootherResult, θ^(m))
```

### Artifacts

`regime_posteriors.csv` (online filtered posteriors, Stage 5), `loglikelihood_history.csv` (training LL per iteration).

### Tests

14 unit tests: numerical stability, conjugacy with smoother, monotone log-likelihood under EM.

### Status

**LIVE** — called on every EM iteration and every post-fit diagnostic pass.

---

## 5. `smoother.rs` — Kim backward smoother $\hat\alpha_{t|T}$

### Theoretical object

Revise filtered posteriors using future observations (Kim / Hamilton–Kim smoother):

$$\hat\alpha_{T|T}(j) = \hat\alpha_{T|T}(j) \qquad \text{(terminal condition)}$$

$$\hat\alpha_{t|T}(i) = \hat\alpha_{t|t}(i)\,\sum_j \frac{A_{ij}\,\hat\alpha_{t+1|T}(j)}{\hat\alpha_{t+1|t}(j)}, \quad t = T-1,\dots,1$$

with `DENOM_FLOOR = 1e-300` guarding zero-division on negligible predicted probabilities. Each step is renormalised so probabilities remain on the simplex.

### Key types

| Type / function | Role |
|---|---|
| `SmootherResult { t, k, smoothed }` | `smoothed[t][j]` = $\hat\alpha_{t+1|T}(j)$, 0-based indexing matching `FilterResult` |
| `smooth(filter_result, params) -> Result<SmootherResult>` | Consumes only `FilterResult` and `ModelParams`; never re-evaluates emissions |

### Stage usage

Called **only during EM training** (Stage 4), inside `e_step`. Smoothed posteriors $\gamma_t(j)$ are the weights for the M-step mean and variance updates. They are not written to disk by default (only the online filtered posteriors appear in `regime_posteriors.csv`).

### Call chain

```
e_step → filter → FilterResult
       → smooth(FilterResult, θ) → SmootherResult
       → pairwise(FilterResult, SmootherResult, θ) → PairwiseResult
```

### Artifacts

None direct — smoothed posteriors are consumed internally by `pairwise` and M-step; they do not appear in exported files.

### Tests

Conjugacy with `filter`, marginal-consistency on stationary chains, finite-difference vs. brute-force on K=2, T≤5.

### Status

**LIVE** — called on every EM E-step iteration.

---

## 6. `pairwise.rs` — Joint posteriors $\xi_t(i,j)$ and expected transitions

### Theoretical object

$$\xi_t(i,j) = p(S_{t-1}=i,\, S_t=j \mid y_{1:T}), \quad t = 2,\dots,T$$

$$\xi_t(i,j) = \frac{\hat\alpha_{t-1|t-1}(i)\,A_{ij}\,\hat\alpha_{t|T}(j)}{\hat\alpha_{t|t-1}(j)}$$

Accumulated expected transition counts used by the M-step:

$$N_{ij}^{\text{exp}} = \sum_t \xi_t(i,j)$$

### Structural invariants

- Non-negativity: $\xi_t(i,j) \ge 0$
- Unit sum over all $(i,j)$: $\sum_{i,j} \xi_t(i,j) = 1$
- Column marginal: $\sum_i \xi_t(i,j) = \gamma_t(j)$
- Row marginal: $\sum_j \xi_t(i,j) = \gamma_{t-1}(i)$

### Key types

| Type / function | Role |
|---|---|
| `PairwiseResult { xi, expected_transitions }` | Per-step $\xi_t$ tensor + accumulated $N^{\text{exp}}$ |
| `pairwise(filter_result, smoother_result, params) -> Result<PairwiseResult>` | Computes xi and sums to expected_transitions; consumes no raw observations |

### Stage usage

Called **only during EM training** (Stage 4), as the third call in `e_step`. `expected_transitions` is the direct input to the M-step transition update:

$$A_{ij}^{(m+1)} = N_{ij}^{\text{exp}} \;/\; \sum_j N_{ij}^{\text{exp}}$$

(rows with $\sum_j N_{ij}^{\text{exp}} < \text{WEIGHT\_FLOOR}$ are frozen at current values).

### Artifacts

Indirect — drives the updated $A$ in `model_params.json`.

### Tests

12 unit tests including all four marginal-consistency invariants.

### Status

**LIVE** — called on every EM E-step iteration.

---

## 7. `em.rs` — Baum–Welch EM estimation `fit_em`

### Theoretical object

Iterative maximisation of the expected complete-data log-likelihood:

$$\theta^{(m+1)} = \arg\max_\theta\; Q(\theta \mid \theta^{(m)})$$

### E-step (one pass)

$$\gamma_t^{(m)}(j) = p(S_t=j \mid y_{1:T};\,\theta^{(m)}) \qquad \text{(smoothed marginals via `smooth`)}$$

$$\xi_t^{(m)}(i,j) = p(S_{t-1}=i,S_t=j \mid y_{1:T};\,\theta^{(m)}) \qquad \text{(via `pairwise`)}$$

### M-step (closed-form Gaussian)

$$\pi_j^{(m+1)} = \gamma_1^{(m)}(j)$$

$$A_{ij}^{(m+1)} = N_{ij}^{(m)} \;/\; M_i^{(m)}, \quad N_{ij}^{(m)} = \sum_t \xi_t^{(m)}(i,j), \quad M_i^{(m)} = \sum_{t=1}^{T-1} \gamma_t^{(m)}(i)$$

$$\mu_j^{(m+1)} = \frac{\sum_t \gamma_t^{(m)}(j)\,y_t}{W_j^{(m)}}, \qquad (\sigma_j^2)^{(m+1)} = \frac{\sum_t \gamma_t^{(m)}(j)(y_t - \mu_j^{(m+1)})^2}{W_j^{(m)}}$$

### Degenerate-case guards

| Constant | Value | Purpose |
|---|---|---|
| `WEIGHT_FLOOR` | `1e-10` | Skip transition / mean / variance update when regime weight $W_j < \epsilon$ |
| `VAR_FLOOR` | `1e-6` | Clamp variance after M-step to prevent collapse |
| `MONOTONE_TOL` | `1e-8` | Warn (not error) if LL decreases by more than $\epsilon$ across one iteration |

### Key types

| Type / function | Role |
|---|---|
| `EmConfig { tol, max_iter, var_floor }` | Convergence configuration |
| `EStepResult { smoothed, expected_transitions, log_likelihood }` | Bundle from one E-step; sole input to M-step |
| `EmResult { params, ll_history, n_iter, converged }` | Output of full EM run |
| `fit_em(obs, init_params, config) -> Result<EmResult>` | Main entry point; orchestrates E/M loop |

### Internal call graph

```
fit_em
  └── loop until convergence:
        e_step(obs, θ):
          filter(obs, θ)        → FilterResult      [filter.rs]
          smooth(filter, θ)     → SmootherResult    [smoother.rs]
          pairwise(f, s, θ)     → PairwiseResult    [pairwise.rs]
          return EStepResult { smoothed, expected_transitions, log_likelihood }
        m_step(obs, e_step_result, θ) → θ_new
        check |ΔLL| < tol
```

### Stage usage

**Stage 4** (`train_or_load_model`) when `TrainingMode::FitOffline`. Multi-start variant wraps `fit_em` in a loop over `em_n_starts` random initialisations and keeps the best LL run.

### Artifacts

`fit_summary.json`, `model_params.json`, `loglikelihood_history.csv`.

### Tests

15 unit tests: monotonicity, recovery of true θ on synthetic data, behaviour at variance floor.

### Status

**LIVE** — core of Stage 4 for every training experiment.

---

## 8. `diagnostics.rs` — Post-fit trust layer

### Theoretical object

After EM converges, verify that the fitted θ̂ is mathematically valid, numerically coherent, and represents genuinely distinct persistent regimes. Six questions:

1. **Parameter validity** — is θ̂ on the constraint set?
2. **Posterior coherence** — do filtered / smoothed / pairwise outputs satisfy their invariants?
3. **EM convergence quality** — was the LL path monotone and well-behaved?
4. **Regime interpretation** — distinct means, non-degenerate occupancy, finite expected durations?
5. **Multi-start stability** — do `em_n_starts` > 1 runs agree?
6. **Trust flag** — is the model safe to pass downstream?

### Key types

| Type / function | Role |
|---|---|
| `FittedModelDiagnostics` | Full bundle: `param_validity`, `posterior_validity`, `convergence`, `regimes`, `warnings`, `is_trustworthy` |
| `ParamValidity` | max simplex deviation, min variance |
| `PosteriorValidity` | max filtered / smoothed / pairwise deviation from 1-normalization |
| `ConvergenceSummary { stop_reason, n_iter, initial_ll, final_ll, ll_gain, is_monotone }` | EM path quality |
| `RegimeSummary { means, variances, occupancy_shares, expected_durations }` | $1/(1-p_{jj})$ per regime |
| `MultiStartSummary { runs, best_ll, ll_spread }` | Cross-start comparison |
| `diagnose(em_result, obs) -> Result<FittedModelDiagnostics>` | Single-model diagnostics |
| `compare_runs(runs) -> MultiStartSummary` | Cross-start stability report |

### Diagnostic thresholds

| Constant | Value | Trigger |
|---|---|---|
| `NEAR_ZERO_VAR_THRESHOLD` | `1e-4` | `NearZeroVariance` warning |
| `NEARLY_UNUSED_REGIME_SHARE` | `0.01` | `NearlyUnusedRegime` warning |
| `SUSPICIOUS_PERSISTENCE_P` | `1 - 1e-6` | `SuspiciousPersistence` warning |
| `UNSTABLE_STARTS_LL_GAP` | `1.0` | `UnstableAcrossStarts` warning |
| `POSTERIOR_TOL` | `1e-8` | Sets `is_trustworthy = false` |

### Internal call graph

`diagnose` reruns the full inference stack under θ̂ as a validation pass:

```
diagnose(em_result, obs)
  filter(obs, θ̂)     → final FilterResult
  smooth(filter, θ̂)  → final SmootherResult
  pairwise(f, s, θ̂)  → final PairwiseResult
  check all invariants → FittedModelDiagnostics
```

### Stage usage

Called at the tail of **Stage 4**, after `fit_em` returns. Warnings are propagated into the run-level `warnings` array. `diagnostics.json` is written when `output.write_json = true`.

### Artifacts

`diagnostics.json`, `multi_start_summary.json` (when `em_n_starts > 1`).

### Tests

20 unit tests covering all six diagnostic axes.

### Status

**LIVE** — always called at Stage 4 conclusion for every training run.

---

## 9. `likelihood.rs` — Standalone log-likelihood evaluator

### Theoretical object

$$\log L(\theta) = \sum_t \log c_t$$

This is a **thin wrapper** over `filter`. It holds no statistical logic; its only purpose is to expose the scalar likelihood as a callable for optimizer code that does not need the full `FilterResult`.

### Key types

| Function | Role |
|---|---|
| `log_likelihood(obs, params) -> Result<f64>` | Calls `filter` once; returns `filter_result.log_likelihood` |
| `log_likelihood_contributions(obs, params) -> Result<Vec<f64>>` | Returns per-step `log_predictive` vector |

### Stage usage

**Not called in the primary EM training loop** (EM reads `filter_result.log_likelihood` directly). Called by:

- `direct_optimize` / `cmd_param_search` — objective for gradient-free optimisation.
- `compare_runs` in `diagnostics.rs` — LL comparison across multi-start results.

Note: the file carries `#![allow(dead_code)]` because the primary EM path does not import it directly; the optimizer path does.

### Artifacts

`loglikelihood_history.csv` (EM path, populated by `filter` inside `fit_em`, not by this module directly), `fit_summary.json`.

### Tests

10 unit tests.

### Status

**LIVE (OPTIMIZER)** — on the optimizer / multi-start comparison path; not on the primary EM training path.

---

## 10. `validation.rs` — Excluded

`src/model/validation.rs` contains scenario test fixtures (scenarios A–H). It is only compiled under `#[cfg(test)]` scenarios and has no live caller in the experiment pipeline. It is **excluded from this document**.

---

## End-to-end model data-flow summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2 — resolve_data (synthetic backend only)                             │
│   ModelParams (init θ)                                                      │
│     └── simulate(θ, T, rng)                                                 │
│           → SimulationResult.observations  → DataBundle.observations        │
│           → SimulationResult.changepoints()→ DataBundle.changepoint_truth   │
└─────────────────────────────────────────────────────────────────────────────┘
           │
           ▼ (observations flow to Stage 3: feature extraction)
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4 — train_or_load_model                                               │
│   fit_em(obs_features, init_params, config)                                 │
│     foreach iteration m:                                                    │
│       E-step:                                                               │
│         filter(obs, θ^m)                                                    │
│           Emission::log_density(y, j)  ← called K times per bar            │
│           → FilterResult { predicted, filtered, log_predictive, ll }        │
│         smooth(FilterResult, θ^m)                                           │
│           → SmootherResult { smoothed }                                     │
│         pairwise(FilterResult, SmootherResult, θ^m)                         │
│           → PairwiseResult { xi, expected_transitions }                     │
│         → EStepResult { smoothed, expected_transitions, log_likelihood }    │
│       M-step(obs, EStepResult, θ^m) → θ^{m+1}                              │
│       check |ΔLL| < tol                                                     │
│   → EmResult { params=θ̂, ll_history, n_iter, converged }                   │
│                                                                             │
│   diagnose(EmResult, obs_features)                                          │
│     filter(obs, θ̂) + smooth + pairwise (one final validation pass)          │
│   → FittedModelDiagnostics                                                  │
│                                                                             │
│   Artifacts: model_params.json, fit_summary.json, loglikelihood_history.csv │
│              diagnostics.json, [multi_start_summary.json]                   │
└─────────────────────────────────────────────────────────────────────────────┘
           │
           ▼ (θ̂ flows to Stage 5: online inference)
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5 — run_online  [src/online/mod.rs — not a model/ file]              │
│   OnlineFilterState::step(y_t, θ̂)                                          │
│     mirrors Hamilton recursion using same ModelParams + Emission            │
│   → OnlineStepResult { posterior, log_predictive }                          │
│                                                                             │
│   Artifacts: regime_posteriors.csv, score_trace.csv                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Thesis chapter mapping (model files)

| Thesis section | Primary code anchor | Equation |
|---|---|---|
| MSM — state-space parametrisation | [src/model/params.rs](../../src/model/params.rs) | θ = (π, A, μ, σ²) |
| MSM — observation law | [src/model/emission.rs](../../src/model/emission.rs) | $y_t \mid S_t=j \sim \mathcal{N}(\mu_j,\sigma_j^2)$ |
| Synthetic DGP | [src/model/simulate.rs](../../src/model/simulate.rs) | Markov chain + Gaussian draws |
| Inference — forward filter | [src/model/filter.rs](../../src/model/filter.rs) | Hamilton recursion, log-sum-exp |
| Inference — backward smoother | [src/model/smoother.rs](../../src/model/smoother.rs) | Kim smoother, `DENOM_FLOOR` |
| Inference — pairwise posteriors | [src/model/pairwise.rs](../../src/model/pairwise.rs) | $\xi_t(i,j)$, marginal consistency |
| Estimation — EM algorithm | [src/model/em.rs](../../src/model/em.rs) | Baum–Welch, M-step closed-form, floors |
| Estimation — trust & diagnostics | [src/model/diagnostics.rs](../../src/model/diagnostics.rs) | Expected duration $1/(1-p_{jj})$, `is_trustworthy` |
| Estimation — log-likelihood | [src/model/likelihood.rs](../../src/model/likelihood.rs) | $\log L(\theta) = \sum_t \log c_t$ |
