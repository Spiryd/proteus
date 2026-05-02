# Synthetic-to-Real Calibration

**Phase 17 — Markov Switching Model Project**

---

## 1. Motivation: Why Worlds A and B Must Be Connected

The project has two experimental worlds:

- **World A (synthetic):** known changepoints, controlled regime structure,
  rigorous benchmark truth.
- **World B (real):** financial relevance, practical validity, external realism.

If these worlds are disconnected, synthetic performance may not transfer to
real data. In particular, arbitrary synthetic parameter choices can create
regime dynamics unlike empirical market behavior (incorrect volatility scale,
unrealistic persistence, implausible change frequency).

Phase 17 defines a **calibration layer** that anchors synthetic scenarios to
real-market summary structure without attempting to replicate the full market
law.

---

## 2. Formal Definition of Calibration

Let:

- $X^{\text{real}} = (y_1^{\text{real}},\dots,y_T^{\text{real}})$ be the
  empirical observation process actually modeled by the detector,
- $X^{\text{syn}}(\vartheta)$ be a synthetic generator parameterized by
  $\vartheta$,
- $T_1,\dots,T_m$ be selected summary functionals.

Calibration seeks parameters $\vartheta$ such that

$$
T_\ell\big(X^{\text{syn}}(\vartheta)\big) \approx T_\ell\big(X^{\text{real}}\big),
\quad \ell=1,\dots,m.
$$

Define the calibration operator

$$
\mathcal{K}:
\big(T_1^{\text{real}},\dots,T_m^{\text{real}}\big)
\mapsto
\vartheta.
$$

The objective is **statistical anchoring**, not perfect distributional cloning.

---

## 3. Feature-Consistency Requirement

Calibration must be computed on the same observed process $y_t$ used by the
Markov Switching detector.

If Phase 16 defines

$$
y_t = \Phi(\text{market data up to } t),
$$

then calibration statistics must be extracted from

$$
\{y_t^{\text{real}}\}_{t=1}^T,
$$

not from unrelated raw-price summaries. This is a hard consistency condition:
calibrating on raw price levels while the model consumes returns or volatility
breaks methodological coherence.

---

## 4. Calibration Targets

A practical target set for this thesis is:

1. **Mean level:** $\bar y = \frac{1}{T}\sum_t y_t$.
2. **Dispersion scale:** $s_y^2 = \frac{1}{T}\sum_t (y_t-\bar y)^2$.
3. **Quantiles:** $q_{0.01}, q_{0.05}, q_{0.50}, q_{0.95}, q_{0.99}$.
4. **Tail-event frequency:** $\Pr(|y_t| > c)$ (or quantile-exceedance rates).
5. **Persistence proxies:** e.g., autocorrelation of $|y_t|$ or $y_t^2$.
6. **Change-frequency proxies:** sign-change rate or threshold-crossing rate.
7. **Episode duration proxies:** average high-vol vs low-vol run lengths.

These targets are sufficient to anchor both marginal behavior and regime-like
structure while remaining interpretable and reproducible.

---

## 5. Marginal vs Regime-Structural Calibration

### 5.1 Marginal Calibration

Matches first- and second-order distribution summaries:

- mean,
- variance,
- quantiles,
- tail frequencies.

This is the minimum viable calibration layer.

### 5.2 Regime-Structural Calibration

Matches transition/persistence behavior:

- expected regime durations,
- calm vs turbulent variance separation,
- change-frequency structure,
- clustering proxies.

Because the model itself is Markov-switching, regime-structural calibration is
especially important for scientific coherence.

---

## 6. Mapping Empirical Structure to MS Parameters

For Gaussian MS emissions,

$$
y_t \mid S_t=j \sim \mathcal{N}(\mu_j,\sigma_j^2),
$$

with transition matrix $P$.

Calibration maps empirical summaries to:

$$
\vartheta = \{K,\mu_1,\dots,\mu_K,\sigma_1^2,\dots,\sigma_K^2,P,\text{horizon},\text{optional jump settings}\}.
$$

### 6.1 Duration-to-persistence map

Given target expected duration $d_j$ for regime $j$:

$$
p_{jj} = 1 - \frac{1}{d_j}, \qquad d_j > 1.
$$

Remaining row mass $1-p_{jj}$ is allocated across off-diagonal entries
(symmetrically or by a chosen routing policy).

### 6.2 Variance anchoring

Use empirical quantiles or variance ratios to define low/high volatility
regimes. For instance, in two-state calibration:

- calm variance from lower-scale empirical region,
- turbulent variance from upper-scale empirical region.

### 6.3 Mean anchoring

Use one of:

- zero-centered means,
- symmetric around empirical mean,
- empirical baseline mean shared across regimes.

In many financial return settings, mean effects are smaller than variance
regime effects; variance and persistence are usually dominant.

---

## 7. Optional Shock Contamination

A robust extension introduces jumps:

- with probability $\lambda$, add shock noise with scale multiplier $\kappa$,
- otherwise emit from regime Gaussian law.

This supports heavier-tailed synthetic behavior while retaining the Gaussian
MS backbone as the core detector model.

---

## 8. Calibration Verification

After mapping and synthetic generation, compute synthetic summaries and compare
against empirical targets.

For selected metrics (mean, variance, quantiles, turbulence proxies), evaluate
absolute/relative discrepancy and declare pass/fail against configured
tolerances.

The output is a reproducible artifact:

- empirical summary,
- mapped synthetic parameters,
- synthetic summary,
- discrepancy report,
- tolerance verdict.

This closes the loop and prevents silent drift between intended and realized
synthetic structure.

---

## 9. Reproducibility and Leakage Policy

1. Calibration uses a **designated calibration partition** (default:
   training-only).
2. The mapping configuration and random seed are stored.
3. Synthetic scenarios are generated from stored calibrated parameters.
4. Verification outputs are exportable for thesis tables and audit trails.

No test-period information should be used when building calibration targets.

---

## 10. Scenario Configuration

Scenario parameters (`k`, `horizon`, mean/variance policy, target durations)
are supplied directly via `CalibrationMappingConfig` when calling
`run_calibration_workflow()`. Typical patterns:

- **Two-state calm/turbulent:** `k=2`, short + long target durations,
  `VariancePolicy::QuantileAnchored`.
- **Persistent states:** `k=2`, longer target durations,
  `VariancePolicy::RatioAroundEmpirical`.
- **Shock contamination:** as calm/turbulent with `JumpContamination` set
  in the mapping config (note: jump injection in the simulator is a future
  extension — the parameter is recorded in the calibration artifact).
- **Asset specific:** `k=3` with per-asset quantile targets.

---

## 11. Phase 17 Outcome

Phase 17 should end with a calibrated bridge:

$$
\text{Empirical feature stream}
\xrightarrow{\text{summaries}}
\text{Calibration map } \mathcal{K}
\xrightarrow{}
\text{Synthetic MS parameters}
\xrightarrow{\text{simulation + verification}}
\text{Calibrated scenario family}.
$$

This transforms synthetic experiments from arbitrary toys into controlled,
empirically anchored stress tests aligned with the real-data part of the
thesis.
