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

### 8.1 Default tolerances

The default `VerificationTolerance` checks `abs_acf1_abs_max ≤ 0.20` (i.e., the absolute difference in lag-1 autocorrelation between the empirical and synthetic sample does not exceed 0.20).

### 8.2 Shock-contaminated scenarios

For `shock_contaminated` scenarios, a **wider ACF1 tolerance of 0.40** is applied automatically by the `calibrate` CLI command. This is intentional: jump contamination introduces heavier tails and larger sample-to-sample autocorrelation variability. The standard 0.20 tolerance is too tight for two independent shock-contaminated synthetic draws and would cause the verification to fail spuriously. The wider tolerance acknowledges this structural property without changing the calibration mapping itself.

### 8.3 Scale-consistency contract (one-sided z-scoring)

When synthetic data is used to train a model that will be evaluated on real
data, the two streams must live on the same numeric scale.  The codebase
enforces a **one-sided policy** (B′1):

- The `FittedScaler` (e.g., z-score from `FeatureStream::fit_transform`) is
  fit **only** on the real training partition.
- The synthetic stream is then passed through that scaler via
  `scaler.transform(&synthetic_observations)` — it is **not** refit.
- After scaling, both the real-train and the calibrated-synthetic streams
  share the same z-axis; the model trained on synthetic-scaled data therefore
  applies meaningfully to real-scaled inputs at test time.

To detect violations of this contract, `scale_consistency_check(&emp_summary,
&syn_summary, tol)` in `src/calibration/verify.rs` compares the standard
deviations of the empirical and synthetic streams (post-scaler) and returns a
warning when `|syn.std - emp.std| / emp.std > tol`.  The default tolerance is
`DEFAULT_SCALE_TOLERANCE = 0.10` (10%).  Results are surfaced both in the
calibration report and in the `synthetic_training_provenance.json` artifact
emitted by sim-to-real runs (see §13).

---

## 13. Calibration strategies

The `CalibrationStrategy` enum on `CalibrationMappingConfig` selects how
empirical structure is mapped to synthetic generator parameters.

### 13.1 `CalibrationStrategy::Summary` (default)

Computes the summary functionals listed in §4 (mean, variance, quantiles,
turbulence proxies) and routes them through the policy-driven mapping
described in §6 (`MeanPolicy`, `VariancePolicy`, `DurationPolicy`).  This is
the historical pathway and remains the default for backwards compatibility.

### 13.2 `CalibrationStrategy::QuickEm { max_iter, tol }`

Runs a short Baum–Welch EM on the real training partition and uses the
resulting `ModelParams` *directly* as the synthetic generator parameters
($\pi$, transition matrix $P$, regime means and variances).  This is the
strongest available form of calibration: the synthetic stream is literally a
sample from the same `ModelParams` that EM extracted from the real data.

```rust
CalibrationMappingConfig {
    k: 2,
    horizon: 2000,
    strategy: CalibrationStrategy::QuickEm { max_iter: 100, tol: 1e-6 },
    ..CalibrationMappingConfig::default()
}
```

Quick-EM is the recommended strategy for `SimToRealBackend` runs because it
eliminates the duration/quantile heuristics in `Summary` mode and produces
parameters that are guaranteed to be consistent with a Gaussian MS likelihood
on the real data.  Convergence is bounded by `max_iter`; non-convergence is
recorded but does not abort the run.

---

## 14. Sim-to-real artifacts

When `ExperimentMode::SimToReal` is used (see `docs/experiment_runner.md`),
two additional artifacts are written to the run directory:

- `synthetic_training_provenance.json` — the calibration profile (empirical
  summary), strategy, calibrated `ModelParams`, scale-consistency check
  result, number of synthetic and real observations.  This replaces the usual
  `split_summary.json` artifact when in SimToReal mode.
- `sim_to_real_summary.json` — collates `train_source`, `test_source`,
  `model_params_synthetic_trained`, `eval_real_metrics` (Route A + Route B)
  for the figure pipeline.

Use `compare-sim-vs-real --id <simreal_id>` to additionally produce
`sim_vs_real_comparison.json` with side-by-side metrics from the
synthetic-trained run and an automatically-derived real-trained variant of
the same experiment.

---

## 15. Variance policy variants (C′1, C′2)

`VariancePolicy` extends the three legacy policies with one
empirically-conditioned variant and a global ratio guard:

- **`MagnitudeConditioned`** (C′1) — splits the raw partition observations
  on `|y|` at the sample median and uses the within-half sample variances as
  the low / high regime variances. Requires `EmpiricalCalibrationProfile::
  observations` to be non-empty; otherwise it falls back to
  `QuantileAnchored` and records the fallback in `mapping_notes`.
- **`CalibrationMappingConfig::min_high_low_ratio`** (C′2, default `1.0`) —
  enforces a minimum `sigma_high / sigma_low` ratio for `QuantileAnchored`
  and `MagnitudeConditioned`. When the empirical ratio falls below the
  floor, the variances are rescaled log-symmetrically around the geometric
  mean to meet the guard; the action is recorded as a `C′2` mapping note.
  Use values around `2.0`–`3.0` when calibrating against assets whose
  empirical q05/q95 spread collapses (e.g. heavily-trimmed series).

The guard composes with the post-mapping flat-spread safety net so a
near-degenerate empirical distribution still yields a well-separated
synthetic generator.

### 15.1 π-policy for Quick-EM (`PiPolicy`)

`CalibrationMappingConfig::pi_policy` selects how the initial distribution
`π` of the calibrated generator is chosen after Quick-EM converges:

- **`PiPolicy::Fitted`** — keep the EM-fitted `π̂` as the maximum-likelihood
  starting state of the training partition.
- **`PiPolicy::Stationary`** (default) — replace `π̂` with the stationary
  distribution `π★ = π★ · P̂` of the fitted transition matrix via
  power iteration (`mapping::stationary_pi`).  The swap is recorded in
  `mapping_notes` with the pre- and post-replacement vectors.

`Stationary` is the principled choice when no prior information about the
test-stream starting regime is available, and it avoids the
degenerate-`π` collapse (`π̂ ≈ (1, 0)`) that Quick-EM produces on real
series whose lag-1 autocorrelation is structurally outside the model class
(e.g. raw daily log-returns on equity indices, where
`ρ̂₁(y) ≈ -0.15`).  Switching to a feature whose autocorrelation is
inside the model class (e.g. AbsReturn with `ρ̂₁ ≈ +0.37`) together with
`PiPolicy::Stationary` is what unlocks the canonical sim-to-real entries;
see Chapter 5 §5.8.4–5.8.5 of the thesis and
`verification/sim_to_real_recovery_2026_05_15/sweep_summary.md`.

### 15.2 Non-finite-observation guard

Quick-EM and `summarize_observation_values` both filter non-finite values
from the input before sorting / EM.  This is required because:

- modern Rust slice-sort validates total order and panics on NaN;
- some real series (notably WTI 2020-04-20 with its negative spot print)
  produce NaN under `LogReturn` / `AbsReturn`.

The drop count is recorded as a `mapping_notes` entry on the Quick-EM
path; `summarize_observation_values` silently filters.

## 16. Policy-aware verifier mask (C′4)

`VerificationTargetMask` (one bool per checked field) drives both the global
`within_tolerance` verdict and the per-field record bundle written to
`CalibrationReportView::field_results`.  Use
`VerificationTargetMask::for_policy(&CalibrationMappingConfig)` to derive
the mask from the active calibration strategy:

- `MeanPolicy::ZeroCentered` ⇒ `mean` masked off (the synthetic mean is
  fixed by construction so comparing it to the empirical drift is a
  tautology that misfires when the data has nonzero baseline return).
- `CalibrationStrategy::QuickEm { .. }` ⇒ `abs_acf1` and `sign_change_rate`
  masked off (those dependence summaries are side-effects of the fitted
  model rather than calibration targets).

Masked-off fields still produce a `FieldVerification { checked: false, .. }`
record so the JSON consumer can distinguish "not checked" from
"checked and passed".  `CalibrationReportView` additionally serialises
`mapping_notes` (E4) and the bar-aware `frequency` tag (E3, e.g.
`"intraday_15min"`).

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

---

## 12. Calibration-to-Registry Workflow

Calibration is a **design-time activity**.  The output of `cargo run -- calibrate --id <id>`
is a `calibration_report.json` file containing fitted `ModelParams`.  Those
parameters are **not** loaded at runtime — they are read offline by the
developer and manually transcribed into `src/experiments/registry.rs` as
static Rust constructors.

The full loop is:

```
1. cargo run -- calibrate --id real_spy_daily
       ↓
   data/calibration/real_spy_daily/calibration_report.json
       ↓
2. Developer reads: pi, transition, means, variances
       ↓
3. Update src/experiments/registry.rs:
       DataConfig::Synthetic { scenario_id: "scenario_calibrated", … }
       (ModelParams pulled from registry's build_synthetic_params helper)
       ↓
4. cargo run -- e2e   # runs all registered synthetic experiments
```

### Why this is intentional

The manual step keeps the registry **type-safe and IDE-navigable**.  An
automated JSON-loader approach would lose compile-time validation of
parameter shapes.  The cost is that `registry.rs` must be updated when
calibration results change.

### What to document in the thesis

The thesis should make this loop explicit:

- Present $\mathcal{K}$ as a mapping from empirical statistics to synthetic
  parameters (§4.15).
- State which assets were calibrated and what the resulting $(K, \mu, \sigma^2,
  P)$ values are (a table in §4.15 or an appendix).
- Explain that the calibrated parameters appear in `registry.rs` under the
  `"scenario_calibrated"` scenario ID and are used by all 7 synthetic
  registered experiments.
- Note that re-calibration (e.g., on a new date range) requires updating
  `registry.rs` and re-running `e2e`.

### Notes for $K \geq 3$

For two-state calibration, `target_durations` can be inferred from the
empirical episode distribution.  For $K \geq 3$, the durations must be
provided explicitly in `CalibrationMappingConfig::target_durations` because
the unsupervised episode labelling is ambiguous beyond two states.
