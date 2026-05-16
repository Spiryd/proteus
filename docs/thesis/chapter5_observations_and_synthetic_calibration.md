# Chapter 5 — Synthetic-to-Real Calibration

**Relation to previous chapters.** Chapters 3 and 4 developed the theory of
the Gaussian Markov Switching Model and its online changepoint detector. Both
chapters worked with an abstract observation sequence $y_{1:T}$ and assumed
known model parameters $\theta$. This chapter addresses the central
methodological question of the thesis: *how do we choose $\theta$ so that the
synthetic data a detector trains on resembles the real market data it must
detect on?*

The answer is the **calibration operator** $\mathcal{K}$, a deterministic map
from empirical summary statistics of the real training stream to the
parameters of a synthetic generator. The operator is the novel contribution
of this thesis. Its definition (§5.1), two concrete implementations (§5.2,
§5.4), the variance policies it supports (§5.3), the initial-distribution
pathology it must handle (§5.5), and the verification protocol that audits its
output (§5.6) form the six sections of this chapter.

---

## 5.1 The calibration problem

A purely hand-picked $\vartheta$ produces synthetic data that may have nothing
to do with markets. The calibration operator addresses this by choosing
$\vartheta$ from empirical summary functionals of the real training stream
$y^{\text{real,train}}_{1:T}$:

$$
\mathcal{K}:\; \big( T_1(y^{\text{real,train}}),\, \ldots,\, T_m(y^{\text{real,train}}) \big)
\;\longmapsto\; \vartheta = (\pi,\, P,\, \mu_{1:K},\, \sigma^2_{1:K}).
$$

The objective is **statistical anchoring, not distributional cloning**. A
perfect distributional match would defeat the purpose of having an
interpretable $K$-regime model whose hidden states have unambiguous financial
meaning. The operator must therefore be expressive enough to reproduce the
dominant empirical features that drive detector behaviour — marginal scale,
tail asymmetry, regime persistence — while remaining deliberately silent on
higher-order structure (e.g. long memory, heavy tails) that the Gaussian MSM
cannot capture.

Two requirements are non-negotiable:

- **Feature consistency.** Each $T_i$ is evaluated on the same feature stream
  $\phi(P_{1:T})$ that the detector consumes, not on raw prices.
- **Partition restriction.** Each $T_i$ sees only $\mathcal{T}_{\text{train}}$,
  so that calibration does not leak information from validation or test
  partitions.

Two strategies satisfy these requirements. The transparent summary-based map
$\mathcal{K}_{\text{sum}}$ (§5.2) computes $\vartheta$ from a small set of
closed-form rules, making every parameter traceable to a named empirical
quantity. The EM-based map $\mathcal{K}_{\text{EM}}$ (§5.4) runs Baum–Welch
directly on the training partition and is required when the synthetic stream
must be a maximum-likelihood match to the real data — as in the sim-to-real
experiments of Chapter 6.

---

## 5.2 Summary-based calibration $\mathcal{K}_{\text{sum}}$

### 5.2.1 The empirical summary

$\mathcal{K}_{\text{sum}}$ begins by computing a fixed empirical summary
$\mathcal{E}(y^{\text{real,train}})$ that collects three blocks of statistics,
each calibrating one block of generator parameters:

| Block | Symbols | Purpose |
|---|---|---|
| Marginal anchors | $\bar y,\, s_y^2,\, q_{0.05}, q_{0.50}, q_{0.95}$ | calibrate $\mu_j,\, \sigma_j^2$ |
| Dependence proxies | $\hat\rho_1(y),\, \hat\rho_1(\lvert y\rvert),\, \mathrm{sgn\text{-}rate}$ | validate the existence of regime structure |
| Episode durations | $\hat d_{\text{low}},\, \hat d_{\text{high}}$ | calibrate $P_{ii}$ |

The episode-duration estimator thresholds $\lvert y_k\rvert$ at $q_{0.95}$ and
returns the mean run lengths on each side. This estimator is *deliberately
coarse*: a finer estimator (e.g. fitting an HMM) would entangle the
calibration with the very inference procedure under evaluation.

### 5.2.2 Mean policy

The default policy `EmpiricalBaseline` sets $\mu_j = \bar y$ for all $j$.
Regime separation is carried entirely by variance, matching the dominant
pattern in financial returns where conditional means are small relative to
conditional standard deviations. Alternative policies shift the means by
$\pm \tfrac{1}{4} s_y$ around either $0$ or $\bar y$ for mean-shift
sensitivity studies; the $1/4$ factor is deliberately small so the detector
cannot succeed on obvious mean differences alone.

A second alternative, `ZeroCentered`, fixes $\mu_j = 0$ for all $j$. It is
used when the feature is a signed transformation (e.g. log-return) whose
unconditional mean is approximately zero and whose regime structure is carried
entirely in the variance.

### 5.2.3 Duration-to-persistence map

If $S_k = j$ self-transitions with probability $p_{jj}$ (Bernoulli), dwell
time in regime $j$ is geometrically distributed with mean $1/(1 - p_{jj})$.
Inverting:

$$
\boxed{\;p_{jj} = 1 - \frac{1}{\hat d_j}.\;}
$$

Off-diagonal mass $1 - p_{jj}$ is split uniformly across the remaining $K-1$
regimes — the most uninformative choice given only duration targets. Boundary
conditions: $\hat d_j > 1$ is required (a regime that cannot persist for one
step on average is incompatible with Markov structure), and $p_{jj}$ is
clamped to $[10^{-6},\, 1 - 10^{-6}]$ to keep EM and the forward filter
numerically interior. The initial distribution defaults to $\pi_j = 1/K$.

### 5.2.4 Complete operator for $K = 2$

Composing the mean policy (§5.2.2), the default variance policy (§5.3.1), and
the duration map (§5.2.3), $\mathcal{K}_{\text{sum}}$ maps the summary tuple
$(\bar y,\, q_{0.05}, q_{0.50}, q_{0.95},\, \hat d_{\text{low}}, \hat d_{\text{high}})$
to

$$
\begin{aligned}
\mu &= (\bar y,\ \bar y),\\
\sigma^2 &= \!\left(\,\big(\tfrac{|q_{0.50}-q_{0.05}|}{1.6449}\big)^2,\ \big(\tfrac{|q_{0.95}-q_{0.50}|}{1.6449}\big)^2\right)\!\text{ sorted}\!\uparrow,\\
P &= \begin{pmatrix} 1 - \tfrac{1}{\hat d_{\text{low}}} & \tfrac{1}{\hat d_{\text{low}}} \\[2pt] \tfrac{1}{\hat d_{\text{high}}} & 1 - \tfrac{1}{\hat d_{\text{high}}} \end{pmatrix},\\
\pi &= (1/2,\ 1/2).
\end{aligned}
$$

Every quantity on the right is an explicit empirical summary; every quantity
on the left is a generator parameter. The map is deterministic and fully
auditable.

---

## 5.3 Variance policies

The variance block of $\mathcal{K}_{\text{sum}}$ is the most sensitive to
feature choice and asset class. Three policies are implemented; all produce
a set of $K$ variances sorted ascending so regime $0$ is canonically the
calm regime.

### 5.3.1 `QuantileAnchored` (default)

The low and high regime standard deviations are read off the empirical
quantile spread by inverting the standard Gaussian CDF at the 5th and 95th
percentiles:

$$
\sigma_{\text{low}} = \frac{\lvert q_{0.50} - q_{0.05} \rvert}{\Phi^{-1}(0.95)},\qquad
\sigma_{\text{high}} = \frac{\lvert q_{0.95} - q_{0.50} \rvert}{\Phi^{-1}(0.95)},
$$

where $\Phi^{-1}(0.95) \approx 1.6449$. For $K > 2$, regime standard
deviations are linearly interpolated between $\sigma_{\text{low}}$ and
$\sigma_{\text{high}}$. The policy is appropriate when the feature
distribution is approximately symmetric and the quantile spread is a reliable
proxy for conditional scale.

### 5.3.2 `MagnitudeConditioned` (C′1)

Rather than relying on marginal quantiles, `MagnitudeConditioned` *directly
partitions the training observations* by their magnitude: observations with
$\lvert y_k \rvert \le \mathrm{median}(\lvert y\rvert)$ form the low group;
all others form the high group. The regime variances are the sample variances
of the two groups:

$$
\sigma_j^2 = \frac{1}{n_j}\sum_{k \in \mathcal{G}_j} (y_k - \bar y_j)^2,
\qquad j \in \{\text{low},\, \text{high}\},
$$

where $\mathcal{G}_{\text{low}} = \{k : \lvert y_k\rvert \le \mathrm{med}\}$
and $\mathcal{G}_{\text{high}}$ is its complement. This policy is preferred
for asymmetric or heavy-tailed features (e.g. AbsReturn, SquaredReturn) where
quantile-based inversion is unreliable. If the training partition is empty,
the policy falls back to `QuantileAnchored` with a recorded note in
`mapping_notes`.

For $K > 2$, regime standard deviations are linearly interpolated between
$\sqrt{\sigma^2_{\text{low}}}$ and $\sqrt{\sigma^2_{\text{high}}}$ using the
same spacing as `QuantileAnchored`.

### 5.3.3 The ratio guard C′2

After variances are sorted, both `QuantileAnchored` and `MagnitudeConditioned`
enforce a minimum $\sigma_{\text{high}} / \sigma_{\text{low}}$ ratio to avoid
degenerate identifiability — a situation where both regimes have nearly
identical variance and the detector cannot distinguish them. If the raw ratio
falls below the configured `min_high_low_ratio` $> 1$, the variances are
rescaled log-symmetrically around their geometric mean until the ratio
constraint is satisfied:

$$
\sigma_j \;\leftarrow\; \sigma_j \cdot \alpha^{2\,\text{frac}_j - 1},
\qquad
\alpha = \sqrt{\frac{\texttt{min\_high\_low\_ratio}}{\sigma_{\text{high}}/\sigma_{\text{low}}}},
$$

where $\text{frac}_j \in [0, 1]$ is the regime's position in the sorted
order. The geometric mean of the variances is preserved by construction,
so the overall empirical scale is unchanged.

---

## 5.4 Quick-EM calibration $\mathcal{K}_{\text{EM}}$

$\mathcal{K}_{\text{sum}}$ is interpretable but heuristic: there is no
guarantee that $\vartheta = \mathcal{K}_{\text{sum}}(\mathcal{E})$ is a
maximum-likelihood description of the training partition. For **sim-to-real**
experiments — where a detector is *trained* on synthetic data and *tested*
on real data — a stronger calibration is needed: the synthetic stream must be
a sample from the same generative law that EM would extract from the real
training data, otherwise the train and test distributions diverge before the
detector has processed a single observation.

The Quick-EM strategy realises this by running Baum–Welch directly on
$y^{\text{real,train}}_{1:T}$:

**Algorithm 5.1 ($\mathcal{K}_{\text{EM}}$).**

1. Compute $\vartheta_0$ via $\mathcal{K}_{\text{sum}}$ (used only as an EM
   warm start, not as the final output).
2. Run `fit_em`$(y^{\text{real,train}},\; \vartheta_0,\; \texttt{max\_iter},\; \texttt{tol})$
   to convergence or iteration limit.
3. Return $\hat\vartheta_{\text{EM}}$ as the generator parameters.

Symbolically,

$$
\mathcal{K}_{\text{EM}}:\; y^{\text{real,train}}_{1:T_{\text{train}}}
\;\xrightarrow{\;\text{Baum–Welch}\;}\;
\hat\vartheta_{\text{EM}} = (\hat\pi,\, \hat P,\, \hat\mu,\, \hat\sigma^2).
$$

The synthetic stream $y^{\text{syn}} \sim \mathrm{MSM}(\hat\vartheta_{\text{EM}})$
then samples from the law that EM would have used to score the real training
data — the natural train-time analogue of the test-time likelihood.

In the SPY daily verification run, $\hat\vartheta_{\text{EM}}$ was obtained
in $31$ Baum–Welch iterations to $\log\mathcal{L} = 4581.29$, with fitted
expected durations $\hat d = (46.76,\, 24.23)$.

---

## 5.5 Initial distribution policy

### 5.5.1 The degenerate-$\pi$ pathology

The Quick-EM operator $\mathcal{K}_{\text{EM}}$ returns four estimators
$(\hat\pi, \hat P, \hat\mu, \hat\Sigma)$. The initial distribution $\hat\pi$
is the *maximum-likelihood* assignment of probability mass to the starting
regime of the *training partition*, which can collapse to a near-degenerate
point mass on series whose lag-1 autocorrelation lies outside the expressive
range of the model class. A concrete instance: z-scored SPY daily log-returns
have $\hat\rho_1(y) \approx -0.15$, which a Gaussian MSM with $\mu_0 = \mu_1$
cannot reproduce. EM responds by placing $\hat\pi \approx (1,\, 9.4
\cdot 10^{-12})$ — "the series started in the volatile regime and immediately
mean-reverted." For *detection*, $\pi$ is then used twice:

1. as the initial state distribution of the synthetic data *generator*, and
2. as the filter prior at time $t = 1$ of the real test stream.

A near-degenerate $\hat\pi$ forces the filter to start with near-certainty in
one regime and never cross the detector threshold, producing zero alarms over
the entire test horizon.

### 5.5.2 `PiPolicy::Stationary` versus `PiPolicy::Fitted`

The principled resolution is to replace $\hat\pi$ with the stationary
distribution $\pi^\star$ of the fitted chain, defined by the fixed-point
equation $\pi^\star = \pi^\star \hat P$ (computed by power iteration to
$10^{-10}$ tolerance). The pipeline supports this via a
`PiPolicy::{Stationary, Fitted}` knob; the default is `Stationary`.

For the SPY daily example, this replaces $\hat\pi = (1,\, 9.4 \cdot 10^{-12})$
with $\pi^\star \approx (0.286,\, 0.714)$, which is the long-run frequency the
chain visits each regime. The filter initialised with $\pi^\star$ is then
agnostic about which regime the series is in at $t = 1$ — the correct
epistemic state when no prior information is available.

`PiPolicy::Stationary` is necessary but not sufficient: it cannot rescue a
structurally misspecified feature choice. On SPY daily *log-returns*, even
with $\pi^\star$ the detector fires zero alarms because the feature
autocorrelation is outside the model class. On SPY daily *absolute returns*,
whose $\hat\rho_1 \approx 0.37$ is inside the model class,
`PiPolicy::Stationary` recovers a working sim-to-real detector. The
combination of feature selection and $\pi$-policy is therefore the correct
unit of analysis, not either factor in isolation.

---

## 5.6 Calibration verification

### 5.6.1 Field-wise tolerance bounds

A calibrated $\vartheta$ is useful only if the synthetic data it produces
actually resembles the empirical target on the summary functionals used to
construct it. The verifier $\mathcal{V}(\vartheta;\, \mathcal{E})$ samples
$y^{\text{syn}}_{1:T} \sim \mathrm{MSM}(\vartheta)$, recomputes
$\mathcal{E}^{\text{syn}}$, and returns the per-field residual

$$
\Delta_i = T_i(y^{\text{syn}}) - T_i(y^{\text{real}})
$$

together with a per-field tolerance check $\lvert \Delta_i \rvert \le \tau_i$
(or, for variance, $\lvert \Delta_i \rvert / T_i(y^{\text{real}}) \le \tau_i$).
Default tolerances are deliberately loose:

| Field | Default $\tau_i$ |
|---|---|
| Mean | $0.25$ (absolute) |
| Variance | $0.50$ (relative) |
| Quantiles $q_{0.05}, q_{0.50}, q_{0.95}$ | $0.30$ (absolute) |
| $\lvert\hat\rho_1\rvert$ | $0.20$ (absolute) |
| Sign-change rate | $0.20$ (absolute) |

The goal is *anchoring*, not cloning. Tight tolerances would require a more
expressive generator class, defeating the interpretability objective.

### 5.6.2 Masking

A target mask selects which fields participate in the global pass/fail verdict
while preserving per-field records. The mask is derived from the active policy:

- `MeanPolicy::ZeroCentered` disables the mean check (the synthetic mean is
  fixed at zero by construction; a deviation is expected and harmless).
- `CalibrationStrategy::QuickEm` disables the dependence proxies
  $\hat\rho_1(\lvert y\rvert)$ and the sign-change rate (which emerge as
  side-effects of EM rather than as explicit calibration targets).

Masked fields are recorded as `checked: false` in the verification output,
distinguishing a *skipped* check from a *passed* one for downstream consumers.

### 5.6.3 Scale-consistency check

In the sim-to-real pipeline, the synthetic stream is *trained on* and the real
stream is *tested on* using a single scaler $\mathcal{S}$ fitted on the real
training partition. For the trained model to transfer, both streams must share
a common numeric scale after applying $\mathcal{S}$. The pipeline enforces:

$$
\text{rel\_error} = \frac{\lvert \hat\sigma(y^{\text{syn,scaled}}) - \hat\sigma(y^{\text{real,scaled}}) \rvert}
{\hat\sigma(y^{\text{real,scaled}})} \;\le\; 0.10,
$$

recorded in `synthetic_training_provenance.json`. The verification run on
SPY daily produced

$$
\hat\sigma_{\text{real}} = 0.01302,\quad
\hat\sigma_{\text{syn}} = 0.01234,\quad
\text{rel\_error} = 0.0524,
$$

passing the 10 % bound.

### 5.6.4 The calibration verdict

When $\mathcal{V}$ reports `within_tolerance: false` — as it does for SPY
daily under $\mathcal{K}_{\text{sum}}$ because the empirical
$\hat\rho_1(y) \approx -0.15$ cannot be reproduced by any $K = 2$ Gaussian
MSM with $\mu_0 = \mu_1$ — the response is *not* to abort. The chosen
scenario is a deliberately simpler analogue of reality, not a faithful clone.
The synthetic ground truth remains valid for detector benchmarking; what is
recorded is a precise *limitation*: the specific fields that failed tolerance
and the direction of the residuals. The methodological cost of using an
interpretable generator is therefore auditable rather than hidden.

Concretely, a `within_tolerance: false` verdict on the autocorrelation field
is the signal to re-examine the feature choice. A feature for which
$\mathcal{V}$ passes all active tolerance checks — such as AbsReturn on
GOLD daily — is one for which the sim-to-real transfer argument can be made
without caveat.

---

## 5.7 Theory–code correspondence

The table below maps each section of this chapter to the Rust source files
and the specific types or functions that implement the described concept.
All paths are relative to the workspace root.

| Section | Concept | File | Key types / functions |
|---|---|---|---|
| §5.1 | Calibration operator $\mathcal{K}$ (top-level entry point) | `src/calibration/mapping.rs` | `calibrate_to_synthetic()`, `CalibrationMappingConfig`, `CalibrationStrategy` |
| §5.1 | End-to-end calibration workflow (simulate → verify) | `src/calibration/report.rs` | `run_calibration_workflow()`, `CalibrationReport`, `CalibrationReportView` |
| §5.1 | Module root / re-exports | `src/calibration/mod.rs` | — |
| §5.2.1 | Empirical summary functionals $T_1 \ldots T_m$ | `src/calibration/summary.rs` | `EmpiricalSummary` (mean, variance, quantiles, acf1, abs\_acf1, sign\_change\_rate, episode durations), `summarize_observation_values()` |
| §5.2.1 | Calibration profile bundle (summary + observations + tag) | `src/calibration/summary.rs` | `EmpiricalCalibrationProfile`, `CalibrationDatasetTag`, `CalibrationPartition`, `SummaryTargetSet` |
| §5.2.2 | Mean policy ($\bar y$, zero-centred, shifted) | `src/calibration/mapping.rs` | `MeanPolicy::{EmpiricalBaseline, ZeroCentered, ShiftedAroundMean, ShiftedAroundZero}` |
| §5.2.3 | Duration-to-persistence map $p_{jj} = 1 - 1/\hat d_j$ | `src/calibration/mapping.rs` | `calibrate_via_summary()` (transition-matrix block) |
| §5.2.4 | Complete $K = 2$ summary operator | `src/calibration/mapping.rs` | `calibrate_via_summary()` (full body), `CalibratedSyntheticParams` |
| §5.3.1 | `QuantileAnchored` variance policy | `src/calibration/mapping.rs` | `VariancePolicy::QuantileAnchored` |
| §5.3.2 | `MagnitudeConditioned` variance policy (C′1) | `src/calibration/mapping.rs` | `VariancePolicy::MagnitudeConditioned` |
| §5.3.3 | Ratio guard (C′2) — log-symmetric rescaling | `src/calibration/mapping.rs` | ratio-guard block inside `calibrate_via_summary()` |
| §5.4 | Quick-EM strategy $\mathcal{K}_{\text{EM}}$ (Baum–Welch on real train) | `src/calibration/mapping.rs` | `CalibrationStrategy::QuickEm { max_iter, tol }`, `calibrate_to_synthetic()` QuickEm branch |
| §5.4 | Sim-to-real experiment backend (full pipeline) | `src/experiments/sim_to_real_backend.rs` | `SimToRealBackend` |
| §5.5.1 | Degenerate-$\pi$ pathology | `src/calibration/mapping.rs` | `PiPolicy`, documented in `CalibrationMappingConfig::pi_policy` |
| §5.5.2 | `PiPolicy::Stationary` / `Fitted` | `src/calibration/mapping.rs` | `PiPolicy::{Stationary, Fitted}`, power-iteration block in `calibrate_to_synthetic()` |
| §5.6.1 | Field-wise tolerance bounds | `src/calibration/verify.rs` | `VerificationTolerance`, `FieldVerification`, `FieldResults` |
| §5.6.2 | Verification target mask | `src/calibration/verify.rs` | `VerificationTargetMask`, `VerificationTargetMask::for_policy()` |
| §5.6.3 | Scale-consistency check (rel\_error ≤ 10 %) | `src/calibration/verify.rs` | `ScaleConsistencyCheck`, `scale_consistency_check()`, `DEFAULT_SCALE_TOLERANCE` |
| §5.6.4 | Calibration verdict (`within_tolerance`) | `src/calibration/verify.rs` | `CalibrationVerification`, `verify_calibration_masked()` |
| §5.6 | Serialised verification output | `src/calibration/report.rs` | `CalibrationReportView` fields: `verification_passed`, `mapping_notes`, `field_results`, `verification_mask` |
