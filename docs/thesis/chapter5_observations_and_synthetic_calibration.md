# Chapter 5 — Observations: From Markets to Synthetic Streams

**Relation to previous chapters.** Chapter 2 fixed the Gaussian Markov-switching
model (MSM), Chapter 3 estimated its parameters by EM, and Chapter 4 used the
resulting filter posterior to build a real-time detector. All three chapters
worked with an abstract observation sequence $y_{1:T}$. This chapter defines
how that sequence is constructed, in the two settings the rest of the thesis
uses:

- the **empirical pipeline** that maps raw market prices to a model-ready
  stream $y^{\text{real}}_{1:T}$ (§5.2–§5.5);
- the **synthetic pipeline** that samples $(y^{\text{syn}}_{1:T},\, S_{1:T})$
  from a calibrated MSM, providing the only setting in which changepoint
  detection has access to ground truth (§5.6–§5.8).

The two pipelines are not parallel decorations: they share the same
observation contract (§5.1) and are bridged by a single calibration operator
$\mathcal{K}$ (§5.7) that anchors the synthetic generator to empirical
summaries computed on the *same* $y^{\text{real}}_{1:T}$ the detector
consumes.

---

## 5.1 The observation contract

Before specifying any pipeline, we state the properties any candidate stream
$y_{1:T}$ must satisfy for the model and detector to be well-defined.

**Definition 5.1 (Admissible observation stream).** A sequence
$(t_k, y_k)_{k=1}^T \in (\mathcal{T} \times \mathbb{R})^T$ is *admissible* if

1. **(C1) Strict order.** $t_1 < t_2 < \cdots < t_T$.
2. **(C2) Causality.** Each $y_k$ is a measurable function of the price
   filtration $\mathcal{F}_{t_k} = \sigma(P_{s} : s \le t_k)$.
3. **(C3) Leakage safety.** Any preprocessing step whose parameters depend on
   the data must be fitted on a *prefix* $y_{1:n_{\text{train}}}$ and applied
   identically to $y_{n_{\text{train}}+1:T}$.
4. **(C4) Stationarity-friendly representation.** $y_k$ is a transformation of
   prices designed so that the within-regime law $y_k \mid S_k = j$ is
   approximately stationary; in particular, raw prices are excluded.
5. **(C5) Regime-conditional Gaussianity.** Conditional on the latent state
   $S_k$, $y_k \sim \mathcal{N}(\mu_{S_k}, \sigma_{S_k}^2)$ holds with
   tolerable misspecification.

C1–C3 are *enforced* by the pipeline; C4 is *promoted* by the choice of
feature; C5 is the model assumption from Chapter 2 that the pipeline is
designed to keep credible.

The remainder of this chapter is organised so that every section discharges
one or more of these conditions. The synthetic pipeline (§5.6) is the unique
setting in which C5 holds *by construction*; the empirical pipeline (§5.2–5.5)
is the unique setting in which the data are financially real but C5 is an
empirical hypothesis.

---

## 5.2 Empirical pipeline: raw prices to a clean series

Daily and intraday equity ETFs (SPY, QQQ) and twelve commodity series are
served by a single REST vendor and persisted to a local cache; all
experiments read exclusively from the cache, so training never issues a
network call. The exact endpoint inventory and cache schema are recorded in
[docs/alphavantage_client.md](../alphavantage_client.md) and
[docs/duckdb_cache.md](../duckdb_cache.md); for the chapter what matters is
that every experiment starts from a deterministic tabular series
$(t_k, P_k)_{k=1}^N$.

### 5.2.1 Two data modes

The pipeline distinguishes daily and intraday data because they differ in two
substantive ways:

| Behaviour | Daily | Intraday (bar of $\Delta$ minutes) |
|---|---|---|
| Timestamp granularity | calendar date at 00:00 | bar-open in US Eastern Time |
| Calendar gaps | expected (weekends, holidays) | flagged when $\Delta t > 3\,\Delta$ |
| Session filter | not applied | $\mathrm{RTH}(t) \iff 09{:}30 \le \mathrm{time}(t) < 16{:}00$ ET |

Mode selection is encoded once per registered experiment; daily and intraday
streams cannot be silently mixed.

### 5.2.2 Cleaning algorithm

Given a vendor response $(t_k, P_k)_{k=1}^M$ in arbitrary order, the cleaning
stage produces a `CleanSeries` by:

**Algorithm 5.1 (Series cleaning).**

1. Record `had_unsorted_input` $\mathrel{:=} \mathbb{1}[\exists k : t_{k+1} < t_k]$.
2. Sort $(t_k, P_k)$ ascending by $t_k$ (stable).
3. Deduplicate by $t_k$ with keep-first; record `n_dropped_duplicates`.
4. If intraday: scan consecutive pairs and record every $k$ with
   $t_{k+1} - t_k > 3\,\Delta$ as a gap; **do not interpolate**.

The pipeline never silently drops valid data: duplicate removal is the only
deletion, and gaps are reported rather than filled. The audit record
`data_quality.json` carries `n_input`, `n_output`, `n_dropped_duplicates`,
`had_unsorted_input` and the list of gaps; for the SPY-daily window used in
the verification of this thesis, $n_{\text{input}} = 6662$ raw bars are
delivered, no duplicates are removed and no gaps are reported.

### 5.2.3 Session filter for intraday data

When the experiment requests `session_aware = true`, the cleaning output is
restricted to RTH bars before any further processing. This is a substantive
empirical decision, not a convenience: pre-market and post-market bars have
much thinner participation and different volatility regimes, and overnight
returns (16:00 ET on day $d$ to 09:30 ET on day $d+1$) dominate the intraday
variance. Including them produces a spurious changepoint at every session
open. Empirically, the filter reduces a 24-hour 15-minute SPY history of
about $378{,}000$ bars to roughly $25{,}000$ RTH bars, a factor matching the
ratio $6.5 / 96$ of the trading day to the full bar-grid day.

---

## 5.3 Chronological partition

For each clean series of length $N$ the timeline is partitioned by two
boundaries $\tau_1 < \tau_2$ into

$$
\mathcal{T}_{\text{train}} = \{t : t < \tau_1\},\quad
\mathcal{T}_{\text{val}} = \{t : \tau_1 \le t < \tau_2\},\quad
\mathcal{T}_{\text{test}} = \{t : t \ge \tau_2\}.
$$

The default policy is a chronological 70/15/15 split: $\tau_1$ is the
$\lfloor 0.70\, N\rfloor$-th timestamp and $\tau_2$ the
$\lfloor 0.85\, N\rfloor$-th. **Random shuffling is forbidden by
construction**, and the partition is performed *before* any feature is
computed.

For the SPY-daily window (2018-01-02 → present, $N = 2091$ after the time
window is applied), the realised cut points are $\tau_1 =$ 2023-10-25 and
$\tau_2 =$ 2025-01-28, yielding $n_{\text{train}} = 1463$ and
$n_{\text{val}} = n_{\text{test}} = 314$ as recorded in
`split_summary.json`.

**Proposition 5.1 (Leakage contract).** *Any preprocessing step whose
parameters depend on the data sees only $\mathcal{T}_{\text{train}}$.*

This is discharged in two places: §5.5 enforces it for the scaler, and §5.7
enforces it for the calibration operator. Together with C2 it implies
condition C3 of Definition 5.1.

---

## 5.4 Observation operators

The model assumes $y_k \mid S_k = j \sim \mathcal{N}(\mu_j, \sigma_j^2)$ but
says nothing about what $y_k$ is. The **observation design** decision selects
a causal operator $\phi: P_{1:k} \mapsto y_k$. Five families are studied in
this thesis. Letting $r_k = \log P_k - \log P_{k-1}$ denote the log-return:

| Family | Symbol | Definition | Warmup |
|---|---|---|---|
| LogReturn | $\phi^{\text{LR}}$ | $r_k$ | $1$ |
| AbsReturn | $\phi^{\text{AR}}$ | $\lvert r_k \rvert$ | $1$ |
| SquaredReturn | $\phi^{\text{SR}}$ | $r_k^2$ | $1$ |
| RollingVol$(w)$ | $\phi^{\text{RV},w}$ | $\sqrt{\tfrac{1}{w}\sum_{i=0}^{w-1}(r_{k-i}-\bar r_k^{(w)})^2}$ | $w$ |
| StandardizedReturn$(w,\varepsilon)$ | $\phi^{\text{ZR},w,\varepsilon}$ | $r_k / (\phi^{\text{RV},w}_k + \varepsilon)$ | $w$ |

Each operator depends only on prices at times $\le t_k$ (condition C2), and
each is undefined at the first warmup bars where its inputs do not yet exist;
those bars are dropped *with their timestamps preserved*, so the warmup count
is exact and recorded in `feature_summary.json`.

**Why not raw prices.** Raw prices fail C4: they are non-stationary, scale
dependent (a 1 % move on SPY at \$200 and at \$600 has different magnitudes),
drift, and exhibit level asymmetry. No registered feature family produces
$P_k$ unchanged.

**Session-aware variants.** For intraday data with `session_aware = true`,
$\phi^{\text{LR}}$ is overridden by a partial operator $\phi^{\text{LR,SA}}$
that is undefined whenever $t_{k-1}$ and $t_k$ fall in different RTH
sessions; the rolling-window operators reset their state at each session
boundary. The resulting stream contains no session-crossing returns.

**Population versus sample variance.** $\phi^{\text{RV},w}$ uses the $1/w$
divisor rather than $1/(w-1)$ because the window is treated as a
finite-sample *volatility proxy*, not an unbiased variance estimator;
this matches GARCH-style realised-volatility usage.

---

## 5.5 Leakage-safe scaling

A fitted scaler $\mathcal{S}: \mathbb{R} \to \mathbb{R}$ is applied to the
feature stream after warmup trimming. The default policy is z-scoring with
parameters $(\hat\mu, \hat\sigma)$ estimated *only* on the first
$n_{\text{train}}$ post-warmup feature values:

$$
\hat\mu = \frac{1}{n_{\text{train}}} \sum_{k=1}^{n_{\text{train}}} \phi_k,\qquad
\hat\sigma^2 = \frac{1}{n_{\text{train}}} \sum_{k=1}^{n_{\text{train}}} (\phi_k - \hat\mu)^2,
$$

$$
\mathcal{S}(x) = \begin{cases}(x - \hat\mu)/\hat\sigma & \hat\sigma > 0,\\ x - \hat\mu & \hat\sigma = 0.\end{cases}
$$

The fitted $\mathcal{S}$ is then *frozen* and applied uniformly to the
training, validation, test and online streams. The scaler has no
`refit_on(\cdot)` operation: re-estimation on later partitions is
structurally impossible, discharging Proposition 5.1 for the scaler.

For the SPY-daily verification run, fitting on $n_{\text{train}} = 1462$
post-warmup log-returns produced a scaler under which the *full* stream of
2090 z-scored observations has mean $0.012$ and standard deviation
$0.934$. The deviation from $1.0$ is the contribution of the
validation+test partitions; the training partition is, by construction, at
unit variance.

---

## 5.6 The synthetic generator

The empirical pipeline produces a financially real $y^{\text{real}}_{1:T}$
but provides no ground-truth changepoints. The synthetic pipeline samples
from the same MSM family as Chapter 2 with a *known* hidden path, providing
the only setting in which event-matching metrics are computable without
circularity.

### 5.6.1 Sampling law

Given a parameter vector $\vartheta = (\pi, P, \mu, \sigma^2)$ with $K \ge 2$
regimes:

$$
S_1 \sim \pi,\qquad
S_k \mid S_{k-1} = i \sim \mathrm{Cat}(P_{i,1}, \ldots, P_{i,K}),\qquad
y_k \mid S_k = j \sim \mathcal{N}(\mu_j, \sigma_j^2).
$$

**Algorithm 5.2 (Forward sampling).** *Inputs:* parameters $\vartheta$,
horizon $T$, RNG state $\omega$. *Outputs:* $(y_{1:T}, S_{1:T})$.

1. Draw $S_1 \sim \mathrm{Cat}(\pi)$.
2. For $k = 2, \ldots, T$: draw $S_k \sim \mathrm{Cat}(P_{S_{k-1}, \cdot})$.
3. For $k = 1, \ldots, T$: draw $y_k \sim \mathcal{N}(\mu_{S_k}, \sigma_{S_k}^2)$.

Before sampling, $\vartheta$ is validated against nine invariants ($K \ge 2$;
$\pi$ and rows of $P$ non-negative and sum to one within $10^{-9}$; all
$\sigma_j^2 > 0$; dimension consistency). All categorical draws use
$\mathrm{rand\_distr::WeightedIndex}$, which is deterministic given the RNG
state.

### 5.6.2 Ground-truth changepoints

The set of true changepoints is defined operationally:

$$
\mathcal{C}(S_{1:T}) = \{\,k \in \{2, \ldots, T\} \;:\; S_k \ne S_{k-1}\,\}.
$$

This is the *unique* source of truth used in the thesis. The benchmark
matcher in Chapter 7 consumes $\mathcal{C}$ with the same 1-based indexing
and the same boundary convention $1 < \tau \le T$, and is required to handle
$\mathcal{C} = \emptyset$ explicitly (a stream that stays in one regime makes
every alarm a false positive). Re-running the experiment with the same RNG
seed reproduces $\mathcal{C}$ exactly, so the ground truth is not persisted
to disk: it is *regenerated* from the seed at evaluation time, eliminating a
class of drift bugs.

### 5.6.3 The synthetic stream as an admissible observation

A synthetic stream satisfies C1–C5 trivially: timestamps are integer indices
(C1); $y_k$ is a function of $\omega$ only (C2); no preprocessing means no
leakage (C3); the law is by construction normal within each regime (C4–C5).
The remaining question is whether the synthetic distribution resembles the
empirical one along the dimensions that matter for detector evaluation. That
is the role of the calibration operator in §5.7.

---

## 5.7 The calibration operator $\mathcal{K}$

A purely hand-picked $\vartheta$ produces synthetic data that may have
nothing to do with markets. The calibration operator chooses $\vartheta$ from
empirical summary functionals of $y^{\text{real,train}}_{1:T}$:

$$
\mathcal{K}: \big( T_1(y^{\text{real,train}}),\, \ldots,\, T_m(y^{\text{real,train}}) \big) \;\longmapsto\; \vartheta.
$$

The objective is **statistical anchoring, not distributional cloning**: a
perfect distributional match would defeat the purpose of having an
interpretable two-state model whose hidden states have unambiguous meaning.
Two requirements are non-negotiable:

- **Feature consistency.** $T_i$ is evaluated on the same feature stream
  $\phi(P_{1:T})$ that the detector consumes, not on raw prices.
- **Partition restriction.** $T_i$ sees only $\mathcal{T}_{\text{train}}$,
  inheriting Proposition 5.1.

Two strategies are implemented: a transparent summary-based map
$\mathcal{K}_{\text{sum}}$ (§5.7.1), and an EM-based map
$\mathcal{K}_{\text{EM}}$ used for the sim-to-real experiments (§5.7.2).

### 5.7.1 Summary-based mapping $\mathcal{K}_{\text{sum}}$

The empirical summary $\mathcal{E}(y^{\text{real,train}})$ collects three
blocks of statistics, each calibrating one block of generator parameters:

| Block | Symbols | Purpose |
|---|---|---|
| Marginal anchors | $\bar y,\, s_y^2,\, q_{0.05}, q_{0.50}, q_{0.95}$ | calibrate $\mu_j,\, \sigma_j^2$ |
| Dependence proxies | $\hat\rho_1(y),\, \hat\rho_1(\lvert y\rvert),\, \mathrm{sgn}$-rate | validate the existence of regime structure |
| Episode durations | $\hat d_{\text{low}},\, \hat d_{\text{high}}$ | calibrate $P_{ii}$ |

The episode-duration estimator thresholds $\lvert y_k\rvert$ at $q_{0.95}$ and
returns the mean run lengths on the two sides. This estimator is
*deliberately coarse*: a finer estimator (e.g. fitting an HMM) would
entangle the calibration with the very inference procedure under test.

The operator is then composed of three rules.

**Mean policy (default `EmpiricalBaseline`).**
$\mu_j = \bar y$ for all $j$. Regime separation is carried by variance, which
matches the dominant pattern in financial returns where conditional means are
small relative to conditional standard deviations. Alternative policies
shift the means by $\pm \tfrac{1}{4} s_y$ around either $0$ or $\bar y$ for
mean-shift sensitivity studies; the $1/4$ factor is small by design to
prevent the detector from succeeding on obvious mean shifts alone.

**Variance policy (default `QuantileAnchored`).** Inverting the standard
Gaussian quantile $\Phi^{-1}(0.95) \approx 1.6449$:

$$
\sigma_{\text{low}} = \frac{\lvert q_{0.50} - q_{0.05} \rvert}{1.6449},\qquad
\sigma_{\text{high}} = \frac{\lvert q_{0.95} - q_{0.50} \rvert}{1.6449},
$$

with variances then sorted ascending so regime $0$ is canonically the calm
regime. A floor `min_high_low_ratio` is enforced on
$\sigma_{\text{high}}/\sigma_{\text{low}}$ to avoid degenerate identifiability.

**Duration-to-persistence map.** If $S_k = j$ stays in regime $j$ with
Bernoulli$(p_{jj})$ self-stay probability, the dwell time is geometric with
mean $1/(1 - p_{jj})$, hence

$$
\boxed{\;p_{jj} = 1 - \frac{1}{d_j}.\;}
$$

Off-diagonal mass $1 - p_{jj}$ is split uniformly across the remaining
$K-1$ regimes (symmetric off-diagonal — the most uninformative choice given
only duration targets). Boundary conditions: $d_j > 1$ is required (a
regime that cannot persist for one step on average is mathematically
incompatible with Markov structure), and $p_{jj}$ is clamped to
$[10^{-6},\, 1 - 10^{-6}]$ to keep EM and the forward filter numerically
interior. The initial distribution defaults to $\pi_j = 1/K$.

**The complete operator for $K = 2$.** Composing the three rules,
$\mathcal{K}_{\text{sum}}$ maps
$(\bar y,\, q_{0.05}, q_{0.50}, q_{0.95},\, \hat d_{\text{low}}, \hat d_{\text{high}})$ to

$$
\begin{aligned}
\mu &= (\bar y,\ \bar y),\\
\sigma^2 &= \!\left(\,\big(\tfrac{|q_{0.50}-q_{0.05}|}{1.6449}\big)^2,\ \big(\tfrac{|q_{0.95}-q_{0.50}|}{1.6449}\big)^2\right)\!\text{ sorted}\!\uparrow,\\
P &= \begin{pmatrix} 1 - \tfrac{1}{\hat d_{\text{low}}} & \tfrac{1}{\hat d_{\text{low}}} \\[2pt] \tfrac{1}{\hat d_{\text{high}}} & 1 - \tfrac{1}{\hat d_{\text{high}}} \end{pmatrix},\\
\pi &= (1/2,\ 1/2).
\end{aligned}
$$

Every quantity on the right is an explicit empirical summary; every quantity
on the left is a generator parameter; the map is deterministic and
auditable.

### 5.7.2 EM-based mapping $\mathcal{K}_{\text{EM}}$ (Quick-EM)

$\mathcal{K}_{\text{sum}}$ is interpretable but heuristic: there is no
guarantee that $\vartheta = \mathcal{K}_{\text{sum}}(\mathcal{E})$ is a
maximum-likelihood description of the training partition. For **sim-to-real**
experiments — where a detector is *trained* on synthetic data and *tested*
on real data — a stronger calibration is needed: the synthetic stream must
be a sample from the same generative law that EM would extract from the real
training data, otherwise the train and test distributions diverge before the
detector sees them.

The Quick-EM strategy realises this by running Baum–Welch directly on
$y^{\text{real,train}}_{1:T}$:

**Algorithm 5.3 ($\mathcal{K}_{\text{EM}}$).**

1. Initialise $\vartheta_0$ via the quantile heuristic of §5.7.1 (used only
   as an EM starting point).
2. Run `fit_em(observations, $\vartheta_0$, max_iter, tol)` to convergence
   or to `max_iter` iterations.
3. Return $\hat\vartheta_{\text{EM}}$ as the generator parameters.

Symbolically,

$$
\mathcal{K}_{\text{EM}}:\; y^{\text{real,train}}_{1:T_{\text{train}}}
\;\xrightarrow{\;\text{Baum–Welch}\;}\;
\hat\vartheta_{\text{EM}} = (\hat\pi,\, \hat P,\, \hat\mu,\, \hat\sigma^2).
$$

The synthetic stream $y^{\text{syn}}\!\!\sim\! \mathrm{MSM}(\hat\vartheta_{\text{EM}})$ then
samples exactly from the law that EM would have used to score the real
training data — the natural train-time analogue of the test-time
likelihood.

---

## 5.8 Verification and the sim-to-real bridge

### 5.8.1 Calibration verification

A calibrated $\vartheta$ is useful only if the synthetic data it produces
actually resembles the empirical target on the same summary functionals.
The verifier $\mathcal{V}(\vartheta;\, \mathcal{E})$ samples
$y^{\text{syn}}_{1:T} \sim \mathrm{MSM}(\vartheta)$, recomputes
$\mathcal{E}^{\text{syn}}$, and returns the per-field difference

$$
\Delta_i = T_i(y^{\text{syn}}) - T_i(y^{\text{real}})
$$

together with a per-field tolerance check $\lvert \Delta_i \rvert \le \tau_i$
(or, for variance, $\lvert \Delta_i \rvert / T_i(y^{\text{real}}) \le \tau_i$).
The defaults are deliberately loose ($\tau_{\text{mean}} = 0.25$,
$\tau_{\text{var}} = 0.50$ relative, $\tau_{q} = 0.30$,
$\tau_{|\rho_1|} = 0.20$, $\tau_{\text{sgn}} = 0.20$): the goal is anchoring,
not cloning.

**Masking.** A target mask selects which fields participate in the global
pass/fail verdict while preserving per-field records. The mask is derived
from the active policy:

- `MeanPolicy::ZeroCentered` disables the mean check (the synthetic mean is
  fixed at zero by construction).
- `CalibrationStrategy::QuickEm` disables the dependence summaries
  $\hat\rho_1(\lvert y\rvert)$ and the sign-change rate (which emerge as
  side-effects of EM rather than as calibration targets).

Masked fields are recorded as `checked: false` so that downstream consumers
can distinguish a *skipped* check from a *passed* one.

### 5.8.2 The sim-to-real bridge

The `SimToReal` experiment mode composes the synthetic pipeline (§5.6) and
the empirical pipeline (§5.2–§5.5) into a single workflow whose stages map
onto the canonical runner from Chapter 7:

| Stage | Effect under SimToReal |
|---|---|
| `resolve_data` | Load $y^{\text{real}}$; run $\mathcal{K}_{\text{EM}}$ on the train partition; sample $y^{\text{syn}}$ of length `horizon`; record the scale-consistency check below |
| `build_features` | Fit a scaler $\mathcal{S}$ on $y^{\text{real,train}}$ log-returns; apply it to both streams |
| `train_or_load_model` | Fit EM **only** on $\mathcal{S}(y^{\text{syn}})$ → $\theta^{\text{frozen}}$ |
| `run_online` | Score the synthetic-trained $\theta^{\text{frozen}}$ over the *real* validation+test stream |
| `evaluate_real` | Route A (proxy events) and Route B (segmentation coherence) on the real test partition |

**Scale-consistency policy.** Because the detector trained on synthetic
features is evaluated on real features, the two streams must share a common
numeric scale. The pipeline enforces a one-sided z-scoring policy: a single
scaler $\mathcal{S}$ is fitted on the real training partition and then
*applied* (not refitted) to the synthetic stream. After scaling,

$$
\text{rel\_error} \;=\; \frac{\lvert \hat\sigma(y^{\text{syn,scaled}}) - \hat\sigma(y^{\text{real,scaled}}) \rvert}{\hat\sigma(y^{\text{real,scaled}})}
\;\le\; 0.10
$$

is required and recorded in `synthetic_training_provenance.json`. The
verification run on SPY daily produced

$$
\hat\sigma_{\text{real}} = 0.01302,\quad \hat\sigma_{\text{syn}} = 0.01234,\quad \text{rel\_error} = 0.0524,
$$

passing the 10 % bound, with $\hat\vartheta_{\text{EM}}$ obtained in
$31$ Baum–Welch iterations to $\log\mathcal{L} = 4581.29$ and
$\hat d = (46.76,\, 24.23)$.

### 5.8.3 The role of the calibration verdict

When $\mathcal{V}$ reports `within_tolerance: false` — as it does for SPY
daily under $\mathcal{K}_{\text{sum}}$ because the empirical
$\hat\rho_1(y) \approx -0.15$ cannot be reproduced by any $K = 2$ Gaussian
MSM with $\mu_0 = \mu_1$ — the response is *not* to abort. The chosen
scenario family is a deliberately simpler analogue, not a faithful clone.
The synthetic ground truth remains valid for detector benchmarking; what is
recorded is the *limitation*: detector performance on this synthetic stream
does not predict performance on the real stream's full serial structure
one-for-one. This is the methodological cost of using an interpretable
generator, and the calibration verdict makes it auditable.

---

## 5.9 The observation streams used in this thesis

### 5.9.1 End-to-end transformation flow

$$
\begin{array}{c}
\text{Vendor response (newest-first)} \\
\big\downarrow\ \text{clean: sort, dedup, gap-check (Alg.\ 5.1)} \\
\text{CleanSeries (ascending, gap-reported)} \\
\big\downarrow\ \text{RTH filter if } \text{session\_aware}=\text{true} \\
\text{CleanSeries (RTH only)} \\
\big\downarrow\ \text{chronological partition (§5.3)} \\
\mathcal{T}_{\text{train}} \sqcup \mathcal{T}_{\text{val}} \sqcup \mathcal{T}_{\text{test}} \\
\big\downarrow\ \text{feature operator } \phi \text{ (§5.4) and scaler } \mathcal{S}_{|train} \text{ (§5.5)} \\
y^{\text{real}}_{1:T} \\[6pt]
\hline\\[-6pt]
\text{Under \texttt{SimToReal}: } \mathcal{K}_{\text{EM}}(y^{\text{real,train}}) = \hat\vartheta \\
\big\downarrow\ \text{Alg.\ 5.2 with seed } \omega \\
(y^{\text{syn}}_{1:T},\, S_{1:T}) \\
\big\downarrow\ \mathcal{S}_{|real\text{-train}} \text{ applied to } y^{\text{syn}} \\
y^{\text{syn,scaled}}_{1:T}\ \text{(train)}\quad y^{\text{real,scaled}}_{1:T}\ \text{(test)}
\end{array}
$$

The lower block is exercised only in `SimToReal` mode; in pure-real mode the
detector is both trained and tested on $y^{\text{real}}$, and in
pure-synthetic mode there is no real stream at all.

### 5.9.2 Provenance artifacts

Every run writes a small set of JSON audit files that make the chain
reproducible:

| Artifact | Source | Contents |
|---|---|---|
| `data_quality.json` | §5.2.2 | $n_{\text{input}}, n_{\text{output}}, n_{\text{dropped\_duplicates}}, \text{had\_unsorted\_input}, \text{gaps}$ |
| `split_summary.json` | §5.3 | $n_{\text{train}}, n_{\text{val}}, n_{\text{test}}, \tau_1, \tau_2$, partition timestamps |
| `feature_summary.json` | §5.4–§5.5 | feature label, $n_{\text{raw}}, n_{\text{feature}}, \text{warmup\_trimmed}, \text{train\_n}$, scaler stats |
| `synthetic_training_provenance.json` | §5.7.2 + §5.8.2 | strategy, $\hat\vartheta_{\text{EM}}$, $\hat d$, `scale_check {empirical\_std, synthetic\_std, rel\_error, within\_tolerance}`, seed |
| `sim_to_real_summary.json` | §5.8.2 | `train_source`, `test_source`, $\hat\vartheta$, Route A & Route B metrics |

For the verification batch used throughout the empirical chapters
(`verify_2026_05_15`), the SPY-daily numbers cited above
($n_{\text{train}} = 1463$, $\tau_1 =$ 2023-10-25,
$\hat\sigma(\text{scaled stream}) = 0.934$, rel\_error $= 0.0524$) are
quoted from these artifacts directly.

### 5.9.3 What the model now sees

By the time $y_{1:T}$ enters EM and the online detector:

1. **(C1)** the sequence is strictly chronological;
2. **(C2)** every $y_k$ is a measurable function of $P_{1:k}$ (operators of
   §5.4 are causal by signature);
3. **(C3)** the scaler is fitted only on the training prefix (§5.5), and
   so is $\mathcal{K}$ (§5.7); validation and test never feed back;
4. **(C4)** $y_k$ is a stationarity-friendly transformation of prices,
   never the price itself;
5. **(C5)** in synthetic mode the regime-conditional Gaussian assumption
   holds by construction; in real and sim-to-real modes the calibration
   verifier of §5.8 quantifies how far it is violated and records the
   verdict.

Conditions C1–C4 are *enforced* by the pipeline; condition C5 is *audited*.
Every guarantee in this list is the responsibility of a separate module of
the codebase, and the union of those modules is exercised by 328 passing
tests in the verification batch.

The observation sequence on which all subsequent results in this thesis are
computed is exactly this $y_{1:T}$, together with the synthetic ground
truth $\mathcal{C}(S_{1:T})$ where applicable.
