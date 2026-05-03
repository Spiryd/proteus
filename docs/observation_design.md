# Observation Design and Feature Engineering

**Phase 16 — Markov Switching Model Project**

---

## 1. The Central Question

The Markov Switching Model assumes

$$y_t \mid S_t = j \;\sim\; \mathcal{N}(\mu_j,\, \sigma_j^2), \qquad j \in \{1, \dots, K\},$$

with a first-order hidden Markov chain $S_t$ governed by transition matrix
$P$ and initial distribution $\pi$.  The model says nothing about what
$y_t$ *is*.  For real market data, that is a scientific choice.

> **Observation Design** is the act of deciding which time-dependent
> statistic of market prices to treat as the modeled process $y_t$.

This is not a pre-processing detail.  Feature choice determines what kind of
market change the detector is sensitive to.  Comparing detectors across
feature families is therefore itself an empirical contribution of the thesis.

---

## 2. Why Raw Prices Are Unsuitable

Let $P_t$ denote the price of an asset at time $t$.  Feeding $y_t = P_t$
directly into the model is almost always a weak design for regime detection.
The main pathologies are:

| Pathology | Consequence |
|---|---|
| Non-stationarity (trend) | EM will fit regime means to slowly drifting levels, not genuine distributional changes |
| Scale dependence | The same regime can span a wide price range, so $\mu_j$ loses interpretability |
| Long-run drift | A detector may fire on ordinary trend continuation rather than true structural change |
| Price-level asymmetry | A 1-point move on a 10-dollar stock and on a 500-dollar stock are incomparable |

In contrast, financial econometrics studies the *return process* and related
dispersion statistics, which are approximately scale-free and conditionally
stationary.  The thesis explicitly motivates detecting "changes in the
distribution of the observed process based on changes in time-series
statistics used in econometric practice."  That commitment rules out raw
prices as the primary observed process.

---

## 3. Observation Families

All families below are defined on the **log-return** $r_t$, which is
computed from consecutive prices:

$$r_t = \log P_t - \log P_{t-1}, \qquad t = 1, 2, \dots$$

The first observation $P_0$ is consumed as a seed; the return series begins
at $t = 1$.

### 3.1 Log Returns

$$\boxed{y_t = r_t = \log P_t - \log P_{t-1}}$$

**Warmup:** 1 price bar (the initial seed price).

**Detector sensitivity:** Changes in the *distribution* of returns — mean
shift, volatility change, or both.

**Interpretation:** In the Gaussian MSM, each regime $j$ corresponds to a
distribution $\mathcal{N}(\mu_j, \sigma_j^2)$ over log returns.  A regime
with $\mu_j > 0$ is a trending-up state; one with $\sigma_j^2$ large is a
volatile state.  This is the baseline observation design for the thesis.

**Causality:** $r_t$ depends only on $P_{t-1}$ and $P_t$, both of which
are observable at time $t$.  ✓

---

### 3.2 Absolute Returns

$$\boxed{y_t = a_t = |r_t|}$$

**Warmup:** 1 price bar.

**Detector sensitivity:** Changes in *return magnitude*, irrespective of
direction.

**Interpretation:** Absolute returns are a simple volatility proxy.
High-$a_t$ periods correspond to turbulent market states, regardless of
whether the market is moving up or down.  Using $y_t = a_t$ steers the
detector toward activity and stress rather than directional trend.

**Causality:** ✓ (same as log return).

---

### 3.3 Squared Returns

$$\boxed{y_t = q_t = r_t^2}$$

**Warmup:** 1 price bar.

**Detector sensitivity:** Changes in *second-moment structure*.

**Interpretation:** $q_t$ is a classical volatility proxy (it is the
pointwise estimator of $\mathbb{E}[r_t^2]$).  It tends to be noisier than
rolling volatility but is strictly causal and requires no window parameter.
It is approximately proportional to $a_t^2$, so it amplifies large moves
more than $a_t$ does.

**Causality:** ✓

---

### 3.4 Rolling Volatility Proxy

Define the trailing mean over a window of size $w$:

$$\bar{r}_t^{(w)} = \frac{1}{w} \sum_{k=0}^{w-1} r_{t-k},$$

and the trailing population standard deviation:

$$\boxed{v_t^{(w)} = \sqrt{\frac{1}{w} \sum_{k=0}^{w-1} \bigl(r_{t-k} - \bar{r}_t^{(w)}\bigr)^2}}$$

**Warmup:** $w$ price bars (the first $w$ prices yield $w-1$ returns, and
the first full $v_t^{(w)}$ is defined when the return buffer contains $w$
values, i.e., at price index $w$).

**Detector sensitivity:** Changes in the *local volatility level*.

**Interpretation:** $v_t^{(w)}$ smooths over the noisiness of $q_t$ by
averaging over a window.  In turbulent regimes it rises; in calm regimes it
falls.  It is especially natural for financial-crisis detection and
volatility-regime analysis.

**Normalization convention:** The divisor is $w$ (population variance of
the window), not $w-1$.  The choice is appropriate because the window is
treated as a fixed-width volatility proxy, not as an unbiased sample-variance
estimator of an infinite population.

> **Thesis requirement:** This choice must be stated explicitly in the thesis.
> Any alternative (e.g., $w-1$ Bessel correction) would produce slightly
> different volatility values.  All registered experiments, calibration
> statistics, and real-data evaluations use the $1/w$ population formula.
> The difference is negligible for large windows ($w \geq 20$) but should
> be documented for reproducibility.

**Causality:** The window $\{r_{t-w+1}, \dots, r_t\}$ uses only past and
present returns.  ✓

**Window selection:** The choice of $w$ is a modeling decision.  Larger $w$
produces a smoother, slower-reacting volatility estimate; smaller $w$
reacts faster but is noisier.  Recommended starting values:
- Daily data: $w \in \{20, 60\}$ (approximately 1 month, 3 months).
- Intraday 5-min: $w \in \{12, 24\}$ (approximately 1 hour, 2 hours).

---

### 3.5 Standardized Returns

$$\boxed{z_t = \frac{r_t}{v_t^{(w)} + \varepsilon}}$$

where $\varepsilon > 0$ is a small numerical stabilizer (default $10^{-8}$).

**Warmup:** $w$ price bars (same as rolling volatility).

**Detector sensitivity:** Changes in the *relative shock size* — unusually
large moves relative to the recent volatility level.

**Interpretation:** $z_t$ separates "large move in a calm market" from "large
move in an already-volatile market."  It can detect structural breaks that
are hidden when volatility is high (because absolute returns look normal) or
false alarms in calm markets (because small absolute moves are amplified when
divided by near-zero volatility, moderated by $\varepsilon$).

**Note on self-reference:** The current return $r_t$ appears in both the
numerator and the denominator (via $v_t^{(w)}$), since the window includes
$r_t$ itself.  This is causal (no future information used) but means $z_t$
is not a pure leave-one-out normalized statistic.  This is a standard
practice in rolling-normalization filters and does not create bias problems.

**Causality:** ✓

---

## 4. Comparative Feature Summary

| Family | Symbol | Warmup | Sensitive to | Free parameters |
|---|---|---|---|---|
| Log return | $r_t$ | 1 | Mean and variance of returns | — |
| Absolute return | $a_t$ | 1 | Return magnitude / volatility | — |
| Squared return | $q_t$ | 1 | Second moment | — |
| Rolling volatility | $v_t^{(w)}$ | $w$ | Volatility level | $w$, session reset |
| Standardized return | $z_t$ | $w$ | Normalized shock size | $w$, $\varepsilon$, session reset |

The minimal recommended set for the thesis is $\{r_t,\; a_t,\; v_t^{(w)}\}$,
giving one directional, one magnitude, and one level-of-volatility observable.
This creates a three-way comparison along the axis:

$$\text{MS detector on returns} \quad\text{vs.}\quad \text{MS detector on volatility proxy}$$

which is itself a well-posed empirical question.

---

## 5. Causality Requirement

Because the detector is evaluated **online**, any feature used as $y_t$ must
satisfy the strict causality constraint:

$$y_t = \Phi(P_0, P_1, \dots, P_t) \qquad \text{(no future observations)}$$

Equivalently, for a rolling feature with window $w$:

$$y_t = g(r_{t-w+1}, r_{t-w+2}, \dots, r_t)$$

The window must be **trailing**, never centered or forward-looking.  A
centered window of the form $g(r_{t-w/2}, \dots, r_{t+w/2})$ would
constitute look-ahead bias and would make the entire online evaluation
invalid.

**Implementation check:** Every rolling feature in this project uses a ring
buffer that accepts values one-at-a-time in forward chronological order and
emits a result only after the buffer is full.  No future values are buffered
ahead of time.

---

## 6. Warmup Policy

Several families require a burn-in period during which the feature value is
undefined:

- $r_t$: undefined at $t = 0$ (no predecessor price).
- $v_t^{(w)}$: undefined for the first $w - 1$ returns after the seed price
  (i.e., the first defined value is at price index $w$).

**Policy:** Drop the warmup prefix.  The output series begins at the first
time index where the feature value is defined.  The dropped count is
recorded in `FeatureStreamMeta::n_warmup_dropped`.

This policy is preferred over imputing undefined values (e.g., with 0 or
the first valid value), because:
- imputation introduces artificial structure at the series start,
- the EM algorithm would fit regime parameters partly on fabricated data,
- the warmup prefix is typically small relative to the full series length.

**Effective start time:** The caller can recover the original timestamp of
the first output observation from `FeatureStream::meta.first_ts`.

---

## 7. Daily vs Intraday Feature Policies

The choice of feature family interacts with data frequency.

### 7.1 Daily Data

Daily log returns are well-defined and cross-day returns carry full
overnight price information.  No session boundary policy is needed: every
calendar-day observation is already one "session."

Recommended starting features:
- $r_t$ — standard baseline.
- $v_t^{(w)}$ with $w \in \{20, 60\}$ — 1-month and 3-month volatility.
- $a_t$ — activity proxy.

### 7.2 Intraday Data

Intraday data has **session structure**: bars within a trading day belong to
one session; consecutive sessions are separated by an overnight gap during
which no trading occurs.

This creates two policy decisions for rolling features:

**Decision A — Cross-session returns.**
Should a log return be computed across the overnight gap?

- If `session_aware = true`: the pair $(P_{\text{last bar of day 1}},\,
  P_{\text{first bar of day 2}})$ does **not** produce a return.  The first
  bar of each session is silently dropped from the return stream.
- If `session_aware = false`: the overnight pair produces a return that
  includes the full overnight price move.

**Recommended default:** `session_aware = true`.  Overnight gaps are
qualitatively different from within-session moves (market closure, news
accumulation, etc.) and should not be fed into intraday volatility
estimates without explicit justification.

**Decision B — Rolling window reset at session open.**
Should the rolling window be cleared at each session open?

- If `session_reset = true`: the ring buffer is emptied at each session
  open.  The first $w$ bars of each session are warmup; no feature value is
  emitted for those bars.
- If `session_reset = false`: the rolling window accumulates across sessions,
  so the volatility estimate at the start of a new session is partly based
  on the previous session's returns.

**Recommended default:** `session_reset = true` for intraday rolling
volatility.  Mixing overnight gaps into the volatility window distorts the
intraday volatility estimate and conflates within-day activity with
overnight risk.

---

## 8. Train-Only Normalization

If scaling is applied (ZScore or RobustZScore), the normalization parameters
must be estimated **on the training partition only**.

### 8.1 ZScore

$$y_t' = \frac{y_t - \hat\mu_{\text{train}}}{\hat\sigma_{\text{train}}}$$

where $\hat\mu_{\text{train}}$ and $\hat\sigma_{\text{train}}$ are the mean
and (population) standard deviation of the **training** feature values.

### 8.2 Robust ZScore

$$y_t' = \frac{y_t - \hat m_{\text{train}}}{\hat s_{\text{train}}}$$

where $\hat m$ is the training-set median and $\hat s = Q_{75} - Q_{25}$ is
the training-set interquartile range.  Robust scaling is preferable when the
return series contains outliers (e.g., large single-day crashes) that would
inflate the standard deviation in the ZScore.

### 8.3 Leakage invariant

The `FittedScaler` is constructed from training values only and then
**frozen**.  The same frozen scaler is applied to validation, test, and the
live online stream.  The parameters $(\hat\mu, \hat\sigma)$ or
$(\hat m, \hat s)$ are never updated after the training phase.  This mirrors
exactly the practice for model parameters in the EM phase: once fitted on
training data, the model is held fixed for all subsequent use.

**Degenerate case:** If $\hat\sigma = 0$ (constant feature series), the
scaler uses $\hat\sigma = 1$ (identity transform) to avoid division by zero.

---

## 9. Architecture: Feature Layer

The feature layer sits between the data pipeline (Phase 15) and the model
fitting / online detection phases.

```
Phase 15 output
    CleanSeries { observations: Vec<Observation>, meta: DatasetMeta }
         │
         │  FeatureConfig {
         │      family: FeatureFamily,   // which feature to compute
         │      scaling: ScalingPolicy,  // None / ZScore / RobustZScore
         │      n_train: usize,          // training prefix length (for scaler)
         │      session_aware: bool,     // intraday session policy
         │  }
         ▼
  FeatureStream::build(prices, data_meta, config)
         │
         ▼
  FeatureStream {
      observations: Vec<Observation>,  // y_t, model-ready
      meta: FeatureStreamMeta,         // provenance
      scaler: FittedScaler,            // frozen, for online use
  }
         │
         ├── .values()  →  Vec<f64>     →  EM fitting
         ├── online bar by bar          →  StreamingSession<D>
         └── .experiment_label()       →  benchmark reporting
```

**Separation of concerns:**
- Phase 15 (`data/`) owns: fetching, caching, validation, RTH filtering,
  session labelling, chronological splitting.
- Phase 16 (`features/`) owns: observation design, causal feature
  computation, scaling.
- Phases 2–10 (`model/`) own: EM, filtering, smoothing, diagnostics.
- Phases 11–13 (`online/`, `detector/`) own: online inference, alarm logic.
- Phase 14 (`benchmark/`) owns: benchmark evaluation.

No leakage of feature-design logic into the model or detector layers.

---

## 10. Observation Design as Scientific Hypothesis

In the thesis, the choice of $y_t$ is not a technicality — it is part of
the scientific question.

Each observation design implicitly answers:

| Design | Thesis question |
|---|---|
| $y_t = r_t$ | Do financial regimes correspond to changes in the return distribution? |
| $y_t = a_t$ or $v_t^{(w)}$ | Are market states better identified through volatility structure than directional returns? |
| $y_t = z_t$ | Are structural breaks revealed by normalized shocks rather than raw returns? |

Running the same Markov Switching detector under different observation designs
and comparing the resulting alarm patterns, regime classifications, and
benchmark metrics is a meaningful empirical contribution.  The feature layer
in Phase 16 is designed to make that comparison reproducible and
configuration-driven.

---

## 11. Module Structure

| File | Contents |
|---|---|
| `src/features/family.rs` | `FeatureFamily` enum; per-family warmup and label metadata |
| `src/features/transform.rs` | Pointwise transforms: `log_return`, `abs_return`, `squared_return`; session-aware batch variants |
| `src/features/rolling.rs` | `RollingStats` ring buffer; `rolling_vol`, `standardized_returns`; session-reset variants |
| `src/features/scaler.rs` | `ScalingPolicy`, `FittedScaler::fit` / `transform` / `inverse_transform` |
| `src/features/stream.rs` | `FeatureConfig`, `FeatureStream`, `FeatureStreamMeta` — main pipeline entry point |
| `src/features/mod.rs` | Module scaffold and flat re-exports |

---

## 12. Test Coverage Summary

| Test class | File | Count |
|---|---|---|
| Warmup metadata correctness | `family.rs` | 4 |
| Log/abs/squared return formula | `transform.rs` | 5 |
| Session-aware return skipping | `transform.rs` | 4 |
| `different_day` predicate | `transform.rs` | 1 |
| `RollingStats` ring-buffer | `rolling.rs` | 6 |
| Batch `rolling_vol` | `rolling.rs` | 3 |
| Session-reset rolling vol | `rolling.rs` | 2 |
| Standardized returns | `rolling.rs` | 2 |
| `FittedScaler` (ZScore/Robust/None) | `scaler.rs` | 10 |
| Leakage invariant (train-only fit) | `scaler.rs` | 1 |
| Inverse transform roundtrip | `scaler.rs` | 1 |
| `FeatureStream` end-to-end | `stream.rs` | 9 |
| **Total** | | **~50** |

All tests are deterministic and run under `cargo test` with no external dependencies.

---

## 13. Common Mistakes to Avoid

**M1 — Centered rolling windows:** Using a window $[t - w/2, t + w/2]$
instead of $[t - w + 1, t]$ introduces look-ahead bias.  Always use
trailing windows.

**M2 — Cross-session overnight returns in intraday volatility:** The
overnight price gap is structurally different from within-session moves.
Use `session_aware = true` unless there is specific justification to include
overnight moves.

**M3 — Fitting the scaler on the full dataset before splitting:** The scaler
must be fitted on the training partition only.  Fitting on all data uses
test-set distributional information and inflates apparent stationarity.

**M4 — Treating feature choice as a coding detail:** The observation design
determines the scientific content of the detector.  Document and justify
each experiment's feature choice.

**M5 — Too many feature families without justification:** Start with the
minimal set $\{r_t, a_t, v_t^{(w)}\}$.  Add extensions only when they
serve a clearly stated thesis question.

---

## 14. Open Questions for Later Phases

- **Realized volatility blocks:** For intraday data, define
  $\mathrm{RV}_t = \sum_{k \in \mathcal{B}_t} r_k^2$ over a fixed block
  $\mathcal{B}_t$ (e.g., one trading session).  This converts the intraday
  return series into a daily realized-variance series, which may be a more
  stable EM input.

- **Volume features:** If reliable volume data is available, log volume and
  volume surprise may serve as auxiliary features.

- **Multivariate observations:** The current EM and detector assume
  univariate $y_t$.  A bivariate $(r_t, v_t^{(w)})$ observation would
  require a multivariate Gaussian emission, which is a non-trivial extension.

- **Window sensitivity analysis:** Running the detector under multiple window
  values $w$ and reporting stability of alarm patterns would quantify the
  sensitivity of conclusions to the rolling-window hyperparameter.
