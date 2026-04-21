# Real-Data Evaluation Protocol (Route A + Route B Only)

**Phase 18 — Markov Switching Model Project**

---

## 1. Problem Setting

For synthetic data, changepoint evaluation is supervised because latent regimes
and true changepoint times are known. For real financial data, true changepoints
are not observed. Therefore, real-data evaluation is an unsupervised assessment
problem and must be defined by explicit criteria rather than direct label
recovery.

In this phase, evaluation is restricted to two routes:

- **Route A (Proxy Events):** external alignment of alarms with meaningful market anchors.
- **Route B (Self-Consistency):** internal quality of detector-induced segmentation.

Route C (downstream predictive usefulness) is explicitly excluded.

---

## 2. Formal Objects

Let the detector produce alarms at ordered indices

$$
\mathcal{A} = \{a_1,\dots,a_N\}, \quad 1 \le a_1 < \cdots < a_N \le T.
$$

Let proxy events be

$$
\mathcal{E} = \{e_1,\dots,e_M\},
$$

where each event is either:

- point anchor: $e_m = \tau_m$,
- window anchor: $e_m = [\ell_m, u_m]$, with $\ell_m \le u_m$.

Proxy events are evaluation anchors, not exact ground-truth changepoints.

---

## 3. Route A: Proxy Event Alignment

### 3.1 Matching windows

For a window event $[\ell_m, u_m]$, alarm $a_n$ is aligned iff

$$
a_n \in [\ell_m, u_m].
$$

For a point event $\tau_m$, define tolerance window

$$
[\tau_m - w^-,\, \tau_m + w^+].
$$

Then $a_n$ is aligned iff $a_n$ lies in that window.

For causal-only matching, use

$$
[\tau_m,\, \tau_m + w^+],
$$

which forbids pre-event matches.

### 3.2 Metrics

1. **Event Coverage**
$$
\text{Coverage} = \frac{\#\{e_m:\exists a_n \text{ aligned to } e_m\}}{M}.
$$

2. **Alarm Relevance**
$$
\text{Relevance} = \frac{\#\{a_n:\exists e_m \text{ aligned to } a_n\}}{N}.
$$

3. **Aligned Delay**
For point-like reference time $\tau_m$, if $a(e_m)$ is first aligned alarm:
$$
d_m = a(e_m) - \tau_m.
$$
Report summary (mean/median/min/max) over matched events.

4. **Alarm Density Around Events**
Compare alarm density inside event windows vs outside event windows.

### 3.3 Interpretation constraint

Route A asks whether alarms are concentrated near meaningful disruptions. It does
not prove exact changepoint truth, and unmatched alarms are not automatically
false discoveries.

---

## 4. Route B: Segmentation Self-Consistency

### 4.1 Alarm-induced segments

Alarms partition the series into half-open intervals:

$$
[1,a_1),\, [a_1,a_2),\,\dots,\,[a_N,T].
$$

In 1-based indexing, alarm $a_i$ belongs to the segment that starts at $a_i$.

### 4.2 Segment statistics

For segment $I_s$ with observations $y_t$:

1. Mean
$$
\bar y^{(s)} = \frac{1}{|I_s|}\sum_{t\in I_s} y_t.
$$

2. Variance
$$
\operatorname{Var}^{(s)}(y)=\frac{1}{|I_s|}\sum_{t\in I_s}(y_t-\bar y^{(s)})^2.
$$

3. Volatility summary (for return-like series): e.g. $\operatorname{mean}(|y_t|)$,
quantiles, tail rates.

4. Optional dependence summary: lag-1 autocorrelation.

### 4.3 Adjacent contrasts

For adjacent segments $I_s, I_{s+1}$:

$$
\Delta_\mu^{(s)} = \bar y^{(s+1)} - \bar y^{(s)},
$$
$$
\Delta_{\sigma^2}^{(s)} = \operatorname{Var}^{(s+1)}(y)-\operatorname{Var}^{(s)}(y).
$$

Also report effect-size style summaries (for example, mean shift scaled by
pooled segment standard deviation).

### 4.4 Global coherence

A coherent segmentation should show stronger between-segment differences and
lower within-segment dispersion. Useful global summaries:

- mean within-segment variance,
- mean absolute adjacent mean contrast,
- mean absolute adjacent variance contrast,
- derived coherence ratio.

### 4.5 Minimum segment length

Very short segments are unstable for statistical summaries. Apply a minimum
length policy (flag-only or exclusion from global summaries).

---

## 5. Why Route A + Route B is Strong

- Route A provides **external relevance**.
- Route B provides **internal statistical coherence**.

A detector that is strong on both is more convincing than one evaluated only on
one axis. This dual protocol is appropriate for unsupervised real-data
changepoint evaluation in finance.

---

## 6. Protocol Boundaries

This phase intentionally excludes downstream predictive usefulness. Claims from
this phase are restricted to:

- alignment of alarms with proxy market anchors,
- statistical coherence of detector-induced segmentation.

No direct claim of true latent-regime recovery on real data is made.

---

## 7. Canonical Evaluation Flow

1. Run fixed-parameter online detector and collect alarm indices.
2. Route A: compute proxy-event alignment metrics.
3. Route B: build segmentation and compute segment/coherence summaries.
4. Export combined run artifact for tables/plots.

This flow defines a reproducible real-data evaluation standard for the thesis.
