# Chapter 4 — Online Changepoint Detection Based on Markov-Switching Inference

**Scope:** `src/detector/`, `src/online/mod.rs`, `src/detector/frozen.rs` — the complete online detection pipeline as implemented in the live experiment code.

**Relation to previous chapters:** Chapter 2 established the Gaussian MSM and its EM estimator. Chapter 3 described the forward filter, backward smoother, and pairwise posteriors used during training. This chapter describes how those trained parameters are frozen and a causally filtered posterior is converted into a real-time changepoint signal.

---

## 4.1 From regime inference to changepoint detection

A Gaussian Markov-switching model supplies, at every time $t$, a filtered posterior over the $K$ latent regimes:

$$\alpha_{t|t}(j) \;=\; P(S_t = j \mid y_{1:t}), \qquad j = 1,\dots,K.$$

This is a *soft* statement about the current state of the world. It is not, by itself, a changepoint signal: it says nothing about *whether a transition occurred*, only about *which regime is currently probable*. Converting the filtered posterior into a binary alarm requires an explicit detection layer.

### 4.1.1 The changepoint detection problem

Following the formulation of Adams & MacKay (2007) and the classical setup of Page (1954), a changepoint detector is a mapping from the data history $y_{1:t}$ to a scalar score $s_t \in \mathbb{R}_{\geq 0}$, together with a rule for issuing a binary alarm $a_t \in \{0, 1\}$:

$$a_t = \mathbf{1}\bigl[s_t \geq \tau\bigr]$$

for threshold $\tau > 0$. The two canonical approaches in the literature are:

- **CUSUM (Page, 1954).** Accumulates a running sum of log-likelihood ratios $\ell_t = \log[f_1(y_t)/f_0(y_t)]$ under a null and alternative hypothesis, resetting when the sum drops below zero. Sensitive to sustained shifts but requires pre-specifying the pre- and post-change distributions.

- **Bayesian Online Changepoint Detection (BOCPD, Adams & MacKay, 2007).** Maintains an exact posterior over the *run length* — the number of steps since the last changepoint. Naturally integrates over unknown post-change parameters but the full recursion has cost linear in run length.

The present work takes a third path: rather than running a separate change-detection model, it *re-uses the filtered posterior of an already-estimated MSM* as the input to a lightweight detection function. This is computationally cheaper than BOCPD for long runs and richer than a fixed-alternative CUSUM because the regime distributions are data-driven rather than pre-specified.

### 4.1.2 Three approaches to turning regime posteriors into alarms

The core idea can be expressed abstractly. Let $\boldsymbol{\alpha}_{t|t} = (\alpha_{t|t}(1),\dots,\alpha_{t|t}(K))$ denote the full filtered posterior at time $t$, and let $c_t$ denote the predictive density (defined in §4.6). A *detector* is a function

$$s_t = g\!\left(\boldsymbol{\alpha}_{t|t},\, \boldsymbol{\alpha}_{t-1|t-1},\, c_t\right)$$

that produces a score, plus an alarm rule $a_t = \mathbf{1}[s_t \geq \tau]$. The three detectors implemented in Proteus differ in which inputs they use and what property of the posterior they measure:

| Detector | Score domain | Primary input | Measures |
|---|---|---|---|
| Hard switch (§4.4) | $\{0, 1\}$ | $\boldsymbol{\alpha}_{t|t}$, $\boldsymbol{\alpha}_{t-1|t-1}$ | Dominant-regime label change |
| Posterior transition (§4.5) | $[0, 1]$ | $\boldsymbol{\alpha}_{t|t}$, $\boldsymbol{\alpha}_{t-1|t-1}$ | Posterior mass shift |
| Predictive surprise (§4.6) | $[0, \infty)$ | $c_t$ | Predictive density drop |

---

## 4.2 Offline-trained, online-filtered detector architecture

### 4.2.1 Two-stage design

The implementation enforces a strict separation between training and detection:

1. **Offline stage.** The EM algorithm (Chapter 3) is run on a historical training window $y_{1:T_{\mathrm{train}}}$, producing a maximum-likelihood estimate $\hat\theta = (\hat\pi, \hat A, \hat\mu, \hat\sigma^2)$ and an `EmResult` struct.

2. **Online stage.** $\hat\theta$ is frozen into an immutable `FrozenModel`. A `StreamingSession` is opened; it receives new observations one at a time and updates only the filter state and detector state, never the parameters.

The key invariant is stated explicitly in the source:

```rust
// src/detector/frozen.rs
pub struct FrozenModel {
    params: ModelParams,
}
// params is private; no mutation method is exposed
```

```rust
pub struct StreamingSession<D: Detector> {
    model: FrozenModel,
    filter_state: OnlineFilterState,
    detector: D,
}
```

`FrozenModel` is constructed once via `FrozenModel::from_em_result(&em_result)` and is never mutated thereafter. The `ModelParams` field has no public setter. Only `filter_state` and `detector` carry mutable state across calls to `step`.

### 4.2.2 The `step` method

Each new observation $y_t$ is processed by `StreamingSession::step(y)`:

```
StreamingSession::step(y_t)
    ├── filter_state.step(y_t, model.params())   →  OnlineStepResult
    └── detector.update(&DetectorInput::from(&filter_out))  →  DetectorOutput
```

The output of `filter_state.step` is wrapped into a `DetectorInput`:

```rust
pub struct DetectorInput<'a> {
    pub filtered: &'a [f64],          // α_{t|t}
    pub predicted_next: &'a [f64],    // α_{t+1|t}
    pub predictive_density: f64,      // c_t
    pub log_predictive: f64,          // log c_t
    pub t: usize,
}
```

The detector returns:

```rust
pub struct DetectorOutput {
    pub score: f64,
    pub alarm: bool,
    pub alarm_event: Option<AlarmEvent>,
    pub t: usize,
    pub ready: bool,   // false during warmup
}
```

The `ready` flag is `false` on the first step of detectors that require a previous-step posterior (hard switch, posterior transition) or that are initialising an EMA baseline (surprise with EMA). Callers must discard outputs with `ready = false`.

### 4.2.3 Module dependencies

The online detection pipeline is fully self-contained within `src/detector/` and `src/online/`:

```
src/online/mod.rs          ← OnlineFilterState, 4-step Hamilton recursion
src/detector/frozen.rs     ← FrozenModel, StreamingSession<D>
src/detector/mod.rs        ← Detector trait, DetectorInput/Output, PersistencePolicy
src/detector/hard_switch.rs
src/detector/posterior_transition.rs
src/detector/surprise.rs
```

No import from `src/model/em.rs`, `src/model/smoother.rs`, `src/model/pairwise.rs`, or any diagnostics module is present in the online path. The only model dependency is `src/model/params.rs` (the parameter container) and `src/model/emission.rs` (Gaussian density evaluation), both of which are read-only at runtime.

---

## 4.3 Causality constraint in online detection

### 4.3.1 The filtered vs. smoothed posterior

The batch smoother (Chapter 3) computes the *smoothed* posterior:

$$\gamma_{t|T}(j) \;=\; P(S_t = j \mid y_{1:T}), \qquad t < T,$$

which uses *all* observations including those after time $t$. The smoothed posterior is more accurate for retrospective analysis but is not available in real time at time $t$.

The online filter computes the *filtered* posterior:

$$\alpha_{t|t}(j) \;=\; P(S_t = j \mid y_{1:t}),$$

using only observations up to and including $y_t$. This is causal: it is computable the instant $y_t$ arrives.

**Definition 4.1 (Causality).** A detector is *causal* if its score $s_t$ is a function of $y_{1:t}$ only, i.e. of observations that have arrived by time $t$.

All three detectors in Proteus satisfy this definition by construction: their only inputs are $\boldsymbol{\alpha}_{t|t}$, $\boldsymbol{\alpha}_{t-1|t-1}$, and $c_t$, each of which is computed from $y_{1:t}$ alone.

### 4.3.2 Implementation guarantee

The causality constraint is enforced structurally:

- `StreamingSession::step` accepts a single scalar $y_t$ and advances the filter by exactly one step. There is no batch-processing method that would allow future data to enter.
- The `OnlineFilterState` carries only the filtered vector $\boldsymbol{\alpha}_{t|t}$ and the time index $t$; it has no buffer of past observations.
- The `FrozenModel` is immutable, so the parameters cannot be updated in response to incoming data.

Neither the smoother recursion nor the pairwise pass is invoked during live detection. The comment in `src/online/mod.rs` captures this explicitly:

```rust
// This module implements the one-step-ahead Hamilton filter for online (causal)
// regime inference. It mirrors the recursion in src/model/filter.rs but operates
// on a single observation at a time and carries no dependency on the smoother,
// pairwise, or EM modules.
```

### 4.3.3 The online Hamilton recursion

For completeness, the four-step recursion executed by `OnlineFilterState::step(y_t, params)` is:

**Step 1 — Predict:**
$$\alpha_{t|t-1}(j) \;=\; \sum_{i=1}^K \hat{A}_{ij}\,\alpha_{t-1|t-1}(i)$$

**Step 2 — Emission:**
$$f_j(y_t) \;=\; \mathcal{N}(y_t;\,\hat\mu_j,\,\hat\sigma_j^2)$$

**Step 3 — Predictive density:**
$$c_t \;=\; \sum_{j=1}^K f_j(y_t)\,\alpha_{t|t-1}(j)$$

**Step 4 — Bayes update:**
$$\alpha_{t|t}(j) \;=\; \frac{f_j(y_t)\,\alpha_{t|t-1}(j)}{c_t}$$

The initial condition is $\boldsymbol{\alpha}_{0|0} = \hat\pi$ (the estimated stationary distribution). The cumulative log-score $\sum_{t=1}^T \log c_t$ is accumulated in `OnlineFilterState::cumulative_log_score` for diagnostic use; it equals the log-likelihood $\log p(y_{1:T} \mid \hat\theta)$ computed without storing the full filter pass.

---

## 4.4 Detector A: Hard switch detector

### 4.4.1 Motivation

The simplest notion of a changepoint in a two-regime model is a *label switch*: the most probable regime at time $t$ differs from the most probable regime at time $t-1$. This is the intuition behind the hard switch detector.

### 4.4.2 Score definition

Define the *dominant regime* at time $t$ as the argmax of the filtered posterior:

$$\hat{S}_t \;=\; \arg\max_j\, \alpha_{t|t}(j).$$

The hard switch score is the indicator of a dominant-regime change subject to a confidence gate:

$$s_t^{\mathrm{HS}} \;=\; \mathbf{1}\!\left[\hat{S}_t \neq \hat{S}_{t-1}\right] \cdot \mathbf{1}\!\left[\max_j \alpha_{t|t}(j) \geq \tau_{\mathrm{conf}}\right].$$

The confidence threshold $\tau_{\mathrm{conf}} \in (0, 1]$ suppresses alarms when the model is uncertain about the current regime even after the label switch. With $K = 2$ regimes, $\tau_{\mathrm{conf}} = 0.5$ corresponds to requiring that the posterior assigns at least $50\%$ probability to the new dominant regime; the natural default is $\tau_{\mathrm{conf}} = 0.6$.

### 4.4.3 Implementation

```rust
// src/detector/hard_switch.rs
pub struct HardSwitchConfig {
    pub confidence_threshold: f64,     // τ_conf
    pub persistence: PersistencePolicy,
}

pub struct HardSwitchDetector {
    config: HardSwitchConfig,
    prev_dominant: Option<usize>,      // Ŝ_{t-1}
}
```

The `update` method:

1. **Warmup:** if `prev_dominant` is `None` (first step), stores $\hat{S}_1$ and returns `ready = false`.
2. **Confidence gate:** checks $\max_j \alpha_{t|t}(j) \geq \tau_{\mathrm{conf}}$.
3. **Label comparison:** checks $\hat{S}_t \neq \hat{S}_{t-1}$.
4. **Score:** `score = 1.0` if both conditions hold, else `0.0`.
5. **Alarm policy:** passes `score ≥ 0.5` (i.e. `score == 1.0`) through `PersistencePolicy::check`.
6. **State update:** stores $\hat{S}_t$ as `prev_dominant`.

### 4.4.4 Key types

| Symbol | Type / field | Location |
|---|---|---|
| $\tau_{\mathrm{conf}}$ | `HardSwitchConfig::confidence_threshold` | `src/detector/hard_switch.rs` |
| $\hat{S}_t$ | result of `dominant_regime(filtered)` | `src/detector/mod.rs` |
| $s_t^{\mathrm{HS}}$ | `DetectorOutput::score` | `src/detector/mod.rs` |

### 4.4.5 Properties

- **Score domain:** $\{0, 1\}$ — binary per step.
- **Regime-label dependency:** yes; the score depends on which regime is labelled $j = 1$ vs. $j = 2$. If EM converges to a permuted labelling, the detector behaves identically because it only tests *change*, not identity.
- **Latency:** one step — alarm is issued at the first crossing after the regime label flips.
- **Weaknesses:** insensitive to gradual posterior drift; may trigger spuriously when the posterior oscillates near $0.5$ on consecutive steps.

---

## 4.5 Detector B: Posterior transition detector

### 4.5.1 Motivation

The hard switch converts the filtered posterior to a binary label, discarding the continuous probability mass. The posterior transition detector instead measures *how much the posterior has moved* between consecutive steps, using the full distributional information in $\boldsymbol{\alpha}_{t|t}$.

Two scores are provided, corresponding to two different geometric notions of posterior movement.

### 4.5.2 Score A — Leave-previous

Let $r_{t-1} = \hat{S}_{t-1} = \arg\max_j \alpha_{t-1|t-1}(j)$ be the previously dominant regime. The leave-previous score measures how much posterior mass has left regime $r_{t-1}$:

$$s_t^{\mathrm{LP}} \;=\; 1 \;-\; \alpha_{t|t}(r_{t-1}).$$

If the model was confident in regime $r_{t-1}$ and remains so ($\alpha_{t|t}(r_{t-1}) \approx 1$), then $s_t^{\mathrm{LP}} \approx 0$. If the posterior has shifted ($\alpha_{t|t}(r_{t-1}) \approx 0$), then $s_t^{\mathrm{LP}} \approx 1$.

**Score domain:** $[0, 1]$.

### 4.5.3 Score B — Total variation

The total variation distance between consecutive filtered posteriors is:

$$s_t^{\mathrm{TV}} \;=\; \frac{1}{2}\sum_{j=1}^K \bigl|\alpha_{t|t}(j) \;-\; \alpha_{t-1|t-1}(j)\bigr|.$$

For a probability simplex, this equals the $L^1$ distance divided by two, so $s_t^{\mathrm{TV}} \in [0, 1]$. Unlike the leave-previous score, total variation is symmetric and does not require identifying a "previous dominant" regime: it measures the overall shift in the distributional shape.

**Score domain:** $[0, 1]$.

### 4.5.4 Implementation

```rust
// src/detector/posterior_transition.rs
pub enum PosteriorTransitionScoreKind {
    LeavePrevious,
    TotalVariation,
}

pub struct PosteriorTransitionConfig {
    pub score_kind: PosteriorTransitionScoreKind,
    pub threshold: f64,
    pub persistence: PersistencePolicy,
}

pub struct PosteriorTransitionDetector {
    config: PosteriorTransitionConfig,
    prev_filtered: Option<Vec<f64>>,   // α_{t-1|t-1}
}
```

The `update` method:

1. **Warmup:** if `prev_filtered` is `None`, stores $\boldsymbol{\alpha}_{t|t}$ and returns `ready = false`.
2. **Score computation:** dispatches on `score_kind` to compute $s_t^{\mathrm{LP}}$ or $s_t^{\mathrm{TV}}$.
3. **Alarm policy:** checks `score ≥ threshold` through `PersistencePolicy::check`.
4. **State update:** replaces `prev_filtered` with the current $\boldsymbol{\alpha}_{t|t}$.

### 4.5.5 Key types

| Symbol | Type / field | Location |
|---|---|---|
| $s_t^{\mathrm{LP}}$ | `LeavePrevious` branch | `src/detector/posterior_transition.rs` |
| $s_t^{\mathrm{TV}}$ | `TotalVariation` branch | `src/detector/posterior_transition.rs` |
| Default threshold | `0.5` | `PosteriorTransitionConfig::default()` |

### 4.5.6 Properties

- **Score domain:** $[0, 1]$ (both variants).
- **Regime-label dependency:** `LeavePrevious` depends on identifying the previously dominant label; `TotalVariation` is fully label-invariant (permuting the regime labels does not change $s_t^{\mathrm{TV}}$).
- **Sensitivity:** both scores respond to *gradual* posterior drift that would not trigger the hard switch, because they measure distributional distance rather than an argmax flip.
- **Relationship to hard switch:** if the posterior moves abruptly from $(1, 0)$ to $(0, 1)$, all three detectors agree. The posterior transition detectors additionally fire on intermediate moves.

---

## 4.6 Detector C: Predictive surprise detector

### 4.6.1 Motivation

The previous two detectors are functions of the filtered posterior $\boldsymbol{\alpha}_{t|t}$, which is a *post-hoc* quantity computed after observing $y_t$. The predictive surprise detector instead uses the *one-step-ahead predictive density* $c_t$, which measures how well the model anticipated $y_t$ given $y_{1:t-1}$. A sudden drop in $c_t$ indicates that $y_t$ was surprising relative to the model's prediction — a natural signal for structural change.

### 4.6.2 Raw surprise score

The predictive density at time $t$ is the normalisation constant from Step 3 of the Hamilton recursion:

$$c_t \;=\; \sum_{j=1}^K f_j(y_t)\,\alpha_{t|t-1}(j) \;=\; p(y_t \mid y_{1:t-1},\, \hat\theta).$$

The raw surprise score is the negative log predictive density:

$$s_t^{\mathrm{surp}} \;=\; -\log c_t.$$

This is a non-negative quantity (since $c_t \leq 1$ for many practical cases) and is large when $y_t$ falls far from both mixture components. It is directly related to the per-step log-likelihood contribution: $\sum_t \log c_t = \log p(y_{1:T} \mid \hat\theta)$.

### 4.6.3 EMA-adjusted surprise score

Over long windows the baseline level of $-\log c_t$ varies with the volatility regime. To isolate *changes* in surprise rather than its absolute level, an exponential moving average (EMA) baseline is subtracted:

$$s_t^{\mathrm{adj}} \;=\; s_t^{\mathrm{surp}} \;-\; b_{t-1},$$

where the baseline is updated *after* computing the score:

$$b_t \;=\; \alpha_{\mathrm{EMA}}\cdot s_t^{\mathrm{surp}} \;+\; (1 - \alpha_{\mathrm{EMA}})\cdot b_{t-1}.$$

Using $b_{t-1}$ (not $b_t$) in the score ensures that the baseline itself is causal: it does not depend on the current observation. The parameter $\alpha_{\mathrm{EMA}} \in (0, 1)$ controls the memory of the baseline; smaller values yield a slower-moving baseline that removes longer-run drift.

**Warmup:** when EMA is enabled, the first step initialises $b_1 = s_1^{\mathrm{surp}}$ and returns `ready = false`. The adjusted score is only meaningful from step $t = 2$ onward.

### 4.6.4 Implementation

```rust
// src/detector/surprise.rs
pub struct SurpriseConfig {
    pub threshold: f64,           // τ; default 3.0
    pub ema_alpha: Option<f64>,   // α_EMA; None = raw score
    pub persistence: PersistencePolicy,
}

pub struct SurpriseDetector {
    config: SurpriseConfig,
    ema_baseline: Option<f64>,    // b_{t-1}
}
```

The `update` method:

1. **Raw score:** `score = -log_predictive` (taken directly from `DetectorInput::log_predictive`).
2. **EMA branch (if enabled):**
   - If `ema_baseline` is `None` (first step): initialise `ema_baseline = score`; return `ready = false`.
   - Else: `adjusted = score - ema_baseline`; update `ema_baseline = alpha * score + (1 - alpha) * ema_baseline`; use `adjusted` as the reported score.
3. **Alarm policy:** checks `score ≥ threshold` through `PersistencePolicy::check`.

### 4.6.5 Key types

| Symbol | Type / field | Location |
|---|---|---|
| $c_t$ | `DetectorInput::predictive_density` | `src/detector/mod.rs` |
| $-\log c_t$ | `DetectorInput::log_predictive` (negated) | `src/detector/mod.rs` |
| $b_t$ | `SurpriseDetector::ema_baseline` | `src/detector/surprise.rs` |
| $\alpha_{\mathrm{EMA}}$ | `SurpriseConfig::ema_alpha` | `src/detector/surprise.rs` |
| Default $\tau$ | `3.0` | `SurpriseConfig::default()` |

### 4.6.6 Properties

- **Score domain:** $[0, \infty)$ (raw); $(-\infty, \infty)$ (adjusted; negative when observation is less surprising than baseline).
- **Regime-label dependency:** none. $c_t$ is a marginal quantity that does not reference regime indices; permuting the regime labels leaves $c_t$ unchanged.
- **Sensitivity:** responds to volatility spikes and distribution mismatches that may not shift the regime posterior (e.g. an outlier that both regimes find improbable).
- **Relationship to log-likelihood:** $\sum_{t=1}^T s_t^{\mathrm{surp}} = -\log p(y_{1:T} \mid \hat\theta)$. Large aggregate surprise corresponds to poor model fit.

---

## 4.7 Alarm policy: thresholds, persistence, and cooldown

### 4.7.1 Threshold crossing

For every detector, the basic alarm rule is a threshold crossing:

$$a_t^{\mathrm{raw}} \;=\; \mathbf{1}[s_t \geq \tau].$$

The threshold $\tau$ is a scalar hyperparameter specific to each detector:

| Detector | Default $\tau$ | Range |
|---|---|---|
| Hard switch | $0.5$ (i.e. `score == 1.0`) | effectively $\{0, 1\}$ |
| Posterior transition | $0.5$ | $[0, 1]$ |
| Predictive surprise | $3.0$ | $[0, \infty)$ |

### 4.7.2 Persistence filter

A single threshold crossing may be a transient fluctuation rather than a genuine structural change. The `PersistencePolicy` requires $N_{\mathrm{req}}$ *consecutive* crossings before issuing an alarm:

$$a_t \;=\; \mathbf{1}\!\left[\sum_{s=t-N_{\mathrm{req}}+1}^{t} a_s^{\mathrm{raw}} = N_{\mathrm{req}}\right].$$

The running count of consecutive crossings is stored in `PersistencePolicy::consecutive_count` and reset to zero whenever a step fails to cross the threshold. The default is $N_{\mathrm{req}} = 1$ (immediate alarm on first crossing).

### 4.7.3 Cooldown period

After an alarm is issued, a cooldown of $C$ steps is enforced: no further alarms can be issued until $C$ steps have elapsed. This prevents repeated alarms during a sustained post-change transition:

$$a_t = 0 \quad \text{if } t - t^* < C,$$

where $t^*$ is the time of the most recent alarm. The cooldown is tracked in `PersistencePolicy::cooldown_remaining`, which is decremented each step and set to $C$ after each alarm. The default is $C = 0$ (no cooldown).

### 4.7.4 Implementation

```rust
// src/detector/mod.rs
pub struct PersistencePolicy {
    pub required_consecutive: usize,  // N_req; default 1
    pub cooldown: usize,              // C; default 0
    consecutive_count: usize,
    cooldown_remaining: usize,
}

impl PersistencePolicy {
    pub fn check(&mut self, threshold_crossed: bool) -> bool {
        // Cooldown active: suppress alarm and count down
        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            self.consecutive_count = 0;
            return false;
        }
        if threshold_crossed {
            self.consecutive_count += 1;
            if self.consecutive_count >= self.required_consecutive {
                self.consecutive_count = 0;
                self.cooldown_remaining = self.cooldown;
                return true;
            }
        } else {
            self.consecutive_count = 0;
        }
        false
    }
}
```

`PersistencePolicy` is embedded in every detector's configuration struct. It is the single point of alarm-issuing logic across all three detectors.

### 4.7.5 Alarm event record

When an alarm is issued, the detector constructs an `AlarmEvent`:

```rust
pub struct AlarmEvent {
    pub t: usize,
    pub score: f64,
    pub detector_kind: DetectorKind,
    pub dominant_regime_before: usize,
    pub dominant_regime_after: usize,
}
```

The `dominant_regime_before` and `dominant_regime_after` fields record the argmax of the filtered posterior at times $t-1$ and $t$ respectively, regardless of which detector issued the alarm. These fields are used by the benchmark layer (Chapter 5) to match detected events against ground-truth changepoints.

---

## 4.8 Discussion of detector semantics

### 4.8.1 What each detector actually measures

The three detectors measure fundamentally different aspects of the filtering output, and their behaviour diverges in instructive ways:

**Hard switch** measures a *discrete event*: the argmax label flipped. It is maximally interpretable — the dominant regime changed — but has resolution limited to the binary grid. In a two-regime model, the posterior must cross $0.5$ for the argmax to flip, so the hard switch is insensitive to posterior drift that does not cross that threshold. It will also not fire twice in quick succession unless the argmax alternates, making it naturally robust to sustained transitions.

**Posterior transition (LeavePrevious)** measures how much probability mass has drained from the previously dominant regime. It fires earlier than the hard switch in a gradual transition (when the posterior drifts from $0.9$ to $0.7$ to $0.5$) and can fire without the argmax ever switching (if the model is always uncertain). The leave-previous score is partially regime-label-dependent: it tracks mass in the *previously dominant* regime, not an absolute regime index.

**Posterior transition (TotalVariation)** is the most symmetric of the three: it measures the $L^1$ distance between consecutive posteriors and is invariant to permutations of the regime labels. It fires whenever the distributional shape changes, regardless of which regime is dominant. It is the natural choice when regime interpretations are unstable across EM restarts.

**Predictive surprise** operates in a different space entirely: it measures predictive fit rather than posterior movement. A high-surprise event may occur with no change in regime label (an outlier in a stable regime) or with a regime change (the model was confidently in the wrong regime). It is the only detector that is *completely* decoupled from the filtered posterior, depending only on $c_t$. This makes it more sensitive to volatility jumps and parametric misfit but also more susceptible to false alarms during high-volatility regimes that are well-modelled.

### 4.8.2 Sensitivity ordering

Under a clean hard-switch scenario (posterior moves abruptly from $(1,0)$ to $(0,1)$):

$$s_t^{\mathrm{TV}} = 1, \quad s_t^{\mathrm{LP}} = 1, \quad s_t^{\mathrm{HS}} = 1.$$

Under a gradual transition (posterior moves from $(0.9, 0.1)$ to $(0.6, 0.4)$ over several steps, never crossing $0.5$):

$$s_t^{\mathrm{TV}} > 0, \quad s_t^{\mathrm{LP}} > 0, \quad s_t^{\mathrm{HS}} = 0.$$

Under an outlier with no regime shift (posterior stays near $(1,0)$ but $y_t$ is far from $\hat\mu_1$):

$$s_t^{\mathrm{surp}} \text{ is large}, \quad s_t^{\mathrm{TV}} \approx 0, \quad s_t^{\mathrm{LP}} \approx 0, \quad s_t^{\mathrm{HS}} = 0.$$

This sensitivity hierarchy guides detector selection: hard switch for sharp, clean transitions; posterior transition for gradual drift; predictive surprise for regimes where volatility itself signals the change.

### 4.8.3 Causal validity of all three detectors

All three detectors are causal by construction (Definition 4.1). No future observation enters any score. The EMA baseline in the surprise detector uses $b_{t-1}$, not $b_t$, specifically to preserve this property. The persistence policy introduces a delay of up to $N_{\mathrm{req}}$ steps between the first threshold crossing and the alarm, but this delay is bounded and known at configuration time.

### 4.8.4 Relationship to the frozen-parameter assumption

The offline-trained, online-filtered design carries an implicit assumption: the regime distributions learned on the training window remain valid in the deployment period. If the underlying distributions shift (distribution shift / covariate shift), the filtered posterior and predictive density degrade. The surprise detector provides the earliest signal of such degradation (rising $s_t^{\mathrm{surp}}$ even without label switches), while the hard switch and posterior transition detectors may remain silent if the shifted distribution still concentrates posterior mass in one regime.

This motivates the calibration protocol described in Chapter 6, which compares synthetic-domain detector behaviour against real-data behaviour to assess whether the frozen-parameter assumption is benign in practice.

---

## Summary

| Component | Theoretical object | Implementation | Location |
|---|---|---|---|
| Offline-trained parameters | $\hat\theta$ from EM | `FrozenModel` (immutable) | `src/detector/frozen.rs` |
| Online filter | $\boldsymbol{\alpha}_{t\|t} = P(S_t \mid y_{1:t})$ | `OnlineFilterState::step` | `src/online/mod.rs` |
| Predictive density | $c_t = p(y_t \mid y_{1:t-1}, \hat\theta)$ | `DetectorInput::predictive_density` | `src/online/mod.rs` |
| Detector interface | $s_t = g(\boldsymbol{\alpha}_{t\|t}, \boldsymbol{\alpha}_{t-1\|t-1}, c_t)$ | `Detector` trait | `src/detector/mod.rs` |
| Hard switch | $s_t^{\mathrm{HS}} \in \{0,1\}$ | `HardSwitchDetector` | `src/detector/hard_switch.rs` |
| Leave-previous | $s_t^{\mathrm{LP}} = 1 - \alpha_{t\|t}(r_{t-1})$ | `PosteriorTransitionDetector` | `src/detector/posterior_transition.rs` |
| Total variation | $s_t^{\mathrm{TV}} = \tfrac{1}{2}\sum_j \|\alpha_{t\|t}(j) - \alpha_{t-1\|t-1}(j)\|$ | `PosteriorTransitionDetector` | `src/detector/posterior_transition.rs` |
| Raw surprise | $s_t^{\mathrm{surp}} = -\log c_t$ | `SurpriseDetector` | `src/detector/surprise.rs` |
| EMA-adjusted surprise | $s_t^{\mathrm{adj}} = s_t^{\mathrm{surp}} - b_{t-1}$ | `SurpriseDetector` (EMA branch) | `src/detector/surprise.rs` |
| Alarm policy | Persist $N$ crossings; suppress $C$ steps | `PersistencePolicy::check` | `src/detector/mod.rs` |
| Alarm record | $(\hat{S}_{t-1}, \hat{S}_t, s_t, t)$ | `AlarmEvent` | `src/detector/mod.rs` |
