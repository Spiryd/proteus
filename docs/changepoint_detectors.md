# Changepoint Detection in the Gaussian Markov Switching Model

## Phase 12 — Detector Family

---

## 1. Why Latent Regime Inference Is Not Changepoint Detection

The online filter computes, at each time $t$, the filtered posterior over
the latent regime:

$$
\alpha_{t|t}(j) = \Pr(S_t = j \mid y_{1:t}), \quad j = 1, \dots, K.
$$

This is a **probabilistic belief state**: a soft, continuously updating
distribution over which regime the system is currently in. It is not a
decision. The filter knows the model is in one of $K$ regimes at all times;
regime transitions are expected behavior, governed by the Markov transition
matrix $P$. Nothing in the filter directly answers the question:

> At time $t$, has a meaningful change occurred that warrants an alarm?

A **changepoint detector** must additionally define:

1. a **score** $s_t$ measuring how much evidence of change exists at time $t$,
2. a **decision rule** specifying when $s_t$ is large enough to trigger an alarm,
3. optionally, a **persistence policy** stabilizing alarms against transient
   score spikes.

This separation is the central architecture of Phase 12:

$$
\underbrace{\text{Online Filter}}_{\text{probabilistic inference}}
\;\longrightarrow\;
\underbrace{s_t}_{\text{score}}
\;\longrightarrow\;
\underbrace{\text{alarm rule}}_{\text{decision}}
\;\longrightarrow\;
\underbrace{\text{AlarmEvent}}_{\text{output}}.
$$

The filter and the detector are not the same layer.

---

## 2. Causal Boundary

All quantities available to the detector at time $t$ are restricted to the
causal set:

$$
\mathcal{C}_t = \bigl\{ y_1, \dots, y_t, \;\alpha_{1|1}, \dots, \alpha_{t|t}, \;\alpha_{2|1}, \dots, \alpha_{t+1|t}, \;c_1, \dots, c_t \bigr\}.
$$

**Permitted at time $t$:**

| Quantity | Notation | Description |
|---|---|---|
| Filtered posterior | $\alpha_{t\vert t}(j)$ | $\Pr(S_t=j \mid y_{1:t})$ |
| Predicted posterior | $\alpha_{t\vert t-1}(j)$ | $\Pr(S_t=j \mid y_{1:t-1})$ |
| One-step-ahead prediction | $\alpha_{t+1\vert t}(j)$ | $\Pr(S_{t+1}=j \mid y_{1:t})$ |
| Predictive density | $c_t$ | $f(y_t \mid y_{1:t-1})$ |
| Log predictive density | $\log c_t$ | |

**Forbidden at time $t$ (offline-only):**

| Quantity | Why forbidden |
|---|---|
| Smoothed marginals $\gamma_t(j) = \Pr(S_t = j \mid y_{1:T})$ | Require future observations |
| Pairwise posteriors $\xi_t(i,j)$ (smoothed) | Require backward pass |
| EM parameter estimates (online update) | Require full-sample E-step |

The detector module (`src/detector/`) has no dependency on
`model::smoother`, `model::pairwise`, `model::em`, or `model::diagnostics`.

---

## 3. Dominant Regime

Let the **dominant regime** at time $t$ be:

$$
\hat{S}_t = \arg\max_{j \in \{1,\dots,K\}} \alpha_{t|t}(j).
$$

Ties are broken by the lower index. This is the maximum-a-posteriori (MAP)
regime estimate at time $t$ given all observations up to and including $y_t$.

---

## 4. Three Detector Variants

The three detector variants represent distinct mathematical answers to the
question *"what does 'change' mean in a Markov Switching Model?"* They
share the same online filtering backbone but define different score
functions and carry different semantics.

---

### 4.1 Hard Switch Detector

**Definition.** The hard-switch score is the indicator that the dominant
regime has changed:

$$
s_t^{\text{hard}} =
\begin{cases}
1 & \text{if } \hat{S}_t \neq \hat{S}_{t-1} \text{ and } \max_j \alpha_{t|t}(j) \geq \tau_{\text{conf}} \\
0 & \text{otherwise,}
\end{cases}
$$

where $\tau_{\text{conf}} \geq 0$ is an optional confidence gate. Setting
$\tau_{\text{conf}} = 0$ disables the gate.

An alarm fires when $s_t^{\text{hard}} = 1$ and the persistence policy
approves.

**Semantics.** This detector defines change as a discrete switch in the
MAP regime estimate. It is the simplest possible regime-change rule and
acts as a natural baseline.

**Strengths.** Directly interpretable; closely tied to the latent regime
story; easy to visualize.

**Weaknesses.** Ignores posterior uncertainty; sensitive to posterior
oscillations near boundaries between regimes; binary score gives no
continuous gradation of evidence.

**Confidence gate.** Requiring $\max_j \alpha_{t|t}(j) \geq \tau_{\text{conf}}$
suppresses alarms when the posterior is too diffuse, preventing spurious
switches from near-ties. A typical value is $\tau_{\text{conf}} = 0.6$–$0.8$.

**Warmup.** No alarm at $t=1$; no previous dominant regime is available.

---

### 4.2 Posterior Transition Detector

Two continuous score variants are available.

#### 4.2.1 Leave-Previous-Regime Score

Let $r_{t-1} = \hat{S}_{t-1} = \arg\max_j \alpha_{t-1|t-1}(j)$.

$$
s_t^{\text{leave}} = 1 - \alpha_{t|t}(r_{t-1}) \;\in [0, 1].
$$

Large when the current posterior has little mass on the previously dominant
regime. Zero only when $\alpha_{t|t}(r_{t-1}) = 1$ (certain presence in the
old regime). Equals $1 - \alpha_{t|t}(r_{t-1})$, which may be nonzero even
if the dominant regime has not switched — capturing gradual mass migration.

#### 4.2.2 Total-Variation (Posterior-Shift) Score

$$
s_t^{\text{TV}} = \frac{1}{2} \sum_{j=1}^{K}
\bigl| \alpha_{t|t}(j) - \alpha_{t-1|t-1}(j) \bigr| \;\in [0, 1].
$$

This is the total variation distance between consecutive filtered
posteriors. It equals $0$ iff the two posteriors are identical, and equals
$1$ iff they are supported on disjoint regimes (maximal shift). Unlike
$s_t^{\text{leave}}$, it is symmetric and captures movement in the full
$K$-dimensional distribution.

**Semantics.** The posterior transition detector defines change as the
posterior probability mass migrating away from the previously dominant
regime (or shifting across the full distribution). It is more stable than
the hard-switch rule because the score is continuous — small oscillations
produce small scores rather than binary on/off signals.

**Strengths.** Uses posterior uncertainty continuously; sensitive to
gradual structural drift; both score variants lie in $[0,1]$ and are
directly thresholdable.

**Weaknesses.** May respond slowly if the Markov chain has strong
persistence (high diagonal entries in $P$); may remain elevated for
multiple steps after a single true change.

**Warmup.** First step stores $\alpha_{1|1}$; no score at $t=1$.

---

### 4.3 Surprise Detector

**Definition.** The raw surprise score is the negative log predictive
density:

$$
s_t^{\text{surp}} = -\log c_t = -\log f(y_t \mid y_{1:t-1}).
$$

Large values indicate that the new observation was highly unexpected under
the model's one-step mixture prediction. The predictive density $c_t$ is
already computed by the online filter at each step; no additional inference
is required.

**Baseline-adjusted variant.** To normalize against the model's own recent
predictive track record, define an exponentially weighted moving average
(EWM) baseline:

$$
b_1 = s_1^{\text{surp}}, \qquad
b_t = \alpha \cdot s_t^{\text{surp}} + (1 - \alpha) \cdot b_{t-1}, \quad \alpha \in (0,1].
$$

The adjusted score uses the **previous** baseline (causal — $b_{t-1}$ is
known before $y_t$ arrives):

$$
s_t^{\text{adj}} = s_t^{\text{surp}} - b_{t-1}.
$$

This removes systematic surprise in genuinely noisy regimes and focuses
alarms on abnormal prediction failures relative to recent history.

**Semantics.** The surprise detector captures a **different notion of
change** from the two posterior-migration detectors. It fires when the
model's predictive structure has become incompatible with the current
observation — potentially *before* the filtered posterior has fully
migrated to a new regime. This is especially useful for detecting abrupt
structural breaks.

**Strengths.** Directly tied to model-prediction mismatch; can detect
changes early; intuitive connection to sequential hypothesis testing.

**Weaknesses.** Can react strongly to isolated outliers even without any
structural change; globally noisy regimes with high intrinsic variance
produce elevated baseline surprise, so the baseline-adjusted variant is
usually preferred in practice.

**Warmup (with EWM baseline).** First step initializes $b_1$ and produces
`ready = false`. Adjusted scores are available from $t = 2$ onward.

---

## 5. Persistence Policy

The **persistence policy** is a reusable alarm-stabilization mechanism
shared across all three detector variants. It prevents isolated single-step
score spikes from generating alarms.

**Definition.** Let $\mathbf{1}_t = \mathbf{1}\{s_t \geq \tau\}$ be the
threshold-crossing indicator at step $t$. An alarm fires at time $t$ iff:

$$
\mathbf{1}_{t - m + 1} = \mathbf{1}_{t - m + 2} = \cdots = \mathbf{1}_t = 1,
$$

where $m$ is the `required_consecutive` parameter. A miss ($\mathbf{1}_t = 0$)
resets the consecutive counter to zero.

**Cooldown.** After each alarm, a cooldown of $c$ steps suppresses further
alarms and resets the consecutive counter. No additional alarm can fire
until the cooldown expires.

**Parameters:**

| Parameter | Meaning |
|---|---|
| `required_consecutive` ($m \geq 1$) | Crossings needed before alarm |
| `cooldown` ($c \geq 0$) | Steps of suppression after alarm |

**Defaults.** $m = 1$ (immediate alarm), $c = 0$ (no cooldown). These
defaults make the policy transparent when a single threshold crossing
is already sufficient evidence.

The policy is identical for all three detectors, allowing fair comparison
of the detectors' score functions in isolation of the alarm rule.

---

## 6. Detector Comparison

| Detector | Score range | Change semantics | Requires prev. state |
|---|---|---|---|
| Hard Switch | $\{0, 1\}$ | MAP regime label switched | prev dominant regime |
| Posterior Transition (Leave) | $[0, 1]$ | Posterior mass left old regime | prev filtered posterior |
| Posterior Transition (TV) | $[0, 1]$ | Full posterior shifted | prev filtered posterior |
| Surprise (raw) | $[0, +\infty)$ | Observation surprised model | none (ready at $t=1$) |
| Surprise (EWM-adjusted) | $(-\infty, +\infty)$ | Surprise above recent average | EWM baseline (warmup 1 step) |

These three detectors represent a theoretically meaningful basis for
comparison in a thesis because they capture distinct notions of change:

1. **Hard Switch** — change in the MAP label (discrete, regime-identity).
2. **Posterior Transition** — change in the probabilistic belief state
   (continuous, magnitude of migration).
3. **Surprise** — change in model-predictive compatibility (continuous,
   observation-level mismatch).

They differ in sensitivity, in delay between true change and alarm, and in
their false-alarm behavior under different regimes. This motivates the
comparative experimental design of later phases.

---

## 7. Alarm Event

Every alarm, regardless of which detector produced it, is represented as an
`AlarmEvent` containing:

| Field | Type | Description |
|---|---|---|
| `t` | `usize` | Time index at which the alarm fires |
| `score` | `f64` | Score value that triggered the alarm |
| `detector_kind` | `DetectorKind` | Which variant produced the alarm |
| `dominant_regime_before` | `Option<usize>` | $\hat{S}_{t-1}$ if tracked |
| `dominant_regime_after` | `usize` | $\hat{S}_t$ at the alarm step |

The `dominant_regime_before` field is `None` for the surprise detector
(which does not track regime transitions) and `Some(j)` for the hard-switch
and posterior-transition detectors.

---

## 8. Runtime Invariants

### Hard Switch Detector
- `dominant_regime_after` must be a valid index in $\{0, \dots, K-1\}$.
- `score` $\in \{0.0, 1.0\}$.
- No alarm at $t=1$ (warmup); `ready = false`.

### Posterior Transition Detector
- `score` $\in [0, 1]$ for both `LeavePrevious` and `TotalVariation`.
- `TotalVariation` score is exactly $0.0$ when consecutive posteriors are identical.
- No alarm at $t=1$ (warmup); `ready = false`.

### Surprise Detector (raw)
- `score = -log_predictive ≥ 0.0` when `log_predictive ≤ 0` (i.e. $c_t \leq 1$,
  which holds when the predictive density is a proper probability density).
- `ready = true` from $t = 1$ when `ema_alpha = None`.

### Surprise Detector (EWM-adjusted)
- `ready = false` at $t=1$ (baseline initialization).
- EWM baseline is always finite and updated causally.

### Shared
- `alarm_event.is_some()` iff `alarm = true`.
- Alarm timestamps are nondecreasing.
- Cooldown and consecutive counters remain non-negative.
- No detector accesses future data.

---

## 9. Module Structure

```
src/detector/
    mod.rs                    ← Detector trait, common types, PersistencePolicy
    hard_switch.rs            ← HardSwitchDetector
    posterior_transition.rs   ← PosteriorTransitionDetector
    surprise.rs               ← SurpriseDetector
```

The `detector` module depends only on `crate::online::OnlineStepResult`
(for the `From<OnlineStepResult>` conversion on `DetectorInput`). It has
no dependency on `model::smoother`, `model::pairwise`, `model::em`, or
`model::diagnostics`.

```
Offline stack (src/model/):
  params → emission → filter → smoother → pairwise → em → diagnostics

Online stack (src/online/):
  OnlineFilterState → step → OnlineStepResult

Detector stack (src/detector/):
  OnlineStepResult → DetectorInput → Detector::update → DetectorOutput / AlarmEvent
```

---

## 10. Usage Pattern

```rust
use crate::online::OnlineFilterState;
use crate::detector::{DetectorInput, HardSwitchDetector, Detector};

let mut state = OnlineFilterState::new(&fitted_params);
let mut det   = HardSwitchDetector::default();

for y in stream {
    let step  = state.step(y, &fitted_params)?;
    let input = DetectorInput::from(&step);
    let out   = det.update(&input);

    if out.alarm {
        let ev = out.alarm_event.unwrap();
        // ev.t, ev.score, ev.dominant_regime_before, ev.dominant_regime_after
    }
}
```

To apply a persistence policy, pass a `PersistencePolicy` inside the
detector config:

```rust
use crate::detector::{HardSwitchConfig, PersistencePolicy};

let config = HardSwitchConfig {
    confidence_threshold: 0.7,
    persistence: PersistencePolicy::new(3, 5), // 3 consecutive, 5-step cooldown
};
let mut det = HardSwitchDetector::new(config);
```

To use the baseline-adjusted surprise detector:

```rust
use crate::detector::{SurpriseConfig, SurpriseDetector};

let config = SurpriseConfig {
    threshold: 2.0,
    ema_alpha: Some(0.1),       // slow-moving baseline
    persistence: PersistencePolicy::new(2, 10),
};
let mut det = SurpriseDetector::new(config);
```

---

## 11. What This Phase Does Not Do

Phase 12 defines the detector family and alarm logic. It deliberately
defers:

- **Threshold calibration** — choosing $\tau$ to achieve a target false-alarm
  rate (requires empirical or analytical calibration, e.g. via training-data
  quantiles).
- **Benchmark protocol** — defining the ground-truth changepoint set and
  detection-delay metrics needed for comparison with BOCPD, CUSUM, etc.
- **Adaptive parameter updates** — re-estimating $\mu_j$, $\sigma_j^2$, or
  $P$ online as the data stream evolves.
- **Final experimental evaluation** — comparison studies and ROC-style
  analysis.

These are left to later phases. The detector family defined here is the
prerequisite for all of them.

---

## 12. Tests

The test suite covers:

| Module | Test | Invariant checked |
|---|---|---|
| `mod` | `persistence_default_fires_immediately` | $m=1, c=0$: alarm on first crossing |
| `mod` | `persistence_requires_consecutive_crossings` | $m=3$: fires only after 3 crossings |
| `mod` | `persistence_miss_resets_counter` | Any miss resets counter to 0 |
| `mod` | `persistence_cooldown_suppresses_refire` | $c=2$: 2 suppressed steps then re-fires |
| `mod` | `persistence_reset_clears_state` | `reset()` restores initial state |
| `mod` | `detector_input_from_online_step_result` | `From<OnlineStepResult>` is correct |
| `hard_switch` | `not_ready_on_first_step` | `ready=false`, `alarm=false` at $t=1$ |
| `hard_switch` | `alarm_when_dominant_changes` | Alarm fires; event fields correct |
| `hard_switch` | `no_alarm_when_regime_stays_same` | No alarm; `score=0` |
| `hard_switch` | `confidence_gate_suppresses_uncertain_switch` | Gate with $\tau_{\text{conf}}=0.8$ |
| `hard_switch` | `persistence_requires_consecutive_switches` | $m=2$ needs 2 switches |
| `posterior_transition` | `not_ready_on_first_step` | Warmup at $t=1$ |
| `posterior_transition` | `leave_score_correct_value` | $s = 1 - \alpha_{t\|t}(r_{t-1})$ |
| `posterior_transition` | `leave_score_in_unit_interval` | Score $\in [0,1]$ |
| `posterior_transition` | `tv_score_extremes` | TV=1 for antipodal posteriors |
| `posterior_transition` | `tv_score_zero_when_identical` | TV=0 for identical posteriors |
| `posterior_transition` | `alarm_when_score_exceeds_threshold` | Threshold logic correct |
| `surprise` | `score_equals_neg_log_predictive` | $s = -\log c_t$ |
| `surprise` | `no_alarm_below_threshold` | No alarm when $s < \tau$ |
| `surprise` | `alarm_above_threshold` | Alarm when $s \geq \tau$ |
| `surprise` | `ema_not_ready_on_first_step` | Warmup when EWM baseline enabled |
| `surprise` | `ema_adjusted_score_and_alarm` | Adjusted score $= s_t - b_{t-1}$ |

Total: **22 tests** in the `detector` module.

---

## 13. References

1. Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.
2. Kim, C.-J. (1994). Dynamic linear models with Markov-switching. *Journal of Econometrics*, 60(1–2), 1–22.
3. Adams, R. P., & MacKay, D. J. C. (2007). Bayesian online changepoint detection. *arXiv:0710.3742*.
4. Fearnhead, P., & Liu, Z. (2007). On-line inference for multiple changepoint problems. *Journal of the Royal Statistical Society: Series B*, 69(4), 589–605.
