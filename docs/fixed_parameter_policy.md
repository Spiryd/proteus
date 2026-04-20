# Fixed-Parameter Policy: Offline-Trained, Online-Filtered Detector

## Phase 13

---

## 1. The Central Design Decision

This phase locks down one of the most fundamental modeling choices of the
streaming detector:

$$
\boxed{\text{Learn parameters once offline. Detect online with frozen parameters.}}
$$

The Markov Switching Model has two structurally distinct layers:

**Layer A — structural parameters** $\widehat{\Theta}$:
$$
\widehat{\Theta} = \bigl(\widehat{\pi},\;\widehat{P},\;\widehat{\mu}_1,\dots,\widehat{\mu}_K,\;\widehat{\sigma}_1^2,\dots,\widehat{\sigma}_K^2\bigr).
$$

**Layer B — online latent-state beliefs**:
$$
\alpha_{t|t}(j) = \Pr(S_t = j \mid y_{1:t};\,\widehat{\Theta}), \quad j=1,\dots,K.
$$

The chosen design holds Layer A **constant** throughout the streaming run and
updates only Layer B. This is the **offline-trained, online-filtered
fixed-parameter detector** (abbreviated: *fixed-parameter MS detector* or
*frozen-parameter detector*).

---

## 2. Why This Is the Right First Architecture

### 2.1 Preserving the meaning of "change"

If $\widehat{\Theta}$ is fixed, then a shift in the detector output has a
clear interpretation:

> The incoming data no longer fit the regime structure learned offline.

The detector measures deviation from a stable reference. If $\widehat{\Theta}$
were allowed to adapt, a genuine structural change could be partially absorbed
into parameter updates, making the detector less sensitive and harder to
interpret.

### 2.2 Clean separation of estimation from detection

Estimation is a batch inference problem solved once on historical data.
Detection is a sequential decision problem solved causally on the live stream.
Merging them into a single adaptive loop conflates two distinct statistical
goals and makes it harder to attribute performance to either.

### 2.3 Fairer benchmarking

When comparing the MS-based detector against conventional online methods
(BOCPD, CUSUM, etc.), fixed parameters put all methods on equal footing: each
receives the same streaming observations and makes causal decisions without
access to future data. No method retrains mid-stream.

### 2.4 Interpretable attribution of performance

With frozen parameters, any variation in detector performance can be attributed
to:
- the learned regime structure (quality of the offline fit),
- the online filtering dynamics,
- the choice of score function,
- the alarm policy.

Without frozen parameters, estimation quality, adaptation speed, forgetting
behavior, and detection logic become entangled.

---

## 3. Formal Definition

### 3.1 Offline training stage

Given a training sequence $y_1^{(\text{tr})}, \dots, y_{T_0}^{(\text{tr})}$,
estimate parameters via EM:

$$
\widehat{\Theta} = \arg\max_{\Theta}\; \log p\!\bigl(y_1^{(\text{tr})},\dots,y_{T_0}^{(\text{tr})} \mid \Theta\bigr).
$$

The result $\widehat{\Theta}$ is **frozen** after this stage. Only the
parameter vector is retained for the streaming runtime; all EM history,
diagnostics, and retrospective posterior quantities are offline artifacts.

### 3.2 Online detection stage

For each new observation $y_t$ ($t = 1, 2, \dots$), the system executes the
following causal recursion under **fixed** $\widehat{\Theta}$:

**Prediction:**
$$
\alpha_{t|t-1}(j)
= \sum_{i=1}^{K} \widehat{p}_{ij}\;\alpha_{t-1|t-1}(i), \quad j=1,\dots,K.
$$

**Emission evaluation:**
$$
\widehat{f}_j(y_t)
= \mathcal{N}\!\bigl(y_t;\;\widehat{\mu}_j,\;\widehat{\sigma}_j^2\bigr), \quad j=1,\dots,K.
$$

**Predictive density:**
$$
c_t = \sum_{j=1}^{K} \widehat{f}_j(y_t)\;\alpha_{t|t-1}(j).
$$

**Bayes update:**
$$
\alpha_{t|t}(j)
= \frac{\widehat{f}_j(y_t)\;\alpha_{t|t-1}(j)}{c_t}, \quad j=1,\dots,K.
$$

At every step, the hat-quantities remain unchanged.

### 3.3 Detector layer

Given the online filter output at time $t$, the detector computes a
changepoint score $s_t$ and applies an alarm policy:

$$
s_t = \mathcal{S}\!\bigl(\alpha_{t|t},\;\alpha_{t-1|t-1},\;c_t\bigr),
$$

where $\mathcal{S}$ is one of the three score functions defined in Phase 12
(hard-switch, posterior-transition, or surprise). The alarm policy maps $s_t$
to a binary decision via a threshold and optional persistence rule.

---

## 4. What Changes Online and What Stays Fixed

| Quantity | Fixed or dynamic? | Layer |
|---|---|---|
| $\widehat{\pi}$ | **Fixed** | Parameter |
| $\widehat{P}$ | **Fixed** | Parameter |
| $\widehat{\mu}_j$ | **Fixed** | Parameter |
| $\widehat{\sigma}_j^2$ | **Fixed** | Parameter |
| $K$ | **Fixed** | Parameter |
| Alarm threshold $\tau$ | **Fixed** (pre-calibrated) | Detector config |
| $\alpha_{t\vert t}(j)$ | **Dynamic** | Online posterior |
| $\alpha_{t+1\vert t}(j)$ | **Dynamic** | Online posterior |
| $c_t$ | **Dynamic** | Online posterior |
| Detector score $s_t$ | **Dynamic** | Detector state |
| Persistence counter | **Dynamic** | Detector state |
| Cooldown counter | **Dynamic** | Detector state |
| Alarm history | **Dynamic** | Detector state |

---

## 5. What Is Intentionally Excluded

The fixed-parameter baseline **does not** support:

- online EM re-estimation of any parameter,
- sliding-window or rolling refits,
- forgetting factors applied to parameter estimates,
- adaptive regime means or variances,
- adaptive transition rows,
- detector thresholds that update from streaming data,
- any query to the smoothed posterior $\gamma_t(j)$ during live operation.

These exclusions are deliberate design constraints, not omissions. They
define the boundary of the baseline detector and establish the clean
reference point needed for later benchmarking.

---

## 6. Two-Stage System Architecture

```
┌─────────────────────────────────────┐
│         OFFLINE TRAINING STAGE      │
│                                     │
│  historical data → fit_em()         │
│                  → EmResult         │
│                  → FrozenModel      │
│                                     │
│  Artifacts retained: FrozenModel    │
│  Artifacts discarded: ll_history,   │
│    n_iter, smoothed marginals, etc. │
└──────────────┬──────────────────────┘
               │  FrozenModel (immutable)
               ▼
┌─────────────────────────────────────┐
│         ONLINE DETECTION STAGE      │
│                                     │
│  StreamingSession {                 │
│      model:        FrozenModel,     │  ← fixed
│      filter_state: OnlineFilterState│  ← mutable
│      detector:     D: Detector,     │  ← mutable
│  }                                  │
│                                     │
│  for each yₜ:                       │
│    1. filter_state.step(yₜ, params) │
│    2. detector.update(input)        │
│    3. emit Option<AlarmEvent>       │
└─────────────────────────────────────┘
```

---

## 7. Runtime Invariants

### Parameter invariants

Throughout the entire online run:

$$
\widehat{\pi},\;\widehat{P},\;\widehat{\mu}_1,\dots,\widehat{\mu}_K,\;\widehat{\sigma}_1^2,\dots,\widehat{\sigma}_K^2 \text{ are byte-identical to their values at construction.}
$$

In Rust: the `FrozenModel` exposes only `&ModelParams`, never `&mut ModelParams`.

### Posterior normalization invariants

At each step $t$:

$$
\sum_{j=1}^{K} \alpha_{t|t}(j) = 1, \qquad
\sum_{j=1}^{K} \alpha_{t+1|t}(j) = 1, \qquad
c_t > 0.
$$

These are enforced by the online filter (runtime checks with tolerance
`NORM_TOL = 1e-9`).

### Architectural invariants

- `StreamingSession::step` passes `frozen.params()` by immutable reference to
  `OnlineFilterState::step`; no parameter copy is made.
- Detector code (`src/detector/`) has no import from `model::em`,
  `model::smoother`, `model::pairwise`, or `model::diagnostics`.
- `FrozenModel` carries only `ModelParams`; offline fit history is not stored.

---

## 8. Module Structure

```
src/detector/
    mod.rs                   ← Detector trait, common types, PersistencePolicy
    frozen.rs                ← FrozenModel, StreamingSession, SessionStepOutput
    hard_switch.rs           ← HardSwitchDetector
    posterior_transition.rs  ← PosteriorTransitionDetector
    surprise.rs              ← SurpriseDetector
```

The `FrozenModel` is the boundary object between the offline and online worlds.
`StreamingSession` is the canonical entry point for live detection.

---

## 9. Conversion from Offline Fit to Runtime Object

`FrozenModel::from_em_result(&em)` is the standard conversion:

```rust
let em: EmResult = fit_em(&obs, init_params, EmConfig::default())?;
let frozen = FrozenModel::from_em_result(&em)?;
```

This discards:
- `ll_history` (EM convergence curve),
- `n_iter`,
- `converged` flag,
- any retrospective diagnostics attached to the `EmResult`.

It retains only `params: ModelParams`, validated before freezing.

---

## 10. Canonical Streaming Workflow

```rust
// --- Stage 1: train offline ---
let em     = fit_em(&training_obs, init_params, EmConfig::default())?;
let frozen = FrozenModel::from_em_result(&em)?;

// --- Stage 2: detect online ---
let filter  = OnlineFilterState::new(frozen.params());
let det     = HardSwitchDetector::new(HardSwitchConfig {
    confidence_threshold: 0.7,
    persistence: PersistencePolicy::new(2, 10),
});
let mut session = StreamingSession::new(frozen, filter, det);

for y in live_stream {
    let out = session.step(y)?;
    if out.detector.alarm {
        let ev = out.detector.alarm_event.unwrap();
        println!("alarm at t={}, score={:.4}", ev.t, ev.score);
    }
}
```

---

## 11. Thesis Naming

Use one of these names consistently across code, documentation, plots, and
benchmarks to make the design choice visible to readers:

- **Fixed-parameter MS detector**
- **Offline-trained / online-filtered MS detector**
- **Frozen-parameter regime-switching detector**

The choice between them is a matter of taste; the important thing is
consistency.

---

## 12. Tests

| Test | Invariant verified |
|---|---|
| `frozen_model_new_accepts_valid_params` | `FrozenModel::new` validates and wraps |
| `frozen_model_from_em_result_discards_history` | Conversion keeps only `params` |
| `frozen_model_params_is_immutable_reference` | Public API is `&ModelParams` only |
| `streaming_session_step_produces_output` | One step yields `t=1`, normalized posterior |
| `streaming_session_step_batch_length_matches` | Batch output length = input length |
| `streaming_session_params_unchanged_after_steps` | Transition matrix byte-identical after 50 steps |
| `streaming_session_reset_restores_initial_state` | `reset()` sets `t=0`, no param mutation |
| `streaming_session_surprise_detector_composes` | `SurpriseDetector` composes cleanly |

---

## 13. What This Phase Does Not Do

- Threshold calibration (choosing $\tau$ to achieve a target false-alarm rate).
- Benchmark protocol and comparison against BOCPD / CUSUM.
- Adaptive parameter updates (reserved for later extensions).
- Experimental evaluation and detection-delay metrics.

This phase establishes the **fixed-parameter baseline architecture** that
those later phases build upon.
