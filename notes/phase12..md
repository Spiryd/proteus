# Phase 12 — Define What Counts as a Changepoint in a Markov-Switching Model

## Goal

In this phase, turn the online Markov Switching inference layer into an actual **online changepoint detection mechanism**.

By the end of Phase 11, you already have a causal streaming backbone:

- fixed fitted parameters,
- one-step online filtering,
- current filtered regime probabilities
  \[
  \alpha_{t|t}(j)=\Pr(S_t=j \mid y_{1:t}),
  \]
- one-step predicted probabilities,
- predictive density
  \[
  c_t=f(y_t \mid y_{1:t-1}).
  \]

But this still does **not** define a changepoint detector.

A Markov Switching Model tells you how likely each latent regime is.  
A changepoint detector must additionally answer:

> At time \(t\), should I declare that a meaningful change has occurred?

That is a different problem.

So this phase is about defining the **decision layer** that sits on top of the online regime posterior.

---

## 1. Why this phase is necessary

A latent regime model and a changepoint detector are not the same thing.

The online filter gives you:

- a probabilistic belief over regimes,
- dynamic predictions,
- observation-level surprise information.

But it does **not automatically** tell you:

- when a change should be declared,
- how much evidence is enough,
- whether a brief posterior fluctuation is noise or a true shift,
- how to avoid excessive false alarms.

That means the online Markov Switching Model needs a second layer:

\[
\text{online inference} \quad \longrightarrow \quad \text{changepoint score} \quad \longrightarrow \quad \text{alarm rule}.
\]

This phase defines that layer.

---

## 2. Why you should pursue multiple approaches

You said you want **2 or 3 approaches** to compare, learn from, and diversify your thesis.

That is exactly the right move.

A Markov Switching backbone admits several distinct changepoint definitions, and they are not equivalent.  
Comparing them is valuable because they emphasize different notions of “change”:

- **regime identity change,**
- **transition probability into a new regime,**
- **observation surprise or instability,**
- **persistent posterior migration into a new regime.**

These correspond to different algorithmic choices and different theoretical interpretations.

So in this phase, you should not commit to only one detector.  
Instead, you should define a **family of online detection rules** built on top of the same streaming regime model.

That gives you:

- one shared probabilistic backbone,
- multiple detector variants,
- a more interesting and defensible thesis comparison.

---

## 3. The core conceptual distinction

The online Markov Switching filter computes:

\[
\alpha_{t|t}(j)=\Pr(S_t=j \mid y_{1:t}).
\]

This is a belief state.

A changepoint detector requires:

- a **score**
  \[
  s_t,
  \]
- and a **decision rule**
  \[
  \text{alarm at time } t \iff s_t \ge \tau
  \]
  or some more structured equivalent.

So the problem splits into two parts.

## Part A — Score design
Define what signal should indicate a change.

## Part B — Alarm policy
Define when that signal becomes strong enough to trigger a changepoint declaration.

This separation should be explicit in both theory and code.

---

## 4. The three recommended approaches

For your thesis, I recommend building **three detector variants** on top of the same online filtering backbone.

They should be treated as three different answers to the question:

> What does “change” mean in a Markov Switching Model?

---

# Approach 1 — Hard Switch Event Detector

## Definition

Declare a changepoint when the most probable regime changes:

\[
\hat{S}_t = \arg\max_j \alpha_{t|t}(j),
\]

and then raise an alarm if

\[
\hat{S}_t \neq \hat{S}_{t-1}.
\]

This is the simplest interpretation of a changepoint:
- the posterior modal regime at time \(t\) is different from the one at time \(t-1\).

---

## Interpretation

This detector says:

> A changepoint occurs when the model’s best regime explanation switches.

This is the most intuitive regime-based rule and the easiest to explain.

---

## Strengths

- very simple,
- easy to visualize,
- directly tied to latent regime interpretation,
- easy to implement and debug.

---

## Weaknesses

- extremely sensitive to small posterior fluctuations,
- may produce noisy alarm sequences,
- ignores uncertainty magnitude,
- can fire even when the two top regimes are nearly tied.

So by itself, this detector is often too brittle unless paired with a persistence or confidence condition.

---

## Better refined form

Instead of using only

\[
\hat{S}_t \neq \hat{S}_{t-1},
\]

require also that the winning regime has sufficient confidence:

\[
\max_j \alpha_{t|t}(j) \ge \tau_{\text{conf}}.
\]

This avoids declaring a changepoint when the posterior is too diffuse.

---

## Deliverables for Approach 1

### Mathematical deliverables
- formal definition of the hard-switch score,
- optional confidence threshold,
- optional cooldown / refractory rule.

### Code changes
Add:
- a `HardSwitchDetector`,
- posterior-mode extraction logic,
- optional confidence threshold configuration,
- optional refractory-period support.

---

# Approach 2 — Posterior Transition Alarm

## Definition

Instead of looking only at the winning regime label, define the change score using posterior mass assigned to moving away from the previous dominant regime.

A simple version is:

1. identify the previously dominant regime
   \[
   r_{t-1} = \arg\max_j \alpha_{t-1|t-1}(j),
   \]
2. define the change score as posterior probability of **not** being in that same regime at time \(t\):
   \[
   s_t^{\text{trans}} = 1 - \alpha_{t|t}(r_{t-1}).
   \]

A more structured version uses the predicted or filtered transition logic and measures mass shifting into alternative regimes.

---

## Interpretation

This detector says:

> A changepoint occurs when the model becomes confident that the system has left the previously dominant regime.

This is more stable than the hard-switch rule because it uses a continuous posterior-change signal rather than only discrete label flipping.

---

## Stronger transition-style score

Another useful version is:

\[
s_t^{\text{new}} = \sum_{j \neq r_{t-1}} \alpha_{t|t}(j).
\]

This is mathematically equivalent to the previous expression, but conceptually emphasizes “posterior mass on new regimes.”

A still richer variant compares filtered probabilities over time:

\[
s_t^{\Delta} = \frac{1}{2}\sum_{j=1}^K \left|\alpha_{t|t}(j)-\alpha_{t-1|t-1}(j)\right|.
\]

This total-variation-style shift score measures how much the full posterior moved between consecutive time steps.

That gives you two closely related transition-style detectors:

- **leave-previous-regime score,**
- **posterior-shift magnitude score.**

You can treat them as subvariants if useful.

---

## Strengths

- uses posterior uncertainty continuously,
- less brittle than hard label switching,
- naturally tied to regime migration,
- easy to threshold and compare.

---

## Weaknesses

- still depends strongly on posterior dynamics,
- may react slowly if persistence is strong,
- may remain elevated for multiple steps after one true change.

So it often benefits from alarm suppression logic or persistence rules.

---

## Deliverables for Approach 2

### Mathematical deliverables
- formal definition of the posterior-transition score,
- optional alternative definition via posterior shift magnitude,
- threshold rule for alarming,
- optional hysteresis / cooldown logic.

### Code changes
Add:
- a `PosteriorTransitionDetector`,
- access to previous filtered posterior,
- score functions for:
  - leaving previous dominant regime,
  - posterior shift magnitude,
- threshold and suppression configuration.

---

# Approach 3 — Surprise / Instability Detector

## Definition

Use the predictive density from the online filter:

\[
c_t = f(y_t \mid y_{1:t-1}).
\]

Then define a change score as the negative log predictive density:

\[
s_t^{\text{surp}} = -\log c_t.
\]

Large values mean that the new observation was highly surprising under the model’s one-step prediction.

---

## Interpretation

This detector says:

> A changepoint occurs when the incoming observation becomes unexpectedly incompatible with what the model predicted from the recent regime dynamics.

This is less about “which regime changed” and more about “the current data no longer fit the existing predictive structure.”

---

## Why this is useful

This gives you a conceptually different type of detector:

- the first two approaches are **posterior regime-change detectors**,
- this one is a **predictive surprise detector**.

That makes it especially valuable in your thesis because it diversifies the comparison while staying within the same backbone model.

---

## Stronger practical form

Instead of thresholding raw surprise directly, define a relative instability score using a local baseline:

\[
z_t^{\text{surp}} = s_t^{\text{surp}} - \text{baseline surprise}.
\]

For example, the baseline could be:
- recent moving average of surprise,
- exponential moving average,
- or calibration quantile from training data.

This helps distinguish:
- globally noisy regimes,
- from genuinely abnormal jumps in surprise.

---

## Strengths

- directly uses predictive model mismatch,
- conceptually close to conventional sequential surprise detection,
- may detect changes even before regime probabilities have fully migrated.

---

## Weaknesses

- can react strongly to isolated outliers,
- may confuse noise bursts with structural change,
- needs careful calibration.

So this detector often benefits from persistence logic or robust thresholding.

---

## Deliverables for Approach 3

### Mathematical deliverables
- formal definition of surprise score,
- optional baseline-adjusted score,
- threshold / persistence rule.

### Code changes
Add:
- a `SurpriseDetector`,
- access to predictive density from online inference,
- log-score computation,
- optional rolling / exponentially weighted baseline tracker,
- thresholding logic.

---

## 5. Optional Approach 4 — Persistent-Change Rule

This should not necessarily be a standalone detector.  
It works especially well as a **meta-rule** layered on top of Approaches 1–3.

## Definition

Do not raise an alarm immediately when a score crosses threshold.  
Require that the signal remain strong for \(m\) consecutive steps.

Example:

\[
\text{alarm at } t
\iff
s_{t-m+1},\dots,s_t \text{ all exceed threshold.}
\]

Or in posterior language, require the new regime posterior to remain high:

\[
\alpha_{u|u}(j_{\text{new}}) \ge \tau
\quad \text{for } u=t-m+1,\dots,t.
\]

---

## Interpretation

This rule says:

> A changepoint should only be declared when the evidence is sustained, not transient.

This is extremely useful in online settings because it reduces false alarms caused by:
- noisy observations,
- single-step outliers,
- short-lived posterior oscillations.

---

## Deliverables for Persistent-Change Rule

### Mathematical deliverables
- a persistence parameter \(m\),
- a sustained-evidence alarm definition,
- optional reset rule.

### Code changes
Add:
- a generic `PersistencePolicy`,
- step counters or score buffers,
- integration with all detector variants.

This is best implemented as reusable detector infrastructure rather than detector-specific code.

---

## 6. Recommended thesis structure for the detector family

For your thesis, the cleanest detector comparison built on the same Markov Switching backbone is:

## Detector A
**Hard Switch Detector**
- simplest regime-change interpretation.

## Detector B
**Posterior Transition Detector**
- continuous posterior-mass migration interpretation.

## Detector C
**Predictive Surprise Detector**
- model-mismatch interpretation.

Then optionally apply a shared **Persistence Policy** to each one.

This gives you:

- one common online filtering backbone,
- three clearly distinct definitions of change,
- a natural comparative study,
- and a stronger thesis than using only one rule.

---

## 7. Unified detector architecture

This phase should establish a general detection abstraction.

The online filtering backbone should produce a causal update object.  
On top of that, detector variants should compute scores and optionally emit alarms.

So architecturally:

\[
\text{Online Filter}
\longrightarrow
\text{Detector Score}
\longrightarrow
\text{Alarm Policy}
\longrightarrow
\text{Alarm Event}
\]

This separation is critical.

---

## 8. Step-by-step guide for this phase

## Step 1 — Define the detector boundary formally

Decide that the online filter is **not** the detector itself.

The filter provides:
- filtered regime posterior,
- predicted posterior,
- predictive density,
- current time index.

The detector consumes that output and produces:
- a changepoint score,
- optional alarm decision,
- optional detector state update.

### Deliverable
A formal separation between inference and decision.

### Code changes
Add a dedicated `detector` layer separate from:
- `online` filtering,
- `offline` training,
- `diagnostics`.

This is the main architectural change of the phase.

---

## Step 2 — Define a common detector interface

All detector variants should share a common interface.

Conceptually, each detector should:
- receive the latest online filter update,
- maintain its own internal state if needed,
- compute a score,
- decide whether to emit an alarm.

### Deliverable
A detector abstraction that supports multiple detector variants.

### Code changes
Add:
- a common detector trait / interface,
- a detector input type wrapping online filter outputs,
- a detector output type containing score and optional alarm.

This is important for fair comparison later.

---

## Step 3 — Implement Approach 1: hard switch detector

Define:
- previous dominant regime,
- current dominant regime,
- optional posterior confidence threshold.

Alarm if:
- dominant regime changed,
- and optional confidence conditions hold.

### Deliverable
A baseline discrete regime-switch detector.

### Code changes
Add:
- `HardSwitchDetector`,
- dominant-regime extraction logic,
- optional confidence threshold config,
- optional refractory rule.

---

## Step 4 — Implement Approach 2: posterior transition detector

Define one or both continuous scores:

### Leave-previous-regime score
\[
s_t = 1 - \alpha_{t|t}(r_{t-1})
\]

### Posterior-shift score
\[
s_t = \frac{1}{2}\sum_{j=1}^K |\alpha_{t|t}(j)-\alpha_{t-1|t-1}(j)|
\]

Alarm when score exceeds threshold.

### Deliverable
A soft posterior-shift detector.

### Code changes
Add:
- `PosteriorTransitionDetector`,
- storage of previous filtered posterior,
- threshold config,
- optional hysteresis or cooldown support.

---

## Step 5 — Implement Approach 3: surprise detector

Define:
\[
s_t = -\log c_t
\]

Optionally normalize using a rolling baseline.

Alarm when surprise exceeds threshold.

### Deliverable
A predictive-instability detector.

### Code changes
Add:
- `SurpriseDetector`,
- predictive-density consumption from online filter output,
- log-score computation,
- optional rolling baseline tracker,
- threshold config.

---

## Step 6 — Add a reusable persistence policy

Instead of duplicating sustained-evidence logic inside every detector, define a reusable policy layer.

It should support:
- consecutive-threshold crossing requirement,
- optional cooldown after alarm,
- optional minimum separation between alarms.

### Deliverable
A general alarm-stabilization mechanism.

### Code changes
Add:
- `PersistencePolicy`,
- `CooldownPolicy` or equivalent,
- detector-composition support.

This is one of the most useful code deliverables of the phase.

---

## Step 7 — Define the alarm event object

A real detector needs a standard alarm representation.

At minimum, an alarm should contain:
- timestamp \(t\),
- detector identifier,
- score value,
- optional confidence,
- optional dominant-regime transition summary.

### Deliverable
A unified alarm event representation.

### Code changes
Add:
- `AlarmEvent`,
- optional alarm metadata fields,
- common serialization / reporting support if needed for experiments.

This will matter later when benchmarking against conventional detectors.

---

## 9. Mathematical summaries of the recommended detectors

## 9.1 Hard switch detector

Let

\[
\hat{S}_t = \arg\max_j \alpha_{t|t}(j).
\]

Then define

\[
s_t^{\text{hard}}=
\mathbf{1}\{\hat{S}_t \neq \hat{S}_{t-1}\}.
\]

Optional confidence-gated version:

\[
\text{alarm at } t
\iff
\hat{S}_t \neq \hat{S}_{t-1}
\quad \text{and} \quad
\max_j \alpha_{t|t}(j)\ge \tau_{\text{conf}}.
\]

---

## 9.2 Posterior transition detector

Let

\[
r_{t-1}=\arg\max_j \alpha_{t-1|t-1}(j).
\]

Then define

\[
s_t^{\text{leave}} = 1-\alpha_{t|t}(r_{t-1}).
\]

Or define the full posterior shift magnitude

\[
s_t^{\text{TV}}
=
\frac{1}{2}\sum_{j=1}^K |\alpha_{t|t}(j)-\alpha_{t-1|t-1}(j)|.
\]

Alarm if score exceeds threshold.

---

## 9.3 Surprise detector

Define

\[
s_t^{\text{surp}}=-\log c_t.
\]

Optional baseline-adjusted version:

\[
s_t^{\text{adj}} = s_t^{\text{surp}} - b_t,
\]

where \(b_t\) is a rolling or exponentially smoothed baseline.

Alarm if the chosen surprise score exceeds threshold.

---

## 10. How to compare the approaches conceptually

These three detectors represent three different semantics of change:

| Detector | Meaning of change |
|---|---|
| Hard switch | the most likely regime changed |
| Posterior transition | posterior mass moved away from the previous regime |
| Surprise detector | the current observation became unexpectedly incompatible with model prediction |

This is exactly why they make a strong thesis trio:
- same probabilistic backbone,
- different change semantics,
- different false-alarm / delay behavior.

---

## 11. Runtime invariants and trust checks

Each detector should have its own local checks.

## Hard switch detector
- dominant regime index must be valid,
- posterior vector must be normalized.

## Posterior transition detector
- previous posterior must exist after warmup,
- score must lie in \([0,1]\) for leave-previous-regime version,
- total-variation score must lie in \([0,1]\).

## Surprise detector
- predictive density must be strictly positive,
- log-score must be finite.

## Shared detector checks
- alarm timestamps must be nondecreasing,
- cooldown / persistence counters must remain valid,
- no detector should access future data.

### Deliverable
A runtime validation layer for detector outputs.

### Code changes
Add:
- detector-specific validation helpers,
- score-bound checks,
- optional debug assertions,
- detector-state sanity checks.

---

## 12. Warmup and initialization issues

Some detectors require one previous posterior to exist.

For example:
- hard switch detector needs a previous dominant regime,
- posterior transition detector needs the previous filtered posterior,
- persistence rules need a score history.

So every detector should define its warmup policy explicitly.

### Possible warmup choices
- no alarms on the first step,
- no alarms until enough history exists,
- soft initialization with default state.

### Deliverable
An explicit detector warmup policy.

### Code changes
Add:
- detector-state initialization logic,
- optional `NotReady` / warmup status in detector outputs.

This is important for correctness and clean experiment design.

---

## 13. What this phase does **not** do yet

This phase defines detector variants, but does **not yet** fully solve:

- threshold calibration,
- benchmark protocol,
- performance comparison against BOCPD / CUSUM / others,
- adaptive parameter updating,
- final evaluation metrics.

Those are later phases.

This phase only defines the detector family and the alarm logic built on the MS backbone.

---

## 14. Common conceptual mistakes to avoid

### Mistake 1 — Thinking a regime posterior is already a changepoint detector
It is not.  
You still need a score and an alarm rule.

### Mistake 2 — Using only a discrete hard switch rule
That is a useful baseline, but too brittle as the only detector.

### Mistake 3 — Mixing detector logic into the filter itself
Keep inference and decision layers separate.

### Mistake 4 — Treating outlier surprise as automatically a true changepoint
The surprise detector measures predictive mismatch, not guaranteed structural change.

### Mistake 5 — Forgetting persistence / cooldown mechanisms
Online detectors often need alarm stabilization.

---

## 15. Deliverables of Phase 12

By the end of this phase, you should have:

### Mathematical deliverables
- a formal definition of what “change” means for each detector variant,
- at least three detector score definitions:
  - hard switch,
  - posterior transition,
  - surprise,
- threshold-based or policy-based alarm definitions,
- optional persistence rule defined independently of detector type.

### Architectural deliverables
- a strict separation between:
  - online inference backbone,
  - detector score layer,
  - alarm policy layer,
- a reusable detector abstraction,
- a unified alarm event representation.

### Code-structure deliverables
You should add or revise, where appropriate:

- a dedicated `detector` module,
- a common detector trait / interface,
- a detector input type wrapping online filter outputs,
- a detector output type,
- `HardSwitchDetector`,
- `PosteriorTransitionDetector`,
- `SurpriseDetector`,
- reusable persistence / cooldown policy types,
- `AlarmEvent`,
- warmup-state handling,
- detector-specific validation checks.

### Thesis deliverables
- three clearly distinct Markov-Switching-based changepoint approaches,
- a stronger comparative framework,
- a better basis for later benchmarking against conventional methods.

---

## 16. Minimal final summary

Phase 12 is the step that finally turns the online Markov Switching backbone into an actual changepoint detection framework.

The central idea is:

\[
\boxed{
\text{latent regime inference} \neq \text{changepoint detection}
}
\]

You must explicitly define:
- what score measures change,
- how that score triggers alarms,
- and which interpretation of “change” your detector is using.

For your thesis, the strongest path is to build and compare three detector variants on top of the same online filtering backbone:

1. **Hard Switch Detector**
2. **Posterior Transition Detector**
3. **Surprise Detector**

with optional persistence logic shared across them.

That gives you a diverse, coherent, and theoretically meaningful detector family to compare later against conventional online changepoint methods.