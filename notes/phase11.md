# Phase 11 — Recast the Model for Online Inference Only

## Goal

In this phase, redesign the Markov Switching Model pipeline so that it can operate in a **causal, streaming, online setting**.

Up to this point, the project has been built primarily around an **offline workflow**:

- fit model parameters,
- run forward filtering,
- run backward smoothing,
- compute pairwise posteriors,
- estimate parameters with EM,
- inspect diagnostics after seeing the full sample.

That workflow is correct for offline statistical modeling, but it is **not** the right inference structure for real-time changepoint detection.

In an online / data-stream setting, when a new observation \(y_t\) arrives, the algorithm is allowed to use only:

\[
y_{1:t} = (y_1,\dots,y_t),
\]

not future data \(y_{t+1:T}\).

So the central inferential object in this phase becomes:

\[
\Pr(S_t=j \mid y_{1:t}),
\]

not

\[
\Pr(S_t=j \mid y_{1:T}).
\]

This phase formalizes that shift.

---

## 1. Why this phase is necessary

A Markov Switching Model can serve as the **backbone** of an online changepoint detector, but only if the inference pipeline itself is causal.

That means:

- no backward smoothing in live inference,
- no pairwise full-sample posteriors in live inference,
- no decisions based on future observations,
- no repeated whole-sample EM during detection time.

Without this phase, the model is still fundamentally an offline latent-state model with retrospective inference tools.

So Phase 11 is the point where you establish a strict distinction between:

## Offline mode
Used for:
- historical fitting,
- calibration,
- diagnostics,
- simulation studies,
- threshold tuning.

## Online mode
Used for:
- one-step causal inference,
- stream updates,
- real-time scoring,
- real-time changepoint alarms.

This separation is the foundation for everything that follows.

---

## 2. The main conceptual shift

Before this phase, the strongest latent-state inference object was:

\[
\gamma_t(j) = \Pr(S_t=j \mid y_{1:T}),
\]

the smoothed full-sample posterior.

That object is excellent for:
- retrospective interpretation,
- estimation,
- diagnostics,
- visualizing latent regimes after the sample is complete.

But in an online system, it is unusable for real-time decisions because it depends on future data.

So in online mode, the highest-level state belief you are allowed to use is:

\[
\alpha_{t|t}(j)=\Pr(S_t=j \mid y_{1:t}).
\]

This is the filtered posterior probability.

That means the forward filter is no longer just one inference component among others.  
In online mode, it becomes the **primary inference engine**.

---

## 3. Formal definition of streaming mode

Streaming mode means the following.

At time \(t\), the system has access to:

- the fitted parameter set \(\Theta\),
- the previous online inference state,
- the new observation \(y_t\),
- and nothing beyond time \(t\).

The online inference update must produce:

- updated predicted regime probabilities,
- updated filtered regime probabilities,
- predictive density / surprise score if needed,
- optional detector-facing summary values.

It must **not** produce:

- smoothed probabilities,
- pairwise full-sample transition probabilities,
- EM parameter updates based on the full sample,
- any output requiring future observations.

This is the formal rule that defines the causal boundary of the online algorithm.

---

## 4. What remains offline vs what becomes online

This phase should make the project architecture explicitly bifurcated.

## 4.1 Offline components

These remain offline-only:

### Model fitting
- EM estimation,
- parameter initialization experiments,
- multi-start comparison.

### Retrospective inference
- smoothing,
- pairwise posterior transitions.

### Calibration
- threshold selection,
- validation-set tuning,
- experiment design.

### Diagnostics
- post-fit trust checks,
- regime interpretation,
- expected-duration summaries,
- stability analysis.

These are all legitimate, but they happen **before deployment** or **outside the live stream**.

---

## 4.2 Online components

These become live-stream components:

### One-step prediction
Use the current filtered posterior and transition matrix to predict the next regime distribution.

### One-step update
Use the new observation to update the filtered regime posterior.

### Predictive density tracking
Compute
\[
c_t = f(y_t \mid y_{1:t-1})
\]
if needed for online scoring.

### Alarm-facing outputs
Expose only causal quantities that can be used by a changepoint decision layer later.

This is the online runtime layer.

---

## 5. The online inference object

The key online state summary at time \(t\) is:

\[
\alpha_{t|t}(j)=\Pr(S_t=j \mid y_{1:t}).
\]

This is the posterior regime distribution after processing the stream up to time \(t\).

The online algorithm should also naturally maintain:

## Predicted probabilities
\[
\alpha_{t+1|t}(j)=\Pr(S_{t+1}=j \mid y_{1:t})
\]

These are useful for:
- the next update step,
- predictive likelihood computation,
- future scoring logic.

## Predictive density
\[
c_t = f(y_t \mid y_{1:t-1})
\]

This is useful for:
- online anomaly / surprise scoring,
- log-score monitoring,
- potential changepoint scoring later.

These quantities form the minimal online inference state.

---

## 6. Online recursion remains the forward filter

Mathematically, the online algorithm is the same forward recursion you already built.  
What changes is **how it is used**.

At time \(t\), given the previous filtered state:

### Step 1 — Predict
\[
\alpha_{t|t-1}(j)
=
\sum_{i=1}^K p_{ij}\,\alpha_{t-1|t-1}(i).
\]

### Step 2 — Evaluate the new observation
\[
f_j(y_t)=f(y_t \mid S_t=j;\theta_j).
\]

### Step 3 — Compute predictive density
\[
c_t=\sum_{j=1}^K f_j(y_t)\alpha_{t|t-1}(j).
\]

### Step 4 — Update filtered posterior
\[
\alpha_{t|t}(j)
=
\frac{f_j(y_t)\alpha_{t|t-1}(j)}{c_t}.
\]

This is now the **core streaming step**.

---

## 7. What must be forbidden in online mode

This phase must explicitly define forbidden operations in causal inference mode.

## Forbidden operation 1 — backward smoothing
Anything requiring:

\[
\Pr(S_t=j \mid y_{1:T})
\]

is offline-only.

### Why
It uses future data.

---

## Forbidden operation 2 — pairwise full-sample posteriors
Anything requiring:

\[
\Pr(S_{t-1}=i,S_t=j \mid y_{1:T})
\]

is offline-only.

### Why
It also depends on future information through smoothing.

---

## Forbidden operation 3 — full-sample EM re-estimation in the live loop
The live detector should not rerun whole-sample EM every time a new point arrives.

### Why
That breaks the intended streaming architecture and mixes offline fitting with online inference.

---

## Forbidden operation 4 — alarms based on retrospective quantities
No detection decision in streaming mode may use:
- smoothed regime probabilities,
- future-corrected transition probabilities,
- hindsight segmentation.

### Why
That would make the reported online performance invalid.

This rule is extremely important for fair benchmark comparisons later.

---

## 8. Step-by-step guide for Phase 11

## Step 1 — Define two explicit operating modes

Introduce two formally separate modes:

### Offline mode
Used for:
- training,
- diagnostics,
- calibration,
- evaluation.

### Online mode
Used for:
- causal filtering,
- real-time summary scores,
- detection-layer input.

These two modes should not be mixed implicitly.

### Deliverable
A formal project-level distinction between offline and online workflows.

### Code changes
You should add a conceptual split in the architecture, such as:
- `offline` / `training` path,
- `online` / `streaming` path.

This can be reflected through:
- modules,
- service objects,
- mode-specific result types,
- or explicit runtime APIs.

---

## Step 2 — Define the online inference contract

The online layer should expose a one-step causal update interface.

Conceptually, it should:

### Input
- fitted model parameters,
- previous online state,
- new observation \(y_t\).

### Output
- new predicted probabilities,
- new filtered probabilities,
- predictive density,
- optional cumulative log-score or other online diagnostics.

### Deliverable
A clear one-step online inference contract.

### Code changes
Add a dedicated online inference abstraction, for example:
- `OnlineFilterState`,
- `OnlineUpdateInput`,
- `OnlineUpdateOutput`,
- or an `OnlineInferenceEngine`.

This should be distinct from batch forward-filter result objects.

---

## Step 3 — Define persistent online state

In offline filtering, you often think in terms of arrays over the full time axis.

In streaming mode, the algorithm must instead maintain a persistent state object containing only what is needed for the next update.

At minimum, the online state should track:

- current filtered regime probabilities,
- current time index,
- optionally cumulative log-likelihood / log-score,
- optional recent score history if later phases need it,
- possibly last predicted probabilities if useful for debugging.

### Deliverable
A stateful online filtering representation.

### Code changes
Add a state container specifically for streaming inference, separate from:
- full-sample filter matrices,
- smoother outputs,
- EM outputs.

This is a major architectural deliverable of the phase.

---

## Step 4 — Freeze the model parameters in online mode

For the first online version, assume the fitted parameter set \(\Theta\) is fixed.

That means online mode does **not** update:
- \(\pi\),
- \(P\),
- \(\mu_j\),
- \(\sigma_j^2\).

Instead, it only updates posterior regime beliefs.

### Why
This is the cleanest first streaming design and avoids entangling online inference with online learning.

### Deliverable
A fixed-parameter online operating assumption.

### Code changes
Your online inference API should take a read-only or immutable fitted parameter object and use it without mutating it.

This should be enforced by design if possible.

---

## Step 5 — Expose only causal outputs to future detector layers

The online filter should expose only quantities valid at time \(t\), such as:

- \(\alpha_{t|t}(j)\),
- \(\alpha_{t+1|t}(j)\),
- \(c_t\),
- optional cumulative online score.

It should not expose:
- smoothed probabilities,
- pairwise smoothed transitions,
- hindsight classification summaries.

### Deliverable
A causal output policy for online inference.

### Code changes
Define distinct online result types so that later changepoint detector code cannot accidentally depend on offline-only fields.

This is an important design safeguard.

---

## Step 6 — Add runtime invariants for online mode

At every online update, verify:

### Filtered normalization
\[
\sum_{j=1}^K \alpha_{t|t}(j)=1.
\]

### Predicted normalization
\[
\sum_{j=1}^K \alpha_{t+1|t}(j)=1.
\]

### Positivity of predictive density
\[
c_t > 0.
\]

### Finite score values
No NaN or infinite values in:
- filtered probabilities,
- predictive density,
- cumulative log-score.

### Deliverable
A streaming safety-check layer.

### Code changes
Add lightweight online validation utilities that can run:
- always in debug mode,
- optionally in release mode,
- or through a configurable diagnostics level.

---

## Step 7 — Define the boundary to the later changepoint detector

This phase does **not** yet define the changepoint rule.  
But it should prepare for it.

That means the online filter should produce outputs that a later detector can consume, such as:

- filtered regime posterior,
- predictive density,
- posterior shift information derivable from the current and previous filtered states.

### Deliverable
A detector-ready online inference output surface.

### Code changes
Design the online result object so that later phases can build changepoint scores on top of it without having to rework the filter.

---

## 9. Online vs offline result objects

This phase should make result types mode-specific.

## Offline fit / inference outputs may include
- full filtered matrix,
- smoothed matrix,
- pairwise posterior tensor,
- EM history,
- diagnostics summaries.

## Online outputs should include only
- current time,
- current filtered posterior,
- next predicted posterior,
- current predictive density,
- optional cumulative online score,
- optional local validation flags.

This distinction is important because it prevents accidental use of hindsight information.

### Deliverable
Separate result abstractions for offline and online use.

### Code changes
Refactor result types if necessary so that:
- batch/offline types remain rich,
- online types remain minimal and causal.

---

## 10. The online filter as a state machine

A useful way to think about the streaming inference engine is as a state machine.

At time \(t-1\), it stores:
- current posterior regime beliefs.

When \(y_t\) arrives, it:
1. predicts,
2. scores,
3. updates,
4. advances internal time,
5. emits a new online summary.

This mental model is better for streaming implementation than thinking in terms of whole-sample arrays.

### Deliverable
A state-machine interpretation of the online inference layer.

### Code changes
Design the API around repeated `step(...)` updates instead of batch `run(...)` calls.

This is one of the key implementation consequences of Phase 11.

---

## 11. What should happen to smoothing code

Smoothing should remain in the project, but it must become clearly labeled as offline-only.

It still matters for:
- training,
- diagnostics,
- retrospective evaluation,
- simulation studies,
- EM.

But it must not be reachable through the online runtime path.

### Deliverable
A clear classification of smoothing as an offline facility.

### Code changes
You should separate smoothing-related code paths from the online module boundary, so that online code cannot accidentally invoke them.

This is both a conceptual and architectural safeguard.

---

## 12. What should happen to EM code

Similarly, EM remains useful for:

- offline parameter learning,
- retraining on historical windows,
- calibration studies.

But it should not be part of the per-observation online update path in this first streaming design.

### Deliverable
A clear classification of EM as a training-time component.

### Code changes
Keep estimation modules separate from streaming modules and avoid any runtime dependency from the online step function into the EM machinery.

---

## 13. Minimal online mathematical summary

In online mode, the causal inference state at time \(t\) is built using only \(y_{1:t}\).

For each new observation \(y_t\):

### Prediction
\[
\alpha_{t|t-1}(j)
=
\sum_{i=1}^K p_{ij}\alpha_{t-1|t-1}(i).
\]

### Emission evaluation
\[
f_j(y_t)=f(y_t \mid S_t=j;\theta_j).
\]

### Predictive density
\[
c_t=\sum_{j=1}^K f_j(y_t)\alpha_{t|t-1}(j).
\]

### Update
\[
\alpha_{t|t}(j)
=
\frac{f_j(y_t)\alpha_{t|t-1}(j)}{c_t}.
\]

The key rule of online mode is:

\[
\text{Only quantities measurable with respect to } y_{1:t} \text{ may be used at time } t.
\]

That is the mathematical heart of this phase.

---

## 14. What this phase does **not** solve yet

Phase 11 does **not** yet define:

- what counts as a changepoint,
- how alarms are triggered,
- how thresholds are calibrated,
- how online performance is benchmarked,
- how conventional detectors will be compared.

Those are later phases.

This phase only ensures that the underlying regime model is being used in a truly online way.

That distinction is very important.

---

## 15. Common conceptual mistakes to avoid

### Mistake 1 — Calling the model “online” while still using smoothing
If a decision uses \(\Pr(S_t=j \mid y_{1:T})\), it is not online.

### Mistake 2 — Reusing full-sample batch result types in streaming mode
Those result objects often contain hindsight information and are unsafe for online use.

### Mistake 3 — Updating parameters inside the streaming loop too early
For the first streaming version, keep parameters fixed.

### Mistake 4 — Confusing filtering with detection
Filtering produces regime beliefs.  
Detection rules still need to be defined later.

### Mistake 5 — Not enforcing the boundary in code
If the architecture allows online code to accidentally access smoothing or EM results, later benchmarks may become invalid.

---

## 16. Deliverables of Phase 11

By the end of this phase, you should have:

### Mathematical deliverables
- a formal definition of online / streaming mode,
- a clear statement that the primary online state posterior is
  \[
  \Pr(S_t=j \mid y_{1:t}),
  \]
  not
  \[
  \Pr(S_t=j \mid y_{1:T}),
  \]
- a rule that only causal quantities may be used for future detection decisions,
- a formal distinction between training-time and runtime inference.

### Architectural deliverables
- a strict separation between offline and online workflows,
- an online inference state machine design,
- a one-step update interface,
- separate offline and online result objects,
- explicit classification of smoothing and EM as offline-only components.

### Code-structure deliverables
You should add or revise, where appropriate:

- a dedicated `online` or `streaming` module,
- an `OnlineFilterState`-style state container,
- a one-step update API,
- online result types exposing only causal quantities,
- runtime normalization / finiteness checks for online updates,
- explicit separation from smoothing code paths,
- explicit separation from EM / training code paths.

### Trust deliverables
- confidence that the regime model can now run causally on a stream,
- a clean foundation for the next phase, where the actual changepoint score and alarm logic will be defined.

---

## 17. Minimal final summary

Phase 11 is the phase that converts the Markov Switching Model from an offline latent-state analysis tool into a causally usable streaming inference backbone.

The central principle is:

\[
\boxed{
\text{At time } t,\ \text{the online system may use only } y_{1:t}.
}
\]

In practice, that means:

- filtering is the live inference engine,
- smoothing is offline-only,
- EM is offline-only,
- online outputs must be causal by construction.

Once this phase is complete, the project is ready for the next step: defining how these causal regime posteriors will become an actual online changepoint detector.