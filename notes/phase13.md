# Phase 13 — Fix the Parameter Policy: Offline-Trained, Online-Filtered Detector

## Goal

In this phase, make an explicit and final modeling decision about how parameters behave in the streaming detector.

For your case, the chosen design is:

\[
\boxed{
\text{train parameters offline once, then keep them fixed during online detection}
}
\]

This means the detector will operate as:

- **offline-trained** — parameters are estimated in advance using historical data,
- **online-filtered** — latent regime beliefs are updated causally as new observations arrive,
- **fixed-parameter** — the streaming detector does **not** re-estimate or adapt model parameters during live operation.

This is the cleanest and most defensible first version for your thesis because it gives you a stable, interpretable benchmark and avoids mixing **detection** with **online learning**.

---

## 1. Why this phase matters

A streaming Markov Switching detector has two conceptually different layers:

## Layer A — model parameters
These are the structural quantities:

- initial distribution \(\pi\),
- transition matrix \(P\),
- regime means \(\mu_j\),
- regime variances \(\sigma_j^2\).

## Layer B — online latent-state inference
These are the evolving posterior quantities:

- predicted regime probabilities,
- filtered regime probabilities,
- predictive density,
- detector scores,
- alarm state.

The question in this phase is:

> During streaming detection, should Layer A remain fixed, or should it evolve together with the data stream?

For your chosen version, the answer is:

\[
\text{Layer A fixed, Layer B dynamic.}
\]

That is the key architectural and methodological decision of this phase.

---

## 2. Why fixed parameters are the right first choice

For an online changepoint detector, using frozen parameters has several major advantages.

## 2.1 It preserves the meaning of “change”

If parameters are fixed, then a change in the detector output really means:

- the new data no longer fit the previously learned regime structure,
- or the posterior is moving between the previously learned regimes.

That is exactly what you want in a changepoint detector.

If the model were allowed to adapt continuously, then a true change could be partially absorbed into the parameter updates, making the detector less sensitive.

---

## 2.2 It makes benchmarking fairer

You said you want to compare the model against conventional online changepoint detectors.

That comparison is cleaner if your method is also a pure online detector and not a hybrid detector-plus-online-learner.

With frozen parameters, the comparison becomes:

- all methods receive the same streaming observations,
- all methods react online,
- none of them get to retrain on future data during the run.

This makes the benchmark more interpretable.

---

## 2.3 It isolates the source of performance

With fixed parameters, if the detector performs well or badly, the reasons are easier to interpret.

Performance can be attributed to:
- the learned regime structure,
- the online filtering dynamics,
- the detector score,
- the alarm policy.

Without fixed parameters, performance becomes harder to interpret because:
- estimation quality,
- adaptation speed,
- forgetting behavior,
- and detection logic

all become entangled.

---

## 2.4 It is much easier to implement correctly

A fixed-parameter streaming detector requires only:

- a fitted model object,
- a causal online filter,
- a detector layer on top.

It does **not** require:
- online EM,
- forgetting factors,
- windowed re-estimation,
- dynamic parameter shrinkage,
- or change-sensitive adaptation rules.

That keeps the first streaming version much more manageable.

---

## 3. What “offline-trained, online-filtered” means precisely

This approach should be understood as a two-stage system.

## Stage 1 — Offline training stage

Use historical or training data to estimate:

\[
\widehat{\Theta}
=
(\widehat{\pi}, \widehat{P}, \widehat{\mu}_1,\dots,\widehat{\mu}_K,\widehat{\sigma}_1^2,\dots,\widehat{\sigma}_K^2).
\]

This uses the offline pipeline already built in earlier phases:
- EM estimation,
- diagnostics,
- model checking,
- optional multi-start fitting.

At the end of this stage, the parameter set is frozen.

---

## Stage 2 — Online detection stage

Given the fixed fitted parameter set \(\widehat{\Theta}\), process the stream online.

At time \(t\), the system updates:

\[
\alpha_{t|t}(j)=\Pr(S_t=j \mid y_{1:t};\widehat{\Theta}),
\]

and then computes changepoint scores and alarms using only online quantities.

The key rule is:

\[
\widehat{\Theta} \text{ remains unchanged throughout the streaming run.}
\]

This is the formal definition of the chosen parameter policy.

---

## 4. What changes online and what stays fixed

This phase should make the distinction explicit.

## 4.1 Fixed during streaming

The following remain constant:

- \(\pi\),
- \(P\),
- \(\mu_j\),
- \(\sigma_j^2\),
- number of regimes \(K\),
- detector thresholds, unless deliberately calibrated beforehand,
- alarm policy configuration.

These are part of the detector configuration.

---

## 4.2 Updated during streaming

The following evolve as new observations arrive:

- predicted probabilities,
- filtered probabilities,
- predictive density,
- cumulative online score if tracked,
- detector internal state,
- persistence / cooldown counters,
- emitted alarm history.

So streaming dynamics happen entirely in the **posterior state layer** and **detector layer**, not in the parameter layer.

---

## 5. Mathematical definition of the online-fixed-parameter detector

Let the fitted offline parameter set be

\[
\widehat{\Theta} = (\widehat{\pi},\widehat{P},\widehat{\theta}_1,\dots,\widehat{\theta}_K),
\]

where

\[
\widehat{\theta}_j = (\widehat{\mu}_j,\widehat{\sigma}_j^2).
\]

Then for each new observation \(y_t\), online inference is performed as:

### Prediction
\[
\alpha_{t|t-1}(j)
=
\sum_{i=1}^K \widehat{p}_{ij}\,\alpha_{t-1|t-1}(i).
\]

### Emission evaluation
\[
\widehat{f}_j(y_t)
=
f(y_t \mid S_t=j;\widehat{\theta}_j).
\]

### Predictive density
\[
c_t
=
\sum_{j=1}^K \widehat{f}_j(y_t)\alpha_{t|t-1}(j).
\]

### Update
\[
\alpha_{t|t}(j)
=
\frac{
\widehat{f}_j(y_t)\alpha_{t|t-1}(j)
}{
c_t
}.
\]

At every step, the parameter values with hats remain unchanged.

That is the full causal inference rule of the chosen design.

---

## 6. Why adaptive parameter updates are intentionally excluded

This phase should state explicitly what is **not** being done.

The detector does **not**:
- re-estimate means online,
- re-estimate variances online,
- adapt the transition matrix online,
- rerun EM on sliding windows,
- introduce forgetting factors,
- update thresholds based on future information.

This exclusion is not a limitation of the theory.  
It is a deliberate design choice for the first benchmarkable version.

### Why this is a good thesis decision
Because it gives you:
- a clear baseline,
- strong interpretability,
- fair comparison against standard online detectors,
- and a clean path for future extension.

You can later present adaptive versions as a natural extension, but they should not be mixed into the first core detector.

---

## 7. Practical interpretation of the frozen-parameter detector

The frozen-parameter detector answers the question:

> How do incoming data compare to the regime structure learned offline?

That is an important viewpoint.

This means the detector is not trying to “keep up” with the stream by changing itself.  
Instead, it is using a stable learned model to judge whether the stream is still behaving in a way consistent with the trained regime system.

This is especially suitable when your thesis goal is:
- comparing an MS-based detector to conventional online methods,
- rather than building a fully adaptive streaming learner.

---

## 8. Step-by-step guide for Phase 13

## Step 1 — Define the parameter policy formally

Write down explicitly that the streaming detector uses a fixed fitted parameter object:

\[
\widehat{\Theta}
\]

estimated offline and held constant during the online run.

This should be part of your theoretical specification, not merely an implementation habit.

### Deliverable
A formal fixed-parameter policy statement.

### Code changes
You should add a clearly named configuration or design concept such as:
- `FixedParameterPolicy`,
- `FrozenModel`,
- or a documented invariant in the online detector architecture.

Even if you do not encode this as an enum yet, the design should make the fixed-parameter assumption explicit.

---

## Step 2 — Separate training artifacts from runtime artifacts

The model fitted offline should become a reusable runtime artifact.

That means distinguishing:

### Offline fit result
Contains:
- EM history,
- diagnostics,
- multi-start summaries,
- optional retrospective posterior summaries.

### Runtime detector model
Contains only what is needed online:
- fitted parameters,
- detector configuration,
- optional calibration thresholds.

### Deliverable
A clean boundary between fitted-model artifacts and runtime detector artifacts.

### Code changes
You should introduce or refine:
- a compact runtime parameter object,
- conversion from offline fit result to online detector model,
- a separation between heavy fit results and lightweight runtime state.

This is a major structural improvement.

---

## Step 3 — Make fitted parameters immutable during streaming

The online inference layer should not mutate the model parameters.

Only the streaming posterior state and detector state may change.

### Deliverable
An immutability rule for runtime model parameters.

### Code changes
Design the streaming API so that:
- the fitted model is passed by immutable reference,
- online state is the only mutable evolving object.

This is a strong design choice and should be visible in the Rust structure.

---

## Step 4 — Define the runtime online state separately

Since parameters are fixed, the runtime online state should only contain:

- current time index,
- current filtered posterior,
- optional previous filtered posterior,
- predictive density history if needed,
- detector-specific counters or buffers,
- alarm history if stored.

### Deliverable
A minimal and clean streaming state representation.

### Code changes
Refine or add:
- `OnlineFilterState`,
- detector state objects,
- score history or policy state as needed.

Make sure none of these objects store mutable copies of the parameters.

---

## Step 5 — Ensure detectors consume only online outputs

Each detector should operate only on:
- filtered regime posteriors,
- predictive density,
- detector-local state,
- fixed thresholds or calibration constants.

They must not request:
- parameter updates,
- smoothing outputs,
- batch re-estimation.

### Deliverable
A detector layer that is strictly downstream of frozen-parameter online filtering.

### Code changes
Review detector interfaces so they consume:
- online filter outputs,
- not fit-time structures or offline diagnostics.

This prevents hidden leakage of offline functionality into runtime logic.

---

## Step 6 — Add explicit documentation that this is the baseline detector mode

Because you may later add adaptive variants, this phase should define the current approach as the baseline.

A useful name is something like:

\[
\text{Offline-trained, online-filtered fixed-parameter detector}
\]

This should become the reference model in the thesis.

### Deliverable
A clearly named baseline detector mode.

### Code changes
Add explicit naming in:
- documentation,
- module naming,
- experiment configuration,
- benchmark labels.

This helps later comparison against adaptive or conventional detectors.

---

## 9. What the runtime API should conceptually look like

Even without writing code here, the runtime workflow should look like:

### Initialization
- load or construct frozen fitted model,
- initialize online posterior state,
- initialize detector state.

### Streaming step
For each new \(y_t\):
1. update online filter using frozen parameters,
2. compute detector score,
3. update alarm policy,
4. emit optional alarm event.

### Finalization
- return or export alarm sequence,
- optionally export online score trajectory,
- optionally export filtered posterior trajectory.

This workflow should be the canonical streaming execution path.

---

## 10. Invariants that should be enforced in this phase

Because the whole design depends on the frozen-parameter policy, the following invariants should hold.

## Parameter invariants
Throughout the online run:
- transition matrix remains unchanged,
- regime means remain unchanged,
- regime variances remain unchanged,
- number of regimes remains unchanged.

## Posterior invariants
At each online step:
\[
\sum_{j=1}^K \alpha_{t|t}(j)=1.
\]

\[
\sum_{j=1}^K \alpha_{t+1|t}(j)=1.
\]

\[
c_t>0.
\]

## Architectural invariants
- online code must not call EM,
- online code must not call smoothing,
- detector code must not mutate parameters.

These invariants are central to the correctness of the chosen design.

---

## 11. Why this phase helps the thesis experimentally

Using an offline-trained, online-filtered baseline gives you a very useful experimental reference point.

It lets you answer questions like:

- How much value does a learned regime model add without online adaptation?
- Does a fixed Markov Switching backbone already compete with conventional detectors?
- Which detector rule works best when the underlying latent model is fixed?
- Are failures due to the model backbone or due to alarm logic?

These are cleaner questions than if you immediately introduced online adaptation.

So this phase is not just a simplification.  
It is also a stronger experimental design choice.

---

## 12. How this phase interacts with the detector family from Phase 12

The detector variants from the previous phase remain valid here:

- Hard Switch Detector,
- Posterior Transition Detector,
- Surprise Detector.

What changes now is that all of them operate under the same frozen model parameters.

This is good because it ensures that differences in performance arise from:

- different score definitions,
- different alarm policies,

rather than from different online adaptation behavior.

That is exactly the kind of controlled comparison you want in a thesis.

---

## 13. What you should **not** add yet

Because this phase is tightly scoped to the offline-trained, online-filtered approach, do **not** add:

- sliding-window refits,
- adaptive means or variances,
- transition-matrix adaptation,
- forgetting factors,
- online EM,
- periodic retraining inside the stream,
- detector thresholds that learn from future data.

Those belong to later extensions if you choose to pursue them.

For now, the value of this phase comes from locking down a clear and disciplined baseline.

---

## 14. Suggested naming for this approach in the thesis

A few strong names you can use consistently:

- **Fixed-parameter MS detector**
- **Offline-trained / online-filtered MS detector**
- **Frozen-parameter regime-switching detector**
- **Causal filtering detector with fixed offline calibration**

Pick one and use it consistently across:
- documentation,
- code labels,
- plots,
- benchmarks.

That consistency will help the thesis read more cleanly.

---

## 15. Common conceptual mistakes to avoid

### Mistake 1 — Saying the model is adaptive when only the posterior changes
Posterior regime beliefs changing is not parameter adaptation.

### Mistake 2 — Treating runtime state as part of the fitted model
The online posterior state is dynamic and separate from the learned parameter object.

### Mistake 3 — Reusing full fit-result objects directly in streaming mode
Those objects often contain offline-only information and create unnecessary coupling.

### Mistake 4 — Calling any periodic batch refit “online filtering”
That is a different design and should not be mixed into this baseline.

### Mistake 5 — Hiding the fixed-parameter assumption
This assumption should be explicit because it shapes how results are interpreted.

---

## 16. Deliverables of Phase 13

By the end of this phase, you should have:

### Mathematical deliverables
- a formal definition of the offline-trained, online-filtered detector,
- a clear statement that
  \[
  \widehat{\Theta}
  \]
  is estimated offline and then frozen during the stream,
- a precise distinction between dynamic posterior updates and fixed structural parameters,
- a documented exclusion of adaptive parameter updates in the baseline design.

### Architectural deliverables
- a clean separation between:
  - offline fit artifacts,
  - frozen runtime model,
  - mutable online filter state,
  - mutable detector state,
- a canonical baseline detector mode for experiments.

### Code-structure deliverables
You should add or revise, where appropriate:

- a dedicated runtime model object for frozen fitted parameters,
- conversion from offline fit result to runtime detector model,
- immutable parameter access in the streaming API,
- a streaming state object that excludes mutable parameters,
- clear separation between online and offline result types,
- detector interfaces that consume only online filter outputs,
- optional configuration type documenting the fixed-parameter policy,
- explicit labels / naming for this baseline in experiment configs and outputs.

### Trust deliverables
- confidence that your online detector is truly a **detector** and not a hidden adaptive learner,
- a stable baseline architecture for later benchmarking,
- a clean foundation for the next phase, where benchmarking protocol and experimental comparison can be built on top.

---

## 17. Minimal final summary

Phase 13 locks down the parameter behavior of your streaming detector.

The central rule is:

\[
\boxed{
\text{Learn once offline, then detect online with frozen parameters.}
}
\]

That means:

- parameters are estimated offline,
- online filtering updates only posterior beliefs,
- detectors operate on causal posterior and predictive outputs,
- no online re-estimation is performed.

For your thesis, this is the right first benchmarkable Markov Switching detector because it is:
- interpretable,
- fair to compare,
- easy to reason about,
- and methodologically clean.