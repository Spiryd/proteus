# Phase 6 — Validate Inference on Simulated Data

## Goal

In this phase, validate the **forward filter** on synthetic data before moving to:

- smoothing,
- EM estimation,
- model comparison,
- or any more advanced extension.

The purpose of this phase is to answer one question:

> Does the forward recursion behave correctly, consistently, and plausibly when the true data-generating process is known?

You are no longer building new theory here.  
You are testing whether the theory from the previous phases has been translated into a reliable inference mechanism.

This phase is essential because the forward filter is the foundation for everything that comes next.  
If the filter is wrong, then:

- smoothed probabilities will be wrong,
- likelihood values will be misleading,
- EM updates will be based on incorrect posteriors,
- later debugging will become much harder.

So Phase 6 is the point where you stop building and start **proving to yourself that the current layer works**.

---

## 1. What “validation” means in this phase

Validation does **not** mean proving the model is true.  
It means checking whether your implementation behaves as the model says it should behave.

Because you already have a simulator from Phase 2, you can generate data where you know:

- the true transition matrix,
- the true regime parameters,
- the true hidden regime path.

This lets you compare:

- what the filter infers,
- with what actually generated the sample.

The validation task is therefore:

1. simulate data under controlled parameter settings,
2. run the forward filter using the true parameters,
3. inspect the predicted and filtered probabilities,
4. check whether the output is mathematically valid and qualitatively sensible.

---

## 2. Why this phase must come before smoothing and estimation

There are several reasons to validate the filter now rather than later.

### 2.1 The filter is the base inference layer
Smoothing and EM both depend on filtering output.  
If filtering is wrong, later phases inherit the error.

### 2.2 Synthetic data gives you controlled experiments
On real data, if something looks strange, you do not know whether the issue is:
- the model,
- the implementation,
- or the data itself.

On simulated data, the true structure is known.

### 2.3 Validation isolates errors early
At this stage, you only have:
- the simulator,
- the emission model,
- the forward filter,
- the log-likelihood layer.

That makes debugging much cleaner than after adding smoothing and estimation.

---

## 3. What exactly you are validating

Phase 6 is about validating both **mathematical correctness** and **statistical behavior**.

## 3.1 Mathematical correctness
You want to check that the recursion respects fundamental invariants:

- predicted probabilities sum to 1,
- filtered probabilities sum to 1,
- predictive densities are positive,
- log-likelihood is finite,
- no probability becomes negative or exceeds 1.

These are implementation-level correctness checks.

---

## 3.2 Statistical behavior
You also want to check whether the filter reacts to different parameter regimes in the way theory predicts.

For example:

- when regimes are clearly separated, filtered probabilities should become sharp,
- when regimes overlap strongly, filtered probabilities should remain more ambiguous,
- when transitions are highly persistent, regime beliefs should evolve smoothly,
- when switching is frequent, beliefs should change more rapidly.

These are model-behavior checks.

---

## 4. Core validation principle

The single most important principle of this phase is:

> Use simulated data with known parameters, and validate whether inference behaves in a way consistent with the known data-generating mechanism.

You are **not yet** asking whether parameters can be estimated well.  
That comes later.

Here you are asking whether, under known parameters:

- the filtering recursion is numerically valid,
- the probabilities look sensible,
- the posterior regime beliefs react correctly to the observed sample.

---

## 5. Step-by-step validation workflow

## Step 1 — Choose a simulation scenario

Start by selecting a parameter configuration for:

- number of regimes \(K\),
- transition matrix \(P\),
- initial distribution \(\pi\),
- means \(\mu_j\),
- variances \(\sigma_j^2\),
- sample length \(T\).

At this phase, you should deliberately choose **different classes of scenarios**, not just one.

### Why
A filter can appear correct in one easy setting and still fail in harder ones.

### Output of this step
A controlled synthetic experiment design.

---

## Step 2 — Simulate a dataset

Using the simulator from Phase 2, generate:

- hidden regime path \(S_{1:T}\),
- observations \(y_{1:T}\).

Retain the hidden path.

### Why
The hidden path is your reference for later comparison.

### Output of this step
A complete synthetic dataset with known ground truth.

---

## Step 3 — Run the forward filter with the true parameters

Use the same parameter set that generated the data.

That means:

- the same transition matrix,
- the same initial distribution,
- the same emission parameters.

The purpose here is **not** parameter recovery.  
It is pure inference validation.

### Why
If the filter behaves badly even when given the true parameters, the issue is almost certainly in the filtering implementation or in the interpretation of the results.

### Output of this step
For each time \(t\), obtain:
- predicted probabilities,
- filtered probabilities,
- predictive density contribution \(c_t\),
- cumulative log-likelihood.

---

## Step 4 — Check mathematical invariants first

Before looking at regime interpretation, verify the core structural conditions.

For every time point \(t\), check:

### Predicted probabilities
\[
\sum_{j=1}^K \Pr(S_t=j \mid y_{1:t-1}) = 1.
\]

### Filtered probabilities
\[
\sum_{j=1}^K \Pr(S_t=j \mid y_{1:t}) = 1.
\]

### Bounds
Each probability must lie in \([0,1]\).

### Predictive density
\[
c_t = f(y_t \mid y_{1:t-1}) > 0.
\]

### Log-likelihood
\[
\log L = \sum_{t=1}^T \log c_t
\]
must remain finite.

### Why this matters
If any of these fail, you have an implementation bug or a severe numerical problem.  
There is no point interpreting the results before these checks pass.

### Output of this step
Basic mathematical confidence in the recursion.

---

## Step 5 — Compare filtered probabilities to the true hidden states

Now use the known latent path from the simulator.

For each time \(t\), compare:

- the true state \(S_t\),
- the filtered probability vector
  \[
  \Pr(S_t=j \mid y_{1:t}).
  \]

Do not expect perfect classification at every time point.  
That is not the goal.

What you want to check is whether the filter is **directionally reasonable**:

- when the observation strongly supports the true regime, the filter should usually place high posterior weight there,
- when regimes overlap heavily, the posterior can remain uncertain,
- when the state just changed, the filter may lag slightly depending on persistence and observation noise.

### Output of this step
A qualitative comparison between inferred regime beliefs and true latent structure.

---

## Step 6 — Examine how the filter behaves near regime changes

Pay special attention to time points where the true state switches.

These are often the most informative points for debugging.

### What to look for
- Does the posterior shift when the hidden regime changes?
- Is the shift immediate or gradual?
- Does strong regime persistence delay adaptation?
- Does a very informative observation overcome prior persistence quickly?

### Why this matters
A filter that looks reasonable in stationary stretches can still behave incorrectly at switching boundaries.

### Output of this step
Confidence that the filter handles regime transitions plausibly.

---

## Step 7 — Evaluate behavior under different scenario classes

Do not stop after one synthetic run.  
Repeat the same validation logic across multiple qualitatively different scenarios.

This is where Phase 6 becomes a real stress test rather than a single sanity check.

---

## 6. Scenario family A — Strongly separated regimes

Choose regimes with clearly different means and/or variances.

For example:
- one low-mean regime,
- one high-mean regime,
- moderate and not excessively large variances,
- reasonably persistent transitions.

### Expected behavior
- filtered probabilities should often be sharp,
- the correct regime should often receive high posterior weight,
- switches should usually be visible in the posterior sequence.

### What this validates
This is the easiest case.  
If the filter struggles here, there is likely a conceptual or implementation problem.

### Why this scenario matters
It tests the filter in a regime-identifiable setting, where the signal from observations is strong.

---

## 7. Scenario family B — Weakly separated regimes

Choose regimes with similar means and similar variances.

### Expected behavior
- filtered probabilities should often remain diffuse,
- posterior uncertainty should be much higher,
- the filter may not strongly favor one regime at many time points.

### What this validates
This scenario checks whether the filter behaves realistically when the data do not strongly distinguish regimes.

### Why this matters
A correct filter should reflect ambiguity rather than invent artificial certainty.

---

## 8. Scenario family C — Mean-switching only

Set regime means different, but variances equal.

### Expected behavior
- regime identification should mainly respond to the location of observations,
- when observations cluster near one regime mean, that regime’s posterior should rise.

### What this validates
It isolates the contribution of the mean parameter.

### Why this matters
It helps confirm that the emission model is interpreting location differences correctly.

---

## 9. Scenario family D — Variance-switching only

Set regime means equal or very similar, but variances different.

### Expected behavior
- the filter should react to differences in dispersion rather than differences in level,
- isolated observations may remain ambiguous,
- clusters of unusually volatile or unusually calm observations should influence posterior beliefs.

### What this validates
It checks whether the filter properly uses scale information, not only mean differences.

### Why this matters
Variance-driven regimes are common in practice and are harder to reason about than mean-switching cases.

---

## 10. Scenario family E — Highly persistent transitions

Choose a transition matrix with large diagonal entries, such as regimes that tend to last a long time.

### Expected behavior
- predicted probabilities should strongly favor staying in the current regime,
- filtered probabilities should usually evolve smoothly,
- regime switches should require stronger observational evidence.

### What this validates
This checks whether the prediction step is using the transition matrix correctly.

### Why this matters
Persistence is one of the key structural features of Markov switching models.

---

## 11. Scenario family F — Weak persistence / frequent switching

Choose a transition matrix with smaller diagonal entries and more switching.

### Expected behavior
- predicted probabilities should be less concentrated,
- filtered probabilities should adapt faster,
- the model should be more responsive to individual observations.

### What this validates
It tests whether the filter can respond appropriately when the Markov chain implies less inertia.

### Why this matters
It isolates the effect of transition dynamics on posterior updating.

---

## 12. Scenario family G — Short samples

Use small \(T\).

### Expected behavior
- regime probabilities may remain less stable,
- early observations may have disproportionate influence,
- inference may be more uncertain overall.

### What this validates
This checks whether the recursion behaves sensibly when information is limited.

### Why this matters
Not all practical datasets are long, and short-sample behavior often reveals initialization issues.

---

## 13. Scenario family H — Long samples

Use large \(T\).

### Expected behavior
- the filter should remain numerically stable,
- probability normalization should remain valid throughout,
- cumulative log-likelihood should remain finite,
- long-run behavior should reflect the model parameters more clearly.

### What this validates
This checks stability across many recursive updates.

### Why this matters
A filter that works for short samples but drifts or breaks on long samples is not reliable.

---

## 14. What to inspect in each validation run

For each synthetic experiment, examine the following outputs.

## 14.1 Hidden state path
This is the ground truth.

## 14.2 Observation path
This tells you what information the filter is actually seeing.

## 14.3 Predicted probabilities
These show what the model believes before seeing the current observation.

## 14.4 Filtered probabilities
These show how the observation updates those beliefs.

## 14.5 Predictive densities
These indicate how surprising each observation is under the model.

## 14.6 Cumulative log-likelihood
This confirms that the forward recursion is producing a coherent overall objective value.

---

## 15. What good validation looks like

A successful validation phase does **not** mean:

- every time point is classified perfectly,
- every filtered posterior is near 0 or 1,
- every scenario is easy.

Instead, good validation means:

### Mathematical integrity
- all invariants hold,
- no impossible values appear,
- no obvious numerical breakdown occurs.

### Behavioral plausibility
- strong separation leads to sharp posterior beliefs,
- weak separation leads to uncertainty,
- persistence affects the smoothness of posterior evolution,
- observations can override prior persistence when sufficiently informative.

### Internal consistency
- the filter responds to the model parameters in the expected direction.

---

## 16. What bad validation looks like

The following are warning signs.

### Warning 1 — Probabilities do not sum to 1
This is a direct bug.

### Warning 2 — Negative or undefined predictive densities
This indicates a serious numerical or logical problem.

### Warning 3 — Posterior probabilities behave erratically in easy scenarios
If strongly separated regimes still produce diffuse and unstable posteriors, the filter may be wrong.

### Warning 4 — Transition persistence appears to have no effect
If changing the transition matrix barely changes predicted probabilities, the prediction step may be faulty.

### Warning 5 — Observation separation appears to have no effect
If changing means or variances barely changes filtered probabilities, the emission layer may be faulty.

### Warning 6 — Long samples break stability
If the recursion becomes invalid or numerically unstable over long runs, the implementation needs strengthening before proceeding.

---

## 17. Recommended validation checklist

For each scenario, go through the following checklist.

### Structural checks
- [ ] predicted probabilities sum to 1 at every time,
- [ ] filtered probabilities sum to 1 at every time,
- [ ] all probabilities lie in \([0,1]\),
- [ ] all predictive densities are positive,
- [ ] total log-likelihood is finite.

### Behavioral checks
- [ ] posterior sharpness increases when regimes are well separated,
- [ ] posterior ambiguity increases when regimes overlap,
- [ ] higher persistence produces smoother regime beliefs,
- [ ] lower persistence produces more adaptive regime beliefs.

### Comparison-to-truth checks
- [ ] filtered probabilities are broadly aligned with the true hidden path,
- [ ] regime changes induce visible shifts in posterior beliefs,
- [ ] easy scenarios look easier than hard scenarios.

---

## 18. How this phase connects to identifiability

Phase 6 is closely tied to the idea of **identifiability**, even though you are not yet estimating parameters.

The key idea is:

- when regimes are well separated and sufficiently persistent, the data tend to carry strong information about the latent state,
- when regimes overlap heavily or switching is too frequent, latent-state inference becomes more ambiguous.

So validation is not just about catching bugs.  
It is also about learning how the structure of the model affects inferential difficulty.

This is important later because if estimation performs poorly in a weakly identified setting, that may be a property of the model and data—not necessarily a programming error.

---

## 19. What should remain fixed during this phase

To keep Phase 6 clean, do **not** vary too many things at once.

At this stage, you should:

- use the true parameters in the filter,
- keep the model class fixed,
- avoid optimization,
- avoid smoothing,
- avoid EM,
- avoid model extensions.

The purpose is to isolate the behavior of the forward recursion itself.

---

## 20. Common conceptual mistakes to avoid

### Mistake 1 — Expecting perfect recovery of the hidden state
The filter computes posterior beliefs, not guaranteed exact classification.

### Mistake 2 — Interpreting ambiguity as failure
If regimes overlap substantially, diffuse posterior probabilities are the correct behavior.

### Mistake 3 — Changing parameters and model structure at the same time
Validation should isolate causes.  
If too many things change at once, the results become hard to interpret.

### Mistake 4 — Using only one synthetic scenario
A single successful run is not enough to validate a recursive inference algorithm.

### Mistake 5 — Moving to EM before validating filtering
This makes later debugging much harder because estimation errors and inference errors become entangled.

---

## 21. Minimal mathematical summary of Phase 6

For each simulated dataset generated under known parameters \(\Theta\):

1. run the forward filter using the same \(\Theta\),
2. compute:
   - predicted probabilities
     \[
     \Pr(S_t=j \mid y_{1:t-1}),
     \]
   - filtered probabilities
     \[
     \Pr(S_t=j \mid y_{1:t}),
     \]
   - predictive densities
     \[
     c_t = f(y_t \mid y_{1:t-1}),
     \]
   - log-likelihood
     \[
     \log L(\Theta)=\sum_{t=1}^T \log c_t,
     \]
3. check structural invariants,
4. compare posterior behavior across easy and hard scenarios,
5. compare inferred regime beliefs with the known true hidden path.

The output of this phase is not a fitted model.  
It is confidence that the forward recursion behaves correctly.

---

## 22. Output of Phase 6

At the end of this phase, you should have:

- a set of validated synthetic test scenarios,
- evidence that the forward filter satisfies its probability invariants,
- evidence that the filter reacts plausibly to regime separation, persistence, variance structure, and sample length,
- qualitative comparisons between filtered probabilities and true hidden states,
- confidence that the forward recursion is reliable enough to support the next phase.

If this phase is done properly, you are ready to move on to smoothing with much greater confidence.