# Phase 7 — Add Backward Smoothing

## Goal

In this phase, extend the inference layer from **filtering** to **smoothing**.

At this point, you already have a working forward filter that computes, for each time \(t\),

- predicted probabilities
  \[
  \Pr(S_t=j \mid y_{1:t-1}),
  \]
- filtered probabilities
  \[
  \Pr(S_t=j \mid y_{1:t}),
  \]
- predictive densities,
- and the observed-data log-likelihood.

Now you want to answer a stronger inferential question:

> Given the **entire observed sample** \(y_{1:T}\), what is the posterior probability that regime \(j\) was active at time \(t\)?

This is the smoothing problem.

Formally, smoothing computes:

\[
\Pr(S_t=j \mid y_{1:T}),
\qquad t=1,\dots,T.
\]

This phase is the first point where your inference stops being purely online / sequential and becomes fully retrospective.

---

## 1. What smoothing is

Filtering and smoothing answer different questions.

## Filtering
Filtering computes:

\[
\Pr(S_t=j \mid y_{1:t}).
\]

This uses only:
- past observations,
- the current observation.

It is the correct object for **online inference**, where future data are not yet available.

---

## Smoothing
Smoothing computes:

\[
\Pr(S_t=j \mid y_{1:T}).
\]

This uses:
- past observations,
- the current observation,
- and all future observations up to time \(T\).

It is the correct object for **retrospective inference**, where the full sample is already observed.

---

## Key difference
Filtering asks:

> “What do I think about the regime at time \(t\) given what I knew at time \(t\)?”

Smoothing asks:

> “Now that I have seen the whole sample, what do I think the regime at time \(t\) was?”

That distinction is central.

---

## 2. Why smoothing matters

Smoothing matters because future data often contain information about past regimes.

This is especially true near regime changes.

For example:

- an observation at time \(t\) may be ambiguous,
- but the observations at times \(t+1,t+2,t+3\) may strongly indicate which regime the process had already entered.

The forward filter cannot use that future evidence.  
The smoother can.

So smoothing usually gives:

- more informative posterior state probabilities,
- cleaner retrospective regime interpretation,
- better latent-state summaries for plotting and analysis,
- the correct latent expectations needed for EM and later estimation stages.

---

## 3. What the smoother adds to the project

Before this phase, your inference layer produces:

- predicted probabilities,
- filtered probabilities.

After this phase, it should also produce:

- smoothed probabilities.

That means your project now supports both:

### Online inference
\[
\Pr(S_t=j \mid y_{1:t})
\]

### Full-sample retrospective inference
\[
\Pr(S_t=j \mid y_{1:T})
\]

This is an important conceptual milestone because it completes the basic hidden-state inference pipeline.

---

## 4. The core smoothing object

Define the smoothed probabilities as

\[
\gamma_t(j)
=
\Pr(S_t=j \mid y_{1:T}).
\]

This notation is convenient because it separates smoothing from filtering notation.

So in this phase you now have three distinct regime-probability objects:

### Predicted probabilities
\[
\alpha_{t|t-1}(j)=\Pr(S_t=j \mid y_{1:t-1})
\]

### Filtered probabilities
\[
\alpha_{t|t}(j)=\Pr(S_t=j \mid y_{1:t})
\]

### Smoothed probabilities
\[
\gamma_t(j)=\Pr(S_t=j \mid y_{1:T})
\]

These should remain conceptually distinct throughout the project.

---

## 5. The intuition behind backward smoothing

The smoother is a **backward pass** built on top of the forward filter.

The basic logic is:

1. the forward filter summarizes what the data up to time \(t\) say about the current state,
2. the backward smoother incorporates what the data after time \(t\) imply about that same state.

So smoothing combines:

- the filtered belief at time \(t\),
- transition information from \(t\) to \(t+1\),
- the future information already encoded in the smoothed distribution at time \(t+1\).

This is why smoothing is naturally done **backward in time**.

---

## 6. Why the backward pass is possible

The smoother works because the hidden-state model has a recursive structure.

Once you know:

- the filtered probabilities at time \(t\),
- the transition matrix,
- the predicted probabilities at time \(t+1\),
- the smoothed probabilities at time \(t+1\),

you can propagate full-sample information backward from time \(t+1\) to time \(t\).

This is possible because the hidden process is Markov and the forward filter has already summarized the past efficiently.

So the smoother does not need to revisit the full raw data history at every step.  
It reuses the sufficient recursive quantities produced by the forward pass.

---

## 7. Step-by-step conceptual structure

The backward smoother should be thought of in two layers:

## Layer A — Forward pass
Run the forward filter first and store:

- predicted probabilities,
- filtered probabilities,
- optionally likelihood-related quantities already produced in Phase 4 and 5.

## Layer B — Backward pass
Start from the end of the sample and move backward, using the forward results plus the transition dynamics.

This means smoothing is **not** a standalone inference procedure.  
It depends on the forward filter being correct and complete.

---

## 8. Boundary condition at the final time point

Start at time \(T\).

At the final observation, there is no future information beyond time \(T\).  
So the smoothed and filtered distributions coincide:

\[
\gamma_T(j)=\Pr(S_T=j \mid y_{1:T})=\Pr(S_T=j \mid y_{1:T})=\alpha_{T|T}(j).
\]

So the terminal condition for the backward recursion is simply:

\[
\gamma_T(j)=\alpha_{T|T}(j).
\]

### Interpretation
At the final time point, “full-sample” information and “current-time” information are the same thing.

### Output of this step
A starting vector for the backward recursion.

---

## 9. Backward recursion formula

For \(t=T-1,T-2,\dots,1\), compute:

\[
\gamma_t(i)
=
\alpha_{t|t}(i)
\sum_{j=1}^K
\frac{p_{ij}\,\gamma_{t+1}(j)}{\alpha_{t+1|t}(j)}.
\]

This is the standard form of the backward smoothing recursion for the hidden-state model you are building.

Here:

- \(\alpha_{t|t}(i)\) is the filtered probability at time \(t\),
- \(p_{ij}\) is the transition probability from regime \(i\) to regime \(j\),
- \(\alpha_{t+1|t}(j)\) is the predicted probability of regime \(j\) at time \(t+1\),
- \(\gamma_{t+1}(j)\) is the smoothed probability at time \(t+1\).

---

## 10. Interpretation of the recursion

The recursion can be read in the following way.

Suppose you want the full-sample posterior probability that the process was in state \(i\) at time \(t\).

Start with:

\[
\alpha_{t|t}(i),
\]

which is what the forward filter already concluded using data up to time \(t\).

Then correct it by asking:

> “If the process were in state \(i\) at time \(t\), how compatible would that be with what we later learned about time \(t+1\) and beyond?”

That correction is exactly the backward multiplier

\[
\sum_{j=1}^K
\frac{p_{ij}\,\gamma_{t+1}(j)}{\alpha_{t+1|t}(j)}.
\]

So the smoother is:

- filtered belief,
- multiplied by a future-information correction factor.

This is the core intuition of the backward pass.

---

## 11. Why predicted probabilities appear in the denominator

The denominator

\[
\alpha_{t+1|t}(j)
=
\Pr(S_{t+1}=j \mid y_{1:t})
\]

appears because smoothing compares:

- what the forward pass expected about state \(j\) at time \(t+1\),
- with what the full sample ultimately implies about state \(j\) at time \(t+1\).

So the ratio

\[
\frac{\gamma_{t+1}(j)}{\alpha_{t+1|t}(j)}
\]

measures how future information changes the belief about regime \(j\) at time \(t+1\).

That correction is then transported backward through the transition matrix.

This is one of the reasons why predicted probabilities from the forward pass must be stored carefully.

---

## 12. Alternative intuition: filtering vs smoothing

It helps to compare the two directly.

## Filtered probability
\[
\Pr(S_t=i \mid y_{1:t})
\]

This tells you what is plausible at time \(t\) using data up to that point.

## Smoothed probability
\[
\Pr(S_t=i \mid y_{1:T})
\]

This tells you what is plausible at time \(t\) after the entire future path has also been seen.

So smoothing should never be thought of as a completely different kind of inference.  
It is a refinement of filtering using future information.

---

## 13. Step-by-step guide for this phase

## Step 1 — Ensure the forward filter stores the right quantities

Before implementing smoothing, verify that the forward pass stores at least:

- filtered probabilities \(\alpha_{t|t}(j)\),
- predicted probabilities \(\alpha_{t+1|t}(j)\) or equivalently \(\alpha_{t|t-1}(j)\) aligned correctly across time.

### Why
The backward recursion depends directly on these quantities.

### Output of this step
A forward-pass result structure rich enough to support smoothing.

---

## Step 2 — Define the terminal smoothed distribution

Set

\[
\gamma_T(j)=\alpha_{T|T}(j).
\]

### Why
There is no future information after time \(T\).

### Output of this step
The starting point for the backward pass.

---

## Step 3 — Move backward from \(T-1\) to \(1\)

For each time \(t\), compute \(\gamma_t(i)\) for every regime \(i\) using:

\[
\gamma_t(i)
=
\alpha_{t|t}(i)
\sum_{j=1}^K
\frac{p_{ij}\,\gamma_{t+1}(j)}{\alpha_{t+1|t}(j)}.
\]

### Why
This combines:
- local filtered evidence at time \(t\),
- future information encoded in \(\gamma_{t+1}\),
- and the transition model linking time \(t\) to time \(t+1\).

### Output of this step
A full vector of smoothed probabilities at time \(t\).

---

## Step 4 — Repeat until the start of the sample

Continue until all times \(t=1,\dots,T\) have a smoothed regime distribution.

### Output of this step
A full matrix of smoothed probabilities over time.

---

## Step 5 — Compare filtering and smoothing

Once smoothing is available, compare:

- \(\alpha_{t|t}(j)\),
- \(\gamma_t(j)\).

### What to expect
In many cases:
- smoothing is more decisive than filtering,
- smoothing adjusts beliefs near regime changes,
- smoothing aligns better with the true hidden path on synthetic data.

### Why
This is the whole point of using future data retrospectively.

### Output of this step
A qualitative understanding of what the smoother adds beyond filtering.

---

## 14. Mathematical derivation sketch

It is useful to understand where the recursion comes from, at least conceptually.

You want:

\[
\Pr(S_t=i \mid y_{1:T}).
\]

Start from the filtered probability at time \(t\):

\[
\Pr(S_t=i \mid y_{1:t}).
\]

Then incorporate future observations \(y_{t+1:T}\).  
Because of the Markov structure, the future influences \(S_t\) only through the transition to \(S_{t+1}\).

So the full-sample posterior at time \(t\) can be obtained by combining:

- the filtered belief at time \(t\),
- transition probabilities to possible next states,
- the effect of future information on those next states.

This is exactly what the backward recursion formalizes.

You do not need the full derivation memorized for implementation, but you should understand that it is a consequence of:

- Bayes’ rule,
- the Markov property,
- and the factorization structure of the hidden-state model.

---

## 15. Normalization and numerical consistency

In exact arithmetic, the smoothing recursion should produce valid probabilities automatically.  
But in practical computation, numerical errors can accumulate.

So conceptually, you should expect and verify:

\[
\sum_{j=1}^K \gamma_t(j)=1
\qquad \text{for all } t.
\]

Also:

\[
0 \le \gamma_t(j) \le 1.
\]

These are mandatory structural properties of the smoothed probabilities.

---

## 16. What to store conceptually

After this phase, the inference layer should be able to retain all of the following:

- predicted probabilities,
- filtered probabilities,
- smoothed probabilities,
- log-likelihood information already produced by the forward pass.

Why store all three probability objects?

### Predicted probabilities
Needed inside the backward recursion and later for pairwise transition posteriors.

### Filtered probabilities
Needed as the base distributions being corrected by future information.

### Smoothed probabilities
Needed for regime interpretation, visualization, and later EM updates.

---

## 17. How smoothing behaves in practice

## Case 1 — Long stable regime
If the system remains in one regime for a long stretch, filtering and smoothing may look very similar in the interior of that stretch.

### Why
The current and nearby observations already provide strong information.

---

## Case 2 — Near a regime change
Smoothing often differs most strongly from filtering around transition points.

### Why
Future observations may reveal that a regime change had already occurred slightly earlier than the filter could confidently detect in real time.

---

## Case 3 — Ambiguous observations
If one observation is individually ambiguous but later observations strongly support one regime, smoothing can become much more decisive than filtering at that earlier time.

### Why
The smoother pools information across time.

---

## 18. What you should expect on simulated data

Since the true hidden path is known in synthetic experiments, smoothing should usually show the following pattern:

- smoothed probabilities align better with the true hidden regimes than filtered probabilities,
- the difference is especially visible near regime boundaries,
- smoothed posteriors are often less noisy and more coherent over time.

This is not a formal guarantee of perfect classification.  
If regimes are weakly separated, smoothing may still remain uncertain.  
But on average it should provide a more informed posterior than filtering alone.

---

## 19. Why this phase is essential for EM later

In later phases, you will estimate parameters with latent-state methods.

For that, you need expectations under the full-sample posterior.

Filtering alone gives:

\[
\Pr(S_t=j \mid y_{1:t}),
\]

but EM needs full-sample latent expectations, which are based on:

\[
\Pr(S_t=j \mid y_{1:T})
\]

and later also pairwise transition probabilities.

So smoothing is not just a nice visualization tool.  
It is a structural prerequisite for EM.

---

## 20. Common conceptual mistakes to avoid

### Mistake 1 — Thinking smoothing replaces filtering
It does not.  
Smoothing depends on filtering and comes after it.

### Mistake 2 — Forgetting the terminal condition
The backward recursion must start from

\[
\gamma_T = \alpha_{T|T}.
\]

### Mistake 3 — Confusing predicted and filtered probabilities
The smoother needs both, and they play different roles.

### Mistake 4 — Expecting smoothing to always be sharper
In many cases it is, but if regimes are highly overlapping, full-sample posteriors can still remain diffuse.

### Mistake 5 — Treating smoothing as optional for estimation
For later EM-style estimation, smoothing is a core requirement.

---

## 21. Validation checklist for Phase 7

Once smoothing is implemented, verify the following.

### Structural checks
- [ ] \(\gamma_T = \alpha_{T|T}\),
- [ ] \(\sum_j \gamma_t(j)=1\) for all \(t\),
- [ ] all smoothed probabilities lie in \([0,1]\),
- [ ] no undefined values appear in the recursion.

### Behavioral checks
- [ ] smoothing differs from filtering mainly where future information should matter,
- [ ] regime changes are reflected more clearly in smoothed probabilities,
- [ ] in easy synthetic settings, smoothed probabilities are broadly closer to the true hidden path than filtered probabilities.

### Conceptual checks
- [ ] the backward pass uses stored forward-pass quantities rather than recomputing the entire model,
- [ ] the role of predicted probabilities in the denominator is clearly understood.

---

## 22. Minimal mathematical summary of Phase 7

Define the smoothed posterior probabilities by

\[
\gamma_t(j)=\Pr(S_t=j \mid y_{1:T}).
\]

Initialize at the end of the sample:

\[
\gamma_T(j)=\alpha_{T|T}(j).
\]

Then for \(t=T-1,\dots,1\), compute:

\[
\gamma_t(i)
=
\alpha_{t|t}(i)
\sum_{j=1}^K
\frac{p_{ij}\,\gamma_{t+1}(j)}{\alpha_{t+1|t}(j)}.
\]

This recursion combines:

- filtered probabilities at time \(t\),
- transition probabilities from \(t\) to \(t+1\),
- future information summarized by the smoothed distribution at time \(t+1\).

The result is the full-sample posterior probability of each regime at every time point.

---

## 23. Output of Phase 7

At the end of this phase, you should have:

- a conceptual specification of a backward smoothing component,
- a full time series of smoothed regime probabilities,
- a clear distinction between predicted, filtered, and smoothed probabilities,
- a validated understanding of how future information modifies past regime beliefs,
- a stronger latent-state inference layer ready for:
  - retrospective regime interpretation,
  - visualization,
  - and the next phase, where pairwise state-transition posteriors will be added for EM.