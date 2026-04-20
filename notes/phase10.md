# Phase 10 — Add Diagnostics and Trust Checks

## Goal

In this phase, turn the estimator from something that merely **runs** into something that can be **trusted, inspected, and compared**.

By the end of Phase 9, you should have a working EM estimator for the basic Gaussian Markov Switching Model.  
But a fitted hidden-state model can still fail in subtle ways even if:

- it converges numerically,
- it returns valid-looking parameters,
- and it produces plausible smoothed regime probabilities.

So Phase 10 is about adding a full **diagnostic and interpretation layer** on top of the estimation pipeline.

This phase has two purposes:

1. **correctness diagnostics** — verify that the fitted model is mathematically and numerically valid,
2. **interpretive diagnostics** — help you understand whether the fitted regimes are meaningful, stable, and credible.

This is the phase where the project stops being only an implementation exercise and becomes a real statistical tool.

---

## 1. What diagnostics mean in this project

A Markov Switching Model can fail in several ways.

It may produce:

- invalid probabilities,
- nearly degenerate variances,
- meaningless regime decompositions,
- unstable solutions across initializations,
- poor convergence behavior,
- regime labels that are mathematically valid but substantively useless.

So diagnostics are not an optional extra.  
They are a structural part of the modeling pipeline.

In this project, diagnostics should answer at least the following questions:

### Structural validity
- Is the estimated parameter set mathematically valid?

### Inference validity
- Are the filtered, smoothed, and pairwise posteriors internally coherent?

### Optimization validity
- Did EM converge in a credible way?

### Interpretive validity
- Do the regimes correspond to sensible patterns in the data?

### Stability validity
- Does the model return similar high-quality solutions under different starts?

---

## 2. Why this phase comes after estimation

Before Phase 9, most checks were about isolated inference components:

- filter normalization,
- smoother consistency,
- pairwise posterior coherence.

Now that the whole estimation loop exists, diagnostics must operate at the **fitted-model level**.

That means checking both:

- the final parameter estimates,
- and the full iterative path by which they were obtained.

So Phase 10 depends on everything before it:

- simulation,
- filtering,
- smoothing,
- pairwise posteriors,
- EM estimation.

This is the phase where all those layers are audited together.

---

## 3. The main diagnostic categories

For this phase, diagnostics should be organized into six categories.

## 3.1 Parameter validity diagnostics
Check whether the fitted parameter values define a mathematically valid model.

## 3.2 Posterior probability diagnostics
Check whether the latent-state probabilities produced by the model are coherent and normalized.

## 3.3 EM convergence diagnostics
Check whether the optimization path behaved like a valid EM run.

## 3.4 Regime interpretation diagnostics
Check whether the estimated regimes have meaningful statistical structure.

## 3.5 Multi-start stability diagnostics
Check whether different initializations lead to similar or competing solutions.

## 3.6 Summary reporting diagnostics
Package the key diagnostic outputs in a form that can be inspected and compared easily.

This decomposition should guide both the mathematical design and the code structure.

---

## 4. Parameter validity diagnostics

These are the most basic checks and should always run after fitting.

## 4.1 Initial distribution validity

The fitted initial regime distribution must satisfy:

\[
\pi_j \ge 0,
\qquad
\sum_{j=1}^K \pi_j = 1.
\]

### What to check
- every entry is nonnegative,
- the sum is numerically close to 1.

### Why
If this fails, the fitted model is invalid at the most basic level.

---

## 4.2 Transition matrix validity

The fitted transition matrix

\[
P=(p_{ij})_{i,j=1}^K
\]

must satisfy:

\[
p_{ij} \ge 0,
\qquad
\sum_{j=1}^K p_{ij}=1
\quad \text{for each row } i.
\]

### What to check
- all entries lie in \([0,1]\),
- every row sums to 1 within tolerance.

### Why
The transition matrix is the core of the hidden-state dynamics.  
A row-stochastic failure means the model is not a valid Markov chain.

---

## 4.3 Variance positivity

For every regime \(j\), the estimated variance must satisfy

\[
\sigma_j^2 > 0.
\]

### What to check
- strict positivity,
- not just numerical nonnegativity,
- absence of near-zero degenerate values unless expected by design.

### Why
A variance collapsing to zero is often a sign of:
- numerical instability,
- overfitting,
- or a regime that has effectively collapsed onto a tiny number of observations.

So this is both a structural and substantive warning sign.

---

## 4.4 Mean finiteness

Each estimated regime mean

\[
\mu_j
\]

should be finite.

### What to check
- no NaN,
- no infinity,
- values in a numerically plausible range relative to the data.

### Why
Even if formulas are simple, numerical corruption can propagate through iterative estimation.

---

## 5. Posterior probability diagnostics

Once the model is fitted, you should run one final forward/backward inference pass under the fitted parameters and verify all posterior layers.

These checks validate the fitted inference output itself.

---

## 5.1 Filtered probability normalization

For every time \(t\),

\[
\sum_{j=1}^K \Pr(S_t=j \mid y_{1:t}) = 1.
\]

### What to check
- every filtered probability vector sums to 1,
- entries stay in \([0,1]\).

### Why
This confirms that the fitted model still produces coherent filtering output.

---

## 5.2 Smoothed probability normalization

For every time \(t\),

\[
\sum_{j=1}^K \Pr(S_t=j \mid y_{1:T}) = 1.
\]

### What to check
- every smoothed probability vector sums to 1,
- entries stay in \([0,1]\).

### Why
This verifies the correctness of the full-sample latent-state posterior under the fitted model.

---

## 5.3 Pairwise posterior normalization

For every transition time \(t=2,\dots,T\),

\[
\sum_{i=1}^K \sum_{j=1}^K \xi_t(i,j)=1.
\]

### What to check
- every pairwise posterior matrix sums to 1.

### Why
Exactly one hidden transition occurred between \(t-1\) and \(t\), so its posterior distribution must be properly normalized.

---

## 5.4 Marginal consistency of pairwise posteriors

Check:

\[
\sum_{i=1}^K \xi_t(i,j)=\gamma_t(j),
\]

and

\[
\sum_{j=1}^K \xi_t(i,j)=\gamma_{t-1}(i).
\]

### Why
This confirms that:
- the pairwise transition posteriors,
- and the single-time smoothed marginals

fit together coherently.

This is one of the strongest internal consistency checks in the entire latent-state pipeline.

---

## 6. EM convergence diagnostics

The next layer is about whether the estimation process itself behaved credibly.

---

## 6.1 Log-likelihood monotonicity

A key property of EM is that the observed-data log-likelihood should not decrease from one iteration to the next, up to numerical tolerance.

If the iteration index is \(m\), then ideally:

\[
\log L(\Theta^{(m+1)}) \ge \log L(\Theta^{(m)}).
\]

### What to check
- every iteration’s log-likelihood,
- whether the sequence is nondecreasing,
- whether any decreases are tiny numerical effects or real failures.

### Why
A strong violation often means:
- an M-step bug,
- an indexing bug,
- invalid posterior computations,
- or broken parameter updates.

This is one of the most important diagnostics of the estimator.

---

## 6.2 Convergence criterion behavior

If you use log-likelihood difference as a stopping rule, inspect:

\[
\left|\log L(\Theta^{(m+1)})-\log L(\Theta^{(m)})\right|.
\]

### What to check
- whether the sequence shrinks as expected,
- whether convergence is smooth,
- whether the algorithm stalls or oscillates.

### Why
This tells you whether the stopping rule is meaningful and whether the estimator is numerically well-behaved.

---

## 6.3 Iteration count

Track how many EM iterations were required.

### What to check
- convergence in a reasonable number of iterations,
- whether the estimator often hits the maximum iteration cap,
- whether some starts converge much slower than others.

### Why
Frequent iteration-cap termination may signal:
- poor initialization,
- weak identifiability,
- or unstable update logic.

---

## 6.4 Final convergence status

Every estimation run should report whether it ended because of:

- tolerance convergence,
- iteration cap,
- invalid parameter update,
- or numerical failure.

### Why
A fitted model is not trustworthy if you cannot explain why the fitting loop stopped.

---

## 7. Regime interpretation diagnostics

Once the model is valid and converged, interpret the estimated hidden-state structure.

This is where diagnostics become substantively useful.

---

## 7.1 Regime means and variances

Inspect:

\[
\mu_1,\dots,\mu_K,
\qquad
\sigma_1^2,\dots,\sigma_K^2.
\]

### What to check
- whether regimes differ meaningfully,
- whether some regimes are nearly duplicates,
- whether one regime has implausibly extreme variance,
- whether the parameter values align with the observed data scale.

### Why
A fitted model with nearly indistinguishable regimes may be mathematically valid but substantively weak.

---

## 7.2 Posterior regime occupancy

Compute, for each regime \(j\),

\[
\sum_{t=1}^T \gamma_t(j).
\]

Optionally normalize by \(T\) to get expected posterior regime frequency.

### What to check
- whether some regimes are barely used,
- whether occupancy is highly imbalanced,
- whether one regime appears effectively empty.

### Why
A regime with negligible occupancy may indicate:
- overfitting,
- too many regimes,
- or a local optimum.

---

## 7.3 Hard state classification summary

Although the model is probabilistic, it is often useful to summarize the most probable regime at each time:

\[
\hat{S}_t = \arg\max_j \gamma_t(j).
\]

### What to check
- how often each regime is assigned,
- whether the classification path is coherent,
- whether regime switches occur implausibly often or rarely.

### Why
This provides a concrete interpretive summary of the latent sequence.

---

## 7.4 Expected regime duration

For each regime \(j\), with self-transition probability \(p_{jj}\), define the expected duration as

\[
\mathbb{E}[\text{duration in regime } j]
=
\frac{1}{1-p_{jj}}.
\]

### Why this formula appears
If the process is currently in regime \(j\), the probability of staying there one more period is \(p_{jj}\).  
So the duration in that regime follows a geometric structure, and its expectation is:

\[
\frac{1}{1-p_{jj}}.
\]

### What to check
- whether durations are finite,
- whether they are plausible relative to the time scale of the data,
- whether they reveal extremely sticky or extremely unstable regimes.

### Interpretation
- \(p_{jj}\) close to 1 implies long expected duration,
- small \(p_{jj}\) implies short-lived regimes.

This is one of the most important interpretive outputs of a Markov Switching Model.

---

## 7.5 Persistence profile

Collect the diagonal of the transition matrix:

\[
p_{11}, p_{22}, \dots, p_{KK}.
\]

### What to check
- which regimes are persistent,
- whether persistence differs sharply across regimes,
- whether estimated persistence makes sense with the smoothed posterior paths.

### Why
Regime persistence is one of the defining features of switching models and should be explicitly reported.

---

## 8. Multi-start diagnostics

A single EM fit is often not enough.

Because the likelihood surface is non-convex, different initializations may converge to:

- the same optimum,
- equivalent label-switched optima,
- or genuinely different local optima.

So multi-start comparison is a core diagnostic layer, not just a convenience.

---

## 8.1 Why multiple starts matter

EM is sensitive to initialization.  
If you only trust one run, you risk accepting:

- a poor local optimum,
- a degenerate solution,
- or an unstable regime decomposition.

So diagnostics should compare multiple runs explicitly.

---

## 8.2 What to compare across starts

For each fit, track at least:

- final log-likelihood,
- convergence status,
- number of iterations,
- fitted means,
- fitted variances,
- fitted transition matrix,
- expected regime durations,
- occupancy summaries.

### Why
Two runs may have similar likelihood but very different regime interpretations.

---

## 8.3 Label switching awareness

If two fits differ only by permuting regime labels, they are effectively the same solution.

So multi-start comparison should be done with awareness that:

- regime 1 in one fit may correspond to regime 2 in another.

### Practical diagnostic implication
Introduce a consistent reporting convention, such as ordering regimes by:
- increasing mean,
- or increasing variance,
- or another fixed criterion.

This helps make diagnostics comparable across runs.

---

## 8.4 Best-run selection

When comparing multiple starts, the primary criterion is usually the final observed-data log-likelihood.

So a standard rule is:

- choose the converged run with the highest final log-likelihood.

But this should not be the only thing reported.

### Also report
- whether competing runs are close,
- whether solutions are stable across starts,
- whether the top solution is clearly dominant.

---

## 9. Diagnostic summary object

At this stage, diagnostics should not live only as ad hoc print statements or scattered checks.

They should become a formal output layer of the project.

That means a fitted model should produce not only parameter estimates, but also:

- structural validity checks,
- convergence history,
- persistence summaries,
- occupancy summaries,
- duration summaries,
- and multi-start comparison summaries where applicable.

This is one of the most important architectural additions of Phase 10.

---

## 10. Step-by-step guide for this phase

## Step 1 — Add structural parameter checks after every fit

After the EM loop finishes, validate:

- \(\pi\) normalization,
- transition matrix row sums,
- transition entry bounds,
- variance positivity,
- finiteness of all fitted parameters.

### Deliverable
A structural validation routine for fitted parameters.

### Code changes
You should add:
- a post-fit parameter validation function,
- reusable numerical tolerance handling,
- structured diagnostic messages or error flags for invalid fits.

---

## Step 2 — Add posterior consistency checks

After running final inference under the fitted parameters, validate:

- filtered normalization,
- smoothed normalization,
- pairwise normalization,
- pairwise-to-marginal consistency.

### Deliverable
A posterior-consistency diagnostic routine.

### Code changes
You should add:
- a diagnostics layer that consumes final inference outputs,
- consistency-check helpers for vectors and matrices of posterior probabilities,
- explicit reporting of the largest normalization deviation.

---

## Step 3 — Add EM-path diagnostics

Track and inspect the full EM history:

- log-likelihood at every iteration,
- likelihood increments,
- iteration count,
- convergence reason.

### Deliverable
A convergence-diagnostics summary for every fit.

### Code changes
Your estimator result type should now include, or be extended to include:
- `log_likelihood_history`,
- `likelihood_deltas`,
- `iterations`,
- `converged`,
- `stop_reason`.

This is one of the most important concrete code deliverables of Phase 10.

---

## Step 4 — Add regime interpretation summaries

Compute and report:

- regime means,
- regime variances,
- posterior occupancies,
- expected durations,
- persistence profile,
- optionally hard regime classification summaries.

### Deliverable
An interpretable regime-summary layer.

### Code changes
Add a summary object or reporting layer that derives:
- occupancy by regime,
- self-transition probabilities,
- expected durations,
- classification counts,
from the fitted model and final posterior outputs.

---

## Step 5 — Add multi-start comparison support

If your estimator already supports multiple starts or will soon support them, diagnostics must compare runs systematically.

### Deliverable
A multi-start comparison table or structured summary.

### Code changes
You should add:
- a run-level summary type,
- an aggregate result type for multiple runs,
- ranking logic by final log-likelihood,
- consistent regime-ordering logic for comparability.

Even if full multi-start fitting is not yet implemented, Phase 10 should prepare the diagnostic design for it.

---

## Step 6 — Add a “trust report” for every fitted model

At the end of a fit, the model should be able to produce a concise summary answering:

- Is the fit valid?
- Did EM converge?
- Were constraints respected?
- Are the regimes nondegenerate?
- How persistent are the regimes?
- How sensitive is the result to initialization?

### Deliverable
A compact fitted-model diagnostic summary.

### Code changes
Add a fitted-model report layer or diagnostics struct that aggregates:
- validity flags,
- warnings,
- summary statistics,
- interpretive metrics.

This should become part of the standard output surface of the project.

---

## 11. Suggested diagnostic metrics

For a fitted model, useful scalar summaries include:

### Parameter validity
- maximum row-sum deviation of \(P\),
- minimum estimated variance,
- minimum / maximum parameter finiteness checks.

### Posterior validity
- maximum deviation of filtered normalization from 1,
- maximum deviation of smoothed normalization from 1,
- maximum deviation of pairwise normalization from 1,
- maximum marginal-consistency error.

### EM behavior
- initial log-likelihood,
- final log-likelihood,
- minimum likelihood increment,
- largest negative likelihood increment, if any,
- number of iterations.

### Regime interpretation
- posterior occupancy share per regime,
- expected duration per regime,
- self-transition probabilities,
- hard classification counts.

These scalar diagnostics are especially useful for automated tests and comparisons across runs.

---

## 12. Warnings the diagnostics layer should be able to emit

A good diagnostics layer should not only compute summaries; it should also raise warnings.

Examples include:

### Warning: invalid transition matrix
If any row sum deviates too much from 1.

### Warning: near-zero variance
If a regime variance is extremely small.

### Warning: nearly unused regime
If posterior occupancy of a regime is very small.

### Warning: EM non-monotonicity
If the log-likelihood decreases beyond tolerance.

### Warning: suspicious persistence
If \(p_{jj}\) is extremely close to 1, implying an enormous expected duration that may reflect degeneracy.

### Warning: unstable solution across starts
If different runs converge to substantially different high-likelihood solutions.

This warning system is an important practical deliverable of the phase.

---

## 13. How Phase 10 connects to later phases

Phase 10 is not the end of the project.  
It prepares the next major steps.

### Connection to initialization and robustness
The multi-start diagnostics directly feed into later work on initialization strategy.

### Connection to model selection
Occupancy, degeneracy, and cross-run instability often signal whether \(K\) is too large.

### Connection to extensions
When you later move to switching regression or switching autoregression, the same diagnostic architecture should still apply.

So Phase 10 should be designed generically enough that diagnostics remain reusable as the model family grows.

---

## 14. Common conceptual mistakes to avoid

### Mistake 1 — Thinking convergence implies correctness
A converged EM run can still produce a poor or degenerate solution.

### Mistake 2 — Reporting only fitted parameters
A hidden-state model needs posterior and persistence summaries too.

### Mistake 3 — Ignoring multiple starts
For non-convex latent-variable models, this is often unsafe.

### Mistake 4 — Treating expected durations as optional
They are a central interpretation of the fitted Markov chain.

### Mistake 5 — Hiding diagnostics inside debugging output
Diagnostics should be first-class structured outputs, not scattered console checks.

---

## 15. Minimal mathematical summary of Phase 10

For a fitted model, verify:

### Parameter constraints
\[
\pi_j \ge 0, \qquad \sum_j \pi_j = 1,
\]
\[
p_{ij}\ge 0, \qquad \sum_j p_{ij}=1,
\]
\[
\sigma_j^2 > 0.
\]

### Posterior normalization
\[
\sum_j \Pr(S_t=j \mid y_{1:t}) = 1,
\]
\[
\sum_j \Pr(S_t=j \mid y_{1:T}) = 1,
\]
\[
\sum_{i,j} \xi_t(i,j)=1.
\]

### EM monotonicity
\[
\log L(\Theta^{(m+1)}) \ge \log L(\Theta^{(m)})
\quad \text{up to tolerance}.
\]

### Expected regime duration
For each regime \(j\),
\[
\mathbb{E}[\text{duration in regime } j]
=
\frac{1}{1-p_{jj}}.
\]

This formula summarizes persistence of the fitted hidden Markov chain and should be part of the standard post-fit interpretation.

---

## 16. Deliverables of Phase 10

By the end of this phase, you should have:

### Mathematical deliverables
- a formal diagnostic framework for parameter validity,
- posterior normalization and consistency checks,
- EM convergence diagnostics,
- regime persistence and expected-duration summaries,
- multi-start comparison logic.

### Interpretation deliverables
- regime means and variances summary,
- posterior occupancy summary,
- expected regime duration summary,
- persistence profile,
- warnings for degenerate or weakly used regimes.

### Code-structure deliverables
You should add or revise, where appropriate:

- a dedicated diagnostics module / file boundary,
- a fitted-model diagnostics struct,
- parameter-validation helpers,
- posterior-consistency helpers,
- EM-history summary support,
- expected-duration computation,
- occupancy and persistence summary computation,
- warning / alert generation,
- multi-run comparison result types,
- regime-ordering logic for cross-run comparability.

### Result-object deliverables
Your fitted estimator output should now include, or make easily derivable:

- fitted parameters,
- final posterior summaries,
- log-likelihood history,
- convergence metadata,
- structural validity flags,
- interpretive summaries,
- diagnostic warnings.

### Trust deliverable
A fitted model that can be inspected and justified, not just executed.

---

## 17. Minimal final summary

Phase 10 turns the estimator into a trustworthy modeling pipeline.

The central idea is:

\[
\boxed{\text{A fitted hidden-state model must be validated, interpreted, and stress-checked.}}
\]

This phase should end with a diagnostics layer that tells you:

- whether the fit is mathematically valid,
- whether EM behaved properly,
- whether the regimes are interpretable,
- how persistent those regimes are,
- and whether the solution is stable across initializations.

That is the point where the model becomes something you can actually use with confidence.