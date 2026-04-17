# Phase 4 — Implement the Forward Filter

## Goal

In this phase, build the **first real inference mechanism** of the Markov Switching Model: the **forward filter**.

This is the algorithm that processes the observed series sequentially and, at each time \(t\), updates your belief about which regime is currently active.

More precisely, the filter should allow you to compute, for every time point:

- the **predicted regime probabilities**
  \[
  \Pr(S_t=j \mid y_{1:t-1}),
  \]
- the **filtered regime probabilities**
  \[
  \Pr(S_t=j \mid y_{1:t}),
  \]
- the **one-step predictive density**
  \[
  f(y_t \mid y_{1:t-1}),
  \]
- and the cumulative **log-likelihood**
  \[
  \log L(\Theta) = \sum_{t=1}^T \log f(y_t \mid y_{1:t-1};\Theta).
  \]

This phase is the transition from:

- **model definition** and **simulation**
  
to

- **probabilistic inference from observed data**.

---

## 1. What the forward filter does conceptually

At each time \(t\), the model has two sources of information:

### A. Prior information from the hidden-state dynamics
From the Markov chain, you already have beliefs about how likely each regime is before seeing the current observation.

These are the **predicted probabilities**.

### B. Information from the current observation
From the emission model, you can measure how plausible the observed value \(y_t\) is under each regime.

These are the **regime-conditional likelihoods** or **emission densities**.

The forward filter combines these two ingredients using Bayes’ rule.

So the logic at each time step is:

1. predict which regime is likely **before** seeing \(y_t\),
2. evaluate how well each regime explains \(y_t\),
3. update regime probabilities **after** seeing \(y_t\),
4. extract the contribution of \(y_t\) to the total likelihood.

---

## 2. Why this phase is foundational

The forward filter is not just one algorithm among many.  
It is the core engine behind almost everything that follows.

It is needed for:

- likelihood evaluation,
- regime inference,
- smoothing,
- EM estimation,
- model diagnostics,
- later extensions such as switching regression or switching autoregression.

If the forward filter is correct and well-structured, the rest of the project becomes much more manageable.  
If it is incorrect, every later phase will inherit the error.

---

## 3. The probability objects you must define

Before implementing anything, define the main forward-probability objects clearly.

## 3.1 Predicted probabilities

These are the probabilities of the current regime before incorporating the current observation:

\[
\alpha_{t|t-1}(j)
=
\Pr(S_t=j \mid y_{1:t-1}).
\]

Interpretation:

- this is the model’s belief about the regime at time \(t\),
- using only past observations \(y_1,\dots,y_{t-1}\).

These probabilities come from the previous filtered probabilities and the transition matrix.

---

## 3.2 Filtered probabilities

These are the probabilities of the current regime after incorporating the current observation:

\[
\alpha_{t|t}(j)
=
\Pr(S_t=j \mid y_{1:t}).
\]

Interpretation:

- this is the updated belief about the regime after seeing \(y_t\),
- it combines prediction from the Markov chain with evidence from the observation model.

These are the main posterior regime probabilities produced by the filter.

---

## 3.3 Emission densities

For each regime \(j\), define:

\[
f_j(y_t)
=
f(y_t \mid S_t=j;\theta_j).
\]

For the Gaussian switching model:

\[
f_j(y_t)
=
\frac{1}{\sqrt{2\pi\sigma_j^2}}
\exp\!\left(
-\frac{(y_t-\mu_j)^2}{2\sigma_j^2}
\right).
\]

These values tell you how compatible the observation \(y_t\) is with each candidate regime.

---

## 3.4 Predictive density

The unconditional density of \(y_t\) given the past is

\[
f(y_t \mid y_{1:t-1})
=
\sum_{j=1}^K f_j(y_t)\,\alpha_{t|t-1}(j).
\]

Interpretation:

- before seeing which regime is active, the observation comes from a mixture over all regimes,
- the mixture weights are the predicted probabilities.

This quantity is critical because:

- it normalizes the Bayesian update,
- its logarithm contributes to the total likelihood.

---

## 4. The structure of the recursion

The filter is a recursive algorithm.  
At each time \(t\), it takes the filtered probabilities from time \(t-1\) and transforms them into filtered probabilities at time \(t\).

The recursion has two main stages:

## Stage A — Prediction
Project the previous regime probabilities forward using the transition matrix.

## Stage B — Update
Use the current observation to update those predicted probabilities.

This is the standard predict-update structure of Bayesian filtering.

---

## 5. Step-by-step forward recursion

## Step 1 — Initialize at \(t=1\)

At the beginning of the sample, there is no previous observation history.  
So the prior belief about the regime is just the initial distribution:

\[
\alpha_{1|0}(j) = \Pr(S_1=j) = \pi_j.
\]

This is the **predicted probability** for time \(t=1\).

Then evaluate the emission density for each regime:

\[
f_j(y_1) = f(y_1 \mid S_1=j;\theta_j).
\]

Now compute the predictive density of the first observation:

\[
f(y_1)
=
\sum_{j=1}^K f_j(y_1)\pi_j.
\]

Then update using Bayes’ rule:

\[
\alpha_{1|1}(j)
=
\frac{f_j(y_1)\pi_j}{f(y_1)}.
\]

### Interpretation
At \(t=1\), the model starts from the prior \(\pi\), then uses the observation \(y_1\) to update the regime probabilities.

### Output of this step
You obtain:
- predicted probabilities at time 1,
- filtered probabilities at time 1,
- the first likelihood contribution.

---

## Step 2 — Predict the regime at time \(t \ge 2\)

For each regime \(j\), compute:

\[
\alpha_{t|t-1}(j)
=
\Pr(S_t=j \mid y_{1:t-1})
=
\sum_{i=1}^K
\Pr(S_t=j \mid S_{t-1}=i)\Pr(S_{t-1}=i \mid y_{1:t-1}).
\]

Using the notation of the model:

\[
\alpha_{t|t-1}(j)
=
\sum_{i=1}^K p_{ij}\,\alpha_{t-1|t-1}(i).
\]

### Interpretation
This is just the law of total probability applied to the hidden Markov chain.

You are saying:

- the model could have been in any regime \(i\) at time \(t-1\),
- each such regime transitions to \(j\) with probability \(p_{ij}\),
- so the total probability of regime \(j\) at time \(t\) is the weighted sum over previous regimes.

### Output of this step
A full vector of predicted probabilities at time \(t\).

---

## Step 3 — Evaluate the current observation under each regime

For each regime \(j\), compute:

\[
f_j(y_t)=f(y_t \mid S_t=j;\theta_j).
\]

For the Gaussian model:

\[
f_j(y_t)
=
\frac{1}{\sqrt{2\pi\sigma_j^2}}
\exp\!\left(
-\frac{(y_t-\mu_j)^2}{2\sigma_j^2}
\right).
\]

### Interpretation
This is the observation-level evidence supporting regime \(j\).

A regime gets a high value here when:
- its mean is close to the observation,
- and/or its variance makes the observation plausible.

### Output of this step
A vector of regime-specific observation densities.

---

## Step 4 — Compute the predictive density of \(y_t\)

Combine the predicted regime probabilities and the emission densities:

\[
f(y_t \mid y_{1:t-1})
=
\sum_{j=1}^K f_j(y_t)\,\alpha_{t|t-1}(j).
\]

### Interpretation
This is the model’s total density for the current observation before knowing the hidden regime.

It is a weighted average of the regime-specific densities, where the weights come from the predicted regime probabilities.

### Why this matters
This quantity plays two roles:
1. it is the denominator in Bayes’ rule,
2. it contributes to the overall log-likelihood.

### Output of this step
A scalar predictive density for \(y_t\).

---

## Step 5 — Update the regime probabilities

Now apply Bayes’ rule:

\[
\alpha_{t|t}(j)
=
\Pr(S_t=j \mid y_{1:t})
=
\frac{
f_j(y_t)\,\alpha_{t|t-1}(j)
}{
f(y_t \mid y_{1:t-1})
}.
\]

### Interpretation
The posterior probability of regime \(j\) is proportional to:

- how plausible regime \(j\) was before seeing \(y_t\),
- times how well regime \(j\) explains \(y_t\).

This is the central Bayesian update step of the filter.

### Output of this step
A vector of filtered probabilities at time \(t\).

---

## Step 6 — Accumulate the log-likelihood

At each time \(t\), the contribution to the log-likelihood is:

\[
\log f(y_t \mid y_{1:t-1}).
\]

So the total sample log-likelihood is

\[
\log L(\Theta)
=
\sum_{t=1}^T \log f(y_t \mid y_{1:t-1};\Theta).
\]

### Interpretation
This is the likelihood of the observed series under the model after integrating out the hidden regimes.

### Output of this step
A cumulative log-likelihood value.

---

## 6. Compact mathematical form of the recursion

For \(t=1\):

\[
\alpha_{1|0}(j)=\pi_j,
\]

\[
c_1 = \sum_{j=1}^K f_j(y_1)\pi_j,
\]

\[
\alpha_{1|1}(j)=\frac{f_j(y_1)\pi_j}{c_1}.
\]

For \(t=2,\dots,T\):

\[
\alpha_{t|t-1}(j)=\sum_{i=1}^K p_{ij}\alpha_{t-1|t-1}(i),
\]

\[
c_t=\sum_{j=1}^K f_j(y_t)\alpha_{t|t-1}(j),
\]

\[
\alpha_{t|t}(j)=\frac{f_j(y_t)\alpha_{t|t-1}(j)}{c_t}.
\]

Then:

\[
\log L(\Theta)=\sum_{t=1}^T \log c_t.
\]

This is the forward filter in its essential form.

---

## 7. Interpretation of the recursion

The recursion can be read as:

### Predict
Use the Markov chain to forecast the next regime probabilities.

### Score
Use the emission model to score the current observation under each regime.

### Normalize
Convert those scores into posterior regime probabilities.

### Accumulate
Record how surprising or unsurprising the observation was under the model.

This interpretation is important because it keeps the algorithm conceptually transparent.

---

## 8. What this phase should produce architecturally

Even without writing code here, you should already think of the filter as a distinct conceptual unit.

It should take as input:

- observations \(y_{1:T}\),
- initial distribution \(\pi\),
- transition matrix \(P\),
- emission parameters \(\theta_1,\dots,\theta_K\).

It should return:

- predicted regime probabilities for each \(t\),
- filtered regime probabilities for each \(t\),
- predictive density contributions \(c_t\),
- total log-likelihood.

This is the minimal output surface of the forward filter.

---

## 9. Why predicted and filtered probabilities must be kept separate

Do not collapse the two notions.

## Predicted probabilities
\[
\Pr(S_t=j \mid y_{1:t-1})
\]

These come from the hidden-state dynamics alone.

## Filtered probabilities
\[
\Pr(S_t=j \mid y_{1:t})
\]

These incorporate the current observation.

### Why this distinction matters
If you do not preserve this distinction, it becomes difficult later to:
- derive the smoother,
- compute pairwise transition posteriors,
- reason about likelihood contributions,
- debug whether problems come from the prediction or the update step.

---

## 10. The forward filter as a hidden-state analog of recursive Bayesian updating

It is useful to understand the filter at a more abstract level.

At each time \(t\), the model performs Bayesian updating in two stages:

### Prior
\[
\Pr(S_t=j \mid y_{1:t-1})
\]

### Likelihood
\[
f(y_t \mid S_t=j)
\]

### Posterior
\[
\Pr(S_t=j \mid y_{1:t})
\]

So the filter is just repeated Bayes updating where the prior itself is produced recursively by a Markov transition model.

This is why the forward filter is simultaneously:

- an HMM forward algorithm,
- a regime inference algorithm,
- a likelihood evaluation mechanism.

---

## 11. What to verify at every time step

When you later implement the filter, several conditions should hold at every \(t\).

## Check A — Predicted probabilities sum to 1
\[
\sum_{j=1}^K \alpha_{t|t-1}(j)=1.
\]

## Check B — Filtered probabilities sum to 1
\[
\sum_{j=1}^K \alpha_{t|t}(j)=1.
\]

## Check C — Predictive density is positive
\[
c_t = f(y_t \mid y_{1:t-1}) > 0.
\]

## Check D — No negative probabilities
All predicted and filtered probabilities must lie in \([0,1]\).

## Check E — Log-likelihood remains finite
The cumulative log-likelihood should stay well-defined.

These checks are essential for debugging.

---

## 12. Intuition through simple scenarios

## Scenario 1 — Observation strongly favors one regime
Suppose one regime has a mean very close to \(y_t\) and the others do not.

Then its emission density will be much larger, so its filtered probability should increase sharply.

### Interpretation
The current observation dominates the update.

---

## Scenario 2 — Transition matrix strongly favors persistence
Suppose the current filtered probability at time \(t-1\) strongly favors regime 1 and \(p_{11}\) is very large.

Then the predicted probability at time \(t\) should also strongly favor regime 1.

### Interpretation
The Markov chain carries information forward even before seeing the next observation.

---

## Scenario 3 — Observation and transition information conflict
Suppose the predicted probability strongly favors regime 1, but the new observation fits regime 2 much better.

Then the filtered posterior is a compromise between:
- dynamic persistence,
- current-data evidence.

### Interpretation
This is exactly the kind of tension the forward filter is designed to resolve.

---

## 13. Common conceptual mistakes to avoid

### Mistake 1 — Forgetting the prediction step
The current regime probabilities are not determined solely by the current observation.  
They must first be forecast from the previous time step using the transition matrix.

### Mistake 2 — Using filtered probabilities where predicted probabilities are needed
The predictive density must be based on the pre-update probabilities
\[
\alpha_{t|t-1}(j),
\]
not the post-update probabilities.

### Mistake 3 — Treating emission densities as probabilities over regimes
The quantities \(f_j(y_t)\) are densities over observations, not posterior probabilities of regimes.

They become part of posterior regime probabilities only after normalization.

### Mistake 4 — Ignoring initialization
The filter must start from \(\pi\).  
The initial regime uncertainty is part of the model and cannot be skipped.

### Mistake 5 — Not storing intermediate results
Later phases, especially smoothing and EM, will need access to predicted and filtered probabilities across all time points.

---

## 14. Numerical perspective

Although this phase is still conceptual, you should already be aware of an important practical issue:

The predictive densities \(c_t\) can become very small, and repeated multiplication of probabilities can lead to numerical underflow.

That is one reason the log-likelihood is accumulated as

\[
\log L(\Theta)=\sum_{t=1}^T \log c_t.
\]

Later, your actual implementation may also require scaling strategies or careful use of log-space arithmetic, but conceptually the recursion remains the same.

---

## 15. How Phase 4 connects to earlier phases

## Connection to Phase 1
Phase 1 defined the model mathematically.

## Connection to Phase 2
Phase 2 used the model in the forward generative direction:
- state \(\to\) observation.

## Connection to Phase 3
Phase 3 isolated the emission density:
- observation plausibility under a candidate regime.

## What Phase 4 does
Phase 4 combines:
- the transition model from Phase 1,
- the simulation intuition from Phase 2,
- the emission model from Phase 3,

into the first real inference recursion.

This is the first time the project answers:

> “Given the observed data so far, which regime is currently most plausible?”

---

## 16. Minimal mathematical summary of Phase 4

For \(t=1\):

\[
\alpha_{1|0}(j)=\pi_j,
\qquad
c_1=\sum_{j=1}^K f_j(y_1)\pi_j,
\qquad
\alpha_{1|1}(j)=\frac{f_j(y_1)\pi_j}{c_1}.
\]

For \(t \ge 2\):

\[
\alpha_{t|t-1}(j)=\sum_{i=1}^K p_{ij}\alpha_{t-1|t-1}(i),
\]

\[
c_t=\sum_{j=1}^K f_j(y_t)\alpha_{t|t-1}(j),
\]

\[
\alpha_{t|t}(j)=\frac{f_j(y_t)\alpha_{t|t-1}(j)}{c_t}.
\]

And the log-likelihood is

\[
\log L(\Theta)=\sum_{t=1}^T \log c_t.
\]

This recursion is the complete conceptual forward filter for the first-order Gaussian Markov Switching Model.

---

## 17. Output of Phase 4

At the end of this phase, you should have a precise conceptual specification of a forward-filtering component that:

- starts from the initial regime distribution,
- propagates regime probabilities through the transition matrix,
- evaluates the current observation under each regime,
- updates regime beliefs using Bayes’ rule,
- computes the predictive density at each time,
- accumulates the total log-likelihood.

This is the first full inference engine of the project and the foundation for all later phases.