# Phase 5 — Build the Log-Likelihood Formally

## Goal

In this phase, isolate and formalize the **log-likelihood** of the Markov Switching Model.

At this point, you already have:

- the mathematical model specification,
- the generative interpretation,
- the emission model,
- the forward filter recursion.

Now you want to understand, with complete precision, **what quantity is actually being optimized** when the model is fitted.

This phase is about turning the filtering recursion into a **statistical objective function**.

That objective function is the log-likelihood of the observed data under the model parameters.

---

## 1. What the likelihood means in this model

Suppose the observed data are

$$
y_1,\dots,y_T.
$$

The model depends on parameters

$$
\Theta = (\pi, P, \theta_1,\dots,\theta_K),
$$

where, in the first Gaussian switching model,

$$
\theta_j = (\mu_j,\sigma_j^2).
$$

The likelihood answers the question:

> How plausible is the entire observed sample \(y_{1:T}\) under the parameter set \(\Theta\)?

Formally, the likelihood is

$$
L(\Theta) = f(y_1,\dots,y_T;\Theta),
$$

and the log-likelihood is

$$
\log L(\Theta) = \log f(y_1,\dots,y_T;\Theta).
$$

In hidden-state models, the log-likelihood is usually preferred because:

- products over many time points become sums,
- it is numerically more stable,
- it is the standard objective in maximum likelihood estimation.

---

## 2. Why this phase must be separate

It is easy to think that once the filter is implemented, the likelihood is “already done.”  
Conceptually, that is not enough.

You want a separate phase for the likelihood because you need to understand:

- what the filter is computing statistically,
- why the predictive densities \(c_t\) are the correct likelihood contributions,
- why the hidden regimes do not appear explicitly in the observed-data likelihood,
- how the inference recursion becomes an estimation objective.

This phase is the bridge between:

- **inference under fixed parameters**,
- and later **parameter estimation**.

---

## 3. Observed-data likelihood vs complete-data likelihood

Before defining the likelihood formally, distinguish two different objects.

## 3.1 Complete-data likelihood

If the hidden regimes were observed, the likelihood would be based on both:

- the hidden path \(S_{1:T}\),
- the observations \(y_{1:T}\).

That joint object factorizes as

$$
\Pr(S_{1:T}, y_{1:T})
=
\pi_{S_1} f(y_1 \mid S_1)
\prod_{t=2}^T p_{S_{t-1},S_t} f(y_t \mid S_t).
$$

This is the **complete-data** structure.

But in the actual problem, the regimes are hidden.

---

## 3.2 Observed-data likelihood

What you observe is only

$$
y_{1:T}.
$$

So the likelihood relevant for estimation is the **observed-data likelihood**:

$$
L(\Theta) = f(y_{1:T};\Theta).
$$

This is obtained by integrating out, or summing over, all possible hidden regime paths.

That is why the problem is nontrivial:

- the hidden states matter,
- but they are not observed directly.

The forward filter is what makes this computation tractable.

---

## 4. Likelihood factorization over time

The key identity is the standard time-series factorization:

$$
f(y_1,\dots,y_T;\Theta)
=
f(y_1;\Theta)\prod_{t=2}^T f(y_t \mid y_{1:t-1};\Theta).
$$

Taking logs gives

$$
\log L(\Theta)
=
\log f(y_1;\Theta)
+
\sum_{t=2}^T \log f(y_t \mid y_{1:t-1};\Theta).
$$

Equivalently, using the same notation for all \(t\),

$$
\log L(\Theta)
=
\sum_{t=1}^T \log f(y_t \mid y_{1:t-1};\Theta),
$$

where for \(t=1\), the “past” is empty and the first factor is understood as the marginal density of \(y_1\).

This is the fundamental decomposition of the observed-data likelihood.

---

## 5. Why the predictive density is a regime mixture

At time \(t\), the regime \(S_t\) is unknown.

So the conditional density of \(y_t\) given the past must average over all possible current regimes:

$$
f(y_t \mid y_{1:t-1};\Theta)
=
\sum_{j=1}^K
f(y_t \mid S_t=j;\theta_j)\,
\Pr(S_t=j \mid y_{1:t-1};\Theta).
$$

Using the notation from Phase 4:

- the regime-conditional density is
  $$
  f_j(y_t)=f(y_t \mid S_t=j;\theta_j),
  $$
- the predicted regime probability is
  $$
  \alpha_{t|t-1}(j)=\Pr(S_t=j \mid y_{1:t-1}).
  $$

So the predictive density becomes

$$
c_t
=
f(y_t \mid y_{1:t-1})
=
\sum_{j=1}^K f_j(y_t)\alpha_{t|t-1}(j).
$$

This scalar \(c_t\) is the one-step likelihood contribution at time \(t\).

---

## 6. The role of the forward filter in likelihood evaluation

The forward filter computes exactly the quantities needed to evaluate the log-likelihood.

At each time \(t\), it gives you:

1. predicted regime probabilities,
2. regime-conditional observation densities,
3. the predictive density
   $$
   c_t = f(y_t \mid y_{1:t-1}),
   $$
4. filtered regime probabilities.

For the likelihood, the critical object is \(c_t\).

The filter therefore does two things simultaneously:

- it updates latent regime beliefs,
- it evaluates the observed-data likelihood recursively.

This is one of the central conceptual facts of hidden-state models.

---

## 7. Step-by-step derivation of the log-likelihood

## Step 1 — Start from the joint density of the observations

The observed-data likelihood is

$$
L(\Theta)=f(y_{1:T};\Theta).
$$

By the chain rule of probability,

$$
f(y_{1:T};\Theta)
=
\prod_{t=1}^T f(y_t \mid y_{1:t-1};\Theta).
$$

This is true for any stochastic process, not just Markov switching models.

---

## Step 2 — Express each factor using hidden regimes

At time \(t\), the current regime is hidden, so apply the law of total probability:

$$
f(y_t \mid y_{1:t-1};\Theta)
=
\sum_{j=1}^K
f(y_t, S_t=j \mid y_{1:t-1};\Theta).
$$

Now factor each term:

$$
f(y_t, S_t=j \mid y_{1:t-1};\Theta)
=
f(y_t \mid S_t=j, y_{1:t-1};\Theta)\Pr(S_t=j \mid y_{1:t-1};\Theta).
$$

In the basic Gaussian switching model, once the current regime is known, the current observation depends on the regime parameters and not directly on the past. So:

$$
f(y_t \mid S_t=j, y_{1:t-1};\Theta)
=
f(y_t \mid S_t=j;\theta_j).
$$

Therefore,

$$
f(y_t \mid y_{1:t-1};\Theta)
=
\sum_{j=1}^K
f(y_t \mid S_t=j;\theta_j)\Pr(S_t=j \mid y_{1:t-1};\Theta).
$$

This is the predictive density formula used in Phase 4.

---

## Step 3 — Define the one-step contribution \(c_t\)

Let

$$
c_t
=
f(y_t \mid y_{1:t-1};\Theta).
$$

Then:

$$
c_t = \sum_{j=1}^K f_j(y_t)\alpha_{t|t-1}(j).
$$

This quantity is:

- the model’s density for the current observation,
- a normalization constant for the Bayesian update,
- the likelihood contribution from time \(t\).

---

## Step 4 — Build the full likelihood

Because the sample density factorizes sequentially,

$$
L(\Theta)=\prod_{t=1}^T c_t.
$$

Taking logs gives

$$
\log L(\Theta)=\sum_{t=1}^T \log c_t.
$$

This is the observed-data log-likelihood of the Markov Switching Model.

This formula is one of the most important in the entire project.

---

## 8. Initialization and the first time step

At \(t=1\), there is no previous observation history, so the predictive regime probabilities are just the initial distribution:

$$
\alpha_{1|0}(j)=\pi_j.
$$

Therefore the first predictive density is

$$
c_1=f(y_1)
=
\sum_{j=1}^K f_j(y_1)\pi_j.
$$

The first contribution to the log-likelihood is

$$
\log c_1.
$$

This means the initial distribution \(\pi\) is part of the likelihood and should not be treated as an afterthought.

---

## 9. Interpretation of the log-likelihood

The log-likelihood measures how well the entire model explains the observed sample.

In this model, “explains” means two things simultaneously:

### Hidden-state dynamics fit
The transition matrix \(P\) should generate plausible regime evolution.

### Observation fit
The emission parameters \((\mu_j,\sigma_j^2)\) should generate plausible observation values within each regime.

The likelihood rewards parameter values that balance both effects.

For example:

- a transition matrix with unrealistic switching frequency may reduce the likelihood,
- emission parameters that assign low density to the observations may also reduce the likelihood.

So the log-likelihood is the single scalar objective that combines both parts of the model.

---

## 10. What the log-likelihood is *not*

It is important not to confuse the observed-data log-likelihood with other quantities.

## Not the same as posterior regime probability
A high posterior probability for one regime at one time point does not mean the overall log-likelihood is high.

## Not the same as complete-data likelihood
The complete-data likelihood conditions on a particular hidden path.  
The observed-data likelihood sums over all possible hidden paths implicitly through the filter.

## Not the same as a sum of emission log-densities alone
You cannot just pick one regime at each time and add Gaussian log-densities.  
The regime uncertainty must be accounted for through mixture weighting and Markov dynamics.

---

## 11. Step-by-step guide for this phase

## Step 1 — Fix the statistical objective formally

Write down, in your project notes, that the quantity of interest for estimation is

$$
\log L(\Theta)=\sum_{t=1}^T \log c_t,
\qquad
c_t=f(y_t \mid y_{1:t-1};\Theta).
$$

This is the objective your model will later maximize.

### Output of this step
A formal definition of the estimation target.

---

## Step 2 — Define each one-step contribution explicitly

For each time \(t\), define

$$
c_t = \sum_{j=1}^K f_j(y_t)\alpha_{t|t-1}(j).
$$

This should be treated as a first-class object in your notes and later in the implementation design.

### Why
It is the central scalar linking:
- filtering,
- likelihood,
- normalization,
- estimation.

### Output of this step
A fixed mathematical definition of one-step predictive density.

---

## Step 3 — Connect the filter to the likelihood

Recognize that the forward filter is the mechanism that computes the \(\alpha_{t|t-1}(j)\) needed inside \(c_t\).

So the likelihood is not a separate disconnected formula.  
It is built directly from the recursive filtering structure.

### Output of this step
A clear understanding that filtering and likelihood evaluation are inseparable.

---

## Step 4 — Separate per-time contributions from the total sum

Conceptually distinguish:

### Local quantity
\[
c_t
\]
or
\[
\log c_t
\]

### Global quantity
\[
\log L(\Theta)=\sum_{t=1}^T \log c_t
\]

This distinction matters because later diagnostics often work time-point by time-point.

### Output of this step
A decomposition of the objective into interpretable local contributions.

---

## Step 5 — Think of the log-likelihood as reusable infrastructure

Later phases will need to use the log-likelihood in different ways:

- direct numerical optimization,
- EM convergence monitoring,
- model comparison,
- debugging,
- AIC and BIC computation.

So in your conceptual design, the log-likelihood should be treated as a reusable output of the filtering procedure, not as a side effect.

### Output of this step
A stable conceptual role for the likelihood inside the project.

---

## 12. Why logs are used instead of raw likelihoods

The likelihood is a product:

$$
L(\Theta)=\prod_{t=1}^T c_t.
$$

For long time series, multiplying many small positive numbers can lead to severe numerical underflow.

Taking logs turns this into:

$$
\log L(\Theta)=\sum_{t=1}^T \log c_t.
$$

This is better because:

- sums are numerically safer than products,
- optimization routines usually work with log-likelihoods,
- diagnostics become easier,
- each time point contributes additively.

Even though this phase is still conceptual, you should already think in terms of the log-likelihood, not the raw likelihood.

---

## 13. Local interpretation of \(\log c_t\)

Each term

$$
\log c_t
\]

measures how well the model explains observation \(y_t\) given the past observations.

Interpretation:

- if \(c_t\) is large, the observation was relatively unsurprising,
- if \(c_t\) is small, the observation was relatively surprising,
- if many observations are surprising, the total log-likelihood becomes low.

So the log-likelihood can be seen as the cumulative fit of the model to the sequence over time.

---

## 14. Relationship between filtering update and likelihood normalization

Recall the filtering update:

$$
\alpha_{t|t}(j)=\frac{f_j(y_t)\alpha_{t|t-1}(j)}{c_t}.
$$

The denominator \(c_t\) is exactly the predictive density.

This means the same quantity plays two simultaneous roles:

### Role 1 — Statistical
It contributes to the observed-data likelihood.

### Role 2 — Bayesian
It normalizes the posterior regime probabilities so that they sum to 1.

This dual role is a fundamental structural feature of the model.

---

## 15. What should be stored conceptually

Even before implementation, decide what quantities the forward pass should conceptually produce and retain.

At minimum:

- predicted probabilities \(\alpha_{t|t-1}(j)\),
- filtered probabilities \(\alpha_{t|t}(j)\),
- predictive densities \(c_t\),
- log-contributions \(\log c_t\),
- total log-likelihood.

### Why this matters
Later phases will require:
- the total log-likelihood for estimation,
- the filtered probabilities for smoothing,
- the per-time contributions for debugging and diagnostics.

So the likelihood should be thought of both:
- globally,
- and as a sequence of local terms.

---

## 16. Sanity checks for the likelihood layer

Before moving to parameter estimation, the likelihood logic should satisfy several conceptual checks.

## Check A — Positivity of predictive density
For every \(t\),

$$
c_t > 0.
$$

Otherwise \(\log c_t\) is undefined.

---

## Check B — Finite log-likelihood
The total

$$
\log L(\Theta)=\sum_{t=1}^T \log c_t
$$

should be finite under valid parameter values.

---

## Check C — Dependence on both transition and emission parameters
If you change:
- the transition matrix,
- or the emission parameters,

the log-likelihood should in general change.

This confirms that the objective incorporates the full model.

---

## Check D — Decomposability
The total log-likelihood should equal the sum of the per-time contributions.

This is both a theoretical identity and a practical debugging aid.

---

## 17. Intuition through simple scenarios

## Scenario 1 — Observation strongly matches a likely regime
Suppose a regime already has high predicted probability and also assigns high density to the current observation.

Then \(c_t\) will tend to be relatively large.

### Interpretation
The model predicted the observation well.

---

## Scenario 2 — Observation mismatches likely regimes
Suppose the predicted regime probabilities are concentrated on regimes that assign low density to the current observation.

Then \(c_t\) will be relatively small.

### Interpretation
The model is surprised by the observation.

---

## Scenario 3 — Observation fits a low-probability regime
Suppose a regime that was unlikely before seeing \(y_t\) actually explains \(y_t\) very well.

Then \(c_t\) may still be moderate, and the posterior probabilities may shift strongly after the update.

### Interpretation
The likelihood contribution reflects the mixture of prior regime uncertainty and current-data evidence.

---

## 18. Common conceptual mistakes to avoid

### Mistake 1 — Thinking the likelihood comes from one chosen regime path
The observed-data likelihood sums over hidden-regime uncertainty.  
It is not based on a single hard regime assignment.

### Mistake 2 — Confusing \(c_t\) with a posterior probability
The predictive density \(c_t\) is a density of the observation, not a probability of a regime.

### Mistake 3 — Using filtered instead of predicted probabilities in \(c_t\)
The predictive density must be computed using the pre-update probabilities:

\[
\alpha_{t|t-1}(j),
\]

not the post-update probabilities \(\alpha_{t|t}(j)\).

### Mistake 4 — Treating the first time step differently in an ad hoc way
It is different only because it uses \(\pi\) rather than a previous filtered distribution, but it belongs to the same likelihood structure.

### Mistake 5 — Forgetting that the likelihood is an observed-data quantity
The hidden states are latent and must be integrated out.

---

## 19. Minimal mathematical summary of Phase 5

The observed-data likelihood factorizes as

$$
L(\Theta)=f(y_{1:T};\Theta)=\prod_{t=1}^T c_t,
$$

where

$$
c_t=f(y_t \mid y_{1:t-1};\Theta)
=
\sum_{j=1}^K f_j(y_t)\alpha_{t|t-1}(j).
$$

Therefore the log-likelihood is

$$
\log L(\Theta)=\sum_{t=1}^T \log c_t.
$$

At \(t=1\),

$$
\alpha_{1|0}(j)=\pi_j,
\qquad
c_1=\sum_{j=1}^K f_j(y_1)\pi_j.
$$

For \(t\ge 2\), the \(\alpha_{t|t-1}(j)\) are obtained from the forward prediction step.

This is the formal statistical objective of the basic Markov Switching Model.

---

## 20. Output of Phase 5

At the end of this phase, you should have:

- a precise definition of the observed-data likelihood,
- a clear distinction between complete-data and observed-data likelihood,
- a formal definition of the one-step predictive density \(c_t\),
- the identity
  $$
  \log L(\Theta)=\sum_{t=1}^T \log c_t,
  $$
- a clear understanding that the forward filter is not only a state-inference algorithm, but also the mechanism that computes the likelihood recursively,
- a stable conceptual basis for Phase 6, where the forward filter will be validated carefully on simulated data.