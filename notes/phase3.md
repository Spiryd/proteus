# Phase 3 — Isolate the Emission Model

## Goal

In this phase, isolate the **observation mechanism** of the Markov Switching Model and treat it as an independent conceptual component.

You are **not** yet doing filtering, smoothing, or estimation.  
You are only defining, as cleanly as possible, how an observation \(y_t\) is generated **conditional on a given regime**.

This phase matters because the hidden-state machinery and the observation machinery play different roles:

- the **Markov chain** governs how regimes evolve over time,
- the **emission model** governs what the data look like once a regime is active.

If this separation is clean, later phases become much easier to implement and debug.

---

## 1. What the emission model is

The hidden regime at time \(t\) is

$$
S_t \in \{1,\dots,K\}.
$$

Conditional on \(S_t=j\), the observation \(y_t\) is generated from a regime-specific distribution.

For the first model, that distribution is Gaussian:

$$
y_t \mid (S_t=j) \sim \mathcal N(\mu_j,\sigma_j^2).
$$

So each regime \(j\) has its own emission parameters

$$
\theta_j = (\mu_j,\sigma_j^2).
$$

The emission model is therefore the collection of regime-conditional densities

$$
f_j(y_t) = f(y_t \mid S_t=j;\theta_j).
$$

For the Gaussian case,

$$
f_j(y_t)
=
\frac{1}{\sqrt{2\pi\sigma_j^2}}
\exp\!\left(
-\frac{(y_t-\mu_j)^2}{2\sigma_j^2}
\right).
$$

This is the mathematical object you are isolating in this phase.

---

## 2. Why this phase is separate

It is tempting to think of the emission model as a trivial detail because the Gaussian density is familiar.  
That is a mistake.

Later, filtering and likelihood computation will repeatedly ask the same question:

> “If regime \(j\) were active at time \(t\), how plausible would the observed \(y_t\) be?”

The answer to that question is exactly the emission density.

So the emission model becomes the bridge between:

- latent regime probabilities,
- observed data,
- likelihood contributions,
- posterior updates.

If the emission model is unclear or inconsistent, then:

- filtered probabilities will be wrong,
- the total likelihood will be wrong,
- EM updates will be wrong,
- extensions to regression or AR switching will be much harder.

---

## 3. What you should define precisely

In this phase, fix the observation model as a formal contract.

For each regime \(j\), define:

### 3.1 Mean
$$
\mu_j \in \mathbb{R}
$$

This is the center of the Gaussian associated with regime \(j\).

### 3.2 Variance
$$
\sigma_j^2 > 0
$$

This determines dispersion in regime \(j\).

### 3.3 Conditional density
$$
f(y_t \mid S_t=j;\theta_j)
=
\mathcal N(y_t;\mu_j,\sigma_j^2)
$$

### 3.4 Log-density
Later, numerical work will almost certainly rely on the log-density:

$$
\log f(y_t \mid S_t=j;\theta_j)
=
-\frac{1}{2}\log(2\pi)
-\frac{1}{2}\log(\sigma_j^2)
-\frac{(y_t-\mu_j)^2}{2\sigma_j^2}.
$$

Even if you are not implementing numerics yet, you should already treat the log-density as part of the conceptual specification.

---

## 4. Interpretation of the Gaussian emission model

The Gaussian emission model says:

- when regime \(j\) is active,
- observations fluctuate around \(\mu_j\),
- with dispersion measured by \(\sigma_j^2\).

This creates different possible regime structures.

### Mean-switching interpretation
If the \(\mu_j\) differ and the variances are similar, regimes correspond mainly to **different levels** of the series.

Example intuition:
- regime 1 = low level,
- regime 2 = high level.

### Variance-switching interpretation
If the means are similar and the \(\sigma_j^2\) differ, regimes correspond mainly to **different volatility levels**.

Example intuition:
- regime 1 = calm / low-noise,
- regime 2 = turbulent / high-noise.

### Mean-and-variance switching
If both differ, regimes differ in both location and dispersion.

This is the most general Gaussian version of your first model.

---

## 5. The role of the emission model in the full MS structure

The complete model has two layers:

### Hidden-state layer
This determines how likely each regime is before seeing the current observation.

### Emission layer
This determines how compatible the current observation is with each regime.

Together they produce the key quantity used later in filtering:

$$
f(y_t \mid y_{1:t-1})
=
\sum_{j=1}^K
f(y_t \mid S_t=j;\theta_j)\,
\Pr(S_t=j \mid y_{1:t-1}).
$$

This equation shows why the emission model matters so much:

- the Markov chain provides the weights,
- the emission model provides the regime-specific likelihoods.

The observation model is therefore not isolated forever; it is isolated now so that it can later be used cleanly inside the inference machinery.

---

## 6. What this phase should produce conceptually

At the end of this phase, you should be able to answer the following question unambiguously:

> Given an observation \(y_t\) and a candidate regime \(j\), what is the model-assigned density or log-density of that observation under regime \(j\)?

That is the entire purpose of Phase 3.

This sounds small, but it is one of the most important interfaces in the whole project.

---

## 7. Step-by-step guide

## Step 1 — Fix the emission family

For the first implementation, choose exactly one family:

$$
y_t \mid (S_t=j) \sim \mathcal N(\mu_j,\sigma_j^2).
$$

Do not generalize yet.

Do **not** start with:
- Student-\(t\),
- multivariate Gaussian,
- switching regression,
- autoregressive conditional mean,
- nonparametric emissions.

Keep the family fixed and simple.

### Why
The project should first validate the hidden-state mechanics with the simplest nontrivial observation law.

### Output of this step
A fixed choice of Gaussian regime-dependent emissions.

---

## Step 2 — Define regime-specific parameterization

For each regime \(j\), define

$$
\theta_j = (\mu_j,\sigma_j^2).
$$

You should decide conceptually that:

- \(\mu_j\) is unrestricted,
- \(\sigma_j^2\) must be strictly positive.

### Why
This is the first place where parameter constraints enter the project.

Later, your implementation will need a consistent representation of these constraints.

### Output of this step
A clear regime-parameter specification.

---

## Step 3 — Write the density formula explicitly

Write the Gaussian density in your own project notes exactly once and keep it fixed:

$$
f_j(y_t)
=
\frac{1}{\sqrt{2\pi\sigma_j^2}}
\exp\!\left(
-\frac{(y_t-\mu_j)^2}{2\sigma_j^2}
\right).
$$

Do the same for the log-density:

$$
\log f_j(y_t)
=
-\frac{1}{2}\log(2\pi)
-\frac{1}{2}\log(\sigma_j^2)
-\frac{(y_t-\mu_j)^2}{2\sigma_j^2}.
$$

### Why
You do not want ambiguity later about whether your emission layer is based on:
- density,
- log-density,
- variance,
- standard deviation.

This is where that notation gets fixed.

### Output of this step
A fully explicit mathematical definition of the emission function.

---

## Step 4 — Understand the geometry of the density

Before implementing anything later, make sure you understand how the density behaves.

For a fixed regime \(j\):

### If \(y_t\) is close to \(\mu_j\)
then \(f_j(y_t)\) is relatively large.

### If \(y_t\) is far from \(\mu_j\)
then \(f_j(y_t)\) is relatively small.

### If \(\sigma_j^2\) is small
the regime is more selective: only values near \(\mu_j\) get high density.

### If \(\sigma_j^2\) is large
the regime is more diffuse: a broader range of values gets moderate density.

### Why this matters
This is exactly what later drives posterior regime updating.

A regime becomes more probable when it assigns high likelihood to the current observation.

### Output of this step
An intuitive understanding of how regime plausibility will later be computed.

---

## Step 5 — Separate density evaluation from state dynamics

At this point, remind yourself:

- the emission model does **not** know anything about transitions,
- the transition matrix does **not** know anything about observation values.

The emission layer only answers:

> “How likely is this data point under regime \(j\)?”

It does **not** answer:

> “How likely is regime \(j\) overall at time \(t\)?”

That second question depends on both:
- state prediction from the Markov chain,
- and emission evaluation.

### Why this matters
This conceptual separation is essential for a clean Rust design later.

### Output of this step
A clear separation between:
- **regime dynamics**,
- **regime-conditional observation law**.

---

## Step 6 — Think in terms of a reusable interface

Even without writing code, you should now define what the emission layer conceptually takes as input and returns.

### Input
- observation \(y_t\),
- regime index \(j\),
- regime-specific parameters \(\theta_j\).

### Output
- density \(f_j(y_t)\),
- or log-density \(\log f_j(y_t)\).

### Why
This will later become the boundary between:
- the model-specific observation law,
- and the generic inference engine.

That means the filter should not need to know the internal structure of the Gaussian formula; it should only ask the emission layer for regime-conditional plausibility.

### Output of this step
A conceptual “emission interface” for the project.

---

## Step 7 — Check identifiability intuition

Before moving on, think through how identifiable the regimes are under your chosen parameters.

### Easy case
If regime means are very different, for example:
- one regime centered low,
- one regime centered high,

then the observations provide strong information about the active regime.

### Hard case
If regime means and variances are very similar, then the data alone will not distinguish regimes well.

### Variance-only switching
If means are equal but variances differ, then single observations may be ambiguous, but sequences of noisy vs stable observations can become informative.

### Why this matters
This will later affect:
- how sharp filtered probabilities are,
- how well EM recovers parameters,
- how sensitive the model is to initialization.

### Output of this step
A qualitative understanding of when the emission model helps or hurts state recovery.

---

## 8. How Phase 3 connects to Phase 2

In Phase 2, you simulated data using the regime-dependent Gaussian law.

There, the emission model was used in the **forward generative direction**:

- state known,
- generate observation.

In Phase 3, you isolate the same object for later **inferential use**:

- observation given,
- evaluate how plausible it is under each possible regime.

So Phase 2 and Phase 3 use the same mathematical object in opposite directions:

### Phase 2
From regime to data.

### Phase 3
From data to regime plausibility.

That is why this phase is the natural next step after simulation.

---

## 9. What to verify before moving on

Before starting filtering, you should be sure of the following.

## Check A — Parameter meaning
You can explain, without hesitation, what each of these does:
- \(\mu_j\),
- \(\sigma_j^2\),
- \(f_j(y_t)\),
- \(\log f_j(y_t)\).

---

## Check B — Density behavior
You understand how density changes when:
- \(y_t\) moves away from \(\mu_j\),
- \(\sigma_j^2\) increases or decreases.

---

## Check C — Model separation
You are clear that:
- transition probabilities control regime evolution,
- emission densities control observation compatibility.

---

## Check D — Future role
You understand that the emission density will later appear inside:
- filtering,
- likelihood evaluation,
- smoothing-related expectations,
- EM estimation.

---

## 10. Common conceptual mistakes to avoid

### Mistake 1 — Treating the emission model as the whole model
It is only one layer.  
A Gaussian mixture by itself is not yet a Markov switching model.

### Mistake 2 — Confusing variance and standard deviation
The parameterization should stay consistent throughout the project.

If you define the regime by variance \(\sigma_j^2\), then all formulas and interpretation should use that consistently.

### Mistake 3 — Forgetting the log-density
Even if you think mainly in terms of densities, later numerical work will usually require log-densities.

### Mistake 4 — Mixing regime probability with observation likelihood
A regime can have a high predicted probability because of the transition matrix, even if the current observation is not especially compatible with it.  
Likewise, a regime can fit the observation well but have low prior weight before seeing that observation.

Filtering combines both effects; the emission model alone does not.

### Mistake 5 — Generalizing too early
Do not add regression or AR structure before the basic emission model is conceptually complete.

---

## 11. Minimal mathematical summary of Phase 3

For each regime \(j=1,\dots,K\), define

$$
\theta_j = (\mu_j,\sigma_j^2),
\qquad \sigma_j^2 > 0,
$$

and the regime-conditional observation law

$$
y_t \mid (S_t=j) \sim \mathcal N(\mu_j,\sigma_j^2).
$$

Then define the emission density

$$
f_j(y_t)
=
\frac{1}{\sqrt{2\pi\sigma_j^2}}
\exp\!\left(
-\frac{(y_t-\mu_j)^2}{2\sigma_j^2}
\right),
$$

and the corresponding log-density

$$
\log f_j(y_t)
=
-\frac{1}{2}\log(2\pi)
-\frac{1}{2}\log(\sigma_j^2)
-\frac{(y_t-\mu_j)^2}{2\sigma_j^2}.
$$

This is the function that later tells the model how compatible an observation is with each possible regime.

---

## 12. Output of Phase 3

At the end of this phase, you should have:

- a fully explicit definition of the regime-specific Gaussian observation law,
- a fixed notation for \(\mu_j\), \(\sigma_j^2\), \(\theta_j\), \(f_j(y_t)\), and \(\log f_j(y_t)\),
- a clear conceptual separation between hidden-state dynamics and observation likelihood,
- an intuitive understanding of how the observation model influences later regime inference,
- a stable basis for Phase 4, where the emission model will be inserted into the forward filter.