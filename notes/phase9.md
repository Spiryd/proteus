# Phase 9 — Implement Likelihood-Based Estimation with Fixed Structure

## Goal

In this phase, move from **inference under fixed parameters** to **actual parameter estimation**.

Up to this point, your Markov Switching Model pipeline has assumed that the parameter set

\[
\Theta = (\pi, P, \mu_1,\dots,\mu_K,\sigma_1^2,\dots,\sigma_K^2)
\]

is known. Under that assumption, you built:

- simulation,
- emission evaluation,
- forward filtering,
- observed-data log-likelihood evaluation,
- smoothing,
- pairwise posterior transition probabilities.

Now the task changes.

You no longer want to ask:

> “Given the parameters, what are the latent regime probabilities?”

You now want to ask:

> “Given only the observed data, which parameter values make the model fit best?”

This phase introduces that estimation layer.

---

## 1. What estimation means here

You observe

\[
y_{1:T} = (y_1,\dots,y_T),
\]

but you do **not** observe the hidden regime path

\[
S_{1:T} = (S_1,\dots,S_T).
\]

The model parameters are unknown, and must be inferred from the observed data.

For the basic Gaussian Markov Switching Model, the unknowns are:

- the initial regime distribution \(\pi\),
- the transition matrix \(P\),
- the regime-specific means \(\mu_j\),
- the regime-specific variances \(\sigma_j^2\).

So estimation means constructing a procedure that takes the observed series \(y_{1:T}\) and returns fitted values

\[
\widehat{\Theta}.
\]

---

## 2. Why this phase is called “likelihood-based estimation with fixed structure”

The phrase **fixed structure** is important.

At this stage, you are **not** trying to choose:

- the number of regimes \(K\),
- the observation family,
- whether the model should be switching mean, switching variance, switching regression, or switching AR.

All of that remains fixed.

You are estimating parameters **within a fixed model structure**:

- finite number of regimes \(K\),
- first-order Markov chain,
- Gaussian regime-dependent emissions.

So this phase is about fitting a given model specification, not comparing different model classes.

---

## 3. The two routes to estimation

There are two main routes.

## Route A — Direct maximum likelihood

Treat the model as a parameterized black box:

1. input a candidate parameter vector,
2. run the forward filter,
3. compute the observed-data log-likelihood,
4. use a numerical optimizer to maximize it.

This is conceptually straightforward but can be harder to manage because:

- the parameters are constrained,
- the likelihood surface is non-convex,
- the hidden-state structure is not used explicitly in the optimization logic.

---

## Route B — EM algorithm

Use the latent-state structure directly.

Alternate between:

### E-step
Given current parameters, compute:
- filtered probabilities,
- smoothed probabilities,
- pairwise transition probabilities.

### M-step
Update parameters using the posterior latent expectations from the E-step.

For the basic Gaussian Markov Switching Model, the M-step has simple weighted closed-form updates for:

- \(\pi\),
- \(P\),
- \(\mu_j\),
- \(\sigma_j^2\).

This is why EM is usually the better first estimation method for a from-scratch implementation.

---

## 4. Recommended route for this project

For this first Rust implementation, the recommended route is:

\[
\boxed{\text{EM first, direct ML later if needed}}
\]

Why EM is the better starting point:

- it matches the hidden-state structure of the model,
- it reuses the inference layers you already built,
- the transition matrix update is transparent,
- Gaussian regime-parameter updates are weighted sample statistics,
- debugging is easier because each iteration decomposes into interpretable substeps.

So Phase 9 should primarily be understood as:

> Build a full EM estimator for the basic Gaussian Markov Switching Model.

You may still keep direct maximum likelihood in mind as a later extension or cross-check, but the main implementation target here should be EM.

---

## 5. What the estimation layer sits on top of

This phase assumes that the following components already exist and are trusted:

- a simulator,
- an emission model,
- a forward filter,
- a log-likelihood layer,
- a backward smoother,
- pairwise posterior transition probabilities.

These earlier layers are now reused inside the estimation procedure.

So the estimation layer does **not** replace inference.  
It orchestrates repeated inference steps under changing parameters.

---

## 6. The EM viewpoint

The EM algorithm is designed for models with latent variables.

In your case, the latent variables are the hidden regimes

\[
S_1,\dots,S_T.
\]

The key idea is:

- if the hidden regimes were known, parameter estimation would be much easier,
- since they are not known, use their posterior expectations instead.

So EM alternates between:

### E-step
Estimate the latent structure probabilistically under current parameters.

### M-step
Pretend those posterior expectations are the relevant sufficient statistics and update the parameters accordingly.

This is exactly the right logic for a hidden Markov / Markov switching model.

---

## 7. The objective being optimized

The ultimate objective remains the observed-data log-likelihood:

\[
\log L(\Theta)
=
\sum_{t=1}^T \log c_t,
\qquad
c_t = f(y_t \mid y_{1:t-1};\Theta).
\]

EM is not optimizing a different criterion.  
It is simply an iterative way to increase this same observed-data log-likelihood by exploiting latent-state structure.

So throughout this phase, keep the distinction clear:

- **target objective:** observed-data log-likelihood,
- **estimation mechanism:** EM updates.

---

## 8. What the E-step must produce

At iteration \(m\), suppose the current parameter estimate is

\[
\Theta^{(m)}.
\]

Then the E-step runs the inference machinery under \(\Theta^{(m)}\) and produces:

### 8.1 Smoothed marginal probabilities
\[
\gamma_t^{(m)}(j)
=
\Pr(S_t=j \mid y_{1:T};\Theta^{(m)}).
\]

These represent expected regime occupancy.

---

### 8.2 Pairwise smoothed transition probabilities
\[
\xi_t^{(m)}(i,j)
=
\Pr(S_{t-1}=i,S_t=j \mid y_{1:T};\Theta^{(m)}),
\qquad t=2,\dots,T.
\]

These represent expected one-step transitions.

---

### 8.3 Observed-data log-likelihood
\[
\log L(\Theta^{(m)}).
\]

This is needed for monitoring convergence.

So the E-step is nothing more than a full inference pass under the current parameters.

---

## 9. What the M-step must update

Given the posterior expectations from the E-step, update the parameters.

For the basic Gaussian switching model, all updates are explicit.

---

## 10. M-step update for the initial distribution

A natural update is:

\[
\pi_j^{(m+1)}
=
\gamma_1^{(m)}(j).
\]

Interpretation:

- the posterior probability that regime \(j\) was active at time \(1\) becomes the new estimate of the initial regime probability.

This is the most direct latent-variable update.

---

## 11. M-step update for the transition matrix

The updated transition probability from regime \(i\) to regime \(j\) is

\[
p_{ij}^{(m+1)}
=
\frac{
\sum_{t=2}^T \xi_t^{(m)}(i,j)
}{
\sum_{t=1}^{T-1} \gamma_t^{(m)}(i)
}.
\]

Interpretation:

- numerator = posterior expected number of \(i \to j\) transitions,
- denominator = posterior expected number of times state \(i\) appears as the previous state.

This is exactly the latent-data analog of estimating a transition matrix from observed state counts.

---

## 12. M-step update for regime means

For each regime \(j\), update the mean using a weighted average:

\[
\mu_j^{(m+1)}
=
\frac{
\sum_{t=1}^T \gamma_t^{(m)}(j)\, y_t
}{
\sum_{t=1}^T \gamma_t^{(m)}(j)
}.
\]

Interpretation:

- each observation contributes to every regime,
- but with weight equal to the posterior probability that the regime was active.

So the regime mean becomes the posterior-weighted sample mean.

---

## 13. M-step update for regime variances

For each regime \(j\), update the variance using the posterior-weighted second moment around the updated mean:

\[
(\sigma_j^2)^{(m+1)}
=
\frac{
\sum_{t=1}^T \gamma_t^{(m)}(j)\,
\bigl(y_t-\mu_j^{(m+1)}\bigr)^2
}{
\sum_{t=1}^T \gamma_t^{(m)}(j)
}.
\]

Interpretation:

- this is the weighted sample variance within regime \(j\),
- where the weights are posterior state probabilities.

This is the closed-form Gaussian M-step update.

---

## 14. The full EM iteration

A single EM iteration looks like this.

## Step 1 — Start from current parameters
\[
\Theta^{(m)}.
\]

## Step 2 — Run E-step
Compute:
- forward filter,
- log-likelihood,
- backward smoother,
- pairwise posterior transitions.

## Step 3 — Run M-step
Update:
- \(\pi\),
- \(P\),
- \(\mu_j\),
- \(\sigma_j^2\).

## Step 4 — Form the new parameter set
\[
\Theta^{(m+1)}.
\]

## Step 5 — Check convergence
Use log-likelihood change, parameter change, or both.

Then repeat until stopping criteria are satisfied.

---

## 15. Step-by-step implementation guide

## Step 1 — Define the estimator boundary

Conceptually define a top-level estimation component whose role is:

- take observations \(y_{1:T}\),
- take initial parameter guesses,
- repeatedly run E-step and M-step,
- return fitted parameters and convergence diagnostics.

### Deliverable
A clearly named estimation layer in your project design.

### Code changes
At this point, you should add a dedicated conceptual module or file boundary for estimation, separate from:
- emission logic,
- filtering,
- smoothing,
- pairwise probabilities.

For example, in structural terms, your project should now gain something like:

- an `estimation` layer,
- an `em` sub-layer or equivalent estimator unit,
- a result type for fitted parameters and diagnostics.

---

## Step 2 — Define the parameter container for fitting

You already have model parameters conceptually, but now they must behave as mutable iterative objects.

The parameter representation should support:

- initialization,
- repeated updates,
- validation after each M-step,
- log-likelihood association.

### Deliverable
A parameter object suitable for iterative estimation.

### Code changes
You should now introduce or refine:
- a unified parameter struct for \(\pi\), \(P\), means, and variances,
- validation routines for constraints,
- cloning / copying semantics suitable for iterative updates,
- a way to compare old vs new parameter values.

---

## Step 3 — Define the E-step result object

The E-step should return a coherent bundle of all quantities needed by the M-step.

At minimum, that bundle should contain:

- filtered probabilities,
- predicted probabilities,
- smoothed probabilities,
- pairwise transition probabilities,
- log-likelihood.

### Deliverable
A well-defined E-step output contract.

### Code changes
You should introduce a new result structure that groups all latent posterior quantities needed for estimation, rather than passing many separate arrays around ad hoc.

This is important for:
- clarity,
- testability,
- M-step cleanliness.

---

## Step 4 — Define the M-step as parameter updates from sufficient statistics

The M-step should be designed as a deterministic transformation:

\[
(\text{current posterior expectations}) \longrightarrow (\text{updated parameters})
\]

This should be conceptually independent of how the E-step was computed.

### Deliverable
A clean separation between latent inference output and parameter-update logic.

### Code changes
You should add a dedicated M-step layer or function boundary that:
- accepts the E-step result,
- returns a new parameter object,
- does not recompute filtering internally.

This separation is very important for debugging.

---

## Step 5 — Add an EM iteration controller

Now add the iterative layer that repeatedly performs:

1. E-step,
2. M-step,
3. convergence check.

### Deliverable
A full EM loop specification.

### Code changes
You should add an iteration controller that tracks:
- iteration number,
- current parameters,
- previous parameters,
- log-likelihood history,
- convergence status.

This controller is the core of Phase 9 from a software-architecture perspective.

---

## Step 6 — Add stopping criteria

You need a principled rule for stopping the EM iterations.

Typical choices:

### Log-likelihood change
Stop when

\[
\left|\log L(\Theta^{(m+1)}) - \log L(\Theta^{(m)})\right|
\]

is below a tolerance.

### Parameter change
Stop when the maximum parameter difference becomes small.

### Iteration cap
Stop if a maximum number of iterations is reached.

For a first implementation, use:
- log-likelihood tolerance,
- plus a hard iteration limit.

### Deliverable
A formal convergence policy.

### Code changes
Add convergence configuration and convergence reporting to the estimator output.

---

## Step 7 — Add monotonicity monitoring

A key EM diagnostic is that the observed-data log-likelihood should not decrease across iterations, up to small numerical noise.

So at iteration \(m\), track

\[
\log L(\Theta^{(m)}).
\]

### Deliverable
A likelihood history for every estimation run.

### Code changes
Your estimator result should now include:
- full log-likelihood trajectory,
- final log-likelihood,
- number of iterations,
- convergence flag.

This is one of the most useful diagnostics in the entire project.

---

## Step 8 — Add initialization support

EM is sensitive to initialization.

So the estimator should not assume that one arbitrary starting point is sufficient.

At minimum, support:
- user-specified initial parameters,
- possibly later multiple starts.

For now, keep the initialization interface explicit.

### Deliverable
A well-defined initialization entry point for estimation.

### Code changes
Add a parameter initialization pathway that is separate from the EM loop itself.

This will matter a lot in later phases.

---

## 16. Mathematical summary of the EM estimator

At iteration \(m\):

### E-step
Compute:
\[
\gamma_t^{(m)}(j)=\Pr(S_t=j \mid y_{1:T};\Theta^{(m)})
\]
and
\[
\xi_t^{(m)}(i,j)=\Pr(S_{t-1}=i,S_t=j \mid y_{1:T};\Theta^{(m)}).
\]

### M-step
Update:
\[
\pi_j^{(m+1)}=\gamma_1^{(m)}(j),
\]

\[
p_{ij}^{(m+1)}
=
\frac{
\sum_{t=2}^T \xi_t^{(m)}(i,j)
}{
\sum_{t=1}^{T-1} \gamma_t^{(m)}(i)
},
\]

\[
\mu_j^{(m+1)}
=
\frac{
\sum_{t=1}^T \gamma_t^{(m)}(j)y_t
}{
\sum_{t=1}^T \gamma_t^{(m)}(j)
},
\]

\[
(\sigma_j^2)^{(m+1)}
=
\frac{
\sum_{t=1}^T \gamma_t^{(m)}(j)(y_t-\mu_j^{(m+1)})^2
}{
\sum_{t=1}^T \gamma_t^{(m)}(j)
}.
\]

Repeat until convergence.

This is the core estimation algorithm of Phase 9.

---

## 17. Relationship to direct maximum likelihood

Even though EM is the recommended route now, it is useful to understand how it relates to direct ML.

Direct ML would require:

1. parameterizing the constrained parameters safely,
2. calling the forward filter for every candidate parameter vector,
3. passing the resulting log-likelihood to a numerical optimizer.

This approach is valid, but for the first implementation it is less transparent because:

- the transition matrix has row-stochastic constraints,
- the variances must remain positive,
- optimization may be sensitive to parameter scaling and local optima.

So direct ML is best thought of here as:
- a later extension,
- a possible benchmark,
- not the primary first implementation target.

---

## 18. What to validate in Phase 9

At this phase, validation is still crucial.

You should check:

### Structural checks
- updated \(\pi\) sums to 1,
- each row of \(P\) sums to 1,
- all variances remain positive.

### EM checks
- log-likelihood is finite at every iteration,
- log-likelihood is nondecreasing up to numerical tolerance,
- parameter updates stabilize when converged.

### Behavioral checks
- on easy simulated data, fitted parameters move toward the true ones,
- regime means become sensible,
- transition persistence becomes plausible,
- the latent posteriors remain coherent after refitting.

---

## 19. Common conceptual mistakes to avoid

### Mistake 1 — Treating EM as a black box optimizer
It is not.  
Its strength is that each step has an interpretable latent-variable meaning.

### Mistake 2 — Mixing E-step and M-step responsibilities
The E-step computes posterior quantities under fixed parameters.  
The M-step updates parameters from those quantities.

Do not blur that boundary.

### Mistake 3 — Forgetting that log-likelihood remains the real objective
Even though EM is the mechanism, the observed-data log-likelihood is still the criterion being improved.

### Mistake 4 — Updating parameters without checking constraints
Every M-step output must still define a valid model.

### Mistake 5 — Not keeping iteration diagnostics
Without iteration history, debugging convergence becomes much harder.

---

## 20. Deliverables of Phase 9

By the end of this phase, you should have:

### Mathematical deliverables
- a full EM formulation for the basic Gaussian Markov Switching Model,
- explicit updates for \(\pi\), \(P\), \(\mu_j\), and \(\sigma_j^2\),
- a clear distinction between the E-step and M-step,
- a convergence criterion tied to the observed-data log-likelihood.

### Inference/estimation deliverables
- an estimation layer that repeatedly runs inference and parameter updates,
- an E-step result bundle,
- an M-step update procedure,
- an EM iteration controller,
- iteration diagnostics including log-likelihood history.

### Code-structure deliverables
You should have added or revised, where appropriate:

- a dedicated estimation module / file boundary,
- a fitted-parameter result type,
- an E-step output type,
- an EM controller / estimator type,
- convergence configuration,
- likelihood-history tracking,
- parameter-validation logic integrated into the estimation cycle.

### Validation deliverables
- at least one synthetic-data estimation run,
- evidence that EM increases the log-likelihood,
- evidence that parameter estimates stabilize,
- and a basis for Phase 10, where initialization and robustness become central.

---

## 21. Minimal final summary

Phase 9 is the point where the project becomes a true fitted model rather than only an inference engine under known parameters.

The recommended implementation target is:

\[
\boxed{\text{EM estimation for the basic Gaussian Markov Switching Model}}
\]

because it reuses all previously built latent-state inference components and yields simple, interpretable updates for the model parameters.

This phase should end with a working estimation loop that can take observed data and return fitted model parameters together with convergence diagnostics.