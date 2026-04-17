# Phase 8 — Add Pairwise Regime Probabilities

## Goal

In this phase, extend the latent-state inference layer beyond **single-time smoothed probabilities** and compute the **pairwise smoothed transition probabilities**

\[
\Pr(S_{t-1}=i, S_t=j \mid y_{1:T}),
\qquad t=2,\dots,T.
\]

These quantities describe the posterior probability that the process was in regime \(i\) at time \(t-1\) and in regime \(j\) at time \(t\), given the full observed sample.

This phase is essential because smoothing alone gives you only:

\[
\Pr(S_t=j \mid y_{1:T}),
\]

which answers:

> “How likely was regime \(j\) at time \(t\)?”

But for EM and transition-matrix estimation, you also need to answer:

> “How likely was the transition \(i \to j\) between times \(t-1\) and \(t\)?”

That is the purpose of Phase 8.

---

## 1. What this phase adds beyond smoothing

By the end of Phase 7, you already have:

- predicted probabilities
  \[
  \Pr(S_t=j \mid y_{1:t-1}),
  \]
- filtered probabilities
  \[
  \Pr(S_t=j \mid y_{1:t}),
  \]
- smoothed probabilities
  \[
  \Pr(S_t=j \mid y_{1:T}).
  \]

These are all **marginal posterior probabilities** at a single time point.

However, transition estimation requires **joint posterior probabilities across two adjacent time points**.

So this phase introduces a new inference object:

\[
\xi_t(i,j)
=
\Pr(S_{t-1}=i, S_t=j \mid y_{1:T}),
\qquad t=2,\dots,T.
\]

This is the posterior probability of a specific regime transition between two consecutive times.

---

## 2. Why pairwise probabilities matter

The transition matrix is defined by

\[
p_{ij} = \Pr(S_t=j \mid S_{t-1}=i).
\]

So if you want to estimate \(p_{ij}\), you need information about:

- how often the model believes state \(i\) occurred at time \(t-1\),
- how often that was followed by state \(j\) at time \(t\).

Single-time smoothed probabilities are not enough for that.

They tell you:

- how often state \(i\) appeared,
- how often state \(j\) appeared,

but **not** how often the specific transition \(i \to j\) occurred.

That transition-level information is exactly what \(\xi_t(i,j)\) provides.

---

## 3. The new probability object

Define, for each \(t=2,\dots,T\),

\[
\xi_t(i,j)
=
\Pr(S_{t-1}=i, S_t=j \mid y_{1:T}).
\]

This is the key object of Phase 8.

You should think of it as the two-time analog of the smoothed probabilities:

- \(\gamma_t(j)\) is a one-time posterior,
- \(\xi_t(i,j)\) is a two-time posterior over adjacent states.

These pairwise probabilities are sometimes called:

- smoothed transition probabilities,
- pairwise posterior state probabilities,
- expected transition indicators.

All of these refer to the same underlying object.

---

## 4. What \(\xi_t(i,j)\) means intuitively

For a fixed time \(t\), the quantity

\[
\xi_t(i,j)
\]

measures how plausible it is, given the **entire sample**, that:

- the hidden regime at time \(t-1\) was \(i\),
- and the hidden regime at time \(t\) was \(j\).

So for each time step, you do not just infer “which regime was likely,” but also “which transition was likely.”

This gives a much richer description of the latent dynamics.

---

## 5. Why this phase comes after smoothing

You should not compute \(\xi_t(i,j)\) before you have:

- the forward filter,
- the backward smoother.

That is because the pairwise posterior depends on both:

- information up to time \(t-1\),
- information from time \(t\) onward,
- and the transition structure linking the two times.

Conceptually, \(\xi_t(i,j)\) sits between filtering and smoothing:

- it uses filtered information from the past,
- predicted/emission information for the transition step,
- and full-sample normalization through the observed data.

This is why it naturally comes after Phase 7.

---

## 6. Deriving the pairwise posterior

You want to compute:

\[
\xi_t(i,j)
=
\Pr(S_{t-1}=i, S_t=j \mid y_{1:T}).
\]

Start from the proportional form:

\[
\Pr(S_{t-1}=i, S_t=j \mid y_{1:T})
\propto
\Pr(S_{t-1}=i, S_t=j, y_{1:T}).
\]

Using the structure of the hidden-state model, this joint term factors into:

1. the filtered probability of state \(i\) at time \(t-1\),
2. the transition probability \(p_{ij}\),
3. the likelihood contribution of \(y_t\) under state \(j\),
4. the influence of future data beyond time \(t\).

There are multiple equivalent ways to express the recursion depending on what quantities you store. For your project, the cleanest route is to express \(\xi_t(i,j)\) using quantities you already have from the forward and backward passes.

---

## 7. Clean formula using filtered and smoothed quantities

A very useful expression is:

\[
\xi_t(i,j)
=
\frac{
\alpha_{t-1|t-1}(i)\, p_{ij}\, f_j(y_t)\,
\frac{\gamma_t(j)}{\alpha_{t|t}(j)}
}{
c_t
},
\qquad t=2,\dots,T.
\]

Here:

- \(\alpha_{t-1|t-1}(i)=\Pr(S_{t-1}=i \mid y_{1:t-1})\) is the filtered probability at time \(t-1\),
- \(p_{ij}\) is the transition probability from \(i\) to \(j\),
- \(f_j(y_t)=f(y_t \mid S_t=j)\) is the emission density at time \(t\) under regime \(j\),
- \(c_t=f(y_t \mid y_{1:t-1})\) is the predictive density at time \(t\),
- \(\gamma_t(j)=\Pr(S_t=j \mid y_{1:T})\) is the smoothed probability at time \(t\),
- \(\alpha_{t|t}(j)=\Pr(S_t=j \mid y_{1:t})\) is the filtered probability at time \(t\).

This formula is especially useful because it ties directly into the quantities already built in Phases 4–7.

---

## 8. Equivalent and often cleaner formula using predicted probabilities

Using the identity

\[
\alpha_{t|t}(j)
=
\frac{f_j(y_t)\alpha_{t|t-1}(j)}{c_t},
\]

the previous formula simplifies to

\[
\xi_t(i,j)
=
\alpha_{t-1|t-1}(i)\, p_{ij}\,
\frac{\gamma_t(j)}{\alpha_{t|t-1}(j)}.
\]

This is often the cleanest conceptual formula.

It says:

- start from the filtered probability that the process was in state \(i\) at time \(t-1\),
- multiply by the transition probability from \(i\) to \(j\),
- then correct by how much full-sample information changes the belief about state \(j\) at time \(t\).

That correction factor is

\[
\frac{\gamma_t(j)}{\alpha_{t|t-1}(j)}.
\]

This is the exact same kind of future-information correction that already appeared in the smoother.

---

## 9. Interpretation of the formula

The pairwise posterior

\[
\xi_t(i,j)
\]

can be understood as a three-part object.

### Part 1 — Current belief about the previous state
\[
\alpha_{t-1|t-1}(i)
\]

This tells you how likely state \(i\) was at time \(t-1\) based on data up to that point.

### Part 2 — Transition dynamics
\[
p_{ij}
\]

This tells you how likely a move from \(i\) to \(j\) is under the Markov chain.

### Part 3 — Future correction
\[
\frac{\gamma_t(j)}{\alpha_{t|t-1}(j)}
\]

This measures how much the full sample changes your belief about state \(j\) at time \(t\) relative to what was predicted before seeing \(y_t\) and beyond.

So \(\xi_t(i,j)\) is the posterior plausibility of the transition \(i \to j\) after combining:

- past information,
- transition structure,
- present observation,
- future information.

---

## 10. Why these are “expected transitions”

Suppose you introduce an indicator variable

\[
I_t(i,j)
=
\mathbf{1}\{S_{t-1}=i, S_t=j\}.
\]

Then

\[
\mathbb{E}[I_t(i,j)\mid y_{1:T}]
=
\Pr(S_{t-1}=i, S_t=j \mid y_{1:T})
=
\xi_t(i,j).
\]

So \(\xi_t(i,j)\) is literally the posterior expected indicator that the transition \(i \to j\) occurred at time \(t\).

This is why people often say that Phase 8 computes **expected regime transitions**.

---

## 11. Step-by-step guide for this phase

## Step 1 — Ensure the required quantities are available

Before adding pairwise probabilities, make sure your inference layer already stores:

- filtered probabilities \(\alpha_{t|t}(j)\),
- filtered probabilities at the previous time \(\alpha_{t-1|t-1}(i)\),
- predicted probabilities \(\alpha_{t|t-1}(j)\),
- smoothed probabilities \(\gamma_t(j)\),
- transition matrix \(P\),
- and, depending on the chosen formula, possibly \(f_j(y_t)\) and \(c_t\).

### Why
The pairwise recursion is built entirely out of these already-computed quantities.

### Output of this step
A forward/backward inference state rich enough to support pairwise posteriors.

---

## Step 2 — Define the new tensor of pairwise posteriors

For each time \(t=2,\dots,T\), define a \(K \times K\) matrix:

\[
\Xi_t = \big(\xi_t(i,j)\big)_{i,j=1}^K.
\]

So over the full sample, you will have a sequence of \(T-1\) such matrices.

### Interpretation
Each matrix \(\Xi_t\) describes the posterior distribution over all possible transitions between times \(t-1\) and \(t\).

### Output of this step
A precise mathematical target for the phase.

---

## Step 3 — Compute \(\xi_t(i,j)\) for every pair \((i,j)\)

For each time \(t=2,\dots,T\) and each pair of regimes \((i,j)\), compute either of the equivalent forms:

### Form A
\[
\xi_t(i,j)
=
\frac{
\alpha_{t-1|t-1}(i)\, p_{ij}\, f_j(y_t)\,
\frac{\gamma_t(j)}{\alpha_{t|t}(j)}
}{
c_t
}
\]

or the simpler

### Form B
\[
\xi_t(i,j)
=
\alpha_{t-1|t-1}(i)\, p_{ij}\,
\frac{\gamma_t(j)}{\alpha_{t|t-1}(j)}.
\]

### Why
This gives the posterior probability of every possible one-step transition.

### Output of this step
A full set of pairwise posterior matrices \(\Xi_2,\dots,\Xi_T\).

---

## Step 4 — Check normalization at each time step

For each \(t\), the pairwise posterior should define a proper probability distribution over all regime pairs:

\[
\sum_{i=1}^K \sum_{j=1}^K \xi_t(i,j)=1.
\]

### Why
At each time step, exactly one transition \(S_{t-1} \to S_t\) occurred, even though it is hidden.

So the posterior over all such possibilities must sum to 1.

### Output of this step
Confidence that each \(\Xi_t\) is a valid two-time posterior distribution.

---

## Step 5 — Check consistency with smoothed marginals

The pairwise probabilities must be consistent with the one-time smoothed probabilities.

For each \(t=2,\dots,T\):

### Marginal over previous state
\[
\sum_{i=1}^K \xi_t(i,j)=\gamma_t(j).
\]

### Marginal over next state
\[
\sum_{j=1}^K \xi_t(i,j)=\gamma_{t-1}(i).
\]

These identities are extremely important.

### Why
They verify that the pairwise posterior and the single-time smoothed posteriors are mutually coherent.

### Output of this step
Strong structural validation of the pairwise recursion.

---

## 12. How pairwise probabilities connect to transition counts

Once you have \(\xi_t(i,j)\), you can sum over time to obtain the posterior expected number of transitions from regime \(i\) to regime \(j\):

\[
N_{ij}^{\text{exp}}
=
\sum_{t=2}^T \xi_t(i,j).
\]

This is one of the main deliverables of the phase.

Interpretation:

- if \(N_{ij}^{\text{exp}}\) is large, the model believes the transition \(i \to j\) happened often,
- if \(N_{ii}^{\text{exp}}\) is large, regime \(i\) appears persistent,
- if off-diagonal expected counts are large, switching between regimes is common.

These expected counts are exactly what the EM algorithm will use later.

---

## 13. Expected regime occupancy from smoothed probabilities

To estimate transition probabilities later, you also need the expected number of times regime \(i\) served as the previous state in a transition.

That quantity is

\[
M_i^{\text{exp}}
=
\sum_{t=1}^{T-1} \gamma_t(i).
\]

Equivalently, by consistency of marginals,

\[
M_i^{\text{exp}}
=
\sum_{t=2}^T \sum_{j=1}^K \xi_t(i,j).
\]

This shows again how the smoothed marginals and pairwise posteriors fit together.

---

## 14. Why this phase is exactly what EM needs

The M-step for the transition matrix will later update

\[
p_{ij}
\]

using the ratio:

\[
p_{ij}^{\text{new}}
=
\frac{
\text{expected number of } i\to j \text{ transitions}
}{
\text{expected number of times state } i \text{ appears as previous state}
}.
\]

That is,

\[
p_{ij}^{\text{new}}
=
\frac{
\sum_{t=2}^T \xi_t(i,j)
}{
\sum_{t=1}^{T-1} \gamma_t(i)
}.
\]

So without Phase 8, you do not have the posterior expected transition counts needed to update the transition matrix.

This is why Phase 8 is a structural prerequisite for EM.

---

## 15. What to expect on simulated data

On synthetic data, where the true hidden path is known, the pairwise probabilities should behave sensibly.

### Easy cases
If regimes are strongly separated and persistent, then for many time points:
- one specific pair \((i,j)\) should dominate,
- often the posterior will strongly favor either staying in the same regime or making a clearly visible switch.

### Hard cases
If regimes overlap strongly:
- pairwise probabilities may remain diffuse,
- several transitions may receive nontrivial posterior mass.

This is not necessarily a bug.  
It may simply reflect genuine latent-state uncertainty.

---

## 16. Interpretation near regime changes

Pairwise posteriors are especially informative near true switching times.

Suppose the latent chain likely changed from regime 1 to regime 2 around time \(t\).

Then you may see:

- high posterior mass on \(\xi_t(1,2)\),
- lower mass on \(\xi_t(1,1)\) or \(\xi_t(2,2)\) at that same time step,
- a clear posterior signal that the transition occurred between \(t-1\) and \(t\).

This makes pairwise probabilities more informative about transition timing than single-time smoothed marginals alone.

---

## 17. Structural checks for Phase 8

Once pairwise probabilities are added, verify the following carefully.

### Check A — Nonnegativity
\[
\xi_t(i,j)\ge 0
\qquad \text{for all } t,i,j.
\]

### Check B — Per-time normalization
\[
\sum_{i=1}^K\sum_{j=1}^K \xi_t(i,j)=1
\qquad \text{for all } t.
\]

### Check C — Consistency with next-time smoothed marginal
\[
\sum_{i=1}^K \xi_t(i,j)=\gamma_t(j).
\]

### Check D — Consistency with previous-time smoothed marginal
\[
\sum_{j=1}^K \xi_t(i,j)=\gamma_{t-1}(i).
\]

### Check E — Expected transition counts are finite
\[
\sum_{t=2}^T \xi_t(i,j)
\]
should be finite and nonnegative for all \(i,j\).

These checks are essential before moving to EM.

---

## 18. Common conceptual mistakes to avoid

### Mistake 1 — Thinking smoothed marginals alone are enough for EM
They are not.  
They give expected state occupancy, but not expected transitions.

### Mistake 2 — Confusing \(\gamma_t(j)\) with \(\xi_t(i,j)\)
The first is a one-time posterior.  
The second is a two-time posterior.

### Mistake 3 — Forgetting the time indexing
\(\xi_t(i,j)\) refers to the transition from time \(t-1\) to time \(t\), not from \(t\) to \(t+1\).

### Mistake 4 — Not checking marginal consistency
If the pairwise matrices are not consistent with the smoothed marginals, something is wrong in the recursion or indexing.

### Mistake 5 — Treating diffuse pairwise posteriors as automatically incorrect
In weakly identified settings, that may be the correct behavior.

---

## 19. Minimal mathematical summary of Phase 8

Define the pairwise posterior transition probabilities by

\[
\xi_t(i,j)=\Pr(S_{t-1}=i, S_t=j \mid y_{1:T}),
\qquad t=2,\dots,T.
\]

A clean formula is

\[
\xi_t(i,j)
=
\alpha_{t-1|t-1}(i)\, p_{ij}\,
\frac{\gamma_t(j)}{\alpha_{t|t-1}(j)}.
\]

These probabilities satisfy:

\[
\sum_{i=1}^K \sum_{j=1}^K \xi_t(i,j)=1,
\]

\[
\sum_{i=1}^K \xi_t(i,j)=\gamma_t(j),
\]

\[
\sum_{j=1}^K \xi_t(i,j)=\gamma_{t-1}(i).
\]

Then the posterior expected number of transitions from \(i\) to \(j\) is

\[
N_{ij}^{\text{exp}}=\sum_{t=2}^T \xi_t(i,j).
\]

This is the key deliverable needed for transition-matrix updates in EM.

---

## 20. Output of Phase 8

At the end of this phase, you should have:

- a formal definition of the pairwise posterior transition probabilities,
- a sequence of \(K\times K\) posterior transition matrices across time,
- expected transition counts for every regime pair,
- posterior measures of regime persistence and switching frequency,
- consistency checks linking pairwise probabilities to smoothed marginals,
- and the full latent-transition information needed for the next phase: EM estimation.