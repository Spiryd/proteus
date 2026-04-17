
## Goal

Before thinking about architecture or Rust modules, fix the model **precisely and completely** in a single notation system.  
This phase is the mathematical contract for the rest of the implementation.

Your first implementation target is a **first-order Gaussian Markov Switching Model** with a finite number of regimes.

---

## 1. Time index and observations

Let the observed time series be

$$
y_1, y_2, \dots, y_T,
$$

where each $y_t \in \mathbb{R}$ is the observed scalar value at time $t$.

For the first version, assume the data are univariate.  
This keeps the model simple and allows you to focus entirely on the hidden-regime machinery.

---

## 2. Hidden regime process

Introduce an unobserved discrete-valued process

$$
S_t \in \{1,2,\dots,K\},
$$

where $S_t$ denotes the regime active at time $t$.

The process $\{S_t\}$ is assumed to be a **first-order Markov chain**, meaning that

$$
\Pr(S_t = j \mid S_{t-1}, S_{t-2}, \dots, S_1)
=
\Pr(S_t = j \mid S_{t-1})
$$

for all $t \ge 2$.

So the current regime depends only on the previous regime.

---

## 3. Transition matrix

Define the transition probabilities by

$$
p_{ij} = \Pr(S_t = j \mid S_{t-1} = i),
\qquad i,j \in \{1,\dots,K\}.
$$

Collect them into the transition matrix

$$
P = (p_{ij})_{i,j=1}^K.
$$

This matrix must satisfy:

1. $p_{ij} \ge 0$ for all $i,j$,
2. each row sums to 1:
   $$
   \sum_{j=1}^K p_{ij} = 1 \qquad \text{for all } i.
   $$

Interpretation:

- row $i$ describes where the process can go **next** if it is currently in regime $i$,
- diagonal terms $p_{ii}$ measure **regime persistence**,
- off-diagonal terms measure **switching intensity**.

---

## 4. Initial regime distribution

Define the initial state probabilities by

$$
\pi_j = \Pr(S_1 = j), \qquad j=1,\dots,K.
$$

Collect them into the vector

$$
\pi = (\pi_1,\dots,\pi_K).
$$

This vector must satisfy:

1. $\pi_j \ge 0$ for all $j$,
2. $$
   \sum_{j=1}^K \pi_j = 1.
   $$

Interpretation:

- $\pi$ determines the regime uncertainty at the start of the sample,
- later, during filtering, this acts as the base prior for the recursion.

---

## 5. Regime-specific parameters

Each regime $j$ has its own parameter vector

$$
\theta_j.
$$

For the first model, use a Gaussian observation law with regime-specific mean and variance:

$$
\theta_j = (\mu_j, \sigma_j^2),
$$

where

- $\mu_j \in \mathbb{R}$ is the mean in regime $j$,
- $\sigma_j^2 > 0$ is the variance in regime $j$.

The full parameter set of the model is therefore

$$
\Theta = \left( \pi,\; P,\; \theta_1,\dots,\theta_K \right).
$$

For the Gaussian case:

$$
\Theta = \left( \pi,\; P,\; \mu_1,\dots,\mu_K,\; \sigma_1^2,\dots,\sigma_K^2 \right).
$$

---

## 6. Observation model

Conditional on the active regime, the observation is Gaussian:

$$
y_t \mid (S_t = j) \sim \mathcal N(\mu_j,\sigma_j^2).
$$

Equivalently, the conditional density is

$$
f(y_t \mid S_t = j; \theta_j)
=
\frac{1}{\sqrt{2\pi\sigma_j^2}}
\exp\left(
-\frac{(y_t-\mu_j)^2}{2\sigma_j^2}
\right).
$$

This is the **emission distribution** or **state-dependent observation density**.

Interpretation:

- the hidden chain tells you **which regime is active**,
- the regime determines **which Gaussian distribution generated the observation**.

So the model is a mixture over time, but the mixture weights are not independent across $t$; they evolve according to a Markov chain.

---

## 7. Independence structure

The model imposes two key structural assumptions.

### 7.1 Markov property of the hidden states

$$
\Pr(S_t \mid S_{t-1}, S_{t-2}, \dots, S_1)
=
\Pr(S_t \mid S_{t-1}).
$$

### 7.2 Conditional independence of observations

Given the regime path, observations are conditionally independent across time in the basic model:

$$
\Pr(y_1,\dots,y_T \mid S_1,\dots,S_T)
=
\prod_{t=1}^T \Pr(y_t \mid S_t).
$$

This is what makes the model an HMM-style latent-state model.

---

## 8. Joint distribution of states and observations

The full joint law factors as

$$
\Pr(S_1,\dots,S_T, y_1,\dots,y_T)
=
\Pr(S_1)\Pr(y_1 \mid S_1)
\prod_{t=2}^T \Pr(S_t \mid S_{t-1}) \Pr(y_t \mid S_t).
$$

Using the notation above:

$$
\Pr(S_1,\dots,S_T, y_1,\dots,y_T)
=
\pi_{S_1}
\, f(y_1 \mid S_1)
\prod_{t=2}^T
p_{S_{t-1},S_t}
\, f(y_t \mid S_t).
$$

This factorization is the core mathematical object behind:

- simulation,
- filtering,
- smoothing,
- likelihood evaluation,
- EM estimation.

---

## 9. Quantities you will later infer

Even though you are not implementing inference yet, you should already define the main probability objects.

### Predicted regime probabilities

$$
\Pr(S_t = j \mid y_{1:t-1})
$$

These represent uncertainty about the current regime **before seeing** $y_t$.

### Filtered regime probabilities

$$
\Pr(S_t = j \mid y_{1:t})
$$

These represent uncertainty about the current regime **after seeing** $y_t$.

### Smoothed regime probabilities

$$
\Pr(S_t = j \mid y_{1:T})
$$

These represent full-sample posterior probabilities of the regime at time $t$.

### Pairwise smoothed transition probabilities

$$
\Pr(S_{t-1}=i, S_t=j \mid y_{1:T})
$$

These will later be needed for EM updates of the transition matrix.

---

## 10. Likelihood object

The sample likelihood is based on the predictive densities

$$
f(y_t \mid y_{1:t-1}),
$$

and the total log-likelihood is

$$
\log L(\Theta)
=
\sum_{t=1}^T \log f(y_t \mid y_{1:t-1};\Theta).
$$

Since the current regime is hidden, the one-step predictive density is a mixture over regimes:

$$
f(y_t \mid y_{1:t-1};\Theta)
=
\sum_{j=1}^K
f(y_t \mid S_t=j;\theta_j)\,
\Pr(S_t=j \mid y_{1:t-1}).
$$

This equation is one of the most important in the entire model.  
It is the bridge between the hidden-state process and observable likelihood evaluation.

---

## 11. Interpretation of the first model

Your first model is therefore:

- a **finite-state latent Markov chain** $\{S_t\}$,
- coupled with
- a **regime-dependent Gaussian observation model**.

It is simultaneously:

- a **Markov Switching Model** in econometrics language,
- and a **Hidden Markov Model with Gaussian emissions** in statistical / algorithmic language.

That equivalence is useful because:

- Hamilton-type sources explain the time-series interpretation,
- HMM sources explain the inference mechanics.

---

## 12. Constraints that must always hold

For the model to be valid, the following constraints must always hold:

### Hidden-state layer
- $K \ge 2$,
- $p_{ij} \ge 0$,
- $\sum_{j=1}^K p_{ij}=1$ for every row $i$,
- $\pi_j \ge 0$,
- $\sum_{j=1}^K \pi_j = 1$.

### Emission layer
- $\mu_j \in \mathbb{R}$,
- $\sigma_j^2 > 0$ for every regime $j$.

These constraints should eventually shape how parameters are represented and estimated.

---

## 13. Minimal notation summary

Use the following notation consistently everywhere:

| Symbol | Meaning |
|---|---|
| $T$ | sample length |
| $K$ | number of regimes |
| $y_t$ | observed scalar at time $t$ |
| $S_t$ | hidden regime at time $t$ |
| $P=(p_{ij})$ | transition matrix |
| $\pi$ | initial regime distribution |
| $\mu_j$ | regime-$j$ mean |
| $\sigma_j^2$ | regime-$j$ variance |
| $\theta_j$ | regime-$j$ parameters |
| $\Theta$ | full model parameter set |

---

## 14. Final model statement

The first model to implement is:

$$
S_1 \sim \pi,
$$

$$
S_t \mid S_{t-1}=i \sim \text{Categorical}(p_{i1},\dots,p_{iK}),
\qquad t=2,\dots,T,
$$

and

$$
y_t \mid S_t=j \sim \mathcal N(\mu_j,\sigma_j^2),
\qquad t=1,\dots,T.
$$

This is the complete mathematical object you should keep fixed before moving to simulation, filtering, smoothing, or estimation.

---

## 15. Output of Phase 1

At the end of this phase, you should have a one-page personal reference containing:

- the hidden-state definition,
- the transition matrix definition,
- the initial distribution,
- the Gaussian regime-specific observation model,
- the joint factorization,
- the predictive / filtered / smoothed probability objects,
- the likelihood definition,
- the parameter constraints.

If this page is precise, the rest of the implementation becomes much cleaner.