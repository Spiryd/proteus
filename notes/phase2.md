# Phase 2 — Build the Model Generatively Before Inferentially

## Goal

Before implementing filtering, smoothing, or estimation, build the model as a **generative process**.

This phase is about making the mathematical object from Phase 1 operational in the simplest possible way:

1. generate a hidden regime path,
2. generate observations conditional on that path,
3. verify that the simulated data actually behaves like the model says it should.

The purpose of this phase is not just to “have fake data.”  
It is to ensure that you fully understand the model as a stochastic system before you try to infer anything from it.

---

## 1. What you are building in this phase

You are building a simulator for the model

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

So the simulator must produce two sequences:

- a hidden regime sequence
  $$
  S_1,\dots,S_T,
  $$
- an observation sequence
  $$
  y_1,\dots,y_T.
  $$

These two sequences are the complete synthetic dataset.

---

## 2. Why this phase comes before inference

This phase should come before filtering for several reasons.

### 2.1 It gives you ground truth
When you later implement filtering or EM, you will need a known reference.

With simulated data, you know:

- the true transition matrix,
- the true regime means and variances,
- the true hidden state path.

This allows you to test whether later inference is working.

### 2.2 It forces you to understand the model causally
The model is not just a set of formulas.  
It is a stochastic mechanism.

The causal story is:

1. choose an initial regime,
2. evolve the regime via a Markov chain,
3. generate an observation from the regime-specific distribution.

If that story is not perfectly clear, inference code will become fragile.

### 2.3 It becomes your permanent test bench
You should reuse this simulator throughout the project for:

- forward filter tests,
- smoother tests,
- EM recovery experiments,
- stress tests under difficult parameter settings.

---

## 3. Inputs to the simulator

Your simulator should conceptually take the following inputs.

### 3.1 Number of regimes
$$
K \in \mathbb{N}, \qquad K \ge 2.
$$

This determines the size of the hidden state space.

### 3.2 Sample length
$$
T \in \mathbb{N}, \qquad T \ge 1.
$$

This determines how many time points to generate.

### 3.3 Initial distribution
$$
\pi = (\pi_1,\dots,\pi_K),
$$

where

$$
\pi_j = \Pr(S_1=j), \qquad \sum_{j=1}^K \pi_j = 1.
$$

### 3.4 Transition matrix
$$
P=(p_{ij})_{i,j=1}^K,
$$

where each row is a probability vector:

$$
p_{ij} \ge 0,
\qquad
\sum_{j=1}^K p_{ij}=1.
$$

### 3.5 Regime-specific Gaussian parameters
For each regime $j$:

$$
\mu_j \in \mathbb{R}, \qquad \sigma_j^2 > 0.
$$

Equivalently, the observation parameters are

$$
\theta_j = (\mu_j,\sigma_j^2).
$$

---

## 4. Conceptual decomposition of the simulator

Do not think of simulation as one monolithic block.  
Split it mentally into two layers.

### Layer A — hidden-state simulation
This produces the regime path

$$
S_1,\dots,S_T.
$$

### Layer B — observation simulation
Given the regime path, this produces

$$
y_1,\dots,y_T.
$$

This decomposition mirrors the model itself and is important because later debugging often depends on checking whether the issue is in:

- state generation,
- or observation generation.

---

## 5. Step-by-step procedure

## Step 1 — Validate the parameter set

Before generating anything, verify that the parameter set is mathematically valid.

### Check:
- $K \ge 2$,
- $T \ge 1$,
- every entry of $\pi$ is nonnegative,
- $\sum_j \pi_j = 1$,
- every entry of $P$ is nonnegative,
- every row of $P$ sums to 1,
- every variance $\sigma_j^2$ is strictly positive.

### Why this matters
The simulator is the first place where invalid parameterization becomes visible.

If you do not enforce constraints here, later bugs become harder to localize.

### Output of this step
A validated parameter object for the model.

---

## Step 2 — Sample the initial hidden regime

Generate the first regime:

$$
S_1 \sim \pi.
$$

This means:

- treat $\pi$ as a categorical probability vector,
- draw one regime index from it.

### Interpretation
This initializes the latent chain.  
At time \(t=1\), the process has not yet transitioned from anything earlier, so the model starts from the prior distribution \(\pi\).

### What to verify
- repeated draws from the simulator should approximately match $\pi$ across many independent runs,
- no impossible regime should ever be sampled.

### Output of this step
The first hidden state \(S_1\).

---

## Step 3 — Simulate the rest of the hidden regime path

For each time

$$
t = 2,3,\dots,T,
$$

generate

$$
S_t \mid S_{t-1}=i \sim \text{Categorical}(p_{i1},\dots,p_{iK}).
$$

In words:

- look at the previous regime \(S_{t-1}\),
- take the corresponding row of the transition matrix,
- sample the next regime from that row.

### Interpretation
The hidden process evolves as a first-order Markov chain.

That means the current state depends only on the immediately previous state, not on the full past.

### What to verify
Across long simulations:

- high diagonal values \(p_{ii}\) should produce long runs in the same regime,
- large off-diagonal values should produce more frequent switching,
- empirical transition frequencies should roughly match the specified matrix when the sample is long enough.

### Output of this step
A full hidden regime path

$$
S_1,\dots,S_T.
$$

---

## Step 4 — Generate observations conditional on the hidden states

For each time \(t\), once \(S_t\) is known, generate

$$
y_t \mid S_t=j \sim \mathcal N(\mu_j,\sigma_j^2).
$$

In words:

- look up the active regime \(j=S_t\),
- use that regime’s mean and variance,
- sample a Gaussian observation.

### Interpretation
The hidden state determines which observation distribution is active at time \(t\).

So the observed series is generated by switching between different Gaussian laws over time.

### What to verify
If you separate observations by their true regime labels, then:

- their sample means should be close to the corresponding \(\mu_j\),
- their sample variances should be close to the corresponding \(\sigma_j^2\),
- regimes with clearly different means or variances should visibly generate different observation patterns.

### Output of this step
A full observation sequence

$$
y_1,\dots,y_T.
$$

---

## Step 5 — Package the simulation output cleanly

The result of a single simulation run should contain at least:

- the sample length \(T\),
- the number of regimes \(K\),
- the hidden state path \(S_{1:T}\),
- the observation path \(y_{1:T}\),
- the parameters used to generate the sample.

### Why this matters
Later, when testing filtering or EM, you will want to compare:

- estimated quantities,
- true parameters,
- true hidden states.

So the simulator output should not discard the hidden states.

### Output of this step
A reusable synthetic dataset object with both latent and observed information.

---

## 6. What this phase is teaching you theoretically

This phase corresponds to the **generative interpretation** of a hidden Markov model / Markov switching model.

The model is defined by:

1. a latent Markov chain,
2. an observation distribution indexed by the latent state.

Formally, the joint distribution factors as

$$
\Pr(S_1,\dots,S_T,y_1,\dots,y_T)
=
\Pr(S_1)\Pr(y_1\mid S_1)
\prod_{t=2}^T \Pr(S_t\mid S_{t-1})\Pr(y_t\mid S_t).
$$

Using your notation:

$$
\Pr(S_1,\dots,S_T,y_1,\dots,y_T)
=
\pi_{S_1} f(y_1\mid S_1)
\prod_{t=2}^T p_{S_{t-1},S_t} f(y_t\mid S_t).
$$

Simulation is nothing more than sampling from this factorization in the natural chronological order.

This is the reason Phase 2 comes directly after Phase 1:  
Phase 1 defines the object; Phase 2 makes that object operational.

---

## 7. What to test before moving on

Do not move to filtering until the simulator passes simple sanity checks.

## Test A — Initial-state sanity
Run many short independent simulations and check whether the empirical frequency of the initial regime is approximately \(\pi\).

### Purpose
Confirms that the initial distribution is being sampled correctly.

---

## Test B — Transition sanity
Run one long simulation and compute empirical transition frequencies:

- count how often \(i \to j\) occurs,
- compare with \(p_{ij}\).

### Purpose
Confirms that the hidden chain follows the specified transition matrix.

---

## Test C — Persistence sanity
Choose a matrix with large diagonal terms, for example with very persistent regimes.

### Expected behavior
The regime path should show long consecutive runs in the same state.

### Purpose
Confirms that diagonal entries are being interpreted correctly.

---

## Test D — Switching sanity
Choose a matrix with strong off-diagonal probabilities.

### Expected behavior
The regime path should switch frequently.

### Purpose
Confirms that switching behavior is responsive to the transition specification.

---

## Test E — Emission sanity
Choose regimes with clearly separated means, for example one low-mean regime and one high-mean regime.

### Expected behavior
Observations generated under the two regimes should appear visibly different.

### Purpose
Confirms that the observation model is conditioned correctly on the hidden state.

---

## Test F — Variance sanity
Choose equal means but different variances.

### Expected behavior
The hidden regimes should generate observations with similar centers but different dispersion.

### Purpose
Confirms that variance switching is implemented correctly.

---

## 8. Recommended simulation scenarios

Before moving on, simulate several qualitatively different cases.

### Scenario 1 — Strongly separated regimes
- very different means,
- moderate variances,
- persistent transitions.

### Why
This is the easiest case and should later be easy for the filter to recover.

---

### Scenario 2 — Weakly separated regimes
- similar means,
- similar variances.

### Why
This is harder and helps you understand identification difficulty.

---

### Scenario 3 — Mean switching only
- different means,
- same variances.

### Why
This isolates location effects.

---

### Scenario 4 — Variance switching only
- same means,
- different variances.

### Why
This isolates scale effects.

---

### Scenario 5 — Very persistent chain
- diagonal entries close to 1.

### Why
This helps you see long regime durations.

---

### Scenario 6 — Frequent switching chain
- diagonal entries only moderately large.

### Why
This helps you see rapid transitions and check responsiveness.

---

## 9. Practical lessons you should extract from the simulator

By the end of this phase, you should be able to answer these questions clearly.

### 9.1 What does the transition matrix actually do?
It does not directly affect the observation values.  
It affects how long the model tends to remain in each regime and how often it jumps between regimes.

### 9.2 What do the Gaussian parameters do?
They determine the shape of the observation distribution within each regime.

### 9.3 What makes regimes easier or harder to recover later?
Regimes are easier to recover when:

- their means or variances are clearly different,
- the chain is reasonably persistent,
- the sample is long enough.

Regimes are harder to recover when:

- their observation distributions overlap heavily,
- switching is very frequent,
- one regime is rarely visited.

### 9.4 Why keep the hidden states?
Because later you want to compare:

- true regime path,
- filtered probabilities,
- smoothed probabilities,
- hard classifications.

---

## 10. Common conceptual mistakes to avoid

### Mistake 1 — Confusing regime labels with observations
The regimes are hidden categorical variables.  
The observations are real-valued outputs conditional on those regimes.

### Mistake 2 — Treating the transition matrix as symmetric by default
There is no reason to assume

$$
p_{ij} = p_{ji}.
$$

The chain can be asymmetric.

### Mistake 3 — Forgetting that simulation is chronological
The correct order is:

1. sample state,
2. sample observation conditional on state.

Not the other way around.

### Mistake 4 — Throwing away the hidden path
For this project, the hidden path is part of the test infrastructure and should always be retained.

### Mistake 5 — Using only one simulation setting
A simulator is useful only if you test the model under multiple qualitatively different parameter regimes.

---

## 11. Minimal mathematical summary of Phase 2

The simulation algorithm is exactly:

### Step 1
Draw

$$
S_1 \sim \pi.
$$

### Step 2
For each \(t=2,\dots,T\), draw

$$
S_t \mid S_{t-1} \sim P(S_{t-1},\cdot).
$$

### Step 3
For each \(t=1,\dots,T\), draw

$$
y_t \mid S_t=j \sim \mathcal N(\mu_j,\sigma_j^2).
$$

This produces a sample from the joint model

$$
\Pr(S_{1:T},y_{1:T})
=
\pi_{S_1} f(y_1\mid S_1)
\prod_{t=2}^T p_{S_{t-1},S_t} f(y_t\mid S_t).
$$

---

## 12. Output of Phase 2

At the end of this phase, you should have:

- a simulator for the hidden regime sequence,
- a simulator for the observed sequence conditional on the hidden regimes,
- a reusable synthetic dataset structure,
- several test scenarios with qualitatively different transition and emission behavior,
- confidence that the model behaves as expected before you attempt inference.

If this phase is done properly, Phase 3 becomes much easier because you will already have trusted synthetic data for all later debugging and validation.