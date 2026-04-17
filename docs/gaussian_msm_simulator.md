# Proteus — Gaussian Markov Switching Model Simulator (Phase 2)

Phase 2 builds the model as a **generative process** before any inference.  
The simulator produces ground-truth hidden regime paths and observations that are reused throughout the project as a test bench for filtering, smoothing, and EM.

---

## Dependencies added

| Crate | Purpose |
|---|---|
| `rand 0.10` | Random number generation, `Rng` trait, `SeedableRng` |
| `rand_distr 0.6` | `WeightedIndex` (categorical draws), `Normal` (Gaussian draws) |

---

## Module layout

```
src/
  model/
    mod.rs        — re-exports ModelParams, SimulationResult, simulate
    params.rs     — ModelParams struct and validate()
    simulate.rs   — simulate_hidden_path, simulate_observations, simulate
```

---

## The model

The simulator samples from the joint distribution of a Gaussian hidden Markov model:

```
S₁ ~ π
Sₜ | S_{t-1}=i ~ Categorical(p_{i1}, …, p_{iK})   t = 2, …, T
yₜ | Sₜ=j      ~ N(μⱼ, σⱼ²)                       t = 1, …, T
```

i.e. the latent chain governs which Gaussian distribution is active at each time step.

---

## `ModelParams`

Defined in `src/model/params.rs`.

```rust
pub struct ModelParams {
    pub k: usize,              // number of regimes K ≥ 2
    pub pi: Vec<f64>,          // initial distribution  (length K)
    pub transition: Vec<f64>,  // transition matrix P   (K×K, row-major)
    pub means: Vec<f64>,       // regime means μⱼ       (length K)
    pub variances: Vec<f64>,   // regime variances σⱼ²  (length K)
}
```

### Constructor

```rust
ModelParams::new(pi, transition_rows, means, variances) -> ModelParams
```

`transition_rows` is a `Vec<Vec<f64>>` (one inner `Vec` per row) that is flattened internally.

### `transition_row(i) -> &[f64]`

Returns a slice into row `i` of the stored flat transition matrix.

### `validate() -> anyhow::Result<()>`

Checks that the parameter set is mathematically valid before any sampling:

| Check | Condition |
|---|---|
| Enough regimes | K ≥ 2 |
| π length | len(π) == K |
| π non-negative | every πⱼ ≥ 0 |
| π sums to 1 | \|∑πⱼ − 1\| < 1e-9 |
| P shape | K² entries |
| P non-negative | every pᵢⱼ ≥ 0 |
| P rows sum to 1 | \|∑ⱼ pᵢⱼ − 1\| < 1e-9 for each i |
| Variances positive | σⱼ² > 0 for each j |
| Correct lengths | means and variances both length K |

Returns an error with a descriptive message on the first violation found.

---

## `SimulationResult`

Defined in `src/model/simulate.rs`.

```rust
pub struct SimulationResult {
    pub t: usize,                // T — number of time steps
    pub k: usize,                // K — number of regimes
    pub states: Vec<usize>,      // hidden path S₁,…,S_T  (0-based indices)
    pub observations: Vec<f64>,  // observation sequence y₁,…,y_T
    pub params: ModelParams,     // parameter set that generated this sample
}
```

Both the hidden path and the observations are retained so the result can serve as ground truth for later inference tests.

---

## `simulate`

```rust
pub fn simulate(
    params: ModelParams,
    t: usize,
    rng: &mut impl Rng,
) -> anyhow::Result<SimulationResult>
```

Entry point. Validates `params`, then runs the two-layer simulation:

1. **Layer A** — hidden regime path via `simulate_hidden_path`
2. **Layer B** — observations via `simulate_observations`

The caller is responsible for constructing and seeding `rng`, which keeps reproducibility fully under caller control.

Returns an error if `t < 1` or if `params.validate()` fails.

### Example

```rust
use proteus::model::{ModelParams, simulate};
use rand::{SeedableRng, rngs::SmallRng};

let params = ModelParams::new(
    vec![0.5, 0.5],
    vec![vec![0.99, 0.01], vec![0.01, 0.99]],
    vec![-5.0, 5.0],
    vec![1.0, 1.0],
);
let mut rng = SmallRng::seed_from_u64(42);
let result = simulate(params, 1_000, &mut rng).unwrap();

println!("T={}, K={}", result.t, result.k);
println!("First 5 states:  {:?}", &result.states[..5]);
println!("First 5 obs:     {:?}", &result.observations[..5]);
```

---

## Simulation decomposition

### `simulate_hidden_path`

```rust
fn simulate_hidden_path(params: &ModelParams, t: usize, rng: &mut impl Rng) -> Vec<usize>
```

- Draws `S₁ ~ π` using `WeightedIndex::new(&params.pi)`.
- For each subsequent step, draws `Sₜ ~ WeightedIndex(transition_row(S_{t-1}))`.

### `simulate_observations`

```rust
fn simulate_observations(params: &ModelParams, states: &[usize], rng: &mut impl Rng) -> Vec<f64>
```

- For each time `t`, reads the active regime `j = states[t]`.
- Draws `yₜ ~ Normal(μⱼ, σⱼ)` using `rand_distr::Normal::new(μⱼ, √σⱼ²)`.

---

## Test suite

All 10 tests are in `src/model/simulate.rs` under `#[cfg(test)]`.  
Run with `cargo test model::simulate`.

```
running 10 tests
test model::simulate::tests::test_validate_rejects_bad_transition_row ... ok
test model::simulate::tests::test_validate_rejects_bad_pi_sum         ... ok
test model::simulate::tests::test_validate_rejects_k1                 ... ok
test model::simulate::tests::test_validate_rejects_nonpositive_variance ... ok
test model::simulate::tests::test_e_emission_means                    ... ok
test model::simulate::tests::test_f_emission_variances                ... ok
test model::simulate::tests::test_a_initial_state_distribution        ... ok
test model::simulate::tests::test_b_transition_frequencies            ... ok
test model::simulate::tests::test_d_switching                         ... ok
test model::simulate::tests::test_c_persistence                       ... ok
test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; finished in 0.08s
```

### Statistical sanity tests (A–F)

| Test | Scenario | What is verified | Tolerance |
|---|---|---|---|
| A | 20 000 single-step runs | Empirical frequency of `S₁` ≈ π | ±3% absolute |
| B | 1 long run of T=100 000 | Empirical transition counts ≈ P | ±1% absolute per cell |
| C | p_{ii}=0.99 (persistent), T=200 000 | Mean run length > 50 | — |
| D | p_{ij}=0.5 (uniform), T=100 000 | Mean run length < 5 | — |
| E | μ₀=−10, μ₁=10, T=50 000 | Per-regime sample mean ≈ μⱼ | ±0.1 absolute |
| F | σ²₀=0.1, σ²₁=10, T=50 000 | Per-regime sample variance ≈ σⱼ² | ±10% relative |

### Validation rejection tests

| Test | Invalid input | Expected outcome |
|---|---|---|
| `test_validate_rejects_k1` | K=1 | `validate()` returns `Err` |
| `test_validate_rejects_bad_pi_sum` | π sums to 0.8 | `validate()` returns `Err` |
| `test_validate_rejects_bad_transition_row` | Row 0 sums to 1.5 | `validate()` returns `Err` |
| `test_validate_rejects_nonpositive_variance` | σ²=−1 | `validate()` returns `Err` |

---

## Recommended simulation scenarios

| Scenario | π | P diagonal | μ | σ² | Notes |
|---|---|---|---|---|---|
| Strongly separated | (0.5, 0.5) | 0.99 | (−10, 10) | (1, 1) | Easiest for filter recovery |
| Weakly separated | (0.5, 0.5) | 0.9 | (−1, 1) | (1, 1) | Tests identification difficulty |
| Mean switching only | (0.5, 0.5) | 0.95 | (−5, 5) | (2, 2) | Isolates mean effect |
| Variance switching only | (0.5, 0.5) | 0.95 | (0, 0) | (0.5, 5) | Isolates variance effect |
| Three regimes | (⅓, ⅓, ⅓) | 0.9 | (−5, 0, 5) | (1, 1, 1) | Tests K>2 path |

---

## What comes next (Phase 3)

Phase 3 will implement the **forward filter** (Hamilton filter):

- compute filtered probabilities `P(Sₜ | y₁,…,yₜ)` recursively,
- use the `SimulationResult` from Phase 2 as ground-truth input,
- verify that filtered state probabilities concentrate on the true hidden states.
