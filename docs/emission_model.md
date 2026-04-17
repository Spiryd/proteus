# Proteus — Emission Model (Phase 3)

Phase 3 isolates the **observation mechanism** of the Markov Switching Model as a clean, independent component.
It does not implement filtering, smoothing, or estimation.
Its sole purpose is to answer:

> Given an observation $y_t$ and a candidate regime $j$, what is the model-assigned density or log-density of that observation under regime $j$?

---

## The emission model

The hidden regime at time $t$ is $S_t \in \{1,\dots,K\}$.  
Conditional on $S_t = j$, the observation is drawn from a regime-specific Gaussian:

$$
y_t \mid (S_t = j) \;\sim\; \mathcal{N}(\mu_j,\, \sigma_j^2)
$$

Each regime $j$ carries a parameter pair:

$$
\theta_j = (\mu_j,\, \sigma_j^2), \qquad \mu_j \in \mathbb{R},\quad \sigma_j^2 > 0.
$$

The regime-conditional density is:

$$
f_j(y_t) \;=\; \frac{1}{\sqrt{2\pi\sigma_j^2}}\exp\!\left(-\frac{(y_t - \mu_j)^2}{2\sigma_j^2}\right)
$$

The log-density (used in all numerical work to avoid underflow) is:

$$
\log f_j(y_t) \;=\; -\tfrac{1}{2}\log(2\pi) \;-\; \tfrac{1}{2}\log(\sigma_j^2) \;-\; \frac{(y_t - \mu_j)^2}{2\sigma_j^2}
$$

---

## Module layout

```
src/
  model/
    emission.rs   — Emission struct: log_density, density, and their vector forms
    params.rs     — ModelParams (Markov chain + emission parameters combined)
    simulate.rs   — Phase 2 generative simulator
    mod.rs        — re-exports Emission, ModelParams, SimulationResult, simulate
```

`Emission` is **intentionally independent** of `ModelParams`. It does not know about the transition matrix $P$ or the initial distribution $\pi$. This boundary is the stable interface the forward filter will call in Phase 4.

---

## `Emission`

Defined in `src/model/emission.rs`.

```rust
pub struct Emission {
    pub k: usize,            // K — number of regimes, inferred from means.len()
    pub means: Vec<f64>,     // μⱼ for j = 0..k
    pub variances: Vec<f64>, // σⱼ² for j = 0..k  (must be > 0)
}
```

### Constructor

```rust
Emission::new(means: Vec<f64>, variances: Vec<f64>) -> Emission
```

### `validate() -> anyhow::Result<()>`

| Check | Condition |
|---|---|
| Length match | `means.len() == variances.len()` |
| Non-empty | K ≥ 1 |
| Variance positivity | $\sigma_j^2 > 0$ for all $j$ |

### `log_density(y: f64, j: usize) -> f64`

Returns $\log f_j(y)$ using the formula above.  
Internally computes the standardized residual $z = (y - \mu_j) / \sigma_j$ and evaluates:

```
-½ ln(2π) - ½ ln(σⱼ²) - ½ z²
```

The constant $\tfrac{1}{2}\ln(2\pi) \approx 0.9189$ is stored as a precomputed constant `HALF_LN_2PI`.

### `density(y: f64, j: usize) -> f64`

Returns $f_j(y) = \exp(\log f_j(y))$.  
Use `log_density` in all numerical work; this is provided for tests and inspection.

### `log_density_vec(y: f64) -> Vec<f64>`

Returns a `Vec` of length $K$ where entry $j$ is $\log f_j(y)$.  
**This is the method the forward filter will call once per time step.**

### `density_vec(y: f64) -> Vec<f64>`

Returns a `Vec` of length $K$ where entry $j$ is $f_j(y)$.

### Example

```rust
use proteus::model::Emission;

let emission = Emission::new(
    vec![-5.0, 5.0],  // μ₀ = -5,  μ₁ = 5
    vec![1.0,  1.0],  // σ₀² = 1,  σ₁² = 1
);
emission.validate().unwrap();

let y = 4.8;
println!("log f₀(y) = {:.4}", emission.log_density(y, 0)); // ≈ -52.7
println!("log f₁(y) = {:.4}", emission.log_density(y, 1)); // ≈ -0.94

// What the filter will call:
let log_likelihoods = emission.log_density_vec(y);
// log_likelihoods[j] = log fⱼ(y) for j = 0, 1
```

---

## Why log-density, not density

The forward filter accumulates products of the form $\prod_t f_{S_t}(y_t)$.
For long sequences, this collapses to zero in floating-point.

Working in log-space converts products to sums:

$$
\log \prod_t f_{S_t}(y_t) \;=\; \sum_t \log f_{S_t}(y_t)
$$

Exponentiation is applied only when needed (e.g., normalizing a filtered distribution), not at every step.
`log_density_vec` is the primary output of this module for exactly this reason.

---

## Geometry of the Gaussian emission

Understanding how $f_j(y)$ behaves drives the intuition behind filtering.

| Situation | Effect on $f_j(y)$ |
|---|---|
| $y$ close to $\mu_j$ | $f_j(y)$ large — regime $j$ is a plausible explanation |
| $y$ far from $\mu_j$ | $f_j(y)$ small — regime $j$ is unlikely to explain $y$ |
| $\sigma_j^2$ small | Selective: only values near $\mu_j$ get high density |
| $\sigma_j^2$ large | Diffuse: a broad range of values gets moderate density |

This is what drives posterior regime updating in the filter:  
**a regime gains probability weight when it assigns high likelihood to the current observation.**

---

## Design separation from the Markov chain

The emission model answers only:

$$
\underbrace{f_j(y_t)}_{\text{observation compatibility with regime } j}
$$

It does **not** answer:

$$
\Pr(S_t = j \mid y_{1:t}) \quad \text{// filtered probability — Phase 4}
$$

That second quantity depends on both:
- state prediction from the Markov chain: $\Pr(S_t = j \mid y_{1:t-1})$,
- and emission evaluation: $f_j(y_t)$.

Keeping these two roles in separate modules means the filter can later be written as an explicit combination:

$$
\Pr(S_t = j \mid y_{1:t}) \;\propto\; f_j(y_t) \cdot \Pr(S_t = j \mid y_{1:t-1})
$$

---

## Notation fixed for this project

| Symbol | Meaning | Type |
|---|---|---|
| $K$ | number of regimes | `usize` |
| $j$ | regime index (0-based in code, 1-based in math) | `usize` |
| $\theta_j$ | regime parameters $(\mu_j, \sigma_j^2)$ | conceptual pair |
| $\mu_j$ | regime mean | `f64` (unrestricted) |
| $\sigma_j^2$ | regime variance | `f64` (strictly positive) |
| $f_j(y_t)$ | regime-conditional density | `f64` ∈ (0, ∞) |
| $\log f_j(y_t)$ | regime-conditional log-density | `f64` ∈ (−∞, 0) |

All subsequent phases (filter, smoother, EM) use this notation without change.

---

## Test suite

All 11 tests are in `src/model/emission.rs` under `#[cfg(test)]`.  
Run with `cargo test model::emission`.

```
running 11 tests
test model::emission::tests::test_density_decreases_away_from_mean     ... ok
test model::emission::tests::test_density_symmetric_around_mean        ... ok
test model::emission::tests::test_larger_variance_lower_peak           ... ok
test model::emission::tests::test_validate_rejects_zero_variance       ... ok
test model::emission::tests::test_log_density_matches_ln_density       ... ok
test model::emission::tests::test_peak_density_at_mean                 ... ok
test model::emission::tests::test_validate_passes_valid_emission       ... ok
test model::emission::tests::test_validate_rejects_length_mismatch     ... ok
test model::emission::tests::test_validate_rejects_negative_variance   ... ok
test model::emission::tests::test_vec_methods_length_and_consistency   ... ok
test model::emission::tests::test_log_density_formula_explicit         ... ok
test result: ok. 11 passed; 0 failed; 0 ignored; finished in 0.00s
```

### Test descriptions

| Test | What is verified |
|---|---|
| `test_peak_density_at_mean` | $f_j(\mu_j) = (2\pi\sigma_j^2)^{-\tfrac{1}{2}}$ exactly |
| `test_log_density_matches_ln_density` | $\log f_j(y) = \ln(f_j(y))$ for all sample points |
| `test_density_symmetric_around_mean` | $f_j(\mu_j + \delta) = f_j(\mu_j - \delta)$ |
| `test_density_decreases_away_from_mean` | $f_j$ is monotonically decreasing away from $\mu_j$ |
| `test_larger_variance_lower_peak` | Wider $\sigma^2$ gives lower peak density |
| `test_vec_methods_length_and_consistency` | `log_density_vec` and `density_vec` match single-regime calls |
| `test_log_density_formula_explicit` | Closed-form check: $\mu=0, \sigma^2=1, y=1$ |
| `test_validate_rejects_zero_variance` | $\sigma^2 = 0$ fails validation |
| `test_validate_rejects_negative_variance` | $\sigma^2 < 0$ fails validation |
| `test_validate_rejects_length_mismatch` | Mismatched `means`/`variances` lengths fail validation |
| `test_validate_passes_valid_emission` | Valid two-regime `Emission` passes validation |

---

## Extensibility path

Phase 3 keeps the observation family fixed as Gaussian.  
When the project extends to **switching regression** or **switching autoregression**, only `Emission` changes — the hidden-state machinery, the forward filter, and the smoother remain untouched.

The conceptual interface is already established:

```
Input:   y_t (f64),  j (usize),  θⱼ stored in Emission
Output:  log fⱼ(y_t) (f64)
```

A future `EmissionAR` or `EmissionRegression` struct would satisfy the same calling convention.

---

## What comes next (Phase 4)

Phase 4 implements the **forward filter** (Hamilton filter):

$$
\Pr(S_t = j \mid y_{1:t}) \;\propto\; f_j(y_t) \;\cdot\; \sum_{i=1}^K p_{ij}\,\Pr(S_{t-1} = i \mid y_{1:t-1})
$$

The emission model provides $f_j(y_t)$ via `Emission::log_density_vec`.  
The Markov chain provides the predicted weights $\sum_i p_{ij}\,\Pr(S_{t-1}=i \mid \cdot)$ via `ModelParams::transition_row`.
