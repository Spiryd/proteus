# Chapter 6 — Synthetic Scenario Generation and Calibration

**Scope:** `src/model/simulate.rs`, `src/model/params.rs`, `src/calibration/`, `src/experiments/synthetic_backend.rs` (the `resolve_data` stage and the `build_synthetic_params` helper), `src/experiments/registry.rs` (synthetic entries), and `src/benchmark/truth.rs`. This chapter covers everything between *"there is a generative model"* and *"a synthetic experiment can be benchmarked against ground truth"*, including how synthetic scenarios are anchored to real-market summary statistics rather than chosen arbitrarily.

**Relation to previous chapters:** Chapter 2 specified the Gaussian MSM as a generative process. Chapter 3 fitted it from data with EM. Chapter 5 defined the empirical observation sequence `y_{1:T}` that the detector consumes. This chapter closes the loop: it uses the *same* model from Chapter 2, calibrated by *empirical statistics* from Chapter 5, to produce synthetic streams with **known** hidden-state paths that serve as the ground truth against which the detector of Chapter 4 is evaluated.

**Source files analysed and cross-checked with docs:**

| Source file | Purpose | Cross-referenced doc |
|---|---|---|
| `src/model/params.rs` | `ModelParams` struct + validate | `docs/gaussian_msm_simulator.md` |
| `src/model/simulate.rs` | `simulate_hidden_path`, `simulate_observations`, `SimulationResult` | `docs/gaussian_msm_simulator.md` |
| `src/calibration/summary.rs` | Empirical summary functionals $T_1, \dots, T_m$ | `docs/synthetic_to_real_calibration.md` |
| `src/calibration/mapping.rs` | Calibration operator $\mathcal{K}$: targets → $\vartheta$ | `docs/synthetic_to_real_calibration.md` |
| `src/calibration/verify.rs` | Synthetic-vs-empirical tolerance checks | `docs/synthetic_to_real_calibration.md` |
| `src/calibration/report.rs` | End-to-end workflow + artifact serialisation | `docs/synthetic_to_real_calibration.md` |
| `src/calibration/mod.rs` | Module orchestration / re-exports | `docs/synthetic_to_real_calibration.md` |
| `src/experiments/synthetic_backend.rs` | `resolve_data` for synthetic mode, `build_synthetic_params` | `docs/experiment_runner.md` |
| `src/experiments/registry.rs` | Registered scenario IDs (`hard_switch`, `posterior_transition`, `surprise`) | `docs/experiment_runner.md` |
| `src/benchmark/truth.rs` | `ChangePointTruth` from regime path | `docs/benchmark_framework.md` |

---

## 6.1 Purpose of synthetic experiments

Real-data experiments suffer from a structural problem: nobody knows where the *true* regime changes are. The empirical record contains only prices, not latent states. Any claim that a detector "found a change" can therefore only be evaluated against a *proxy* for truth (a financial-news label, a volatility breakpoint inferred ex-post, etc.), and proxies are themselves estimates with their own errors.

Synthetic experiments solve this by *generating* the data from a known stochastic process. Because the generator emits the hidden path $S_{1:T}$ alongside the observations $y_{1:T}$, every changepoint is known **by construction, not by inference**. This makes the synthetic side of the project the only setting where the standard event-matching metrics (recall, precision, alarm-time error) are computable without circularity.

[docs/synthetic_to_real_calibration.md](docs/synthetic_to_real_calibration.md) §1 frames the same idea in terms of two "worlds":

| World | Strength | Weakness |
|---|---|---|
| A — Synthetic | Known changepoints, rigorous benchmark truth | Risk of being a toy unrelated to markets |
| B — Real | Financial relevance, external validity | No ground truth |

A purely synthetic study is methodologically clean but financially irrelevant. A purely real study is financially relevant but cannot be benchmarked. The calibration layer (§6.4–6.6) is the bridge that lets the synthetic world inherit empirical realism without losing its ground-truth guarantee.

---

## 6.2 The synthetic Markov-switching generator

The generator is the same Gaussian MSM as Chapter 2, sampled forward. The full implementation lives in [src/model/simulate.rs](src/model/simulate.rs).

The sampling distribution is

$$
S_1 \sim \pi, \qquad
S_t \mid S_{t-1} = i \sim \mathrm{Cat}(p_{i,1}, \dots, p_{i,K}), \qquad
y_t \mid S_t = j \sim \mathcal{N}(\mu_j, \sigma_j^2),
$$

with the parameter set $\vartheta = (\pi, P, \mu, \sigma^2)$ packaged as [`ModelParams`](src/model/params.rs):

```rust
pub struct ModelParams {
    pub k: usize,
    pub pi: Vec<f64>,           // length K
    pub transition: Vec<f64>,   // K×K row-major
    pub means: Vec<f64>,
    pub variances: Vec<f64>,
}
```

`ModelParams::validate()` enforces nine invariants before any sampling: $K \ge 2$; $\pi, P$ rows are non-negative; $\pi$ sums to $1$; each row of $P$ sums to $1$ (tolerance $10^{-9}$); variances strictly positive; lengths consistent. Sampling cannot start with an invalid parameter set.

The forward sampler is split into two stages for clarity:

```rust
// src/model/simulate.rs (lines 56–91)
fn simulate_hidden_path(params: &ModelParams, t: usize, rng: &mut impl Rng) -> Vec<usize> {
    let mut states = Vec::with_capacity(t);
    let init_dist = WeightedIndex::new(&params.pi).expect("pi was already validated");
    states.push(init_dist.sample(rng));
    for _ in 1..t {
        let prev = *states.last().unwrap();
        let row = params.transition_row(prev);
        let row_dist = WeightedIndex::new(row).expect("transition row was already validated");
        states.push(row_dist.sample(rng));
    }
    states
}

fn simulate_observations(params: &ModelParams, states: &[usize], rng: &mut impl Rng) -> Vec<f64> {
    states.iter().map(|&j| {
        let mu = params.means[j];
        let sigma = params.variances[j].sqrt();
        Normal::new(mu, sigma).expect("variance was already validated").sample(rng)
    }).collect()
}
```

Two implementation points matter for reproducibility:

1. **`rand_distr::WeightedIndex`** is used for the categorical draws of $S_1$ and the transition rows. This is an $O(K)$ alias-free implementation that is deterministic given the RNG state.
2. **`rand_distr::Normal::new(mu, sigma)`** takes the *standard deviation*, not the variance. The cast `params.variances[j].sqrt()` makes the contract explicit and is the unique conversion point — the rest of the codebase always works with variances.

The driver `simulate(params, t, rng) -> SimulationResult` calls the two stages sequentially and returns both the observations and the hidden path together so that the latter can be retained as ground truth.

---

## 6.3 Known changepoints and benchmark ground truth

A changepoint, in the synthetic world, is defined operationally: **the index $t \in \{2, \dots, T\}$ at which $S_t \ne S_{t-1}$**. This is the single source of truth that the project uses.

The definition appears in two places that are required to agree:

```rust
// src/model/simulate.rs (lines 44–52)
pub fn changepoints(&self) -> Vec<usize> {
    (1..self.states.len())
        .filter(|&i| self.states[i] != self.states[i - 1])
        .map(|i| i + 1) // 0-based → 1-based time step
        .collect()
}
```

```rust
// src/benchmark/truth.rs (from_regime_sequence)
pub fn from_regime_sequence(regimes: &[usize]) -> Result<Self> {
    let stream_len = regimes.len();
    let times: Vec<usize> = (1..stream_len)
        .filter(|&i| regimes[i] != regimes[i - 1])
        .map(|i| i + 1)
        .collect();
    Self::new(times, stream_len)
}
```

Three properties of this construction are non-negotiable downstream:

1. **1-based time index.** The detector's `OnlineStepResult::t` is 1-based, and the benchmark matcher uses the same convention. Every changepoint $\tau$ satisfies $1 < \tau \le T$.
2. **Strictly increasing.** `ChangePointTruth::new` rejects duplicates and out-of-order entries, so any bug that produced an invalid path would be caught before evaluation.
3. **No-change is a valid case.** An empty changepoint list with `stream_len = T` represents a stream that stays in one regime for all $T$ observations; in that case every detector alarm is a false positive. The matcher and metrics in [src/benchmark/](src/benchmark/) handle this case explicitly.

In `SyntheticBackend::resolve_data` ([src/experiments/synthetic_backend.rs](src/experiments/synthetic_backend.rs#L48-L70)) the truth is generated alongside the data:

```rust
let params = build_synthetic_params(scenario_id, cfg.model.k_regimes)?;
let sim = simulate(params, horizon, &mut rng)?;
let changepoints = sim.changepoints();
```

and packaged in the `DataBundle` as `changepoint_truth: Some(changepoints)`. The real backend ([src/experiments/real_backend.rs](src/experiments/real_backend.rs)) returns `None` here — the type system itself records the fact that ground truth is only available for synthetic runs.

### 6.3.1 Reproducibility of the truth

Because both the hidden path and the observations are drawn from the same `StdRng` seeded by `cfg.reproducibility.seed`, re-running the experiment with the same seed produces *exactly* the same changepoint set. `regenerate_changepoints` ([src/experiments/synthetic_backend.rs](src/experiments/synthetic_backend.rs#L344-L362)) exploits this to recover the truth at the evaluation stage without persisting it explicitly — a cleaner design than caching ground-truth files that could drift out of sync with the data they describe.

---

## 6.4 Synthetic-to-real calibration

If the synthetic generator parameters are chosen by hand, the resulting series may have nothing to do with the volatility scale, persistence, or tail behaviour of the market. The calibration layer formalises the alternative: choose $\vartheta$ so that selected empirical summary functionals of $y^{\text{syn}}$ approximately match those of $y^{\text{real}}$.

The calibration operator, stated formally in [docs/synthetic_to_real_calibration.md](docs/synthetic_to_real_calibration.md) §2, is

$$
\mathcal{K}: \big( T_1(y^{\text{real}}),\, \dots,\, T_m(y^{\text{real}}) \big) \;\longmapsto\; \vartheta.
$$

The objective is **statistical anchoring**, not distributional cloning. A perfect match between synthetic and real distributions would defeat the purpose of having a generator at all: the Gaussian MSM is deliberately *simpler* than real markets so that its parameters are interpretable and its hidden states have an unambiguous meaning.

A second hard requirement is the **feature-consistency condition** (§3 of the same document, enforced by the type signature of [`summarize_feature_stream`](src/calibration/summary.rs)): the empirical summaries must be computed on the *same* `FeatureStream::observations` that the detector consumes, not on raw prices. Calibrating on raw prices while the model observes log-returns would be methodologically incoherent.

```rust
// src/calibration/summary.rs
/// - Do not calibrate on raw prices if the model observes transformed values.
/// - Compute summaries on `FeatureStream::observations` (the actual y_t).
pub fn summarize_feature_stream(
    stream: &FeatureStream,
    partition: CalibrationPartition,
    n_train: usize,
    n_validation: usize,
    targets: SummaryTargetSet,
) -> EmpiricalCalibrationProfile { /* ... */ }
```

A third requirement is **leakage safety**: the default partition is `CalibrationPartition::TrainOnly`, so empirical summaries never see validation or test data. The same partition discipline already enforced for the scaler (Chapter 5 §5.8.2) is inherited verbatim.

---

## 6.5 Empirical summary statistics

The set of summary functionals computed by `summarize_observation_values` is:

| Field | Symbol | Meaning |
|---|---|---|
| `mean` | $\bar y$ | First moment of $y_t$ |
| `variance`, `std_dev` | $s_y^2, s_y$ | Second moment, population normalisation |
| `q01, q05, q50, q95, q99` | $q_p$ | Linear-interpolation quantiles |
| `tail_freq_q95` | $\Pr(y_t > q_{0.95})$ | Right-tail rate |
| `abs_exceed_2std` | $\Pr(|y_t| > 2 s_y)$ | Two-sigma exceedance |
| `acf1`, `abs_acf1` | $\hat\rho_1(y), \hat\rho_1(|y|)$ | Lag-1 dependence proxies |
| `sign_change_rate` | — | Fraction of consecutive pairs with opposite signs |
| `high_episode_mean_duration` | $\hat d_{\text{high}}$ | Mean run length of $\{|y_t| > q_{0.95}\}$ |
| `low_episode_mean_duration` | $\hat d_{\text{low}}$ | Mean run length of $\{|y_t| \le q_{0.95}\}$ |

The full struct is in [src/calibration/summary.rs](src/calibration/summary.rs) (`EmpiricalSummary`).

Three blocks of statistics serve three distinct roles:

- **Marginal anchors** (`mean`, `variance`, quantiles) tie the synthetic emission distribution to the empirical marginal shape — they will calibrate $\mu_j, \sigma_j^2$.
- **Dependence proxies** (`acf1`, `abs_acf1`, `sign_change_rate`) measure short-range structure that an i.i.d. emission cannot produce; persistence in $|y_t|$ is the signature of regime structure (volatility clustering) and the principal motivation for using a switching model at all.
- **Episode-duration proxies** (`high_episode_mean_duration`, `low_episode_mean_duration`) translate "how long does turbulence last?" into a number, and feed directly into the duration-to-persistence mapping of §6.6.

The episode-duration estimator thresholds $|y_t|$ at $q_{0.95}$ and computes the mean run length of consecutive bars on each side. This is intentionally coarse: a finer estimator (e.g. fitting an HMM to the same data) would entangle the calibration with the very inference procedure under test.

For the verification run on `real_spy_daily_hard_switch` (SPY daily log-returns, train-partition, $n=1462$ feature observations), the computed empirical summary in [verification/verify_2026_05_03b/calibration/real/calibration_summary.json](verification/verify_2026_05_03b/calibration/real/calibration_summary.json) is:

| Field | Value |
|---|---|
| `mean` | $7.79 \times 10^{-4}$ (z-scored returns) |
| `std_dev` | $0.99990$ |
| `q05`, `q50`, `q95` | $-1.573,\ 0.022,\ 1.307$ |
| `acf1` | $-0.148$ |
| `abs_acf1` | $0.374$ |
| `sign_change_rate` | $0.508$ |
| `low_episode_mean_dur` | $10.18$ |
| `high_episode_mean_dur` | $1.43$ |

The numbers tell a coherent story: returns are essentially zero-mean with unit variance after scaling, mildly negatively autocorrelated, but **strongly persistent in magnitude** (`abs_acf1 ≈ 0.37`) with low-magnitude episodes lasting ~10 bars on average and high-magnitude episodes ~1.4 bars. This is the empirical signature that the calibration mapping in §6.6 will encode into the synthetic transition matrix.

---

## 6.6 The calibration mapping

The mapping operator $\mathcal{K}$ is implemented in [src/calibration/mapping.rs](src/calibration/mapping.rs) by `calibrate_to_synthetic`, which is configured by `CalibrationMappingConfig`:

```rust
pub struct CalibrationMappingConfig {
    pub k: usize,
    pub horizon: usize,
    pub mean_policy: MeanPolicy,           // ZeroCentered | SymmetricAroundEmpirical | EmpiricalBaseline
    pub variance_policy: VariancePolicy,   // QuantileAnchored | RatioAroundEmpirical { low_mult, high_mult } | MagnitudeConditioned
    pub target_durations: Vec<f64>,        // d_1, ..., d_K  (empty → infer from episode durations)
    pub symmetric_offdiag: bool,
    pub jump: Option<JumpContamination>,
    pub strategy: CalibrationStrategy,     // Summary | QuickEm { max_iter, tol }
    pub min_high_low_ratio: f64,           // C′2 — minimum sigma_high/sigma_low under quantile-derived policies (default 1.0)
}
```

`VariancePolicy::MagnitudeConditioned` (C′1) median-splits the raw partition
observations on $|y|$ and uses the within-half sample variances as the
low/high regime variances; it falls back to `QuantileAnchored` (recorded in
`mapping_notes`) when raw observations are unavailable.
`min_high_low_ratio` (C′2) enforces a floor on
$\sigma_{\text{high}}/\sigma_{\text{low}}$ for `QuantileAnchored` and
`MagnitudeConditioned`, rescaling log-symmetrically around the geometric
mean and recording the action in `mapping_notes`.

Each policy field corresponds to one block of empirical targets from §6.5.

### 6.6.1 Mean policy

Three policies are available, encoded in `map_means`:

- `EmpiricalBaseline` (default): $\mu_j = \bar y$ for all $j$. Regime separation is carried entirely by variance, which matches the dominant pattern in financial returns where conditional means are small relative to conditional standard deviations.
- `ZeroCentered`: $\mu = (-\tfrac{1}{4} s_y,\ +\tfrac{1}{4} s_y)$ for $K=2$. Used for studying mean-shift sensitivity in isolation.
- `SymmetricAroundEmpirical`: $\mu = (\bar y - \tfrac{1}{4} s_y,\ \bar y + \tfrac{1}{4} s_y)$ for $K=2$. Combines the previous two.

The $1/4$ separation factor is a fixed design choice: it is small enough to keep the synthetic series statistically close to a single-regime null, ensuring that the detector's success cannot be attributed merely to obvious mean shifts.

### 6.6.2 Variance policy

The default `QuantileAnchored` policy extracts low- and high-volatility scales from the empirical quantile spread by inverting the Gaussian quantile function. For a $\mathcal{N}(0, \sigma^2)$ distribution, $q_{0.95} = 1.64485\,\sigma$, so the policy estimates

$$
\sigma_{\text{low}} = \frac{|q_{0.50} - q_{0.05}|}{1.64485},
\qquad
\sigma_{\text{high}} = \frac{|q_{0.95} - q_{0.50}|}{1.64485}.
$$

```rust
// src/calibration/mapping.rs
let low_sigma  = ((profile.summary.q50 - profile.summary.q05).abs() / 1.64485).max(1e-6);
let high_sigma = ((profile.summary.q95 - profile.summary.q50).abs() / 1.64485).max(1e-6);
```

The variances are then sorted ascending (`sort_by`) so the regime labels are canonical: regime 0 is always the calm regime, regime 1 the turbulent one. If the sorted vector is degenerate (e.g. perfectly symmetric quantiles), the code spreads the variances by a $10\%$ multiplicative grid to avoid identifiability failure in the downstream EM step.

For the SPY calibration above, the policy produces $\sigma_{\text{low}}^2 = 0.610$ and $\sigma_{\text{high}}^2 = 0.941$ — both close to $1$ because the input was z-scored, but separated enough to define two distinguishable regimes.

The alternative `RatioAroundEmpirical { low_mult, high_mult }` multiplies the empirical variance by user-specified factors and is used when the researcher wants to study a specific volatility ratio independent of the empirical quantile shape.

### 6.6.3 Duration-to-persistence mapping

This is the central formula of the calibration layer:

$$
\boxed{\;p_{jj} = 1 - \frac{1}{d_j}\;}
$$

where $d_j$ is the *expected* time spent in regime $j$ before a transition. The derivation is elementary: if $S_t = j$ and transitions are i.i.d. Bernoulli with self-stay probability $p_{jj}$, the dwell time is geometric with mean $1/(1 - p_{jj})$, so $\mathbb{E}[\text{dwell}_j] = d_j \iff p_{jj} = 1 - 1/d_j$.

The mapping is implemented exactly in [src/calibration/mapping.rs](src/calibration/mapping.rs):

```rust
/// Map target expected durations d_j to transition probabilities using
/// p_jj = 1 - 1/d_j and symmetric off-diagonal allocation.
fn durations_to_transition_rows(
    durations: &[f64],
    symmetric_offdiag: bool,
) -> anyhow::Result<Vec<Vec<f64>>> {
    let k = durations.len();
    let mut rows = vec![vec![0.0; k]; k];
    for i in 0..k {
        let d = durations[i];
        if d <= 1.0 || !d.is_finite() {
            anyhow::bail!("duration d[{i}] must be >1 and finite, got {d}");
        }
        let p_ii = (1.0 - 1.0 / d).clamp(1e-6, 1.0 - 1e-6);
        rows[i][i] = p_ii;
        let rem = 1.0 - p_ii;
        if symmetric_offdiag {
            let each = rem / (k - 1) as f64;
            for (j, cell) in rows[i].iter_mut().enumerate() {
                if j != i { *cell = each; }
            }
        } else {
            let next = (i + 1) % k;
            rows[i][next] = rem;
        }
    }
    Ok(rows)
}
```

Three design choices are worth pointing out:

1. **The hard bound $d_j > 1$.** A duration of $\le 1$ would imply $p_{jj} \le 0$, i.e. a regime that cannot persist for even one step on average. This is mathematically incompatible with the Markov structure and is rejected explicitly.
2. **Clamping $p_{jj}$ to $[10^{-6},\ 1 - 10^{-6}]$.** Both extremes (perfect persistence and instantaneous switching) cause numerical instability in EM and the forward filter. The clamp guarantees an interior point.
3. **Symmetric off-diagonal allocation.** When `symmetric_offdiag = true`, the remaining $1 - p_{jj}$ probability mass is split uniformly across the other $K-1$ regimes. This is the most uninformative choice given only the duration target; encoding a preferred next-regime would require additional empirical inputs that the project does not estimate.

For the SPY calibration the inferred durations are $d = (10.18,\ 2.00)$, taken from `(low_episode_mean_duration, high_episode_mean_duration)` with the lower bound at 2 to satisfy $d_j > 1$. Applying $p_{jj} = 1 - 1/d_j$ yields

$$
p_{00} = 1 - \frac{1}{10.18} = 0.9018, \qquad p_{11} = 1 - \frac{1}{2.00} = 0.5,
$$

which matches the persisted `transition` matrix in [verification/verify_2026_05_03b/calibration/real/calibrated_scenario.json](verification/verify_2026_05_03b/calibration/real/calibrated_scenario.json) to seven decimal places.

### 6.6.4 The complete mapping in one place

End-to-end, the operator $\mathcal{K}$ for $K = 2$ is

$$
\begin{aligned}
\mu_j &= \bar y && \text{(EmpiricalBaseline)} \\
\sigma_j^2 &\in \left\{\, \big(\tfrac{|q_{0.50}-q_{0.05}|}{1.64485}\big)^2,\ \big(\tfrac{|q_{0.95}-q_{0.50}|}{1.64485}\big)^2 \,\right\}, && \text{(QuantileAnchored, sorted)} \\
p_{jj} &= 1 - \frac{1}{d_j}, \quad d = (\hat d_{\text{low}},\ \hat d_{\text{high}}) && \\
p_{ij} &= \frac{1 - p_{ii}}{K-1}, \quad i \ne j && \text{(symmetric off-diagonal)} \\
\pi_j &= 1/K && \text{(uniform initial distribution)}
\end{aligned}
$$

Every quantity on the right-hand side is an explicit empirical summary; every quantity on the left-hand side is a `ModelParams` field. The mapping is deterministic, auditable, and JSON-serialisable.

---

## 6.7 Calibrated scenario families

Three synthetic scenarios are pre-registered in [src/experiments/synthetic_backend.rs](src/experiments/synthetic_backend.rs#L292-L336) via `build_synthetic_params(scenario_id, k)`. They are exposed in the registry ([src/experiments/registry.rs](src/experiments/registry.rs#L28-L45)) as `hard_switch`, `posterior_transition`, and `surprise`.

| Scenario ID | $K$ | $\pi$ | Transition $P$ | $\mu$ | $\sigma^2$ | Designed to test |
|---|---|---|---|---|---|---|
| `scenario_calibrated` / `calm_turbulent` | 2 | $(0.5, 0.5)$ | $\begin{pmatrix} 0.96 & 0.04 \\ 0.12 & 0.88 \end{pmatrix}$ | $(2 \!\times\! 10^{-4},\ -1 \!\times\! 10^{-4})$ | $(10^{-4},\ 4 \!\times\! 10^{-4})$ | Baseline calm-vs-turbulent regime split |
| `persistent_states` | 2 | $(0.5, 0.5)$ | $\begin{pmatrix} 0.975 & 0.025 \\ 0.067 & 0.933 \end{pmatrix}$ | $(10^{-4},\ -10^{-4})$ | $(8 \!\times\! 10^{-5},\ 3.2 \!\times\! 10^{-4})$ | Long dwell times: $d \approx (40, 15)$ |
| `shock_contaminated` | 2 | $(0.5, 0.5)$ | $\begin{pmatrix} 0.96 & 0.04 \\ 0.12 & 0.88 \end{pmatrix}$ | $(2 \!\times\! 10^{-4},\ -5 \!\times\! 10^{-4})$ | $(10^{-4},\ 1.6 \!\times\! 10^{-3})$ | Asymmetric high-vol regime with large $\sigma_2$ |
| (`k = 3` fallback) | 3 | $(0.34, 0.33, 0.33)$ | $\begin{pmatrix} 0.971 & 0.020 & 0.009 \\ 0.013 & 0.930 & 0.057 \\ 0.007 & 0.136 & 0.857 \end{pmatrix}$ | $(3 \!\times\! 10^{-4},\ 0,\ -2 \!\times\! 10^{-4})$ | $(9 \!\times\! 10^{-5},\ 1.8 \!\times\! 10^{-4},\ 4 \!\times\! 10^{-4})$ | Three-regime extension for capacity tests |

Each transition matrix is consistent with the $p_{jj} = 1 - 1/d_j$ map for plausible $d_j$ values. The `persistent_states` row of $P$, for example, gives $p_{00} = 0.975$ which inverts to $d_0 = 1/(1-0.975) = 40$, matching the docstring comment "durations ~40 and ~15".

The three scenarios are **not** parameter sweeps of a single design. They differ along orthogonal axes — switching frequency, dwell length, asymmetry — so that experiments comparing detectors across scenarios test different aspects of detector behaviour. This is why all four detector variants (`HardSwitch`, `PosteriorTransition`, `Surprise`, `Combined`) are evaluated on each scenario in the joint-optimisation grid; see verification Step 18.

---

## 6.8 Verification of synthetic realism

A calibrated scenario is useful only if the synthetic data it produces actually resembles the empirical target on the same summary functionals. The verification layer in [src/calibration/verify.rs](src/calibration/verify.rs) closes the loop with two artifacts:

```rust
pub struct CalibrationDiff {
    pub mean_abs_err:        f64,
    pub variance_rel_err:    f64,
    pub q05_abs_err:         f64,
    pub q95_abs_err:         f64,
    pub abs_acf1_abs_err:    f64,
    pub sign_change_abs_err: f64,
}

pub struct VerificationTolerance {
    pub mean_abs_max:         f64,  // default 0.25
    pub variance_rel_max:     f64,  // default 0.50
    pub quantile_abs_max:     f64,  // default 0.30
    pub abs_acf1_abs_max:     f64,  // default 0.20
    pub sign_change_abs_max:  f64,  // default 0.20
}
```

`verify_calibration` simulates the calibrated model for the same horizon as the empirical sample, recomputes the same summaries on the synthetic series, and reports per-field absolute / relative errors plus a boolean `within_tolerance` flag. The defaults are deliberately loose because the calibration target is statistical anchoring, not distributional cloning — strict tolerances would force the rejection of any model whose marginal kurtosis or autocorrelation structure cannot be perfectly reproduced by a finite-state Gaussian MSM.

A `VerificationTargetMask` (C′4) selects which fields participate in the
global pass/fail verdict, while still recording per-field
`FieldVerification { checked, passed, diff, tolerance }` results for every
target.  `VerificationTargetMask::for_policy(&CalibrationMappingConfig)`
derives the mask from the active strategy: `MeanPolicy::ZeroCentered`
disables the mean check (the synthetic mean is fixed at zero by
construction), and `CalibrationStrategy::QuickEm` disables the dependence
summaries `abs_acf1` and `sign_change_rate` (which emerge as side-effects
of the EM fit rather than calibration targets).  Masked fields surface in
the report as `checked: false` so JSON consumers can distinguish a skipped
check from a passed one.

The workflow that ties §6.5 → §6.6 → §6.8 together is `run_calibration_workflow` in [src/calibration/report.rs](src/calibration/report.rs), which serialises three artifacts:

| Artifact | Contents |
|---|---|
| `calibrated_scenario.json` | The fitted `ModelParams` (k, π, transition, means, variances) and feature label |
| `calibration_summary.json` | The empirical summary, the chosen expected durations, the verification diff, and the boolean pass/fail |
| `synthetic_vs_empirical_summary.json` | Side-by-side per-field comparison for inspection / plotting |

### 6.8.1 A worked verification example

The verification run produced two complete calibration artifacts, one synthetic and one real.

**Synthetic baseline** ([verification/verify_2026_05_03b/calibration/synthetic/calibration_summary.json](verification/verify_2026_05_03b/calibration/synthetic/calibration_summary.json)): calibrating from the `hard_switch` scenario back to itself, the resampled synthetic data matched its own targets cleanly (`verification_passed: true`, note: *"synthetic summaries are within configured tolerance"*). This is a sanity check on the operator $\mathcal{K}$ and the simulator together — they are mutually consistent.

**Real-data calibration** ([verification/verify_2026_05_03b/calibration/real/calibration_summary.json](verification/verify_2026_05_03b/calibration/real/calibration_summary.json)): calibrating from SPY daily log-returns produced a model with $p_{00} = 0.9018$, $p_{11} = 0.5$, $\sigma^2 = (0.610, 0.941)$, but `verification_passed: false`. The flag is triggered by `acf1 = -0.148` in the empirical SPY series — a *negative* lag-1 autocorrelation that a $K=2$ Gaussian MSM cannot match within $\pm 0.20$, because the MSM's marginal `acf1` is bounded by the squared mean separation, which is essentially zero under `EmpiricalBaseline`.

This is not a code bug — it is the calibration layer correctly reporting that the chosen scenario family cannot replicate every feature of the SPY return process. The methodological response is to record the failure in `verification_notes`, retain the calibrated parameters for downstream use, and treat the resulting synthetic series as a deliberately *simpler* analogue rather than a faithful clone. The synthetic ground truth remains usable for detector benchmarking; the limitation is just that detector performance on this synthetic stream does not predict performance on SPY's serial structure one-for-one.

### 6.8.2 What the chapter has delivered

By the end of this chapter, the project has the four ingredients needed for a defensible synthetic study:

1. **A validated generator** — `simulate(params, T, rng)` with nine pre-sampling invariants on `ModelParams`.
2. **Known changepoints** — `SimulationResult::changepoints` and the matching `ChangePointTruth::from_regime_sequence`, both using the same `S_t ≠ S_{t-1}` rule and 1-based indexing.
3. **An empirical anchor** — `EmpiricalSummary` computed on the same `FeatureStream` the detector consumes, restricted by default to the train partition.
4. **A deterministic, auditable calibration operator** — `calibrate_to_synthetic` mapping $(\bar y, q_{0.05}, q_{0.50}, q_{0.95}, \hat d_{\text{low}}, \hat d_{\text{high}})$ to $(\pi, P, \mu, \sigma^2)$ via three transparent rules: `EmpiricalBaseline` means, `QuantileAnchored` variances, and $p_{jj} = 1 - 1/d_j$ durations — plus a verifier that quantifies how well the result tracks its empirical targets.

The synthetic streams used in every subsequent benchmark in this thesis are produced by this pipeline, and the ground truth against which the detector is scored in Chapter 7 is the unmodified `SimulationResult::states` from §6.2.

---

## 6.9 Quick-EM calibration and the sim-to-real bridge

The mapping $\mathcal{K}$ described in §6.6 is *parametric* — it routes a small
fixed set of summary statistics through configurable policies
(`MeanPolicy`, `VariancePolicy`, `DurationPolicy`).  It is interpretable but
necessarily heuristic: there is no guarantee that the resulting
`ModelParams` $\theta^*$ are a maximum-likelihood description of the real
training partition under the Gaussian MS model.

For the **sim-to-real** experiments in this thesis, where a detector is
trained on synthetic data and tested on real data, we need a stronger
calibration: the synthetic stream must be a *sample from the same generative
model* that EM would extract from the real data, otherwise the train and test
distributions diverge before the detector has even seen the data.

### 6.9.1 The Quick-EM strategy

`CalibrationStrategy::QuickEm { max_iter, tol }` (Rust enum in
`src/calibration/mapping.rs`) replaces the summary-based pipeline of §6.6
with a short Baum–Welch run on the real training partition:

1. Initialise $\theta_0$ via the quantile-based heuristic
   `quantile_init_params` (used purely as an EM starting point).
2. Run `fit_em(observations, theta_0, max_iter, tol)` to convergence (or to
   `max_iter` iterations).
3. Use the resulting $\hat{\theta}_{\text{EM}}$ **directly** as the synthetic
   generator parameters.

In symbolic form:

$$
\mathcal{K}_{\text{QuickEm}}: \{y_t^{\text{real,train}}\}_{t=1}^{T_{\text{train}}}
\;\xrightarrow{\;\text{Baum-Welch}\;}\;
\hat{\theta}_{\text{EM}}
\;=\;\bigl(\hat\pi,\,\hat P,\,\hat\mu,\,\hat\sigma^2\bigr).
$$

The synthetic stream $X^{\text{syn}}(\hat{\theta}_{\text{EM}})$ then samples
exactly from the law that EM would have used to score the real training data,
which is the natural train-time analogue of the test-time likelihood.

### 6.9.2 The one-sided scale-consistency policy

Because the model trained on synthetic features is evaluated on real
features, the two streams must share the same numeric scale.  The codebase
enforces this via a **one-sided z-scoring policy** (`B′1`):

- A `FittedScaler` is fit **only** on the real training partition.
- The synthetic stream is then *transformed* (not re-fit) by that scaler.

The function `scale_consistency_check(&empirical_summary, &synthetic_summary,
tol)` in `src/calibration/verify.rs` computes
$|\hat\sigma_{\text{syn}} - \hat\sigma_{\text{emp}}| / \hat\sigma_{\text{emp}}$
and flags any violation of `DEFAULT_SCALE_TOLERANCE = 0.10`.  The result is
embedded in the `synthetic_training_provenance.json` artifact described
below.

### 6.9.3 The `SimToReal` experiment mode

`ExperimentMode::SimToReal` and the matching `DataConfig::CalibratedSynthetic`
variant compose the pieces above into a single end-to-end backend
(`SimToRealBackend`, `src/experiments/sim_to_real_backend.rs`).  Its five
stages map cleanly onto the canonical runner workflow:

| Stage | Effect under `SimToReal` |
|---|---|
| `resolve_data` | Load real series, run Quick-EM on real-train, simulate synthetic stream of length `horizon`, record scale-consistency check in provenance |
| `build_features` | Fit `FittedScaler` on real-train log-returns, apply it to both real and synthetic streams |
| `train_or_load_model` | Fit a fresh EM **only** on the scaled synthetic stream → `FrozenModel` |
| `run_online` | Run the synthetic-trained `FrozenModel` over the real validation+test partition |
| `evaluate_real` | Route A (proxy events) and Route B (segmentation coherence) on the real test partition, identical to `RealBackend` |

The artifacts written under SimToReal mode (in addition to the standard
real-mode set from §6.7) are:

- `synthetic_training_provenance.json` — empirical calibration profile,
  strategy used, fitted $\hat{\theta}_{\text{EM}}$, scale-consistency check,
  $|X^{\text{syn}}|$ and $|y^{\text{real,train}}|$.
- `sim_to_real_summary.json` — `train_source` / `test_source` labels,
  $\hat{\theta}_{\text{EM}}$, and the real-eval metric tuples (Route A +
  Route B).

### 6.9.4 The train-on-real comparator

To quantify the cost of training on synthetic data instead of on the target
distribution itself, the CLI subcommand
`compare-sim-vs-real --id <simreal_id>` automatically derives a
train-on-real variant of any `simreal_*` experiment — mode flipped to
`Real`, data config flattened from `CalibratedSynthetic { real_asset,
real_frequency, real_dataset_id, … }` to `Real { asset, frequency,
dataset_id, … }`, all other settings (features, model, detector,
evaluation) preserved — and writes the two evaluation metric tuples
side-by-side to `sim_vs_real_comparison.json`.  This artifact is the basis
for the sim-to-real generalisation-gap figure used in the empirical chapter.

### 6.9.5 The canonical sim-to-real registry entry

`simreal_spy_daily_hard_switch` (in `src/experiments/registry.rs`) is the
flagship sim-to-real experiment used in this thesis:

- SPY daily, full sample from 2018-01-01;
- Quick-EM calibration with `max_iter = 100`, `tol = 1e-6`, $K = 2$,
  horizon = 2000;
- HardSwitch detector with threshold 0.55, persistence 2, cooldown 5;
- ZScore feature scaling, log-return family;
- Real evaluation via Route A (with `data/proxy_events/spy.json`) and
  Route B.

Both `cargo run -- run-real --id simreal_spy_daily_hard_switch` and
`cargo run -- compare-sim-vs-real --id simreal_spy_daily_hard_switch`
operate against this entry and emit the artifacts cited above.
