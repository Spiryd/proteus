# Proteus

Changepoint detection for commodity and equity time series using a Gaussian Markov Switching Model (MSM). Built as a Master's thesis project.

The system fits a K-regime MSM offline via EM (Baum-Welch), then runs a causal online filter to score and alarm on regime shifts in real time. Three detector variants are provided: Hard Switch, Posterior Transition, and Surprise.

---

## Quick Start

### Prerequisites

- Rust (stable) — install via [rustup](https://rustup.rs)

For **real-data** experiments (optional, deferred): a free Alpha Vantage API key — [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key). Synthetic experiments work with no API key or config file.

### Running (synthetic — no config needed)

```
cargo run              # interactive menu
cargo run -- e2e       # run all 3 registered synthetic experiments end-to-end
cargo run -r -- e2e    # release build (recommended for EM training runs)
cargo run -- help      # direct CLI help
```

### Configuration (real data only)

Copy the example config and fill in your API key:

```
cp config.example.toml config.toml
```

Edit `config.toml`:

```toml
[alphavantage]
api_key = "your_api_key_here"
rate_limit_per_minute = 75   # default, can be omitted

[cache]
path = "data/commodities.duckdb"   # default, can be omitted

[ingest]
series = [
    { commodity = "spy",         interval = "15min"   },
    { commodity = "qqq",         interval = "15min"   },
    { commodity = "wti",         interval = "daily"   },
    { commodity = "brent",       interval = "daily"   },
    { commodity = "natural_gas", interval = "daily"   },
    { commodity = "gold",        interval = "daily"   },
    { commodity = "silver",      interval = "daily"   },
]
```

> `config.toml` contains your API key — never commit it.

---

## Usage

### Interactive Mode

`cargo run` launches a 9-category guided menu. Navigate with arrow keys; press **Esc** to go back at any prompt.

```
Main menu:
  Data          — ingest, inspect, and refresh market data
  Features      — feature families and observation pipeline
  Calibration   — synthetic-to-real scenario calibration
  Models        — Gaussian MSM fitting and inspection
  Detection     — detector variants and alarm configuration
  Evaluation    — synthetic and real-data evaluation
  Experiments   — run single or batch experiments
  Reporting     — plots, tables, and artifact export
  Inspect Runs  — browse and view saved run artifacts
  Exit
```

See [docs/interactive_cli.md](docs/interactive_cli.md) for the full menu reference.

### Direct CLI Mode

Pass a subcommand as the first argument to skip the interactive menu (useful for scripting):

```
cargo run -- e2e                                          # run all registered experiments end-to-end
cargo run -- run-experiment  --config experiment_config.json
cargo run -- run-batch       --config a.json --config b.json
cargo run -- param-search    --id <experiment_id>         # grid search over detector params
cargo run -- inspect         --dir ./runs/synthetic/my_run/run_001
cargo run -- status          [--config path/to/config.toml]
cargo run -- help
```

Experiment configs are JSON files. A template is printed by the **Experiments > Show Config Template** menu item.

---

## Model

### Gaussian Markov Switching Model

Hidden state S_t in {1,...,K} with first-order Markov dynamics:

    P(S_t = j | S_{t-1} = i) = A_{ij}

Observations are Gaussian given the regime:

    y_t | S_t = j  ~  N(mu_j, sigma_j^2)

Parameters theta = (pi, A, mu_{1:K}, sigma^2_{1:K}) are fitted offline via the EM algorithm (Baum-Welch), then frozen for online use.

See [docs/gaussian_msm_simulator.md](docs/gaussian_msm_simulator.md) and [docs/em_estimation.md](docs/em_estimation.md).

### Inference Pipeline

| Phase | Component | Doc |
|-------|-----------|-----|
| Emission density | N(y_t; mu_j, sigma_j^2) | [emission_model.md](docs/emission_model.md) |
| Forward filter | alpha_{t|t}(j) = Pr(S_t=j | y_{1:t}) | [forward_filter.md](docs/forward_filter.md) |
| Log-likelihood | log p(y_{1:T}) from filter normalisation constants | [log_likelihood.md](docs/log_likelihood.md) |
| Backward smoother | gamma_t(j) = Pr(S_t=j | y_{1:T}) | [backward_smoother.md](docs/backward_smoother.md) |
| Pairwise posteriors | xi_t(i,j) = Pr(S_{t-1}=i, S_t=j | y_{1:T}) | [pairwise_posteriors.md](docs/pairwise_posteriors.md) |
| EM estimation | Baum-Welch until convergence | [em_estimation.md](docs/em_estimation.md) |
| Diagnostics | Validity checks on fitted parameters | [diagnostics.md](docs/diagnostics.md) |
| Online inference | Causal streaming filter, no future data | [online_inference.md](docs/online_inference.md) |

### Detector Variants

All detectors consume one-step causal filter output and apply a score + alarm policy (persistence + cooldown). See [docs/changepoint_detectors.md](docs/changepoint_detectors.md).

| Detector | Score | Alarm trigger |
|----------|-------|---------------|
| **HardSwitch** | Dominant regime label argmax_j alpha_{t|t}(j) | Label changes vs previous step |
| **PosteriorTransition** | Off-diagonal posterior transition mass | Score exceeds threshold |
| **Surprise** | -log c_t (negative log predictive density) | Score exceeds threshold |

The fixed-parameter policy (offline-fit, online-freeze) is described in [docs/fixed_parameter_policy.md](docs/fixed_parameter_policy.md).

---

## Observation Pipeline

Raw prices are transformed into the observation sequence y_t before fitting or streaming:

| Family | Formula |
|--------|---------|
| `LogReturn` | log(P_t / P_{t-1}) |
| `AbsReturn` | absolute value of log return |
| `SquaredReturn` | (log return)^2 |
| `RollingVol` | Rolling std of log returns over window w |
| `StandardizedReturn` | log return / rolling std |

Scaling options: `None`, `ZScore`, `RobustZScore`. All transforms are strictly causal.

See [docs/observation_design.md](docs/observation_design.md) for the full pipeline and session-aware variants.

---

## Calibration

Synthetic scenarios are calibrated against real empirical data so that benchmark experiments are grounded. The workflow maps empirical statistics (mean, variance, jump contamination) to MSM parameters and verifies the discrepancy.

See [docs/synthetic_to_real_calibration.md](docs/synthetic_to_real_calibration.md).

---

## Evaluation

### Synthetic Benchmark

Evaluated on simulated data with known changepoints using an event-window matching protocol. Metrics: coverage, precision-like score, mean detection delay.

See [docs/benchmark_protocol.md](docs/benchmark_protocol.md).

### Real-Data Evaluation

No ground truth is available, so two routes are used:

- **Route A — Proxy Event Alignment:** alarm timing vs. known market events (earnings, macro announcements).
- **Route B — Segmentation Self-Consistency:** within-segment homogeneity and between-segment contrast.

See [docs/real_data_evaluation.md](docs/real_data_evaluation.md).

---

## Experiments

Experiments are fully described by a JSON `ExperimentConfig` and run through `ExperimentRunner`. Each run produces a deterministic run ID (from config hash + seed), a structured artifact directory, and a serialised `ExperimentResult`.

```
runs/
  synthetic/
    <run_label>/
      <run_id>/
        config/             — experiment_config.json snapshot
        metadata/           — run_metadata.json, result.json
        results/            — evaluation_summary.json
        model_params.json   — fitted FittedParamsSummary (used by LoadFrozen)
        score_trace.csv     — per-step detector score (if save_traces + write_csv)
        alarms.csv          — alarm timestamps (if save_traces + write_csv)
        metrics.md          — metrics table (Markdown)
        metrics.csv         — metrics table (CSV)
        metrics.tex         — metrics table (LaTeX)
        plots/              — PNG plots (if enabled)
```

The three registered synthetic experiments (`hard_switch`, `posterior_transition`, `surprise`) can all be run at once:

```
cargo run -- e2e
```

**Sample output:**

```
[1/3] hard_switch
  TrainOrLoadModel: LL=-1948.67  iter=124  converged=true
  RunOnline:        n_steps=1999  n_alarms=38
  Evaluate:         precision=0.6579  recall=0.2066  n_events=121
  Metrics:  prec=0.6579  recall=0.2066  n_events=121  n_alarms=38
  Miss rate: 0.7934  FAR=0.006500
  Delay:    mean=10.8  median=11.0
  Model:    K=2  LL=-1948.6659  iter=124  converged=true
  Regime 0: mu=-0.038  var=0.655  pi=0.000
  Regime 1: mu=0.140   var=2.243  pi=1.000
  Trans[0]: [0.9466, 0.0534]
  Trans[1]: [0.1971, 0.8029]
...
  Completed: 3  Failed: 0
```

See [docs/experiment_runner.md](docs/experiment_runner.md).

---

## Reporting

The reporting layer generates tables (and, in future work, plots) from run artifacts. See [docs/reporting_and_export.md](docs/reporting_and_export.md).

| Output | Description | Status |
|--------|-------------|--------|
| `metrics.md / .csv / .tex` | Coverage / precision / recall / delay per run | ✓ Generated |
| `comparison_table.csv` | Side-by-side comparison across runs | ✓ Generated (E2E aggregate) |
| `model_params.json` | Fitted model summary (reloadable via LoadFrozen) | ✓ Generated |
| `alarms.csv` | Alarm step indices | ✓ Generated (save_traces=true) |
| `score_trace.csv` | Per-step detector score | ✓ Generated (save_traces=true) |
| `signal_alarms.png` | Observation series with alarm markers | Pending |
| `detector_scores.png` | Score trace with threshold line | Pending |
| `regime_posteriors.png` | Filtered posterior traces per regime | Pending |
| `delay_distribution.png` | Detection delay histogram (synthetic) | Pending |

---

## Data

### Sources

Data is sourced from the [Alpha Vantage API](https://www.alphavantage.co). Supported series:

**Commodities** (daily / weekly / monthly / quarterly / annual): WTI, Brent, Natural Gas, Copper, Aluminum, Wheat, Corn, Cotton, Sugar, Coffee, Gold, Silver, All Commodities Index.

**Equities** (SPY, QQQ): daily, weekly, monthly, and intraday (`1min`, `5min`, `15min`, `30min`, `60min`).

The HTTP client is rate-limited (default 75 req/min, token-bucket). See [docs/alphavantage_client.md](docs/alphavantage_client.md).

### Caching

All fetched data is persisted in a local [DuckDB](https://duckdb.org) database (default: `data/commodities.duckdb`, created automatically). Each series is stored as `(symbol, interval, date, value)` rows. Re-ingest does a full replace.

See [docs/duckdb_cache.md](docs/duckdb_cache.md) for schema details.

---

## Architecture

```
src/
  main.rs                    — dual-mode dispatch (interactive / direct CLI)
  config.rs                  — TOML config structs
  alphavantage/
    client.rs                — async HTTP client with rate limiting
    commodity.rs             — endpoint/interval types + deserialisation
    rate_limiter.rs          — token-bucket rate limiter
  cache/
    mod.rs                   — DuckDB persistence layer
  data_service/
    mod.rs                   — cache-first orchestration, bulk ingest
  cli/
    mod.rs                   — interactive menu + direct subcommand dispatch
  features/
    mod.rs                   — feature families, scaling, session-aware pipeline
  model/
    params.rs                — ModelParams (K, pi, A, mu, sigma^2)
    simulate.rs              — Gaussian MSM generative sampler
    filter.rs                — Hamilton forward filter
    smoother.rs              — backward smoother (RTS)
    pairwise.rs              — pairwise posterior pass
    em.rs                    — Baum-Welch EM estimator
    diagnostics.rs           — fitted-model validity checks
  online/
    mod.rs                   — causal streaming filter state machine
  detector/
    hard_switch.rs           — Hard Switch detector
    posterior_transition.rs  — Posterior Transition detector
    surprise.rs              — Surprise (-log predictive) detector
    frozen.rs                — FrozenModel + StreamingSession
  calibration/
    mod.rs                   — empirical summary + synthetic mapping
    report.rs                — CalibrationReport workflow
  benchmark/
    mod.rs                   — event-window evaluation protocol
  real_eval/
    route_a.rs               — proxy event alignment
    route_b.rs               — segmentation self-consistency
  experiments/
    config.rs                — ExperimentConfig (fully serialisable)
    runner.rs                — ExperimentRunner<B> + ExperimentBackend trait
    synthetic_backend.rs     — SyntheticBackend: real EM + detection + evaluation
    batch.rs                 — BatchConfig + run_batch
    result.rs                — ExperimentResult, RunStatus, EvaluationSummary, FittedParamsSummary
    registry.rs              — registered experiment definitions
    search.rs                — param-search grid (detector threshold / persistence)
    artifact.rs              — run directory layout + snapshot helpers
  reporting/
    artifact.rs              — ArtifactRootConfig, RunArtifactLayout
    export/                  — JSON / CSV export (schema, json, csv)
    plot/                    — plotters-based renderers (5 plot types)
    table/                   — MetricsTableBuilder, ComparisonTableBuilder
    report.rs                — RunReporter, AggregateReporter
```

---

## Documentation Index

| Doc | Topic |
|-----|-------|
| [alphavantage_client.md](docs/alphavantage_client.md) | Alpha Vantage HTTP client and rate limiting |
| [duckdb_cache.md](docs/duckdb_cache.md) | DuckDB schema and cache API |
| [data_service.md](docs/data_service.md) | DataService orchestration layer |
| [data_pipeline.md](docs/data_pipeline.md) | Real financial data pipeline |
| [interactive_cli.md](docs/interactive_cli.md) | Interactive CLI full reference |
| [observation_design.md](docs/observation_design.md) | Feature families and observation pipeline |
| [gaussian_msm_simulator.md](docs/gaussian_msm_simulator.md) | Generative MSM simulator |
| [emission_model.md](docs/emission_model.md) | Gaussian emission density |
| [forward_filter.md](docs/forward_filter.md) | Hamilton forward filter |
| [filter_validation.md](docs/filter_validation.md) | Filter validation on simulated data |
| [log_likelihood.md](docs/log_likelihood.md) | Observed-data log-likelihood |
| [backward_smoother.md](docs/backward_smoother.md) | RTS backward smoother |
| [pairwise_posteriors.md](docs/pairwise_posteriors.md) | Pairwise posterior transition probabilities |
| [em_estimation.md](docs/em_estimation.md) | Baum-Welch EM estimator |
| [diagnostics.md](docs/diagnostics.md) | Fitted-model diagnostics and trust checks |
| [online_inference.md](docs/online_inference.md) | Online (streaming) causal inference |
| [changepoint_detectors.md](docs/changepoint_detectors.md) | Detector variants (HardSwitch, PosteriorTransition, Surprise) |
| [fixed_parameter_policy.md](docs/fixed_parameter_policy.md) | Offline-trained, online-frozen parameter policy |
| [benchmark_protocol.md](docs/benchmark_protocol.md) | Synthetic benchmark and event-window evaluation |
| [synthetic_to_real_calibration.md](docs/synthetic_to_real_calibration.md) | Synthetic-to-real calibration workflow |
| [real_data_evaluation.md](docs/real_data_evaluation.md) | Real-data evaluation (Route A + B) |
| [experiment_runner.md](docs/experiment_runner.md) | Experiment runner and reproducibility layer |
| [reporting_and_export.md](docs/reporting_and_export.md) | Reporting, plots, tables, and artifact export |

---

## Tests

```
cargo test
```

323 tests covering all core components: filter/smoother correctness, EM convergence, detector alarm logic, calibration mapping, benchmark matching, experiment runner orchestration, and artifact serialisation.
