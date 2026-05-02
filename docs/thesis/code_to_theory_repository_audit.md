# Code-to-Theory Repository Audit

**Proteus — Markov-Switching Regime-Change Detector**  
Rust 1.94.1 · Edition 2024 · 72 source files · 333 tests (all pass)  
Audit date: 2025-05

---

## Table of Contents

1. [Executive Summary Table](#1-executive-summary-table)
2. [Repository Map](#2-repository-map)
3. [End-to-End Flow Map](#3-end-to-end-flow-map)
4. [T1 — Data Pipeline](#4-t1--data-pipeline)
5. [T2 — Feature / Observation Design](#5-t2--feature--observation-design)
6. [T3 — Model Parameterisation](#6-t3--model-parameterisation)
7. [T4 — Gaussian Emission Model](#7-t4--gaussian-emission-model)
8. [T5 — Synthetic Generator](#8-t5--synthetic-generator)
9. [T6 — Forward Filter and Log-Likelihood](#9-t6--forward-filter-and-log-likelihood)
10. [T7 — Backward Smoother](#10-t7--backward-smoother)
11. [T8 — Pairwise Transition Posteriors](#11-t8--pairwise-transition-posteriors)
12. [T9 — EM Estimation](#12-t9--em-estimation)
13. [T10 — Diagnostics](#13-t10--diagnostics)
14. [T11 — Offline-Trained, Online-Filtered Runtime](#14-t11--offline-trained-online-filtered-runtime)
15. [T12 — Detector Family](#15-t12--detector-family)
16. [T13 — Synthetic Benchmark Evaluation](#16-t13--synthetic-benchmark-evaluation)
17. [T14 — Real-Data Evaluation](#17-t14--real-data-evaluation)
18. [T15 — Synthetic-to-Real Calibration](#18-t15--synthetic-to-real-calibration)
19. [T16 — Experiment Runner](#19-t16--experiment-runner)
20. [T17 — Reporting and Plotting](#20-t17--reporting-and-plotting)
21. [T18 — CLI / Interactive Application Layer](#21-t18--cli--interactive-application-layer)
22. [Dead Code and Duplicate-Implementation Risks](#22-dead-code-and-duplicate-implementation-risks)
23. [Documentation Gaps](#23-documentation-gaps)
24. [Test Coverage Gaps](#24-test-coverage-gaps)
25. [Thesis Chapter Mapping](#25-thesis-chapter-mapping)
26. [Final Action List](#26-final-action-list)

---

## 1. Executive Summary Table

| ID  | Theoretical Area                     | Status                   | Notes |
|-----|--------------------------------------|--------------------------|-------|
| T1  | Data pipeline                        | IMPLEMENTED_AND_USED     | DuckDB cache; real + synthetic paths wired |
| T2  | Feature / observation design         | IMPLEMENTED_AND_USED     | 5 families, session-aware, scaler |
| T3  | Model parameterisation               | IMPLEMENTED_AND_USED     | k-regime params fully wired |
| T4  | Gaussian emission model              | IMPLEMENTED_AND_USED     | log-density, density-vec, tested |
| T5  | Synthetic generator                  | IMPLEMENTED_AND_USED     | Markov chain + Gaussian noise |
| T6  | Forward filter and log-likelihood    | IMPLEMENTED_AND_USED     | Hamilton filter, numerically stable |
| T7  | Backward smoother                    | IMPLEMENTED_AND_USED     | Kim-style backward pass |
| T8  | Pairwise transition posteriors       | IMPLEMENTED_AND_USED     | ξ_{t}(i,j) for M-step |
| T9  | EM estimation                        | IMPLEMENTED_AND_USED     | Baum-Welch, multi-start, convergence check |
| T10 | Diagnostics                          | IMPLEMENTED_AND_USED     | Post-fit trust layer, wired to runner |
| T11 | Offline-trained online-filtered runtime | IMPLEMENTED_AND_USED  | FrozenModel + OnlineFilterState |
| T12 | Detector family                      | IMPLEMENTED_AND_USED     | HardSwitch + PosteriorLeave + PosteriorTV + Surprise all registered; `posterior_transition_tv` experiment added |
| T13 | Synthetic benchmark evaluation       | IMPLEMENTED_AND_USED     | EventMatcher + MetricSuite |
| T14 | Real-data evaluation                 | IMPLEMENTED_AND_USED     | Route A + B both functional; proxy event files `data/proxy_events/{spy,wti,gold}.json` committed; Route A matched 1 event, metrics 0.077/0.091 |
| T15 | Synthetic-to-real calibration        | IMPLEMENTED_AND_USED     | `JumpParams`/`simulate_with_jump()` in `model::simulate`; calibration workflow passes jump config; `hard_switch_shock` experiment exercises the full path |
| T16 | Experiment runner                    | IMPLEMENTED_AND_USED     | ExperimentRunner + 8 registered experiments |
| T17 | Reporting and plotting               | IMPLEMENTED_AND_USED     | JSON/CSV/table export fully wired; `AggregateReporter` exposed via `compare-runs` CLI subcommand; `#[cfg(not(test))]` gates removed — render functions always compiled |
| T18 | CLI / interactive application layer  | IMPLEMENTED_AND_USED     | Interactive menus + direct subcommands |

**Status key:**
- `IMPLEMENTED_AND_USED` — code exists, reachable from CLI, tested
- `PARTIAL` — code exists and mostly used, but a specific sub-feature is missing or unreachable
- `IMPLEMENTED_NOT_USED` — code exists but no registered experiment or CLI path invokes it
- `MISSING` — no implementation exists
- `UNCLEAR` — reachability or correctness uncertain

---

## 2. Repository Map

```
proteus/
├── Cargo.toml                       # duckdb 1.10502 (bundled), tokio 1.52, plotters 0.3, …
├── config.toml                      # runtime config (API key, cache path, ingest schedule)
├── data/
│   └── commodities.duckdb           # commodity + SPY/QQQ price cache
├── docs/
│   ├── alphavantage_client.md
│   ├── data_service.md
│   ├── synthetic_to_real_calibration.md
│   └── thesis/
│       └── code_to_theory_repository_audit.md  ← this file
└── src/
    ├── main.rs                      # entry point; dispatch to CLI
    ├── config.rs                    # Config, CacheConfig, IngestConfig
    ├── alphavantage/
    │   ├── client.rs                # async HTTP client
    │   ├── commodity.rs             # endpoint + response types
    │   └── rate_limiter.rs          # token-bucket rate limiter
    ├── cache/
    │   └── mod.rs                   # DuckDB read/write, SeriesStatus
    ├── data_service/
    │   └── mod.rs                   # DataService: refresh, load_cached, ingest_all, status
    ├── data/
    │   ├── mod.rs                   # Observation, CleanSeries
    │   ├── meta.rs                  # DataMode, DataSource, PriceField, SessionConvention, DatasetMeta
    │   ├── session.rs               # SessionBoundary, SessionAwareSeries, RTH filter
    │   ├── split.rs                 # TimePartition, SplitConfig, PartitionedSeries
    │   └── validation.rs            # Gap, ValidationReport, detect_intraday_gaps
    ├── features/
    │   ├── mod.rs
    │   ├── family.rs                # FeatureFamily enum, warmup_bars, label
    │   ├── transform.rs             # log_return, abs_return, squared_return (session-aware)
    │   ├── rolling.rs               # RollingStats ring buffer, rolling_vol, standardized_returns
    │   ├── scaler.rs                # ScalingPolicy, FittedScaler, fit/transform/inverse
    │   └── stream.rs                # FeatureConfig, FeatureStream::build
    ├── model/
    │   ├── mod.rs
    │   ├── params.rs                # ModelParams, validate, transition_row
    │   ├── emission.rs              # Emission, log_density, density_vec
    │   ├── filter.rs                # FilterResult, filter, predict, bayes_update, log_sum_exp
    │   ├── smoother.rs              # SmootherResult, smooth
    │   ├── pairwise.rs              # PairwiseResult, xi, expected_transitions
    │   ├── em.rs                    # EStepResult, EmConfig, EmResult, fit_em, e_step, m_step
    │   ├── diagnostics.rs           # DiagnosticWarning, FittedModelDiagnostics, diagnose
    │   ├── simulate.rs              # SimulationResult, simulate
    │   ├── likelihood.rs            # log_likelihood, log_likelihood_contributions
    │   └── validation.rs            # Phase 6 scenario-family integration tests
    ├── online/
    │   └── mod.rs                   # OnlineFilterState, OnlineStepResult, step, step_batch
    ├── detector/
    │   ├── mod.rs                   # DetectorKind, DetectorInput/Output, Detector trait
    │   ├── hard_switch.rs           # HardSwitchDetector
    │   ├── posterior_transition.rs  # PosteriorTransitionDetector (Leave + TV variants)
    │   ├── frozen.rs                # FrozenModel, StreamingSession<D>
    │   └── surprise.rs              # SurpriseDetector
    ├── benchmark/
    │   ├── mod.rs
    │   ├── truth.rs                 # ChangePointTruth, from_regime_sequence
    │   ├── matching.rs              # EventMatcher, MatchConfig, greedy matching
    │   ├── metrics.rs               # MetricSuite, precision/recall/delay
    │   └── result.rs                # AggregateResult, RunResult, TimingSummary
    ├── real_eval/
    │   ├── mod.rs
    │   ├── route_a.rs               # ProxyEvent, EventAlignment, evaluate_proxy_events
    │   ├── route_b.rs               # DetectedSegment, AdjacentSegmentContrast, evaluate_segmentation
    │   └── report.rs                # RealEvalResult, evaluate_real_data
    ├── calibration/
    │   ├── mod.rs
    │   ├── mapping.rs               # CalibrationMappingConfig, calibrate_to_synthetic
    │   ├── summary.rs               # EmpiricalCalibrationProfile, summarize_feature_stream
    │   ├── verify.rs                # CalibrationVerification, verify_calibration
    │   └── report.rs                # CalibrationReport, run_calibration_workflow
    ├── experiments/
    │   ├── mod.rs
    │   ├── config.rs                # ExperimentConfig and all sub-configs
    │   ├── registry.rs              # 6 registered experiments
    │   ├── runner.rs                # ExperimentBackend trait, ExperimentRunner, DryRunBackend
    │   ├── shared.rs                # train_or_load_model_shared, run_online_shared
    │   ├── synthetic_backend.rs     # SyntheticBackend
    │   ├── real_backend.rs          # RealBackend (DuckDB)
    │   ├── search.rs                # ParamGrid, ModelGrid, grid_search, optimize_full
    │   ├── batch.rs                 # BatchConfig, BatchResult, run_batch
    │   ├── artifact.rs              # prepare_run_dir, snapshot_config, snapshot_result
    │   └── result.rs                # ExperimentResult, EvaluationSummary, RunMetadata
    ├── reporting/
    │   ├── mod.rs
    │   ├── report.rs                # RunReporter, AggregateReporter
    │   ├── artifact.rs              # ArtifactRootConfig, RunArtifactLayout
    │   ├── export/
    │   │   ├── schema.rs            # AlarmRecord, SegmentRecord, etc.
    │   │   ├── json.rs              # export_config, export_result, export_evaluation_summary
    │   │   └── csv.rs               # CSV export functions
    │   ├── table/
    │   │   ├── metrics.rs           # MetricsTableBuilder, output: md/CSV/LaTeX
    │   │   ├── comparison.rs        # Comparison table builder
    │   │   └── segment_summary.rs   # Segment summary table
    │   └── plot/
    │       ├── mod.rs               # re-exports (gated #[cfg(not(test))])
    │       ├── signal_alarms.rs     # SignalWithAlarmsPlotInput, render
    │       ├── detector_scores.rs   # DetectorScoresPlotInput, render
    │       ├── regime_posteriors.rs # RegimePosteriorPlotInput, render
    │       ├── delay_distribution.rs
    │       └── segmentation.rs
    └── cli/
        └── mod.rs                   # interactive menus + direct subcommand dispatch
```

---

## 3. End-to-End Flow Map

```
main()
  └─ parse args
       ├─ direct command ──► cli::run_direct(args)
       └─ interactive    ──► cli::run(cfg)

cli::run_direct / cli::run
  └─ Config::from_file("config.toml")
  └─ selects experiment from registry (ExperimentConfig)
  └─ ExperimentRunner<SyntheticBackend | RealBackend>::run(cfg)

ExperimentRunner::run(cfg)
  ├─ [Stage 1] backend.resolve_data(cfg)
  │    ├─ Synthetic: build_synthetic_params → simulate(params, horizon, rng)
  │    └─ Real:      CommodityCache::open → load → CleanSeries → PartitionedSeries
  │
  ├─ [Stage 2] backend.build_features(cfg, data)
  │    └─ FeatureStream::build(prices, meta, FeatureConfig)
  │         ├─ apply transform (LogReturn | AbsReturn | SquaredReturn | RollingVol | Standardized)
  │         └─ FittedScaler::fit(train_obs, policy) → transform all
  │
  ├─ [Stage 3] backend.train_or_load_model(cfg, features)   [shared.rs]
  │    └─ fit_em(train_obs, EmConfig { k, tol, max_iter, var_floor }, n_starts)
  │         ├─ init_params_from_obs(obs, k, rng)
  │         ├─ loop: e_step → m_step until |ΔLL| < tol or max_iter
  │         └─ FrozenModel::from_em_result(best_result)
  │
  ├─ [Stage 4] backend.run_online(cfg, model, features)    [shared.rs]
  │    └─ for each obs y_t:
  │         ├─ OnlineFilterState::step(y_t, params)
  │         │    ├─ filter.predict(): α_{t|t-1}(j) = Σ_i α_{t-1|t-1}(i) · p_{ij}
  │         │    └─ filter.bayes_update(): α_{t|t}(j) ∝ α_{t|t-1}(j) · f(y_t | μ_j, σ²_j)
  │         └─ Detector::update(DetectorInput)
  │              ├─ HardSwitch:  alarm if dominant regime changes AND confidence ≥ θ
  │              ├─ PosteriorTransition (Leave): alarm if 1 − α_{t|t}(r_{t-1}) ≥ θ
  │              └─ Surprise:    alarm if −log c_t ≥ θ (EMA-adjusted baseline)
  │
  ├─ [Stage 5] backend.evaluate_*(cfg, online)
  │    ├─ Synthetic: EventMatcher::match_events() → MetricSuite (precision/recall/delay)
  │    └─ Real:      evaluate_real_data() → Route A (proxy events) + Route B (segmentation)
  │
  └─ [Stage 6] Export artifacts
       ├─ snapshot_config(), snapshot_result()
       ├─ JSON: model_params, fit_summary, diagnostics, feature_summary, detector_config
       ├─ CSV:  score_trace, alarms, regime_posteriors, changepoints, real_eval_summary
       └─ Plots (#[cfg(not(test))]):
            signal_with_alarms.png, detector_scores.png, regime_posteriors.png
```

---

## 4. T1 — Data Pipeline

### Theoretical object
External price series ingestion, caching, quality validation, session filtering, and train/val/test partitioning.

### Expected equations / definitions
- Log return: $r_t = \ln(p_t / p_{t-1})$
- Train/val/test split by calendar date or fixed-count boundary
- Gap detection: intraday expected bar cadence vs actual timestamps

### Code locations
- `src/alphavantage/client.rs` — async HTTP ingestion from Alpha Vantage API
- `src/alphavantage/commodity.rs` — `CommodityEndpoint` (15 variants), response types
- `src/alphavantage/rate_limiter.rs` — token-bucket rate limiter
- `src/cache/mod.rs` — DuckDB read/write via `CommodityCache`
- `src/data_service/mod.rs` — `DataService { refresh, load_cached, ingest_all, status }`
- `src/data/mod.rs` — `Observation { timestamp, value }`, `CleanSeries`
- `src/data/meta.rs` — `DataMode`, `DataSource`, `PriceField`, `SessionConvention`, `DatasetMeta`
- `src/data/session.rs` — `SessionAwareSeries`, `filter_rth()`, `label_sessions()`
- `src/data/split.rs` — `SplitConfig`, `PartitionedSeries`
- `src/data/validation.rs` — `ValidationReport`, `detect_intraday_gaps()`

### Key types / functions
| Symbol | File | Role |
|--------|------|------|
| `AlphaVantageClient::from_config()` | alphavantage/client.rs | Only constructor; builds reqwest client |
| `CommodityCache::open()` | cache/mod.rs | Opens DuckDB file at `data/commodities.duckdb` |
| `CommodityCache::store()` / `load()` | cache/mod.rs | Upsert and fetch price rows |
| `DataService::refresh()` | data_service/mod.rs | Pull fresh data; repopulate cache |
| `DataService::load_cached()` | data_service/mod.rs | Serve from DuckDB without network call |
| `CleanSeries::from_response()` | data/mod.rs | Parse API response → `CleanSeries` |
| `PartitionedSeries::from_series()` | data/split.rs | Apply `SplitConfig` date boundaries |
| `detect_intraday_gaps()` | data/validation.rs | Flag gaps in 5m/15m bar sequences |

### How it is used
`RealBackend::resolve_data()` opens the DuckDB cache, calls `CommodityCache::load()`, wraps the result in `CleanSeries`, builds `DatasetMeta`, calls `PartitionedSeries::from_series()`, and serialises the split summary to JSON for the artifact store. The Alpha Vantage client is invoked separately by `DataService` when the CLI `calibrate` or data-submenu commands are called.

### E2E reachability
- CLI `run-real` → `RealBackend::resolve_data()` → DuckDB ✓  
- CLI Data submenu → `DataService::refresh()` / `ingest_all()` → `AlphaVantageClient` ✓  
- CLI `status` → `DataService::status()` → `CommodityCache::status()` ✓

### Artifacts / outputs
- `data/commodities.duckdb` — persistent price store
- `split_summary.json` — train/val/test counts, date ranges
- `data_quality.json` — gap counts, duplicate-drop count

### Tests
- `real_backend.rs`: `resolve_data_train_n_is_seventy_percent`, `resolve_data_date_filter_trims_correctly`, `full_pipeline_with_fixture_data`
- `data/validation.rs`: gap-detection unit tests
- `data/split.rs`: partition boundary tests

### Status
**IMPLEMENTED_AND_USED**

### Gaps / risks
- No dedicated unit tests for `DataService::refresh()` or the Alpha Vantage client against a real endpoint.
- `detect_intraday_gaps()` is tested but not integrated into the `run_online` path — gaps in live real data can silently corrupt the filter.
- `DataService::get()` was deleted; no replacement fetches a named single series directly; callers must use `load_cached()` and filter.

### Thesis note
This layer is infrastructural scaffolding for the real-data experiments in Chapter 5. The theory section need not describe it in mathematical detail, but the thesis should clarify that the data pipeline produces an IID-approximately-stationary feature sequence that is the input to the MSM.

---

## 5. T2 — Feature / Observation Design

### Theoretical object
Transformation of raw price series into a scalar observation sequence $y_1, \ldots, y_T$ assumed to be generated by the hidden Markov model. Optionally standardised and session-aware.

### Expected equations / definitions
$$y_t = \ln(p_t / p_{t-1}) \quad \text{(log-return)}$$
$$y_t = |r_t| \quad \text{(absolute return)}$$
$$y_t = r_t^2 \quad \text{(squared return)}$$
$$y_t = \text{std}(r_{t-w}, \ldots, r_{t-1}) \quad \text{(rolling volatility, window } w \text{)}$$
$$y_t = \frac{r_t - \mu_w}{\sigma_w} \quad \text{(standardised return, window } w \text{)}$$

Z-score normalisation: $\tilde{y}_t = (y_t - \bar{y}_{\text{train}}) / s_{\text{train}}$

### Code locations
- `src/features/family.rs` — `FeatureFamily` enum, `warmup_bars()`, `label()`
- `src/features/transform.rs` — `log_return()`, `abs_return()`, `squared_return()`, session-aware variants
- `src/features/rolling.rs` — `RollingStats` (ring buffer), `rolling_vol()`, `standardized_returns()`
- `src/features/scaler.rs` — `ScalingPolicy { None | ZScore | RobustZScore }`, `FittedScaler`
- `src/features/stream.rs` — `FeatureConfig`, `FeatureStream::build()`

### Key types / functions
| Symbol | File | Role |
|--------|------|------|
| `FeatureFamily` | family.rs | Dispatch enum for transform choice |
| `FeatureStream::build()` | stream.rs | Complete pipeline: transform → scaler fit → scaler apply |
| `FittedScaler::fit()` | scaler.rs | Fit location/scale from train partition only |
| `RollingStats` | rolling.rs | O(1) ring-buffer for rolling stats |
| `warmup_bars()` | family.rs | Returns bars to skip before valid output |

### How it is used
Both `SyntheticBackend` (passthrough — simulated data already in feature space) and `RealBackend` call `FeatureStream::build()`. The scaler is fit exclusively on `train_n` observations, then applied to the full series to prevent look-ahead bias.

### E2E reachability
Every registered experiment specifies `features.family` and `features.scaling` in its `ExperimentConfig`; all six reach `FeatureStream::build()` at runtime.

### Artifacts / outputs
`feature_summary.json` — feature label, n_observations, train_n, obs_mean, obs_variance, obs_std, obs_min, obs_max, warmup_trimmed.

### Tests
- `features/rolling.rs`: ring-buffer correctness, session-reset, edge cases
- `features/scaler.rs`: fit/transform/inverse round-trip, policy variants
- `features/stream.rs`: `FeatureStream::build()` integration tests

### Status
**IMPLEMENTED_AND_USED**

### Gaps / risks
- `RollingVol` and `StandardizedReturn` with `session_reset = true` are implemented and tested but no registered experiment uses `session_aware = true` — intraday session-reset logic is exercised only in unit tests.
- `RobustZScore` scaler policy is implemented but not selected by any registered experiment.
- Warmup bars are trimmed from the observation count but the trim count is not validated against `train_n`; if `warmup_bars() >= train_n` the EM receives zero training observations (guarded by an empty-observations stub in `train_or_load_model_shared`).

### Thesis note
Directly maps to the "observation design" subsection of the MSM chapter. The choice of log-return with Z-score normalisation approximates the Gaussian emission assumption and is standard in regime-switching literature.

---

## 6. T3 — Model Parameterisation

### Theoretical object
The parameter vector $\theta = (\pi, A, \mu, \sigma^2)$ of a $k$-state discrete-time hidden Markov model with Gaussian emissions.

### Expected equations / definitions
- Initial distribution: $\pi \in \Delta^{k-1}$, $\sum_{j=1}^k \pi_j = 1$
- Transition matrix: $A \in \mathbb{R}^{k \times k}$, $A_{ij} \geq 0$, $\sum_j A_{ij} = 1$
- Emission: $y_t | s_t = j \sim \mathcal{N}(\mu_j, \sigma^2_j)$
- Stored flat: `transition[i*k + j]` = $A_{ij}$

### Code locations
- `src/model/params.rs`

### Key types / functions
| Symbol | Role |
|--------|------|
| `ModelParams { k, pi, transition, means, variances }` | Complete parameter set |
| `ModelParams::validate()` | Checks row-sum, positivity, finite values |
| `ModelParams::transition_row(i)` | Returns slice `transition[i*k..(i+1)*k]` |

### How it is used
`ModelParams` is the output of `fit_em()` and the input to every downstream stage: `filter()`, `smooth()`, `pairwise()`, `simulate()`, `OnlineFilterState::step()`. Stored as JSON in `model_params.json` and `fit_summary.json`.

### E2E reachability
All 6 registered experiments produce a `ModelParams` value after the training stage.

### Artifacts / outputs
`model_params.json`, `fit_summary.json` — serialised `FittedParamsSummary`.

### Tests
- `model/params.rs`: validate rejects bad row sums, negative variances, non-finite values

### Status
**IMPLEMENTED_AND_USED**

### Gaps / risks
- Flat `Vec<f64>` storage for the transition matrix requires careful index arithmetic; `transition_row()` is the only safe accessor and should be used consistently.
- No test verifies that `transition_row()` panics correctly on out-of-bounds `i`.

### Thesis note
§ "Model specification" — quote `ModelParams::validate()` as the formal contract for $\theta$.

---

## 7. T4 — Gaussian Emission Model

### Theoretical object
Univariate Gaussian emission density for regime $j$:
$$f(y \mid j) = \frac{1}{\sqrt{2\pi\sigma^2_j}} \exp\!\left(-\frac{(y - \mu_j)^2}{2\sigma^2_j}\right)$$

### Expected equations / definitions
Log-density: $\ell_j(y) = -\tfrac{1}{2}\ln(2\pi\sigma^2_j) - \tfrac{(y-\mu_j)^2}{2\sigma^2_j}$

### Code locations
- `src/model/emission.rs`

### Key types / functions
| Symbol | Role |
|--------|------|
| `Emission { k, means, variances }` | Holds emission parameters for all regimes |
| `log_density(y, j)` | Returns $\ell_j(y)$ |
| `density_vec(y)` | Returns `[f(y|0), …, f(y|k-1)]` used by filter and smoother |

### How it is used
`Emission::density_vec()` is called inside `filter::bayes_update()` and `smoother::smooth()`. `log_density()` is called in `likelihood::log_likelihood_contributions()`.

### E2E reachability
Called on every observation in every experiment run.

### Artifacts / outputs
No direct artifact; contributes to `log_likelihood` in `fit_summary.json`.

### Tests
- `test_peak_density_at_mean` — density is maximised at $\mu_j$
- `test_log_density_matches_ln_density` — log/exp consistency
- `test_density_vec_sums_not_necessarily_one` — sanity on un-normalised densities
- `test_log_density_extreme_sigma` — numerical stability with large/small variance
- Several boundary tests

### Status
**IMPLEMENTED_AND_USED**

### Gaps / risks
- No variance floor is enforced inside `Emission` itself; the floor is applied in the EM M-step (`EmConfig::var_floor`). If `Emission` is constructed with `variance = 0`, `log_density` returns `NaN`/`-Inf`. Only `ModelParams::validate()` and the EM guard against this.

### Thesis note
Direct implementation of § "Emission distribution". The Gaussian assumption is restrictive for fat-tailed financial returns; the thesis should acknowledge this and note that the Z-score normalisation partially mitigates it.

---

## 8. T5 — Synthetic Generator

### Theoretical object
Forward simulation of a hidden Markov model to produce labelled regime sequences and Gaussian observations for benchmarking.

### Expected equations / definitions
1. Draw $s_1 \sim \pi$
2. For $t = 2, \ldots, T$: draw $s_t \sim \text{Categorical}(A_{s_{t-1}, \cdot})$
3. Draw $y_t \sim \mathcal{N}(\mu_{s_t}, \sigma^2_{s_t})$

### Code locations
- `src/model/simulate.rs`
- `src/experiments/synthetic_backend.rs` — `build_synthetic_params()`

### Key types / functions
| Symbol | Role |
|--------|------|
| `simulate(params, t, rng)` | Main entry; returns `SimulationResult` |
| `SimulationResult { states, observations, params }` | Full simulation output |
| `build_synthetic_params(scenario_id, k)` | Constructs calibrated params for a named scenario |

### How it is used
`SyntheticBackend::resolve_data()` calls `build_synthetic_params()` then `simulate()`. The `states` vector provides ground-truth changepoint labels; `observations` becomes the observation sequence fed to the EM and online filter.

### E2E reachability
Three registered synthetic experiments (`hard_switch`, `posterior_transition`, `surprise`) all use `SyntheticBackend` → `simulate()`.

### Artifacts / outputs
`changepoints.csv` — ground-truth changepoint times (when `write_csv = true`).

### Tests
- Phase 2 spec tests (a–f): correct sequence length, state distribution, observation distribution, reproducibility with fixed seed

### Status
**IMPLEMENTED_AND_USED**

### Gaps / risks
- `build_synthetic_params()` defines its own hardcoded parameter tables. These are not derived from the calibration workflow; they are approximate hand-crafted values. Running the full calibration pipeline (`calibrate` CLI command) produces `CalibratedSyntheticParams` but the registered experiments do not consume them automatically.

### Thesis note
The simulation is used to validate the entire inference pipeline under known ground truth. Chapter 4 (experiments) should note the relationship between `build_synthetic_params` scenario definitions and the true target parameter ranges.

---

## 9. T6 — Forward Filter and Log-Likelihood

### Theoretical object
The Hamilton (1989) forward filter: recursive computation of the filtered state posterior $\alpha_{t|t}(j) = P(s_t = j \mid y_{1:t})$ and the predictive density $c_t = p(y_t \mid y_{1:t-1})$.

### Expected equations / definitions
**Predict step:**
$$\alpha_{t|t-1}(j) = \sum_{i=1}^k \alpha_{t-1|t-1}(i) \cdot A_{ij}$$

**Update step:**
$$\alpha_{t|t}(j) = \frac{\alpha_{t|t-1}(j) \cdot f(y_t \mid j)}{c_t}, \quad c_t = \sum_{j=1}^k \alpha_{t|t-1}(j) \cdot f(y_t \mid j)$$

**Log-likelihood:**
$$\mathcal{L}(\theta) = \sum_{t=1}^T \ln c_t$$

### Code locations
- `src/model/filter.rs`
- `src/model/likelihood.rs`

### Key types / functions
| Symbol | Role |
|--------|------|
| `FilterResult { predicted, filtered, log_predictive, log_likelihood }` | Complete filter output |
| `filter(params, obs)` | Main entry; runs full forward pass |
| `predict(params, filtered_prev)` | One predict step |
| `bayes_update(predicted, densities)` | One update step |
| `log_sum_exp(log_probs)` | Numerically stable logsumexp |
| `log_likelihood(params, obs)` | Convenience wrapper returning scalar LL |
| `log_likelihood_contributions(params, obs)` | Per-step $\ln c_t$ |

### How it is used
- Offline: `filter()` called by `e_step()` in `fit_em()` during every EM iteration
- Online: `OnlineFilterState::step()` replicates one predict+update step using the same arithmetic
- Diagnostics: `FittedModelDiagnostics` runs `filter()` on held-out obs

### E2E reachability
Called on every EM iteration for all 6 registered experiments, and on every online observation.

### Artifacts / outputs
`loglikelihood_history.csv` — per-iteration LL trace from EM (when `write_csv = true`).

### Tests
- `test_t1_exact_posterior` — analytical 2-state single-observation posterior
- `test_filter_log_likelihood_finite` — no NaN/Inf for valid inputs
- `test_filter_preserves_normalisation` — filtered posteriors sum to 1.0
- `test_predict_step_stochastic_matrix` — predict output sums to 1
- `test_bayes_update_normalises` — update output sums to 1
- 6 additional scenario-invariant tests

### Status
**IMPLEMENTED_AND_USED**

### Gaps / risks
- `log_sum_exp` is local to `filter.rs`; the same computation is duplicated inline in `emission.rs` and `online/mod.rs`. Divergence risk if one copy is updated.
- The `log_predictive` field stores $\ln c_t$ but its sign convention (positive = log-probability, negative = surprise) is not documented in the struct; `SurpriseDetector` uses `-log_predictive` to compute surprise scores.

### Thesis note
Core algorithmic contribution. Present the predict + Bayes update in full matrix form, then show the log-space implementation in `filter.rs` as the direct code realisation.

---

## 10. T7 — Backward Smoother

### Theoretical object
Kim (1994) backward smoother: computation of the smoothed state posterior $\gamma_t(j) = P(s_t = j \mid y_{1:T})$.

### Expected equations / definitions
**Terminal condition:** $\gamma_T(j) = \alpha_{T|T}(j)$

**Backward recursion:**
$$\gamma_{t}(j) = \alpha_{t|t}(j) \sum_{i=1}^k A_{ji} \frac{\gamma_{t+1}(i)}{\alpha_{t+1|t}(i)}$$

### Code locations
- `src/model/smoother.rs`

### Key types / functions
| Symbol | Role |
|--------|------|
| `SmootherResult { smoothed }` | $T \times k$ matrix of $\gamma_t(j)$ |
| `smooth(params, filter_result)` | Runs backward pass; calls `filter()` internally if needed |

### How it is used
Called by `e_step()` in `fit_em()` after the forward filter. The smoothed posteriors are consumed by `pairwise()` to compute $\xi_t(i,j)$.

### E2E reachability
Every EM iteration for all 6 registered experiments.

### Artifacts / outputs
No direct artifact; intermediate step for EM M-step.

### Tests
- `terminal_condition_exact` — $\gamma_T = \alpha_{T|T}$
- `smoothed_sums_to_one` — $\sum_j \gamma_t(j) = 1$ for all $t$
- `smoother_more_decisive_at_regime_boundary` — smoothed is more concentrated than filtered near a true changepoint
- 6 additional tests

### Status
**IMPLEMENTED_AND_USED**

### Gaps / risks
- Division by $\alpha_{t+1|t}(i)$ is numerically unsafe if the predicted probability is exactly zero (degenerate regime). Protected by a zero-check in the implementation, but no test exercises the exact-zero case.

### Thesis note
Present alongside T6 as the E-step pair: forward filter computes $\alpha_{t|t}$, backward smoother computes $\gamma_t$.

---

## 11. T8 — Pairwise Transition Posteriors

### Theoretical object
The pairwise smoothed posterior $\xi_t(i,j) = P(s_{t-1} = i, s_t = j \mid y_{1:T})$ required by the M-step update for the transition matrix.

### Expected equations / definitions
$$\xi_t(i,j) = \frac{\alpha_{t-1|t-1}(i) \cdot A_{ij} \cdot \gamma_t(j)}{\alpha_{t|t-1}(j)}$$

$$\hat{A}_{ij}^{\text{new}} = \frac{\sum_{t=2}^T \xi_t(i,j)}{\sum_{t=2}^T \gamma_{t-1}(i)}$$

### Code locations
- `src/model/pairwise.rs`

### Key types / functions
| Symbol | Role |
|--------|------|
| `PairwiseResult { xi, expected_transitions }` | Per-step $\xi_t(i,j)$ and summed $\hat{A}_{ij}$ numerators |
| `pairwise(params, filter_result, smoother_result)` | Computes full $\xi$ tensor |

### How it is used
`e_step()` calls `pairwise()` and passes the result to `m_step()` which reads `expected_transitions` to update $A$.

### E2E reachability
Every EM iteration for all 6 registered experiments.

### Artifacts / outputs
No direct artifact.

### Tests
- `pairwise_sums_to_smoother` — $\sum_j \xi_t(i,j) = \gamma_{t-1}(i)$
- `pairwise_expected_transitions_row_sums` — consistency with M-step denominator
- 7 additional structural tests

### Status
**IMPLEMENTED_AND_USED**

### Gaps / risks
- `PairwiseResult::xi` is a 3-D `Vec<Vec<Vec<f64>>>` ($T \times k \times k$); for large $T$ and $k$ this can be memory-intensive. Currently $k = 2$ in all experiments, so no practical issue.

### Thesis note
Present the $\xi_t(i,j)$ formula clearly as the bridge between the E-step and the M-step transition update.

---

## 12. T9 — EM Estimation

### Theoretical object
Baum-Welch EM algorithm for maximum-likelihood estimation of $\theta$ given observations $y_{1:T}$.

### Expected equations / definitions
**E-step:** Compute $\gamma_t(j)$ (smoother) and $\xi_t(i,j)$ (pairwise).

**M-step updates:**
$$\hat{\pi}_j = \gamma_1(j)$$
$$\hat{A}_{ij} = \frac{\sum_{t=2}^T \xi_t(i,j)}{\sum_{t=2}^T \gamma_{t-1}(i)}$$
$$\hat{\mu}_j = \frac{\sum_{t=1}^T \gamma_t(j) \cdot y_t}{\sum_{t=1}^T \gamma_t(j)}$$
$$\hat{\sigma}^2_j = \max\!\left(\text{var\_floor},\ \frac{\sum_{t=1}^T \gamma_t(j) (y_t - \hat{\mu}_j)^2}{\sum_{t=1}^T \gamma_t(j)}\right)$$

**Convergence:** $|\mathcal{L}^{(n+1)} - \mathcal{L}^{(n)}| < \text{tol}$ or `max_iter` reached.

**Multi-start:** Run $n_{\text{starts}}$ independent initialisations; return run with highest final LL.

### Code locations
- `src/model/em.rs`
- `src/experiments/shared.rs` — `train_or_load_model_shared()`

### Key types / functions
| Symbol | Role |
|--------|------|
| `EmConfig { tol, max_iter, var_floor }` | Hyperparameters |
| `EmResult { params, log_likelihood, ll_history, n_iter, converged }` | Fit output |
| `fit_em(obs, config, k, n_starts, rng)` | Main entry; orchestrates multi-start |
| `e_step(params, obs)` | Returns `EStepResult { filter, smoother, pairwise }` |
| `m_step(e_step_result, obs)` | Returns updated `ModelParams` |

### How it is used
`train_or_load_model_shared()` calls `fit_em()` on the train partition. All 6 registered experiments use `TrainingMode::FitOffline` → `fit_em()` with `em_n_starts = 1`.

### E2E reachability
Training stage for every registered experiment.

### Artifacts / outputs
`fit_summary.json` (convergence flag, LL, n_iter, final params), `loglikelihood_history.csv`.

### Tests
- `log_likelihood_nondecreasing` — LL never decreases between EM iterations
- `fit_em_basic_convergence` — fits a 2-regime model from clean synthetic data
- `m_step_pi_sums_to_one` — $\hat{\pi}$ is a valid distribution
- `m_step_transition_rows_sum_to_one`
- `var_floor_applied` — variance never goes below `var_floor`
- 1 additional multi-start test

### Status
**IMPLEMENTED_AND_USED**

### Gaps / risks
- All registered experiments set `em_n_starts: 1`. Multi-start EM is fully implemented and tested but is not exercised in any registered experiment. The `multi_start_summary.json` artifact is never produced in practice.
- `init_params_from_obs()` is defined in **both** `src/experiments/shared.rs` (line ~253) and `src/experiments/synthetic_backend.rs`. This is a silent duplicate with divergence risk if one copy is updated. See [§22](#22-dead-code-and-duplicate-implementation-risks).

### Thesis note
The EM chapter should present the non-decreasing LL property (guaranteed by the EM construction) and discuss the multi-start protocol as a safeguard against local optima.

---

## 13. T10 — Diagnostics

### Theoretical object
Post-fit trust assessment: flag fitted parameters that suggest numerical or statistical pathologies before the model is used for online inference.

### Expected equations / definitions
- Near-zero variance: $\sigma^2_j < \epsilon$ for some threshold $\epsilon$
- Nearly unused regime: $\sum_t \gamma_t(j) < \delta \cdot T$
- EM non-monotonicity: $\mathcal{L}^{(n+1)} < \mathcal{L}^{(n)} - \epsilon$
- Suspicious persistence: $A_{jj} > 1 - 10^{-4}$ (near-unit-root regime)
- Instability across starts: $\max_i \max_j |A_{ij}^{(i)} - A_{ij}^{(0)}| > \delta$

### Code locations
- `src/model/diagnostics.rs`

### Key types / functions
| Symbol | Role |
|--------|------|
| `DiagnosticWarning` enum | 5 warning kinds |
| `FittedModelDiagnostics { warnings, … }` | Per-run diagnostic output |
| `MultiStartSummary` | Cross-start parameter stability |
| `diagnose(result, obs)` | Runs all checks; returns `FittedModelDiagnostics` |
| `compare_runs(results)` | Computes instability across starts |

### How it is used
`train_or_load_model_shared()` calls `diagnose()` after `fit_em()`. Warnings are propagated into `result.warnings` and serialised to `diagnostics.json`. `diagnostics_ok` field in `ModelArtifact` controls whether the runner marks the run as `RunStatus::Degraded`.

### E2E reachability
Every training stage for all 6 registered experiments. Warnings visible in CLI output and `diagnostics.json`.

### Artifacts / outputs
`diagnostics.json` — all warnings with descriptions, severity, per-regime statistics.

### Tests
- `test_diagnose_no_warnings_on_healthy_params`
- `test_diagnose_near_zero_variance`
- `test_diagnose_nearly_unused_regime`
- `test_compare_runs_unstable`

### Status
**IMPLEMENTED_AND_USED**

### Gaps / risks
- `diagnose()` always runs even when `n_starts = 1`, in which case the `UnstableAcrossStarts` warning can never fire. The check is benign but wasteful.

### Thesis note
The diagnostics layer is not a theoretical contribution but is essential validation infrastructure. A footnote in the experiments chapter should mention that all reported results passed the diagnostic checks.

---

## 14. T11 — Offline-Trained, Online-Filtered Runtime

### Theoretical object
Deployment pattern: train $\hat{\theta}$ offline on historical data; freeze parameters; run the Hamilton filter online (causal, one-step-ahead) on new observations without re-fitting.

### Expected equations / definitions
Same predict + update as T6, but applied sequentially and causally:
- At each step $t$, only $y_1, \ldots, y_t$ is seen.
- Filtered posterior $\alpha_{t|t}$ is the live state estimate.
- Predictive density $c_t$ and predicted posterior $\alpha_{t|t-1}$ are the inputs to detectors.

### Code locations
- `src/online/mod.rs` — `OnlineFilterState`, `OnlineStepResult`
- `src/detector/frozen.rs` — `FrozenModel`, `StreamingSession<D>`
- `src/experiments/shared.rs` — `run_online_shared()`

### Key types / functions
| Symbol | Role |
|--------|------|
| `OnlineFilterState { filtered, t, cumulative_log_score }` | Running filter state |
| `OnlineFilterState::new(params)` | Initialises from $\pi$ |
| `OnlineFilterState::step(y, params)` | One predict + update; returns `OnlineStepResult` |
| `OnlineFilterState::step_batch(obs, params)` | Runs the full batch at once |
| `FrozenModel { params }` | Immutable parameter holder |
| `FrozenModel::from_em_result(result)` | Wraps fitted params |
| `StreamingSession<D: Detector>` | Pairs `FrozenModel` with a `Detector` instance |

### How it is used
`run_online_shared()` constructs `OnlineFilterState::new()` then iterates over features calling `step()` and the selected detector's `update()` on each step. Score traces and alarm indices are collected.

### E2E reachability
All 6 registered experiments; the online pass runs on the full feature sequence (train + val + test) after model fitting.

### Artifacts / outputs
`score_trace.csv`, `alarms.csv`, `regime_posteriors.csv` (when `save_traces = true`).

### Tests
- `online/mod.rs`: `test_step_produces_valid_posteriors`, `test_cumulative_log_score_finite`, `test_step_batch_matches_sequential`

### Status
**IMPLEMENTED_AND_USED**

### Gaps / risks
- `StreamingSession<D>` is fully implemented with a clean generics interface but is **never instantiated** from the experiment runner. `run_online_shared()` constructs `OnlineFilterState` and detectors independently, bypassing `StreamingSession`. This is a latent API that is tested only indirectly through integration.
- `TrainingMode::LoadFrozen { artifact_id }` config path is implemented in `DryRunBackend` and `train_or_load_model_shared()` but no registered experiment uses it, and the artifact path resolver for re-loading from disk is not implemented in `RealBackend` or `SyntheticBackend`.

### Thesis note
The offline-train / online-filter architecture is the central deployment design decision. Emphasise that frozen parameters mean the filter is deterministic given $\hat{\theta}$: reproducibility and computational efficiency are guaranteed.

---

## 15. T12 — Detector Family

### Theoretical object
A stateful alarm-generation layer that consumes filtered state estimates and fires an alarm when a regime-change signal exceeds a threshold, optionally with persistence and cooldown logic.

### Expected equations / definitions
**HardSwitch score:**
$$s_t = \mathbf{1}\bigl[\operatorname{argmax}_j \alpha_{t|t}(j) \neq \operatorname{argmax}_j \alpha_{t-1|t-1}(j)\ \wedge\ \max_j \alpha_{t|t}(j) \geq \theta_{\text{conf}}\bigr]$$

**PosteriorTransition (LeavePrevious) score:**
$$s_t^{\text{leave}} = 1 - \alpha_{t|t}(r_{t-1})$$

**PosteriorTransition (TotalVariation) score:**
$$s_t^{\text{TV}} = \frac{1}{2}\sum_{j=1}^k \bigl|\alpha_{t|t}(j) - \alpha_{t-1|t-1}(j)\bigr|$$

**Surprise score:**
$$s_t^{\text{surp}} = -\ln c_t - \text{EMA}_{t-1}(-\ln c)$$

**Persistence policy:** alarm fires only if score $\geq \theta$ for $p$ consecutive steps.  
**Cooldown policy:** suppress further alarms for $d$ steps after each alarm.

### Code locations
- `src/detector/mod.rs` — `Detector` trait, `DetectorInput`, `DetectorOutput`, `PersistencePolicy`
- `src/detector/hard_switch.rs` — `HardSwitchDetector`
- `src/detector/posterior_transition.rs` — `PosteriorTransitionDetector`
- `src/detector/surprise.rs` — `SurpriseDetector`
- `src/detector/frozen.rs` — `StreamingSession<D>`

### Key types / functions
| Symbol | File | Role |
|--------|------|------|
| `Detector` trait | mod.rs | `update(input) → DetectorOutput`, `reset()` |
| `dominant_regime(filtered)` | mod.rs | `argmax_j α_{t\|t}(j)` |
| `HardSwitchDetector` | hard_switch.rs | Discrete switch + confidence gate |
| `PosteriorTransitionDetector { score_kind }` | posterior_transition.rs | LeavePrevious or TotalVariation |
| `SurpriseDetector { ema_alpha }` | surprise.rs | EMA-adjusted log-predictive score |

### How it is used
`run_online_shared()` instantiates the detector specified by `cfg.detector.detector_type` and calls `detector.update()` on every online step. The three registered synthetic experiments each test one detector type. The three real experiments use `HardSwitch` and `Surprise`.

### E2E reachability
- `HardSwitch`: `hard_switch` (synthetic), `hard_switch_shock` (synthetic), `real_spy_daily_hard_switch`, `real_spy_intraday_hard_switch`
- `PosteriorTransition (Leave)`: `posterior_transition` (synthetic)
- `PosteriorTransition (TV)`: `posterior_transition_tv` (synthetic) — **registered and exercised**
- `Surprise`: `surprise` (synthetic), `real_wti_daily_surprise`

### Artifacts / outputs
`detector_config.json`, `score_trace.csv`, `alarms.csv`.

### Tests
- `hard_switch.rs`: threshold, persistence, cooldown unit tests
- `posterior_transition.rs`: LeavePrevious and TotalVariation score correctness
- `surprise.rs`: EMA baseline, threshold, persistence
- Integration via `run_online_shared()` tests in `experiments/`

### Status
**IMPLEMENTED_AND_USED** — all four detector variants (HardSwitch, PosteriorLeave, PosteriorTV, Surprise) are registered, reachable, and exercised by named experiments.

### Gaps / risks
- No gradient-based sensitivity test: it is unknown how detector performance changes with small Δθ around the threshold.

---

## 16. T13 — Synthetic Benchmark Evaluation

### Theoretical object
Ground-truth evaluation of detection delay and accuracy against a known changepoint schedule.

### Expected equations / definitions
**Greedy matching protocol:** Alarm $a_n$ is a true positive for changepoint $\tau_m$ iff $\tau_m \leq a_n < \tau_m + w$ and $\tau_m$ is unmatched.

$$\text{Precision} = \frac{|\text{TP}|}{|\text{TP}| + |\text{FP}|}$$
$$\text{Recall} = \frac{|\text{detected}|}{N_{\text{CP}}}$$
$$\text{Delay} = a_n - \tau_m \quad \text{for matched pairs}$$

### Code locations
- `src/benchmark/truth.rs` — `ChangePointTruth`, `from_regime_sequence()`
- `src/benchmark/matching.rs` — `EventMatcher`, `MatchConfig { window }`, greedy protocol
- `src/benchmark/metrics.rs` — `MetricSuite`
- `src/benchmark/result.rs` — `AggregateResult`, `RunResult`

### Key types / functions
| Symbol | Role |
|--------|------|
| `ChangePointTruth::from_regime_sequence(states)` | Extract changepoint times from simulated states |
| `EventMatcher::match_events(truth, alarms, config)` | Returns `MatchResult` |
| `MetricSuite::from_match(match_result)` | Computes precision/recall/delay/FAR |

### How it is used
`SyntheticBackend::evaluate_synthetic()` calls `ChangePointTruth::from_regime_sequence()` on the simulated states, then `EventMatcher::match_events()`, then `MetricSuite::from_match()`. Results populate `EvaluationSummary::Synthetic`.

### E2E reachability
Three synthetic registered experiments.

### Artifacts / outputs
`summary.json` — evaluation summary with precision/recall/delay metrics.

### Tests
- `matching.rs`: exact greedy matching on known alarm/CP sequences, edge cases (no alarms, all alarms, window = 1)
- `metrics.rs`: MetricSuite field correctness, precision/recall boundary conditions

### Status
**IMPLEMENTED_AND_USED**

### Gaps / risks
- `AggregateResult` and multi-run `BenchmarkLabel` in `result.rs` are implemented but not populated by any CLI command. Aggregate across multiple runs is only available programmatically.
- The matching window $w$ is fixed per experiment in `ExperimentConfig`; no sensitivity sweep is registered.

### Thesis note
The delay and precision/recall metrics are the primary quantitative outputs for the thesis results tables. The greedy matching protocol should be described and justified against the optimal bipartite matching alternative.

---

## 17. T14 — Real-Data Evaluation

### Theoretical object
Reference-free assessment of detector quality on real financial data where ground-truth labels do not exist.

### Expected equations / definitions
**Route A — Proxy-event alignment:**
- `event_coverage` = fraction of proxy events with at least one aligned alarm within tolerance
- `alarm_relevance` = fraction of alarms that fall within any event window

**Route B — Segmentation self-consistency:**
$$\text{coherence\_score} = \frac{\text{mean between-segment } |\Delta\mu|}{\text{mean within-segment } \sigma}$$

### Code locations
- `src/real_eval/route_a.rs` — `ProxyEvent`, `EventAlignment`, `evaluate_proxy_events()`
- `src/real_eval/route_b.rs` — `DetectedSegment`, `AdjacentSegmentContrast`, `evaluate_segmentation()`
- `src/real_eval/report.rs` — `RealEvalResult`, `evaluate_real_data()`
- `src/experiments/real_backend.rs` — `RealBackend::evaluate_real()`

### Key types / functions
| Symbol | Role |
|--------|------|
| `ProxyEvent { id, event_type, label, anchor }` | External reference event |
| `RouteBConfig { min_len, short_segment_policy }` | Segmentation params |
| `SegmentationGlobalSummary { coherence_score }` | Aggregate Route B metric |
| `evaluate_real_data(meta, obs, alarm_indices, cfg)` | Orchestrator |

### How it is used
`RealBackend::evaluate_real()` calls `evaluate_real_data()`. Route A reads proxy event fixtures from disk (path derived from `cfg.meta.run_label`). Route B computes statistics directly from `alarm_indices` and the observation series.

### E2E reachability
Three registered real experiments. Route B always runs. Route A runs only if proxy event fixture files exist.

### Artifacts / outputs
`real_eval_summary.csv` — event_coverage, alarm_relevance, segmentation_coherence.  
`route_a_result.json`, `route_b_result.json` — detailed per-event and per-segment data.

### Tests
- `route_a.rs`: `evaluate_proxy_events()` unit tests with fixture events
- `route_b.rs`: segmentation correctness, short-segment policy, coherence formula

### Status
**IMPLEMENTED_AND_USED** — Route A and Route B both fully functional. Proxy event fixture files `data/proxy_events/spy.json` (13 events), `data/proxy_events/wti.json`, and `data/proxy_events/gold.json` are committed to the repository. Route A produced 1 matched event for SPY daily with `event_coverage=0.077`, `alarm_relevance=0.091`; low metric values reflect detector sensitivity relative to event density, not missing infrastructure.

### Gaps / risks
- The coherence score formula in `evaluate_segmentation()` uses mean absolute delta mean normalised by mean within-segment variance, but the exact formula is not documented in a docstring; the reader must infer it from code.

### Thesis note
Route A and Route B are original methodological contributions. The thesis should present them as the "ground-truth-free" evaluation framework and explain the proxy-event construction protocol.

---

## 18. T15 — Synthetic-to-Real Calibration

### Theoretical object
Mapping empirical statistics from a real data series to a synthetic MSM parameter set, so that simulated benchmarks reflect the statistical character of real financial data.

### Expected equations / definitions
**Regime persistence:** $\hat{A}_{jj} = 1 - 1/\bar{d}_j$ where $\bar{d}_j$ is target mean episode duration.

**Mean policies:**
- `ZeroCentered`: $\hat{\mu}_j = 0$
- `SymmetricAroundEmpirical`: $\hat{\mu}_j = \pm \delta$
- `EmpiricalBaseline`: $\hat{\mu}_j = \bar{y}_j^{\text{emp}}$

**Variance policies:**
- `QuantileAnchored`: variances from empirical quantiles of the per-regime variance distribution
- `RatioAroundEmpirical`: variances as multiples of empirical global variance

### Code locations
- `src/calibration/summary.rs` — `EmpiricalCalibrationProfile`, `summarize_feature_stream()`
- `src/calibration/mapping.rs` — `CalibrationMappingConfig`, `calibrate_to_synthetic()`
- `src/calibration/verify.rs` — `CalibrationVerification`, `verify_calibration()`
- `src/calibration/report.rs` — `CalibrationReport`, `run_calibration_workflow()`
- `src/cli/mod.rs` — `cmd_calibrate()` wires CLI → `run_calibration_workflow()`

### Key types / functions
| Symbol | Role |
|--------|------|
| `EmpiricalCalibrationProfile` | Summary statistics from real feature stream |
| `CalibratedSyntheticParams` | Output: full MSM params matched to empirical profile |
| `JumpParams { prob, scale_mult }` | Jump-contamination params for `model::simulate` |
| `calibrate_to_synthetic(profile, config)` | Mapping function |
| `verify_calibration(calibrated, profile, tol)` | Sanity-check output against input |
| `run_calibration_workflow(cfg, obs, meta)` | Orchestrator called by CLI |
| `simulate_with_jump(params, t, rng, jump)` | Core simulation with optional Bernoulli shock overlay |

### How it is used
CLI `calibrate` command → `run_calibration_workflow()` → `summarize_feature_stream()` → `calibrate_to_synthetic()` → `verify_calibration()`. The result is printed to the terminal and optionally serialised.

When `CalibrationMappingConfig.jump` is `Some(JumpContamination)`, `run_calibration_workflow()` converts it to `JumpParams` and calls `simulate_with_jump()`, overlaying Bernoulli shocks (probability `jump_prob`, scale `jump_scale_mult × σⱼ`) on the Gaussian regime emissions. The `hard_switch_shock` registered experiment (scenario `shock_contaminated`) exercises this path via `calibrate --id hard_switch_shock`.

### E2E reachability
Reachable via `calibrate --id <id>`. Jump contamination path exercised by `calibrate --id hard_switch_shock`.

### Artifacts / outputs
`calibration_summary.json`, `synthetic_vs_empirical_summary.json`, `calibrated_scenario.json` (saved to `./runs/calibration/<id>/`).

### Tests
- `calibration/mapping.rs`: round-trip tests, boundary conditions, duration-to-transition formula
- `calibration/verify.rs`: tolerance checks

### Status
**IMPLEMENTED_AND_USED** — `JumpContamination` config fully wired from `CalibrationMappingConfig` through `CalibratedSyntheticParams` to `simulate_with_jump()`. The `hard_switch_shock` registered experiment exercises the complete jump-contamination calibration path.

### Gaps / risks
- Calibrated params are not consumed by any registered experiment; the calibration workflow is a standalone analysis tool disconnected from the experiment pipeline.
- `CalibrationReport` is printed but not written to a stable file path. There is no `calibration_report.json` artifact.

---

## 19. T16 — Experiment Runner

### Theoretical object
Orchestration layer that composes data, features, model, online filter, and evaluation into a reproducible, serialisable experiment run.

### Expected equations / definitions
No new mathematics. The runner is pure software engineering: pipeline composition, timing, artifact management, reproducibility.

### Code locations
- `src/experiments/runner.rs` — `ExperimentBackend` trait, `ExperimentRunner<B>`, `DryRunBackend`
- `src/experiments/config.rs` — `ExperimentConfig` and all sub-configs
- `src/experiments/registry.rs` — 8 registered experiments
- `src/experiments/shared.rs` — shared training and online stages
- `src/experiments/synthetic_backend.rs` — `SyntheticBackend`
- `src/experiments/real_backend.rs` — `RealBackend`
- `src/experiments/batch.rs` — `run_batch()`
- `src/experiments/search.rs` — `grid_search()`, `optimize_full()`
- `src/experiments/artifact.rs` — artifact file management
- `src/experiments/result.rs` — `ExperimentResult`, `EvaluationSummary`

### Key types / functions
| Symbol | Role |
|--------|------|
| `ExperimentBackend` trait | 6 methods: resolve_data, build_features, train_or_load_model, run_online, evaluate_synthetic, evaluate_real |
| `ExperimentRunner::run(cfg)` | Main entry; 6 stages with timing and error recovery |
| `DryRunBackend` | Deterministic stub for orchestration tests |
| `run_batch(configs, backend)` | Runs multiple experiments sequentially |
| `grid_search(backend, base_cfg, param_grid)` | Threshold/persistence sweep |
| `optimize_full(backend, base_cfg, model_grid, param_grid)` | Full model + detector grid search |

### Registered experiments
| ID | Mode | Asset | Detector | Horizon |
|----|------|-------|----------|---------|
| `hard_switch` | Synthetic | — | HardSwitch | 2000 |
| `posterior_transition` | Synthetic | — | PosteriorTransition (Leave) | 2000 |
| `posterior_transition_tv` | Synthetic | — | PosteriorTransition (TV) | 2000 |
| `surprise` | Synthetic | — | Surprise | 2000 |
| `hard_switch_shock` | Synthetic | — | HardSwitch (shock_contaminated) | 2000 |
| `real_spy_daily_hard_switch` | Real | SPY daily | HardSwitch | — |
| `real_wti_daily_surprise` | Real | WTI daily | Surprise | — |
| `real_spy_intraday_hard_switch` | Real | SPY 15min | HardSwitch | — |

### How it is used
CLI `run-experiment <id>` → `registry::get(id)` → `ExperimentRunner::run()`. `run-batch` calls `run_batch()`. `param-search` calls `grid_search()`. `optimize` calls `optimize_full()`.

### E2E reachability
All 8 experiments reachable via direct CLI commands.

### Artifacts / outputs
Run directory tree under `runs/<mode>/<label>/<run_id>/`:
- `config/experiment_config.json`
- `metadata/run_metadata.json`
- `results/evaluation_summary.json`, `summary.json`
- `model_params.json`, `fit_summary.json`, `diagnostics.json`
- `feature_summary.json`, `detector_config.json`
- `score_trace.csv`, `alarms.csv`, `regime_posteriors.csv` (if save_traces)
- `changepoints.csv` (synthetic), `real_eval_summary.csv` (real)
- `split_summary.json`, `data_quality.json` (real)

### Tests
- `experiments/runner.rs`: `test_runner_dry_run_completes`, `test_runner_timing_stages_recorded`, `test_runner_invalid_config_fails_gracefully`, reproducibility tests
- `experiments/real_backend.rs`: `resolve_data_train_n_is_seventy_percent`, `full_pipeline_with_fixture_data`
- `experiments/batch.rs`: batch orchestration tests

### Status
**IMPLEMENTED_AND_USED**

### Gaps / risks
- `#![allow(dead_code)]` suppressor at the top of `runner.rs`, `search.rs`, `batch.rs` — indicates public APIs that are not yet all called from tests or CLI. Should be audited when the codebase stabilises.
- `init_params_from_obs()` duplicate (see T9 and §22).
- `TrainingMode::LoadFrozen` path is not implemented in `SyntheticBackend` or `RealBackend` — calling it returns a `not implemented` error.

### Thesis note
The runner architecture allows the thesis to present experiments as reproducible, identified by `run_id` (derived from config hash + timestamp). This is a software-engineering contribution that enables the repeatability claims.

---

## 20. T17 — Reporting and Plotting

### Theoretical object
Artefact generation: export run results as JSON/CSV for archival, as Markdown/LaTeX tables for the thesis, and as PNG plots for visualisation.

### Expected equations / definitions
No new mathematics. Metrics tables present T13/T14 outputs. Plots render time-series with alarm markers, posterior probability traces, score traces.

### Code locations
- `src/reporting/mod.rs` — re-exports; plot render functions always compiled (no test gate)
- `src/reporting/report.rs` — `RunReporter { export_run, generate_tables, generate_plots }`, `AggregateReporter`
- `src/reporting/artifact.rs` — `RunArtifactLayout` path resolver
- `src/reporting/export/json.rs` — `export_config`, `export_result`, `export_evaluation_summary`
- `src/reporting/export/csv.rs` — CSV export helpers
- `src/reporting/export/schema.rs` — serialisable record types
- `src/reporting/table/metrics.rs` — `MetricsTableBuilder` (markdown/CSV/LaTeX output)
- `src/reporting/table/comparison.rs` — comparison table
- `src/reporting/table/segment_summary.rs` — segment summary
- `src/reporting/plot/signal_alarms.rs` — `render_signal_with_alarms`
- `src/reporting/plot/detector_scores.rs` — `render_detector_scores`
- `src/reporting/plot/regime_posteriors.rs` — `render_regime_posteriors`
- `src/reporting/plot/delay_distribution.rs`
- `src/reporting/plot/segmentation.rs`

### Key types / functions
| Symbol | Role |
|--------|------|
| `RunReporter::export_run()` | Writes JSON config + result |
| `RunReporter::generate_tables()` | Writes metrics.md / metrics.csv / metrics.tex |
| `RunReporter::generate_plots()` | Renders PNG plots |
| `AggregateReporter::generate_comparison_table()` | Multi-run comparison, exposed via `compare-runs` CLI |
| `MetricsTableBuilder` | Fluent builder for Markdown/CSV/LaTeX metric tables |

### How it is used
`ExperimentRunner::run()` directly calls the export functions inline (JSON, CSV). `RunReporter` is instantiated for full reporting. Plot render functions are always compiled; they are called during non-test experiment runs. `AggregateReporter` is reachable via `compare-runs --dir <dir> [--dir <dir> ...] [--save <path>]`.

### E2E reachability
JSON and CSV exports: all 8 registered experiments (when `write_json = true`).  
Table generation: called after every `run-experiment` command.  
Plot generation: called for all experiment runs.  
`AggregateReporter`: **reachable via `compare-runs` CLI subcommand**.

### Artifacts / outputs
See [T16 artifacts](#19-t16--experiment-runner). Additionally:
- `tables/metrics_table.md`, `tables/metrics_table.tex`
- `plots/signal_with_alarms.png`, `plots/detector_scores.png`, `plots/regime_posteriors.png`

### Tests
- `reporting/export/schema.rs`: serialisation round-trip for `AlarmRecord`
- Plot input struct serialisation tested

### Status
**IMPLEMENTED_AND_USED** — JSON/CSV/table export wired to all experiment runs; `#[cfg(not(test))]` gates removed so render functions are always compiled; `AggregateReporter` exposed via `compare-runs` CLI subcommand.

### Gaps / risks
- `#![allow(dead_code)]` on `reporting/report.rs`, `reporting/artifact.rs`, `reporting/export/schema.rs` — some struct fields or methods may be unused.
- No integration test exercises the full plot-generation path (render functions call the plotters bitmap backend which writes files; this is excluded from unit tests to keep test execution fast).

### Thesis note
The LaTeX table output from `MetricsTableBuilder` can be embedded directly in the thesis results chapter. The plot PNG files are suitable for figures.

---

## 21. T18 — CLI / Interactive Application Layer

### Theoretical object
User interface: interactive terminal menus (using `inquire`) and direct subcommand dispatch for headless / scripted use.

### Expected equations / definitions
None.

### Code locations
- `src/main.rs` — entry point, `is_direct_command()`, `main()`
- `src/cli/mod.rs` — `run()` (interactive), `run_direct()` (direct)

### Key types / functions
| Symbol | Role |
|--------|------|
| `is_direct_command(arg)` | Returns true if arg matches any known subcommand |
| `cli::run(cfg)` | Interactive mode with `inquire` menus |
| `cli::run_direct(args)` | Direct subcommand dispatch |

### Interactive menus
```
Main menu:
  Data management
    Refresh data  →  DataService::refresh()
    Show status   →  DataService::status()
    Ingest all    →  DataService::ingest_all()
  Experiments
    run hard_switch
    run posterior_transition
    run surprise
    run real_spy_daily_hard_switch
    run real_wti_daily_surprise
    run real_spy_intraday_hard_switch
  Inspect Runs    →  list run directories
  E2E Run         →  synthetic e2e pipeline
  Exit
```

### Direct subcommands
| Subcommand | Action |
|------------|--------|
| `e2e` | Synthetic end-to-end test |
| `param-search` | Threshold/persistence grid search |
| `run-experiment <id>` | Single registered experiment |
| `run-batch` | All registered experiments |
| `run-real <id>` | Real experiment |
| `calibrate` | Calibration workflow |
| `optimize` | Full model + detector grid search |
| `inspect` | List run artefact directories |
| `status` | Data cache status |
| `help` | Usage |

### E2E reachability
All subcommands tested manually. Automated integration test: `DryRunBackend` tests cover the `run-experiment` path end-to-end.

### Artifacts / outputs
Interactive: formatted terminal output.  
Direct: same artifacts as T16/T17.

### Tests
- CLI dispatch logic is tested through `experiments/runner.rs` dry-run tests
- No dedicated CLI unit tests

### Status
**IMPLEMENTED_AND_USED**

### Gaps / risks
- No automated test for `is_direct_command()` or the menu dispatch logic.
- `inspect` and `status` commands are not tested.
- `help` output is hardcoded — will drift from actual subcommand set if commands are added.

### Thesis note
The CLI layer is an implementation detail; the thesis may describe it briefly as "a command-line interface providing access to all experimental workflows."

---

## 22. Dead Code and Duplicate-Implementation Risks

### High-severity risks

#### 1. `init_params_from_obs()` defined twice
- **Files:** `src/experiments/shared.rs` (line ~253) and `src/experiments/synthetic_backend.rs`
- **Risk:** Silent divergence. If one copy is updated (e.g., better initialisation heuristic) and the other is not, the two backends will initialise EM with different strategies without a compile error.
- **Recommended fix:** Delete one copy; have the other call through.

#### 2. `JumpContamination` — **RESOLVED**
- `JumpParams` added to `model::simulate`; `simulate_with_jump()` applies Bernoulli shocks; `run_calibration_workflow()` passes jump config; `hard_switch_shock` experiment exercises the path.

#### 3. Route A proxy event fixtures — **RESOLVED**
- `data/proxy_events/spy.json` (13 events), `wti.json`, and `gold.json` are committed. Route A is functional; SPY daily produced 1 match with `event_coverage=0.077`, `alarm_relevance=0.091`.

### Medium-severity risks

#### 4. `StreamingSession<D>` — implemented but never instantiated from runner
- **File:** `src/detector/frozen.rs`
- **Risk:** `StreamingSession` provides a clean, type-safe pairing of `FrozenModel` and `Detector`. The runner bypasses it entirely, creating a gap between the public API design and actual usage.
- **Recommended fix:** Refactor `run_online_shared()` to use `StreamingSession`, or document clearly that it is a lower-level API for embedding use.

#### 5. `TrainingMode::LoadFrozen` — config path with no backend implementation
- **File:** `src/experiments/config.rs`, `shared.rs`
- **Risk:** A user can write a config with `training = "LoadFrozen"` but it will error at runtime for `SyntheticBackend` and `RealBackend`.
- **Recommended fix:** Implement artifact re-loading in both backends, or remove the variant with a `todo!()` + doc note.

#### 6. `PosteriorTransitionTV` — **RESOLVED**
- `posterior_transition_tv` registered experiment added; the TV detector is now exercised end-to-end.

#### 7. `AggregateReporter::generate_comparison_table()` — **RESOLVED**
- `compare-runs --dir <dir> [--dir <dir> ...] [--save <path>]` CLI subcommand added.

#### 8. `em_n_starts = 1` for all registered experiments
- **Risk:** Multi-start EM is implemented and tested but never exercised in registered experiments. `multi_start_summary.json` is never produced.
- **Recommended fix:** Set `em_n_starts: 3` in at least one registered experiment.

### Low-severity (code hygiene)

#### Files with top-level `#![allow(dead_code)]` suppressors
These files have module-wide suppression, meaning any dead item is silently ignored by the compiler:

| File | Suppressor |
|------|-----------|
| `src/experiments/runner.rs` | `#![allow(dead_code)]` |
| `src/experiments/search.rs` | `#![allow(dead_code)]` |
| `src/experiments/batch.rs` | `#![allow(dead_code)]` |
| `src/calibration/mapping.rs` | `#![allow(dead_code)]` |
| `src/reporting/report.rs` | `#![allow(dead_code)]` |
| `src/reporting/artifact.rs` | `#![allow(dead_code)]` |
| `src/reporting/export/schema.rs` | `#![allow(dead_code)]` |
| `src/benchmark/matching.rs` | `#![allow(dead_code)]` |
| `src/benchmark/metrics.rs` | `#![allow(dead_code)]` |
| `src/benchmark/truth.rs` | `#![allow(dead_code)]` |
| `src/benchmark/result.rs` | `#![allow(dead_code)]` |
| `src/data/meta.rs` | `#![allow(dead_code)]` |
| `src/data/mod.rs` | `#![allow(unused_imports, dead_code)]` |
| `src/data/session.rs` | `#![allow(dead_code)]` |
| `src/data/split.rs` | `#![allow(dead_code)]` |
| `src/data/validation.rs` | `#![allow(dead_code)]` |
| `src/detector/frozen.rs` | `#![allow(dead_code)]` |
| `src/detector/mod.rs` | `#![allow(unused_imports, dead_code)]` |
| `src/detector/posterior_transition.rs` | `#![allow(dead_code)]` |
| `src/calibration/summary.rs` | `#![allow(dead_code)]` |
| `src/calibration/report.rs` | `#![allow(dead_code)]` |
| `src/features/family.rs` | `#![allow(dead_code)]` |

---

## 23. Documentation Gaps

| Area | Gap | Priority |
|------|-----|----------|
| `StreamingSession<D>` | No explanation of why it is bypassed by the runner | Medium |
| `log_predictive` in `FilterResult` | Sign convention not documented | Medium |
| `coherence_score` formula in Route B | No docstring or mathematical reference | Medium |
| `build_synthetic_params()` | Parameters are not justified against calibration workflow | Medium |
| `TrainingMode::LoadFrozen` | No documentation that backend implementation is missing | Medium |
| `run_calibration_workflow()` | Output written to `./runs/calibration/<id>/` but path not documented in the workflow docstring | Low |
| All plot input structs | No docstrings explaining coordinate conventions | Low |

---

## 24. Test Coverage Gaps

| Component | Gap | Priority |
|-----------|-----|----------|
| Route A (real_eval) | No end-to-end test with real proxy event fixture | High |
| `DataService::refresh()` | No test against network or mock HTTP | Medium |
| `AlphaVantageClient` | No test — deleted `new()` / `with_base_url()` means test construction path is unclear | Medium |
| `StreamingSession<D>` | Only tested indirectly; no direct session-level test | Medium |
| `TrainingMode::LoadFrozen` | Not tested in any backend | Medium |
| `PosteriorTransitionTV` | No end-to-end experiment test | Medium |
| `AggregateReporter` | No test for `generate_comparison_table()` | Low |
| `is_direct_command()` | No unit test | Low |
| Plot rendering | No integration test (by design; `#[cfg(not(test))]`) | Low |
| `init_params_from_obs()` duplicate | Only one copy exercised per test run | Low |

---

## 25. Thesis Chapter Mapping

| Thesis chapter | Theory IDs | Primary source files |
|----------------|------------|---------------------|
| Ch. 2 — Background: HMM and regime-switching | T3, T4 | model/params.rs, model/emission.rs |
| Ch. 3 — Inference algorithms | T6, T7, T8, T9 | model/filter.rs, smoother.rs, pairwise.rs, em.rs |
| Ch. 3.5 — Diagnostics and trust layer | T10 | model/diagnostics.rs |
| Ch. 3.6 — Online deployment | T11 | online/mod.rs, detector/frozen.rs |
| Ch. 4 — Detector design | T12 | detector/*.rs |
| Ch. 4.5 — Synthetic simulation | T5 | model/simulate.rs |
| Ch. 4.6 — Calibration methodology | T15 | calibration/*.rs |
| Ch. 5 — Experiments: synthetic benchmark | T13, T2, T5 | benchmark/*.rs, features/*.rs |
| Ch. 5 — Experiments: real data | T1, T2, T14 | data/*.rs, real_eval/*.rs |
| Ch. 6 — Implementation and reproducibility | T16, T17, T18 | experiments/*.rs, reporting/*.rs, cli/mod.rs |
| Appendix A — Feature engineering | T2 | features/*.rs |
| Appendix B — Data pipeline | T1 | alphavantage/*.rs, cache/mod.rs, data_service/mod.rs |

---

## 26. Final Action List

### Critical (must fix before thesis submission)

1. **Fix JumpContamination gap** — either implement jump injection in `model/simulate.rs` or remove `JumpContamination` from `calibration/mapping.rs` and document it as future work. The current state is misleading.

2. **Create proxy event fixture files** — without at least one fixture file per real experiment, Route A metrics are permanently zero and the real-data evaluation is incomplete. Minimum: commit `data/events/real_spy_daily_hard_switch.json` and `data/events/real_wti_daily_surprise.json`.

3. **Remove `init_params_from_obs()` duplicate** — consolidate to one copy in `shared.rs`; delete from `synthetic_backend.rs`. Add a test that both backends produce the same init params for the same input.

### Important (fix before experiments chapter is written)

4. **Register a `PosteriorTransitionTV` experiment** — add it to `registry.rs` so all detector variants have at least one registered experiment and end-to-end test.

5. **Implement `TrainingMode::LoadFrozen` in `SyntheticBackend` and `RealBackend`** — or add a runtime `anyhow::bail!` with a clear error message, and document the gap.

6. **Expose `AggregateReporter` from CLI** — add a `compare-runs` direct subcommand; the comparison table is needed for the thesis parameter search results.

7. **Set `em_n_starts: 3` in at least one registered experiment** — exercises multi-start EM and produces `multi_start_summary.json` artifact.

8. **Document `log_predictive` sign convention** in `FilterResult` — clarify whether it is $\ln c_t$ (positive, log-probability) or $-\ln c_t$ (positive, surprise). `SurpriseDetector` currently negates it; confirm this is correct.

### Good practice (improve before final review)

9. **Audit and remove `#![allow(dead_code)]` suppressors** module by module — use `#[cfg_attr(not(test), allow(dead_code))]` on individual items where appropriate, or add tests / CLI paths.

10. **Add docstring for `coherence_score`** in `route_b.rs` with the formula and normalisation convention.

11. **Add docstring to `build_synthetic_params()`** explaining the relationship (and disconnect) from `run_calibration_workflow()`.

12. **Add test for `is_direct_command()`** to prevent the help text and dispatch logic from drifting apart.

13. **Implement `AlphaVantageClient` mock/test harness** so `DataService::refresh()` can be tested without a live API key.

14. **Consider refactoring `run_online_shared()` to use `StreamingSession<D>`** — closes the gap between the public API design and the actual runtime path.
