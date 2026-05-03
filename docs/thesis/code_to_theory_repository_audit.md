# Proteus — Code-to-Theory Repository Audit

**Project:** Proteus (Rust Markov-switching change-point detection)
**Audit scope:** `src/`, `docs/`, `runs/`, `data/`, `notes/`, top-level Rust workspace
**Audit purpose:** Determine, for every theoretical component required by the MSc thesis, whether the implementation exists, is wired into the live experiment pipeline, is exercised by tests, and produces artifacts under `runs/`.

Status legend (used throughout):

- **IMPLEMENTED_AND_USED** — code exists, called from a CLI-reachable path, produces artifacts, has tests.
- **IMPLEMENTED_NOT_USED** — code exists but no live caller (only unit tests, or `#[allow(dead_code)]`).
- **PARTIAL** — main path exists, but some sub-piece (e.g. typed export, summary row) is unwired.
- **MISSING** — required by theory but absent from the codebase.
- **UNCLEAR** — caller graph could not be confirmed from static analysis alone.

---

## 1. Executive summary

| Area | Status | Used in live pipeline? | CLI entry | Evidence | Risk for thesis |
|---|---|---|---|---|---|
| Data pipeline (clean, partition, validate) | IMPLEMENTED_AND_USED | Yes (real backend) | `run-real`, `e2e`, `calibrate` | [src/data/mod.rs](src/data/mod.rs), [src/data/split.rs](src/data/split.rs), [src/data/validation.rs](src/data/validation.rs) | Low |
| Session awareness (RTH labelling) | IMPLEMENTED_NOT_USED | **No** — `SessionAwareSeries` never constructed by backends | n/a | [src/data/session.rs](src/data/session.rs) | **Medium** — claim of intraday session handling overstated |
| Feature families & scaling | IMPLEMENTED_AND_USED | Yes (real backend) | `run-real`, `run-experiment` | [src/features/family.rs](src/features/family.rs), [src/features/stream.rs](src/features/stream.rs), [src/features/scaler.rs](src/features/scaler.rs) | Low |
| `ModelParams` / Gaussian emission | IMPLEMENTED_AND_USED | Yes | all | [src/model/params.rs](src/model/params.rs), [src/model/emission.rs](src/model/emission.rs) | Low |
| Forward filter (Hamilton recursion) | IMPLEMENTED_AND_USED | Yes (offline + online) | all | [src/model/filter.rs](src/model/filter.rs), [src/online/mod.rs](src/online/mod.rs) | Low |
| Backward (Kim) smoother | IMPLEMENTED_AND_USED | Yes (E-step) | all train paths | [src/model/smoother.rs](src/model/smoother.rs) | Low |
| Pairwise posteriors $\xi_t(i,j)$ | IMPLEMENTED_AND_USED | Yes (E-step) | all train paths | [src/model/pairwise.rs](src/model/pairwise.rs) | Low |
| Log-likelihood | IMPLEMENTED_AND_USED | Yes | all | [src/model/likelihood.rs](src/model/likelihood.rs) | Low |
| EM estimation (`fit_em`) | IMPLEMENTED_AND_USED | Yes | `run-experiment`, `e2e`, `optimize`, `param-search` | [src/model/em.rs](src/model/em.rs) | Low |
| Diagnostics (regime ID, multi-start) | IMPLEMENTED_AND_USED | Yes | `run-experiment`, `optimize` | [src/model/diagnostics.rs](src/model/diagnostics.rs) | Low |
| Gaussian MSM simulator | IMPLEMENTED_AND_USED | Yes (synthetic backend) | `run-experiment`, `e2e` | [src/model/simulate.rs](src/model/simulate.rs) | Low |
| Frozen / fixed-parameter policy | IMPLEMENTED_AND_USED | Yes (`hard_switch_frozen` experiment) | `run-experiment` | [src/detector/frozen.rs](src/detector/frozen.rs) | Low |
| Online filter (1-step Hamilton) | IMPLEMENTED_AND_USED | Yes | all run paths | [src/online/mod.rs](src/online/mod.rs) | Low |
| Hard-switch detector | IMPLEMENTED_AND_USED | Yes | many experiments | [src/detector/hard_switch.rs](src/detector/hard_switch.rs) | Low |
| Posterior-transition detector (LP / TV) | IMPLEMENTED_AND_USED | Yes | `posterior_transition`, `posterior_transition_tv` | [src/detector/posterior_transition.rs](src/detector/posterior_transition.rs) | Low |
| Surprise detector (with EMA baseline) | IMPLEMENTED_AND_USED | Yes | `surprise`, `surprise_ema`, `squared_return_surprise` | [src/detector/surprise.rs](src/detector/surprise.rs) | Low |
| Benchmark protocol (truth, matching, metrics) | IMPLEMENTED_AND_USED | Yes (synthetic backend) | `run-experiment`, `run-batch` | [src/benchmark/](src/benchmark/) | Low |
| Real-data evaluation Route A (proxy events) | IMPLEMENTED_AND_USED | Yes | `run-real` | [src/real_eval/route_a.rs](src/real_eval/route_a.rs) | Low |
| Real-data evaluation Route B (segmentation) | PARTIAL | Yes — but `RealEvalSummaryRow` aggregator is dead | `run-real` | [src/real_eval/route_b.rs](src/real_eval/route_b.rs), [src/real_eval/report.rs](src/real_eval/report.rs) | Low |
| Calibration (synthetic ↔ real) | IMPLEMENTED_AND_USED | Yes | `calibrate` | [src/calibration/](src/calibration/) | Low |
| Experiment runner / backends | IMPLEMENTED_AND_USED | Yes | all | [src/experiments/runner.rs](src/experiments/runner.rs), [src/experiments/synthetic_backend.rs](src/experiments/synthetic_backend.rs), [src/experiments/real_backend.rs](src/experiments/real_backend.rs) | Low |
| Reporting — JSON/CSV/PNG/MD/TEX (inline) | IMPLEMENTED_AND_USED | Yes | all | [src/reporting/](src/reporting/) | Low |
| Reporting — `RunReporter` / `AggregateReporter` | IMPLEMENTED_NOT_USED | **No** | n/a | [src/reporting/report.rs](src/reporting/report.rs) | Low (cosmetic) |
| Reporting — typed CSV export module | IMPLEMENTED_NOT_USED | **No** — `#![allow(dead_code)]` | n/a | [src/reporting/export/csv.rs](src/reporting/export/csv.rs) | Low (cosmetic) |
| AlphaVantage client + DuckDB cache | IMPLEMENTED_AND_USED | Yes | `run-real`, `calibrate` | [src/alphavantage/](src/alphavantage/), [src/cache/mod.rs](src/cache/mod.rs) | Low |

**Bottom line:** All theory required by the thesis is implemented, wired into the CLI, and produces persisted artifacts under `runs/`. The only items not used at runtime are (a) `SessionAwareSeries` (intraday session labelling), (b) the typed `csv` export module, and (c) the `RunReporter`/`AggregateReporter` orchestration types in `src/reporting/report.rs`. None of these gate any thesis claim; (a) is the most relevant gap because the README/docs may imply session-aware processing of intraday SPY data.

---

## 2. Repository map

Grouped by module; only first-party `src/**/*.rs` files (≈80 files).

| Path | Role | Key types / functions | Notes |
|---|---|---|---|
| [src/main.rs](src/main.rs) | Binary entrypoint, dispatches direct vs interactive CLI | `is_direct_command`, `main` | Lists known subcommands |
| [src/config.rs](src/config.rs) | Top-level TOML config | `Config::from_file` | |
| [src/cli/mod.rs](src/cli/mod.rs) | Interactive (`inquire`) + direct subcommand routing, all command handlers | `run`, `run_direct`, `direct_run_experiment`, `direct_run_batch`, `direct_run_real`, `direct_calibrate`, `direct_compare_runs`, `direct_optimize`, `direct_inspect`, `direct_generate_report`, `cmd_e2e_run`, `cmd_param_search`, `cmd_calibrate_scenario`, `cmd_run_real_experiment`, `cmd_status` | Single 2400+ line module |
| [src/alphavantage/mod.rs](src/alphavantage/mod.rs) | Module root | re-exports | |
| [src/alphavantage/client.rs](src/alphavantage/client.rs) | REST client | `AlphaVantageClient` | |
| [src/alphavantage/commodity.rs](src/alphavantage/commodity.rs) | Commodity endpoints | `CommodityEndpoint`, `CommodityResponse` | |
| [src/alphavantage/rate_limiter.rs](src/alphavantage/rate_limiter.rs) | Token-bucket rate limiter | `RateLimiter` | |
| [src/cache/mod.rs](src/cache/mod.rs) | DuckDB persistence layer | `CommodityCache` | |
| [src/data/mod.rs](src/data/mod.rs) | Clean OHLCV → tidy series | `CleanSeries::from_response` | 4 tests |
| [src/data/meta.rs](src/data/meta.rs) | Dataset metadata | `DataMode`, `DataSource`, `PriceField`, `DatasetMeta` | |
| [src/data/session.rs](src/data/session.rs) | Intraday session labelling (RTH) | `SessionAwareSeries`, `is_rth_bar`, `label_sessions` | **Not wired into live code** |
| [src/data/split.rs](src/data/split.rs) | Train / verify / test partition | `PartitionedSeries::from_series`, `TimePartition`, `SplitConfig` | Used by [src/experiments/real_backend.rs](src/experiments/real_backend.rs) |
| [src/data/validation.rs](src/data/validation.rs) | Gap / NaN / monotonicity checks | `ValidationReport`, `validate` | |
| [src/data_service/mod.rs](src/data_service/mod.rs) | Façade combining client + cache | `DataService` | |
| [src/features/family.rs](src/features/family.rs) | Feature definitions | `FeatureFamily{LogReturn, AbsReturn, SquaredReturn, RollingVol, StandardizedReturn}`, `warmup_bars` | |
| [src/features/rolling.rs](src/features/rolling.rs) | Streaming O(w) ring buffer | `RollingStats` | population std |
| [src/features/scaler.rs](src/features/scaler.rs) | Train-only z-score / robust-z | `ScalingPolicy`, `FittedScaler` | |
| [src/features/transform.rs](src/features/transform.rs) | log / abs / squared returns | `log_return`, `abs_return`, `squared_return` | 11 tests |
| [src/features/stream.rs](src/features/stream.rs) | Pipeline glue | `FeatureStream::build`, `FeatureStreamMeta` | |
| [src/model/params.rs](src/model/params.rs) | $\theta=(\pi,A,\mu,\sigma^2)$ | `ModelParams::new`, `validate` | serde |
| [src/model/emission.rs](src/model/emission.rs) | Gaussian log-density | `Emission::log_density`, `HALF_LN_2PI` | 11 tests |
| [src/model/filter.rs](src/model/filter.rs) | Hamilton forward filter | `filter`, `FilterResult` | log-sum-exp; 14 tests |
| [src/model/smoother.rs](src/model/smoother.rs) | Kim backward smoother | `smooth`, `DENOM_FLOOR=1e-300` | renormalised per step |
| [src/model/pairwise.rs](src/model/pairwise.rs) | $\xi_t(i,j)$ + $\sum_t \xi_t$ | `pairwise`, `PairwiseResult` | 12 tests |
| [src/model/likelihood.rs](src/model/likelihood.rs) | $\log p(y_{1:T})$ + per-step contributions | `log_likelihood`, `log_likelihood_contributions` | 10 tests |
| [src/model/em.rs](src/model/em.rs) | Baum–Welch EM | `fit_em`, `EmConfig`, `EmResult`, floors `WEIGHT_FLOOR=1e-10`, `VAR_FLOOR=1e-6`, `MONOTONE_TOL=1e-8` | 15 tests |
| [src/model/diagnostics.rs](src/model/diagnostics.rs) | Regime ID, multi-start | `diagnose`, `compare_runs`, `FittedModelDiagnostics`, `MultiStartSummary` | 20 tests |
| [src/model/validation.rs](src/model/validation.rs) | Scenario validation A–H | scenario tests | |
| [src/model/simulate.rs](src/model/simulate.rs) | Gaussian MSM simulator | `simulate`, `simulate_with_jump`, `JumpParams`, `SimulationResult::changepoints` | 17 tests |
| [src/online/mod.rs](src/online/mod.rs) | 1-step Hamilton update | `OnlineFilterState::step`, `OnlineStepResult`, `cumulative_log_score` | 15 tests |
| [src/detector/mod.rs](src/detector/mod.rs) | Detector trait + types | `Detector`, `DetectorKind{HardSwitch, PosteriorTransition, Surprise}`, `AlarmEvent`, `PersistencePolicy` | 6 tests |
| [src/detector/frozen.rs](src/detector/frozen.rs) | Frozen-parameter runtime | `FrozenModel`, `StreamingSession<D>` | 8 tests |
| [src/detector/hard_switch.rs](src/detector/hard_switch.rs) | argmax-jump detector | `HardSwitchDetector` | 5 tests |
| [src/detector/posterior_transition.rs](src/detector/posterior_transition.rs) | LeavePrevious + TotalVariation | `PosteriorTransitionDetector`, `PosteriorTransitionVariant` | 2+ tests |
| [src/detector/surprise.rs](src/detector/surprise.rs) | $-\log c_t$ with EMA baseline | `SurpriseDetector`, `BaselineMode` | 3+ tests |
| [src/benchmark/matching.rs](src/benchmark/matching.rs) | TP/FP/FN matching | `EventMatcher`, `MatchConfig` | 11 tests |
| [src/benchmark/metrics.rs](src/benchmark/metrics.rs) | Precision/recall/F1/delay | `MetricSuite`, `compute` | 10 tests |
| [src/benchmark/result.rs](src/benchmark/result.rs) | Result records | `BenchmarkResult` | |
| [src/benchmark/truth.rs](src/benchmark/truth.rs) | Truth vector | `ChangePointTruth` | |
| [src/real_eval/route_a.rs](src/real_eval/route_a.rs) | Proxy-event alignment | `RouteAConfig`, `ProxyEvent`, `evaluate_route_a` | 6 tests |
| [src/real_eval/route_b.rs](src/real_eval/route_b.rs) | Segmentation self-consistency | `RouteBConfig`, `ShortSegmentPolicy`, `evaluate_route_b` | 4 tests |
| [src/real_eval/report.rs](src/real_eval/report.rs) | Glue + JSON | `evaluate_real_data`, `RealEvalResult`, `RealEvalMeta`, `RealEvalSummaryRow` | `RealEvalSummaryRow` unused |
| [src/calibration/mapping.rs](src/calibration/mapping.rs) | Empirical → MSM mapping | `MeanPolicy`, `VariancePolicy`, $p_{jj}=1-1/d_j$ | 4 tests |
| [src/calibration/summary.rs](src/calibration/summary.rs) | Empirical statistics | `EmpiricalCalibrationProfile` | 5 tests |
| [src/calibration/report.rs](src/calibration/report.rs) | Calibration report | `CalibrationReport` | 2 tests |
| [src/calibration/verify.rs](src/calibration/verify.rs) | Round-trip verification | `verify_round_trip` | 2 tests |
| [src/experiments/registry.rs](src/experiments/registry.rs) | Registered scenarios | 12 entries (see §4 below) | |
| [src/experiments/runner.rs](src/experiments/runner.rs) | Stage pipeline + bundles | `ExperimentRunner<B>`, `RunStage`, `DataBundle`, `FeatureBundle`, `ModelArtifact`, `OnlineRunArtifact`, `SyntheticEvalArtifact`, `RealEvalArtifact` | |
| [src/experiments/synthetic_backend.rs](src/experiments/synthetic_backend.rs) | Synthetic `ExperimentBackend` impl | `SyntheticBackend`, `resolve_data`, `evaluate_synthetic` | uses `simulate`, `EventMatcher`, `ChangePointTruth` |
| [src/experiments/real_backend.rs](src/experiments/real_backend.rs) | Real `ExperimentBackend` impl | `RealBackend`, `load_clean_series`, `evaluate_real` | uses `CommodityCache`, `PartitionedSeries`, `RouteA/RouteB` |
| [src/reporting/artifact.rs](src/reporting/artifact.rs) | Run-directory layout | `RunArtifactLayout` | |
| [src/reporting/export/json.rs](src/reporting/export/json.rs) | Typed JSON export | used by runner | |
| [src/reporting/export/csv.rs](src/reporting/export/csv.rs) | Typed CSV export module | `#![allow(dead_code)]` — **unused** | live runs use inline `csv` writes |
| [src/reporting/plot/*.rs](src/reporting/plot/) | Plotters PNG renderers | `signal_alarms`, `detector_scores`, `regime_posteriors`, `segmentation`, `delay_distribution` | wired via `generate_plots` |
| [src/reporting/table/*.rs](src/reporting/table/) | MD/TEX tables | `metrics`, `comparison`, `segment_summary` | |
| [src/reporting/report.rs](src/reporting/report.rs) | High-level orchestration | `RunReporter`, `AggregateReporter` | **unused** |

---

## 3. End-to-end flow map

The single live pipeline is the `RunStage` ladder driven by `ExperimentRunner<B>` over the `ExperimentBackend` trait. Stages are identical for synthetic and real backends; only stage implementations differ.

| # | Stage | Backend code path | Functions / types | Confirmed used? | Evidence (artifact written) |
|---|---|---|---|---|---|
| 1 | CLI dispatch | `main.rs::is_direct_command` → `cli::run_direct` / `cli::run` | `direct_*`, `cmd_*` | Yes | terminal output, `runs/**/run_*` directories created |
| 2 | Resolve data (synthetic) | `SyntheticBackend::resolve_data` → `model::simulate::simulate` | `SimulationResult`, `changepoints()` | Yes | `result.json` contains simulated truth |
| 2′ | Resolve data (real) | `RealBackend::load_clean_series` → `CommodityCache` → `CleanSeries::from_response` → `PartitionedSeries::from_series` → `validate` | `DataBundle{ split_summary_json, validation_report_json }` | Yes | Validation report embedded in run JSON |
| 3 | Build features | `RealBackend::build_features` → `FeatureStream::build` (synthetic backend = passthrough) | `FeatureBundle`, `FeatureStreamMeta` | Yes | `feature_summary.json` |
| 4 | Train / load model | `train_or_load_model` → `model::em::fit_em` (uses `filter`, `smoother`, `pairwise`, `likelihood`) | `ModelArtifact{params, fit_summary, loglikelihood_history, diagnostics}` | Yes | `model_params.json`, `fit_summary.json`, `diagnostics.json`, `loglikelihood_history.csv` |
| 4′ | Train (frozen) | `FrozenModel::load` (used by `hard_switch_frozen`) | `FrozenModel{params}` | Yes | `data/frozen_models/hard_switch_frozen/` referenced |
| 5 | Run online | `run_online` → `online::OnlineFilterState::step` per bar; `Detector::step` per bar via `StreamingSession<D>` | `OnlineRunArtifact{posteriors, scores, alarms}` | Yes | `regime_posteriors.{csv,png}`, `score_trace.csv`, `detector_scores.png`, `alarms.csv`, `signal_alarms.png` |
| 6 | Evaluate (synthetic) | `evaluate_synthetic` → `ChangePointTruth::new` → `EventMatcher::match_events` → `MetricSuite::compute` | `SyntheticEvalArtifact{metrics, matches, delays}` | Yes | `metrics.{csv,md,tex}`, `delay_distribution.png`, `changepoints.csv` |
| 6′ | Evaluate (real) | `evaluate_real` → `real_eval::report::evaluate_real_data` → Route A (`evaluate_route_a`) and Route B (`evaluate_route_b`) | `RealEvalArtifact{route_a, route_b}` | Yes | `signal_alarms.png`, `segmentation.png`, JSON in `result.json` |
| 7 | Export | `RunArtifactLayout::write_*`, `reporting::export::json::write`, plotters renderers, table renderers | files under `runs/<kind>/<experiment>/run_<hash>_<seed>/` | Yes | full file set verified at `runs/synthetic/hard_switch/run_b0348319eb7e2e7f_42/` |

**Verified artifact set** for a typical synthetic run (`runs/synthetic/hard_switch/run_b0348319eb7e2e7f_42/`):

```
alarms.csv                  feature_summary.json   model_params.json       result.json
changepoints.csv            fit_summary.json       regime_posteriors.csv   score_trace.csv
config.snapshot.json        loglikelihood_history.csv  regime_posteriors.png  signal_alarms.png
delay_distribution.png      metrics.csv            summary.json
detector_config.json        metrics.md
detector_scores.png         metrics.tex
diagnostics.json
```

For a typical real run (`runs/real/real_spy_daily_hard_switch/run_549cd0a25ac79a7a_19deea24de9/`): `config.snapshot.json`, `segmentation.png`, `signal_alarms.png` (Route A/B JSON is embedded in `result.json` of richer runs; some short real runs only persist plots + config snapshot).

**Aggregator outputs** (e.g. `runs/batch_summary.json`, `runs/metrics_table.md`) confirm the cross-run roll-up path is invoked from `run-batch` / `compare-runs`.

---

## 4. Theory-to-code audit

The fixed template per component:
- **Theoretical object**
- **Expected equations / invariants**
- **Code locations** (file, type, function)
- **Key types**
- **How used** (which stage, which CLI command, which artifact)
- **End-to-end reachability**
- **Artifacts emitted**
- **Tests**
- **Status**
- **Gaps / risks**
- **Thesis note**

### 4.1 Markov-switching state-space model — parameters $\theta=(\pi,A,\mu,\sigma^2)$

- **Theoretical object.** Initial distribution $\pi\in\Delta^{K-1}$, row-stochastic transition matrix $A\in\mathbb{R}^{K\times K}$, regime means $\mu\in\mathbb{R}^K$, variances $\sigma^2\in\mathbb{R}_+^K$.
- **Expected equations.** $\sum_i \pi_i=1$; $\sum_j A_{ij}=1$; $\sigma^2_k>0$.
- **Code locations.** [src/model/params.rs](src/model/params.rs).
- **Key types.** `ModelParams { pi, transition, means, variances }`, `ModelParams::new(...)`, `ModelParams::validate()`, serde-derived.
- **How used.** Constructed in `fit_em` output, in `simulate`, in `FrozenModel`, in synthetic-backend ground truth. Persisted to `model_params.json`.
- **E2E reachability.** Yes — every CLI subcommand that runs an experiment writes `model_params.json`.
- **Artifacts.** `model_params.json`, `config.snapshot.json` (echoes init params).
- **Tests.** `validate()` rejects malformed simplex / non-stochastic rows / non-positive variances (in-file tests).
- **Status.** **IMPLEMENTED_AND_USED**.
- **Gaps.** None.
- **Thesis note.** Cite as the canonical $\theta$ container; no separate label-permutation invariant lives here — that lives in `model::diagnostics` (§4.10).

### 4.2 Gaussian emission $p(y_t \mid s_t=k)$

- **Theoretical object.** $y_t\mid s_t=k\sim\mathcal{N}(\mu_k,\sigma_k^2)$.
- **Expected equations.** $\log p(y\mid k)=-\tfrac12\ln(2\pi\sigma_k^2)-\tfrac{(y-\mu_k)^2}{2\sigma_k^2}$.
- **Code locations.** [src/model/emission.rs](src/model/emission.rs).
- **Key types.** `Emission`, `Emission::log_density(y, k, params)`, const `HALF_LN_2PI`.
- **How used.** Called inside `model::filter::filter`, `model::likelihood`, `online::OnlineFilterState::step`.
- **E2E reachability.** Yes.
- **Artifacts.** Indirectly via posteriors and likelihood outputs.
- **Tests.** 11 unit tests (closed-form checks at boundary variance, large $|y|$, etc.).
- **Status.** **IMPLEMENTED_AND_USED**.
- **Gaps.** Only Gaussian (univariate). Thesis correctly scopes to univariate Gaussian MSM.
- **Thesis note.** Cite `Emission::log_density` as the operational form of equation (Gaussian density).

### 4.3 Gaussian MSM simulator (synthetic data)

- **Theoretical object.** Sample $(s_{1:T},y_{1:T})$ from $\theta$.
- **Expected equations.** $s_1\sim\pi$, $s_t\mid s_{t-1}\sim A_{s_{t-1},\cdot}$, $y_t\sim\mathcal{N}(\mu_{s_t},\sigma^2_{s_t})$.
- **Code locations.** [src/model/simulate.rs](src/model/simulate.rs).
- **Key types.** `simulate(params, T, rng)`, `simulate_with_jump(params, T, JumpParams, rng)`, `SimulationResult { states, observations }`, `SimulationResult::changepoints()`, `JumpParams`.
- **How used.** `SyntheticBackend::resolve_data` (stage 2). Drives every synthetic experiment.
- **E2E reachability.** Yes — `run-experiment` synthetic kinds + `e2e`.
- **Artifacts.** `changepoints.csv` (truth derived from simulated states), `result.json`.
- **Tests.** 17 unit tests (state-distribution convergence, deterministic seeding, jump injection).
- **Status.** **IMPLEMENTED_AND_USED**.
- **Gaps.** None for current scope.
- **Thesis note.** This is the synthetic data-generating process used in all benchmark sections.

### 4.4 Forward (Hamilton) filter

- **Theoretical object.** $p(s_t\mid y_{1:t})$ via predict / update / normalise.
- **Expected equations.** $\hat\alpha_{t|t-1}=A^\top \hat\alpha_{t-1|t-1}$; $\hat\alpha_{t|t}\propto \hat\alpha_{t|t-1}\odot p(y_t\mid s_t)$; $\log p(y_{1:T})=\sum_t \log c_t$.
- **Code locations.** [src/model/filter.rs](src/model/filter.rs).
- **Key types.** `filter(obs, params) -> FilterResult { predicted, filtered, log_predictive, log_likelihood }`. Internally uses log-sum-exp.
- **How used.** Called by `fit_em` (E-step), by `model::likelihood`, by `model::diagnostics`, and (in 1-step form) by `online::OnlineFilterState::step`.
- **E2E reachability.** Yes — every fit and every online run.
- **Artifacts.** `regime_posteriors.csv` (filtered $\hat\alpha_{t|t}$), `loglikelihood_history.csv` (training).
- **Tests.** 14 unit tests (numerical stability, conjugacy with smoother, monotone log-lik).
- **Status.** **IMPLEMENTED_AND_USED**.
- **Gaps.** None.
- **Thesis note.** `FilterResult.log_predictive` exposes per-step $\log c_t$, which feeds the surprise detector (§4.16).

### 4.5 Backward (Kim) smoother

- **Theoretical object.** $p(s_t\mid y_{1:T})$.
- **Expected equations.** $\hat\alpha_{t|T,k}=\hat\alpha_{t|t,k}\sum_j \frac{A_{kj}\hat\alpha_{t+1|T,j}}{[A^\top \hat\alpha_{t|t}]_j}$ with safeguards.
- **Code locations.** [src/model/smoother.rs](src/model/smoother.rs); `DENOM_FLOOR=1e-300`; per-step renormalisation.
- **Key types.** `smooth(filter_result, params) -> Vec<[f64; K]>`.
- **How used.** Called by `fit_em` E-step (consumed by `pairwise`).
- **E2E reachability.** Yes (training only).
- **Artifacts.** Smoothed posteriors are not exported as a separate file by default — they feed the M-step.
- **Tests.** Conjugacy with `filter`, marginal-consistency under stationary chains, finite-difference vs. brute-force on $K=2,T\le 5$.
- **Status.** **IMPLEMENTED_AND_USED**.
- **Gaps.** Smoothed posteriors not exported as artifact (only the online filtered posteriors are written). Acceptable, but could be added under a flag.
- **Thesis note.** Documented in [docs/backward_smoother.md](docs/backward_smoother.md).

### 4.6 Pairwise posteriors $\xi_t(i,j)$ and expected transitions

- **Theoretical object.** $\xi_t(i,j)=p(s_t=i,s_{t+1}=j\mid y_{1:T})$ and $\sum_t \xi_t$.
- **Expected equations.** Marginal consistency $\sum_j \xi_t(i,j)=\hat\alpha_{t|T,i}$.
- **Code locations.** [src/model/pairwise.rs](src/model/pairwise.rs).
- **Key types.** `pairwise(filter_result, smoothed, params) -> PairwiseResult { xi: Vec<[[f64;K];K]>, expected_transitions: [[f64;K];K] }`.
- **How used.** EM M-step uses `expected_transitions` to update $A$.
- **E2E reachability.** Yes (training).
- **Artifacts.** Indirect via `model_params.json` (updated $A$).
- **Tests.** 12 unit tests including marginal-consistency invariants.
- **Status.** **IMPLEMENTED_AND_USED**.
- **Gaps.** None.
- **Thesis note.** Documented in [docs/pairwise_posteriors.md](docs/pairwise_posteriors.md).

### 4.7 Log-likelihood

- **Theoretical object.** $\log p(y_{1:T}\mid\theta)=\sum_t \log c_t$.
- **Code locations.** [src/model/likelihood.rs](src/model/likelihood.rs).
- **Key types.** `log_likelihood(obs, params)`, `log_likelihood_contributions(obs, params)`.
- **How used.** EM monotonicity check (`MONOTONE_TOL=1e-8`); diagnostic comparison; fed to optimiser in `cmd_param_search` / `direct_optimize`.
- **E2E reachability.** Yes.
- **Artifacts.** `loglikelihood_history.csv`, `fit_summary.json`.
- **Tests.** 10 unit tests.
- **Status.** **IMPLEMENTED_AND_USED**.
- **Gaps.** None.
- **Thesis note.** Cite the EM convergence section against this monotonicity guard.

### 4.8 EM (Baum–Welch) estimation

- **Theoretical object.** Iterative $\theta^{(r+1)}=\arg\max_\theta Q(\theta\mid\theta^{(r)})$.
- **Code locations.** [src/model/em.rs](src/model/em.rs).
- **Key types.** `fit_em(obs, init_params, EmConfig) -> EmResult`. Floors: `WEIGHT_FLOOR=1e-10`, `VAR_FLOOR=1e-6`. M-step closed-form for Gaussian emissions.
- **How used.** Stage 4 in synthetic and real pipelines (when `train` mode); also driven by multi-start in `direct_optimize`.
- **E2E reachability.** Yes.
- **Artifacts.** `fit_summary.json`, `loglikelihood_history.csv`, `model_params.json`.
- **Tests.** 15 unit tests (monotonicity, recovery of true $\theta$ on synthetic, behaviour at variance floor).
- **Status.** **IMPLEMENTED_AND_USED**.
- **Gaps.** None.
- **Thesis note.** [docs/em_estimation.md](docs/em_estimation.md) describes the implementation.

### 4.9 Multi-start / fit diagnostics

- **Theoretical object.** Identifiability under label-switching; basin discovery.
- **Code locations.** [src/model/diagnostics.rs](src/model/diagnostics.rs).
- **Key types.** `diagnose`, `compare_runs`, `FittedModelDiagnostics`, `MultiStartSummary`. Reports expected duration $1/(1-p_{jj})$.
- **How used.** Called from EM result builder; multi-start aggregation in `direct_optimize`.
- **E2E reachability.** Yes.
- **Artifacts.** `diagnostics.json`.
- **Tests.** 20 unit tests.
- **Status.** **IMPLEMENTED_AND_USED**.
- **Gaps.** None.
- **Thesis note.** Use to discuss identifiability / label-switching mitigation.

### 4.10 Frozen / fixed-parameter policy

- **Theoretical object.** Use a fitted $\theta^*$ unchanged across the test stream (no re-estimation online).
- **Code locations.** [src/detector/frozen.rs](src/detector/frozen.rs).
- **Key types.** `FrozenModel { params }`, `StreamingSession<D: Detector>`. Persisted under `data/frozen_models/hard_switch_frozen/`.
- **How used.** Loaded by experiment kinds `hard_switch_frozen` and (by reference) other frozen variants.
- **E2E reachability.** Yes.
- **Artifacts.** Reuses standard run artifacts; `model_params.json` reflects the frozen $\theta$.
- **Tests.** 8 unit tests.
- **Status.** **IMPLEMENTED_AND_USED**.
- **Gaps.** Some accessor methods on `FrozenModel` are exercised only by tests, not by live calls; not a correctness issue.
- **Thesis note.** Documented in [docs/fixed_parameter_policy.md](docs/fixed_parameter_policy.md).

### 4.11 Online filter (1-step Hamilton update)

- **Theoretical object.** Streaming $p(s_t\mid y_{1:t})$ given fixed $\theta$.
- **Code locations.** [src/online/mod.rs](src/online/mod.rs).
- **Key types.** `OnlineFilterState::step(y, params) -> OnlineStepResult { posterior, log_predictive }`, `cumulative_log_score`. Implements the 4-step recursion (predict, emission, joint, normalise).
- **How used.** Stage 5 for every run (synthetic and real).
- **E2E reachability.** Yes.
- **Artifacts.** `regime_posteriors.csv`, `score_trace.csv`.
- **Tests.** 15 unit tests (matches batch `filter` to $\le 10^{-12}$).
- **Status.** **IMPLEMENTED_AND_USED**.
- **Gaps.** None.
- **Thesis note.** [docs/online_inference.md](docs/online_inference.md).

### 4.12 Detector trait abstraction

- **Theoretical object.** Detection rule mapping $(\hat\alpha_t,\log c_t)\to s_t\in\{0,1\}$.
- **Code locations.** [src/detector/mod.rs](src/detector/mod.rs).
- **Key types.** `Detector` trait, `DetectorKind`, `AlarmEvent`, `PersistencePolicy`.
- **How used.** Implemented by the three detectors below; consumed by `StreamingSession<D>`.
- **E2E reachability.** Yes.
- **Tests.** 6 unit tests.
- **Status.** **IMPLEMENTED_AND_USED**.

### 4.13 Hard-switch detector

- **Theoretical object.** Alarm at time $t$ iff $\arg\max_k \hat\alpha_{t,k}\neq \arg\max_k \hat\alpha_{t-1,k}$, with confidence threshold and persistence.
- **Code locations.** [src/detector/hard_switch.rs](src/detector/hard_switch.rs).
- **Key types.** `HardSwitchDetector`.
- **How used.** Experiments: `hard_switch`, `hard_switch_shock`, `hard_switch_frozen`, `hard_switch_multi_start`, `real_spy_daily_hard_switch`, `real_spy_intraday_hard_switch`.
- **Artifacts.** `alarms.csv`, `signal_alarms.png`.
- **Tests.** 5 unit tests.
- **Status.** **IMPLEMENTED_AND_USED**.

### 4.14 Posterior-transition detector (LP and TV variants)

- **Theoretical object.** Variant `LeavePrevious`: $1-\alpha_{s_{t-1}^*}(t)$. Variant `TotalVariation`: $\tfrac12\sum_k|\hat\alpha_{t,k}-\hat\alpha_{t-1,k}|$.
- **Code locations.** [src/detector/posterior_transition.rs](src/detector/posterior_transition.rs).
- **Key types.** `PosteriorTransitionDetector`, `PosteriorTransitionVariant::{LeavePrevious, TotalVariation}`.
- **How used.** Experiments: `posterior_transition`, `posterior_transition_tv`.
- **Artifacts.** `score_trace.csv`, `detector_scores.png`, `alarms.csv`.
- **Tests.** ≥2 unit tests covering both variants.
- **Status.** **IMPLEMENTED_AND_USED**.

### 4.15 Surprise detector (with optional EMA baseline)

- **Theoretical object.** Score $-\log c_t$ where $c_t=p(y_t\mid y_{1:t-1})$, optionally relative to an EMA baseline $-\log c_t - \mathrm{EMA}(-\log c_{1:t-1})$.
- **Code locations.** [src/detector/surprise.rs](src/detector/surprise.rs).
- **Key types.** `SurpriseDetector`, `BaselineMode { None, Ema { alpha } }`.
- **How used.** Experiments: `surprise`, `surprise_ema`, `squared_return_surprise`, `real_wti_daily_surprise`.
- **Artifacts.** `score_trace.csv`, `detector_scores.png`, `alarms.csv`.
- **Tests.** ≥3 unit tests.
- **Status.** **IMPLEMENTED_AND_USED**.

### 4.16 Benchmark protocol (truth, matching, metrics)

- **Theoretical object.** TP/FP/FN matching of detected alarms against ground-truth change points within a window; precision, recall, F1, mean and median detection delay.
- **Code locations.** [src/benchmark/truth.rs](src/benchmark/truth.rs), [src/benchmark/matching.rs](src/benchmark/matching.rs), [src/benchmark/metrics.rs](src/benchmark/metrics.rs), [src/benchmark/result.rs](src/benchmark/result.rs).
- **Key types.** `ChangePointTruth`, `EventMatcher`, `MatchConfig { window }`, `MetricSuite::compute`.
- **How used.** `SyntheticBackend::evaluate_synthetic` (stage 6).
- **Artifacts.** `metrics.{csv,md,tex}`, `delay_distribution.png`, `changepoints.csv`.
- **Tests.** 11 (matching) + 10 (metrics) unit tests.
- **Status.** **IMPLEMENTED_AND_USED**.
- **Thesis note.** [docs/benchmark_protocol.md](docs/benchmark_protocol.md).

### 4.17 Real-data evaluation — Route A (proxy events) and Route B (segmentation)

- **Theoretical object.**
  - Route A: distance-based alignment between detector alarms and external proxy events $E\subset\{1,\dots,T\}$ (e.g. macro releases). Reports precision/recall/F1 + delay vs. window.
  - Route B: segmentation self-consistency. Detector alarms partition $y_{1:T}$ into segments; report between- vs. within-segment statistics, with short-segment merging.
- **Code locations.** [src/real_eval/route_a.rs](src/real_eval/route_a.rs), [src/real_eval/route_b.rs](src/real_eval/route_b.rs), [src/real_eval/report.rs](src/real_eval/report.rs).
- **Key types.** `RouteAConfig`, `ProxyEvent`, `PointMatchPolicy`; `RouteBConfig`, `ShortSegmentPolicy`; `evaluate_real_data`, `RealEvalResult`, `RealEvalMeta`.
- **How used.** `RealBackend::evaluate_real` (stage 6′). Driven by `run-real`.
- **Artifacts.** `signal_alarms.png`, `segmentation.png`, embedded JSON in `result.json`. Proxy events read from `data/proxy_events/{spy,gold,wti}.json`.
- **Tests.** 6 (Route A) + 4 (Route B) unit tests.
- **Status.** **PARTIAL** — main path is fully wired and tested; the `RealEvalSummaryRow` aggregator type in `report.rs` has no live caller (cross-asset summary currently produced inline by the runner / `compare-runs`).
- **Gaps.** Wire `RealEvalSummaryRow` into `compare-runs`, or remove it.
- **Thesis note.** [docs/real_data_evaluation.md](docs/real_data_evaluation.md) covers the design.

### 4.18 Calibration (synthetic ↔ real)

- **Theoretical object.** Map empirical statistics of a real series to MSM parameters (e.g. $p_{jj}=1-1/d_j$ from average regime duration $d_j$); verify by simulating from the calibrated $\theta$ and re-fitting.
- **Code locations.** [src/calibration/mapping.rs](src/calibration/mapping.rs), [src/calibration/summary.rs](src/calibration/summary.rs), [src/calibration/report.rs](src/calibration/report.rs), [src/calibration/verify.rs](src/calibration/verify.rs).
- **Key types.** `MeanPolicy`, `VariancePolicy`, `EmpiricalCalibrationProfile`, `CalibrationReport`, `verify_round_trip`.
- **How used.** `direct_calibrate` / `cmd_calibrate_scenario`. Output frozen models live under `data/frozen_models/`.
- **Artifacts.** Calibration reports written under `runs/optimize/` and consumed by frozen-model experiments.
- **Tests.** 4 + 5 + 2 + 2 = 13 unit tests across the four files.
- **Status.** **IMPLEMENTED_AND_USED**.
- **Thesis note.** [docs/synthetic_to_real_calibration.md](docs/synthetic_to_real_calibration.md).

---

## 5. Usage verification (per component)

For each component, "Used by" lists the concrete call sites; "CLI reachable via" lists the user-facing subcommand(s); "Artifacts" lists the on-disk evidence.

| Component | Used by | CLI reachable via | Artifacts |
|---|---|---|---|
| `model::params::ModelParams` | `simulate`, `fit_em`, `FrozenModel`, `OnlineFilterState::step`, all detectors | every subcommand | `model_params.json` |
| `model::emission::log_density` | `filter`, `likelihood`, `online::step` | every subcommand | indirect |
| `model::filter::filter` | `fit_em` E-step, `likelihood`, `diagnostics`, batch eval | `run-experiment`, `optimize`, `param-search` | `regime_posteriors.csv` (offline path) |
| `model::smoother::smooth` | `fit_em` E-step | training paths | indirect (via $\theta$ updates) |
| `model::pairwise::pairwise` | `fit_em` E-step | training paths | indirect |
| `model::likelihood::log_likelihood` | `fit_em` monotonicity, `compare_runs`, optimiser | `run-experiment`, `optimize`, `param-search` | `loglikelihood_history.csv`, `fit_summary.json` |
| `model::em::fit_em` | runner stage 4 | `run-experiment`, `e2e`, `optimize`, `param-search` | `fit_summary.json`, `model_params.json`, `diagnostics.json` |
| `model::diagnostics::diagnose` / `compare_runs` | runner stage 4, multi-start | `run-experiment`, `optimize` | `diagnostics.json` |
| `model::simulate::simulate` | `SyntheticBackend::resolve_data` | synthetic experiments | `changepoints.csv`, `result.json` |
| `online::OnlineFilterState::step` | runner stage 5 (every backend) | every run | `regime_posteriors.csv`, `score_trace.csv` |
| `detector::frozen::FrozenModel` + `StreamingSession` | `hard_switch_frozen`, frozen variants | `run-experiment` (kind=frozen) | standard run artifacts |
| `detector::hard_switch::HardSwitchDetector` | 6 experiment kinds | `run-experiment`, `run-real` | `alarms.csv`, `signal_alarms.png` |
| `detector::posterior_transition::*` | 2 experiment kinds | `run-experiment` | `score_trace.csv`, `detector_scores.png` |
| `detector::surprise::*` | 4 experiment kinds | `run-experiment`, `run-real` | `score_trace.csv`, `detector_scores.png` |
| `benchmark::matching` + `benchmark::metrics` + `benchmark::truth` | `SyntheticBackend::evaluate_synthetic` | synthetic experiments, `run-batch` | `metrics.{csv,md,tex}`, `delay_distribution.png` |
| `real_eval::route_a::evaluate_route_a` | `RealBackend::evaluate_real` | `run-real` | `signal_alarms.png`, `result.json` |
| `real_eval::route_b::evaluate_route_b` | `RealBackend::evaluate_real` | `run-real` | `segmentation.png`, `result.json` |
| `real_eval::report::RealEvalSummaryRow` | **no live caller** | n/a | none (dead) |
| `calibration::*` | `cmd_calibrate_scenario`, `direct_calibrate` | `calibrate` | calibration JSON + frozen model dir |
| `data::CleanSeries`, `PartitionedSeries`, `validate` | `RealBackend::load_clean_series` | `run-real`, `calibrate` | embedded in `result.json` |
| `data::session::SessionAwareSeries` | **no live caller** | n/a | none (dead) |
| `features::FeatureStream::build` | `RealBackend::build_features` | `run-real`, `run-experiment` (real) | `feature_summary.json` |
| `reporting::artifact::RunArtifactLayout` | runner export | every run | run directory |
| `reporting::export::json` | runner export | every run | `*.json` files |
| `reporting::export::csv` (typed module) | **no live caller** (`#![allow(dead_code)]`) | n/a | none |
| `reporting::plot::*` (signal/scores/posteriors/segmentation/delay) | `generate_plots` | every run | `*.png` files |
| `reporting::table::*` | runner export | every run | `metrics.{md,tex}` |
| `reporting::report::RunReporter` / `AggregateReporter` | **no live caller** | n/a | none (dead) |

---

## 6. Dead-code and duplicate audit

| Item | File | Symptom | Action |
|---|---|---|---|
| Typed CSV export module | [src/reporting/export/csv.rs](src/reporting/export/csv.rs) | `#![allow(dead_code)]`; live runner uses inline `csv` calls inside `RunArtifactLayout` | Either wire `RunArtifactLayout` to delegate to this module (consolidates schema in one place) or delete the file. |
| `RunReporter`, `AggregateReporter` | [src/reporting/report.rs](src/reporting/report.rs) | Public types with no live constructors outside their own tests | Either replace runner export call-sites with `RunReporter`, or delete `report.rs`. |
| `RealEvalSummaryRow` | [src/real_eval/report.rs](src/real_eval/report.rs) | Defined and serialised in tests, never aggregated by `compare-runs` / `run-batch` | Wire into `cli::direct_compare_runs` aggregator or delete. |
| `SessionAwareSeries`, `is_rth_bar`, `label_sessions` | [src/data/session.rs](src/data/session.rs) | Module exists with 7 unit tests but no backend constructs `SessionAwareSeries`; intraday SPY currently flows through plain `CleanSeries` + `PartitionedSeries` | **Most important to address:** wire into `RealBackend::load_clean_series` for `Intraday` mode (gates session_reset semantics in `FeatureFamily::RollingVol/StandardizedReturn`), or downgrade thesis claim to "session-naive intraday processing". |
| `FrozenModel` accessor methods | [src/detector/frozen.rs](src/detector/frozen.rs) | A few read-only accessors only exercised by tests | Cosmetic; safe to leave. |

No genuine duplicates were found (e.g. there is exactly one Hamilton recursion, one EM, one matcher).

---

## 7. Documentation gaps

| Topic | Doc file present? | Code aligned? | Notes |
|---|---|---|---|
| AlphaVantage client | [docs/alphavantage_client.md](docs/alphavantage_client.md) | Yes | |
| Backward smoother | [docs/backward_smoother.md](docs/backward_smoother.md) | Yes | |
| Benchmark protocol | [docs/benchmark_protocol.md](docs/benchmark_protocol.md) | Yes | |
| Change-point detectors | [docs/changepoint_detectors.md](docs/changepoint_detectors.md) | Yes | covers all three detectors |
| Data pipeline | [docs/data_pipeline.md](docs/data_pipeline.md) | **Partially** | Should explicitly note that `SessionAwareSeries` is **not** invoked in the live pipeline. |
| Data service | [docs/data_service.md](docs/data_service.md) | Yes | |
| Diagnostics | [docs/diagnostics.md](docs/diagnostics.md) | Yes | |
| DuckDB cache | [docs/duckdb_cache.md](docs/duckdb_cache.md) | Yes | |
| EM estimation | [docs/em_estimation.md](docs/em_estimation.md) | Yes | |
| Emission model | [docs/emission_model.md](docs/emission_model.md) | Yes | |
| Experiment runner | [docs/experiment_runner.md](docs/experiment_runner.md) | Yes | should list current 12 registry entries explicitly |
| Filter validation | [docs/filter_validation.md](docs/filter_validation.md) | Yes | |
| Fixed-parameter policy | [docs/fixed_parameter_policy.md](docs/fixed_parameter_policy.md) | Yes | |
| Forward filter | [docs/forward_filter.md](docs/forward_filter.md) | Yes | |
| Gaussian MSM simulator | [docs/gaussian_msm_simulator.md](docs/gaussian_msm_simulator.md) | Yes | |
| Interactive CLI | [docs/interactive_cli.md](docs/interactive_cli.md) | Yes | should mention `direct_*` companions |
| Log-likelihood | [docs/log_likelihood.md](docs/log_likelihood.md) | Yes | |
| Observation design | [docs/observation_design.md](docs/observation_design.md) | Yes | |
| Online inference | [docs/online_inference.md](docs/online_inference.md) | Yes | |
| Pairwise posteriors | [docs/pairwise_posteriors.md](docs/pairwise_posteriors.md) | Yes | |
| Real data evaluation | [docs/real_data_evaluation.md](docs/real_data_evaluation.md) | Yes | should note that `RealEvalSummaryRow` is currently unused. |
| Reporting / export | [docs/reporting_and_export.md](docs/reporting_and_export.md) | **Partially** | Should disclose that the typed `csv.rs` module and `RunReporter`/`AggregateReporter` are not on the active path. |
| Synthetic ↔ real calibration | [docs/synthetic_to_real_calibration.md](docs/synthetic_to_real_calibration.md) | Yes | |
| **Missing**: detector trait + `DetectorKind` registry | — | — | Add a short doc cross-referencing [src/detector/mod.rs](src/detector/mod.rs). |
| **Missing**: dataset partitioning (`SplitConfig`, `TimePartition`) | — | — | Add a short doc; currently only inline rustdoc. |
| **Missing**: features (family / scaler / stream) | — | — | Add a doc for the feature stack as it appears in [src/features/](src/features/). |

---

## 8. Test coverage gaps

| Area | Tests today | Gap |
|---|---|---|
| `model::filter` | 14 | None — strong coverage. |
| `model::smoother` | conjugacy + finite-diff | None. |
| `model::pairwise` | 12 | None. |
| `model::em` | 15 | None. |
| `model::diagnostics` | 20 | None. |
| `model::simulate` | 17 | None. |
| `online` | 15 | None. |
| `detector::*` | 5 + 2 + 3 + 8 (frozen) | OK — could add a property-based test that all detectors are time-shift invariant on stationary segments. |
| `benchmark::*` | 11 + 10 | None. |
| `real_eval::route_a` | 6 | OK. |
| `real_eval::route_b` | 4 | Add a test exercising `ShortSegmentPolicy::Merge` end-to-end with a real fixture. |
| `data::session` | 7 | Tests pass but the module is unwired — adding an integration test through `RealBackend` would force wiring or removal. |
| `features` | 11 (transform) + family/scaler unit tests | Missing: integration test that `FeatureStream::build` round-trips through `RealBackend`. |
| `calibration` | 13 across 4 files | OK. |
| `reporting` | sparse (mostly rendered into `runs/`) | Missing snapshot tests for `RunArtifactLayout` schema. |

No `MISSING` test category blocks any thesis claim.

---

## 9. Thesis chapter mapping

Suggested mapping from chapter sections to repository evidence:

| Thesis chapter / section | Code anchor | Doc anchor | Run evidence |
|---|---|---|---|
| Ch. Background — Markov-switching model | [src/model/params.rs](src/model/params.rs), [src/model/emission.rs](src/model/emission.rs) | [docs/observation_design.md](docs/observation_design.md), [docs/emission_model.md](docs/emission_model.md) | any `model_params.json` |
| Ch. Inference — forward filter | [src/model/filter.rs](src/model/filter.rs) | [docs/forward_filter.md](docs/forward_filter.md), [docs/filter_validation.md](docs/filter_validation.md) | `regime_posteriors.csv` |
| Ch. Inference — backward smoother | [src/model/smoother.rs](src/model/smoother.rs) | [docs/backward_smoother.md](docs/backward_smoother.md) | training-only |
| Ch. Inference — pairwise posteriors | [src/model/pairwise.rs](src/model/pairwise.rs) | [docs/pairwise_posteriors.md](docs/pairwise_posteriors.md) | training-only |
| Ch. Inference — log-likelihood | [src/model/likelihood.rs](src/model/likelihood.rs) | [docs/log_likelihood.md](docs/log_likelihood.md) | `loglikelihood_history.csv` |
| Ch. Estimation — EM | [src/model/em.rs](src/model/em.rs) | [docs/em_estimation.md](docs/em_estimation.md) | `fit_summary.json` |
| Ch. Estimation — diagnostics & multi-start | [src/model/diagnostics.rs](src/model/diagnostics.rs) | [docs/diagnostics.md](docs/diagnostics.md) | `diagnostics.json`, `runs/optimize/` |
| Ch. Online inference | [src/online/mod.rs](src/online/mod.rs) | [docs/online_inference.md](docs/online_inference.md) | `score_trace.csv`, `regime_posteriors.csv` |
| Ch. Detectors — hard switch | [src/detector/hard_switch.rs](src/detector/hard_switch.rs) | [docs/changepoint_detectors.md](docs/changepoint_detectors.md) | `runs/synthetic/hard_switch/` |
| Ch. Detectors — posterior transition | [src/detector/posterior_transition.rs](src/detector/posterior_transition.rs) | [docs/changepoint_detectors.md](docs/changepoint_detectors.md) | `runs/synthetic/posterior_transition*/` |
| Ch. Detectors — surprise | [src/detector/surprise.rs](src/detector/surprise.rs) | [docs/changepoint_detectors.md](docs/changepoint_detectors.md) | `runs/synthetic/surprise*/`, `runs/real/real_wti_daily_surprise/` |
| Ch. Synthetic benchmarks | [src/benchmark/](src/benchmark/), [src/experiments/synthetic_backend.rs](src/experiments/synthetic_backend.rs) | [docs/benchmark_protocol.md](docs/benchmark_protocol.md) | `metrics.{csv,md,tex}`, `delay_distribution.png` |
| Ch. Frozen models / fixed-parameter policy | [src/detector/frozen.rs](src/detector/frozen.rs), [src/model/simulate.rs](src/model/simulate.rs) | [docs/fixed_parameter_policy.md](docs/fixed_parameter_policy.md) | `runs/synthetic/hard_switch_frozen/`, `data/frozen_models/hard_switch_frozen/` |
| Ch. Calibration | [src/calibration/](src/calibration/) | [docs/synthetic_to_real_calibration.md](docs/synthetic_to_real_calibration.md) | `runs/optimize/` |
| Ch. Real data — Route A | [src/real_eval/route_a.rs](src/real_eval/route_a.rs) | [docs/real_data_evaluation.md](docs/real_data_evaluation.md) | `runs/real/real_spy_daily_hard_switch/`, `runs/real/real_wti_daily_surprise/` |
| Ch. Real data — Route B | [src/real_eval/route_b.rs](src/real_eval/route_b.rs) | [docs/real_data_evaluation.md](docs/real_data_evaluation.md) | `segmentation.png` in real runs |
| Ch. Engineering — pipeline & artifacts | [src/experiments/runner.rs](src/experiments/runner.rs), [src/reporting/](src/reporting/) | [docs/experiment_runner.md](docs/experiment_runner.md), [docs/reporting_and_export.md](docs/reporting_and_export.md) | full run dirs |

---

## 10. Prioritised action list

### Critical (do before submission)

1. **Decide the fate of `SessionAwareSeries`.** Either (a) wire `RealBackend::load_clean_series` to construct `SessionAwareSeries` in `Intraday` mode and pass session boundaries into `FeatureFamily::RollingVol { session_reset: true }` and `StandardizedReturn { session_reset: true }`, or (b) state explicitly in the thesis and in [docs/data_pipeline.md](docs/data_pipeline.md) that intraday processing is currently session-naïve. As-is, the code in [src/data/session.rs](src/data/session.rs) is dead and the `session_reset` flags in feature families are not exercised on real data.
2. **Disclose the unused reporting machinery.** Update [docs/reporting_and_export.md](docs/reporting_and_export.md) to flag that the typed [src/reporting/export/csv.rs](src/reporting/export/csv.rs) module and `RunReporter`/`AggregateReporter` in [src/reporting/report.rs](src/reporting/report.rs) are not part of the live path. Then either delete them or wire them in.
3. **Resolve `RealEvalSummaryRow`.** Wire it into `direct_compare_runs` cross-asset aggregation, or delete it.

### Important (improves audit defensibility)

4. Add a tiny integration test that drives a synthetic experiment end-to-end through the CLI (`run-experiment`) and asserts the full artifact set listed in §3.
5. Add docs for (a) the detector trait & registry, (b) `SplitConfig`/`TimePartition`, (c) features stack ([src/features/](src/features/)).
6. Add a property-based test for `EventMatcher` (idempotent under shifting truth and detections by the same offset within tolerance).
7. Add a Route B integration test exercising `ShortSegmentPolicy::Merge` on a fixture series.

### Nice-to-have

8. Persist smoothed posteriors as an optional artifact (`smoothed_posteriors.csv`) gated by a config flag, for direct use in the thesis figures.
9. Snapshot test the schema of `RunArtifactLayout` outputs to catch silent breakage.
10. Cross-link [notes/Verification.md](notes/Verification.md) and the per-phase notes into [docs/](docs/) so the thesis bibliography can cite a single artefact rather than `notes/`.

---
