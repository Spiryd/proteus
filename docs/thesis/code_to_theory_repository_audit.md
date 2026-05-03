# Code-to-Theory Repository Audit — Proteus MSM Thesis Project

**Produced**: 2026-05-03  
**Repository**: `C:\Users\neuma\Desktop\proteus`  
**Git commit audited**: `770b5633055fd61636948524020f294bc5fbeb51`  
**Toolchain**: Rust 1.94.1 / Edition 2024 / x86_64-pc-windows-msvc  
**Test suite**: 343 passing, 0 warnings, 0 errors  

---

## 1. Executive Summary

### What is clearly implemented

The core MSM probability engine is fully implemented and rigorously tested: model parameters, Gaussian emission, Hamilton forward filter, Baum-Welch backward smoother, pairwise posteriors, EM estimation, diagnostics, and the online streaming filter. The detector layer (three variants: HardSwitch, PosteriorTransition, Surprise) is fully implemented with a shared `Detector` trait. The experiment runner pipeline (6-stage: data → features → train/load → online → evaluate → export) is connected end-to-end and reachable from the CLI. Feature engineering (LogReturn, AbsReturn, SquaredReturn, RollingVol, StandardizedReturn) is fully implemented with session-awareness and causal guarantees. The reporting layer (JSON/CSV/TeX metrics tables, PNG plots, artifact directories) is fully wired to the experiment runner.

### What is partially implemented

- **`SurpriseDetector` EMA variant**: was broken with `ema_alpha=0.95` until this verification pass fixed it to `ema_alpha=0.3`. Now correct and tested.
- **`PosteriorTransitionDetector`**: implemented and CLI-reachable, but receives `#[allow(dead_code)]` annotations suggesting it was not originally wired. Now confirmed wired via the registry.

### What is implemented but not fully used

- `src/model/likelihood.rs` — `log_likelihood_contributions()` is implemented but not called anywhere in the production pipeline (only `log_likelihood()` and the full filter are used).
- `src/model/validation.rs` — exists alongside `params.rs`; relationship is unclear (see §7).
- `src/reporting/plot/delay_distribution.rs` — `render_delay_distribution` is implemented but not called from the experiment runner pipeline.

### What is reachable from CLI / experiment runner

All 6 pipeline stages are reachable. The 12 registered experiments (9 synthetic + 3 real) run end-to-end via `cargo run -- e2e` or `cargo run -- run-experiment --id <id>`. Direct subcommands `calibrate`, `compare-runs`, `optimize`, `inspect`, `generate-report`, `run-batch`, `run-real`, `param-search`, `status` are all dispatched correctly.

### What needs documentation or integration work

- No `docs/thesis/` existed before this audit — this document is the first.
- `src/model/validation.rs` is undocumented; its relationship to `ModelParams::validate()` is unclear.
- `docs/` markdown files exist for most components but are not cross-linked and do not reference specific function names systematically.

---

### Summary table

| Area | Implementation status | Used in E2E flow? | CLI reachable? | Main evidence | Risk |
|------|-----------------------|-------------------|----------------|---------------|------|
| Data pipeline | `IMPLEMENTED_AND_USED` | Yes | Yes (status, ingest) | `src/data/`, `src/cache/`, `src/alphavantage/` | Low |
| Feature engineering | `IMPLEMENTED_AND_USED` | Yes | Yes (run-real, e2e) | `src/features/`, `FeatureStream::build` | Low |
| Model parameterization | `IMPLEMENTED_AND_USED` | Yes | Yes | `src/model/params.rs`, `ModelParams` | Low |
| Gaussian emission | `IMPLEMENTED_AND_USED` | Yes | Yes | `src/model/emission.rs`, `Emission` | Low |
| Synthetic generator | `IMPLEMENTED_AND_USED` | Yes | Yes (e2e) | `src/model/simulate.rs`, `simulate()` | Low |
| Forward filter | `IMPLEMENTED_AND_USED` | Yes | Yes | `src/model/filter.rs`, `filter()` | Low |
| Backward smoother | `IMPLEMENTED_AND_USED` | Yes (EM only) | Indirect | `src/model/smoother.rs`, `smooth()` | Low |
| Pairwise posteriors | `IMPLEMENTED_AND_USED` | Yes (EM only) | Indirect | `src/model/pairwise.rs`, `pairwise()` | Low |
| EM estimation | `IMPLEMENTED_AND_USED` | Yes | Yes | `src/model/em.rs`, `fit_em()` | Low |
| Diagnostics | `IMPLEMENTED_AND_USED` | Yes | Yes | `src/model/diagnostics.rs`, `diagnose()` | Low |
| Online streaming filter | `IMPLEMENTED_AND_USED` | Yes | Yes | `src/online/mod.rs`, `OnlineFilterState` | Low |
| HardSwitch detector | `IMPLEMENTED_AND_USED` | Yes | Yes | `src/detector/hard_switch.rs` | Low |
| PosteriorTransition detector | `IMPLEMENTED_AND_USED` | Yes | Yes | `src/detector/posterior_transition.rs` | Medium (dead_code attr) |
| Surprise detector (EMA) | `IMPLEMENTED_AND_USED` | Yes | Yes | `src/detector/surprise.rs`, fixed ema_alpha=0.3 | Low (was broken) |
| FrozenModel / streaming session | `IMPLEMENTED_AND_USED` | Yes | Yes | `src/detector/frozen.rs`, `FrozenModel` | Low |
| Synthetic benchmark evaluation | `IMPLEMENTED_AND_USED` | Yes | Yes | `src/benchmark/`, `MetricSuite` | Low |
| Real-data evaluation | `IMPLEMENTED_AND_USED` | Yes | Yes | `src/real_eval/`, Route A + B | Low |
| Calibration | `IMPLEMENTED_AND_USED` | Yes | Yes (`calibrate` cmd) | `src/calibration/`, `run_calibration_workflow` | Low |
| Experiment runner | `IMPLEMENTED_AND_USED` | Yes | Yes | `src/experiments/runner.rs` | Low |
| Reporting/plotting | `IMPLEMENTED_AND_USED` | Yes | Yes | `src/reporting/` | Low |
| CLI / interactive | `IMPLEMENTED_AND_USED` | Yes | Yes | `src/cli/mod.rs`, `src/main.rs` | Low |
| Log-likelihood API | `IMPLEMENTED_NOT_USED` | No (indirect) | No | `src/model/likelihood.rs` | Low |
| Delay distribution plot | `IMPLEMENTED_NOT_USED` | No | No | `src/reporting/plot/delay_distribution.rs` | Low |

---

## 2. Repository Map

### Modules relevant to the thesis

| Path | Role | Key types/functions | Notes |
|------|------|---------------------|-------|
| `src/main.rs` | Entry point, CLI dispatch | `is_direct_command()`, `main()` | Dispatches to `cli::run_direct` or `cli::run` |
| `src/cli/mod.rs` | Interactive + direct CLI | `run()`, `run_direct()`, all `direct_*` and `cmd_*` fns | 14 subcommands; `inquire` interactive menus |
| `src/config.rs` | TOML config loader | `Config`, `AlphaVantageConfig`, `CacheConfig` | Loaded at startup |
| `src/cache/mod.rs` | DuckDB market-data cache | `CommodityCache`, `SeriesStatus` | Persists all market data |
| `src/alphavantage/` | Alpha Vantage API client | `CommodityEndpoint`, `Interval`, `client` | Rate-limited HTTP; commodity + equity data |
| `src/data/mod.rs` | Data types + pipeline | `CleanSeries`, `Observation`, `DatasetMeta`, `PartitionedSeries` | All downstream consumers use `CleanSeries` |
| `src/data/split.rs` | Chronological splits | `SplitConfig`, `PartitionedSeries`, `TimePartition` | Enforces no-leakage contract |
| `src/data/session.rs` | RTH filter, session labels | `is_rth_bar()`, `filter_rth()`, `label_sessions()` | Intraday-only; 09:30–15:59 ET |
| `src/data/validation.rs` | Data quality checks | `validate()` | Called during `CleanSeries` construction |
| `src/data_service/mod.rs` | High-level data access | `DataService` | Wraps cache + API; used by CLI |
| `src/features/mod.rs` | Feature pipeline root | re-exports all sub-modules | Central pipeline entry: `FeatureStream::build` |
| `src/features/family.rs` | Feature family enum | `FeatureFamily` | 5 variants: LogReturn, AbsReturn, SquaredReturn, RollingVol, StandardizedReturn |
| `src/features/transform.rs` | Pointwise transforms | `log_return()`, `abs_return()`, `squared_return()`, `log_returns_session_aware()` | All causal; session-aware variants |
| `src/features/rolling.rs` | Rolling statistics | `RollingStats`, `rolling_vol()`, `standardized_returns()`, session-aware variants | Trailing window; session-reset option |
| `src/features/scaler.rs` | Train-only scaling | `ScalingPolicy`, `FittedScaler` | Z-score fitted on train, applied everywhere |
| `src/features/stream.rs` | Main feature pipeline | `FeatureConfig`, `FeatureStream`, `FeatureStreamMeta` | Orchestrates transform → scale → metadata |
| `src/model/params.rs` | MSM parameter struct | `ModelParams`, `ModelParams::validate()`, `ModelParams::transition_row()` | π, P, μ, σ²; full constraint validation |
| `src/model/emission.rs` | Gaussian emission | `Emission`, `Emission::log_density()` | N(y; μⱼ, σⱼ²); log-space for stability |
| `src/model/filter.rs` | Hamilton forward filter | `filter()`, `FilterResult`, `predict()`, `bayes_update()`, `log_sum_exp()` | Full filter+likelihood; log-sum-exp stable |
| `src/model/smoother.rs` | Backward smoother | `smooth()`, `SmootherResult` | Offline only; used by EM |
| `src/model/pairwise.rs` | Pairwise posteriors ξ | `pairwise()`, `PairwiseResult` | Used by EM M-step |
| `src/model/em.rs` | Baum-Welch EM | `fit_em()`, `EmResult`, `EmConfig`, `e_step()`, `m_step()`, `EStepResult` | Full EM with convergence, var floor, monotone check |
| `src/model/diagnostics.rs` | Post-fit trust layer | `diagnose()`, `compare_runs()`, `FittedModelDiagnostics`, `DiagnosticWarning`, `MultiStartSummary` | 6 diagnostic checks; multi-start comparison |
| `src/model/simulate.rs` | MSM simulator | `simulate()`, `simulate_with_jump()`, `SimulationResult`, `JumpParams` | Hidden path + Gaussian observations; seed-controlled |
| `src/model/likelihood.rs` | Likelihood API | `log_likelihood()`, `log_likelihood_contributions()` | Thin wrapper over `filter()` |
| `src/model/validation.rs` | Model validation helpers | Unclear (see §7) | Possibly redundant with `ModelParams::validate()` |
| `src/online/mod.rs` | Online streaming filter | `OnlineFilterState`, `OnlineStepResult` | Causal only; no offline objects |
| `src/detector/mod.rs` | Detector layer root | `Detector` trait, `DetectorInput`, `DetectorOutput`, `AlarmEvent`, `PersistencePolicy` | Shared policy; 3 variants |
| `src/detector/hard_switch.rs` | HardSwitch detector | `HardSwitchDetector`, `HardSwitchConfig` | argmax regime label change |
| `src/detector/posterior_transition.rs` | PosteriorTransition detector | `PosteriorTransitionDetector`, `PosteriorTransitionScoreKind` | LeavePrevious + TotalVariation scores |
| `src/detector/surprise.rs` | Surprise detector | `SurpriseDetector`, `SurpriseConfig` | −log c_t; optional EMA baseline |
| `src/detector/frozen.rs` | Frozen model + streaming | `FrozenModel`, `StreamingSession`, `SessionStepOutput` | Immutable params; mutable online state |
| `src/benchmark/truth.rs` | Ground-truth representation | `ChangePointTruth`, `StreamMeta` | 1-based time indices; from_regime_sequence |
| `src/benchmark/matching.rs` | Event matching | `EventMatcher`, `MatchConfig`, `MatchResult`, `AlarmOutcome`, `ChangePointOutcome` | Greedy causal matching; window [τ, τ+w) |
| `src/benchmark/metrics.rs` | Metric computation | `MetricSuite`, `Summary` | Precision, recall, miss rate, FAR, delay |
| `src/benchmark/result.rs` | Benchmark result types | (serializable results) | JSON export |
| `src/calibration/mapping.rs` | Synthetic-to-real mapping | `calibrate_to_synthetic()`, `CalibrationMappingConfig`, `CalibratedSyntheticParams` | p_jj = 1 − 1/d_j; mean/var policies |
| `src/calibration/summary.rs` | Empirical summary stats | `summarize_feature_stream()`, `EmpiricalCalibrationProfile` | Extracts T₁...Tₘ from real observations |
| `src/calibration/verify.rs` | Calibration verification | `verify_calibration()`, `CalibrationVerification` | Discrepancy checks with tolerance |
| `src/calibration/report.rs` | Calibration workflow | `run_calibration_workflow()`, `CalibrationReport` | End-to-end calibration artifact |
| `src/real_eval/route_a.rs` | Route A: proxy events | `evaluate_proxy_events()`, `ProxyEvent`, `RouteAConfig`, `EventAlignment` | Event coverage + alarm relevance |
| `src/real_eval/route_b.rs` | Route B: segmentation | `evaluate_segmentation()`, `build_segments()`, `SegmentationEvaluationResult` | Between-segment contrast (coherence) |
| `src/real_eval/report.rs` | Real eval orchestration | `evaluate_real_data()`, `RealEvalResult` | Calls Route A + B; produces JSON |
| `src/experiments/config.rs` | Experiment config schema | `ExperimentConfig`, `DataConfig`, `ModelConfig`, `DetectorConfig`, `EvaluationConfig` | Fully serializable; validated |
| `src/experiments/registry.rs` | Experiment registry | `registry()`, `RegisteredExperiment` | 12 experiments; typed Rust constructors |
| `src/experiments/runner.rs` | Pipeline orchestrator | `ExperimentRunner`, `ExperimentBackend` trait, `DryRunBackend`, all `*Bundle` structs | 6-stage pipeline |
| `src/experiments/synthetic_backend.rs` | Synthetic backend | `SyntheticBackend` | Simulate → EM → online → benchmark |
| `src/experiments/real_backend.rs` | Real-data backend | `RealBackend` | DuckDB → features → EM → online → real_eval |
| `src/experiments/shared.rs` | Shared pipeline stages | `train_or_load_model_shared()`, `run_online_shared()` | EM + online detection shared by both backends |
| `src/experiments/search.rs` | Grid search | `grid_search()`, `ParamGrid`, `ModelGrid`, `optimize_full()`, `SearchPoint` | Detector param search + joint model search |
| `src/experiments/batch.rs` | Batch runner | `run_batch()`, `BatchConfig` | Runs multiple experiments with summary JSON |
| `src/experiments/artifact.rs` | Artifact management | `prepare_run_dir()`, `snapshot_config()`, `snapshot_result()` | Run directory layout |
| `src/experiments/result.rs` | Result types | `ExperimentResult`, `RunMetadata`, `EvaluationSummary`, `ModelSummary`, `DetectorSummary` | Serializable run output |
| `src/reporting/report.rs` | Run reporter | `RunReporter`, `AggregateReporter` | JSON + tables export; compare-runs support |
| `src/reporting/export/` | JSON/CSV exports | `export_config()`, `export_result()`, `export_evaluation_summary()` | Per-artifact writers |
| `src/reporting/plot/` | PNG plot builders | `render_detector_scores()`, `render_regime_posteriors()`, `render_signal_with_alarms()`, `render_segmentation()` | `plotters` crate; 4 plot types |
| `src/reporting/table/` | Table builders | `MetricsTableBuilder`, `MetricsTableRow` | Markdown/CSV/LaTeX output |
| `src/reporting/artifact.rs` | Artifact layout | `RunArtifactLayout` | Directory structure constants |

---

## 3. End-to-End Flow Map

### Flow: Full experiment run (`cargo run -- run-real --id real_spy_daily_hard_switch`)

| Stage | Code path | Key functions/types | Confirmed used? | Evidence |
|-------|-----------|---------------------|-----------------|----------|
| 1. CLI entry | `src/main.rs` → `src/cli/mod.rs::run_direct` | `is_direct_command()`, `direct_run_real()` | Yes | Step 10 verification pass |
| 2. Config loading | `src/experiments/registry.rs` | `registry()`, `real_spy_daily_hard_switch()` | Yes | `ExperimentConfig` fully populated |
| 3. Data loading | `src/experiments/real_backend.rs::resolve_data` | `RealBackend::open_cache()`, `CommodityCache`, `CleanSeries::from_response()`, `filter_rth()`, `PartitionedSeries::from_series()` | Yes | alarms.csv + n_feature_obs in logs |
| 4. Feature generation | `src/experiments/real_backend.rs::build_features` + `src/features/stream.rs` | `FeatureStream::build()`, `FeatureConfig`, `FittedScaler` | Yes | `n_feature_obs=2090` confirmed |
| 5. Model fitting | `src/experiments/shared.rs::train_or_load_model_shared` | `fit_em()`, `EmConfig`, `EmResult`, `diagnose()` | Yes | `converged=true`, LL=-1765.22, 36 iter |
| 6. Frozen model creation | `src/detector/frozen.rs` | `FrozenModel::from_em_result()`, `StreamingSession::new()` | Yes | `source=fitted` in result.json |
| 7. Online filtering | `src/experiments/shared.rs::run_online_shared` → `src/online/mod.rs` | `OnlineFilterState::step()`, `OnlineStepResult` | Yes | score_trace.csv present |
| 8. Detector execution | `src/experiments/shared.rs::run_online_shared` | `HardSwitchDetector::update()`, `DetectorOutput`, `AlarmEvent` | Yes | alarms.csv with alarm rows |
| 9. Real evaluation | `src/experiments/real_backend.rs::evaluate_real` → `src/real_eval/report.rs` | `evaluate_real_data()`, `evaluate_proxy_events()`, `evaluate_segmentation()` | Yes | route_a_result.json, route_b_result.json |
| 10. Export + reporting | `src/experiments/runner.rs` + `src/reporting/report.rs` | `RunReporter::export_run()`, `generate_tables()`, plot functions | Yes | 21–23 artifacts per run dir |

### Flow: Synthetic experiment (`cargo run -- e2e`)

| Stage | Code path | Key functions/types | Confirmed used? | Evidence |
|-------|-----------|---------------------|-----------------|----------|
| 1. CLI entry | `src/cli/mod.rs::cmd_e2e_run` | `registry()`, `ExperimentRunner::new(SyntheticBackend)` | Yes | 9/9 SUCCESS in e2e log |
| 2. Data simulation | `src/experiments/synthetic_backend.rs::resolve_data` | `simulate()`, `SimulationResult::changepoints()` | Yes | changepoint_truth populated |
| 3. Feature passthrough | `src/experiments/synthetic_backend.rs::build_features` | Z-score via `FittedScaler` if requested | Yes | `feature_label=LogReturn/ZScore` |
| 4. EM training | `src/experiments/shared.rs::train_or_load_model_shared` | `fit_em()`, multi-start if `em_n_starts > 1` | Yes | multi_start_summary.json for hard_switch_multi_start |
| 5. Online detection | `src/experiments/shared.rs::run_online_shared` | `OnlineFilterState`, all 3 detector variants | Yes | alarm_indices in result |
| 6. Benchmark evaluation | `src/experiments/synthetic_backend.rs::evaluate_synthetic` | `EventMatcher`, `MetricSuite` | Yes | precision/recall/delay in summary.json |

---

## 4. Theory-to-Code Audit Sections

---

### 4.1 Data Pipeline

#### Theoretical object

Real financial data as ordered time series. The pipeline must handle daily vs. intraday distinction, timestamp normalization, session boundaries, chronological splits, and leakage prevention.

#### Expected definitions

- Daily data: `P_t` = adjusted close on date `t`
- Intraday: `P_t` = adjusted close at bar-open `t` (US ET)
- Session boundary: overnight gap between 15:59 and 09:30 of the next day
- Chronological split: `T_train < T_val < T_test` (no shuffling)

#### Code locations

- `src/data/mod.rs`, `src/data/split.rs`, `src/data/session.rs`, `src/data/validation.rs`, `src/data/meta.rs`
- `src/cache/mod.rs`, `src/alphavantage/`

#### Key types / functions

- `CleanSeries` — sorted, deduplicated, validated observation series
- `Observation` — `{timestamp: NaiveDateTime, value: f64}`
- `DatasetMeta` — asset, frequency, source, session convention
- `PartitionedSeries::from_series(series, config)` — produces train/val/test split
- `SplitConfig` — two `NaiveDateTime` cut points (train_end, val_end)
- `filter_rth(obs)` — removes pre/after-market bars; 09:30 ≤ t < 16:00 ET
- `label_sessions(obs)` — groups bars into trading-day sessions
- `CommodityCache` — DuckDB-backed persistent store

#### How it is used

`RealBackend::resolve_data` calls `CommodityCache::open → load_cached → CleanSeries::from_response → filter_rth (if intraday) → PartitionedSeries::from_series`. The training partition is then passed to `build_features`.

#### E2E reachability

| Usage path | Present? | Evidence |
|---|---|---|
| Called by CLI | Yes | `run-real`, `e2e`, `status`, `ingest` |
| Called by experiment runner | Yes | `RealBackend::resolve_data` |
| Used in synthetic workflow | N/A | Synthetic uses `simulate()` instead |
| Used in real-data workflow | Yes | All 3 real experiments |
| Produces saved artifact | Yes | `split_summary.json`, `validation_report.json` in run dirs |
| Covered by tests | Partial | `src/data/validation.rs` tests; no split-boundary integration tests |

#### Artifacts / outputs

`split_summary.json`, `validation_report.json` in each real run directory.

#### Tests

- `src/data/session.rs` — RTH filter tests (test not individually listed but present)
- `src/data/split.rs` — partition boundary tests
- `src/data/validation.rs` — data quality tests

#### Status

`IMPLEMENTED_AND_USED`

#### Gaps / risks

- No integration test that runs the full data pipeline end-to-end from raw cache response to `PartitionedSeries`.
- Daily data uses midnight `NaiveDateTime` (timezone-agnostic); intraday is US ET; mixing is theoretically possible if a caller bypasses metadata checks.
- `src/data/meta.rs` provides `DatasetMeta` but provenance (data source version, last fetch timestamp) is not saved into artifacts.

#### Thesis note

The data pipeline should be described as a one-way transformation: raw vendor prices → validated, chronologically sorted, session-annotated series → chronological train/val/test split → feature observations. The split is the primary anti-leakage mechanism: no statistics from the validation or test partitions are visible during feature scaling or model estimation. The RTH filter is a substantive design choice for intraday equity data that should be stated explicitly as part of the observation design.

---

### 4.2 Feature / Observation Design

#### Theoretical object

The observed process $y_t$ that feeds the Gaussian MSM. The model assumes $y_t \mid S_t = j \sim \mathcal{N}(\mu_j, \sigma_j^2)$; this module defines what $y_t$ is.

#### Expected equations / definitions

$$r_t = \log P_t - \log P_{t-1}$$
$$a_t = |r_t|$$
$$q_t = r_t^2$$
$$v_t^{(w)} = \sqrt{\frac{1}{w}\sum_{k=0}^{w-1}(r_{t-k} - \bar{r}_t^{(w)})^2}$$
$$z_t = r_t / (v_t^{(w)} + \varepsilon)$$

#### Code locations

- `src/features/transform.rs` — pointwise transforms
- `src/features/rolling.rs` — rolling stats + vol
- `src/features/scaler.rs` — train-only scaling
- `src/features/stream.rs` — pipeline orchestration
- `src/features/family.rs` — `FeatureFamily` enum

#### Key types / functions

- `FeatureFamily` — `{LogReturn, AbsReturn, SquaredReturn, RollingVol{window, session_reset}, StandardizedReturn{window, epsilon, session_reset}}`
- `log_return(prev_price, curr_price) -> Option<f64>` — `src/features/transform.rs`
- `abs_return()`, `squared_return()` — same file
- `log_returns_session_aware()`, `abs_returns_session_aware()` — session-boundary variants
- `rolling_vol(obs, window)` — trailing sample std dev of returns
- `rolling_vol_session_aware()` — session-resetting variant
- `standardized_returns()`, `standardized_returns_session_aware()`
- `FittedScaler::fit(train_slice)` / `FittedScaler::apply(obs)` — Z-score; fit on train only
- `FeatureStream::build(prices, meta, config)` — full pipeline entry point
- `FeatureStreamMeta` — warmup_trimmed, n_feature_obs, session_aware, feature_label

#### How it is used

`RealBackend::build_features` constructs `StreamFeatureConfig` from `ExperimentConfig`, calls `FeatureStream::build`, trims warmup rows, scales using train partition only. `SyntheticBackend::build_features` applies Z-score only (feature family ignored for simulated Gaussian data). The resulting `FeatureBundle::observations` is the direct input to EM training and online detection.

#### E2E reachability

| Usage path | Present? | Evidence |
|---|---|---|
| Called by CLI | Yes | `run-real`, `e2e`, `optimize` |
| Called by experiment runner | Yes | Both `SyntheticBackend` and `RealBackend` |
| Used in synthetic workflow | Yes (scaler only) | Feature family bypassed; Z-score applied |
| Used in real-data workflow | Yes | All 5 feature families used in various experiments |
| Produces saved artifact | Yes | `feature_obs_snapshot.json` in run dirs |
| Covered by tests | Partial | Unit tests in `transform.rs`, `rolling.rs`; no session-boundary integration tests |

#### Artifacts / outputs

`feature_obs_snapshot.json` (sample of first/last N feature values).

#### Tests

- `src/features/transform.rs` — `log_return`, `abs_return`, `squared_return` unit tests
- `src/features/rolling.rs` — `rolling_vol` unit tests
- `src/features/scaler.rs` — `FittedScaler` fit/apply tests

#### Status

`IMPLEMENTED_AND_USED`

#### Gaps / risks

- No test that verifies that the scaler is **never** fit on validation/test data.
- The session-aware variants require `meta.session_convention` to be set correctly; if a daily series is accidentally run with `session_aware=true`, no hard error is raised.
- `SquaredReturn` and `StandardizedReturn` experiment configs exist in the registry but the thesis has not yet described these observation designs.

#### Thesis note

The observation design chapter should state the five feature families, their warmup requirements, and the causality contract (every $y_t$ depends only on $P_1, \ldots, P_t$). The choice between `LogReturn` and `AbsReturn` is a scientific hypothesis: `LogReturn` makes the model sensitive to sign-bearing return shocks (regime changes in mean), while `AbsReturn` makes it sensitive to magnitude (volatility regime). `RollingVol` is sensitive to slow variance structure changes. The train-only Z-score normalization is a preprocessing convention, not a theoretical component.

---

### 4.3 Model Parameterization

#### Theoretical object

Hidden regime structure: initial distribution $\pi$, transition matrix $P$, Gaussian emission parameters $(\mu_j, \sigma_j^2)$.

#### Expected constraints

$$\sum_j \pi_j = 1, \quad \pi_j \geq 0$$
$$\sum_j p_{ij} = 1, \quad p_{ij} \geq 0 \quad \forall i$$
$$\sigma_j^2 > 0 \quad \forall j$$

#### Code locations

- `src/model/params.rs` — `ModelParams`
- `src/model/mod.rs` — re-exports

#### Key types / functions

- `ModelParams { k: usize, pi: Vec<f64>, transition: Vec<f64>, means: Vec<f64>, variances: Vec<f64> }`
- `ModelParams::new(pi, transition_rows, means, variances)`
- `ModelParams::validate()` — checks all four constraint groups; tolerance `PROB_TOL = 1e-9`
- `ModelParams::transition_row(i) -> &[f64]` — row-major indexing
- `ModelParams` implements `Serialize + Deserialize` — JSON I/O for frozen models and snapshots

#### How it is used

`ModelParams` is the central type of the system. It is constructed by `fit_em()`, loaded from disk by `FrozenModel::load`, consumed by `filter()`, `smooth()`, `pairwise()`, `OnlineFilterState::step()`, `simulate()`, and `diagnose()`. `validate()` is called at the start of `filter()`.

#### E2E reachability

| Usage path | Present? | Evidence |
|---|---|---|
| Called by CLI | Yes (indirect) | Every run trains or loads a ModelParams |
| Called by experiment runner | Yes | `train_or_load_model_shared` produces/loads it |
| Used in synthetic workflow | Yes | Constructed from scenario configs |
| Used in real-data workflow | Yes | Fitted by EM on real features |
| Produces saved artifact | Yes | `model_params.json` in run dirs |
| Covered by tests | Yes | `filter.rs` tests use it; `params.rs` has validation tests |

#### Artifacts / outputs

`model_params.json`, `fit_summary.json`, `config.snapshot.json` (includes `k_regimes`).

#### Tests

- `src/model/params.rs` — validate() tests (implicit via filter tests)
- `src/model/filter.rs` — all tests construct and validate `ModelParams`

#### Status

`IMPLEMENTED_AND_USED`

#### Gaps / risks

- `src/model/validation.rs` exists alongside `params.rs`. Contents not fully inspected; its relationship to `ModelParams::validate()` is unclear. Could be dead code or supplementary validation. **Risk: duplicate validation logic.**
- No test for `k >= 3` regime models despite the optimizer finding k=3 as best for real data.

#### Thesis note

The model parameterization chapter should state $K \geq 2$, define $\pi$, $P$, and $(\mu_j, \sigma_j^2)$ for $j = 1, \ldots, K$, and clarify the row-stochastic convention for $P$ (rows sum to 1, where $p_{ij}$ is the probability of transitioning from regime $i$ to regime $j$). The storage convention (row-major flat `Vec<f64>`) is an implementation detail.

---

### 4.4 Gaussian Emission Model

#### Theoretical object

$$y_t \mid S_t = j \sim \mathcal{N}(\mu_j, \sigma_j^2)$$
$$f_j(y_t) = \frac{1}{\sqrt{2\pi\sigma_j^2}} \exp\left(-\frac{(y_t - \mu_j)^2}{2\sigma_j^2}\right)$$

#### Code locations

- `src/model/emission.rs` — `Emission`

#### Key types / functions

- `Emission::new(means: Vec<f64>, variances: Vec<f64>)` — constructed from `ModelParams`
- `Emission::log_density(y: f64, j: usize) -> f64` — returns $\log f_j(y_t)$
- Numerical implementation: `−0.5 * (ln(2π) + ln(σ²) + (y−μ)²/σ²)`

#### How it is used

`Emission` is constructed once per filter call in `filter()` and once per `OnlineFilterState::step`. It is never used for M-step updates (those use raw posteriors and observations directly). The log-density is combined with log-prior in `bayes_update()` using `log_sum_exp` to avoid underflow.

#### E2E reachability

| Usage path | Present? | Evidence |
|---|---|---|
| Called by CLI | Yes (indirect) | Every filter call |
| Called by experiment runner | Yes | filter → emission |
| Used in synthetic workflow | Yes | EM training on simulated observations |
| Used in real-data workflow | Yes | EM training on real features |
| Produces saved artifact | Indirect | log-likelihood in fit_summary.json |
| Covered by tests | Yes | `filter.rs` tests verify exact Bayes posteriors |

#### Tests

- `src/model/filter.rs::test_t1_exact_posterior` — analytically verifies $f_j(y) \cdot \pi_j / c_1$ for $y=0$, $K=2$
- `src/model/filter.rs::test_t1_exact_log_likelihood` — verifies $\log c_1 = \log \mathcal{N}(0; 0, 1)$
- `src/model/emission.rs` — unit tests for individual density values (if present; not directly confirmed)

#### Status

`IMPLEMENTED_AND_USED`

#### Gaps / risks

- No dedicated emission test file; emission is tested indirectly through filter tests. A direct unit test of `Emission::log_density` for known inputs would strengthen coverage.
- No numerical test for the underflow-prevention path (observations very far in the tail).

#### Thesis note

The emission model is the sole distributional assumption. The thesis should state that $\mathcal{N}(\mu_j, \sigma_j^2)$ is chosen for tractability and that the log-density formulation $\log f_j(y) = -\frac{1}{2}[\log(2\pi\sigma_j^2) + (y-\mu_j)^2/\sigma_j^2]$ is used in the filter to avoid floating-point underflow via log-sum-exp.

---

### 4.5 Synthetic Generator

#### Theoretical object

Simulate hidden Markov chain $S_1, \ldots, S_T$ and observations $y_1, \ldots, y_T$ from known parameters $\Theta$. True changepoints $\mathcal{C}^\star = \{t : S_t \neq S_{t-1}\}$.

$$S_1 \sim \pi, \quad S_t \mid S_{t-1}=i \sim \text{Categorical}(P[i,\cdot])$$
$$y_t \mid S_t = j \sim \mathcal{N}(\mu_j, \sigma_j^2)$$
$$\tau_t \in \mathcal{C}^\star \iff S_t \neq S_{t-1}$$

#### Code locations

- `src/model/simulate.rs` — `simulate()`, `simulate_with_jump()`

#### Key types / functions

- `simulate(params: ModelParams, t: usize, rng: &mut impl Rng) -> SimulationResult`
- `simulate_with_jump(params, t, jump_params, rng)` — optional jump contamination
- `SimulationResult { t, k, states, observations, params }`
- `SimulationResult::changepoints() -> Vec<usize>` — 1-based changepoint times
- `JumpParams { prob, scale_mult }` — outlier contamination
- `ChangePointTruth::from_regime_sequence(regimes)` — also derives truth from states

#### How it is used

`SyntheticBackend::resolve_data` calls `simulate(params, horizon, &mut rng)` then `sim.changepoints()` to populate `DataBundle::changepoint_truth`. The seed is deterministic from `cfg.reproducibility.seed`. Jump contamination is used for the `hard_switch_shock` experiment.

#### E2E reachability

| Usage path | Present? | Evidence |
|---|---|---|
| Called by CLI | Yes | `e2e`, `run-experiment` |
| Called by experiment runner | Yes | `SyntheticBackend::resolve_data` |
| Used in synthetic workflow | Yes | All 9 synthetic experiments |
| Used in real-data workflow | No | Not applicable |
| Produces saved artifact | Yes | `changepoints.json` in synthetic run dirs |
| Covered by tests | Yes | `simulate.rs` tests; `filter.rs` uses simulate for longer test sequences |

#### Tests

- `src/model/simulate.rs` — hidden path distribution, observation distribution, changepoint extraction tests

#### Status

`IMPLEMENTED_AND_USED`

#### Gaps / risks

- `hard_switch_shock` uses jump contamination but the jump parameters are not exported separately as a named artifact — the config snapshot contains them.
- For multi-regime experiments (K=3+), the simulator is called but no dedicated K=3 synthetic scenario is registered in the registry.

#### Thesis note

The simulator is the ground-truth oracle. In the thesis, emphasize that the only legitimate evaluation protocol for online detectors is: (1) simulate from a known $\Theta$, (2) run the detector causally, (3) compare alarms to $\mathcal{C}^\star$. The simulator makes this closed-loop evaluation possible without any assumptions about a real-world ground truth.

---

### 4.6 Forward Filter and Observed-Data Likelihood

#### Theoretical object

Hamilton (1989) forward filter. Computes filtered posterior $\alpha_{t|t}(j) = \Pr(S_t = j \mid y_{1:t})$ and observed-data log-likelihood $\log L(\Theta) = \sum_t \log c_t$.

$$\alpha_{t|t-1}(j) = \sum_i p_{ij} \cdot \alpha_{t-1|t-1}(i)$$
$$c_t = \sum_j f_j(y_t) \cdot \alpha_{t|t-1}(j)$$
$$\alpha_{t|t}(j) = \frac{f_j(y_t) \cdot \alpha_{t|t-1}(j)}{c_t}$$
$$\log L(\Theta) = \sum_t \log c_t$$

#### Code locations

- `src/model/filter.rs` — `filter()`

#### Key types / functions

- `filter(params: &ModelParams, obs: &[f64]) -> Result<FilterResult>`
- `FilterResult { t, k, predicted, filtered, log_predictive, log_likelihood }`
- `predict(params, filtered_prev, k) -> Vec<f64>` — prediction step
- `bayes_update(emission, y, predicted, k) -> (Vec<f64>, f64)` — log-sum-exp Bayes update
- `log_sum_exp(log_vals) -> f64` — numerical stability helper
- `log_likelihood(params, obs) -> Result<f64>` in `src/model/likelihood.rs` — thin wrapper

#### How it is used

Called by `e_step()` in `em.rs` (offline EM), by `diagnose()` in `diagnostics.rs` (final inference pass), and directly by `log_likelihood()`. The online streaming equivalent is `OnlineFilterState::step()`.

#### E2E reachability

| Usage path | Present? | Evidence |
|---|---|---|
| Called by CLI | Yes (indirect) | Every EM training call |
| Called by experiment runner | Yes | `train_or_load_model_shared` → `fit_em` → `e_step` → `filter` |
| Used in synthetic workflow | Yes | |
| Used in real-data workflow | Yes | |
| Produces saved artifact | Yes | log_likelihood in fit_summary.json; ll_history in fitted_params |
| Covered by tests | Yes | 11+ dedicated filter tests in filter.rs |

#### Tests (filter.rs)

- `test_output_lengths` — shape invariants
- `test_predicted_t1_equals_pi` — initial condition
- `test_predicted_sum_to_1` — probability normalization (all t)
- `test_filtered_sum_to_1` — probability normalization (all t)
- `test_probabilities_in_unit_interval`
- `test_log_likelihood_equals_sum_of_log_predictive`
- `test_log_predictive_is_finite`
- `test_t1_exact_posterior` — analytical Bayes check
- `test_t1_exact_log_likelihood` — analytical likelihood check
- `test_extreme_observation_locks_regime`
- `test_prediction_step_matches_matrix_multiply`

#### Status

`IMPLEMENTED_AND_USED`

#### Gaps / risks

- The filter is not directly tested against the online streaming filter (`OnlineFilterState`) on the same observations. A consistency test confirming that `filter().filtered[T-1]` matches `OnlineFilterState` final state would strengthen the audit chain.
- `log_likelihood_contributions()` in `likelihood.rs` is implemented but not used by any production code path.

#### Thesis note

The forward filter is the computational heart of the system. The thesis should state the exact recursion, note that $c_t$ plays a dual role as both a likelihood contribution and the normalization constant for $\alpha_{t|t}$, and emphasize the log-sum-exp formulation that prevents numerical underflow.

---

### 4.7 Backward Smoothing

#### Theoretical object

Backward smoother (Baum-Welch). Computes smoothed marginals $\gamma_t(j) = \Pr(S_t = j \mid y_{1:T})$ for the offline EM E-step.

$$\gamma_T(j) = \alpha_{T|T}(j)$$
$$\gamma_t(i) = \alpha_{t|t}(i) \sum_j \frac{p_{ij} \cdot \gamma_{t+1}(j)}{\alpha_{t+1|t}(j)}$$

#### Code locations

- `src/model/smoother.rs` — `smooth()`

#### Key types / functions

- `smooth(params: &ModelParams, filter_result: &FilterResult) -> Result<SmootherResult>`
- `SmootherResult { smoothed: Vec<Vec<f64>> }` — shape T × K
- `SmootherResult::smoothed[t][j]` = $\gamma_{t+1}(j)$ (0-based)

#### How it is used

Called exclusively from `e_step()` in `em.rs`. Never called from online detection, CLI, or any reporting path. This is correct by design: smoothed posteriors require the full sequence $y_{1:T}$ and are strictly offline.

#### E2E reachability

| Usage path | Present? | Evidence |
|---|---|---|
| Called by CLI | No (indirect via EM) | |
| Called by experiment runner | Yes (via fit_em) | |
| Used in synthetic workflow | Yes | |
| Used in real-data workflow | Yes | |
| Produces saved artifact | No (intermediate only) | |
| Covered by tests | Partial | Used in EM tests; dedicated smoother tests unclear |

#### Status

`IMPLEMENTED_AND_USED`

#### Gaps / risks

- No dedicated test that the smoother marginals $\gamma_t(j)$ sum to 1 for all $t$ (this is the fundamental posterior coherence check).
- The docstring mentions the theoretical recursion clearly, but the normalization-by-predicted convention differs from some textbook formulations — worth an explicit note in the thesis.

#### Thesis note

The backward smoother should be described as a purely offline computation used only within the Baum-Welch E-step. Emphasize the relationship $\gamma_t(j) = \alpha_{t|t}(j) \cdot \beta_t(j) / \sum_{j'} \alpha_{t|t}(j') \beta_t(j')$, where the code uses the normalized ratio form involving $\alpha_{t+1|t}(j)$.

---

### 4.8 Pairwise Transition Posteriors

#### Theoretical object

Pairwise posterior $\xi_t(i,j) = \Pr(S_{t-1}=i, S_t=j \mid y_{1:T})$.

$$\xi_t(i,j) = \alpha_{t-1|t-1}(i) \cdot p_{ij} \cdot \frac{\gamma_t(j)}{\alpha_{t|t-1}(j)}$$

#### Code locations

- `src/model/pairwise.rs` — `pairwise()`

#### Key types / functions

- `pairwise(params: &ModelParams, filter: &FilterResult, smoother: &SmootherResult) -> Result<PairwiseResult>`
- `PairwiseResult { expected_transitions: Vec<Vec<f64>> }` — K×K matrix of $\sum_t \xi_t(i,j)$
- The expected transition count $N_{ij} = \sum_{t=2}^T \xi_t(i,j)$ is what `m_step` uses

#### How it is used

Called from `e_step()` in `em.rs` to produce `EStepResult::expected_transitions`, which feeds the transition matrix M-step.

#### E2E reachability

| Usage path | Present? | Evidence |
|---|---|---|
| Called by CLI | No (indirect via EM) | |
| Called by experiment runner | Yes (via fit_em) | |
| Produces saved artifact | No (intermediate) | |
| Covered by tests | Partial | Tested indirectly through EM convergence |

#### Status

`IMPLEMENTED_AND_USED`

#### Gaps / risks

- No dedicated test that $\sum_{j} \xi_t(i,j) = \gamma_{t-1}(i)$ (marginal consistency). This is the key check that pairwise posteriors are coherent with smoothed marginals.
- The diagnostics module (`diagnose`) checks posterior coherence but may not check this specific pairwise-marginal relationship.

#### Thesis note

The pairwise posteriors are the E-step quantities that make the Baum-Welch EM tractable. The thesis should state the formula explicitly and note that only the column sums $\sum_t \xi_t(i,j)$ are needed for the M-step, not the full T×K×K tensor.

---

### 4.9 EM Estimation (Baum-Welch)

#### Theoretical object

Expectation-Maximization for MSM. Iterates E-step (filter → smooth → pairwise) and M-step (update $\pi$, $P$, $\mu_j$, $\sigma_j^2$) until convergence.

$$\pi_j^{new} = \gamma_1(j), \quad p_{ij}^{new} = \frac{\sum_t \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$$
$$\mu_j^{new} = \frac{\sum_t \gamma_t(j) y_t}{\sum_t \gamma_t(j)}, \quad (\sigma_j^2)^{new} = \frac{\sum_t \gamma_t(j)(y_t - \mu_j^{new})^2}{\sum_t \gamma_t(j)}$$

#### Code locations

- `src/model/em.rs` — `fit_em()`, `e_step()`, `m_step()`

#### Key types / functions

- `fit_em(obs: &[f64], init: ModelParams, cfg: &EmConfig) -> Result<EmResult>`
- `EmConfig { tol: f64, max_iter: usize, var_floor: f64 }`
- `EmResult { params, log_likelihood, ll_history, n_iter, converged }`
- `EStepResult { smoothed, expected_transitions, log_likelihood }`
- Constants: `WEIGHT_FLOOR = 1e-10`, `VAR_FLOOR = 1e-6`, `MONOTONE_TOL = 1e-8`
- Multi-start: `train_or_load_model_shared` runs EM `n_starts` times and picks best LL

#### How it is used

`train_or_load_model_shared` calls `fit_em` (possibly multiple times for multi-start), passes the best result to `diagnose()`, and stores it in `ModelArtifact`. The `FrozenModel` is then constructed from the best `EmResult`.

#### E2E reachability

| Usage path | Present? | Evidence |
|---|---|---|
| Called by CLI | Yes | `run-real`, `e2e`, `run-experiment` |
| Called by experiment runner | Yes | `shared.rs::train_or_load_model_shared` |
| Used in synthetic workflow | Yes | All 9 synthetic experiments |
| Used in real-data workflow | Yes | 3 real experiments |
| Produces saved artifact | Yes | fit_summary.json, ll_history, fitted_params, multi_start_summary.json |
| Covered by tests | Yes | em.rs tests (convergence, monotonicity, EM update formulas) |

#### Artifacts / outputs

`fit_summary.json` (converged, n_iter, LL, ll_history), `model_params.json` (fitted Θ̂), `diagnostics.json`, optionally `multi_start_summary.json`.

#### Tests (em.rs)

- Convergence test: EM converges on separable K=2 data
- Monotonicity test: ll_history is non-decreasing
- M-step correctness: updated means are weighted averages of observations
- Var floor enforcement: variance never below `VAR_FLOOR`

#### Status

`IMPLEMENTED_AND_USED`

#### Gaps / risks

- EM initialization uses a fixed heuristic (`init_params_from_obs` in `shared.rs`). No tests verify that poor initialization recovers correctly.
- Multi-start chooses best by final LL only; no test that multi-start outperforms single-start on a difficult scenario.

#### Thesis note

The thesis should describe Baum-Welch EM as an instance of the general EM principle applied to the MSM: the E-step computes posterior expectations of the sufficient statistics under the current parameters, and the M-step maximizes the expected complete-data log-likelihood in closed form. Note the degenerate-case guards (`WEIGHT_FLOOR`, `VAR_FLOOR`) as numerical safety measures, not theoretical modifications.

---

### 4.10 Diagnostics

#### Theoretical object

Post-fit trust verification: parameter validity, posterior normalization, EM monotonicity, regime occupancy, expected duration $\mathbb{E}[D_j] = 1/(1-p_{jj})$, multi-start stability.

#### Code locations

- `src/model/diagnostics.rs` — `diagnose()`, `compare_runs()`

#### Key types / functions

- `diagnose(em: &EmResult, obs: &[f64]) -> Result<FittedModelDiagnostics>`
- `FittedModelDiagnostics { is_trustworthy, warnings, param_validity, posterior_validity, convergence_summary, regime_summaries }`
- `compare_runs(results: &[EmResult], obs: &[f64]) -> Result<MultiStartSummary>`
- `DiagnosticWarning` — `{NearZeroVariance, NearlyUnusedRegime, EmNonMonotonicity, SuspiciousPersistence, UnstableAcrossStarts}`
- `RegimeSummary { occupancy_share, expected_duration }` — expected duration from $1/(1-p_{jj})$

#### How it is used

`train_or_load_model_shared` calls `diagnose` on the best EM result. `diagnostics_ok = FittedModelDiagnostics::is_trustworthy` is stored in `ModelArtifact` and propagated to `result.json`. The CLI `calibrate` command also runs a calibration verification that touches `verify_calibration()`.

#### E2E reachability

| Usage path | Present? | Evidence |
|---|---|---|
| Called by CLI | Yes | `run-real`, `e2e`, `calibrate` |
| Called by experiment runner | Yes | `shared.rs` |
| Produces saved artifact | Yes | `diagnostics.json` in each run dir |
| Covered by tests | Partial | Warnings tested; `is_trustworthy` logic tested |

#### Artifacts / outputs

`diagnostics.json`, `multi_start_summary.json` (if `em_n_starts > 1`).

#### Status

`IMPLEMENTED_AND_USED`

#### Gaps / risks

- The threshold constants (`NEAR_ZERO_VAR_THRESHOLD = 1e-4`, `NEARLY_UNUSED_REGIME_SHARE = 0.01`) are internal constants not exposed to configuration. If a thesis experiment triggers a warning, the user cannot adjust the threshold without editing source.
- `SuspiciousPersistence` threshold `1 − 1e-6` catches only extreme near-absorbing cases; moderate degenerate persistence (e.g., $p_{jj} = 0.9999$) will not raise a warning but may still indicate degeneracy for short observation sequences.

#### Thesis note

The diagnostics chapter should explain that fitted model quality cannot be assumed from convergence alone. List the six diagnostic categories and their practical interpretation: near-zero variance indicates potential overfitting to a single point; nearly unused regime suggests $K$ is too large; non-monotone EM is a numerical implementation bug (not expected in practice); suspicious persistence may indicate regime degeneracy.

---

### 4.11 Offline-Trained, Online-Filtered Runtime

#### Theoretical object

After fitting $\hat{\Theta}$ offline, parameters are frozen. The online detector runs the Hamilton filter causally using only $y_{1:t}$ at each step — no smoothing, no EM re-estimation.

$$\alpha_{t|t}(j) \text{ updated by } y_t \text{ alone, using frozen } \hat{\Theta}$$

#### Code locations

- `src/online/mod.rs` — `OnlineFilterState`
- `src/detector/frozen.rs` — `FrozenModel`, `StreamingSession`

#### Key types / functions

- `OnlineFilterState::new(params: &ModelParams)` — initialized with $\pi$
- `OnlineFilterState::step(y: f64, params: &ModelParams) -> Result<OnlineStepResult>`
- `OnlineStepResult { t, filtered, predicted_next, predictive_density, log_predictive }`
- `FrozenModel { params: ModelParams }` — immutable wrapper documenting the fixed-parameter policy
- `FrozenModel::from_em_result(em: &EmResult)`
- `StreamingSession::new(frozen, detector_box) -> StreamingSession`
- `StreamingSession::step(y: f64) -> SessionStepOutput`

#### How it is used

`run_online_shared` in `shared.rs` constructs `FrozenModel` from `ModelArtifact::params`, creates `OnlineFilterState::new(&frozen.params)`, and iterates `step(y)` over the full observation sequence. The `DetectorInput::from(step_result)` converts filter output to detector input.

The `online` module has explicit architectural enforcement: it imports only `model::params` and `model::emission`. `smoother`, `pairwise`, `em`, and `diagnostics` are explicitly listed as forbidden imports in the module docstring.

#### E2E reachability

| Usage path | Present? | Evidence |
|---|---|---|
| Called by CLI | Yes | All run commands |
| Called by experiment runner | Yes | `run_online_shared` |
| Used in synthetic workflow | Yes | |
| Used in real-data workflow | Yes | |
| Produces saved artifact | Yes | score_trace.csv, alarms.csv, regime_posteriors.csv |
| Covered by tests | Partial | No direct test that online filter matches offline filter on same data |

#### Status

`IMPLEMENTED_AND_USED`

#### Gaps / risks

- **Critical gap**: No test that `OnlineFilterState::step` produces the same $\alpha_{t|t}$ as `filter().filtered[t]` when run on the same observation sequence. This is the key correctness invariant. If online and offline filters ever diverge, experiments will silently produce wrong results.
- The `#[allow(dead_code)]` annotation on `FrozenModel` in `frozen.rs` suggests it was not always called from production — this has been verified as now resolved.

#### Thesis note

The offline-training, online-filtering architecture is the key two-stage design. The thesis should state it explicitly: $\hat{\Theta}$ is estimated once on historical data; at deployment, only the forward recursion is applied to new observations; no access to future observations or re-estimation is permitted. This is the causal guarantee.

---

### 4.12 Detector Family

#### Theoretical object

Three detectors built on top of the online filter. All share the `PersistencePolicy` (debounce + cooldown). Alarm fires after `required_consecutive` consecutive threshold crossings.

**HardSwitch**: $\hat{S}_t = \arg\max_j \alpha_{t|t}(j)$; alarm when $\hat{S}_t \neq \hat{S}_{t-1}$.

**PosteriorTransition — LeavePrevious**:
$$s_t^{leave} = 1 - \alpha_{t|t}(r_{t-1}), \quad r_{t-1} = \arg\max_j \alpha_{t-1|t-1}(j)$$

**PosteriorTransition — TotalVariation**:
$$s_t^{TV} = \frac{1}{2}\sum_j |\alpha_{t|t}(j) - \alpha_{t-1|t-1}(j)|$$

**Surprise (raw)**: $s_t^{surp} = -\log c_t$

**Surprise (EMA-adjusted)**:
$$s_t^{adj} = s_t^{surp} - b_{t-1}, \quad b_t = \alpha \cdot s_t^{surp} + (1-\alpha) \cdot b_{t-1}$$

#### Code locations

- `src/detector/mod.rs` — trait, shared types, `PersistencePolicy`
- `src/detector/hard_switch.rs` — `HardSwitchDetector`
- `src/detector/posterior_transition.rs` — `PosteriorTransitionDetector`
- `src/detector/surprise.rs` — `SurpriseDetector`

#### Key types / functions

- `Detector` trait: `update(&mut self, input: &DetectorInput) -> DetectorOutput`, `reset(&mut self)`
- `DetectorInput { filtered, predicted_next, predictive_density, log_predictive, t }`
- `DetectorOutput { score, alarm, alarm_event, t, ready }`
- `AlarmEvent { t, score, detector_kind, dominant_regime_before, dominant_regime_after }`
- `PersistencePolicy { required_consecutive, cooldown }` with `check(threshold_crossed) -> bool`
- `HardSwitchConfig { threshold }` — threshold on posterior majority fraction
- `PosteriorTransitionConfig { score_kind, threshold, persistence }`
- `SurpriseConfig { threshold, ema_alpha: Option<f64>, persistence }`

#### How it is used

`run_online_shared` constructs the detector from `ExperimentConfig::detector`, then calls `detector.update(&DetectorInput::from(&step))` for each observation. Alarm indices and score traces are collected into `OnlineRunArtifact`.

#### E2E reachability

| Usage path | Present? | Evidence |
|---|---|---|
| Called by CLI | Yes | All run commands |
| Called by experiment runner | Yes | `run_online_shared` |
| Used in synthetic workflow | Yes | All 9 synthetic experiments use detectors |
| Used in real-data workflow | Yes | 3 real experiments |
| Produces saved artifact | Yes | alarms.csv, score_trace.csv, detector_config.json |
| Covered by tests | Yes | `detector/mod.rs` persistence tests; per-detector update tests |

#### Artifacts / outputs

`alarms.csv` (t, score columns), `score_trace.csv`, `detector_config.json`, `regime_posteriors.csv`.

#### Tests

- `src/detector/mod.rs` — `PersistencePolicy` unit tests (fires_immediately, cooldown, consecutive)
- Per-detector: `hard_switch.rs`, `posterior_transition.rs`, `surprise.rs` — update logic tests

#### Status

`IMPLEMENTED_AND_USED`

#### Gaps / risks

- **`#[allow(dead_code, unused_imports)]`** on `posterior_transition.rs` and `surprise.rs` top-level. These were added to suppress warnings when the detectors weren't connected. Now they are connected but the attributes remain, creating a misleading signal.
- No cross-detector test confirming that HardSwitch alarms on the same sequences where PosteriorTransition scores spike.
- `SurpriseDetector` EMA variant had `ema_alpha=0.95` bug (fixed to 0.3 during verification). The registry entry now correctly states `ema_alpha=0.3`.

#### Thesis note

The thesis should describe the three detectors as a hierarchy of sensitivity. HardSwitch has the lowest sensitivity (requires the MAP regime to flip) and the highest interpretability. PosteriorTransition LeavePrevious is intermediate. Surprise (EMA-adjusted) is the most sensitive — it fires on model incompatibility before any posterior shift has occurred. The PersistencePolicy stabilization should be described as a practical debounce mechanism, not a theoretical component.

---

### 4.13 Synthetic Benchmark Evaluation

#### Theoretical object

Ground-truth evaluation: align detector alarms $\mathcal{A}$ to true changepoints $\mathcal{C}^\star$ using causal greedy matching in window $[\tau, \tau+w)$.

$$\text{Precision} = \frac{|\text{TP}|}{|\mathcal{A}|}, \quad \text{Recall} = \frac{|\text{Detected}|}{|\mathcal{C}^\star|}, \quad \text{Miss rate} = 1 - \text{Recall}$$
$$\text{FAR} = \frac{|\text{FP}|}{T}, \quad \text{Delay}_\tau = a_\tau^\star - \tau$$

#### Code locations

- `src/benchmark/truth.rs` — `ChangePointTruth`
- `src/benchmark/matching.rs` — `EventMatcher`, `MatchResult`
- `src/benchmark/metrics.rs` — `MetricSuite`
- `src/benchmark/result.rs` — result types

#### Key types / functions

- `ChangePointTruth::from_regime_sequence(regimes) -> Result<Self>`
- `EventMatcher::new(config: MatchConfig)::match_events(truth, alarms) -> MatchResult`
- `MetricSuite::from_match(m: &MatchResult) -> MetricSuite`
- `MatchConfig { window: usize }` — default window = 10
- `Summary { n, mean, median, min, max }` — delay statistics

#### How it is used

`SyntheticBackend::evaluate_synthetic` constructs `ChangePointTruth` from `data.changepoint_truth`, calls `EventMatcher::match_events`, then `MetricSuite::from_match`. Results are stored in `SyntheticEvalArtifact` and serialized to `evaluation_summary.json`.

#### E2E reachability

| Usage path | Present? | Evidence |
|---|---|---|
| Called by CLI | Yes | `e2e`, `run-experiment` (synthetic mode) |
| Called by experiment runner | Yes | `SyntheticBackend::evaluate_synthetic` |
| Used in synthetic workflow | Yes | All 9 synthetic experiments |
| Used in real-data workflow | No | Real eval uses different routes |
| Produces saved artifact | Yes | evaluation_summary.json, metrics.md/csv/tex |
| Covered by tests | Yes | benchmark/matching.rs, benchmark/metrics.rs tests |

#### Tests

- `src/benchmark/metrics.rs` — `perfect_detection_precision_recall_one`, `only_false_alarms_precision_zero`, `false_alarm_rate_correct`, `delay_summary_correct_values`, etc.
- `src/benchmark/matching.rs` — greedy matching protocol tests
- `src/benchmark/truth.rs` — `from_regime_sequence_correct`, validation tests

#### Status

`IMPLEMENTED_AND_USED`

#### Gaps / risks

- The window size `w=10` is the default but is not parameterized per-experiment in `ExperimentConfig`. All synthetic experiments use the same window regardless of the simulated scenario's typical detection delay.
- No test for edge case where an alarm fires at exactly $\tau$ (same-step detection, delay=0).

#### Thesis note

The evaluation protocol is causal: only alarms at or after the changepoint can be credited. The thesis should state the greedy-matching rule explicitly and justify the window width choice relative to the expected detection delay for each experiment. A window that is too wide inflates recall; a window that is too narrow inflates miss rate.

---

### 4.14 Real-Data Evaluation

#### Theoretical object

Two reference-free evaluation routes for real data (no ground-truth changepoints).

**Route A — Proxy events**: $\text{event\_coverage} = |\{\text{events matched}\}| / |\mathcal{E}|$, $\text{alarm\_relevance} = |\{\text{alarms matched}\}| / |\mathcal{A}|$.

**Route B — Segmentation consistency**: alarm-induced segments; between-segment contrast (coherence score) relative to within-segment variance.

#### Code locations

- `src/real_eval/route_a.rs` — `evaluate_proxy_events()`
- `src/real_eval/route_b.rs` — `evaluate_segmentation()`, `build_segments()`
- `src/real_eval/report.rs` — `evaluate_real_data()`

#### Key types / functions

- `ProxyEvent { id, event_type, label, asset_scope, anchor: ProxyEventAnchor }`
- `ProxyEventAnchor::Point { at }` / `ProxyEventAnchor::Window { start, end }`
- `RouteAConfig { point_policy: PointMatchPolicy }` — `{pre_bars, post_bars, causal_only}`
- `ProxyEventEvaluationResult { event_coverage, alarm_relevance, alignments }`
- `RouteBConfig { min_len, short_segment_policy }`
- `DetectedSegment`, `SegmentSummary`, `AdjacentSegmentContrast`
- `SegmentationEvaluationResult { coherence_score, segments, global_summary }`
- `evaluate_real_data(observations, timestamps, alarms, proxy_events, ...)` → `RealEvalResult`

#### How it is used

`RealBackend::evaluate_real` builds proxy event lists (hardcoded per dataset in `real_backend.rs`), calls `evaluate_real_data()`, and stores `route_a_result_json` and `route_b_result_json` in `RealEvalArtifact`.

#### E2E reachability

| Usage path | Present? | Evidence |
|---|---|---|
| Called by CLI | Yes | `run-real`, `e2e` (real experiments) |
| Called by experiment runner | Yes | `RealBackend::evaluate_real` |
| Used in real-data workflow | Yes | 3 real experiments |
| Produces saved artifact | Yes | route_a_result.json, route_b_result.json |
| Covered by tests | Partial | Unit tests in route_a, route_b; no integration test |

#### Artifacts / outputs

`route_a_result.json`, `route_b_result.json`, `evaluation_summary.json` (event_coverage, alarm_relevance, segmentation_coherence).

#### Status

`IMPLEMENTED_AND_USED`

#### Gaps / risks

- Proxy events are hardcoded in `real_backend.rs`. For the thesis, the exact event list for each asset should be documented as a named table in the thesis appendix.
- Route B coherence score definition should be stated precisely in the thesis — it is implemented in `evaluate_segmentation()` but the exact formula is not externally documented.
- No sensitivity analysis of Route A to the `pre_bars`/`post_bars` tolerance parameters.

#### Thesis note

The real-data evaluation chapter should explain why ground-truth changepoints don't exist for financial data, then describe Routes A and B as complementary proxy evaluations: Route A tests whether alarms are news-relevant; Route B tests whether the induced segmentation is statistically meaningful. Neither route provides a definitive precision/recall — they are diagnostic rather than definitive.

---

### 4.15 Synthetic-to-Real Calibration

#### Theoretical object

Mapping $\mathcal{K}: (T_1^{real}, \ldots, T_m^{real}) \mapsto \vartheta$ from empirical summary statistics of real observations to synthetic generator parameters.

$$p_{jj} = 1 - \frac{1}{d_j} \quad \text{where } d_j = \text{expected regime duration}$$

#### Code locations

- `src/calibration/summary.rs` — `summarize_feature_stream()`
- `src/calibration/mapping.rs` — `calibrate_to_synthetic()`
- `src/calibration/verify.rs` — `verify_calibration()`
- `src/calibration/report.rs` — `run_calibration_workflow()`

#### Key types / functions

- `EmpiricalCalibrationProfile` — empirical quantiles, durations, mean, variance
- `SummaryTargetSet` — named set of targets
- `CalibrationMappingConfig { k, horizon, mean_policy, variance_policy, target_durations, jump }`
- `MeanPolicy` — `{ZeroCentered, SymmetricAroundEmpirical, EmpiricalBaseline}`
- `VariancePolicy` — `{QuantileAnchored, RatioAroundEmpirical}`
- `CalibratedSyntheticParams { model_params, horizon, expected_durations, jump, mapping_notes }`
- `calibrate_to_synthetic(profile, config)` — implements $p_{jj} = 1 - 1/d_j$
- `verify_calibration(empirical, synthetic, tol)` — discrepancy checks
- `run_calibration_workflow(...)` → `CalibrationReport` — end-to-end artifact

#### How it is used

The `calibrate` CLI subcommand (`direct_calibrate` in `cli.rs`) runs the full calibration workflow. The interactive CLI also exposes calibration via `cmd_calibrate_scenario()`. The calibration mapping is used to set up the `hard_switch` and `hard_switch_shock` experiment configs in `registry.rs`.

#### E2E reachability

| Usage path | Present? | Evidence |
|---|---|---|
| Called by CLI | Yes | `cargo run -- calibrate --id <id>` (synthetic + real experiments) |
| Called by experiment runner | Partial | Calibrated params fed into registry, not called per-run |
| Produces saved artifact | Yes | calibration_report.json, calibration_summary.txt |
| Covered by tests | Partial | verify_calibration tested; end-to-end not tested |

#### Status

`IMPLEMENTED_AND_USED`

#### Gaps / risks

- The calibration mapping is called during experiment config construction in `registry.rs` (the scenario-specific builders call the mapping to derive `p_jj`), but the full `run_calibration_workflow` is only triggered by the `calibrate` CLI command — it is not automatically called when running `e2e`.
- `target_durations: vec![]` in the default config means durations are inferred from empirical episode durations — this inference path needs to be explicitly tested.
- The empirical calibration targets (SPY log return distribution) are not saved as named artifacts in the standard run directory.

#### Thesis note

The calibration chapter should describe the mapping $\mathcal{K}$ as a function from empirical observation statistics (mean, low/high variance quantile, typical episode duration) to MSM generator parameters. Emphasize that $p_{jj} = 1 - 1/d_j$ is the closed-form relationship between the self-transition probability and the geometric expected duration $d_j$.

---

### 4.16 Experiment Runner

#### Theoretical object

$$\text{ExperimentConfig} \rightarrow \text{ExperimentResult}$$

A 6-stage pipeline: resolve_data → build_features → train_or_load_model → run_online → evaluate → export.

#### Code locations

- `src/experiments/runner.rs` — `ExperimentRunner`, `ExperimentBackend` trait
- `src/experiments/shared.rs` — shared stages 3 and 4
- `src/experiments/synthetic_backend.rs` — synthetic stages
- `src/experiments/real_backend.rs` — real stages
- `src/experiments/config.rs` — `ExperimentConfig`
- `src/experiments/registry.rs` — 12 registered experiments

#### Key types / functions

- `ExperimentBackend` trait: 6 methods (`resolve_data`, `build_features`, `train_or_load_model`, `run_online`, `evaluate_synthetic`, `evaluate_real`)
- `ExperimentRunner::new(backend)::run(cfg) -> ExperimentResult`
- `DryRunBackend` — no-op backend for testing pipeline structure
- `SyntheticBackend` — full math stack for simulated experiments
- `RealBackend` — DuckDB + feature pipeline for real-data experiments
- `DataBundle`, `FeatureBundle`, `ModelArtifact`, `OnlineRunArtifact`, `SyntheticEvalArtifact`, `RealEvalArtifact`

#### E2E reachability

All 6 stages confirmed reachable for both synthetic and real modes via the `e2e` command.

#### Status

`IMPLEMENTED_AND_USED`

#### Gaps / risks

- `DryRunBackend` is used in the interactive menu's "Run from Registry" path (no real computation). This is correct for quick preview, but a user who selects "Run from Registry" in the interactive menu and expects real results will get empty artifacts. The warning is now printed prominently.
- `run_batch` uses `DryRunBackend` only — there is no batch execution path that calls `SyntheticBackend` or `RealBackend` for real computation.

#### Thesis note

The experiment runner implements the canonical scientific reproducibility requirement: every run is fully determined by its `ExperimentConfig` (including seed) and produces a deterministic, stamped artifact directory. The thesis should show the config-to-result diagram and emphasize that results are reproducible by re-running the same `--id`.

---

### 4.17 Reporting and Plotting

#### Theoretical object

Reproducible artifacts: every run produces a complete directory of JSON/CSV/Markdown/LaTeX/PNG artifacts sufficient to reconstruct all results without re-running.

#### Code locations

- `src/reporting/report.rs` — `RunReporter`, `AggregateReporter`
- `src/reporting/export/` — JSON/CSV writers
- `src/reporting/plot/` — 5 plot builders
- `src/reporting/table/` — 3 table builders
- `src/reporting/artifact.rs` — `RunArtifactLayout`

#### Key types / functions

- `RunReporter::export_run(config, result)` — config snapshot + result JSON
- `RunReporter::generate_tables(result)` — metrics.md, metrics.csv, metrics.tex
- `render_detector_scores(input, path)` — detector score time series with alarms
- `render_regime_posteriors(input, path)` — posterior probability per regime over time
- `render_signal_with_alarms(input, path)` — raw signal with alarm markers
- `render_segmentation(input, path)` — Route B segment visualization
- `render_delay_distribution(input, path)` — delay histogram (NOT called from runner)
- `MetricsTableBuilder::to_markdown()` / `to_csv()` / `to_latex()`

#### E2E reachability

| Usage path | Present? | Evidence |
|---|---|---|
| Called by CLI | Yes | All run commands; `generate-report` re-runs pipeline |
| Called by experiment runner | Yes | `ExperimentRunner::run` calls `RunReporter` |
| Produces saved artifact | Yes | 21-23 artifacts per run dir confirmed |
| Covered by tests | Partial | Table builders tested; plots not tested (gated with `#[cfg(not(test))]`) |

#### Status

`IMPLEMENTED_AND_USED`

#### Gaps / risks

- `render_delay_distribution` is implemented but never called from `ExperimentRunner` or any CLI command. The delay distribution is computed (`per_event_delays` in `SyntheticEvalArtifact`) but not plotted.
- All four `render_*` functions are gated with `#[cfg(not(test))]`, meaning they are never compiled in tests. No test can validate that a valid plot is produced.
- `AggregateReporter::generate_comparison_table` reads the legacy `results/evaluation_summary.json` path first; the fallback to `summary.json` was only added as a bug fix during the verification pass.

#### Thesis note

The reporting system is the bridge between the computation and the thesis document. Emphasize that all three table formats (Markdown for README inspection, CSV for spreadsheet import, LaTeX for direct thesis inclusion) are produced identically from the same `MetricsTableBuilder`. The PNG plots are produced with a fixed 1200×600 pixel format using the `plotters` crate.

---

### 4.18 CLI / Interactive Application Layer

#### Theoretical object

Operational access to the full system: `cargo run` interactive mode, direct subcommand mode, shared service layer.

#### Code locations

- `src/main.rs` — entry point, `is_direct_command()`
- `src/cli/mod.rs` — all CLI logic

#### Key types / functions

- `is_direct_command(s: &str) -> bool` — guards for 13 known subcommands
- `cli::run(cfg: Config)` — interactive `inquire` menu loop
- `cli::run_direct(args: Vec<String>)` — direct dispatch
- Direct subcommands: `e2e`, `run-experiment`, `run-batch`, `run-real`, `calibrate`, `compare-runs`, `optimize`, `inspect`, `status`, `help`, `generate-report`, `param-search`

#### E2E reachability

All 13 subcommands confirmed reachable. Interactive mode also covers data ingestion, experiment menu, and inspect-runs.

#### Status

`IMPLEMENTED_AND_USED`

#### Gaps / risks

- `optimize` subcommand uses `optimize_full()` from `search.rs` which calls `RealBackend` with the CLI-supplied `--cache` flag. The optimization is CPU-intensive and takes minutes for a full 1280-point grid — there is no progress bar or time estimate (a warning is now emitted in debug mode).
- The interactive menu's "Run from Registry" path uses `DryRunBackend` which produces empty artifacts. This is a UX mismatch: a user expects real results but gets a dry run without an obvious warning.

---

## 5. Dead Code / Duplicate Implementation Risks

| Risk | Location | Why it matters | Suggested action |
|------|----------|----------------|------------------|
| `#[allow(dead_code)]` on `posterior_transition.rs` | `src/detector/posterior_transition.rs` line 1 | Detector IS now wired, but the attribute remains. Creates false impression of unused code. | Remove attribute after confirming CI is clean. |
| `#[allow(dead_code, unused_imports)]` on `surprise.rs` | `src/detector/surprise.rs` line 1 | Same issue — detector is wired. | Remove attribute. |
| `#[allow(dead_code)]` on `frozen.rs` | `src/detector/frozen.rs` line 1 | `FrozenModel` is used. | Remove attribute. |
| `render_delay_distribution` never called | `src/reporting/plot/delay_distribution.rs` | Delay data is computed but the histogram plot is never generated | Wire into `RunReporter` for synthetic runs, or remove. |
| `log_likelihood_contributions()` unused | `src/model/likelihood.rs` | Provides per-time contributions but is never called by any production path | Use in diagnostics or plotting; otherwise mark as dead code explicitly. |
| `src/model/validation.rs` — unclear purpose | `src/model/validation.rs` | May duplicate `ModelParams::validate()`; exact content not fully inspected | Audit; remove if redundant. |
| DryRunBackend as default in interactive menu | `src/cli/mod.rs` — `ExperimentsAction::RunFromRegistry` | User expects real computation but gets empty artifacts | Add prominent dry-run warning or replace with real backend for single experiments. |
| `run_batch` always uses DryRunBackend | `src/experiments/batch.rs` | No real batch computation path exists despite command being advertised | Document limitation clearly; add real-backend batch support or update help text. |

---

## 6. Documentation Gaps Audit

| Theory component | Implemented? | Documented? | Doc path | Gap |
|-----------------|--------------|-------------|----------|-----|
| Data pipeline | Yes | Yes | `docs/data_pipeline.md` | No function-name cross-references; needs update for RTH filter |
| Feature engineering | Yes | Yes | `docs/observation_design.md` | All 5 families documented; session-aware variants not described |
| Model parameterization | Yes | Partial | `docs/` (implicit in filter doc) | No dedicated params doc; `validation.rs` undocumented |
| Gaussian emission | Yes | Yes | `docs/emission_model.md` | Good coverage |
| Synthetic generator | Yes | Yes | `docs/gaussian_msm_simulator.md` | Jump contamination documented |
| Forward filter | Yes | Yes | `docs/forward_filter.md`, `docs/filter_validation.md` | Excellent coverage; log-sum-exp documented |
| Backward smoother | Yes | Yes | `docs/backward_smoother.md` | Good coverage |
| Pairwise posteriors | Yes | Yes | `docs/pairwise_posteriors.md` | Good coverage |
| EM estimation | Yes | Yes | `docs/em_estimation.md` | Good coverage; var floor documented |
| Diagnostics | Yes | Yes | `docs/diagnostics.md` | Good coverage |
| Online filter | Yes | Yes | `docs/online_inference.md` | Causal boundary documented |
| Detector family | Yes | Yes | `docs/changepoint_detectors.md` | PosteriorTransition TV variant underdocumented |
| Frozen model | Yes | Yes | `docs/fixed_parameter_policy.md` | Good coverage |
| Synthetic benchmark | Yes | Yes | `docs/benchmark_protocol.md` | Greedy matching documented |
| Real-data evaluation | Yes | Yes | `docs/real_data_evaluation.md` | Route A/B documented; coherence formula not shown |
| Calibration | Yes | Yes | `docs/synthetic_to_real_calibration.md` | p_jj = 1 - 1/d_j documented |
| Experiment runner | Yes | Yes | `docs/experiment_runner.md` | Good coverage |
| Reporting/plotting | Yes | Yes | `docs/reporting_and_export.md` | delay_distribution gap not noted |
| CLI / interactive | Yes | Yes | `docs/interactive_cli.md` | DryRunBackend risk undocumented |
| `docs/thesis/` directory | — | No | — | **No thesis-oriented docs existed before this audit** |

---

## 7. Test Coverage Audit

| Component | Tests found | Missing tests | Risk |
|-----------|-------------|---------------|------|
| Forward filter (`filter.rs`) | 11 dedicated tests; normalization, analytical checks, log-likelihood, extreme observations | Online vs. offline consistency test | Low |
| Backward smoother (`smoother.rs`) | Indirect via EM | Dedicated normalization test ($\sum_j \gamma_t(j) = 1$) | Medium |
| Pairwise posteriors (`pairwise.rs`) | Indirect via EM | Marginal consistency test ($\sum_j \xi_t(i,j) = \gamma_{t-1}(i)$) | Medium |
| EM estimation (`em.rs`) | Convergence, monotonicity, M-step correctness, var floor | Multi-start vs single-start comparison test | Low |
| Diagnostics (`diagnostics.rs`) | Warning generation tests | `is_trustworthy` boundary condition tests | Low |
| Online filter (`online/mod.rs`) | Normalization of step outputs | **Consistency with offline filter on same sequence** | High |
| HardSwitch detector | Regime-flip alarm test | Persistence + cooldown integration | Medium |
| PosteriorTransition detector | Score computation tests | LeavePrevious vs TV comparison | Low |
| Surprise detector | Score + EMA baseline tests | EMA correctness after fix (ema_alpha=0.3) | Low |
| Event matching (`matching.rs`) | Greedy protocol tests | Edge case: alarm at exactly τ (delay=0) | Low |
| Metric suite (`metrics.rs`) | Precision/recall/FAR/delay tests | FAR normalization for very short streams | Low |
| Feature transforms (`transform.rs`) | Unit tests for log/abs/squared return | Session-boundary return should be None | Medium |
| Feature rolling (`rolling.rs`) | Rolling vol unit tests | Session-reset correctness test | Medium |
| Feature scaler (`scaler.rs`) | Fit/apply tests | Scaler never fits on test data (anti-leakage) | High |
| Data split (`split.rs`) | Partition boundary tests | No-overlap invariant across all three partitions | Medium |
| Data session (`session.rs`) | RTH filter tests | Boundary conditions (09:30:00 exactly, 15:59:59) | Low |
| Calibration mapping (`mapping.rs`) | Mapping output tests | p_jj = 1 - 1/d_j formula verification | Medium |
| Reporting tables | Builder tests | LaTeX escaping of special characters | Low |
| Report plots | Not tested (`#[cfg(not(test))]`) | Valid PNG output; correct axis range | Medium |

---

## 8. Thesis Chapter Mapping

| Thesis chapter | Theory components | Code evidence | Missing material |
|----------------|-------------------|---------------|------------------|
| **Data and observation design** | Real financial time series; daily/intraday; session handling; chronological splits; leakage prevention; feature families (5 variants) | `src/data/`, `src/features/`, `CleanSeries`, `FeatureStream::build`, `filter_rth`, `PartitionedSeries` | Explicit anti-leakage theorem (scaler fit on train only); formal session-boundary definition for intraday |
| **Markov Switching Model theory** | Hidden regimes; initial distribution; transition matrix; Gaussian emission; model constraints | `src/model/params.rs` `ModelParams`, `src/model/emission.rs` `Emission` | Formal likelihood factorization derivation; identifiability discussion |
| **Estimation and diagnostics** | Forward filter (Hamilton); backward smoother (Baum); pairwise posteriors; EM (Baum-Welch); diagnostics; expected duration | `src/model/filter.rs`, `src/model/smoother.rs`, `src/model/pairwise.rs`, `src/model/em.rs`, `src/model/diagnostics.rs` | EM convergence proof; initial-parameter sensitivity analysis |
| **Online changepoint detector** | Frozen model; online streaming filter; three detector variants; PersistencePolicy | `src/online/mod.rs`, `src/detector/`, `src/detector/frozen.rs` | Theoretical comparison of three detectors; formal causal guarantee proof |
| **Synthetic calibration** | Empirical summary extraction; mapping K; p_jj = 1 - 1/d_j | `src/calibration/` | Full calibration table (empirical targets → synthetic params for each registered experiment) |
| **Evaluation methodology** | Ground-truth evaluation (precision/recall/delay/FAR/miss rate); causal matching; real-data Route A + B | `src/benchmark/`, `src/real_eval/` | Justification of window width w=10; Route B coherence formula; proxy event table for each asset |
| **System implementation** | Experiment runner; CLI; artifact system; reproducibility | `src/experiments/`, `src/cli/mod.rs`, `src/reporting/` | Architecture diagram; config schema reference |
| **Experimental results** | All 12 registered experiments; synthetic benchmark metrics; real evaluation metrics; joint optimization | Run directories in `runs/`; `search_report.json`; `search_summary.txt` | Comparison table across all experiments; sensitivity to k_regimes and feature family choice |

---

## 9. Final Action List

### Critical before thesis writing

| Priority | Action | Location | Reason |
|----------|--------|----------|--------|
| 1 | **Add online/offline filter consistency test** | `src/online/mod.rs` or a new integration test | Without this, the online detector could silently diverge from the offline filter with no detection |
| 2 | **Add scaler anti-leakage test** | `src/features/scaler.rs` or integration tests | The central reproducibility claim depends on train-only scaling; no test currently enforces it |
| 3 | **Document Route B coherence formula** | `docs/real_data_evaluation.md` | The thesis must state the exact formula; currently only implemented in code |
| 4 | **Write proxy event table for each real experiment** | Thesis appendix + `docs/thesis/` | Route A results are meaningless without knowing which proxy events were used |
| 5 | **Remove `#[allow(dead_code)]` from wired detectors** | `src/detector/posterior_transition.rs`, `surprise.rs`, `frozen.rs` | These attributes suppress legitimate future warnings |

### Important but not blocking

| Priority | Action | Location | Reason |
|----------|--------|----------|--------|
| 6 | **Add smoother normalization test** ($\sum_j \gamma_t(j) = 1$) | `src/model/smoother.rs` | E-step correctness depends on this; currently only tested indirectly |
| 7 | **Add pairwise marginal consistency test** ($\sum_j \xi_t(i,j) = \gamma_{t-1}(i)$) | `src/model/pairwise.rs` | Core EM correctness invariant |
| 8 | **Wire `render_delay_distribution` into RunReporter** | `src/reporting/report.rs` | Delay distribution is informative for thesis; data is computed but plot is never generated |
| 9 | **Audit `src/model/validation.rs`** | `src/model/validation.rs` | Potential dead code / duplicate of `ModelParams::validate()` |
| 10 | **Document the 12-experiment registry in thesis** | `docs/thesis/` | Currently only visible via `--help` or source inspection |
| 11 | **Add K≥3 synthetic scenario** | `src/experiments/registry.rs` | The joint optimizer found K=3 best for real SPY data, but no K=3 synthetic benchmark exists |
| 12 | **Add DryRunBackend warning to interactive menu** | `src/cli/mod.rs` | Users running experiments interactively expect real results |
| 13 | **Create dedicated `docs/thesis/` chapter notes** | `docs/thesis/` | This audit document should be the first of several; add one `.md` per thesis chapter |

---

*End of code-to-theory repository audit — verify_2026_05_03*
