# Chapter 5 — Data and Observation Design

**Scope:** `src/alphavantage/`, `src/cache/`, `src/data_service/`, `src/data/`, `src/features/`, plus the `resolve_data` / `build_features` stages of `src/experiments/real_backend.rs`. This chapter covers every code path between an external vendor response and the model-ready observation sequence `y_{1:T}` consumed by the EM trainer (Chapter 3) and the online detector (Chapter 4).

**Relation to previous chapters:** Chapter 2 fixed the Gaussian MSM, Chapter 3 estimated its parameters by EM, and Chapter 4 used the resulting filter posterior to build a real-time detector. All three chapters assumed an abstract observation sequence `y_{1:T}` was already available. This chapter defines exactly how that sequence is constructed from raw market data, why each transformation step is necessary, and what guarantees the pipeline provides so that downstream evaluation results are scientifically reproducible.

**Source files analysed and cross-checked with docs:**

| Source file | Purpose | Cross-referenced doc |
|---|---|---|
| `src/alphavantage/client.rs` | REST client, intraday pagination | `docs/alphavantage_client.md` |
| `src/alphavantage/commodity.rs` | Endpoint enum, response parsing, unit normalisation | `docs/alphavantage_client.md` |
| `src/alphavantage/rate_limiter.rs` | Token-bucket rate limiter | `docs/alphavantage_client.md` |
| `src/cache/mod.rs` | DuckDB persistence layer | `docs/duckdb_cache.md` |
| `src/data_service/mod.rs` | Cache–API orchestrator | `docs/data_service.md` |
| `src/data/mod.rs` | `Observation`, `CleanSeries` types | `docs/data_pipeline.md` |
| `src/data/meta.rs` | `DataMode`, `DatasetMeta`, `PriceField` | `docs/data_pipeline.md` |
| `src/data/validation.rs` | Ordering, dedup, gap detection | `docs/data_pipeline.md` |
| `src/data/session.rs` | RTH filter, session labelling | `docs/data_pipeline.md` |
| `src/data/split.rs` | Chronological train/val/test split | `docs/data_pipeline.md` |
| `src/features/family.rs` | `FeatureFamily` enum (LogReturn, AbsReturn, …) | `docs/observation_design.md` |
| `src/features/transform.rs` | Causal pointwise transforms | `docs/observation_design.md` |
| `src/features/rolling.rs` | Trailing-window rolling statistics | `docs/observation_design.md` |
| `src/features/scaler.rs` | Leakage-safe z-score scaling | `docs/observation_design.md` |
| `src/features/stream.rs` | `FeatureStream` assembly | `docs/observation_design.md` |
| `src/experiments/real_backend.rs` | `resolve_data` + `build_features` wiring | `docs/experiment_runner.md` |

---

## 5.1 Data sources

Three classes of asset are supported, all served via the [Alpha Vantage](https://www.alphavantage.co/) REST API and cached locally:

| Asset class | Examples | Endpoint family | Frequency |
|---|---|---|---|
| Commodities | WTI, Brent, Natural Gas, Copper, Aluminum, Wheat, Corn, Cotton, Sugar, Coffee, Gold, Silver | `WTI`, `BRENT`, …, `GOLD_SILVER_HISTORY` | Daily |
| Equity ETFs (EOD) | SPY, QQQ | `TIME_SERIES_DAILY_ADJUSTED` | Daily |
| Equity ETFs (intraday) | SPY, QQQ | `TIME_SERIES_INTRADAY` | 5-min, 15-min |

The endpoint enum is defined once in [src/alphavantage/commodity.rs](src/alphavantage/commodity.rs):

```rust
pub enum CommodityEndpoint {
    Wti, Brent, NaturalGas, Copper, Aluminum,
    Wheat, Corn, Cotton, Sugar, Coffee,
    AllCommodities, Gold, Silver,
    Spy, Qqq,
}
```

For commodity endpoints, `function_name()` maps directly to the Alpha Vantage `function` query parameter. For equity endpoints, `equity_function_name(interval)` selects the correct adjusted-close function based on whether the interval is daily or intraday. **Adjusted close** is mandatory for ETFs so that dividends and splits do not appear as artificial price jumps; the choice is enforced one layer up in `RealBackend::to_price_field` ([src/experiments/real_backend.rs](src/experiments/real_backend.rs#L110-L115)):

```rust
fn to_price_field(endpoint: &CommodityEndpoint) -> PriceField {
    match endpoint {
        CommodityEndpoint::Spy | CommodityEndpoint::Qqq => PriceField::AdjustedClose,
        _                                                => PriceField::Value,
    }
}
```

For commodity series the only available field is the spot/settlement `value`, so the distinction is moot.

### 5.1.1 The DuckDB local cache

Every successful API response is persisted to a DuckDB file (default `data/commodities.duckdb`). The schema, in [src/cache/mod.rs](src/cache/mod.rs#L41-L65), has two tables:

```sql
CREATE TABLE commodity_meta (
    symbol VARCHAR, interval VARCHAR, name VARCHAR, unit VARCHAR,
    PRIMARY KEY (symbol, interval)
);
CREATE TABLE commodity_prices (
    symbol VARCHAR, interval VARCHAR, date TIMESTAMP,
    value DOUBLE, fetched_at TIMESTAMP,
    PRIMARY KEY (symbol, interval, date)
);
```

`CommodityCache::store` truncates and re-loads each `(symbol, interval)` series via DuckDB's bulk `Appender` so that ingesting full-history intraday data (~378k bars for SPY 15-min since 2000) completes in seconds. `CommodityCache::status` is what the `cargo run -- status` CLI invocation reads to print the inventory.

Cross-references: [docs/duckdb_cache.md](docs/duckdb_cache.md), [docs/alphavantage_client.md](docs/alphavantage_client.md).

### 5.1.2 The cache-first data service

[src/data_service/mod.rs](src/data_service/mod.rs) defines `DataService`, the orchestrator that decides whether a request is satisfied from the cache or escalated to the API:

- `load_cached(endpoint, interval)` — read-only, never hits the network. This is the path used by every model experiment, ensuring training is fully reproducible offline.
- `refresh(endpoint, interval)` — force re-fetch, used by the `ingest` workflow.
- `ingest_all(series, force)` — bulk refresh of every series listed under `[ingest]` in `config.toml`; skips fresh series unless `force = true`.

This separation guarantees that **no experiment ever issues an API call**; all training and evaluation reads exclusively from DuckDB.

---

## 5.2 Daily and intraday data modes

The pipeline distinguishes two modes via [src/data/meta.rs](src/data/meta.rs):

```rust
pub enum DataMode {
    Daily,
    Intraday { bar_minutes: u32 },
}
```

The mode controls four downstream behaviours:

| Behaviour | Daily | Intraday |
|---|---|---|
| Timestamp granularity | Date at midnight | Bar-open time in US Eastern Time |
| Gap detection | Disabled (calendar gaps are normal) | Flagged when Δt > 3× expected bar duration |
| RTH filter | N/A | Applied when `session_aware = true` |
| Rolling-window session reset | N/A | Optional, controlled by `session_reset` |

Mode selection is hard-wired per experiment in the registry (`RealFrequency::Daily`, `Intraday5m`, `Intraday15m`) and translated by `RealBackend::to_data_mode`. There is no path by which daily and intraday data can be silently mixed.

---

## 5.3 Data cleaning and timestamp handling

The cleaning pipeline is implemented in [src/data/validation.rs](src/data/validation.rs) and executed in `CleanSeries::from_response` ([src/data/mod.rs](src/data/mod.rs)). Four operations are performed in order:

1. **Ordering check.** Record whether the input was already ascending (Alpha Vantage returns newest-first, so this flag is almost always `false` and is reported in `ValidationReport::had_unsorted_input`).
2. **Stable ascending sort by timestamp.** Required because EM, the forward filter, and the online detector all assume strictly increasing time.
3. **Deduplication.** After the stable sort, repeated timestamps are collapsed with a keep-first policy. This is essential for intraday data, where the month-by-month pagination of `fetch_equity_intraday_full` in [src/alphavantage/client.rs](src/alphavantage/client.rs) can return the same bar twice at month boundaries.
4. **Intraday gap detection.** For `DataMode::Intraday`, any pair of consecutive observations separated by more than `3 × bar_minutes` is recorded as a `Gap`. **Gaps are reported but never filled** — the pipeline does not impute price dynamics that the vendor did not report.

The resulting `ValidationReport` is serialised to `data_quality.json` in every real-data run artifact. The contract is documented in module-level prose:

```rust
// src/data/validation.rs
// "The validator never silently drops valid data; it only removes
//  provably duplicate entries."
```

### 5.3.1 Timezone contract

`docs/data_pipeline.md` makes this explicit and the code matches it:

- **Daily** timestamps are stored as midnight `NaiveDateTime`. They represent a calendar trading date with no time-of-day meaning.
- **Intraday** timestamps are bar-open times in **US Eastern Time** as returned by Alpha Vantage. No conversion is performed inside the pipeline.

The contract is invariant: callers must not mix sources with different conventions. There is no timezone-conversion code anywhere in `src/data/`, and the `chrono::NaiveDateTime` type is used precisely because it carries no timezone (forcing the discipline at compile time).

### 5.3.2 Intraday session handling

For intraday equity data, [src/data/session.rs](src/data/session.rs) provides:

- `is_rth_bar(ts)` — `true` iff `09:30:00 ≤ time < 16:00:00` ET.
- `filter_rth(obs)` — keeps only RTH bars; pre-market and after-hours are dropped.
- `label_sessions(obs)` — partitions the survivor sequence into `SessionBoundary` records, one per calendar trading day.

The RTH filter is applied when the experiment registers `session_aware = true`. The verification run for `real_spy_intraday_hard_switch` shows the effect: starting from 378 149 raw 15-minute bars, the filter reduces the model-ready stream to roughly 25 075 RTH bars — about a 14× reduction, matching the ratio of the 6.5-hour trading day to the full 24-hour calendar bar count.

Overnight gaps between 15:59 ET on day $d$ and 09:30 ET on day $d+1$ are **not** hidden. They appear in the timestamp sequence but produce no return crossing the boundary because of the session-aware feature primitive described in §5.6.

---

## 5.4 Train–validation–test split protocol

Splitting is performed in [src/data/split.rs](src/data/split.rs) by `PartitionedSeries::from_series`. The contract is uniform across every experiment:

- Splits are **chronological**, defined by two timestamps `train_end` and `val_end`.
- An observation at time `t` is assigned by:
  - **Train**: `t < train_end`
  - **Validation**: `train_end ≤ t < val_end`
  - **Test**: `t ≥ val_end`

```rust
pub struct SplitConfig {
    pub train_end: NaiveDateTime,
    pub val_end:   NaiveDateTime,
}
```

The split occurs **before any feature construction** so that the scaler (§5.6.3) sees only training-partition information. Random shuffling is forbidden by construction — there is no shuffle code path in the module.

### 5.4.1 Default 70/15/15 split

For real experiments, [src/experiments/real_backend.rs](src/experiments/real_backend.rs#L289-L300) selects timestamp-anchored cut points at the 70th and 85th percentile of the sorted series:

```rust
let train_idx = (n as f64 * 0.70) as usize;
let val_idx   = (n as f64 * 0.85) as usize;
let train_end_ts = timestamps[train_idx];
let val_end_ts   = timestamps[val_idx];
```

The exact boundaries used for each run are serialised to `split_summary.json` (asset, train_end, val_end, partition sizes, first timestamp of each segment). The verification run for SPY daily (2018-01-02 → present) yielded `n_train = 1463`, `n_validation = 314`, `n_test = 314`, with `train_end = 2023-10-25` and `val_end = 2025-01-28`.

### 5.4.2 The leakage-prevention contract

The split is the foundation of the leakage-prevention contract stated in module-level documentation:

> "Preprocessing steps that depend on data statistics (e.g., mean, variance normalization) must be fit on the train partition only, then applied to validation and test."

This contract is enforced in code by passing `n_train` (the post-warmup training size) through `FeatureConfig::n_train` to the scaler (§5.6.3) and to the EM trainer. There is no API surface that would allow a downstream step to recompute its parameters using validation or test data.

---

## 5.5 Observation design

Once a `CleanSeries` is split into partitions, the value sequence is still raw prices. The model assumes

$$y_t \mid S_t = j \sim \mathcal{N}(\mu_j, \sigma_j^2)$$

but says nothing about what `y_t` is. The choice of transformation is the **observation design** decision, formalised in [src/features/family.rs](src/features/family.rs):

```rust
pub enum FeatureFamily {
    LogReturn,
    AbsReturn,
    SquaredReturn,
    RollingVol         { window: usize, session_reset: bool },
    StandardizedReturn { window: usize, epsilon: f64, session_reset: bool },
}
```

Each variant has a documented warmup requirement (`FeatureFamily::warmup_bars`):

| Family | Symbol | Warmup (bars) | Detector sensitive to |
|---|---|---|---|
| `LogReturn` | $r_t$ | 1 | Mean and dispersion of returns |
| `AbsReturn` | $\lvert r_t \rvert$ | 1 | Return magnitude / activity |
| `SquaredReturn` | $r_t^2$ | 1 | Second-moment structure |
| `RollingVol` | $v_t^{(w)}$ | $w$ | Recent volatility level |
| `StandardizedReturn` | $z_t$ | $w$ | Normalised shock size |

Comparing these families across the same asset and detector is itself an empirical contribution of the thesis (cf. [docs/observation_design.md](docs/observation_design.md) §1). The verification pass exercises `LogReturn`, `AbsReturn`, `SquaredReturn`, `RollingVol{5}`, `RollingVol{20}` through the joint-optimisation grid search (1 280 grid points, see Step 18 of `verify_2026_05_03b`).

### 5.5.1 Why raw prices are unsuitable

[docs/observation_design.md](docs/observation_design.md) §2 enumerates four pathologies (non-stationarity, scale dependence, drift, level asymmetry) that disqualify $y_t = P_t$ as a default observation. The code makes raw prices structurally inaccessible: no `FeatureFamily` variant produces `value` unchanged, and there is no parameter in `ExperimentConfig` that selects "no transformation".

---

## 5.6 Log returns

The log-return transformation is the canonical default. It is implemented as a single pointwise primitive in [src/features/transform.rs](src/features/transform.rs):

```rust
#[inline]
pub fn log_return(prev_price: f64, curr_price: f64) -> Option<f64> {
    if prev_price <= 0.0 || curr_price <= 0.0 { None }
    else { Some(curr_price.ln() - prev_price.ln()) }
}
```

Definition:

$$r_t = \log P_t - \log P_{t-1}, \qquad t = 1, \dots, n$$

The module-level documentation states the causality contract explicitly:

> "Every function in this module is **causal**: it reads only the current and past price values; it never looks ahead. This is a hard requirement for the online detection setting."

Three properties of `log_return` make it the right primitive:

1. **Approximately scale-free.** A 1 % move on SPY at 200 USD and at 600 USD produces the same `r_t`.
2. **Approximately stationary** under mild conditions, which matches the i.i.d.-within-regime assumption of the Gaussian MSM.
3. **Additive.** Multi-period returns sum: $r_{t-w+1} + \dots + r_t = \log P_t - \log P_{t-w}$. This makes rolling-window statistics interpretable.

### 5.6.1 The `None` guard

`log_return` returns `None` for non-positive prices. This is not paranoia: occasional vendor anomalies (zero or negative values) would otherwise inject `-inf` into the observation stream and cascade through the filter. The batch wrapper `log_returns` in the same module drops `None` results with the original timestamps preserved, so the warmup count is exact.

### 5.6.2 Session-aware variant

For intraday data with `session_aware = true`, the variant `log_returns_session_aware` consults `different_day(prev_ts, curr_ts)` and emits `None` whenever the two timestamps fall on different calendar days. This **explicitly suppresses overnight returns**, which would otherwise dominate the intraday variance and create spurious changepoints at every 09:30 session open.

The choice is documented in [src/features/transform.rs](src/features/transform.rs):

> "For intraday data, the caller is responsible for deciding whether session-crossing returns are allowed. If `allow_cross_session = false`, the functions that take `prev_value` should receive `None` at each session-open bar, which will cause that bar to be skipped (no return emitted)."

---

## 5.7 Volatility-related features

Three families produce volatility-related observations.

### 5.7.1 Absolute and squared returns

`|r_t|` and `r_t^2` are computed pointwise in [src/features/transform.rs](src/features/transform.rs):

```rust
pub fn abs_return(prev, curr)     -> Option<f64> { log_return(prev, curr).map(f64::abs) }
pub fn squared_return(prev, curr) -> Option<f64> { log_return(prev, curr).map(|r| r * r) }
```

Both have warmup 1 and are causal in the same sense as `log_return`. The squared return is particularly relevant when comparing against the GARCH-style literature: $r_t^2$ is the natural one-step volatility proxy under a zero-conditional-mean assumption.

### 5.7.2 Rolling volatility

[src/features/rolling.rs](src/features/rolling.rs) implements trailing-window volatility:

$$v_t^{(w)} = \sqrt{\frac{1}{w} \sum_{k=0}^{w-1} (r_{t-k} - \bar r_t^{(w)})^2}$$

Two design choices documented in the module:

- **Population normalisation** ($1/w$, not $1/(w-1)$) because the window is treated as a finite-sample volatility proxy, not an unbiased variance estimator.
- **Ring-buffer accumulator** (`RollingStats`) — $O(w)$ per update, simpler and numerically better-conditioned than a Welford-style $O(1)$ update for the bar counts in this project.

Warmup is $w$ price observations (i.e., $w$ returns). The first $w-1$ feature values are `None` and trimmed.

### 5.7.3 Standardised returns

$$z_t = \frac{r_t}{v_t^{(w)} + \varepsilon}, \qquad \varepsilon = 10^{-8}\ \text{(default)}$$

This is the most "engineered" of the families: the denominator is itself a feature, so the observation is already partially regime-corrected. Switching to `StandardizedReturn` typically reduces detector sensitivity to ordinary high-volatility periods while preserving sensitivity to genuinely *new* dynamics.

### 5.7.4 Session reset

Both `RollingVol` and `StandardizedReturn` carry a `session_reset: bool` flag. When `true` (intraday data only), the ring buffer is cleared at every calendar-day boundary. This prevents the prior session's closing volatility from contaminating the next session's first $w$ bars and is the rolling-window analogue of the `session_aware` log-return policy in §5.6.2.

---

## 5.8 Causality and leakage prevention

The pipeline has four leakage-prevention guarantees, each enforced by a different code-level mechanism.

### 5.8.1 No future information in features

Every primitive in [src/features/transform.rs](src/features/transform.rs) and [src/features/rolling.rs](src/features/rolling.rs) accepts the *current* observation and possibly *past* observations (via ring buffer or a single `prev_*` argument). None accepts a future observation. The module-level prose makes this explicit and the function signatures make it structural.

### 5.8.2 Scaler fitted only on training partition

[src/features/scaler.rs](src/features/scaler.rs) defines `FittedScaler` constructed via `fit_on(train_values)` and applied via `apply(value)`. The scaler-fitting flow inside `FeatureStream::build` ([src/features/stream.rs](src/features/stream.rs)) consumes only the first `n_train` post-warmup feature values:

> "If anything other than `ScalingPolicy::None`, the scaler is fitted on the first `n_train` observations of the feature series."

```rust
pub struct FeatureConfig {
    pub family:        FeatureFamily,
    pub scaling:       ScalingPolicy,
    pub n_train:       usize,   // — fit scaler on observations [0..n_train)
    pub session_aware: bool,
}
```

The fitted `FittedScaler` is then frozen and applied uniformly to training, validation, test, and the online stream. There is no API method `refit_on(...)` on `FittedScaler` — re-estimation is structurally impossible.

For the degenerate case where the training-partition standard deviation is zero, the scaler falls back to the identity (no division by zero).

### 5.8.3 Chronological split before feature construction

The leakage contract requires `PartitionedSeries::from_series` to run before any feature pipeline. `RealBackend::resolve_data` enforces this order: prices are split, then the timestamp boundaries `train_end_ts` / `val_end_ts` are passed to feature construction through `n_train`. Any attempt to reverse the order would have to bypass the `ExperimentBackend` trait.

### 5.8.4 Frozen model parameters during online evaluation

This is the Chapter 4 contract restated here for completeness. After EM fits the model on the training partition, `FrozenModel` exposes no setter and `StreamingSession` consumes only `filter_state` and `detector` as mutable. The detector therefore cannot adapt to the validation or test distribution.

---

## 5.9 Final model-ready observation streams

The `build_features` stage of [src/experiments/real_backend.rs](src/experiments/real_backend.rs) produces the `FeatureBundle` consumed by Chapters 3 and 4:

```rust
pub struct FeatureBundle {
    pub feature_label: String,
    pub n_observations: usize,
    pub observations: Vec<f64>,
    pub train_n: usize,
    pub timestamps: Vec<chrono::NaiveDateTime>,
}
```

### 5.9.1 End-to-end transformation flow

```
CommodityResponse              (vendor JSON: newest-first)
        │
        │  CleanSeries::from_response
        │    sort↑ · dedup · gap-check · annotate
        ▼
CleanSeries                    (ascending, gap-reported, mode-tagged)
        │
        │  filter_rth (only when session_aware = true and DataMode::Intraday)
        ▼
CleanSeries (RTH only)         (~14× reduction for 15-min equity data)
        │
        │  PartitionedSeries::from_series  with SplitConfig { train_end, val_end }
        ▼
PartitionedSeries              (train / validation / test, contract: t < train_end → Train)
        │
        │  FeatureStream::build(series, FeatureConfig { family, scaling, n_train, session_aware })
        │    compute family · trim warmup · fit scaler on train only · apply to all
        ▼
FeatureStream                  (observations: Vec<f64>, scaler: FittedScaler frozen)
        │
        │  packaged into FeatureBundle
        ▼
FeatureBundle                  → EM (Chapter 3) → FrozenModel → online detector (Chapter 4)
```

### 5.9.2 Provenance preserved at every step

Three serialised artifacts make the pipeline auditable:

| Artifact | Source | Contents |
|---|---|---|
| `data_quality.json` | `ValidationReport` from §5.3 | `n_input`, `n_dropped_duplicates`, `had_unsorted_input`, `gaps` |
| `split_summary.json` | `PartitionedSeries` from §5.4 | `n_train`, `n_validation`, `n_test`, `train_end`, `val_end`, first timestamps |
| `feature_summary.json` | `FeatureStreamMeta` from §5.6 | `feature_label`, `n_feature_observations`, `train_n`, `warmup_trimmed`, scaling stats |

The verification run for `real_spy_daily_hard_switch` shows the chain end-to-end:
- `n_input = 2091` (raw daily bars 2018–present, post-clean) → `n_output = 2091`, no duplicates, no gaps reported.
- Split: `n_train = 1463`, `n_validation = 314`, `n_test = 314`.
- Features: `LogReturn`, `ZScore` scaling, `warmup_trimmed = 1`, `train_n = 1462`, `n_feature_observations = 2090`, scaled `obs_std ≈ 0.934` on the validation+test partitions (training partition was z-scored to ≈ 1.0 by construction).

### 5.9.3 What the model sees

By the time `y_{1:T}` is handed to EM:

1. **Strict chronological order.**
2. **No future information in any single `y_t`.**
3. **No partition leakage** — the scaler used to produce `y_t` for `t ≥ train_end` was fitted on `t < train_end`.
4. **Stationarity-friendly transformation** — `LogReturn` (or one of the four variants), never raw prices.
5. **Session-aware handling** for intraday data — no spurious overnight returns.
6. **Explicit warmup record** — the first `warmup_bars` price observations were consumed to produce the first feature value, and this is logged in `feature_summary.json`.

Every guarantee in the list is asserted by a separate module, and every module is unit-tested (see the test counts cited at the end of each `docs/*.md` for `data/`, `data_service/`, `features/`). The end-to-end test suite at the time of `verify_2026_05_03b` ran 343 tests with 0 failures across these modules and their consumers.

This is the observation sequence on which all subsequent results in this thesis are computed.
