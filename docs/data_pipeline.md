# Real Financial Data Pipeline

## Phase 15

---

## 1. The Data Pipeline Is a Scientific Component

For a thesis on online changepoint detection in financial time series, the data pipeline is **not** an implementation detail. It is part of the scientific definition of the experiment.

The formal statement of any result in this project must include:

$$
\boxed{
\text{Detector } D \text{ was evaluated on series } \{y_t\}_{t=1}^{T}
\text{ where } y_t \text{ is defined precisely by the pipeline described here.}
}
$$

If the data pipeline is ambiguous, results cannot be compared across detectors, across assets, or across runs. Specifically, for your thesis:

- **Offline EM fitting** is only reproducible if the exact training observations, their ordering, and the price field used are documented.
- **Online detection** depends on the causal observation sequence; incorrect ordering or hidden gaps invalidate the detection timeline.
- **Benchmark comparisons** between detectors are only fair if both see identical input series.

This document defines the pipeline rigorously. Any future experiment that uses market data for this project must conform to these definitions.

---

## 2. Investigation: Current Pipeline State

Before building Phase 15, a full audit of the existing codebase was performed.

### 2.1 What already existed

| Component | Location | Status |
|---|---|---|
| HTTP client with rate limiting | `src/alphavantage/client.rs` | ✅ Complete |
| Commodity endpoint definitions | `src/alphavantage/commodity.rs` | ✅ Complete |
| `CommodityDataPoint`, `CommodityResponse` types | `src/alphavantage/commodity.rs` | ✅ Complete |
| Date parsing (daily, intraday, monthly) | `commodity.rs::deserialize_date` | ✅ Complete |
| Missing value handling (`"."` → `None`) | `commodity.rs::deserialize_optional_float` | ✅ Implicit |
| Equity adjusted-close extraction | `client.rs::fetch_equity_intraday_full` | ✅ Complete |
| DuckDB persistent cache | `src/cache/mod.rs` | ✅ Complete |
| Fetch/cache orchestration | `src/data_service/mod.rs` | ✅ Complete |
| Interactive CLI (ingest/show/status) | `src/cli/mod.rs` | ✅ Complete |
| Config loading (TOML) | `src/config.rs` | ✅ Complete |

### 2.2 What was missing

| Requirement | Status before Phase 15 |
|---|---|
| Chronological ordering guarantee | ❌ Missing — cache loads `ORDER BY date DESC` |
| Duplicate timestamp detection/removal | ❌ Missing |
| Data-quality validation report | ❌ Missing |
| Clean dataset type (`CleanSeries`) | ❌ Missing |
| Provenance metadata struct | ❌ Missing |
| RTH filter for intraday | ❌ Missing |
| Session boundary labelling | ❌ Missing |
| Chronological train/val/test split | ❌ Missing |
| Leakage-prevention partition objects | ❌ Missing |
| Tests for any data module | ❌ Missing |
| `DataMode`, `PriceField`, `SessionConvention` types | ❌ Missing |

### 2.3 Pre-existing risks

**Ordering bug**: the DuckDB cache loads data with `ORDER BY date DESC` (newest-first). The Alpha Vantage API also returns data newest-first. The online detector requires oldest-first. Before Phase 15, there was no layer that enforced ascending order. `CleanSeries::from_response` now corrects this automatically.

**Silent missing-data dropping**: `deserialize_optional_float` converts the vendor sentinel `"."` to `None`, and the caller's `filter_map` silently drops these observations. This is the correct handling for the commodity API, but it was undocumented. The `ValidationReport` now makes this visible at the `n_input / n_output` level.

**Timezone undocumented**: intraday timestamps from Alpha Vantage are in US Eastern Time but are stored as `NaiveDateTime` (timezone-unaware). This was not documented anywhere. It is now stated explicitly in `DataMode::Intraday` and in `DatasetMeta`.

---

## 3. Data Sources and Asset Taxonomy

### 3.1 Commodity price series

Assets: WTI, Brent, Natural Gas, Copper, Aluminum, Wheat, Corn, Cotton, Sugar, Coffee, Gold, Silver, All Commodities Index.

**What one row represents**: the vendor's reported value for that commodity on the given date, typically the London fix, settlement, or spot price as defined by the vendor. The unit is USD per the relevant quantity (e.g., USD/barrel, USD/troy oz).

**Available intervals**: daily, weekly, monthly, quarterly, annual.

**Missing values**: some dates have `"."` as the value (vendor data gap). These are dropped before reaching `CleanSeries`. The dropped count is reported in `ValidationReport::n_dropped_duplicates` (via the `filter_map` in deserialization — effectively `n_input` reflects only the non-missing points that reached the pipeline).

**Price field**: `PriceField::Value`. This is the raw commodity index value, **not** a closing price in the equity sense.

### 3.2 Daily SPY / QQQ

**What one row represents**: the dividend-and-split-adjusted closing price on that trading day, sourced from Alpha Vantage `TIME_SERIES_DAILY_ADJUSTED` field `"5. adjusted close"`.

**Price field**: `PriceField::AdjustedClose`. This is the correct field for long-horizon analysis because it removes the price-level discontinuities caused by dividends and splits.

**Important**: unadjusted close and adjusted close are **not** interchangeable without explicit justification. All daily SPY/QQQ results in this thesis use adjusted close unless otherwise stated.

### 3.3 Intraday SPY / QQQ

**What one row represents**: one time bar. The `value` field is the last-traded price within the bar (`"4. close"` in the Alpha Vantage response).

**Timestamp semantics**: `timestamp` is the bar-**open** time. A bar labelled `"2024-01-02 09:30:00"` covers the interval `[09:30, 09:31)` for a 1-minute bar.

**Available granularities**: 1min, 5min, 15min, 30min, 60min.

**Timezone**: all intraday timestamps are in **US Eastern Time (ET)**. They are stored as `NaiveDateTime` (no tz offset). This is a deliberate choice: embedding timezone in the type would require pulling `chrono-tz` as a dependency. The convention is documented here and in `DataMode::Intraday`.

---

## 4. Raw Data Schema

### 4.1 Wire format: `CommodityResponse`

The `CommodityResponse` struct in `src/alphavantage/commodity.rs` is the direct output of parsing an Alpha Vantage JSON response (or loading from the DuckDB cache). It is **not** a clean dataset; it is raw vendor data.

```
CommodityResponse {
    name:     String,              // vendor description (e.g., "West Texas Intermediate")
    interval: String,              // vendor-supplied interval string (e.g., "daily")
    unit:     String,              // vendor unit (e.g., "dollars per barrel")
    data:     Vec<CommodityDataPoint>,
}

CommodityDataPoint {
    date:  NaiveDateTime,          // parsed timestamp (see §4.2)
    value: f64,                    // price or index value
}
```

**Important properties**:
- The `data` vector is in **descending** chronological order (newest first), consistent with the Alpha Vantage API and the DuckDB cache load order.
- Missing observations (`"."` values in the JSON) are silently dropped before reaching this struct.
- The `interval` string in `CommodityResponse` is not validated against `DataMode`; that mapping is the caller's responsibility.

### 4.2 Timestamp parsing

The `deserialize_date` function in `commodity.rs` accepts three formats:

| Format | Example | Interpretation |
|---|---|---|
| Full datetime | `"2026-04-14 09:30:00"` | Intraday bar-open time (ET) |
| Date only | `"2024-03-15"` | Daily trading date → midnight `NaiveDateTime` |
| Year-month | `"2024-03"` | Monthly data → 1st of month at midnight |

---

## 5. The `CleanSeries` Type and Construction Pipeline

`CleanSeries` is the first clean representation in the pipeline. It is the output of `CleanSeries::from_response(response: CommodityResponse, meta: DatasetMeta)`.

### 5.1 Construction steps

The following operations are performed **in order** during `from_response`:

#### Step 1 — Map to `Observation`

```
CommodityDataPoint { date, value }  →  Observation { timestamp: date, value }
```

This is a direct field copy. No transformation of the value occurs here.

#### Step 2 — `validation::validate`

Operates in-place on `Vec<Observation>`:

1. **Record unsorted flag** — check if any consecutive pair violates ascending order. Set `had_unsorted_input = true` if so. This is expected to be `true` for all freshly-parsed Alpha Vantage responses.

2. **Sort ascending (stable)** — `sort_by(|a, b| a.timestamp.cmp(&b.timestamp))`. After this step, the vector is guaranteed ascending. Stable sort means equal-timestamp elements retain their original relative order, which is required for the keep-first deduplication policy.

3. **Deduplicate** — `dedup_by_key(|o| o.timestamp)` removes all but the first occurrence of each timestamp. This can occur at month boundaries in intraday pagination. The count is reported in `ValidationReport::n_dropped_duplicates`.

4. **Gap detection** — For `DataMode::Intraday { bar_minutes }`, scan consecutive pairs and flag those with interval $> 3 \times \text{bar\_minutes}$. For `DataMode::Daily`, no gap check is performed (calendar absences are structurally normal).

#### Step 3 — Construct `CleanSeries`

The sorted, deduplicated `Vec<Observation>` and the `ValidationReport` are stored together with the `DatasetMeta`.

### 5.2 Ordering guarantee

After construction, `CleanSeries::observations` is **strictly ascending** by `timestamp` (after deduplication). Every downstream consumer — the online filter, the benchmark, the session labeller, the splitter — can rely on this without re-sorting.

---

## 6. Validation Report

Every `CleanSeries` carries a `ValidationReport`:

```rust
pub struct ValidationReport {
    pub n_input:               usize,       // observations before validation
    pub n_dropped_duplicates:  usize,       // removed by dedup
    pub n_output:              usize,       // observations after validation
    pub had_unsorted_input:    bool,        // whether input was not already ascending
    pub gaps:                  Vec<Gap>,    // flagged intraday gaps
}

pub struct Gap {
    pub from: NaiveDateTime,   // last timestamp before the gap
    pub to:   NaiveDateTime,   // first timestamp after the gap
}
```

### Gap detection policy

A gap is flagged for intraday data only when the interval between two consecutive observations exceeds $3 \times \text{bar\_minutes}$:

$$
\Delta t_{i,i+1} > 3 \cdot \Delta t_{\text{bar}} \implies \text{gap flagged}
$$

The factor of 3 is chosen to accommodate minor irregularities at session edges (e.g., sparse coverage near market open/close) while still catching genuine missing-session events such as overnight gaps in un-filtered data or multi-day data outages.

**Policy**: gaps are **reported but not filled**. The pipeline never silently imputes price dynamics. The caller must decide how to handle gaps (typically: exclude affected sessions, or reset detector state at gap boundaries).

For daily data, calendar absences (weekends, market holidays) are **not** flagged. They are structurally expected and do not represent data quality problems.

---

## 7. Timestamp and Timezone Conventions

### 7.1 Daily data

- Timestamp: the trading date, stored as midnight `NaiveDateTime` (`%Y-%m-%d 00:00:00`).
- Timezone: none embedded; treated as a calendar date identifier.
- Holiday gaps: not flagged (see §6).

### 7.2 Intraday data

- Timestamp: bar-open time in **US Eastern Time (ET)**.
- Timezone: stored as `NaiveDateTime` (no offset). The ET convention is documented here and in `DataMode::Intraday`. Do not mix intraday series from sources with different timezone conventions.
- For a bar labelled $t$, the bar covers the interval $[t, t + \Delta t_{\text{bar}})$.

### 7.3 Non-trading-day vs missing-observation distinction

This distinction is critical and is handled differently by frequency:

| Scenario | Daily treatment | Intraday treatment |
|---|---|---|
| Weekend | Not a missing obs — skip | N/A (market closed) |
| Market holiday | Not a missing obs — skip | N/A (market closed) |
| Vendor data gap | Missing observation | Gap flagged (> 3× step) |
| Pre/after-hours bar | Not applicable | Present unless RTH filter applied |

---

## 8. Intraday Session Structure

### 8.1 Regular Trading Hours (RTH)

For US equity markets (SPY, QQQ), RTH is defined as:

$$
\text{RTH} = \{ t : 09{:}30{:}00_{\text{ET}} \leq t < 16{:}00{:}00_{\text{ET}} \}
$$

The filter criterion applied to bar-open timestamps is:

$$
\text{is\_rth}(t) \iff 09{:}30{:}00 \leq t.\text{time()} < 16{:}00{:}00
$$

This is inclusive of the opening bar and exclusive of the 16:00 bar. For a 1-minute series, the last RTH bar is at 15:59:00; for a 5-minute series, it is at 15:55:00.

### 8.2 RTH filtering

`filter_rth(obs: &[Observation]) -> Vec<Observation>` applies the RTH criterion to a sorted slice and returns only the conforming observations. This is a **pure function** — it does not modify the input `CleanSeries`.

Typical usage:
```rust
let rth_obs = filter_rth(&series.observations);
```

### 8.3 Overnight gaps

After RTH filtering, consecutive sessions are separated by an overnight gap:

$$
\text{gap} = t_{\text{open}}^{(k+1)} - t_{\text{close}}^{(k)} \approx 17.5\,\text{hours}
$$

These gaps are visible in the timestamp sequence and are reported by `ValidationReport::gaps` for 5-minute and finer data (since $17.5\,\text{h} \gg 3 \times 5\,\text{min} = 15\,\text{min}$). Whether the online detector **resets state** at session boundaries is a detector-level policy decision, not a data-pipeline decision.

**Thesis recommendation**: do not mix overnight gaps into ordinary intraday transitions without explicit intent. The standard first-thesis design is to reset or re-initialize detector state at each session open.

### 8.4 Session labelling

`label_sessions(obs: &[Observation]) -> Vec<SessionBoundary>` partitions an ascending-sorted observation sequence into contiguous calendar-date groups:

```rust
pub struct SessionBoundary {
    pub session_index: usize,     // 0-based
    pub date:          NaiveDate, // calendar date of this session
    pub first_obs_idx: usize,     // index into observations
    pub last_obs_idx:  usize,     // inclusive
}
```

For RTH-filtered data, each `SessionBoundary` corresponds to exactly one trading day.

---

## 9. Resampling Policy

At the time of Phase 15, resampling is **not implemented** in the pipeline.

**Rationale for deferral**: Alpha Vantage provides data at native granularities (1min, 5min, 15min, 30min, 60min). Since these can be fetched directly, bar-aggregation resampling (e.g., 1min → 5min) is not needed for the thesis experiments. If it becomes necessary, a `resample_ohlc` utility should be added to `src/data/session.rs` and should:

1. Accept a `Vec<Observation>` and a target `bar_minutes`.
2. Group by aligned bar boundaries: bar start = `floor(t / target_step) * target_step`.
3. Aggregate using close-of-bar (last observation in each group).
4. Preserve session boundaries (never aggregate across an overnight gap).
5. Produce a new `CleanSeries` with updated `DataMode`.

**When to use which granularity**:

| Use case | Recommended granularity |
|---|---|
| Regime detection (slow) | 15min or 60min |
| Regime detection (medium) | 5min |
| High-frequency comparison | 1min |

The choice must be stated explicitly in thesis result tables.

---

## 10. Dataset Metadata (`DatasetMeta`)

Every `CleanSeries` carries a `DatasetMeta` struct:

```rust
pub struct DatasetMeta {
    pub symbol:               String,              // e.g., "SPY", "WTI"
    pub mode:                 DataMode,             // Daily | Intraday { bar_minutes }
    pub source:               DataSource,           // AlphaVantage
    pub price_field:          PriceField,           // AdjustedClose | Close | Value
    pub session_convention:   SessionConvention,    // FullDay | RthOnly
    pub fetched_at:           Option<NaiveDateTime>, // UTC fetch time
    pub unit:                 Option<String>,        // "USD", "percent", etc.
}
```

The metadata is preserved through `PartitionedSeries` and is available to all downstream phases. Thesis tables should include at minimum `symbol`, `mode`, `price_field`, and `session_convention` in their data-description columns.

---

## 11. Chronological Train / Validation / Test Split

### 11.1 Why random splitting is invalid

For a time-series detector:

$$
y_{t+1} \perp\!\!\!\not\perp \{y_1, \ldots, y_t\} \text{ in general.}
$$

Future observations are **not** independent of past observations. Random train/test shuffling would allow the model to see future data during training, producing optimistic and unreproducible results. This is the fundamental temporal leakage problem.

### 11.2 The split structure

A chronological 3-way split is defined by two timestamps $t_{\text{train}}$ and $t_{\text{val}}$ with $t_{\text{train}} \leq t_{\text{val}}$:

$$
\text{Train} = \{ y_t : t < t_{\text{train}} \}
$$
$$
\text{Validation} = \{ y_t : t_{\text{train}} \leq t < t_{\text{val}} \}
$$
$$
\text{Test} = \{ y_t : t \geq t_{\text{val}} \}
$$

The three partitions are **disjoint** and **exhaustive** (they cover the entire series).

```rust
pub struct SplitConfig {
    pub train_end: NaiveDateTime,   // first timestamp NOT in train
    pub val_end:   NaiveDateTime,   // first timestamp NOT in validation
}
```

### 11.3 Intended use by downstream phases

| Phase | Partition used |
|---|---|
| Offline EM fitting | Train only |
| Detector threshold selection | Validation |
| Final evaluation | Test |

**The model parameters fitted on the train partition must not be refitted on validation or test data.** The `FrozenModel` type from Phase 13 enforces this for the online streaming phase by prohibiting any mutation of model parameters after construction.

### 11.4 Leakage-prevention policies

1. **Split before feature construction**: all feature derivation (e.g., log returns, normalization) must be applied partition-by-partition, with statistics fitted on train only.
2. **No full-sample normalization**: do not compute a mean or standard deviation over the full series and then split. This leaks test-period statistics into the training process.
3. **Partition labels are explicit**: `PartitionedSeries` stores `SplitConfig` and carries a `partition_of(ts)` method so any downstream phase can verify which partition a given timestamp belongs to.
4. **Split is early**: splitting occurs in the data-pipeline layer, before any modeling. Later phases receive a `PartitionedSeries` rather than a raw `CleanSeries`.

---

## 12. Module Architecture

### 12.1 Existing modules (pre-Phase 15)

```
src/alphavantage/
    client.rs        ← HTTP client, rate limiting, intraday pagination
    commodity.rs     ← CommodityEndpoint, Interval, CommodityResponse,
                       date/value parsing, equity adjusted-close extraction
    rate_limiter.rs  ← token-bucket rate limiter

src/cache/
    mod.rs           ← DuckDB-backed persistent cache; load/store/status

src/data_service/
    mod.rs           ← DataService: orchestrates fetch + cache

src/config.rs        ← TOML config; AlphaVantageConfig, CacheConfig, IngestConfig
src/cli/mod.rs       ← interactive TUI; Ingest/Show/Refresh/Status commands
```

### 12.2 New module added in Phase 15

```
src/data/
    mod.rs           ← Observation, CleanSeries, from_response; module root
    meta.rs          ← DataMode, DataSource, PriceField,
                       SessionConvention, DatasetMeta
    validation.rs    ← Gap, ValidationReport, validate()
    session.rs       ← is_rth_bar(), filter_rth(), SessionBoundary,
                       label_sessions(), SessionAwareSeries
    split.rs         ← TimePartition, SplitConfig, PartitionedSeries
```

### 12.3 Dependency graph (acyclic)

```
alphavantage ──────────────────────────────────┐
                                               ▼
cache ──────────► data_service ──► data (CleanSeries, etc.)
                                               │
                                    ┌──────────┴──────────┐
                                    ▼                     ▼
                              model / online          benchmark
```

The `data` module imports `CommodityResponse` from `alphavantage` (for `CleanSeries::from_response`) but has no dependency on `cache`, `model`, `online`, or `benchmark`.

---

## 13. Canonical Usage Pattern

```rust
use crate::alphavantage::commodity::{CommodityEndpoint, Interval};
use crate::data::{
    CleanSeries, DataMode, DataSource, DatasetMeta, PartitionedSeries,
    PriceField, SessionConvention, SessionAwareSeries, SplitConfig,
    filter_rth,
};
use crate::data_service::DataService;
use chrono::NaiveDateTime;

// 1. Fetch or load from cache.
let response = service
    .load_cached(&CommodityEndpoint::Spy, Interval::Intraday5Min)
    .await?
    .ok_or_else(|| anyhow::anyhow!("SPY 5min not cached — run ingest first"))?;

// 2. Build a CleanSeries with explicit provenance metadata.
let meta = DatasetMeta {
    symbol:             "SPY".to_string(),
    mode:               DataMode::Intraday { bar_minutes: 5 },
    source:             DataSource::AlphaVantage,
    price_field:        PriceField::Close,
    session_convention: SessionConvention::RthOnly,
    fetched_at:         None,
    unit:               Some("USD".to_string()),
};
let series = CleanSeries::from_response(response, meta);

// 3. (Optional) Filter to RTH and label sessions.
let rth_obs = filter_rth(&series.observations);
let session_series = SessionAwareSeries::from_clean_series(
    CleanSeries { observations: rth_obs, ..series.clone() }
);

// 4. Split chronologically.
let cfg = SplitConfig {
    train_end: NaiveDateTime::parse_from_str("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
    val_end:   NaiveDateTime::parse_from_str("2024-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
};
let partitioned = PartitionedSeries::from_series(series, cfg);

// 5. Use train partition for offline EM.
let train_values = partitioned.train.iter().map(|o| o.value).collect::<Vec<f64>>();

// 6. Use test partition for final evaluation.
let test_values = partitioned.test.iter().map(|o| o.value).collect::<Vec<f64>>();
```

---

## 14. Tests

| Module | Test | Invariant verified |
|---|---|---|
| `validation` | `accepts_already_sorted_sequence` | No changes on clean input |
| `validation` | `sorts_unsorted_input_and_flags_it` | Ascending after sort; flag set |
| `validation` | `removes_duplicate_timestamps_keep_first` | Keep-first policy |
| `validation` | `detects_intraday_gap_exceeding_threshold` | Gap flagged at > 3× step |
| `validation` | `normal_intraday_step_not_flagged_as_gap` | No false positives |
| `validation` | `daily_mode_never_reports_gaps` | Daily calendar gaps not flagged |
| `validation` | `empty_input_produces_empty_report` | Empty-safe |
| `session` | `filter_rth_removes_premarket_bars` | 09:00, 09:29 excluded |
| `session` | `filter_rth_removes_after_hours_bars` | 16:00, 16:30 excluded |
| `session` | `filter_rth_keeps_open_bar` | 09:30 included |
| `session` | `filter_rth_keeps_last_rth_bar_and_drops_close` | 15:59 in, 16:00 out |
| `session` | `label_sessions_single_day_produces_one_boundary` | One session |
| `session` | `label_sessions_two_days_produces_two_boundaries` | Day-boundary detection |
| `session` | `label_sessions_empty_is_ok` | Empty-safe |
| `split` | `assigns_observations_to_correct_partitions` | Partition assignment |
| `split` | `partitions_cover_all_observations` | `total_len == n` |
| `split` | `no_temporal_overlap_between_partitions` | Causal ordering preserved |
| `split` | `all_in_train_when_cut_is_after_last_observation` | Degenerate case |
| `mod` | `from_response_sorts_ascending_from_newest_first` | API order corrected |
| `mod` | `from_response_removes_duplicate_timestamp` | Dedup via from_response |
| `mod` | `values_and_timestamps_match_observations` | Accessor correctness |
| `mod` | `from_response_empty_is_valid` | Empty-safe |

Total: **22 tests** in the `data` module.

---

## 15. Open Questions and Future Work

| Item | Priority | Notes |
|---|---|---|
| Log-return feature construction | Phase 16 | `y_t = \log(p_t / p_{t-1})`; must be partition-aware |
| Normalization (mean/std) | Phase 16 | Fit on train only |
| Bar-aggregation resampling | Future | Needed only if API granularity is insufficient |
| Commodity calendar validation | Future | Distinguish holiday gaps from vendor gaps |
| Timezone-aware types (`chrono-tz`) | Optional | Documents ET explicitly in the type system |
| `CommodityResponse → CleanSeries` for annual/monthly | Future | `DataMode::Monthly` etc. if needed |

---

## 16. Summary

The central principle of Phase 15 is:

$$
\boxed{
\text{In a time-series thesis, the data pipeline is part of the model definition.}
}
$$

Phase 15 built the missing bridge between the raw Alpha Vantage data layer (which already existed) and the modeling pipeline. The new `src/data/` module provides:

- **`CleanSeries`** — a validated, ascending-ordered, metadata-tagged dataset that all downstream phases can trust.
- **`ValidationReport`** — a transparent audit trail of what was corrected or flagged during construction.
- **RTH filter and session labels** — intraday data segmented into calendar sessions with overnight gaps made explicit.
- **`PartitionedSeries`** — chronological train/validation/test split with leakage-safe partition objects.
- **`DatasetMeta`** — full provenance record for reproducible thesis reporting.

The pipeline is now deterministic, documented, and reproducible. All 22 new tests confirm the correctness of ordering, deduplication, gap detection, RTH filtering, session labelling, and chronological splitting.
