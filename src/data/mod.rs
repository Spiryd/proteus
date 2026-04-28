#![allow(unused_imports, dead_code)]
/// Real financial market-data pipeline.
///
/// This module is the bridge between the raw Alpha Vantage API responses
/// (or the DuckDB cache) and the modeling pipeline.  It is the only place
/// where "raw vendor data" becomes "validated, annotated data ready for
/// the Markov Switching Model."
///
/// # Architecture
///
/// ```text
/// AlphaVantage API / DuckDB cache
///         │  CommodityResponse
///         ▼
///   CleanSeries::from_response(response, meta)
///         │  sort ↑ · dedup · gap-check · annotate
///         ▼
///      CleanSeries  (ascending, gap-reported, metadata-tagged)
///         │
///   ┌─────┴──────────────────────────────┐
///   │  (optional) RTH filter             │
///   │  filter_rth(&series.observations)  │
///   └─────┬──────────────────────────────┘
///         │
///   ┌─────┴────────────────────────────────┐
///   │  (optional) session labels           │
///   │  SessionAwareSeries::from_clean(…)   │
///   └─────┬────────────────────────────────┘
///         │
///   PartitionedSeries::from_series(series, config)
///         │  train / validation / test
///         ▼
///   Offline EM  ·  Online streaming  ·  Benchmark
/// ```
///
/// # Ordering contract
///
/// Every `CleanSeries` is **guaranteed** to have `observations` sorted in
/// strictly ascending chronological order.  This is enforced by
/// `validation::validate` during construction regardless of the source
/// order (Alpha Vantage returns data newest-first).
///
/// # Timezone contract
///
/// - **Daily data**: `timestamp` is the trading date stored as midnight
///   `NaiveDateTime` (timezone-unaware but conventionally UTC calendar date).
/// - **Intraday data**: `timestamp` is the bar-open time in **US Eastern
///   Time (ET)**, as returned by Alpha Vantage.  No conversion is applied;
///   callers must not mix sources with different timezone conventions.
pub mod meta;
pub mod session;
pub mod split;
pub mod validation;

pub use meta::{DataMode, DataSource, DatasetMeta, PriceField, SessionConvention};
pub use session::{SessionAwareSeries, SessionBoundary, filter_rth, is_rth_bar, label_sessions};
pub use split::{PartitionedSeries, SplitConfig, TimePartition};
pub use validation::{Gap, ValidationReport};

use chrono::NaiveDateTime;

use crate::alphavantage::commodity::CommodityResponse;

// =========================================================================
// Observation
// =========================================================================

/// A single time-stamped observation ready for the modeling pipeline.
///
/// The `value` field holds exactly the price or commodity index value
/// specified by `DatasetMeta::price_field`.  No further transformation
/// (e.g., log returns) is applied at this layer — that belongs to the
/// feature-construction phase.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    /// Bar-open time (intraday, US ET) or trading-date midnight (daily).
    pub timestamp: NaiveDateTime,
    /// Price or value as extracted from the vendor response.
    pub value: f64,
}

// =========================================================================
// CleanSeries
// =========================================================================

/// A validated, ascending-ordered, metadata-annotated market-data series.
///
/// `CleanSeries` is the central output type of the data pipeline.  Every
/// instance is guaranteed to have:
///
/// - `observations` sorted strictly ascending by `timestamp`,
/// - no duplicate timestamps (keep-first deduplication applied),
/// - a `ValidationReport` documenting what was changed during construction.
///
/// `CleanSeries` does **not** guarantee gap-free data — gaps are reported
/// in `report.gaps` but are not filled (imputing price dynamics would
/// compromise the scientific integrity of the pipeline).
#[derive(Debug, Clone)]
pub struct CleanSeries {
    /// Observations in strictly ascending chronological order.
    pub observations: Vec<Observation>,
    /// Provenance and configuration metadata.
    pub meta: DatasetMeta,
    /// Data-quality report produced during construction.
    pub report: ValidationReport,
}

impl CleanSeries {
    /// Build a `CleanSeries` from a raw `CommodityResponse`.
    ///
    /// `meta` must be pre-populated by the caller with at least `symbol`,
    /// `mode`, `source`, `price_field`, and `session_convention`.
    ///
    /// The conversion performs:
    /// 1. Map `CommodityDataPoint` → `Observation` (both fields are direct).
    /// 2. Call `validation::validate` (sort, dedup, gap-detect).
    pub fn from_response(response: CommodityResponse, meta: DatasetMeta) -> Self {
        let mut observations: Vec<Observation> = response
            .data
            .into_iter()
            .map(|dp| Observation {
                timestamp: dp.date,
                value: dp.value,
            })
            .collect();
        let report = validation::validate(&mut observations, &meta.mode);
        Self {
            observations,
            meta,
            report,
        }
    }

    /// Values in ascending chronological order — direct model input.
    ///
    /// The slice corresponds to the `y_1, …, y_T` sequence consumed by the
    /// EM estimator and the online filter.
    pub fn values(&self) -> Vec<f64> {
        self.observations.iter().map(|o| o.value).collect()
    }

    /// Timestamps in ascending chronological order.
    pub fn timestamps(&self) -> Vec<NaiveDateTime> {
        self.observations.iter().map(|o| o.timestamp).collect()
    }

    /// Number of clean observations.
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphavantage::commodity::{CommodityDataPoint, CommodityResponse};
    use chrono::NaiveDateTime;

    fn ts(s: &str) -> NaiveDateTime {
        NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").unwrap()
    }

    fn make_response(dates: &[(&str, f64)]) -> CommodityResponse {
        CommodityResponse {
            name: "TEST".to_string(),
            interval: "daily".to_string(),
            unit: "USD".to_string(),
            data: dates
                .iter()
                .map(|(s, v)| CommodityDataPoint {
                    date: ts(s),
                    value: *v,
                })
                .collect(),
        }
    }

    fn daily_meta(symbol: &str) -> DatasetMeta {
        DatasetMeta {
            symbol: symbol.to_string(),
            mode: DataMode::Daily,
            source: DataSource::AlphaVantage,
            price_field: PriceField::AdjustedClose,
            session_convention: SessionConvention::FullDay,
            fetched_at: None,
            unit: Some("USD".to_string()),
        }
    }

    #[test]
    fn from_response_sorts_ascending_from_newest_first() {
        // Alpha Vantage typically returns data newest-first.
        let response = make_response(&[
            ("2024-03-01 00:00:00", 100.0),
            ("2024-02-01 00:00:00", 99.0),
            ("2024-01-01 00:00:00", 98.0),
        ]);
        let series = CleanSeries::from_response(response, daily_meta("SPY"));
        assert_eq!(series.len(), 3);
        assert!(series.report.had_unsorted_input);
        assert!(series.observations[0].timestamp < series.observations[1].timestamp);
        assert!(series.observations[1].timestamp < series.observations[2].timestamp);
        assert_eq!(series.observations[0].value, 98.0); // oldest first
    }

    #[test]
    fn from_response_removes_duplicate_timestamp() {
        let response = make_response(&[
            ("2024-01-02 00:00:00", 1.0),
            ("2024-01-02 00:00:00", 2.0), // duplicate
            ("2024-01-03 00:00:00", 3.0),
        ]);
        let series = CleanSeries::from_response(response, daily_meta("WTI"));
        assert_eq!(series.len(), 2);
        assert_eq!(series.report.n_dropped_duplicates, 1);
    }

    #[test]
    fn values_and_timestamps_match_observations() {
        let response =
            make_response(&[("2024-01-01 00:00:00", 10.0), ("2024-01-02 00:00:00", 20.0)]);
        let series = CleanSeries::from_response(response, daily_meta("GOLD"));
        assert_eq!(series.values(), vec![10.0, 20.0]);
        assert_eq!(
            series.timestamps(),
            vec![ts("2024-01-01 00:00:00"), ts("2024-01-02 00:00:00")]
        );
    }

    #[test]
    fn from_response_empty_is_valid() {
        let response = make_response(&[]);
        let series = CleanSeries::from_response(response, daily_meta("SPY"));
        assert!(series.is_empty());
        assert_eq!(series.report.n_input, 0);
    }
}
