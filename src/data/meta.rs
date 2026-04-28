#![allow(dead_code)]
/// Provenance and configuration metadata for a cleaned market-data series.
///
/// `DatasetMeta` travels with every `CleanSeries` and `PartitionedSeries` so
/// that downstream modeling phases can report exactly what data they consumed,
/// how it was preprocessed, and which partition they are operating on.
use chrono::NaiveDateTime;

// =========================================================================
// DataMode
// =========================================================================

/// Observation frequency / temporal mode.
///
/// The mode drives:
/// - gap-detection thresholds in the validation layer,
/// - session-boundary semantics,
/// - documentation of expected bar duration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataMode {
    /// One observation per trading day.
    ///
    /// Timestamps are calendar dates stored as midnight `NaiveDateTime` in
    /// the convention of the source (Alpha Vantage daily data uses the
    /// trading date, not UTC end-of-day).
    Daily,

    /// One observation per bar of `bar_minutes` minutes.
    ///
    /// Timestamps are bar-open times in **US Eastern Time (ET)**, as
    /// supplied directly by Alpha Vantage.  The timezone is documented here
    /// but is **not** embedded in the `NaiveDateTime` type — callers must
    /// not mix sources with different timezone conventions.
    Intraday { bar_minutes: u32 },
}

impl DataMode {
    /// Bar duration in minutes, or `None` for daily data.
    pub fn bar_minutes(&self) -> Option<u32> {
        match self {
            Self::Daily => None,
            Self::Intraday { bar_minutes } => Some(*bar_minutes),
        }
    }

    pub fn is_intraday(&self) -> bool {
        matches!(self, Self::Intraday { .. })
    }

    pub fn is_daily(&self) -> bool {
        matches!(self, Self::Daily)
    }
}

// =========================================================================
// DataSource
// =========================================================================

/// Original data vendor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataSource {
    /// Alpha Vantage JSON REST API (<https://www.alphavantage.co>).
    AlphaVantage,
}

// =========================================================================
// PriceField
// =========================================================================

/// Which price / value field was extracted from the raw API response.
///
/// This must be stated explicitly for thesis reproducibility: "adjusted close"
/// and "unadjusted close" are not interchangeable for quantitative research.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PriceField {
    /// Dividend-and-split-adjusted closing price.
    ///
    /// Source: Alpha Vantage `TIME_SERIES_DAILY_ADJUSTED` field
    /// `"5. adjusted close"` (also weekly/monthly adjusted variants).
    /// Used for: SPY, QQQ daily/weekly/monthly.
    AdjustedClose,

    /// Unadjusted last-traded price within the bar.
    ///
    /// Source: Alpha Vantage `TIME_SERIES_INTRADAY` field `"4. close"`.
    /// Used for: SPY, QQQ intraday.
    Close,

    /// Commodity spot/settlement index value.
    ///
    /// Source: Alpha Vantage commodity endpoints (WTI, Brent, etc.) `"value"`.
    /// Units vary by commodity (USD/barrel, USD/troy oz, etc.).
    Value,
}

// =========================================================================
// SessionConvention
// =========================================================================

/// Session-filtering policy for intraday data.
///
/// The choice of session convention must be stated in any thesis result that
/// uses intraday data, because overnight gaps and pre/after-market data can
/// distort online-detection behavior.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionConvention {
    /// No session filtering — all available bars retained.
    ///
    /// For intraday data this may include pre-market (before 09:30 ET) and
    /// after-hours (after 16:00 ET) bars.  Overnight transitions between
    /// sessions are **not** marked or removed.
    FullDay,

    /// Regular Trading Hours only: bars with open-time in `[09:30, 16:00)` ET.
    ///
    /// Overnight gaps between the 15:59 close bar and the next 09:30 open
    /// bar are visible as gaps in the time index.  Session boundaries are
    /// labelled by the `session` module.
    RthOnly,
}

// =========================================================================
// DatasetMeta
// =========================================================================

/// Complete provenance record for one cleaned market-data series.
#[derive(Debug, Clone)]
pub struct DatasetMeta {
    /// Asset identifier matching the Alpha Vantage cache key
    /// (e.g., `"SPY"`, `"WTI"`, `"GOLD"`).
    pub symbol: String,
    /// Temporal resolution and timestamp convention.
    pub mode: DataMode,
    /// Original data vendor.
    pub source: DataSource,
    /// Which price field was extracted from the vendor response.
    pub price_field: PriceField,
    /// Session convention applied during preprocessing (intraday only;
    /// for daily data, use `SessionConvention::FullDay`).
    pub session_convention: SessionConvention,
    /// UTC timestamp at which the raw data was fetched from the vendor.
    /// `None` if the fetch time is unavailable (e.g., legacy cache entries).
    pub fetched_at: Option<NaiveDateTime>,
    /// Measurement unit as reported by the vendor (e.g., `"USD"`, `"percent"`).
    pub unit: Option<String>,
}
