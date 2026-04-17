use chrono::{NaiveDate, NaiveDateTime};
use serde::{Deserialize, Deserializer};
use std::collections::HashMap;
use std::fmt;

/// All supported commodity endpoints from Alpha Vantage.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CommodityEndpoint {
    Wti,
    Brent,
    NaturalGas,
    Copper,
    Aluminum,
    Wheat,
    Corn,
    Cotton,
    Sugar,
    Coffee,
    AllCommodities,
    Gold,
    Silver,
    /// S&P 500 ETF (daily/weekly/monthly adjusted close)
    Spy,
    /// Nasdaq-100 ETF (daily/weekly/monthly adjusted close)
    Qqq,
}

impl CommodityEndpoint {
    pub fn function_name(&self) -> &'static str {
        match self {
            Self::Wti => "WTI",
            Self::Brent => "BRENT",
            Self::NaturalGas => "NATURAL_GAS",
            Self::Copper => "COPPER",
            Self::Aluminum => "ALUMINUM",
            Self::Wheat => "WHEAT",
            Self::Corn => "CORN",
            Self::Cotton => "COTTON",
            Self::Sugar => "SUGAR",
            Self::Coffee => "COFFEE",
            Self::AllCommodities => "ALL_COMMODITIES",
            Self::Gold | Self::Silver => "GOLD_SILVER_HISTORY",
            // Equities use interval-specific function names; see equity_function_name().
            Self::Spy | Self::Qqq => unreachable!("call equity_function_name() for equity endpoints"),
        }
    }

    /// Function name for equity endpoints — encodes the interval.
    /// Returns `None` for non-equity endpoints.
    /// For intraday intervals use `TIME_SERIES_INTRADAY` directly in the client.
    pub fn equity_function_name(&self, interval: Interval) -> Option<&'static str> {
        if !matches!(self, Self::Spy | Self::Qqq) {
            return None;
        }
        Some(match interval {
            Interval::Daily   => "TIME_SERIES_DAILY_ADJUSTED",
            Interval::Weekly  => "TIME_SERIES_WEEKLY_ADJUSTED",
            Interval::Monthly => "TIME_SERIES_MONTHLY_ADJUSTED",
            _ => unreachable!("use TIME_SERIES_INTRADAY for intraday equity intervals"),
        })
    }

    /// Stable unique cache key (differs from function_name for Gold/Silver).
    pub fn cache_key(&self) -> &'static str {
        match self {
            Self::Wti => "WTI",
            Self::Brent => "BRENT",
            Self::NaturalGas => "NATURAL_GAS",
            Self::Copper => "COPPER",
            Self::Aluminum => "ALUMINUM",
            Self::Wheat => "WHEAT",
            Self::Corn => "CORN",
            Self::Cotton => "COTTON",
            Self::Sugar => "SUGAR",
            Self::Coffee => "COFFEE",
            Self::AllCommodities => "ALL_COMMODITIES",
            Self::Gold => "GOLD",
            Self::Silver => "SILVER",
            Self::Spy => "SPY",
            Self::Qqq => "QQQ",
        }
    }

    pub fn symbol(&self) -> Option<&'static str> {
        match self {
            Self::Gold => Some("GOLD"),
            Self::Silver => Some("SILVER"),
            _ => None,
        }
    }

    /// Ticker symbol for equity index endpoints (SPY / QQQ).
    pub fn equity_ticker(&self) -> Option<&'static str> {
        match self {
            Self::Spy => Some("SPY"),
            Self::Qqq => Some("QQQ"),
            _ => None,
        }
    }

    pub fn supported_intervals(&self) -> &'static [Interval] {
        match self {
            Self::Wti | Self::Brent | Self::NaturalGas | Self::Gold | Self::Silver => {
                &[Interval::Daily, Interval::Weekly, Interval::Monthly]
            }
            Self::Spy | Self::Qqq => &[
                Interval::Daily,
                Interval::Weekly,
                Interval::Monthly,
                Interval::Intraday1Min,
                Interval::Intraday5Min,
                Interval::Intraday15Min,
                Interval::Intraday30Min,
                Interval::Intraday60Min,
            ],
            _ => &[Interval::Monthly, Interval::Quarterly, Interval::Annual],
        }
    }
}

impl fmt::Display for CommodityEndpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.cache_key())
    }
}

impl std::str::FromStr for CommodityEndpoint {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().replace('-', "_").as_str() {
            "wti" => Ok(Self::Wti),
            "brent" => Ok(Self::Brent),
            "natural_gas" | "naturalgas" | "gas" => Ok(Self::NaturalGas),
            "copper" | "cu" => Ok(Self::Copper),
            "aluminum" | "aluminium" | "al" => Ok(Self::Aluminum),
            "wheat" => Ok(Self::Wheat),
            "corn" => Ok(Self::Corn),
            "cotton" => Ok(Self::Cotton),
            "sugar" => Ok(Self::Sugar),
            "coffee" => Ok(Self::Coffee),
            "all_commodities" | "all" => Ok(Self::AllCommodities),
            "gold" | "xau" => Ok(Self::Gold),
            "silver" | "xag" => Ok(Self::Silver),
            "spy" => Ok(Self::Spy),
            "qqq" => Ok(Self::Qqq),
            other => anyhow::bail!(
                "Unknown commodity '{other}'. Valid: wti, brent, natural_gas, copper, \
                 aluminum, wheat, corn, cotton, sugar, coffee, gold, silver, all_commodities, spy, qqq"
            ),
        }
    }
}

/// Time resolution for a commodity query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interval {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Annual,
    Intraday1Min,
    Intraday5Min,
    Intraday15Min,
    Intraday30Min,
    Intraday60Min,
}

impl Interval {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Daily          => "daily",
            Self::Weekly         => "weekly",
            Self::Monthly        => "monthly",
            Self::Quarterly      => "quarterly",
            Self::Annual         => "annual",
            Self::Intraday1Min   => "1min",
            Self::Intraday5Min   => "5min",
            Self::Intraday15Min  => "15min",
            Self::Intraday30Min  => "30min",
            Self::Intraday60Min  => "60min",
        }
    }

    pub fn is_intraday(self) -> bool {
        matches!(
            self,
            Self::Intraday1Min
                | Self::Intraday5Min
                | Self::Intraday15Min
                | Self::Intraday30Min
                | Self::Intraday60Min
        )
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for Interval {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "daily" | "d"       => Ok(Self::Daily),
            "weekly" | "w"      => Ok(Self::Weekly),
            "monthly" | "m"     => Ok(Self::Monthly),
            "quarterly" | "q"   => Ok(Self::Quarterly),
            "annual" | "yearly" | "y" | "a" => Ok(Self::Annual),
            "1min"              => Ok(Self::Intraday1Min),
            "5min"              => Ok(Self::Intraday5Min),
            "15min"             => Ok(Self::Intraday15Min),
            "30min"             => Ok(Self::Intraday30Min),
            "60min"             => Ok(Self::Intraday60Min),
            other => anyhow::bail!(
                "Unknown interval '{other}'. Valid: daily, weekly, monthly, quarterly, annual, \
                 1min, 5min, 15min, 30min, 60min"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Deserialisation helpers
// ---------------------------------------------------------------------------

fn deserialize_date<'de, D: Deserializer<'de>>(d: D) -> Result<NaiveDateTime, D::Error> {
    let s = String::deserialize(d)?;
    // Full datetime (intraday): "2026-04-14 20:00:00"
    if let Ok(dt) = NaiveDateTime::parse_from_str(&s, "%Y-%m-%d %H:%M:%S") {
        return Ok(dt);
    }
    // Date only: "2024-03-15" → midnight
    if let Ok(d) = NaiveDate::parse_from_str(&s, "%Y-%m-%d") {
        return Ok(d.and_hms_opt(0, 0, 0).unwrap());
    }
    // Year-month: "2024-03" → 1st of month at midnight
    NaiveDate::parse_from_str(&format!("{s}-01"), "%Y-%m-%d")
        .map(|d| d.and_hms_opt(0, 0, 0).unwrap())
        .map_err(serde::de::Error::custom)
}

fn deserialize_optional_float<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
    let s = String::deserialize(d)?;
    if s == "." {
        Ok(None)
    } else {
        s.parse::<f64>().map(Some).map_err(serde::de::Error::custom)
    }
}

// ---------------------------------------------------------------------------
// Private raw deserialization types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub(crate) struct RawStandardPoint {
    #[serde(deserialize_with = "deserialize_date")]
    date: NaiveDateTime,
    #[serde(deserialize_with = "deserialize_optional_float")]
    value: Option<f64>,
}

#[derive(Deserialize)]
pub(crate) struct RawStandardResponse {
    pub name: String,
    #[allow(dead_code)]
    pub interval: String,
    pub unit: String,
    pub data: Vec<RawStandardPoint>,
}

impl RawStandardResponse {
    pub fn into_response(self) -> CommodityResponse {
        let data = self
            .data
            .into_iter()
            .filter_map(|p| p.value.map(|v| CommodityDataPoint { date: p.date, value: v }))
            .collect();
        CommodityResponse { name: self.name, interval: self.interval, unit: self.unit, data }
    }
}

#[derive(Deserialize)]
pub(crate) struct RawGoldSilverPoint {
    #[serde(deserialize_with = "deserialize_date")]
    date: NaiveDateTime,
    #[serde(deserialize_with = "deserialize_optional_float")]
    price: Option<f64>,
}

#[derive(Deserialize)]
pub(crate) struct RawGoldSilverResponse {
    pub nominal: String,
    pub data: Vec<RawGoldSilverPoint>,
}

impl RawGoldSilverResponse {
    pub fn into_response(self) -> CommodityResponse {
        let data = self
            .data
            .into_iter()
            .filter_map(|p| p.price.map(|v| CommodityDataPoint { date: p.date, value: v }))
            .collect();
        CommodityResponse {
            name: self.nominal,
            interval: String::new(),
            unit: String::from("USD"),
            data,
        }
    }
}

/// Equity time series from TIME_SERIES_{DAILY,WEEKLY,MONTHLY}_ADJUSTED.
/// The outer JSON key varies by interval so we flatten into a generic map.
#[derive(Deserialize)]
pub(crate) struct RawEquityResponse {
    #[serde(rename = "Meta Data")]
    meta: RawEquityMeta,
    #[serde(flatten)]
    series: HashMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct RawEquityMeta {
    #[serde(rename = "2. Symbol")]
    symbol: String,
}

impl RawEquityResponse {
    pub fn into_response(self, interval_str: &str) -> anyhow::Result<CommodityResponse> {
        // The time series sits under the only key that isn't "Meta Data".
        let time_obj = self
            .series
            .into_values()
            .find(|v| v.is_object())
            .and_then(|v| match v {
                serde_json::Value::Object(m) => Some(m),
                _ => None,
            })
            .ok_or_else(|| anyhow::anyhow!("Could not locate time series in equity response"))?;

        let mut data: Vec<CommodityDataPoint> = time_obj
            .into_iter()
            .filter_map(|(date_str, entry)| {
                // Parse full datetime (intraday) or date-only (daily/weekly/monthly).
                let timestamp = NaiveDateTime::parse_from_str(&date_str, "%Y-%m-%d %H:%M:%S")
                    .ok()
                    .or_else(|| {
                        NaiveDate::parse_from_str(&date_str, "%Y-%m-%d")
                            .ok()
                            .and_then(|d| d.and_hms_opt(0, 0, 0))
                    })?;
                // Prefer adjusted close (daily/weekly/monthly), fall back to close (intraday).
                let value: f64 = entry
                    .get("5. adjusted close")
                    .or_else(|| entry.get("4. close"))
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse().ok())?;
                Some(CommodityDataPoint { date: timestamp, value })
            })
            .collect();

        data.sort_by(|a, b| b.date.cmp(&a.date));

        Ok(CommodityResponse {
            name: format!("{} Adjusted Close", self.meta.symbol),
            interval: interval_str.to_string(),
            unit: String::from("USD"),
            data,
        })
    }
}

// ---------------------------------------------------------------------------
// Public response types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CommodityDataPoint {
    pub date: NaiveDateTime,
    pub value: f64,
}

#[derive(Debug, Clone)]
pub struct CommodityResponse {
    pub name: String,
    #[allow(dead_code)]
    pub interval: String,
    pub unit: String,
    pub data: Vec<CommodityDataPoint>,
}
