use chrono::NaiveDate;
use serde::{Deserialize, Deserializer};
use std::fmt;

/// All supported commodity endpoints from Alpha Vantage.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CommodityEndpoint {
    /// West Texas Intermediate crude oil (daily/weekly/monthly)
    Wti,
    /// Brent crude oil (daily/weekly/monthly)
    Brent,
    /// Henry Hub natural gas spot price (daily/weekly/monthly)
    NaturalGas,
    /// Global copper price (monthly/quarterly/annual)
    Copper,
    /// Global aluminum price (monthly/quarterly/annual)
    Aluminum,
    /// Global wheat price (monthly/quarterly/annual)
    Wheat,
    /// Global corn price (monthly/quarterly/annual)
    Corn,
    /// Global cotton price (monthly/quarterly/annual)
    Cotton,
    /// Global sugar price (monthly/quarterly/annual)
    Sugar,
    /// Global coffee price (monthly/quarterly/annual)
    Coffee,
    /// Global price index of all commodities (monthly/quarterly/annual)
    AllCommodities,
    /// Historical gold price (daily/weekly/monthly)
    Gold,
    /// Historical silver price (daily/weekly/monthly)
    Silver,
}

impl CommodityEndpoint {
    /// Alpha Vantage `function` parameter value.
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
        }
    }

    /// Some endpoints require an extra `symbol` query param (gold/silver).
    pub fn symbol(&self) -> Option<&'static str> {
        match self {
            Self::Gold => Some("GOLD"),
            Self::Silver => Some("SILVER"),
            _ => None,
        }
    }

    /// Intervals accepted by this endpoint.
    pub fn supported_intervals(&self) -> &'static [Interval] {
        match self {
            Self::Wti | Self::Brent | Self::NaturalGas | Self::Gold | Self::Silver => {
                &[Interval::Daily, Interval::Weekly, Interval::Monthly]
            }
            _ => &[Interval::Monthly, Interval::Quarterly, Interval::Annual],
        }
    }
}

impl fmt::Display for CommodityEndpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.function_name())
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
}

impl Interval {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Daily => "daily",
            Self::Weekly => "weekly",
            Self::Monthly => "monthly",
            Self::Quarterly => "quarterly",
            Self::Annual => "annual",
        }
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Custom deserialization for the date and value fields in the API response.
fn deserialize_date<'de, D: Deserializer<'de>>(d: D) -> Result<NaiveDate, D::Error> {
    let s = String::deserialize(d)?;
    // Alpha Vantage returns dates as "YYYY-MM-DD" or "YYYY-MM" (monthly/quarterly).
    // Try full date first, then year-month with day defaulting to 1.
    NaiveDate::parse_from_str(&s, "%Y-%m-%d")
        .or_else(|_| NaiveDate::parse_from_str(&format!("{}-01", s), "%Y-%m-%d"))
        .map_err(serde::de::Error::custom)
}

/// Parses a stringified float. Returns `None` for `"."`, which Alpha Vantage
/// uses as a missing-data sentinel in long historical series.
fn deserialize_optional_float<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
    let s = String::deserialize(d)?;
    if s == "." {
        Ok(None)
    } else {
        s.parse::<f64>().map(Some).map_err(serde::de::Error::custom)
    }
}

// ---------------------------------------------------------------------------
// Private raw types — used only during deserialization, never exposed publicly.
// ---------------------------------------------------------------------------

/// Standard commodity endpoints: { name, interval, unit, data: [{date, value}] }
#[derive(Deserialize)]
pub(crate) struct RawStandardPoint {
    #[serde(deserialize_with = "deserialize_date")]
    date: NaiveDate,
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
    /// Convert into the public type, silently dropping missing-value entries.
    pub fn into_response(self) -> CommodityResponse {
        let data = self
            .data
            .into_iter()
            .filter_map(|p| p.value.map(|v| CommodityDataPoint { date: p.date, value: v }))
            .collect();
        CommodityResponse {
            name: self.name,
            interval: self.interval,
            unit: self.unit,
            data,
        }
    }
}

/// GOLD_SILVER_HISTORY endpoint: { nominal, data: [{date, price}] }
/// Different shape — no `name`/`interval`/`unit`, value field is `price`.
#[derive(Deserialize)]
pub(crate) struct RawGoldSilverPoint {
    #[serde(deserialize_with = "deserialize_date")]
    date: NaiveDate,
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

/// A single date/value pair returned by the API.
#[derive(Debug, Clone)]
pub struct CommodityDataPoint {
    pub date: NaiveDate,
    pub value: f64,
}

/// Top-level response from any commodity history endpoint.
#[derive(Debug)]
pub struct CommodityResponse {
    pub name: String,
    #[allow(dead_code)]
    pub interval: String,
    pub unit: String,
    pub data: Vec<CommodityDataPoint>,
}
