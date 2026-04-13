use chrono::NaiveDate;
use serde::{Deserialize, Deserializer};
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
        }
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
        }
    }

    pub fn symbol(&self) -> Option<&'static str> {
        match self {
            Self::Gold => Some("GOLD"),
            Self::Silver => Some("SILVER"),
            _ => None,
        }
    }

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
            other => anyhow::bail!(
                "Unknown commodity '{other}'. Valid: wti, brent, natural_gas, copper, \
                 aluminum, wheat, corn, cotton, sugar, coffee, gold, silver, all_commodities"
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

impl std::str::FromStr for Interval {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "daily" | "d" => Ok(Self::Daily),
            "weekly" | "w" => Ok(Self::Weekly),
            "monthly" | "m" => Ok(Self::Monthly),
            "quarterly" | "q" => Ok(Self::Quarterly),
            "annual" | "yearly" | "y" | "a" => Ok(Self::Annual),
            other => anyhow::bail!(
                "Unknown interval '{other}'. Valid: daily, weekly, monthly, quarterly, annual"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Deserialisation helpers
// ---------------------------------------------------------------------------

fn deserialize_date<'de, D: Deserializer<'de>>(d: D) -> Result<NaiveDate, D::Error> {
    let s = String::deserialize(d)?;
    NaiveDate::parse_from_str(&s, "%Y-%m-%d")
        .or_else(|_| NaiveDate::parse_from_str(&format!("{s}-01"), "%Y-%m-%d"))
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

// ---------------------------------------------------------------------------
// Public response types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CommodityDataPoint {
    pub date: NaiveDate,
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
