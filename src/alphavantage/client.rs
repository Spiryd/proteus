use super::commodity::{
    CommodityEndpoint, CommodityResponse, Interval, RawGoldSilverResponse, RawStandardResponse,
};
use super::rate_limiter::RateLimiter;
use crate::config::Config;

const DEFAULT_BASE_URL: &str = "https://www.alphavantage.co/query";
const DEFAULT_RATE_LIMIT: u32 = 75;

pub struct AlphaVantageClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    rate_limiter: RateLimiter,
}

impl AlphaVantageClient {
    pub fn from_config(config: &Config) -> Self {
        let base_url = config
            .alphavantage
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let rpm = config.alphavantage.rate_limit_per_minute.unwrap_or(DEFAULT_RATE_LIMIT);
        Self {
            http: reqwest::Client::new(),
            api_key: config.alphavantage.api_key.clone(),
            base_url,
            rate_limiter: RateLimiter::new(rpm),
        }
    }

    #[allow(dead_code)]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            rate_limiter: RateLimiter::new(DEFAULT_RATE_LIMIT),
        }
    }

    #[allow(dead_code)]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub async fn commodity_history(
        &self,
        endpoint: &CommodityEndpoint,
        interval: Interval,
    ) -> anyhow::Result<CommodityResponse> {
        let supported = endpoint.supported_intervals();
        if !supported.contains(&interval) {
            let supported_str =
                supported.iter().map(|i| i.as_str()).collect::<Vec<_>>().join(", ");
            anyhow::bail!(
                "Interval '{}' is not supported by {}. Accepted: {}",
                interval,
                endpoint,
                supported_str
            );
        }

        self.rate_limiter.acquire().await;

        let mut url = format!(
            "{}?function={}&interval={}&datatype=json&apikey={}",
            self.base_url,
            endpoint.function_name(),
            interval.as_str(),
            self.api_key,
        );
        if let Some(sym) = endpoint.symbol() {
            url.push_str("&symbol=");
            url.push_str(sym);
        }

        let response = self
            .http
            .get(url)
            .send()
            .await?
            .error_for_status()
            .map_err(|e| anyhow::anyhow!("HTTP error: {e}"))?;

        let commodity_response = if endpoint.symbol().is_some() {
            response
                .json::<RawGoldSilverResponse>()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to parse gold/silver response: {e}"))?
                .into_response()
        } else {
            response
                .json::<RawStandardResponse>()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to parse response: {e}"))?
                .into_response()
        };

        Ok(commodity_response)
    }
}
