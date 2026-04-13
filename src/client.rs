use crate::commodity::{CommodityEndpoint, CommodityResponse, Interval, RawGoldSilverResponse, RawStandardResponse};
use crate::config::Config;
use crate::rate_limiter::RateLimiter;

const DEFAULT_BASE_URL: &str = "https://www.alphavantage.co/query";
const DEFAULT_RATE_LIMIT: u32 = 75;

/// Async Alpha Vantage client. Build one per process and reuse it.
pub struct AlphaVantageClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    rate_limiter: RateLimiter,
}

impl AlphaVantageClient {
    /// Create a client from a loaded [`Config`].
    pub fn from_config(config: &Config) -> Self {
        let base_url = config
            .alphavantage
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let rpm = config
            .alphavantage
            .rate_limit_per_minute
            .unwrap_or(DEFAULT_RATE_LIMIT);
        Self {
            http: reqwest::Client::new(),
            api_key: config.alphavantage.api_key.clone(),
            base_url,
            rate_limiter: RateLimiter::new(rpm),
        }
    }

    /// Create a client with an explicit API key.
    #[allow(dead_code)]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            rate_limiter: RateLimiter::new(DEFAULT_RATE_LIMIT),
        }
    }

    /// Override the base URL (useful for testing or proxies).
    #[allow(dead_code)]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Fetch historical commodity prices.
    ///
    /// # Errors
    /// Returns an error if `interval` is not supported by the given `endpoint`,
    /// or if the HTTP request fails.
    pub async fn commodity_history(
        &self,
        endpoint: &CommodityEndpoint,
        interval: Interval,
    ) -> anyhow::Result<CommodityResponse> {
        // Validate interval against endpoint capabilities.
        let supported = endpoint.supported_intervals();
        if !supported.contains(&interval) {
            let supported_str = supported
                .iter()
                .map(|i| i.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            anyhow::bail!(
                "Interval '{}' is not supported by {}. Accepted: {}",
                interval,
                endpoint,
                supported_str
            );
        }

        // Consume one rate-limit token before sending the request.
        self.rate_limiter.acquire().await;

        // Build the query URL explicitly to avoid type-inference issues with
        // reqwest 0.13's RequestBuilder::query generics.
        let mut url = format!(
            "{}?function={}&interval={}&datatype=json&apikey={}",
            self.base_url,
            endpoint.function_name(),
            interval.as_str(),
            self.api_key,
        );

        // GOLD_SILVER_HISTORY requires an extra `symbol` param.
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

        // GOLD_SILVER_HISTORY has a different JSON shape than every other endpoint.
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
