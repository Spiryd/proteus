use chrono::{Datelike, Utc};
use super::commodity::{
    CommodityDataPoint, CommodityEndpoint, CommodityResponse, Interval, RawEquityResponse,
    RawGoldSilverResponse, RawStandardResponse,
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

        // Equity endpoints branch based on whether the interval is intraday.
        if let Some(ticker) = endpoint.equity_ticker() {
            if interval.is_intraday() {
                // Full historical intraday data requires month-by-month pagination.
                return self.fetch_equity_intraday_full(ticker, interval).await;
            }
            self.rate_limiter.acquire().await;
            let function = endpoint
                .equity_function_name(interval)
                .expect("equity_function_name returned None for non-intraday equity");
            let url = format!(
                "{}?function={}&symbol={}&outputsize=full&datatype=json&apikey={}",
                self.base_url, function, ticker, self.api_key,
            );
            let response = self
                .http
                .get(url)
                .send()
                .await?
                .error_for_status()
                .map_err(|e| anyhow::anyhow!("HTTP error: {e}"))?;
            return response
                .json::<RawEquityResponse>()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to parse equity response for {ticker}: {e}"))?
                .into_response(interval.as_str());
        }

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

    /// Fetch the complete intraday history for an equity ticker by paginating
    /// through every calendar month backwards from today until the API returns
    /// no data.
    async fn fetch_equity_intraday_full(
        &self,
        ticker: &str,
        interval: Interval,
    ) -> anyhow::Result<CommodityResponse> {
        let mut all_points: Vec<CommodityDataPoint> = Vec::new();
        let mut meta: Option<(String, String)> = None;
        let mut months_fetched: u32 = 0;
        let total_start = std::time::Instant::now();

        let today = Utc::now().naive_utc().date();
        let mut year = today.year();
        let mut month = today.month();

        println!("[intraday] {ticker} ({}) — fetching full history month by month …", interval.as_str());

        loop {
            let month_str = format!("{year:04}-{month:02}");
            self.rate_limiter.acquire().await;

            print!("[intraday] {ticker}  {month_str} … ");
            let _ = std::io::Write::flush(&mut std::io::stdout());
            let req_start = std::time::Instant::now();

            let url = format!(
                "{}?function=TIME_SERIES_INTRADAY&symbol={}&interval={}&month={}&outputsize=full&datatype=json&apikey={}",
                self.base_url, ticker, interval.as_str(), month_str, self.api_key,
            );

            let json: serde_json::Value = self
                .http
                .get(&url)
                .send()
                .await?
                .error_for_status()
                .map_err(|e| anyhow::anyhow!("HTTP error fetching {ticker} {month_str}: {e}"))?
                .json()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to read {ticker} {month_str}: {e}"))?;

            // Alpha Vantage may signal an error via top-level keys.
            if let Some(msg) = json
                .get("Error Message")
                .or_else(|| json.get("Information"))
                .or_else(|| json.get("Note"))
                .and_then(|v| v.as_str())
            {
                let is_rate_limit = msg.to_ascii_lowercase().contains("rate limit")
                    || msg.to_ascii_lowercase().contains("per minute");

                if is_rate_limit {
                    println!("rate limited — waiting 60s then retrying …");
                    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                    // Do NOT advance the month counter; retry the same month.
                    continue;
                } else {
                    println!("stopped ({msg})");
                    break;
                }
            }

            let raw: RawEquityResponse = serde_json::from_value(json).map_err(|e| {
                anyhow::anyhow!("Failed to parse equity response for {ticker} {month_str}: {e}")
            })?;
            let month_resp = raw.into_response(interval.as_str())?;

            // When a requested month has no data Alpha Vantage silently returns the most recent
            // available period instead of an error. Filter to only points that actually belong to
            // the month we asked for; if none match, we've exhausted available history.
            let points_in_month: Vec<CommodityDataPoint> = month_resp
                .data
                .into_iter()
                .filter(|dp| dp.date.year() == year && dp.date.month() == month)
                .collect();

            if points_in_month.is_empty() {
                println!("no data — history exhausted");
                break;
            }

            months_fetched += 1;
            println!("{} points  (total so far: {})  [{:.1}s]",
                points_in_month.len(),
                all_points.len() + points_in_month.len(),
                req_start.elapsed().as_secs_f32()
            );

            if meta.is_none() {
                meta = Some((month_resp.name.clone(), month_resp.unit.clone()));
            }
            all_points.extend(points_in_month);

            // Move to the previous month.
            if month == 1 {
                year -= 1;
                month = 12;
            } else {
                month -= 1;
            }
            if year < 1990 {
                break;
            }
        }

        all_points.sort_unstable_by_key(|dp| dp.date);

        println!(
            "[intraday] {ticker} ({}) — done: {} months, {} total points  (total {:.1}s)",
            interval.as_str(),
            months_fetched,
            all_points.len(),
            total_start.elapsed().as_secs_f32()
        );

        let (name, unit) = meta.unwrap_or_else(|| (ticker.to_string(), "USD".to_string()));
        Ok(CommodityResponse {
            name,
            interval: interval.as_str().to_string(),
            unit,
            data: all_points,
        })
    }
}
