use std::sync::Arc;

use crate::alphavantage::client::AlphaVantageClient;
use crate::alphavantage::commodity::{CommodityEndpoint, CommodityResponse, Interval};
use crate::cache::{CommodityCache, SeriesStatus};
use crate::config::{Config, IngestSeriesItem};

pub struct DataService {
    client: AlphaVantageClient,
    cache: Arc<CommodityCache>,
}

impl DataService {
    pub fn new(config: &Config) -> anyhow::Result<Self> {
        let client = AlphaVantageClient::from_config(config);
        let cache = CommodityCache::open(config.cache.resolved_path())?;
        Ok(Self {
            client,
            cache: Arc::new(cache),
        })
    }

    /// Force-fetch from the API regardless of cache state.
    pub async fn refresh(
        &self,
        endpoint: &CommodityEndpoint,
        interval: Interval,
    ) -> anyhow::Result<CommodityResponse> {
        let response = self.client.commodity_history(endpoint, interval).await?;
        self.write_cache(endpoint, interval, response.clone())
            .await?;
        Ok(response)
    }

    /// Read-only cache lookup — never calls the API.
    pub async fn load_cached(
        &self,
        endpoint: &CommodityEndpoint,
        interval: Interval,
    ) -> anyhow::Result<Option<CommodityResponse>> {
        let symbol = endpoint.cache_key().to_string();
        let interval_str = interval.as_str().to_string();
        let cache = Arc::clone(&self.cache);
        tokio::task::spawn_blocking(move || cache.load(&symbol, &interval_str)).await?
    }

    /// Ingest all series listed in `[ingest]`, skipping fresh ones unless `force = true`.
    pub async fn ingest_all(&self, series: &[IngestSeriesItem], force: bool) -> anyhow::Result<()> {
        for item in series {
            let endpoint: CommodityEndpoint = item.commodity.parse()?;
            let interval: Interval = item.interval.parse()?;
            let symbol = endpoint.cache_key().to_string();
            let interval_str = interval.as_str().to_string();

            if !force {
                let cache = Arc::clone(&self.cache);
                let sym = symbol.clone();
                let iv = interval_str.clone();
                let cached =
                    tokio::task::spawn_blocking(move || cache.last_fetched(&sym, &iv)).await??;

                if cached.is_some() {
                    println!("[skip]  {symbol} ({interval_str}) — already cached");
                    continue;
                }
            }

            println!("[fetch] {symbol} ({interval_str}) ...");
            let series_start = std::time::Instant::now();
            match self.client.commodity_history(&endpoint, interval).await {
                Ok(response) => {
                    let count = response.data.len();
                    let fetch_elapsed = series_start.elapsed();
                    self.write_cache(&endpoint, interval, response).await?;
                    println!(
                        "[ok]    {} ({}) — {} points  (fetch {:.1}s, total {:.1}s)",
                        symbol,
                        interval_str,
                        count,
                        fetch_elapsed.as_secs_f32(),
                        series_start.elapsed().as_secs_f32()
                    );
                }
                Err(e) => {
                    eprintln!("[error] {symbol} ({interval_str}): {e}");
                }
            }
        }
        Ok(())
    }

    /// Return cache status for all stored series.
    pub async fn status(&self) -> anyhow::Result<Vec<SeriesStatus>> {
        let cache = Arc::clone(&self.cache);
        tokio::task::spawn_blocking(move || cache.status()).await?
    }

    async fn write_cache(
        &self,
        endpoint: &CommodityEndpoint,
        interval: Interval,
        response: CommodityResponse,
    ) -> anyhow::Result<()> {
        let symbol = endpoint.cache_key().to_string();
        let interval_str = interval.as_str().to_string();
        let cache = Arc::clone(&self.cache);
        tokio::task::spawn_blocking(move || cache.store(&symbol, &interval_str, &response))
            .await??;
        Ok(())
    }
}
