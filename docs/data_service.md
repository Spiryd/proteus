# Data Service

`DataService` (`src/data_service/mod.rs`) is the orchestration layer between the interactive CLI and the two underlying subsystems — the Alpha Vantage HTTP client and the DuckDB cache.

## Construction

```rust
let service = DataService::new(&cfg)?;
```

This opens the cache database and builds the rate-limited HTTP client in one step.

## Methods

### `get` — cache-first read

```rust
pub async fn get(endpoint: &CommodityEndpoint, interval: Interval) -> anyhow::Result<CommodityResponse>
```

Returns cached data when available. Only calls the API when no data exists for that series. Results are written back to the cache automatically.

### `refresh` — force API fetch

```rust
pub async fn refresh(endpoint: &CommodityEndpoint, interval: Interval) -> anyhow::Result<CommodityResponse>
```

Always fetches from the API and overwrites the cache. Used by the **Refresh** action in the interactive CLI.

### `load_cached` — read-only

```rust
pub async fn load_cached(endpoint: &CommodityEndpoint, interval: Interval) -> anyhow::Result<Option<CommodityResponse>>
```

Returns cached data without ever making an HTTP request. Returns `None` if the series has not been ingested yet. Used by **Show**.

### `ingest_all` — bulk ingest

```rust
pub async fn ingest_all(series: &[IngestSeriesItem], force: bool) -> anyhow::Result<()>
```

Iterates the `[ingest]` series list from `config.toml`. Series that already have data in the cache are skipped unless `force = true`. Each fetch respects the rate limiter.

### `status` — cache overview

```rust
pub async fn status() -> anyhow::Result<Vec<SeriesStatus>>
```

Returns one `SeriesStatus` row per cached series with point count, date range, and last-fetch timestamp.

## Async / blocking boundary

DuckDB uses a synchronous API. All cache operations are dispatched with `tokio::task::spawn_blocking` so the async runtime is never blocked.
