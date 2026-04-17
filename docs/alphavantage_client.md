# Proteus — Alpha Vantage Commodity Client

Proteus is an async Rust client for the [Alpha Vantage](https://www.alphavantage.co) API, focused on historic commodity price data.

---

## Dependencies

| Crate | Purpose |
|---|---|
| `reqwest` | Async HTTP (JSON feature) |
| `serde` / `serde_json` | JSON deserialisation |
| `chrono` | Typed date/datetime values (`NaiveDateTime`) |
| `tokio` | Async runtime |
| `toml` | Config file parsing |
| `anyhow` | Error handling |
| `duckdb` | Embedded DuckDB database (bundled + chrono features) |
| `inquire` | Interactive terminal prompts |

---

## Configuration

Copy `config.example.toml` to `config.toml` and fill in your key:

```toml
[alphavantage]
api_key = "your_api_key_here"

# Max API requests per minute (defaults to 75 when omitted)
rate_limit_per_minute = 75

# Optionally override the base URL (e.g. for a proxy)
# base_url = "https://www.alphavantage.co/query"
```

The file is read at startup via `Config::from_file("config.toml")` and passed into `AlphaVantageClient::from_config(&cfg)`. Add `config.toml` to your `.gitignore` to avoid committing credentials.

---

## Module Layout

```
src/
  main.rs          — entry point / demo queries
  client.rs        — AlphaVantageClient
  commodity.rs     — CommodityEndpoint, Interval, response types
  config.rs        — Config + AlphaVantageConfig (TOML structs)
  rate_limiter.rs  — Token-bucket RateLimiter
```

---

## Core Types

### `AlphaVantageClient`

```rust
pub struct AlphaVantageClient { /* … */ }

impl AlphaVantageClient {
    // Build from a parsed Config (recommended)
    pub fn from_config(config: &Config) -> Self;

    // Build with an explicit key
    pub fn new(api_key: impl Into<String>) -> Self;

    // Override base URL (useful for proxies / testing)
    pub fn with_base_url(self, base_url: impl Into<String>) -> Self;

    // Fetch historic prices for a commodity endpoint + interval.
    // For intraday equity (SPY/QQQ + Intraday* intervals) this transparently
    // paginates through every available calendar month to return the full history.
    pub async fn commodity_history(
        &self,
        endpoint: &CommodityEndpoint,
        interval: Interval,
    ) -> anyhow::Result<CommodityResponse>;
}
```

### `Config` / `AlphaVantageConfig`

```rust
/// Loaded from config.toml
pub struct Config {
    pub alphavantage: AlphaVantageConfig,
}

pub struct AlphaVantageConfig {
    pub api_key:                String,
    pub base_url:               Option<String>,  // defaults to official API
    pub rate_limit_per_minute:  Option<u32>,     // defaults to 75
}

impl Config {
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self>;
}
```

### `CommodityDataPoint`

Each data point in the response holds typed Rust values — no manual parsing required:

```rust
pub struct CommodityDataPoint {
    pub date:  chrono::NaiveDateTime,  // parsed from "YYYY-MM-DD HH:MM:SS", "YYYY-MM-DD", or "YYYY-MM"
    pub value: f64,                    // parsed from the API's string representation
}
```

`date` handles three formats:
- Intraday: `"2026-04-14 20:00:00"` — stored as-is.
- Daily / weekly: `"2024-03-15"` — stored at midnight (`00:00:00`).
- Monthly / quarterly: `"2024-03"` — normalised to the 1st of the month at midnight.

### `CommodityResponse`

```rust
pub struct CommodityResponse {
    pub name:     String,
    pub interval: String,   // e.g. "monthly", "daily", "60min"
    pub unit:     String,
    pub data:     Vec<CommodityDataPoint>,
}
```

---

## Intervals

The `Interval` enum covers all resolutions exposed by the API:

| Variant | `as_str()` | Supported by |
|---|---|---|
| `Daily` | `"daily"` | all commodities, SPY, QQQ |
| `Weekly` | `"weekly"` | all commodities, SPY, QQQ |
| `Monthly` | `"monthly"` | all commodities, SPY, QQQ |
| `Quarterly` | `"quarterly"` | commodity indices only |
| `Annual` | `"annual"` | commodity indices only |
| `Intraday1Min` | `"1min"` | SPY, QQQ |
| `Intraday5Min` | `"5min"` | SPY, QQQ |
| `Intraday15Min` | `"15min"` | SPY, QQQ |
| `Intraday30Min` | `"30min"` | SPY, QQQ |
| `Intraday60Min` | `"60min"` | SPY, QQQ |

`Interval::is_intraday()` returns `true` for the five intraday variants. The client uses this flag to select the `TIME_SERIES_INTRADAY` endpoint (with an `interval` query parameter) instead of the adjusted daily/weekly/monthly endpoints.

### Intraday full-history pagination

`TIME_SERIES_INTRADAY` returns at most one month of data per request. When `commodity_history` is called with an intraday interval for SPY or QQQ, it automatically paginates backwards month-by-month from today, collecting all available data.

**Termination**: Alpha Vantage does not return an error for unavailable months — it silently falls back to the most recent period. The client detects exhaustion by filtering each response to only data points that fall within the requested month; an empty result stops further pagination.

**API cost**: each month is one request, so fetching several years of 1-minute data will consume many rate-limit tokens. With the default 75 rpm limit, a full SPY history (~30 years ≈ 360 months) takes approximately 5 minutes.

---

## Rate Limiting

Every call to `commodity_history` automatically acquires a token from an internal `RateLimiter` before the HTTP request is dispatched. If the budget is exhausted the call suspends (async, no thread blocking) until a token becomes available.

### How it works

The limiter is a **token bucket** built on `tokio::sync::Semaphore`:

| Step | Detail |
|---|---|
| Startup | Semaphore seeded with `rate_limit_per_minute` permits |
| Per request | One permit consumed via `acquire()` before the HTTP call |
| Refill | Background task adds 1 permit every `60 000 / rpm` ms |
| Burst cap | `add_permits` is skipped when the semaphore is already full |

With the default of 75 rpm, one permit is refilled every **800 ms**.

### Configuring the limit

Set `rate_limit_per_minute` in `config.toml`:

```toml
[alphavantage]
api_key = "your_api_key_here"
rate_limit_per_minute = 75   # omit to use the default of 75
```

The value is picked up by `AlphaVantageClient::from_config` — no code changes needed.

### `RateLimiter` API

```rust
pub struct RateLimiter { /* … */ }

impl RateLimiter {
    // Create a new limiter; spawns a background refill task.
    pub fn new(requests_per_minute: u32) -> Self;

    // Async wait for a permit. Called automatically by commodity_history.
    pub async fn acquire(&self);
}
```

You do not need to interact with `RateLimiter` directly — `AlphaVantageClient` owns one and calls `acquire()` transparently.

---

## Supported Endpoints

### Energy (daily / weekly / monthly)

| Variant | Alpha Vantage function |
|---|---|
| `CommodityEndpoint::Wti` | `WTI` |
| `CommodityEndpoint::Brent` | `BRENT` |
| `CommodityEndpoint::NaturalGas` | `NATURAL_GAS` |

### Metals (daily / weekly / monthly)

| Variant | Alpha Vantage function |
|---|---|
| `CommodityEndpoint::Gold` | `GOLD_SILVER_HISTORY` (symbol=GOLD) |
| `CommodityEndpoint::Silver` | `GOLD_SILVER_HISTORY` (symbol=SILVER) |

### Agricultural & Industrial (monthly / quarterly / annual)

| Variant | Alpha Vantage function |
|---|---|
| `CommodityEndpoint::Copper` | `COPPER` |
| `CommodityEndpoint::Aluminum` | `ALUMINUM` |
| `CommodityEndpoint::Wheat` | `WHEAT` |
| `CommodityEndpoint::Corn` | `CORN` |
| `CommodityEndpoint::Cotton` | `COTTON` |
| `CommodityEndpoint::Sugar` | `SUGAR` |
| `CommodityEndpoint::Coffee` | `COFFEE` |
| `CommodityEndpoint::AllCommodities` | `ALL_COMMODITIES` |

Passing an unsupported interval (e.g. `Interval::Daily` to `Copper`) returns an `anyhow::Error` with a descriptive message listing accepted values.

---

## Usage Example

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = Config::from_file("config.toml")?;
    let client = AlphaVantageClient::from_config(&cfg);

    let resp = client
        .commodity_history(&CommodityEndpoint::Wti, Interval::Monthly)
        .await?;

    println!("{} ({})", resp.name, resp.unit);
    for point in resp.data.iter().take(5) {
        // point.date  -> chrono::NaiveDateTime
        // point.value -> f64
        println!("{} — ${:.2}", point.date, point.value);
    }

    Ok(())
}
```

---

## Error Handling

All public async methods return `anyhow::Result<T>`. Errors are transparently propagated with context:

- Invalid interval for endpoint → descriptive bail message
- HTTP errors → wrapped `reqwest::Error` with status context
- JSON parse failure → wrapped with field context

Use `anyhow::Context` from the caller side for additional chaining:

```rust
use anyhow::Context;

let resp = client
    .commodity_history(&CommodityEndpoint::Gold, Interval::Daily)
    .await
    .context("failed to fetch gold prices")?;
```
