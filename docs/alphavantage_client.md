# Proteus — Alpha Vantage Commodity Client

Proteus is an async Rust client for the [Alpha Vantage](https://www.alphavantage.co) API, focused on historic commodity price data.

---

## Dependencies

| Crate | Purpose |
|---|---|
| `reqwest` | Async HTTP (JSON feature) |
| `serde` / `serde_json` | JSON deserialisation |
| `chrono` | Typed date values (`NaiveDate`) |
| `tokio` | Async runtime |
| `toml` | Config file parsing |
| `anyhow` | Error handling |

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

    // Fetch historic prices for a commodity endpoint + interval
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
    pub date:  chrono::NaiveDate,   // parsed from "YYYY-MM-DD" or "YYYY-MM"
    pub value: f64,                 // parsed from the API's string representation
}
```

`date` handles both full dates (`2024-03-15`) and year-month strings (`2024-03`), normalising the latter to the 1st of that month.

### `CommodityResponse`

```rust
pub struct CommodityResponse {
    pub name:     String,
    pub interval: String,
    pub unit:     String,
    pub data:     Vec<CommodityDataPoint>,
}
```

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

## Intervals

```rust
pub enum Interval {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Annual,
}
```

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
        // point.date  -> chrono::NaiveDate
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
