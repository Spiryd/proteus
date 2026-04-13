# proteus
Change Point Detection Algorithm Based on a Markov Switching Model for Commodities made for my Master Thesis

## Setup

### Prerequisites

- Rust (stable) — install via [rustup](https://rustup.rs)
- A free Alpha Vantage API key — get one at [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)

### Configuration

Copy the example config and fill in your API key:

```
cp config.example.toml config.toml
```

Edit `config.toml`:

```toml
[alphavantage]
api_key = "your_api_key_here"
rate_limit_per_minute = 75   # default, can be omitted

[cache]
path = "data/commodities.duckdb"   # default, can be omitted

[ingest]
series = [
    { commodity = "wti",         interval = "monthly"   },
    { commodity = "brent",       interval = "monthly"   },
    { commodity = "natural_gas", interval = "weekly"    },
    { commodity = "gold",        interval = "monthly"   },
    { commodity = "silver",      interval = "monthly"   },
]
```

> **Important:** `config.toml` contains your API key — never commit it.

### Running

```
cargo run
```

An optional path to a different config file can be passed as the first argument:

```
cargo run -- path/to/other.toml
```

## Usage

Proteus has a fully interactive terminal UI. After `cargo run` you will see:

```
? What would you like to do?
> Ingest    — fetch & cache configured series
  Show      — view cached data for a series
  Refresh   — force-refresh a series from the API
  Status    — cache overview
  Quit
```

Navigate with the arrow keys and confirm with Enter. Pressing **Esc** at any sub-prompt cancels back to the main menu.

| Action | Description |
|--------|-------------|
| **Ingest** | Fetches every series in `[ingest]` and stores it in DuckDB. Already-cached series are skipped unless you choose force re-fetch. |
| **Show** | Displays cached data for a chosen commodity and interval — never calls the API. |
| **Refresh** | Force-fetches one series from the API and overwrites the cache. |
| **Status** | Tabular overview of all cached series: point count, date range, last-fetch time. |

See [docs/interactive_cli.md](docs/interactive_cli.md) for full details.

## Implementation Details

### Data Sources

Data is sourced from the [Alpha Vantage API](https://www.alphavantage.co), which provides full historical commodity price series. Supported commodities include WTI, Brent, Natural Gas, Copper, Aluminum, Wheat, Corn, Cotton, Sugar, Coffee, Gold, Silver, and the All Commodities Index, each available at daily, weekly, monthly, quarterly, or annual resolution (depending on the endpoint).

The HTTP client is rate-limited to 75 requests per minute by default (configurable). See [docs/alphavantage_client.md](docs/alphavantage_client.md) for API details.

### Data Caching

Fetched data is persisted in a local [DuckDB](https://duckdb.org) database (`data/commodities.duckdb` by default). The database is created automatically on first run.

Each series is stored as a flat table of `(date, value)` rows keyed by `(symbol, interval)`. Re-ingesting a series does a full replace, keeping the cache in sync with the API without manual diffing.

See [docs/duckdb_cache.md](docs/duckdb_cache.md) for the schema and API.

### Architecture

```
src/
  main.rs                  — entry point
  config.rs                — TOML config structs
  alphavantage/
    client.rs              — async HTTP client
    commodity.rs           — endpoint/interval types + deserialisation
    rate_limiter.rs        — token-bucket rate limiter
  cache/
    mod.rs                 — DuckDB persistence layer
  data_service/
    mod.rs                 — orchestration (cache-first, bulk ingest)
  cli/
    mod.rs                 — interactive terminal UI (inquire)
```

See [docs/data_service.md](docs/data_service.md) for the orchestration layer details.

