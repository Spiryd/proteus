# DuckDB Cache

Commodity data is persisted locally in a [DuckDB](https://duckdb.org) database so that the API is only called once per series.

## Location

Configured via `config.toml`:

```toml
[cache]
path = "data/commodities.duckdb"   # default when omitted
```

The directory is created automatically if it does not exist.

## Schema

### `commodity_meta`

Stores the display name and unit for each series.

| Column     | Type    | Notes                        |
|------------|---------|------------------------------|
| `symbol`   | VARCHAR | e.g. `WTI`, `GOLD`           |
| `interval` | VARCHAR | e.g. `monthly`, `weekly`     |
| `name`     | VARCHAR | Human-readable series name   |
| `unit`     | VARCHAR | e.g. `dollars per barrel`    |

Primary key: `(symbol, interval)`

### `commodity_prices`

Stores the individual date/value data points.

| Column       | Type      | Notes                    |
|--------------|-----------|--------------------------|
| `symbol`     | VARCHAR   |                          |
| `interval`   | VARCHAR   |                          |
| `date`       | DATE      | Normalised to first of month for monthly/quarterly data |
| `value`      | DOUBLE    |                          |
| `fetched_at` | TIMESTAMP | UTC timestamp of the fetch |

Primary key: `(symbol, interval, date)`

## Write behaviour

Each `store()` call does a full replace for that series:

1. `DELETE` all existing rows for `(symbol, interval)`
2. `INSERT` the new data points

This keeps the cache in sync with the API response without manual diff logic.

## `CommodityCache` API

```rust
// Open (or create) the database. Schema is initialised on first open.
CommodityCache::open(path: &str) -> anyhow::Result<Self>

// Timestamp of the most recent fetch for a series, or None if uncached.
cache.last_fetched(symbol, interval) -> anyhow::Result<Option<NaiveDateTime>>

// Load a full series from the cache. Returns None if not present.
cache.load(symbol, interval) -> anyhow::Result<Option<CommodityResponse>>

// Persist a full series response (replaces existing data).
cache.store(symbol, interval, &response) -> anyhow::Result<usize>

// Summary row for every series in the cache.
cache.status() -> anyhow::Result<Vec<SeriesStatus>>
```

## Thread safety

`duckdb::Connection` is `Send` but not `Sync`. The connection is wrapped in a `Mutex<Connection>` and `CommodityCache` manually implements `unsafe impl Sync` — safe because all access is serialised through the mutex.

All cache calls from async code go through `tokio::task::spawn_blocking` to avoid blocking the async runtime.
