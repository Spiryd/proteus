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
| `date`       | TIMESTAMP | Intraday: full datetime. Daily/weekly: midnight. Monthly: 1st of month at midnight. |
| `value`      | DOUBLE    |                          |
| `fetched_at` | TIMESTAMP | UTC timestamp of the fetch |

Primary key: `(symbol, interval, date)`

## Write behaviour

Each `store()` call does a full replace for that series:

1. `DELETE` all existing rows for `(symbol, interval)`
2. Bulk-insert the new data points via DuckDB's `Appender`

Bulk insert uses the `Appender` API with `append_row(params![symbol, interval, date, value, fetched_at])` + `flush()`. This is significantly faster than individual `INSERT` statements for large intraday series (e.g., 378k rows for SPY 15min).

## Schema migration

On every startup `init_schema()` silently runs:

```sql
ALTER TABLE commodity_prices ALTER COLUMN date TYPE TIMESTAMP;
```

This is a no-op when the column is already `TIMESTAMP`. It upgrades existing databases that were created with the legacy `DATE` column type before intraday support was added.

## `CommodityCache` API

```rust
// Open (or create) the database. Schema is initialised on first open.
CommodityCache::open(path: &str) -> anyhow::Result<Self>

// Timestamp of the most recent fetch for a series, or None if uncached.
// Uses ORDER BY fetched_at DESC LIMIT 1 rather than MAX() to avoid a
// DuckDB INTERNAL error on empty tables with .query_row().
cache.last_fetched(symbol, interval) -> anyhow::Result<Option<NaiveDateTime>>

// Load a full series from the cache. Returns None if not present.
cache.load(symbol, interval) -> anyhow::Result<Option<CommodityResponse>>

// Persist a full series response (replaces existing data via Appender bulk insert).
cache.store(symbol, interval, &response) -> anyhow::Result<usize>

// Summary row for every series in the cache.
cache.status() -> anyhow::Result<Vec<SeriesStatus>>
```

## Thread safety

`duckdb::Connection` is `Send` but not `Sync`. The connection is wrapped in a `Mutex<Connection>` and `CommodityCache` manually implements `unsafe impl Sync` — safe because all access is serialised through the mutex.

All cache calls from async code go through `tokio::task::spawn_blocking` to avoid blocking the async runtime.
