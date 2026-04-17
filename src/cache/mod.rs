use chrono::{NaiveDateTime, Utc};
use duckdb::OptionalExt;
use std::sync::Mutex;

use crate::alphavantage::commodity::{CommodityDataPoint, CommodityResponse};

#[allow(dead_code)]
pub struct SeriesStatus {
    pub symbol: String,
    pub interval: String,
    pub name: String,
    pub unit: String,
    pub point_count: i64,
    pub from_date: NaiveDateTime,
    pub to_date: NaiveDateTime,
    pub last_fetched: NaiveDateTime,
}

pub struct CommodityCache {
    conn: Mutex<duckdb::Connection>,
}

// duckdb::Connection implements Send, so Mutex<Connection> is Send + Sync.
unsafe impl Sync for CommodityCache {}

impl CommodityCache {
    pub fn open(path: &str) -> anyhow::Result<Self> {
        if let Some(parent) = std::path::Path::new(path).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let conn = duckdb::Connection::open(path)
            .map_err(|e| anyhow::anyhow!("Failed to open DuckDB at '{path}': {e}"))?;
        let cache = Self {
            conn: Mutex::new(conn),
        };
        cache.init_schema()?;
        Ok(cache)
    }

    fn init_schema(&self) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        // Migrate existing databases: upgrade date column from DATE to TIMESTAMP.
        // Ignored if table doesn't exist yet or column is already TIMESTAMP.
        let _ =
            conn.execute_batch("ALTER TABLE commodity_prices ALTER COLUMN date TYPE TIMESTAMP;");
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS commodity_meta (
                symbol   VARCHAR NOT NULL,
                interval VARCHAR NOT NULL,
                name     VARCHAR NOT NULL,
                unit     VARCHAR NOT NULL,
                PRIMARY KEY (symbol, interval)
            );
            CREATE TABLE IF NOT EXISTS commodity_prices (
                symbol     VARCHAR    NOT NULL,
                interval   VARCHAR    NOT NULL,
                date       TIMESTAMP  NOT NULL,
                value      DOUBLE     NOT NULL,
                fetched_at TIMESTAMP  NOT NULL,
                PRIMARY KEY (symbol, interval, date)
            );",
        )
        .map_err(|e| anyhow::anyhow!("Schema init failed: {e}"))?;
        Ok(())
    }

    /// Timestamp of the most recent fetch for a series, or `None` if uncached.
    pub fn last_fetched(
        &self,
        symbol: &str,
        interval: &str,
    ) -> anyhow::Result<Option<NaiveDateTime>> {
        let conn = self.conn.lock().unwrap();
        let result: Option<NaiveDateTime> = conn
            .query_row(
                "SELECT MAX(fetched_at) FROM commodity_prices WHERE symbol = ? AND interval = ?",
                duckdb::params![symbol, interval],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| anyhow::anyhow!("last_fetched query failed: {e}"))?
            .flatten();
        Ok(result)
    }

    /// Load a cached series. Returns `None` if no data is stored.
    pub fn load(&self, symbol: &str, interval: &str) -> anyhow::Result<Option<CommodityResponse>> {
        let conn = self.conn.lock().unwrap();

        let meta: Option<(String, String)> = conn
            .query_row(
                "SELECT name, unit FROM commodity_meta WHERE symbol = ? AND interval = ?",
                duckdb::params![symbol, interval],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(|e| anyhow::anyhow!("Meta load failed: {e}"))?;

        let Some((name, unit)) = meta else {
            return Ok(None);
        };

        let mut stmt = conn
            .prepare(
                "SELECT date, value FROM commodity_prices
                 WHERE symbol = ? AND interval = ?
                 ORDER BY date DESC",
            )
            .map_err(|e| anyhow::anyhow!("Prepare failed: {e}"))?;

        let points: Vec<CommodityDataPoint> = stmt
            .query_map(duckdb::params![symbol, interval], |row| {
                Ok(CommodityDataPoint {
                    date: row.get(0)?,
                    value: row.get(1)?,
                })
            })
            .map_err(|e| anyhow::anyhow!("Data load failed: {e}"))?
            .collect::<Result<_, _>>()
            .map_err(|e| anyhow::anyhow!("Row mapping failed: {e}"))?;

        if points.is_empty() {
            return Ok(None);
        }

        Ok(Some(CommodityResponse {
            name,
            interval: interval.to_string(),
            unit,
            data: points,
        }))
    }

    /// Persist a full series response, replacing any existing data for that key.
    pub fn store(
        &self,
        symbol: &str,
        interval: &str,
        response: &CommodityResponse,
    ) -> anyhow::Result<usize> {
        let conn = self.conn.lock().unwrap();
        let fetched_at = Utc::now().naive_utc();
        let total_start = std::time::Instant::now();

        // Wrap everything in a single transaction — avoids one disk sync per row,
        // which is the main reason row-by-row inserts are slow.
        conn.execute_batch("BEGIN")
            .map_err(|e| anyhow::anyhow!("BEGIN failed: {e}"))?;

        let t = std::time::Instant::now();
        conn.execute(
            "DELETE FROM commodity_prices WHERE symbol = ? AND interval = ?",
            duckdb::params![symbol, interval],
        )
        .map_err(|e| anyhow::anyhow!("Delete prices failed: {e}"))?;
        conn.execute(
            "DELETE FROM commodity_meta WHERE symbol = ? AND interval = ?",
            duckdb::params![symbol, interval],
        )
        .map_err(|e| anyhow::anyhow!("Delete meta failed: {e}"))?;
        println!("  [db] delete:       {:.3}s", t.elapsed().as_secs_f32());

        conn.execute(
            "INSERT INTO commodity_meta (symbol, interval, name, unit) VALUES (?, ?, ?, ?)",
            duckdb::params![symbol, interval, &response.name, &response.unit],
        )
        .map_err(|e| anyhow::anyhow!("Insert meta failed: {e}"))?;

        let t = std::time::Instant::now();
        let mut stmt = conn
            .prepare(
                "INSERT INTO commodity_prices (symbol, interval, date, value, fetched_at)
                 VALUES (?, ?, ?, ?, ?)",
            )
            .map_err(|e| anyhow::anyhow!("Prepare insert failed: {e}"))?;

        for point in &response.data {
            stmt.execute(duckdb::params![
                symbol,
                interval,
                point.date,
                point.value,
                fetched_at
            ])
            .map_err(|e| anyhow::anyhow!("Insert point failed: {e}"))?;
        }
        println!(
            "  [db] insert {} rows: {:.3}s",
            response.data.len(),
            t.elapsed().as_secs_f32()
        );

        let t = std::time::Instant::now();
        conn.execute_batch("COMMIT")
            .map_err(|e| anyhow::anyhow!("COMMIT failed: {e}"))?;
        println!("  [db] commit:       {:.3}s", t.elapsed().as_secs_f32());
        println!(
            "  [db] total store:  {:.3}s  ({} points)",
            total_start.elapsed().as_secs_f32(),
            response.data.len()
        );

        Ok(response.data.len())
    }

    /// Summary of all series currently in the cache.
    pub fn status(&self) -> anyhow::Result<Vec<SeriesStatus>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn
            .prepare(
                "SELECT p.symbol, p.interval,
                        COALESCE(m.name, ''), COALESCE(m.unit, ''),
                        COUNT(*)        AS point_count,
                        MIN(p.date)     AS from_date,
                        MAX(p.date)     AS to_date,
                        MAX(p.fetched_at) AS last_fetched
                 FROM   commodity_prices p
                 LEFT JOIN commodity_meta m
                        ON p.symbol = m.symbol AND p.interval = m.interval
                 GROUP BY p.symbol, p.interval, m.name, m.unit
                 ORDER BY p.symbol, p.interval",
            )
            .map_err(|e| anyhow::anyhow!("Status prepare failed: {e}"))?;

        let statuses: Vec<SeriesStatus> = stmt
            .query_map([], |row| {
                Ok(SeriesStatus {
                    symbol: row.get(0)?,
                    interval: row.get(1)?,
                    name: row.get(2)?,
                    unit: row.get(3)?,
                    point_count: row.get(4)?,
                    from_date: row.get(5)?,
                    to_date: row.get(6)?,
                    last_fetched: row.get(7)?,
                })
            })
            .map_err(|e| anyhow::anyhow!("Status query failed: {e}"))?
            .collect::<Result<_, _>>()
            .map_err(|e| anyhow::anyhow!("Status row mapping failed: {e}"))?;

        Ok(statuses)
    }
}
