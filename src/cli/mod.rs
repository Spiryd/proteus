use std::fmt;

use inquire::{Confirm, InquireError, Select, Text};

use crate::alphavantage::commodity::{CommodityEndpoint, Interval};
use crate::config::Config;
use crate::data_service::DataService;

// ---------------------------------------------------------------------------
// Main menu
// ---------------------------------------------------------------------------

enum Action {
    Ingest,
    Show,
    Refresh,
    Status,
    Quit,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Ingest => "Ingest    — fetch & cache configured series",
            Self::Show => "Show      — view cached data for a series",
            Self::Refresh => "Refresh   — force-refresh a series from the API",
            Self::Status => "Status    — cache overview",
            Self::Quit => "Quit",
        })
    }
}

// ---------------------------------------------------------------------------
// Commodity selection helper
// ---------------------------------------------------------------------------

struct CommodityChoice(CommodityEndpoint);

impl fmt::Display for CommodityChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match &self.0 {
            CommodityEndpoint::Wti => "WTI          — West Texas Intermediate crude",
            CommodityEndpoint::Brent => "Brent        — North Sea crude",
            CommodityEndpoint::NaturalGas => "Natural Gas  — Henry Hub spot price",
            CommodityEndpoint::Copper => "Copper",
            CommodityEndpoint::Aluminum => "Aluminum",
            CommodityEndpoint::Wheat => "Wheat",
            CommodityEndpoint::Corn => "Corn",
            CommodityEndpoint::Cotton => "Cotton",
            CommodityEndpoint::Sugar => "Sugar",
            CommodityEndpoint::Coffee => "Coffee",
            CommodityEndpoint::Gold => "Gold",
            CommodityEndpoint::Silver => "Silver",
            CommodityEndpoint::AllCommodities => "All Commodities Index",
            CommodityEndpoint::Spy => "SPY          — S&P 500 ETF (adjusted close)",
            CommodityEndpoint::Qqq => "QQQ          — Nasdaq-100 ETF (adjusted close)",
        })
    }
}

fn commodity_choices() -> Vec<CommodityChoice> {
    vec![
        CommodityChoice(CommodityEndpoint::Wti),
        CommodityChoice(CommodityEndpoint::Brent),
        CommodityChoice(CommodityEndpoint::NaturalGas),
        CommodityChoice(CommodityEndpoint::Copper),
        CommodityChoice(CommodityEndpoint::Aluminum),
        CommodityChoice(CommodityEndpoint::Wheat),
        CommodityChoice(CommodityEndpoint::Corn),
        CommodityChoice(CommodityEndpoint::Cotton),
        CommodityChoice(CommodityEndpoint::Sugar),
        CommodityChoice(CommodityEndpoint::Coffee),
        CommodityChoice(CommodityEndpoint::Gold),
        CommodityChoice(CommodityEndpoint::Silver),
        CommodityChoice(CommodityEndpoint::AllCommodities),
        CommodityChoice(CommodityEndpoint::Spy),
        CommodityChoice(CommodityEndpoint::Qqq),
    ]
}

// ---------------------------------------------------------------------------
// Prompt helpers — return None on Esc / Ctrl+C
// ---------------------------------------------------------------------------

fn prompt_commodity() -> anyhow::Result<Option<CommodityEndpoint>> {
    match Select::new("Select commodity:", commodity_choices())
        .without_help_message()
        .prompt()
    {
        Ok(c) => Ok(Some(c.0)),
        Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn prompt_interval(endpoint: &CommodityEndpoint) -> anyhow::Result<Option<Interval>> {
    let intervals = endpoint.supported_intervals().to_vec();
    match Select::new("Select interval:", intervals)
        .without_help_message()
        .prompt()
    {
        Ok(i) => Ok(Some(i)),
        Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

// ---------------------------------------------------------------------------
// Entry point — interactive loop
// ---------------------------------------------------------------------------

pub async fn run(cfg: Config) -> anyhow::Result<()> {
    let service = DataService::new(&cfg)?;

    loop {
        println!();
        let action = match Select::new(
            "What would you like to do?",
            vec![
                Action::Ingest,
                Action::Show,
                Action::Refresh,
                Action::Status,
                Action::Quit,
            ],
        )
        .without_help_message()
        .prompt()
        {
            Ok(a) => a,
            Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => break,
            Err(e) => return Err(e.into()),
        };

        let result = match action {
            Action::Ingest => cmd_ingest(&service, &cfg).await,
            Action::Show => cmd_show(&service).await,
            Action::Refresh => cmd_refresh(&service).await,
            Action::Status => cmd_status(&service).await,
            Action::Quit => break,
        };

        if let Err(e) = result {
            eprintln!("\nError: {e:#}");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Command handlers
// ---------------------------------------------------------------------------

async fn cmd_ingest(service: &DataService, cfg: &Config) -> anyhow::Result<()> {
    let ingest_cfg = cfg
        .ingest
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("No [ingest] section found in config.toml"))?;

    if ingest_cfg.series.is_empty() {
        println!("No series configured under [ingest]. Nothing to do.");
        return Ok(());
    }

    let force = match Confirm::new("Force re-fetch all series even if already cached?")
        .with_default(false)
        .prompt()
    {
        Ok(v) => v,
        Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => return Ok(()),
        Err(e) => return Err(e.into()),
    };

    println!("\nIngesting {} series ...\n", ingest_cfg.series.len());
    service.ingest_all(&ingest_cfg.series, force).await?;
    println!("\nDone.");
    Ok(())
}

async fn cmd_show(service: &DataService) -> anyhow::Result<()> {
    let Some(endpoint) = prompt_commodity()? else {
        return Ok(());
    };
    let Some(interval) = prompt_interval(&endpoint)? else {
        return Ok(());
    };

    let limit_str = match Text::new("Data points to display:")
        .with_default("10")
        .prompt()
    {
        Ok(s) => s,
        Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    let limit: usize = limit_str.trim().parse().unwrap_or(10);

    let response = service
        .load_cached(&endpoint, interval)
        .await?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "No cached data for {} ({}). Run Ingest first.",
                endpoint,
                interval
            )
        })?;

    println!(
        "\n=== {} — {} ({}) ===",
        endpoint, response.name, response.unit
    );
    let intraday = response.interval.ends_with("min");
    if intraday {
        println!("{:<20}  {:>14}", "Timestamp", "Close");
        println!("{:-<20}  {:->14}", "", "");
        for dp in response.data.iter().take(limit) {
            println!(
                "{:<20}  {:>14.4}",
                dp.date.format("%Y-%m-%d %H:%M"),
                dp.value
            );
        }
    } else {
        println!("{:<12}  {:>14}", "Date", "Value");
        println!("{:-<12}  {:->14}", "", "");
        for dp in response.data.iter().take(limit) {
            println!("{:<12}  {:>14.4}", dp.date.date(), dp.value);
        }
    }
    println!(
        "\n{} of {} total data points shown.",
        limit.min(response.data.len()),
        response.data.len()
    );
    Ok(())
}

async fn cmd_refresh(service: &DataService) -> anyhow::Result<()> {
    let Some(endpoint) = prompt_commodity()? else {
        return Ok(());
    };
    let Some(interval) = prompt_interval(&endpoint)? else {
        return Ok(());
    };

    println!("\nRefreshing {} ({}) ...", endpoint, interval);
    let response = service.refresh(&endpoint, interval).await?;
    println!(
        "Done. {} — {} data points stored.",
        response.name,
        response.data.len()
    );
    Ok(())
}

async fn cmd_status(service: &DataService) -> anyhow::Result<()> {
    let statuses = service.status().await?;

    if statuses.is_empty() {
        println!("Cache is empty. Run Ingest to populate it.");
        return Ok(());
    }

    println!(
        "\n{:<20} {:<10} {:<30} {:<8} {:<12} {:<12} Last Fetched",
        "Symbol", "Interval", "Name", "Points", "From", "To"
    );
    println!("{:-<112}", "");
    for s in &statuses {
        println!(
            "{:<20} {:<10} {:<30} {:<8} {:<12} {:<12} {}",
            s.symbol,
            s.interval,
            s.name,
            s.point_count,
            s.from_date.date(),
            s.to_date.date(),
            s.last_fetched.format("%Y-%m-%d %H:%M:%S")
        );
    }
    println!();
    Ok(())
}
