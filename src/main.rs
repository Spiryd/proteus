mod client;
mod commodity;
mod config;
mod rate_limiter;

use client::AlphaVantageClient;
use commodity::{CommodityEndpoint, Interval};
use config::Config;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cfg = Config::from_file("config.toml")?;
    let av = AlphaVantageClient::from_config(&cfg);

    // ---------------------------------------------------------------
    // Configure which endpoints + intervals you want to query here.
    // ---------------------------------------------------------------
    let queries: &[(CommodityEndpoint, Interval)] = &[
        (CommodityEndpoint::Wti, Interval::Monthly),
        (CommodityEndpoint::Brent, Interval::Monthly),
        (CommodityEndpoint::NaturalGas, Interval::Weekly),
        (CommodityEndpoint::Copper, Interval::Quarterly),
        (CommodityEndpoint::Wheat, Interval::Monthly),
        (CommodityEndpoint::Gold, Interval::Monthly),
        (CommodityEndpoint::Silver, Interval::Monthly),
    ];

    for (endpoint, interval) in queries {
        println!(
            "\n=== {} ({}) ===",
            endpoint.function_name(),
            interval.as_str()
        );

        match av.commodity_history(endpoint, *interval).await {
            Ok(resp) => {
                println!("Name   : {}", resp.name);
                println!("Unit   : {}", resp.unit);
                println!("Points : {}", resp.data.len());
                println!("Latest 5 entries:");
                for dp in resp.data.iter().take(5) {
                    println!("  {} -> {:.4}", dp.date, dp.value);
                }
            }
            Err(e) => eprintln!("Error fetching {}: {e}", endpoint.function_name()),
        }
    }

    Ok(())
}

