mod alphavantage;
mod benchmark;
mod cache;
mod calibration;
mod cli;
mod config;
mod data;
mod data_service;
mod detector;
mod experiments;
mod features;
mod model;
mod online;
mod real_eval;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Optional: pass a config path as the first argument (defaults to "config.toml").
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "config.toml".to_string());
    let cfg = config::Config::from_file(&config_path)?;
    cli::run(cfg).await
}
