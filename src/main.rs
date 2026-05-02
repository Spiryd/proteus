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
mod reporting;

/// Known direct subcommands (first arg after binary name).
fn is_direct_command(s: &str) -> bool {
    matches!(
        s,
        "e2e" | "param-search" | "run-experiment" | "run-batch" | "run-real" | "calibrate"
            | "optimize" | "inspect" | "status" | "help"
    )
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    match args.first().map(String::as_str) {
        // Direct subcommand mode
        Some(cmd) if is_direct_command(cmd) => cli::run_direct(args).await,
        // Explicit config file (interactive mode with that config)
        Some(cfg_path) => {
            let cfg = config::Config::from_file(cfg_path)?;
            cli::run(cfg).await
        }
        // No arguments — interactive mode with default config
        None => {
            let cfg = config::Config::from_file("config.toml")?;
            cli::run(cfg).await
        }
    }
}
