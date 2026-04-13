use serde::Deserialize;
use std::path::Path;

/// Root configuration loaded from `config.toml`.
#[derive(Debug, Deserialize)]
pub struct Config {
    pub alphavantage: AlphaVantageConfig,
}

/// `[alphavantage]` section of `config.toml`.
#[derive(Debug, Deserialize)]
pub struct AlphaVantageConfig {
    /// Your Alpha Vantage API key.
    pub api_key: String,
    /// Optional override for the base URL (defaults to the official API).
    pub base_url: Option<String>,
    /// Max API requests per minute. Defaults to 75 when omitted.
    pub rate_limit_per_minute: Option<u32>,
}

impl Config {
    /// Load and parse a TOML config file from `path`.
    pub fn from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Cannot read config file '{}': {e}", path.display()))?;
        toml::from_str(&contents)
            .map_err(|e| anyhow::anyhow!("Failed to parse config file '{}': {e}", path.display()))
    }
}
