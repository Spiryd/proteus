# proteus
Change Point Detection Algorithm Based on a Markov Switching Model for Commodities made for my Master Thesis

## Setup

### Configuration
The project is configured via a `config.toml` file in the project root. Copy the example and fill in your API key:

```
cp config.example.toml config.toml
```

Then edit `config.toml`:

```toml
[alphavantage]
api_key = "your_api_key_here"

# Max API requests per minute (defaults to 75 when omitted)
rate_limit_per_minute = 75
```

> **Important:** `config.toml` contains your API key, never commit it.

Get a free API key at [alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key).

## Implementation Details

### Data Sources
The data used in this project is sourced from the [Alpha Vantage API](https://www.alphavantage.co). The API provides historical commodity price data, which is essential for the change point detection algorithm implemented in this project.
To interact with the API we have a rate limited client that you can read about in the [docs](docs/alphavantage_client.md).

### Data Caching
To optimize performance and reduce redundant API calls, the project implements a caching mechanism using DuckDB. //TODO: expand on this after implementing the caching layer
