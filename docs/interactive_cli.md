# Interactive CLI

Proteus supports two modes:
- **Interactive mode**: fully arrow-key driven menus powered by [inquire](https://github.com/mikaelmello/inquire).
- **Direct CLI mode**: subcommand-based, useful for scripting and reproducibility.

## Starting the app

```
cargo run              # interactive menu
cargo run -- e2e       # run all registered experiments end-to-end
cargo run -- param-search --id <experiment_id>
cargo run -- run-experiment --config experiment_config.json
cargo run -- run-batch --config a.json --config b.json [--save <dir>]
cargo run -- run-real  --id <experiment_id> [--cache <path.duckdb>] [--save <dir>]
cargo run -- calibrate --id <experiment_id> [--out <dir>]
cargo run -- inspect --dir ./runs/real/my_run/run_001
cargo run -- status
cargo run -- help
```

## Main menu (interactive mode)

```
? What would you like to do?
> Data          — ingest, inspect, and refresh market data
  Experiments   — run, search, and inspect experiments
  Inspect Runs  — browse and view saved run artifacts
  E2E Run       — run all registered experiments end-to-end
  Exit
```

Navigate with the arrow keys, confirm with Enter. Press **Esc** or **Ctrl+C** at any prompt to cancel back to the main menu.

## Data sub-menu

Selecting **Data** opens the data actions menu:

```
? Data action:
> Ingest    — fetch & cache configured series
  Show      — view cached data for a series
  Refresh   — force-refresh a series from the API
  Status    — cache overview
  Back
```

## Data Actions

### Ingest

Fetches every series listed under `[ingest]` in `config.toml` and stores the results in DuckDB.

A confirmation prompt asks whether to force re-fetch series that are already cached (`Y/n`).

Series already in the cache are skipped by default:

```
[skip]  WTI (monthly) — already cached
[fetch] BRENT (monthly) ...
[ok]    BRENT (monthly) — 276 data points stored
```

### Show

Displays cached data for a chosen series — never makes an API call.

1. Select commodity (scrollable list with descriptions)
2. Select interval (filtered to only intervals that endpoint supports)
3. Enter number of data points to display (default: 10)

Non-intraday series show only the date:

```
=== WTI — West Texas Intermediate (dollars per barrel) ===
Date            Value
------------  --------------
2025-12-01       71.1700
2025-11-01       69.5500
...

10 of 276 total data points shown.
```

Intraday series (interval ends with `min`) show the full timestamp:

```
=== SPY — SPDR S&P 500 ETF Trust (USD) ===
Timestamp               Close
--------------------  --------------
2026-04-14 20:00       552.1300
2026-04-14 19:00       551.9800
...

10 of 390 total data points shown.
```

### Refresh

Force-fetches one series from the API regardless of cache contents.

1. Select commodity
2. Select interval
3. Fetches and overwrites the cache entry

### Status

Tabular overview of everything currently in the cache:

```
Symbol               Interval   Name                           Points   From         To           Last Fetched
----------------------------------------------------------------------------------------------------------------
BRENT                daily      Crude Oil Prices Brent         9874     1987-05-20 2026-04-20 2026-04-29 18:31:54
GOLD                 daily      XAUUSD                         5256     2011-06-01 2026-04-28 2026-04-29 18:31:55
QQQ                  15min      QQQ Adjusted Close             364015   2000-01-03 2026-04-28 2026-04-29 18:31:52
QQQ                  daily      QQQ Adjusted Close             6662     1999-11-01 2026-04-28 2026-04-29 18:27:33
SPY                  15min      SPY Adjusted Close             378149   2000-01-03 2026-04-28 2026-04-29 18:27:30
SPY                  daily      SPY Adjusted Close             6662     1999-11-01 2026-04-28 2026-04-29 18:27:32
WTI                  daily      Crude Oil Prices WTI           10143    1986-01-02 2026-04-20 2026-04-29 18:31:55
```

The **From** / **To** columns always show the date part only (no time component), regardless of whether the series is intraday.

## Direct Subcommands Reference

### `run-real`

Runs a registered real-data experiment end-to-end using the DuckDB cache as input:

```
cargo run -- run-real --id real_spy_daily_hard_switch
cargo run -- run-real --id real_spy_intraday_hard_switch --cache data/commodities.duckdb --save ./output
```

The six registered experiment IDs are:

| ID | Type | Description |
|----|------|-------------|
| `hard_switch` | Synthetic | HardSwitch, 2-regime, LogReturn/ZScore |
| `posterior_transition` | Synthetic | PosteriorTransition, 2-regime, LogReturn/ZScore |
| `surprise` | Synthetic | Surprise, 2-regime, LogReturn/ZScore |
| `real_spy_daily_hard_switch` | Real | SPY daily, HardSwitch, 2018–present |
| `real_wti_daily_surprise` | Real | WTI daily, Surprise, 2018–present |
| `real_spy_intraday_hard_switch` | Real | SPY 15-min session-aware, HardSwitch, 2022–2025 |

Each real run produces 16 artifacts in `runs/real/<id>/<run_id>/`.

### `calibrate`

Calibrates a synthetic experiment against empirical data and writes three JSON files:

```
cargo run -- calibrate --id hard_switch --out ./calibration
```

Output files:
- `calibration_summary.json` — empirical vs synthetic statistics, verification pass/fail
- `synthetic_vs_empirical_summary.json` — side-by-side comparison table
- `calibrated_scenario.json` — calibrated `ModelParams` ready for use in synthetic runs

### `run-batch`

Runs a list of JSON experiment configs in sequence:

```
cargo run -- run-batch --config a.json --config b.json --save ./batch_out
```

Writes `batch_summary.json` to the save directory with per-run status, IDs, and metrics.

## Cancellation

Pressing **Esc** or **Ctrl+C** at any sub-prompt cancels that action and returns to the main menu cleanly — no partial state is written.
