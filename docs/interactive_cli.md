# Interactive CLI

Proteus uses a fully interactive terminal UI powered by [inquire](https://github.com/mikaelmello/inquire). There are no command-line flags — everything is driven by arrow-key menus, confirms, and text inputs.

## Starting the app

```
cargo run
```

An optional config path can be passed as the first argument (defaults to `config.toml`):

```
cargo run -- path/to/other.toml
```

## Main menu

```
? What would you like to do?
> Ingest    — fetch & cache configured series
  Show      — view cached data for a series
  Refresh   — force-refresh a series from the API
  Status    — cache overview
  Quit
```

Navigate with the arrow keys, confirm with Enter. Press **Esc** or **Ctrl+C** at any prompt to cancel back to the main menu.

## Actions

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
SPY                  60min      SPDR S&P 500 ETF Trust         390      2026-04-13   2026-04-14   2026-04-15 09:01:44
BRENT                monthly    Brent (ICE) Crude Oil Prices   276      1987-05-01   2025-12-01   2026-04-13 14:32:01
WTI                  monthly    West Texas Intermediate...     276      1983-01-01   2025-12-01   2026-04-13 14:31:58
```

The **From** / **To** columns always show the date part only (no time component), regardless of whether the series is intraday.

## Cancellation

Pressing **Esc** or **Ctrl+C** at any sub-prompt cancels that action and returns to the main menu cleanly — no partial state is written.
