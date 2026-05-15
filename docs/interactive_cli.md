# Interactive CLI

Proteus supports two modes:
- **Interactive mode**: fully arrow-key driven menus powered by [inquire](https://github.com/mikaelmello/inquire).
- **Direct CLI mode**: subcommand-based, useful for scripting and reproducibility.

## Starting the app

```
cargo run              # interactive menu
cargo run -- e2e       # run all registered experiments end-to-end
cargo run -- param-search --id <experiment_id>
cargo run -- optimize  --id <experiment_id> [--cache <path>] [--save <dir>] [--top <n>] [--model]
cargo run -- run-experiment --config experiment_config.json
cargo run -- run-batch --config a.json --config b.json [--save <dir>]
cargo run -- run-real  --id <experiment_id> [--cache <path.duckdb>] [--save <dir>]
cargo run -- calibrate --id <experiment_id> [--out <dir>]
cargo run -- inspect --dir ./runs/real/my_run/run_001
cargo run -- generate-report --dir ./runs/real/my_run/run_001 [--cache <path.duckdb>]
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
| `posterior_transition_tv` | Synthetic | PosteriorTransitionTV, 2-regime, LogReturn/ZScore |
| `hard_switch_shock` | Synthetic | HardSwitch, shock-contaminated (jump noise path) |
| `hard_switch_frozen` | Synthetic | HardSwitch, loads pre-fitted model from `data/frozen_models/hard_switch_frozen` |
| `hard_switch_multi_start` | Synthetic | HardSwitch, multi-start EM (3 starts) |
| `real_spy_daily_hard_switch` | Real | SPY daily, HardSwitch, 2018–present |
| `real_wti_daily_surprise` | Real | WTI daily, Surprise, 2018–present |
| `real_spy_intraday_hard_switch` | Real | SPY 15-min RTH session-aware, HardSwitch, 2022–2025 |

For intraday experiments (`real_spy_intraday_hard_switch`), the pipeline automatically applies a **Regular Trading Hours (RTH) filter** — only bars within 09:30–15:59 ET are kept. Pre-market and after-hours bars are excluded before training and detection.

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

### `optimize`

Two-phase parameter search for real-data experiments. Phase 1 runs a grid on real
data (artifact writes disabled for speed) and ranks every point by a combined
coverage + precision score. Phase 2 re-runs the best configuration with full
artifact output.

Two search modes are available:

- **Detector-only** (default): sweeps `threshold`, `persistence_required`, and
  `cooldown`. Feature family and `k_regimes` are fixed at base-config values.
- **Joint model + detector** (`--model`): additionally sweeps `k_regimes` ∈ {2, 3}
  and five feature families (LogReturn, AbsReturn, SquaredReturn, RollingVol{5},
  RollingVol{20}). Intraday experiments use `session_reset: true` rolling-vol
  variants automatically.

```
# Detector-only
cargo run -- optimize --id real_spy_daily_hard_switch
cargo run -- optimize --id real_wti_daily_surprise --save ./runs/optimize/wti --top 15

# Joint model + detector
cargo run -- optimize --id real_spy_daily_hard_switch --model
cargo run -- optimize --id real_spy_intraday_hard_switch --model --top 20
```

Flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--id` | (required) | Registered experiment ID |
| `--cache` | `data/commodities.duckdb` | DuckDB price cache path |
| `--save` | `./runs/optimize/<id>/` | Output directory |
| `--top` | `10` | Rows shown in the printed ranking table |
| `--model` | off | Enable joint model + detector search |

> **Note (debug builds):** In debug mode (`cargo run` without `--release`) each EM
> fit is approximately 200× slower. The `optimize` command prints an estimated
> time for your grid size when running in debug mode and recommends
> `cargo build --release` for any grid larger than a few dozen points.

Default grids by detector type:

| Detector | Threshold range | Grid points (detector) | Joint grid points (×10 model combos) |
|----------|----------------|----------------------|--------------------------------------|
| `HardSwitch` | 0.30 – 0.80 | 128 | 1 280 |
| `Surprise` | 1.0 – 6.0 | 128 | 1 280 |
| `PosteriorTransition` | 0.10 – 0.50 | 84 | 840 |

Artifacts written to `--save`:

| File | Contents |
|------|----------|
| `search_report.json` | Full ranked grid — all N scored points |
| `search_summary.txt` | Human-readable top-N table + best params |
| `result.json` | Full `ExperimentResult` from best-config run |
| `config.snapshot.json` | Exact `ExperimentConfig` used for best run |
| `signal_alarms.png` | Alarm timeline (best params) |
| `detector_scores.png` | Detector score trace (best params) |
| `regime_posteriors.png` | Filtered posterior heatmap (best params) |
| All standard CSVs + JSONs | Full run artifact set |

### `run-batch`

Runs a list of JSON experiment configs in sequence. Each config is dispatched
through the backend matching its `mode` field (`Synthetic` → `SyntheticBackend`,
`Real` → `RealBackend`, `SimToReal` → `SimToRealBackend`):

```
cargo run -- run-batch --config a.json --config b.json --save ./batch_out
```

Writes `batch_summary.json` to the save directory with per-run status, IDs, and metrics.

### `generate-report`

Regenerates all plots and JSON artifacts for an existing run by replaying its recorded `config.snapshot.json`:

```
cargo run -- generate-report --dir ./runs/real/my_run/run_001
cargo run -- generate-report --dir ./runs/real/my_run/run_001 --cache data/commodities.duckdb
```

The command reads `<dir>/config.snapshot.json`, re-runs the full pipeline (including EM fitting), and writes a fresh artifact set with a new `run_id`. Files in the original run directory are **not** overwritten. Use `--cache` to specify a DuckDB path for real-data experiments (defaults to `data/commodities.duckdb`).

## Cancellation

Pressing **Esc** or **Ctrl+C** at any sub-prompt cancels that action and returns to the main menu cleanly — no partial state is written.
