use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use inquire::{Confirm, InquireError, Select, Text};

use crate::alphavantage::commodity::{CommodityEndpoint, Interval};
use crate::config::Config;
use crate::data_service::DataService;
use crate::experiments::batch::BatchResult;
use crate::experiments::config::{DataConfig, EvaluationConfig, ExperimentConfig, TrainingMode};
use crate::experiments::real_backend::RealBackend;
use crate::experiments::registry;
use crate::experiments::result::{EvaluationSummary, ExperimentResult, RunStage, RunStatus};
use crate::experiments::runner::{DryRunBackend, ExperimentRunner};
use crate::experiments::search::{ParamGrid, SearchPoint, grid_search};
use crate::experiments::synthetic_backend::SyntheticBackend;
use crate::reporting::table::{MetricsTableBuilder, MetricsTableRow};

// ---------------------------------------------------------------------------
// Top-level menu — only items that actually do something
// ---------------------------------------------------------------------------

enum MainMenu {
    Data,
    Experiments,
    InspectRuns,
    E2eRun,
    Exit,
}

impl fmt::Display for MainMenu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            MainMenu::Data => "Data          — ingest, inspect, and refresh market data",
            MainMenu::Experiments => "Experiments   — run, search, and inspect experiments",
            MainMenu::InspectRuns => "Inspect Runs  — browse and view saved run artifacts",
            MainMenu::E2eRun => "E2E Run       — run all registered experiments end-to-end",
            MainMenu::Exit => "Exit",
        })
    }
}

// ---------------------------------------------------------------------------
// Data sub-menu
// ---------------------------------------------------------------------------

enum DataAction {
    Ingest,
    Show,
    Refresh,
    Status,
    Back,
}

impl fmt::Display for DataAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DataAction::Ingest => "Ingest    — fetch & cache configured series",
            DataAction::Show => "Show      — view cached data for a series",
            DataAction::Refresh => "Refresh   — force-refresh a series from the API",
            DataAction::Status => "Status    — cache overview",
            DataAction::Back => "<- Back",
        })
    }
}

// ---------------------------------------------------------------------------
// Commodity selection helpers
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
// Entry points
// ---------------------------------------------------------------------------

/// Interactive mode: `cargo run` or `cargo run -- <config.toml>`
pub async fn run(cfg: Config) -> anyhow::Result<()> {
    println!();
    println!("  Proteus — Markov-Switching Changepoint Detection Platform");
    println!("  ---------------------------------------------------------");
    println!("  Press Esc or Ctrl+C at any prompt to go back / exit.\n");

    let service = DataService::new(&cfg)?;

    loop {
        println!();
        let action = match Select::new(
            "Main menu:",
            vec![
                MainMenu::Data,
                MainMenu::Experiments,
                MainMenu::InspectRuns,
                MainMenu::E2eRun,
                MainMenu::Exit,
            ],
        )
        .without_help_message()
        .prompt()
        {
            Ok(a) => a,
            Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => break,
            Err(e) => return Err(e.into()),
        };

        let result: anyhow::Result<()> = match action {
            MainMenu::Data => menu_data(&service, &cfg).await,
            MainMenu::Experiments => menu_experiments(),
            MainMenu::InspectRuns => menu_inspect_runs(),
            MainMenu::E2eRun => cmd_e2e_run(),
            MainMenu::Exit => break,
        };

        if let Err(e) = result {
            eprintln!("\n  Error: {e:#}");
        }
    }

    println!("\n  Goodbye.\n");
    Ok(())
}

/// Direct CLI dispatch: `cargo run -- <subcommand> [options]`
pub async fn run_direct(args: Vec<String>) -> anyhow::Result<()> {
    let cmd = args.first().map_or("help", String::as_str);
    match cmd {
        "run-experiment" => direct_run_experiment(&args),
        "run-batch" => direct_run_batch(&args),
        "run-real" => direct_run_real(&args),
        "calibrate" => direct_calibrate(&args),
        "compare-runs" => direct_compare_runs(&args),
        "compare-sim-vs-real" => direct_compare_sim_vs_real(&args),
        "e2e" => cmd_e2e_run(),
        "param-search" => {
            let id = flag_value(&args, "--id").unwrap_or_else(|| "surprise".to_string());
            cmd_param_search(&id)
        }
        "optimize" => direct_optimize(&args),
        "inspect" => direct_inspect(&args),
        "generate-report" => direct_generate_report(&args),
        "status" => {
            let config_path =
                flag_value(&args, "--config").unwrap_or_else(|| "config.toml".to_string());
            let cfg = crate::config::Config::from_file(&config_path)?;
            let service = DataService::new(&cfg)?;
            cmd_status(&service).await
        }
        _ => {
            print_help();
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Data commands
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
            anyhow::anyhow!("No cached data for {endpoint} ({interval}). Run Ingest first.")
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

    println!("\nRefreshing {endpoint} ({interval}) ...");
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
        "\n{:<20} {:<10} {:<30} {:<8} {:<8} {:<12} {:<12} Last Fetched",
        "Symbol", "Interval", "Name", "Unit", "Points", "From", "To"
    );
    println!("{:-<120}", "");
    for s in &statuses {
        println!(
            "{:<20} {:<10} {:<30} {:<8} {:<8} {:<12} {:<12} {}",
            s.symbol,
            s.interval,
            s.name,
            s.unit,
            s.point_count,
            s.from_date.date(),
            s.to_date.date(),
            s.last_fetched.format("%Y-%m-%d %H:%M:%S")
        );
    }
    println!();
    Ok(())
}

// ---------------------------------------------------------------------------
// Data sub-menu
// ---------------------------------------------------------------------------

async fn menu_data(service: &DataService, cfg: &Config) -> anyhow::Result<()> {
    loop {
        println!();
        let action = match Select::new(
            "Data:",
            vec![
                DataAction::Ingest,
                DataAction::Show,
                DataAction::Refresh,
                DataAction::Status,
                DataAction::Back,
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
            DataAction::Ingest => cmd_ingest(service, cfg).await,
            DataAction::Show => cmd_show(service).await,
            DataAction::Refresh => cmd_refresh(service).await,
            DataAction::Status => cmd_status(service).await,
            DataAction::Back => break,
        };

        if let Err(e) = result {
            eprintln!("\n  Error: {e:#}");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Experiments sub-menu
// ---------------------------------------------------------------------------

fn menu_experiments() -> anyhow::Result<()> {
    enum ExperimentsAction {
        ListRegistry,
        RunFromRegistry,
        RunAllE2E,
        RunRealExperiment,
        CalibrateScenario,
        ParamSearch,
        ValidateConfig,
        ShowConfigTemplate,
        Back,
    }
    impl fmt::Display for ExperimentsAction {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(match self {
                ExperimentsAction::ListRegistry => {
                    "List Registry           — show all registered experiments"
                }
                ExperimentsAction::RunFromRegistry => {
                    "Run from Registry       — pick and run a registered experiment"
                }
                ExperimentsAction::RunAllE2E => {
                    "Run All (E2E)           — run every registered experiment with full logging"
                }
                ExperimentsAction::RunRealExperiment => {
                    "Run Real Experiment     — run a real-data experiment from the registry"
                }
                ExperimentsAction::CalibrateScenario => {
                    "Calibrate Scenario      — calibrate a synthetic scenario to empirical targets"
                }
                ExperimentsAction::ParamSearch => {
                    "Parameter Search        — grid search over detector parameters"
                }
                ExperimentsAction::ValidateConfig => {
                    "Validate Config File    — load and validate an experiment config file"
                }
                ExperimentsAction::ShowConfigTemplate => {
                    "Show Config Template    — print an example ExperimentConfig as JSON"
                }
                ExperimentsAction::Back => "<- Back",
            })
        }
    }
    loop {
        println!();
        let action = match Select::new(
            "Experiments:",
            vec![
                ExperimentsAction::ListRegistry,
                ExperimentsAction::RunFromRegistry,
                ExperimentsAction::RunAllE2E,
                ExperimentsAction::RunRealExperiment,
                ExperimentsAction::CalibrateScenario,
                ExperimentsAction::ParamSearch,
                ExperimentsAction::ValidateConfig,
                ExperimentsAction::ShowConfigTemplate,
                ExperimentsAction::Back,
            ],
        )
        .without_help_message()
        .prompt()
        {
            Ok(a) => a,
            Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => break,
            Err(e) => return Err(e.into()),
        };
        match action {
            ExperimentsAction::ListRegistry => {
                println!();
                println!("  Registered Experiments:");
                println!("  -----------------------");
                for (i, entry) in registry::registry().iter().enumerate() {
                    println!("  [{}] {}  —  {}", i + 1, entry.id, entry.description);
                }
            }
            ExperimentsAction::RunFromRegistry => {
                let reg = registry::registry();
                let choices: Vec<String> = reg
                    .iter()
                    .map(|e| format!("{} — {}", e.id, e.description))
                    .collect();
                let idx = match Select::new("Pick experiment:", choices).prompt() {
                    Ok(choice) => reg
                        .iter()
                        .position(|e| choice.starts_with(e.id))
                        .unwrap_or(0),
                    Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => {
                        continue;
                    }
                    Err(e) => return Err(e.into()),
                };
                let entry = &reg[idx];
                let cfg = (entry.build)();
                println!();
                print_config_block(&cfg);
                println!();
                println!("  Running '{}'...", cfg.meta.run_label);
                let runner = ExperimentRunner::new(DryRunBackend);
                let result = runner.run(cfg.clone());
                println!();
                println!("  Pipeline:");
                print_stage_log(&cfg, &result);
                println!();
                print_run_result_summary(&result);
            }
            ExperimentsAction::RunAllE2E => {
                cmd_e2e_run()?;
            }
            ExperimentsAction::RunRealExperiment => {
                cmd_run_real_experiment()?;
            }
            ExperimentsAction::CalibrateScenario => {
                cmd_calibrate_scenario()?;
            }
            ExperimentsAction::ParamSearch => {
                let reg = registry::registry();
                let choices: Vec<String> = reg
                    .iter()
                    .map(|e| format!("{} — {}", e.id, e.description))
                    .collect();
                let idx = match Select::new("Base experiment for search:", choices).prompt() {
                    Ok(choice) => reg
                        .iter()
                        .position(|e| choice.starts_with(e.id))
                        .unwrap_or(0),
                    Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => {
                        continue;
                    }
                    Err(e) => return Err(e.into()),
                };
                cmd_param_search(reg[idx].id)?;
            }
            ExperimentsAction::ValidateConfig => {
                let config_path = match Text::new("Path to experiment config (JSON or TOML):")
                    .with_default("experiment_config.json")
                    .prompt()
                {
                    Ok(s) => s,
                    Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => {
                        continue;
                    }
                    Err(e) => return Err(e.into()),
                };
                match load_experiment_config(&config_path) {
                    Ok(cfg) => match cfg.validate() {
                        Ok(()) => {
                            println!("\n  [ok] Config is valid.");
                            print_config_block(&cfg);
                        }
                        Err(e) => eprintln!("\n  [!] Validation failed: {e}"),
                    },
                    Err(e) => eprintln!("\n  [!] Could not load config: {e}"),
                }
            }
            ExperimentsAction::ShowConfigTemplate => {
                let tmpl = make_template_config();
                let pretty = serde_json::to_string_pretty(&tmpl).unwrap_or_default();
                println!();
                println!("  Experiment Config Template (JSON):");
                println!("  -----------------------------------");
                for line in pretty.lines() {
                    println!("  {line}");
                }
            }
            ExperimentsAction::Back => break,
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Calibrate Scenario — empirical profile from a synthetic run → CalibrationReport
// ---------------------------------------------------------------------------

fn cmd_calibrate_scenario() -> anyhow::Result<()> {
    use crate::calibration::{
        CalibrationDatasetTag, CalibrationMappingConfig, CalibrationPartition,
        EmpiricalCalibrationProfile, SummaryTargetSet, VerificationTolerance,
        run_calibration_workflow, summarize_observation_values,
    };
    use crate::experiments::config::{ExperimentMode, FeatureFamilyConfig};
    use crate::experiments::runner::ExperimentBackend;
    use crate::features::FeatureFamily;

    // Only synthetic experiments make sense for the offline calibration workflow.
    let reg = registry::registry();
    let syn_entries: Vec<_> = reg
        .iter()
        .filter(|e| (e.build)().mode == ExperimentMode::Synthetic)
        .collect();

    if syn_entries.is_empty() {
        println!("\n  No synthetic experiments found in registry.");
        return Ok(());
    }

    let choices: Vec<String> = syn_entries
        .iter()
        .map(|e| format!("{} — {}", e.id, e.description))
        .collect();
    let idx = match Select::new("Pick synthetic experiment to calibrate:", choices).prompt() {
        Ok(choice) => syn_entries
            .iter()
            .position(|e| choice.starts_with(e.id))
            .unwrap_or(0),
        Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => {
            return Ok(());
        }
        Err(e) => return Err(e.into()),
    };

    let entry = syn_entries[idx];
    let cfg = (entry.build)();

    println!();
    println!("  Calibrating '{}'...", cfg.meta.run_label);
    println!("  (Simulating data and computing empirical targets — this is fast.)");
    println!();

    // Run the synthetic backend just to get feature observations.
    let backend = SyntheticBackend::new();
    let data = backend.resolve_data(&cfg)?;
    let features = backend.build_features(&cfg, &data)?;

    if features.observations.is_empty() {
        println!("  [!] No feature observations produced — cannot calibrate.");
        return Ok(());
    }

    // Summarise empirical targets from the training partition only.
    let train_obs = &features.observations[..features.train_n.min(features.observations.len())];
    let empirical_summary = summarize_observation_values(train_obs);

    // Build the calibration profile.
    let tag = CalibrationDatasetTag {
        asset: match &cfg.data {
            DataConfig::Synthetic { scenario_id, .. } => format!("synthetic:{scenario_id}"),
            DataConfig::Real { asset, .. } => asset.clone(),
            DataConfig::CalibratedSynthetic { real_asset, .. } => real_asset.clone(),
        },
        frequency: "synthetic".to_string(),
        feature_label: format!("{:?}", cfg.features.family),
        partition: CalibrationPartition::TrainOnly,
    };
    let feature_family = match &cfg.features.family {
        FeatureFamilyConfig::LogReturn => FeatureFamily::LogReturn,
        FeatureFamilyConfig::AbsReturn => FeatureFamily::AbsReturn,
        FeatureFamilyConfig::SquaredReturn => FeatureFamily::SquaredReturn,
        FeatureFamilyConfig::RollingVol {
            window,
            session_reset,
        } => FeatureFamily::RollingVol {
            window: *window,
            session_reset: *session_reset,
        },
        FeatureFamilyConfig::StandardizedReturn {
            window,
            epsilon,
            session_reset,
        } => FeatureFamily::StandardizedReturn {
            window: *window,
            epsilon: *epsilon,
            session_reset: *session_reset,
        },
    };
    let profile = EmpiricalCalibrationProfile {
        tag,
        feature_family,
        targets: SummaryTargetSet::Full,
        summary: empirical_summary,
        observations: Vec::new(),
    };

    // Run full calibration workflow.
    let seed = cfg.reproducibility.seed.unwrap_or(42);
    let mapping = CalibrationMappingConfig {
        k: cfg.model.k_regimes,
        horizon: match &cfg.data {
            DataConfig::Synthetic { horizon, .. } => *horizon,
            DataConfig::Real { .. } => 2000,
            DataConfig::CalibratedSynthetic { horizon, .. } => *horizon,
        },
        ..CalibrationMappingConfig::default()
    };
    let report =
        run_calibration_workflow(profile, mapping, VerificationTolerance::default(), seed)?;
    let view = report.view();

    println!("  Calibration Report");
    println!("  ------------------");
    println!("  Asset         : {}", view.asset);
    println!("  Feature       : {}", view.feature_label);
    println!("  Horizon       : {}", view.horizon);
    println!("  Empirical n   : {}", view.empirical_n);
    println!("  Synthetic n   : {}", view.synthetic_n);
    println!(
        "  Verification  : {}",
        if view.verification_passed {
            "PASSED"
        } else {
            "FAILED"
        }
    );
    if !view.verification_notes.is_empty() {
        println!("  Notes:");
        for note in &view.verification_notes {
            println!("    - {note}");
        }
    }
    println!("  Expected durations:");
    for (j, d) in view.expected_durations.iter().enumerate() {
        println!(
            "    Regime {j}: {d:.1} bars (p_jj = {:.4})",
            1.0 - 1.0 / d.max(1.0)
        );
    }

    // Empirical statistics
    let s = &report.empirical_profile.summary;
    println!();
    println!("  Empirical Summary (train partition):");
    println!(
        "    mean={:.6}  var={:.6}  std={:.6}",
        s.mean, s.variance, s.std_dev
    );
    println!(
        "    q01={:.4}  q05={:.4}  q50={:.4}  q95={:.4}  q99={:.4}",
        s.q01, s.q05, s.q50, s.q95, s.q99
    );
    println!(
        "    acf1={:.4}  abs_acf1={:.4}  sign_change_rate={:.4}",
        s.acf1, s.abs_acf1, s.sign_change_rate
    );
    println!(
        "    high_episode_mean_dur={:.1}  low_episode_mean_dur={:.1}",
        s.high_episode_mean_duration, s.low_episode_mean_duration
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Run Real Experiment — interactive selection + RealBackend execution
// ---------------------------------------------------------------------------

fn cmd_run_real_experiment() -> anyhow::Result<()> {
    use crate::experiments::config::ExperimentMode;

    // Filter registry to real experiments only.
    let reg = registry::registry();
    let real_entries: Vec<_> = reg
        .iter()
        .filter(|e| (e.build)().mode == ExperimentMode::Real)
        .collect();

    if real_entries.is_empty() {
        println!();
        println!("  No real-data experiments found in the registry.");
        println!("  Add some in src/experiments/registry.rs using ExperimentMode::Real.");
        return Ok(());
    }

    println!();
    println!("  Real-Data Experiments:");
    println!("  ----------------------");
    println!("  These experiments require cached market data.");
    println!("  Run 'Data > Ingest' first if you have not done so.");
    println!();

    let choices: Vec<String> = real_entries
        .iter()
        .map(|e| format!("{} — {}", e.id, e.description))
        .collect();

    let idx = match Select::new("Pick real experiment:", choices).prompt() {
        Ok(choice) => real_entries
            .iter()
            .position(|e| choice.starts_with(e.id))
            .unwrap_or(0),
        Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => {
            return Ok(());
        }
        Err(e) => return Err(e.into()),
    };

    let entry = real_entries[idx];
    let cfg = (entry.build)();

    // Prompt for cache path (default from the standard location).
    let cache_path = match Text::new("DuckDB cache path:")
        .with_default("data/commodities.duckdb")
        .prompt()
    {
        Ok(s) => s,
        Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => {
            return Ok(());
        }
        Err(e) => return Err(e.into()),
    };

    println!();
    print_config_block(&cfg);
    println!();
    println!("  Cache : {cache_path}");
    println!();
    println!("  Running '{}'...", cfg.meta.run_label);
    println!("  (This may take several seconds while loading data and running EM.)");
    println!();

    let backend = RealBackend::new(cache_path);
    let runner = ExperimentRunner::new(backend);
    let result = runner.run(cfg.clone());

    println!();
    println!("  Pipeline:");
    print_stage_log(&cfg, &result);
    println!();
    print_run_result_summary(&result);
    println!();

    // Print real-eval metrics if available.
    if let Some(crate::experiments::result::EvaluationSummary::Real {
        event_coverage,
        alarm_relevance,
        segmentation_coherence,
    }) = &result.evaluation_summary
    {
        println!("  Real-Eval Metrics:");
        println!("  ------------------");
        println!("  Route A  event_coverage     = {event_coverage:.4}");
        println!("  Route A  alarm_relevance     = {alarm_relevance:.4}");
        println!("  Route B  segmentation_coherence = {segmentation_coherence:.4}");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// E2E run — verbose, step-by-step, with metrics table
// ---------------------------------------------------------------------------

fn cmd_e2e_run() -> anyhow::Result<()> {
    use crate::experiments::config::ExperimentMode;

    let reg = registry::registry();
    // E2E run only covers synthetic experiments; real experiments require
    // cached market data and are run separately via Run Real Experiment.
    let synthetic_entries: Vec<_> = reg
        .iter()
        .filter(|e| (e.build)().mode == ExperimentMode::Synthetic)
        .collect();
    let total = synthetic_entries.len();

    println!();
    println!("  ==============================================================");
    println!("  E2E Run — {total} synthetic experiments");
    println!("  (Real-data experiments skipped — use 'Run Real Experiment')");
    println!("  ==============================================================");

    let runner = ExperimentRunner::new(SyntheticBackend::new());
    let mut all_results: Vec<(ExperimentConfig, ExperimentResult)> = Vec::new();

    for (i, entry) in synthetic_entries.iter().enumerate() {
        let cfg = (entry.build)();

        println!();
        println!("  +-------------------------------------------------------------");
        println!("  | [{}/{}]  {}", i + 1, total, entry.id);
        println!("  |         {}", entry.description);
        println!("  +-------------------------------------------------------------");

        // -- Config ----------------------------------------------------------
        println!();
        print_config_block(&cfg);

        // -- Run -------------------------------------------------------------
        println!();
        println!("  Pipeline:");
        let result = runner.run(cfg.clone());
        print_stage_log(&cfg, &result);

        // -- Per-run result ---------------------------------------------------
        println!();
        let ok = result.is_success();
        println!(
            "  Result  : {}  run_id={}",
            if ok { "SUCCESS" } else { "FAILED" },
            result.metadata.run_id
        );

        if let Some(eval) = &result.evaluation_summary {
            match eval {
                EvaluationSummary::Synthetic {
                    coverage,
                    precision_like,
                    n_events,
                    precision,
                    recall,
                    miss_rate,
                    false_alarm_rate,
                    delay_mean,
                    delay_median,
                    ..
                } => {
                    let prec = precision.unwrap_or(*precision_like);
                    let rec = recall.unwrap_or(*coverage);
                    println!(
                        "  Metrics : prec={:.4}  recall={:.4}  n_events={}  n_alarms={}",
                        prec,
                        rec,
                        n_events,
                        result.detector_summary.as_ref().map_or(0, |d| d.n_alarms)
                    );
                    if let Some(mr) = miss_rate {
                        println!(
                            "  Miss rate: {mr:.4}  FAR: {:.6}",
                            false_alarm_rate.unwrap_or(0.0)
                        );
                    }
                    if let Some(dm) = delay_mean {
                        println!(
                            "  Delay   : mean={dm:.1}  median={:.1}",
                            delay_median.unwrap_or(0.0)
                        );
                    }
                }
                EvaluationSummary::Real {
                    event_coverage,
                    alarm_relevance,
                    segmentation_coherence,
                } => {
                    println!(
                        "  Metrics : event_cov={event_coverage:.4}  relevance={alarm_relevance:.4}  coherence={segmentation_coherence:.4}"
                    );
                }
            }
        }

        // Show fitted model params if available
        if let Some(ms) = &result.model_summary
            && let Some(fp) = &ms.fitted_params
        {
            println!(
                "  Model   : K={}  LL={:.4}  iter={}  converged={}",
                ms.k_regimes, fp.log_likelihood, fp.n_iter, fp.converged
            );
            for j in 0..ms.k_regimes {
                println!(
                    "  Regime {}: mu={:.6}  var={:.6}  pi={:.4}",
                    j, fp.means[j], fp.variances[j], fp.pi[j]
                );
            }
            for i in 0..ms.k_regimes {
                let row: Vec<String> = fp.transition[i].iter().map(|v| format!("{v:.4}")).collect();
                println!("  Trans[{}]: [{}]", i, row.join(", "));
            }
        }

        if !result.artifacts.is_empty()
            && let Some(first) = result.artifacts.first()
        {
            let p = Path::new(&first.path);
            if let Some(dir) = p.parent() {
                println!("  Artifacts: {}", dir.display());
            }
        }

        for w in &result.warnings {
            println!("  Warning  : {w}");
        }

        all_results.push((cfg, result));
    }

    // -- Aggregate metrics table ----------------------------------------------
    println!();
    println!("  ==============================================================");
    println!("  Aggregate Metrics");
    println!("  ==============================================================");

    let results_only: Vec<&ExperimentResult> = all_results.iter().map(|(_, r)| r).collect();
    let table_md = build_metrics_table(&results_only);

    if !table_md.is_empty() {
        println!();
        for line in table_md.lines() {
            println!("  {line}");
        }
        if let Err(e) = fs::create_dir_all("./runs") {
            eprintln!("  [!] Could not create ./runs: {e}");
        } else {
            let table_path = PathBuf::from("./runs/metrics_table.md");
            match fs::write(&table_path, &table_md) {
                Ok(()) => println!("\n  Metrics table written to: {}", table_path.display()),
                Err(e) => eprintln!("  [!] Could not write metrics table: {e}"),
            }
        }
    }

    let n_success = all_results.iter().filter(|(_, r)| r.is_success()).count();
    let n_failed = all_results.len() - n_success;
    println!();
    println!("  Completed: {n_success}  Failed: {n_failed}");
    Ok(())
}

// ---------------------------------------------------------------------------
// Parameter search
// ---------------------------------------------------------------------------

fn cmd_param_search(experiment_id: &str) -> anyhow::Result<()> {
    let reg = registry::registry();
    let entry = reg.iter().find(|e| e.id == experiment_id).ok_or_else(|| {
        anyhow::anyhow!(
            "Unknown experiment id '{}'. Known: {}",
            experiment_id,
            reg.iter().map(|e| e.id).collect::<Vec<_>>().join(", ")
        )
    })?;

    let base = (entry.build)();
    let grid = ParamGrid::default();

    println!();
    println!("  Parameter Search — base: {}", entry.id);
    println!("  --------------------------------------------------------------");
    println!("  Detector   : {:?}", base.detector.detector_type);
    println!(
        "  Grid size  : {} points ({} thresholds x {} persistence x {} cooldown)",
        grid.n_points(),
        grid.thresholds.len(),
        grid.persistence_values.len(),
        grid.cooldown_values.len()
    );
    println!("  Thresholds : {:?}", grid.thresholds);
    println!("  Persistence: {:?}", grid.persistence_values);
    println!("  Cooldown   : {:?}", grid.cooldown_values);
    println!();
    println!("  Running {} evaluations...", grid.n_points());

    let runner = ExperimentRunner::new(DryRunBackend);
    let results = grid_search(&runner, &base, &grid);

    println!();
    print_search_results(&results, results.len());
    Ok(())
}

// ---------------------------------------------------------------------------
// Inspect Runs sub-menu
// ---------------------------------------------------------------------------

fn menu_inspect_runs() -> anyhow::Result<()> {
    enum InspectAction {
        ListRuns,
        ViewResult,
        ViewConfigSnapshot,
        ViewArtifactTree,
        Back,
    }
    impl fmt::Display for InspectAction {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(match self {
                InspectAction::ListRuns => "List Runs           — show all saved runs",
                InspectAction::ViewResult => "View Result         — show result.json for a run",
                InspectAction::ViewConfigSnapshot => {
                    "View Config         — show config.snapshot.json for a run"
                }
                InspectAction::ViewArtifactTree => {
                    "View Artifact Tree  — show all files in a run directory"
                }
                InspectAction::Back => "<- Back",
            })
        }
    }
    loop {
        println!();
        let action = match Select::new(
            "Inspect Runs:",
            vec![
                InspectAction::ListRuns,
                InspectAction::ViewResult,
                InspectAction::ViewConfigSnapshot,
                InspectAction::ViewArtifactTree,
                InspectAction::Back,
            ],
        )
        .without_help_message()
        .prompt()
        {
            Ok(a) => a,
            Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => break,
            Err(e) => return Err(e.into()),
        };
        match action {
            InspectAction::ListRuns => {
                let runs_base = Path::new("./runs");
                if !runs_base.exists() {
                    println!(
                        "\n  No ./runs directory found. Run experiments first to create run artifacts."
                    );
                    continue;
                }
                let mut count = 0usize;
                println!();
                println!("  Saved Runs (./runs):");
                println!("  --------------------");
                list_runs_recursive(runs_base, &mut count);
                if count == 0 {
                    println!("  (no runs found)");
                } else {
                    println!("\n  Total: {count} run(s)");
                }
            }
            InspectAction::ViewResult => {
                let run_dir = match Text::new("Run directory path:").prompt() {
                    Ok(s) => s,
                    Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => {
                        continue;
                    }
                    Err(e) => return Err(e.into()),
                };
                let path = PathBuf::from(&run_dir).join("result.json");
                show_json_file(&path);
            }
            InspectAction::ViewConfigSnapshot => {
                let run_dir = match Text::new("Run directory path:").prompt() {
                    Ok(s) => s,
                    Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => {
                        continue;
                    }
                    Err(e) => return Err(e.into()),
                };
                let path = PathBuf::from(&run_dir).join("config.snapshot.json");
                show_json_file(&path);
            }
            InspectAction::ViewArtifactTree => {
                let run_dir = match Text::new("Run directory path:").prompt() {
                    Ok(s) => s,
                    Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => {
                        continue;
                    }
                    Err(e) => return Err(e.into()),
                };
                println!();
                show_artifact_tree(Path::new(&run_dir), 0);
            }
            InspectAction::Back => break,
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Direct CLI handlers
// ---------------------------------------------------------------------------

fn direct_run_experiment(args: &[String]) -> anyhow::Result<()> {
    let config_path = flag_value(args, "--config")
        .ok_or_else(|| anyhow::anyhow!("Usage: run-experiment --config <path.json>"))?;
    let cfg = load_experiment_config(&config_path)?;
    cfg.validate()?;

    // Prominently warn the user that run-experiment uses DryRunBackend and
    // produces mock metrics.  No real data is loaded and the math stack is
    // not exercised.  Users who want real results should use `run-real`.
    eprintln!();
    eprintln!("  WARNING: run-experiment uses DryRunBackend — no real data is loaded.");
    eprintln!("  All metrics (alarms, coverage, precision) are synthetic mock values.");
    eprintln!("  For real-data experiments use:  run-real --id <experiment_id>");
    eprintln!("  For end-to-end synthetic runs use:  e2e");
    eprintln!();

    println!();
    print_config_block(&cfg);
    println!();
    println!("Running '{}'...", cfg.meta.run_label);
    let runner = ExperimentRunner::new(DryRunBackend);
    let result = runner.run(cfg.clone());
    println!();
    println!("Pipeline:");
    print_stage_log(&cfg, &result);
    println!();
    print_run_result_summary(&result);
    Ok(())
}

fn direct_run_batch(args: &[String]) -> anyhow::Result<()> {
    use crate::experiments::config::ExperimentMode;

    let config_paths = collect_flag_values(args, "--config");
    if config_paths.is_empty() {
        anyhow::bail!(
            "Usage: run-batch --config a.json --config b.json ... [--cache <path>] [--stop-on-error]"
        );
    }

    let cache =
        flag_value(args, "--cache").unwrap_or_else(|| "data/commodities.duckdb".to_string());

    let mut configs = Vec::new();
    for path in &config_paths {
        let cfg = load_experiment_config(path)?;
        cfg.validate()?;
        configs.push(cfg);
    }
    let stop_on_error = args.contains(&"--stop-on-error".to_string());

    // Dispatch per-config by mode so real and sim-to-real configs load real
    // data via the proper backends instead of returning mock metrics.
    println!("Running batch of {} experiments...", configs.len());
    let mut results: Vec<ExperimentResult> = Vec::with_capacity(configs.len());
    let mut n_success = 0usize;
    let mut n_failed = 0usize;
    for cfg in configs {
        let r = match cfg.mode {
            ExperimentMode::Synthetic => {
                let runner = ExperimentRunner::new(
                    crate::experiments::synthetic_backend::SyntheticBackend::new(),
                );
                runner.run(cfg)
            }
            ExperimentMode::Real => {
                let runner = ExperimentRunner::new(
                    crate::experiments::real_backend::RealBackend::new(&cache),
                );
                runner.run(cfg)
            }
            ExperimentMode::SimToReal => {
                let runner = ExperimentRunner::new(
                    crate::experiments::sim_to_real_backend::SimToRealBackend::new(&cache),
                );
                runner.run(cfg)
            }
        };
        if r.is_success() {
            n_success += 1;
        } else {
            n_failed += 1;
            if stop_on_error {
                results.push(r);
                break;
            }
        }
        results.push(r);
    }
    let batch = BatchResult {
        run_results: results,
        n_success,
        n_failed,
    };
    println!("Completed: {}  Failed: {}", batch.n_success, batch.n_failed);
    let results_only: Vec<&ExperimentResult> = batch.run_results.iter().collect();
    let table_md = build_metrics_table(&results_only);
    if !table_md.is_empty() {
        println!();
        for line in table_md.lines() {
            println!("{line}");
        }
    }

    // Save batch_summary.json if --save is specified or always.
    let save_dir = flag_value(args, "--save");
    {
        let out_dir = save_dir.as_deref().unwrap_or("./runs");
        let _ = fs::create_dir_all(out_dir);
        let run_summaries: Vec<_> = batch
            .run_results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "run_id": r.metadata.run_id,
                    "run_label": r.metadata.run_label,
                    "mode": r.metadata.mode,
                    "status": format!("{:?}", r.status),
                    "evaluation": r.evaluation_summary,
                })
            })
            .collect();
        let batch_summary = serde_json::json!({
            "n_runs": batch.run_results.len(),
            "n_success": batch.n_success,
            "n_failed": batch.n_failed,
            "runs": run_summaries,
        });
        let summary_path = PathBuf::from(out_dir).join("batch_summary.json");
        if let Ok(json) = serde_json::to_string_pretty(&batch_summary) {
            let _ = fs::write(&summary_path, json);
            println!("\nBatch summary written to: {}", summary_path.display());
        }
    }
    Ok(())
}

fn direct_inspect(args: &[String]) -> anyhow::Result<()> {
    let dir = flag_value(args, "--dir")
        .ok_or_else(|| anyhow::anyhow!("Usage: inspect --dir <run-directory>"))?;
    let path = Path::new(&dir);
    if !path.exists() {
        anyhow::bail!("Directory not found: {dir}");
    }
    println!("\nArtifact tree for: {dir}");
    show_artifact_tree(path, 0);
    Ok(())
}

/// Regenerate all plots and JSON artifacts for an existing run by replaying
/// the experiment from its `config.snapshot.json`.
///
/// Usage: `generate-report --dir <run-directory> [--cache <path>]`
///
/// Reads `<dir>/config.snapshot.json`, re-runs the experiment (including EM
/// fitting if the experiment mode is FitOffline), and writes fresh artifacts
/// to the same `output.root_dir` that was recorded in the snapshot.  The new
/// run will receive a fresh `run_id` (non-deterministic) so it does not
/// overwrite the original run directory.
///
/// For real-data experiments the `--cache` flag overrides the DuckDB path
/// (defaults to `data/commodities.duckdb`).
fn direct_generate_report(args: &[String]) -> anyhow::Result<()> {
    use crate::experiments::config::ExperimentMode;

    let dir = flag_value(args, "--dir").ok_or_else(|| {
        anyhow::anyhow!("Usage: generate-report --dir <run-directory> [--cache <path>]")
    })?;
    let cache =
        flag_value(args, "--cache").unwrap_or_else(|| "data/commodities.duckdb".to_string());

    let snapshot_path = PathBuf::from(&dir).join("config.snapshot.json");
    if !snapshot_path.exists() {
        anyhow::bail!(
            "No config.snapshot.json found in '{dir}'.\n\
             This does not appear to be a valid run directory.\n\
             Hint: run 'inspect --dir <path>' to verify the directory structure."
        );
    }

    let json = fs::read_to_string(&snapshot_path)?;
    let mut cfg: ExperimentConfig = serde_json::from_str(&json)
        .map_err(|e| anyhow::anyhow!("Failed to parse config.snapshot.json: {e}"))?;

    // Ensure all artifact types are written.
    cfg.output.write_json = true;
    cfg.output.write_csv = true;
    cfg.output.save_traces = true;
    // Use a fresh non-deterministic run_id so this doesn't collide with the
    // original run directory.
    cfg.reproducibility.deterministic_run_id = false;

    println!();
    println!("Regenerating report from: {dir}");
    print_config_block(&cfg);
    println!();

    let result = match cfg.mode {
        ExperimentMode::Synthetic => {
            println!("Running SyntheticBackend (re-fitting model)...");
            println!();
            let backend = crate::experiments::synthetic_backend::SyntheticBackend::new();
            let runner = crate::experiments::runner::ExperimentRunner::new(backend);
            runner.run(cfg.clone())
        }
        ExperimentMode::Real => {
            println!("Running RealBackend (cache: {cache}) — re-fitting model...");
            println!();
            let backend = crate::experiments::real_backend::RealBackend::new(&cache);
            let runner = crate::experiments::runner::ExperimentRunner::new(backend);
            runner.run(cfg.clone())
        }
        ExperimentMode::SimToReal => {
            println!("Running SimToRealBackend (cache: {cache}) — re-fitting model...");
            println!();
            let backend = crate::experiments::sim_to_real_backend::SimToRealBackend::new(&cache);
            let runner = crate::experiments::runner::ExperimentRunner::new(backend);
            runner.run(cfg.clone())
        }
    };

    println!();
    println!("Pipeline:");
    print_stage_log(&cfg, &result);
    println!();
    println!(
        "Status: {}",
        if result.is_success() {
            "SUCCESS"
        } else {
            "FAILED"
        }
    );

    if let Some(first) = result.artifacts.first() {
        let p = PathBuf::from(&first.path);
        if let Some(run_dir) = p.parent() {
            println!("\nArtifacts written to: {}", run_dir.display());
        }
    }
    for w in &result.warnings {
        println!("Warning: {w}");
    }
    Ok(())
}

/// Run a real-data experiment by registry ID using `RealBackend`.
///
/// Usage: `run-real --id <experiment_id> [--cache <path>] [--save <out_dir>]`
///
/// The `--save` flag copies the run directory to `<out_dir>` for archiving.
fn direct_run_real(args: &[String]) -> anyhow::Result<()> {
    let id = flag_value(args, "--id").ok_or_else(|| {
        anyhow::anyhow!("Usage: run-real --id <experiment_id> [--cache <path>] [--save <out_dir>]")
    })?;
    let cache =
        flag_value(args, "--cache").unwrap_or_else(|| "data/commodities.duckdb".to_string());
    let save_to = flag_value(args, "--save");

    let reg = crate::experiments::registry::registry();
    let entry = reg.iter().find(|e| e.id == id).ok_or_else(|| {
        let known = reg.iter().map(|e| e.id).collect::<Vec<_>>().join(", ");
        anyhow::anyhow!("Unknown experiment id '{id}'. Known: {known}")
    })?;

    let cfg = (entry.build)();
    println!();
    print_config_block(&cfg);
    println!();
    println!("Cache : {cache}");
    println!();
    println!("Running '{}'...", cfg.meta.run_label);
    println!("(Loading data and running EM — this may take a few seconds.)");
    println!();

    // Dispatch by mode so SimToReal experiments use SimToRealBackend.
    let result = match cfg.mode {
        crate::experiments::config::ExperimentMode::SimToReal => {
            let backend = crate::experiments::sim_to_real_backend::SimToRealBackend::new(&cache);
            crate::experiments::runner::ExperimentRunner::new(backend).run(cfg.clone())
        }
        _ => {
            let backend = crate::experiments::real_backend::RealBackend::new(&cache);
            crate::experiments::runner::ExperimentRunner::new(backend).run(cfg.clone())
        }
    };

    println!();
    println!("Pipeline:");
    print_stage_log(&cfg, &result);
    println!();
    print_run_result_summary(&result);

    if let Some(crate::experiments::result::EvaluationSummary::Real {
        event_coverage,
        alarm_relevance,
        segmentation_coherence,
    }) = &result.evaluation_summary
    {
        println!();
        println!("Real-Eval Metrics:");
        println!("  Route A  event_coverage          = {event_coverage:.4}");
        println!("  Route A  alarm_relevance          = {alarm_relevance:.4}");
        println!("  Route B  segmentation_coherence   = {segmentation_coherence:.4}");
    }

    // Show model parameters.
    if let Some(ms) = &result.model_summary
        && let Some(fp) = &ms.fitted_params
    {
        println!();
        println!("Fitted Model:");
        println!(
            "  K={}  LL={:.4}  iter={}  converged={}",
            ms.k_regimes, fp.log_likelihood, fp.n_iter, fp.converged
        );
        for j in 0..ms.k_regimes {
            println!(
                "  Regime {j}: mu={:.6}  var={:.6}  pi={:.4}",
                fp.means[j], fp.variances[j], fp.pi[j]
            );
        }
        for i in 0..ms.k_regimes {
            let row: Vec<String> = fp.transition[i].iter().map(|v| format!("{v:.4}")).collect();
            println!("  Trans[{i}]: [{}]", row.join(", "));
        }
    }

    // List artifact paths.
    if !result.artifacts.is_empty() {
        println!();
        println!("Artifacts:");
        for a in &result.artifacts {
            println!("  {} : {}", a.name, a.path);
        }
    }

    for w in &result.warnings {
        println!("Warning: {w}");
    }

    // Optionally copy artifacts to a save directory.
    if let Some(ref save_dir) = save_to
        && let Some(first_art) = result.artifacts.first()
    {
        let run_path = PathBuf::from(&first_art.path);
        if let Some(run_dir_path) = run_path.parent() {
            let dest = PathBuf::from(save_dir);
            let _ = fs::create_dir_all(&dest);
            for art in &result.artifacts {
                let src = PathBuf::from(&art.path);
                if src.exists() {
                    let fname = src.file_name().unwrap_or_default();
                    let _ = fs::copy(&src, dest.join(fname));
                }
            }
            println!();
            println!("Artifacts copied to: {save_dir}");
            let _ = run_dir_path; // suppress unused warning
        }
    }

    Ok(())
}

/// D′2 — `compare-sim-vs-real --id <simreal_id>`
///
/// Runs the registered sim-to-real experiment with `SimToRealBackend`, then
/// derives a "train-on-real" variant from the same config (Real mode + the
/// underlying real series as the training source) and runs it with
/// `RealBackend`.  Emits `sim_vs_real_comparison.json` summarising both
/// metric tuples side-by-side.
fn direct_compare_sim_vs_real(args: &[String]) -> anyhow::Result<()> {
    use crate::experiments::config::{DataConfig, ExperimentMode};
    use crate::experiments::result::EvaluationSummary;
    use crate::experiments::runner::ExperimentRunner;

    let id = flag_value(args, "--id").ok_or_else(|| {
        anyhow::anyhow!(
            "Usage: compare-sim-vs-real --id <simreal_id> [--cache <path>] [--save <out_dir>]"
        )
    })?;
    let cache =
        flag_value(args, "--cache").unwrap_or_else(|| "data/commodities.duckdb".to_string());
    let save_to = flag_value(args, "--save").unwrap_or_else(|| format!("./runs/comparison/{id}"));

    let reg = crate::experiments::registry::registry();
    let entry = reg
        .iter()
        .find(|e| e.id == id)
        .ok_or_else(|| anyhow::anyhow!("Unknown experiment id '{id}'"))?;
    let sim_cfg = (entry.build)();

    if !matches!(sim_cfg.mode, ExperimentMode::SimToReal) {
        anyhow::bail!(
            "compare-sim-vs-real requires a SimToReal-mode experiment; '{id}' is {:?}",
            sim_cfg.mode
        );
    }

    // Derive the train-on-real variant: same data series + features + model +
    // detector + evaluation, but train directly on the real series.
    let mut real_cfg = sim_cfg.clone();
    real_cfg.mode = ExperimentMode::Real;
    real_cfg.meta.run_label = format!("{}__trainreal", sim_cfg.meta.run_label);
    real_cfg.data = match &sim_cfg.data {
        DataConfig::CalibratedSynthetic {
            real_asset,
            real_frequency,
            real_dataset_id,
            real_date_start,
            real_date_end,
            ..
        } => DataConfig::Real {
            asset: real_asset.clone(),
            frequency: real_frequency.clone(),
            dataset_id: real_dataset_id.clone(),
            date_start: real_date_start.clone(),
            date_end: real_date_end.clone(),
        },
        other => anyhow::bail!(
            "SimToReal config must use CalibratedSynthetic data; got {:?}",
            std::mem::discriminant(other)
        ),
    };

    println!();
    println!("compare-sim-vs-real  id = {id}");
    println!("  cache = {cache}");
    println!();

    // 1. Run the synthetic-trained pipeline.
    println!("[1/2] Running SimToReal pipeline (synthetic-trained, real-tested)...");
    let sim_backend = crate::experiments::sim_to_real_backend::SimToRealBackend::new(&cache);
    let sim_runner = ExperimentRunner::new(sim_backend);
    let sim_result = sim_runner.run(sim_cfg.clone());
    print_run_result_summary(&sim_result);

    // 2. Run the train-on-real comparator.
    println!();
    println!("[2/2] Running RealBackend (real-trained, real-tested)...");
    let real_backend = crate::experiments::real_backend::RealBackend::new(&cache);
    let real_runner = ExperimentRunner::new(real_backend);
    let real_result = real_runner.run(real_cfg.clone());
    print_run_result_summary(&real_result);

    // Extract metrics into a comparable shape.
    let metrics = |r: &ExperimentResult| -> Option<(f64, f64, f64)> {
        if let Some(EvaluationSummary::Real {
            event_coverage,
            alarm_relevance,
            segmentation_coherence,
        }) = &r.evaluation_summary
        {
            Some((*event_coverage, *alarm_relevance, *segmentation_coherence))
        } else {
            None
        }
    };

    let metrics_json = |m: Option<(f64, f64, f64)>| -> serde_json::Value {
        match m {
            Some((ec, ar, sc)) => serde_json::json!({
                "event_coverage": ec,
                "alarm_relevance": ar,
                "segmentation_coherence": sc,
            }),
            None => serde_json::Value::Null,
        }
    };

    let sim_metrics = metrics(&sim_result);
    let real_metrics = metrics(&real_result);

    // D′2.2 — Compute sim-to-real gap (real_trained − sim_trained).  A
    // positive value means real-trained is better; negative means the
    // synthetic-trained model already matches or exceeds the real-trained
    // baseline on that metric.
    let deltas_json = match (sim_metrics, real_metrics) {
        (Some((sec, sar, ssc)), Some((rec, rar, rsc))) => serde_json::json!({
            "event_coverage_gap":        rec - sec,
            "alarm_relevance_gap":       rar - sar,
            "segmentation_coherence_gap": rsc - ssc,
        }),
        _ => serde_json::Value::Null,
    };

    let comparison = serde_json::json!({
        "id": id,
        "sim_trained": {
            "run_id": sim_result.metadata.run_id,
            "run_label": sim_result.metadata.run_label,
            "metrics": metrics_json(sim_metrics),
        },
        "real_trained": {
            "run_id": real_result.metadata.run_id,
            "run_label": real_result.metadata.run_label,
            "metrics": metrics_json(real_metrics),
        },
        "deltas_real_minus_sim": deltas_json,
    });

    fs::create_dir_all(&save_to)?;
    let out_path = PathBuf::from(&save_to).join("sim_vs_real_comparison.json");
    fs::write(&out_path, serde_json::to_string_pretty(&comparison)?)?;
    println!();
    println!("Wrote comparison: {}", out_path.display());

    // D′2.3 — Markdown table alongside the JSON.
    let fmt = |v: Option<f64>| v.map_or("n/a".to_string(), |x| format!("{x:.4}"));
    let (sec, sar, ssc) =
        sim_metrics.map_or((None, None, None), |(a, b, c)| (Some(a), Some(b), Some(c)));
    let (rec, rar, rsc) =
        real_metrics.map_or((None, None, None), |(a, b, c)| (Some(a), Some(b), Some(c)));
    let delta = |s: Option<f64>, r: Option<f64>| match (s, r) {
        (Some(a), Some(b)) => format!("{:+.4}", b - a),
        _ => "n/a".to_string(),
    };
    let md = format!(
        "# Sim-vs-Real Comparison — `{id}`\n\
        \n\
        | Metric | Sim-trained | Real-trained | Δ (real − sim) |\n\
        |---|---:|---:|---:|\n\
        | event_coverage          | {sec_s} | {rec_s} | {ec_d} |\n\
        | alarm_relevance         | {sar_s} | {rar_s} | {ar_d} |\n\
        | segmentation_coherence  | {ssc_s} | {rsc_s} | {sc_d} |\n\
        \n\
        - Sim-trained run id : `{sim_run}`\n\
        - Real-trained run id: `{real_run}`\n",
        id = id,
        sec_s = fmt(sec),
        rec_s = fmt(rec),
        ec_d = delta(sec, rec),
        sar_s = fmt(sar),
        rar_s = fmt(rar),
        ar_d = delta(sar, rar),
        ssc_s = fmt(ssc),
        rsc_s = fmt(rsc),
        sc_d = delta(ssc, rsc),
        sim_run = sim_result.metadata.run_id,
        real_run = real_result.metadata.run_id,
    );
    let md_path = PathBuf::from(&save_to).join("sim_vs_real_comparison.md");
    fs::write(&md_path, &md)?;
    println!("Wrote markdown:   {}", md_path.display());
    println!();
    for line in md.lines() {
        println!("{line}");
    }

    Ok(())
}

/// Run calibration for a registered experiment and save JSON artifacts.
///
/// Works for **both** synthetic and real experiments:
/// - Synthetic: simulates data from the scenario, computes empirical targets,
///   maps targets back to synthetic generator parameters, and verifies the
///   round-trip.
/// - Real: loads real market data from the DuckDB cache, builds the feature
///   stream, computes empirical statistics on the training partition, then
///   maps those statistics to calibrated synthetic generator parameters.
///
/// Usage: `calibrate --id <experiment_id> [--cache <path>] [--out <dir>]`
fn direct_calibrate(args: &[String]) -> anyhow::Result<()> {
    use crate::calibration::{
        CalibrationDatasetTag, CalibrationMappingConfig, CalibrationPartition,
        EmpiricalCalibrationProfile, SummaryTargetSet, VerificationTolerance,
        run_calibration_workflow, summarize_observation_values,
    };
    use crate::experiments::config::{ExperimentMode, FeatureFamilyConfig};
    use crate::experiments::runner::ExperimentBackend;
    use crate::features::FeatureFamily;

    let id = flag_value(args, "--id").ok_or_else(|| {
        anyhow::anyhow!("Usage: calibrate --id <experiment_id> [--cache <path>] [--out <dir>]")
    })?;
    let out_dir = flag_value(args, "--out").unwrap_or_else(|| format!("./runs/calibration/{id}"));

    let reg = crate::experiments::registry::registry();
    let entry = reg
        .iter()
        .find(|e| e.id == id)
        .ok_or_else(|| anyhow::anyhow!("Unknown experiment id '{id}'"))?;

    let cfg = (entry.build)();

    // For real experiments we need the DuckDB cache path.
    let cache =
        flag_value(args, "--cache").unwrap_or_else(|| "data/commodities.duckdb".to_string());

    println!();
    match cfg.mode {
        ExperimentMode::Synthetic => {
            println!("Calibrating '{id}'  [mode: Synthetic]");
            println!("(Simulating synthetic data and computing empirical targets.)");
        }
        ExperimentMode::Real => {
            println!("Calibrating '{id}'  [mode: Real]");
            println!("(Loading real market data from cache: {cache})");
            println!("(Computing empirical feature statistics -> synthetic generator params.)");
        }
        ExperimentMode::SimToReal => {
            println!("Calibrating '{id}'  [mode: SimToReal]");
            println!(
                "(Loading real market data from cache: {cache} and running quick-EM calibration.)"
            );
        }
    }
    println!();

    let (_data, features) = match cfg.mode {
        ExperimentMode::Synthetic => {
            let backend = crate::experiments::synthetic_backend::SyntheticBackend::new();
            let data = backend.resolve_data(&cfg)?;
            let features = backend.build_features(&cfg, &data)?;
            (data, features)
        }
        ExperimentMode::Real => {
            let backend = crate::experiments::real_backend::RealBackend::new(&cache);
            let data = backend.resolve_data(&cfg)?;
            let features = backend.build_features(&cfg, &data)?;
            (data, features)
        }
        ExperimentMode::SimToReal => {
            let backend = crate::experiments::sim_to_real_backend::SimToRealBackend::new(&cache);
            let data = backend.resolve_data(&cfg)?;
            let features = backend.build_features(&cfg, &data)?;
            (data, features)
        }
    };

    if features.observations.is_empty() {
        anyhow::bail!("No feature observations produced — cannot calibrate.");
    }

    let train_obs = &features.observations[..features.train_n.min(features.observations.len())];
    let empirical_summary = summarize_observation_values(train_obs);

    let tag = CalibrationDatasetTag {
        asset: match &cfg.data {
            DataConfig::Synthetic { scenario_id, .. } => format!("synthetic:{scenario_id}"),
            DataConfig::Real { asset, .. } => asset.clone(),
            DataConfig::CalibratedSynthetic { real_asset, .. } => {
                format!("calibrated_synthetic:{real_asset}")
            }
        },
        frequency: match &cfg.data {
            DataConfig::Synthetic { .. } => "synthetic".to_string(),
            DataConfig::Real { frequency, .. } => format!("{frequency:?}").to_lowercase(),
            DataConfig::CalibratedSynthetic { real_frequency, .. } => {
                format!("{real_frequency:?}").to_lowercase()
            }
        },
        feature_label: format!("{:?}", cfg.features.family),
        partition: CalibrationPartition::TrainOnly,
    };
    let feature_family = match &cfg.features.family {
        FeatureFamilyConfig::LogReturn => FeatureFamily::LogReturn,
        FeatureFamilyConfig::AbsReturn => FeatureFamily::AbsReturn,
        FeatureFamilyConfig::SquaredReturn => FeatureFamily::SquaredReturn,
        FeatureFamilyConfig::RollingVol {
            window,
            session_reset,
        } => FeatureFamily::RollingVol {
            window: *window,
            session_reset: *session_reset,
        },
        FeatureFamilyConfig::StandardizedReturn {
            window,
            epsilon,
            session_reset,
        } => FeatureFamily::StandardizedReturn {
            window: *window,
            epsilon: *epsilon,
            session_reset: *session_reset,
        },
    };

    let profile = EmpiricalCalibrationProfile {
        tag,
        feature_family,
        targets: SummaryTargetSet::Full,
        summary: empirical_summary.clone(),
        observations: Vec::new(),
    };

    let seed = cfg.reproducibility.seed.unwrap_or(42);
    let scenario_id = match &cfg.data {
        DataConfig::Synthetic { scenario_id, .. } => scenario_id.clone(),
        DataConfig::Real { .. } => String::new(),
        DataConfig::CalibratedSynthetic { .. } => String::new(),
    };
    let jump = if scenario_id == "shock_contaminated" {
        use crate::calibration::mapping::JumpContamination;
        Some(JumpContamination {
            jump_prob: 0.05,
            jump_scale_mult: 3.0,
        })
    } else {
        None
    };
    let mapping = CalibrationMappingConfig {
        k: cfg.model.k_regimes,
        horizon: match &cfg.data {
            DataConfig::Synthetic { horizon, .. } => *horizon,
            // For real experiments use the training-partition length as the
            // synthetic horizon — this gives a calibrated generator that
            // produces sequences of the same length as the training data.
            DataConfig::Real { .. } => features.train_n.max(100),
            DataConfig::CalibratedSynthetic { horizon, .. } => *horizon,
        },
        jump,
        ..CalibrationMappingConfig::default()
    };

    // Shock-contaminated scenarios have inherently high ACF1 variance; two
    // independently-drawn shock samples diverge beyond the default 0.20
    // tolerance.  Use a wider bound so the verification reflects whether the
    // calibration is structurally sound, not just whether two random draws
    // agree on a high-variance statistic.
    let tol = if scenario_id == "shock_contaminated" {
        VerificationTolerance {
            abs_acf1_abs_max: 0.40,
            ..VerificationTolerance::default()
        }
    } else {
        VerificationTolerance::default()
    };

    let report = run_calibration_workflow(profile, mapping, tol, seed)?;
    let view = report.view();

    println!("Calibration Report");
    println!("------------------");
    println!("  Asset        : {}", view.asset);
    println!("  Feature      : {}", view.feature_label);
    println!("  Horizon      : {}", view.horizon);
    println!("  Empirical n  : {}", view.empirical_n);
    println!("  Synthetic n  : {}", view.synthetic_n);
    println!(
        "  Verification : {}",
        if view.verification_passed {
            "PASSED"
        } else {
            "FAILED"
        }
    );
    for note in &view.verification_notes {
        println!("  Note: {note}");
    }
    let s = &report.empirical_profile.summary;
    println!(
        "  Empirical: mean={:.6}  var={:.6}  std={:.6}",
        s.mean, s.variance, s.std_dev
    );
    println!(
        "  acf1={:.4}  abs_acf1={:.4}  sign_change_rate={:.4}",
        s.acf1, s.abs_acf1, s.sign_change_rate
    );

    // Save artifacts.
    fs::create_dir_all(&out_dir)?;

    // calibration_summary.json
    let cal_summary = serde_json::json!({
        "experiment_id": id,
        "asset": view.asset,
        "feature_label": view.feature_label,
        "frequency": view.frequency,
        "horizon": view.horizon,
        "empirical_n": view.empirical_n,
        "synthetic_n": view.synthetic_n,
        "verification_passed": view.verification_passed,
        "verification_notes": view.verification_notes,
        "verification_mask": view.verification_mask,
        "field_results": view.field_results,
        "mapping_notes": view.mapping_notes,
        "scale_check": view.scale_check,
        "rng_seed": view.rng_seed,
        "expected_durations": view.expected_durations,
        "empirical": {
            "mean": s.mean, "variance": s.variance, "std_dev": s.std_dev,
            "q01": s.q01, "q05": s.q05, "q50": s.q50, "q95": s.q95, "q99": s.q99,
            "acf1": s.acf1, "abs_acf1": s.abs_acf1, "sign_change_rate": s.sign_change_rate,
            "high_episode_mean_dur": s.high_episode_mean_duration,
            "low_episode_mean_dur": s.low_episode_mean_duration,
        },
        "calibrated_model": {
            "k": report.calibrated.model_params.k,
            "pi": report.calibrated.model_params.pi.clone(),
            "transition": (0..report.calibrated.model_params.k)
                .map(|i| report.calibrated.model_params.transition_row(i).to_vec())
                .collect::<Vec<_>>(),
            "means": report.calibrated.model_params.means.clone(),
            "variances": report.calibrated.model_params.variances.clone(),
        },
    });
    let cal_path = PathBuf::from(&out_dir).join("calibration_summary.json");
    fs::write(&cal_path, serde_json::to_string_pretty(&cal_summary)?)?;
    println!("\nSaved: {}", cal_path.display());

    // synthetic_vs_empirical_summary.json
    let mp = &report.calibrated.model_params;
    let vs_summary = serde_json::json!({
        "empirical_mean": s.mean,
        "empirical_variance": s.variance,
        "empirical_acf1": s.acf1,
        "calibrated_regime_means": mp.means.clone(),
        "calibrated_regime_variances": mp.variances.clone(),
        "calibrated_regime_pi": mp.pi.clone(),
        "calibrated_expected_durations": view.expected_durations,
        "horizon": view.horizon,
        "verification_passed": view.verification_passed,
    });
    let vs_path = PathBuf::from(&out_dir).join("synthetic_vs_empirical_summary.json");
    fs::write(&vs_path, serde_json::to_string_pretty(&vs_summary)?)?;
    println!("Saved: {}", vs_path.display());

    // calibrated_scenario.json (the calibrated scenario parameters)
    let params = &report.calibrated.model_params;
    let scenario_json = serde_json::json!({
        "scenario_type": "calibrated_from_empirical",
        "source_experiment": id,
        "feature_family": format!("{:?}", cfg.features.family),
        "k_regimes": params.k,
        "horizon": view.horizon,
        "pi": params.pi.clone(),
        "regime_means": params.means.clone(),
        "regime_variances": params.variances.clone(),
        "transition_matrix": (0..params.k)
            .map(|i| params.transition_row(i).to_vec())
            .collect::<Vec<_>>(),
    });
    let sc_path = PathBuf::from(&out_dir).join("calibrated_scenario.json");
    fs::write(&sc_path, serde_json::to_string_pretty(&scenario_json)?)?;
    println!("Saved: {}", sc_path.display());

    println!();
    println!("Calibration artifacts written to: {out_dir}");
    Ok(())
}

/// Grid-search for optimal detector parameters on a real-data experiment,
/// then run a full E2E with the best config and write a search report.
///
/// Usage: `optimize --id <experiment_id> [--cache <path>] [--save <dir>] [--top <n>]`
///
/// Artifacts written to `<save>` (default `./runs/optimize/<id>/`):
///   search_report.json   — full ranked grid (all points)
///   search_summary.txt   — human-readable top-N table + best params
///   result.json          — full ExperimentResult from the best-config run
///   + all standard run artifacts (config.snapshot.json, plots, CSV traces, …)
fn direct_optimize(args: &[String]) -> anyhow::Result<()> {
    use crate::experiments::search::{
        ModelGrid, ParamGrid, feature_family_name, optimize, optimize_full,
    };
    use std::io::Write;

    let id = flag_value(args, "--id").ok_or_else(|| {
        anyhow::anyhow!(
            "Usage: optimize --id <experiment_id> [--cache <path>] [--save <dir>] [--top <n>] [--model]"
        )
    })?;
    let cache =
        flag_value(args, "--cache").unwrap_or_else(|| "data/commodities.duckdb".to_string());
    let save_dir = flag_value(args, "--save").unwrap_or_else(|| format!("./runs/optimize/{id}"));
    let top_n: usize = flag_value(args, "--top")
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let model_opt = args.iter().any(|a| a == "--model");

    let reg = crate::experiments::registry::registry();
    let entry = reg.iter().find(|e| e.id == id).ok_or_else(|| {
        let known = reg.iter().map(|e| e.id).collect::<Vec<_>>().join(", ");
        anyhow::anyhow!("Unknown experiment id '{id}'. Known: {known}")
    })?;

    let base_cfg = (entry.build)();

    // Choose the detector grid appropriate for this detector type.
    let grid = ParamGrid::for_real_detector(&base_cfg.detector.detector_type);

    // Choose the model grid when --model is requested.
    let model_grid = if model_opt {
        Some(if base_cfg.features.session_aware {
            ModelGrid::for_intraday()
        } else {
            ModelGrid::default()
        })
    } else {
        None
    };

    println!();
    println!("  Optimize — experiment : {id}");
    println!(
        "  Detector              : {:?}",
        base_cfg.detector.detector_type
    );
    println!(
        "  Mode                  : {}",
        if model_opt {
            "joint model + detector"
        } else {
            "detector only"
        }
    );

    // In debug builds each EM fit is ~200× slower than release.  Give an
    // actionable estimate so the user can decide whether to wait or kill.
    #[cfg(debug_assertions)]
    {
        let n_pts = if let Some(mg) = &model_grid {
            mg.n_points() * grid.n_points()
        } else {
            grid.n_points()
        };
        eprintln!();
        eprintln!("  NOTE: running in debug build — EM is ~200× slower than release.");
        eprintln!(
            "  Estimated time: {n_pts} grid points × ~10–30s/fit ≈ {:.0}–{:.0} min.",
            n_pts as f64 * 10.0 / 60.0,
            n_pts as f64 * 30.0 / 60.0
        );
        eprintln!(
            "  Run  cargo build --release  and use the release binary for production searches."
        );
        eprintln!();
    }
    if let Some(mg) = &model_grid {
        println!("  k_regimes values      : {:?}", mg.k_regimes_values);
        println!(
            "  Feature families      : {}",
            mg.feature_families
                .iter()
                .map(feature_family_name)
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let total_pts = if let Some(mg) = &model_grid {
        mg.n_points() * grid.n_points()
    } else {
        grid.n_points()
    };
    println!(
        "  Grid size             : {} points ({} thresholds × {} persistence × {} cooldown{})",
        total_pts,
        grid.thresholds.len(),
        grid.persistence_values.len(),
        grid.cooldown_values.len(),
        if let Some(mg) = &model_grid {
            format!(" × {} model combos", mg.n_points())
        } else {
            String::new()
        },
    );
    println!("  Thresholds            : {:?}", grid.thresholds);
    println!("  Persistence           : {:?}", grid.persistence_values);
    println!("  Cooldown              : {:?}", grid.cooldown_values);
    println!("  Cache                 : {cache}");
    println!("  Save to               : {save_dir}");
    println!();
    println!("  Phase 1/2 — grid search (all artifact writes disabled for speed)...");
    println!();

    // Dispatch the search backend by experiment mode so synthetic experiments
    // are evaluated against synthetic data and real experiments against real
    // data (rather than RealBackend rejecting synthetic configs).
    let progress_full = |idx: usize, total: usize| {
        if idx.is_multiple_of(4) || idx == total - 1 {
            print!("\r  [{:>4}/{total}] searching...", idx + 1);
            let _ = std::io::stdout().flush();
        }
    };
    let progress_det = |idx: usize, total: usize| {
        if idx.is_multiple_of(4) || idx == total - 1 {
            print!("\r  [{:>3}/{total}] searching...", idx + 1);
            let _ = std::io::stdout().flush();
        }
    };
    let opt = match base_cfg.mode {
        crate::experiments::config::ExperimentMode::Synthetic => {
            let runner = crate::experiments::runner::ExperimentRunner::new(
                crate::experiments::synthetic_backend::SyntheticBackend::new(),
            );
            if let Some(mg) = &model_grid {
                optimize_full(&runner, &base_cfg, mg, &grid, progress_full)
            } else {
                optimize(&runner, &base_cfg, &grid, progress_det)
            }
        }
        crate::experiments::config::ExperimentMode::SimToReal => {
            let runner = crate::experiments::runner::ExperimentRunner::new(
                crate::experiments::sim_to_real_backend::SimToRealBackend::new(&cache),
            );
            if let Some(mg) = &model_grid {
                optimize_full(&runner, &base_cfg, mg, &grid, progress_full)
            } else {
                optimize(&runner, &base_cfg, &grid, progress_det)
            }
        }
        crate::experiments::config::ExperimentMode::Real => {
            let runner = crate::experiments::runner::ExperimentRunner::new(
                crate::experiments::real_backend::RealBackend::new(&cache),
            );
            if let Some(mg) = &model_grid {
                optimize_full(&runner, &base_cfg, mg, &grid, progress_full)
            } else {
                optimize(&runner, &base_cfg, &grid, progress_det)
            }
        }
    };

    println!("\r  [{n}/{n}] done.         ", n = opt.n_evaluated);
    println!();

    // Print top-N results.
    if model_opt {
        println!("  Top-{top_n} results (sorted by score desc):");
        println!(
            "  {:>5}  {:>8}  {:>13}  {:>10}  {:>8}  {:>9}  {:>9}  {:>8}",
            "rank", "k_reg", "feature", "threshold", "persist", "coverage", "precision", "score"
        );
        println!("  {}", "-".repeat(85));
        for (rank, pt) in opt.points.iter().take(top_n).enumerate() {
            println!(
                "  {:>5}  {:>8}  {:>13}  {:>10.4}  {:>8}  {:>9.4}  {:>9.4}  {:>8.4}",
                rank + 1,
                pt.k_regimes,
                feature_family_name(&pt.feature_family),
                pt.threshold,
                pt.persistence_required,
                pt.coverage,
                pt.precision_like,
                pt.score,
            );
        }
    } else {
        println!("  Top-{top_n} results (sorted by score desc):");
        println!(
            "  {:>6}  {:>11}  {:>11}  {:>8}  {:>9}  {:>9}  {:>8}",
            "rank", "threshold", "persistence", "cooldown", "coverage", "precision", "score"
        );
        println!("  {}", "-".repeat(72));
        for (rank, pt) in opt.points.iter().take(top_n).enumerate() {
            println!(
                "  {:>6}  {:>11.4}  {:>11}  {:>8}  {:>9.4}  {:>9.4}  {:>8.4}",
                rank + 1,
                pt.threshold,
                pt.persistence_required,
                pt.cooldown,
                pt.coverage,
                pt.precision_like,
                pt.score,
            );
        }
    }
    println!();

    let best = opt
        .points
        .first()
        .expect("grid must have at least one point");
    println!("  Best params:");
    if model_opt {
        println!("    k_regimes           = {}", best.k_regimes);
        println!(
            "    feature_family      = {}",
            feature_family_name(&best.feature_family)
        );
    }
    println!("    threshold           = {}", best.threshold);
    println!("    persistence_required= {}", best.persistence_required);
    println!("    cooldown            = {}", best.cooldown);
    println!("    score               = {:.6}", best.score);
    println!("    coverage            = {:.6}", best.coverage);
    println!("    precision_like      = {:.6}", best.precision_like);
    println!("    n_alarms            = {}", best.n_alarms);
    println!();
    println!("  Phase 2/2 — full E2E run with best params (artifact writes enabled)...");
    println!();

    // Re-run with best config, with full artifact output.
    let mut best_cfg = opt.best_config.clone();
    best_cfg.output.write_json = true;
    best_cfg.output.write_csv = true;
    best_cfg.output.save_traces = true;
    best_cfg.output.root_dir.clone_from(&save_dir);

    // Dispatch the Phase-2 backend by mode (same as Phase 1) so synthetic
    // configs are not rejected by RealBackend.
    let result = match best_cfg.mode {
        crate::experiments::config::ExperimentMode::Synthetic => {
            let backend2 = crate::experiments::synthetic_backend::SyntheticBackend::new();
            crate::experiments::runner::ExperimentRunner::new(backend2).run(best_cfg.clone())
        }
        crate::experiments::config::ExperimentMode::SimToReal => {
            let backend2 = crate::experiments::sim_to_real_backend::SimToRealBackend::new(&cache);
            crate::experiments::runner::ExperimentRunner::new(backend2).run(best_cfg.clone())
        }
        crate::experiments::config::ExperimentMode::Real => {
            let backend2 = crate::experiments::real_backend::RealBackend::new(&cache);
            crate::experiments::runner::ExperimentRunner::new(backend2).run(best_cfg.clone())
        }
    };

    println!("Pipeline:");
    print_stage_log(&best_cfg, &result);
    println!();
    print_run_result_summary(&result);

    if let Some(crate::experiments::result::EvaluationSummary::Real {
        event_coverage,
        alarm_relevance,
        segmentation_coherence,
    }) = &result.evaluation_summary
    {
        println!();
        println!("  Real-Eval Metrics (best config):");
        println!("    event_coverage        = {event_coverage:.4}");
        println!("    alarm_relevance       = {alarm_relevance:.4}");
        println!("    segmentation_coherence= {segmentation_coherence:.4}");
    }

    // Write search_report.json
    fs::create_dir_all(&save_dir)?;
    let report_points: Vec<_> = opt
        .points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            serde_json::json!({
                "rank": i + 1,
                "k_regimes": p.k_regimes,
                "feature_family": feature_family_name(&p.feature_family),
                "threshold": p.threshold,
                "persistence_required": p.persistence_required,
                "cooldown": p.cooldown,
                "score": p.score,
                "coverage": p.coverage,
                "precision_like": p.precision_like,
                "n_alarms": p.n_alarms,
                "status": format!("{:?}", p.status),
            })
        })
        .collect();

    let model_grid_json = if let Some(mg) = &model_grid {
        serde_json::json!({
            "k_regimes_values": mg.k_regimes_values,
            "feature_families": mg.feature_families.iter().map(feature_family_name).collect::<Vec<_>>(),
            "n_points": mg.n_points(),
        })
    } else {
        serde_json::json!(null)
    };

    let search_report = serde_json::json!({
        "experiment_id": id,
        "detector_type": format!("{:?}", base_cfg.detector.detector_type),
        "mode": if model_opt { "joint" } else { "detector_only" },
        "model_grid": model_grid_json,
        "detector_grid": {
            "thresholds": grid.thresholds,
            "persistence_values": grid.persistence_values,
            "cooldown_values": grid.cooldown_values,
            "n_points": grid.n_points(),
        },
        "n_evaluated": opt.n_evaluated,
        "best": {
            "k_regimes": best.k_regimes,
            "feature_family": feature_family_name(&best.feature_family),
            "threshold": best.threshold,
            "persistence_required": best.persistence_required,
            "cooldown": best.cooldown,
            "score": best.score,
            "coverage": best.coverage,
            "precision_like": best.precision_like,
            "n_alarms": best.n_alarms,
        },
        "all_points": report_points,
    });
    let report_path = PathBuf::from(&save_dir).join("search_report.json");
    fs::write(&report_path, serde_json::to_string_pretty(&search_report)?)?;
    println!("\nSearch report  : {}", report_path.display());

    // Write search_summary.txt
    let mut summary_lines = vec![
        format!("Proteus Optimize — {id}"),
        format!(
            "Date           : {}",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
        ),
        format!("Experiment ID  : {id}"),
        format!("Detector       : {:?}", base_cfg.detector.detector_type),
        format!(
            "Mode           : {}",
            if model_opt {
                "joint model + detector"
            } else {
                "detector only"
            }
        ),
        format!("Grid points    : {}", opt.n_evaluated),
        String::new(),
        format!("Best Params:"),
    ];
    if model_opt {
        summary_lines.push(format!("  k_regimes           = {}", best.k_regimes));
        summary_lines.push(format!(
            "  feature_family      = {}",
            feature_family_name(&best.feature_family)
        ));
    }
    summary_lines.extend([
        format!("  threshold           = {}", best.threshold),
        format!("  persistence_required= {}", best.persistence_required),
        format!("  cooldown            = {}", best.cooldown),
        format!("  score               = {:.6}", best.score),
        format!("  coverage            = {:.6}", best.coverage),
        format!("  precision_like      = {:.6}", best.precision_like),
        format!("  n_alarms            = {}", best.n_alarms),
        String::new(),
        format!("Top-{top_n} Grid Results:"),
    ]);
    if model_opt {
        summary_lines.push(format!(
            "  {:>4}  {:>6}  {:>13}  {:>10}  {:>7}  {:>9}  {:>9}  {:>8}",
            "rank", "k_reg", "feature", "threshold", "persist", "coverage", "precision", "score"
        ));
        summary_lines.push(format!("  {}", "-".repeat(82)));
        for (rank, pt) in opt.points.iter().take(top_n).enumerate() {
            summary_lines.push(format!(
                "  {:>4}  {:>6}  {:>13}  {:>10.4}  {:>7}  {:>9.4}  {:>9.4}  {:>8.4}",
                rank + 1,
                pt.k_regimes,
                feature_family_name(&pt.feature_family),
                pt.threshold,
                pt.persistence_required,
                pt.coverage,
                pt.precision_like,
                pt.score,
            ));
        }
    } else {
        summary_lines.push(format!(
            "  {:>4}  {:>10}  {:>11}  {:>8}  {:>9}  {:>9}  {:>8}",
            "rank", "threshold", "persistence", "cooldown", "coverage", "precision", "score"
        ));
        summary_lines.push(format!("  {}", "-".repeat(65)));
        for (rank, pt) in opt.points.iter().take(top_n).enumerate() {
            summary_lines.push(format!(
                "  {:>4}  {:>10.4}  {:>11}  {:>8}  {:>9.4}  {:>9.4}  {:>8.4}",
                rank + 1,
                pt.threshold,
                pt.persistence_required,
                pt.cooldown,
                pt.coverage,
                pt.precision_like,
                pt.score,
            ));
        }
    }
    summary_lines.push(String::new());
    summary_lines.push(format!("Artifacts saved to: {save_dir}"));

    let summary_path = PathBuf::from(&save_dir).join("search_summary.txt");
    fs::write(&summary_path, summary_lines.join("\n"))?;
    println!("Search summary : {}", summary_path.display());
    println!();
    println!("  Optimization complete.");
    Ok(())
}

/// Compare multiple run directories by aggregating their evaluation summaries.
///
/// Usage: `compare-runs --dir <run-dir> [--dir <run-dir> ...] [--save <path>]`
///
/// Each `--dir` argument is the root of a run directory (the folder that
/// contains `results/evaluation_summary.json`).  The combined Markdown table
/// is printed to stdout and, if `--save <path>` is given, also written to
/// that file.
fn direct_compare_runs(args: &[String]) -> anyhow::Result<()> {
    use crate::reporting::{AggregateReporter, RunArtifactLayout};

    let dirs = collect_flag_values(args, "--dir");
    if dirs.is_empty() {
        anyhow::bail!("Usage: compare-runs --dir <run-dir> [--dir <run-dir> ...] [--save <path>]");
    }
    let save_to = flag_value(args, "--save");

    let layouts: Vec<RunArtifactLayout> = dirs
        .iter()
        .map(|dir| {
            let root = PathBuf::from(dir);
            let run_id = root
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| dir.clone());
            RunArtifactLayout { run_id, root }
        })
        .collect();

    let reporter = AggregateReporter::new(layouts);
    let table = reporter.generate_comparison_table()?;

    println!("\nAggregate Comparison Table");
    println!("==========================");
    println!("{table}");

    if let Some(path) = save_to {
        fs::write(&path, &table)?;
        println!("Saved comparison table to: {path}");
    }

    Ok(())
}

fn print_help() {
    println!();
    println!("USAGE:");
    println!("  cargo run                             # interactive menu");
    println!("  cargo run -- <config.toml>            # interactive, specific config");
    println!("  cargo run -- <subcommand> [options]   # direct command");
    println!();
    println!("SUBCOMMANDS:");
    println!(
        "  e2e                                          Run all registered synthetic experiments"
    );
    println!(
        "  param-search  [--id <experiment_id>]         Grid search over detector params (DryRun)"
    );
    println!(
        "  optimize      --id <id> [--cache <path>]     Grid search on real data then full E2E"
    );
    println!("                [--save <dir>] [--top <n>]      Add --model to also sweep k_regimes");
    println!(
        "                [--model]                        and feature families (joint search)"
    );
    println!(
        "  run-experiment  --config <path.json>         Run single experiment (DryRun/Synthetic)"
    );
    println!("  run-batch       --config a.json [...]        Run batch from files");
    println!(
        "  run-real        --id <id> [--cache <path>]   Run a real-data experiment by registry ID"
    );
    println!(
        "  calibrate       --id <id> [--cache <path>]   Calibrate any experiment (synthetic or real)"
    );
    println!("  compare-runs    --dir <dir> [--dir <dir> ...]  Aggregate metrics across run dirs");
    println!("                  [--save <path>]");
    println!(
        "  compare-sim-vs-real --id <simreal_id>        Run SimToReal + train-on-real comparator"
    );
    println!("                  [--cache <path>] [--save <out_dir>]");
    println!("  inspect         --dir <run-directory>        Inspect run artifacts");
    println!(
        "  generate-report --dir <run-directory>        Regenerate plots/artifacts from config snapshot"
    );
    println!(
        "                  [--cache <path>]              (re-runs EM; use for real-data reports)"
    );
    println!("  status          [--config <path.toml>]       Show data cache status");
    println!("  help                                         Show this message");
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

fn print_config_block(cfg: &ExperimentConfig) {
    let training_str = match &cfg.model.training {
        TrainingMode::FitOffline => "FitOffline".to_string(),
        TrainingMode::LoadFrozen { artifact_id } => format!("LoadFrozen:{artifact_id}"),
    };
    println!("  Config:");
    println!("    Label      : {}", cfg.meta.run_label);
    println!("    Mode       : {:?}", cfg.mode);
    match &cfg.data {
        DataConfig::Synthetic {
            scenario_id,
            horizon,
            ..
        } => {
            println!("    Data       : Synthetic  scenario={scenario_id}  horizon={horizon}");
        }
        DataConfig::Real {
            asset,
            frequency,
            dataset_id,
            date_start,
            date_end,
        } => {
            let range = match (date_start.as_deref(), date_end.as_deref()) {
                (Some(s), Some(e)) => format!("{s} .. {e}"),
                (Some(s), None) => format!("{s} .. (latest)"),
                (None, Some(e)) => format!("(earliest) .. {e}"),
                (None, None) => "all available".to_string(),
            };
            println!("    Data       : Real  asset={asset}  {frequency:?}  dataset={dataset_id}");
            println!("    Date range : {range}");
        }
        DataConfig::CalibratedSynthetic {
            real_asset,
            real_frequency,
            real_dataset_id,
            horizon,
            ..
        } => {
            println!(
                "    Data       : CalibratedSynthetic  real={real_asset}  {real_frequency:?}  dataset={real_dataset_id}  horizon={horizon}"
            );
        }
    }
    println!(
        "    Feature    : {:?}  scaling={:?}  session_aware={}",
        cfg.features.family, cfg.features.scaling, cfg.features.session_aware
    );
    println!(
        "    Model      : K={}  {}  em_max_iter={}  tol={}",
        cfg.model.k_regimes, training_str, cfg.model.em_max_iter, cfg.model.em_tol
    );
    println!(
        "    Detector   : {:?}  threshold={}  persistence={}  cooldown={}",
        cfg.detector.detector_type,
        cfg.detector.threshold,
        cfg.detector.persistence_required,
        cfg.detector.cooldown
    );
    match &cfg.evaluation {
        EvaluationConfig::Synthetic { matching_window } => {
            println!("    Evaluation : Synthetic  matching_window={matching_window}");
        }
        EvaluationConfig::Real {
            route_a_point_pre_bars,
            route_a_point_post_bars,
            route_b_min_segment_len,
            ..
        } => {
            println!(
                "    Evaluation : Real  route_a=[{route_a_point_pre_bars},{route_a_point_post_bars})  route_b_min_seg={route_b_min_segment_len}"
            );
        }
    }
    if let Some(seed) = cfg.reproducibility.seed {
        println!("    Seed       : {}  output={}", seed, cfg.output.root_dir);
    } else {
        println!("    Seed       : (none)  output={}", cfg.output.root_dir);
    }
}

fn print_stage_log(cfg: &ExperimentConfig, result: &ExperimentResult) {
    let n_stages = result.timings.len();
    for (i, timing) in result.timings.iter().enumerate() {
        let failed = if let RunStatus::Failed { stage: ref fs, .. } = result.status {
            *fs == timing.stage
        } else {
            false
        };
        let marker = if failed { "x" } else { "v" };

        let detail = match &timing.stage {
            RunStage::ResolveData => match &cfg.data {
                DataConfig::Synthetic {
                    scenario_id,
                    horizon,
                    ..
                } => format!("dataset=synthetic:{scenario_id}  n={horizon}"),
                DataConfig::Real { dataset_id, .. } => format!("dataset=real:{dataset_id}"),
                DataConfig::CalibratedSynthetic {
                    real_dataset_id,
                    horizon,
                    ..
                } => {
                    format!("dataset=simreal:{real_dataset_id}  horizon={horizon}")
                }
            },
            RunStage::BuildFeatures => {
                let n = match &cfg.data {
                    DataConfig::Synthetic { horizon, .. } => horizon.saturating_sub(1),
                    DataConfig::Real { .. } => result.n_feature_obs.unwrap_or(0),
                    DataConfig::CalibratedSynthetic { .. } => result.n_feature_obs.unwrap_or(0),
                };
                format!(
                    "family={:?}  scaling={:?}  session_aware={}  n_feature_obs={}",
                    cfg.features.family, cfg.features.scaling, cfg.features.session_aware, n
                )
            }
            RunStage::TrainOrLoadModel => result
                .model_summary
                .as_ref()
                .map(|ms| {
                    let base = format!(
                        "source={}  K={}  diagnostics={}",
                        ms.training_mode,
                        ms.k_regimes,
                        if ms.diagnostics_ok { "ok" } else { "FAIL" }
                    );
                    if let Some(fp) = &ms.fitted_params {
                        format!(
                            "{}  LL={:.2}  iter={}  converged={}",
                            base, fp.log_likelihood, fp.n_iter, fp.converged
                        )
                    } else {
                        base
                    }
                })
                .unwrap_or_default(),
            RunStage::RunOnline => {
                let n_steps = match &cfg.data {
                    DataConfig::Synthetic { horizon, .. } => horizon.saturating_sub(1),
                    DataConfig::Real { .. } => result
                        .timings
                        .iter()
                        .find(|t| t.stage == RunStage::RunOnline)
                        .map_or(0, |_| {
                            result
                                .detector_summary
                                .as_ref()
                                .map_or(0, |ds| ds.alarm_indices.last().copied().unwrap_or(0))
                        }),
                    DataConfig::CalibratedSynthetic { .. } => result.n_feature_obs.unwrap_or(0),
                };
                result
                    .detector_summary
                    .as_ref()
                    .map(|ds| {
                        format!(
                            "detector={:?}  thr={:.3}  persistence={}  cooldown={}  n_steps≈{}  n_alarms={}",
                            cfg.detector.detector_type,
                            cfg.detector.threshold,
                            cfg.detector.persistence_required,
                            cfg.detector.cooldown,
                            n_steps,
                            ds.n_alarms
                        )
                    })
                    .unwrap_or_default()
            }
            RunStage::Evaluate => match &result.evaluation_summary {
                Some(EvaluationSummary::Synthetic {
                    coverage,
                    precision_like,
                    n_events,
                    recall,
                    precision,
                    ..
                }) => {
                    let prec = precision
                        .map(|p| format!("{p:.4}"))
                        .unwrap_or_else(|| format!("{precision_like:.4}"));
                    let rec = recall
                        .map(|r| format!("{r:.4}"))
                        .unwrap_or_else(|| format!("{coverage:.4}"));
                    format!("precision={prec}  recall={rec}  n_events={n_events}")
                }
                Some(EvaluationSummary::Real {
                    event_coverage,
                    alarm_relevance,
                    segmentation_coherence,
                }) => format!(
                    "event_cov={event_coverage:.4}  relevance={alarm_relevance:.4}  coherence={segmentation_coherence:.4}"
                ),
                None => String::new(),
            },
            RunStage::Export => format!("artifacts={}", result.artifacts.len()),
        };

        println!(
            "    [{}/{}] [{}] {:<20} {}  ({}ms)",
            i + 1,
            n_stages,
            marker,
            format!("{:?}", timing.stage),
            detail,
            timing.duration_ms
        );
    }
    for w in &result.warnings {
        println!("    [!] Warning: {w}");
    }
}

fn print_run_result_summary(result: &ExperimentResult) {
    let ok = result.is_success();
    println!(
        "  Result   : {}  run_id={}",
        if ok { "SUCCESS" } else { "FAILED" },
        result.metadata.run_id
    );
    if let Some(eval) = &result.evaluation_summary {
        match eval {
            EvaluationSummary::Synthetic {
                coverage,
                precision_like,
                n_events,
                precision,
                recall,
                miss_rate,
                false_alarm_rate,
                delay_mean,
                delay_median,
                ..
            } => {
                let prec = precision.unwrap_or(*precision_like);
                let rec = recall.unwrap_or(*coverage);
                println!(
                    "  Metrics  : prec={:.4}  recall={:.4}  n_events={}  n_alarms={}",
                    prec,
                    rec,
                    n_events,
                    result.detector_summary.as_ref().map_or(0, |d| d.n_alarms)
                );
                if let Some(mr) = miss_rate {
                    println!(
                        "  Miss rate: {mr:.4}  FAR: {:.6}",
                        false_alarm_rate.unwrap_or(0.0)
                    );
                }
                if let Some(dm) = delay_mean {
                    println!(
                        "  Delay    : mean={dm:.1}  median={:.1}",
                        delay_median.unwrap_or(0.0)
                    );
                }
            }
            EvaluationSummary::Real {
                event_coverage,
                alarm_relevance,
                segmentation_coherence,
            } => {
                println!(
                    "  Metrics  : event_cov={event_coverage:.4}  relevance={alarm_relevance:.4}  coherence={segmentation_coherence:.4}"
                );
            }
        }
    }
    if !result.warnings.is_empty() {
        println!("  Warnings : {}", result.warnings.len());
    }
}

fn build_metrics_table(results: &[&ExperimentResult]) -> String {
    let mut builder = MetricsTableBuilder::new();
    for result in results {
        let (coverage, precision, delay_mean_val, delay_median_val, n_alarms) =
            match &result.evaluation_summary {
                Some(EvaluationSummary::Synthetic {
                    coverage,
                    precision_like,
                    precision,
                    recall,
                    delay_mean,
                    delay_median,
                    ..
                }) => (
                    Some(recall.unwrap_or(*coverage)),
                    Some(precision.unwrap_or(*precision_like)),
                    *delay_mean,
                    *delay_median,
                    result.detector_summary.as_ref().map_or(0, |d| d.n_alarms),
                ),
                Some(EvaluationSummary::Real {
                    event_coverage,
                    alarm_relevance,
                    ..
                }) => (
                    Some(*event_coverage),
                    Some(*alarm_relevance),
                    None,
                    None,
                    result.detector_summary.as_ref().map_or(0, |d| d.n_alarms),
                ),
                None => (None, None, None, None, 0),
            };
        let (detector_type, threshold) = result.detector_summary.as_ref().map_or_else(
            || ("unknown".to_string(), 0.0),
            |d| (d.detector_type.clone(), d.threshold),
        );
        builder.add_row(MetricsTableRow {
            run_id: result.metadata.run_id.clone(),
            scenario_or_asset: result.metadata.run_label.clone(),
            detector_type,
            threshold,
            n_alarms,
            coverage,
            precision,
            delay_mean: delay_mean_val,
            delay_median: delay_median_val,
        });
    }
    builder.to_markdown()
}

fn print_search_results(results: &[SearchPoint], top_n: usize) {
    let show = top_n.min(results.len());
    println!(
        "  Top {} / {} grid points (sorted by score descending):",
        show,
        results.len()
    );
    println!();
    println!(
        "  {:<8} {:<12} {:<12} {:<10} {:<10} {:<10}",
        "Score", "Threshold", "Persistence", "Cooldown", "Coverage", "Precision"
    );
    println!("  {}", "-".repeat(64));
    for pt in results.iter().take(show) {
        println!(
            "  {:<8.4} {:<12.3} {:<12} {:<10} {:<10.4} {:<10.4}",
            pt.score,
            pt.threshold,
            pt.persistence_required,
            pt.cooldown,
            pt.coverage,
            pt.precision_like
        );
    }
    if let Some(best) = results.first() {
        println!();
        println!(
            "  Best: threshold={:.3}  persistence={}  cooldown={}  score={:.4}",
            best.threshold, best.persistence_required, best.cooldown, best.score
        );
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn load_experiment_config(path: &str) -> anyhow::Result<ExperimentConfig> {
    let content =
        fs::read_to_string(path).map_err(|e| anyhow::anyhow!("Cannot read '{path}': {e}"))?;
    if std::path::Path::new(path)
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("toml"))
    {
        toml::from_str(&content).map_err(|e| anyhow::anyhow!("TOML parse error: {e}"))
    } else {
        serde_json::from_str(&content).map_err(|e| anyhow::anyhow!("JSON parse error: {e}"))
    }
}

fn make_template_config() -> serde_json::Value {
    serde_json::json!({
        "meta": {"run_label": "my_run", "notes": null},
        "mode": "Synthetic",
        "data": {"Synthetic": {"scenario_id": "scenario_calibrated", "horizon": 1000, "dataset_id": null}},
        "features": {"family": "LogReturn", "scaling": "ZScore", "session_aware": false},
        "model": {"k_regimes": 2, "training": "FitOffline", "em_max_iter": 200, "em_tol": 1e-6},
        "detector": {"detector_type": "Surprise", "threshold": 2.0, "persistence_required": 1, "cooldown": 5},
        "evaluation": {"Synthetic": {"matching_window": 20}},
        "output": {"root_dir": "./runs", "write_json": true, "write_csv": false, "save_traces": false},
        "reproducibility": {"seed": 42, "deterministic_run_id": true, "save_config_snapshot": true, "record_git_info": false}
    })
}

fn show_json_file(path: &Path) {
    match fs::read_to_string(path) {
        Ok(content) => {
            let display = serde_json::from_str::<serde_json::Value>(&content)
                .and_then(|v| serde_json::to_string_pretty(&v))
                .unwrap_or(content);
            println!();
            println!("  {}:", path.display());
            println!("  {}", "-".repeat(60.min(path.to_string_lossy().len() + 1)));
            for line in display.lines().take(60) {
                println!("  {line}");
            }
            let n = display.lines().count();
            if n > 60 {
                println!("  ... ({} more lines)", n - 60);
            }
        }
        Err(e) => eprintln!("\n  [!] Cannot read {}: {e}", path.display()),
    }
}

fn show_artifact_tree(path: &Path, depth: usize) {
    let indent = "  ".repeat(depth + 1);
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    if path.is_dir() {
        if depth == 0 {
            println!();
            println!("  {}/", path.display());
        } else {
            println!("{indent}{name}/");
        }
        if let Ok(mut entries) = fs::read_dir(path) {
            let mut children: Vec<_> = entries.by_ref().flatten().collect();
            children.sort_by_key(std::fs::DirEntry::file_name);
            for entry in children {
                show_artifact_tree(&entry.path(), depth + 1);
            }
        }
    } else {
        let size = fs::metadata(path)
            .map(|m| format!("{} B", m.len()))
            .unwrap_or_default();
        println!("{indent}{name}  ({size})");
    }
}

fn list_runs_recursive(path: &Path, count: &mut usize) {
    if !path.is_dir() {
        return;
    }
    if path.join("result.json").exists() {
        *count += 1;
        let info = fs::read_to_string(path.join("result.json"))
            .ok()
            .and_then(|s| {
                let v: serde_json::Value = serde_json::from_str(&s).ok()?;
                let label = v["metadata"]["run_label"].as_str()?.to_string();
                let mode = v["metadata"]["mode"].as_str()?.to_string();
                let status = v["status"].as_str()?.to_string();
                Some(format!("{label}  [{mode}]  {status}"))
            })
            .unwrap_or_else(|| path.display().to_string());
        println!("  [{count}] {info}");
        return;
    }
    if let Ok(mut entries) = fs::read_dir(path) {
        let mut children: Vec<_> = entries.by_ref().flatten().collect();
        children.sort_by_key(std::fs::DirEntry::file_name);
        for entry in children {
            list_runs_recursive(&entry.path(), count);
        }
    }
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}

fn collect_flag_values(args: &[String], flag: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut iter = args.iter().peekable();
    while let Some(arg) = iter.next() {
        if arg == flag
            && let Some(val) = iter.next()
        {
            values.push(val.clone());
        }
    }
    values
}
