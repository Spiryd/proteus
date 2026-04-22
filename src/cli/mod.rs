use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use inquire::{Confirm, InquireError, Select, Text};

use crate::alphavantage::commodity::{CommodityEndpoint, Interval};
use crate::config::Config;
use crate::data_service::DataService;
use crate::experiments::batch::{BatchConfig, run_batch};
use crate::experiments::config::ExperimentConfig;
use crate::experiments::runner::{DryRunBackend, ExperimentRunner};

// ---------------------------------------------------------------------------
// Top-level interactive menu
// ---------------------------------------------------------------------------

enum MainMenu {
    Data,
    Features,
    Calibration,
    Models,
    Detection,
    Evaluation,
    Experiments,
    Reporting,
    InspectRuns,
    Exit,
}

impl fmt::Display for MainMenu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            MainMenu::Data => "Data          — ingest, inspect, and refresh market data",
            MainMenu::Features => "Features      — feature families and observation pipeline",
            MainMenu::Calibration => "Calibration   — synthetic-to-real scenario calibration",
            MainMenu::Models => "Models        — Gaussian MSM fitting and inspection",
            MainMenu::Detection => "Detection     — detector variants and alarm configuration",
            MainMenu::Evaluation => "Evaluation    — synthetic and real-data evaluation",
            MainMenu::Experiments => "Experiments   — run single or batch experiments",
            MainMenu::Reporting => "Reporting     — plots, tables, and artifact export",
            MainMenu::InspectRuns => "Inspect Runs  — browse and view saved run artifacts",
            MainMenu::Exit => "Exit",
        })
    }
}

// ---------------------------------------------------------------------------
// Data sub-menu actions
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
// Entry points
// ---------------------------------------------------------------------------

/// Interactive mode: `cargo run` or `cargo run -- <config.toml>`
pub async fn run(cfg: Config) -> anyhow::Result<()> {
    println!();
    println!("  Proteus — Markov-Switching Changepoint Detection Platform");
    println!("  ─────────────────────────────────────────────────────────");
    println!("  Press Esc or Ctrl+C at any prompt to go back / exit.\n");

    let service = DataService::new(&cfg)?;

    loop {
        println!();
        let action = match Select::new(
            "Main menu:",
            vec![
                MainMenu::Data,
                MainMenu::Features,
                MainMenu::Calibration,
                MainMenu::Models,
                MainMenu::Detection,
                MainMenu::Evaluation,
                MainMenu::Experiments,
                MainMenu::Reporting,
                MainMenu::InspectRuns,
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
            MainMenu::Features => menu_features(),
            MainMenu::Calibration => menu_calibration(),
            MainMenu::Models => menu_models(),
            MainMenu::Detection => menu_detection(),
            MainMenu::Evaluation => menu_evaluation(),
            MainMenu::Experiments => menu_experiments(),
            MainMenu::Reporting => menu_reporting(),
            MainMenu::InspectRuns => menu_inspect_runs(),
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
    let cmd = args.first().map(|s| s.as_str()).unwrap_or("help");
    match cmd {
        "run-experiment" => direct_run_experiment(&args),
        "run-batch" => direct_run_batch(&args),
        "inspect" => direct_inspect(&args),
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
// Features sub-menu
// ---------------------------------------------------------------------------

fn menu_features() -> anyhow::Result<()> {
    enum FeaturesAction {
        ShowFamilies,
        ShowConfig,
        Back,
    }
    impl fmt::Display for FeaturesAction {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(match self {
                FeaturesAction::ShowFamilies => {
                    "Feature Families  — show available feature transforms"
                }
                FeaturesAction::ShowConfig => "Config Structure  — show FeatureConfig fields",
                FeaturesAction::Back => "<- Back",
            })
        }
    }
    loop {
        println!();
        let action = match Select::new(
            "Features:",
            vec![
                FeaturesAction::ShowFamilies,
                FeaturesAction::ShowConfig,
                FeaturesAction::Back,
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
            FeaturesAction::ShowFamilies => {
                println!();
                println!("  Available Feature Families:");
                println!("  ───────────────────────────");
                println!("  LogReturn              — log(P_t / P_{{t-1}})");
                println!("  AbsReturn              — |log return|");
                println!("  SquaredReturn          — (log return)^2");
                println!("  RollingVol             — rolling std of log returns over window w");
                println!("  StandardizedReturn     — log return / rolling_std, session-aware");
                println!();
                println!("  All transforms are causal (no future leakage).");
                println!("  Scaling options: None | ZScore | RobustZScore.");
            }
            FeaturesAction::ShowConfig => {
                println!();
                println!("  FeatureConfig JSON structure:");
                println!("  ──────────────────────────────");
                println!(r#"  {{"#);
                println!(r#"    "family": "LogReturn",     // or AbsReturn, SquaredReturn"#);
                println!(
                    r#"    //         {{"RollingVol": {{"window": 20, "session_reset": false}}}}"#
                );
                println!(
                    r#"    //         {{"StandardizedReturn": {{"window": 20, "epsilon": 1e-8, "session_reset": false}}}}"#
                );
                println!(r#"    "scaling": "ZScore",       // None | ZScore | RobustZScore"#);
                println!(r#"    "session_aware": false"#);
                println!(r#"  }}"#);
            }
            FeaturesAction::Back => break,
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Calibration sub-menu
// ---------------------------------------------------------------------------

fn menu_calibration() -> anyhow::Result<()> {
    enum CalibrationAction {
        ShowWorkflow,
        ShowScenarios,
        Back,
    }
    impl fmt::Display for CalibrationAction {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(match self {
                CalibrationAction::ShowWorkflow => "Calibration Workflow — steps and inputs",
                CalibrationAction::ShowScenarios => {
                    "Preset Scenarios     — available calibrated presets"
                }
                CalibrationAction::Back => "<- Back",
            })
        }
    }
    loop {
        println!();
        let action = match Select::new(
            "Calibration:",
            vec![
                CalibrationAction::ShowWorkflow,
                CalibrationAction::ShowScenarios,
                CalibrationAction::Back,
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
            CalibrationAction::ShowWorkflow => {
                println!();
                println!("  Calibration Workflow:");
                println!("  ─────────────────────");
                println!("  1. Build feature stream from real price data");
                println!("  2. Summarize empirical statistics (summarize_feature_stream)");
                println!("  3. Map empirical profile to MSM params (calibrate_to_synthetic)");
                println!("  4. Verify discrepancy (verify_calibration)");
                println!("  5. Save CalibrationReport to JSON artifact");
                println!();
                println!("  Key types: CalibrationMappingConfig, MeanPolicy, VariancePolicy,");
                println!("  JumpContamination, CalibratedSyntheticParams, CalibrationReport.");
            }
            CalibrationAction::ShowScenarios => {
                println!();
                println!("  Preset Scenario IDs:");
                println!("  ────────────────────");
                println!("  scenario_calibrated   — standard calibrated from real data");
                println!("  scenario_extreme      — amplified jumps / high variance");
                println!("  scenario_slow         — slow regime switching");
                println!("  scenario_fast         — fast regime switching");
                println!();
                println!("  Reference scenario_id in ExperimentConfig.data.Synthetic.scenario_id.");
            }
            CalibrationAction::Back => break,
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Models sub-menu
// ---------------------------------------------------------------------------

fn menu_models() -> anyhow::Result<()> {
    enum ModelsAction {
        ShowEmConfig,
        ShowArchitecture,
        ValidateConfig,
        Back,
    }
    impl fmt::Display for ModelsAction {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(match self {
                ModelsAction::ShowEmConfig => "EM Config         — ModelConfig fields and defaults",
                ModelsAction::ShowArchitecture => {
                    "MSM Architecture  — K-regime Gaussian model summary"
                }
                ModelsAction::ValidateConfig => {
                    "Validate Config   — check an experiment config file"
                }
                ModelsAction::Back => "<- Back",
            })
        }
    }
    loop {
        println!();
        let action = match Select::new(
            "Models:",
            vec![
                ModelsAction::ShowEmConfig,
                ModelsAction::ShowArchitecture,
                ModelsAction::ValidateConfig,
                ModelsAction::Back,
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
            ModelsAction::ShowEmConfig => {
                println!();
                println!("  ModelConfig JSON structure:");
                println!("  ───────────────────────────");
                println!(
                    r#"  {{"k_regimes": 2, "training": "FitOffline", "em_max_iter": 200, "em_tol": 1e-6}}"#
                );
                println!();
                println!("  training options:");
                println!(r#"    "FitOffline"                          — fit EM from data"#);
                println!(r#"    {{"LoadFrozen": {{"artifact_id": "..."}}}}  — load saved model"#);
            }
            ModelsAction::ShowArchitecture => {
                println!();
                println!("  K-Regime Gaussian MSM:");
                println!("  ──────────────────────");
                println!("  Hidden state S_t in {{1,...,K}} with first-order Markov dynamics.");
                println!("  Emission:   y_t | S_t=j ~ N(mu_j, sigma^2_j)");
                println!("  Transition: P(S_t=j | S_{{t-1}}=i) = A_ij");
                println!("  Fitted via EM (Baum-Welch) with multiple restarts.");
                println!("  Online inference: causal forward filter (exact).");
            }
            ModelsAction::ValidateConfig => {
                let path = match Text::new("Path to experiment config (JSON):")
                    .with_default("experiment_config.json")
                    .prompt()
                {
                    Ok(s) => s,
                    Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => {
                        continue;
                    }
                    Err(e) => return Err(e.into()),
                };
                match load_experiment_config(&path) {
                    Ok(cfg) => match cfg.validate() {
                        Ok(()) => {
                            println!("\n  [ok] Config is valid.");
                            println!("    Label    : {}", cfg.meta.run_label);
                            println!("    Mode     : {:?}", cfg.mode);
                            println!("    K regimes: {}", cfg.model.k_regimes);
                        }
                        Err(e) => eprintln!("\n  [!] Validation failed: {e}"),
                    },
                    Err(e) => eprintln!("\n  [!] Could not load config: {e}"),
                }
            }
            ModelsAction::Back => break,
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Detection sub-menu
// ---------------------------------------------------------------------------

fn menu_detection() -> anyhow::Result<()> {
    enum DetectionAction {
        ShowDetectors,
        ShowDetectorConfig,
        Back,
    }
    impl fmt::Display for DetectionAction {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(match self {
                DetectionAction::ShowDetectors => "Detector Variants    — available detector types",
                DetectionAction::ShowDetectorConfig => {
                    "Detector Config      — DetectorConfig JSON structure"
                }
                DetectionAction::Back => "<- Back",
            })
        }
    }
    loop {
        println!();
        let action = match Select::new(
            "Detection:",
            vec![
                DetectionAction::ShowDetectors,
                DetectionAction::ShowDetectorConfig,
                DetectionAction::Back,
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
            DetectionAction::ShowDetectors => {
                println!();
                println!("  Detector Variants:");
                println!("  ──────────────────");
                println!("  HardSwitch           — triggers on dominant regime change");
                println!("  PosteriorTransition  — monitors posterior transition mass");
                println!("  Surprise             — negative log predictive density");
                println!();
                println!("  All variants share persistence_required + cooldown controls.");
            }
            DetectionAction::ShowDetectorConfig => {
                println!();
                println!("  DetectorConfig JSON structure:");
                println!("  ──────────────────────────────");
                println!(r#"  {{"#);
                println!(
                    r#"    "detector_type": "Surprise",   // HardSwitch | PosteriorTransition | Surprise"#
                );
                println!(r#"    "threshold": 2.0,"#);
                println!(r#"    "persistence_required": 1,"#);
                println!(r#"    "cooldown": 5"#);
                println!(r#"  }}"#);
            }
            DetectionAction::Back => break,
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Evaluation sub-menu
// ---------------------------------------------------------------------------

fn menu_evaluation() -> anyhow::Result<()> {
    enum EvaluationAction {
        ShowSynthetic,
        ShowReal,
        Back,
    }
    impl fmt::Display for EvaluationAction {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(match self {
                EvaluationAction::ShowSynthetic => "Synthetic Evaluation — event-window protocol",
                EvaluationAction::ShowReal => {
                    "Real Evaluation      — Route A (proxy) + Route B (segments)"
                }
                EvaluationAction::Back => "<- Back",
            })
        }
    }
    loop {
        println!();
        let action = match Select::new(
            "Evaluation:",
            vec![
                EvaluationAction::ShowSynthetic,
                EvaluationAction::ShowReal,
                EvaluationAction::Back,
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
            EvaluationAction::ShowSynthetic => {
                println!();
                println!("  Synthetic Evaluation (EventMatcher protocol):");
                println!("  ─────────────────────────────────────────────");
                println!("  Ground truth: known changepoint positions.");
                println!("  Each changepoint gets a detection window [t, t+w].");
                println!("  Metrics: coverage, precision_like, delay_mean.");
                println!("  Config: EvaluationConfig.Synthetic.matching_window.");
            }
            EvaluationAction::ShowReal => {
                println!();
                println!("  Real-data Evaluation:");
                println!("  ─────────────────────");
                println!("  Route A — Proxy Event Alignment:");
                println!("    Alarm timing vs. known market events.");
                println!("    Config: route_a_point_pre_bars, route_a_point_post_bars.");
                println!();
                println!("  Route B — Segmentation Self-Consistency:");
                println!("    Within-segment homogeneity + between-segment contrast.");
                println!("    Config: route_b_min_segment_len.");
            }
            EvaluationAction::Back => break,
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Experiments sub-menu
// ---------------------------------------------------------------------------

fn menu_experiments() -> anyhow::Result<()> {
    enum ExperimentsAction {
        RunSingle,
        ValidateConfig,
        ShowConfigTemplate,
        Back,
    }
    impl fmt::Display for ExperimentsAction {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(match self {
                ExperimentsAction::RunSingle => {
                    "Run Single Experiment   — provide config file -> dry-run"
                }
                ExperimentsAction::ValidateConfig => "Validate Config         — check config file",
                ExperimentsAction::ShowConfigTemplate => {
                    "Show Config Template    — print example experiment config"
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
                ExperimentsAction::RunSingle,
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
            ExperimentsAction::RunSingle => {
                let config_path = match Text::new("Path to experiment config (JSON):")
                    .with_default("experiment_config.json")
                    .prompt()
                {
                    Ok(s) => s,
                    Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => {
                        continue;
                    }
                    Err(e) => return Err(e.into()),
                };
                let cfg = match load_experiment_config(&config_path) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("\n  [!] Failed to load config: {e}");
                        continue;
                    }
                };
                if let Err(e) = cfg.validate() {
                    eprintln!("\n  [!] Config validation failed: {e}");
                    continue;
                }
                let confirmed = match Confirm::new(&format!(
                    "Run '{}' (dry-run — backend validation only)?",
                    cfg.meta.run_label
                ))
                .with_default(true)
                .prompt()
                {
                    Ok(v) => v,
                    Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => {
                        continue;
                    }
                    Err(e) => return Err(e.into()),
                };
                if confirmed {
                    println!("\n  Running '{}'...", cfg.meta.run_label);
                    let runner = ExperimentRunner::new(DryRunBackend);
                    let result = runner.run(cfg);
                    println!("  Status  : {:?}", result.status);
                    println!("  Run ID  : {}", result.metadata.run_id);
                    println!("  Stages  : {}", result.timings.len());
                    println!("  Warnings: {}", result.warnings.len());
                }
            }
            ExperimentsAction::ValidateConfig => {
                let config_path = match Text::new("Path to experiment config (JSON):")
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
                            println!("    Label    : {}", cfg.meta.run_label);
                            println!("    Mode     : {:?}", cfg.mode);
                            println!("    K regimes: {}", cfg.model.k_regimes);
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
                println!("  ───────────────────────────────────");
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
// Reporting sub-menu
// ---------------------------------------------------------------------------

fn menu_reporting() -> anyhow::Result<()> {
    enum ReportingAction {
        InspectRunArtifacts,
        ShowPlotTypes,
        ShowTableTypes,
        Back,
    }
    impl fmt::Display for ReportingAction {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(match self {
                ReportingAction::InspectRunArtifacts => {
                    "Inspect Run Artifacts  — list files in a run directory"
                }
                ReportingAction::ShowPlotTypes => {
                    "Show Plot Types        — available plot renderers"
                }
                ReportingAction::ShowTableTypes => {
                    "Show Table Types       — available table builders"
                }
                ReportingAction::Back => "<- Back",
            })
        }
    }
    loop {
        println!();
        let action = match Select::new(
            "Reporting:",
            vec![
                ReportingAction::InspectRunArtifacts,
                ReportingAction::ShowPlotTypes,
                ReportingAction::ShowTableTypes,
                ReportingAction::Back,
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
            ReportingAction::InspectRunArtifacts => {
                let run_dir = match Text::new("Path to run directory:")
                    .with_default("./runs")
                    .prompt()
                {
                    Ok(s) => s,
                    Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => {
                        continue;
                    }
                    Err(e) => return Err(e.into()),
                };
                show_artifact_tree(Path::new(&run_dir), 0);
            }
            ReportingAction::ShowPlotTypes => {
                println!();
                println!("  Available Plot Renderers (plotters):");
                println!("  ─────────────────────────────────────");
                println!("  render_signal_with_alarms    — observation series + alarm markers");
                println!("  render_detector_scores       — score trace + threshold line");
                println!("  render_regime_posteriors     — filtered posterior probability traces");
                println!("  render_segmentation          — segment boundaries on real data");
                println!(
                    "  render_delay_distribution    — histogram of detection delays (synthetic)"
                );
                println!();
                println!("  All renderers output PNG via BitMapBackend (1200x600).");
                println!("  Use RunArtifactLayout to resolve output paths.");
            }
            ReportingAction::ShowTableTypes => {
                println!();
                println!("  Available Table Builders:");
                println!("  ─────────────────────────");
                println!("  MetricsTableBuilder    — coverage / precision / delay per detector");
                println!("  ComparisonTableBuilder — side-by-side comparison across runs");
                println!("  SegmentSummaryBuilder  — segment statistics (Route B evaluation)");
                println!();
                println!("  Output formats: Markdown | CSV | LaTeX.");
            }
            ReportingAction::Back => break,
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Inspect Runs sub-menu
// ---------------------------------------------------------------------------

fn menu_inspect_runs() -> anyhow::Result<()> {
    enum InspectAction {
        ListRuns,
        ViewRunSummary,
        ViewConfigSnapshot,
        ViewArtifactTree,
        Back,
    }
    impl fmt::Display for InspectAction {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(match self {
                InspectAction::ListRuns => "List Runs           — show all saved runs",
                InspectAction::ViewRunSummary => "View Run Summary    — view run metadata JSON",
                InspectAction::ViewConfigSnapshot => {
                    "View Config         — view experiment config snapshot"
                }
                InspectAction::ViewArtifactTree => "View Artifact Tree  — show all files in a run",
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
                InspectAction::ViewRunSummary,
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
                        "\n  No ./runs directory found. Runs are created by executing experiments."
                    );
                    continue;
                }
                let mut count = 0usize;
                println!();
                println!("  Saved Runs (./runs):");
                println!("  ────────────────────");
                list_runs_recursive(runs_base, &mut count);
                if count == 0 {
                    println!("  (no runs found)");
                } else {
                    println!("\n  Total: {count} run(s)");
                }
            }
            InspectAction::ViewRunSummary => {
                let run_dir = match Text::new("Run directory path:").prompt() {
                    Ok(s) => s,
                    Err(InquireError::OperationCanceled | InquireError::OperationInterrupted) => {
                        continue;
                    }
                    Err(e) => return Err(e.into()),
                };
                let path = PathBuf::from(&run_dir)
                    .join("metadata")
                    .join("run_metadata.json");
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
                let path = PathBuf::from(&run_dir)
                    .join("config")
                    .join("experiment_config.json");
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
    println!("Running experiment '{}' ...", cfg.meta.run_label);
    let runner = ExperimentRunner::new(DryRunBackend);
    let result = runner.run(cfg);
    println!("Status   : {:?}", result.status);
    println!("Run ID   : {}", result.metadata.run_id);
    println!("Warnings : {}", result.warnings.len());
    Ok(())
}

fn direct_run_batch(args: &[String]) -> anyhow::Result<()> {
    let config_paths = collect_flag_values(args, "--config");
    if config_paths.is_empty() {
        anyhow::bail!("Usage: run-batch --config a.json --config b.json ...");
    }
    let mut configs = Vec::new();
    for path in &config_paths {
        let cfg = load_experiment_config(path)?;
        cfg.validate()?;
        configs.push(cfg);
    }
    let stop_on_error = args.contains(&"--stop-on-error".to_string());
    let runner = ExperimentRunner::new(DryRunBackend);
    let batch_cfg = BatchConfig {
        runs: configs,
        stop_on_error,
    };
    println!("Running batch of {} experiments ...", batch_cfg.runs.len());
    let result = run_batch(&runner, batch_cfg);
    println!("Completed: {}", result.n_success);
    println!("Failed   : {}", result.n_failed);
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

fn print_help() {
    println!("Proteus — Markov-Switching Changepoint Detection Platform");
    println!();
    println!("USAGE:");
    println!("  cargo run                             # interactive mode (default)");
    println!("  cargo run -- <config.toml>            # interactive, specific config");
    println!("  cargo run -- <subcommand> [options]   # direct command mode");
    println!();
    println!("SUBCOMMANDS:");
    println!("  run-experiment  --config <path.json>         Run single experiment");
    println!("  run-batch       --config a.json --config b.json  Run batch");
    println!("  inspect         --dir <run-directory>        Inspect run artifacts");
    println!("  status          [--config <path.toml>]       Show data cache status");
    println!("  help                                         Show this message");
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

fn load_experiment_config(path: &str) -> anyhow::Result<ExperimentConfig> {
    let content =
        fs::read_to_string(path).map_err(|e| anyhow::anyhow!("Cannot read '{}': {}", path, e))?;
    if path.ends_with(".toml") {
        toml::from_str(&content).map_err(|e| anyhow::anyhow!("TOML parse error: {}", e))
    } else {
        serde_json::from_str(&content).map_err(|e| anyhow::anyhow!("JSON parse error: {}", e))
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
            children.sort_by_key(|e| e.file_name());
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
    // A "run" directory contains a "metadata" subdirectory.
    if path.join("metadata").is_dir() {
        *count += 1;
        println!("  [{}] {}", count, path.display());
        return;
    }
    if let Ok(mut entries) = fs::read_dir(path) {
        let mut children: Vec<_> = entries.by_ref().flatten().collect();
        children.sort_by_key(|e| e.file_name());
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
        if arg == flag {
            if let Some(val) = iter.next() {
                values.push(val.clone());
            }
        }
    }
    values
}
