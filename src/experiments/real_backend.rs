/// Real-data experiment backend (Phase 22).
///
/// Implements [`ExperimentBackend`] for experiments that run on real market
/// data held in the DuckDB commodity cache.  The EM training and online
/// detection stages are shared with `SyntheticBackend` via `shared.rs`.
///
/// # Evaluation methodology
///
/// For real financial data there are no latent ground-truth changepoint labels.
/// Instead, we assess detector quality via two reference-free routes:
///
/// * **Route A — Proxy-event alignment**: Do alarm times coincide with
///   externally documented macro events (FOMC decisions, oil-supply shocks,
///   financial crises)?  The `event_coverage` metric measures the fraction of
///   events that received at least one aligned alarm within a tolerance window.
///   `alarm_relevance` measures the fraction of alarms that fall within *any*
///   event window.
///
/// * **Route B — Segmentation self-consistency**: The detector partitions the
///   series into segments by its alarms.  Are those segments statistically
///   distinguishable from their neighbours?  The `coherence_score` summarises
///   how large the between-segment mean/variance contrast is relative to
///   within-segment variance.
///
/// Both routes are theoretically grounded: a detector that fires randomly will
/// score poorly on Route A (low event coverage, low relevance) and on Route B
/// (segments indistinguishable from each other).
use chrono::NaiveDate;

use crate::alphavantage::commodity::{CommodityEndpoint, Interval};
use crate::cache::CommodityCache;
use crate::data::{CleanSeries, DataMode, DataSource, DatasetMeta, Observation, PriceField,
                  SessionConvention};
use crate::data::split::{PartitionedSeries, SplitConfig};
use crate::features::family::FeatureFamily;
use crate::features::scaler::ScalingPolicy;
use crate::features::stream::{FeatureConfig as StreamFeatureConfig, FeatureStream};
use crate::real_eval::report::{RealEvalMeta, evaluate_real_data};
use crate::real_eval::route_a::{PointMatchPolicy, ProxyEvent, RouteAConfig};
use crate::real_eval::route_b::{RouteBConfig, ShortSegmentPolicy};

use super::config::{
    DataConfig, EvaluationConfig, ExperimentConfig, FeatureFamilyConfig,
    RealFrequency, ScalingPolicyConfig,
};
use super::runner::{
    DataBundle, ExperimentBackend, FeatureBundle, ModelArtifact, OnlineRunArtifact,
    RealEvalArtifact, SyntheticEvalArtifact,
};
use super::shared::{run_online_shared, train_or_load_model_shared};

// =========================================================================
// RealBackend
// =========================================================================

/// Backend that runs real-market-data experiments from the DuckDB cache.
///
/// Construct from a cache file path:
/// ```rust,no_run
/// use crate::experiments::real_backend::RealBackend;
/// let backend = RealBackend::new("data/commodities.duckdb");
/// ```
#[derive(Debug, Clone)]
pub struct RealBackend {
    /// Path to the DuckDB commodity-cache file (passed to `CommodityCache::open`).
    pub cache_path: String,
}

impl RealBackend {
    pub fn new(cache_path: impl Into<String>) -> Self {
        Self {
            cache_path: cache_path.into(),
        }
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Open the commodity cache.  Returns a descriptive error if the DuckDB
    /// file cannot be opened.
    fn open_cache(&self) -> anyhow::Result<CommodityCache> {
        CommodityCache::open(&self.cache_path)
    }

    /// Map `RealFrequency` → `Interval` for cache look-ups.
    fn to_interval(freq: &RealFrequency) -> Interval {
        match freq {
            RealFrequency::Daily => Interval::Daily,
            RealFrequency::Intraday5m => Interval::Intraday5Min,
            RealFrequency::Intraday15m => Interval::Intraday15Min,
        }
    }

    /// Map `RealFrequency` → `DataMode` for `DatasetMeta`.
    fn to_data_mode(freq: &RealFrequency) -> DataMode {
        match freq {
            RealFrequency::Daily => DataMode::Daily,
            RealFrequency::Intraday5m => DataMode::Intraday { bar_minutes: 5 },
            RealFrequency::Intraday15m => DataMode::Intraday { bar_minutes: 15 },
        }
    }

    /// Choose the correct `PriceField` for an endpoint.
    ///
    /// Equity ETFs (SPY, QQQ) use the dividend-adjusted close price.
    /// Commodity indices (WTI, Brent, etc.) use the raw spot/settlement value.
    fn to_price_field(endpoint: &CommodityEndpoint) -> PriceField {
        match endpoint {
            CommodityEndpoint::Spy | CommodityEndpoint::Qqq => PriceField::AdjustedClose,
            _ => PriceField::Value,
        }
    }

    /// Load, clean, and optionally date-filter a series from the cache.
    ///
    /// Returns the `CleanSeries` sorted in ascending chronological order.
    fn load_clean_series(
        &self,
        asset: &str,
        frequency: &RealFrequency,
        date_start: Option<&str>,
        date_end: Option<&str>,
    ) -> anyhow::Result<CleanSeries> {
        let endpoint: CommodityEndpoint = asset.parse()?;
        let interval = Self::to_interval(frequency);
        let mode = Self::to_data_mode(frequency);
        let price_field = Self::to_price_field(&endpoint);

        let cache = self.open_cache()?;
        let response = cache
            .load(endpoint.cache_key(), interval.as_str())?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No cached data for {} ({}).\n\
                     Hint: run 'cargo run -- status' to see what is cached, then\n\
                     'cargo run -- ingest' (or Data > Ingest from the interactive menu)\n\
                     to fetch the series.",
                    asset,
                    interval.as_str()
                )
            })?;

        let meta = DatasetMeta {
            symbol: endpoint.cache_key().to_string(),
            mode,
            source: DataSource::AlphaVantage,
            price_field,
            session_convention: match frequency {
                RealFrequency::Daily => SessionConvention::FullDay,
                _ => SessionConvention::RthOnly,
            },
            fetched_at: None,
            unit: Some(response.unit.clone()),
        };

        let mut series = CleanSeries::from_response(response, meta);

        // Apply date-range filter.
        if let Some(start_str) = date_start
            && !start_str.is_empty()
        {
            let start = NaiveDate::parse_from_str(start_str, "%Y-%m-%d")
                .map_err(|_| anyhow::anyhow!("Invalid date_start '{start_str}': expected YYYY-MM-DD"))?
                .and_hms_opt(0, 0, 0)
                .unwrap();
            series.observations.retain(|o| o.timestamp >= start);
        }
        if let Some(end_str) = date_end
            && !end_str.is_empty()
        {
            // End of the given day
            let end = NaiveDate::parse_from_str(end_str, "%Y-%m-%d")
                .map_err(|_| anyhow::anyhow!("Invalid date_end '{end_str}': expected YYYY-MM-DD"))?
                .and_hms_opt(23, 59, 59)
                .unwrap();
            series.observations.retain(|o| o.timestamp <= end);
        }

        if series.observations.len() < 10 {
            anyhow::bail!(
                "Too few observations after date filtering for {} ({}): {} bars found, minimum is 10.",
                asset, interval.as_str(), series.observations.len()
            );
        }

        Ok(series)
    }

    /// Convert `config::FeatureFamilyConfig` → `features::FeatureFamily`.
    fn to_feature_family(cfg_family: &FeatureFamilyConfig) -> FeatureFamily {
        match cfg_family {
            FeatureFamilyConfig::LogReturn => FeatureFamily::LogReturn,
            FeatureFamilyConfig::AbsReturn => FeatureFamily::AbsReturn,
            FeatureFamilyConfig::SquaredReturn => FeatureFamily::SquaredReturn,
            FeatureFamilyConfig::RollingVol { window, session_reset } => {
                FeatureFamily::RollingVol {
                    window: *window,
                    session_reset: *session_reset,
                }
            }
            FeatureFamilyConfig::StandardizedReturn {
                window,
                epsilon,
                session_reset,
            } => FeatureFamily::StandardizedReturn {
                window: *window,
                epsilon: *epsilon,
                session_reset: *session_reset,
            },
        }
    }

    /// Convert `config::ScalingPolicyConfig` → `features::ScalingPolicy`.
    fn to_scaling_policy(cfg_scale: &ScalingPolicyConfig) -> ScalingPolicy {
        match cfg_scale {
            ScalingPolicyConfig::None => ScalingPolicy::None,
            ScalingPolicyConfig::ZScore => ScalingPolicy::ZScore,
            ScalingPolicyConfig::RobustZScore => ScalingPolicy::RobustZScore,
        }
    }

    /// Load proxy events from the JSON file specified in the evaluation config.
    ///
    /// Returns an empty list if `proxy_events_path` is empty.
    fn load_proxy_events(proxy_events_path: &str) -> anyhow::Result<Vec<ProxyEvent>> {
        if proxy_events_path.is_empty() {
            return Ok(vec![]);
        }
        let json = std::fs::read_to_string(proxy_events_path).map_err(|e| {
            anyhow::anyhow!(
                "Cannot read proxy events file '{proxy_events_path}': {e}"
            )
        })?;
        serde_json::from_str(&json).map_err(|e| {
            anyhow::anyhow!(
                "Failed to parse proxy events JSON from '{proxy_events_path}': {e}"
            )
        })
    }
}

// =========================================================================
// ExperimentBackend impl
// =========================================================================

impl ExperimentBackend for RealBackend {
    // ------------------------------------------------------------------
    // Stage 1 — Resolve data
    // ------------------------------------------------------------------

    /// Load price data from DuckDB, validate, sort, and date-filter.
    ///
    /// The cache **must** be pre-populated (run ingest first).  Raw prices are
    /// stored in `DataBundle::observations`; corresponding bar timestamps are
    /// stored in `DataBundle::timestamps`.  The feature pipeline in stage 2
    /// will convert raw prices to model-ready observations (log-returns, etc.).
    fn resolve_data(&self, cfg: &ExperimentConfig) -> anyhow::Result<DataBundle> {
        let (asset, frequency, dataset_id, date_start, date_end) = match &cfg.data {
            DataConfig::Real { asset, frequency, dataset_id, date_start, date_end } => {
                (asset.as_str(), frequency, dataset_id.as_str(), date_start.as_deref(), date_end.as_deref())
            }
            DataConfig::Synthetic { dataset_id, .. } => {
                anyhow::bail!(
                    "RealBackend::resolve_data called with Synthetic DataConfig (dataset_id={dataset_id:?})"
                );
            }
        };

        let series = self.load_clean_series(asset, frequency, date_start, date_end)?;
        let n = series.observations.len();

        let prices: Vec<f64> = series.observations.iter().map(|o| o.value).collect();
        let timestamps: Vec<chrono::NaiveDateTime> =
            series.observations.iter().map(|o| o.timestamp).collect();

        // Compute timestamp-based 70/15/15 split boundaries from the sorted series.
        let train_idx = (n as f64 * 0.70) as usize;
        let val_idx = (n as f64 * 0.85) as usize;
        let epoch_max = chrono::NaiveDate::from_ymd_opt(9999, 12, 31)
            .unwrap()
            .and_hms_opt(23, 59, 59)
            .unwrap();
        let train_end_ts = timestamps.get(train_idx).copied().unwrap_or(epoch_max);
        let val_end_ts = timestamps.get(val_idx).copied().unwrap_or(epoch_max);

        // Serialize ValidationReport before moving series into PartitionedSeries.
        let validation_report_json = serde_json::to_string_pretty(&serde_json::json!({
            "n_input": series.report.n_input,
            "n_dropped_duplicates": series.report.n_dropped_duplicates,
            "n_output": series.report.n_output,
            "had_unsorted_input": series.report.had_unsorted_input,
            "n_gaps": series.report.gaps.len(),
            "gaps": series.report.gaps.iter().map(|g| serde_json::json!({
                "from": g.from.to_string(),
                "to": g.to.to_string(),
            })).collect::<Vec<_>>(),
        })).ok();

        // Build the chronological split (consumes series).
        let split_cfg = SplitConfig { train_end: train_end_ts, val_end: val_end_ts };
        let partitioned = PartitionedSeries::from_series(series, split_cfg);
        let train_n = partitioned.train.len();

        // Serialize the split summary for auditing.
        let split_summary_json = serde_json::to_string_pretty(&serde_json::json!({
            "asset": asset,
            "n_total": n,
            "n_train": train_n,
            "n_validation": partitioned.validation.len(),
            "n_test": partitioned.test.len(),
            "train_end": train_end_ts.to_string(),
            "val_end": val_end_ts.to_string(),
            "train_start": partitioned.train.first().map(|o| o.timestamp.to_string()),
            "val_start": partitioned.validation.first().map(|o| o.timestamp.to_string()),
            "test_start": partitioned.test.first().map(|o| o.timestamp.to_string()),
        })).ok();

        let interval = Self::to_interval(frequency);
        Ok(DataBundle {
            dataset_key: format!("real:{}:{}:{}", asset, interval.as_str(), dataset_id),
            n_observations: n,
            observations: prices,
            changepoint_truth: None,
            train_n,
            timestamps,
            split_summary_json,
            validation_report_json,
        })
    }

    // ------------------------------------------------------------------
    // Stage 2 — Build features
    // ------------------------------------------------------------------

    /// Apply the causal feature transform to raw prices.
    ///
    /// Uses the full `FeatureStream` pipeline from `src/features/`, which
    /// guarantees causality (no look-ahead) and fits the scaler on the
    /// training partition only.
    fn build_features(
        &self,
        cfg: &ExperimentConfig,
        data: &DataBundle,
    ) -> anyhow::Result<FeatureBundle> {
        if data.observations.is_empty() || data.timestamps.is_empty() {
            return Ok(FeatureBundle {
                feature_label: format!("{:?}", cfg.features.family),
                n_observations: 0,
                observations: vec![],
                train_n: 0,
                timestamps: vec![],
            });
        }

        // Reconstruct Observation slice from prices + timestamps.
        let price_obs: Vec<Observation> = data
            .timestamps
            .iter()
            .zip(data.observations.iter())
            .map(|(&ts, &v)| Observation { timestamp: ts, value: v })
            .collect();

        // Build DatasetMeta from config.
        let (asset, frequency, _) = match &cfg.data {
            DataConfig::Real { asset, frequency, dataset_id, .. } => (asset, frequency, dataset_id),
            _ => anyhow::bail!("RealBackend::build_features: expected Real DataConfig"),
        };
        let endpoint: CommodityEndpoint = asset.parse()?;
        let mode = Self::to_data_mode(frequency);
        let price_field = Self::to_price_field(&endpoint);

        let data_meta = DatasetMeta {
            symbol: endpoint.cache_key().to_string(),
            mode,
            source: DataSource::AlphaVantage,
            price_field,
            session_convention: match frequency {
                RealFrequency::Daily => SessionConvention::FullDay,
                _ => SessionConvention::RthOnly,
            },
            fetched_at: None,
            unit: None,
        };

        let feature_family = Self::to_feature_family(&cfg.features.family);
        let scaling = Self::to_scaling_policy(&cfg.features.scaling);

        let stream_cfg = StreamFeatureConfig {
            family: feature_family,
            scaling,
            n_train: data.train_n,
            session_aware: cfg.features.session_aware,
        };

        let stream = FeatureStream::build(&price_obs, data_meta, stream_cfg);

        // Compute training partition size: after warmup trimming, the number
        // of feature observations in train = data.train_n minus warmup bars.
        let actual_train_n = stream.meta.n_obs.min(
            data.train_n.saturating_sub(stream.meta.n_warmup_dropped)
        );

        let feature_timestamps: Vec<chrono::NaiveDateTime> = stream.timestamps();
        let values = stream.values();
        let n = values.len();

        Ok(FeatureBundle {
            feature_label: stream.experiment_label(),
            n_observations: n,
            observations: values,
            train_n: actual_train_n,
            timestamps: feature_timestamps,
        })
    }

    // ------------------------------------------------------------------
    // Stage 3 — Train or load model
    // ------------------------------------------------------------------
    fn train_or_load_model(
        &self,
        cfg: &ExperimentConfig,
        features: &FeatureBundle,
    ) -> anyhow::Result<ModelArtifact> {
        train_or_load_model_shared(cfg, features)
    }

    // ------------------------------------------------------------------
    // Stage 4 — Online detection
    // ------------------------------------------------------------------
    fn run_online(
        &self,
        cfg: &ExperimentConfig,
        model: &ModelArtifact,
        features: &FeatureBundle,
    ) -> anyhow::Result<OnlineRunArtifact> {
        run_online_shared(cfg, model, features)
    }

    // ------------------------------------------------------------------
    // Stage 5a — Evaluate synthetic (not valid for real data)
    // ------------------------------------------------------------------
    fn evaluate_synthetic(
        &self,
        _cfg: &ExperimentConfig,
        _online: &OnlineRunArtifact,
    ) -> anyhow::Result<SyntheticEvalArtifact> {
        anyhow::bail!(
            "RealBackend does not support synthetic evaluation. \
             Use SyntheticBackend for synthetic experiments."
        )
    }

    // ------------------------------------------------------------------
    // Stage 5b — Evaluate real (Route A + Route B)
    // ------------------------------------------------------------------

    /// Evaluate detector output on real data.
    ///
    /// This stage re-loads and transforms the data (same as stages 1 and 2)
    /// so that timestamps and feature values are available for the evaluation
    /// functions.  This is intentionally consistent with how `SyntheticBackend`
    /// re-simulates to obtain changepoints in its evaluation stage.
    ///
    /// **Route A** requires `Vec<ProxyEvent>` loaded from the JSON file
    /// specified in `EvaluationConfig::Real::proxy_events_path`.  An empty
    /// path is allowed and results in zero-coverage Route A metrics.
    ///
    /// **Route B** operates on the feature observations directly, ensuring
    /// that the segmentation coherence is measured in the same space as the
    /// model emission distributions.
    fn evaluate_real(
        &self,
        cfg: &ExperimentConfig,
        online: &OnlineRunArtifact,
    ) -> anyhow::Result<RealEvalArtifact> {
        let (asset, frequency, date_start, date_end, proxy_events_path,
             route_a_pre, route_a_post, route_a_causal, route_b_min_seg) =
            match &cfg.data {
                DataConfig::Real { asset, frequency, date_start, date_end, .. } => {
                    match &cfg.evaluation {
                        EvaluationConfig::Real {
                            proxy_events_path,
                            route_a_point_pre_bars,
                            route_a_point_post_bars,
                            route_a_causal_only,
                            route_b_min_segment_len,
                        } => (
                            asset.as_str(),
                            frequency,
                            date_start.as_deref(),
                            date_end.as_deref(),
                            proxy_events_path.as_str(),
                            *route_a_point_pre_bars,
                            *route_a_point_post_bars,
                            *route_a_causal_only,
                            *route_b_min_segment_len,
                        ),
                        _ => anyhow::bail!(
                            "RealBackend::evaluate_real: EvaluationConfig must be Real"
                        ),
                    }
                }
                _ => anyhow::bail!(
                    "RealBackend::evaluate_real: DataConfig must be Real"
                ),
            };

        // Re-load and clean the series (same as resolve_data).
        let series = self.load_clean_series(asset, frequency, date_start, date_end)?;

        // Apply the feature transform to get model-space observations.
        let endpoint: CommodityEndpoint = asset.parse()?;
        let data_meta = DatasetMeta {
            symbol: endpoint.cache_key().to_string(),
            mode: Self::to_data_mode(frequency),
            source: DataSource::AlphaVantage,
            price_field: Self::to_price_field(&endpoint),
            session_convention: match frequency {
                RealFrequency::Daily => SessionConvention::FullDay,
                _ => SessionConvention::RthOnly,
            },
            fetched_at: None,
            unit: None,
        };

        let n_raw = series.observations.len();
        let train_n = (n_raw as f64 * 0.70) as usize;

        let feature_family = Self::to_feature_family(&cfg.features.family);
        let scaling = Self::to_scaling_policy(&cfg.features.scaling);

        let stream_cfg = StreamFeatureConfig {
            family: feature_family,
            scaling,
            n_train: train_n,
            session_aware: cfg.features.session_aware,
        };
        let stream = FeatureStream::build(&series.observations, data_meta.clone(), stream_cfg);

        // Route A config.
        let route_a_cfg = RouteAConfig {
            point_policy: PointMatchPolicy {
                pre_bars: route_a_pre,
                post_bars: route_a_post,
                causal_only: route_a_causal,
            },
        };

        // Route B config.
        let route_b_cfg = RouteBConfig {
            min_len: route_b_min_seg,
            short_segment_policy: ShortSegmentPolicy::FlagOnly,
        };

        // Load proxy events (empty = skip Route A matching).
        let proxy_events = Self::load_proxy_events(proxy_events_path)?;

        // Build eval metadata.
        let eval_meta = RealEvalMeta {
            asset: endpoint.cache_key().to_string(),
            frequency: Self::to_interval(frequency).as_str().to_string(),
            feature_label: stream.experiment_label(),
            detector_label: format!("{:?}", cfg.detector.detector_type),
        };

        // Run Route A + B via the report orchestrator.
        let result = evaluate_real_data(
            &stream.observations,
            &online.alarm_indices,
            &proxy_events,
            &route_a_cfg,
            &route_b_cfg,
            eval_meta,
        )?;

        Ok(RealEvalArtifact {
            event_coverage: result.route_a.event_coverage,
            alarm_relevance: result.route_a.alarm_relevance,
            segmentation_coherence: result.route_b.global.coherence_score,
            route_a_result_json: serde_json::to_string_pretty(&result.route_a).ok(),
            route_b_result_json: serde_json::to_string_pretty(&result.route_b).ok(),
        })
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiments::config::{
        DataConfig, DetectorConfig, DetectorType, EvaluationConfig, ExperimentConfig,
        ExperimentMode, FeatureConfig, FeatureFamilyConfig, ModelConfig, OutputConfig,
        RealFrequency, ReproducibilityConfig, RunMetaConfig, ScalingPolicyConfig, TrainingMode,
    };
    use crate::experiments::runner::ExperimentRunner;

    fn make_real_cfg(proxy_path: &str) -> ExperimentConfig {
        ExperimentConfig {
            meta: RunMetaConfig {
                run_label: "test_real".to_string(),
                notes: None,
            },
            mode: ExperimentMode::Real,
            data: DataConfig::Real {
                asset: "SPY".to_string(),
                frequency: RealFrequency::Daily,
                dataset_id: "spy_daily_test".to_string(),
                date_start: None,
                date_end: None,
            },
            features: FeatureConfig {
                family: FeatureFamilyConfig::LogReturn,
                scaling: ScalingPolicyConfig::ZScore,
                session_aware: false,
            },
            model: ModelConfig {
                k_regimes: 2,
                training: TrainingMode::FitOffline,
                em_max_iter: 10,
                em_tol: 1e-4,
                em_n_starts: 1,
            },
            detector: DetectorConfig {
                detector_type: DetectorType::HardSwitch,
                threshold: 0.6,
                persistence_required: 1,
                cooldown: 0,
                ema_alpha: None,
            },
            evaluation: EvaluationConfig::Real {
                proxy_events_path: proxy_path.to_string(),
                route_a_point_pre_bars: 3,
                route_a_point_post_bars: 5,
                route_a_causal_only: false,
                route_b_min_segment_len: 5,
            },
            output: OutputConfig {
                root_dir: std::env::temp_dir().to_string_lossy().to_string(),
                write_json: false,
                write_csv: false,
                save_traces: false,
            },
            reproducibility: ReproducibilityConfig {
                seed: Some(42),
                deterministic_run_id: true,
                save_config_snapshot: false,
                record_git_info: false,
            },
        }
    }

    /// `resolve_data` must fail gracefully when cache is empty / file missing.
    #[test]
    fn resolve_data_missing_cache_returns_error() {
        let backend = RealBackend::new("/nonexistent/path/commodities.duckdb");
        let cfg = make_real_cfg("");
        let result = backend.resolve_data(&cfg);
        assert!(result.is_err(), "expected error for missing cache");
    }

    /// `resolve_data` must reject when called with a Synthetic config.
    #[test]
    fn resolve_data_rejects_synthetic_config() {
        let backend = RealBackend::new("data/commodities.duckdb");
        let cfg = ExperimentConfig {
            meta: RunMetaConfig {
                run_label: "t".to_string(),
                notes: None,
            },
            mode: ExperimentMode::Synthetic,
            data: DataConfig::Synthetic {
                scenario_id: "x".to_string(),
                horizon: 100,
                dataset_id: None,
            },
            features: FeatureConfig {
                family: FeatureFamilyConfig::LogReturn,
                scaling: ScalingPolicyConfig::None,
                session_aware: false,
            },
            model: ModelConfig {
                k_regimes: 2,
                training: TrainingMode::FitOffline,
                em_max_iter: 10,
                em_tol: 1e-5,
                em_n_starts: 1,
            },
            detector: DetectorConfig {
                detector_type: DetectorType::Surprise,
                threshold: 2.0,
                persistence_required: 1,
                cooldown: 0,
                ema_alpha: None,
            },
            evaluation: EvaluationConfig::Synthetic { matching_window: 5 },
            output: OutputConfig {
                root_dir: std::env::temp_dir().to_string_lossy().to_string(),
                write_json: false,
                write_csv: false,
                save_traces: false,
            },
            reproducibility: ReproducibilityConfig {
                seed: Some(1),
                deterministic_run_id: true,
                save_config_snapshot: false,
                record_git_info: false,
            },
        };
        let result = backend.resolve_data(&cfg);
        assert!(result.is_err(), "expected error for Synthetic DataConfig");
        assert!(
            result.unwrap_err().to_string().contains("Synthetic"),
            "error message should mention Synthetic"
        );
    }

    /// `evaluate_synthetic` must always return an error for `RealBackend`.
    #[test]
    fn evaluate_synthetic_is_not_supported() {
        let backend = RealBackend::new("data/commodities.duckdb");
        let cfg = make_real_cfg("");
        let dummy_online = OnlineRunArtifact {
            n_steps: 0,
            n_alarms: 0,
            alarm_indices: vec![],
            score_trace: vec![],
            regime_posteriors: vec![],
        };
        assert!(backend.evaluate_synthetic(&cfg, &dummy_online).is_err());
    }

    /// Loading proxy events from an empty path returns an empty list.
    #[test]
    fn load_proxy_events_empty_path() {
        let result = RealBackend::load_proxy_events("");
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    /// Loading proxy events from a non-existent path returns a descriptive error.
    #[test]
    fn load_proxy_events_missing_file() {
        let result = RealBackend::load_proxy_events("/no/such/file.json");
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("proxy events"),
            "error should mention proxy events"
        );
    }

    /// `build_features` returns empty FeatureBundle when observations are empty.
    #[test]
    fn build_features_empty_data_returns_empty() {
        let backend = RealBackend::new("data/commodities.duckdb");
        let cfg = make_real_cfg("");
        let data = DataBundle {
            dataset_key: "real:SPY:daily:test".to_string(),
            n_observations: 0,
            observations: vec![],
            changepoint_truth: None,
            train_n: 0,
            timestamps: vec![],
            split_summary_json: None,
            validation_report_json: None,
        };
        let result = backend.build_features(&cfg, &data);
        assert!(result.is_ok());
        let fb = result.unwrap();
        assert_eq!(fb.n_observations, 0);
        assert!(fb.observations.is_empty());
    }

    /// Feature-family and scaling conversions are consistent round-trips.
    #[test]
    fn feature_family_conversion_covers_all_variants() {
        let families = [
            FeatureFamilyConfig::LogReturn,
            FeatureFamilyConfig::AbsReturn,
            FeatureFamilyConfig::SquaredReturn,
            FeatureFamilyConfig::RollingVol { window: 5, session_reset: false },
            FeatureFamilyConfig::StandardizedReturn { window: 10, epsilon: 1e-8, session_reset: true },
        ];
        for f in &families {
            // Should not panic.
            let _ = RealBackend::to_feature_family(f);
        }
    }

    /// Scaling policy conversion covers all variants.
    #[test]
    fn scaling_policy_conversion_covers_all_variants() {
        assert!(matches!(RealBackend::to_scaling_policy(&ScalingPolicyConfig::None), ScalingPolicy::None));
        assert!(matches!(RealBackend::to_scaling_policy(&ScalingPolicyConfig::ZScore), ScalingPolicy::ZScore));
        assert!(matches!(RealBackend::to_scaling_policy(&ScalingPolicyConfig::RobustZScore), ScalingPolicy::RobustZScore));
    }

    /// Date filtering with both bounds removes out-of-range bars.
    ///
    /// This uses a tiny in-memory DuckDB to avoid requiring a live API key.
    #[test]
    fn resolve_data_date_filter_trims_correctly() {
        use crate::alphavantage::commodity::{CommodityDataPoint, CommodityResponse};
        use crate::cache::CommodityCache;
        use chrono::NaiveDateTime;

        // Write a small in-memory DB to a temp file.
        let tmp = tempfile_path("test_real_backend_date_filter.duckdb");
        let cache = CommodityCache::open(&tmp).expect("open temp db");

        // 20 daily points: 2023-01-02 to 2023-01-27 (weekdays only).
        let dates = [
            "2023-01-02 00:00:00", "2023-01-03 00:00:00", "2023-01-04 00:00:00",
            "2023-01-05 00:00:00", "2023-01-06 00:00:00", "2023-01-09 00:00:00",
            "2023-01-10 00:00:00", "2023-01-11 00:00:00", "2023-01-12 00:00:00",
            "2023-01-13 00:00:00", "2023-01-17 00:00:00", "2023-01-18 00:00:00",
            "2023-01-19 00:00:00", "2023-01-20 00:00:00", "2023-01-23 00:00:00",
            "2023-01-24 00:00:00", "2023-01-25 00:00:00", "2023-01-26 00:00:00",
            "2023-01-27 00:00:00", "2023-01-30 00:00:00",
        ];
        let data: Vec<CommodityDataPoint> = dates
            .iter()
            .enumerate()
            .map(|(i, &d)| CommodityDataPoint {
                date: NaiveDateTime::parse_from_str(d, "%Y-%m-%d %H:%M:%S").unwrap(),
                value: 380.0 + i as f64,
            })
            .collect();

        let response = CommodityResponse {
            name: "S&P 500 ETF".to_string(),
            interval: "daily".to_string(),
            unit: "USD".to_string(),
            data,
        };
        cache.store("SPY", "daily", &response).expect("store");
        drop(cache);

        let backend = RealBackend::new(&tmp);

        // With date range 2023-01-05 to 2023-01-27: should keep 16 bars
        // (excludes Jan 2, 3, 4 from the start and Jan 30 from the end).
        let series = backend
            .load_clean_series(
                "spy",
                &RealFrequency::Daily,
                Some("2023-01-05"),
                Some("2023-01-27"),
            )
            .expect("load");

        // Expected dates in range: Jan 5,6,9,10,11,12,13,17,18,19,20,23,24,25,26,27 = 16 bars
        assert_eq!(
            series.observations.len(),
            16,
            "expected 16 bars in Jan 05-27 range"
        );

        // Clean up temp file.
        let _ = std::fs::remove_file(&tmp);
    }

    /// `train_n` is 70 % of the total bar count.
    #[test]
    fn resolve_data_train_n_is_seventy_percent() {
        use crate::alphavantage::commodity::{CommodityDataPoint, CommodityResponse};
        use crate::cache::CommodityCache;

        let tmp = tempfile_path("test_real_backend_train_n.duckdb");
        let cache = CommodityCache::open(&tmp).expect("open");

        // Generate 100 synthetic daily points.
        let data: Vec<CommodityDataPoint> = (0..100_u64)
            .map(|i| CommodityDataPoint {
                date: chrono::DateTime::from_timestamp(
                    // 2020-01-01 + i days
                    1577836800 + i as i64 * 86400,
                    0,
                )
                .unwrap()
                .naive_utc(),
                value: 100.0 + i as f64,
            })
            .collect();
        let response = CommodityResponse {
            name: "WTI".to_string(),
            interval: "daily".to_string(),
            unit: "USD/bbl".to_string(),
            data,
        };
        cache.store("WTI", "daily", &response).expect("store");
        drop(cache);

        let backend = RealBackend::new(&tmp);
        let cfg = ExperimentConfig {
            meta: RunMetaConfig { run_label: "t".to_string(), notes: None },
            mode: ExperimentMode::Real,
            data: DataConfig::Real {
                asset: "wti".to_string(),
                frequency: RealFrequency::Daily,
                dataset_id: "wti_test".to_string(),
                date_start: None,
                date_end: None,
            },
            features: FeatureConfig {
                family: FeatureFamilyConfig::LogReturn,
                scaling: ScalingPolicyConfig::None,
                session_aware: false,
            },
            model: ModelConfig {
                k_regimes: 2,
                training: TrainingMode::FitOffline,
                em_max_iter: 10,
                em_tol: 1e-4,
                em_n_starts: 1,
            },
            detector: DetectorConfig {
                detector_type: DetectorType::Surprise,
                threshold: 2.0,
                persistence_required: 1,
                cooldown: 0,
                ema_alpha: None,
            },
            evaluation: EvaluationConfig::Real {
                proxy_events_path: String::new(),
                route_a_point_pre_bars: 3,
                route_a_point_post_bars: 3,
                route_a_causal_only: false,
                route_b_min_segment_len: 5,
            },
            output: OutputConfig {
                root_dir: std::env::temp_dir().to_string_lossy().to_string(),
                write_json: false,
                write_csv: false,
                save_traces: false,
            },
            reproducibility: ReproducibilityConfig {
                seed: Some(1),
                deterministic_run_id: true,
                save_config_snapshot: false,
                record_git_info: false,
            },
        };

        let bundle = backend.resolve_data(&cfg).expect("resolve");
        assert_eq!(bundle.n_observations, 100);
        assert_eq!(bundle.train_n, 70); // floor(100 * 0.70) = 70

        let _ = std::fs::remove_file(&tmp);
    }

    /// Full pipeline with dummy DuckDB data: resolve → features → train → online → evaluate.
    #[test]
    fn full_pipeline_with_fixture_data() {
        use crate::alphavantage::commodity::{CommodityDataPoint, CommodityResponse};
        use crate::cache::CommodityCache;

        let tmp = tempfile_path("test_real_backend_full_pipeline.duckdb");
        let cache = CommodityCache::open(&tmp).expect("open");

        // 500 synthetic daily SPY points with realistic price level.
        let data: Vec<CommodityDataPoint> = (0..500_u64)
            .map(|i| {
                // Simple random walk: start at 400, ±1 % per day.
                let price = 400.0 * (1.0 + 0.001 * (i as f64 % 17.0 - 8.0));
                CommodityDataPoint {
                    date: chrono::DateTime::from_timestamp(
                        1577836800 + i as i64 * 86400,
                        0,
                    )
                    .unwrap()
                    .naive_utc(),
                    value: price,
                }
            })
            .collect();
        let response = CommodityResponse {
            name: "S&P 500 ETF".to_string(),
            interval: "daily".to_string(),
            unit: "USD".to_string(),
            data,
        };
        cache.store("SPY", "daily", &response).expect("store");
        drop(cache);

        let backend = RealBackend::new(&tmp);
        let cfg = ExperimentConfig {
            meta: RunMetaConfig { run_label: "full_pipeline_test".to_string(), notes: None },
            mode: ExperimentMode::Real,
            data: DataConfig::Real {
                asset: "spy".to_string(),
                frequency: RealFrequency::Daily,
                dataset_id: "spy_full_pipeline".to_string(),
                date_start: None,
                date_end: None,
            },
            features: FeatureConfig {
                family: FeatureFamilyConfig::LogReturn,
                scaling: ScalingPolicyConfig::ZScore,
                session_aware: false,
            },
            model: ModelConfig {
                k_regimes: 2,
                training: TrainingMode::FitOffline,
                em_max_iter: 50,
                em_tol: 1e-5,
                em_n_starts: 1,
            },
            detector: DetectorConfig {
                detector_type: DetectorType::HardSwitch,
                threshold: 0.55,
                persistence_required: 2,
                cooldown: 5,
                ema_alpha: None,
            },
            evaluation: EvaluationConfig::Real {
                proxy_events_path: String::new(),
                route_a_point_pre_bars: 3,
                route_a_point_post_bars: 5,
                route_a_causal_only: false,
                route_b_min_segment_len: 5,
            },
            output: OutputConfig {
                root_dir: std::env::temp_dir().to_string_lossy().to_string(),
                write_json: false,
                write_csv: false,
                save_traces: true,
            },
            reproducibility: ReproducibilityConfig {
                seed: Some(99),
                deterministic_run_id: true,
                save_config_snapshot: false,
                record_git_info: false,
            },
        };

        let runner = ExperimentRunner::new(backend);
        let result = runner.run(cfg);

        assert!(
            result.is_success(),
            "pipeline should succeed; status = {:?}; warnings = {:?}",
            result.status,
            result.warnings
        );
        assert!(
            result.evaluation_summary.is_some(),
            "evaluation summary must be populated"
        );
        match result.evaluation_summary.unwrap() {
            crate::experiments::result::EvaluationSummary::Real {
                event_coverage,
                alarm_relevance,
                segmentation_coherence,
            } => {
                // When proxy_events_path is empty, Route A has 0 events →
                // event_coverage is NaN (0/0, undefined but not an error).
                // alarm_relevance may also be NaN if no alarms fired.
                assert!(
                    event_coverage.is_nan() || (event_coverage >= 0.0 && event_coverage <= 1.0),
                    "event_coverage out of range: {event_coverage}"
                );
                assert!(
                    alarm_relevance.is_nan() || (alarm_relevance >= 0.0 && alarm_relevance <= 1.0),
                    "alarm_relevance out of range: {alarm_relevance}"
                );
                assert!(
                    segmentation_coherence.is_nan() || segmentation_coherence >= 0.0,
                    "segmentation_coherence negative: {segmentation_coherence}"
                );
            }
            _ => panic!("expected Real evaluation summary"),
        }

        let _ = std::fs::remove_file(&tmp);
    }

    // Helper: build a deterministic temp-file path for DuckDB tests.
    fn tempfile_path(name: &str) -> String {
        let mut p = std::env::temp_dir();
        p.push(name);
        p.to_string_lossy().to_string()
    }
}
