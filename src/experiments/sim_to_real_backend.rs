/// Sim-to-real experiment backend (Phase A′2).
///
/// Implements the full sim-to-real pipeline:
///
/// 1. **Resolve data** — load the real series, partition 70/15/15, fit the
///    calibration mapping `K` on real-train log-returns (Quick-EM by default),
///    and simulate a synthetic stream of length `horizon` from the calibrated
///    `ModelParams`.  Both the real prices and the raw synthetic observations
///    are stored in the resulting `DataBundle`.
/// 2. **Build features** — apply the configured feature transform to the real
///    prices and fit a single `FittedScaler` on the real training partition.
///    The same scaler is then applied to the synthetic stream so that the
///    EM-training data and the online-detection data share scale (B′1
///    contract).  The bundle's `observations` field carries the scaled real
///    series; `synthetic_train_obs` carries the scaled synthetic stream.
/// 3. **Train or load model** — Baum-Welch EM is fit on the synthetic stream
///    only.  This is the defining sim-to-real step.
/// 4. **Run online** — the synthetic-trained `FrozenModel` is streamed across
///    the full real series; alarms, scores and posteriors are collected.
/// 5. **Evaluate real** — identical Route A + Route B evaluation to
///    `RealBackend::evaluate_real`, computed on the real series.
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::alphavantage::commodity::{CommodityEndpoint, Interval};
use crate::cache::CommodityCache;
use crate::calibration::{
    CalibrationDatasetTag, CalibrationPartition, DEFAULT_SCALE_TOLERANCE,
    EmpiricalCalibrationProfile, SummaryTargetSet, calibrate_to_synthetic, scale_consistency_check,
    summarize_observation_values,
};
use crate::data::split::{PartitionedSeries, SplitConfig};
use crate::data::{
    CleanSeries, DataMode, DataSource, DatasetMeta, Observation, PriceField, SessionConvention,
};
use crate::features::family::FeatureFamily;
use crate::features::scaler::ScalingPolicy;
use crate::features::stream::{FeatureConfig as StreamFeatureConfig, FeatureStream};
use crate::model::simulate;
use crate::model::simulate::JumpParams;
use crate::real_eval::report::{RealEvalMeta, evaluate_real_data};
use crate::real_eval::route_a::{PointMatchPolicy, ProxyEvent, RouteAConfig};
use crate::real_eval::route_b::{RouteBConfig, ShortSegmentPolicy};

use super::config::{
    DataConfig, EvaluationConfig, ExperimentConfig, FeatureFamilyConfig, RealFrequency,
    ScalingPolicyConfig,
};
use super::runner::{
    DataBundle, ExperimentBackend, FeatureBundle, ModelArtifact, OnlineRunArtifact,
    RealEvalArtifact, SyntheticEvalArtifact,
};
use super::shared::{run_online_shared, train_or_load_model_shared};

/// Sim-to-real backend (stub for A′1; full pipeline lands in A′2).
#[derive(Debug, Clone)]
pub struct SimToRealBackend {
    pub cache_path: String,
}

impl SimToRealBackend {
    pub fn new(cache_path: impl Into<String>) -> Self {
        Self {
            cache_path: cache_path.into(),
        }
    }

    pub(crate) fn to_interval(freq: &RealFrequency) -> Interval {
        match freq {
            RealFrequency::Daily => Interval::Daily,
            RealFrequency::Intraday5m => Interval::Intraday5Min,
            RealFrequency::Intraday15m => Interval::Intraday15Min,
        }
    }

    pub(crate) fn to_data_mode(freq: &RealFrequency) -> DataMode {
        match freq {
            RealFrequency::Daily => DataMode::Daily,
            RealFrequency::Intraday5m => DataMode::Intraday { bar_minutes: 5 },
            RealFrequency::Intraday15m => DataMode::Intraday { bar_minutes: 15 },
        }
    }

    fn to_feature_family(cfg_family: &FeatureFamilyConfig) -> FeatureFamily {
        match cfg_family {
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
        }
    }

    fn to_scaling_policy(cfg_scale: &ScalingPolicyConfig) -> ScalingPolicy {
        match cfg_scale {
            ScalingPolicyConfig::None => ScalingPolicy::None,
            ScalingPolicyConfig::ZScore => ScalingPolicy::ZScore,
            ScalingPolicyConfig::RobustZScore => ScalingPolicy::RobustZScore,
        }
    }

    fn load_proxy_events(path: &str) -> anyhow::Result<Vec<ProxyEvent>> {
        if path.is_empty() {
            return Ok(vec![]);
        }
        let json = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Cannot read proxy events file '{path}': {e}"))?;
        serde_json::from_str(&json)
            .map_err(|e| anyhow::anyhow!("Failed to parse proxy events JSON from '{path}': {e}"))
    }
}

impl ExperimentBackend for SimToRealBackend {
    fn resolve_data(&self, cfg: &ExperimentConfig) -> anyhow::Result<DataBundle> {
        let (
            real_asset,
            real_freq,
            real_dataset_id,
            real_date_start,
            real_date_end,
            horizon,
            mapping_cfg,
        ) = match &cfg.data {
            DataConfig::CalibratedSynthetic {
                real_asset,
                real_frequency,
                real_dataset_id,
                real_date_start,
                real_date_end,
                horizon,
                mapping,
                ..
            } => (
                real_asset.as_str(),
                real_frequency,
                real_dataset_id.as_str(),
                real_date_start.as_deref(),
                real_date_end.as_deref(),
                *horizon,
                mapping.clone(),
            ),
            _ => anyhow::bail!(
                "SimToRealBackend::resolve_data requires DataConfig::CalibratedSynthetic"
            ),
        };

        let series = load_clean_series(
            &self.cache_path,
            real_asset,
            real_freq,
            real_date_start,
            real_date_end,
        )?;
        let n = series.observations.len();
        let prices: Vec<f64> = series.observations.iter().map(|o| o.value).collect();
        let timestamps: Vec<chrono::NaiveDateTime> =
            series.observations.iter().map(|o| o.timestamp).collect();
        let train_idx = (n as f64 * 0.70) as usize;
        let val_idx = (n as f64 * 0.85) as usize;
        let epoch_max = chrono::NaiveDate::from_ymd_opt(9999, 12, 31)
            .unwrap()
            .and_hms_opt(23, 59, 59)
            .unwrap();
        let train_end_ts = timestamps.get(train_idx).copied().unwrap_or(epoch_max);
        let val_end_ts = timestamps.get(val_idx).copied().unwrap_or(epoch_max);

        let split_cfg = SplitConfig {
            train_end: train_end_ts,
            val_end: val_end_ts,
        };
        let partitioned = PartitionedSeries::from_series(series, split_cfg);
        let train_n = partitioned.train.len();

        // Calibrate from real-train log-returns (raw, pre-scaling).
        let train_prices: Vec<f64> = partitioned.train.iter().map(|o| o.value).collect();
        let train_log_returns: Vec<f64> = if train_prices.len() >= 2 {
            (1..train_prices.len())
                .map(|i| (train_prices[i] / train_prices[i - 1]).ln())
                .collect()
        } else {
            Vec::new()
        };
        let summary = summarize_observation_values(&train_log_returns);
        let profile = EmpiricalCalibrationProfile {
            tag: CalibrationDatasetTag {
                asset: real_asset.to_string(),
                frequency: format!("{:?}", real_freq).to_lowercase(),
                feature_label: "log_return".to_string(),
                partition: CalibrationPartition::TrainOnly,
            },
            feature_family: FeatureFamily::LogReturn,
            targets: SummaryTargetSet::Full,
            summary,
            observations: train_log_returns,
        };
        let mut mapping_cfg = mapping_cfg;
        mapping_cfg.k = cfg.model.k_regimes;
        mapping_cfg.horizon = horizon;
        let calibrated = calibrate_to_synthetic(&profile, &mapping_cfg)?;

        // Simulate the synthetic training stream (deterministic from seed).
        let seed = cfg.reproducibility.seed.unwrap_or(42);
        let mut rng = SmallRng::seed_from_u64(seed);
        let jump_params: Option<JumpParams> = calibrated.jump.as_ref().map(|j| JumpParams {
            prob: j.jump_prob,
            scale_mult: j.jump_scale_mult,
        });
        let sim = simulate::simulate_with_jump(
            calibrated.model_params.clone(),
            calibrated.horizon,
            &mut rng,
            jump_params.as_ref(),
        )?;

        // B′1 scale-consistency check: synthetic vs empirical std on the raw
        // (pre-scaling) log-return space.  Captured in the provenance blob
        // so the downstream report carries the calibration health signal.
        let empirical_summary = summarize_observation_values(&profile.observations);
        let synthetic_summary = summarize_observation_values(&sim.observations);
        let scale_check = scale_consistency_check(
            &empirical_summary,
            &synthetic_summary,
            DEFAULT_SCALE_TOLERANCE,
        );

        let interval = Self::to_interval(real_freq);
        let dataset_key = format!(
            "simreal:{}:{}:{}",
            real_asset,
            interval.as_str(),
            real_dataset_id
        );

        let provenance_json = serde_json::to_string_pretty(&serde_json::json!({
            "real_asset": real_asset,
            "real_frequency": format!("{:?}", real_freq),
            "real_dataset_id": real_dataset_id,
            "real_n_train": train_n,
            "calibration_strategy": format!("{:?}", mapping_cfg.strategy),
            "mapping_notes": calibrated.mapping_notes,
            "expected_durations": calibrated.expected_durations,
            "horizon": horizon,
            "seed": seed,
            "scale_check": {
                "empirical_std": scale_check.empirical_std,
                "synthetic_std": scale_check.synthetic_std,
                "relative_error": scale_check.relative_error,
                "tolerance": scale_check.tolerance,
                "within_tolerance": scale_check.within_tolerance,
            },
        }))
        .ok();

        Ok(DataBundle {
            dataset_key,
            n_observations: n,
            observations: prices,
            changepoint_truth: None,
            train_n,
            timestamps,
            split_summary_json: provenance_json,
            validation_report_json: None,
            synthetic_observations: sim.observations,
        })
    }

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
                synthetic_train_obs: Vec::new(),
            });
        }

        // Build the real feature stream (with scaler fitted on real-train).
        let price_obs: Vec<Observation> = data
            .timestamps
            .iter()
            .zip(data.observations.iter())
            .map(|(&ts, &v)| Observation {
                timestamp: ts,
                value: v,
            })
            .collect();

        let (asset, frequency) = match &cfg.data {
            DataConfig::CalibratedSynthetic {
                real_asset,
                real_frequency,
                ..
            } => (real_asset.as_str(), real_frequency),
            _ => anyhow::bail!(
                "SimToRealBackend::build_features: expected CalibratedSynthetic DataConfig"
            ),
        };
        let endpoint: CommodityEndpoint = asset.parse()?;
        let data_meta = DatasetMeta {
            symbol: endpoint.cache_key().to_string(),
            mode: Self::to_data_mode(frequency),
            source: DataSource::AlphaVantage,
            price_field: match &endpoint {
                CommodityEndpoint::Spy | CommodityEndpoint::Qqq => PriceField::AdjustedClose,
                _ => PriceField::Value,
            },
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

        // Real feature observations.
        let feature_timestamps: Vec<chrono::NaiveDateTime> = stream.timestamps();
        let real_values = stream.values();
        let n = real_values.len();
        let actual_train_n = stream
            .meta
            .n_obs
            .min(data.train_n.saturating_sub(stream.meta.n_warmup_dropped));

        // Apply the SAME scaler to the synthetic stream (one-sided policy).
        // The synthetic observations are emitted on the same scale as the
        // raw feature values (log-returns), so the fitted scaler maps them
        // into the same space as the real features.
        let scaled_synth: Vec<f64> = stream.scaler.transform(&data.synthetic_observations);

        Ok(FeatureBundle {
            feature_label: stream.experiment_label(),
            n_observations: n,
            observations: real_values,
            train_n: actual_train_n,
            timestamps: feature_timestamps,
            synthetic_train_obs: scaled_synth,
        })
    }

    fn train_or_load_model(
        &self,
        cfg: &ExperimentConfig,
        features: &FeatureBundle,
    ) -> anyhow::Result<ModelArtifact> {
        if features.synthetic_train_obs.is_empty() {
            anyhow::bail!(
                "SimToRealBackend::train_or_load_model: synthetic_train_obs is empty; \
                 cannot fit EM. Did resolve_data/build_features produce a synthetic stream?"
            );
        }
        // Construct an inner bundle whose `observations` are the synthetic
        // training stream, so that `train_or_load_model_shared` fits EM on
        // synthetic data only.
        let inner = FeatureBundle {
            feature_label: features.feature_label.clone(),
            n_observations: features.synthetic_train_obs.len(),
            observations: features.synthetic_train_obs.clone(),
            train_n: features.synthetic_train_obs.len(),
            timestamps: Vec::new(),
            synthetic_train_obs: Vec::new(),
        };
        let mut artifact = train_or_load_model_shared(cfg, &inner)?;
        artifact.source = format!("{}:synthetic_trained", artifact.source);
        Ok(artifact)
    }

    fn run_online(
        &self,
        cfg: &ExperimentConfig,
        model: &ModelArtifact,
        features: &FeatureBundle,
    ) -> anyhow::Result<OnlineRunArtifact> {
        // The synthetic-trained `FrozenModel` is streamed across the *real*
        // feature observations.  Use the real series as-is.
        run_online_shared(cfg, model, features)
    }

    fn evaluate_synthetic(
        &self,
        _cfg: &ExperimentConfig,
        _online: &OnlineRunArtifact,
    ) -> anyhow::Result<SyntheticEvalArtifact> {
        anyhow::bail!("SimToRealBackend evaluates on real data; use evaluate_real")
    }

    fn evaluate_real(
        &self,
        cfg: &ExperimentConfig,
        online: &OnlineRunArtifact,
    ) -> anyhow::Result<RealEvalArtifact> {
        let (
            asset,
            frequency,
            date_start,
            date_end,
            proxy_events_path,
            route_a_pre,
            route_a_post,
            route_a_causal,
            route_b_min_seg,
        ) = match (&cfg.data, &cfg.evaluation) {
            (
                DataConfig::CalibratedSynthetic {
                    real_asset,
                    real_frequency,
                    real_date_start,
                    real_date_end,
                    ..
                },
                EvaluationConfig::Real {
                    proxy_events_path,
                    route_a_point_pre_bars,
                    route_a_point_post_bars,
                    route_a_causal_only,
                    route_b_min_segment_len,
                },
            ) => (
                real_asset.as_str(),
                real_frequency,
                real_date_start.as_deref(),
                real_date_end.as_deref(),
                proxy_events_path.as_str(),
                *route_a_point_pre_bars,
                *route_a_point_post_bars,
                *route_a_causal_only,
                *route_b_min_segment_len,
            ),
            _ => anyhow::bail!(
                "SimToRealBackend::evaluate_real: expected CalibratedSynthetic + Real eval"
            ),
        };

        // Re-load and feature-transform the real series.
        let series = load_clean_series(&self.cache_path, asset, frequency, date_start, date_end)?;
        let endpoint: CommodityEndpoint = asset.parse()?;
        let data_meta = DatasetMeta {
            symbol: endpoint.cache_key().to_string(),
            mode: Self::to_data_mode(frequency),
            source: DataSource::AlphaVantage,
            price_field: match &endpoint {
                CommodityEndpoint::Spy | CommodityEndpoint::Qqq => PriceField::AdjustedClose,
                _ => PriceField::Value,
            },
            session_convention: match frequency {
                RealFrequency::Daily => SessionConvention::FullDay,
                _ => SessionConvention::RthOnly,
            },
            fetched_at: None,
            unit: None,
        };
        let n_raw = series.observations.len();
        let train_n = (n_raw as f64 * 0.70) as usize;
        let stream_cfg = StreamFeatureConfig {
            family: Self::to_feature_family(&cfg.features.family),
            scaling: Self::to_scaling_policy(&cfg.features.scaling),
            n_train: train_n,
            session_aware: cfg.features.session_aware,
        };
        let stream = FeatureStream::build(&series.observations, data_meta.clone(), stream_cfg);

        let route_a_cfg = RouteAConfig {
            point_policy: PointMatchPolicy {
                pre_bars: route_a_pre,
                post_bars: route_a_post,
                causal_only: route_a_causal,
            },
        };
        let route_b_cfg = RouteBConfig {
            min_len: route_b_min_seg,
            short_segment_policy: ShortSegmentPolicy::FlagOnly,
        };
        let proxy_events = Self::load_proxy_events(proxy_events_path)?;
        let eval_meta = RealEvalMeta {
            asset: endpoint.cache_key().to_string(),
            frequency: Self::to_interval(frequency).as_str().to_string(),
            feature_label: stream.experiment_label(),
            detector_label: format!("{:?}", cfg.detector.detector_type),
        };
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

fn load_clean_series(
    cache_path: &str,
    asset: &str,
    frequency: &RealFrequency,
    date_start: Option<&str>,
    date_end: Option<&str>,
) -> anyhow::Result<CleanSeries> {
    let cache = CommodityCache::open(cache_path)?;
    let endpoint: CommodityEndpoint = asset.parse()?;
    let interval = SimToRealBackend::to_interval(frequency);
    let mode = SimToRealBackend::to_data_mode(frequency);
    let price_field = match &endpoint {
        CommodityEndpoint::Spy | CommodityEndpoint::Qqq => PriceField::AdjustedClose,
        _ => PriceField::Value,
    };
    let response = cache
        .load(endpoint.cache_key(), interval.as_str())?
        .ok_or_else(|| {
            anyhow::anyhow!(
                "No cached data for {} ({}). Run ingest first.",
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
    if let Some(start_str) = date_start
        && !start_str.is_empty()
    {
        let start = chrono::NaiveDate::parse_from_str(start_str, "%Y-%m-%d")
            .map_err(|_| anyhow::anyhow!("Invalid date_start '{start_str}': expected YYYY-MM-DD"))?
            .and_hms_opt(0, 0, 0)
            .unwrap();
        series.observations.retain(|o| o.timestamp >= start);
    }
    if let Some(end_str) = date_end
        && !end_str.is_empty()
    {
        let end = chrono::NaiveDate::parse_from_str(end_str, "%Y-%m-%d")
            .map_err(|_| anyhow::anyhow!("Invalid date_end '{end_str}': expected YYYY-MM-DD"))?
            .and_hms_opt(23, 59, 59)
            .unwrap();
        series.observations.retain(|o| o.timestamp <= end);
    }
    Ok(series)
}
