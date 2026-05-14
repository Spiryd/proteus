/// Model-ready observation stream: the final output of the feature pipeline.
///
/// # Pipeline summary
///
/// ```text
/// CleanSeries  (Phase 15: ascending, gap-checked, metadata-tagged)
///      â”‚
///      â”‚  FeatureConfig { family, scaling, session_aware }
///      â–¼
/// FeatureStream::build(series, config)
///      â”‚   â€¢ compute log-returns (or other family)
///      â”‚   â€¢ apply session-boundary policy (if intraday + session_aware)
///      â”‚   â€¢ compute rolling statistics (if applicable)
///      â”‚   â€¢ trim warmup prefix
///      â”‚   â€¢ fit scaler on training partition (if scaling â‰  None)
///      â”‚   â€¢ apply scaler to all partitions
///      â–¼
/// FeatureStream
///      â”œâ”€â”€ observations: Vec<Observation>  â† y_t, ready for EM / detector
///      â”œâ”€â”€ meta: FeatureStreamMeta         â† provenance
///      â””â”€â”€ scaler: FittedScaler            â† frozen for online use
/// ```
///
/// # Causal guarantee
///
/// No step in `build` uses future information relative to the bar being
/// computed.  The scaler is fitted on the explicitly identified training
/// prefix only.
///
/// # Warmup trimming
///
/// The first `family.warmup_bars()` input bars are consumed to produce the
/// first defined feature value.  This is recorded in
/// `FeatureStreamMeta::n_warmup_dropped`.  The caller should add
/// `n_warmup_dropped` to any absolute-time index when mapping output indices
/// back to the original price series.
use chrono::NaiveDateTime;

use crate::data::{DataMode, DatasetMeta, Observation};

use super::{
    family::FeatureFamily,
    rolling::{
        rolling_vol, rolling_vol_session_aware, standardized_returns,
        standardized_returns_session_aware,
    },
    scaler::{FittedScaler, ScalingPolicy},
    transform::{
        abs_returns, abs_returns_session_aware, different_day, log_returns,
        log_returns_session_aware, squared_returns,
    },
};

// =========================================================================
// FeatureConfig
// =========================================================================

/// Full configuration for one feature-engineering experiment run.
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Which observation family to compute.
    pub family: FeatureFamily,

    /// Normalization policy.
    ///
    /// If anything other than `ScalingPolicy::None`, the scaler is fitted on
    /// the first `n_train` observations of the feature series.
    pub scaling: ScalingPolicy,

    /// Number of feature observations that belong to the training partition.
    ///
    /// Used to fit the scaler without leakage.  Set to `usize::MAX` or the
    /// full series length to fit on everything (useful for offline
    /// exploration, but not for rigorous evaluation).
    pub n_train: usize,

    /// For intraday data: whether to reset session-local rolling windows and
    /// skip cross-session returns at session boundaries.
    ///
    /// Ignored for daily data (has no effect on daily series).
    pub session_aware: bool,
}

// =========================================================================
// FeatureStreamMeta
// =========================================================================

/// Provenance record attached to every `FeatureStream`.
#[derive(Debug, Clone)]
pub struct FeatureStreamMeta {
    /// The source-data metadata (asset, mode, source).
    pub data_meta: DatasetMeta,

    /// The feature family that was applied.
    pub family: FeatureFamily,

    /// Number of price bars dropped as warmup at the beginning of the series.
    ///
    /// `feature_observations[0]` corresponds to
    /// `price_series[n_warmup_dropped]`.
    pub n_warmup_dropped: usize,

    /// Total number of feature observations.
    pub n_obs: usize,
}

// =========================================================================
// FeatureStream
// =========================================================================

/// Model-ready observation stream $y_t$.
///
/// This is the final output of the feature pipeline.  `observations` contains
/// the transformed, optionally scaled values ready for:
/// - offline EM fitting (`.values()` slice),
/// - online streaming (one bar at a time),
/// - benchmark evaluation.
#[derive(Debug, Clone)]
pub struct FeatureStream {
    /// Time-indexed feature values $y_t$.
    pub observations: Vec<Observation>,

    /// Provenance metadata.
    pub meta: FeatureStreamMeta,

    /// The scaler fitted on the training partition.
    ///
    /// For `ScalingPolicy::None` this is an identity transform.
    /// Stored here so it can be applied to live online data without
    /// re-fitting.
    pub scaler: FittedScaler,
}

impl FeatureStream {
    /// Build a `FeatureStream` from a cleaned price series.
    ///
    /// Steps:
    /// 1. Compute raw feature observations from price series.
    /// 2. Apply scaler fitted on the first `config.n_train` feature bars.
    /// 3. Record provenance in `FeatureStreamMeta`.
    ///
    /// # Arguments
    ///
    /// * `prices` â€” ascending-ordered price observations (as from `CleanSeries`).
    /// * `data_meta` â€” provenance metadata from Phase 15.
    /// * `config`   â€” feature configuration.
    pub fn build(prices: &[Observation], data_meta: DatasetMeta, config: FeatureConfig) -> Self {
        let is_intraday = matches!(data_meta.mode, DataMode::Intraday { .. });
        let use_session = config.session_aware && is_intraday;

        // 1. Compute raw (un-scaled) feature observations.
        let raw_obs = compute_raw_features(prices, &config.family, use_session);

        // 2. Fit scaler on train prefix only.
        let n_train = config.n_train.min(raw_obs.len());
        let train_values: Vec<f64> = raw_obs[..n_train].iter().map(|o| o.value).collect();
        let scaler = FittedScaler::fit(&train_values, config.scaling.clone());

        // 3. Apply scaler to all observations.
        let observations: Vec<Observation> = raw_obs
            .into_iter()
            .map(|o| Observation {
                value: scaler.transform_value(o.value),
                ..o
            })
            .collect();

        let n_warmup_dropped = config.family.warmup_bars();
        let n_obs = observations.len();

        let meta = FeatureStreamMeta {
            data_meta,
            family: config.family,
            n_warmup_dropped,
            n_obs,
        };

        Self {
            observations,
            meta,
            scaler,
        }
    }

    /// Feature values as a plain `Vec<f64>`, ready for EM or detector input.
    pub fn values(&self) -> Vec<f64> {
        self.observations.iter().map(|o| o.value).collect()
    }

    /// Timestamps of feature observations.
    pub fn timestamps(&self) -> Vec<NaiveDateTime> {
        self.observations.iter().map(|o| o.timestamp).collect()
    }

    /// Short experiment label combining asset symbol and feature family label.
    ///
    /// Example: `"SPY_log_return"`, `"WTI_rolling_vol_w20_sr0"`.
    pub fn experiment_label(&self) -> String {
        format!(
            "{}_{}",
            self.meta.data_meta.symbol,
            self.meta.family.label()
        )
    }
}

// =========================================================================
// Raw feature computation (private)
// =========================================================================

fn compute_raw_features(
    prices: &[Observation],
    family: &FeatureFamily,
    use_session: bool,
) -> Vec<Observation> {
    match family {
        FeatureFamily::LogReturn => {
            if use_session {
                log_returns_session_aware(prices, different_day)
            } else {
                log_returns(prices)
            }
        }

        FeatureFamily::AbsReturn => {
            if use_session {
                abs_returns_session_aware(prices, different_day)
            } else {
                abs_returns(prices)
            }
        }

        FeatureFamily::SquaredReturn => {
            // SquaredReturn is always pointwise; session-awareness via session
            // predicate on log_returns then square.
            if use_session {
                log_returns_session_aware(prices, different_day)
                    .into_iter()
                    .map(|o| Observation {
                        value: o.value * o.value,
                        ..o
                    })
                    .collect()
            } else {
                squared_returns(prices)
            }
        }

        FeatureFamily::RollingVol {
            window,
            session_reset,
        } => {
            let returns = if use_session {
                log_returns_session_aware(prices, different_day)
            } else {
                log_returns(prices)
            };
            if use_session && *session_reset {
                rolling_vol_session_aware(&returns, *window, different_day)
            } else {
                rolling_vol(&returns, *window)
            }
        }

        FeatureFamily::StandardizedReturn {
            window,
            epsilon,
            session_reset,
        } => {
            let returns = if use_session {
                log_returns_session_aware(prices, different_day)
            } else {
                log_returns(prices)
            };
            if use_session && *session_reset {
                standardized_returns_session_aware(&returns, *window, *epsilon, different_day)
            } else {
                standardized_returns(&returns, *window, *epsilon)
            }
        }
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::meta::{DataMode, DataSource, DatasetMeta, PriceField, SessionConvention};

    fn make_meta(symbol: &str, mode: DataMode) -> DatasetMeta {
        DatasetMeta {
            symbol: symbol.to_string(),
            mode,
            source: DataSource::AlphaVantage,
            price_field: PriceField::AdjustedClose,
            session_convention: SessionConvention::FullDay,
            fetched_at: None,
            unit: None,
        }
    }

    fn obs(ts_str: &str, value: f64) -> Observation {
        Observation {
            timestamp: NaiveDateTime::parse_from_str(ts_str, "%Y-%m-%d %H:%M:%S").unwrap(),
            value,
        }
    }

    fn daily_prices(values: &[(f64, &str)]) -> Vec<Observation> {
        values
            .iter()
            .map(|(v, d)| obs(&format!("{d} 00:00:00"), *v))
            .collect()
    }

    // ---- LogReturn feature stream ----

    #[test]
    fn log_return_stream_length_and_values() {
        let prices = daily_prices(&[
            (100.0, "2024-01-02"),
            (110.0, "2024-01-03"),
            (105.0, "2024-01-04"),
        ]);
        let meta = make_meta("TEST", DataMode::Daily);
        let config = FeatureConfig {
            family: FeatureFamily::LogReturn,
            scaling: ScalingPolicy::None,
            n_train: 10,
            session_aware: false,
        };
        let stream = FeatureStream::build(&prices, meta, config);
        assert_eq!(stream.observations.len(), 2);
        let expected_r0 = (110f64 / 100.0).ln();
        assert!((stream.observations[0].value - expected_r0).abs() < 1e-12);
    }

    #[test]
    fn warmup_metadata_is_one_for_log_return() {
        let prices = daily_prices(&[(100.0, "2024-01-02"), (110.0, "2024-01-03")]);
        let meta = make_meta("TEST", DataMode::Daily);
        let config = FeatureConfig {
            family: FeatureFamily::LogReturn,
            scaling: ScalingPolicy::None,
            n_train: 10,
            session_aware: false,
        };
        let stream = FeatureStream::build(&prices, meta, config);
        assert_eq!(stream.meta.n_warmup_dropped, 1);
    }

    // ---- AbsReturn ----

    #[test]
    fn abs_return_values_are_nonneg() {
        let prices = daily_prices(&[
            (100.0, "2024-01-02"),
            (90.0, "2024-01-03"),
            (95.0, "2024-01-04"),
        ]);
        let meta = make_meta("TEST", DataMode::Daily);
        let config = FeatureConfig {
            family: FeatureFamily::AbsReturn,
            scaling: ScalingPolicy::None,
            n_train: 10,
            session_aware: false,
        };
        let stream = FeatureStream::build(&prices, meta, config);
        for o in &stream.observations {
            assert!(o.value >= 0.0);
        }
    }

    // ---- RollingVol ----

    #[test]
    fn rolling_vol_warmup_is_window() {
        // 10 price bars, window=5 â†’ 4 returns consumed as warmup â†’ 5 outputs
        let prices: Vec<Observation> = (0..10)
            .map(|i| obs(&format!("2024-01-{:02} 00:00:00", i + 2), 100.0 + i as f64))
            .collect();
        let meta = make_meta("TEST", DataMode::Daily);
        let config = FeatureConfig {
            family: FeatureFamily::RollingVol {
                window: 5,
                session_reset: false,
            },
            scaling: ScalingPolicy::None,
            n_train: 100,
            session_aware: false,
        };
        let stream = FeatureStream::build(&prices, meta, config);
        // 9 returns, window=5 â†’ 5 outputs
        assert_eq!(stream.observations.len(), 5);
        assert_eq!(stream.meta.n_warmup_dropped, 5);
    }

    // ---- ZScore scaling fits on train only ----

    #[test]
    fn zscore_train_prefix_determines_scaler() {
        // 4 prices â†’ 3 returns; n_train=2 â†’ scaler fitted on first 2 returns only
        let prices = daily_prices(&[
            (100.0, "2024-01-02"),
            (110.0, "2024-01-03"),
            (105.0, "2024-01-04"),
            (115.0, "2024-01-05"),
        ]);
        let meta = make_meta("TEST", DataMode::Daily);
        let config = FeatureConfig {
            family: FeatureFamily::LogReturn,
            scaling: ScalingPolicy::ZScore,
            n_train: 2,
            session_aware: false,
        };
        let stream = FeatureStream::build(&prices, meta, config);
        // Scaler location should match the mean of first 2 raw returns.
        let r0 = (110f64 / 100.0).ln();
        let r1 = (105f64 / 110.0).ln();
        let expected_location = (r0 + r1) / 2.0;
        assert!((stream.scaler.location - expected_location).abs() < 1e-12);
    }

    // ---- experiment_label ----

    #[test]
    fn experiment_label_contains_symbol_and_family() {
        let prices = daily_prices(&[(100.0, "2024-01-02"), (110.0, "2024-01-03")]);
        let meta = make_meta("SPY", DataMode::Daily);
        let config = FeatureConfig {
            family: FeatureFamily::LogReturn,
            scaling: ScalingPolicy::None,
            n_train: 10,
            session_aware: false,
        };
        let stream = FeatureStream::build(&prices, meta, config);
        assert_eq!(stream.experiment_label(), "SPY_log_return");
    }

    // ---- Intraday session-aware stream ----

    #[test]
    fn intraday_session_aware_skips_overnight_returns() {
        // 2 sessions of 2 bars each; session_aware=true skips the cross-day pair
        let prices = vec![
            obs("2024-01-02 09:30:00", 100.0),
            obs("2024-01-02 09:35:00", 101.0),
            obs("2024-01-03 09:30:00", 102.0), // cross-session: skipped
            obs("2024-01-03 09:35:00", 103.0),
        ];
        let meta = make_meta("SPY", DataMode::Intraday { bar_minutes: 5 });
        let config = FeatureConfig {
            family: FeatureFamily::LogReturn,
            scaling: ScalingPolicy::None,
            n_train: 10,
            session_aware: true,
        };
        let stream = FeatureStream::build(&prices, meta, config);
        // Returns at 09:35 day1 + 09:35 day2 = 2 (cross-day return skipped)
        assert_eq!(stream.observations.len(), 2);
    }

    #[test]
    fn daily_mode_session_aware_flag_has_no_effect() {
        // session_aware flag ignored for daily data
        let prices = daily_prices(&[
            (100.0, "2024-01-02"),
            (110.0, "2024-01-03"),
            (105.0, "2024-01-04"),
        ]);
        let meta_no_sa = make_meta("TEST", DataMode::Daily);
        let meta_sa = make_meta("TEST", DataMode::Daily);

        let cfg_no = FeatureConfig {
            family: FeatureFamily::LogReturn,
            scaling: ScalingPolicy::None,
            n_train: 10,
            session_aware: false,
        };
        let cfg_yes = FeatureConfig {
            family: FeatureFamily::LogReturn,
            scaling: ScalingPolicy::None,
            n_train: 10,
            session_aware: true,
        };

        let s_no = FeatureStream::build(&prices, meta_no_sa, cfg_no);
        let s_yes = FeatureStream::build(&prices, meta_sa, cfg_yes);

        assert_eq!(s_no.observations.len(), s_yes.observations.len());
        for (a, b) in s_no.observations.iter().zip(s_yes.observations.iter()) {
            assert!((a.value - b.value).abs() < 1e-12);
        }
    }

    // ---- Empty and degenerate inputs ----

    #[test]
    fn empty_price_series_produces_empty_stream() {
        let meta = make_meta("TEST", DataMode::Daily);
        let config = FeatureConfig {
            family: FeatureFamily::LogReturn,
            scaling: ScalingPolicy::None,
            n_train: 0,
            session_aware: false,
        };
        let stream = FeatureStream::build(&[], meta, config);
        assert!(stream.observations.is_empty());
            }

    #[test]
    fn single_price_produces_empty_log_return_stream() {
        let prices = daily_prices(&[(100.0, "2024-01-02")]);
        let meta = make_meta("TEST", DataMode::Daily);
        let config = FeatureConfig {
            family: FeatureFamily::LogReturn,
            scaling: ScalingPolicy::None,
            n_train: 0,
            session_aware: false,
        };
        let stream = FeatureStream::build(&prices, meta, config);
        assert!(stream.observations.is_empty());
    }
}

