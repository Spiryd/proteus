#![allow(unused_imports)]
/// Feature engineering and observation design for real financial data.
///
/// # Phase 16 — Observation Design
///
/// This module is the answer to the question:
///
/// > What is the observed process $y_t$ that the Markov Switching detector
/// > should model?
///
/// The model assumes $y_t \mid S_t = j \sim \mathcal{N}(\mu_j, \sigma_j^2)$.
/// This module defines what $y_t$ *is* for real market data.
///
/// # Architecture
///
/// ```text
/// CleanSeries  (Phase 15)
///      │   Vec<Observation> { timestamp, price }
///      │
///      │   FeatureConfig { family, scaling, n_train, session_aware }
///      ▼
/// FeatureStream::build(prices, data_meta, config)
///      │   • log/abs/squared returns  (transform.rs)
///      │   • rolling vol / std return (rolling.rs)
///      │   • session-boundary policy  (transform.rs + rolling.rs)
///      │   • warmup trimming
///      │   • scaler fitted on train only (scaler.rs)
///      ▼
/// FeatureStream { observations: Vec<Observation>, meta, scaler }
///      │
///      ▼
/// Offline EM · Online Detector · Benchmark
/// ```
///
/// # Causality guarantee
///
/// Every transformation is trailing/causal.  Feature value $y_t$ depends
/// only on prices at times $\leq t$.  This is a hard requirement for the
/// online-detection setting.
///
/// # Module structure
///
/// | Sub-module | Responsibility |
/// |---|---|
/// | `family`    | `FeatureFamily` enum and per-family warmup metadata |
/// | `transform` | Causal pointwise transforms: log/abs/squared return; session-aware variants |
/// | `rolling`   | `RollingStats` accumulator; `rolling_vol`, `standardized_returns`; session-reset variants |
/// | `scaler`    | `ScalingPolicy`, `FittedScaler` (fit on train, apply everywhere) |
/// | `stream`    | `FeatureConfig`, `FeatureStream`, `FeatureStreamMeta` — the public pipeline entry point |
pub mod family;
pub mod rolling;
pub mod scaler;
pub mod stream;
pub mod transform;

// Flat re-exports for ergonomic use by downstream phases.

pub use family::FeatureFamily;
pub use rolling::{
    RollingStats, rolling_vol, rolling_vol_session_aware, standardized_returns,
    standardized_returns_session_aware,
};
pub use scaler::{FittedScaler, ScalingPolicy};
pub use stream::{FeatureConfig, FeatureStream, FeatureStreamMeta};
pub use transform::{
    abs_return, abs_returns, abs_returns_session_aware, different_day, log_return, log_returns,
    log_returns_session_aware, squared_return, squared_returns,
};
