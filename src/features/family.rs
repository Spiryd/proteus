#![allow(dead_code)]
//! Observation-family configuration for the financial feature pipeline.
//!
//! # The central design decision
//!
//! The Markov Switching Model assumes
//!
//! ```text
//! y_t | S_t = j  ~  N(μ_j, σ_j²)
//! ```
//!
//! but the model says nothing about what `y_t` *is*.  For real market data,
//! `y_t` is an observation-design choice.  This module enumerates the
//! supported families and their configuration parameters.
//!
//! # Family taxonomy
//!
//! | Family | Symbol | Detector is sensitive to … |
//! |---|---|---|
//! | `LogReturn` | $r_t$ | changes in return distribution (mean, dispersion) |
//! | `AbsReturn` | $\|r_t\|$ | changes in return magnitude / activity |
//! | `SquaredReturn` | $r_t^2$ | changes in second-moment structure |
//! | `RollingVol` | $v_t^{(w)}$ | changes in recent rolling volatility level |
//! | `StandardizedReturn` | $z_t$ | changes in normalized shock size |
//!
//! The choice of family is part of the scientific hypothesis:
//! returning to a null-model where raw prices are fed directly would be
//! almost never defensible for long-horizon regime analysis.

// =========================================================================
// FeatureFamily
// =========================================================================

/// One member of the supported observation-design family.
///
/// Each variant carries the configuration parameters relevant to that
/// transformation.  `FeatureFamily` values are cheaply cloneable and
/// can be serialized or embedded in experiment labels.
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureFamily {
    /// $y_t = r_t = \log P_t - \log P_{t-1}$.
    ///
    /// Warmup: 1 preceding price observation required.
    LogReturn,

    /// $y_t = |r_t|$.
    ///
    /// Warmup: 1 preceding price observation required (same as `LogReturn`).
    AbsReturn,

    /// $y_t = r_t^2$.
    ///
    /// Warmup: 1 preceding price observation required.
    SquaredReturn,

    /// $y_t = v_t^{(w)}$ — trailing sample standard deviation of the last `w`
    /// log returns.
    ///
    /// $$v_t^{(w)} = \sqrt{\frac{1}{w}\sum_{k=0}^{w-1}(r_{t-k} - \bar r_t^{(w)})^2}$$
    ///
    /// Warmup: `w` preceding price observations (i.e., `w` returns) required.
    RollingVol {
        /// Rolling window size in bars.
        window: usize,
        /// If `true`, the rolling window resets at each calendar-day session
        /// boundary for intraday data.  Ignored for daily data.
        session_reset: bool,
    },

    /// $y_t = r_t / (v_t^{(w)} + \varepsilon)$ — return standardized by
    /// the `window`-bar trailing volatility.
    ///
    /// Warmup: `window` preceding price observations required.
    StandardizedReturn {
        /// Rolling window size for the denominator volatility estimate.
        window: usize,
        /// Numerical stabilizer added to the denominator: $v_t^{(w)} + \varepsilon$.
        /// Default: 1e-8.
        epsilon: f64,
        /// If `true`, the volatility window resets at each session boundary.
        session_reset: bool,
    },
}

impl FeatureFamily {
    /// Minimum number of **price** observations that must precede the first
    /// defined feature value.
    ///
    /// The first `warmup_bars()` price observations in the input are consumed
    /// to produce the first feature value.  Equivalently, the output series
    /// is `warmup_bars()` observations shorter than the input price series.
    pub fn warmup_bars(&self) -> usize {
        match self {
            Self::LogReturn | Self::AbsReturn | Self::SquaredReturn => 1,
            Self::RollingVol { window, .. } => *window,
            Self::StandardizedReturn { window, .. } => *window,
        }
    }

    /// Human-readable short label suitable for experiment tables.
    pub fn label(&self) -> String {
        match self {
            Self::LogReturn => "log_return".to_string(),
            Self::AbsReturn => "abs_return".to_string(),
            Self::SquaredReturn => "sq_return".to_string(),
            Self::RollingVol {
                window,
                session_reset,
            } => {
                format!("rolling_vol_w{window}_sr{}", u8::from(*session_reset))
            }
            Self::StandardizedReturn {
                window,
                session_reset,
                ..
            } => {
                format!("std_return_w{window}_sr{}", u8::from(*session_reset))
            }
        }
    }

    /// Whether this family requires a rolling-window computation.
    pub fn is_rolling(&self) -> bool {
        matches!(
            self,
            Self::RollingVol { .. } | Self::StandardizedReturn { .. }
        )
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_for_pointwise_families_is_one() {
        assert_eq!(FeatureFamily::LogReturn.warmup_bars(), 1);
        assert_eq!(FeatureFamily::AbsReturn.warmup_bars(), 1);
        assert_eq!(FeatureFamily::SquaredReturn.warmup_bars(), 1);
    }

    #[test]
    fn warmup_for_rolling_is_window() {
        let fam = FeatureFamily::RollingVol {
            window: 20,
            session_reset: false,
        };
        assert_eq!(fam.warmup_bars(), 20);
        let fam2 = FeatureFamily::StandardizedReturn {
            window: 10,
            epsilon: 1e-8,
            session_reset: true,
        };
        assert_eq!(fam2.warmup_bars(), 10);
    }

    #[test]
    fn label_roundtrip_contains_window() {
        let fam = FeatureFamily::RollingVol {
            window: 15,
            session_reset: true,
        };
        assert!(fam.label().contains("15"));
        assert!(fam.label().contains("sr1"));
    }

    #[test]
    fn is_rolling_flag_correct() {
        assert!(!FeatureFamily::LogReturn.is_rolling());
        assert!(!FeatureFamily::AbsReturn.is_rolling());
        assert!(
            FeatureFamily::RollingVol {
                window: 5,
                session_reset: false
            }
            .is_rolling()
        );
        assert!(
            FeatureFamily::StandardizedReturn {
                window: 5,
                epsilon: 1e-8,
                session_reset: false
            }
            .is_rolling()
        );
    }
}
