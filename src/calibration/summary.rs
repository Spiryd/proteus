/// Empirical summary extraction for Phase 17 synthetic-to-real calibration.
///
/// This module computes feature-family-aware calibration statistics from the
/// same modeled observation process used by the detector (Phase 16 output).
///
/// The core principle is:
///
/// - Do not calibrate on raw prices if the model observes transformed values.
/// - Compute summaries on `FeatureStream::observations` (the actual `y_t`).
use serde::{Deserialize, Serialize};

use crate::features::FeatureFamily;

/// Which partition is allowed for calibration fitting.
///
/// Calibration must be leakage-safe; the default policy is `TrainOnly`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalibrationPartition {
    TrainOnly,
    TrainAndValidation,
    FullSeries,
}

/// Lightweight metadata identifying the empirical source used for calibration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationDatasetTag {
    pub asset: String,
    pub frequency: String,
    pub feature_label: String,
    pub partition: CalibrationPartition,
}

/// Calibrated summary functionals T_1..T_m computed from empirical `y_t`.
///
/// These are the quantities the calibration mapping attempts to match in the
/// synthetic generator, either exactly (for some parameters) or approximately.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmpiricalSummary {
    pub n: usize,
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,

    // Distribution-shape anchors
    pub q01: f64,
    pub q05: f64,
    pub q50: f64,
    pub q95: f64,
    pub q99: f64,

    // Tail-event frequencies
    pub tail_freq_q95: f64, // P(y_t > q95)
    pub abs_exceed_2std: f64,

    // Dependence / turbulence proxies
    pub acf1: f64,
    pub abs_acf1: f64,

    // Change-frequency proxy: share of points where sign changes.
    pub sign_change_rate: f64,

    // Episode-duration proxies from thresholded |y_t|.
    pub high_episode_mean_duration: f64,
    pub low_episode_mean_duration: f64,
}

/// Selection policy for which calibration statistics should be considered
/// active by downstream mapping logic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SummaryTargetSet {
    MarginalOnly,
    RegimeStructural,
    Full,
}

/// Bundle of summary + dataset identity + target set.
#[derive(Debug, Clone, PartialEq)]
pub struct EmpiricalCalibrationProfile {
    pub tag: CalibrationDatasetTag,
    pub feature_family: FeatureFamily,
    pub targets: SummaryTargetSet,
    pub summary: EmpiricalSummary,
    /// Raw partition observations (used by `CalibrationStrategy::QuickEm`).
    /// May be empty when only summary statistics are available.
    pub observations: Vec<f64>,
}

pub fn summarize_observation_values(values: &[f64]) -> EmpiricalSummary {
    if values.is_empty() {
        return EmpiricalSummary {
            n: 0,
            mean: 0.0,
            variance: 0.0,
            std_dev: 0.0,
            q01: 0.0,
            q05: 0.0,
            q50: 0.0,
            q95: 0.0,
            q99: 0.0,
            tail_freq_q95: 0.0,
            abs_exceed_2std: 0.0,
            acf1: 0.0,
            abs_acf1: 0.0,
            sign_change_rate: 0.0,
            high_episode_mean_duration: 0.0,
            low_episode_mean_duration: 0.0,
        };
    }

    let n = values.len();
    let n_f = n as f64;

    let mean = values.iter().sum::<f64>() / n_f;
    let variance = values.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n_f;
    let std_dev = variance.sqrt();

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q01 = percentile_sorted(&sorted, 0.01);
    let q05 = percentile_sorted(&sorted, 0.05);
    let q50 = percentile_sorted(&sorted, 0.50);
    let q95 = percentile_sorted(&sorted, 0.95);
    let q99 = percentile_sorted(&sorted, 0.99);

    let tail_freq_q95 = values.iter().filter(|&&x| x > q95).count() as f64 / n_f;
    let abs_exceed_2std = if std_dev > 0.0 {
        values.iter().filter(|&&x| x.abs() > 2.0 * std_dev).count() as f64 / n_f
    } else {
        0.0
    };

    let acf1 = lag1_autocorr(values);
    let abs_values: Vec<f64> = values.iter().map(|x| x.abs()).collect();
    let abs_acf1 = lag1_autocorr(&abs_values);

    let sign_change_rate = sign_change_fraction(values);
    let (high_episode_mean_duration, low_episode_mean_duration) =
        episode_duration_means(&abs_values, q95.abs());

    EmpiricalSummary {
        n,
        mean,
        variance,
        std_dev,
        q01,
        q05,
        q50,
        q95,
        q99,
        tail_freq_q95,
        abs_exceed_2std,
        acf1,
        abs_acf1,
        sign_change_rate,
        high_episode_mean_duration,
        low_episode_mean_duration,
    }
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = p * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

fn lag1_autocorr(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let num = values
        .windows(2)
        .map(|w| (w[0] - mean) * (w[1] - mean))
        .sum::<f64>();
    let den = values.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>();
    if den == 0.0 { 0.0 } else { num / den }
}

fn sign_change_fraction(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mut changes = 0usize;
    let mut total = 0usize;
    for w in values.windows(2) {
        if w[0] == 0.0 || w[1] == 0.0 {
            continue;
        }
        total += 1;
        if w[0].is_sign_positive() != w[1].is_sign_positive() {
            changes += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        changes as f64 / total as f64
    }
}

fn episode_duration_means(abs_values: &[f64], high_threshold: f64) -> (f64, f64) {
    if abs_values.is_empty() {
        return (0.0, 0.0);
    }
    let mut high_runs = Vec::new();
    let mut low_runs = Vec::new();

    let mut current_is_high = abs_values[0] > high_threshold;
    let mut len = 1usize;
    for &v in &abs_values[1..] {
        let is_high = v > high_threshold;
        if is_high == current_is_high {
            len += 1;
        } else {
            if current_is_high {
                high_runs.push(len);
            } else {
                low_runs.push(len);
            }
            current_is_high = is_high;
            len = 1;
        }
    }
    if current_is_high {
        high_runs.push(len);
    } else {
        low_runs.push(len);
    }

    let high_mean = if high_runs.is_empty() {
        0.0
    } else {
        high_runs.iter().sum::<usize>() as f64 / high_runs.len() as f64
    };
    let low_mean = if low_runs.is_empty() {
        0.0
    } else {
        low_runs.iter().sum::<usize>() as f64 / low_runs.len() as f64
    };
    (high_mean, low_mean)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarize_values_basic_moments() {
        let s = summarize_observation_values(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(s.n, 4);
        assert!((s.mean - 2.5).abs() < 1e-12);
        assert!(s.variance > 0.0);
        assert!(s.q50 >= 2.0 && s.q50 <= 3.0);
    }

    #[test]
    fn summarize_values_empty_safe() {
        let s = summarize_observation_values(&[]);
        assert_eq!(s.n, 0);
        assert_eq!(s.mean, 0.0);
        assert_eq!(s.std_dev, 0.0);
    }

    #[test]
    fn sign_change_fraction_correct() {
        let x = vec![1.0, -1.0, 2.0, -3.0];
        let r = sign_change_fraction(&x);
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn episode_durations_nonzero() {
        let x = vec![0.1, 0.2, 5.0, 6.0, 0.1, 0.2];
        let (hi, lo) = episode_duration_means(&x, 1.0);
        assert!(hi > 0.0);
        assert!(lo > 0.0);
    }
}
