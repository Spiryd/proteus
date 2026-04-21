/// Leakage-safe feature normalization / scaling.
///
/// # The leakage problem
///
/// A scaler that is fitted on the *full* dataset (train + validation + test)
/// uses distributional information from the future to transform past
/// observations.  In the regime-detection context this means the detector
/// would be responding to an anomaly defined relative to a standard it
/// could not have known at deployment time.  This invalidates evaluation
/// results.
///
/// The correct procedure is:
///
/// 1. Fit scaler parameters on the **training partition only**.
/// 2. Apply the frozen scaler to all partitions (train, validation, test)
///    and to the live online stream.
/// 3. Never re-estimate the scaler on validation or test data.
///
/// # Supported scaling policies
///
/// | Policy | Formula | Use case |
/// |---|---|---|
/// | `None` | $y_t' = y_t$ | Raw features (recommended first) |
/// | `ZScore` | $y_t' = (y_t - \mu) / \sigma$ | Gaussian-shaped features |
/// | `RobustZScore` | $y_t' = (y_t - m) / s$ | Heavy-tailed or outlier-rich features |
///
/// where $\mu, \sigma$ are the training-set mean and standard deviation,
/// and $m, s$ are the training-set median and interquartile range (IQR).
///
/// # Degenerate case
///
/// If the training-set standard deviation (or IQR) is zero, no scaling is
/// applied (the scaler falls back to the identity).  This prevents division
/// by zero for pathological constant features.

// =========================================================================
// ScalingPolicy
// =========================================================================

/// The normalization policy to apply to a feature series.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalingPolicy {
    /// No normalization.  Feature values are used as-is.
    None,

    /// Subtract training mean and divide by training standard deviation.
    ///
    /// $y_t' = (y_t - \mu_{\text{train}}) / \sigma_{\text{train}}$
    ZScore,

    /// Subtract training median and divide by training IQR.
    ///
    /// $y_t' = (y_t - m_{\text{train}}) / s_{\text{train}}$
    ///
    /// where $s_{\text{train}} = Q_{75} - Q_{25}$ (interquartile range).
    RobustZScore,
}

// =========================================================================
// FittedScaler
// =========================================================================

/// A scaler fitted on training data and ready to transform any partition.
///
/// Construct via [`FittedScaler::fit`].  Apply via [`FittedScaler::transform`]
/// or [`FittedScaler::transform_value`].
///
/// The scaler is cheaply cloneable so it can be stored alongside the
/// `FeatureStream` for provenance.
#[derive(Debug, Clone)]
pub struct FittedScaler {
    /// The policy this scaler implements.
    pub policy: ScalingPolicy,
    /// Location parameter (mean or median). Zero for `None` policy.
    pub location: f64,
    /// Scale parameter (std or IQR). One for `None` policy or degenerate case.
    pub scale: f64,
}

impl FittedScaler {
    /// Fit a scaler from a slice of **training-set** feature values.
    ///
    /// # Panics
    ///
    /// Does not panic; degeneracy (empty or zero-scale) is handled gracefully
    /// by returning the identity transform.
    pub fn fit(values: &[f64], policy: ScalingPolicy) -> Self {
        match &policy {
            ScalingPolicy::None => Self {
                policy,
                location: 0.0,
                scale: 1.0,
            },

            ScalingPolicy::ZScore => {
                if values.is_empty() {
                    return Self {
                        policy,
                        location: 0.0,
                        scale: 1.0,
                    };
                }
                let n = values.len() as f64;
                let mean = values.iter().sum::<f64>() / n;
                let var = values.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
                let std = var.sqrt();
                let scale = if std > 0.0 { std } else { 1.0 };
                Self {
                    policy,
                    location: mean,
                    scale,
                }
            }

            ScalingPolicy::RobustZScore => {
                if values.is_empty() {
                    return Self {
                        policy,
                        location: 0.0,
                        scale: 1.0,
                    };
                }
                let mut sorted: Vec<f64> = values.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = percentile_sorted(&sorted, 0.50);
                let q25 = percentile_sorted(&sorted, 0.25);
                let q75 = percentile_sorted(&sorted, 0.75);
                let iqr = q75 - q25;
                let scale = if iqr > 0.0 { iqr } else { 1.0 };
                Self {
                    policy,
                    location: median,
                    scale,
                }
            }
        }
    }

    /// Apply the fitted transform to a single value.
    #[inline]
    pub fn transform_value(&self, v: f64) -> f64 {
        (v - self.location) / self.scale
    }

    /// Apply the fitted transform in-place to a mutable slice.
    pub fn transform_inplace(&self, values: &mut [f64]) {
        for v in values.iter_mut() {
            *v = self.transform_value(*v);
        }
    }

    /// Return a transformed copy of `values`.
    pub fn transform(&self, values: &[f64]) -> Vec<f64> {
        values.iter().map(|&v| self.transform_value(v)).collect()
    }

    /// Invert the transform: recover original value from scaled value.
    #[inline]
    pub fn inverse_transform_value(&self, scaled: f64) -> f64 {
        scaled * self.scale + self.location
    }
}

// =========================================================================
// Percentile helper (linear interpolation, sorted input)
// =========================================================================

/// Compute the `p`-th percentile of a *sorted* slice via linear interpolation.
///
/// `p` must be in [0, 1].  Returns `NaN` for an empty slice.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    let idx = p * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- None policy ----

    #[test]
    fn none_policy_is_identity() {
        let scaler = FittedScaler::fit(&[1.0, 2.0, 3.0], ScalingPolicy::None);
        assert_eq!(scaler.transform_value(5.0), 5.0);
        assert_eq!(scaler.transform_value(-3.14), -3.14);
    }

    // ---- ZScore ----

    #[test]
    fn zscore_mean_maps_to_zero() {
        let data = vec![2.0, 4.0, 6.0, 8.0]; // mean=5, std=sqrt(5)
        let scaler = FittedScaler::fit(&data, ScalingPolicy::ZScore);
        assert!((scaler.transform_value(5.0)).abs() < 1e-12);
    }

    #[test]
    fn zscore_scaled_values_have_unit_variance() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let scaler = FittedScaler::fit(&data, ScalingPolicy::ZScore);
        let scaled = scaler.transform(&data);
        let n = scaled.len() as f64;
        let mean_s = scaled.iter().sum::<f64>() / n;
        let var_s = scaled
            .iter()
            .map(|x| (x - mean_s) * (x - mean_s))
            .sum::<f64>()
            / n;
        assert!((var_s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn zscore_degenerate_constant_series() {
        // constant series → std=0 → identity transform
        let data = vec![3.0, 3.0, 3.0];
        let scaler = FittedScaler::fit(&data, ScalingPolicy::ZScore);
        assert_eq!(scaler.scale, 1.0);
        // transform(3.0) = (3 - 3) / 1 = 0
        assert_eq!(scaler.transform_value(3.0), 0.0);
    }

    #[test]
    fn zscore_empty_training_set() {
        let scaler = FittedScaler::fit(&[], ScalingPolicy::ZScore);
        assert_eq!(scaler.location, 0.0);
        assert_eq!(scaler.scale, 1.0);
    }

    // ---- RobustZScore ----

    #[test]
    fn robust_zscore_median_maps_to_zero() {
        // odd-length: median is middle element
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scaler = FittedScaler::fit(&data, ScalingPolicy::RobustZScore);
        assert!((scaler.transform_value(3.0)).abs() < 1e-12); // median=3
    }

    #[test]
    fn robust_zscore_iqr_correct() {
        // data [1,2,3,4,5]: Q25=2, Q75=4 → IQR=2
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scaler = FittedScaler::fit(&data, ScalingPolicy::RobustZScore);
        assert!((scaler.scale - 2.0).abs() < 1e-12);
    }

    // ---- Leakage invariant ----

    #[test]
    fn scaler_fitted_on_train_applied_to_test() {
        // Fitting on train=[1,2,3] and transforming test=[10] must use
        // train statistics, not test statistics.
        let train = vec![1.0, 2.0, 3.0];
        let scaler = FittedScaler::fit(&train, ScalingPolicy::ZScore);
        let test_transformed = scaler.transform_value(10.0);
        // Manually: mean_train=2, std_train=sqrt(2/3); scaled = (10-2)/sqrt(2/3)
        let mean = 2.0f64;
        let std = (2.0f64 / 3.0).sqrt();
        let expected = (10.0 - mean) / std;
        assert!((test_transformed - expected).abs() < 1e-10);
    }

    // ---- Inverse transform ----

    #[test]
    fn inverse_transform_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scaler = FittedScaler::fit(&data, ScalingPolicy::ZScore);
        for &v in &data {
            let scaled = scaler.transform_value(v);
            let recovered = scaler.inverse_transform_value(scaled);
            assert!((recovered - v).abs() < 1e-10);
        }
    }

    // ---- Percentile helper ----

    #[test]
    fn percentile_sorted_endpoints() {
        let d = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile_sorted(&d, 0.0), 1.0);
        assert_eq!(percentile_sorted(&d, 1.0), 5.0);
    }

    #[test]
    fn percentile_sorted_median_odd() {
        let d = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile_sorted(&d, 0.5), 3.0);
    }

    #[test]
    fn percentile_sorted_median_even_interpolated() {
        let d = vec![1.0, 2.0, 3.0, 4.0];
        // midpoint between index 1.5 → 2.0*0.5 + 3.0*0.5 = 2.5
        assert!((percentile_sorted(&d, 0.5) - 2.5).abs() < 1e-12);
    }
}
