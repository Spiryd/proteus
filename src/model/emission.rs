/// Regime-conditional Gaussian emission model for the Markov Switching Model.
///
/// # Mathematical specification
///
/// For each regime `j` in `0..k`, define the regime parameter
///
/// ```text
/// θⱼ = (μⱼ, σⱼ²),   σⱼ² > 0
/// ```
///
/// The regime-conditional observation law is
///
/// ```text
/// yₜ | (Sₜ = j)  ~  N(μⱼ, σⱼ²)
/// ```
///
/// with density
///
/// ```text
/// fⱼ(yₜ) = (2π σⱼ²)^{-½} exp( -(yₜ - μⱼ)² / (2σⱼ²) )
/// ```
///
/// and log-density (used in all numerical work)
///
/// ```text
/// log fⱼ(yₜ) = -½ log(2π) - ½ log(σⱼ²) - (yₜ - μⱼ)² / (2σⱼ²)
/// ```
///
/// # Design
///
/// `Emission` is intentionally **independent of the Markov chain**.
/// It does not know about the transition matrix or the initial distribution.
/// Its sole responsibility is answering:
///
/// > "Given observation `y` and regime `j`, what is the model-assigned
/// >  density (or log-density) of that observation under regime `j`?"
///
/// This boundary is the stable interface the forward filter will call in Phase 4.
///
/// # Extensibility note
///
/// If the observation law is later extended to switching regression or
/// switching autoregression, only this struct needs to change;
/// the hidden-state machinery remains untouched.
#[derive(Debug, Clone)]
pub struct Emission {
    /// μⱼ — regime-specific means, `j = 0..k`.  Unrestricted in ℝ.
    pub means: Vec<f64>,
    /// σⱼ² — regime-specific variances, `j = 0..k`.  Must be strictly positive.
    pub variances: Vec<f64>,
}

/// Precomputed constant: ½ ln(2π) ≈ 0.9189385332.
///
/// Used in every log-density evaluation. Isolating it here documents
/// exactly which form of the Gaussian log-density this project uses.
const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;

impl Emission {
    /// Construct an `Emission` from regime means and variances.
    ///
    /// K is inferred from `means.len()`. Lengths must match.
    pub fn new(means: Vec<f64>, variances: Vec<f64>) -> Self {
        Self { means, variances }
    }

    /// Log-density of observation `y` under regime `j`.
    ///
    /// ```text
    /// log fⱼ(y) = -½ log(2π) - ½ log(σⱼ²) - (y - μⱼ)² / (2σⱼ²)
    /// ```
    ///
    /// # Panics
    /// Panics if `j >= self.means.len()`.
    pub fn log_density(&self, y: f64, j: usize) -> f64 {
        let mu = self.means[j];
        let var = self.variances[j];
        let z = (y - mu) / var.sqrt(); // standardized residual: (y - μⱼ) / σⱼ
        -HALF_LN_2PI - 0.5 * var.ln() - 0.5 * z * z
    }
}

// ---------------------------------------------------------------------------
// Tests — density shape, numerical consistency, and parameter validation
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const LN_2PI: f64 = 1.837_877_066_409_345_6; // ln(2π)

    fn unit_emission() -> Emission {
        // Two regimes: μ₀=0, σ₀²=1  |  μ₁=5, σ₁²=4
        Emission::new(vec![0.0, 5.0], vec![1.0, 4.0])
    }

    // -----------------------------------------------------------------------
    // Test 1 — Peak log-density at the mean is -½ log(2π σⱼ²)
    // -----------------------------------------------------------------------
    #[test]
    fn test_peak_density_at_mean() {
        let e = unit_emission();

        let expected0 = -0.5 * LN_2PI; // σ²=1 → -½ ln(2π)
        assert!((e.log_density(0.0, 0) - expected0).abs() < 1e-12);

        let expected1 = -0.5 * LN_2PI - 0.5 * (4.0f64).ln(); // σ²=4 → -½ ln(2π·4)
        assert!((e.log_density(5.0, 1) - expected1).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Test 2 — log_density is symmetric around the mean
    // -----------------------------------------------------------------------
    #[test]
    fn test_density_symmetric_around_mean() {
        let e = unit_emission();
        for &delta in &[0.5, 1.0, 2.0, 3.0] {
            let above = e.log_density(delta, 0);
            let below = e.log_density(-delta, 0);
            assert!((above - below).abs() < 1e-14);
        }
    }

    // -----------------------------------------------------------------------
    // Test 3 — log_density decreases monotonically away from the mean
    // -----------------------------------------------------------------------
    #[test]
    fn test_density_decreases_away_from_mean() {
        let e = unit_emission();
        let points = [0.0_f64, 0.5, 1.0, 2.0, 3.0];
        for window in points.windows(2) {
            let (closer, farther) = (window[0], window[1]);
            assert!(e.log_density(closer, 0) > e.log_density(farther, 0));
        }
    }

    // -----------------------------------------------------------------------
    // Test 4 — Larger variance gives lower peak log-density (more diffuse)
    // -----------------------------------------------------------------------
    #[test]
    fn test_larger_variance_lower_peak() {
        let narrow = Emission::new(vec![0.0], vec![0.5]);
        let wide = Emission::new(vec![0.0], vec![4.0]);
        assert!(narrow.log_density(0.0, 0) > wide.log_density(0.0, 0));
    }

    // -----------------------------------------------------------------------
    // Test 5 — Explicit log-density formula check (regime j=0, y=1)
    // -----------------------------------------------------------------------
    #[test]
    fn test_log_density_formula_explicit() {
        // μ=0, σ²=1, y=1  →  log f = -½ ln(2π) - ½·ln(1) - ½·1² = -½ ln(2π) - 0.5
        let e = Emission::new(vec![0.0], vec![1.0]);
        let expected = -0.5 * LN_2PI - 0.5;
        let got = e.log_density(1.0, 0);
        assert!((got - expected).abs() < 1e-12);
    }
}
