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
    /// K — number of regimes. Derived from `means.len()` at construction.
    pub k: usize,
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
        let k = means.len();
        Self { k, means, variances }
    }

    /// Validate all parameter constraints.
    ///
    /// Checks (in order):
    /// 1. `means` and `variances` have the same length K.
    /// 2. K ≥ 1 (emission requires at least one regime).
    /// 3. Every σⱼ² is strictly positive.
    ///
    /// Returns an error with a descriptive message on the first violation found.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.means.len() != self.variances.len() {
            anyhow::bail!(
                "Emission: means has {} entries but variances has {} — lengths must match",
                self.means.len(),
                self.variances.len()
            );
        }
        if self.k < 1 {
            anyhow::bail!("Emission: at least one regime is required");
        }
        for (j, &v) in self.variances.iter().enumerate() {
            if v <= 0.0 {
                anyhow::bail!(
                    "Emission: variance[{j}] = {v} — σⱼ² must be strictly positive"
                );
            }
        }
        Ok(())
    }

    /// Log-density of observation `y` under regime `j`.
    ///
    /// ```text
    /// log fⱼ(y) = -½ log(2π) - ½ log(σⱼ²) - (y - μⱼ)² / (2σⱼ²)
    /// ```
    ///
    /// **Prefer this over `density` in all numerical work.**
    /// Products of densities in the filter should be computed as sums of log-densities
    /// to avoid underflow.
    ///
    /// # Panics
    /// Panics if `j >= self.k`.
    pub fn log_density(&self, y: f64, j: usize) -> f64 {
        let mu = self.means[j];
        let var = self.variances[j];
        let z = (y - mu) / var.sqrt(); // standardized residual: (y - μⱼ) / σⱼ
        -HALF_LN_2PI - 0.5 * var.ln() - 0.5 * z * z
    }

    /// Density of observation `y` under regime `j`.
    ///
    /// ```text
    /// fⱼ(y) = exp( log fⱼ(y) )
    ///        = (2π σⱼ²)^{-½} exp( -(y - μⱼ)² / (2σⱼ²) )
    /// ```
    ///
    /// When computing the filter, use `log_density` and exponentiate
    /// only where necessary. This method is provided for tests and
    /// small-scale inspection.
    ///
    /// # Panics
    /// Panics if `j >= self.k`.
    pub fn density(&self, y: f64, j: usize) -> f64 {
        self.log_density(y, j).exp()
    }

    /// Log-density of observation `y` under **every** regime.
    ///
    /// Returns a `Vec<f64>` of length K where entry `j` equals `log fⱼ(y)`.
    ///
    /// This is the method the forward filter will call once per time step,
    /// treating the per-regime log-densities as likelihood contributions
    /// before combining them with the predicted state probabilities.
    pub fn log_density_vec(&self, y: f64) -> Vec<f64> {
        (0..self.k).map(|j| self.log_density(y, j)).collect()
    }

    /// Density of observation `y` under **every** regime.
    ///
    /// Returns a `Vec<f64>` of length K where entry `j` equals `fⱼ(y)`.
    pub fn density_vec(&self, y: f64) -> Vec<f64> {
        (0..self.k).map(|j| self.density(y, j)).collect()
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
    // Test 1 — Peak density at the mean is (2π σⱼ²)^{-½}
    // -----------------------------------------------------------------------
    #[test]
    fn test_peak_density_at_mean() {
        let e = unit_emission();

        // Regime 0: μ=0, σ²=1 → peak = 1/√(2π) ≈ 0.39894
        let expected0 = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!(
            (e.density(0.0, 0) - expected0).abs() < 1e-12,
            "peak density regime 0: got {:.6}, expected {:.6}",
            e.density(0.0, 0),
            expected0
        );

        // Regime 1: μ=5, σ²=4 → peak = 1/√(8π) ≈ 0.19947
        let expected1 = 1.0 / (8.0 * std::f64::consts::PI).sqrt();
        assert!(
            (e.density(5.0, 1) - expected1).abs() < 1e-12,
            "peak density regime 1: got {:.6}, expected {:.6}",
            e.density(5.0, 1),
            expected1
        );
    }

    // -----------------------------------------------------------------------
    // Test 2 — log_density is the natural log of density
    // -----------------------------------------------------------------------
    #[test]
    fn test_log_density_matches_ln_density() {
        let e = unit_emission();
        for &y in &[-5.0, -1.0, 0.0, 1.0, 3.0, 5.0, 10.0] {
            for j in 0..2 {
                let log_d = e.log_density(y, j);
                let ln_d = e.density(y, j).ln();
                assert!(
                    (log_d - ln_d).abs() < 1e-12,
                    "y={y}, j={j}: log_density={log_d:.10}, ln(density)={ln_d:.10}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test 3 — Density is symmetric around the mean
    // -----------------------------------------------------------------------
    #[test]
    fn test_density_symmetric_around_mean() {
        let e = unit_emission();
        for &delta in &[0.5, 1.0, 2.0, 3.0] {
            // Regime 0: μ=0, so f(delta) should equal f(-delta)
            let above = e.density(delta, 0);
            let below = e.density(-delta, 0);
            assert!(
                (above - below).abs() < 1e-14,
                "symmetry broken at delta={delta}: f(+delta)={above}, f(-delta)={below}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 4 — Density decreases monotonically away from the mean
    // -----------------------------------------------------------------------
    #[test]
    fn test_density_decreases_away_from_mean() {
        let e = unit_emission(); // regime 0: μ=0, σ²=1
        let points = [0.0_f64, 0.5, 1.0, 2.0, 3.0];
        for window in points.windows(2) {
            let (closer, farther) = (window[0], window[1]);
            assert!(
                e.density(closer, 0) > e.density(farther, 0),
                "density should decrease: f({closer})={} > f({farther})={}",
                e.density(closer, 0),
                e.density(farther, 0)
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 5 — Larger variance gives lower peak density (more diffuse)
    // -----------------------------------------------------------------------
    #[test]
    fn test_larger_variance_lower_peak() {
        let narrow = Emission::new(vec![0.0], vec![0.5]); // σ²=0.5
        let wide = Emission::new(vec![0.0], vec![4.0]);   // σ²=4

        assert!(
            narrow.density(0.0, 0) > wide.density(0.0, 0),
            "narrow σ² peak ({}) should exceed wide σ² peak ({})",
            narrow.density(0.0, 0),
            wide.density(0.0, 0)
        );
    }

    // -----------------------------------------------------------------------
    // Test 6 — density_vec / log_density_vec return K-length vectors
    // -----------------------------------------------------------------------
    #[test]
    fn test_vec_methods_length_and_consistency() {
        let e = unit_emission(); // K=2
        let ld_vec = e.log_density_vec(2.3);
        let d_vec = e.density_vec(2.3);

        assert_eq!(ld_vec.len(), 2);
        assert_eq!(d_vec.len(), 2);
        for j in 0..2 {
            assert!(
                (ld_vec[j] - e.log_density(2.3, j)).abs() < 1e-14,
                "log_density_vec[{j}] mismatch"
            );
            assert!(
                (d_vec[j] - e.density(2.3, j)).abs() < 1e-14,
                "density_vec[{j}] mismatch"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 7 — Explicit log-density formula check (regime j=0, y=1)
    // -----------------------------------------------------------------------
    #[test]
    fn test_log_density_formula_explicit() {
        // μ=0, σ²=1, y=1  →  log f = -½ ln(2π) - ½·ln(1) - ½·1² = -½ ln(2π) - 0.5
        let e = Emission::new(vec![0.0], vec![1.0]);
        let expected = -0.5 * LN_2PI - 0.5;
        let got = e.log_density(1.0, 0);
        assert!(
            (got - expected).abs() < 1e-12,
            "log_density(1.0, 0): got {got:.12}, expected {expected:.12}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 8 — Validate rejects non-positive variance
    // -----------------------------------------------------------------------
    #[test]
    fn test_validate_rejects_zero_variance() {
        let e = Emission::new(vec![0.0, 1.0], vec![1.0, 0.0]);
        assert!(e.validate().is_err(), "zero variance should fail validation");
    }

    #[test]
    fn test_validate_rejects_negative_variance() {
        let e = Emission::new(vec![0.0], vec![-1.0]);
        assert!(e.validate().is_err(), "negative variance should fail validation");
    }

    // -----------------------------------------------------------------------
    // Test 9 — Validate rejects mismatched means/variances lengths
    // -----------------------------------------------------------------------
    #[test]
    fn test_validate_rejects_length_mismatch() {
        let e = Emission::new(vec![0.0, 1.0], vec![1.0]); // 2 means, 1 variance
        assert!(e.validate().is_err(), "length mismatch should fail validation");
    }

    // -----------------------------------------------------------------------
    // Test 10 — Validate passes for a valid two-regime emission
    // -----------------------------------------------------------------------
    #[test]
    fn test_validate_passes_valid_emission() {
        let e = unit_emission();
        assert!(e.validate().is_ok());
    }
}
