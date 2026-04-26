/// Parameters for a first-order Gaussian Markov Switching Model.
///
/// Regime indices are 0-based throughout (`0..k`).
/// The mathematics uses 1-based notation; this mapping is applied at the boundary only.
///
/// Mathematical objects:
/// - `k`          → K  (number of regimes)
/// - `pi`         → π  (initial distribution, length K)
/// - `transition` → P  (K×K row-stochastic transition matrix, stored row-major)
/// - `means`      → (μ₁,…,μ_K)
/// - `variances`  → (σ₁²,…,σ_K²)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelParams {
    /// K — number of regimes (≥ 2).
    pub k: usize,
    /// π — initial distribution over regimes 0..k. Must sum to 1.
    pub pi: Vec<f64>,
    /// P — transition matrix, stored as a flat Vec of length k*k, row-major.
    /// Row i: `transition[i*k .. (i+1)*k]`.
    pub transition: Vec<f64>,
    /// μⱼ  — regime-specific observation means.
    pub means: Vec<f64>,
    /// σⱼ² — regime-specific observation variances (strictly positive).
    pub variances: Vec<f64>,
}

const PROB_TOL: f64 = 1e-9;

impl ModelParams {
    /// Construct parameters from separate row vectors for the transition matrix.
    /// Rows are concatenated internally; `rows[i]` is row i of P.
    pub fn new(
        pi: Vec<f64>,
        transition_rows: Vec<Vec<f64>>,
        means: Vec<f64>,
        variances: Vec<f64>,
    ) -> Self {
        let k = pi.len();
        let transition = transition_rows.into_iter().flatten().collect();
        Self {
            k,
            pi,
            transition,
            means,
            variances,
        }
    }

    /// Validate that all mathematical constraints are satisfied.
    ///
    /// Checks (in order):
    /// 1. K ≥ 2
    /// 2. len(π) = K,  πⱼ ≥ 0,  Σπⱼ = 1
    /// 3. len(P) = K²,  Pᵢⱼ ≥ 0,  Σⱼ Pᵢⱼ = 1 for each row i
    /// 4. len(μ) = K,  len(σ²) = K,  σⱼ² > 0 for all j
    pub fn validate(&self) -> anyhow::Result<()> {
        // 1. Minimum regime count.
        if self.k < 2 {
            anyhow::bail!("ModelParams: k must be ≥ 2, got {}", self.k);
        }

        // 2. Initial distribution.
        if self.pi.len() != self.k {
            anyhow::bail!(
                "ModelParams: pi has length {}, expected k={}",
                self.pi.len(),
                self.k
            );
        }
        for (j, &p) in self.pi.iter().enumerate() {
            if p < 0.0 {
                anyhow::bail!("ModelParams: pi[{j}] = {p} is negative");
            }
        }
        let pi_sum: f64 = self.pi.iter().sum();
        if (pi_sum - 1.0).abs() > PROB_TOL {
            anyhow::bail!("ModelParams: pi sums to {pi_sum}, expected 1");
        }

        // 3. Transition matrix.
        let expected_len = self.k * self.k;
        if self.transition.len() != expected_len {
            anyhow::bail!(
                "ModelParams: transition has {} entries, expected k²={}",
                self.transition.len(),
                expected_len
            );
        }
        for i in 0..self.k {
            for j in 0..self.k {
                let p = self.transition[i * self.k + j];
                if p < 0.0 {
                    anyhow::bail!("ModelParams: P[{i},{j}] = {p} is negative");
                }
            }
            let row_sum: f64 = self.transition[i * self.k..(i + 1) * self.k].iter().sum();
            if (row_sum - 1.0).abs() > PROB_TOL {
                anyhow::bail!("ModelParams: transition row {i} sums to {row_sum}, expected 1");
            }
        }

        // 4. Emission parameters.
        if self.means.len() != self.k {
            anyhow::bail!(
                "ModelParams: means has length {}, expected k={}",
                self.means.len(),
                self.k
            );
        }
        if self.variances.len() != self.k {
            anyhow::bail!(
                "ModelParams: variances has length {}, expected k={}",
                self.variances.len(),
                self.k
            );
        }
        for (j, &v) in self.variances.iter().enumerate() {
            if v <= 0.0 {
                anyhow::bail!("ModelParams: variances[{j}] = {v} must be strictly positive");
            }
        }

        Ok(())
    }

    /// Return row i of the transition matrix as a slice.
    #[inline]
    pub fn transition_row(&self, i: usize) -> &[f64] {
        &self.transition[i * self.k..(i + 1) * self.k]
    }
}
