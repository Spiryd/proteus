/// Calibration mapping: empirical summary functionals -> synthetic generator parameters.
///
/// This module implements the operator
///
///   K: (T_1^real, ..., T_m^real) -> theta
///
/// where `theta` contains the synthetic Markov-switching generator settings.
use serde::{Deserialize, Serialize};

use crate::model::ModelParams;

use super::summary::{EmpiricalCalibrationProfile, SummaryTargetSet};

/// How regime means are assigned from empirical summaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MeanPolicy {
    ZeroCentered,
    SymmetricAroundEmpirical,
    EmpiricalBaseline,
}

/// Top-level calibration strategy: which mapping operator to apply.
///
/// * [`Summary`] — the legacy heuristic operator K (mean / variance / duration
///   policies acting on empirical summary statistics).  Cheap; reproducible;
///   limited by the quality of the summary estimators.
/// * [`QuickEm`] — fit a short Baum-Welch EM directly on the real training
///   partition and use the resulting `ModelParams` as the synthetic generator.
///   The synthetic stream then is, by construction, a sample from the model
///   most likely to have produced the real training partition.  This is the
///   path used by the sim-to-real experiment family.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CalibrationStrategy {
    Summary,
    QuickEm {
        /// EM iteration budget (small — we are calibrating, not fitting).
        max_iter: usize,
        /// Convergence tolerance.
        tol: f64,
    },
}

impl Default for CalibrationStrategy {
    fn default() -> Self {
        Self::Summary
    }
}

/// How regime variances are anchored to empirical summaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VariancePolicy {
    /// Low/high variances from empirical low/high quantile scales.
    QuantileAnchored,
    /// Fixed ratio around empirical variance.
    RatioAroundEmpirical { low_mult: f64, high_mult: f64 },
    /// C′1 — median-split the raw observations on `|y|` and use the sample
    /// variances of each half as the low / high regime variances.  Requires
    /// `profile.observations` to be non-empty; falls back to
    /// `QuantileAnchored` (with a recorded mapping note) when it isn't.
    MagnitudeConditioned,
}

/// Optional jump contamination settings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JumpContamination {
    /// With probability `jump_prob`, add an outlier shock.
    pub jump_prob: f64,
    /// Shock std multiplier relative to sqrt(regime_variance).
    pub jump_scale_mult: f64,
}

/// Synthetic generator parameters produced by calibration.
#[derive(Debug, Clone)]
pub struct CalibratedSyntheticParams {
    pub model_params: ModelParams,
    pub horizon: usize,
    pub expected_durations: Vec<f64>,
    pub jump: Option<JumpContamination>,
    pub mapping_notes: Vec<String>,
}

/// Mapping configuration for phase-17 calibration.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct CalibrationMappingConfig {
    pub k: usize,
    pub horizon: usize,
    pub mean_policy: MeanPolicy,
    pub variance_policy: VariancePolicy,

    /// Target expected durations d_j used in p_jj = 1 - 1/d_j.
    /// If empty, defaults are inferred from empirical episode durations.
    pub target_durations: Vec<f64>,

    /// Off-diagonal row allocation style.
    pub symmetric_offdiag: bool,

    /// Optional jump contamination settings.
    pub jump: Option<JumpContamination>,

    /// Top-level calibration strategy.  Defaults to [`CalibrationStrategy::Summary`]
    /// for backwards compatibility.
    #[serde(default)]
    pub strategy: CalibrationStrategy,

    /// C′2 — minimum ratio `sigma_high / sigma_low` enforced under
    /// `VariancePolicy::QuantileAnchored` (and `MagnitudeConditioned`).
    /// When the empirically derived ratio is below this floor, the high
    /// variance is lifted (and the low variance is lowered symmetrically in
    /// log-space) until the guard is satisfied, and the action is recorded
    /// in `mapping_notes`.  Defaults to `1.0` (i.e. no-op).
    #[serde(default = "default_min_high_low_ratio")]
    pub min_high_low_ratio: f64,
}

fn default_min_high_low_ratio() -> f64 {
    1.0
}

impl Default for CalibrationMappingConfig {
    fn default() -> Self {
        Self {
            k: 2,
            horizon: 2_000,
            mean_policy: MeanPolicy::EmpiricalBaseline,
            variance_policy: VariancePolicy::QuantileAnchored,
            target_durations: vec![],
            symmetric_offdiag: true,
            jump: None,
            strategy: CalibrationStrategy::Summary,
            min_high_low_ratio: default_min_high_low_ratio(),
        }
    }
}

/// Build calibrated synthetic params from empirical calibration profile.
pub fn calibrate_to_synthetic(
    profile: &EmpiricalCalibrationProfile,
    config: &CalibrationMappingConfig,
) -> anyhow::Result<CalibratedSyntheticParams> {
    if config.k < 2 {
        anyhow::bail!("calibration mapping requires k>=2, got {}", config.k);
    }
    if config.horizon < 2 {
        anyhow::bail!("calibration horizon must be >=2, got {}", config.horizon);
    }

    match &config.strategy {
        CalibrationStrategy::Summary => calibrate_via_summary(profile, config),
        CalibrationStrategy::QuickEm { max_iter, tol } => {
            calibrate_via_quick_em(profile, config, *max_iter, *tol)
        }
    }
}

fn calibrate_via_summary(
    profile: &EmpiricalCalibrationProfile,
    config: &CalibrationMappingConfig,
) -> anyhow::Result<CalibratedSyntheticParams> {
    let mut notes = Vec::new();
    notes.push("strategy=Summary (heuristic operator K)".to_string());

    let means = map_means(profile, config);
    notes.push(format!("means mapped with policy {:?}", config.mean_policy));

    let (variances, variance_notes) = map_variances(profile, config)?;
    notes.extend(variance_notes);

    let durations = if config.target_durations.is_empty() {
        infer_durations(profile, config.k)
    } else {
        config.target_durations.clone()
    };
    if durations.len() != config.k {
        anyhow::bail!(
            "target duration length {} does not match k={}.",
            durations.len(),
            config.k
        );
    }

    let transition_rows = durations_to_transition_rows(&durations, config.symmetric_offdiag)?;
    notes.push("transition rows mapped via p_jj = 1 - 1/d_j".to_string());

    let pi = stationary_pi(&transition_rows);
    notes.push(
        "pi set to stationary distribution of transition matrix (power iteration)".to_string(),
    );
    let model_params = ModelParams::new(pi, transition_rows, means, variances);
    model_params.validate()?;

    if matches!(profile.targets, SummaryTargetSet::MarginalOnly) {
        notes.push("targets=MarginalOnly: persistence mapping treated as weak prior".to_string());
    }

    Ok(CalibratedSyntheticParams {
        model_params,
        horizon: config.horizon,
        expected_durations: durations,
        jump: config.jump.clone(),
        mapping_notes: notes,
    })
}

fn map_means(profile: &EmpiricalCalibrationProfile, config: &CalibrationMappingConfig) -> Vec<f64> {
    let m = profile.summary.mean;
    match config.mean_policy {
        MeanPolicy::ZeroCentered => {
            // Symmetric small separation around 0.
            if config.k == 2 {
                vec![
                    -0.25 * profile.summary.std_dev,
                    0.25 * profile.summary.std_dev,
                ]
            } else {
                centered_grid(config.k, 0.5 * profile.summary.std_dev)
            }
        }
        MeanPolicy::SymmetricAroundEmpirical => {
            if config.k == 2 {
                vec![
                    m - 0.25 * profile.summary.std_dev,
                    m + 0.25 * profile.summary.std_dev,
                ]
            } else {
                centered_grid(config.k, 0.5 * profile.summary.std_dev)
                    .into_iter()
                    .map(|x| x + m)
                    .collect()
            }
        }
        MeanPolicy::EmpiricalBaseline => vec![m; config.k],
    }
}

fn map_variances(
    profile: &EmpiricalCalibrationProfile,
    config: &CalibrationMappingConfig,
) -> anyhow::Result<(Vec<f64>, Vec<String>)> {
    let var = profile.summary.variance.max(1e-12);
    let std = var.sqrt();
    let mut notes = Vec::new();
    notes.push(format!(
        "variances mapped with policy {:?}",
        config.variance_policy
    ));
    let vals = match &config.variance_policy {
        VariancePolicy::RatioAroundEmpirical {
            low_mult,
            high_mult,
        } => {
            if *low_mult <= 0.0 || *high_mult <= 0.0 {
                anyhow::bail!("variance multipliers must be >0");
            }
            if config.k == 2 {
                vec![var * low_mult, var * high_mult]
            } else {
                (0..config.k)
                    .map(|i| {
                        let frac = i as f64 / (config.k - 1) as f64;
                        let mult = low_mult + frac * (high_mult - low_mult);
                        var * mult
                    })
                    .collect()
            }
        }
        VariancePolicy::QuantileAnchored => {
            // Convert quantile spread into low/high sigma anchors.
            let low_sigma = ((profile.summary.q50 - profile.summary.q05).abs() / 1.64485).max(1e-6);
            let high_sigma =
                ((profile.summary.q95 - profile.summary.q50).abs() / 1.64485).max(1e-6);
            if config.k == 2 {
                vec![low_sigma * low_sigma, high_sigma * high_sigma]
            } else {
                (0..config.k)
                    .map(|i| {
                        let frac = i as f64 / (config.k - 1) as f64;
                        let s = low_sigma + frac * (high_sigma - low_sigma);
                        s * s
                    })
                    .collect()
            }
        }
        VariancePolicy::MagnitudeConditioned => {
            // C′1 — median-split |y| on the raw observations.
            if profile.observations.is_empty() {
                notes.push(
                    "MagnitudeConditioned fallback: observations empty, using QuantileAnchored"
                        .to_string(),
                );
                let low_sigma =
                    ((profile.summary.q50 - profile.summary.q05).abs() / 1.64485).max(1e-6);
                let high_sigma =
                    ((profile.summary.q95 - profile.summary.q50).abs() / 1.64485).max(1e-6);
                if config.k == 2 {
                    vec![low_sigma * low_sigma, high_sigma * high_sigma]
                } else {
                    (0..config.k)
                        .map(|i| {
                            let frac = i as f64 / (config.k - 1) as f64;
                            let s = low_sigma + frac * (high_sigma - low_sigma);
                            s * s
                        })
                        .collect()
                }
            } else {
                let mut abs_sorted: Vec<f64> =
                    profile.observations.iter().map(|x| x.abs()).collect();
                abs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = abs_sorted[abs_sorted.len() / 2];

                let (low_obs, high_obs): (Vec<f64>, Vec<f64>) = profile
                    .observations
                    .iter()
                    .copied()
                    .partition(|x| x.abs() <= median);

                let sample_var = |slice: &[f64]| -> f64 {
                    if slice.is_empty() {
                        return var;
                    }
                    let m = slice.iter().sum::<f64>() / slice.len() as f64;
                    let v = slice.iter().map(|x| (x - m).powi(2)).sum::<f64>() / slice.len() as f64;
                    v.max(1e-12)
                };

                let v_low = sample_var(&low_obs);
                let v_high = sample_var(&high_obs);
                notes.push(format!(
                    "MagnitudeConditioned: median(|y|)={median:.6}, low_n={}, high_n={}",
                    low_obs.len(),
                    high_obs.len()
                ));
                if config.k == 2 {
                    vec![v_low, v_high]
                } else {
                    // K>2: log-linear interpolation between the two extremes.
                    let s_low = v_low.sqrt();
                    let s_high = v_high.sqrt();
                    (0..config.k)
                        .map(|i| {
                            let frac = i as f64 / (config.k - 1) as f64;
                            let s = s_low + frac * (s_high - s_low);
                            s * s
                        })
                        .collect()
                }
            }
        }
    };

    if vals.iter().any(|v| *v <= 0.0 || !v.is_finite()) {
        anyhow::bail!("mapped variances must be finite and positive");
    }
    // Keep regime ordering from calm->turbulent.
    let mut sorted = vals;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // If nearly flat, spread slightly to avoid duplicate regimes.
    if sorted.first() == sorted.last() {
        let base = std.max(1e-6);
        sorted = (0..config.k)
            .map(|i| {
                let m = 1.0 + 0.1 * i as f64;
                (base * m).powi(2)
            })
            .collect();
    }

    // C′2 — enforce minimum sigma_high / sigma_low ratio for quantile-derived
    // policies (QuantileAnchored and MagnitudeConditioned).  We rescale the
    // sorted variances log-symmetrically around the geometric mean so the
    // empirical centre of mass is preserved.
    let guarded_policy = matches!(
        config.variance_policy,
        VariancePolicy::QuantileAnchored | VariancePolicy::MagnitudeConditioned
    );
    if guarded_policy && config.min_high_low_ratio > 1.0 && sorted.len() >= 2 {
        let s_low = sorted.first().copied().unwrap_or(1.0).sqrt();
        let s_high = sorted.last().copied().unwrap_or(1.0).sqrt();
        let ratio = (s_high / s_low).max(1.0);
        if ratio < config.min_high_low_ratio {
            // Need to inflate the spread.  Scale low by alpha^-1 and high by
            // alpha so that the new ratio matches min_high_low_ratio.
            let alpha = (config.min_high_low_ratio / ratio).sqrt();
            for (i, v) in sorted.iter_mut().enumerate() {
                let frac = if sorted_len_minus_one(config.k) == 0 {
                    0.0
                } else {
                    i as f64 / (config.k - 1) as f64
                };
                // Scale exponent ranges linearly from -1 (lowest) to +1 (highest).
                let exp = 2.0 * frac - 1.0;
                let factor = alpha.powf(exp);
                *v *= factor * factor; // applied to variance => factor^2
            }
            notes.push(format!(
                "C′2: lifted sigma_high/sigma_low from {:.3} to {:.3} via log-symmetric rescaling",
                ratio, config.min_high_low_ratio
            ));
        }
    }
    Ok((sorted, notes))
}

fn sorted_len_minus_one(k: usize) -> usize {
    k.saturating_sub(1)
}

fn infer_durations(profile: &EmpiricalCalibrationProfile, k: usize) -> Vec<f64> {
    let low = profile.summary.low_episode_mean_duration.max(2.0);
    let high = profile.summary.high_episode_mean_duration.max(2.0);
    if k == 2 {
        vec![low, high]
    } else {
        (0..k)
            .map(|i| {
                let frac = i as f64 / (k - 1) as f64;
                low + frac * (high - low)
            })
            .collect()
    }
}

/// Map target expected durations d_j to transition probabilities using
/// p_jj = 1 - 1/d_j and symmetric off-diagonal allocation.
fn durations_to_transition_rows(
    durations: &[f64],
    symmetric_offdiag: bool,
) -> anyhow::Result<Vec<Vec<f64>>> {
    let k = durations.len();
    if k < 2 {
        anyhow::bail!("durations must have len>=2");
    }

    let mut rows = vec![vec![0.0; k]; k];
    for i in 0..k {
        let d = durations[i];
        if d <= 1.0 || !d.is_finite() {
            anyhow::bail!("duration d[{i}] must be >1 and finite, got {d}");
        }
        let p_ii = (1.0 - 1.0 / d).clamp(1e-6, 1.0 - 1e-6);
        rows[i][i] = p_ii;
        let rem = 1.0 - p_ii;
        if symmetric_offdiag {
            let each = rem / (k - 1) as f64;
            for (j, cell) in rows[i].iter_mut().enumerate() {
                if j != i {
                    *cell = each;
                }
            }
        } else {
            // Simple deterministic next-regime bias.
            let next = (i + 1) % k;
            rows[i][next] = rem;
        }
    }
    Ok(rows)
}

fn centered_grid(k: usize, scale: f64) -> Vec<f64> {
    if k == 1 {
        return vec![0.0];
    }
    let mid = (k - 1) as f64 / 2.0;
    (0..k).map(|i| (i as f64 - mid) * scale).collect()
}

/// E1 — Stationary distribution π from a transition matrix via power
/// iteration.  Falls back to the uniform distribution when iteration fails
/// to converge or produces an invalid result.  The returned vector is
/// length-K, non-negative, and sums to 1.
pub(crate) fn stationary_pi(transition: &[Vec<f64>]) -> Vec<f64> {
    let k = transition.len();
    let uniform = vec![1.0 / k as f64; k];
    if k == 0 {
        return uniform;
    }
    if !transition.iter().all(|row| row.len() == k) {
        return uniform;
    }
    let mut pi = uniform.clone();
    let max_iter = 256;
    let tol = 1e-10;
    for _ in 0..max_iter {
        let mut next = vec![0.0; k];
        for i in 0..k {
            for j in 0..k {
                next[j] += pi[i] * transition[i][j];
            }
        }
        let sum: f64 = next.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return uniform;
        }
        for v in &mut next {
            *v /= sum;
        }
        let diff: f64 = next.iter().zip(pi.iter()).map(|(a, b)| (a - b).abs()).sum();
        pi = next;
        if diff < tol {
            break;
        }
    }
    if pi.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return uniform;
    }
    pi
}

// -------------------------------------------------------------------------
// Quick-EM calibration (C′3)
// -------------------------------------------------------------------------

/// Build a quantile-split initial `ModelParams` from raw observations.
///
/// Mirrors `experiments::shared::init_params_from_obs` but keeps the
/// calibration module independent of the experiments module.
fn quantile_init_params(obs: &[f64], k: usize) -> anyhow::Result<ModelParams> {
    if obs.is_empty() || k < 2 {
        anyhow::bail!("quantile_init_params: need at least 1 observation and k >= 2");
    }
    let mut sorted = obs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let chunk = (n / k).max(1);
    let mut means = Vec::with_capacity(k);
    let mut variances = Vec::with_capacity(k);
    for j in 0..k {
        let start = j * chunk;
        let end = if j == k - 1 { n } else { (j + 1) * chunk };
        let slice = &sorted[start..end];
        let m = slice.iter().sum::<f64>() / slice.len() as f64;
        let v = slice.iter().map(|x| (x - m).powi(2)).sum::<f64>() / slice.len() as f64;
        means.push(m);
        variances.push(v.max(1e-6));
    }
    let pi = vec![1.0 / k as f64; k];
    let off_diag = 0.1 / (k - 1) as f64;
    let transition_rows: Vec<Vec<f64>> = (0..k)
        .map(|i| {
            (0..k)
                .map(|j| if i == j { 0.9 } else { off_diag })
                .collect()
        })
        .collect();
    Ok(ModelParams::new(pi, transition_rows, means, variances))
}

/// Calibrate the synthetic generator by fitting a short Baum-Welch EM
/// directly on the real training observations.
///
/// The resulting `ModelParams` is used as the synthetic generator — so the
/// synthetic stream is, by construction, a sample from the model best fit to
/// the real data.  This sidesteps the heuristic duration / variance
/// estimators entirely.
fn calibrate_via_quick_em(
    profile: &EmpiricalCalibrationProfile,
    config: &CalibrationMappingConfig,
    max_iter: usize,
    tol: f64,
) -> anyhow::Result<CalibratedSyntheticParams> {
    let mut notes = Vec::new();
    notes.push(format!(
        "strategy=QuickEm (max_iter={}, tol={:.1e})",
        max_iter, tol
    ));

    if profile.observations.is_empty() {
        anyhow::bail!(
            "CalibrationStrategy::QuickEm requires profile.observations to be \
             non-empty; the legacy summary-only path is not supported by Quick-EM"
        );
    }

    let init = quantile_init_params(&profile.observations, config.k)?;
    let em_cfg = crate::model::em::EmConfig {
        tol,
        max_iter,
        ..Default::default()
    };
    let em_result = crate::model::em::fit_em(&profile.observations, init, &em_cfg)?;
    notes.push(format!(
        "EM converged={} after {} iters; ll={:.4}",
        em_result.converged, em_result.n_iter, em_result.log_likelihood
    ));

    // Sort regimes calm→turbulent by variance so downstream consumers can rely
    // on the ordering convention used by the summary path.
    let fitted = em_result.params;
    let order = {
        let mut idx: Vec<usize> = (0..fitted.k).collect();
        idx.sort_by(|&a, &b| {
            fitted.variances[a]
                .partial_cmp(&fitted.variances[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx
    };
    let means: Vec<f64> = order.iter().map(|&i| fitted.means[i]).collect();
    let variances: Vec<f64> = order.iter().map(|&i| fitted.variances[i]).collect();
    let pi: Vec<f64> = order.iter().map(|&i| fitted.pi[i]).collect();
    let k = fitted.k;
    let transition_rows: Vec<Vec<f64>> = (0..k)
        .map(|new_i| {
            let i = order[new_i];
            (0..k)
                .map(|new_j| fitted.transition[i * k + order[new_j]])
                .collect()
        })
        .collect();
    let model_params = ModelParams::new(pi, transition_rows.clone(), means, variances);
    model_params.validate()?;

    // Expected dwell durations from the diagonal of P: d_j = 1 / (1 - p_jj).
    let expected_durations: Vec<f64> = (0..k)
        .map(|i| {
            let p_ii = transition_rows[i][i].clamp(1e-6, 1.0 - 1e-6);
            1.0 / (1.0 - p_ii)
        })
        .collect();

    Ok(CalibratedSyntheticParams {
        model_params,
        horizon: config.horizon,
        expected_durations,
        jump: config.jump.clone(),
        mapping_notes: notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::summary::{
        CalibrationDatasetTag, CalibrationPartition, EmpiricalSummary, SummaryTargetSet,
    };
    use crate::features::FeatureFamily;

    #[test]
    fn stationary_pi_matches_known_two_state() {
        // For [[0.9, 0.1], [0.2, 0.8]] the stationary distribution is
        // pi_0 = q/(p+q), pi_1 = p/(p+q) with p=0.1, q=0.2 -> (2/3, 1/3).
        let p = stationary_pi(&[vec![0.9, 0.1], vec![0.2, 0.8]]);
        assert!((p[0] - 2.0 / 3.0).abs() < 1e-6, "got {:?}", p);
        assert!((p[1] - 1.0 / 3.0).abs() < 1e-6, "got {:?}", p);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn stationary_pi_falls_back_to_uniform_on_bad_input() {
        // Non-square matrix -> uniform.
        let p = stationary_pi(&[vec![0.5, 0.5], vec![0.1]]);
        assert!((p[0] - 0.5).abs() < 1e-12);
        assert!((p[1] - 0.5).abs() < 1e-12);
    }

    fn profile() -> EmpiricalCalibrationProfile {
        EmpiricalCalibrationProfile {
            tag: CalibrationDatasetTag {
                asset: "SPY".to_string(),
                frequency: "daily".to_string(),
                feature_label: "log_return".to_string(),
                partition: CalibrationPartition::TrainOnly,
            },
            feature_family: FeatureFamily::LogReturn,
            targets: SummaryTargetSet::Full,
            summary: EmpiricalSummary {
                n: 1000,
                mean: 0.001,
                variance: 0.0004,
                std_dev: 0.02,
                q01: -0.05,
                q05: -0.03,
                q50: 0.0,
                q95: 0.03,
                q99: 0.05,
                tail_freq_q95: 0.05,
                abs_exceed_2std: 0.04,
                acf1: 0.1,
                abs_acf1: 0.2,
                sign_change_rate: 0.45,
                high_episode_mean_duration: 6.0,
                low_episode_mean_duration: 20.0,
            },
            observations: Vec::new(),
        }
    }

    #[test]
    fn duration_mapping_formula_matches() {
        let rows = durations_to_transition_rows(&[10.0, 5.0], true).unwrap();
        assert!((rows[0][0] - 0.9).abs() < 1e-12);
        assert!((rows[1][1] - 0.8).abs() < 1e-12);
        assert!((rows[0][1] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn calibrate_outputs_valid_model_params() {
        let cfg = CalibrationMappingConfig::default();
        let out = calibrate_to_synthetic(&profile(), &cfg).unwrap();
        out.model_params.validate().unwrap();
        assert_eq!(out.model_params.k, 2);
        assert_eq!(out.horizon, 2000);
    }

    #[test]
    fn calibrate_respects_target_duration() {
        let cfg = CalibrationMappingConfig {
            target_durations: vec![30.0, 5.0],
            ..CalibrationMappingConfig::default()
        };
        let out = calibrate_to_synthetic(&profile(), &cfg).unwrap();
        assert_eq!(out.expected_durations, vec![30.0, 5.0]);
        let p00 = out.model_params.transition_row(0)[0];
        assert!((p00 - (1.0 - 1.0 / 30.0)).abs() < 1e-10);
    }

    #[test]
    fn quantile_variance_mapping_positive() {
        let (vars, _notes) =
            map_variances(&profile(), &CalibrationMappingConfig::default()).unwrap();
        assert!(vars.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn quick_em_strategy_requires_observations() {
        let cfg = CalibrationMappingConfig {
            strategy: CalibrationStrategy::QuickEm {
                max_iter: 30,
                tol: 1e-5,
            },
            ..CalibrationMappingConfig::default()
        };
        let err = calibrate_to_synthetic(&profile(), &cfg).unwrap_err();
        assert!(err.to_string().contains("observations"));
    }

    #[test]
    fn quick_em_recovers_two_regime_structure() {
        // Generate a deterministic toy 2-regime sequence: low-var around 0,
        // high-var around 0, with long dwell times.
        let mut obs = Vec::new();
        // Pseudo-random but deterministic using LCG.
        let mut state: u64 = 0xDEAD_BEEF;
        let mut next_u01 = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / ((1u64 << 31) as f64)
        };
        // Box-Muller helper.
        let mut next_normal = || {
            let u1 = next_u01().max(1e-12);
            let u2 = next_u01();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        // 800 samples low-var, 800 high-var, alternating in blocks of 200.
        let block = 200;
        for blk in 0..8 {
            let high = blk % 2 == 1;
            let sigma = if high { 1.5 } else { 0.3 };
            for _ in 0..block {
                obs.push(sigma * next_normal());
            }
        }
        let mut prof = profile();
        prof.observations = obs;
        let cfg = CalibrationMappingConfig {
            strategy: CalibrationStrategy::QuickEm {
                max_iter: 100,
                tol: 1e-6,
            },
            ..CalibrationMappingConfig::default()
        };
        let out = calibrate_to_synthetic(&prof, &cfg).unwrap();
        out.model_params.validate().unwrap();
        // Calm regime variance should be much smaller than turbulent.
        let v_lo = out.model_params.variances[0];
        let v_hi = out.model_params.variances[1];
        assert!(
            v_hi / v_lo > 3.0,
            "expected separation v_hi/v_lo > 3, got {} / {} = {}",
            v_hi,
            v_lo,
            v_hi / v_lo
        );
        // Expected durations should be > 1 (high persistence).
        assert!(out.expected_durations.iter().all(|d| *d > 1.5));
    }

    fn profile_with_obs(obs: Vec<f64>) -> EmpiricalCalibrationProfile {
        let mut p = profile();
        p.observations = obs;
        p
    }

    #[test]
    fn magnitude_conditioned_splits_on_abs_y() {
        // Two-mode mixture: half small ~ N(0, 0.1^2), half large ~ N(0, 1.0^2).
        let mut obs = Vec::new();
        let mut state: u64 = 0xC0FF_EE42;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as f64 / (u32::MAX as f64) - 0.5
        };
        for _ in 0..500 {
            obs.push(0.1 * next());
        }
        for _ in 0..500 {
            obs.push(1.0 * next());
        }
        let cfg = CalibrationMappingConfig {
            variance_policy: VariancePolicy::MagnitudeConditioned,
            ..CalibrationMappingConfig::default()
        };
        let (vars, notes) = map_variances(&profile_with_obs(obs), &cfg).unwrap();
        assert_eq!(vars.len(), 2);
        assert!(vars[1] > 10.0 * vars[0], "vars = {:?}", vars);
        assert!(notes.iter().any(|n| n.contains("MagnitudeConditioned")));
    }

    #[test]
    fn magnitude_conditioned_falls_back_when_no_observations() {
        let cfg = CalibrationMappingConfig {
            variance_policy: VariancePolicy::MagnitudeConditioned,
            ..CalibrationMappingConfig::default()
        };
        let (vars, notes) = map_variances(&profile(), &cfg).unwrap();
        assert_eq!(vars.len(), 2);
        assert!(notes.iter().any(|n| n.contains("fallback")));
    }

    #[test]
    fn min_high_low_ratio_lifts_collapsed_variances() {
        // Build a profile whose q05/q50/q95 are very close (small natural ratio).
        let mut p = profile();
        p.summary.q05 = -0.01;
        p.summary.q50 = 0.0;
        p.summary.q95 = 0.011; // ratio ~1.1
        let cfg = CalibrationMappingConfig {
            variance_policy: VariancePolicy::QuantileAnchored,
            min_high_low_ratio: 3.0,
            ..CalibrationMappingConfig::default()
        };
        let (vars, notes) = map_variances(&p, &cfg).unwrap();
        let ratio = vars[1].sqrt() / vars[0].sqrt();
        assert!((ratio - 3.0).abs() < 1e-6, "ratio after guard = {ratio}");
        assert!(
            notes
                .iter()
                .any(|n| n.contains("C′2") || n.contains("lifted"))
        );
    }

    #[test]
    fn min_high_low_ratio_noop_when_already_above() {
        // Profile with naturally wide q95 vs q05 ratio so we are above 1.5×.
        let mut p = profile();
        p.summary.q05 = -0.01;
        p.summary.q95 = 0.05; // s_high/s_low ≈ 5
        let cfg = CalibrationMappingConfig {
            variance_policy: VariancePolicy::QuantileAnchored,
            min_high_low_ratio: 1.5,
            ..CalibrationMappingConfig::default()
        };
        let (_vars, notes) = map_variances(&p, &cfg).unwrap();
        assert!(!notes.iter().any(|n| n.contains("lifted")));
    }
}
