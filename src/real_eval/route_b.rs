use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

use crate::data::Observation;

/// Policy for handling segments shorter than `min_len`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShortSegmentPolicy {
    FlagOnly,
    ExcludeFromGlobalStats,
}

/// Route B configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteBConfig {
    pub min_len: usize,
    pub short_segment_policy: ShortSegmentPolicy,
}

impl Default for RouteBConfig {
    fn default() -> Self {
        Self {
            min_len: 5,
            short_segment_policy: ShortSegmentPolicy::FlagOnly,
        }
    }
}

/// Detector-induced segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectedSegment {
    pub segment_index: usize,
    pub start_idx_1based: usize,
    pub end_idx_1based_inclusive: usize,
    pub start_ts: NaiveDateTime,
    pub end_ts: NaiveDateTime,
    pub duration: usize,
    pub values: Vec<f64>,
}

/// Descriptive statistics for one segment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentSummary {
    pub segment_index: usize,
    pub n: usize,
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub abs_mean: f64,
    pub q05: f64,
    pub q50: f64,
    pub q95: f64,
    pub acf1: f64,
    pub is_short: bool,
}

/// Contrast across one detected boundary between adjacent segments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdjacentSegmentContrast {
    pub boundary_alarm_t: usize,
    pub left_segment_index: usize,
    pub right_segment_index: usize,
    pub delta_mean: f64,
    pub delta_variance: f64,
    pub delta_abs_mean: f64,
    pub effect_size_mean: f64,
}

/// Global coherence summary for the detector-induced segmentation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentationGlobalSummary {
    pub n_segments: usize,
    pub n_short_segments: usize,
    pub mean_segment_len: f64,
    pub mean_within_segment_variance: f64,
    pub mean_adjacent_abs_delta_mean: f64,
    pub mean_adjacent_abs_delta_variance: f64,
    pub coherence_score: f64,
    pub warnings: Vec<String>,
}

/// Full Route B output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentationEvaluationResult {
    pub segments: Vec<DetectedSegment>,
    pub segment_summaries: Vec<SegmentSummary>,
    pub adjacent_contrasts: Vec<AdjacentSegmentContrast>,
    pub global: SegmentationGlobalSummary,
}

/// Build detector-induced segments and evaluate self-consistency metrics.
pub fn evaluate_segmentation(
    observations: &[Observation],
    alarm_indices_1based: &[usize],
    config: &RouteBConfig,
) -> SegmentationEvaluationResult {
    let segments = build_segments(observations, alarm_indices_1based);
    let segment_summaries: Vec<SegmentSummary> = segments
        .iter()
        .map(|s| summarize_segment(s, config.min_len))
        .collect();

    let mut alarm_clean = sanitize_alarm_indices(alarm_indices_1based, observations.len());
    alarm_clean.retain(|a| *a > 1 && *a <= observations.len());

    let adjacent_contrasts = build_adjacent_contrasts(&segment_summaries, &alarm_clean);
    let global = summarize_global(&segment_summaries, &adjacent_contrasts, config);

    SegmentationEvaluationResult {
        segments,
        segment_summaries,
        adjacent_contrasts,
        global,
    }
}

/// Convert alarms a_1 < ... < a_N into [1,a1), [a1,a2), ..., [aN,T].
pub fn build_segments(
    observations: &[Observation],
    alarm_indices_1based: &[usize],
) -> Vec<DetectedSegment> {
    if observations.is_empty() {
        return vec![];
    }

    let n = observations.len();
    let alarms = sanitize_alarm_indices(alarm_indices_1based, n);

    let mut starts = vec![1usize];
    for &a in &alarms {
        if a > 1 && a <= n {
            starts.push(a);
        }
    }
    starts.sort_unstable();
    starts.dedup();

    let mut segments = Vec::new();
    for (i, &s) in starts.iter().enumerate() {
        let e = if i + 1 < starts.len() {
            starts[i + 1].saturating_sub(1)
        } else {
            n
        };
        if s > e || s == 0 {
            continue;
        }

        let slice = &observations[(s - 1)..e];
        segments.push(DetectedSegment {
            segment_index: segments.len(),
            start_idx_1based: s,
            end_idx_1based_inclusive: e,
            start_ts: slice.first().unwrap().timestamp,
            end_ts: slice.last().unwrap().timestamp,
            duration: e - s + 1,
            values: slice.iter().map(|o| o.value).collect(),
        });
    }

    segments
}

fn summarize_segment(seg: &DetectedSegment, min_len: usize) -> SegmentSummary {
    let n = seg.values.len();
    let n_f = n as f64;
    let mean = if n == 0 {
        f64::NAN
    } else {
        seg.values.iter().sum::<f64>() / n_f
    };
    let variance = if n == 0 {
        f64::NAN
    } else {
        seg.values
            .iter()
            .map(|x| (x - mean) * (x - mean))
            .sum::<f64>()
            / n_f
    };
    let std_dev = variance.sqrt();
    let abs_mean = if n == 0 {
        f64::NAN
    } else {
        seg.values.iter().map(|x| x.abs()).sum::<f64>() / n_f
    };
    let mut sorted = seg.values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    SegmentSummary {
        segment_index: seg.segment_index,
        n,
        mean,
        variance,
        std_dev,
        abs_mean,
        q05: percentile_sorted(&sorted, 0.05),
        q50: percentile_sorted(&sorted, 0.50),
        q95: percentile_sorted(&sorted, 0.95),
        acf1: lag1_autocorr(&seg.values),
        is_short: n < min_len,
    }
}

fn build_adjacent_contrasts(
    summaries: &[SegmentSummary],
    alarm_indices_1based: &[usize],
) -> Vec<AdjacentSegmentContrast> {
    let mut out = Vec::new();
    for i in 0..summaries.len().saturating_sub(1) {
        let left = &summaries[i];
        let right = &summaries[i + 1];

        let pooled_std = f64::midpoint(left.variance, right.variance).sqrt();
        let effect_size_mean = if pooled_std > 0.0 {
            (right.mean - left.mean) / pooled_std
        } else {
            0.0
        };

        out.push(AdjacentSegmentContrast {
            boundary_alarm_t: alarm_indices_1based.get(i).copied().unwrap_or(0),
            left_segment_index: left.segment_index,
            right_segment_index: right.segment_index,
            delta_mean: right.mean - left.mean,
            delta_variance: right.variance - left.variance,
            delta_abs_mean: right.abs_mean - left.abs_mean,
            effect_size_mean,
        });
    }
    out
}

fn summarize_global(
    summaries: &[SegmentSummary],
    contrasts: &[AdjacentSegmentContrast],
    cfg: &RouteBConfig,
) -> SegmentationGlobalSummary {
    let n_segments = summaries.len();
    let n_short_segments = summaries.iter().filter(|s| s.is_short).count();

    let included: Vec<&SegmentSummary> = match cfg.short_segment_policy {
        ShortSegmentPolicy::FlagOnly => summaries.iter().collect(),
        ShortSegmentPolicy::ExcludeFromGlobalStats => {
            summaries.iter().filter(|s| !s.is_short).collect()
        }
    };

    let mean_segment_len = if summaries.is_empty() {
        f64::NAN
    } else {
        summaries.iter().map(|s| s.n).sum::<usize>() as f64 / summaries.len() as f64
    };

    let mean_within_segment_variance = if included.is_empty() {
        f64::NAN
    } else {
        included.iter().map(|s| s.variance).sum::<f64>() / included.len() as f64
    };

    let mean_adjacent_abs_delta_mean = if contrasts.is_empty() {
        f64::NAN
    } else {
        contrasts.iter().map(|c| c.delta_mean.abs()).sum::<f64>() / contrasts.len() as f64
    };

    let mean_adjacent_abs_delta_variance = if contrasts.is_empty() {
        f64::NAN
    } else {
        contrasts
            .iter()
            .map(|c| c.delta_variance.abs())
            .sum::<f64>()
            / contrasts.len() as f64
    };

    let coherence_score = if mean_within_segment_variance.is_finite()
        && mean_within_segment_variance > 0.0
        && mean_adjacent_abs_delta_mean.is_finite()
    {
        mean_adjacent_abs_delta_mean / mean_within_segment_variance.sqrt()
    } else {
        f64::NAN
    };

    let mut warnings = Vec::new();
    if n_short_segments > 0 {
        warnings.push(format!(
            "{} segments shorter than min_len={}",
            n_short_segments, cfg.min_len
        ));
    }

    SegmentationGlobalSummary {
        n_segments,
        n_short_segments,
        mean_segment_len,
        mean_within_segment_variance,
        mean_adjacent_abs_delta_mean,
        mean_adjacent_abs_delta_variance,
        coherence_score,
        warnings,
    }
}

fn sanitize_alarm_indices(alarm_indices_1based: &[usize], n: usize) -> Vec<usize> {
    let mut out: Vec<usize> = alarm_indices_1based
        .iter()
        .copied()
        .filter(|x| *x >= 1 && *x <= n)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
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
        return f64::NAN;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(i: usize, v: f64) -> Observation {
        Observation {
            timestamp: NaiveDateTime::parse_from_str(
                &format!("2024-01-{i:02} 00:00:00"),
                "%Y-%m-%d %H:%M:%S",
            )
            .unwrap(),
            value: v,
        }
    }

    fn series(vals: &[f64]) -> Vec<Observation> {
        vals.iter()
            .enumerate()
            .map(|(i, &v)| obs(i + 1, v))
            .collect()
    }

    #[test]
    fn segments_follow_half_open_rule() {
        let s = series(&[1.0, 1.0, 1.0, 10.0, 10.0]);
        let seg = build_segments(&s, &[4]);
        assert_eq!(seg.len(), 2);
        assert_eq!(seg[0].start_idx_1based, 1);
        assert_eq!(seg[0].end_idx_1based_inclusive, 3);
        assert_eq!(seg[1].start_idx_1based, 4);
        assert_eq!(seg[1].end_idx_1based_inclusive, 5);
    }

    #[test]
    fn segment_stats_capture_mean_shift() {
        let s = series(&[0.0, 0.1, -0.1, 3.0, 3.1, 2.9]);
        let out = evaluate_segmentation(
            &s,
            &[4],
            &RouteBConfig {
                min_len: 2,
                short_segment_policy: ShortSegmentPolicy::FlagOnly,
            },
        );
        assert_eq!(out.segment_summaries.len(), 2);
        assert!(out.segment_summaries[1].mean > out.segment_summaries[0].mean);
        assert_eq!(out.adjacent_contrasts.len(), 1);
        assert!(out.adjacent_contrasts[0].delta_mean > 0.0);
    }

    #[test]
    fn min_len_policy_flags_short_segments() {
        let s = series(&[1.0, 2.0, 3.0]);
        let out = evaluate_segmentation(
            &s,
            &[2, 3],
            &RouteBConfig {
                min_len: 2,
                short_segment_policy: ShortSegmentPolicy::FlagOnly,
            },
        );
        assert!(out.global.n_short_segments > 0);
        assert!(!out.global.warnings.is_empty());
    }

    #[test]
    fn empty_series_is_safe() {
        let out = evaluate_segmentation(&[], &[], &RouteBConfig::default());
        assert_eq!(out.segments.len(), 0);
        assert_eq!(out.global.n_segments, 0);
    }
}
