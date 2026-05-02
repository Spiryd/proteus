use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};

/// Point or window event anchor used as external reference for Route A.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProxyEventAnchor {
    Point {
        at: NaiveDateTime,
    },
    Window {
        start: NaiveDateTime,
        end: NaiveDateTime,
    },
}

/// Proxy event definition. These are evaluation anchors, not ground truth labels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProxyEvent {
    pub id: String,
    pub event_type: String,
    pub label: String,
    pub asset_scope: Option<String>,
    pub anchor: ProxyEventAnchor,
}

impl ProxyEvent {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.id.trim().is_empty() {
            anyhow::bail!("proxy event id must be non-empty");
        }
        if self.event_type.trim().is_empty() {
            anyhow::bail!("proxy event type must be non-empty");
        }
        if let ProxyEventAnchor::Window { start, end } = self.anchor
            && start > end
        {
            anyhow::bail!("proxy event window start must be <= end");
        }
        Ok(())
    }
}

/// Matching policy for point events.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PointMatchPolicy {
    /// Number of bars allowed before the event anchor.
    pub pre_bars: usize,
    /// Number of bars allowed after the event anchor.
    pub post_bars: usize,
    /// If true, ignore pre-event tolerance and require alarm index >= event index.
    pub causal_only: bool,
}

impl Default for PointMatchPolicy {
    fn default() -> Self {
        Self {
            pre_bars: 3,
            post_bars: 3,
            causal_only: false,
        }
    }
}

/// Route A evaluation configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RouteAConfig {
    pub point_policy: PointMatchPolicy,
}

/// Per-event alignment details.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EventAlignment {
    pub event_id: String,
    pub matched_alarm_indices: Vec<usize>,
    pub first_alarm_index: Option<usize>,
    /// Delay in bars: first_alarm_index - event_reference_index.
    pub first_delay_bars: Option<isize>,
}

/// Aggregate Route A metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProxyEventEvaluationResult {
    pub n_events: usize,
    pub n_alarms: usize,
    pub n_events_covered: usize,
    pub n_alarms_aligned: usize,

    /// Events with >=1 aligned alarm divided by total events.
    pub event_coverage: f64,
    /// Alarms aligned to any event divided by total alarms.
    pub alarm_relevance: f64,

    /// Delay summary for first aligned alarms (bars).
    pub mean_delay_bars: f64,
    pub median_delay_bars: f64,
    pub min_delay_bars: Option<isize>,
    pub max_delay_bars: Option<isize>,

    /// Alarms per bar inside event windows.
    pub alarm_density_in_event_windows: f64,
    /// Alarms per bar outside event windows.
    pub alarm_density_outside_event_windows: f64,

    pub alignments: Vec<EventAlignment>,
}

/// Evaluate detector alarms against proxy events (Route A).
///
/// `alarm_indices_1based` uses the same 1-based convention as detector output.
pub fn evaluate_proxy_events(
    timestamps: &[NaiveDateTime],
    alarm_indices_1based: &[usize],
    events: &[ProxyEvent],
    config: &RouteAConfig,
) -> anyhow::Result<ProxyEventEvaluationResult> {
    for e in events {
        e.validate()?;
    }

    let n = timestamps.len();
    let alarms = sanitize_alarm_indices(alarm_indices_1based, n);

    let mut alignments = Vec::with_capacity(events.len());
    let mut aligned_alarm_set = std::collections::BTreeSet::new();
    let mut delay_values = Vec::new();
    let mut window_mask = vec![false; n];

    for e in events {
        let (l, u, reference_idx) = event_window_indices(timestamps, e, &config.point_policy)?;
        if let Some((ll, uu)) = l.zip(u) {
            for mask in &mut window_mask[ll..=uu] {
                *mask = true;
            }
            let matched: Vec<usize> = alarms
                .iter()
                .copied()
                .filter(|a| {
                    let idx = *a - 1;
                    idx >= ll && idx <= uu
                })
                .collect();

            for a in &matched {
                aligned_alarm_set.insert(*a);
            }

            let first_alarm_index = matched.first().copied();
            let first_delay_bars = match (first_alarm_index, reference_idx) {
                (Some(a), Some(r)) => {
                    let d = a as isize - r as isize;
                    delay_values.push(d);
                    Some(d)
                }
                _ => None,
            };

            alignments.push(EventAlignment {
                event_id: e.id.clone(),
                matched_alarm_indices: matched,
                first_alarm_index,
                first_delay_bars,
            });
        } else {
            alignments.push(EventAlignment {
                event_id: e.id.clone(),
                matched_alarm_indices: vec![],
                first_alarm_index: None,
                first_delay_bars: None,
            });
        }
    }

    let n_events = events.len();
    let n_alarms = alarms.len();
    let n_events_covered = alignments
        .iter()
        .filter(|x| !x.matched_alarm_indices.is_empty())
        .count();
    let n_alarms_aligned = aligned_alarm_set.len();

    let event_coverage = if n_events == 0 {
        f64::NAN
    } else {
        n_events_covered as f64 / n_events as f64
    };
    let alarm_relevance = if n_alarms == 0 {
        f64::NAN
    } else {
        n_alarms_aligned as f64 / n_alarms as f64
    };

    delay_values.sort_unstable();
    let mean_delay_bars = if delay_values.is_empty() {
        f64::NAN
    } else {
        delay_values.iter().sum::<isize>() as f64 / delay_values.len() as f64
    };
    let median_delay_bars = if delay_values.is_empty() {
        f64::NAN
    } else if delay_values.len() % 2 == 1 {
        delay_values[delay_values.len() / 2] as f64
    } else {
        let i = delay_values.len() / 2;
        f64::midpoint(delay_values[i - 1] as f64, delay_values[i] as f64)
    };

    let in_count = window_mask.iter().filter(|&&x| x).count();
    let out_count = n.saturating_sub(in_count);
    let in_alarms = alarms.iter().filter(|&&a| window_mask[a - 1]).count();
    let out_alarms = alarms.len().saturating_sub(in_alarms);

    let alarm_density_in_event_windows = if in_count == 0 {
        f64::NAN
    } else {
        in_alarms as f64 / in_count as f64
    };
    let alarm_density_outside_event_windows = if out_count == 0 {
        f64::NAN
    } else {
        out_alarms as f64 / out_count as f64
    };

    Ok(ProxyEventEvaluationResult {
        n_events,
        n_alarms,
        n_events_covered,
        n_alarms_aligned,
        event_coverage,
        alarm_relevance,
        mean_delay_bars,
        median_delay_bars,
        min_delay_bars: delay_values.first().copied(),
        max_delay_bars: delay_values.last().copied(),
        alarm_density_in_event_windows,
        alarm_density_outside_event_windows,
        alignments,
    })
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

fn event_window_indices(
    timestamps: &[NaiveDateTime],
    event: &ProxyEvent,
    point_policy: &PointMatchPolicy,
) -> anyhow::Result<(Option<usize>, Option<usize>, Option<usize>)> {
    match event.anchor {
        ProxyEventAnchor::Window { start, end } => {
            let l = first_index_at_or_after(timestamps, start);
            let u = last_index_at_or_before(timestamps, end);
            Ok((l, u, l.map(|x| x + 1)))
        }
        ProxyEventAnchor::Point { at } => {
            let Some(anchor) = first_index_at_or_after(timestamps, at) else {
                return Ok((None, None, None));
            };
            let pre = if point_policy.causal_only {
                0
            } else {
                point_policy.pre_bars
            };
            let l = anchor.saturating_sub(pre);
            let u = (anchor + point_policy.post_bars).min(timestamps.len().saturating_sub(1));
            Ok((Some(l), Some(u), Some(anchor + 1)))
        }
    }
}

fn first_index_at_or_after(ts: &[NaiveDateTime], x: NaiveDateTime) -> Option<usize> {
    let idx = ts.partition_point(|t| *t < x);
    if idx < ts.len() { Some(idx) } else { None }
}

fn last_index_at_or_before(ts: &[NaiveDateTime], x: NaiveDateTime) -> Option<usize> {
    let idx = ts.partition_point(|t| *t <= x);
    if idx == 0 { None } else { Some(idx - 1) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(day: u32) -> NaiveDateTime {
        NaiveDateTime::parse_from_str(&format!("2024-01-{day:02} 00:00:00"), "%Y-%m-%d %H:%M:%S")
            .unwrap()
    }

    fn base_timestamps(n: u32) -> Vec<NaiveDateTime> {
        (1..=n).map(ts).collect()
    }

    #[test]
    fn point_event_alignment_with_symmetric_window() {
        let t = base_timestamps(10);
        let events = vec![ProxyEvent {
            id: "e1".to_string(),
            event_type: "stress".to_string(),
            label: "Event".to_string(),
            asset_scope: None,
            anchor: ProxyEventAnchor::Point { at: ts(5) },
        }];
        let alarms = vec![4, 9];
        let cfg = RouteAConfig {
            point_policy: PointMatchPolicy {
                pre_bars: 1,
                post_bars: 1,
                causal_only: false,
            },
        };
        let r = evaluate_proxy_events(&t, &alarms, &events, &cfg).unwrap();
        assert_eq!(r.n_events_covered, 1);
        assert_eq!(r.n_alarms_aligned, 1);
        assert!((r.event_coverage - 1.0).abs() < 1e-12);
        assert!((r.alarm_relevance - 0.5).abs() < 1e-12);
    }

    #[test]
    fn causal_only_rejects_pre_event_alarm() {
        let t = base_timestamps(10);
        let events = vec![ProxyEvent {
            id: "e1".to_string(),
            event_type: "stress".to_string(),
            label: "Event".to_string(),
            asset_scope: None,
            anchor: ProxyEventAnchor::Point { at: ts(5) },
        }];
        let alarms = vec![4];
        let cfg = RouteAConfig {
            point_policy: PointMatchPolicy {
                pre_bars: 3,
                post_bars: 0,
                causal_only: true,
            },
        };
        let r = evaluate_proxy_events(&t, &alarms, &events, &cfg).unwrap();
        assert_eq!(r.n_events_covered, 0);
    }

    #[test]
    fn window_event_alignment_works() {
        let t = base_timestamps(10);
        let events = vec![ProxyEvent {
            id: "e1".to_string(),
            event_type: "window".to_string(),
            label: "Window".to_string(),
            asset_scope: None,
            anchor: ProxyEventAnchor::Window {
                start: ts(3),
                end: ts(6),
            },
        }];
        let alarms = vec![2, 3, 7];
        let r = evaluate_proxy_events(&t, &alarms, &events, &RouteAConfig::default()).unwrap();
        assert_eq!(r.n_alarms_aligned, 1);
    }

    #[test]
    fn delay_is_first_alarm_minus_event_anchor() {
        let t = base_timestamps(10);
        let events = vec![ProxyEvent {
            id: "e1".to_string(),
            event_type: "point".to_string(),
            label: "Point".to_string(),
            asset_scope: None,
            anchor: ProxyEventAnchor::Point { at: ts(5) },
        }];
        let alarms = vec![6];
        let cfg = RouteAConfig {
            point_policy: PointMatchPolicy {
                pre_bars: 0,
                post_bars: 2,
                causal_only: true,
            },
        };
        let r = evaluate_proxy_events(&t, &alarms, &events, &cfg).unwrap();
        assert_eq!(r.alignments[0].first_delay_bars, Some(1));
    }

    #[test]
    fn empty_events_and_alarms_is_safe() {
        let t = base_timestamps(10);
        let r = evaluate_proxy_events(&t, &[], &[], &RouteAConfig::default()).unwrap();
        assert_eq!(r.n_events, 0);
        assert_eq!(r.n_alarms, 0);
        assert!(r.event_coverage.is_nan());
        assert!(r.alarm_relevance.is_nan());
    }
}
