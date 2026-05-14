use serde::{Deserialize, Serialize};

use crate::data::Observation;

use super::route_a::{ProxyEvent, ProxyEventEvaluationResult, RouteAConfig, evaluate_proxy_events};
use super::route_b::{RouteBConfig, SegmentationEvaluationResult, evaluate_segmentation};

/// Metadata for one real-data evaluation run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealEvalMeta {
    pub asset: String,
    pub frequency: String,
    pub feature_label: String,
    pub detector_label: String,
}

/// Combined Route A + Route B output for one run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RealEvalResult {
    pub meta: RealEvalMeta,
    pub n_observations: usize,
    pub n_alarms: usize,
    pub route_a: ProxyEventEvaluationResult,
    pub route_b: SegmentationEvaluationResult,
}

/// Evaluate real-data detector output using Route A + Route B only.
pub fn evaluate_real_data(
    observations: &[Observation],
    alarm_indices_1based: &[usize],
    proxy_events: &[ProxyEvent],
    route_a_cfg: &RouteAConfig,
    route_b_cfg: &RouteBConfig,
    meta: RealEvalMeta,
) -> anyhow::Result<RealEvalResult> {
    let timestamps: Vec<_> = observations.iter().map(|o| o.timestamp).collect();

    let route_a =
        evaluate_proxy_events(&timestamps, alarm_indices_1based, proxy_events, route_a_cfg)?;
    let route_b = evaluate_segmentation(observations, alarm_indices_1based, route_b_cfg);

    Ok(RealEvalResult {
        meta,
        n_observations: observations.len(),
        n_alarms: alarm_indices_1based.len(),
        route_a,
        route_b,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::real_eval::route_a::{PointMatchPolicy, ProxyEventAnchor};

    fn obs(i: usize, v: f64) -> Observation {
        Observation {
            timestamp: chrono::NaiveDateTime::parse_from_str(
                &format!("2024-01-{i:02} 00:00:00"),
                "%Y-%m-%d %H:%M:%S",
            )
            .unwrap(),
            value: v,
        }
    }

    #[test]
    fn combined_eval_returns_summary() {
        let observations = vec![obs(1, 0.0), obs(2, 0.1), obs(3, 3.0), obs(4, 2.8)];
        let alarms = vec![3];
        let events = vec![ProxyEvent {
            id: "e1".to_string(),
            event_type: "stress".to_string(),
            label: "Stress".to_string(),
            asset_scope: Some("SPY".to_string()),
            anchor: ProxyEventAnchor::Point {
                at: observations[2].timestamp,
            },
        }];

        let out = evaluate_real_data(
            &observations,
            &alarms,
            &events,
            &RouteAConfig {
                point_policy: PointMatchPolicy {
                    pre_bars: 0,
                    post_bars: 1,
                    causal_only: true,
                },
            },
            &RouteBConfig::default(),
            RealEvalMeta {
                asset: "SPY".to_string(),
                frequency: "daily".to_string(),
                feature_label: "log_return".to_string(),
                detector_label: "surprise".to_string(),
            },
        )
        .unwrap();

        assert_eq!(out.meta.asset, "SPY");
        assert_eq!(out.n_alarms, 1);
        assert_eq!(out.route_b.global.n_segments, 2);
    }
}
