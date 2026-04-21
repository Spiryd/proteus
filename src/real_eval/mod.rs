/// Phase 18 real-data evaluation layer.
///
/// This module evaluates detector alarms on real market data without latent
/// ground-truth changepoints using two routes only:
/// - Route A: proxy-event alignment,
/// - Route B: detector-induced segmentation self-consistency.
///
/// Route C (downstream predictive usefulness) is explicitly out of scope.
pub mod report;
pub mod route_a;
pub mod route_b;

pub use report::{RealEvalMeta, RealEvalResult, RealEvalSummaryRow, evaluate_real_data};
pub use route_a::{
    EventAlignment, PointMatchPolicy, ProxyEvent, ProxyEventAnchor, ProxyEventEvaluationResult,
    RouteAConfig, evaluate_proxy_events,
};
pub use route_b::{
    AdjacentSegmentContrast, DetectedSegment, RouteBConfig, SegmentSummary,
    SegmentationEvaluationResult, SegmentationGlobalSummary, ShortSegmentPolicy, build_segments,
    evaluate_segmentation,
};
