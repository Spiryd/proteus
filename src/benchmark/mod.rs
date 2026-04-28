#![allow(unused_imports)]
/// Online benchmarking protocol for Markov-Switching-based changepoint detectors.
///
/// # Purpose
///
/// This module defines the complete evaluation pipeline for assessing the
/// statistical and computational quality of an online changepoint detector.
/// It is strictly an **evaluation layer**: it consumes detector outputs that
/// have already been produced causally and measures how well those outputs
/// correspond to a known ground truth.
///
/// # Protocol overview
///
/// The benchmark proceeds in three stages:
///
/// ```text
/// Ground truth  ─────────────────────────────────────────┐
///                                                         ▼
/// Detector run  →  [AlarmEvent, ...]  →  EventMatcher  →  MatchResult
///                                                         ▼
///                                               MetricSuite (per stream)
///                                                         ▼
///                                         BenchmarkAggregateResult (N runs)
/// ```
///
/// # Evaluation semantics
///
/// Online changepoint detection is an **event-level** problem, not pointwise
/// classification.  The detector emits a sparse alarm sequence
/// `{a₁, a₂, …, aₙ}` which must be compared against the ordered true
/// changepoint times `{τ₁, τ₂, …, τₘ}`.
///
/// The benchmark uses a **causal detection-window rule**: for each true
/// changepoint τₘ, the first alarm in the half-open interval `[τₘ, τₘ + w)`
/// counts as a successful detection.
///
/// # Sub-modules
///
/// - [`truth`] — `ChangePointTruth` and stream metadata
/// - [`matching`] — event matcher with detection-window logic
/// - [`metrics`] — per-stream metric computation
/// - [`result`] — per-stream and aggregate result objects, timing
pub mod matching;
pub mod metrics;
pub mod result;
pub mod truth;

pub use matching::{EventMatcher, MatchConfig, MatchResult};
pub use metrics::MetricSuite;
pub use result::{AggregateResult, BenchmarkLabel, RunResult, TimingSummary};
pub use truth::{ChangePointTruth, StreamMeta};
