pub mod delay_distribution;
pub mod detector_scores;
pub mod regime_posteriors;
pub mod segmentation;
pub mod signal_alarms;

// Plot rendering is only exercised in non-test builds (CI / headless
// environments crash the plotters font stack).  All re-exports of the
// plot inputs and renderers are gated to match the call site in
// `experiments::runner::generate_plots`.
#[cfg(not(test))]
pub use delay_distribution::{DelayDistributionPlotInput, render_delay_distribution};
#[cfg(not(test))]
pub use detector_scores::{DetectorScoresPlotInput, render_detector_scores};
#[cfg(not(test))]
pub use regime_posteriors::{RegimePosteriorPlotInput, render_regime_posteriors};
#[cfg(not(test))]
pub use segmentation::{SegmentationPlotInput, render_segmentation};
#[cfg(not(test))]
pub use signal_alarms::{SignalWithAlarmsPlotInput, render_signal_with_alarms};
