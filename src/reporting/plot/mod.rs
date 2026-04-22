pub mod common;
pub mod signal_alarms;
pub mod detector_scores;
pub mod regime_posteriors;
pub mod segmentation;
pub mod delay_distribution;

pub use signal_alarms::{SignalWithAlarmsPlotInput, render_signal_with_alarms};
pub use detector_scores::{DetectorScoresPlotInput, render_detector_scores};
pub use regime_posteriors::{RegimePosteriorPlotInput, render_regime_posteriors};
pub use segmentation::{SegmentationPlotInput, render_segmentation};
pub use delay_distribution::{DelayDistributionPlotInput, render_delay_distribution};
