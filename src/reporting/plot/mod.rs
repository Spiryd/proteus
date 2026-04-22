pub mod common;
pub mod delay_distribution;
pub mod detector_scores;
pub mod regime_posteriors;
pub mod segmentation;
pub mod signal_alarms;

pub use delay_distribution::{DelayDistributionPlotInput, render_delay_distribution};
pub use detector_scores::{DetectorScoresPlotInput, render_detector_scores};
pub use regime_posteriors::{RegimePosteriorPlotInput, render_regime_posteriors};
pub use segmentation::{SegmentationPlotInput, render_segmentation};
pub use signal_alarms::{SignalWithAlarmsPlotInput, render_signal_with_alarms};
