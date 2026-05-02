#![allow(unused_imports)]
pub mod common;
pub mod delay_distribution;
pub mod detector_scores;
pub mod regime_posteriors;
pub mod segmentation;
pub mod signal_alarms;

pub use delay_distribution::{DelayDistributionPlotInput, render_delay_distribution};
pub use detector_scores::DetectorScoresPlotInput;
#[cfg(not(test))]
pub use detector_scores::render_detector_scores;
pub use regime_posteriors::RegimePosteriorPlotInput;
#[cfg(not(test))]
pub use regime_posteriors::render_regime_posteriors;
pub use segmentation::{SegmentationPlotInput, render_segmentation};
pub use signal_alarms::SignalWithAlarmsPlotInput;
#[cfg(not(test))]
pub use signal_alarms::render_signal_with_alarms;
