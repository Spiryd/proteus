use std::path::Path;
use csv::Writer;
use super::schema::*;

pub fn export_alarms(path: &Path, alarms: &[AlarmRecord]) -> anyhow::Result<()> {
    let mut wtr = Writer::from_path(path)?;
    for alarm in alarms {
        wtr.serialize(alarm)?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn export_matched_events(
    path: &Path,
    events: &[MatchedEventRecord],
) -> anyhow::Result<()> {
    let mut wtr = Writer::from_path(path)?;
    for event in events {
        wtr.serialize(event)?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn export_segments(path: &Path, segments: &[SegmentRecord]) -> anyhow::Result<()> {
    let mut wtr = Writer::from_path(path)?;
    for segment in segments {
        wtr.serialize(segment)?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn export_feature_trace(
    path: &Path,
    trace: &[FeatureTraceRecord],
) -> anyhow::Result<()> {
    let mut wtr = Writer::from_path(path)?;
    for record in trace {
        wtr.serialize(record)?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn export_score_trace(path: &Path, trace: &[ScoreTraceRecord]) -> anyhow::Result<()> {
    let mut wtr = Writer::from_path(path)?;
    for record in trace {
        wtr.serialize(record)?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn export_regime_posterior(
    path: &Path,
    posteriors: &[RegimePosteriorRecord],
) -> anyhow::Result<()> {
    let mut wtr = Writer::from_path(path)?;
    for record in posteriors {
        wtr.serialize(record)?;
    }
    wtr.flush()?;
    Ok(())
}
