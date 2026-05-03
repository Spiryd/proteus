use chrono::{DateTime, Utc};
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorScoresPlotInput {
    pub timestamps: Vec<DateTime<Utc>>,
    pub scores: Vec<f64>,
    pub threshold: f64,
    pub alarms: Vec<(DateTime<Utc>, bool)>,
    pub title: String,
}

#[cfg(not(test))]
pub fn render_detector_scores(
    input: &DetectorScoresPlotInput,
    output_path: &Path,
) -> anyhow::Result<()> {
    use chrono::{TimeZone, Utc};

    let root = BitMapBackend::new(output_path, (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    if input.timestamps.is_empty() {
        return Ok(());
    }

    let min_ts = input.timestamps[0].timestamp() as f64;
    let max_ts = input.timestamps.last().unwrap().timestamp() as f64;
    let min_score = input.scores.iter().copied().fold(f64::INFINITY, f64::min);
    let max_score = input
        .scores
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .max(input.threshold * 1.2);

    let mut chart = ChartBuilder::on(&root)
        .caption(&input.title, ("sans-serif", 30))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min_ts..max_ts, min_score..max_score)?;

    chart
        .configure_mesh()
        .x_label_style(("sans-serif", 12))
        .x_label_formatter(&|ts: &f64| {
            Utc.timestamp_opt(*ts as i64, 0)
                .single()
                .map(|dt| dt.format("%Y-%m-%d").to_string())
                .unwrap_or_default()
        })
        .y_label_style(("sans-serif", 15))
        .y_desc("Detector Score")
        .draw()?;

    chart.draw_series(input.scores.iter().enumerate().map(|(i, &score)| {
        let x = input.timestamps[i].timestamp() as f64;
        Circle::new((x, score), 1, BLUE)
    }))?;

    chart.draw_series(std::iter::once(PathElement::new(
        vec![(min_ts, input.threshold), (max_ts, input.threshold)],
        RED,
    )))?;

    for (ts, is_alarm) in &input.alarms {
        if *is_alarm {
            let x = ts.timestamp() as f64;
            if let Some(pos) = input
                .timestamps
                .iter()
                .position(|t| t.timestamp() == ts.timestamp())
                && let Some(&score) = input.scores.get(pos)
            {
                chart.draw_series(std::iter::once(Circle::new((x, score), 4, RED.filled())))?;
            }
        }
    }

    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_scores_input_serialization() {
        let input = DetectorScoresPlotInput {
            timestamps: vec![Utc::now()],
            scores: vec![0.7],
            threshold: 0.5,
            alarms: vec![(Utc::now(), false)],
            title: "Detector Scores".to_string(),
        };

        let json = serde_json::to_string(&input).unwrap();
        let _deserialized: DetectorScoresPlotInput = serde_json::from_str(&json).unwrap();
    }
}
