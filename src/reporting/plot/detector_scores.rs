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
    let root = BitMapBackend::new(output_path, (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    if input.timestamps.is_empty() {
        return Ok(());
    }

    let min_ts = input.timestamps[0].timestamp();
    let max_ts = input.timestamps.last().unwrap().timestamp();
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
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d((min_ts as f64)..(max_ts as f64), min_score..max_score)?;

    chart
        .configure_mesh()
        .y_label_style(("sans-serif", 15))
        .y_desc("Detector Score")
        .draw()?;

    chart.draw_series(input.scores.iter().enumerate().map(|(i, &score)| {
        let x = input.timestamps[i].timestamp() as f64;
        Circle::new((x, score), 1, BLUE)
    }))?;

    chart.draw_series(std::iter::once(PathElement::new(
        vec![
            (min_ts as f64, input.threshold),
            (max_ts as f64, input.threshold),
        ],
        RED,
    )))?;

    for (ts, is_alarm) in &input.alarms {
        if *is_alarm {
            let x = ts.timestamp() as f64;
            if let Some(&score) = input
                .scores
                .get(input.timestamps.iter().position(|&t| t == *ts).unwrap_or(0))
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
