#[cfg(not(test))]
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
#[cfg(not(test))]
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentationPlotInput {
    pub timestamps: Vec<chrono::DateTime<chrono::Utc>>,
    pub observations: Vec<f64>,
    pub segments: Vec<(
        chrono::DateTime<chrono::Utc>,
        chrono::DateTime<chrono::Utc>,
        bool,
    )>,
    pub title: String,
}

#[cfg(not(test))]
pub fn render_segmentation(
    input: &SegmentationPlotInput,
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
    let min_obs = input
        .observations
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_obs = input
        .observations
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption(&input.title, ("sans-serif", 30))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min_ts..max_ts, min_obs..max_obs)?;

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
        .y_desc("Observations")
        .draw()?;

    // Plot observations
    chart.draw_series(input.observations.iter().enumerate().map(|(i, &y)| {
        let x = input.timestamps[i].timestamp() as f64;
        Circle::new((x, y), 2, BLUE)
    }))?;

    // Highlight detected segments
    for (start, end, is_detected) in &input.segments {
        let x_start = start.timestamp() as f64;
        let x_end = end.timestamp() as f64;
        let color = if *is_detected {
            RED.mix(0.2)
        } else {
            GREEN.mix(0.1)
        };
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x_start, min_obs), (x_end, max_obs)],
            color,
        )))?;
    }

    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segmentation_input_serialization() {
        let now = chrono::Utc::now();
        let input = SegmentationPlotInput {
            timestamps: vec![now],
            observations: vec![1.0],
            segments: vec![(now, now, true)],
            title: "Segmentation".to_string(),
        };

        let json = serde_json::to_string(&input).unwrap();
        let deserialized: SegmentationPlotInput = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.title, "Segmentation");
        assert_eq!(deserialized.observations.len(), 1);
    }
}
