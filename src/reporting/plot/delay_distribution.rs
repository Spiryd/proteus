#[cfg(not(test))]
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
#[cfg(not(test))]
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelayDistributionPlotInput {
    pub delays: Vec<usize>,
    pub title: String,
}

#[cfg(not(test))]
pub fn render_delay_distribution(
    input: &DelayDistributionPlotInput,
    output_path: &Path,
) -> anyhow::Result<()> {
    let root = BitMapBackend::new(output_path, (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    if input.delays.is_empty() {
        return Ok(());
    }

    // Calculate histogram bins
    let max_delay = *input.delays.iter().max().unwrap_or(&1);
    let mut bins = vec![0usize; max_delay + 1];
    for &delay in &input.delays {
        if delay < bins.len() {
            bins[delay] += 1;
        }
    }

    let max_count = *bins.iter().max().unwrap_or(&1);

    let mut chart = ChartBuilder::on(&root)
        .caption(&input.title, ("sans-serif", 30))
        .margin(15)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0f64..(max_delay as f64 + 1.0),
            0f64..(max_count as f64 + 1.0),
        )?;

    chart
        .configure_mesh()
        .y_label_style(("sans-serif", 15))
        .y_desc("Count")
        .x_desc("Delay (steps)")
        .draw()?;

    // Draw histogram bars
    for (i, &count) in bins.iter().enumerate() {
        if count > 0 {
            let x = i as f64;
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x - 0.4, 0.0), (x + 0.4, count as f64)],
                BLUE.filled(),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delay_distribution_input_serialization() {
        let input = DelayDistributionPlotInput {
            delays: vec![1, 2, 2, 3],
            title: "Delays".to_string(),
        };

        let json = serde_json::to_string(&input).unwrap();
        let deserialized: DelayDistributionPlotInput = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.delays.len(), 4);
        assert_eq!(deserialized.title, "Delays");
    }
}
