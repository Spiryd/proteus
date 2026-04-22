use chrono::{DateTime, Utc};
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimePosteriorPlotInput {
    pub timestamps: Vec<DateTime<Utc>>,
    pub posteriors: Vec<Vec<f64>>,
    pub title: String,
}

pub fn render_regime_posteriors(
    input: &RegimePosteriorPlotInput,
    output_path: &Path,
) -> anyhow::Result<()> {
    let root = BitMapBackend::new(output_path, (1200, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    if input.timestamps.is_empty() {
        return Ok(());
    }

    let min_ts = input.timestamps[0].timestamp();
    let max_ts = input.timestamps.last().unwrap().timestamp();
    let k = input.posteriors.first().map(|p| p.len()).unwrap_or(1);

    let mut chart = ChartBuilder::on(&root)
        .caption(&input.title, ("sans-serif", 30))
        .margin(15)
        .x_label_area_size(30)
        .y_label_area_size(60)
        .build_cartesian_2d((min_ts as f64)..(max_ts as f64), 0f64..1f64)?;

    chart
        .configure_mesh()
        .y_label_style(("sans-serif", 15))
        .y_desc("Posterior Probability")
        .draw()?;

    let colors: Vec<&RGBColor> = vec![&RED, &BLUE, &GREEN, &MAGENTA, &CYAN];

    for j in 0..k {
        let color = colors[j % colors.len()];
        chart.draw_series(input.posteriors.iter().enumerate().map(|(i, post_vec)| {
            let x = input.timestamps[i].timestamp() as f64;
            let y = post_vec.get(j).cloned().unwrap_or(0.0);
            Circle::new((x, y), 1, color)
        }))?;
    }

    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_posterior_input_serialization() {
        let input = RegimePosteriorPlotInput {
            timestamps: vec![Utc::now()],
            posteriors: vec![vec![0.3, 0.5, 0.2]],
            title: "Regime Posteriors".to_string(),
        };

        let json = serde_json::to_string(&input).unwrap();
        let _deserialized: RegimePosteriorPlotInput = serde_json::from_str(&json).unwrap();
    }
}
