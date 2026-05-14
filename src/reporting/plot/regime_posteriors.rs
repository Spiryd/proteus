use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
#[cfg(not(test))]
use plotters::prelude::*;
#[cfg(not(test))]
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimePosteriorPlotInput {
    pub timestamps: Vec<DateTime<Utc>>,
    pub posteriors: Vec<Vec<f64>>,
    pub title: String,
}

#[cfg(not(test))]
pub fn render_regime_posteriors(
    input: &RegimePosteriorPlotInput,
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
    let k = input.posteriors.first().map_or(1, Vec::len);

    let mut chart = ChartBuilder::on(&root)
        .caption(&input.title, ("sans-serif", 30))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min_ts..max_ts, 0f64..1f64)?;

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
        .y_desc("Posterior Probability")
        .draw()?;

    let colors: Vec<&RGBColor> = vec![&RED, &BLUE, &GREEN, &MAGENTA, &CYAN];

    for j in 0..k {
        let color = colors[j % colors.len()];
        chart.draw_series(input.posteriors.iter().enumerate().map(|(i, post_vec)| {
            let x = input.timestamps[i].timestamp() as f64;
            let y = post_vec.get(j).copied().unwrap_or(0.0);
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
