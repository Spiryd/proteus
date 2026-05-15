use chrono::{DateTime, Utc};
#[cfg(not(test))]
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
#[cfg(not(test))]
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalWithAlarmsPlotInput {
    pub timestamps: Vec<DateTime<Utc>>,
    pub observations: Vec<f64>,
    pub alarms: Vec<(DateTime<Utc>, bool)>,
    pub true_changepoints: Option<Vec<DateTime<Utc>>>,
    pub title: String,
    pub y_label: String,
}

#[cfg(not(test))]
pub fn render_signal_with_alarms(
    input: &SignalWithAlarmsPlotInput,
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

    // Estimate one-bar width (seconds) for marker rectangles.
    let half_bar_secs: f64 = if input.timestamps.len() > 1 {
        (max_ts - min_ts) / input.timestamps.len() as f64 / 2.0
    } else {
        43200.0 // 12 hours
    };

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
        .y_desc(&input.y_label)
        .draw()?;

    chart.draw_series(input.observations.iter().enumerate().map(|(i, &y)| {
        let x = input.timestamps[i].timestamp() as f64;
        Circle::new((x, y), 2, BLUE)
    }))?;

    if let Some(cps) = &input.true_changepoints {
        for cp in cps {
            let x = cp.timestamp() as f64;
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x - half_bar_secs, min_obs), (x + half_bar_secs, max_obs)],
                BLUE.mix(0.3),
            )))?;
        }
    }

    for (ts, is_alarm) in &input.alarms {
        if *is_alarm {
            let x = ts.timestamp() as f64;
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x - half_bar_secs, min_obs), (x + half_bar_secs, max_obs)],
                RED.mix(0.2),
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
    fn test_signal_with_alarms_input_serialization() {
        let input = SignalWithAlarmsPlotInput {
            timestamps: vec![Utc::now()],
            observations: vec![1.0],
            alarms: vec![(Utc::now(), true)],
            true_changepoints: None,
            title: "Test".to_string(),
            y_label: "Value".to_string(),
        };

        let json = serde_json::to_string(&input).unwrap();
        let _deserialized: SignalWithAlarmsPlotInput = serde_json::from_str(&json).unwrap();
    }
}
