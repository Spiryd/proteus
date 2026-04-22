use plotters::style::RGBColor;

pub mod colors {
    use plotters::style::RGBColor;

    pub const ALARM: RGBColor = RGBColor(255, 0, 0);
    pub const CHANGEPOINT: RGBColor = RGBColor(0, 0, 255);
    pub const OBSERVATION: RGBColor = RGBColor(100, 100, 100);
    pub const THRESHOLD: RGBColor = RGBColor(200, 100, 100);
    pub const PRIMARY: RGBColor = RGBColor(31, 119, 180);
    pub const SECONDARY: RGBColor = RGBColor(255, 127, 14);
}

pub type PlotResult = Result<String, Box<dyn std::error::Error>>;

pub fn timestamp_to_x(ts: &chrono::DateTime<chrono::Utc>, min_ts: i64, max_ts: i64) -> f64 {
    let ts_i64 = ts.timestamp();
    let range = (max_ts - min_ts) as f64;
    if range == 0.0 {
        0.5
    } else {
        (ts_i64 - min_ts) as f64 / range
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_to_x() {
        let min_ts = 1000;
        let max_ts = 2000;

        let ts_mid = chrono::DateTime::<chrono::Utc>::from_timestamp(1500, 0).unwrap();
        let x = timestamp_to_x(&ts_mid, min_ts, max_ts);

        assert!((x - 0.5).abs() < 0.01);
    }
}
