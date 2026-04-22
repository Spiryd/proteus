use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlarmRecord {
    pub timestamp: DateTime<Utc>,
    pub detector_score: f64,
    pub threshold: f64,
    pub is_alarm: bool,
    pub persistence_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedEventRecord {
    pub true_changepoint: DateTime<Utc>,
    pub matched_alarm_time: Option<DateTime<Utc>>,
    pub delay_steps: Option<usize>,
    pub is_detected: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentRecord {
    pub segment_id: usize,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub duration_steps: usize,
    pub is_detected: bool,
    pub mean_shift: f64,
    pub contrast_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureTraceRecord {
    pub timestamp: DateTime<Utc>,
    pub observation: f64,
    pub feature_value: f64,
    pub regime: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreTraceRecord {
    pub timestamp: DateTime<Utc>,
    pub detector_score: f64,
    pub threshold: f64,
    pub is_alarm: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimePosteriorRecord {
    pub timestamp: DateTime<Utc>,
    pub posteriors: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alarm_record_serialization() {
        let record = AlarmRecord {
            timestamp: Utc::now(),
            detector_score: 0.8,
            threshold: 0.5,
            is_alarm: true,
            persistence_count: 2,
        };

        let json = serde_json::to_string(&record).unwrap();
        let deserialized: AlarmRecord = serde_json::from_str(&json).unwrap();

        assert_eq!(record.detector_score, deserialized.detector_score);
        assert_eq!(record.is_alarm, deserialized.is_alarm);
    }
}
