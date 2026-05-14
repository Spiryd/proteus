/// Chronological train / validation / test splitting for time series.
///
/// # Why chronological splitting is mandatory
///
/// For a time-series changepoint detector, the model is trained on historical
/// data and evaluated on future data.  Random train/test shuffling would
/// create **temporal leakage**: information from future observations would
/// be visible during training, invalidating the causal evaluation.
///
/// The split is therefore defined by two cut points in calendar time rather
/// than by observation counts or random indices.
///
/// # Split structure
///
/// The series is divided into three non-overlapping, contiguous partitions:
///
/// ```text
/// [epoch ············ train_end) | [train_end ········ val_end) | [val_end ·· end]
///         Train                           Validation                   Test
/// ```
///
/// An observation at time `t` belongs to:
/// - **Train**      iff `t < train_end`
/// - **Validation** iff `train_end ≤ t < val_end`
/// - **Test**       iff `t ≥ val_end`
///
/// # Leakage-prevention contract
///
/// - Preprocessing steps that depend on data statistics (e.g., mean, variance
///   normalization) must be fit on the **train** partition only, then applied
///   to validation and test.
/// - The split is performed **before** any feature construction.
/// - `PartitionedSeries` carries `SplitConfig` so downstream phases can
///   document exactly which boundaries were used.
use chrono::NaiveDateTime;

use super::{CleanSeries, DatasetMeta, Observation};

// =========================================================================
// TimePartition
// =========================================================================

/// Partition label for one observation or timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimePartition {
    Train,
    Validation,
    Test,
}

// =========================================================================
// SplitConfig
// =========================================================================

/// Defines the two time cut points for a chronological 3-way split.
///
/// Both boundaries are **exclusive lower bounds** for the subsequent
/// partition (i.e., the boundary timestamp itself belongs to the later
/// partition).
#[derive(Debug, Clone)]
pub struct SplitConfig {
    /// First timestamp **not** in the train set.
    /// All observations with `timestamp < train_end` are in `Train`.
    pub train_end: NaiveDateTime,
    /// First timestamp **not** in the validation set.
    /// All observations with `train_end ≤ timestamp < val_end` are in `Validation`.
    pub val_end: NaiveDateTime,
}

// =========================================================================
// PartitionedSeries
// =========================================================================

/// A `CleanSeries` split into train / validation / test partitions.
///
/// Each partition is a `Vec<Observation>` in ascending chronological order,
/// consistent with the parent series ordering.  The three vecs are
/// collectively exhaustive and mutually exclusive over the parent series.
#[derive(Debug, Clone)]
pub struct PartitionedSeries {
    pub train: Vec<Observation>,
    pub validation: Vec<Observation>,
    pub test: Vec<Observation>,
    /// Metadata from the parent `CleanSeries`.
    pub meta: DatasetMeta,
    /// The split boundaries used.
    pub config: SplitConfig,
}

impl PartitionedSeries {
    /// Split `series` into three chronological partitions according to `config`.
    ///
    /// # Panics
    ///
    /// Panics if `config.train_end > config.val_end`.
    pub fn from_series(series: CleanSeries, config: SplitConfig) -> Self {
        assert!(
            config.train_end <= config.val_end,
            "split invariant violated: train_end ({}) must be ≤ val_end ({})",
            config.train_end,
            config.val_end
        );

        let mut train = Vec::new();
        let mut validation = Vec::new();
        let mut test = Vec::new();

        for obs in series.observations {
            if obs.timestamp < config.train_end {
                train.push(obs);
            } else if obs.timestamp < config.val_end {
                validation.push(obs);
            } else {
                test.push(obs);
            }
        }

        Self {
            train,
            validation,
            test,
            meta: series.meta,
            config,
        }
    }

    /// Assign a `TimePartition` label to any arbitrary timestamp.
    pub fn partition_of(&self, ts: NaiveDateTime) -> TimePartition {
        if ts < self.config.train_end {
            TimePartition::Train
        } else if ts < self.config.val_end {
            TimePartition::Validation
        } else {
            TimePartition::Test
        }
    }

    /// Total observations across all partitions (equals the parent series length).
    pub fn total_len(&self) -> usize {
        self.train.len() + self.validation.len() + self.test.len()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{
        DatasetMeta, Observation, ValidationReport,
        meta::{DataMode, DataSource, PriceField, SessionConvention},
    };
    use chrono::NaiveDateTime;

    fn ts(s: &str) -> NaiveDateTime {
        NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").unwrap()
    }

    fn obs(s: &str) -> Observation {
        Observation {
            timestamp: ts(s),
            value: 1.0,
        }
    }

    fn empty_report() -> ValidationReport {
        ValidationReport {
            n_input: 0,
            n_dropped_duplicates: 0,
            n_output: 0,
            had_unsorted_input: false,
            gaps: vec![],
        }
    }

    fn meta() -> DatasetMeta {
        DatasetMeta {
            symbol: "TEST".to_string(),
            mode: DataMode::Daily,
            source: DataSource::AlphaVantage,
            price_field: PriceField::Value,
            session_convention: SessionConvention::FullDay,
            fetched_at: None,
            unit: None,
        }
    }

    fn make_series(timestamps: &[&str]) -> CleanSeries {
        CleanSeries {
            observations: timestamps.iter().map(|&s| obs(s)).collect(),
            meta: meta(),
            report: empty_report(),
        }
    }

    #[test]
    fn assigns_observations_to_correct_partitions() {
        // 3 observations: 2024-01-01 (train), 2024-04-01 (val), 2024-07-01 (test)
        let series = make_series(&[
            "2024-01-01 00:00:00",
            "2024-04-01 00:00:00",
            "2024-07-01 00:00:00",
        ]);
        let cfg = SplitConfig {
            train_end: ts("2024-04-01 00:00:00"),
            val_end: ts("2024-07-01 00:00:00"),
        };
        let p = PartitionedSeries::from_series(series, cfg);
        assert_eq!(p.train.len(), 1);
        assert_eq!(p.validation.len(), 1);
        assert_eq!(p.test.len(), 1);
    }

    #[test]
    fn partitions_cover_all_observations() {
        let series = make_series(&[
            "2024-01-01 00:00:00",
            "2024-02-01 00:00:00",
            "2024-03-01 00:00:00",
            "2024-04-01 00:00:00",
            "2024-05-01 00:00:00",
        ]);
        let n = series.observations.len();
        let cfg = SplitConfig {
            train_end: ts("2024-03-01 00:00:00"),
            val_end: ts("2024-04-15 00:00:00"),
        };
        let p = PartitionedSeries::from_series(series, cfg);
        assert_eq!(p.total_len(), n);
    }

    #[test]
    fn no_temporal_overlap_between_partitions() {
        let series = make_series(&[
            "2024-01-01 00:00:00",
            "2024-06-01 00:00:00",
            "2024-12-01 00:00:00",
        ]);
        let cfg = SplitConfig {
            train_end: ts("2024-04-01 00:00:00"),
            val_end: ts("2024-08-01 00:00:00"),
        };
        let p = PartitionedSeries::from_series(series, cfg);
        // Max of train < min of val < min of test.
        if let (Some(tr_last), Some(v_first)) = (p.train.last(), p.validation.first()) {
            assert!(tr_last.timestamp < v_first.timestamp);
        }
        if let (Some(v_last), Some(te_first)) = (p.validation.last(), p.test.first()) {
            assert!(v_last.timestamp < te_first.timestamp);
        }
    }

    #[test]
    fn all_in_train_when_cut_is_after_last_observation() {
        let series = make_series(&["2024-01-01 00:00:00", "2024-02-01 00:00:00"]);
        let cfg = SplitConfig {
            train_end: ts("2030-01-01 00:00:00"),
            val_end: ts("2030-06-01 00:00:00"),
        };
        let p = PartitionedSeries::from_series(series, cfg);
        assert_eq!(p.train.len(), 2);
        assert!(p.validation.is_empty());
        assert!(p.test.is_empty());
    }
}
