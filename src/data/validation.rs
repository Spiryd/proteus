/// Chronology validation and data-quality reporting for market time series.
///
/// # Validation pipeline
///
/// Given a raw `Vec<Observation>` (as parsed from an API response or cache),
/// `validate` performs the following operations in order:
///
/// 1. **Record ordering status** — check whether the input is already
///    ascending before any transformation.
/// 2. **Sort ascending** (stable, by `timestamp`) — Alpha Vantage returns
///    data newest-first; the modeling pipeline requires oldest-first.
/// 3. **Deduplicate** — remove all but the first occurrence of each
///    timestamp (keep-first policy after stable sort).
/// 4. **Detect intraday gaps** — for `DataMode::Intraday`, flag pairs of
///    consecutive observations whose interval exceeds 3× the expected bar
///    duration.  Daily data is not gap-checked because calendar absences
///    (weekends, holidays) are structurally normal.
///
/// The caller receives a `ValidationReport` describing every change made
/// and every issue detected.  The validator **never** silently drops valid
/// data; it only removes provably duplicate entries.
use chrono::{Duration, NaiveDateTime};

use super::Observation;
use super::meta::DataMode;

// =========================================================================
// Gap
// =========================================================================

/// A detected temporal gap between two consecutive observations.
///
/// For intraday data a gap is flagged when consecutive observations are
/// separated by more than **3× the expected bar duration**.  The multiplier
/// of 3 accommodates minor irregularities (extended-hours coverage changes,
/// sparse vendor data near session edges) while still flagging genuine
/// missing-session events.
///
/// **Gaps are reported but not filled** — the pipeline never silently
/// imputes price dynamics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gap {
    /// Timestamp of the last observation before the gap.
    pub from: NaiveDateTime,
    /// Timestamp of the first observation after the gap.
    pub to: NaiveDateTime,
}

// =========================================================================
// ValidationReport
// =========================================================================

/// Data-quality summary produced during `CleanSeries` construction.
///
/// Carry this struct alongside the cleaned data so that thesis tables can
/// report how many observations were dropped or flagged at each stage.
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Number of observations supplied to the validator (before any changes).
    pub n_input: usize,
    /// Number of duplicate timestamps removed.
    ///
    /// The keep-first policy is applied: after stable ascending sort, the
    /// first occurrence of each timestamp is retained.  For Alpha Vantage
    /// intraday pagination, the same bar can appear at month boundaries.
    pub n_dropped_duplicates: usize,
    /// Number of observations remaining after validation.
    pub n_output: usize,
    /// Whether the input sequence had any non-ascending timestamps before sorting.
    ///
    /// `true` does **not** indicate an error — the validator corrects this
    /// automatically.  Alpha Vantage consistently returns data newest-first,
    /// so `true` is the expected value for freshly fetched responses.
    pub had_unsorted_input: bool,
    /// Intraday gaps detected (always empty for `DataMode::Daily`).
    pub gaps: Vec<Gap>,
}

// =========================================================================
// Public entry point
// =========================================================================

/// Validate and normalize `obs` in-place; return a quality report.
///
/// See module-level docs for the full operation sequence.
pub fn validate(obs: &mut Vec<Observation>, mode: &DataMode) -> ValidationReport {
    let n_input = obs.len();

    // 1. Check whether input was already ascending.
    let had_unsorted_input = obs.windows(2).any(|w| w[0].timestamp > w[1].timestamp);

    // 2. Sort ascending (stable — equal timestamps retain their input order,
    //    so keep-first is well-defined after dedup).
    obs.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

    // 3. Remove consecutive duplicates (consecutive after sort = all duplicates).
    let before_dedup = obs.len();
    obs.dedup_by_key(|o| o.timestamp);
    let n_dropped_duplicates = before_dedup - obs.len();

    // 4. Detect intraday gaps.
    let gaps = if let DataMode::Intraday { bar_minutes } = mode {
        detect_intraday_gaps(obs, *bar_minutes)
    } else {
        vec![]
    };

    ValidationReport {
        n_input,
        n_dropped_duplicates,
        n_output: obs.len(),
        had_unsorted_input,
        gaps,
    }
}

// =========================================================================
// Private helpers
// =========================================================================

fn detect_intraday_gaps(obs: &[Observation], bar_minutes: u32) -> Vec<Gap> {
    if obs.len() < 2 {
        return vec![];
    }
    // Threshold = 3× expected step.
    let threshold = Duration::minutes(i64::from(bar_minutes) * 3);
    obs.windows(2)
        .filter_map(|w| {
            if w[1].timestamp - w[0].timestamp > threshold {
                Some(Gap {
                    from: w[0].timestamp,
                    to: w[1].timestamp,
                })
            } else {
                None
            }
        })
        .collect()
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Observation;

    fn ts(s: &str) -> NaiveDateTime {
        NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").unwrap()
    }

    fn obs(s: &str) -> Observation {
        Observation {
            timestamp: ts(s),
            value: 1.0,
        }
    }

    #[test]
    fn accepts_already_sorted_sequence() {
        let mut v = vec![obs("2024-01-02 09:30:00"), obs("2024-01-02 09:31:00")];
        let r = validate(&mut v, &DataMode::Daily);
        assert!(!r.had_unsorted_input);
        assert_eq!(r.n_dropped_duplicates, 0);
        assert_eq!(r.n_output, 2);
    }

    #[test]
    fn sorts_unsorted_input_and_flags_it() {
        let mut v = vec![obs("2024-01-02 09:31:00"), obs("2024-01-02 09:30:00")];
        let r = validate(&mut v, &DataMode::Daily);
        assert!(r.had_unsorted_input);
        assert!(
            v[0].timestamp < v[1].timestamp,
            "must be ascending after validate"
        );
    }

    #[test]
    fn removes_duplicate_timestamps_keep_first() {
        // After stable sort, "a" and "b" at 09:30 appear in original order;
        // "b" is the one removed.
        let mut v = vec![
            Observation {
                timestamp: ts("2024-01-02 09:30:00"),
                value: 1.0,
            },
            Observation {
                timestamp: ts("2024-01-02 09:30:00"),
                value: 2.0,
            }, // dup
            Observation {
                timestamp: ts("2024-01-02 09:31:00"),
                value: 3.0,
            },
        ];
        let r = validate(&mut v, &DataMode::Intraday { bar_minutes: 1 });
        assert_eq!(r.n_dropped_duplicates, 1);
        assert_eq!(r.n_output, 2);
        assert_eq!(v[0].value, 1.0, "keep-first: original value 1.0 retained");
    }

    #[test]
    fn detects_intraday_gap_exceeding_threshold() {
        // bar_minutes=5 → threshold=15min.  Step from 09:30→10:00 is 30min → flagged.
        let t0 = ts("2024-01-02 09:30:00");
        let t1 = ts("2024-01-02 10:00:00"); // 30-min gap
        let t2 = ts("2024-01-02 10:05:00");
        let mut v = vec![
            Observation {
                timestamp: t0,
                value: 1.0,
            },
            Observation {
                timestamp: t1,
                value: 1.0,
            },
            Observation {
                timestamp: t2,
                value: 1.0,
            },
        ];
        let r = validate(&mut v, &DataMode::Intraday { bar_minutes: 5 });
        assert_eq!(r.gaps.len(), 1);
        assert_eq!(r.gaps[0].from, t0);
        assert_eq!(r.gaps[0].to, t1);
    }

    #[test]
    fn normal_intraday_step_not_flagged_as_gap() {
        // bar_minutes=5 → threshold=15min.  Step of 5min is not flagged.
        let mut v = vec![
            obs("2024-01-02 09:30:00"),
            obs("2024-01-02 09:35:00"),
            obs("2024-01-02 09:40:00"),
        ];
        let r = validate(&mut v, &DataMode::Intraday { bar_minutes: 5 });
        assert!(r.gaps.is_empty());
    }

    #[test]
    fn daily_mode_never_reports_gaps() {
        // Even a 5-month jump should not be reported as a gap in daily mode.
        let mut v = vec![obs("2024-01-01 00:00:00"), obs("2024-06-01 00:00:00")];
        let r = validate(&mut v, &DataMode::Daily);
        assert!(r.gaps.is_empty());
    }

    #[test]
    fn empty_input_produces_empty_report() {
        let mut v: Vec<Observation> = vec![];
        let r = validate(&mut v, &DataMode::Daily);
        assert_eq!(r.n_input, 0);
        assert_eq!(r.n_output, 0);
        assert!(!r.had_unsorted_input);
        assert!(r.gaps.is_empty());
    }
}
