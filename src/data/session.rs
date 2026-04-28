#![allow(dead_code)]
/// Intraday session handling: RTH filtering and session-boundary labelling.
///
/// # Trading session conventions
///
/// For US equity markets (SPY, QQQ), the Regular Trading Hours (RTH) are:
///
/// - **Open**: 09:30:00 US Eastern Time (ET)
/// - **Close**: 15:59:59 ET (last bar; bars beginning at 16:00:00 are excluded)
///
/// The filter criterion for bar-open timestamps is:
///
/// ```text
/// 09:30:00 ≤ bar_open_time < 16:00:00  (ET)
/// ```
///
/// Alpha Vantage intraday timestamps represent bar-open time and are already
/// in US Eastern Time.  No timezone conversion is performed here; callers
/// must not mix data from sources with different timezone conventions.
///
/// # Overnight gaps
///
/// RTH filtering removes pre-market and after-hours bars.  The resulting
/// series contains visible overnight gaps between the 15:59 bar of one
/// session and the 09:30 bar of the next.  These gaps are **not** hidden —
/// they are reported by the validation layer and are visible in the timestamp
/// sequence.  The online detector should reset state at session boundaries
/// or otherwise be made aware of them; that decision belongs to the
/// streaming phase, not the data-pipeline phase.
///
/// # Session labelling
///
/// `label_sessions` partitions an ascending-ordered observation sequence
/// into contiguous calendar-day groups.  Each group is one `SessionBoundary`.
/// For RTH-filtered data, each boundary corresponds to one trading day.
use chrono::{NaiveDate, NaiveTime};

use super::Observation;

// =========================================================================
// RTH filter
// =========================================================================

/// Return `true` iff the bar-open time falls within Regular Trading Hours.
///
/// The RTH window is **inclusive** of 09:30:00 and **exclusive** of 16:00:00.
pub fn is_rth_bar(ts: chrono::NaiveDateTime) -> bool {
    let t = ts.time();
    t >= NaiveTime::from_hms_opt(9, 30, 0).unwrap()
        && t < NaiveTime::from_hms_opt(16, 0, 0).unwrap()
}

/// Filter a slice of observations to Regular Trading Hours only.
///
/// Returns a new `Vec<Observation>` containing only bars whose open time
/// satisfies `09:30:00 ≤ time < 16:00:00`.
///
/// The input must already be sorted ascending (as guaranteed by
/// `CleanSeries`).  The output preserves the ascending order.
pub fn filter_rth(obs: &[Observation]) -> Vec<Observation> {
    obs.iter()
        .filter(|o| is_rth_bar(o.timestamp))
        .cloned()
        .collect()
}

// =========================================================================
// Session boundary
// =========================================================================

/// Extent of one contiguous trading session in an observation vector.
///
/// A session is defined as a maximal contiguous run of observations sharing
/// the same calendar date.  For RTH-filtered intraday data, one session
/// corresponds to one trading day.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionBoundary {
    /// 0-based session number within the series.
    pub session_index: usize,
    /// Calendar date of this session.
    pub date: NaiveDate,
    /// Index of the first observation in this session (in the parent
    /// `observations` vector).
    pub first_obs_idx: usize,
    /// Index of the last observation in this session (inclusive).
    pub last_obs_idx: usize,
}

/// Partition `obs` into sessions by calendar-date transitions.
///
/// `obs` must be sorted ascending (as guaranteed by `CleanSeries`).
///
/// Returns an empty `Vec` if `obs` is empty.
pub fn label_sessions(obs: &[Observation]) -> Vec<SessionBoundary> {
    if obs.is_empty() {
        return vec![];
    }
    let mut sessions = Vec::new();
    let mut session_idx = 0usize;
    let mut session_start = 0usize;
    let mut current_date: NaiveDate = obs[0].timestamp.date();

    for (i, o) in obs.iter().enumerate() {
        let date = o.timestamp.date();
        if date != current_date {
            sessions.push(SessionBoundary {
                session_index: session_idx,
                date: current_date,
                first_obs_idx: session_start,
                last_obs_idx: i - 1,
            });
            session_idx += 1;
            session_start = i;
            current_date = date;
        }
    }
    // Push the last (possibly only) session.
    sessions.push(SessionBoundary {
        session_index: session_idx,
        date: current_date,
        first_obs_idx: session_start,
        last_obs_idx: obs.len() - 1,
    });
    sessions
}

// =========================================================================
// Session-aware series
// =========================================================================

/// A `CleanSeries` annotated with explicit session boundaries.
///
/// Produced by `SessionAwareSeries::from_clean_series`.  The session
/// boundaries are consistent with the ascending observation order.
#[derive(Debug, Clone)]
pub struct SessionAwareSeries {
    pub series: super::CleanSeries,
    pub sessions: Vec<SessionBoundary>,
}

impl SessionAwareSeries {
    /// Label sessions by calendar-date transitions.
    pub fn from_clean_series(series: super::CleanSeries) -> Self {
        let sessions = label_sessions(&series.observations);
        Self { series, sessions }
    }

    /// Borrow the observations belonging to session `idx`.
    ///
    /// Returns `None` if `idx ≥ n_sessions`.
    pub fn session_obs(&self, idx: usize) -> Option<&[Observation]> {
        let s = self.sessions.get(idx)?;
        Some(&self.series.observations[s.first_obs_idx..=s.last_obs_idx])
    }

    pub fn n_sessions(&self) -> usize {
        self.sessions.len()
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Observation;
    use chrono::NaiveDateTime;

    fn obs(s: &str) -> Observation {
        Observation {
            timestamp: NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S").unwrap(),
            value: 1.0,
        }
    }

    // --- RTH filter ---------------------------------------------------------

    #[test]
    fn filter_rth_removes_premarket_bars() {
        // 09:00 and 09:29 are before the 09:30 open.
        let input = vec![obs("2024-01-02 09:00:00"), obs("2024-01-02 09:29:00")];
        assert!(filter_rth(&input).is_empty());
    }

    #[test]
    fn filter_rth_removes_after_hours_bars() {
        // 16:00 is the exclusive upper bound; 16:30 is deep after-hours.
        let input = vec![obs("2024-01-02 16:00:00"), obs("2024-01-02 16:30:00")];
        assert!(filter_rth(&input).is_empty());
    }

    #[test]
    fn filter_rth_keeps_open_bar() {
        // 09:30:00 is the inclusive lower bound.
        let input = vec![obs("2024-01-02 09:30:00")];
        assert_eq!(filter_rth(&input).len(), 1);
    }

    #[test]
    fn filter_rth_keeps_last_rth_bar_and_drops_close() {
        // 15:59 is the last RTH bar; 16:00 is excluded.
        let input = vec![obs("2024-01-02 15:59:00"), obs("2024-01-02 16:00:00")];
        let out = filter_rth(&input);
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0].timestamp,
            NaiveDateTime::parse_from_str("2024-01-02 15:59:00", "%Y-%m-%d %H:%M:%S").unwrap()
        );
    }

    // --- label_sessions -----------------------------------------------------

    #[test]
    fn label_sessions_single_day_produces_one_boundary() {
        let obs = vec![
            obs("2024-01-02 09:30:00"),
            obs("2024-01-02 09:35:00"),
            obs("2024-01-02 09:40:00"),
        ];
        let sessions = label_sessions(&obs);
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].first_obs_idx, 0);
        assert_eq!(sessions[0].last_obs_idx, 2);
    }

    #[test]
    fn label_sessions_two_days_produces_two_boundaries() {
        let obs = vec![
            obs("2024-01-02 09:30:00"),
            obs("2024-01-02 09:35:00"),
            obs("2024-01-03 09:30:00"), // new day
            obs("2024-01-03 09:35:00"),
        ];
        let sessions = label_sessions(&obs);
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].last_obs_idx, 1);
        assert_eq!(sessions[1].first_obs_idx, 2);
        assert_eq!(sessions[1].last_obs_idx, 3);
    }

    #[test]
    fn label_sessions_empty_is_ok() {
        let sessions = label_sessions(&[]);
        assert!(sessions.is_empty());
    }
}
