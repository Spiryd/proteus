/// Causal, pointwise financial feature transformations.
///
/// # Causality contract
///
/// Every function in this module is **causal**: it reads only the current
/// and past price values; it never looks ahead.  This is a hard requirement
/// for the online detection setting.
///
/// # Price-to-return mapping
///
/// Given a price series $P_0, P_1, \dots, P_n$:
///
/// $$r_t = \log P_t - \log P_{t-1}, \quad t = 1, \dots, n$$
///
/// The first defined log-return has index 1 (i.e., warmup = 1 price bar).
///
/// # Session-boundary contract
///
/// For intraday data, the caller is responsible for deciding whether
/// session-crossing returns are allowed.  If `allow_cross_session = false`,
/// the functions that take `prev_value` should receive `None` at each
/// session-open bar, which will cause that bar to be skipped (no return
/// emitted).
///
/// These session-cross decisions are made in the higher-level
/// `FeatureTransformer` in `stream.rs`; the primitives below are
/// session-agnostic.
use crate::data::Observation;

use chrono::NaiveDateTime;

// =========================================================================
// Single-step primitives
// =========================================================================

/// Compute one log return: $r_t = \log p_t - \log p_{t-1}$.
///
/// Returns `None` if either price is non-positive (guard against corrupt
/// vendor data; log of zero or negative is undefined).
#[inline]
pub fn log_return(prev_price: f64, curr_price: f64) -> Option<f64> {
    if prev_price <= 0.0 || curr_price <= 0.0 {
        None
    } else {
        Some(curr_price.ln() - prev_price.ln())
    }
}

/// Compute one absolute return: $a_t = |r_t|$.
///
/// Returns `None` if either price is non-positive.
#[inline]
pub fn abs_return(prev_price: f64, curr_price: f64) -> Option<f64> {
    log_return(prev_price, curr_price).map(f64::abs)
}

/// Compute one squared return: $q_t = r_t^2$.
///
/// Returns `None` if either price is non-positive.
#[inline]
pub fn squared_return(prev_price: f64, curr_price: f64) -> Option<f64> {
    log_return(prev_price, curr_price).map(|r| r * r)
}

// =========================================================================
// Batch transformations — price slice → feature observation slice
// =========================================================================

/// Apply a pointwise two-price function over a sorted price-observation slice.
///
/// `f(prev_price, curr_price) -> Option<f64>`: returning `None` skips the
/// output bar (warmup or invalid).
///
/// The output length is `obs.len().saturating_sub(1)` in the no-skip case.
/// Each output `Observation` inherits the *current* bar's timestamp.
fn apply_pairwise(obs: &[Observation], f: impl Fn(f64, f64) -> Option<f64>) -> Vec<Observation> {
    obs.windows(2)
        .filter_map(|w| {
            f(w[0].value, w[1].value).map(|v| Observation {
                timestamp: w[1].timestamp,
                value: v,
            })
        })
        .collect()
}

/// Compute log returns for every consecutive pair in `obs`.
///
/// Returns a series with `obs.len() - 1` observations (the first price bar
/// is consumed as the warm-up predecessor).  Bars with non-positive prices
/// are silently dropped (both the bar and the immediately following one).
pub fn log_returns(obs: &[Observation]) -> Vec<Observation> {
    apply_pairwise(obs, log_return)
}

/// Compute absolute returns for every consecutive pair in `obs`.
pub fn abs_returns(obs: &[Observation]) -> Vec<Observation> {
    apply_pairwise(obs, abs_return)
}

/// Compute squared returns for every consecutive pair in `obs`.
pub fn squared_returns(obs: &[Observation]) -> Vec<Observation> {
    apply_pairwise(obs, squared_return)
}

// =========================================================================
// Session-respecting pairwise transform
// =========================================================================

/// Like `apply_pairwise`, but refuses to form a return across a session
/// boundary.
///
/// `is_new_session(prev_ts, curr_ts)` returns `true` if the two timestamps
/// belong to different calendar sessions (e.g., different days for RTH
/// intraday data).  When it returns `true`, the pair is skipped — no return
/// is emitted for the current bar.
///
/// This is the correct policy for intraday log returns because the overnight
/// price gap is not a within-session return.
pub fn log_returns_session_aware(
    obs: &[Observation],
    is_new_session: impl Fn(NaiveDateTime, NaiveDateTime) -> bool,
) -> Vec<Observation> {
    obs.windows(2)
        .filter_map(|w| {
            if is_new_session(w[0].timestamp, w[1].timestamp) {
                None
            } else {
                log_return(w[0].value, w[1].value).map(|v| Observation {
                    timestamp: w[1].timestamp,
                    value: v,
                })
            }
        })
        .collect()
}

/// Absolute returns with session-boundary skip.
pub fn abs_returns_session_aware(
    obs: &[Observation],
    is_new_session: impl Fn(NaiveDateTime, NaiveDateTime) -> bool,
) -> Vec<Observation> {
    log_returns_session_aware(obs, is_new_session)
        .into_iter()
        .map(|o| Observation {
            value: o.value.abs(),
            ..o
        })
        .collect()
}

// =========================================================================
// Convenience: default session predicate (different calendar day)
// =========================================================================

/// `true` iff `prev` and `curr` lie on different calendar days.
///
/// Use this as the `is_new_session` predicate for RTH-filtered intraday
/// data, which has at most one session per calendar day.
pub fn different_day(prev: NaiveDateTime, curr: NaiveDateTime) -> bool {
    prev.date() != curr.date()
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDateTime;

    fn obs(ts_str: &str, value: f64) -> Observation {
        Observation {
            timestamp: NaiveDateTime::parse_from_str(ts_str, "%Y-%m-%d %H:%M:%S").unwrap(),
            value,
        }
    }

    // --- pointwise primitives ---

    #[test]
    fn log_return_correct() {
        // ln(110/100) ≈ 0.09531
        let r = log_return(100.0, 110.0).unwrap();
        let expected = (110f64).ln() - (100f64).ln();
        assert!((r - expected).abs() < 1e-12);
    }

    #[test]
    fn log_return_rejects_non_positive() {
        assert!(log_return(0.0, 100.0).is_none());
        assert!(log_return(100.0, 0.0).is_none());
        assert!(log_return(-1.0, 100.0).is_none());
    }

    #[test]
    fn abs_return_is_nonnegative() {
        // price drop
        let a = abs_return(110.0, 100.0).unwrap();
        let r = log_return(110.0, 100.0).unwrap();
        assert!((a - r.abs()).abs() < 1e-12);
        assert!(a >= 0.0);
    }

    #[test]
    fn squared_return_equals_log_return_squared() {
        let q = squared_return(100.0, 105.0).unwrap();
        let r = log_return(100.0, 105.0).unwrap();
        assert!((q - r * r).abs() < 1e-12);
    }

    // --- batch transformations ---

    #[test]
    fn log_returns_length_is_n_minus_one() {
        let obs_vec = vec![
            obs("2024-01-02 00:00:00", 100.0),
            obs("2024-01-03 00:00:00", 102.0),
            obs("2024-01-04 00:00:00", 101.0),
        ];
        let rets = log_returns(&obs_vec);
        assert_eq!(rets.len(), 2);
    }

    #[test]
    fn log_returns_timestamps_are_current_bars() {
        let obs_vec = vec![
            obs("2024-01-02 00:00:00", 100.0),
            obs("2024-01-03 00:00:00", 105.0),
        ];
        let rets = log_returns(&obs_vec);
        assert_eq!(rets[0].timestamp, obs_vec[1].timestamp);
    }

    #[test]
    fn log_returns_values_match_manual() {
        let obs_vec = vec![
            obs("2024-01-02 00:00:00", 100.0),
            obs("2024-01-03 00:00:00", 110.0),
        ];
        let rets = log_returns(&obs_vec);
        let expected = (110f64 / 100.0).ln();
        assert!((rets[0].value - expected).abs() < 1e-12);
    }

    // --- session-aware log returns ---

    #[test]
    fn session_aware_skips_cross_session_pair() {
        // Two bars on different days — should produce 0 returns.
        let obs_vec = vec![
            obs("2024-01-02 15:59:00", 100.0),
            obs("2024-01-03 09:30:00", 102.0),
        ];
        let rets = log_returns_session_aware(&obs_vec, different_day);
        assert_eq!(rets.len(), 0);
    }

    #[test]
    fn session_aware_keeps_within_session_pairs() {
        // Two bars on same day.
        let obs_vec = vec![
            obs("2024-01-02 09:30:00", 100.0),
            obs("2024-01-02 09:35:00", 101.0),
            obs("2024-01-02 09:40:00", 102.0),
        ];
        let rets = log_returns_session_aware(&obs_vec, different_day);
        assert_eq!(rets.len(), 2);
    }

    #[test]
    fn session_aware_two_day_sequence() {
        // Day 1: 2 bars (1 return); overnight gap (skip); Day 2: 2 bars (1 return).
        let obs_vec = vec![
            obs("2024-01-02 09:30:00", 100.0),
            obs("2024-01-02 09:35:00", 101.0),
            obs("2024-01-03 09:30:00", 101.5), // cross-session: skipped
            obs("2024-01-03 09:35:00", 103.0),
        ];
        let rets = log_returns_session_aware(&obs_vec, different_day);
        assert_eq!(rets.len(), 2);
        // Both returned bars are on distinct days.
        assert_eq!(rets[0].timestamp, obs_vec[1].timestamp);
        assert_eq!(rets[1].timestamp, obs_vec[3].timestamp);
    }

    #[test]
    fn different_day_predicate_correct() {
        let d1 = NaiveDateTime::parse_from_str("2024-01-02 15:59:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let d2 = NaiveDateTime::parse_from_str("2024-01-03 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap();
        let same =
            NaiveDateTime::parse_from_str("2024-01-02 09:30:00", "%Y-%m-%d %H:%M:%S").unwrap();
        assert!(different_day(d1, d2));
        assert!(!different_day(d1, same));
    }
}
