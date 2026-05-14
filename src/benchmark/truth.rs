/// Ground-truth changepoint representation.
///
/// A [`ChangePointTruth`] holds the ordered sequence of true changepoint
/// times τ₁ < τ₂ < … < τₘ and the total stream length T for a single
/// evaluation stream.  Time indices are **1-based** throughout, matching the
/// convention used by the online filter (`OnlineStepResult::t`).
///
/// For a simulated Markov Switching stream, a changepoint at time t means
/// `S_t ≠ S_{t-1}` — i.e. the latent regime changed entering observation t.
///
/// # No-change streams
///
/// A stream with no true changepoints is valid: construct with an empty `times`
/// vector.  All alarms on such a stream are false positives.
use anyhow::Result;

// =========================================================================
// ChangePointTruth
// =========================================================================

/// Ordered sequence of true changepoint times for one evaluation stream.
///
/// Time indices are 1-based and must satisfy `1 ≤ τ₁ < τ₂ < … < τₘ ≤ T`.
#[derive(Debug, Clone)]
pub struct ChangePointTruth {
    /// True changepoint times τ₁ < … < τₘ.  Empty for a no-change stream.
    pub times: Vec<usize>,
    /// Total stream length T (number of observations).
    pub stream_len: usize,
}

impl ChangePointTruth {
    /// Construct from a list of changepoint times and a stream length.
    ///
    /// Validates:
    /// 1. All times are in `[1, stream_len]`.
    /// 2. Times are strictly increasing (no duplicates).
    pub fn new(times: Vec<usize>, stream_len: usize) -> Result<Self> {
        for &t in &times {
            if t == 0 || t > stream_len {
                anyhow::bail!(
                    "ChangePointTruth: changepoint time {t} is out of range \
                     [1, {stream_len}]"
                );
            }
        }
        for w in times.windows(2) {
            if w[0] >= w[1] {
                anyhow::bail!(
                    "ChangePointTruth: times must be strictly increasing; \
                     found {} followed by {}",
                    w[0],
                    w[1]
                );
            }
        }
        Ok(Self { times, stream_len })
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truth_accepts_valid_times() {
        let t = ChangePointTruth::new(vec![10, 50, 90], 100).unwrap();
        assert_eq!(t.times.len(), 3);
    }

    #[test]
    fn truth_accepts_empty_no_change() {
        let t = ChangePointTruth::new(vec![], 200).unwrap();
        assert!(t.times.is_empty());
    }

    #[test]
    fn truth_rejects_out_of_bounds() {
        assert!(ChangePointTruth::new(vec![0], 100).is_err());
        assert!(ChangePointTruth::new(vec![101], 100).is_err());
    }

    #[test]
    fn truth_rejects_non_increasing() {
        assert!(ChangePointTruth::new(vec![20, 10], 100).is_err());
        assert!(ChangePointTruth::new(vec![20, 20], 100).is_err());
    }
}
