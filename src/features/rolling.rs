#![allow(dead_code)]
/// Causal, trailing-window rolling statistics for financial feature engineering.
///
/// # Trailing-window invariant (causality)
///
/// Every statistic produced here uses **only past and present** observations.
/// For window size $w$, at time index $t$ (0-based into the return series)
/// the computation uses $r_{t-w+1}, \dots, r_t$.
///
/// This is mandatory for an online-detection pipeline: centered or
/// future-looking windows would constitute look-ahead bias and would
/// invalidate detector evaluation.
///
/// # Rolling volatility definition
///
/// $$v_t^{(w)} = \sqrt{\frac{1}{w}\sum_{k=0}^{w-1}(r_{t-k} - \bar r_t^{(w)})^2}$$
///
/// where $\bar r_t^{(w)} = \frac{1}{w}\sum_{k=0}^{w-1} r_{t-k}$.
///
/// The normalisation is $1/w$ (not $1/(w-1)$): this is the *population*
/// standard deviation of the window, which is the natural choice when
/// treating the window as a finite-sample volatility proxy rather than
/// an unbiased sample-variance estimator.
///
/// # Incremental computation
///
/// The `RollingStats` struct maintains a ring buffer of the most recent `w`
/// values and updates in $O(w)$ time.  For large windows, a Welford-style
/// online update could reduce this to $O(1)$, but the ring-buffer approach
/// is simpler, numerically well-conditioned, and fast enough for the bar
/// counts found in typical financial time series.
///
/// # Session-reset semantics
///
/// When `session_reset = true` the accumulator is cleared at every
/// session-open bar.  This prevents overnight price gaps from entering
/// the rolling window and distorting intraday volatility estimates.  The
/// first `w` bars of each session are warmup; no feature value is emitted
/// until the window is full within the current session.
///
/// For daily data `session_reset` is irrelevant (every bar is already its
/// own "session") and should be ignored by callers.
use crate::data::Observation;

// =========================================================================
// RollingStats
// =========================================================================

/// Incremental trailing-window accumulator for mean and population variance.
///
/// Feed values one at a time with [`push`](Self::push).
/// Call [`vol`](Self::vol) to get the current rolling volatility $v_t^{(w)}$,
/// or [`mean`](Self::mean) to get $\bar r_t^{(w)}$.
///
/// `vol()` / `mean()` return `None` until the buffer contains exactly `w`
/// values (warmup phase).
pub struct RollingStats {
    window: usize,
    buf: Vec<f64>, // ring buffer, length ≤ window
    head: usize,   // next write position (mod window)
    full: bool,    // true once buf has been filled once
}

impl RollingStats {
    /// Create a new accumulator with the given window size.
    ///
    /// # Panics
    ///
    /// Panics if `window == 0`.
    pub fn new(window: usize) -> Self {
        assert!(window > 0, "rolling window must be > 0");
        Self {
            window,
            buf: vec![0.0; window],
            head: 0,
            full: false,
        }
    }

    /// Reset the accumulator to the empty state (used for session resets).
    pub fn reset(&mut self) {
        self.buf.fill(0.0);
        self.head = 0;
        self.full = false;
    }

    /// Push a new value into the window, evicting the oldest if full.
    pub fn push(&mut self, value: f64) {
        self.buf[self.head] = value;
        self.head = (self.head + 1) % self.window;
        if !self.full && self.head == 0 {
            self.full = true;
        }
    }

    /// Number of values currently in the buffer (0..=window).
    pub fn count(&self) -> usize {
        if self.full { self.window } else { self.head }
    }

    /// `true` iff the buffer contains exactly `window` values (warmup done).
    pub fn is_ready(&self) -> bool {
        self.full
    }

    /// Rolling mean $\bar r_t^{(w)}$.
    ///
    /// Returns `None` during warmup.
    pub fn mean(&self) -> Option<f64> {
        if !self.full {
            return None;
        }
        Some(self.buf.iter().sum::<f64>() / self.window as f64)
    }

    /// Rolling population standard deviation $v_t^{(w)}$.
    ///
    /// Returns `None` during warmup.
    pub fn vol(&self) -> Option<f64> {
        let m = self.mean()?;
        let var = self.buf.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / self.window as f64;
        Some(var.sqrt())
    }

    /// Current slice of buffered values (length = `count()`).
    /// Intended for testing; not required for normal feature computation.
    #[cfg(test)]
    pub fn values(&self) -> Vec<f64> {
        if self.full {
            // rotate so index 0 is oldest
            let mut v = Vec::with_capacity(self.window);
            v.extend_from_slice(&self.buf[self.head..]);
            v.extend_from_slice(&self.buf[..self.head]);
            v
        } else {
            self.buf[..self.head].to_vec()
        }
    }
}

// =========================================================================
// Batch rolling-volatility computation
// =========================================================================

/// Compute the rolling volatility feature series from a return series.
///
/// # Arguments
///
/// * `returns` — causal log-return observations, ascending by timestamp.
/// * `window`  — rolling window size $w$.
///
/// # Returns
///
/// A series of `Observation` values where each value is $v_t^{(w)}$.
/// The first `window - 1` return bars are consumed as warmup; the output
/// length is `returns.len() - (window - 1)` (when every return is
/// finite; bars producing `NaN` or `Inf` are skipped).
///
/// # Causality
///
/// Each output value at timestamp $t$ depends only on returns at or before
/// time $t$.
pub fn rolling_vol(returns: &[Observation], window: usize) -> Vec<Observation> {
    assert!(window > 0, "window must be > 0");
    let mut stats = RollingStats::new(window);
    let mut out = Vec::with_capacity(returns.len().saturating_sub(window - 1));
    for r in returns {
        if r.value.is_finite() {
            stats.push(r.value);
        }
        if let Some(v) = stats.vol() {
            out.push(Observation {
                timestamp: r.timestamp,
                value: v,
            });
        }
    }
    out
}

/// Compute the standardized return series $z_t = r_t / (v_t^{(w)} + \varepsilon)$.
///
/// # Arguments
///
/// * `returns` — causal log-return observations.
/// * `window`  — rolling window size for the denominator volatility.
/// * `epsilon` — numerical stabilizer (add to denominator to avoid divide-by-zero).
///
/// # Warmup
///
/// The first `window - 1` bars are warmup for the rolling vol; the first
/// defined output aligns with the `window`-th return bar.
///
/// # Causality
///
/// $v_t^{(w)}$ is estimated from returns $r_{t-w+1}, \dots, r_t$ (the
/// current bar is included in the vol estimate before dividing, which is
/// causal but note: $r_t$ appears in both numerator and denominator).
pub fn standardized_returns(
    returns: &[Observation],
    window: usize,
    epsilon: f64,
) -> Vec<Observation> {
    assert!(window > 0, "window must be > 0");
    let mut stats = RollingStats::new(window);
    let mut out = Vec::with_capacity(returns.len().saturating_sub(window - 1));
    for r in returns {
        if r.value.is_finite() {
            stats.push(r.value);
        }
        if let Some(v) = stats.vol() {
            let z = r.value / (v + epsilon);
            out.push(Observation {
                timestamp: r.timestamp,
                value: z,
            });
        }
    }
    out
}

// =========================================================================
// Session-reset variants
// =========================================================================

/// Rolling volatility with optional session reset.
///
/// `is_new_session(prev_ts, curr_ts)` → `true` iff the two timestamps
/// belong to different sessions.  When a session boundary is detected the
/// accumulator is cleared before processing the current bar, so the first
/// `window` bars of each session are warmup.
///
/// For daily data, pass `|_, _| false` (never reset).
pub fn rolling_vol_session_aware(
    returns: &[Observation],
    window: usize,
    is_new_session: impl Fn(chrono::NaiveDateTime, chrono::NaiveDateTime) -> bool,
) -> Vec<Observation> {
    assert!(window > 0);
    let mut stats = RollingStats::new(window);
    let mut out = Vec::with_capacity(returns.len().saturating_sub(window - 1));
    let mut prev_ts: Option<chrono::NaiveDateTime> = None;
    for r in returns {
        if let Some(prev) = prev_ts
            && is_new_session(prev, r.timestamp)
        {
            stats.reset();
        }
        prev_ts = Some(r.timestamp);
        if r.value.is_finite() {
            stats.push(r.value);
        }
        if let Some(v) = stats.vol() {
            out.push(Observation {
                timestamp: r.timestamp,
                value: v,
            });
        }
    }
    out
}

/// Standardized returns with optional session reset for the vol estimator.
pub fn standardized_returns_session_aware(
    returns: &[Observation],
    window: usize,
    epsilon: f64,
    is_new_session: impl Fn(chrono::NaiveDateTime, chrono::NaiveDateTime) -> bool,
) -> Vec<Observation> {
    assert!(window > 0);
    let mut stats = RollingStats::new(window);
    let mut out = Vec::with_capacity(returns.len().saturating_sub(window - 1));
    let mut prev_ts: Option<chrono::NaiveDateTime> = None;
    for r in returns {
        if let Some(prev) = prev_ts
            && is_new_session(prev, r.timestamp)
        {
            stats.reset();
        }
        prev_ts = Some(r.timestamp);
        if r.value.is_finite() {
            stats.push(r.value);
        }
        if let Some(v) = stats.vol() {
            let z = r.value / (v + epsilon);
            out.push(Observation {
                timestamp: r.timestamp,
                value: z,
            });
        }
    }
    out
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Observation;
    use chrono::NaiveDateTime;

    fn obs(ts_str: &str, value: f64) -> Observation {
        Observation {
            timestamp: NaiveDateTime::parse_from_str(ts_str, "%Y-%m-%d %H:%M:%S").unwrap(),
            value,
        }
    }

    // ---- RollingStats unit tests ----

    #[test]
    fn rolling_stats_not_ready_before_full() {
        let mut rs = RollingStats::new(3);
        rs.push(1.0);
        rs.push(2.0);
        assert!(!rs.is_ready());
        assert!(rs.vol().is_none());
        assert!(rs.mean().is_none());
    }

    #[test]
    fn rolling_stats_ready_after_window_values() {
        let mut rs = RollingStats::new(3);
        rs.push(1.0);
        rs.push(2.0);
        rs.push(3.0);
        assert!(rs.is_ready());
        // mean should be 2.0
        assert!((rs.mean().unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn rolling_stats_vol_manual_window3() {
        // values: 1, 2, 3 → mean=2 → var=(1+0+1)/3 = 2/3 → std=sqrt(2/3)
        let mut rs = RollingStats::new(3);
        rs.push(1.0);
        rs.push(2.0);
        rs.push(3.0);
        let expected = (2.0f64 / 3.0).sqrt();
        assert!((rs.vol().unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn rolling_stats_evicts_oldest() {
        let mut rs = RollingStats::new(3);
        rs.push(1.0);
        rs.push(2.0);
        rs.push(3.0);
        rs.push(4.0); // 1 should be evicted
        let m = rs.mean().unwrap();
        // window now [2,3,4] → mean=3
        assert!((m - 3.0).abs() < 1e-12);
    }

    #[test]
    fn rolling_stats_reset_clears_state() {
        let mut rs = RollingStats::new(3);
        rs.push(1.0);
        rs.push(2.0);
        rs.push(3.0);
        assert!(rs.is_ready());
        rs.reset();
        assert!(!rs.is_ready());
        assert_eq!(rs.count(), 0);
    }

    #[test]
    fn rolling_stats_constant_series_has_zero_vol() {
        let mut rs = RollingStats::new(4);
        for _ in 0..4 {
            rs.push(5.0);
        }
        assert_eq!(rs.vol().unwrap(), 0.0);
    }

    // ---- Batch rolling_vol ----

    #[test]
    fn rolling_vol_output_length() {
        // 5 returns, window=3 → 3 outputs (bars 2,3,4 in 0-index)
        let rets: Vec<Observation> = (0..5)
            .map(|i| obs(&format!("2024-01-0{} 00:00:00", i + 2), 0.01))
            .collect();
        let vols = rolling_vol(&rets, 3);
        assert_eq!(vols.len(), 3);
    }

    #[test]
    fn rolling_vol_trailing_window_matches_manual() {
        // returns: [0.01, 0.02, 0.03], window=3
        // mean = 0.02, var = ((−0.01)²+0²+0.01²)/3 = 2e-4/3 ≈ 6.667e-5
        let rets = vec![
            obs("2024-01-02 00:00:00", 0.01),
            obs("2024-01-03 00:00:00", 0.02),
            obs("2024-01-04 00:00:00", 0.03),
        ];
        let vols = rolling_vol(&rets, 3);
        assert_eq!(vols.len(), 1);
        let expected = {
            let mean = 0.02f64;
            let var = ((0.01 - mean).powi(2) + (0.02 - mean).powi(2) + (0.03 - mean).powi(2)) / 3.0;
            var.sqrt()
        };
        assert!((vols[0].value - expected).abs() < 1e-12);
    }

    #[test]
    fn rolling_vol_timestamp_matches_last_window_bar() {
        let rets = vec![
            obs("2024-01-02 00:00:00", 0.01),
            obs("2024-01-03 00:00:00", 0.02),
            obs("2024-01-04 00:00:00", 0.03),
        ];
        let vols = rolling_vol(&rets, 3);
        assert_eq!(vols[0].timestamp, rets[2].timestamp);
    }

    // ---- Session-aware rolling_vol ----

    #[test]
    fn session_reset_clears_window_at_session_start() {
        // 3 bars day1, 3 bars day2; window=3 with session_reset=true
        // Expect: 1 vol on day1 (bars 0,1,2), 1 vol on day2 (bars 0,1,2)
        let rets = vec![
            obs("2024-01-02 09:30:00", 0.01),
            obs("2024-01-02 09:35:00", 0.02),
            obs("2024-01-02 09:40:00", 0.01),
            obs("2024-01-03 09:30:00", 0.01), // session reset
            obs("2024-01-03 09:35:00", 0.02),
            obs("2024-01-03 09:40:00", 0.01),
        ];
        let vols = rolling_vol_session_aware(&rets, 3, |p, c| p.date() != c.date());
        // exactly 2 outputs — one at end of each session's 3-bar window
        assert_eq!(vols.len(), 2);
    }

    #[test]
    fn no_session_reset_accumulates_across_days() {
        // Same data but no reset — continuous rolling window
        let rets = vec![
            obs("2024-01-02 09:30:00", 0.01),
            obs("2024-01-02 09:35:00", 0.02),
            obs("2024-01-02 09:40:00", 0.01),
            obs("2024-01-03 09:30:00", 0.01),
            obs("2024-01-03 09:35:00", 0.02),
            obs("2024-01-03 09:40:00", 0.01),
        ];
        let vols = rolling_vol_session_aware(&rets, 3, |_, _| false);
        // With no reset, output from bar 2 onward → 4 outputs
        assert_eq!(vols.len(), 4);
    }

    // ---- Standardized returns ----

    #[test]
    fn standardized_return_formula_manual() {
        // 3 returns all equal to 0.01: vol=0, standardized → r/(0+ε)
        let rets = vec![
            obs("2024-01-02 00:00:00", 0.01),
            obs("2024-01-03 00:00:00", 0.01),
            obs("2024-01-04 00:00:00", 0.01),
        ];
        let eps = 1e-8;
        let z = standardized_returns(&rets, 3, eps);
        assert_eq!(z.len(), 1);
        let expected = 0.01 / (0.0 + eps);
        assert!((z[0].value - expected).abs() < 1.0); // large but finite
    }

    #[test]
    fn standardized_returns_warmup_matches_window() {
        let rets: Vec<Observation> = (0..5)
            .map(|i| {
                obs(
                    &format!("2024-01-0{} 00:00:00", i + 2),
                    0.01 * (i as f64 + 1.0),
                )
            })
            .collect();
        let z = standardized_returns(&rets, 4, 1e-8);
        assert_eq!(z.len(), 2); // 5 - (4-1) = 2
    }
}
