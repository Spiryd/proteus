# Online Benchmarking Protocol for Markov-Switching Changepoint Detectors

## Phase 14

---

## 1. Why an Event-Level Protocol Is Required

Online changepoint detection is **not** a pointwise binary classification problem.
At each time step $t$, a classifier would independently decide "change" or "no
change." A changepoint detector instead emits a sparse sequence of alarm events
in continuous time, and those alarms must be interpreted relative to the positions
of true changepoints.

This distinction has immediate evaluation consequences:

- A single true changepoint at $\tau$ may trigger several consecutive alarms;
  only the first should be credited as a detection.
- An alarm at time $a$ that is close to but strictly before $\tau$ may still
  represent predictive early warning — or it may be a false alarm if it occurs
  before any nearby changepoint.
- Two true changepoints that are close together may produce overlapping detection
  windows and require an unambiguous matching policy.

For these reasons, the benchmark protocol operates at the **event level**: it
matches an ordered set of true changepoint times against an ordered set of alarm
times, applies a principled matching rule, and derives evaluation metrics from
the resulting pairing.

The central principle is:

$$
\boxed{\text{Online changepoint detectors must be evaluated at the event level, not just by vague alarm counts.}}
$$

---

## 2. Formal Setup

### 2.1 Ground-truth changepoints

For a single stream of length $T$, the true changepoints form an ordered finite
set:

$$
\mathcal{C}^\star = \{\tau_1, \tau_2, \dots, \tau_M\}, \quad \tau_1 < \tau_2 < \cdots < \tau_M,
\quad \tau_m \in \{1, \dots, T\}.
$$

Each $\tau_m$ is the **first time index** at which the generating process has
changed. For a simulated Markov Switching stream with regime sequence
$(S_1, \dots, S_T)$, a changepoint occurs at time $t$ iff $S_t \neq S_{t-1}$:

$$
\mathcal{C}^\star = \bigl\{ t \in \{2, \dots, T\} : S_t \neq S_{t-1} \bigr\}.
$$

A stream with $M = 0$ (no true changepoints) is a valid benchmark stream; all
alarms on such a stream are by definition false positives.

### 2.2 Detector alarms

The detector produces an ordered sequence of alarm events:

$$
\mathcal{A} = \{a_1, a_2, \dots, a_N\}, \quad a_1 \leq a_2 \leq \cdots \leq a_N,
\quad a_n \in \{1, \dots, T\}.
$$

Each $a_n$ is the time index at which the detector fired. The alarm sequence is
produced causally: $a_n$ depends only on $y_1, \dots, y_{a_n}$.

### 2.3 Causal constraint

An alarm $a_n$ can be credited for detecting $\tau_m$ only if $a_n \geq \tau_m$.
Alarms before a changepoint cannot be retroactively credited as detections of
that changepoint:

$$
\text{alarm } a_n \text{ is eligible for } \tau_m \iff a_n \geq \tau_m.
$$

---

## 3. The Detection-Window Matching Rule

### 3.1 Detection window

For each true changepoint $\tau_m$, define an **allowed detection window**:

$$
W_m = [\tau_m,\; \tau_m + w),
$$

where $w \geq 1$ is the **window width** (measured in observations). The window
is half-open: alarm $a_n$ matches $\tau_m$ iff

$$
\tau_m \leq a_n < \tau_m + w.
$$

The window width $w$ encodes the maximum allowed detection delay: an alarm
arriving after $\tau_m + w - 1$ is too late to count as a detection of $\tau_m$.

### 3.2 Greedy chronological matching

Matching proceeds by processing changepoints in ascending order $\tau_1, \tau_2,
\dots, \tau_M$ and greedily assigning the earliest unused alarm within each
window:

> For changepoint $\tau_m$, find the earliest unused alarm $a_n \in W_m$.
> If found: assign the pair $(\tau_m, a_n)$ as a match and mark $a_n$ as used.
> If not found: $\tau_m$ is a miss.

**One alarm per changepoint:** only the first eligible alarm in the window counts
as a detection; later alarms in the same window remain unmatched (and may become
false positives).

**One changepoint per alarm:** each alarm is assigned to at most one changepoint.

**Causal matching only:** alarms before $\tau_m$ are ineligible regardless of
proximity.

These four policies together constitute the **causal greedy first-match rule**.

### 3.3 Window width guidance

| Setting | Typical $w$ | Effect |
|---|---|---|
| Fast-reaction regime | 1–5 | Rewards only very rapid detection |
| Standard online | 10–50 | Balanced tolerance for detection delay |
| Slow-acting detector | 50–100 | Permits late but eventual detection |

In thesis comparisons, state $w$ explicitly in every table and plot. Reporting
results at multiple values of $w$ provides a richer picture.

---

## 4. Event Taxonomy

After matching, every true changepoint and every alarm receives exactly one label.

### 4.1 Changepoint outcomes

| Outcome | Symbol | Definition |
|---|---|---|
| Detected | TP$_\tau$ | $\exists$ matched alarm in $W_m$ |
| Missed | FN$_\tau$ | No alarm in $W_m$ |

Let $M_{\text{det}}$ = number of detected changepoints, $M_{\text{miss}} = M - M_{\text{det}}$.

### 4.2 Alarm outcomes

| Outcome | Symbol | Definition |
|---|---|---|
| True positive | TP$_a$ | First matched alarm for some $\tau_m$ |
| False positive | FP$_a$ | Not matched to any changepoint |

Note: TP$_\tau$ = TP$_a$ = number of matched pairs. There is no notion of "true
negative" at the alarm level because alarms are sparse events, not per-step
decisions.

---

## 5. Evaluation Metrics

### 5.1 Detection delay

For each matched pair $(\tau_m, a_m)$, the **detection delay** is:

$$
d_m = a_m - \tau_m \geq 0.
$$

Summary statistics over all matched delays:

$$
\bar{d} = \frac{1}{M_{\text{det}}} \sum_{m:\text{detected}} d_m, \qquad
\tilde{d} = \text{median}\{d_m\}.
$$

A detector that achieves high recall but with large $\bar{d}$ may be unsuitable
for time-critical applications.

### 5.2 Miss rate

$$
\text{Miss Rate} = \frac{M_{\text{miss}}}{M} = 1 - \frac{M_{\text{det}}}{M}.
$$

Undefined (set to $\text{NaN}$) for no-change streams ($M = 0$).

### 5.3 Recall

$$
\text{Recall} = \frac{M_{\text{det}}}{M} = 1 - \text{Miss Rate}.
$$

Undefined for $M = 0$.

### 5.4 Precision

$$
\text{Precision} = \frac{\text{TP}_a}{N}.
$$

Undefined ($\text{NaN}$) when $N = 0$ (detector emitted no alarms).

### 5.5 False alarm rate

$$
\text{FAR} = \frac{\text{FP}_a}{T}.
$$

Reports false positives per observation. Multiply by 1000 for "per 1000
observations." Defined for all stream lengths $T \geq 1$.

### 5.6 Run length to first false alarm

Let $a_{\text{FA}}$ be the time index of the first false-positive alarm:

$$
\text{RFFA} = a_{\text{FA}}.
$$

If no false alarm occurs in the stream, the run contributes no sample to the
aggregate. For no-change streams, every alarm is a false alarm, and RFFA is
especially meaningful.

### 5.7 Computational cost per observation

For a streaming run of $T$ steps with total elapsed time $\Delta t$:

$$
\bar{c} = \frac{\Delta t}{T} \quad \text{(mean step time)}.
$$

This is the primary runtime metric for fair comparison of detectors with
different algorithmic complexity.

---

## 6. No-Change Streams

A no-change stream ($M = 0$) is an important evaluation scenario because it tests
the detector's false-alarm behavior under a stable process.

On such streams:
- every alarm is a false positive,
- recall and miss rate are undefined ($\text{NaN}$),
- FAR and RFFA are the primary metrics.

The benchmark handles empty $\mathcal{C}^\star$ as a valid input and propagates
$\text{NaN}$ for metrics undefined in this case.

---

## 7. Handling Multiple Runs

The benchmark runs the same detector on many streams (typically simulated) and
aggregates results.

### 7.1 Per-run result

Each run produces:
- a `MatchResult` (event-level labels),
- a `MetricSuite` (per-stream metric values),
- an optional `TimingSummary` (runtime),
- `StreamMeta` (detector ID, scenario ID, stream index).

### 7.2 Aggregate result

Across $R$ runs, per-metric values are collected into a `Summary` (mean, median,
min, max). `NaN` values are excluded before computing the summary; `Summary::n`
records how many finite values contributed.

Summary statistics computed for each metric:

| Metric | Aggregation |
|---|---|
| Recall | mean, median over runs with $M > 0$ |
| Precision | mean, median over runs with $N > 0$ |
| Miss rate | mean, median over runs with $M > 0$ |
| FAR | mean, median over all runs |
| Detection delay | mean and median over runs with at least one detection |
| RFFA | mean, median over runs with at least one false alarm |
| Mean step time | mean, median over timed runs |

---

## 8. Benchmark Policies

The following policies are fixed for this implementation and should be stated
explicitly in any thesis section that reports these results:

| Policy | Rule |
|---|---|
| Causal matching only | Alarms before $\tau_m$ are ineligible |
| First-match rule | Only the first alarm in $W_m$ counts as TP |
| One alarm per changepoint | Later alarms in $W_m$ are not additional TPs |
| One changepoint per alarm | Each alarm matched at most once |
| Explicit window | $w$ is always stated in tables and plots |
| $\text{NaN}$ propagation | Metrics undefined for $M=0$ or $N=0$ return NaN |
| NaN exclusion in aggregates | NaN values are excluded before computing summaries |

---

## 9. Module Structure

```
src/benchmark/
    mod.rs          ← module declarations and re-exports
    truth.rs        ← ChangePointTruth, StreamMeta
    matching.rs     ← MatchConfig, EventMatcher, MatchResult,
                       AlarmOutcome, ChangePointOutcome
    metrics.rs      ← MetricSuite, Summary
    result.rs       ← RunResult, AggregateResult, TimingSummary, BenchmarkLabel
```

The `benchmark` module depends on `detector::AlarmEvent` (read-only) and
`benchmark::truth::ChangePointTruth`. It has no dependency on the online filter,
EM, or smoother modules.

---

## 10. Canonical Benchmark Workflow

```rust
use crate::benchmark::{
    ChangePointTruth, EventMatcher, MatchConfig, MetricSuite,
    RunResult, AggregateResult, StreamMeta, TimingSummary,
};
use crate::detector::{FrozenModel, StreamingSession, HardSwitchDetector};
use crate::online::OnlineFilterState;
use std::time::Instant;

// --- One stream ---
fn run_one(
    frozen: &FrozenModel,
    obs: &[f64],
    truth: ChangePointTruth,
    meta: StreamMeta,
) -> RunResult {
    let filter = OnlineFilterState::new(frozen.params());
    let det    = HardSwitchDetector::default();
    let mut session = StreamingSession::new(frozen.clone(), filter, det);

    let mut alarms = Vec::new();
    let t0 = Instant::now();
    for &y in obs {
        let out = session.step(y).unwrap();
        if let Some(ev) = out.detector.alarm_event {
            alarms.push(ev);
        }
    }
    let timing = TimingSummary::new(t0.elapsed(), obs.len());

    let mr = EventMatcher::new(MatchConfig { window: 20 })
        .match_events(&truth, &alarms);
    RunResult::new(mr, Some(timing), meta)
}

// --- Aggregate over N streams ---
let runs: Vec<RunResult> = (0..100)
    .map(|i| /* simulate + run_one */ )
    .collect();
let agg = AggregateResult::from_runs(&runs);
```

---

## 11. Runtime Invariants

### Matching invariants
- `n_true_positive + n_false_positive == n_alarms` (alarm partition is complete).
- `n_detected + n_missed == n_changepoints` (changepoint partition is complete).
- Every matched alarm satisfies `alarm_t >= tau` (causal constraint).
- `delay >= 0` for every matched pair.

### Metric invariants
- `recall + miss_rate == 1.0` (when $M > 0$).
- `precision ∈ [0, 1]` (when $N > 0$).
- `false_alarm_rate ∈ [0, 1]`.
- `delay.min <= delay.median <= delay.max`.

---

## 12. Tests

| Module | Test | Invariant verified |
|---|---|---|
| `truth` | `accepts_valid_times` | Valid construction |
| `truth` | `accepts_empty_no_change` | $M=0$ is legal |
| `truth` | `rejects_out_of_bounds` | Times in $[1,T]$ |
| `truth` | `rejects_non_increasing` | Strict ordering |
| `truth` | `from_regime_sequence_correct` | Change at $S_t \neq S_{t-1}$ |
| `truth` | `from_regime_sequence_no_change` | No-change stream |
| `matching` | `exact_match_single_changepoint` | delay=0, TP=1, FP=0 |
| `matching` | `match_within_window` | delay=5, TP=1 |
| `matching` | `alarm_outside_window_is_false_positive` | FP=1, miss=1 |
| `matching` | `alarm_before_changepoint_is_false_positive` | Causal constraint |
| `matching` | `multiple_alarms_one_changepoint_only_first_counts` | First-match rule |
| `matching` | `multiple_changepoints_matched_in_order` | Chronological matching |
| `matching` | `no_change_stream_all_alarms_false_positive` | $M=0$ case |
| `matching` | `no_alarms_all_changepoints_missed` | $N=0$ case |
| `matching` | `window_boundary_inclusive_start_exclusive_end` | $[τ, τ+w)$ semantics |
| `metrics` | `perfect_detection_precision_recall_one` | Precision=Recall=1 |
| `metrics` | `no_alarms_recall_zero_and_miss_rate_one` | NaN precision when $N=0$ |
| `metrics` | `only_false_alarms_precision_zero` | Precision=0 |
| `metrics` | `false_alarm_rate_correct` | FAR = FP/T |
| `metrics` | `first_false_alarm_none_when_no_fp` | RFFA=None |
| `metrics` | `first_false_alarm_correct_time` | RFFA value |
| `metrics` | `delay_summary_nan_when_nothing_detected` | NaN delay |
| `metrics` | `delay_summary_correct_values` | mean/min/max |
| `metrics` | `no_change_stream_nan_recall_and_miss_rate` | NaN recall |
| `result` | `run_result_metrics_populated` | RunResult construction |
| `result` | `aggregate_from_two_perfect_runs` | mean recall/precision=1 |
| `result` | `aggregate_nan_excluded_from_summaries` | NaN exclusion |
| `result` | `timing_summary_mean_step` | mean_step = total/n |

Total: **29 tests** in the `benchmark` module.

---

## 13. What This Phase Does Not Do

- **Threshold calibration** — choosing $w$ or alarm threshold $\tau$ to achieve a
  target FAR. This requires calibration studies on held-out data.
- **Conventional baseline implementations** — implementing BOCPD, CUSUM, or other
  comparison methods. The protocol defined here can receive any detector's alarm
  sequence.
- **Adaptive online learning** — the benchmark evaluates fixed-parameter detectors.
- **Final experiment conclusions** — experimental comparisons and result
  interpretation are deferred to later phases.
