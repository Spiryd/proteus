# Phase 18 — Real-Data Evaluation Protocol (Proxy Events + Self-Consistency Only)

## Goal

In this phase, define how to evaluate the Markov-Switching-based online changepoint detector on **real financial data**, where true changepoints are not directly observed.

On synthetic data, evaluation was clean because you had:
- known hidden regimes,
- known changepoint times,
- direct event-level benchmarking.

On real market data, that is no longer available.

So the central question becomes:

> What does it mean for an online changepoint detector to perform well on real financial series when the true changepoints are unknown?

For this thesis, the real-data evaluation protocol will use **only** the following two routes:

## Route A — Proxy event evaluation
Compare detector alarms against externally meaningful market events or empirically defined stress episodes.

## Route B — Self-consistency / segmentation evaluation
Evaluate whether the detector’s changepoints split the time series into segments that are statistically different in meaningful financial ways.

You are **not** using Route C (downstream predictive usefulness) in this phase.

That restriction is good: it keeps the thesis tighter and the evaluation more interpretable.

---

## 1. Why this phase is necessary

A detector that works well on synthetic data may still be unconvincing on real market data if you do not define a proper real-data evaluation framework.

Without this phase, you could only show:
- alarm timestamps,
- score curves,
- posterior regime plots.

Those are useful, but not enough for a thesis-grade evaluation.

You need a protocol that answers:

- Did alarms occur near meaningful real-world market disruptions?
- Do detected changepoints correspond to statistically different market regimes?
- Are the resulting segments interpretable as distinct financial states?

This phase gives you that evaluation layer.

---

## 2. Why Route A + B is a strong thesis choice

Using only A and B is a good methodological decision.

## Route A gives external relevance
It checks whether the detector reacts near events that a human or financial analyst would recognize as structurally important.

## Route B gives internal statistical coherence
It checks whether the detector’s segmentation produces statistically meaningful distinctions in the observed process.

Together they give two complementary viewpoints:

- **external alignment** with market-relevant events,
- **internal consistency** of the segmentation itself.

This is a strong and defensible evaluation strategy for a thesis on online changepoint detection in financial series.

---

# Part I — Route A: Proxy Event Evaluation

## 3. What proxy event evaluation means

Proxy event evaluation does **not** claim that external events are the exact ground truth changepoints.

Instead, it treats certain real-world events or episodes as **reference anchors**.

Examples include:
- crisis dates,
- volatility spikes,
- major commodity shocks,
- market stress windows,
- macro-sensitive episodes if you later choose to include them.

The goal is not to say:

> “The true changepoint occurred exactly at this timestamp.”

Instead, the goal is:

> “If the detector is meaningful, it should often react near these externally important market disruptions.”

So proxy events are **evaluation anchors**, not perfect labels.

---

## 4. Proxy event representation

For each real-data experiment, define a set of proxy events:

\[
\mathcal{E} = \{e_1, e_2, \dots, e_M\}.
\]

Each \(e_m\) should contain at least:
- timestamp or date,
- event type,
- asset scope if relevant,
- optional event window,
- optional metadata or textual label.

In some cases, a proxy may be a **point event**:
\[
e_m = \tau_m.
\]

In other cases, it may be an **event window**:
\[
e_m = [\ell_m, u_m].
\]

For real financial data, windows are often more appropriate than points because market stress can develop over several observations rather than one exact instant.

---

## 5. Types of proxy events you should support

For your thesis, proxy events should come from two broad sources.

## 5.1 Externally specified market events
These are dates or windows identified from known market history, for example:
- major selloffs,
- turbulence episodes,
- commodity dislocations,
- broad ETF stress episodes.

These are useful because they are interpretable and easy to explain in the thesis.

## 5.2 Empirically defined event windows
These are events extracted from the data itself using simple external summary rules, for example:
- top volatility days,
- days where realized volatility exceeds a threshold,
- intraday blocks with extreme return magnitude.

These are still external to the detector, even though they are data-derived, because they are defined independently of the MS detector logic.

This is often the most scalable route for your datasets.

---

## 6. Point-event vs window-event evaluation

You should explicitly support both.

## Point events
Used when a real-world event has a meaningful focal date:
\[
\tau_m
\]

## Window events
Used when the effect is distributed over time:
\[
[\ell_m, u_m]
\]

For financial data, window-based proxy evaluation is often better because:
- markets can anticipate events,
- reactions can unfold over several periods,
- EOD and intraday timestamps may not align perfectly with economic reality.

So in the first thesis version, I recommend treating proxy evaluation primarily as a **window alignment problem**.

---

## 7. Alarm-to-event alignment rule

You need a formal rule for when an alarm is considered aligned with a proxy event.

Let the alarm sequence be:

\[
\mathcal{A} = \{a_1, a_2, \dots, a_N\}.
\]

For each proxy event \(e_m\), define an evaluation window.

### If the proxy is already a window
\[
e_m = [\ell_m, u_m]
\]

then an alarm \(a_n\) is aligned if:

\[
a_n \in [\ell_m, u_m].
\]

### If the proxy is a point
\[
e_m = \tau_m
\]

then define a tolerance window:

\[
[\tau_m - w^{-}, \tau_m + w^{+}]
\]

and an alarm is aligned if it falls inside that window.

Because your detector is online, you may also choose a stricter causal variant:

\[
[\tau_m, \tau_m + w^{+}]
\]

which says the event is only considered “detected” if the alarm occurs at or after the event.

Both should be documented clearly if used.

---

## 8. What proxy-event evaluation should measure

The proxy-event layer should compute at least:

## 8.1 Event coverage
How many proxy events had at least one aligned alarm?

\[
\text{Coverage}
=
\frac{\text{number of proxy events with aligned alarms}}{\text{number of proxy events}}
\]

This is the event-level analogue of recall.

---

## 8.2 Alarm relevance
How many alarms align with any proxy event?

\[
\text{Alarm Relevance}
=
\frac{\text{number of alarms aligned with at least one proxy event}}{\text{number of alarms}}
\]

This is similar in spirit to precision, but since proxy events are imperfect anchors, it should be interpreted cautiously.

---

## 8.3 Event delay
For point events or event starts, define the delay of the first aligned alarm relative to the event reference time.

If \(a(e_m)\) is the first aligned alarm for event \(e_m\), then:

\[
d_m = a(e_m) - \tau_m.
\]

This is especially useful for:
- point-like events,
- start-of-window analysis,
- causal interpretation.

---

## 8.4 Alarm density around events
You may also compute how alarm frequency changes near proxy events compared with normal periods.

This gives a softer and often more robust signal of whether the detector is reacting to relevant market episodes.

---

## 9. Important interpretation rule for proxy events

Proxy-event evaluation should be described carefully in the thesis.

It does **not** prove that:
- the detector found the “true” changepoint,
- every unmatched alarm is incorrect,
- every event should necessarily produce exactly one alarm.

Instead, proxy evaluation asks:

> Are detector alarms systematically concentrated near meaningful market disruptions?

That is the right interpretation.

This matters because some alarms may reflect real latent market-state changes that do not correspond to named public events.

---

# Part II — Route B: Self-Consistency / Segmentation Evaluation

## 10. What self-consistency evaluation means

In this route, you do not compare alarms to external events.

Instead, you ask whether the changepoints found by the detector produce a segmentation of the observed time series into intervals with **meaningfully different statistical properties**.

This is one of the strongest internal validation ideas for unsupervised real-data evaluation.

If the detector is useful, then the segments it induces should differ in interpretable ways such as:
- mean,
- variance,
- volatility level,
- autocorrelation,
- tail behavior.

So the evaluation question becomes:

> Do the detector’s changepoints partition the financial series into statistically distinct regimes?

---

## 11. Segment construction from alarms

Suppose the detector produces alarms at times

\[
a_1 < a_2 < \dots < a_N.
\]

Then these alarms induce a segmentation of the observation sequence into intervals such as:

\[
[1, a_1), \quad [a_1, a_2), \quad \dots, \quad [a_N, T].
\]

For real-data evaluation, you may want to restrict attention to:
- alarms surviving persistence/cooldown policy,
- alarms above a given confidence threshold,
- or the final alarm set chosen for the benchmark configuration.

These intervals are your detected segments.

---

## 12. What statistics should be compared across segments

This depends on the observation family from Phase 16.

At minimum, for return-based and volatility-based observations, you should compare:

## 12.1 Segment mean
\[
\bar y^{(s)} = \frac{1}{|I_s|}\sum_{t \in I_s} y_t
\]

## 12.2 Segment variance
\[
\operatorname{Var}^{(s)}(y)
=
\frac{1}{|I_s|}\sum_{t \in I_s}(y_t-\bar y^{(s)})^2
\]

## 12.3 Segment volatility level
If \(y_t\) is return-based, compare volatility summaries within each segment.

## 12.4 Segment autocorrelation
For example, lag-1 autocorrelation:

\[
\rho_1^{(s)} = \operatorname{Corr}(y_t, y_{t-1}), \quad t \in I_s
\]

## 12.5 Tail or dispersion features
Such as:
- upper quantiles,
- absolute-return averages,
- exceedance frequencies.

These give a richer picture of whether adjacent segments are genuinely different.

---

## 13. Adjacent-segment contrast

The most important local question is whether two adjacent segments differ.

Suppose alarms create segments \(I_s\) and \(I_{s+1}\).  
Then you should compute contrast measures such as:

### Mean contrast
\[
\Delta_\mu^{(s)} = \bar y^{(s+1)} - \bar y^{(s)}
\]

### Variance contrast
\[
\Delta_{\sigma^2}^{(s)} = \operatorname{Var}^{(s+1)}(y) - \operatorname{Var}^{(s)}(y)
\]

### Volatility contrast
defined similarly for volatility statistics.

These contrasts help quantify whether each detected changepoint corresponds to a meaningful statistical shift.

---

## 14. Segment-level significance or separation measures

Depending on how formal you want the evaluation to be, you can use either:

## Descriptive separation
Report differences in summary statistics without formal hypothesis testing.

This is simpler and often enough for a thesis if presented carefully.

## Statistical testing
Apply segment-comparison tests, for example:
- mean difference tests,
- variance difference tests,
- distribution comparison tests.

For your thesis, I recommend starting with **descriptive separation plus effect-size style summaries**, rather than over-relying on formal p-values.

That keeps the evaluation interpretable and avoids overcomplicating the analysis.

---

## 15. Global segmentation quality summary

In addition to adjacent contrasts, define a global summary of whether the segmentation is meaningful.

A useful route is to compare:
- within-segment variability,
- against between-segment variability.

For example, if detected segments are valid regimes, then the series should be more homogeneous inside segments than across segment boundaries.

You can summarize:
- average within-segment variance,
- average adjacent-segment variance difference,
- average adjacent-segment mean difference,
- pooled separation metrics.

This provides an overall measure of segmentation coherence.

---

## 16. Route B is detector-internal but still scientifically meaningful

It is important to explain the role of self-consistency evaluation carefully.

This route does **not** prove that the detector found the unique true latent market regimes.

Instead, it asks:

> Given the segmentation induced by the detector, are the resulting segments statistically distinct in ways that are meaningful for financial time series?

That is a strong unsupervised real-data criterion, especially when combined with Route A.

---

# Part III — Combined Real-Data Evaluation Design

## 17. Why A and B complement each other

These two routes evaluate different aspects of the detector.

## Route A — external relevance
Do alarms occur near meaningful market episodes?

## Route B — internal coherence
Do the alarms create statistically distinct segments?

A detector that performs well on both is much more convincing than one that performs well only on one side.

For example:
- a detector may align with news-like events but create weak statistical segmentation,
- or it may create strong statistical segmentation but fail to correspond to recognizable market episodes.

Using both routes helps avoid overclaiming.

---

## 18. Recommended evaluation flow on real data

For each asset / frequency / feature configuration:

### Step 1
Run the fixed-parameter online detector and collect:
- alarms,
- online scores,
- posterior summaries if useful.

### Step 2
Evaluate Route A:
- align alarms with proxy events or stress windows,
- compute coverage, relevance, and delays.

### Step 3
Evaluate Route B:
- construct detector-induced segments,
- compute segment summaries,
- compute adjacent contrasts and global coherence summaries.

### Step 4
Produce qualitative plots:
- price/feature with alarm markers,
- score curves,
- proxy-event overlays,
- segment summary tables.

This should become the canonical real-data experiment structure.

---

## 19. Step-by-step guide for Phase 18

## Step 1 — Define the proxy-event schema

Create a formal representation of real-data evaluation anchors.

Each event should include:
- timestamp or window,
- event type,
- asset scope,
- optional human-readable label,
- optional metadata.

### Deliverable
A standardized proxy-event representation.

### Code changes
You should add:
- a `ProxyEvent` type,
- support for point events and window events,
- event metadata fields,
- validation of event timestamps/windows.

---

## Step 2 — Define proxy-event alignment logic

Implement the rule that determines whether an alarm aligns with a proxy event.

This logic should support:
- point events with tolerance windows,
- window events,
- optional causal-only matching.

### Deliverable
A proxy-event matching protocol.

### Code changes
Add:
- event-alignment utilities,
- configurable tolerance/window policy,
- alignment result objects.

---

## Step 3 — Define Route A metrics

Compute:
- event coverage,
- alarm relevance,
- aligned delay summaries,
- optional alarm density around events.

### Deliverable
A proxy-event evaluation metric layer.

### Code changes
Add:
- Route A metric helpers,
- summary result structs,
- aggregation across assets and runs.

---

## Step 4 — Define the segmentation schema

Formalize how alarms induce segments.

Each segment should include:
- start time,
- end time,
- duration,
- asset metadata,
- feature metadata,
- contained observations.

### Deliverable
A detector-induced segment representation.

### Code changes
Add:
- a `DetectedSegment` type,
- segmentation utilities from alarm sequences,
- duration checks and minimum-length policies if needed.

---

## Step 5 — Define segment statistics

For each segment, compute relevant descriptive summaries.

At minimum:
- mean,
- variance,
- volatility summary,
- optional autocorrelation,
- optional tail statistics.

### Deliverable
A per-segment statistical summary layer.

### Code changes
Add:
- segment-statistics utilities,
- feature-family-aware segment summaries,
- exportable segment summary objects.

---

## Step 6 — Define adjacent-segment contrast summaries

For each detected boundary, compare the segment before and after it.

### Deliverable
A local segmentation-quality layer.

### Code changes
Add:
- adjacent-segment comparison utilities,
- difference/effect-size summaries,
- optional rank-based or descriptive contrast measures.

---

## Step 7 — Define global segmentation summaries

Aggregate segment-level and boundary-level results into global real-data evaluation metrics.

### Deliverable
A global self-consistency evaluation layer.

### Code changes
Add:
- global segmentation summary objects,
- aggregation functions over all segments and boundaries,
- per-run and aggregate summary export.

---

## Step 8 — Add a real-data reporting layer

This phase should produce interpretable outputs, not just raw metric structs.

### Deliverable
A real-data evaluation report format.

### Code changes
Add:
- a `real_eval` module,
- Route A and Route B result objects,
- report builders,
- plot-ready export structures,
- table-ready summaries.

---

# Part IV — Suggested Real-Data Evaluation Objects

## 20. Route A objects

You should likely have conceptual objects such as:

- `ProxyEvent`
- `ProxyEventSet`
- `AlarmProxyAlignment`
- `ProxyEventEvaluationResult`

These should support:
- event coverage,
- alarm alignment,
- delay summaries.

---

## 21. Route B objects

You should likely have conceptual objects such as:

- `DetectedSegment`
- `SegmentSummary`
- `AdjacentSegmentContrast`
- `SegmentationEvaluationResult`

These should support:
- per-segment statistics,
- per-boundary contrast summaries,
- global segmentation quality summaries.

---

## 22. Combined real-evaluation object

A full real-data evaluation run should likely produce something like:

- detector configuration,
- asset / dataset metadata,
- proxy-event evaluation,
- segmentation evaluation,
- alarm summary,
- plotting/reporting metadata.

This combined object becomes the main artifact for real-data analysis.

---

# Part V — Practical and Methodological Constraints

## 23. Asset-specific proxy-event policy

You may not want one identical event set for all datasets.

For example:
- commodity datasets may need commodity-specific stress events,
- SPY/QQQ may use broader equity-market proxy windows.

So the protocol should allow proxy events to be:
- asset-specific,
- frequency-specific,
- and possibly feature-specific if needed.

This should be configurable rather than hardcoded globally.

---

## 24. Minimum segment length policy

Short segments can be statistically unstable.

If a detector produces many very short segments, segment statistics may become unreliable.

So Route B should consider adding a minimum segment length policy:
- either for evaluation only,
- or as a warning condition.

This does not necessarily mean deleting short segments, but they should be flagged or treated carefully.

### Code changes
Add:
- segment-length checks,
- warnings for undersized segments,
- optional exclusion policies in Route B summaries.

---

## 25. Real-data evaluation is not the same as synthetic benchmarking

This should be stated explicitly in the thesis.

Synthetic evaluation answers:
- whether the detector recovers known changes.

Real-data evaluation answers:
- whether the detector reacts near meaningful market episodes,
- and whether its segmentation is statistically coherent.

These are different scientific claims.

That distinction is important and strengthens the thesis rather than weakening it.

---

## 26. Common conceptual mistakes to avoid

### Mistake 1 — Treating proxy events as perfect truth
They are anchors, not exact labels.

### Mistake 2 — Treating unmatched alarms as automatically meaningless
Some alarms may reflect latent regime changes not captured by the proxy-event set.

### Mistake 3 — Using only qualitative plots on real data
You need formal Route A and B summaries as well.

### Mistake 4 — Ignoring segment quality
A detector that produces many alarms but weak statistical segmentation is questionable.

### Mistake 5 — Overclaiming causal explanation from ex post event alignment
Alignment supports relevance, not proof of causation.

### Mistake 6 — Using overly tiny segments for Route B summaries
Statistical summaries of very short segments are unstable and should be treated cautiously.

---

## 27. Testing requirements for Phase 18

This phase also needs dedicated tests.

You should test at least:

## Route A tests
- point-event alignment correct,
- window-event alignment correct,
- causal-only alignment works as intended,
- delay computation correct,
- unmatched alarms/events handled correctly.

## Route B tests
- alarm-to-segment conversion correct,
- segment boundaries correct,
- segment statistics computed correctly,
- adjacent contrast summaries correct,
- minimum segment-length policy behaves correctly.

## Combined real-evaluation tests
- all result objects serialize/export correctly,
- asset metadata preserved,
- empty-event or empty-alarm cases handled gracefully.

### Deliverable
A trustworthy real-data evaluation layer.

### Code changes
Add:
- unit tests for alignment and segmentation,
- integration tests from alarm stream to Route A/B summaries,
- edge-case tests for empty alarms, empty events, and tiny segments.

---

## 28. Deliverables of Phase 18

By the end of this phase, you should have:

### Mathematical / methodological deliverables
- a formal real-data evaluation protocol based on:
  - Route A: proxy event evaluation,
  - Route B: self-consistency / segmentation evaluation,
- precise definitions for:
  - proxy events,
  - event/alarm alignment,
  - event coverage,
  - alarm relevance,
  - aligned delay,
  - detected segments,
  - segment statistics,
  - adjacent-segment contrasts,
  - global segmentation coherence,
- a clear statement that real-data evaluation is not synthetic ground-truth evaluation.

### Architectural deliverables
- a dedicated real-data evaluation layer,
- explicit separation between:
  - detector execution,
  - proxy-event evaluation,
  - segmentation evaluation,
  - reporting/export.

### Code-structure deliverables
You should add or revise, where appropriate:

- a dedicated `real_eval` module,
- `ProxyEvent` and related Route A types,
- event alignment utilities,
- Route A metric computation,
- `DetectedSegment` and related Route B types,
- segment-statistics utilities,
- adjacent-segment contrast utilities,
- global segmentation summary logic,
- combined real-data evaluation result types,
- report/export builders,
- tests for alignment, segmentation, and edge cases.

### Thesis deliverables
- a defensible real-data evaluation methodology,
- a way to argue both:
  - external market relevance of alarms,
  - and internal statistical coherence of the detected segmentation,
- a much stronger basis for the empirical chapter of the thesis.

---

## 29. Minimal final summary

Phase 18 is the step that makes real-market evaluation scientifically meaningful.

The central idea is:

\[
\boxed{
\text{On real financial data, detector quality must be evaluated through external relevance and internal segmentation coherence.}
}
\]

For this thesis, that means using:

- **Route A**: proxy event evaluation,
- **Route B**: self-consistency / segmentation evaluation,

and deliberately **not** relying on downstream predictive usefulness in this phase.

This phase should end with:
- a solid real-data evaluation protocol,
- a clean Rust-oriented `real_eval` layer,
- and thesis-ready outputs for analyzing changepoint alarms on actual market data.