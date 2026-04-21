# Phase 14 — Add the Online Benchmarking Protocol

## Goal

In this phase, define a **rigorous benchmarking protocol** for evaluating the Markov-Switching-based online changepoint detector against conventional online detectors.

By the end of the previous phases, you have:

- an offline-trained, online-filtered Markov Switching backbone,
- a family of detector variants built on top of that backbone,
- causal online operation,
- a fixed-parameter runtime design.

But none of that is enough for a serious thesis comparison unless you also define:

1. **what counts as a correct detection,**
2. **how late a detection is allowed to be,**
3. **what counts as a false alarm,**
4. **how to compare detectors fairly across streams and scenarios,**
5. **how computational cost is measured in an online setting.**

So this phase is about moving from “a detector exists” to “the detector can be evaluated scientifically.”

---

## 1. Why this phase is necessary

Online changepoint detection is not judged only by whether alarms occur near true changes.

A detector may:
- detect almost every true change, but far too late,
- detect quickly, but raise too many false alarms,
- detect well on one stream, but fail badly on another,
- perform well statistically, but be too slow for practical streaming use.

So a proper benchmark must measure both:

## Statistical detection quality
How well the detector identifies actual changepoints.

## Online operational quality
How quickly and efficiently it reacts in a streaming setting.

Without this phase, any comparison against conventional methods will remain anecdotal.

---

## 2. What an online benchmark must evaluate

For your thesis, the benchmark protocol should evaluate at least the following dimensions.

## 2.1 Detection delay
How long after a true changepoint the detector raises an alarm.

## 2.2 False alarms
How often the detector raises alarms that cannot be justified by a true changepoint.

## 2.3 Missed detections
How often a true changepoint occurs without being successfully detected.

## 2.4 Precision / recall
Whether the detector’s alarm stream is both selective and complete.

## 2.5 False-alarm control
How long the detector typically runs before producing a false alarm.

## 2.6 Runtime cost
How expensive each update step is in the streaming setting.

This set of metrics should become the core evaluation surface of the thesis.

---

## 3. The central benchmarking object: event-level comparison

The benchmark must compare two sequences:

## Ground-truth changepoints
A set of true event times

\[
\mathcal{C}^{\star} = \{ \tau_1, \tau_2, \dots, \tau_M \}
\]

where each \(\tau_m\) is a true changepoint time in the simulated or labeled stream.

## Detector alarms
A set of detected event times

\[
\mathcal{A} = \{ a_1, a_2, \dots, a_N \}
\]

where each \(a_n\) is a time at which the detector emitted an alarm.

The benchmark protocol must define how these two sets are matched.

That matching rule is the heart of the evaluation logic.

---

## 4. Why event matching is nontrivial

Online changepoint detection is not ordinary classification.

At each time point, you do not simply label “change” or “no change” independently.  
Instead, the detector emits a sparse sequence of alarms in time, and those alarms must be interpreted relative to nearby true changepoints.

This creates several complications:

- one true changepoint may trigger multiple nearby alarms,
- one alarm may occur slightly before or after the ideal location,
- two true changepoints may be close together,
- a late alarm may become ambiguous if another changepoint occurs soon afterward.

So the benchmark must define a principled matching policy.

---

## 5. Ground-truth representation

Before evaluating any detector, define the ground truth formally.

For each stream, represent the true changepoints as an ordered sequence

\[
\tau_1 < \tau_2 < \dots < \tau_M.
\]

Each \(\tau_m\) should correspond to the time at which the generating process changes regime or distributional structure.

If the simulator emits regime sequences \(S_t\), then in the simplest case a true changepoint occurs whenever

\[
S_t \neq S_{t-1}.
\]

But more generally, the benchmark protocol should treat the ground-truth event list as a first-class object independent of the detector.

### Deliverable
A formal changepoint ground-truth representation.

### Code changes
You should add:
- a `ChangePointTruth` type,
- a per-stream event list representation,
- optional metadata for scenario type, stream ID, and changepoint count.

---

## 6. Alarm representation

Similarly, detector outputs should be represented as an ordered alarm sequence

\[
a_1 < a_2 < \dots < a_N.
\]

Each alarm should already exist from earlier phases as an `AlarmEvent` or equivalent object, but now the benchmarking protocol must consume them as evaluation objects rather than just detector outputs.

Each alarm may carry:
- timestamp,
- detector ID,
- score value,
- optional confidence,
- optional regime metadata.

### Deliverable
A standardized benchmark-facing alarm representation.

### Code changes
Ensure the existing alarm event object is suitable for evaluation, or introduce a thinner benchmark view if needed.

---

## 7. The matching policy

The benchmark must define when an alarm counts as a successful detection of a true changepoint.

The cleanest first protocol is a **detection window** rule.

For each true changepoint \(\tau_m\), define an allowed detection window

\[
[\tau_m, \tau_m + w]
\]

where \(w\) is a user-chosen delay tolerance.

Then:

- the **first alarm** in that window is matched to \(\tau_m\),
- that changepoint is counted as detected,
- additional alarms in the same window are not counted as additional true positives.

This is a standard and defensible online evaluation rule because it respects causality: a true changepoint may only be detected at or after its occurrence.

---

## 8. Why the detection window matters

The window width \(w\) expresses how much delay you are willing to tolerate.

A very small \(w\):
- rewards very fast detectors,
- penalizes slower detectors heavily.

A larger \(w\):
- is more forgiving,
- shifts emphasis from speed to eventual detection.

So the detection window is not just a technical detail.  
It encodes the operational requirements of the changepoint problem.

For a serious thesis, you should either:
- choose \(w\) based on the application setting,
- or report results for multiple window values.

---

## 9. Matched, missed, and false events

Under the event-matching protocol, each stream yields:

## Matched changepoints
True changepoints with a valid alarm in their detection window.

## Missed changepoints
True changepoints with no matched alarm.

## True-positive alarms
Alarms that are the first valid match for some changepoint.

## False-positive alarms
Alarms that do not match any changepoint.

These definitions should be fixed globally across all detectors.

### Deliverable
A benchmark-wide event taxonomy.

### Code changes
You should add:
- a matching result object,
- event-level labels for alarms and changepoints,
- explicit counts of TP / FP / miss.

---

## 10. Detection delay

For each matched changepoint \(\tau_m\) with matched alarm \(a(\tau_m)\), define the detection delay as

\[
d_m = a(\tau_m) - \tau_m.
\]

This is one of the most important online metrics.

You should report at least:

- mean detection delay,
- median detection delay,
- possibly quantiles,
- possibly worst-case delay under matched events.

### Why this matters
A detector that always detects eventually but only after a long lag may be poor in an online setting.

### Deliverable
A latency-aware delay metric suite.

### Code changes
Add:
- delay computation utilities,
- summary statistics for matched delays,
- benchmark result fields for delay summaries.

---

## 11. Miss rate

Let \(M\) be the total number of true changepoints and \(M_{\text{det}}\) the number of matched changepoints.

Then the miss rate is

\[
\text{Miss Rate} = 1 - \frac{M_{\text{det}}}{M}.
\]

This measures how often true changepoints are not detected within the allowed window.

### Deliverable
A changepoint-level recall / miss computation.

### Code changes
Add:
- matched-changepoint counting,
- miss-rate calculation,
- per-stream and aggregate summaries.

---

## 12. Precision and recall

Define:

\[
\text{Precision} = \frac{\text{TP alarms}}{\text{all alarms}},
\]

\[
\text{Recall} = \frac{\text{detected changepoints}}{\text{all true changepoints}}.
\]

These are useful because they summarize two different detector tendencies:

- **precision** penalizes detectors that alarm too often,
- **recall** penalizes detectors that miss true changes.

### Deliverable
Standard event-level precision and recall metrics.

### Code changes
Add:
- alarm-level TP / FP counting,
- aggregate precision / recall logic,
- safe handling of zero-alarm cases.

---

## 13. False alarm rate

False alarms should be measured explicitly.

A simple stream-level false alarm rate is:

\[
\text{False Alarm Rate}
=
\frac{\text{number of false-positive alarms}}{\text{stream length}}
\]

or normalized by time units or number of observations.

Depending on how you want to present results, you may also report:
- false positives per 1000 observations,
- false positives per stream,
- or false positives before first true changepoint.

### Deliverable
A standardized false-alarm metric.

### Code changes
Add:
- false-positive counting,
- normalization policies,
- stream-length-aware summaries.

---

## 14. Average run length to false alarm

For changepoint detection, a common false-alarm-control quantity is the time until the first false alarm.

In your benchmark, this can be defined per stream as:

- if the first unmatched alarm occurs at time \(a_{\text{FA}}\), then the run length to false alarm is \(a_{\text{FA}}\),
- if no false alarm occurs, treat appropriately according to your summary policy.

You may also compute:
- time to first false alarm before the first true changepoint,
- average run length to false alarm across no-change streams.

This metric is especially useful if you include streams with no changepoints or long pre-change segments.

### Deliverable
A first-false-alarm timing metric.

### Code changes
Add:
- first-false-alarm extraction,
- optional special handling for no-false-alarm streams,
- aggregate ARL-style summaries.

---

## 15. Computational cost per observation

Because your setting is streaming, runtime cost matters.

The benchmark should measure:
- update time per observation,
- total time per stream,
- possibly memory footprint or state size if relevant.

At minimum, you should record:

\[
\text{mean step time}, \quad \text{median step time}, \quad \text{total runtime}.
\]

### Why this matters
A detector that performs well statistically but is too expensive per step may be unsuitable for an online deployment scenario.

### Deliverable
A per-step computational-cost measurement protocol.

### Code changes
Add:
- per-step timing instrumentation,
- total-stream timing,
- benchmark result fields for runtime summaries.

---

## 16. Streams with no changepoints

A strong benchmark should include streams where no true changepoint occurs.

This is important because it tests the detector’s tendency to hallucinate changes.

For such streams:
- every alarm is a false alarm,
- miss rate is not the main quantity,
- false alarm rate and run length to false alarm become especially informative.

This kind of scenario is highly valuable for evaluating detector stability.

### Deliverable
A no-change evaluation policy.

### Code changes
Make sure the benchmark logic handles:
- empty ground-truth event lists,
- valid precision/recall behavior under no-change scenarios,
- correct false-alarm summaries.

---

## 17. Streams with multiple close changepoints

The benchmark protocol must define what happens when changepoints are close together.

If two true changepoints occur within overlapping detection windows, naive matching can become ambiguous.

For the first version, use one of these policies:

### Policy A — non-overlapping scenario design
Ensure your simulated streams space changepoints sufficiently far apart.

### Policy B — greedy chronological matching
Match alarms to changepoints in chronological order, with each alarm usable at most once.

For thesis clarity, Policy A is cleaner in early experiments.

### Deliverable
A matching policy for closely spaced changes.

### Code changes
Document and implement the matching rule explicitly rather than leaving it implicit.

---

## 18. Benchmark aggregation across streams

A single stream is not enough.

The benchmark should run over many simulated or labeled streams and aggregate results across:

- streams,
- parameter settings,
- detector variants,
- scenario classes.

This means your result objects should support both:

## Per-stream results
Detailed event-level evaluation for one run.

## Aggregate results
Means, medians, standard deviations, and scenario-level summaries across repeated runs.

### Deliverable
A hierarchical benchmark result structure.

### Code changes
Add:
- per-stream benchmark result type,
- aggregate benchmark summary type,
- grouping by scenario and detector.

---

## 19. Step-by-step guide for Phase 14

## Step 1 — Define the benchmark contract

Formally specify that every benchmark run takes:

- a ground-truth changepoint sequence,
- a detector alarm sequence,
- stream length,
- optional per-step timing data.

and returns:
- event-matching results,
- metric summaries,
- runtime summaries.

### Deliverable
A formal benchmark input/output contract.

### Code changes
Add a dedicated `benchmark` module or equivalent boundary.

---

## Step 2 — Define the ground-truth event representation

Create a standard representation for true changepoints.

### Deliverable
A stream-level truth object with ordered changepoint times.

### Code changes
Add:
- `ChangePointTruth`,
- optional stream metadata fields,
- validation that changepoint times are sorted and in bounds.

---

## Step 3 — Define the benchmark-facing alarm representation

Standardize how detector alarms are passed into the benchmark.

### Deliverable
A reusable alarm sequence representation.

### Code changes
Ensure compatibility between detector outputs and benchmark inputs, or add conversion helpers.

---

## Step 4 — Implement event matching

Implement the core chronological matching logic using the chosen detection-window rule.

### Deliverable
An event matcher that returns:
- matched changepoints,
- matched alarms,
- unmatched alarms,
- missed changepoints.

### Code changes
Add:
- event-matching utilities,
- window-based matching logic,
- per-event labels.

This is the core algorithmic deliverable of the phase.

---

## Step 5 — Implement metric computation

From the matching result, compute:
- detection delays,
- precision,
- recall,
- miss rate,
- false alarm rate,
- first false alarm timing.

### Deliverable
A metric computation layer.

### Code changes
Add:
- metric calculators,
- safe handling of edge cases,
- summary-statistics utilities.

---

## Step 6 — Add runtime timing instrumentation

Instrument the detector loop to record step-level or stream-level runtime.

### Deliverable
A runtime-cost measurement pipeline.

### Code changes
Add:
- timing hooks,
- benchmark timing collectors,
- runtime summary fields in benchmark results.

---

## Step 7 — Add repeated-run aggregation

Support repeated simulations and repeated benchmark runs.

### Deliverable
An aggregate benchmark summary for each detector/scenario combination.

### Code changes
Add:
- aggregation logic,
- mean/median/variance summaries,
- detector-by-scenario comparison structures.

---

## 20. Suggested benchmark result structure

A complete per-stream benchmark result should include at least:

- stream ID,
- detector ID,
- scenario ID,
- true changepoint list,
- alarm list,
- matched event pairs,
- missed changepoints,
- false alarms,
- delay list,
- precision,
- recall,
- miss rate,
- false alarm rate,
- first false alarm timing,
- runtime summaries.

An aggregate result should include:
- number of runs,
- mean / median delay,
- mean precision / recall,
- mean miss rate,
- mean false alarm rate,
- runtime summaries,
- optional quantiles.

### Deliverable
A benchmark result schema suitable for experiments and tables.

### Code changes
Add:
- `BenchmarkRunResult`,
- `BenchmarkAggregateResult`,
- scenario/detector grouping support.

---

## 21. Recommended benchmark policies for thesis clarity

For an initial clean thesis benchmark, I recommend:

### Policy 1 — causal matching only
Only alarms at or after the changepoint may count as detections.

### Policy 2 — first-match rule
Only the first alarm in the allowed window counts as the true detection.

### Policy 3 — one alarm per changepoint
A single changepoint cannot produce multiple true positives.

### Policy 4 — one alarm used once
An alarm cannot be matched to multiple changepoints.

### Policy 5 — explicit window parameter
Always state the detection tolerance window \(w\) in plots and tables.

These policies make the benchmark transparent and reproducible.

---

## 22. Common conceptual mistakes to avoid

### Mistake 1 — Using classification accuracy instead of event-level metrics
Changepoint detection is not ordinary pointwise classification.

### Mistake 2 — Ignoring delay
A detector that alarms eventually but too late may be poor online.

### Mistake 3 — Allowing retrospective matches before the changepoint
That violates causal evaluation.

### Mistake 4 — Counting multiple alarms around one change as multiple successes
That artificially inflates detector performance.

### Mistake 5 — Ignoring no-change streams
Those are essential for evaluating false alarms.

### Mistake 6 — Not measuring runtime
In a streaming thesis, speed is part of the method’s practical quality.

---

## 23. Deliverables of Phase 14

By the end of this phase, you should have:

### Mathematical deliverables
- a formal event-level online benchmarking protocol,
- a ground-truth changepoint representation,
- a causal detection-window matching rule,
- definitions for:
  - detection delay,
  - miss rate,
  - precision,
  - recall,
  - false alarm rate,
  - run length to false alarm,
  - runtime-per-observation summaries.

### Architectural deliverables
- a dedicated benchmarking layer,
- clear separation between:
  - detector execution,
  - event matching,
  - metric computation,
  - aggregation.

### Code-structure deliverables
You should add or revise, where appropriate:

- a `benchmark` module,
- a ground-truth changepoint type,
- benchmark-facing alarm conversion or reuse,
- event-matching logic,
- metric computation helpers,
- delay summary helpers,
- false-alarm summary helpers,
- runtime timing instrumentation,
- per-stream result objects,
- aggregate result objects,
- grouping logic by detector and scenario.

### Experimental deliverables
- a benchmark harness capable of repeated simulation runs,
- comparable metric outputs across detector variants,
- a clean foundation for later comparison against conventional online changepoint methods.

---

## 24. Minimal final summary

Phase 14 is the step that makes your online detector scientifically comparable.

The central principle is:

\[
\boxed{
\text{Online changepoint detectors must be evaluated at the event level, not just by vague alarm counts.}
}
\]

This phase should end with a benchmark protocol that can tell you, for every detector:

- how often it detects true changes,
- how late it detects them,
- how often it raises false alarms,
- how stable it is on no-change streams,
- and how expensive it is per observation.

That protocol becomes the backbone of the comparative experiments in the thesis.