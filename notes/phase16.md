# Phase 16 — Feature Engineering and Observation Design

## Goal

In this phase, define **what the Markov Switching Model actually observes** when applied to real financial data.

This is one of the most important remaining phases because, up to now, the modeling pipeline has mostly been described in abstract statistical terms:

- there is an observed process \(y_t\),
- there is a hidden regime process \(S_t\),
- the detector operates on the online evolution of \(y_t\) through the model.

But for real market data, \(y_t\) is **not automatically given**.

You must decide whether the observed process is:

- raw prices,
- returns,
- absolute returns,
- rolling volatility,
- standardized returns,
- realized volatility,
- or some derived statistic combining several of these.

This phase defines that choice formally and turns it into a reproducible, configurable feature pipeline.

---

## 1. Why this phase is necessary

Your model currently assumes an observed process of the form

\[
y_t \mid S_t=j \sim \mathcal N(\mu_j,\sigma_j^2)
\]

or a related regime-dependent observation law.

That means the statistical behavior of the detector is determined not only by:

- the hidden-state dynamics,
- the estimation procedure,
- the online filter,
- the detector score,

but also by the **meaning of \(y_t\)**.

For financial data, this is crucial because raw prices are usually a poor direct input to regime or changepoint models.

### Why raw prices are problematic
Raw prices often exhibit:
- strong nonstationary level trends,
- scale dependence,
- long-run drift,
- price-level effects that are not the real object of interest for changepoint detection.

In contrast, financial econometrics usually studies:
- returns,
- volatility,
- distributional dispersion,
- standardized shocks,
- activity proxies.

This aligns directly with your thesis description, which emphasizes:

> dynamic detection of changes in the distribution of the observed process based on changes in time-series statistics used in econometric practice.

So Phase 16 is the step where the abstract process \(y_t\) becomes a **financially meaningful statistical observable**.

---

## 2. The main purpose of feature engineering in this thesis

The role of feature engineering here is **not** generic machine-learning feature expansion.

It has a narrower, more scientific role:

1. translate raw market data into a statistically meaningful observed process,
2. make the model more appropriate for financial dynamics,
3. create multiple observation definitions that can be compared in the thesis,
4. make the detector interpretable in terms of market statistics.

So this phase is really about **observation design**, not just feature extraction.

---

## 3. The central question of Phase 16

The core question is:

> What scalar or low-dimensional time-dependent statistic should be modeled as the observed process \(y_t\) in the Markov Switching detector?

This question must be answered separately for:
- daily data,
- intraday data,
- and potentially for different asset classes.

Because your current modeling core is most naturally univariate, the first implementation target should be a family of **univariate observation processes**, even if later extensions may become multivariate.

That makes the thesis cleaner and the detector comparison more interpretable.

---

## 4. Recommended first observation families

For your thesis, I recommend that Phase 16 support at least **three primary observation designs**.

## 4.1 Log returns

Define:

\[
r_t = \log P_t - \log P_{t-1}.
\]

This should be your first and most basic observation process.

### Why
Log returns are the standard baseline in financial econometrics because they:
- remove price level trend,
- are scale-normalized,
- are easier to compare across assets,
- often behave more stationarily than raw prices.

### Interpretation
Using \(y_t = r_t\) means the model is detecting changes in the distribution of returns.

This is a very natural first thesis setting.

---

## 4.2 Absolute returns

Define:

\[
a_t = |r_t|.
\]

### Why
Absolute returns are a simple proxy for volatility or market activity.

They are useful because:
- they often highlight stress periods,
- they are sensitive to volatility regime changes,
- they remain simple and univariate.

### Interpretation
Using \(y_t = |r_t|\) means the detector is reacting to changes in return magnitude rather than signed direction.

This is useful when your interest is more about turbulence than directional trend.

---

## 4.3 Rolling volatility proxy

A simple rolling volatility proxy over a window of size \(w\) can be defined as:

\[
v_t^{(w)}
=
\sqrt{
\frac{1}{w}
\sum_{k=0}^{w-1}
(r_{t-k} - \bar r_t^{(w)})^2
},
\]

where

\[
\bar r_t^{(w)}
=
\frac{1}{w}
\sum_{k=0}^{w-1} r_{t-k}.
\]

### Why
This creates an observed process tied directly to changing dispersion structure.

It is especially relevant for:
- financial crises,
- stress detection,
- volatility-regime analysis,
- intraday instability studies.

### Interpretation
Using \(y_t = v_t^{(w)}\) means the detector is explicitly tracking changes in recent volatility rather than single-step returns.

---

## 5. Strongly recommended secondary observation families

After the three primary families, there are several valuable extensions.

---

## 5.1 Standardized returns

Define standardized returns using a rolling volatility estimate:

\[
z_t = \frac{r_t}{v_t^{(w)} + \varepsilon},
\]

where \(\varepsilon > 0\) is a small numerical stabilizer.

### Why
This separates:
- raw move size,
- from move size relative to local volatility.

This can help distinguish:
- structural market changes,
- from ordinary moves within already volatile periods.

### Interpretation
Using \(y_t = z_t\) means the detector focuses on unusually large normalized shocks.

---

## 5.2 Squared returns

Define:

\[
q_t = r_t^2.
\]

### Why
Squared returns are another classical volatility proxy.

They are often noisier than rolling volatility, but:
- simpler,
- causal,
- and cheap to compute online.

### Interpretation
Using \(y_t = r_t^2\) means the detector reacts to changing second-moment structure.

---

## 5.3 Realized intraday volatility

For intraday settings, if you aggregate returns within a session interval, you can define realized volatility over a block:

\[
RV_t = \sum_{k \in \mathcal{B}_t} r_k^2
\]

for some block \(\mathcal{B}_t\).

### Why
This is especially relevant for intraday market-state analysis and volatility clustering.

### Interpretation
Using \(y_t = RV_t\) means the detector is monitoring local realized variance bursts rather than pointwise returns.

---

## 5.4 Volume-based features

If reliable volume is available, you may define features such as:
- log volume,
- rolling average volume,
- volume surprise.

These should remain optional in the first thesis version because:
- volume data quality and interpretation vary,
- your current core is already strong enough without them.

Still, a volume-based extension could become useful if you want to test whether market-state changes are better captured by liquidity/activity shifts.

---

## 6. Recommended minimal observation set for the thesis

If you want a focused but strong thesis design, I recommend the following three core observation processes:

## Observation Set A — Returns
\[
y_t = r_t
\]

## Observation Set B — Absolute or volatility proxy
\[
y_t = |r_t|
\quad \text{or} \quad
y_t = v_t^{(w)}
\]

## Observation Set C — Return + volatility design through separate experiments
Use one experiment family for returns and one for volatility proxy rather than trying to force both into one multivariate model immediately.

This gives you:
- a directional/statistical-change view,
- a volatility-regime view,
- and a meaningful comparative structure.

---

## 7. Why not use raw prices directly

This deserves an explicit statement in the thesis.

Raw prices are usually a poor direct choice because:

- they are highly nonstationary,
- regime changes in price level may simply reflect long-run drift,
- the same absolute price movement means different things at different price scales,
- online changepoint alarms may become dominated by slow trend effects rather than genuine statistical-distribution changes.

So the default stance of the project should be:

\[
\text{Do not use raw prices as the primary observed process unless strongly justified.}
\]

That is a methodological design decision and should be documented clearly.

---

## 8. Daily vs intraday feature design

Phase 16 should preserve the distinction between daily and intraday data established in Phase 15.

## 8.1 Daily features
For daily data, the most natural features are:
- daily log returns,
- rolling daily volatility,
- absolute daily returns.

These are straightforward and align well with standard econometric practice.

---

## 8.2 Intraday features
For intraday data, you must be more careful.

Possible observation processes include:
- intraday log returns at chosen bar resolution,
- absolute intraday returns,
- rolling intraday volatility,
- block-level realized volatility.

### Important note
Because intraday data have session structure, some rolling features must respect session boundaries.

For example, you should decide whether:
- rolling windows reset at session start,
- or continue across overnight boundaries.

For the first thesis version, I strongly recommend:
- **resetting session-local rolling statistics at session boundaries**,  
or at minimum,
- making the policy explicit and configurable.

That will avoid mixing overnight gaps into ordinary intraday volatility estimation.

---

## 9. Causality requirement for online features

Since the detector is online, the feature pipeline must also be causal.

That means that at time \(t\), the feature \(y_t\) may depend only on:
- current observation,
- past observations,
- but never future observations.

This is especially important for rolling statistics.

So any rolling feature must be based on a trailing window:

\[
y_t = g(x_{t-w+1}, \dots, x_t),
\]

not a centered or future-looking window.

This is a mandatory requirement for the thesis because the detector is evaluated online.

---

## 10. Warmup and undefined early observations

Many useful features are undefined at the very start of the series.

For example:
- log returns need \(P_{t-1}\),
- rolling volatility with window \(w\) needs at least \(w\) returns.

So the feature pipeline must define a warmup policy.

### Recommended options
- drop the first undefined points,
- emit feature values only when defined,
- track warmup length explicitly.

For the first thesis version, I recommend:
- explicitly dropping undefined prefix observations from the detector input stream,
- and recording the resulting effective start time.

This is simpler and more reproducible than trying to patch undefined values.

---

## 11. Scaling and normalization decisions

Feature scaling should be treated carefully.

For financial features, possible scaling choices include:
- no scaling,
- z-scoring on training data,
- robust scaling using training quantiles,
- asset-wise normalization.

### Important principle
Any normalization parameters must be fit on the training set only, then frozen for validation/test and online use.

Otherwise, you create leakage.

### Recommendation
For the first version:
- use raw log returns or volatility proxies without overly aggressive normalization,
- or, if normalizing, do so strictly using train-only statistics.

This should be configurable and clearly documented.

---

## 12. Feature pipeline as a formal transformation

This phase should define the feature layer as a deterministic transformation:

\[
\text{market data stream}
\longrightarrow
\text{observation stream } y_t
\]

More formally:

\[
y_t = \Phi(\text{raw market data up to time } t),
\]

where \(\Phi\) is a chosen feature policy.

This makes the observation definition explicit and reproducible.

It also suggests that your implementation should treat feature generation as its own pipeline layer rather than scattering feature logic across:
- data loaders,
- model code,
- detector code.

---

## 13. Step-by-step guide for Phase 16

## Step 1 — Define the canonical observation families

Write down the exact feature families the thesis will support, at least:

- log returns,
- absolute returns,
- rolling volatility proxy.

Optionally add:
- standardized returns,
- squared returns,
- realized volatility for intraday settings.

### Deliverable
A formal observation-family specification.

### Code changes
You should add a dedicated `features` module or equivalent boundary.

---

## Step 2 — Define causal feature formulas

For each supported feature, specify:
- formula,
- required inputs,
- warmup requirement,
- whether it is daily-only, intraday-only, or both.

### Deliverable
A mathematically explicit feature-definition layer.

### Code changes
Add:
- a formal feature configuration type,
- one implementation path per feature family,
- explicit warmup metadata.

---

## Step 3 — Define session-aware feature behavior for intraday data

For intraday rolling features, decide:
- whether the rolling window resets at each session,
- whether overnight gaps are excluded from feature continuity,
- how partial sessions are handled.

### Deliverable
A session-aware feature policy.

### Code changes
Add:
- session-aware rolling-statistics utilities,
- feature computation that respects session metadata from Phase 15.

---

## Step 4 — Define normalization policy

Specify whether each feature is:
- unnormalized,
- training-set standardized,
- or transformed by another train-only scaling rule.

### Deliverable
A leakage-safe scaling policy.

### Code changes
Add:
- feature-scaler abstractions,
- train-fit / test-apply separation,
- frozen scaler artifacts for runtime use.

---

## Step 5 — Define the final observation stream object

The output of the feature pipeline should be a clean observed series ready for:
- offline fitting,
- online filtering,
- detector evaluation.

It should include:
- transformed values \(y_t\),
- timestamps,
- asset metadata,
- feature metadata,
- warmup trimming information.

### Deliverable
A model-ready observation-series representation.

### Code changes
Add:
- a feature-output dataset type,
- metadata linking observation stream to feature policy and source data.

---

## Step 6 — Support multiple feature experiments cleanly

Because your thesis should compare several observation definitions, the feature layer should make it easy to rerun the full detector pipeline under different feature choices.

### Deliverable
A reproducible feature-experiment interface.

### Code changes
Add:
- configurable feature pipeline composition,
- experiment labels by feature family,
- serialization of feature configuration.

This is essential for later experiment reporting.

---

## 14. Suggested feature architecture

By the end of this phase, your project should conceptually have:

## Raw market data
Output of Phase 15.

## Feature policy
Defines how raw data are transformed into observation values.

## Fitted feature transformer / scaler
Optional train-only fitted normalization object.

## Observation stream
Final time-indexed \(y_t\) sequence consumed by the model and detector.

This is the cleanest architecture because it separates:
- data ingestion,
- feature construction,
- model inference.

---

## 15. Daily and intraday recommended feature set

For your current assets and thesis scope, a strong first experiment set would be:

## Daily commodities
- log returns,
- rolling volatility proxy.

## Daily SPY / QQQ
- log returns,
- absolute returns,
- rolling volatility proxy.

## Intraday SPY / QQQ
- 5-minute or 15-minute log returns,
- session-reset rolling volatility,
- optionally realized volatility blocks.

This gives you a coherent family of real-data observation designs without exploding the experiment count too early.

---

## 16. What should be compared in the thesis

This phase should naturally create one of your first real experimental comparison axes:

\[
\text{same detector backbone} \quad + \quad \text{different observation definitions}
\]

That means you can compare:
- MS detector on returns,
- MS detector on volatility proxy,
- MS detector on standardized or absolute returns.

This is valuable because it helps answer:

> Is change in financial market state better captured through returns, volatility, or another econometric statistic?

That is very aligned with your thesis topic.

---

## 17. Testing requirements for Phase 16

This phase needs dedicated tests because feature bugs can silently distort the whole detector.

You should test at least:

## Return computation tests
- log return formula correct,
- sign and scaling correct,
- daily and intraday handling consistent.

## Rolling feature tests
- rolling windows are trailing only,
- values match expected manual calculations,
- warmup behavior is correct.

## Session-aware tests
- rolling features reset correctly at session boundaries if configured,
- overnight gaps are not accidentally treated as normal continuity.

## Normalization tests
- scaler fitted only on training data,
- same scaler reused on validation/test,
- no leakage across partitions.

## Metadata tests
- feature metadata preserved correctly,
- warmup trimming recorded correctly.

### Deliverable
A trustworthy feature-engineering layer.

### Code changes
Add:
- unit tests for feature formulas,
- integration tests from cleaned market data to observation stream,
- leakage-focused tests for scaling policies.

---

## 18. Common conceptual mistakes to avoid

### Mistake 1 — Feeding raw prices directly into the model without justification
This is usually a weak design for financial changepoint detection.

### Mistake 2 — Using future-looking rolling windows
This would invalidate online evaluation.

### Mistake 3 — Mixing session boundaries into intraday rolling statistics without explicit policy
This can distort volatility features badly.

### Mistake 4 — Normalizing on the full dataset before splitting
That creates leakage.

### Mistake 5 — Treating feature design as a coding detail rather than a modeling decision
In your thesis, feature choice is part of the scientific hypothesis.

### Mistake 6 — Adding too many features too early
A smaller, well-justified feature family is better for a thesis than a noisy combinatorial explosion.

---

## 19. Deliverables of Phase 16

By the end of this phase, you should have:

### Mathematical / methodological deliverables
- a formal definition of the observed process \(y_t\) for real financial experiments,
- documented feature families:
  - log returns,
  - absolute returns,
  - rolling volatility proxy,
  - and any selected extensions,
- a causal rolling-feature policy,
- a session-aware intraday feature policy,
- a train-only normalization policy if scaling is used.

### Architectural deliverables
- a dedicated feature-engineering layer between real-data ingestion and model inference,
- a clear separation between:
  - raw market data,
  - feature policy,
  - fitted scaler,
  - model-ready observation stream,
- support for multiple feature configurations in experiments.

### Code-structure deliverables
You should add or revise, where appropriate:

- a dedicated `features` module,
- feature-family enums or config types,
- log-return computation utilities,
- absolute-return and squared-return utilities,
- rolling-statistics utilities,
- session-aware rolling feature support,
- optional scaler abstractions,
- fitted-scaler artifacts,
- observation-stream output types,
- feature metadata and provenance tracking,
- tests for formula correctness, causality, warmup behavior, session handling, and leakage prevention.

### Experimental deliverables
- at least a small, clearly defined family of financial observation processes,
- the ability to rerun the full detector pipeline under different feature definitions,
- a strong basis for comparing whether market changepoints are better captured by return-based or volatility-based observations.

---

## 20. Minimal final summary

Phase 16 is the phase that answers:

\[
\boxed{
\text{What is the observed financial process that the Markov Switching detector should model?}
}
\]

For your thesis, this should not be left implicit.

This phase should end with:
- a solid econometric observation design,
- a causal and reproducible feature pipeline,
- clear distinction between return-based and volatility-based observation processes,
- and a Rust feature layer that cleanly feeds the model and online detector.