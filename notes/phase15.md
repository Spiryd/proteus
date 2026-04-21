# Phase 15 — Build the Real Financial Data Pipeline

## Goal

In this phase, build a **clean, explicit, reproducible data pipeline** for real financial time series.

At this point in the project, you already have:

- a Markov Switching modeling core,
- offline fitting,
- online filtering,
- detector logic,
- benchmark logic for synthetic settings.

What is still missing is a robust bridge from **raw market data** to a **well-defined observed process** that can actually be used in the thesis experiments.

This phase is that bridge.

Its purpose is to ensure that when you say:

> “I trained the detector on real commodity, SPY, and QQQ data,”

that statement has a precise and reproducible meaning.

---

## 1. Why this phase is necessary

For financial time series, data handling is not a minor implementation detail.  
It is part of the scientific definition of the experiment.

A large fraction of empirical financial work becomes unreliable because the data pipeline is vague about:

- what timestamps mean,
- how missing observations are handled,
- how sessions are defined,
- whether overnight gaps are mixed into intraday returns,
- how splits are performed,
- whether data were resampled before or after filtering.

So for your thesis, the real-data pipeline must be treated as a first-class modeling component.

This is especially important because your thesis explicitly concerns:

- time series,
- market data,
- online detection,
- statistical analysis of change.

If the data layer is inconsistent, then:
- model fitting becomes ambiguous,
- detector comparisons become unfair,
- results become hard to defend in writing.

---

## 2. Scope of this phase

This phase is about **real market data ingestion and preprocessing**, not yet about:
- feature engineering beyond basic structural preparation,
- model fitting itself,
- benchmark comparison,
- real-data evaluation interpretation.

Those come later.

The scope here is to define how raw daily and intraday market data become a clean time-indexed dataset suitable for later phases.

For your setup, the pipeline should support at least:

- daily commodity prices,
- daily SPY and QQQ,
- intraday SPY and QQQ.

---

## 3. What counts as “real financial data pipeline”

In this phase, a real-data pipeline should cover the following sequence:

1. **load raw data**,  
2. **parse and normalize timestamps**,  
3. **sort and validate chronology**,  
4. **clean or flag missing / duplicated observations**,  
5. **define the trading session structure**,  
6. **apply optional resampling**,  
7. **attach metadata**,  
8. **split into train / validation / test by time**,  
9. **export a clean dataset representation for later modeling phases**.

This pipeline should be deterministic and reproducible.

---

## 4. Data domains in your thesis

You have three main data regimes.

## 4.1 Daily commodity prices

These likely behave differently from equity ETF data in:
- trading schedules,
- gaps,
- volatility patterns,
- possible holiday alignment,
- market microstructure assumptions.

For daily data, the key structural issues are:
- calendar alignment,
- missing dates,
- whether to use settlement or close prices,
- consistent ordering across multiple assets.

---

## 4.2 Daily SPY / QQQ

These are simpler than intraday data, but still require care.

Important issues include:
- market holidays,
- missing days,
- dividend-adjusted vs raw prices if relevant,
- whether your input is close, adjusted close, OHLC, or something else.

For a thesis, this must be documented explicitly.

---

## 4.3 Intraday SPY / QQQ

This is the most demanding case.

The main issues are:
- timestamp timezone,
- regular trading hours vs extended hours,
- intraday session boundaries,
- missing bars,
- resampling,
- overnight gaps,
- partial trading days.

This part of the pipeline must be designed especially carefully because your detector is online and session structure matters a lot.

---

## 5. Data model: what the raw dataset should represent

Before preprocessing, define what a raw observation represents.

For the financial pipeline, each raw row should conceptually include at least:

- timestamp,
- asset identifier,
- one or more price fields,
- optional volume,
- optional OHLC fields,
- optional source metadata.

For a daily dataset, one row usually represents one trading day.  
For an intraday dataset, one row usually represents one time bar.

At this phase, you do **not yet** need to decide the final observation feature for the model.  
But you do need to define the **market-data unit** unambiguously.

That is the object the rest of the project will transform later.

---

## 6. Timestamp normalization

This is one of the most important parts of the phase.

A timestamp must have a clear and consistent meaning.

You need to decide and document:

- what timezone raw timestamps are interpreted in,
- whether all timestamps are converted to one canonical timezone,
- whether daily data are stored as dates only,
- whether intraday timestamps are stored at bar open or bar close time.

These decisions matter because the online detector operates step by step, so the timestamp is part of the state ordering.

### For daily data
A clean approach is:
- store a trading date,
- preserve asset identity,
- attach calendar metadata if needed.

### For intraday data
A clean approach is:
- convert everything to one canonical timezone,
- document whether timestamps refer to the start or end of the bar,
- ensure ordering is strictly increasing within each asset stream.

---

## 7. Chronological ordering and duplication checks

A financial time-series detector assumes a valid chronological sequence.

So the pipeline must verify:

- observations are sorted by time,
- there are no duplicated timestamps within an asset series unless explicitly handled,
- there are no backward jumps in time,
- no bars exist outside the intended session definition unless explicitly allowed.

These are not optional checks.  
If a stream is not correctly ordered, online detection loses meaning.

### Deliverable
A chronology validation policy for all datasets.

### Code changes
You should add:
- ordering validation utilities,
- duplicate detection / resolution logic,
- stream consistency checks,
- dataset-level error or warning reporting.

---

## 8. Missing data handling

Financial datasets frequently contain missing observations.

The pipeline must define what “missing” means in each case.

## 8.1 Daily data missingness
Possible causes:
- weekends,
- holidays,
- data vendor gaps,
- suspended trading,
- asset-specific missing entries.

Not every absent calendar day is a missing observation.  
You must distinguish:
- non-trading days,
- actual missing data.

## 8.2 Intraday missingness
Possible causes:
- sparse vendor coverage,
- missing bars,
- exchange issues,
- half days,
- filtering to regular trading hours,
- asset-specific data holes.

For intraday data, missingness is much more important because the online detector expects a regular or at least interpretable temporal flow.

### Policy choices you must define
For both daily and intraday data, decide whether to:
- drop incomplete segments,
- flag missing points,
- forward-fill some fields,
- reconstruct bars,
- or reject corrupted sessions entirely.

For your thesis, I recommend:
- **never silently impute price dynamics**,  
- instead prefer:
  - explicit dropping,
  - explicit gap flags,
  - or session-level exclusion.

This keeps the pipeline more defensible.

### Deliverable
A missing-data policy by data frequency.

### Code changes
You should add:
- missing-observation detection,
- gap reporting,
- explicit gap-handling policies,
- dataset cleaning summaries.

---

## 9. Trading calendar consistency

For real market data, timestamps must be interpreted relative to the actual trading calendar.

This matters because:
- weekends should not be treated as missing daily bars,
- intraday session boundaries depend on exchange hours,
- half trading days can distort intraday structure,
- overnight gaps are not just ordinary inter-bar moves.

So the data pipeline must define a calendar-consistency layer.

At minimum, it should know:
- whether the asset is daily or intraday,
- which exchange/session convention is assumed,
- which observations belong to a regular trading session.

### Why this matters for your detector
If the model later sees overnight gaps treated as ordinary intraday transitions, it may fire spurious changepoints for structural reasons unrelated to market-state changes.

So this phase must decide whether intraday analysis:
- includes overnight transitions,
- or resets at each trading session.

For the first thesis version, I strongly recommend:
- **treat regular trading sessions explicitly**,  
- and do **not** mix overnight gaps into normal intraday transitions unless you deliberately design an overnight-aware model.

---

## 10. Intraday session boundaries

This is one of the most important design decisions for your real-data pipeline.

For intraday SPY / QQQ data, define clearly:

- whether you use only regular trading hours,
- whether premarket / after-hours are excluded,
- whether each trading day is treated as a separate session,
- whether the detector state resets at session boundaries or carries over.

At the data-pipeline level, you do not yet need to finalize detector reset policy, but you do need to mark session boundaries explicitly.

### Recommended first thesis design
For intraday experiments:
- use regular trading hours only,
- explicitly segment the stream into sessions,
- keep session metadata,
- and make overnight gaps visible at the metadata level rather than hiding them inside a continuous time index.

This is much easier to reason about and defend.

### Deliverable
A session-aware intraday dataset format.

### Code changes
You should add:
- session labeling,
- session boundary markers,
- optional per-session slicing,
- regular-trading-hours filtering.

---

## 11. Resampling policy

Intraday data often come at different granularities.  
You need a defined resampling policy.

Examples:
- keep raw 1-minute data,
- resample to 5-minute bars,
- resample to 15-minute bars.

This decision must be explicit because it changes:
- noise level,
- detector sensitivity,
- runtime cost,
- number of alarms,
- and regime interpretation.

### Recommended first thesis design
Support a configurable resampling layer, but standardize at least one or two canonical resolutions, for example:
- 5-minute,
- 15-minute.

That will make comparisons more stable and easier to report.

### Important principle
Resampling must happen in a clearly documented place in the pipeline, before the data become the input stream for the detector.

### Deliverable
A documented resampling policy for intraday data.

### Code changes
You should add:
- bar aggregation / resampling utilities,
- frequency metadata,
- checks that resampling preserves time ordering and session structure.

---

## 12. Asset metadata

Each dataset should carry metadata that later phases will need.

At minimum, metadata should include:
- asset symbol,
- frequency,
- source,
- timezone / date convention,
- session convention,
- price field used,
- resampling rule if any,
- number of observations,
- train/validation/test boundaries.

This is important for reproducibility and for generating thesis tables later.

### Deliverable
A dataset metadata schema.

### Code changes
Add:
- a metadata struct,
- persistence of metadata with cleaned datasets,
- export support for metadata summaries.

---

## 13. Train / validation / test split by time

For time series, splits must be chronological.

This phase should define a strict temporal split policy.

## Why this matters
Random train/test shuffling is invalid in your setting.

You need a split such as:

- **train**: historical period for offline fitting,
- **validation**: period for threshold / detector policy selection,
- **test**: held-out future period for final evaluation.

This applies to both:
- daily data,
- intraday data.

### For the thesis
You should define the split policy explicitly and consistently across assets.

For example:
- earliest segment = training,
- middle segment = validation,
- latest segment = final test.

### Deliverable
A chronological split protocol.

### Code changes
You should add:
- split logic by timestamp/date,
- split metadata,
- safeguards against leakage,
- utilities to export split summaries.

---

## 14. Leakage prevention

This phase must explicitly prevent temporal leakage.

That means:
- no using test data to fit model parameters,
- no using future periods to define thresholds,
- no using full-sample normalization that sees test periods during training preprocessing.

In finance, leakage can creep in through subtle preprocessing decisions.

So your data pipeline should be designed so that:
- splitting occurs early enough,
- and downstream phases clearly know which partition they are using.

### Deliverable
A leakage-aware preprocessing and split policy.

### Code changes
You should add:
- partition-aware dataset objects,
- restrictions or checks preventing accidental cross-partition use,
- explicit split labels attached to outputs.

---

## 15. Daily vs intraday data should remain distinct data modes

Do not force daily and intraday data into one overly generic abstraction too early.

They share some logic, but they differ in:
- session structure,
- missing data meaning,
- calendar semantics,
- resampling needs,
- online interpretation.

So the pipeline should have a shared foundation, but still preserve separate frequency-aware handling.

### Deliverable
A unified-but-frequency-aware data architecture.

### Code changes
You should structure the data layer so that:
- shared parsing and validation utilities are reused,
- but daily and intraday pipelines remain explicitly distinct where needed.

---

## 16. Step-by-step guide for Phase 15

## Step 1 — Define canonical raw data schemas

Explicitly document what raw daily and intraday rows look like.

### Deliverable
A raw-schema definition for each data source type.

### Code changes
Add:
- raw record structs or parsing schemas,
- source-specific loaders,
- validation of required columns.

---

## Step 2 — Normalize timestamps and ordering

Convert timestamps to canonical internal format and sort observations.

### Deliverable
A time-normalized ordered dataset.

### Code changes
Add:
- timestamp parsing,
- timezone normalization,
- chronological sorting,
- duplicate rejection or resolution.

---

## Step 3 — Add data-quality validation

Validate:
- missing timestamps,
- duplicates,
- invalid prices,
- non-monotone time order,
- malformed sessions.

### Deliverable
A data-quality validation layer.

### Code changes
Add:
- validation reports,
- warning/error types,
- summary statistics for cleaning.

---

## Step 4 — Add session-aware intraday preprocessing

For intraday data:
- define sessions,
- filter regular trading hours if chosen,
- mark or split session boundaries,
- handle overnight gaps explicitly.

### Deliverable
A session-aware intraday pipeline.

### Code changes
Add:
- session classifiers,
- session metadata,
- session splitting utilities,
- RTH filtering.

---

## Step 5 — Add resampling support

For intraday data, optionally resample to chosen bar frequency.

### Deliverable
A reproducible resampling stage.

### Code changes
Add:
- aggregation/resampling logic,
- bar-alignment rules,
- output frequency metadata.

---

## Step 6 — Add dataset metadata and provenance

Attach descriptive metadata to every cleaned dataset.

### Deliverable
A dataset object that carries both data and provenance.

### Code changes
Add:
- metadata structs,
- serialization-friendly summaries,
- provenance fields.

---

## Step 7 — Add chronological train/validation/test split logic

Implement split-by-time utilities.

### Deliverable
Leakage-safe data partitions for later training and evaluation.

### Code changes
Add:
- time-based split utilities,
- split result objects,
- partition labels,
- leakage checks.

---

## 17. Suggested data objects

By the end of this phase, your project should probably have at least these conceptual layers.

## Raw market data object
Represents vendor/raw file content.

## Clean market series object
Represents validated, time-ordered, cleaned data for one asset and frequency.

## Session-aware intraday object
Represents cleaned intraday data with session boundaries and frequency metadata.

## Partitioned dataset object
Represents train / validation / test splits.

## Metadata object
Represents provenance, schema, frequency, session convention, and preprocessing decisions.

These do not need to be implemented exactly with these names, but these roles should exist.

---

## 18. Testing requirements for Phase 15

This phase needs serious tests because data bugs are easy to miss and very costly later.

You should test at least:

## Loader tests
- parses expected fields correctly,
- rejects malformed input.

## Timestamp normalization tests
- preserves correct order,
- converts timezones correctly,
- handles daily vs intraday consistently.

## Session tests
- correctly identifies session boundaries,
- excludes out-of-session bars when configured,
- handles overnight transitions correctly.

## Resampling tests
- preserves chronological order,
- produces expected timestamps,
- aggregates bars consistently.

## Split tests
- train/validation/test are disjoint,
- all observations are assigned correctly,
- no temporal leakage occurs.

### Deliverable
A reliable tested data-ingestion foundation.

### Code changes
Add:
- unit tests for parsing and validation,
- integration tests for end-to-end cleaning,
- dataset fixture files if practical.

---

## 19. Common conceptual mistakes to avoid

### Mistake 1 — Using raw prices without documenting what field they are
Daily close, adjusted close, and intraday last price are not interchangeable without documentation.

### Mistake 2 — Treating non-trading days as missing observations
Calendar absence and data absence are not the same thing.

### Mistake 3 — Mixing overnight gaps into ordinary intraday continuity without saying so
This can strongly distort online detection behavior.

### Mistake 4 — Random train/test splitting
This is invalid for your time-series detector setting.

### Mistake 5 — Letting preprocessing see the full future before splitting
That can create leakage.

### Mistake 6 — Hiding frequency-specific logic in one opaque abstraction
Daily and intraday require visibly different treatment.

---

## 20. Deliverables of Phase 15

By the end of this phase, you should have:

### Mathematical / methodological deliverables
- a formally documented real-data preprocessing protocol,
- clear timestamp and session conventions,
- a missing-data policy,
- a resampling policy,
- a chronological split policy,
- a leakage-prevention policy.

### Architectural deliverables
- a dedicated real-data ingestion and preprocessing layer,
- distinct handling paths for daily and intraday data,
- session-aware intraday support,
- metadata/provenance tracking,
- partition-aware dataset representation.

### Code-structure deliverables
You should add or revise, where appropriate:

- a dedicated `data` module,
- loaders for daily and intraday files,
- raw-record schemas,
- clean dataset objects,
- timestamp normalization utilities,
- chronology and duplicate checks,
- missing-data and cleaning reports,
- session-aware intraday preprocessing,
- resampling utilities,
- metadata structs,
- chronological split utilities,
- partition-aware dataset objects,
- tests for loading, cleaning, session handling, resampling, and splitting.

### Trust deliverables
- confidence that later model fitting and detection are driven by a defensible data pipeline,
- reproducible preparation of commodity, SPY, and QQQ datasets,
- a clean foundation for the next phase, where feature/statistics construction for the model input can be built on top of this data layer.

---

## 21. Minimal final summary

Phase 15 is the step that turns raw financial files into a scientifically usable market-data foundation.

The central principle is:

\[
\boxed{
\text{In a time-series thesis, the data pipeline is part of the model definition.}
}
\]

This phase should end with:
- clean daily and intraday market datasets,
- explicit session and timestamp conventions,
- leakage-safe temporal splits,
- and a reproducible Rust data layer that all later modeling and evaluation phases can trust.