# Phase 20 — Build the Experiment Runner and Reproducibility Layer

## Goal

In this phase, build the **experiment orchestration layer** that turns all previously implemented components into a usable research system.

Up to this point, you have built most of the scientific and algorithmic parts:

- Markov Switching model,
- offline fitting,
- online filtering,
- detector variants,
- synthetic benchmarking protocol,
- real-data preprocessing,
- observation design,
- synthetic-to-real calibration,
- real-data evaluation.

What is still missing is the layer that can **run all of this repeatedly, systematically, and reproducibly**.

That is the purpose of Phase 20.

This phase is about building the machinery that can do the following, end to end:

1. choose a dataset or scenario,
2. choose a feature pipeline,
3. train the model if needed,
4. run online detection,
5. run evaluation,
6. export structured results,
7. repeat the same process across many configurations.

Without this phase, the thesis remains a collection of components.  
With this phase, it becomes a **research pipeline**.

---

## 1. Why this phase is necessary

A thesis-grade empirical study needs more than correct algorithms.

It also needs:

- repeatability,
- comparability,
- controlled configuration,
- structured outputs,
- clean separation between experiment inputs and results,
- support for many runs across datasets and detector settings.

If these are not formalized, then:
- experiments become manual,
- comparisons become inconsistent,
- results become hard to reproduce,
- and thesis tables and figures become difficult to trust.

So Phase 20 is the step that turns the project into an actual experimental platform.

---

## 2. What an experiment runner must do

An experiment runner is not just a script that calls a few functions.

It should be understood as a formal orchestration system whose role is to coordinate:

## Input selection
- dataset,
- synthetic scenario or real asset,
- frequency,
- feature family,
- model configuration,
- detector configuration,
- evaluation protocol.

## Execution
- preprocessing,
- feature generation,
- offline fitting if required,
- online detection,
- evaluation,
- diagnostics,
- reporting.

## Output
- structured run metadata,
- fitted parameter summaries,
- benchmark results,
- real-data evaluation results,
- timing,
- export artifacts.

That means the runner must sit above the individual algorithmic layers and coordinate them without mixing their responsibilities.

---

## 3. What this phase must support in your project

For your thesis, the experiment runner should support at least:

## 3.1 Synthetic runs
For controlled experiments with:
- known changepoints,
- calibrated scenario families,
- event-level online benchmarking.

## 3.2 Daily real-data runs
For:
- commodities,
- daily SPY,
- daily QQQ.

## 3.3 Intraday real-data runs
For:
- intraday SPY,
- intraday QQQ,
- session-aware evaluation.

## 3.4 Multiple detector variants
At minimum:
- hard switch detector,
- posterior transition detector,
- surprise detector.

## 3.5 Multiple detector parameterizations
For example:
- different thresholds,
- different persistence settings,
- different cooldown settings.

## 3.6 Multiple observation families
For example:
- returns,
- absolute returns,
- volatility proxy.

This means the runner must be **configuration-driven** rather than hardcoded.

---

## 4. The central idea of this phase

The central idea is:

\[
\boxed{
\text{An experiment should be fully determined by an explicit configuration object.}
}
\]

This is the most important principle of Phase 20.

That means:
- no hidden settings,
- no hardcoded file paths inside evaluation logic,
- no manual tweaking inside code between runs,
- no ambiguous thresholds.

Instead, each run should be reproducible from a configuration that describes:

- what data are used,
- how they are transformed,
- how the model is trained,
- which detector is used,
- how evaluation is performed,
- where outputs are written.

This is what makes the experimental study scientifically reproducible.

---

## 5. What an experiment is, formally

A single experiment run should be thought of as a mapping:

\[
\text{ExperimentConfig}
\longrightarrow
\text{ExperimentResult}
\]

where:

## Input: `ExperimentConfig`
Contains everything needed to specify the run.

## Output: `ExperimentResult`
Contains everything needed to inspect, compare, and reproduce the run.

This is the key abstraction of the phase.

---

## 6. Core categories of experiment configuration

The configuration should be broken into explicit sub-configurations.

## 6.1 Data configuration
Specifies:
- synthetic vs real,
- asset,
- frequency,
- date range,
- session policy,
- source path or dataset ID.

## 6.2 Feature configuration
Specifies:
- feature family,
- rolling window if relevant,
- session-reset behavior,
- scaling policy.

## 6.3 Model configuration
Specifies:
- number of regimes,
- training procedure settings,
- calibration source if synthetic,
- parameter policy (fixed offline-trained online-filtered).

## 6.4 Detector configuration
Specifies:
- detector type,
- thresholds,
- persistence policy,
- cooldown policy.

## 6.5 Evaluation configuration
Specifies:
- synthetic benchmark settings,
- real-data Route A settings,
- real-data Route B settings,
- detection windows,
- alignment policy.

## 6.6 Output configuration
Specifies:
- output directory,
- file naming,
- serialization formats,
- whether plots/tables/raw traces are saved.

This configuration decomposition should be explicit in both theory and code.

---

## 7. Why reproducibility must be built into the runner

This phase must treat reproducibility as a first-class concern, not a side effect.

For synthetic runs, reproducibility requires:
- fixed random seeds,
- deterministic scenario generation,
- stored scenario parameters,
- saved configuration snapshots.

For real-data runs, reproducibility requires:
- saved dataset identifiers,
- explicit date ranges,
- saved feature and detector settings,
- recorded preprocessing metadata.

Without this, you cannot reliably recreate:
- tables,
- figures,
- ablation studies,
- detector comparisons.

So every experiment result should carry enough metadata to reconstruct the run.

---

## 8. Separation between experiment logic and algorithmic logic

A crucial architectural principle of this phase is:

\[
\text{runner orchestration} \neq \text{model implementation}
\]

The experiment runner should not contain:
- filtering math,
- detector score formulas,
- feature formulas,
- fitting algorithms.

Those belong to the lower layers.

The runner should instead:
- instantiate configs,
- call the right components,
- route outputs,
- collect metrics,
- serialize results.

This separation keeps the system maintainable and makes debugging much easier.

---

## 9. Synthetic and real runs should share the same orchestration philosophy

Even though synthetic and real experiments differ in evaluation, they should still pass through a common orchestration framework.

That means a top-level run should always look like:

1. resolve data source,
2. build observation stream,
3. train or load model,
4. run detector,
5. evaluate,
6. serialize results.

The details of evaluation may differ:
- synthetic uses known changepoints,
- real uses Route A + B,

but the runner structure should remain unified.

This is one of the biggest design goals of Phase 20.

---

## 10. Step-by-step guide for Phase 20

## Step 1 — Define the top-level experiment contract

Define what counts as one experiment run.

A run should specify:
- one dataset or scenario,
- one feature pipeline,
- one model configuration,
- one detector configuration,
- one evaluation configuration.

A run should produce:
- one self-contained result object,
- optionally multiple exported artifacts.

### Deliverable
A formal experiment input/output definition.

### Code changes
You should add a dedicated `experiments` module or equivalent orchestration layer.

This is the main architectural addition of the phase.

---

## Step 2 — Define the experiment configuration schema

Create a structured configuration object that contains all necessary subconfigs.

At minimum, the schema should cover:
- data,
- features,
- model,
- detector,
- evaluation,
- output,
- reproducibility metadata (seed, run label, version info if useful).

### Deliverable
A complete configuration schema for experiment runs.

### Code changes
You should add:
- an `ExperimentConfig` type,
- nested config structs/enums for each subsystem,
- serialization support for configs.

This is one of the most important code deliverables of the phase.

---

## Step 3 — Define the experiment result schema

The result should not be a loose collection of prints or temporary objects.

It should contain:

- run metadata,
- status,
- timing,
- model summary,
- detector summary,
- evaluation summary,
- paths to exported artifacts if applicable.

### Deliverable
A structured experiment result object.

### Code changes
You should add:
- an `ExperimentResult` type,
- sub-result objects for synthetic and real runs,
- optional raw trace references,
- serialization support for results.

---

## Step 4 — Add a run planner / resolver

The runner must be able to interpret a configuration and determine what actions are needed.

Examples:
- if run type is synthetic, generate synthetic data,
- if run type is real, load the specified dataset,
- if training mode is “fit”, train the model,
- if training mode is “reuse fitted artifact”, load the saved model,
- if evaluation mode is synthetic, call benchmark module,
- if evaluation mode is real, call Route A + B real-eval layer.

### Deliverable
A deterministic run-resolution policy.

### Code changes
Add:
- experiment dispatch logic,
- scenario/data resolver utilities,
- artifact-loading hooks where needed.

---

## Step 5 — Add a training orchestration layer

For runs that require offline fitting, the experiment runner should:
- build the training observation stream,
- fit the model,
- validate diagnostics,
- freeze the runtime model.

This should not be hidden or implicit.

### Deliverable
A clearly orchestrated training stage within the runner.

### Code changes
Add:
- a training orchestration function or service,
- conversion from fit result to runtime frozen model,
- persisted model artifacts if needed.

---

## Step 6 — Add an online execution layer

Once the runtime model is available, the runner should:
- stream the observation sequence through the online filter,
- run the chosen detector,
- collect alarms,
- store score traces if configured.

### Deliverable
A detector execution stage controlled by the runner.

### Code changes
Add:
- online-run orchestration,
- alarm collection,
- optional trace recording policies.

---

## Step 7 — Add evaluation routing

The runner should then pass the detector outputs to the appropriate evaluation layer.

### Synthetic runs
- call synthetic benchmark logic.

### Real runs
- call Route A proxy-event evaluation,
- call Route B segmentation evaluation.

### Deliverable
A clear evaluation dispatch policy.

### Code changes
Add:
- evaluation router logic,
- unified evaluation result wrapper.

---

## Step 8 — Add output/export orchestration

The runner must save outputs in a structured way.

At minimum, it should be able to export:
- config snapshot,
- result summary,
- model summary,
- metrics,
- alarm list,
- timing,
- optional raw traces.

### Deliverable
A reproducible artifact output protocol.

### Code changes
Add:
- structured output directory creation,
- result serialization,
- config serialization,
- optional CSV/JSON export helpers.

---

## Step 9 — Add reproducibility guarantees

For synthetic runs, require:
- fixed seed,
- saved scenario config.

For all runs, save:
- exact experiment config,
- run timestamp,
- output artifact paths,
- optional code version/tag if available.

### Deliverable
A reproducibility layer.

### Code changes
Add:
- seed handling,
- config snapshot saving,
- run ID generation,
- metadata recording in result objects.

---

## 11. Support for experiment grids

You will not run only one configuration.

You will likely want:
- several assets,
- several frequencies,
- several feature families,
- several detector thresholds,
- several detector types.

So the runner should support not only single runs, but also **experiment grids** or batches.

This means the runner should be able to expand a family of configurations into many concrete runs.

### Deliverable
A batch-experiment capability.

### Code changes
You should add:
- a batch config or run list abstraction,
- iteration over config grids,
- per-run isolation and result collection,
- aggregate run summaries.

This is one of the most useful practical additions for thesis work.

---

## 12. Structured output directories

The output system should not dump everything into one folder.

A clean structure might distinguish:
- synthetic vs real,
- dataset / asset,
- feature family,
- detector type,
- run ID.

For example, conceptually:

- run config,
- summary,
- metrics,
- alarms,
- traces,
- plots (if later generated).

The exact layout is up to you, but it must be:
- deterministic,
- human-readable,
- and easy to map back to configurations.

### Deliverable
A run-artifact directory policy.

### Code changes
Add:
- output-path resolver utilities,
- stable naming conventions,
- conflict-safe run directory creation.

---

## 13. Result objects should support both raw detail and summary

For a thesis, you need:
- detailed outputs for debugging,
- summary outputs for tables and figures.

So each experiment result should support both:

## Detailed outputs
- full alarm list,
- score trace,
- fit diagnostics,
- segmentation summaries,
- per-stream metrics.

## Summary outputs
- high-level metrics,
- final detector settings,
- timing summaries,
- core evaluation results.

This dual-level design prevents later pain when generating thesis materials.

### Deliverable
A layered result schema.

### Code changes
Add:
- detailed vs summary result sub-objects,
- optional raw-trace storage controls.

---

## 14. Synthetic and real experiment modes

The runner should explicitly distinguish at least two experiment modes:

## Synthetic mode
Includes:
- synthetic scenario generation,
- known-changepoint benchmark evaluation.

## Real mode
Includes:
- real dataset loading,
- Route A + B evaluation.

This distinction should be explicit in configs and result types, not hidden behind optional fields.

### Deliverable
A mode-aware experiment system.

### Code changes
Use enums or tagged config/result types so the run mode is explicit and statically clear.

---

## 15. Failure handling and run status

A real experiment runner must handle failures cleanly.

Possible failure points:
- invalid dataset,
- fitting failure,
- detector runtime failure,
- evaluation inconsistency,
- serialization/export failure.

So every run should record a status such as:
- success,
- partial success,
- failed during training,
- failed during evaluation,
- failed during export.

This is important because in thesis-scale runs, some configurations may fail and you need to know exactly why.

### Deliverable
A robust run-status protocol.

### Code changes
Add:
- run status enum,
- structured failure reporting,
- partial result retention where useful.

---

## 16. Logging and traceability

The runner should produce enough traceability to diagnose what happened in a run without opening the code.

At minimum, each run should log:
- run ID,
- config summary,
- major pipeline stages,
- warnings,
- completion status.

This does not need to be overly elaborate, but it should be systematic.

### Deliverable
A run-traceability policy.

### Code changes
Add:
- structured logging hooks,
- stage markers,
- warning collection into result artifacts.

---

## 17. Step-by-step theoretical workflow of one run

A fully specified experiment run should look like this:

### Input
\[
\text{ExperimentConfig}
\]

### Stage 1 — Data resolution
Load/generate the base time series.

### Stage 2 — Feature transformation
Construct the observed process \(y_t\).

### Stage 3 — Training
Fit the model offline or load a frozen fitted model.

### Stage 4 — Online execution
Run causal filtering and detector logic on the target stream.

### Stage 5 — Evaluation
Synthetic benchmark or real-data Route A + B evaluation.

### Stage 6 — Export
Serialize results and metadata.

### Output
\[
\text{ExperimentResult}
\]

This formal structure should appear in your documentation because it defines the experimental pipeline of the thesis.

---

## 18. Suggested configuration dimensions for your thesis

The runner should eventually support variation along these axes:

## Data axis
- synthetic calibrated scenario,
- daily commodity,
- daily SPY,
- daily QQQ,
- intraday SPY,
- intraday QQQ.

## Feature axis
- returns,
- absolute returns,
- rolling volatility.

## Detector axis
- hard switch,
- posterior transition,
- surprise.

## Detector-policy axis
- threshold,
- persistence,
- cooldown.

## Model axis
- number of regimes,
- feature-specific calibration settings.

This means the runner must support a large combinatorial space without turning into unstructured scripting.

---

## 19. Testing requirements for Phase 20

This phase needs tests even though it is orchestration-heavy.

You should test at least:

## Config tests
- configs serialize/deserialize correctly,
- invalid configs are rejected cleanly.

## Run resolution tests
- synthetic config routes to synthetic pipeline,
- real config routes to real pipeline,
- evaluation mode dispatch works correctly.

## Reproducibility tests
- same seed yields same synthetic scenario,
- same config yields equivalent core results under deterministic conditions.

## Artifact tests
- output directories created correctly,
- result and config snapshots written correctly.

## Failure tests
- failed runs produce meaningful status objects,
- errors do not silently corrupt unrelated runs.

### Deliverable
A reliable experiment orchestration layer.

### Code changes
Add:
- tests for config handling,
- integration tests for minimal end-to-end runs,
- failure-mode tests,
- artifact-output tests.

---

## 20. Suggested architecture for Phase 20

By the end of this phase, your project should conceptually have:

## Configuration layer
Defines experiment inputs.

## Runner layer
Dispatches the run through the correct stages.

## Artifact layer
Handles output directories, serialization, and reproducibility.

## Batch layer
Runs multiple configurations systematically.

## Result layer
Stores both per-run detail and summary metrics.

This is the architecture that turns the project into a proper empirical research system.

---

## 21. Common conceptual mistakes to avoid

### Mistake 1 — Running experiments manually with ad hoc parameter edits
This destroys reproducibility.

### Mistake 2 — Letting the runner implement model logic
The runner should orchestrate, not redefine algorithms.

### Mistake 3 — Not saving configs with results
Then successful runs become hard to reconstruct.

### Mistake 4 — Mixing synthetic and real evaluation logic inside one unstructured function
These should be routed explicitly.

### Mistake 5 — Writing outputs without a stable directory policy
That quickly becomes chaotic at thesis scale.

### Mistake 6 — Not preserving failure information
A failed run should still leave a trace explaining what happened.

---

## 22. Deliverables of Phase 20

By the end of this phase, you should have:

### Mathematical / methodological deliverables
- a formally documented end-to-end experiment workflow,
- a configuration-driven definition of a run,
- a reproducibility policy,
- an explicit distinction between synthetic and real experiment modes,
- a clear separation between run orchestration and algorithmic components.

### Architectural deliverables
- a dedicated experiment orchestration layer,
- a configuration schema,
- a result schema,
- a batch-run capability,
- a structured artifact/output system,
- a failure-handling and logging policy.

### Code-structure deliverables
You should add or revise, where appropriate:

- a dedicated `experiments` module,
- an `ExperimentConfig` type,
- nested config types for:
  - data,
  - features,
  - model,
  - detector,
  - evaluation,
  - output,
- an `ExperimentResult` type,
- synthetic vs real run mode handling,
- training and online execution orchestration,
- evaluation routing,
- output-path resolution,
- config/result serialization,
- seed handling,
- batch/grid execution support,
- run status and failure reporting,
- logging/traceability support,
- tests for config handling, reproducibility, routing, failure modes, and artifact writing.

### Thesis deliverables
- a clean and reproducible empirical workflow,
- a system capable of generating thesis-ready experiment artifacts,
- a practical bridge from implemented algorithms to actual empirical study.

---

## 23. Minimal final summary

Phase 20 is the step that transforms your detector project into a reproducible experimental platform.

The central principle is:

\[
\boxed{
\text{Every experiment must be fully defined by configuration and fully reconstructible from its artifacts.}
}
\]

This phase should end with:
- a formal experiment runner,
- structured configs,
- structured results,
- reproducible seeds,
- and a clean orchestration layer for synthetic and real runs.

That gives you the operational backbone needed to produce reliable thesis experiments at scale.