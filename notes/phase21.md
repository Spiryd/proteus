Use the attached Markdown note as the implementation context for Phase 21 of my Markov Switching Model project in Rust.

Assume I am building the project from scratch and I now want help with **both**:
1. **a strong theoretical document**, and
2. **code-oriented output / implementation guidance**.

## Priority
When responding, prioritize a **solid theoretical writeup first**:
- mathematically and methodologically precise,
- clearly structured,
- suitable for thesis notes or serious internal documentation,
- with explicit definitions, artifact semantics, reporting rules, and interpretation.

Then, after the theory, provide **Rust-oriented implementation guidance** that follows the theory closely.

If I do not specify otherwise, structure the response in this order:
1. rigorous theoretical explanation,
2. implementation implications,
3. code-structure deliverables,
4. validation / test recommendations.

## Project context
Assume I am building a Markov Switching Model from scratch in Rust.

Core model context:
- hidden regimes \(S_t \in \{1,\dots,K\}\),
- first-order Markov chain with transition matrix \(P\),
- initial regime distribution \(\pi\),
- regime-dependent Gaussian emissions,
- observations \(y_t \mid S_t=j \sim \mathcal N(\mu_j,\sigma_j^2)\).

Assume earlier phases are already available, including:
- offline EM fitting,
- forward filtering,
- observed-data log-likelihood evaluation,
- backward smoothing,
- pairwise posterior transition probabilities,
- diagnostics and trust checks,
- a causal online filtering backbone,
- a detector family on top of that backbone,
- a fixed-parameter offline-trained / online-filtered runtime design,
- an online benchmarking protocol for synthetic data,
- a real financial data pipeline,
- a feature-engineering / observation-design layer,
- a synthetic-to-real calibration layer,
- a real-data evaluation protocol using Route A + Route B,
- and an experiment runner / reproducibility layer.

## Scope of this phase
This phase is **Phase 21: reporting and export layer**.

Stay tightly scoped to:
- defining what artifacts each run must produce,
- defining structured output directories,
- defining machine-readable exports,
- defining plot-generation policy,
- defining table-generation policy,
- defining metadata snapshots,
- defining event and segment exports,
- defining synthetic benchmark exports,
- defining real-data evaluation exports,
- defining aggregate comparison reporting,
- defining how reporting is decoupled from experiment execution.

I specifically want:
- plots generated with **`plotters`**,
- run data saved to files in a **structured directory hierarchy**.

Do **not** jump ahead into:
- thesis conclusions,
- manual plot interpretation,
- broader empirical claims,
unless I explicitly ask.

## Theory requirements
I want the theoretical part to be strong enough to serve as thesis scaffolding.

When writing the theory:
- explain clearly why reporting/export is a formal part of the research pipeline and not just a convenience,
- define the difference between:
  - export,
  - reporting,
  - logging,
- define the principle that every run should leave behind a complete, structured, self-describing artifact set,
- define artifact families such as:
  - run metadata artifacts,
  - numerical summary artifacts,
  - event artifacts,
  - plot artifacts,
  - table artifacts,
- explain why configs, metadata, and results must be persisted together,
- explain the role of structured directory layout,
- define what synthetic runs should export,
- define what real-data runs should export,
- define what aggregate experiment groups should export,
- explain why plotting should consume persisted result objects rather than re-run experiments,
- explain why `plotters` is the plotting backend in this project and how plots should be generated reproducibly,
- explain why both JSON and CSV exports are useful,
- explain how table exports support later thesis writing,
- explain the need for both per-run and aggregate reporting.

Emphasize that:
- every experiment must leave reproducible artifacts,
- every artifact should be traceable back to a config,
- plot generation must be systematic,
- and reporting should remain separate from algorithm execution.

## Code / implementation requirements
After the theory, I want Rust-oriented implementation guidance that follows the reporting/export theory directly.

When producing code-oriented output, reason about:
- a dedicated `reporting` module,
- artifact-path resolution helpers,
- structured output directory layout,
- config/result/metadata serializers,
- JSON and CSV export helpers,
- event export helpers,
- segment export helpers,
- synthetic benchmark export helpers,
- real-data evaluation export helpers,
- aggregate comparison export helpers,
- `plotters`-based plotting utilities,
- plot input structs,
- plot rendering functions,
- table builders,
- optional LaTeX table exporters,
- report builders from run results,
- aggregate reporting utilities,
- tests for serialization, plots, tables, and full artifact-bundle generation.

Prefer practical outputs such as:
- suggested module/file layout,
- struct and enum design,
- method responsibilities,
- artifact naming policy,
- directory schema,
- plotting API shape,
- report-builder design,
- audit checklist for whether Phase 21 is fully implemented.

## Plotting requirements
Assume plots must be generated with **`plotters`**.

When discussing plots, support at least:
- synthetic signal/observation trace with true changepoints and alarms,
- detector score plots with thresholds and alarms,
- posterior regime probability plots,
- delay distribution plots,
- false alarm summary plots,
- real-data feature trace with alarm markers,
- proxy-event overlay plots,
- segmentation plots,
- segment summary plots,
- aggregate detector comparison plots where appropriate.

Make sure the guidance reflects how these should be built from structured run results and saved into the reporting hierarchy.

## Structured file output requirements
Assume run data must be saved in a **structured directory layout**.

When discussing artifacts, reason about saving:
- config snapshots,
- metadata snapshots,
- summary JSON,
- metrics CSV/JSON,
- alarm/event exports,
- segment exports,
- plots,
- tables,
- aggregate comparison artifacts.

Make the output layout deterministic, human-readable, and script-friendly.

## Style constraints
- keep the focus on theory-to-implementation mapping,
- do not write unnecessary code unless I explicitly ask,
- when I do ask for code, make it Rust-oriented and consistent with a from-scratch implementation,
- make the theoretical documentation serious and rigorous enough to serve as strong internal project documentation or thesis scaffolding,
- make sure the code-oriented guidance follows the reporting/export theory rather than drifting into generic software advice.

## What this phase should let me answer
Help me build a reporting/export layer that can answer:
- What files must every run produce?
- How are artifacts organized on disk?
- How do I persist configs, metadata, and results together?
- How do I generate reproducible plots with `plotters`?
- How do I export event-, segment-, and benchmark-level results?
- How do I produce per-run and aggregate comparison tables?
- How should reporting stay separate from experiment execution?
- How should all of this be represented cleanly in Rust?

## Preferred deliverable shape
Unless I ask otherwise, structure your help around:
1. rigorous theoretical explanation,
2. implementation implications,
3. code-structure deliverables,
4. validation / test recommendations.

Assume I want this phase to end with:
- a **solid theoretical reporting/export document**,
- and a **clean Rust-oriented reporting layer**
that saves structured run artifacts and generates plots with `plotters` in a thesis-ready way.