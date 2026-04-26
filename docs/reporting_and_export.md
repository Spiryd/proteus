# Reporting and Export Layer

**Phase 21 — Markov Switching Model Project**

---

## 1. Overview: Why Reporting is Formal Research Methodology

Reporting and export are not post-hoc conveniences bolted onto a completed experiment.

They are instead a **formal part of the research pipeline** that:

1. Transforms raw algorithmic outputs into reproducible, auditable artifacts.
2. Enables separation between algorithm execution and interpretation.
3. Decouples thesis writing from experiment re-running.
4. Provides structured evidence trails for each run.
5. Support systematic comparison and tabulation across many experiments.

Without systematic reporting, a detector research project becomes:
- Hard to compare configurations,
- Difficult to generate thesis figures,
- Prone to manual error in metric extraction,
- Non-reproducible from published results.

With systematic reporting, every run becomes a self-contained, traceable artifact that can be re-inspected, re-analyzed, or re-plotted without re-running the expensive detector or fitting pipeline.

---

## 2. Definitions: Export, Reporting, and Logging

These three terms have distinct roles in the pipeline:

### Export
**Definition:** Serializing experiment inputs, parameters, and numerical results into persistent, machine-readable files.

**Examples:**
- config snapshot (JSON),
- result summary (JSON),
- alarm list (CSV),
- segmentation list (CSV),
- metrics table (JSON/CSV).

**Purpose:** Ensures all computational outputs can be inspected, combined, and re-analyzed without re-running.

**Artifact format:** Deterministic, structured data files (JSON, CSV, binary if needed).

### Reporting
**Definition:** Synthesizing exported data into human-readable visualizations and summaries.

**Examples:**
- time-series plots with alarms overlaid,
- posterior probability regime plots,
- detector score vs threshold traces,
- summary tables for comparison,
- aggregate results across detector variants.

**Purpose:** Enables thesis writing, figure generation, and evidence presentation.

**Artifact format:** Plots (PNG/PDF), tables (Markdown, LaTeX), summary documents.

### Logging
**Definition:** Recording runtime event traces during algorithm execution for diagnosis and traceability.

**Examples:**
- EM iteration log,
- feature-build progress,
- online filter step messages,
- error and warning messages.

**Purpose:** Supports debugging and understanding what happened during a run.

**Artifact format:** Timestamped text logs, optional JSON-structured events.

---

## 3. Core Principle: Every Run Must Produce a Self-Describing Artifact Set

The central principle of Phase 21 is:

$$
\boxed{\text{Every experiment run leaves behind a complete, structured, self-describing set of artifacts.}}
$$

This means:

1. **Configs are saved alongside results.**
   - The config that produced a result is always present in the same artifact directory.
   - No ambiguity about which settings were used.

2. **Results are serialized in machine-readable form.**
   - JSON for structured metadata and metrics.
   - CSV for tabular data (alarms, segments, events).
   - Plots for visualization.

3. **Artifact directories follow a deterministic, human-readable schema.**
   - Paths encode mode (synthetic/real), run ID, and artifact type.
   - Scripts can reliably find and process artifacts.

4. **All artifacts are traceable back to a config and run ID.**
   - The run ID is a hash of the config plus a seed-based suffix.
   - This ensures reproducibility and avoids accidental overwrites.

5. **Reporting is decoupled from execution.**
   - The experiment runner produces raw exports.
   - A separate reporting layer consumes exports and generates plots/tables.
   - This separation allows re-plotting without re-running expensive detectors.

---

## 4. Artifact Families

Every run produces artifacts in several families. Each family serves a distinct purpose:

### 4.1 Metadata Artifacts
**Purpose:** Capture the experiment configuration and execution context.

**Contents:**
- Config snapshot (JSON): exact ExperimentConfig used.
- Run metadata (JSON): run ID, start/end time, seed, config hash.
- Git info (JSON, optional): commit hash, branch, working directory state.

**File examples:**
- `config.snapshot.json`
- `metadata.json`
- `git_info.json`

**Why:** Without metadata, it is impossible to recreate what was run or debug anomalies.

### 4.2 Numerical Summary Artifacts
**Purpose:** Provide high-level metrics and summaries of the run.

**Contents:**
- Model summary (JSON): regime count, parameter estimates, fit diagnostics.
- Detector summary (JSON): detector type, threshold, persistence, n_alarms.
- Evaluation summary (JSON): mode-specific metrics (synthetic: coverage, precision; real: Routes A and B metrics).
- Aggregate metrics (JSON): timing, success status, warnings.

**File examples:**
- `model_summary.json`
- `detector_summary.json`
- `evaluation_summary.json`
- `metrics.json`

**Why:** These summaries enable rapid extraction of key results without parsing large raw files.

### 4.3 Event Artifacts
**Purpose:** Export all detected alarms/events with their timestamps and metadata.

**Contents:**
- Alarm list (CSV): time, detector score, alarm/no-alarm, persistence state.
- Matched events (CSV, synthetic only): ground-truth changepoint, matched alarm if any, delay if matched.
- Proxy events (CSV, real Route A): marked real events, aligned alarms, agreement/disagreement.

**File examples:**
- `alarms.csv`
- `matched_events.csv`
- `proxy_events.csv`

**Why:** These enable detailed investigation of detector behavior and support custom metrics.

### 4.4 Segment Artifacts
**Purpose:** Export segmentation results (real Route B evaluation).

**Contents:**
- Segment list (CSV): segment ID, start time, end time, detected, mean shift magnitude, occupancy.
- Segment summaries (JSON): per-segment diagnostics, contrast ratios, event coverage.

**File examples:**
- `segments.csv`
- `segment_summaries.json`

**Why:** Support detailed post-hoc analysis of segmentation quality.

### 4.5 Trace Artifacts
**Purpose:** Export detailed time-series data for visualization.

**Contents:**
- Feature trace (CSV, optional): timestamp, observation value, feature value(s).
- Score trace (CSV, optional): timestamp, detector score, threshold, regime posterior probabilities.
- Regime posterior (CSV, optional): timestamp, P(S_t=1|y_1:t), ..., P(S_t=K|y_1:t).

**File examples:**
- `feature_trace.csv`
- `score_trace.csv`
- `regime_posterior.csv`

**Why:** These support plot generation and advanced forensic analysis.

### 4.6 Plot Artifacts
**Purpose:** Visualize results in human-readable form.

**Contents:**
- Signal + alarms plot (PNG/PDF): time series with true changepoints (synthetic) or real events (real), detector alarms overlaid.
- Detector score plot (PNG/PDF): score trace vs threshold, with alarm markers.
- Posterior regime plot (PNG/PDF): regime posterior probabilities stacked over time.
- Segmentation plot (real only, PNG/PDF): segments with boundaries, color-coded by detected/undetected.
- Delay distribution (synthetic only, PNG/PDF): histogram of detection delays or false alarm times.

**File examples:**
- `signal_with_alarms.png`
- `detector_scores.png`
- `regime_posteriors.png`
- `segmentation.png`
- `delay_distribution.png`

**Why:** These enable figure inclusion in thesis chapters and rapid visual inspection.

### 4.7 Table Artifacts
**Purpose:** Support thesis table generation and comparison.

**Contents:**
- Per-run metrics table (Markdown/LaTeX): key metrics formatted for inclusion in thesis tables.
- Aggregate comparison table (CSV/Markdown): metrics across detector variants, allowing export to thesis.

**File examples:**
- `metrics.md` / `metrics.csv` / `metrics.tex`
- `comparison_table.csv`

**Why:** These reduce manual work in thesis preparation.

---

## 5. Structured Directory Layout

Artifacts for a single run must be organized in a deterministic, human-readable hierarchy.

### 5.1 Root Structure
Assume a reporting root directory (e.g., `runs/` or `/tmp/proteus_runs/`):

```
runs/
├── synthetic/
│   ├── scenario_calibrated/
│   │   ├── run_abc123_seed_7/
│   │   │   ├── config/
│   │   │   ├── metadata/
│   │   │   ├── results/
│   │   │   ├── traces/
│   │   │   ├── plots/
│   │   │   └── tables/
│   │   └── run_abc123_seed_9/
│   │       └── ...
│   └── scenario_extreme/
│       └── run_xyz789_seed_1/
│           └── ...
├── real/
│   ├── spy_daily/
│   │   ├── run_def456_seed_42/
│   │   │   ├── config/
│   │   │   ├── metadata/
│   │   │   ├── results/
│   │   │   ├── traces/
│   │   │   ├── plots/
│   │   │   └── tables/
│   │   └── ...
│   └── qqq_intraday/
│       └── ...
└── aggregate/
    ├── comparison_synthetic_detectors.csv
    ├── comparison_real_detectors.csv
    ├── plot_detector_roc_synthetic.png
    └── plot_delay_comparison.png
```

### 5.2 Per-Run Directory Schema
For a given run, the artifact structure is:

```
run_<config_hash_hex16>_<seed>/
├── config/
│   └── experiment_config.json    — full config snapshot
├── metadata/
│   ├── run_metadata.json          — run ID, label, timestamps, seed, hash
│   └── result.json                — full ExperimentResult
├── model_params.json              — FittedParamsSummary (π, P, means, vars, LL history)
├── summary.json                   — EvaluationSummary only
├── score_trace.csv                — per-step detector score (if save_traces + write_csv)
├── alarms.csv                     — alarm step indices (if save_traces + write_csv)
├── metrics.md                     — per-run metrics table (Markdown)
├── metrics.csv                    — per-run metrics table (CSV)
├── metrics.tex                    — per-run metrics table (LaTeX)
└── plots/
    ├── signal_with_alarms.png     — pending
    ├── detector_scores.png        — pending
    ├── regime_posteriors.png      — pending
    ├── delay_distribution.png     — pending (synthetic only)
    └── segmentation.png           — pending (real only)
```

### 5.3 Design Rationale
This structure is chosen to:

1. **Group by mode and dataset first** so experiments are naturally partitioned.
2. **Isolate runs by unique ID** so concurrent or repeated runs don't collide.
3. **Sub-organize by artifact type** (config, metadata, results, traces, plots, tables) for easy navigation.
4. **Use consistent naming** across all runs for script-based processing.
5. **Support lazy loading** — scripts can find a specific artifact type without traversing everything.

---

## 6. Synthetic Run Exports

A synthetic experiment run must export the following in addition to common artifacts:

### 6.1 Specific Exports for Synthetic Runs

**Matched events:**
- Ground-truth changepoints from the scenario,
- Matched alarms (if any) within matching window,
- Detection delay or miss status,
- Exported to `traces/matched_events.csv`.

**Synthetic benchmark metrics:**
- True positive count,
- False positive count,
- Miss count,
- Detection delay distribution,
- False alarm onset time distribution,
- Exported to `results/evaluation_summary.json`.

**Synthetic signal trace:**
- Feature values and observations across the entire synthetic stream,
- True regime sequence,
- Exported to `traces/feature_trace.csv` with optional regime annotation.

**Scenario metadata:**
- Scenario ID, horizon, regime sequence, means, variances, transition matrix,
- Exported to `metadata/scenario_metadata.json`.

### 6.2 Synthetic Plots

- Time series with true changepoints (vertical lines) and detected alarms (markers).
- Detector score vs time with threshold band.
- Regime posterior probabilities stacked area chart.
- Delay histogram (only if detections occurred).
- Precision/recall scatter if multiple runs available.

---

## 7. Real-Data Run Exports

A real-data experiment run must export the following in addition to common artifacts:

### 7.1 Specific Exports for Real Runs

**Route A (point event evaluation):**
- Proxy events (user-marked regime shifts or events of interest),
- Alarm-event alignment (which alarms match which events, if any),
- Point metrics: event coverage, alarm relevance,
- Exported to `traces/proxy_events.csv`.

**Route B (segmentation evaluation):**
- Detected segments (contiguous alarm clusters),
- Segment properties: start, end, duration, mean internal shift, contrast ratio,
- Per-segment match against ground-truth segments if available,
- Exported to `traces/segments.csv` and `tables/segment_summary.csv`.

**Real-data signal:**
- Real feature trace with alarm markers,
- Exported to `traces/feature_trace.csv`.

**Real-data evaluation metrics:**
- Route A metrics (event coverage, alarm relevance, causal consistency),
- Route B metrics (segmentation recall, precision, coherence),
- Exported to `results/evaluation_summary.json`.

### 7.2 Real-Data Plots

- Feature trace with alarm markers and event indicators.
- Detector score plot with alarms.
- Segmentation plot with boundaries and color-coding.
- Route A alignment plot (events vs alarms in a Gantt-like visualization).
- Regime posterior plot if available.

---

## 8. Aggregate Reporting

Beyond individual runs, Phase 21 must support **aggregate reporting** across multiple experiments.

### 8.1 Aggregate Comparison Exports

When a batch of experiments completes (e.g., comparing detectors across scenarios or assets):

**Aggregate metrics table (CSV):**
```
scenario,detector_type,threshold,n_alarms,coverage,precision,delay_mean,delay_median
synthetic_low,Surprise,2.0,5,0.92,0.80,3.2,2.5
synthetic_low,HardSwitch,0.8,6,0.88,0.75,4.1,3.0
synthetic_high,Surprise,2.0,3,0.85,0.95,2.1,1.8
...
```

**Aggregate comparison plots:**
- Detector variant comparison across scenarios (e.g., ROC-like curves if applicable).
- Threshold sensitivity analysis (coverage vs false alarm rate).
- Detector performance across real assets (bar charts of key metrics).

**Aggregate summary (JSON):**
- Batch metadata: date, experiment set name, total runs, success count.
- Per-scenario or per-asset summary statistics.

### 8.2 Report Builders

A **report builder** takes a collection of run results and produces:
- Aggregate summary statistics (mean, median, std dev of key metrics),
- Comparison tables for thesis inclusion,
- Aggregate plots (e.g., boxplot of detection delays across detectors),
- Index document (Markdown or HTML) linking all runs and summaries.

---

## 9. Plot Generation with `plotters`

Plotting is a systematic, reproducible process in Phase 21.

### 9.1 Why `plotters` for This Project

**`plotters` is chosen because:**

1. **Pure Rust:** No external graphics libraries or system dependencies.
2. **Reproducible:** Same code + same data = identical plots.
3. **Lightweight:** Suitable for batch-generating many plots.
4. **Fine-grained control:** Precise axis labels, legends, colors, and annotations.
5. **Multiple output formats:** PNG, SVG, Bitmap.

### 9.2 Plot Generation Principle

Plots are **not generated during the detector run.**

Instead:
1. The detector run exports structured traces (CSV/JSON).
2. A **separate plotting stage** reads these traces.
3. The plotter consumes the traces and produces PNG/SVG files.

**Benefit:** Plots can be re-generated with different styling, colors, or axis ranges without re-running detectors.

### 9.3 Plot Input Structs

Each plot type has a dedicated input struct that captures exactly what data is needed.

**Example: SignalWithAlarmsPlotInput**

```rust
pub struct SignalWithAlarmsPlotInput {
    pub timestamps: Vec<DateTime<Utc>>,
    pub observations: Vec<f64>,
    pub alarms: Vec<(DateTime<Utc>, bool)>,
    pub true_changepoints: Option<Vec<DateTime<Utc>>>,  // synthetic only
    pub title: String,
}
```

This struct:
- Holds only the data needed to draw the plot.
- Can be serialized from run results.
- Decouples plot rendering from result object shape.

### 9.4 Plot Rendering Functions

Each plot type has a function that consumes the input struct and writes to disk.

**Example: `fn render_signal_with_alarms(...) -> Result<PathBuf>`**

This function:
- Takes a `SignalWithAlarmsPlotInput`.
- Constructs a `plotters` figure.
- Renders axes, data, annotations.
- Writes PNG/SVG to disk.
- Returns the path to the output file.

### 9.5 Supported Plot Types

Based on Phase 21 requirements:

1. **Signal with alarms:** Time series, true changepoints, detected alarms.
2. **Detector scores:** Score trace, threshold band, alarm markers.
3. **Regime posteriors:** Stacked area chart of regime probabilities.
4. **Segmentation:** Horizontal bars for segments, color by detected/missed.
5. **Delay distribution:** Histogram of detection delays.
6. **Proxy events:** Event alignment plot (alarms vs marked events).
7. **Route A alignment:** Gantt-like view of events and detections.
8. **Aggregate comparison:** Bar charts or boxplots across detectors.

---

## 10. Table Generation for Thesis Writing

Tables are structured outputs that can be exported to thesis-ready formats.

### 10.1 Table Types

**Per-run metrics table:**
- One row per run, columns for key metrics.
- Exportable to Markdown for direct inclusion in thesis.
- Optional LaTeX format for publication-ready tables.

**Aggregate comparison table:**
- Rows: detector variant, asset/scenario, or configuration.
- Columns: key metrics (coverage, precision, delay, etc.).
- CSV for data analysis, Markdown for thesis.

**Segment summary table (real-data only):**
- Rows: detected segments.
- Columns: start time, duration, detection status, metrics.

### 10.2 Table Builder Pattern

A **table builder** takes run results and produces structured output:

```rust
pub struct MetricsTableBuilder { ... }

impl MetricsTableBuilder {
    pub fn add_run(&mut self, run_id: &str, metrics: &EvaluationSummary) { ... }
    pub fn to_markdown(&self) -> String { ... }
    pub fn to_csv(&self) -> String { ... }
    pub fn to_latex(&self) -> String { ... }
}
```

This pattern:
- Accepts results incrementally.
- Formats output on demand.
- Supports multiple export formats.

---

## 11. JSON and CSV Export Utility

Two serialization formats are used for complementary purposes:

### 11.1 JSON Exports

**Use JSON for:**
- Hierarchical, nested data (model parameters, nested summaries),
- Metadata with optional fields,
- Results that need to be deserialized back into structs.

**Examples:**
- `config.snapshot.json`: full ExperimentConfig struct,
- `model_summary.json`: fitted parameters with diagnostics,
- `evaluation_summary.json`: nested real-data metrics.

**Advantage:** Preserves structure, enables programmatic access via JSON libraries.

### 11.2 CSV Exports

**Use CSV for:**
- Flat, tabular data (time series, event lists),
- Data meant for spreadsheet import,
- Data for external analysis tools.

**Examples:**
- `alarms.csv`: one row per alarm,
- `matched_events.csv`: one row per ground-truth event,
- `segments.csv`: one row per segment.

**Advantage:** Universal spreadsheet compatibility, easy filtering/sorting.

### 11.3 Utility Design

Serialization utilities are factored into:

```rust
pub mod export {
    pub mod json {
        pub fn serialize_config(...) -> anyhow::Result<String> { ... }
        pub fn serialize_result(...) -> anyhow::Result<String> { ... }
    }
    pub mod csv {
        pub fn export_alarms(...) -> anyhow::Result<String> { ... }
        pub fn export_segments(...) -> anyhow::Result<String> { ... }
    }
}
```

Each function:
- Takes input data.
- Returns serialized string or writes to file.
- Handles errors gracefully.

---

## 12. Decoupling Reporting from Execution

A core principle of Phase 21 is that reporting is **separate from the detector pipeline**.

### 12.1 Execution Pipeline vs Reporting Pipeline

**Execution (Phases 16–20):**
- Load data,
- Engineer features,
- Train model,
- Run online detector,
- Evaluate,
- Export raw results.

**Reporting (Phase 21):**
- Read exported results,
- Compute derived metrics if needed,
- Generate plots,
- Generate tables,
- Aggregate across runs.

### 12.2 Benefits of Separation

1. **Speed:** Re-plotting doesn't require re-running expensive detectors.
2. **Flexibility:** Plots can be re-styled without touching detector code.
3. **Scalability:** Reports can be generated on a different machine.
4. **Debuggability:** If a plot is wrong, the issue is in reporting, not detection.
5. **Reproducibility:** Plotting code can be version-controlled separately.

### 12.3 Interface Design

The boundary between execution and reporting is the **export schema**:

- Execution writes structured files to disk.
- Reporting reads from those files.
- A schema document (this Phase 21 document) defines the contract.

```rust
// Execution produces these exports:
pub fn export_run_results(
    run_dir: &Path,
    result: &ExperimentResult,
) -> anyhow::Result<()> {
    // Write config, metadata, traces, summaries
}

// Reporting consumes these exports:
pub fn load_run_artifacts(run_dir: &Path) -> anyhow::Result<RunArtifacts> {
    // Read config, traces, summaries
}
```

---

## 13. Artifact Audit Checklist

For a run to be considered fully reported, the following must be present:

### Metadata Layer
- [ ] `config/experiment_config.json`
- [ ] `metadata/run_metadata.json`
- [ ] `metadata/git_info.json` (if enabled)

### Numerical Results
- [ ] `results/evaluation_summary.json`
- [ ] `results/metrics.json`
- [ ] `results/model_summary.json`
- [ ] `results/detector_summary.json`

### Traces
- [ ] `traces/alarms.csv`
- [ ] `traces/feature_trace.csv` (if save_traces enabled)
- [ ] `traces/score_trace.csv` (if save_traces enabled)
- [ ] `traces/regime_posterior.csv` (if save_traces enabled)

### Synthetic-Specific
- [ ] `traces/matched_events.csv`
- [ ] `metadata/scenario_metadata.json`
- [ ] `plots/signal_with_alarms.png`
- [ ] `plots/delay_distribution.png`

### Real-Specific
- [ ] `traces/proxy_events.csv` (Route A)
- [ ] `traces/segments.csv` (Route B)
- [ ] `tables/segment_summary.csv` (Route B)
- [ ] `plots/segmentation.png`

### Tables
- [ ] `tables/metrics_table.md`
- [ ] `tables/metrics_table.tex` (optional)

---

## 14. Summary: The Reporting Pipeline as Part of Research Methodology

Phase 21 formalizes reporting as a systematic, reproducible stage of the research pipeline.

The reporting pipeline transforms raw detector outputs into publication-ready artifacts:

1. **Exports:** Deterministic, structured serialization of all run data.
2. **Artifacts:** Organized, traceable directory hierarchy.
3. **Plotting:** Systematic, reproducible visualization with `plotters`.
4. **Tables:** Thesis-ready summary tables in multiple formats.
5. **Aggregation:** Batch processing of multiple runs for comparison.

Every artifact is:
- **Traceable:** Back to a specific config and run ID.
- **Reproducible:** Generated the same way every time.
- **Structured:** Organized in a predictable hierarchy.
- **Self-describing:** Includes metadata, summaries, and documentation.

This enables thesis-scale empirical studies where:
- Figures can be regenerated on demand.
- Metrics can be re-extracted or re-analyzed.
- Experiments can be compared systematically.
- Conclusions are grounded in persisted, auditable results.
