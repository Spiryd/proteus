# E2E Thesis Product Checklist

## Goal

This checklist is focused only on what I want from the final product:

- full end-to-end flow,
- model training,
- test results,
- real-data runs,
- visibility into parameters and outputs at every important step,
- structured artifacts ready to support thesis writing.

This is the **practical completion checklist**, not the full architecture checklist.

---

# 1. One complete runnable flow

- [x] I can start the whole system from `cargo run`
- [x] I can use the interactive CLI to execute the full workflow
- [x] I can also run the same workflow through direct CLI/config mode for reproducibility (`cargo run -- e2e`)
- [x] I do not need to edit code to run normal experiments

---

# 2. Data is explicit and inspectable

- [x] I can choose the dataset from the CLI (synthetic scenario ID)
- [x] I can see what asset is used (scenario label shown per run)
- [~] I can see what frequency is used — synthetic only (real data deferred)
- [~] I can see the exact date/time range used — N/A for synthetic; deferred for real
- [x] I can see how the data was split into train / validation / test (train_n = first 70%)
- [~] For intraday data, session handling is explicit — deferred (real data)
- [x] The cleaned dataset metadata is saved in the run artifacts (`config.snapshot.json`)

---

# 3. Observation / feature pipeline is explicit

- [x] I know exactly what the model observes in a run (feature_label shown in stage log)
- [x] The observation family is saved (LogReturn / ZScore; persisted in config snapshot)
- [x] Warmup trimming is explicit (train_n boundary is logged and saved)
- [x] Any scaling/normalization is documented and saved (ZScore policy in config)
- [x] If scaling is used, it is fit only on training data (ZScore fit on train_n slice)
- [~] The final model-ready observation stream can be inspected — score_trace.csv and alarms.csv are exported; raw obs CSV export pending

---

# 4. Training is fully visible

- [x] I can launch model fitting from the CLI (`cargo run -- e2e`)
- [x] The training configuration is saved (`config.snapshot.json`)
- [x] I can inspect fitted parameters after training:
  - [x] initial distribution π (shown in CLI + saved in `model_params.json`)
  - [x] transition matrix P
  - [x] regime means
  - [x] regime variances
- [x] I can inspect EM progress:
  - [x] log-likelihood history (saved in `model_params.json`)
  - [x] iteration count (shown in CLI: `iter=124`)
  - [x] convergence reason (`converged=true` shown in CLI)
- [x] The fitted model artifact is saved (`model_params.json` per run)
- [x] I can reload the fitted model later without retraining (`TrainingMode::LoadFrozen`)

---

# 5. Detection is fully visible

- [x] I can run the detector on synthetic data from the CLI
- [x] I can choose detector type from the CLI (3 registered variants: HardSwitch, PosteriorTransition, Surprise)
- [x] I can inspect detector configuration:
  - [x] threshold (shown in aggregate table)
  - [x] persistence (in config / registry)
  - [x] cooldown (in config / registry)
- [x] I can inspect detector outputs:
  - [x] alarm timestamps (`alarms.csv` written when save_traces=true)
  - [x] detector score trace (`score_trace.csv` written when save_traces=true)
  - [x] filtered regime probabilities (collected in `regime_posteriors`; CSV export pending)
  - [~] optional posterior trace export — pending raw CSV write

---

# 6. Synthetic evaluation works end to end

- [x] I can run a full synthetic experiment from the CLI (`cargo run -- e2e`)
- [x] The system saves true changepoints (regenerated deterministically from same seed)
- [x] The system saves alarm timestamps (`alarms.csv`)
- [x] The system computes and saves:
  - [x] delay (mean + median, shown in CLI and `summary.json`)
  - [x] precision
  - [x] recall
  - [x] miss rate
  - [x] false alarm rate
- [~] I can inspect matched vs unmatched events — aggregate counts shown; per-event CSV pending
- [ ] Synthetic plots are generated automatically — `generate_plots()` pending

---

# 7. Real-data evaluation works end to end

> **Deferred — Phase 19 / 22**

- [ ] I can run the detector on real data from the CLI
- [ ] Route A works: proxy event alignment
- [ ] Route B works: segmentation self-consistency
- [ ] Real-data plots are generated automatically

---

# 8. I can see everything that happened

- [x] Every run has a run ID (deterministic hash + seed)
- [x] Every run saves a config snapshot (`config.snapshot.json`)
- [x] Every run saves a metadata summary (`run_metadata.json`)
- [x] Every run saves a result summary (`result.json`, `summary.json`)
- [x] Every run saves warnings or failure info if something goes wrong (`result.warnings`)
- [x] I can inspect previous runs from the CLI (`cargo run -- inspect`)
- [x] I can understand what happened in a run by opening its artifact folder

---

# 9. Reporting is thesis-ready

- [x] Every run produces a structured artifact bundle
- [x] The artifact bundle includes:
  - [x] config (`config.snapshot.json`)
  - [x] metadata (`run_metadata.json`)
  - [x] fitted model summary (`model_params.json` — FittedParamsSummary with pi, P, means, variances, LL history)
  - [x] detector summary (in `result.json`)
  - [x] evaluation summary (`summary.json`)
  - [x] alarms/events (`alarms.csv`)
  - [~] plots — `generate_plots()` pending
  - [x] tables (`metrics.md`, `metrics.csv`, `metrics.tex`)
- [~] Plots are generated with `plotters` — pending
- [x] Tables are saved in a reusable format (Markdown / CSV / LaTeX)
- [~] I can regenerate reports from saved results without rerunning — `generate_tables()` works; `generate_plots()` pending

---

# 10. Batch experiment support

- [x] I can run multiple experiments in one batch (`cargo run -- e2e` runs 3)
- [x] I can vary: feature family, detector type, detector parameters (registry + param-search)
- [x] I get aggregate summaries across runs (aggregate table printed after all runs)
- [~] I can compare runs from saved artifacts — `inspect` command works; aggregate reload pending

---

# 11. Final CLI expectation

The final product should let me do these things without editing code:

- [x] prepare/inspect data (synthetic; real deferred)
- [x] build/inspect features
- [~] calibrate synthetic scenarios (calibration module exists; CLI wiring partial)
- [x] fit/inspect model
- [x] run detector
- [x] evaluate synthetic runs
- [ ] evaluate real runs (deferred)
- [x] run a full experiment
- [x] run a batch
- [x] build reports / tables
- [~] build plots (pending)
- [x] inspect previous runs

---

# 12. Final "done" definition

I should call the system complete when I can reliably do the following:

1. [x] choose data,
2. [x] build observations,
3. [x] train the model,
4. [x] inspect the fitted parameters,
5. [x] run online detection,
6. [x] inspect alarms and score traces,
7. [x] evaluate synthetic or real results,
8. [~] save plots and tables automatically (tables ✓, plots pending),
9. [x] inspect everything later from saved artifacts,
10. [x] use the outputs directly while writing the thesis.

**Current status:** Functionally complete for the synthetic thesis stage. Real-data evaluation (Phase 19/22) and plot generation are the remaining pending items.

# 11. Final CLI expectation

The final product should let me do these things without editing code:

- [ ] prepare/inspect data
- [ ] build/inspect features
- [ ] calibrate synthetic scenarios
- [ ] fit/inspect model
- [ ] run detector
- [ ] evaluate synthetic runs
- [ ] evaluate real runs
- [ ] run a full experiment
- [ ] run a batch
- [ ] build reports / plots / tables
- [ ] inspect previous runs

---

# 12. Final “done” definition

I should call the system complete when I can reliably do the following:

1. choose data,
2. build observations,
3. train the model,
4. inspect the fitted parameters,
5. run online detection,
6. inspect alarms and score traces,
7. evaluate synthetic or real results,
8. save plots and tables automatically,
9. inspect everything later from saved artifacts,
10. use the outputs directly while writing the thesis.

If all of that works from the CLI without manual code edits, the system is functionally complete for the thesis stage I am currently targeting.