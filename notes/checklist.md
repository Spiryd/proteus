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
- [x] I can also run the same workflow through direct CLI/config mode for reproducibility
- [x] I do not need to edit code to run normal experiments

---

# 2. Data is explicit and inspectable

- [x] I can choose the dataset from the CLI
- [x] I can see what asset is used
- [x] I can see what frequency is used (daily / intraday)
- [x] I can see the exact date/time range used
- [x] I can see how the data was split into train / validation / test
- [x] For intraday data, session handling is explicit
- [x] The cleaned dataset metadata is saved in the run artifacts

---

# 3. Observation / feature pipeline is explicit

- [x] I know exactly what the model observes in a run
- [x] The observation family is saved, e.g.:
  - [x] log returns
  - [x] absolute returns
  - [x] rolling volatility
- [x] Warmup trimming is explicit
- [x] Any scaling/normalization is documented and saved
- [x] If scaling is used, it is fit only on training data
- [x] The final model-ready observation stream can be inspected

---

# 4. Training is fully visible

- [x] I can launch model fitting from the CLI
- [x] The training configuration is saved
- [x] I can inspect fitted parameters after training:
  - [x] initial distribution \(\pi\)
  - [x] transition matrix \(P\)
  - [x] regime means
  - [x] regime variances
- [x] I can inspect EM progress:
  - [x] log-likelihood history
  - [x] iteration count
  - [x] convergence reason
- [x] The fitted model artifact is saved
- [x] I can reload the fitted model later without retraining

---

# 5. Detection is fully visible

- [x] I can run the detector on synthetic, validation, test, or real data from the CLI
- [x] I can choose detector type from the CLI
- [x] I can inspect detector configuration:
  - [x] threshold
  - [x] persistence
  - [x] cooldown
- [x] I can inspect detector outputs:
  - [x] alarm timestamps
  - [x] detector score trace
  - [x] filtered regime probabilities
  - [x] optional posterior trace export

---

# 6. Synthetic evaluation works end to end

- [x] I can run a full synthetic experiment from the CLI
- [x] The system saves true changepoints
- [x] The system saves alarm timestamps
- [x] The system computes and saves:
  - [x] delay
  - [x] precision
  - [x] recall
  - [x] miss rate
  - [x] false alarm rate
- [x] I can inspect matched vs unmatched events
- [x] Synthetic plots are generated automatically

---

# 7. Real-data evaluation works end to end

- [x] I can run the detector on real data from the CLI
- [x] Route A works:
  - [x] proxy events loaded
  - [x] alarm-event alignment computed
  - [x] event coverage summary saved
  - [x] alarm relevance summary saved
  - [x] aligned delays saved
- [x] Route B works:
  - [x] alarms converted into segments
  - [x] segment statistics computed
  - [x] adjacent-segment contrasts computed
  - [x] segmentation summary saved
- [x] Real-data plots are generated automatically

---

# 8. I can see everything that happened

- [x] Every run has a run ID
- [x] Every run saves a config snapshot
- [x] Every run saves a metadata summary
- [x] Every run saves a result summary
- [x] Every run saves warnings or failure info if something goes wrong
- [x] I can inspect previous runs from the CLI
- [x] I can understand what happened in a run by opening its artifact folder

---

# 9. Reporting is thesis-ready

- [x] Every run produces a structured artifact bundle
- [x] The artifact bundle includes:
  - [x] config
  - [x] metadata
  - [x] fitted model summary
  - [x] detector summary
  - [x] evaluation summary
  - [x] alarms/events
  - [x] plots
  - [x] tables
- [x] Plots are generated with `plotters`
- [x] Plots use stable names and clear titles
- [x] Tables are saved in a reusable format
- [x] I can regenerate reports from saved results without rerunning the whole experiment

---

# 10. Batch experiment support

- [x] I can run multiple experiments in one batch
- [x] I can vary:
  - [x] asset
  - [x] frequency
  - [x] feature family
  - [x] detector type
  - [x] detector parameters
- [x] I get aggregate summaries across runs
- [x] I can compare runs from saved artifacts

---

# 11. Final CLI expectation

The final product should let me do these things without editing code:

- [x] prepare/inspect data
- [x] build/inspect features
- [x] calibrate synthetic scenarios
- [x] fit/inspect model
- [x] run detector
- [x] evaluate synthetic runs
- [x] evaluate real runs
- [x] run a full experiment
- [x] run a batch
- [x] build reports / plots / tables
- [x] inspect previous runs

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
