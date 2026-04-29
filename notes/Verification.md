# Manual verification plan
## 0. Freeze the verification target

Before testing anything, create a verification folder for this pass.

**Do**

* Pick one verification label, for example `verify_2026_04_28`.
* Create a clean output root for this verification run.
* Record:

  * git commit hash
  * Rust toolchain version
  * OS
  * timestamp

**Expected result**

* One folder for this verification session.
* One text file like `verification_metadata.txt`.

**Checklist**

* [ ] Verification folder created
* [ ] Commit hash saved
* [ ] Toolchain/version saved
* [ ] Timestamp saved

---

## 1. Build the project from scratch

This verifies that the project is not relying on stale binaries or local hacks.

**Do**

* Run:

  ```bash
  cargo clean
  cargo build
  ```
* Then run:

  ```bash
  cargo test
  ```

**Expected result**

* Build succeeds.
* Tests complete successfully or with only explicitly known/accepted skipped tests.
* Save full terminal logs.

**Tangible artifacts**

* `build.log`
* `test.log`

**Checklist**

* [ ] `cargo build` succeeds
* [ ] `cargo test` succeeds
* [ ] Logs saved

---

## 2. Verify CLI entry behavior

This verifies the primary UX.

**Do**

* Run:

  ```bash
  cargo run
  ```
* Confirm the interactive CLI opens.
* Also test one direct mode command, for example:

  ```bash
  cargo run -- --help
  ```

  or the equivalent command style your app uses.

**Expected result**

* Interactive CLI opens with visible top-level options.
* Direct CLI mode still works.
* Save screenshots or terminal captures.

**Tangible artifacts**

* `cli_interactive_start.txt` or screenshot
* `cli_help.txt`

**Checklist**

* [ ] Interactive mode opens
* [ ] Main menu is readable
* [ ] Direct CLI mode works
* [ ] Help text is meaningful

---

## 3. Verify data discovery and inspection

This verifies that the system can see the datasets you expect to use.

**Do**

* From the CLI, inspect available datasets.
* Confirm that at least the intended real datasets are visible:

  * commodities daily
  * SPY daily
  * QQQ daily
  * SPY intraday
  * QQQ intraday
* For one daily and one intraday dataset, inspect metadata.

**Expected result**

* Dataset list is visible from the CLI.
* Metadata includes at least:

  * asset name
  * frequency
  * date/time range
  * row count
  * source path or identifier

**Tangible artifacts**

* `dataset_list.txt`
* `spy_daily_metadata.json` or equivalent
* `spy_intraday_metadata.json` or equivalent

**Checklist**

* [ ] All intended datasets are discoverable
* [ ] Daily metadata looks correct
* [ ] Intraday metadata looks correct
* [ ] Date ranges match expectations

---

## 4. Verify preprocessing and splits

This checks that the real-data pipeline is operational and leakage-safe.

**Do**

* Run preprocessing for one daily dataset and one intraday dataset.
* Inspect generated train/validation/test splits.
* Confirm session handling exists for intraday data.

**Expected result**

* Preprocessed datasets are saved.
* Split summaries are saved.
* Intraday data clearly indicate session-aware handling.

**Tangible artifacts**

* `prepared_daily/`
* `prepared_intraday/`
* `split_summary_daily.json`
* `split_summary_intraday.json`

**Human checks**

* Train period occurs strictly before validation.
* Validation occurs strictly before test.
* No overlap.
* Intraday sessions do not look merged incorrectly across overnight gaps.

**Checklist**

* [ ] Daily preprocessing succeeds
* [ ] Intraday preprocessing succeeds
* [ ] Splits are chronological
* [ ] No leakage visible
* [ ] Session handling is explicit

---

## 5. Verify feature generation

This checks that the observation design layer is real, not just configured.

**Do**

* Build at least two feature streams for one dataset:

  * returns
  * rolling volatility or absolute returns
* Inspect summaries and sample rows.

**Expected result**

* Feature artifacts are saved.
* Metadata records feature family and any window/scaling parameters.
* Warmup trimming is visible.

**Tangible artifacts**

* `features_returns/`
* `features_volatility/`
* `feature_summary_returns.json`
* `feature_summary_volatility.json`
* optional CSV preview files

**Human checks**

* Returns are not raw prices.
* Rolling features start after an expected warmup region.
* If scaling exists, it is identified clearly.

**Checklist**

* [ ] Returns feature built successfully
* [ ] Volatility-like feature built successfully
* [ ] Warmup behavior visible
* [ ] Feature metadata saved
* [ ] Feature output looks numerically plausible

---

## 6. Verify synthetic-to-real calibration

This checks that calibrated synthetic experiments are grounded in real data.

**Do**

* Run one calibration using a real training dataset and one feature family.
* Save the resulting synthetic scenario config.
* Inspect the calibration summary.

**Expected result**

* Empirical summary statistics are extracted.
* Calibrated synthetic parameters are generated.
* A comparison summary is saved.

**Tangible artifacts**

* `calibration_summary.json`
* `calibrated_scenario.toml` or equivalent
* `synthetic_vs_empirical_summary.json`

**Human checks**

* Calibrated scenario references the same feature family you chose.
* Synthetic regime variances/persistence look plausible relative to empirical summaries.

**Checklist**

* [ ] Calibration runs successfully
* [ ] Scenario artifact is saved
* [ ] Empirical summaries are saved
* [ ] Calibration output is interpretable

---

## 7. Verify model fitting on one training set

This is one of the most important checks.

**Do**

* Fit the model on one prepared training set using one chosen feature family.
* Save the fitted model artifact.
* Inspect the fit summary.

**Expected result**

* Model fitting succeeds.
* The following are visible:

  * initial distribution
  * transition matrix
  * regime means
  * regime variances
  * EM iteration count
  * log-likelihood history
  * convergence reason

**Tangible artifacts**

* `fitted_model.bin` / `fitted_model.json` / equivalent
* `fit_summary.json`
* `loglikelihood_history.csv`
* optional fit plots

**Human checks**

* Transition matrix rows sum to about 1.
* Variances are positive.
* Log-likelihood generally improves through EM.
* Number of regimes matches the requested config.

**Checklist**

* [ ] Fit completes successfully
* [ ] Fitted model is saved
* [ ] Parameter summary is saved
* [ ] EM history is saved
* [ ] Parameters are numerically valid

---

## 8. Verify model reload

This checks reproducibility and artifact usability.

**Do**

* In a fresh command, load the fitted model artifact rather than refitting.
* Inspect the model summary from the loaded artifact.

**Expected result**

* Model can be reused without retraining.
* Loaded parameters match the saved fit summary.

**Tangible artifacts**

* `loaded_model_summary.json`
* terminal log showing reuse path

**Checklist**

* [ ] Reload succeeds
* [ ] Reloaded parameters match saved ones
* [ ] No hidden retraining occurred

---

## 9. Verify one synthetic end-to-end run

This checks the full synthetic pipeline.

**Do**

* Run one complete synthetic experiment:

  * scenario generation
  * feature/observation build if applicable
  * fit or load model
  * online detection
  * synthetic evaluation
  * report generation

**Expected result**

* One complete synthetic run directory is created.
* It includes:

  * config snapshot
  * truth changepoints
  * alarms
  * metrics
  * plots
  * tables

**Tangible artifacts**

* `runs/synthetic/<run_id>/...`

**Human checks**

* Truth changepoints exist.
* Alarm timestamps exist.
* Delay / precision / recall / miss rate / false alarm outputs exist.
* Plots render correctly.

**Checklist**

* [ ] Full synthetic run succeeds
* [ ] Truth events saved
* [ ] Alarm events saved
* [ ] Synthetic metrics saved
* [ ] Synthetic plots saved
* [ ] Synthetic tables saved

---

## 10. Verify one real daily end-to-end run

This checks the daily real-data path.

**Do**

* Run one complete real-data experiment on a daily dataset:

  * data prep or resolved prepared artifact
  * feature build
  * fit/load model
  * detector run
  * Route A evaluation
  * Route B evaluation
  * report generation

**Expected result**

* One complete real daily run directory is created.

**Tangible artifacts**

* `runs/real/<run_id>/...`

**Human checks**

* Route A output contains:

  * proxy events used
  * aligned alarms
  * event coverage
  * alarm relevance
* Route B output contains:

  * segment boundaries
  * segment statistics
  * adjacent contrast summaries
* Daily plots are legible.

**Checklist**

* [ ] Full real daily run succeeds
* [ ] Route A output exists
* [ ] Route B output exists
* [ ] Daily plots saved
* [ ] Daily tables saved

---

## 11. Verify one real intraday end-to-end run

This checks the hardest practical path.

**Do**

* Run one complete real-data experiment on intraday SPY or QQQ.

**Expected result**

* One complete real intraday run directory is created.

**Tangible artifacts**

* `runs/real_intraday/<run_id>/...`

**Human checks**

* Session-aware handling is visible in metadata or logs.
* Feature generation respects intraday assumptions.
* Route A and Route B outputs both exist.
* Intraday time axis and alarms look sensible in plots.

**Checklist**

* [ ] Full real intraday run succeeds
* [ ] Session-aware preprocessing is visible
* [ ] Route A output exists
* [ ] Route B output exists
* [ ] Intraday plots saved
* [ ] Intraday tables saved

---

## 12. Verify detector visibility

This step checks that you can really “see everything happening.”

**Do**

* Inspect one run and confirm you can locate:

  * detector config
  * alarm list
  * score trace
  * filtered regime probabilities
  * optional posterior trace

**Expected result**

* All of these are saved and inspectable.

**Tangible artifacts**

* `detector_config.json`
* `alarms.csv`
* `score_trace.csv`
* `filtered_probs.csv` or equivalent

**Checklist**

* [ ] Detector config saved
* [ ] Alarm list saved
* [ ] Score trace saved
* [ ] Regime probability trace saved or intentionally omitted with explanation

---

## 13. Verify reporting layer

This checks thesis usability.

**Do**

* For one completed run, regenerate the report artifacts without rerunning the full experiment.
* Confirm plots are produced using `plotters`.
* Confirm tables are saved in reusable formats.

**Expected result**

* Reporting can run from saved results.
* Plot files exist and open correctly.
* Tables are accessible as CSV or other chosen export formats.

**Tangible artifacts**

* regenerated `plots/`
* regenerated `tables/`
* `report_summary.json`

**Checklist**

* [ ] Reports can be rebuilt from saved artifacts
* [ ] Plot files open correctly
* [ ] Tables are readable and reusable
* [ ] Reporting is decoupled from experiment execution

---

## 14. Verify run inspection UX

This checks long-term usability.

**Do**

* Use the CLI to list previous runs.
* Inspect one run from the CLI.
* Confirm you can identify:

  * what config was used
  * what dataset was used
  * what model/detector was used
  * where plots/tables are stored

**Expected result**

* Run discovery and run inspection work without manual filesystem hunting.

**Tangible artifacts**

* terminal output screenshots or logs for:

  * list runs
  * inspect run

**Checklist**

* [ ] Run listing works
* [ ] Run inspection works
* [ ] Artifact paths are discoverable
* [ ] Metadata is enough to understand the run

---

## 15. Verify batch execution

This checks whether you can run a study rather than a single test.

**Do**

* Run one small batch over a few configurations, for example:

  * two feature families
  * two detector variants
  * one dataset

**Expected result**

* Multiple run folders are created.
* Aggregate summaries are created.

**Tangible artifacts**

* `batch_summary.json`
* aggregate CSV tables
* optional aggregate plots

**Checklist**

* [ ] Batch run succeeds
* [ ] Multiple runs saved separately
* [ ] Aggregate summary exists
* [ ] Aggregate results are interpretable

---

## 16. Verify reproducibility

This checks whether the work is defensible.

**Do**

* Pick one finished run.
* Confirm the run folder contains:

  * config snapshot
  * metadata snapshot
  * seed if applicable
  * run ID
  * result summary
* Re-run the same config once and compare the core outputs.

**Expected result**

* Configuration and metadata are sufficient to reconstruct the run.
* Synthetic runs are reproducible given the same seed.
* Real runs are reproducible if inputs are unchanged.

**Tangible artifacts**

* side-by-side config snapshots
* summary comparison note

**Checklist**

* [ ] Config snapshot exists
* [ ] Metadata snapshot exists
* [ ] Seed saved where relevant
* [ ] Run is reconstructible
* [ ] Re-run results are consistent enough

---

## 17. Verify docs against reality

This is critical for thesis readiness.

**Do**

* Open your docs and check whether each major system component is actually described:

  * data pipeline
  * feature design
  * calibration
  * model fitting
  * detector family
  * synthetic evaluation
  * real-data evaluation
  * experiment runner
  * reporting/export
  * CLI usage

**Expected result**

* Docs match what the system actually does.

**Tangible artifacts**

* one short `docs_audit.md` file listing mismatches

**Checklist**

* [ ] Docs exist for all major parts
* [ ] Docs match actual behavior
* [ ] No major undocumented workflow remains
* [ ] CLI usage is documented

---

## 18. Final human sign-off checklist

You can call the system practically complete when all of the following are true:

* [ ] I can train the model from the CLI
* [ ] I can inspect fitted parameters after training
* [ ] I can run detection from the CLI
* [ ] I can inspect alarms, scores, and regime probabilities
* [ ] I can evaluate synthetic runs end to end
* [ ] I can evaluate real runs end to end
* [ ] I can generate plots and tables automatically
* [ ] I can inspect previous runs and artifacts later
* [ ] I can run a small batch experiment
* [ ] I can reproduce a run from saved config/artifacts
* [ ] I have enough saved outputs to start writing the thesis without rerunning everything

---

# Suggested execution order

Run the verification in this order:

1. build + test
2. CLI entry behavior
3. data inspection
4. preprocessing + splits
5. feature generation
6. calibration
7. model fitting
8. model reload
9. synthetic full run
10. real daily full run
11. real intraday full run
12. detector visibility
13. reporting regeneration
14. run inspection
15. batch run
16. reproducibility check
17. docs audit
18. final sign-off
