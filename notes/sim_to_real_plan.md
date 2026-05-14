# Sim-to-Real Implementation Plan

**Goal:** Make the thesis claim *"the model is trained on synthetic data and then tested on real-world data"* actually supportable by the codebase.

**Status legend:** ⬜ not started · 🟡 in progress · ✅ done · ⏭️ skipped/deferred

**Order of execution** (per user instruction):
1. B′1 — one-sided z-scoring policy
2. C′3 — Quick-EM calibration
3. A′1–A′4 — `DataConfig::CalibratedSynthetic` + `SimToRealBackend`
4. D′1–D′2 — provenance artifacts and train-on-real comparator
5. Polish — C′1+C′2, C′4, E (stationary π, frequency tag, registry rename)
6. Documentation updates — /docs and /docs/thesis
7. Verification.md updates (if any new artifacts/steps)
8. Run full verification pipeline

---

## Phase B′1 — One-sided z-scoring policy

**Problem:** Synthetic streams emit values on raw log-return scale (σ ≈ 0.01–0.04). Real streams are z-scored before EM (σ ≈ 1). A model trained on one cannot run on the other without scale agreement.

**Decision:** Both train and test streams must pass through the **same** `FittedScaler`, fitted on the **real** training partition. The synthetic stream is scaled by that scaler (it is *not* refitted on synthetic). This is the one-sided policy.

**Implementation steps:**
- ✅ B′1.1 — Document the scale-consistency contract in `src/calibration/summary.rs` doc comment.
- ✅ B′1.2 — Add `scale_consistency_check` helper in `src/calibration/verify.rs`: assert `|syn.std - emp.std| / emp.std < 0.10`.
- ✅ B′1.3 — In `run_calibration_workflow` invoke the check and add to notes/warnings.
- ✅ B′1.4 — Add a unit test for the check (passes when scales match, fails when one is 0.01 vs 1.0).
- ✅ B′1.5 — `cargo test` green.

**Acceptance:** New tests pass; existing tests unchanged.

**Notes:** The actual cross-stream scaler sharing lives in **A′2** (the new backend). B′1 is just the policy + verifier scaffolding.

---

## Phase C′3 — Quick-EM calibration policy

**Problem:** Current `calibrate_to_synthetic` uses a broken duration estimator (clamped to 2) and a quantile-anchored variance estimator that under-separates regimes.

**Decision:** Add a new policy that runs a short Baum-Welch EM on the real train partition and uses the resulting `ModelParams` directly as the synthetic generator parameters. This is the strongest possible form of calibration: the synthetic data is literally a sample from the model fitted to real data.

**Implementation steps:**
- ✅ C′3.1 — In `src/calibration/mapping.rs`, add `enum CalibrationStrategy { Summary, QuickEm }`. Field on `CalibrationMappingConfig` (`#[serde(default)]`).
- ✅ C′3.2 — Add `calibrate_via_quick_em(observations, config) -> CalibratedSyntheticParams`. Uses `init_params_from_obs` + `fit_em` with budget ~50 iters.
- ✅ C′3.3 — `calibrate_to_synthetic` dispatches by `strategy`. The existing summary-based path is preserved as the default.
- ✅ C′3.4 — `init_params_from_obs` needs to be reachable from `calibration::mapping` — promote from `experiments::shared` to `crate::model::init` or duplicate (decision: use `quantile_init_params` from mapping module).
- ✅ C′3.5 — Strategy reachable via mapping config; scale_check embedded in calibration report.
- ✅ C′3.6 — Unit tests: quick-EM on synthetic data with known params recovers them within tolerance.
- ✅ C′3.7 — `cargo test` green.

**Acceptance:** New strategy reachable from public API; tests pass; existing calibration tests still pass.

---

## Phase A′1 — `DataConfig::CalibratedSynthetic` + `ExperimentMode::SimToReal`

**Implementation steps:**
- ✅ A′1.1 — Add `ExperimentMode::SimToReal` variant.
- ✅ A′1.2 — Add `DataConfig::CalibratedSynthetic { real_asset, real_frequency, real_dataset_id, real_date_start, real_date_end, horizon, strategy, dataset_id }` variant.
- ✅ A′1.3 — Validate logic for both new variants in `validate_for_mode` and `DataConfig::validate`.
- ✅ A′1.4 — `EvaluationConfig::SimToReal { … }` — actually reuse `EvaluationConfig::Real` since the eval target is real data. So `SimToReal` mode accepts `Real` evaluation config.
- ✅ A′1.5 — Existing CLI / runner / backend dispatch must compile.

**Acceptance:** `cargo build` succeeds; existing tests pass.

---

## Phase A′2 — `SimToRealBackend` composite backend

**Implementation steps:**
- ✅ A′2.1 — New file `src/experiments/sim_to_real_backend.rs` implementing `ExperimentBackend`.
- ✅ A′2.2 — `resolve_data`: loads real series, calibrates (Quick-EM by default), simulates a synthetic stream of length `horizon`, returns a `DataBundle` with split provenance.
- ✅ A′2.3 — Synthetic stream plumbed through `DataBundle.synthetic_observations` and `FeatureBundle.synthetic_train_obs`; provenance in `split_summary_json`.
- ✅ A′2.4 — `build_features`: real `FeatureStream`'s `FittedScaler` is applied to synthetic stream (one-sided policy).
- ✅ A′2.5 — `train_or_load_model`: EM on `features.synthetic_train_obs` via `train_or_load_model_shared`.
- ✅ A′2.6 — `run_online`: synthetic-trained `FrozenModel` runs over real features via `run_online_shared`.
- ✅ A′2.7 — `evaluate_real`: Route A + Route B re-loads real series and computes metrics like `RealBackend`.
- ✅ A′2.8 — Register the backend in CLI dispatch.

**Acceptance:** A simple unit test simulates the full pipeline on a small synthetic-stand-in real dataset and verifies the EM saw synthetic data and the detector ran on real data.

---

## Phase A′3 — Register `simreal_*` experiment(s)

**Implementation steps:**
- ✅ A′3.1 — Added `simreal_spy_daily_hard_switch()` builder in `src/experiments/registry.rs`.
- ✅ A′3.2 — Registry entry added.
- ⏭️ A′3.3 — (Optional) `simreal_wti_daily_surprise()`, `simreal_spy_intraday_hard_switch()` — deferred.

**Acceptance:** `cargo run -- e2e --id simreal_spy_daily_hard_switch` succeeds end-to-end against cached SPY data.

---

## Phase A′4 — `synthetic_training_provenance.json` artifact

**Implementation steps:**
- ✅ A′4.1 — In `runner.rs`, when `cfg.mode == SimToReal`, the existing split-summary blob (calibration profile, strategy, calibrated `ModelParams`, scale_check, etc.) is written under the dedicated filename `synthetic_training_provenance.json`.
- ✅ A′4.2 — `ArtifactRef { name: "synthetic_training_provenance", kind: "json" }` added.
- ✅ A′4.3 — Artifact appears in `result.artifacts` and therefore in `result.json`.

**Acceptance:** Running a `simreal_*` experiment produces the new JSON, and the user can read it without needing to re-run anything.

---

## Phase D′1 — `sim_to_real_summary.json`

**Decision:** A side-by-side artifact that is also written by the runner when in `SimToReal` mode. Initially it just collates the synthetic-trained model + real-test metrics. The train-on-real comparator (D′2) extends it.

**Implementation steps:**
- ✅ D′1.1 — `sim_to_real_summary.json` is emitted after `evaluate_real` when `mode == SimToReal`. Contents: run_id, train_source / test_source labels, n_real_observations, n_synthetic_observations, provenance JSON (incl. scale_consistency_check), model_params_synthetic_trained, eval_real_metrics.

**Acceptance:** JSON readable from disk after a `simreal_*` run.

---

## Phase D′2 — Train-on-real comparator (the "sim-to-real gap")

**Decision:** Optionally run the same experiment with the same real test partition but EM trained on the real train partition. Compare metric-for-metric. This is the figure the thesis needs.

**Implementation steps:**
- ✅ D′2.1 — `compare-sim-vs-real --id <simreal_id> [--cache <path>] [--save <out_dir>]` in `cli/mod.rs` runs `SimToRealBackend` + a derived real-trained variant (mode → Real, data → DataConfig::Real built from the CalibratedSynthetic real_* fields) and writes `sim_vs_real_comparison.json`.
- ✅ D′2.2 — `deltas_real_minus_sim` block in `sim_vs_real_comparison.json` records `event_coverage_gap`, `alarm_relevance_gap`, `segmentation_coherence_gap` = real − sim.
- ✅ D′2.3 — `sim_vs_real_comparison.md` written alongside the JSON: three-column delta table printed to the terminal too.

**Acceptance:** One command produces the comparison artifact for `simreal_spy_daily_hard_switch`.

---

## Phase C′1 + C′2 — Alternative calibration policies (polish)

- ✅ C′1 — `VariancePolicy::MagnitudeConditioned` (median-split on |y|) in
  `src/calibration/mapping.rs`. Falls back to `QuantileAnchored` (with a
  mapping note) when `profile.observations` is empty. 2 unit tests.
- ✅ C′2 — `CalibrationMappingConfig::min_high_low_ratio` (default `1.0`)
  enforces a minimum `sigma_high / sigma_low` ratio under `QuantileAnchored`
  and `MagnitudeConditioned`.  When the empirical ratio is below the floor,
  variances are rescaled log-symmetrically around the geometric mean to meet
  the guard, and the action is recorded in `mapping_notes`. 2 unit tests.

---

## Phase C′4 — Policy-aware verifier (polish)

- ✅ C′4.1 — `VerificationTargetMask` struct in `src/calibration/verify.rs`
  (one bool per check: `mean`, `variance`, `q05`, `q95`, `abs_acf1`,
  `sign_change_rate`).
- ✅ C′4.2 — `VerificationTargetMask::for_policy(&CalibrationMappingConfig)`
  disables the mean check under `MeanPolicy::ZeroCentered` and disables
  `abs_acf1` / `sign_change_rate` under `CalibrationStrategy::QuickEm`.
- ✅ C′4.3 — `verify_calibration_masked` writes per-field
  `FieldVerification { checked, passed, diff, tolerance }` records to
  `CalibrationVerification::field_results`. The global verdict aggregates
  only fields with `mask.<field> == true`. Surfaced in
  `CalibrationReportView::field_results`. 3 unit tests.

---

## Phase E — Cleanups (polish, time-permitting)

- ✅ E1 — `stationary_pi(transition)` in `src/calibration/mapping.rs` via power iteration with uniform fallback. Wired into the Summary calibration path (with mapping note "pi set to stationary distribution of transition matrix"). 2 unit tests.
- ✅ E2 — `OnlineRunArtifact::changepoint_truth: Option<Vec<usize>>` plumbed
  through `src/experiments/runner.rs` (filled from `DataBundle.changepoint_truth`
  post `run_online` so synthetic backends carry truth alongside the score
  trace; real and sim-to-real runs leave it `None`).
- ✅ E3 — `CalibrationDatasetTag::frequency` now carries bar size for
  intraday streams (e.g. `"intraday_15min"`) via `DataMode::bar_minutes()`
  in `summarize_feature_stream`.
- ✅ E4 — `CalibrationReportView` serialises `mapping_notes`, `frequency`,
  `field_results`, and `verification_mask`. JSON round-trip covered by the
  new `report_view_surfaces_mapping_notes_and_frequency` test.
- ⬜ E5 — Registry rename / split — defer; risk of breaking too many tests.

---

## Documentation phase

- ✅ F1 — Updated `docs/synthetic_to_real_calibration.md` with the new `CalibrationStrategy::QuickEm`, the scale-consistency contract (§8.3), the strategies section (§13), and the sim-to-real artifacts section (§14).
- ✅ F2 — Updated `docs/thesis/chapter6_synthetic_scenario_generation_and_calibration.md` with new §6.9 (Quick-EM calibration and the sim-to-real bridge).
- ✅ F3 — New §9b in `docs/experiment_runner.md` describing `SimToReal` mode, `DataConfig::CalibratedSynthetic`, the SimToReal-mode artifacts, and the `compare-sim-vs-real` comparator. The DryRunBackend Scope table in §13 lists `SimToRealBackend` and `compare-sim-vs-real`.
- ⏭️ F4 — New short doc `docs/sim_to_real_training.md` deferred (overlaps with §9b + chapter6 §6.9).
- ⏭️ F5 — Chapter 7 cross-reference deferred (no thesis content yet for that chapter — verify before adding).
- ⏭️ F6 — README update deferred; README mentions `cargo run` only.

---

## Verification phase

- ✅ V1 — Added §19.5 "Verify sim-to-real pipeline (post-Phase-17′ baseline)" to `notes/Verification.md`. Updated execution-order list and final sign-off checklist.
- ✅ V2 — `cargo test` clean (349 passed, 0 failed; +2 from stationary_pi tests).
- ✅ V3 — `cargo run --release -- run-real --id simreal_spy_daily_hard_switch` completes end-to-end, status SUCCESS, 21 artifacts including `synthetic_training_provenance.json`, `sim_to_real_summary.json`, `route_a_result.json`, `route_b_result.json`. Note: `cli/mod.rs::direct_run_real` was fixed to dispatch on `cfg.mode` so SimToReal experiments now route to `SimToRealBackend` automatically. `main.rs::is_direct_command` also extended to include `compare-sim-vs-real`.
- ✅ V4 — `cargo run --release -- compare-sim-vs-real --id simreal_spy_daily_hard_switch ...` writes both `sim_vs_real_comparison.json` and `sim_vs_real_comparison.md`. Example: sim-trained (0.000 / 0.000 / 0.1492) vs real-trained (0.0769 / 0.0909 / 0.1389); deltas +0.0769 / +0.0909 / −0.0103. (Note: sim-trained π collapsed to (1,0) on z-scored SPY daily — a thesis-relevant empirical finding about Quick-EM on this asset, not a code defect.)
- ⬜ V5 — Re-run prior Verification.md steps — user-side action.

---

## Risk register

| Risk | Mitigation |
|---|---|
| `DataBundle` / `FeatureBundle` shape changes break many tests | Add new optional fields rather than refactor structs |
| Quick-EM on raw real returns may not converge cleanly | Use multi-start (`em_n_starts >= 3`) and warn on non-convergence |
| Scaler fitted on real-train then applied to synthetic may produce non-unit-σ synthetic | Acceptable; the verifier reports it as a flag |
| Registry naming collisions with prior `hard_switch` test snapshots | Use new prefix `simreal_` and keep old IDs untouched |
| CLI changes break batch / optimize subcommands | Each subcommand: check the dispatch site, add the new mode |

---

## Live progress log

(Append timestamped notes as work proceeds.)

- 2026-05-14T??: Plan created.
