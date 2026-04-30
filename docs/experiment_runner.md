# Experiment Runner and Reproducibility Layer

**Phase 20 — Markov Switching Model Project**

---

## 1. Objective

The project already contains model, detection, benchmarking, feature, calibration,
and real-data evaluation components. Phase 20 introduces the orchestration layer
that turns those components into a reproducible empirical system.

The central principle is:

$$
\boxed{\text{Every run is configuration-driven and reconstructible from artifacts.}}
$$

---

## 2. Formal Run Contract

A single run is the mapping

$$
\text{ExperimentConfig} \longrightarrow \text{ExperimentResult}.
$$

- `ExperimentConfig` contains all experiment inputs and policies.
- `ExperimentResult` contains status, outputs, timing, and references needed for
  rerun and audit.

No hidden parameter edits are allowed between runs.

---

## 3. Configuration Semantics

A complete config includes nine categories.

### 3.0 Meta config

Identifies the run for logging and artifact naming:
- run label (human-readable name),
- optional notes.

### 3.1 Data config

Defines source and scope:
- synthetic scenario ID and horizon, or
- real dataset ID, asset, frequency, date range.

### 3.2 Feature config

Defines observed-process construction:
- feature family,
- rolling window semantics,
- session-aware policy,
- scaling policy.

### 3.3 Model config

Defines offline model policy:
- regime count,
- fit-vs-load mode,
- EM settings.

### 3.4 Detector config

Defines online alarm policy:
- detector type,
- threshold,
- persistence,
- cooldown.

### 3.5 Evaluation config

Defines mode-compatible evaluation:
- synthetic benchmark settings, or
- real Route A + Route B settings.

### 3.6 Output config

Defines artifact policy:
- root directory,
- serialization options,
- trace saving policy.

### 3.7 Reproducibility config

Defines deterministic behavior:
- seed,
- deterministic run ID policy,
- config snapshot policy,
- metadata recording policy.

---

## 4. Result Semantics

A complete result must include:

1. Run metadata:
   - run ID,
   - run label,
   - mode,
   - timestamps,
   - seed,
   - config hash.
2. Stage timings:
   - data resolution,
   - feature build,
   - model training/loading,
   - online execution,
   - evaluation,
   - export.
3. Summaries:
   - model summary (includes fitted params: π, P, means, variances, LL history, n_iter, converged),
   - detector summary (includes alarm indices),
   - evaluation summary (synthetic or real, includes full MetricSuite: precision, recall, miss rate, FAR, delay mean + median).
4. Trace data (populated only when `save_traces = true`):
   - score trace (per-step detector score),
   - regime posteriors (T × K filtered probabilities).
5. Artifact references:
   - config snapshot,
   - run result,
   - summary files.
5. Status and warnings:
   - success,
   - partial success,
   - failed with stage + message.

---

## 5. Synthetic vs Real Modes

Mode is explicit and typed.

- **Synthetic mode** uses synthetic data resolution and synthetic benchmark evaluation.
- **Real mode** uses real dataset resolution and real Route A + Route B evaluation.

Mode/evaluation mismatch is invalid configuration.

---

## 6. Orchestration vs Algorithm Separation

The runner must orchestrate only. It must not reimplement:
- filtering mathematics,
- detector score equations,
- feature formulas,
- benchmark internals.

The runner performs controlled dispatch:

1. resolve data,
2. build observation features,
3. train/load model,
4. run online detector,
5. route evaluation,
6. export artifacts.

This separation keeps scientific logic testable and orchestration maintainable.

---

## 7. Reproducibility Policy

Reproducibility is explicit, not implicit.

Required properties:

1. Config snapshot saved with run artifacts.
2. Seed stored in metadata.
3. Config hash stored in metadata.
4. Deterministic run ID optional and seed-governed.
5. Stable output-path policy for all runs.

For synthetic experiments, same config + same seed should yield equivalent core
outputs under deterministic backend assumptions.

---

## 8. Structured Artifact Policy

Artifacts must be written to deterministic, human-readable paths keyed by:

- mode,
- run label,
- run ID.

Minimal export set:

- `config.snapshot.json` — exact ExperimentConfig snapshot,
- `result.json` — full ExperimentResult,
- `summary.json` — lightweight metrics summary,
- `model_params.json` — fitted ModelParams (reloadable via `LoadFrozen`),
- `fit_summary.json` — human-readable EM fit metadata: K, n_iter, converged, log_likelihood_initial/final, convergence_reason,
- `loglikelihood_history.csv` — log-likelihood at each EM iteration (`iteration,log_likelihood`),
- `feature_summary.json` — feature pipeline metadata: label, n_obs, train_n, val_n, obs_mean/variance/std/min/max, scaling, session_aware.

Additional artifacts saved when `save_traces = true`:

- `score_trace.csv` — per-step detector score,
- `alarms.csv` — alarm timestamps and scores,
- `regime_posteriors.csv` — T×K filtered posterior probabilities.

Plot artifacts (`signal_alarms.png`, `detector_scores.png`, `regime_posteriors.png`, `delay_distribution.png`) are written when a font backend is available. On Windows without a font backend, plots are skipped without error.

---

## 9. Batch/Grid Execution

Thesis experiments require many runs over data, feature, detector, and policy
axes. Therefore, orchestration must support batch execution:

- iterate run list or grid expansion,
- preserve per-run isolation,
- collect run-level outputs,
- aggregate success/failure counts,
- optional stop-on-error policy.

Batch support is part of the scientific method because comparisons require
consistent execution conditions.

---

## 10. Failure and Partial Results

Failure handling is first-class:

- failed runs must record stage and error message,
- partial artifacts should be retained where possible,
- partial-success status is valid when core run succeeds but export stage fails.

This prevents silent run loss and improves thesis-scale traceability.

---

## 9. Real-Mode Artifacts

When the experiment is backed by `RealBackend`, the run directory additionally contains:

- `split_summary.json` — train/val/test date boundaries and point counts (70/15/15 split),
- `data_quality.json` — NaN rate, gap detection, out-of-range checks,
- `real_eval_summary.csv` — Route A + Route B metric row,
- `route_a_result.json` — proxy event alignment details (agreement, delay, missed, false alarms),
- `route_b_result.json` — segmentation self-consistency details (n_segments, mean contrast ratio, coverage),
- `segmentation.png` — segment-coloured plot of the real series (requires font backend).

Real runs are launched via the `run-real` direct subcommand:

```
cargo run -- run-real --id real_spy_daily_hard_switch --cache data/commodities.duckdb --save ./output
```

The `DataConfig::Real` section of each registry entry controls which asset, frequency, and date window are used.

---

## 11. Canonical Run Workflow

Given `ExperimentConfig`:

1. Validate configuration.
2. Generate run ID and config hash.
3. Create output directory and snapshot config.
4. Resolve data.
5. Build feature stream.
6. Train or load model.
7. Run online detector.
8. Dispatch evaluation by mode.
9. Export result artifacts.
10. Return `ExperimentResult`.

This workflow defines the operational backbone for synthetic and real
Markov-switching detector experiments.

---

## 12. Parameter Optimization (`optimize`)

The `optimize` subcommand provides automated detector parameter search for
real-data experiments. It runs a two-phase workflow:

### Phase 1 — Grid Search

A `ParamGrid` is selected automatically based on the detector type:

| Detector | Threshold range | Persistence | Cooldown | Points |
|----------|----------------|-------------|----------|--------|
| `HardSwitch` | 0.30 – 0.80 | 1, 2, 3, 5 | 0, 3, 5, 10 | 128 |
| `Surprise` | 1.0 – 6.0 | 1, 2, 3, 5 | 0, 5, 10, 20 | 128 |
| `PosteriorTransition` | 0.10 – 0.50 | 1, 2, 3 | 0, 3, 5, 10 | 84 |

For each grid point the runner:

1. Patches `threshold`, `persistence_required`, `cooldown` onto the base config.
2. Disables artifact writes (`write_json = false`, `write_csv = false`, `save_traces = false`) for speed.
3. Calls `ExperimentRunner::run` on `RealBackend`.
4. Extracts evaluation metrics and computes a combined score:

$$
\text{score} = 0.5 \times \text{coverage} + 0.5 \times \text{precision\_like}
$$

For real-mode evaluation the three Route A + B sub-metrics are averaged before this formula:

$$
\text{coverage} = \text{precision\_like} = \frac{\text{event\_coverage} + \text{alarm\_relevance} + \text{segmentation\_coherence}}{3}
$$

All grid points are sorted by score descending. The top-ranked config is selected as `best_config`.

### Phase 2 — Full E2E Run

The best config is re-run with full artifact output:

- `write_json = true`, `write_csv = true`, `save_traces = true`
- Plots enabled (requires font backend)
- `root_dir` set to the `--save` destination

### Optimize Artifacts

In addition to the standard run artifact set, `optimize` writes:

| File | Contents |
|------|----------|
| `search_report.json` | Full ranked grid — all N scored points |
| `search_summary.txt` | Human-readable top-N table + best params |

### CLI Usage

```
cargo run -- optimize --id real_spy_daily_hard_switch
cargo run -- optimize --id real_wti_daily_surprise --cache data/commodities.duckdb --save ./runs/optimize/wti --top 15
```

The `optimize` function is exported from `experiments::search` and accepts any
`ExperimentBackend`, making it testable with `DryRunBackend` and reusable in
batch workflows.
