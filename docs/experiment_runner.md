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

- `config.snapshot.json`,
- `result.json`,
- summary JSON.

This supports auditability and thesis table/figure regeneration.

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
