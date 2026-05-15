# Sim-to-Real Detector Recovery Plan — `sim_to_real_recovery_2026_05_15`

**Goal.** Close (or characterise) the sim-to-real generalisation gap surfaced
by the `verify_2026_05_15` run, where the synthetic-trained detector on SPY
daily log-returns produced `event_coverage = 0.0` while the real-trained
detector produced `0.0769`.

**Source of the failure (already diagnosed).**

From `verification/verify_2026_05_15/run_sim_to_real/model_params.json`:

```
π          = (1.0, 9.4e-12)       ← degenerate
variances  = (2.33, 0.296)
transition = ((0.943, 0.057), (0.023, 0.977))
log_likelihood = -2372.35, converged in 23 iters
```

Two structural pathologies:

1. **π collapses to (1, 0).** EM finds the maximum-likelihood story for
   z-scored SPY daily log-returns is "we started in a tantrum and slowly
   mean-reverted into calm", because real SPY daily log-returns have
   $\hat\rho_1(y) \approx -0.15$. A Gaussian MSM with $\mu_0 \approx \mu_1$
   cannot reproduce negative serial correlation, so EM expresses it as an
   asymmetric initial state.
2. **The HardSwitch detector is filter-initialised from this fitted π.** With
   $P_{00} = 0.943$ self-stay, posterior probability of regime 0
   monotonically decays but never *crosses* the 0.55 threshold → zero alarms
   over the entire real test stream.

Route-B segmentation coherence (`0.149` sim vs `0.139` real) confirms the
sim-trained model is *internally consistent*; it just never *fires*. The
gap is a **detector-firing pathology rooted in a feature mismatch**, not a
model-quality pathology.

---

## Three-pronged response

This plan implements three interventions, in order of expected payoff. Each
intervention is independent and can be evaluated separately; the final
empirical table will allow attribution.

### Option 1 — Stationary-π post-processing in Quick-EM

**Rationale.** At training time we have no information about which regime
the test stream starts in. The principled prior is the *long-run frequency*
of regime occupation, i.e. the stationary distribution $\pi^\star$ of $\hat
P$, not the EM-fit initial state $\hat\pi$. For the verified SPY daily
matrix, $\pi^\star \approx (0.286,\,0.714)$ instead of $(1, 0)$.

**Implementation.**

1. Add `PiPolicy` enum to `src/calibration/mapping.rs`:
   ```rust
   #[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
   pub enum PiPolicy {
       /// Use the EM-fitted initial distribution \hat\pi as-is.
       Fitted,
       /// Replace \hat\pi with the stationary distribution of \hat P
       /// (left eigenvector of \hat P^T at eigenvalue 1, via power iteration).
       #[default]
       Stationary,
   }
   ```
   Default = `Stationary` so all sim-to-real experiments inherit the cure
   automatically.
2. Add `pi_policy: PiPolicy` field to `CalibrationMappingConfig` with
   `#[serde(default)]`.
3. In `calibrate_via_quick_em`, after the regime canonicalisation block,
   apply the policy:
   ```rust
   let pi = match config.pi_policy {
       PiPolicy::Fitted => pi,                       // current behaviour
       PiPolicy::Stationary => {
           notes.push("Quick-EM: pi replaced with stationary distribution of P".into());
           stationary_pi(&transition_rows)
       }
   };
   ```
   `stationary_pi` already exists at `mapping.rs:466` (`pub(crate)`) and is
   used by the Summary path — no new numerical code needed.
4. Update `CalibrationMappingConfig::default()` to include
   `pi_policy: PiPolicy::default()` (= Stationary).
5. Add a unit test: build a known $(P, \hat\pi_{\text{degenerate}})$, run a
   Quick-EM-like build and assert the returned $\pi$ matches the analytic
   stationary distribution within $10^{-6}$.

### Option 2 — Register the AbsReturn / K=3 sim-to-real entry

**Rationale.** The thesis's own joint `optimize --model` run (verification
§18) found `feature = AbsReturn`, `k_regimes = 3` produces
`event_coverage = 0.846` on real SPY daily. The reason is exactly that
$|r_t|$ has $\hat\rho_1 \approx +0.37$ (volatility clustering), which **is**
inside the expressive range of a Gaussian MSM. Quick-EM on
$|y^{\text{real,train}}|$ does not need a degenerate $\pi$ to fit the data.

**Implementation.**

Register a new experiment in `src/experiments/registry.rs`:

| Field | Value |
|---|---|
| `id` | `simreal_spy_daily_abs_return_k3` |
| `description` | "SPY daily — Sim-to-real | AbsReturn / K=3 / HardSwitch (joint-optimum)" |
| `mode` | `ExperimentMode::SimToReal` |
| `data` | `CalibratedSynthetic { real_asset: "SPY", real_frequency: Daily, horizon: 2000, mapping.k: 3, mapping.strategy: QuickEm{100, 1e-6}, mapping.pi_policy: Stationary }` |
| `features` | `AbsReturn` + `ZScore` |
| `model` | `k_regimes: 3`, `FitOffline`, `em_max_iter: 300`, `em_tol: 1e-7` |
| `detector` | `HardSwitch` with `threshold = 0.55`, `persistence_required = 2`, `cooldown = 5` (joint optimum from §18) |
| `evaluation` | `Real { proxy_events_path: "data/proxy_events/spy.json", route_a_*, route_b_min_segment_len: 10 }` |

### Option 5 — Sim-to-real sweep over (asset, feature, K)

**Rationale.** A single positive result is fragile; a table across multiple
asset / feature combinations is much stronger evidence that the bridge
works. The available dataset inventory supports five daily assets (SPY,
QQQ, BRENT, GOLD, WTI). The sweep also gives the chapter both pass and
fail rows.

**Implementation.** Register five additional sim-to-real entries with the
joint-optimum configuration (AbsReturn / K=3 / HardSwitch / Stationary-π).

| New id | Asset | Frequency | Feature | K | Detector |
|---|---|---|---|---|---|
| `simreal_spy_daily_abs_return_k3` | SPY | Daily | AbsReturn | 3 | HardSwitch(0.55, 2, 5) |
| `simreal_qqq_daily_abs_return_k3` | QQQ | Daily | AbsReturn | 3 | HardSwitch(0.55, 2, 5) |
| `simreal_wti_daily_abs_return_k3` | WTI | Daily | AbsReturn | 3 | HardSwitch(0.55, 2, 5) |
| `simreal_brent_daily_abs_return_k3` | BRENT | Daily | AbsReturn | 3 | HardSwitch(0.55, 2, 5) |
| `simreal_gold_daily_abs_return_k3` | GOLD | Daily | AbsReturn | 3 | HardSwitch(0.55, 2, 5) |

The existing `simreal_spy_daily_hard_switch` (LogReturn / K=2) is retained
as the **documented counterexample** for the chapter.

For SPY and QQQ we have proxy events files (`data/proxy_events/spy.json`,
`qqq.json`). For commodities we only have Route B (segmentation coherence)
unless proxy events are also defined; if not, the entries fall back to
Route B only or use a generic proxy events file. We check
`data/proxy_events/` before registration and gate the evaluation config
accordingly.

---

## Execution order

1. **Implementation plan** (this document).
2. **Option 1 code** — `PiPolicy` enum + `pi_policy` field + `calibrate_via_quick_em` branch + unit test.
3. **Build + cargo test** — confirm all 328 (+ new) tests pass.
4. **Option 2/5 registry entries** — 5 new `simreal_*_abs_return_k3` constructors and registry rows.
5. **Build + clippy** — confirm clean.
6. **Run all 6 sim-to-real experiments** — each via `compare-sim-vs-real --id <id>` to get both sim-trained and real-trained metrics.
7. **Aggregate** results into `verification/sim_to_real_recovery_2026_05_15/sim_to_real_sweep.{json,md}` with one row per id and the three metrics (event_coverage, alarm_relevance, segmentation_coherence) for both sim-trained and real-trained variants plus the delta column.
8. **Update Chapter 5** §5.8.3 ("The role of the calibration verdict") with the counterexample diagnosis and a reference to the table.
9. **Sign-off** — short `SIGN_OFF.md` in the recovery directory listing what passed.

## Acceptance criteria

- All existing tests still pass; at least one new test asserts stationary-π
  is applied in Quick-EM.
- `clippy` is clean.
- At least one of the new AbsReturn / K=3 entries achieves
  `sim_trained.event_coverage > 0` on its target asset (the "decently
  working detector" bar).
- At least one entry reaches `sim_trained.event_coverage ≥ 0.30` *or* a
  documented near-miss; either outcome is publishable as long as the
  diagnostic story is clear.
- The SPY daily LogReturn entry continues to produce a degenerate-π
  diagnostic (probably nonzero now that π is stationary, but still
  significantly worse than AbsReturn) for the counterexample row.

## What this plan does NOT do

- **Multi-restart EM (Option 4).** Worth doing later; orthogonal to the
  stationary-π fix and out of scope here. The existing `em_n_starts`
  knob on `ModelConfig` can be exercised in a follow-up sweep if needed.
- **Detector retuning (Option 3).** We keep the joint-optimum HardSwitch
  parameters from the existing §18 grid search. Per-id detector retuning
  could squeeze further performance but is orthogonal to the structural
  fix.
- **Intraday entries.** Daily-only sweep keeps the running time bounded.
  An intraday extension is a one-line addition once the daily sweep
  validates the recipe.
