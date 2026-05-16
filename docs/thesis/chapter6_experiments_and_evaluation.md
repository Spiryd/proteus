# Chapter 6 — Experiments and Evaluation

**Relation to previous chapters.** Chapters 3 and 4 established the model and
detector theory. Chapter 5 defined the calibration operator $\mathcal{K}$
that maps empirical statistics to synthetic generator parameters. This chapter
operationalises all three components on real market data. It is organised in
four empirical layers of increasing difficulty:

1. **Synthetic experiments** (§6.4) — calibrated synthetic streams with known
   hidden paths; establish baseline detector behaviour and confirm that the
   model class can produce recoverable changepoints.
2. **Real-data experiments** (§6.5) — the detector trained and tested on real
   market observations; first direct assessment against proxy events.
3. **Sim-to-real experiments** (§6.6) — model trained on synthetic data,
   tested on real data; the central empirical argument of the thesis.
4. **Joint optimisation** (§6.7) — grid search over model architecture and
   detector hyperparameters; derives the canonical configuration used in the
   sim-to-real sweep.

Before the experiments, §6.1–§6.3 fix the datasets, preprocessing pipeline,
and evaluation protocol that all four layers share.

---

## 6.1 Datasets

Four primary series are used. All are loaded from a local DuckDB cache; no
network call is issued at experiment time, ensuring reproducibility.

| Series | Asset | Frequency | Source field | Date window | Raw bars |
|---|---|---|---|---|---|
| SPY daily | SPDR S&P 500 ETF | Daily | Adjusted close | 2018-01-01 – present | 6 662 |
| WTI daily | WTI crude oil | Daily | Spot/settlement | 2018-01-01 – present | 10 143 |
| GOLD daily | Gold spot | Daily | Spot/settlement | available start – present | 5 256 |
| SPY 15min | SPDR S&P 500 ETF | Intraday 15 min | Adjusted close | 2022-01-01 – 2025-12-31 | ≈378 149 |

**Daily series** are sorted ascending, deduplicated, and gap-checked.
Calendar gaps (weekends, public holidays) are expected and are not flagged.

**Intraday series** additionally apply the RTH session filter: only bars
with open time $t$ satisfying $09{:}30 \le t < 16{:}00$ ET are retained.
This reduces the SPY 15-minute history from ≈378 000 bars to approximately
25 000 RTH bars — consistent with the ratio $6.5/96$ of the trading day to
the 24-hour bar grid. Pre-market and after-hours bars are excluded because
their distinct volatility regime would produce a spurious changepoint at
every session boundary.

### 6.1.1 Chronological partition

Each series is split chronologically into training / validation / test
partitions at the $70/15/15$ boundary indices. For the SPY daily window the
realised cut points are $\tau_1 =$ 2023-10-25 and $\tau_2 =$ 2025-01-28,
yielding $n_{\text{train}} = 1463$, $n_{\text{val}} = n_{\text{test}} = 314$.
Random shuffling is structurally forbidden. All partition boundaries and
sizes are recorded in `split_summary.json`.

---

## 6.2 Feature families and preprocessing

Each experiment selects one of five causal feature operators $\phi$
(introduced in §5.4 of the thesis framework). For the registered experiments
in this chapter:

| Operator | Symbol | Registered use |
|---|---|---|
| LogReturn | $r_k = \log P_k - \log P_{k-1}$ | all Synthetic; SPY daily real; WTI daily real; SPY intraday; all SimToReal baselines |
| AbsReturn | $\lvert r_k\rvert$ | SimToReal joint-optimum (SPY, WTI, GOLD) |
| SquaredReturn | $r_k^2$ | `squared_return_surprise` synthetic variant |
| RollingVol(5) | σ̂ over 5-bar window | ModelGrid sweep only |
| RollingVol(20) | σ̂ over 20-bar window | ModelGrid sweep only |

**Scaling.** The default is z-scoring: parameters $(\hat\mu, \hat\sigma)$ are
estimated on the training partition only and then frozen; the same transform
is applied to validation, test, and online streams. For intraday data,
rolling-window operators reset state at each session boundary
(`session_reset = true`).

---

## 6.3 Evaluation protocol

Because the three experiment modes produce qualitatively different types of
output, the evaluation protocol splits into two branches.

### 6.3.1 Synthetic evaluation

When $\mathcal{C}(S_{1:T})$ is available (Synthetic mode only), detector
output is assessed against the ground-truth changepoint set via a
**tolerance-window matching** procedure (§3 of the benchmark layer). An alarm
at time $a$ is a *true positive* for changepoint $\tau \in \mathcal{C}$ if
$\lvert a - \tau \rvert \le W$ (default $W = 20$ bars). At most one alarm may
match each changepoint (first-match rule); excess alarms are false positives.

This yields the `MetricSuite`:

| Symbol | Name | Definition |
|---|---|---|
| $M$ | True changepoints | $\lvert\mathcal{C}\rvert$ |
| $N$ | Total alarms | $\lvert\{a_t = 1\}\rvert$ |
| TP | True-positive alarms | alarms within window of a changepoint |
| FP | False-positive alarms | $N - \mathrm{TP}$ |
| $M_{\det}$ | Detected changepoints | changepoints with $\ge 1$ matched alarm |
| Precision | $\mathrm{TP}/N$ | fraction of alarms that are genuine |
| Recall (= `coverage`) | $M_{\det}/M$ | fraction of changepoints detected |
| Miss rate | $(M - M_{\det})/M$ | $1 - $ Recall |
| FAR | $\mathrm{FP}/T$ | false positives per observation |
| Delay | mean/median bars | time from changepoint to first matched alarm |

The summary tuple reported per run is `(coverage, precision_like, n_events, n_alarms)`,
where `precision_like` is the same as Precision and `coverage` is Recall.

### 6.3.2 Real-data evaluation (Route A and Route B)

For real and sim-to-real experiments there are no latent ground-truth labels.
Assessment proceeds via two complementary routes.

**Route A — Proxy-event alignment.** A set of externally documented macro
events $\mathcal{E} = \{e_1, \ldots, e_P\}$ is loaded from
`data/proxy_events/{asset}.json`. Each event $e_p$ has a reference timestamp
$\tau_p^*$. An alarm at index $a$ is *aligned* to $e_p$ if
$a \in [\tau_p^* - \Delta^-, \tau_p^* + \Delta^+]$, where the tolerance
window is asset- and frequency-specific:

| Mode | $\Delta^-$ (pre-bars) | $\Delta^+$ (post-bars) |
|---|---|---|
| Daily (SPY, WTI, GOLD) | 5 | 10 |
| Intraday 15min (SPY) | 20 | 40 |

Two aggregate metrics are computed:

$$
\text{event\_coverage} = \frac{\lvert\{e_p : \exists\, a \in \text{window}(e_p)\}\rvert}{P},
\qquad
\text{alarm\_relevance} = \frac{\lvert\{a : \exists\, e_p \ni a\}\rvert}{N}.
$$

`event_coverage` is recall over events; `alarm_relevance` is precision over
alarms. A random detector will score poorly on both: few events will happen
to have an alarm nearby, and few alarms will land near any documented event.

**Route B — Segmentation self-consistency.** The detector's alarm sequence
partitions the observation stream into $n_{\text{seg}}$ segments. Route B
quantifies whether adjacent segments are statistically distinguishable using a
segment coherence score:

$$
\text{coherence\_score} = \frac{\overline{\lvert\Delta\bar y\rvert}}{\sqrt{\overline{s^2_{\text{within}}}}},
$$

where $\overline{\lvert\Delta\bar y\rvert}$ is the mean absolute between-segment
mean contrast across all adjacent pairs and $\overline{s^2_{\text{within}}}$
is the mean within-segment variance. This is a signal-to-noise ratio for the
segmentation: values $>1$ indicate that the average between-boundary shift
is larger than the typical within-segment standard deviation.
Segments shorter than `min_segment_len` (10 bars for daily, 20 for
intraday) are flagged and, by default, included in the global statistics
(`ShortSegmentPolicy::FlagOnly`).

**The combined optimisation score** used in the joint grid search (§6.7) is
the equal-weight average of the three real metrics:

$$
\text{score} = \tfrac{1}{3}\bigl(\text{event\_coverage} + \text{alarm\_relevance} + \text{coherence\_score}\bigr).
$$

### 6.3.3 Proxy event catalogues

Proxy events represent externally documented structural breaks against which
detector alignment is assessed. They are not ground truth — a detector that
misses a listed event is not necessarily wrong, and the catalogues do not
claim completeness. Their role is to provide a financially meaningful
evaluation anchor that is independent of the detector's own output.

| Asset | Event types in catalogue | Examples |
|---|---|---|
| SPY | Macro shocks, Fed rate-cycle turns | COVID-19 crash (Mar 2020), Fed pivot (Nov 2021), SVB crisis (Mar 2023) |
| WTI | Supply/demand shocks | COVID demand collapse (Mar 2020), OPEC+ production cuts |
| GOLD | Risk-regime shifts | COVID flight-to-safety (Mar 2020), Fed hiking cycle starts |

---

## 6.4 Synthetic experiments

Synthetic experiments serve one purpose: with $\mathcal{C}(S_{1:T})$ known
exactly, every precision and recall number is interpretable without ambiguity.
All synthetic experiments share the base configuration: $K = 2$, LogReturn
feature, z-score scaler, horizon $T = 2000$, seed 42. The `scenario_calibrated`
generator is used as the data source; its parameters are set by
$\mathcal{K}_{\text{sum}}$ from a fixed empirical profile.

### 6.4.1 Registered synthetic experiments

| id | Detector | threshold $\tau$ | persistence $N_{\text{req}}$ | cooldown $C$ | Notes |
|---|---|---|---|---|---|
| `hard_switch` | HardSwitch | 0.5 | 2 | 5 | Core baseline |
| `posterior_transition` | PosteriorTransition (LeavePrevious) | 0.3 | 2 | 5 | Soft-threshold baseline |
| `posterior_transition_tv` | PosteriorTransition (TotalVariation) | 0.3 | 2 | 5 | Label-invariant variant |
| `surprise` | Surprise (raw) | 2.5 | 2 | 5 | Predictive-density baseline |
| `surprise_ema` | Surprise (EMA $\alpha=0.3$) | 2.5 | 2 | 5 | Slow-baseline adjustment |
| `squared_return_surprise` | Surprise (raw) | 2.5 | 2 | 5 | SquaredReturn feature |
| `hard_switch_shock` | HardSwitch | 0.5 | 2 | 5 | `shock_contaminated` scenario; jump contamination |
| `hard_switch_multi_start` | HardSwitch | 0.5 | 2 | 5 | EM multi-start $n_{\text{starts}} = 3$ |
| `hard_switch_frozen` | HardSwitch | 0.5 | 2 | 5 | Loads pre-fitted `FrozenModel` |

### 6.4.2 What the synthetic experiments establish

**Detector recoverability.** The core six experiments (`hard_switch`,
`posterior_transition`, `posterior_transition_tv`, `surprise`, `surprise_ema`,
`squared_return_surprise`) confirm that each of the three detector families
achieves non-trivial recall ($> 0$) on a 2000-step synthetic stream at default
thresholds. The matching window $W = 20$ is deliberately generous to avoid
penalising plausible early/late alarms.

**Sensitivity ordering.** The theoretical ordering from §4.8.2 of Chapter 4
is confirmed: hard switch fires only on clean posterior flips; posterior
transition fires earlier during gradual drift; predictive surprise fires on
outlier observations even without label switches.

**Feature transferability.** `squared_return_surprise` demonstrates that
AbsReturn/SquaredReturn features fed to the Surprise detector also yield
recoverable performance on a calibrated stream, motivating their use in the
sim-to-real experiments (§6.6).

**EM robustness.** `hard_switch_multi_start` confirms that the multi-start EM
($n_{\text{starts}} = 3$) produces detection results consistent with single-
start EM under the seed-42 generator, validating that the local-optimum
sensitivity of Baum–Welch is not a concern at $K = 2$ on streams of this length.

**Jump contamination.** `hard_switch_shock` uses the `shock_contaminated`
scenario (jump probability and scale are non-zero) and confirms that the
HardSwitch detector remains operable under moderate jump contamination; the
expected degradation in precision is recorded as a calibration note.

---

## 6.5 Real-data experiments

Three experiments apply the trained-offline, deployed-online pipeline to
actual market observations. EM is fit on $\mathcal{T}_{\text{train}}$ only;
the resulting `FrozenModel` runs the Hamilton filter causally over the full
series. Route A and Route B metrics are computed on $\mathcal{T}_{\text{test}}$.

### 6.5.1 Experiment: SPY daily — HardSwitch (`real_spy_daily_hard_switch`)

| Parameter | Value |
|---|---|
| Asset / frequency | SPY / daily |
| Feature | LogReturn, z-scored |
| EM | $K = 2$, max 300 iterations, tol $10^{-7}$ |
| Detector | HardSwitch, $\tau = 0.55$, $N_{\text{req}} = 2$, $C = 5$ |
| Route A tolerance | $\Delta^- = 5$, $\Delta^+ = 10$ bars |
| Route B min segment | 10 bars |

The confidence threshold $\tau_{\text{conf}} = 0.55$ requires the posterior
to assign more than $55\%$ probability to the new dominant regime before a
label-switch alarm is eligible. The persistence requirement $N_{\text{req}} =
2$ means two consecutive threshold crossings are needed; this absorbs
single-bar fluctuations where the posterior briefly touches $0.55$ and
immediately retreats.

SPY daily proxy events cover major macro-regime shifts: the COVID-19 crash and
recovery (2020), the Fed rate-hike cycle inception (2022), and the banking
sector stress episode (March 2023). The daily log-return series has empirical
$\hat\rho_1(r) \approx -0.15$, which is outside the expressive range of a
Gaussian MSM with $\mu_0 = \mu_1$; this limits the model's ability to
reproduce serial structure but does not prevent the detector from aligning
with volatility-driven macro events.

### 6.5.2 Experiment: WTI daily — Surprise (`real_wti_daily_surprise`)

| Parameter | Value |
|---|---|
| Asset / frequency | WTI crude oil / daily |
| Feature | LogReturn, z-scored |
| EM | $K = 2$, max 300 iterations, tol $10^{-7}$ |
| Detector | Surprise (raw), $\tau = 3.0$, $N_{\text{req}} = 2$, $C = 10$ |
| Route A tolerance | $\Delta^- = 5$, $\Delta^+ = 10$ bars |

The Surprise detector is chosen for WTI because crude-oil supply shocks
(OPEC production decisions, COVID demand collapse) manifest as abrupt
predictive-density drops rather than smooth posterior migrations. The higher
threshold $\tau = 3.0$ and longer cooldown $C = 10$ reflect the higher
intrinsic volatility of the WTI log-return series relative to SPY; without
them, a low-variance calm regime would fire on every spike above the
conditional mean.

### 6.5.3 Experiment: SPY intraday 15 min — HardSwitch (`real_spy_intraday_hard_switch`)

| Parameter | Value |
|---|---|
| Asset / frequency | SPY / intraday 15 min |
| Feature | LogReturn, z-scored, `session_aware = true` |
| Date window | 2022-01-01 – 2025-12-31 |
| EM | $K = 2$, max 200 iterations, tol $10^{-6}$ |
| Detector | HardSwitch, $\tau = 0.55$, $N_{\text{req}} = 3$, $C = 10$ |
| Route A tolerance | $\Delta^- = 20$, $\Delta^+ = 40$ bars ($\approx$ 5 h and 10 h) |
| Route B min segment | 20 bars ($= 5$ h) |

The `session_aware` flag ensures that the log-return operator is undefined
whenever $t_{k-1}$ and $t_k$ straddle a session boundary, eliminating
overnight returns from the feature stream. The persistence requirement is
raised to $N_{\text{req}} = 3$ (relative to 2 for daily) because intraday
posteriors fluctuate more rapidly and a two-step filter would produce too
many transient alarms within a single session.

The intraday experiment has ≈25 000 RTH bars after the session filter.
Proxy events are the same SPY catalogue as the daily experiment; Route A
tolerance windows of 20/40 bars (5 h / 10 h) correspond to the same
physical time as 5/10 daily bars at a 15-minute resolution.

---

## 6.6 Sim-to-real experiments

The four sim-to-real experiments are the empirical centrepiece of the thesis.
Each follows the pipeline of §5.8.2: $\mathcal{K}_{\text{EM}}$ is run on the
real training partition to produce $\hat\vartheta_{\text{EM}}$; a synthetic
stream of 2000 observations is sampled; EM is fit **only** on the scaled
synthetic stream; the resulting `FrozenModel` is then evaluated on the real
test partition.

All four use the detector configuration that emerged as the joint optimum
(§6.7): HardSwitch with $(\tau_{\text{conf}}, N_{\text{req}}, C) = (0.55, 2, 5)$
and `PiPolicy::Stationary`.

### 6.6.1 Experiment inventory

| id | Asset | Feature | $K$ | Role |
|---|---|---|---|---|
| `simreal_spy_daily_hard_switch` | SPY | LogReturn | 2 | Documented counterexample |
| `simreal_spy_daily_abs_return_k3` | SPY | AbsReturn | 3 | Joint-optimum, partial recovery |
| `simreal_wti_daily_abs_return_k3` | WTI | AbsReturn | 3 | Joint-optimum, Route-B-strong |
| `simreal_gold_daily_abs_return_k3` | GOLD | AbsReturn | 3 | **Canonical working example** |

### 6.6.2 The counterexample: SPY LogReturn ($K = 2$)

`simreal_spy_daily_hard_switch` is the intended negative control. SPY daily
log-returns have $\hat\rho_1(r) \approx -0.15$. A Gaussian MSM with equal
means ($\mu_0 = \mu_1$) cannot reproduce this negative autocorrelation. EM
responds by collapsing $\hat\pi \approx (1, 10^{-11})$; even with
`PiPolicy::Stationary` replacing $\hat\pi$ with $\pi^\star \approx (0.29,
0.71)$, the filter starts in the high-volatility regime on the real test
stream and zero alarms fire over the entire test horizon. `event_coverage =
0.000`, `coherence_score = 0.138`.

This experiment is left in the registry intentionally: it documents the
necessary condition for sim-to-real transfer (the feature autocorrelation
must lie inside the model class) and demonstrates that the calibration verifier
correctly flags the failure before the experiment runs.

### 6.6.3 Generalisation sweep: AbsReturn / $K = 3$

The three AbsReturn / $K = 3$ experiments are evaluated side-by-side against
their real-trained ceiling. The real-trained ceiling is the result of the
corresponding Real experiment run (`real_spy_daily_hard_switch`,
`real_wti_daily_surprise`, and a fresh real GOLD run) with the same feature
and $K$ but with EM fit on real training observations.

| Asset | sim `event_cov` | real `event_cov` | sim `coherence` | real `coherence` |
|---|---|---|---|---|
| SPY AbsReturn $K{=}3$ | 0.077 | 0.692 | 0.301 | 0.459 |
| WTI AbsReturn $K{=}3$ | 0.077 | 0.923 | **0.454** | 0.308 |
| GOLD AbsReturn $K{=}3$ | **0.875** | 1.000 | 0.334 | 0.360 |

**SPY AbsReturn $K = 3$** shows partial recovery: sim-trained event coverage
(0.077) is substantially below the real-trained baseline (0.692), indicating
that the synthetic model misses most SPY proxy events but the coherence gap
(0.301 vs 0.459) is smaller. The AbsReturn feature has $\hat\rho_1(|r|)
\approx 0.37$, which lies inside the model class, so the zero-alarm pathology
is resolved; the remaining gap reflects residual distributional mismatch.

**WTI AbsReturn $K = 3$** exhibits a diagnostic reversal: sim-trained
coherence (0.454) *exceeds* the real-trained baseline (0.308), while
sim-trained event coverage (0.077) is low. This pattern isolates the source
of the sim-to-real gap: the segmentation quality is already comparable to the
real-trained model, but the detector threshold is miscalibrated for the
real-data scale. The fix is threshold tuning (§6.7), not model improvement.

**GOLD AbsReturn $K = 3$** is the existence proof. The sim-trained detector
achieves `event_coverage = 0.875` (7 of 8 proxy events matched) and
`coherence_score = 0.334`, against real-trained baselines of 1.000 and 0.360
respectively — gaps of 12.5 % and 7.3 %. The calibration verifier
`within_tolerance: true` on all active fields for this asset-feature pair;
it is the only configuration in the sweep for which the sim-to-real transfer
argument can be made without qualification.

### 6.6.4 Why feature choice matters more than calibration strategy

The sweep reveals a clear hierarchy: whether the sim-to-real transfer works
at all is determined primarily by *which feature family is chosen*, not by
which calibration strategy ($\mathcal{K}_{\text{sum}}$ vs.
$\mathcal{K}_{\text{EM}}$) is used. The necessary condition is that
$\hat\rho_1(\phi(P))$ lies inside the model class (positive and significant
for AbsReturn, ≈0 for SquaredReturn, negative for LogReturn on equity ETFs).
Once that condition is met, $\mathcal{K}_{\text{EM}}$ produces a higher
maximum-likelihood match than $\mathcal{K}_{\text{sum}}$ and reduces the
scale-consistency residual, but the qualitative pass/fail verdict is already
set by the feature.

---

## 6.7 Joint optimisation

The joint model-and-detector grid search sweeps both the model architecture
(number of regimes $K$ and feature family $\phi$) and the detector
hyperparameters $(\tau, N_{\text{req}}, C)$ in a single combined grid.

### 6.7.1 Grid definition

**`ModelGrid` (default):**
- $K \in \{2, 3\}$
- feature family $\in$ \{LogReturn, AbsReturn, SquaredReturn, RollingVol$(5)$,
  RollingVol$(20)$\}
- Total model configurations: $2 \times 5 = 10$

**`ParamGrid` (real HardSwitch):**
- thresholds $\tau \in \{0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80\}$ — 8 values
- persistence $N_{\text{req}} \in \{1, 2, 3, 5\}$ — 4 values
- cooldown $C \in \{0, 3, 5, 10\}$ — 4 values
- Total detector configurations: $8 \times 4 \times 4 = 128$

**Joint grid size:** $10 \times 128 = 1\,280$ candidate configurations,
evaluated on the registered real experiment `real_spy_daily_hard_switch` via
the `optimize --model` subcommand.

### 6.7.2 Scoring and ranking

Each candidate configuration runs the full E2E pipeline (training, online
detection, Route A + B evaluation) and is scored by the combined metric
from §6.3.2:

$$
\text{score} = \tfrac{1}{3}\bigl(\text{event\_coverage} + \text{alarm\_relevance} + \text{coherence\_score}\bigr).
$$

Results are sorted descending; the top-10 configurations are written to
`search_report.json` with per-row fields `k_regimes`, `feature_family`,
`threshold`, `persistence_required`, `cooldown`, `event_coverage`,
`alarm_relevance`, `coherence_score`, and `combined_score`.

### 6.7.3 Winning configuration

The joint optimum across the 1 280-point grid is:

$$
K = 3,\quad \phi = \text{AbsReturn},\quad
\tau_{\text{conf}} = 0.55,\quad N_{\text{req}} = 2,\quad C = 5.
$$

This configuration is used in all four sim-to-real registered experiments
(§6.6.1) and provides the canonical detector parameters for the thesis. The
selection of $K = 3$ over $K = 2$ reflects the additional expressiveness
available from a third regime (calm / elevated / crisis), which produces
finer posterior differentiation and higher alarm relevance. The selection of
AbsReturn over LogReturn is consistent with the sim-to-real analysis of §6.6.4:
absolute returns have positive autocorrelation that lies inside the model
class, making them the only feature family for which both synthetic and real
evaluation can be performed without the degenerate-$\pi$ pathology.

---

## 6.8 Discussion

### 6.8.1 What the results establish

The four empirical layers answer four distinct questions:

1. *Can the detectors recover changepoints at all?* — Yes, confirmed by
   synthetic experiments with ground truth (§6.4).
2. *Does the detector align with financially meaningful events on real data?*
   — Partially, with the HardSwitch on SPY and GOLD showing meaningful
   event coverage and the Surprise detector on WTI covering supply shocks
   (§6.5).
3. *Can a detector trained entirely on synthetic data work on real data?* —
   Yes, for the right feature choice. GOLD AbsReturn/$K = 3$ achieves 87.5 %
   of real-trained event coverage with sim-trained parameters (§6.6.3).
4. *What is the optimal configuration?* — $K = 3$, AbsReturn,
   HardSwitch$(0.55, 2, 5)$ (§6.7.3).

### 6.8.2 The boundary of sim-to-real transferability

The sweep establishes a clear boundary: sim-to-real transfer is viable when
the feature's empirical autocorrelation structure is inside the expressive
range of the Gaussian MSM, and not viable otherwise. This is a testable
condition: the calibration verifier $\mathcal{V}$ of §5.6 flags failures by
reporting `within_tolerance: false` on the $\hat\rho_1$ field, which is
precisely the signal to switch features before running the full experiment.

### 6.8.3 WTI as a partially resolved case

WTI AbsReturn/$K = 3$ occupies a middle ground: the coherence score
exceeds the real-trained baseline (better segmentation quality) but
event coverage remains low (few proxy events matched). This combination
identifies threshold calibration as the remaining bottleneck — the model
segments the stream correctly but the threshold suppresses most alarms.
A targeted single-parameter search over $\tau$ in $[0.30, 0.80]$, holding
$K$ and $\phi$ fixed, would be the appropriate next step rather than
re-calibrating the generator.

### 6.8.4 Limitations

The evaluation rests on three constraints that bound the conclusions:

1. **Proxy events are sparse.** With 8–13 events per asset catalogue, the
   Route A metrics are coarse: a change of one event matched shifts
   `event_coverage` by 8–12 percentage points. Conclusions based on small
   absolute differences in `event_coverage` should be interpreted with caution.

2. **Frozen parameters.** The `FrozenModel` assumption (§4.2 of Chapter 4)
   means that the detector's regime distributions remain fixed at the
   training-time estimates. If the true marginal distribution of returns
   shifts after the training cutoff (e.g. a structural change in VIX levels),
   the frozen model will degrade silently.

3. **Single real series per asset.** All experiments use a single
   chronological split. There is no cross-validation or multiple disjoint
   test windows, so the test results reflect the specific macro environment
   of 2025–present and may not generalise to other market regimes.

---

## 6.9 Theory–code correspondence

The table below maps each section of this chapter to the Rust source files
and specific types or functions that implement the described concept.
All paths are relative to the workspace root.

| Section | Concept | File | Key types / functions |
|---|---|---|---|
| §6.1 | Raw observations and price series | `src/data/mod.rs` | `Observation`, `CleanSeries` |
| §6.1 | DuckDB commodity cache (data source) | `src/cache/mod.rs` | `CommodityCache` |
| §6.1 | Asset endpoint identifiers and intervals | `src/alphavantage/commodity.rs` | `CommodityEndpoint`, `Interval` |
| §6.1 | Data loading, date filtering, gap checking | `src/experiments/real_backend.rs` | `RealBackend::load_clean_series()`, `RealBackend::resolve_data()` |
| §6.1 | RTH session filter (09:30–16:00 ET) | `src/data/session.rs` | `filter_rth()`, `is_rth_bar()`, `SessionBoundary` |
| §6.1.1 | Chronological 70/15/15 partition | `src/data/split.rs` | `SplitConfig`, `PartitionedSeries`, `PartitionedSeries::from_series()` |
| §6.2 | Feature operator definitions ($\phi$) | `src/features/family.rs` | `FeatureFamily::{LogReturn, AbsReturn, SquaredReturn, RollingVol, StandardizedReturn}` |
| §6.2 | Z-score scaler (train-only fit, frozen apply) | `src/features/scaler.rs` | `ScalingPolicy::{ZScore, RobustZScore, None}` |
| §6.2 | Causal feature pipeline (warmup, session reset) | `src/features/stream.rs` | `FeatureStream`, `FeatureConfig`, `FeatureStreamMeta` |
| §6.3.1 | Tolerance-window matching ($W$ = 20) | `src/benchmark/matching.rs` | `MatchConfig`, `EventMatcher`, `MatchResult`, `EventMatcher::match_events()` |
| §6.3.1 | Full synthetic metric suite (precision, recall, FAR, delay) | `src/benchmark/metrics.rs` | `MetricSuite`, `MetricSuite::from_match()` |
| §6.3.1 | Serialised synthetic evaluation result | `src/experiments/result.rs` | `EvaluationSummary::Synthetic { coverage, precision_like, … }` |
| §6.3.2 | Route A — proxy-event alignment | `src/real_eval/route_a.rs` | `ProxyEvent`, `RouteAConfig`, `PointMatchPolicy`, `evaluate_proxy_events()`, `ProxyEventEvaluationResult` |
| §6.3.2 | Route B — segmentation coherence score | `src/real_eval/route_b.rs` | `RouteBConfig`, `evaluate_segmentation()`, `SegmentationGlobalSummary`, `AdjacentSegmentContrast` |
| §6.3.2 | Real eval orchestration (Route A + B) | `src/real_eval/report.rs` | `evaluate_real_data()`, `RealEvalMeta`, `RealEvalResult` |
| §6.3.2 | Serialised real evaluation result | `src/experiments/result.rs` | `EvaluationSummary::Real { event_coverage, alarm_relevance, segmentation_coherence }` |
| §6.4 | Synthetic experiment configurations | `src/experiments/registry.rs` | registered entries for `hard_switch`, `posterior_transition`, `surprise`, `surprise_ema`, `squared_return_surprise`, `hard_switch_shock`, `hard_switch_multi_start`, `hard_switch_frozen` |
| §6.4 | Synthetic experiment backend (simulate → train → detect) | `src/experiments/synthetic_backend.rs` | `SyntheticBackend` |
| §6.5 | Real experiment configurations | `src/experiments/registry.rs` | registered entries for `real_spy_daily_hard_switch`, `real_wti_daily_surprise`, `real_spy_intraday_hard_switch` |
| §6.5 | Real experiment backend (load → feature → train → detect → evaluate) | `src/experiments/real_backend.rs` | `RealBackend`, `RealBackend::evaluate_real()` |
| §6.5–§6.6 | Shared EM training and online detection stages | `src/experiments/shared.rs` | `train_or_load_model_shared()`, `run_online_shared()` |
| §6.6 | Sim-to-real experiment configurations | `src/experiments/registry.rs` | registered entries for `simreal_spy_daily_hard_switch`, `simreal_spy_daily_abs_return_k3`, `simreal_wti_daily_abs_return_k3`, `simreal_gold_daily_abs_return_k3` |
| §6.6 | Sim-to-real experiment backend ($\mathcal{K}_{\text{EM}}$ → synthetic train → real test) | `src/experiments/sim_to_real_backend.rs` | `SimToRealBackend` |
| §6.6.2–§6.6.3 | Initial distribution policy (counterexample fix) | `src/calibration/mapping.rs` | `PiPolicy::{Stationary, Fitted}` |
| §6.7 | Detector hyperparameter grid | `src/experiments/search.rs` | `ParamGrid`, `ParamGrid::for_real_hard_switch()`, `ParamGrid::for_real_surprise()` |
| §6.7 | Model architecture grid ($K$ × feature family) | `src/experiments/search.rs` | `ModelGrid`, `ModelGrid::default()` |
| §6.7 | Joint grid search execution and scoring | `src/experiments/search.rs` | `optimize_full()`, `optimize()`, `grid_search()`, `SearchPoint`, `OptimizeResult` |
| §6.7 | Experiment runner (E2E pipeline orchestration) | `src/experiments/runner.rs` | `ExperimentBackend` trait, `ExperimentConfig` |
| §6.7 | Experiment configuration schema | `src/experiments/config.rs` | `ExperimentConfig`, `DataConfig`, `EvaluationConfig`, `DetectorType` |
