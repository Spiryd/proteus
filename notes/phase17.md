# Phase 17 — Build the Synthetic-to-Real Calibration Layer

## Goal

In this phase, create a principled bridge between:

- **synthetic regime-generating processes**, and
- **empirical market behavior observed in real financial data**.

Up to now, your project has two partially separate worlds:

## World A — Synthetic experiments
These are useful because:
- true changepoints are known,
- regime structure is controlled,
- benchmarking is rigorous.

## World B — Real market data
These are useful because:
- they reflect actual financial behavior,
- they give the thesis practical relevance,
- they test whether the detector is meaningful outside simulation.

The problem is that without calibration, these two worlds can become disconnected.

If synthetic data are too artificial, then:
- detector success on synthetic streams says little about real markets,
- parameter choices become arbitrary,
- transition dynamics and volatility patterns may be unrealistic,
- the thesis loses coherence.

So the purpose of this phase is:

\[
\boxed{
\text{Design synthetic regimes whose statistical structure is anchored in empirical financial data.}
}
\]

This makes the synthetic benchmark more realistic and the real-data study more interpretable.

---

## 1. Why this phase is necessary

Your thesis description explicitly emphasizes two linked ideas:

1. the observed time series should be modeled by a Markov Switching framework,
2. the model parameters should be adjusted at least partly using statistics derived from generated synthetic data and empirical analysis.

That means the thesis should not present synthetic experiments as purely arbitrary toy examples.

Instead, the synthetic generator should be informed by empirical market statistics such as:
- typical return scales,
- volatility levels,
- persistence of calm vs turbulent regimes,
- jump or shock frequency,
- regime duration.

This phase formalizes that idea.

---

## 2. What “synthetic-to-real calibration” means

Synthetic-to-real calibration means constructing synthetic regime-switching streams so that selected summary properties of the generated process resemble corresponding summary properties of real market data.

This does **not** mean trying to perfectly reproduce the market.

Instead, it means selecting a set of empirical target statistics and calibrating the synthetic generator so that it approximately matches them.

Formally, let:

- \(X^{\text{real}}\) denote the empirical market series,
- \(X^{\text{syn}}(\vartheta)\) denote a synthetic generator with parameter set \(\vartheta\).

Then calibration means choosing \(\vartheta\) such that a set of summary functionals

\[
T_1, T_2, \dots, T_m
\]

satisfy approximately:

\[
T_\ell(X^{\text{syn}}(\vartheta)) \approx T_\ell(X^{\text{real}})
\qquad \text{for } \ell=1,\dots,m.
\]

These \(T_\ell\) are empirical market statistics or stylized facts chosen to be relevant to the thesis.

---

## 3. What should be calibrated

This phase should not attempt to calibrate everything.

Instead, it should focus on a carefully chosen family of statistics that matter most for:

- the Markov Switching structure,
- the online detector behavior,
- financial interpretability.

The most important calibration targets are:

## 3.1 Mean return level
For return-based observation streams, calibrate the average level:

\[
\mu^{\text{real}} = \frac{1}{T}\sum_{t=1}^T r_t.
\]

In many financial contexts, the mean is small relative to volatility, but it still belongs in the calibration framework.

---

## 3.2 Variance / volatility scale
For returns or volatility-based features, calibrate dispersion:

\[
\sigma_{\text{real}}^2 = \frac{1}{T}\sum_{t=1}^T (r_t-\bar r)^2.
\]

Or if the observation process is volatility-like, calibrate empirical volatility levels directly.

This is usually one of the most important targets.

---

## 3.3 Regime persistence
Because your detector is built on a Markov Switching model, the synthetic generator must reflect realistic persistence.

Persistence is encoded by diagonal transition probabilities:

\[
p_{jj} = \Pr(S_t=j \mid S_{t-1}=j).
\]

Equivalent summary:

\[
\mathbb{E}[\text{duration in regime } j] = \frac{1}{1-p_{jj}}.
\]

You should calibrate regime durations or transition persistence to resemble observed market regime persistence.

---

## 3.4 Frequency of changes
Synthetic streams should exhibit a realistic rate of change events.

This can be summarized by:
- expected number of changepoints per horizon,
- average regime duration,
- expected transition frequency.

This matters directly for online detection difficulty.

---

## 3.5 Jump / shock frequency
Financial series often contain abrupt large moves that are not well described by simple Gaussian noise.

Even if your core model remains Gaussian, your synthetic generator should at least allow control over the frequency of unusually large moves.

This can be summarized through:
- tail-event frequency,
- large-return exceedance rates,
- optional jump contamination rate.

---

## 3.6 Volatility clustering or turbulence structure
If you use volatility-based features, synthetic data should reflect the empirical tendency for turbulence to cluster in time.

In the Markov Switching framework, this can often be represented via:
- persistent high-volatility regimes,
- persistent low-volatility regimes,
- asymmetric regime durations.

This is one of the most natural strengths of your MS framework.

---

## 4. Calibration is feature-dependent

A crucial point in this phase is that calibration must be tied to the **actual observation design** from Phase 16.

If your detector runs on:

- log returns,
- absolute returns,
- rolling volatility,

then the calibration statistics should be computed on the same observation family.

That means you should not calibrate synthetic raw-price properties if the model will later observe returns.

Formally, if your feature transformation is

\[
y_t = \Phi(\text{market data up to } t),
\]

then the calibration targets should be defined on the empirical sequence

\[
y_1^{\text{real}}, \dots, y_T^{\text{real}}
\]

and the synthetic generator should be designed to reproduce corresponding synthetic observation statistics.

This makes the calibration layer consistent with the rest of the thesis.

---

## 5. Two levels of calibration

It is useful to distinguish two levels.

## Level 1 — Marginal calibration
Match broad first- and second-order properties such as:
- mean,
- variance,
- tail frequency,
- empirical quantiles.

This is the simplest and most practical first step.

---

## Level 2 — Regime-structural calibration
Match regime-related properties such as:
- calm-regime volatility,
- turbulent-regime volatility,
- transition persistence,
- expected regime durations,
- frequency of detected switches.

This is more specific to your Markov Switching framework and usually more valuable for the thesis.

For your project, you should aim for **both**, but start with Level 1 and then add regime structure.

---

## 6. What should the calibrated synthetic generator look like

At this stage, the synthetic generator should become more structured than a generic toy regime simulator.

A good first calibrated generator might specify:

- \(K\) regimes,
- regime-specific means \(\mu_j\),
- regime-specific variances \(\sigma_j^2\),
- transition matrix \(P\),
- optional jump contamination settings,
- scenario length,
- observation family tied to the same feature logic used later on real data.

Then calibration means choosing these values from empirical statistics rather than arbitrary guesses.

---

## 7. Recommended calibration workflow

A clean Phase 17 workflow is:

## Step 1 — choose an empirical calibration dataset
Select a specific real-data subset, for example:
- daily SPY training period,
- daily commodity training period,
- intraday QQQ training period.

Calibration must be tied to a specific data regime and feature definition.

---

## Step 2 — choose the observation family
For example:
- returns,
- absolute returns,
- rolling volatility.

The calibration targets depend on this choice.

---

## Step 3 — compute empirical summary statistics
Compute the statistics you want to match:
- mean,
- variance,
- quantiles,
- exceedance frequency,
- persistence proxies,
- change frequency proxies.

---

## Step 4 — map empirical statistics to synthetic generator parameters
Define how empirical quantities determine:
- \(\mu_j\),
- \(\sigma_j^2\),
- \(P\),
- regime-duration profile,
- optional shock contamination.

This mapping is the heart of the phase.

---

## Step 5 — generate synthetic calibrated scenarios
Use the calibrated parameters to generate synthetic streams.

---

## Step 6 — verify calibration quality
Compare synthetic and real summary statistics.

If the synthetic process is too far from the empirical targets, adjust the mapping and repeat.

This creates a controlled synthetic regime family grounded in market behavior.

---

## 8. The calibration problem as a formal mapping

This phase should explicitly define a calibration operator:

\[
\mathcal{K}: \text{empirical statistics} \longrightarrow \text{synthetic generator parameters}.
\]

That is,

\[
\vartheta = \mathcal{K}(T_1^{\text{real}}, \dots, T_m^{\text{real}}),
\]

where \(\vartheta\) contains:
- regime means,
- regime variances,
- transition probabilities,
- optional additional generator settings.

You do not need this mapping to be theoretically optimal.  
But you do need it to be:
- explicit,
- reproducible,
- documented.

This is what makes the synthetic experiments defensible.

---

## 9. Example calibration targets for returns

Suppose your observation family is log returns.

Then useful targets include:

### Mean
\[
\bar r = \frac{1}{T}\sum_{t=1}^T r_t
\]

### Volatility
\[
s_r^2 = \frac{1}{T}\sum_{t=1}^T (r_t-\bar r)^2
\]

### Tail quantiles
\[
q_{0.01}, q_{0.99}
\]

### Large-move frequency
\[
\Pr(|r_t| > c)
\]

for some threshold \(c\).

### Serial dependence proxies
If relevant, include:
- autocorrelation of \(|r_t|\),
- autocorrelation of \(r_t^2\).

This is especially useful if your calibrated synthetic data should reflect volatility clustering.

---

## 10. Example calibration targets for volatility features

If your observation family is a rolling volatility proxy \(v_t^{(w)}\), useful targets include:

- mean volatility level,
- median volatility,
- upper quantiles,
- persistence of high-volatility episodes,
- duration of calm vs turbulent periods.

This is often very relevant for financial-regime detection because volatility regimes are one of the clearest real-market regime phenomena.

---

## 11. Calibrating regime durations

Since your model explicitly contains hidden regimes, one particularly important target is regime duration.

If you identify high- and low-volatility market episodes empirically, you can estimate rough duration statistics such as:

- mean calm-period duration,
- mean turbulent-period duration.

Then map those durations to transition probabilities using:

\[
p_{jj} = 1 - \frac{1}{d_j},
\]

where \(d_j\) is the target expected duration for regime \(j\).

This gives a very direct synthetic-to-real link:
- empirical persistence \(\to\) transition matrix persistence.

This is one of the strongest calibration ideas in the entire thesis because it directly ties real market behavior to the MS backbone.

---

## 12. Calibrating regime-specific variance structure

A very natural financial calibration strategy is:

- identify a calm volatility level,
- identify a turbulent volatility level,
- use those as regime-specific variances.

For example, define:
- low-volatility regime variance from lower empirical volatility quantiles,
- high-volatility regime variance from upper empirical volatility quantiles.

This creates synthetic data with realistic heterogeneity between regimes rather than arbitrary variance ratios.

---

## 13. Optional contamination / jump calibration

If you want synthetic streams to better resemble real financial noise, add a contamination layer.

For example:
- with probability \(\lambda\), inject a large shock,
- otherwise draw from the regime-Gaussian observation law.

This can be calibrated from empirical large-move rates.

This is optional, but useful if your synthetic Gaussian data are otherwise too smooth compared with real markets.

For the thesis, even if you do not use this in the first core experiment set, it is valuable as a robustness extension.

---

## 14. Calibrated scenario families

This phase should end with a family of calibrated synthetic scenario types, not just one generator.

A good set is:

## Scenario A — Calm vs turbulent calibrated by volatility levels
Two-regime variance-switching design anchored to empirical volatility quantiles.

## Scenario B — Persistent market-state design
Two-regime or three-regime design calibrated to empirical episode durations.

## Scenario C — Shock-contaminated calibrated design
Same as A or B, but with empirically calibrated jump contamination.

## Scenario D — Asset-specific calibration
Separate synthetic calibration for:
- commodity-like behavior,
- daily SPY/QQQ behavior,
- intraday ETF behavior.

This makes the synthetic study much more relevant to the real-data part of the thesis.

---

## 15. Why this phase improves the thesis scientifically

Without this phase, your thesis risks saying:

- “Here are some synthetic results,”
- and separately,
- “here are some real-data results.”

With this phase, you can instead say:

- “Synthetic scenarios were designed to reflect empirically observed market characteristics, so the synthetic benchmark is a controlled approximation of the real data regimes we later study.”

That is a much stronger scientific story.

---

## 16. Step-by-step guide for Phase 17

## Step 1 — Define calibration datasets

For each asset / frequency family, choose the empirical period used to estimate calibration statistics.

Examples:
- daily SPY training segment,
- daily commodity training segment,
- intraday QQQ training segment.

### Deliverable
A calibration-dataset policy.

### Code changes
You should add:
- configuration for calibration source datasets,
- explicit tagging of which partition is used for calibration.

---

## Step 2 — Define calibration statistics per observation family

For each feature family from Phase 16, define the statistics to extract.

For returns:
- mean,
- variance,
- quantiles,
- tail-event frequency.

For volatility:
- mean/median volatility,
- high-volatility quantiles,
- episode duration statistics.

### Deliverable
A family-specific calibration-statistics definition.

### Code changes
Add:
- empirical-statistics utilities,
- feature-family-aware calibration summaries,
- exportable summary objects.

---

## Step 3 — Define the synthetic-generator parameterization

Specify what synthetic generator parameters are calibrated.

At minimum:
- \(\mu_j\),
- \(\sigma_j^2\),
- \(P\),
- scenario length.

Optionally:
- jump contamination rate,
- outlier scale.

### Deliverable
A formal synthetic-generator parameter schema.

### Code changes
Add:
- synthetic scenario config structs,
- generator parameter structs,
- serialization support for calibrated configs.

---

## Step 4 — Define the calibration mapping

Write down how empirical statistics map to synthetic parameters.

Examples:
- empirical low/high volatility quantiles \(\to\) regime variances,
- empirical episode duration \(\to\) diagonal transition probabilities,
- empirical mean return \(\to\) regime means or baseline mean.

### Deliverable
A reproducible calibration mapping.

### Code changes
Add:
- a calibration module,
- mapping functions from empirical summaries to generator configs,
- explicit calibration reports.

---

## Step 5 — Add synthetic verification reports

After calibration, generate synthetic streams and compare synthetic summaries to empirical targets.

### Deliverable
A calibration-quality verification layer.

### Code changes
Add:
- synthetic-vs-empirical comparison utilities,
- summary reports,
- optional export of calibration diagnostics.

---

## Step 6 — Package calibrated scenario families for experiments

Make calibrated synthetic scenarios easy to reuse in benchmark runs.

### Deliverable
Reusable calibrated scenario definitions.

### Code changes
Add:
- named scenario presets,
- experiment-ready scenario objects,
- configuration-driven scenario generation.

---

## 17. Suggested architecture for this phase

By the end of Phase 17, the project should conceptually have:

## Empirical summary layer
Computes calibration statistics from real training data.

## Calibration layer
Maps empirical summaries to synthetic generator parameters.

## Synthetic scenario layer
Uses calibrated parameters to generate benchmark streams.

## Calibration verification layer
Compares synthetic summary statistics against empirical targets.

This keeps the design clean and reproducible.

---

## 18. What should be exported from this phase

This phase should produce artifacts that can be used both in experiments and in the thesis text.

At minimum, export:

- empirical summary statistics by asset and feature family,
- calibrated synthetic-generator parameters,
- synthetic-vs-empirical comparison summaries,
- scenario labels and metadata.

These artifacts are useful for:
- experiment reproducibility,
- tables in the thesis,
- methodological explanation.

---

## 19. Testing requirements for Phase 17

This phase should be tested carefully.

You should test at least:

## Empirical summary tests
- summary-statistics computations are correct,
- feature-family-specific summaries use the right input stream.

## Calibration mapping tests
- empirical duration targets map correctly to transition persistence,
- empirical variance targets map correctly to regime variances,
- calibrated configs are internally valid.

## Generator validity tests
- calibrated synthetic scenarios produce valid parameter objects,
- transition matrices remain row-stochastic,
- variances remain positive.

## Calibration verification tests
- synthetic summaries move in the correct direction when empirical targets change,
- asset-specific calibrations produce meaningfully different synthetic setups.

### Deliverable
A reliable synthetic-to-real calibration layer.

### Code changes
Add:
- unit tests for calibration formulas,
- validation tests for calibrated configs,
- integration tests from empirical dataset to calibrated synthetic scenario.

---

## 20. Common conceptual mistakes to avoid

### Mistake 1 — Treating synthetic data as arbitrary toys
This weakens the connection between simulation and real-market analysis.

### Mistake 2 — Calibrating on raw prices when the detector uses returns or volatility
Calibration must happen on the actual modeled observation process.

### Mistake 3 — Trying to perfectly reproduce the full market distribution
Calibration should be targeted and interpretable, not overengineered.

### Mistake 4 — Hiding the calibration mapping
The mapping from empirical summaries to synthetic parameters must be explicit and reproducible.

### Mistake 5 — Using future test data to calibrate synthetic scenarios
Calibration should use training or designated calibration periods only.

### Mistake 6 — Calibrating only marginal moments and ignoring persistence
For an MS model, regime duration and persistence are central.

---

## 21. Deliverables of Phase 17

By the end of this phase, you should have:

### Mathematical / methodological deliverables
- a formal definition of synthetic-to-real calibration,
- a chosen set of empirical calibration statistics,
- a documented mapping from empirical statistics to synthetic-generator parameters,
- calibrated synthetic scenario families tied to real financial observation processes,
- a verification procedure comparing synthetic and empirical summaries.

### Architectural deliverables
- a dedicated calibration layer between real-data features and synthetic scenario generation,
- clear separation between:
  - empirical summary extraction,
  - calibration mapping,
  - synthetic generation,
  - calibration verification.

### Code-structure deliverables
You should add or revise, where appropriate:

- a dedicated `calibration` module,
- empirical summary-statistics objects,
- feature-aware calibration summary utilities,
- synthetic scenario config types,
- generator parameter structs,
- calibration mapping functions,
- calibration verification utilities,
- exportable calibration reports,
- named calibrated scenario presets,
- tests for summary extraction, calibration mapping, and scenario validity.

### Experimental deliverables
- synthetic scenarios that are no longer arbitrary,
- a principled bridge between synthetic benchmarking and real market analysis,
- a much stronger thesis narrative linking simulation and real-data experiments.

---

## 22. Minimal final summary

Phase 17 is the step that connects your synthetic benchmark world to your real market world.

The central idea is:

\[
\boxed{
\text{Synthetic regime-switching scenarios should be calibrated using empirical market statistics.}
}
\]

This phase should end with:
- empirical financial summary extraction,
- calibrated synthetic scenario generation,
- verification that synthetic streams roughly reflect the intended real-data structure,
- and a reproducible calibration layer that makes your synthetic experiments much more defensible in the thesis.