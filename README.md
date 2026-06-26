# Forecast–Price Feedback Loop

This repository studies a problem that appears whenever pricing systems learn from the data they create.

In a dynamic pricing system, a demand forecast helps set the price. That price then changes the demand the firm observes. The observed demand is added back into the training data, and the forecaster is retrained. Over time, the system can begin to learn from its own distorted evidence. A pricing rule may eventually settle down and look stable, even though it has converged to the wrong elasticity and the wrong price.

The project builds a simulator for studying this forecast–price feedback loop. The simulator makes the true demand curve known, which means we can measure estimation bias directly. It also lets us compare different ways of correcting the loop: changing the prices to create better variation, correcting the estimate with instrumental-variable or control-function methods, and using decision-focused approaches that account for the pricing decision itself.

The synthetic worlds are calibrated to 26 grocery categories from the Dominick's Finer Foods scanner data. The real data is used to set realistic elasticities, shock behavior, and instrument strength. The feedback loop itself is studied in simulation, where the ground truth remains known.

The main calibrated result is simple:

**When endogeneity is present, instrumental-variable / control-function correction is the strongest fix. It beats the naive baseline in 23 of 24 grocery categories at moderate-to-high endogeneity. But when endogeneity is close to zero, where historical grocery markets appear to sit, no correction has a clear advantage and debiasing can sometimes hurt.**

The practical message is therefore forward-looking. In older, pre-algorithmic grocery settings, the feedback loop appears weak. As pricing becomes more automated and self-referential, correction becomes much more important.

**Author:** Dr. Jose Mendoza, New York University · jose.mendoza@nyu.edu

## Installation

The package requires Python 3.10 or later.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[nn,viz,calib]"
```

Optional dependency groups:

- **core** — NumPy, pandas, scikit-learn, Plotly, and pyarrow. The core package does not require PyTorch.
- **`nn`** — PyTorch support for the feedforward MLP and GRU forecasters.
- **`viz`** — static figure export through kaleido.
- **`calib`** — tools for the Dominick's calibration pipeline, including pyreadstat and linearmodels.
- **`dev`** — testing and linting tools.

To reproduce the main calibrated result, the core package plus `calib` is enough.

## Reproducing the main result

The headline result can be reproduced with the command-line pipeline below. Run these commands from the repository root.

```bash
pip install -e ".[calib]"

# 1. Obtain the Dominick's movement CSVs and unzip them into data/raw/dominicks/
#    See data/README.md for the Kilts Center download instructions.

# 2. Calibrate the categories -> data/calibrated_markets.csv
#    This estimates OLS and IV elasticity, first-stage R^2, shock sigma/rho,
#    and implied lambda for each category.
python -m fploop.calibration.run \
  --categories ANA,BAT,BER,BJC,CER,CHE,CIG,COO,CRA,CSO,DID,FEC,FRD,FRJ,FSF,GRO,LND,OAT,PTW,SDR,SHA,SOA,TBR,TNA,TPA,TTI \
  --data data/raw/dominicks --out data/calibrated_markets.csv

# 3. Build the calibrated-worlds config.
#    This maps each category to a WorldConfig.
python scripts/build_worlds_config.py        # -> data/calibrated_worlds_config.csv

# 4. Run the intervention comparison across the calibrated worlds.
#    The script prints a verdict for each category-lambda cell.
python scripts/run_calibrated.py \
  --arms greedy,controlled_variance,twosls \
  --lambda-grid 0,0.4,0.8 --seeds 0 \
  --out results/sweep.csv
```

Step 4 reports the mean regret for each arm in each `(category, λ)` cell and identifies the winning family. The expected pattern is that causal correction wins most cells once λ reaches 0.4 or higher. At λ = 0, there is no clear winner.

The currently supported headline comparison is **causal correction versus the naive baseline**. The exploration arm is still being repaired, so it should not yet be used as the basis for strong claims about exploration.

## Usage

### Calibrate grocery categories

```bash
python -m fploop.calibration.run \
  --categories CSO,BER,FRJ \
  --data data/raw/dominicks \
  --out data/calibrated_markets.csv
```

This estimates real-market coordinates from the Dominick's data. See [data/README.md](data/README.md) for download instructions.

### Build calibrated worlds

```bash
python scripts/build_worlds_config.py
```

This converts the calibration output into simulator parameters and writes:

```text
data/calibrated_worlds_config.csv
```

### Run intervention comparisons

```bash
python scripts/run_calibrated.py \
  --arms greedy,controlled_variance,twosls \
  --lambda-grid 0,0.4,0.8 \
  --seeds 0,1,2
```

This runs the selected intervention arms across calibrated categories, λ values, and seeds. It writes a tidy outcomes file and prints a winner verdict.

### Run synthetic regime sweeps

```bash
python -m fploop.sweep \
  --scenario base \
  --forecaster gbt \
  --out sweeps/base_gbt
```

This runs offline sweeps over the endogeneity and cost-variation grid. Results are cached as parquet files.

## Project structure

```text
fploop/              the toolkit
  types.py             data containers for configs, states, observations, histories, and results
  loop.py              run_simulation, which turns the closed loop
  oracle.py            full-information benchmark
  features.py          design-matrix construction with optional controls
  generators/          ground-truth demand worlds
  forecasters/         demand models retrained on history
  policies/            baseline, exploration, causal, and decision-focused policies
  arms.py              arm registry
  metrics/             regret, residual elasticity bias, CVaR, and stability
  sweep.py             offline sweep engine and winner aggregation
  crossover.py         crossover-map and vulnerability-atlas figures
  calibration/         Dominick's ingest and IV estimation
    worlds.py            calibration CSV -> WorldConfig objects
  reproduce.py         one-command figure reproduction

scripts/
  build_worlds_config.py   calibration -> calibrated_worlds_config.csv
  run_calibrated.py        calibrated-world intervention runner

data/                calibration inputs and outputs
docs/                project documentation
tests/               pytest suite
```

## How the simulator works

The simulator uses a constant-elasticity demand world. An unobserved demand shock affects both demand and the effective transaction price in the same period. This creates the endogeneity problem: the price is no longer cleanly independent of the demand shock.

A naive forecaster treats the observed price and quantity history as if it were ordinary training data. It estimates demand, the pricing policy sets a price from that estimate, and the new demand observation is added back into the data. If the estimate is biased, the pricing decision can generate data that confirms the bias rather than correcting it. When inventory or capacity limits bind, observed demand can be censored, which makes the problem worse.

The interventions are grouped into three families:

1. **Pre-estimation interventions** vary the price to restore identification. Examples include controlled-variance pricing, epsilon-greedy pricing, and SPSA-style perturbations.

2. **In-estimation interventions** correct the elasticity estimate directly. Examples include instrumental variables, control functions, double machine learning, and censoring-aware estimation.

3. **Post-estimation / decision-focused interventions** change how the estimate is used in the pricing decision. Examples include distributionally robust optimization and smoothed retraining.

The calibrated grocery categories provide realistic starting points for elasticity, shock variance, shock persistence, and instrument strength. The simulator then sweeps λ upward from the historical anchor to study what happens as pricing becomes more algorithmic and the feedback loop becomes stronger.

## Calibration and interpretation

The Dominick's calibration should be read carefully.

The data comes from a single Chicago-area grocery chain in the 1989–1997 period. Pricing was largely pre-algorithmic, driven by category management and promotion calendars. As a result, the data is useful for calibrating demand primitives, but it cannot by itself show a modern forecast-price feedback loop.

In the calibration, most categories sit near λ = 0. That does not mean the feedback loop is impossible. It means that, in this historical grocery setting, the loop was not strongly present. The simulator uses those calibrated categories as the starting point and then moves λ upward to ask a forward-looking question:

**What happens when a market with realistic grocery demand moves into a more algorithmic pricing regime?**

That is where the correction families begin to separate.

## Known limitations

- **The exploration (pre-estimation) arm helps only at low endogeneity.** Its dispersion is annealed over time and truncated each period so prices stay bounded; it restores identification when the problem is under-exploration (low λ, where it beats the baseline) but cannot remove endogeneity bias (high λ), where it is modestly worse than the baseline — the expected economics, since added price variation is still contaminated by the endogeneity that only an instrument can purge. Both the baseline and exploration overprice heavily at high λ once the biased estimate hits the markup ceiling; that is baseline behavior, and the headline causal-vs-baseline comparison is unaffected.

- **The instrument is valid by construction.** In the simulator, the cost shifter is exogenous by design. The result should be interpreted as: when a valid instrument exists, causal correction is highly effective under endogeneity. It does not imply that valid instruments are always available in practice.

- **The calibration is narrow.** Dominick's is one grocery chain, in one metropolitan area, during one historical period, and before modern algorithmic pricing. It anchors the simulator, but it does not validate the loop or the interventions in the field.

## Citation

If you use this software, please cite it as:

```bibtex
@software{mendoza_fploop_2026,
  author  = {Mendoza, Jose},
  title   = {Forecast--Price Feedback Loop: A Simulation Instrument for
             Pricing-Induced Demand Bias and Its Correction},
  year    = {2026},
  url     = {https://github.com/jrmst102/forecast-price-feedback-loop}
}
```

The ISF paper citation will be added once it is available.

## Acknowledgments

This project uses the Dominick's Finer Foods dataset provided by the James M. Kilts Center at the University of Chicago Booth School of Business.

https://www.chicagobooth.edu/research/kilts/research-data/dominicks

The intervention taxonomy is inspired by IBM's AI Fairness 360.

https://github.com/Trusted-AI/AIF360

## License

MIT — see [LICENSE](LICENSE).

## Contact

Dr. Jose Mendoza  
jose.mendoza@nyu.edu
