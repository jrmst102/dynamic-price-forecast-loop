"""
Forecast–Price Feedback Loop
File: tests/test_demo_claim.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Behavioural acceptance test guarding the §4 narrative (foundation-fix regime).

The headline claim is that the causal (control-function / 2SLS) policy beats the
greedy baseline once the readout is honest. We assert it on the demo regime
``cost_shifter_std=0.05`` (weak-but-live instrument), using the clipped
grid-slope readout and the 2SLS causal arm, over five seeds:

- **Regret** (readout-independent): causal cumulative regret <= greedy on every
  seed — small but robust.
- **Bias**: causal second-half mean ``|residual bias|`` below greedy on the
  seed-averaged mean and on at least 4 of 5 seeds.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from fploop.forecasters.gbt import GBTForecaster
from fploop.generators.linear_logit import LinearLogitWorld
from fploop.loop import run_simulation
from fploop.metrics import cumulative_regret, residual_elasticity_bias
from fploop.policies import ControlFunctionPolicy, GreedyBaseline
from fploop.types import WorldConfig

DEMO = WorldConfig(
    elasticity=-1.5,
    endogeneity_strength=0.6,
    shock_std=0.2,
    cost_shifter_std=0.05,
    horizon=200,
)

_POLICIES = {"greedy": GreedyBaseline, "causal": ControlFunctionPolicy}


@cache
def _run(policy_key: str, seed: int) -> tuple[float, float]:
    """Return (second-half mean |bias|, final cumulative regret) for one run."""
    world = LinearLogitWorld(DEMO)
    policy = _POLICIES[policy_key](GBTForecaster(), rng=np.random.default_rng(1))
    result = run_simulation(world, policy, seed=seed)
    bias = residual_elasticity_bias(result.estimated_elasticity, result.true_elasticity).ravel()
    half = len(bias) // 2
    abs_bias = float(np.nanmean(np.abs(bias[half:])))
    regret = float(cumulative_regret(result.realized_revenue, result.oracle_revenue)[-1])
    return abs_bias, regret


SEEDS = range(5)


def test_causal_beats_greedy_on_regret_every_seed():
    for s in SEEDS:
        g_reg = _run("greedy", s)[1]
        c_reg = _run("causal", s)[1]
        assert c_reg <= g_reg, f"seed {s}: causal regret {c_reg:.0f} > greedy {g_reg:.0f}"


def test_causal_reduces_bias_vs_greedy():
    greedy = np.array([_run("greedy", s)[0] for s in SEEDS])
    causal = np.array([_run("causal", s)[0] for s in SEEDS])
    assert causal.mean() < greedy.mean(), (
        f"causal {causal.mean():.3f} !< greedy {greedy.mean():.3f}"
    )
    wins = int(np.sum(causal < greedy))
    assert wins >= 4, f"causal beat greedy on bias on only {wins}/5 seeds (need >=4)"
