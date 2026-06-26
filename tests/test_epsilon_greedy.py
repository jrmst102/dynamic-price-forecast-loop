"""
Forecast–Price Feedback Loop
File: tests/test_epsilon_greedy.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Phase 4b — ε-greedy forced exploration (§B1.3). At epsilon=0 the arm reduces to
greedy; a positive epsilon raises second-half price dispersion while keeping the
price path finite.
"""

from __future__ import annotations

import numpy as np

from fploop.forecasters.gbt import GBTForecaster
from fploop.generators.linear_logit import LinearLogitWorld
from fploop.loop import run_simulation
from fploop.policies import EpsilonGreedyPricing, GreedyBaseline
from fploop.types import WorldConfig

CFG = WorldConfig(
    elasticity=-1.5,
    endogeneity_strength=0.6,
    shock_std=0.2,
    cost_shifter_std=0.05,
    horizon=120,
)


def _run(policy_cls, seed=0, **kw):
    return run_simulation(
        LinearLogitWorld(CFG),
        policy_cls(GBTForecaster(), rng=np.random.default_rng(1), **kw),
        seed=seed,
    )


def _second_half_dispersion(res) -> float:
    logp = np.log(res.prices.ravel())
    return float(np.std(logp[len(logp) // 2 :]))


def test_epsilon_zero_reduces_to_greedy():
    greedy = _run(GreedyBaseline)
    eps0 = _run(EpsilonGreedyPricing, epsilon=0.0)
    assert np.allclose(eps0.prices, greedy.prices)


def test_epsilon_raises_dispersion_and_stays_finite():
    greedy = _run(GreedyBaseline)
    eps = _run(EpsilonGreedyPricing, epsilon=0.2)
    assert _second_half_dispersion(eps) > _second_half_dispersion(greedy)
    assert np.all(np.isfinite(eps.prices))
