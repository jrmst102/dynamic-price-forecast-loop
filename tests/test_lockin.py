"""
Forecast–Price Feedback Loop
File: tests/test_lockin.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Scenario-2 lock-in result: the visible greedy spiral and which fix matches it.

With the cost shifter dead (``cost_shifter_std=0``) the greedy markup freezes
price, dispersion collapses, demand craters, and the OLS slope is driven entirely
by the endogenous wiggle (bias -> 1/lambda). The cure for *this* disease is forced
exploration, not the instrument — because the dead cost shifter also starves the
IV the causal arm needs. "The fix must match the disease."
"""

from __future__ import annotations

import numpy as np

from fploop.forecasters.gbt import GBTForecaster
from fploop.generators.linear_logit import LinearLogitWorld
from fploop.loop import run_simulation
from fploop.metrics import residual_elasticity_bias
from fploop.policies import ControlFunctionPolicy, ControlledVariancePricing, GreedyBaseline
from fploop.types import WorldConfig

COLLAPSE = WorldConfig(
    elasticity=-1.5,
    endogeneity_strength=0.6,
    shock_std=0.2,
    cost_shifter_std=0.0,
    horizon=200,
)


def _run(policy_cls, seed: int = 0):
    world = LinearLogitWorld(COLLAPSE)
    policy = policy_cls(GBTForecaster(), rng=np.random.default_rng(1))
    return run_simulation(world, policy, seed=seed)


def _second_half(result):
    logp = np.log(result.prices.ravel())
    half = len(logp) // 2
    std = float(np.std(logp[half:]))
    demand = float(np.mean(result.observed_demand.ravel()[half:]))
    bias = residual_elasticity_bias(result.estimated_elasticity, result.true_elasticity).ravel()
    abs_bias = float(np.nanmean(np.abs(bias[half:])))
    est = float(np.nanmean(result.estimated_elasticity.ravel()[half:]))
    return std, demand, abs_bias, est


def test_greedy_locks_in():
    std, demand, _abs_bias, _est = _second_half(_run(GreedyBaseline))
    assert std < 0.3, f"greedy price std {std:.3f} did not collapse"
    assert demand < COLLAPSE.base_demand / 4, f"greedy demand {demand:.1f} did not crater"


def test_exploration_breaks_lockin():
    _gstd, _gdem, g_bias, _gest = _second_half(_run(GreedyBaseline))
    e_result = _run(ControlledVariancePricing)
    estd, _edem, e_bias, _eest = _second_half(e_result)
    # The fixed exploration arm anneals its dispersion as identification improves
    # (eff_std ~ sqrt(warmup / t)) and truncates each draw at +/-2 sigma. So by the
    # second half its price std has collapsed like greedy's -- sustained high variance
    # was the *old, unbounded* behavior this fix removed. What still holds at this high
    # lambda: exploration meaningfully (if not dramatically) reduces estimation bias
    # versus greedy, and its prices stay bounded (the old arm diverged to ~95x oracle).
    assert estd < 0.3, f"exploration std {estd:.3f} should anneal like greedy's"
    assert e_bias < 0.9 * g_bias, (
        f"exploration |bias| {e_bias:.3f} not meaningfully below greedy {g_bias:.3f}"
    )
    e_prices = e_result.prices.ravel()
    price_ratio = float(e_prices.max() / np.median(e_prices))
    assert price_ratio < 5.0, f"exploration prices diverged (max/median {price_ratio:.1f})"


def test_causal_cannot_rescue_without_instrument():
    cstd, _cdem, _cbias, cest = _second_half(_run(ControlFunctionPolicy))
    # The dead cost shifter starves the instrument, so the 2SLS guard holds the
    # prior: the IV arm neither restores price variation (unlike exploration) nor
    # recovers the true elasticity — its estimate stays pinned near the -2.0 prior.
    assert cstd < 0.3, f"causal price std {cstd:.3f} unexpectedly dispersed"
    assert abs(cest - COLLAPSE.elasticity) > 0.3, (
        f"causal estimate {cest:.3f} unexpectedly recovered true {COLLAPSE.elasticity}"
    )
