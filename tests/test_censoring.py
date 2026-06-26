"""
Forecast–Price Feedback Loop
File: tests/test_censoring.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Phase 4a — censoring / stockouts (§A.3, §A.6, §A.9). Demand is clipped at capacity
with the censored mask set, uncensored demand passes through, and the optimal price
rises to clear a binding capacity.
"""

from __future__ import annotations

import numpy as np
import pytest

from fploop.generators.linear_logit import LinearLogitWorld, optimal_price
from fploop.types import WorldConfig


def test_demand_truncated_at_capacity():
    # A low price drives latent demand above capacity; observed demand is clipped,
    # the censored mask is set, and revenue uses the truncated quantity.
    cfg = WorldConfig(
        elasticity=-1.5,
        endogeneity_strength=0.0,
        shock_std=0.0,
        cost_shifter_std=0.0,
        base_demand=100.0,
        horizon=10,
        censoring=True,
        capacity=40.0,
    )
    w = LinearLogitWorld(cfg)
    w.reset(0)
    obs = w.step(np.array([1.0]))  # latent demand = 100 * 1^-1.5 = 100 > 40
    assert obs.observed_demand[0] == pytest.approx(40.0)
    assert bool(obs.censored[0]) is True
    assert obs.revenue == pytest.approx(1.0 * 40.0)


def test_uncensored_demand_passes_through():
    cfg = WorldConfig(
        elasticity=-1.5,
        endogeneity_strength=0.0,
        shock_std=0.0,
        cost_shifter_std=0.0,
        base_demand=100.0,
        horizon=10,
        censoring=True,
        capacity=40.0,
    )
    w = LinearLogitWorld(cfg)
    w.reset(0)
    obs = w.step(np.array([5.0]))  # latent demand = 100 * 5^-1.5 = 8.94 < 40
    assert bool(obs.censored[0]) is False
    assert obs.observed_demand[0] == pytest.approx(100.0 * 5.0**-1.5)


def test_optimal_price_clears_binding_capacity():
    # When the unconstrained optimum's demand exceeds capacity, the firm raises
    # price to the capacity-clearing level (>= the unconstrained optimum).
    cfg = WorldConfig(
        elasticity=-1.5,
        endogeneity_strength=0.0,
        shock_std=0.0,
        cost_shifter_std=0.0,
        base_demand=100.0,
        horizon=10,
        censoring=True,
        capacity=15.0,  # binds: demand at the unconstrained optimum is ~19.3
    )
    w = LinearLogitWorld(cfg)
    w.reset(0)
    p_unconstrained = optimal_price(np.array([cfg.marginal_cost]), cfg.elasticity)[0]
    p_star = w.optimal_prices()[0]
    assert p_star > p_unconstrained
    # at p_star the latent demand exactly clears capacity
    latent = cfg.base_demand * p_star**cfg.elasticity
    assert latent == pytest.approx(cfg.capacity, rel=1e-6)
