"""
Forecast–Price Feedback Loop
File: tests/test_competition.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Phase 4a — competition / observed rival price (§A.3, §A.6, §A.9). The rival price
is logged and varies, own demand responds to cross-elasticity, the cross term
vanishes when competition is off, and the constant-elasticity oracle markup is
unchanged by the competitor path.
"""

from __future__ import annotations

import numpy as np
import pytest

from fploop.generators.linear_logit import LinearLogitWorld, optimal_price
from fploop.oracle import full_information_oracle
from fploop.types import WorldConfig


def test_competitor_price_is_logged():
    cfg = WorldConfig(
        elasticity=-1.5,
        cost_shifter_std=0.0,
        horizon=20,
        competition=True,
        cross_elasticity=0.5,
        competitor_price_std=0.2,
    )
    w = LinearLogitWorld(cfg)
    w.reset(0)
    obs = w.step(np.array([2.0]))
    assert obs.competitor_prices is not None
    pcs = [w.step(np.array([2.0])).competitor_prices[0] for _ in range(15)]
    assert np.std(pcs) > 0.0  # competitor price genuinely varies


def test_demand_responds_to_cross_elasticity():
    # Substitutes (cross_elasticity > 0): own demand rises with the rival price.
    cfg = WorldConfig(
        elasticity=-1.5,
        endogeneity_strength=0.0,
        shock_std=0.0,
        cost_shifter_std=0.0,
        horizon=200,
        competition=True,
        cross_elasticity=0.5,
        competitor_price_std=0.2,
    )
    w = LinearLogitWorld(cfg)
    w.reset(0)
    logq, logpc = [], []
    for _ in range(cfg.horizon):
        obs = w.step(np.array([2.0]))  # hold own price fixed
        logq.append(np.log(obs.observed_demand[0]))
        logpc.append(np.log(obs.competitor_prices[0]))
    assert np.corrcoef(logq, logpc)[0, 1] > 0.5


def test_no_cross_effect_when_off():
    # With competition off, pc_t == 1 so the term vanishes regardless of cross_e.
    cfg = WorldConfig(
        elasticity=-1.5,
        endogeneity_strength=0.0,
        shock_std=0.0,
        cost_shifter_std=0.0,
        horizon=5,
        competition=False,
        cross_elasticity=0.5,
    )
    w = LinearLogitWorld(cfg)
    w.reset(0)
    obs = w.step(np.array([2.0]))
    assert obs.observed_demand[0] == pytest.approx(cfg.base_demand * 2.0**cfg.elasticity)
    assert obs.competitor_prices[0] == pytest.approx(1.0)


def test_oracle_prices_under_competition():
    # Constant-elasticity optimum is level-independent, so the competitor path
    # shifts demand but not the oracle's markup price.
    cfg = WorldConfig(
        elasticity=-1.5,
        cost_shifter_std=0.0,
        marginal_cost=1.0,
        horizon=30,
        competition=True,
        cross_elasticity=0.5,
    )
    orc = full_information_oracle(LinearLogitWorld(cfg), seed=0)
    expected = optimal_price(np.array([1.0]), cfg.elasticity)[0]
    assert np.allclose(orc.oracle_prices.ravel(), expected)
