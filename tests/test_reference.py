"""
Forecast–Price Feedback Loop
File: tests/test_reference.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Phase 4a — reference-price effect (§A.3, §A.5, §A.9).

The reference price is an EMA of the firm's own effective prices; omitting it from
a price-only forecaster biases the elasticity readout even with a *live*
instrument — the instrument-independent lock-in channel.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fploop.forecasters.gbt import GBTForecaster
from fploop.generators.linear_logit import LinearLogitWorld
from fploop.loop import run_simulation
from fploop.policies import GreedyBaseline
from fploop.types import WorldConfig


def test_reference_ema_updates_as_specified():
    # r_0 = marginal_cost; r_{t+1} = gamma*r_t + (1-gamma)*p_eff. With lambda=0
    # and no shock, p_eff == the proposed price.
    cfg = WorldConfig(
        elasticity=-2.0,
        reference_effect=0.5,
        reference_memory=0.7,
        marginal_cost=2.0,
        endogeneity_strength=0.0,
        shock_std=0.0,
        cost_shifter_std=0.0,
        horizon=10,
    )
    w = LinearLogitWorld(cfg)
    state = w.reset(0)
    assert state.reference_price[0] == pytest.approx(2.0)  # r_0 = marginal_cost
    w.step(np.array([3.0]))
    expected = 0.7 * 2.0 + 0.3 * 3.0
    assert w.current_state.reference_price[0] == pytest.approx(expected)
    w.step(np.array([3.0]))
    assert w.current_state.reference_price[0] == pytest.approx(0.7 * expected + 0.3 * 3.0)


def test_true_elasticity_is_short_run():
    # The short-run own-price elasticity is beta - theta_ref (the grid-slope target).
    cfg = WorldConfig(elasticity=-1.5, reference_effect=0.5)
    w = LinearLogitWorld(cfg)
    w.reset(0)
    assert w.true_elasticity()[0] == pytest.approx(-2.0)


def test_price_only_readout_is_biased_with_live_instrument():
    # reference_effect>0 with a healthy cost instrument (cost_std=0.15): the
    # price-only grid-slope is biased away from the short-run truth (omitted r_t),
    # yet the run stays finite. Bias is toward zero (the reference coefficient is
    # positive and r_t co-moves with price).
    cfg = WorldConfig(
        elasticity=-1.5,
        reference_effect=0.5,
        endogeneity_strength=0.6,
        shock_std=0.2,
        cost_shifter_std=0.15,
        horizon=200,
    )
    res = run_simulation(
        LinearLogitWorld(cfg),
        GreedyBaseline(GBTForecaster(), rng=np.random.default_rng(1)),
        seed=0,
    )
    est = res.estimated_elasticity.ravel()
    half = len(est) // 2
    assert np.all(np.isfinite(est[half:])), "run did not stay finite"
    short_run = cfg.elasticity - cfg.reference_effect  # -2.0
    bias = float(np.nanmean(est[half:])) - short_run
    assert abs(bias) > 0.3, f"price-only readout not biased ({bias:.3f})"
    assert bias > 0.0, "omitted-variable bias should pull the slope toward zero"


def test_oracle_reference_tracks_oracle_prices():
    # The oracle is a lambda=0 clone; its reference path is the EMA of *its own*
    # prices, not the policy's. Replay the clone and check the recursion holds.
    cfg = WorldConfig(
        elasticity=-1.5,
        reference_effect=0.5,
        reference_memory=0.7,
        endogeneity_strength=0.6,
        shock_std=0.2,
        cost_shifter_std=0.15,
        horizon=40,
    )
    clone = LinearLogitWorld(replace(cfg, endogeneity_strength=0.0))
    clone.reset(0)
    gamma = cfg.reference_memory
    for _ in range(cfg.horizon - 1):
        r_t = clone.current_state.reference_price.copy()
        p = clone.optimal_prices()
        obs = clone.step(p)
        r_next = clone.current_state.reference_price
        assert np.allclose(r_next, gamma * r_t + (1.0 - gamma) * obs.prices)
