"""
Forecast–Price Feedback Loop
File: tests/test_loop.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

End-to-end ``run_simulation`` checks with the GBT forecaster: result arrays have
the right shapes and stay finite after warmup, and the full-information oracle's
total profit dominates the greedy baseline on average across seeds.
"""

from __future__ import annotations

import numpy as np

from fploop.forecasters.gbt import GBTForecaster
from fploop.generators.linear_logit import LinearLogitWorld
from fploop.loop import run_simulation
from fploop.policies import GreedyBaseline
from fploop.types import WorldConfig


def _cfg(horizon=60):
    return WorldConfig(
        elasticity=-1.5,
        endogeneity_strength=0.4,
        shock_std=0.2,
        cost_shifter_std=0.15,
        horizon=horizon,
    )


def test_run_simulation_shapes_and_finite_after_warmup():
    cfg = _cfg(60)
    policy = GreedyBaseline(GBTForecaster(), rng=np.random.default_rng(0))
    res = run_simulation(LinearLogitWorld(cfg), policy, horizon=60, seed=0)

    assert res.prices.shape == (60, 1)
    assert res.observed_demand.shape == (60, 1)
    assert res.estimated_elasticity.shape == (60, 1)
    assert res.true_elasticity.shape == (60, 1)
    assert res.realized_revenue.shape == (60,)
    assert res.oracle_revenue.shape == (60,)

    # Once the forecaster has been fit, the estimate and per-period series are
    # finite (the first fit lands a little after `warmup`, gated by retrain_every).
    est = res.estimated_elasticity.ravel()
    settled = policy.warmup + policy.retrain_every + 1
    assert np.all(np.isfinite(est[settled:]))
    assert np.all(np.isfinite(res.realized_revenue))
    assert np.all(np.isfinite(res.prices))


def test_oracle_profit_dominates_greedy_on_average():
    cfg = _cfg(60)
    oracle_totals, greedy_totals = [], []
    for seed in range(4):
        policy = GreedyBaseline(GBTForecaster(), rng=np.random.default_rng(0))
        res = run_simulation(LinearLogitWorld(cfg), policy, horizon=60, seed=seed)
        oracle_totals.append(res.oracle_revenue.sum())
        greedy_totals.append(res.realized_revenue.sum())
    # Oracle is optimal in expectation; allow tiny slack on the per-seed mean.
    assert np.mean(oracle_totals) >= np.mean(greedy_totals) - 1e-6
