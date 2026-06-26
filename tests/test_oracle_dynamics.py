"""
Forecast–Price Feedback Loop
File: tests/test_oracle_dynamics.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Phase 4a — the oracle as a forward simulation on a lambda=0 clone (§A.6, §A.9).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from fploop.forecasters.gbt import GBTForecaster
from fploop.generators.linear_logit import LinearLogitWorld
from fploop.loop import run_simulation
from fploop.oracle import full_information_oracle
from fploop.policies import GreedyBaseline
from fploop.types import WorldConfig

# The four Phase-4a regimes (each enables one feature on top of the demo world).
WORLDS = {
    "reference": dict(reference_effect=0.5, cost_shifter_std=0.15),
    "drift": dict(drift_kind="gradual", drift_magnitude=0.8, cost_shifter_std=0.05),
    "censoring": dict(censoring=True, capacity=40.0, cost_shifter_std=0.15),
    "competition": dict(competition=True, cross_elasticity=0.5, cost_shifter_std=0.15),
}
BASE = dict(elasticity=-1.5, endogeneity_strength=0.6, shock_std=0.2, horizon=80)


def test_oracle_equals_lambda0_clone_forward_sim():
    cfg = WorldConfig(**BASE, **WORLDS["reference"])
    orc = full_information_oracle(LinearLogitWorld(cfg), seed=0)
    # Manually replay the endogeneity-free clone.
    clone = LinearLogitWorld(replace(cfg, endogeneity_strength=0.0))
    clone.reset(0)
    prices = []
    for _ in range(cfg.horizon):
        p = clone.optimal_prices()
        obs = clone.step(p)
        prices.append(obs.prices)
    assert np.allclose(orc.oracle_prices, np.array(prices).reshape(cfg.horizon, 1))


def test_oracle_beats_greedy_on_average_across_worlds():
    for name, extra in WORLDS.items():
        diffs = []
        for seed in (0, 1):
            cfg = WorldConfig(**BASE, **extra)
            orc = full_information_oracle(LinearLogitWorld(cfg), seed=seed)
            g = run_simulation(
                LinearLogitWorld(cfg),
                GreedyBaseline(GBTForecaster(), rng=np.random.default_rng(1)),
                seed=seed,
            )
            diffs.append(orc.oracle_revenue.sum() - np.asarray(g.realized_revenue).sum())
        assert np.mean(diffs) >= 0.0, f"oracle did not dominate greedy in {name}"


def test_oracle_clone_exogenous_paths_correct():
    # The clone shares the seeded shock/cost/drift/competitor draws with a fresh
    # world (lambda only changes the price map, not the draws).
    cfg = WorldConfig(
        **BASE, drift_kind="abrupt", drift_magnitude=0.7, competition=True, cross_elasticity=0.4
    )
    fresh = LinearLogitWorld(cfg)
    fresh.reset(3)
    clone = LinearLogitWorld(replace(cfg, endogeneity_strength=0.0))
    clone.reset(3)
    assert np.allclose(clone._xi, fresh._xi)
    assert np.allclose(clone._cost, fresh._cost)
    assert np.allclose(clone._alpha, fresh._alpha)  # drift schedule
    assert np.allclose(clone._pc, fresh._pc)  # competitor path
    # drift schedule is the configured abrupt step
    alpha0 = np.log(cfg.base_demand)
    assert np.allclose(clone._alpha[cfg.horizon // 2 :], alpha0 + 0.7)
