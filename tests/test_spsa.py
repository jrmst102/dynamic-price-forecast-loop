"""
Forecast–Price Feedback Loop
File: tests/test_spsa.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Phase 4b — SPSA model-free profit optimiser (§B1.3).
"""

from __future__ import annotations

import numpy as np

from fploop.forecasters.gbt import GBTForecaster
from fploop.generators.linear_logit import LinearLogitWorld, optimal_price
from fploop.loop import run_simulation
from fploop.policies import SPSAPricing
from fploop.types import WorldConfig

# A static world (no shock, no endogeneity, constant cost): the profit landscape
# in log price is concave, so SPSA's centre should climb to the oracle markup.
STATIC = WorldConfig(
    elasticity=-2.0,
    endogeneity_strength=0.0,
    shock_std=0.0,
    cost_shifter_std=0.0,
    marginal_cost=1.0,
    base_demand=100.0,
    horizon=300,
)


def _run():
    return run_simulation(
        LinearLogitWorld(STATIC),
        SPSAPricing(GBTForecaster(), rng=np.random.default_rng(1)),
        seed=0,
    )


def test_spsa_converges_to_oracle_markup():
    res = _run()
    oracle_p = optimal_price(np.array([STATIC.marginal_cost]), STATIC.elasticity)[0]
    logp = res.prices.ravel()
    second_half_mean = float(np.mean(logp[len(logp) // 2 :]))
    assert abs(second_half_mean - oracle_p) < 0.15, (
        f"SPSA price {second_half_mean:.3f} did not converge to oracle {oracle_p:.3f}"
    )


def test_spsa_reports_no_elasticity():
    # Model-free: it estimates no elasticity, so the readout is nan throughout.
    res = _run()
    assert np.all(np.isnan(res.estimated_elasticity.ravel()))
