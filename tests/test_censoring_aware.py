"""
Forecast–Price Feedback Loop
File: tests/test_censoring_aware.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Phase 4b.2 — censoring-aware EM demand unconstraining (§B2.2, §B2.3).

A naive forecaster trains on demand truncated at capacity, so when capacity binds
it underestimates demand and the elasticity readout flattens. The EM/Tobit arm
reconstructs the latent demand above the cap before fitting, recovering the true
own-price elasticity. With censoring off it reduces to the greedy price-only fit.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from fploop.forecasters.gbt import GBTForecaster
from fploop.generators.linear_logit import LinearLogitWorld
from fploop.loop import run_simulation
from fploop.metrics import cumulative_regret
from fploop.policies import CensoringAwarePolicy, GreedyBaseline
from fploop.types import WorldConfig

CENSORED = WorldConfig(
    elasticity=-1.5,
    endogeneity_strength=0.0,
    shock_std=0.2,
    cost_shifter_std=0.15,
    base_demand=100.0,
    horizon=200,
    censoring=True,
    capacity=40.0,
)


def _run(policy_cls, cfg, seed=0):
    world = LinearLogitWorld(cfg)
    policy = policy_cls(GBTForecaster(), rng=np.random.default_rng(1))
    return run_simulation(world, policy, seed=seed)


def _second_half_abs_bias(result, truth):
    est = result.estimated_elasticity.ravel()
    half = len(est) // 2
    return float(np.nanmean(np.abs(est[half:] - truth)))


def test_censoring_aware_recovers_elasticity_under_binding_capacity():
    # Binding capacity flattens the naive greedy slope; the EM unconstraining
    # recovers the true elasticity markedly better.
    truth = CENSORED.elasticity  # -1.5
    g_bias = _second_half_abs_bias(_run(GreedyBaseline, CENSORED), truth)
    c_bias = _second_half_abs_bias(_run(CensoringAwarePolicy, CENSORED), truth)
    assert c_bias < g_bias, (
        f"censoring-aware |bias| {c_bias:.3f} not below naive greedy {g_bias:.3f}"
    )


def _final_regret(result):
    rev = np.asarray(result.realized_revenue)
    oracle = np.asarray(result.oracle_revenue)
    return float(cumulative_regret(rev, oracle)[-1])


# A capacity that binds at the *optimum*: at price~3 (markup 3 over cost~1) demand is
# ~19 > 12, so the unconstrained optimum over-demands the cap and the capacity-aware
# rule must raise price. (CENSORED's cap of 40 only binds during low-price exploration,
# which moves bias but not the optimum's regret.)
TIGHT_CENSORED = replace(CENSORED, capacity=12.0)


def test_censoring_aware_improves_regret_under_binding_capacity():
    # Capacity-aware pricing (2.2): under a binding cap the arm raises price to choke
    # estimated demand to the inferred capacity instead of pricing at the
    # unconstrained optimum, so it now beats greedy on REGRET, not only on bias.
    g_regret = _final_regret(_run(GreedyBaseline, TIGHT_CENSORED))
    c_regret = _final_regret(_run(CensoringAwarePolicy, TIGHT_CENSORED))
    assert c_regret < g_regret, (
        f"censoring-aware regret {c_regret:.1f} not below naive greedy {g_regret:.1f}"
    )


def test_censoring_aware_matches_greedy_when_censoring_off():
    # No binding capacity -> no demand spike to detect -> identical price-only fit.
    uncensored = WorldConfig(
        elasticity=-1.5,
        endogeneity_strength=0.0,
        shock_std=0.2,
        cost_shifter_std=0.15,
        horizon=200,
    )
    g_est = _run(GreedyBaseline, uncensored).estimated_elasticity.ravel()
    c_est = _run(CensoringAwarePolicy, uncensored).estimated_elasticity.ravel()
    half = len(g_est) // 2
    assert np.allclose(np.nan_to_num(g_est[half:]), np.nan_to_num(c_est[half:]), atol=1e-9), (
        "censoring-aware diverged from greedy with censoring off"
    )
