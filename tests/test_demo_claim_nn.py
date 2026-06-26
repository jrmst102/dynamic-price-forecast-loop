"""
Forecast–Price Feedback Loop
File: tests/test_demo_claim_nn.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Behavioural acceptance tests for the NN forecasters (require the ``nn`` extra).

Foundation-fix regime (``cost_shifter_std=0.05``). The feedforward net is smooth,
so its greedy arm does not lock in the way the tree does — instead it cleanly
exhibits the endogeneity *level* bias, which the forecaster-independent 2SLS
causal arm corrects. (Regret is not asserted here: the FF greedy is already a
strong baseline at this regime, and the weak instrument makes the 2SLS regret
edge seed-fragile; the robust NN signal is bias.) The RNN claim stays soft.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from fploop.forecasters import FeedforwardForecaster, RNNForecaster  # noqa: E402
from fploop.forecasters.gbt import GBTForecaster  # noqa: E402
from fploop.generators.linear_logit import LinearLogitWorld  # noqa: E402
from fploop.loop import run_simulation  # noqa: E402
from fploop.metrics import residual_elasticity_bias  # noqa: E402
from fploop.policies import ControlFunctionPolicy, GreedyBaseline  # noqa: E402
from fploop.types import WorldConfig  # noqa: E402

DEMO = WorldConfig(
    elasticity=-1.5,
    endogeneity_strength=0.6,
    shock_std=0.2,
    cost_shifter_std=0.05,
    horizon=200,
)


def _second_half_abs_bias(policy_cls, forecaster_factory, seed: int) -> float:
    world = LinearLogitWorld(DEMO)
    policy = policy_cls(forecaster_factory(), rng=np.random.default_rng(1), retrain_every=2)
    result = run_simulation(world, policy, seed=seed)
    bias = residual_elasticity_bias(result.estimated_elasticity, result.true_elasticity).ravel()
    half = len(bias) // 2
    return float(np.nanmean(np.abs(bias[half:])))


def test_twosls_causal_reduces_bias_vs_feedforward_greedy():
    # The 2SLS causal arm is forecaster-independent (built with a cheap GBT it
    # never fits); compare it to the FF greedy baseline on second-half |bias|.
    seeds = range(3)
    ff_greedy = np.array(
        [_second_half_abs_bias(GreedyBaseline, FeedforwardForecaster, s) for s in seeds]
    )
    causal = np.array(
        [_second_half_abs_bias(ControlFunctionPolicy, GBTForecaster, s) for s in seeds]
    )
    assert ff_greedy.mean() > 0.05, "FF greedy should carry the endogeneity level bias"
    assert causal.mean() < ff_greedy.mean(), (
        f"2SLS-causal bias {causal.mean():.3f} !< FF-greedy {ff_greedy.mean():.3f}"
    )
    wins = int(np.sum(causal < ff_greedy))
    assert wins >= 2, f"2SLS-causal beat FF-greedy on bias on only {wins}/3 seeds"


def test_rnn_greedy_loop_biases_and_is_finite():
    # Softer claim: the greedy loop induces a clear bias on the GRU; the run is
    # finite. No strict control-function win asserted for the windowed model.
    bias = _second_half_abs_bias(GreedyBaseline, RNNForecaster, seed=0)
    assert np.isfinite(bias)
    assert bias > 0.01
