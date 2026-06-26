"""
Forecast–Price Feedback Loop
File: tests/test_loop_forecasters.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

The loop runs end-to-end with each forecaster (requires the ``nn`` extra).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from fploop.forecasters import (  # noqa: E402
    FeedforwardForecaster,
    GBTForecaster,
    RNNForecaster,
)
from fploop.generators.linear_logit import LinearLogitWorld  # noqa: E402
from fploop.loop import run_simulation  # noqa: E402
from fploop.policies import GreedyBaseline  # noqa: E402
from fploop.types import WorldConfig  # noqa: E402

FACTORIES = [GBTForecaster, FeedforwardForecaster, RNNForecaster]


def _cfg(horizon: int = 60) -> WorldConfig:
    return WorldConfig(
        elasticity=-1.5,
        endogeneity_strength=0.4,
        shock_std=0.2,
        cost_shifter_std=0.15,
        horizon=horizon,
    )


@pytest.mark.parametrize("factory", FACTORIES, ids=lambda f: f.__name__)
def test_run_simulation_shapes_and_finite_after_warmup(factory):
    cfg = _cfg(60)
    policy = GreedyBaseline(factory(), rng=np.random.default_rng(0), retrain_every=2)
    res = run_simulation(LinearLogitWorld(cfg), policy, horizon=60, seed=0)

    assert res.prices.shape == (60, 1)
    assert res.observed_demand.shape == (60, 1)
    assert res.estimated_elasticity.shape == (60, 1)
    assert res.true_elasticity.shape == (60, 1)
    assert res.realized_revenue.shape == (60,)

    settled = policy.warmup + policy.retrain_every + 1
    assert np.all(np.isfinite(res.estimated_elasticity.ravel()[settled:]))
    assert np.all(np.isfinite(res.realized_revenue))
    assert np.all(np.isfinite(res.prices))
