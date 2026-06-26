"""
Forecast–Price Feedback Loop
File: tests/test_world.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

LinearLogitWorld behaviour: seed-reproducible shock/cost paths, the true-elasticity
value and shape, the monopoly markup formula for optimal prices, the near-singularity
floor in the ``optimal_price`` helper, and rejection of the deferred strategic toggle
and of inelastic (|beta| <= 1) configurations.
"""

from __future__ import annotations

import numpy as np
import pytest

from fploop.generators.linear_logit import LinearLogitWorld, optimal_price
from fploop.types import WorldConfig


def test_reset_is_seed_reproducible():
    cfg = WorldConfig(horizon=50, shock_std=0.2, cost_shifter_std=0.15)
    w1, w2 = LinearLogitWorld(cfg), LinearLogitWorld(cfg)
    w1.reset(7)
    w2.reset(7)
    assert np.allclose(w1.shock_path(), w2.shock_path())
    assert np.allclose(w1.cost_path(), w2.cost_path())

    w3 = LinearLogitWorld(cfg)
    w3.reset(8)
    assert not np.allclose(w1.shock_path(), w3.shock_path())


def test_true_elasticity_value_and_shape():
    cfg = WorldConfig(elasticity=-1.8)
    world = LinearLogitWorld(cfg)
    world.reset(0)
    te = world.true_elasticity()
    assert te.shape == (1,)
    assert te[0] == pytest.approx(-1.8)


def test_optimal_prices_matches_markup_formula():
    cfg = WorldConfig(elasticity=-1.5, marginal_cost=2.0, cost_shifter_std=0.15)
    world = LinearLogitWorld(cfg)
    state = world.reset(3)
    beta = cfg.elasticity
    expected = state.cost * beta / (beta + 1.0)
    assert np.allclose(world.optimal_prices(), expected)


def test_optimal_price_helper_floors_near_singularity():
    # An estimate at -1.0 would blow up; the floor keeps the markup finite.
    p = optimal_price(np.array([1.0]), -1.0, floor=-1.05)
    assert np.isfinite(p).all()
    assert p[0] == pytest.approx(1.0 * (-1.05 / -0.05))


def test_strategic_toggle_raises():
    # Phase 4a implements drift/censoring/competition; strategic stays deferred (§A.8).
    world = LinearLogitWorld(WorldConfig(strategic=True))
    with pytest.raises(NotImplementedError):
        world.reset(0)


def test_inelastic_beta_raises_value_error():
    world = LinearLogitWorld(WorldConfig(elasticity=-0.8))
    with pytest.raises(ValueError):
        world.reset(0)
