"""
Forecast–Price Feedback Loop
File: tests/test_features.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Phase 4b — design-matrix feature sets (§B1.2). The price-only design and the
ordered control columns (cost, competitor, reference, time) are built correctly,
the reference series falls back to an EMA when unlogged, and an unknown control
raises.
"""

from __future__ import annotations

import numpy as np
import pytest

from fploop.features import CONTROLS, build_design, reference_series
from fploop.generators.linear_logit import LinearLogitWorld
from fploop.types import History, WorldConfig


def _history(cfg: WorldConfig, n: int = 30, seed: int = 0) -> History:
    w = LinearLogitWorld(cfg)
    w.reset(seed)
    h = History()
    rng = np.random.default_rng(1)
    for _ in range(n):
        p = np.array([2.0 * np.exp(rng.normal(0.0, 0.2))])
        h.add(w.step(p))
    return h


CFG = WorldConfig(
    elasticity=-1.5,
    reference_effect=0.4,
    reference_memory=0.7,
    competition=True,
    cross_elasticity=0.3,
    cost_shifter_std=0.15,
)


def test_price_only_design():
    h = _history(CFG)
    X, y = build_design(h)
    assert X.shape == (len(h), 1)
    assert np.allclose(X[:, 0], np.log([p[0] for p in h.prices]))
    assert np.allclose(y, np.log([d[0] for d in h.observed_demand]))


def test_control_columns_in_order():
    h = _history(CFG)
    X, _ = build_design(h, controls=CONTROLS)
    assert X.shape == (len(h), 1 + len(CONTROLS))
    assert np.allclose(X[:, 0], np.log([p[0] for p in h.prices]))
    assert np.allclose(X[:, 1], np.log([c[0] for c in h.costs]))  # cost
    assert np.allclose(X[:, 2], np.log([pc[0] for pc in h.competitor_prices]))  # competitor
    assert np.allclose(X[:, 3], np.log([r[0] for r in h.reference_prices]))  # reference (logged)
    assert np.allclose(X[:, 4], np.arange(len(h), dtype=float))  # time


def test_reference_column_matches_logged_reference():
    h = _history(CFG)
    X, _ = build_design(h, controls=("reference",))
    assert np.allclose(X[:, 1], np.log([r[0] for r in h.reference_prices]))


def test_reference_ema_fallback_when_unlogged():
    h = _history(CFG)
    h.reference_prices = []  # force the EMA fallback
    r = reference_series(h, reference_memory=0.7)
    prices = np.array([p[0] for p in h.prices])
    expected = np.empty_like(prices)
    expected[0] = prices[0]
    for t in range(1, len(prices)):
        expected[t] = 0.7 * expected[t - 1] + 0.3 * prices[t - 1]
    assert np.allclose(r, expected)


def test_unknown_control_raises():
    h = _history(CFG)
    with pytest.raises(ValueError):
        build_design(h, controls=("bogus",))
