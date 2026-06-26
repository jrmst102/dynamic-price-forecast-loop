"""
Forecast–Price Feedback Loop
File: tests/test_metrics.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Unit tests for the core diagnostic metrics: cumulative regret is zero when the
two series match, residual elasticity bias is the elementwise difference, and the
windowed performative gap has leading NaNs before settling to finite values.
"""

from __future__ import annotations

import numpy as np

from fploop.metrics import (
    cumulative_regret,
    performative_gap,
    residual_elasticity_bias,
)


def test_cumulative_regret_zero_when_equal():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    reg = cumulative_regret(x, x)
    assert reg.shape == x.shape
    assert np.allclose(reg, 0.0)


def test_residual_elasticity_bias_is_elementwise_difference():
    est = np.array([-1.0, -2.0, -3.0])
    truth = np.array([-1.5, -1.5, -1.5])
    assert np.allclose(residual_elasticity_bias(est, truth), est - truth)


def test_performative_gap_leading_nans_then_finite():
    rng = np.random.default_rng(0)
    prices = np.exp(rng.normal(0.0, 0.1, size=40))
    window = 10
    gap = performative_gap(prices, window=window)
    assert gap.shape == prices.shape
    assert np.all(np.isnan(gap[: window - 1]))
    assert np.all(np.isfinite(gap[window - 1 :]))
