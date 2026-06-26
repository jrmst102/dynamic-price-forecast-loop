"""
Forecast–Price Feedback Loop
File: tests/test_rnn.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

RNNForecaster (GRU) tests (require the optional ``nn`` extra).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from fploop.forecasters import DesignMatrixForecaster, RNNForecaster  # noqa: E402
from fploop.types import History, Observation  # noqa: E402


def _clean_history(beta: float = -2.0, T: int = 200, seed: int = 0) -> History:
    """Clean exogenous dispersed data straight from the true model (lambda = 0)."""
    rng = np.random.default_rng(seed)
    logp = rng.normal(0.0, 0.4, size=T)
    q = np.exp(np.log(100.0) + beta * logp + rng.normal(0.0, 0.05, size=T))
    prices = np.exp(logp)
    history = History()
    for t in range(T):
        history.add(
            Observation(
                period=t,
                prices=np.array([prices[t]]),
                observed_demand=np.array([q[t]]),
                revenue=float(prices[t] * q[t]),
            )
        )
    return history


def test_is_design_matrix_forecaster_with_window():
    f = RNNForecaster(window=8)
    assert isinstance(f, DesignMatrixForecaster)
    assert f.window == 8


def test_estimated_elasticity_nan_before_finite_after():
    f = RNNForecaster(seed=0)
    assert np.isnan(f.estimated_elasticity()[0])
    f.fit(_clean_history())
    est = f.estimated_elasticity()
    assert est.shape == (1,)
    assert np.isfinite(est[0])


def test_recovers_beta_on_clean_data():
    # Average over seeds: the GRU finite difference is noisier than the MLP's.
    beta = -2.0
    ests = []
    for seed in range(3):
        f = RNNForecaster(seed=seed)
        f.fit(_clean_history(beta=beta, T=200, seed=seed))
        ests.append(float(f.estimated_elasticity()[0]))
    mean_est = float(np.mean(ests))
    assert abs(mean_est - beta) < 0.7, f"mean estimate {mean_est:.3f} not within 0.7 of {beta}"


def test_warm_start_updates_estimate_without_error():
    f = RNNForecaster(seed=0)
    f.fit(_clean_history(beta=-2.0, T=120, seed=0))
    first = float(f.estimated_elasticity()[0])
    f.fit(_clean_history(beta=-2.0, T=240, seed=0))
    second = float(f.estimated_elasticity()[0])
    assert np.isfinite(second)
    assert first != second


def test_fit_design_two_column_family_b():
    history = _clean_history(beta=-2.0, T=200)
    prices = np.array([p[0] for p in history.prices])
    demand = np.array([d[0] for d in history.observed_demand])
    logp = np.log(prices).reshape(-1, 1)
    y = np.log(demand)
    resid = np.random.default_rng(3).normal(0.0, 1.0, size=logp.shape[0]).reshape(-1, 1)
    f = RNNForecaster(seed=0)
    f.fit_design(np.column_stack([logp, resid]), y)
    assert np.isfinite(f.estimated_elasticity()[0])
