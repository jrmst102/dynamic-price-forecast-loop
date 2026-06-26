"""
Forecast–Price Feedback Loop
File: tests/test_feedforward.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

FeedforwardForecaster tests (require the optional ``nn`` extra). It is a
DesignMatrixForecaster, reports NaN before fitting and a finite estimate after,
recovers beta on clean data, warm-starts on refit, and accepts the two-column
Family-B design.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from fploop.forecasters import DesignMatrixForecaster, FeedforwardForecaster  # noqa: E402
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


def test_is_design_matrix_forecaster():
    assert isinstance(FeedforwardForecaster(), DesignMatrixForecaster)


def test_estimated_elasticity_nan_before_finite_after():
    f = FeedforwardForecaster(seed=0)
    assert np.isnan(f.estimated_elasticity()[0])
    f.fit(_clean_history())
    est = f.estimated_elasticity()
    assert est.shape == (1,)
    assert np.isfinite(est[0])


def test_recovers_beta_on_clean_data():
    beta = -2.0
    f = FeedforwardForecaster(seed=0)
    f.fit(_clean_history(beta=beta, T=200))
    est = float(f.estimated_elasticity()[0])
    assert abs(est - beta) < 0.7, f"estimate {est:.3f} not within 0.7 of {beta}"


def test_warm_start_updates_estimate_without_error():
    f = FeedforwardForecaster(seed=0)
    f.fit(_clean_history(beta=-2.0, T=120, seed=0))
    first = float(f.estimated_elasticity()[0])
    # Extend with more data and refit (warm-started): must not error and should move.
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
    f = FeedforwardForecaster(seed=0)
    f.fit_design(np.column_stack([logp, resid]), y)
    assert np.isfinite(f.estimated_elasticity()[0])
