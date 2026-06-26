"""
Forecast–Price Feedback Loop
File: tests/test_forecaster.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Unit tests for GBTForecaster: it recovers the true elasticity on clean,
exogenous, dispersed data and reports NaN before it is fitted.
"""

from __future__ import annotations

import numpy as np

from fploop.forecasters.gbt import GBTForecaster
from fploop.types import History, Observation


def test_estimated_elasticity_recovers_beta_on_clean_data():
    # Clean exogenous data (lambda = 0), dispersed prices, from the true model.
    rng = np.random.default_rng(0)
    beta = -2.0
    alpha = np.log(100.0)
    T = 250
    logp = rng.normal(0.0, 0.4, size=T)  # dispersed log prices
    xi = rng.normal(0.0, 0.05, size=T)  # tiny exogenous noise
    q = np.exp(alpha + beta * logp + xi)

    history = History()
    prices = np.exp(logp)
    for t in range(T):
        history.add(
            Observation(
                period=t,
                prices=np.array([prices[t]]),
                observed_demand=np.array([q[t]]),
                revenue=float(prices[t] * q[t]),
            )
        )

    forecaster = GBTForecaster()
    forecaster.fit(history)
    est = forecaster.estimated_elasticity()
    assert est.shape == (1,)
    assert abs(est[0] - beta) < 0.3


def test_estimated_elasticity_is_nan_before_fit():
    forecaster = GBTForecaster()
    assert np.isnan(forecaster.estimated_elasticity()[0])
