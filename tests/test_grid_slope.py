"""
Forecast–Price Feedback Loop
File: tests/test_grid_slope.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

The clipped grid-slope elasticity readout (§4 of the foundation-fix spec).

These need no torch: they exercise ``DesignMatrixForecaster.estimated_elasticity``
through ``GBTForecaster``.
"""

from __future__ import annotations

import numpy as np

from fploop.forecasters import GBTForecaster


def _clean(beta: float = -2.0, T: int = 200, spread: float = 0.4, seed: int = 0):
    """Clean exogenous dispersed (log p, log q) from the true model (lambda = 0)."""
    rng = np.random.default_rng(seed)
    logp = rng.normal(0.0, spread, size=T)
    y = np.log(100.0) + beta * logp + rng.normal(0.0, 0.05, size=T)
    return logp.reshape(-1, 1), y


def test_grid_slope_low_variance_and_accurate():
    # Clean exogenous dispersed data: fit on independent realisations and measure
    # the readout's spread. The grid-slope's std is far below the old tree FD's
    # (~0.49 on this data); assert std < 0.1 and mean within 0.2 of beta.
    beta = -2.0
    ests = []
    for seed in range(15):
        X, y = _clean(beta=beta, T=200, seed=seed)
        f = GBTForecaster()
        f.fit_design(X, y)
        ests.append(float(f.estimated_elasticity()[0]))
    ests = np.array(ests)
    assert ests.std() < 0.1, f"grid-slope std {ests.std():.3f} not < 0.1"
    assert abs(ests.mean() - beta) < 0.2, f"grid-slope mean {ests.mean():.3f} off from {beta}"


def test_clipping_no_extrapolation_on_narrow_band():
    # Even with prices spanning a narrow band, the 5-95 percentile grid does not
    # extrapolate into flat tree regions, so the readout stays near beta.
    beta = -2.0
    X, y = _clean(beta=beta, T=200, spread=0.15)
    f = GBTForecaster()
    f.fit_design(X, y)
    est = float(f.estimated_elasticity()[0])
    assert abs(est - beta) < 1.0, f"narrow-band readout {est:.3f} off by >1 from {beta}"


def test_collapsed_dispersion_returns_nan():
    # When observed log-prices are all within 1e-3 (frozen), elasticity is
    # undefined and the readout returns nan rather than a fabricated slope.
    f = GBTForecaster()
    X = np.full((50, 1), np.log(5.0))
    y = np.random.default_rng(0).normal(4.0, 0.1, size=50)
    f.fit_design(X, y)
    assert np.isnan(f.estimated_elasticity()[0])
