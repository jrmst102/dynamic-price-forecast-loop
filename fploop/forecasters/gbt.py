"""
Forecast–Price Feedback Loop
File: fploop/forecasters/gbt.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Gradient-boosted-trees demand forecaster. Implements :class:`GBTForecaster`,
backed by :class:`sklearn.ensemble.HistGradientBoostingRegressor` with
deliberately regularised defaults, regressing log-demand on the design matrix.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import HistGradientBoostingRegressor

from fploop.forecasters.base import DesignMatrixForecaster


class GBTForecaster(DesignMatrixForecaster):
    """Gradient-boosted-trees demand forecaster.

    Backed by :class:`sklearn.ensemble.HistGradientBoostingRegressor` (LightGBM
    and XGBoost are intentionally NOT used). Regresses ``log q`` on a design
    matrix whose column 0 is ``log price`` (Family B appends a control-function
    residual). The shared :class:`DesignMatrixForecaster` base supplies the
    finite-difference elasticity readout; this class implements only the fit and
    prediction hooks.
    """

    window = 1

    def __init__(self, *, fd_eps: float = 0.2, **hgbr_kwargs: object) -> None:
        # Defaults are deliberately regularised (shallow, few iterations) and the
        # finite-difference step is wide. A flexible GBT refit every period drives
        # the greedy markup to oscillate so violently that the price dispersion
        # swamps the endogeneity signal and the cost instrument loses all strength;
        # a smooth surface keeps the loop in the regime the demo needs and is
        # ~6x faster to fit. (Phase 2 tuning — preserved here; override via
        # ``hgbr_kwargs`` for clean-data fits.)
        super().__init__()
        self.fd_eps = fd_eps
        self._kwargs: dict = {
            "max_iter": 50,
            "learning_rate": 0.1,
            "max_depth": 2,
            "min_samples_leaf": 30,
            **hgbr_kwargs,
        }
        self._model: HistGradientBoostingRegressor | None = None

    def _fit(self, X: NDArray, y: NDArray) -> None:
        """Fit the GBT regressor on design matrix ``X`` against target ``y``."""
        self._model = HistGradientBoostingRegressor(**self._kwargs).fit(X, y)

    def predict_logq(self, X: NDArray) -> NDArray:
        """Predict log-demand for each row of ``X``."""
        return self._model.predict(np.asarray(X, dtype=float))
