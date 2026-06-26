"""
Forecast–Price Feedback Loop
File: fploop/forecasters/base.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Abstract bases for demand forecasters. Defines the minimal :class:`Forecaster`
interface and :class:`DesignMatrixForecaster`, which regresses log-demand on a
design matrix (column 0 is log price) and supplies the shared grid-slope
elasticity readout; subclasses implement only the fit and prediction hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from fploop.types import History


class Forecaster(ABC):
    """Demand model retrained on accumulating history.

    `estimated_elasticity` exposes the model's current own-price elasticity so it
    can be compared against the world's ground truth.
    """

    @abstractmethod
    def fit(self, history: History) -> None:
        """Fit/retrain on the accumulated history."""

    @abstractmethod
    def predict_demand(self, prices: NDArray, context: dict | None = None) -> NDArray:
        """Predict demand at the given prices."""

    @abstractmethod
    def estimated_elasticity(self) -> NDArray:
        """Current estimated own-price elasticity, shape (n_products,)."""


class DesignMatrixForecaster(Forecaster):
    """Forecasters that regress log-demand on a design matrix whose column 0 is
    log price.

    A design matrix ``X`` has rows in **temporal order**; **column 0 is always
    ``log price``**; any further columns are controls (e.g. the control-function
    residual). The target ``y`` is ``log demand``. Tree and feedforward models
    treat rows independently (``window == 1``); the RNN treats each row as the end
    of a length-``window`` sequence.

    Subclasses implement only :meth:`_fit` and :meth:`predict_logq`; the base
    provides concrete :meth:`fit`, :meth:`fit_design`, :meth:`predict_demand`, and
    :meth:`estimated_elasticity`.
    """

    window: int = 1  # RNN overrides with its lookback length
    fd_eps: float = 0.05  # finite-difference step in log price

    def __init__(self) -> None:
        self._X: NDArray | None = None
        self._y: NDArray | None = None
        self._fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ---- subclass hooks ----
    @abstractmethod
    def _fit(self, X: NDArray, y: NDArray) -> None:
        """Fit the underlying model on the design matrix ``X`` and target ``y``."""

    @abstractmethod
    def predict_logq(self, X: NDArray) -> NDArray:
        """One predicted log-demand per row of X (rows in temporal order).

        ``window == 1`` models predict each row independently; the RNN uses up to
        ``window`` preceding rows (inclusive) as the sequence ending at that row.
        """

    # ---- shared concrete behaviour ----
    def fit_design(self, X: NDArray, y: NDArray) -> None:
        """Fit on an explicit design matrix (column 0 must be ``log price``)."""
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._fit(self._X, self._y)
        self._fitted = True

    def fit(self, history: History) -> None:
        prices = np.array([p[0] for p in history.prices])
        demand = np.array([d[0] for d in history.observed_demand])
        X = np.log(prices).reshape(-1, 1)
        y = np.log(np.clip(demand, 1e-9, None))
        self.fit_design(X, y)

    def predict_demand(self, prices: NDArray, context: dict | None = None) -> NDArray:
        logp = np.log(np.atleast_1d(prices)).reshape(-1, 1)
        X = logp
        if context and "controls" in context:
            X = np.column_stack([logp, np.atleast_2d(context["controls"])])
        return np.exp(self.predict_logq(X))

    def estimated_elasticity(self, n_grid: int = 21) -> NDArray:
        """Slope of predicted log-demand on log-price over a grid clipped to the
        observed price range (controls held at their medians).

        Trees are piecewise-constant, so a two-point finite difference is bimodal
        (≈0 inside a leaf, huge across a split); regressing predictions over a
        grid averages that out (~18x lower variance in testing). The grid is
        clipped to the 5th–95th percentile of observed ``log price`` because
        extrapolating beyond the observed range lands in flat tree regions and
        inflates the readout.

        Returns ``nan`` if the forecaster is unfit or price dispersion has
        collapsed (no slope to read) — an informative gap rather than a
        fabricated number when prices freeze.
        """
        if not self._fitted or self._X is None:
            return np.array([np.nan])
        logp = self._X[:, 0]
        lo, hi = np.quantile(logp, 0.05), np.quantile(logp, 0.95)
        if hi - lo < 1e-3:  # dispersion collapsed -> elasticity undefined
            return np.array([np.nan])
        grid = np.linspace(lo, hi, n_grid)
        med = np.median(self._X, axis=0)
        w = min(self.window, self._X.shape[0])
        base_ctx = self._X[-w:].copy()  # recent rows give the RNN its sequence
        preds = np.empty(n_grid)
        for k, g in enumerate(grid):
            ctx = base_ctx.copy()
            ctx[-1] = med.copy()
            ctx[-1, 0] = g
            preds[k] = float(self.predict_logq(ctx)[-1])
        slope = float(np.polyfit(grid, preds, 1)[0])
        return np.array([slope])
