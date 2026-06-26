"""
Forecast–Price Feedback Loop
File: fploop/policies/base.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Abstract base class for pricing policies: defines the common warm-up phase,
retraining schedule, and the propose-price / update interface every intervention
arm implements.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from fploop.forecasters.base import Forecaster
from fploop.generators.linear_logit import optimal_price
from fploop.types import History, MarketState, Observation


class Policy(ABC):
    """A pricing policy: proposes a price each period and updates on new data.

    `family` tags which intervention family the policy belongs to, mirroring
    AIF360's pre/in/post grouping:
    "baseline" | "exploration" | "causal" | "decision_focused".

    All policies share a warm-up phase: for the first ``warmup`` periods they
    return dispersed prices around a default markup to seed estimation. After
    warm-up they fit in :meth:`update` (every ``retrain_every`` periods) and
    price in :meth:`propose_price`.
    """

    family: str = ""

    def __init__(
        self,
        forecaster: Forecaster,
        *,
        warmup: int = 15,
        retrain_every: int = 4,
        explore_std: float = 0.30,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.forecaster = forecaster
        self.warmup = warmup
        self.retrain_every = retrain_every
        self.explore_std = explore_std
        self.rng = rng or np.random.default_rng(0)
        self._beta_hat = -2.0  # prior estimate before any fit

    def _warmup_price(self, state: MarketState) -> NDArray:
        # dispersed prices around a default markup to seed estimation
        base = optimal_price(state.cost, self._beta_hat)
        return base * np.exp(self.rng.normal(0.0, self.explore_std, size=base.shape))

    def current_elasticity(self) -> NDArray:
        """The elasticity estimate this policy is currently pricing on (for display).

        The default reads the forecaster's grid-slope, returning ``nan`` until the
        forecaster is fit. Policies that price off their own estimate (e.g. the
        2SLS causal arm or the smoothed decision-focused arm) override this.
        """
        if not self.forecaster.is_fitted:
            return np.array([np.nan])
        return self.forecaster.estimated_elasticity()

    @abstractmethod
    def propose_price(self, state: MarketState, history: History) -> NDArray:
        """Choose prices for the current period."""

    @abstractmethod
    def update(self, observation: Observation, history: History) -> None:
        """Incorporate the latest observation (retrain the forecaster as needed)."""
