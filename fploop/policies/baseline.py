"""
Forecast–Price Feedback Loop
File: fploop/policies/baseline.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Baseline pricing arm: the greedy estimate-then-optimize policy that fits demand to
its own prices and re-optimizes against the fit — the arm expected to spiral under
endogeneity, with no exploration or bias correction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fploop.generators.linear_logit import optimal_price
from fploop.policies.base import Policy
from fploop.types import History, MarketState, Observation


class GreedyBaseline(Policy):
    """Arm 0: greedy estimate-then-optimize.

    Fit demand to whatever the current prices produced, then price optimally
    against that fit, repeat — the arm expected to spiral. No exploration, no
    bias correction.
    """

    family = "baseline"

    def propose_price(self, state: MarketState, history: History) -> NDArray:
        beta = self.forecaster.estimated_elasticity()[0]
        if len(history) < self.warmup or not np.isfinite(beta):
            return self._warmup_price(state)
        self._beta_hat = float(beta)
        return optimal_price(state.cost, self._beta_hat)

    def update(self, observation: Observation, history: History) -> None:
        self.forecaster.fit(history)
