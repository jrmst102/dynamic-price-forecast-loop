"""
Forecast–Price Feedback Loop
File: fploop/policies/decision_focused.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Decision-focused pricing arms (Family C): stabilise the closed loop rather than
de-bias it. Provides an EMA-smoothed-elasticity policy that damps oscillation and a
distributionally-robust policy that prices against the worst-case elasticity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fploop.generators.linear_logit import optimal_price
from fploop.policies.base import Policy
from fploop.types import History, MarketState, Observation


class DecisionFocusedPolicy(Policy):
    """Family C: regularised / stabilised closed loop.

    Following Kabra & Hanasusanto (2024) — naive retraining can be unstable. It
    fits like the greedy baseline but prices off an EMA-smoothed elasticity to
    damp oscillation. This improves the performative-stability metric and reduces
    price thrashing; it is **not** a causal correction, so a residual bias
    remains under endogeneity.
    """

    family = "decision_focused"

    def __init__(self, *args: object, kappa: float = 0.3, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.kappa = kappa
        self._beta_smoothed = -2.0

    def propose_price(self, state: MarketState, history: History) -> NDArray:
        beta_raw = self.forecaster.estimated_elasticity()[0]
        if len(history) < self.warmup or not np.isfinite(beta_raw):
            return self._warmup_price(state)
        self._beta_smoothed = (1 - self.kappa) * self._beta_smoothed + self.kappa * float(beta_raw)
        self._beta_hat = self._beta_smoothed
        return optimal_price(state.cost, self._beta_smoothed)

    def update(self, observation: Observation, history: History) -> None:
        self.forecaster.fit(history)

    def current_elasticity(self) -> NDArray:
        return np.array([self._beta_smoothed])


class DRORobustPolicy(Policy):
    """Family C: distributionally-robust pricing over an elasticity confidence set.

    Prices against the most-elastic plausible value ``beta_hat - z*se``, where
    ``se`` is a rolling standard deviation of the grid-slope estimate (an
    uncertainty proxy, floored so it is never zero). More-elastic demand lowers
    profit at any given price, so the worst-case elasticity over the set is its
    elastic endpoint and this closed form *is* the worst-case-optimal price: the
    firm takes a lower markup when uncertain. Its belief (``current_elasticity``)
    stays the central ``beta_hat``; the conservatism is purely a pricing choice,
    aimed at tail-risk (worst-case / CVaR regret) rather than mean regret.
    """

    family = "decision_focused"

    def __init__(
        self,
        *args: object,
        z: float = 1.0,
        window: int = 15,
        se_floor: float = 0.05,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.z, self.window, self.se_floor = z, window, se_floor
        self._beta_hat = -2.0
        self._beta_history: list[float] = []

    def propose_price(self, state: MarketState, history: History) -> NDArray:
        beta = self.forecaster.estimated_elasticity()[0]
        if len(history) < self.warmup or not np.isfinite(beta):
            return self._warmup_price(state)
        self._beta_hat = float(beta)
        self._beta_history.append(self._beta_hat)
        recent = self._beta_history[-self.window :]
        se = max(float(np.std(recent)), self.se_floor) if len(recent) > 1 else self.se_floor
        beta_worst = self._beta_hat - self.z * se  # the elastic (pessimistic) endpoint
        return optimal_price(state.cost, beta_worst)

    def update(self, observation: Observation, history: History) -> None:
        self.forecaster.fit(history)

    def current_elasticity(self) -> NDArray:
        return np.array([self._beta_hat])
