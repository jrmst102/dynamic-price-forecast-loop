"""
Forecast–Price Feedback Loop
File: fploop/generators/base.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Abstract base for demand-world generators. Defines the :class:`DemandWorld`
interface — a ground-truth environment with a known demand curve whose true
elasticity and optimal prices are computable exactly, so estimation bias is
measurable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from numpy.typing import NDArray

from fploop.types import MarketState, Observation, WorldConfig


class DemandWorld(ABC):
    """Ground-truth demand environment with a known demand curve.

    Concrete worlds own the true parameters, so `true_elasticity` and
    `optimal_prices` are computable exactly — this is what makes residual
    estimation bias measurable, the project's headline metric.
    """

    def __init__(self, config: WorldConfig) -> None:
        self.config = config

    @property
    def n_products(self) -> int:
        return self.config.n_products

    @property
    def current_state(self) -> MarketState:
        """State for the upcoming period (set by reset/step). Raises if not reset yet."""
        if getattr(self, "_state", None) is None:
            raise RuntimeError("call reset() before reading current_state")
        return self._state

    @abstractmethod
    def reset(self, seed: int | None = None) -> MarketState:
        """Reset to period 0 and return the initial state."""

    @abstractmethod
    def step(self, prices: NDArray) -> Observation:
        """Apply the firm's prices for the current period and advance one step."""

    @abstractmethod
    def true_elasticity(self) -> NDArray:
        """Ground-truth own-price elasticity at the current state, shape (n_products,)."""

    @abstractmethod
    def optimal_prices(self) -> NDArray:
        """Full-information revenue-optimal prices at the current state."""

    @abstractmethod
    def shock_path(self) -> NDArray:
        """The full demand-shock sequence xi_0..xi_{T-1}, available after reset(seed)."""

    @abstractmethod
    def cost_path(self) -> NDArray:
        """The full marginal-cost sequence c_0..c_{T-1}, available after reset(seed)."""
