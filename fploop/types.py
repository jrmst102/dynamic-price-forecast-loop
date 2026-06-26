"""
Forecast–Price Feedback Loop
File: fploop/types.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Core data structures shared across the simulator: the WorldConfig knobs, the
ground-truth MarketState, the per-period Observation, the accumulating History
log a forecaster/policy trains on, and the RunResult downstream code plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class WorldConfig:
    """Configuration for a ground-truth demand world.

    Toggles default to the simplest regime (single product, exogenous prices,
    no censoring/drift/competition). Phase 2 activates `elasticity` +
    `endogeneity_strength`; later phases activate the remaining switches.
    """

    n_products: int = 1
    horizon: int = 200
    base_demand: float = 100.0
    elasticity: float = -1.5  # true own-price elasticity (negative)
    endogeneity_strength: float = 0.0  # 0 -> exogenous prices; >0 activates loop bias
    shock_std: float = 0.0  # std of the unobserved cost/quality shock
    reference_effect: float = 0.0  # weight on past-price reference term
    marginal_cost: float = 1.0  # mean marginal cost c_bar
    cost_shifter_std: float = 0.15  # sd of the log cost shifter (the IV instrument)
    shock_ar1: float = 0.0  # AR(1) coefficient rho for the demand shock
    reference_memory: float = 0.7  # EMA weight gamma for the reference price
    drift: bool = False  # legacy nonstationarity flag (superseded by drift_kind)
    drift_kind: str = "none"  # "none" | "gradual" | "abrupt"
    drift_magnitude: float = 0.0  # size of the intercept drift
    drift_period: int = 0  # gradual sinusoid period; 0 -> horizon // 2
    censoring: bool = False  # truncate observed demand at capacity
    capacity: float | None = None
    competition: bool = False
    cross_elasticity: float = 0.0  # competitor cross-price elasticity (substitutes > 0)
    competitor_price_mean: float = 1.0
    competitor_price_std: float = 0.10  # sd of log competitor price
    strategic: bool = False
    seed: int | None = None


@dataclass
class MarketState:
    """Ground-truth state of the world at the start of a period."""

    period: int
    reference_price: NDArray  # shape (n_products,)
    latent_params: dict  # true params this period (may drift)
    inventory: NDArray | None = None  # shape (n_products,) when censoring is on
    cost: NDArray | None = None  # current marginal cost, shape (n_products,)
    competitor_price: NDArray | None = None  # rival price this period when competition is on


@dataclass
class Observation:
    """What the firm observes after setting a price for one period."""

    period: int
    prices: NDArray  # chosen prices, shape (n_products,)
    observed_demand: NDArray  # possibly censored, shape (n_products,)
    revenue: float
    competitor_prices: NDArray | None = None
    censored: NDArray | None = None  # bool mask where stockout truncated demand
    cost: NDArray | None = None  # realised marginal cost this period (the instrument)
    reference_price: NDArray | None = None  # r_t used in this period's demand (EMA of past prices)


@dataclass
class History:
    """Accumulating log a forecaster or policy may train on."""

    periods: list[int] = field(default_factory=list)
    prices: list[NDArray] = field(default_factory=list)
    observed_demand: list[NDArray] = field(default_factory=list)
    competitor_prices: list[NDArray] = field(default_factory=list)
    costs: list[NDArray] = field(default_factory=list)
    reference_prices: list[NDArray] = field(default_factory=list)

    def add(self, obs: Observation) -> None:
        """Append a single period's observation to the log."""
        self.periods.append(obs.period)
        self.prices.append(obs.prices)
        self.observed_demand.append(obs.observed_demand)
        if obs.competitor_prices is not None:
            self.competitor_prices.append(obs.competitor_prices)
        if obs.cost is not None:
            self.costs.append(obs.cost)
        if obs.reference_price is not None:
            self.reference_prices.append(obs.reference_price)

    def __len__(self) -> int:
        return len(self.periods)

    def as_frame(self) -> pd.DataFrame:
        """Return the log as a tidy DataFrame (one row per period-product)."""
        rows = []
        for i, period in enumerate(self.periods):
            prices = np.atleast_1d(self.prices[i])
            demand = np.atleast_1d(self.observed_demand[i])
            for j in range(prices.shape[0]):
                rows.append(
                    {
                        "period": period,
                        "product": j,
                        "price": float(prices[j]),
                        "observed_demand": float(demand[j]),
                    }
                )
        return pd.DataFrame(rows)


@dataclass
class RunResult:
    """Per-period log of one simulation run — everything downstream code plots.

    All array fields are shape (T,) for scalars or (T, n_products) for
    per-product series, where T is the number of pricing cycles.
    """

    prices: NDArray
    observed_demand: NDArray
    realized_revenue: NDArray
    oracle_prices: NDArray
    oracle_revenue: NDArray
    estimated_elasticity: NDArray
    true_elasticity: NDArray
    metadata: dict = field(default_factory=dict)
