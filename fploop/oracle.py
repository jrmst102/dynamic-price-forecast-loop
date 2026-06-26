"""
Forecast–Price Feedback Loop
File: fploop/oracle.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

The full-information benchmark: forward-simulates an endogeneity-free clone of the
world (same seeded shock/cost/competitor/drift draws) that prices at the myopic
per-period optimum each step, yielding the optimal price path and its profit
series used as the regret baseline.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from fploop.generators.base import DemandWorld
from fploop.types import RunResult


def full_information_oracle(
    world: DemandWorld,
    *,
    horizon: int | None = None,
    seed: int | None = None,
) -> RunResult:
    """Compute the full-information optimal price path and its profit.

    The reference price is path-dependent on the pricer's own prices, so the
    oracle can no longer be evaluated analytically: it is a **forward simulation
    on an endogeneity-free clone** of ``world``. Setting
    ``endogeneity_strength=0`` leaves the seeded draws (shock, cost, competitor,
    drift) untouched — only the price→effective-price map changes — so the clone
    sees exactly the same exogenous paths as the policy run on the same ``seed``.
    Their reference paths differ (each tracks its own prices), which is correct.

    Each period the clone prices at :meth:`optimal_prices` (the myopic
    full-information per-period optimum, capacity-aware when censoring binds) and
    steps. No estimation: the elasticity series is the true short-run elasticity.

    Notes
    -----
    The ``oracle_revenue`` field carries the **profit** series (not revenue), per
    the Phase 2 profit objective; ``metadata["objective"] == "profit"`` records
    this.
    """
    cfg = replace(world.config, endogeneity_strength=0.0)  # frozen dataclass -> replace
    w = type(world)(cfg)
    w.reset(seed)  # same seed -> same xi/cost/competitor/drift draws
    T = horizon or cfg.horizon
    prices, demand, profit, revenue = [], [], [], []
    for _ in range(T):
        p_opt = w.optimal_prices()
        obs = w.step(p_opt)
        prices.append(obs.prices)
        demand.append(obs.observed_demand)
        profit.append(float(((obs.prices - obs.cost) * obs.observed_demand)[0]))
        revenue.append(obs.revenue)
    e = cfg.elasticity - cfg.reference_effect  # short-run elasticity (= true each period)
    return RunResult(
        prices=np.array(prices).reshape(T, 1),
        observed_demand=np.array(demand).reshape(T, 1),
        realized_revenue=np.array(revenue),
        oracle_prices=np.array(prices).reshape(T, 1),
        oracle_revenue=np.array(profit),  # profit benchmark series
        estimated_elasticity=np.full((T, 1), e),
        true_elasticity=np.full((T, 1), e),
        metadata={"objective": "profit"},
    )
