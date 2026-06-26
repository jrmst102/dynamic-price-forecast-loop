"""
Forecast–Price Feedback Loop
File: fploop/features.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Design-matrix construction with optional controls (Phase 4b §B1.2).

Policies decide what their forecaster conditions on by building the design matrix
themselves. The naive/greedy/exploration arms use price only (``controls=()``),
which is what keeps the omitted-variable lock-in visible; the richer causal arms
(4b.2) condition on the confounders the 4a worlds introduce.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fploop.types import History

CONTROLS = ("cost", "competitor", "reference", "time")


def reference_series(history: History, reference_memory: float = 0.7) -> NDArray:
    """The per-period reference price for each row of ``history``.

    Uses the logged reference price when the world recorded it (exact), otherwise
    falls back to the EMA of past effective prices ``r_{t+1} = gamma*r_t +
    (1-gamma)*p_t`` seeded at the first observed price.
    """
    prices = np.array([p[0] for p in history.prices])
    n = prices.shape[0]
    if len(history.reference_prices) == n:
        return np.array([r[0] for r in history.reference_prices])
    r = np.empty(n)
    r[0] = prices[0]  # no marginal_cost available here; seed at the first price
    for t in range(1, n):
        r[t] = reference_memory * r[t - 1] + (1.0 - reference_memory) * prices[t - 1]
    return r


def build_design(
    history: History,
    controls: tuple[str, ...] = (),
    reference_memory: float = 0.7,
) -> tuple[NDArray, NDArray]:
    """Build ``(X, y)`` for a demand regression in temporal order.

    Column 0 of ``X`` is always ``log price``; the remaining columns are the
    requested ``controls`` (a subset of :data:`CONTROLS`) in the given order.
    ``y`` is ``log`` observed demand. This keeps the grid-slope elasticity readout
    (perturb column 0, controls held at medians) pointed at the own-price slope.
    """
    prices = np.array([p[0] for p in history.prices])
    demand = np.array([d[0] for d in history.observed_demand])
    cols = [np.log(prices)]
    for c in controls:
        if c == "cost":
            cols.append(np.log(np.array([x[0] for x in history.costs])))
        elif c == "competitor":
            cols.append(np.log(np.array([x[0] for x in history.competitor_prices])))
        elif c == "reference":
            cols.append(np.log(reference_series(history, reference_memory)))
        elif c == "time":
            cols.append(np.arange(prices.shape[0], dtype=float))
        else:
            raise ValueError(f"unknown control {c!r}; expected a subset of {CONTROLS}")
    X = np.column_stack(cols)
    y = np.log(np.clip(demand, 1e-9, None))
    return X, y
