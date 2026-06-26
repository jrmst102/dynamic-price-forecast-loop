"""
Forecast–Price Feedback Loop
File: fploop/loop.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

The core simulation driver: turns the forecast -> price -> demand -> retrain loop
for a given world and policy, computes the same-seed full-information oracle for a
fair paired comparison, and logs the per-period prices, demand, profit, and
estimated-vs-true elasticity into a RunResult.
"""

from __future__ import annotations

import numpy as np

from fploop.generators.base import DemandWorld
from fploop.oracle import full_information_oracle
from fploop.policies.base import Policy
from fploop.types import History, RunResult


def run_simulation(
    world: DemandWorld,
    policy: Policy,
    *,
    horizon: int | None = None,
    seed: int | None = None,
) -> RunResult:
    """Turn the forecast -> price -> demand -> retrain loop and log everything.

    For each of ``horizon`` periods: the policy proposes a (base) price from the
    current world state and history, the world returns an observation at the
    effective price, the observation is appended to history, and after warm-up
    the policy updates (retraining its forecaster every ``retrain_every``
    periods). Per-period prices, demand, profit, and the estimated-vs-true
    elasticity are recorded for downstream metrics and figures.

    The oracle is computed first on the same ``seed``; the world is then reset
    again so the policy run sees identical shock/cost paths — a fair comparison.

    Returns
    -------
    RunResult
        Per-period log aligned with the oracle path produced by
        :func:`fploop.oracle.full_information_oracle`. ``realized_revenue``
        carries the **profit** series, matching ``oracle_revenue``.
    """
    T = horizon or world.config.horizon
    oracle = full_information_oracle(world, horizon=T, seed=seed)  # resets world internally
    state = world.reset(seed)  # reset again: identical shocks
    history = History()
    prices, demand, revenue, profit, est_e = [], [], [], [], []
    beta = world.config.elasticity
    for t in range(T):
        p_base = policy.propose_price(state, history)
        obs = world.step(p_base)
        history.add(obs)
        if len(history) > policy.warmup and (t % policy.retrain_every == 0):
            policy.update(obs, history)
        prices.append(obs.prices)
        demand.append(obs.observed_demand)
        revenue.append(obs.revenue)
        profit.append(float(((obs.prices - obs.cost) * obs.observed_demand)[0]))
        est_e.append(policy.current_elasticity())
        state = world.current_state if t < T - 1 else state
    return RunResult(
        prices=np.array(prices).reshape(T, 1),
        observed_demand=np.array(demand).reshape(T, 1),
        realized_revenue=np.array(profit),  # profit series (matches oracle.oracle_revenue)
        oracle_prices=oracle.oracle_prices,
        oracle_revenue=oracle.oracle_revenue,
        estimated_elasticity=np.array(est_e).reshape(T, 1),
        true_elasticity=np.full((T, 1), beta),
        metadata={"objective": "profit", "family": policy.family},
    )
