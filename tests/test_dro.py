"""
Forecast–Price Feedback Loop
File: tests/test_dro.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Phase 4b — distributionally-robust pricing (§B1.4).

DRO's raison d'être is **conservatism under uncertainty**: it prices against the
elastic endpoint ``beta_hat - z*se`` of the elasticity confidence set, taking a
lower markup when the estimate is noisy, to cut tail risk.

Reportable 4a/4b finding: in the *closed loop* the greedy grid-slope locks in to
the markup floor (the project's core lock-in) right after warm-up, so a price-only
DRO arm — which does not itself break dispersion — coincides with greedy and its
``se`` correction never crosses the floor. The conservatism is therefore proved
directly on a controlled, dispersed estimate (where it provably bites), and the
closed-loop test asserts only that DRO never *worsens* the tail. Realising DRO's
tail benefit needs the estimate kept elastic (e.g. DRO composed with exploration)
— a Phase-4c / combination concern, flagged in the PR.
"""

from __future__ import annotations

import numpy as np

from fploop.forecasters.gbt import GBTForecaster
from fploop.generators.linear_logit import LinearLogitWorld, optimal_price
from fploop.loop import run_simulation
from fploop.metrics import cumulative_regret
from fploop.policies import DRORobustPolicy, GreedyBaseline
from fploop.types import History, MarketState, Observation, WorldConfig


def _dispersed_elastic_history(beta_true: float = -2.0, T: int = 200, seed: int = 0):
    """A clean, well-dispersed exogenous (log p, log q) sample and a GBT fit on it,
    so the grid-slope is genuinely elastic (well below the -1.05 markup floor)."""
    rng = np.random.default_rng(seed)
    logp = rng.normal(0.0, 0.4, size=T)
    y = np.log(100.0) + beta_true * logp + rng.normal(0.0, 0.05, size=T)
    f = GBTForecaster()
    f.fit_design(logp.reshape(-1, 1), y)
    hist = History()
    for t in range(T):
        p, q = float(np.exp(logp[t])), float(np.exp(y[t]))
        hist.add(
            Observation(
                period=t,
                prices=np.array([p]),
                observed_demand=np.array([q]),
                revenue=p * q,
                cost=np.array([1.0]),
            )
        )
    return f, hist


def test_dro_prices_more_conservatively_under_uncertainty():
    # With a genuinely elastic estimate and non-trivial estimate variance, DRO
    # prices below the greedy markup (the elastic worst-case endpoint).
    f, hist = _dispersed_elastic_history()
    state = MarketState(
        period=len(hist), reference_price=np.array([1.0]), latent_params={}, cost=np.array([1.0])
    )
    dro = DRORobustPolicy(f, warmup=10)
    dro._beta_history = [-2.3, -1.8, -2.1, -1.7, -2.2, -1.9]  # elevated estimate spread
    p_dro = float(dro.propose_price(state, hist)[0])

    beta_hat = float(f.estimated_elasticity()[0])
    p_greedy = float(optimal_price(np.array([1.0]), beta_hat)[0])
    assert beta_hat < -1.05, f"estimate {beta_hat:.2f} not elastic enough to exercise DRO"
    assert p_dro < p_greedy, f"DRO price {p_dro:.3f} not below greedy markup {p_greedy:.3f}"


def test_dro_belief_is_the_central_estimate():
    # Its reported belief is the central beta_hat; conservatism is only a pricing
    # choice, so current_elasticity matches the greedy grid-slope reading.
    f, hist = _dispersed_elastic_history()
    state = MarketState(
        period=len(hist), reference_price=np.array([1.0]), latent_params={}, cost=np.array([1.0])
    )
    dro = DRORobustPolicy(f, warmup=10)
    dro.propose_price(state, hist)
    assert np.isclose(dro.current_elasticity()[0], f.estimated_elasticity()[0])


def _final_regret(res) -> float:
    return float(
        cumulative_regret(np.asarray(res.realized_revenue), np.asarray(res.oracle_revenue))[-1]
    )


def test_dro_never_worsens_the_tail_in_closed_loop():
    # Closed-loop lock-in pins both arms to the markup floor, so DRO coincides with
    # greedy and — at minimum — never worsens worst-case regret.
    cfg = WorldConfig(
        elasticity=-1.5,
        endogeneity_strength=0.2,
        shock_std=0.35,
        cost_shifter_std=0.15,
        horizon=50,
    )

    def run(cls, seed):
        return run_simulation(
            LinearLogitWorld(cfg), cls(GBTForecaster(), rng=np.random.default_rng(1)), seed=seed
        )

    seeds = range(12)
    g = np.array([_final_regret(run(GreedyBaseline, s)) for s in seeds])
    d = np.array([_final_regret(run(DRORobustPolicy, s)) for s in seeds])
    assert np.percentile(d, 95) <= np.percentile(g, 95) + 1e-6
