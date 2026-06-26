"""
Forecast–Price Feedback Loop
File: tests/test_drift.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Phase 4a — drift / nonstationarity (§A.3, §A.9). The gradual schedule is a
sinusoid and the abrupt one a step, the oracle's demand tracks the drifting
intercept, and a naive pooling forecaster's late-window elasticity bias grows
under drift.
"""

from __future__ import annotations

import numpy as np

from fploop.forecasters.gbt import GBTForecaster
from fploop.generators.linear_logit import LinearLogitWorld
from fploop.loop import run_simulation
from fploop.metrics import residual_elasticity_bias
from fploop.policies import GreedyBaseline
from fploop.types import WorldConfig


def _alpha_path(w: LinearLogitWorld) -> np.ndarray:
    return w._alpha  # the pre-generated drifting intercept


def test_gradual_schedule_matches_sinusoid():
    cfg = WorldConfig(elasticity=-1.5, drift_kind="gradual", drift_magnitude=0.8, horizon=120)
    w = LinearLogitWorld(cfg)
    w.reset(0)
    alpha0 = np.log(cfg.base_demand)
    period = cfg.horizon // 2
    expected = alpha0 + 0.8 * np.sin(2.0 * np.pi * np.arange(cfg.horizon) / period)
    assert np.allclose(_alpha_path(w), expected)


def test_abrupt_schedule_is_a_step():
    cfg = WorldConfig(elasticity=-1.5, drift_kind="abrupt", drift_magnitude=0.6, horizon=100)
    w = LinearLogitWorld(cfg)
    w.reset(0)
    alpha0 = np.log(cfg.base_demand)
    alpha = _alpha_path(w)
    assert np.allclose(alpha[: cfg.horizon // 2], alpha0)
    assert np.allclose(alpha[cfg.horizon // 2 :], alpha0 + 0.6)


def test_oracle_demand_tracks_drift():
    # The oracle price is level-independent, but its realised demand follows the
    # drifting intercept — the oracle "tracks" the drift through demand/profit.
    from fploop.oracle import full_information_oracle

    cfg = WorldConfig(
        elasticity=-1.5,
        endogeneity_strength=0.6,
        shock_std=0.2,
        cost_shifter_std=0.15,
        drift_kind="gradual",
        drift_magnitude=0.8,
        horizon=200,
    )
    w = LinearLogitWorld(cfg)
    orc = full_information_oracle(w, seed=0)
    w.reset(0)
    corr = np.corrcoef(np.log(orc.observed_demand.ravel()), _alpha_path(w))[0, 1]
    assert corr > 0.5, f"oracle demand does not track drift (corr={corr:.2f})"


def _late_bias(res) -> float:
    b = np.abs(residual_elasticity_bias(res.estimated_elasticity, res.true_elasticity).ravel())
    q = int(0.75 * len(b))
    return float(np.nanmean(b[q:]))


def test_drift_inflates_naive_late_bias():
    # A naive accumulating forecaster pools data across the regime shift, so its
    # late-window elasticity bias is larger under (abrupt) drift than under none.
    base = dict(
        elasticity=-1.5,
        endogeneity_strength=0.3,
        shock_std=0.2,
        cost_shifter_std=0.15,
        horizon=200,
    )

    def greedy(cfg):
        return run_simulation(
            LinearLogitWorld(cfg),
            GreedyBaseline(GBTForecaster(), rng=np.random.default_rng(1)),
            seed=0,
        )

    none_bias = _late_bias(greedy(WorldConfig(**base)))
    drift_bias = _late_bias(greedy(WorldConfig(**base, drift_kind="abrupt", drift_magnitude=1.5)))
    assert drift_bias > none_bias, f"drift bias {drift_bias:.3f} not > none {none_bias:.3f}"
