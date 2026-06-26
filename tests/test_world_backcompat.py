"""
Forecast–Price Feedback Loop
File: tests/test_world_backcompat.py
Author: Dr. Jose Mendoza <jose.mendoza@nyu.edu>

Phase 4a — back-compatibility: all toggles off reproduces Phase 3.5 (§A.9).

With ``reference_effect=0``, ``drift_kind="none"``, ``competition=False`` and
``censoring=False`` the unified DGP must reduce *bit-identically* to the 3.5
equation ``log q = alpha + beta*log p_eff + xi``.
"""

from __future__ import annotations

import numpy as np
import pytest

from fploop.generators.linear_logit import LinearLogitWorld
from fploop.types import WorldConfig

CFG_35 = WorldConfig(
    elasticity=-1.8,
    endogeneity_strength=0.6,
    shock_std=0.2,
    cost_shifter_std=0.15,
    shock_ar1=0.3,
    horizon=60,
)


def test_step_matches_phase35_equation():
    w = LinearLogitWorld(CFG_35)
    w.reset(7)
    xi, cost = w.shock_path(), w.cost_path()
    alpha, beta, lam = np.log(CFG_35.base_demand), CFG_35.elasticity, CFG_35.endogeneity_strength
    rng = np.random.default_rng(99)
    for t in range(CFG_35.horizon):
        p = np.array([1.5 + abs(rng.normal(0.0, 0.4))])
        obs = w.step(p)
        log_p_eff = np.log(p) + lam * xi[t]
        p_eff = np.exp(log_p_eff)
        q = np.exp(alpha + beta * log_p_eff + xi[t])
        assert obs.prices[0] == pytest.approx(p_eff[0])
        assert obs.observed_demand[0] == pytest.approx(q[0])
        assert obs.revenue == pytest.approx(float((p_eff * q)[0]))
        assert obs.cost[0] == pytest.approx(cost[t])
        assert bool(obs.censored[0]) is False


def test_strategic_and_multiproduct_raise():
    with pytest.raises(NotImplementedError):
        LinearLogitWorld(WorldConfig(strategic=True)).reset(0)
    with pytest.raises(NotImplementedError):
        LinearLogitWorld(WorldConfig(n_products=2)).reset(0)


def test_inelastic_effective_elasticity_raises():
    # A positive reference_effect only makes short-run demand more elastic; a
    # reference_effect that pushes beta - theta_ref >= -1 (inelastic) is rejected.
    with pytest.raises(ValueError):
        LinearLogitWorld(WorldConfig(elasticity=-1.5, reference_effect=-0.7)).reset(0)


def test_bad_drift_kind_raises():
    with pytest.raises(ValueError):
        LinearLogitWorld(WorldConfig(drift_kind="wobbly")).reset(0)
